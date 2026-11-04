"""`talos setup | run | compile | status`. Wizard prompts go through `ask` so tests can script
them; each loop event is printed to stdout as one line, and every finished iteration ends with a
summary line: outcome, best delta, LLM and compute spend and the wall-clock time left."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Callable
import types
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from talos import __version__, extras, image_updates, workspace
from talos import mainnet as mainnet_api
from talos.base_algorithm import (job_void_reason, read_algorithm_dir, read_base_root,
                                  root_problem, write_base_root)
from talos.budget import (ADD_DIMENSIONS, Budget, BudgetExhausted, Spend, effective_budget,
                          exhausted)
from talos.extend import ADD_FLAGS, BUDGET_FLAGS, ExtensionRefused, extend
from talos.c3_bench import C3CommandError
from talos.c3_jobdir import LocalSettings
from talos.c3_transport import CliTransport, make_transport
from talos.challenges import (CHALLENGES, DEFAULT_CHALLENGE, DEV_IMAGE_TAG, MONOREPO_REF,
                              TEMPLATE_START_FILES,
                              c3_hardware_options, c3_hardware_class,
                              c3_image, hardware_options, hardware_class, local_hardware_class,
                              status_text, status_warning)
from talos.config import (Config, ConfigError, ENV_KEYS, load, resolve_api_key,
                          resolve_c3_api_key, save)
from talos.deploy_digest import marker_name
from talos.diagnostics import first_error
from talos.leak_check import KNOWN_LEAKING
from talos.lineage import AnnealSettings
from talos.local_transport import (DockerInfo, DockerTransport, docker_info, has_gpu_runtime,
                                   prepare)
from talos.mainnet import ChallengeInfo, MainnetError, TrackHyperparameters, fetch_challenge_info
from talos.masked_input import ask_secret
from talos.nonces import HOLDOUT_START, NONCES_PER_TRACK, draw_nonce_sets, new_rand_hash
from talos.providers import DEFAULT_MODELS, KINDS, make_provider, validate_provider
from talos.providers.codex_cli import list_codex_models
from talos.providers.pricing import estimate_cost
from talos.prompts import WHOLE_FILE_FORMAT
from talos.wizard import ask_choice, confirm
from talos.scoring import NOISE_WORDS
from talos.state import JobSpec, JobState, JobStore, user_tacit
from talos.types import Usage

BASELINE_CACHE = Path.home() / ".talos" / "baselines"
CALIBRATION_CACHE = Path.home() / ".talos" / "calibration"
# Deployed as a module from the directory that holds the `talos` package, never by file
# path: a file path makes Modal import the app as top-level `talos_bench`, the serialized
# functions then refer to that name, and no container can load them.
MODAL_APP_MODULE = "talos.modal_app.talos_bench"
MODAL_APP_ROOT = Path(__file__).resolve().parent.parent
CLI_PROVIDERS = ("claude-cli", "codex-cli")
UNMETERED = CLI_PROVIDERS + ("fake",)
DEFAULT_COMPUTE_USD = 20.0
BACKENDS = ("modal", "c3", "local")
LOCAL_DEFAULT_MEMORY_GIB = 8  # when the daemon does not report its memory
LOCAL_MIN_MEMORY_GIB = 4
LOCAL_MEMORY_HEADROOM_GIB = 4
IMAGE_HINT = ("the C3 backend pulls the official TIG dev image from GHCR; check that "
              "`DEV_IMAGE_TAG` in talos/challenges.py names a published tag")


def default_ask(prompt: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    v = ask_secret(f"{prompt}{suffix}: ") if secret else input(f"{prompt}{suffix}: ")
    return v.strip() or (default or "")


def deploy_bench(token_id: str | None, token_secret: str | None, run=subprocess.run) -> None:
    base = [sys.executable, "-m", "modal"]  # the modal this Talos was installed with
    if token_id and token_secret:
        r = run(base + ["token", "set", "--token-id", token_id, "--token-secret", token_secret],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise ConfigError(f"modal token set failed: {r.stderr[-500:]}")
    r = run(base + ["deploy", "-m", MODAL_APP_MODULE], capture_output=True, text=True,
            encoding="utf-8", errors="replace", cwd=str(MODAL_APP_ROOT))
    if r.returncode != 0:
        raise ConfigError(f"modal deploy failed: {(r.stderr or r.stdout)[-2000:]}")


def check_c3(run=None, api_key: str | None = None, transport=None) -> float:
    """Confirms C3 is reachable and authenticated — over MCP when a key is configured, over the
    `c3` CLI otherwise — and returns the credit balance in GBP. 0.0 means "could not read it"."""
    from talos.c3_mcp import McpAuthError

    def key_rejected(e: Exception) -> str:
        return (f"C3 rejected the API key: {e}; check `c3 apikey list` and "
                f"run `talos setup` again")

    if transport is not None:
        t = transport
    elif run is not None:
        t = CliTransport(run=run, api_key=api_key)  # an injected runner means the CLI, keyed or not
    else:
        t = make_transport(api_key, run=subprocess.run)  # resolved now: tests patch cli.subprocess
    try:
        t.whoami()
    except McpAuthError as e:
        raise ConfigError(key_rejected(e)) from None
    except C3CommandError as e:
        keyed = api_key or os.environ.get("C3_API_KEY")
        fix = ("check the C3 API key (`c3 apikey list`)" if keyed
               else "install the `c3` CLI and run `c3 login`, or give setup a C3 API key "
                    "(`c3 apikey create`),")
        raise ConfigError(f"C3 login check failed: {e}; {fix} and retry") from None
    try:
        balance = t.balance_gbp()
    except McpAuthError as e:
        # McpAuthError is a C3CommandError; without this clause it falls into the branch
        # below and only warns, letting a revoked key through as if it had credit
        raise ConfigError(key_rejected(e)) from None
    except C3CommandError as e:
        print(f"warning: could not read the C3 balance: {e}", file=sys.stderr)
        return 0.0
    if balance is None:
        print("warning: could not read the C3 balance", file=sys.stderr)
        return 0.0
    if balance < 1.0:
        print(f"warning: C3 credit balance is low (£{balance:.2f}); run `c3 topup`",
              file=sys.stderr)
    return balance


def default_local_memory_gib(total_gib: int | None) -> int:
    """`total_gib` is what the daemon reports (talos/local_transport.py::docker_info), which on
    Docker Desktop is the VM's allocation, not the host's memory. Never above the total: a floor
    the daemon cannot give would be proposed by setup and then refused by it."""
    if total_gib is None:
        return LOCAL_DEFAULT_MEMORY_GIB
    return min(total_gib, max(LOCAL_MIN_MEMORY_GIB, total_gib - LOCAL_MEMORY_HEADROOM_GIB))


def local_settings(cfg: Config | None, info: DockerInfo | None = None) -> LocalSettings:
    """The container limits `talos setup` wrote. Without them (a `talos compile` in the agentic
    sandbox has no config at all, and scores no nonce, so no hardware class) they are the
    daemon's own figures, never the host's: Docker refuses `--cpus` above its CPU count, and on
    Docker Desktop that count is the VM's."""
    cpus = cfg.local_cpus if cfg else None
    mem = cfg.local_memory_gib if cfg else None
    if not (cpus and mem):
        info = info or docker_info()
        cpus = cpus or info.ncpu or os.cpu_count() or 1
        mem = mem or default_local_memory_gib(info.mem_total_gib)
    return LocalSettings(cpus=cpus, memory_gib=mem)


def check_local(run=None) -> DockerInfo:
    """Docker must answer; returns what the daemon reports (runtimes, CPUs, memory)."""
    try:
        return docker_info(run or subprocess.run)
    except C3CommandError as e:
        raise ConfigError(f"Docker check failed: {e}; install and start Docker "
                          f"(docker.com), then run `talos setup` again") from None


def image_available(challenge: str, fetch=None) -> bool:
    """True when the manifest for the challenge's dev image tag is on GHCR, which is what C3
    pulls. Anonymous access to GHCR needs a pull-scoped token first, so this is two requests.
    `fetch(url, headers)` returns `(status, body)`; status 0 means unreachable."""
    name, tag = c3_image(challenge).removeprefix("ghcr.io/").rsplit(":", 1)
    if fetch is None:
        def fetch(u, headers):
            try:
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return resp.status, resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                return e.code, ""
            except (OSError, http.client.HTTPException):
                # URLError (an OSError) wraps connect failures only; a body read timeout, a
                # reset connection or a truncated response arrive bare. All of them are
                # unreachable: reported as "not available", the message says so
                return 0, ""
    status, body = fetch(f"https://ghcr.io/token?scope=repository:{name}:pull", {})
    if status != 200:
        return False
    try:
        token = json.loads(body)["token"]
    except (ValueError, KeyError, TypeError):
        return False
    accept = ", ".join(("application/vnd.oci.image.index.v1+json",
                        "application/vnd.oci.image.manifest.v1+json",
                        "application/vnd.docker.distribution.manifest.list.v2+json",
                        "application/vnd.docker.distribution.manifest.v2+json"))
    status, _ = fetch(f"https://ghcr.io/v2/{name}/manifests/{tag}",
                      {"Authorization": f"Bearer {token}", "Accept": accept})
    return status == 200


def make_bench(backend: str, run_dir: Path, pending, c3_api_key: str | None = None,
               local: LocalSettings | None = None):
    if backend == "modal":
        from talos.bench import ModalBench
        return ModalBench()
    if backend == "c3":
        from talos.c3_bench import C3Bench
        return C3Bench(run_dir, pending=pending, api_key=c3_api_key)
    if backend == "local":
        from talos.c3_bench import C3Bench
        return C3Bench(run_dir, pending=pending, transport=DockerTransport(),
                       local=local or local_settings(None), usd_per_hour=0.0)
    raise ConfigError(f"unknown backend {backend!r}; run `talos setup`")


def bench_hardware_class(backend: str, challenge: str, local: LocalSettings | None = None,
                         gpu_name: str | None = None, host: str | None = None,
                         hardware: str | None = None) -> str:
    """`hardware` is the Modal GPU or C3 class or profile the job was frozen to
    (JobState.hardware); the local
    backend's class carries the `nvidia-smi` name instead."""
    spec = CHALLENGES[challenge]
    if backend == "local":
        return local_hardware_class(spec, local.cpus, local.memory_gib, gpu_name,
                                    host or socket.gethostname())
    return c3_hardware_class(spec, hardware) if backend == "c3" else hardware_class(spec, hardware)


def freeze_hardware(bench, spec, state, store, backend: str, log=print,
                    clock: Callable[[], float] = time.time) -> None:
    """Fixes the hardware a job runs on, once. The first run probes through the bench and saves
    the choice; a resume hands the saved choice back, so the candidates score on the hardware
    the baseline was measured on. Exported as TALOS_HARDWARE for the sandbox's `talos compile`.
    Where there is nothing to choose (a CPU challenge on Modal, the local backend, the fake
    bench) the choice stays None.

    The probe is a compute call (invariant 5): the budget is checked before it, and what the
    bench charged for it goes on compute spend, as `talos/loop.py::_BudgetedBench` does for
    the baseline."""
    cs = CHALLENGES[spec.challenge]
    if backend == "local":
        options = ()
    else:
        options = c3_hardware_options(cs) if backend == "c3" else hardware_options(cs)
    if state.hardware is None and state.baseline is not None and options:
        # A job from before the choice existed: its baseline ran on the one class there was,
        # the first option. Probing now could move its candidates elsewhere.
        state.hardware = options[0]
        store.save(state)
        store.event("hardware_selected", hardware=state.hardware,
                    reason="job predates the hardware choice")
    if state.hardware is None:
        if options:
            dim = exhausted(effective_budget(spec.budget, state.budget_extensions),
                            state.spend, clock())
            if dim:
                raise BudgetExhausted(dim)
            log(f"Probing capacity for {spec.challenge}...")
        mark = bench.cost_mark()
        try:
            state.hardware = bench.select_hardware(spec.challenge)
        finally:
            state.spend.compute_usd += bench.cost_usd_since(mark)
            store.save(state)
        if state.hardware is not None:
            store.event("hardware_selected", hardware=state.hardware)
            log(f"Hardware: {state.hardware}")
    else:
        bench.select_hardware(spec.challenge, chosen=state.hardware)
    if state.hardware is None:
        os.environ.pop("TALOS_HARDWARE", None)
    else:
        os.environ["TALOS_HARDWARE"] = state.hardware


def _model_prompt(kind: str) -> tuple[str, str | None]:
    """Prompt text and default for the model question. CLI providers know their own models:
    codex publishes a catalog, claude accepts short aliases. The static default stands in when
    the catalog cannot be read."""
    default = DEFAULT_MODELS.get(kind) or None
    if kind == "codex-cli":
        models = list_codex_models()
        if models:
            print("Models your codex CLI accepts: " + ", ".join(models))
            return "Model", models[0]
    if kind == "claude-cli":
        return "Model (an alias such as fable, opus or sonnet, or a full model id)", default
    return "Model", default


BACKEND_OPTIONS = (
    ("modal", "your Modal account; billed per container second"),
    ("c3", "your C3 account; about 12 min fixed overhead per iteration"),
    ("local", "Docker on this machine; no compute charge, slow builds"),
)
PROVIDER_EXPLANATIONS = {
    "anthropic": "Anthropic API; needs an API key",
    "openai": "OpenAI API; needs an API key",
    "google": "Google Gemini API; needs an API key",
    "openrouter": "OpenRouter API; needs an API key",
    "custom": "any OpenAI-compatible endpoint; needs a base URL and an API key",
    "claude-cli": "your local `claude` CLI login; no API key",
    "codex-cli": "your local `codex` CLI login; no API key",
}
MODE_OPTIONS = (
    ("single-shot", "one LLM call per edit"),
    ("agentic", "the CLI agent edits and compiles in a sandbox; roughly 5-20x the tokens"),
)
HYPERPARAMETER_OPTIONS = (
    ("mainnet", "per-track values from the baseline's best mainnet benchmark"),
    ("none", "run without hyperparameters"),
)


def _track_arg(answer: str | None) -> str | None:
    """The focus track a flag or wizard answer names; None (every track) for no answer or
    "all", which is how the wizard prompt and the README spell the default."""
    if answer is None or answer.strip() in ("", "all"):
        return None
    return answer.strip()


def _mainnet_hyperparameters(api, challenge: str, info) -> tuple[dict | None, dict | None,
                                                                 dict | None]:
    """(baseline_algorithm, hyperparameters, hyperparameters_source) for a new job. The algorithm
    is pinned whenever mainnet has one, so the baseline measured later is the code the map
    belongs to. The map is None when no track has a benchmark of it at this fuel."""
    top = api.top_algorithm(challenge)
    if top is None:
        return None, None, None  # resolve_baseline reports the missing algorithm
    name, algorithm_id, adoption = top
    algorithm = {"name": name, "id": algorithm_id, "adoption": adoption}
    per_track = api.top_hyperparameters(algorithm_id, info.tracks, info.max_fuel)
    found = {t: th for t, th in per_track.items() if th.benchmark_id is not None}
    if not found:
        return algorithm, None, None
    return (algorithm, {t: per_track[t].hyperparameters for t in info.tracks},
            {t: th.source() for t, th in found.items()})


def _load_base(args, api, challenge: str) -> tuple[dict, dict[str, str]]:
    """(base_algorithm, root files) for a new `--base-algorithm`/`--base-dir` job. Raises
    ConfigError for a bad choice and MainnetError when mainnet cannot be read."""
    if args.base_dir is not None:
        d = Path(args.base_dir)
        try:
            files = read_algorithm_dir(d)
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigError(f"cannot read --base-dir {d}: {e}") from None
        if not files:
            raise ConfigError(f"no .rs/.cu files under {d}")
        algorithm = {"source": "local", "name": d.resolve().name, "id": None}
    else:
        if args.base_algorithm in KNOWN_LEAKING.get(challenge, ()):
            raise ConfigError(f"{args.base_algorithm} carries the hidden_seed leak, so it cannot "
                              "be a base algorithm; use --base-dir with a copy that removes it")
        found = api.find_algorithm(challenge, args.base_algorithm)
        if found is None:
            raise ConfigError(f"no compiled algorithm {args.base_algorithm!r} on mainnet for "
                              f"{challenge}")
        files = api.fetch_algorithm_files(challenge, found[0])
        algorithm = {"source": "mainnet", "name": found[0], "id": found[1]}
    problem = root_problem(challenge, files)
    if problem:
        raise ConfigError(problem)
    return algorithm, files


def cmd_setup(args, ask) -> int:
    root = workspace.find_root_for_write()
    print(f"Workspace: {root}"
          + ("  (the default workspace; talos --workspace DIR setup keeps it elsewhere)"
             if root == workspace.default_root() else ""))
    try:
        backend = ask_choice(ask, "Compute backend", BACKEND_OPTIONS, "modal")
        if backend == "modal":
            extras.require("modal", "the modal backend")
        # "fake" is the in-process test double: it is in KINDS, but setting it up would write a
        # config whose runs never touch an LLM at all.
        kind = ask_choice(ask, "Provider",
                          [(k, PROVIDER_EXPLANATIONS.get(k, "")) for k in KINDS if k != "fake"],
                          "anthropic")
        if kind == "anthropic":
            extras.require("anthropic", "the anthropic provider")
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    model = ask(*_model_prompt(kind))
    api_base = ask("API base URL") if kind == "custom" else None
    api_key = None
    if kind in ("anthropic", "openai", "google", "openrouter", "custom"):
        api_key = ask("API key", secret=True)
    mode = "single-shot"
    if kind in CLI_PROVIDERS:
        try:
            mode = ask_choice(ask, "Mode", MODE_OPTIONS, "single-shot")
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    token_id = token_secret = c3_api_key = None
    if backend == "c3":
        c3_api_key = ask("C3 API key (blank to use your `c3 login` session)", secret=True) or None
    if backend == "modal":
        token_id = ask("Modal token id (create at modal.com/settings/tokens)")
        token_secret = ask("Modal token secret", secret=True)
    local = info = None
    if backend == "local":
        # Before the limits are asked: their defaults and their ceilings are the daemon's.
        try:
            info = check_local()
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 1
        try:
            local = LocalSettings(
                cpus=_ask_number(ask, "CPUs for the local container",
                                 str(info.ncpu or os.cpu_count() or 1), int),
                memory_gib=_ask_number(ask, "Memory for the local container in GiB",
                                       str(default_local_memory_gib(info.mem_total_gib)), int))
        except ConfigError as e:  # three non-numbers, as the run wizard treats it
            print(str(e), file=sys.stderr)
            return 2
        if local.cpus < 1 or local.memory_gib < 1:
            print("CPUs and memory for the local container must each be at least 1",
                  file=sys.stderr)
            return 2
        if ((info.ncpu and local.cpus > info.ncpu)
                or (info.mem_total_gib and local.memory_gib > info.mem_total_gib)):
            # Docker refuses --cpus above its count at every deploy; --memory above its total
            # is accepted, so only this check stops it. On Docker Desktop both are the VM's.
            print(f"Docker has {info.ncpu} CPUs and {info.mem_total_gib} GiB; the container "
                  f"cannot have more (Docker Desktop: raise them under Settings > Resources)",
                  file=sys.stderr)
            return 2
    provider = make_provider(kind, model, api_key=api_key, api_base=api_base)
    err = validate_provider(provider)
    if err:
        print(f"Provider check failed: {err}", file=sys.stderr)
        return 1
    try:
        if backend == "c3":
            env_key = os.environ.get("C3_API_KEY") or None
            check_key = c3_api_key or env_key
            if c3_api_key:
                print("C3: using the hosted MCP endpoint with your API key; no c3 CLI needed.")
            elif env_key:
                print("C3: using the hosted MCP endpoint with the C3_API_KEY in your "
                      "environment; a shell without it falls back to the c3 CLI.")
            else:
                print("C3: using the c3 CLI and its `c3 login` session.")
            check_c3(api_key=check_key)
        elif backend == "local":
            print("Local: jobs run in Docker on this machine. GPU challenges: "
                  + ("available (nvidia runtime found)." if "nvidia" in info.runtimes
                     else "not available (no nvidia runtime)."))
        else:
            deploy_bench(token_id or None, token_secret or None)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    root.mkdir(parents=True, exist_ok=True)
    save(root, Config(provider=kind, model=model, mode=mode, api_base=api_base,
                      backend=backend, local_cpus=local.cpus if local else None,
                      local_memory_gib=local.memory_gib if local else None),
         api_key, c3_api_key=c3_api_key)
    print("Setup complete. Run `talos run` to start a job.")
    return 0


def _ask_number(ask, prompt: str, default: str, cast=float, tries: int = 3):
    """Wizard answers are typed by a human: "abc" or "20 usd" must re-prompt rather than
    traceback out of `float(...)` and throw away everything already answered."""
    for _ in range(tries):
        raw = ask(prompt, default)
        try:
            return cast(raw)
        except (TypeError, ValueError):
            print(f"not a number: {raw!r}", file=sys.stderr)
    raise ConfigError(f"{prompt}: no number given after {tries} tries")


def _challenge_options() -> list[tuple[str, str]]:
    """spec §5.2: GPU challenges are marked, and each shows its status, so the expensive and
    the untested choices are visible before one is picked."""
    return [(name, f"{'GPU' if cs.is_gpu else 'CPU'}, {status_text(cs)}")
            for name, cs in CHALLENGES.items()]


def _codex_refused(cfg: Config, start: str = "mainnet") -> bool:
    """spec §10: say so before a single dollar is spent, not at the first iteration."""
    from talos.agentic import (CODEX_AGENTIC_REFUSAL, TEMPLATE_CODEX_REFUSAL,
                               codex_agentic_refused)
    if codex_agentic_refused(cfg.provider, cfg.mode, start):
        print(TEMPLATE_CODEX_REFUSAL if start == "template" else CODEX_AGENTIC_REFUSAL,
              file=sys.stderr)
        return True
    return False


def _budget_from_args(args) -> Budget:
    b = Budget(usd=args.budget_usd, hours=args.budget_hours, iterations=args.budget_iterations,
               compute_usd=args.budget_compute_usd)
    b.validate()
    return b


# ── the fake end-to-end run (provider "fake"): no Modal, no network ───────────

def _fake_scores(challenge: str, files: dict[str, str], ns):
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [100 + k - 1 for _ in ns.nonces()]  # k=1 is baseline parity


def _fake_script(system: str, user: str) -> str:
    if '"strategy_tag" (one of' in system:
        return ('{"title": "bump k", "description": "raise k by one", '
                '"strategy_tag": "local_search"}')
    if WHOLE_FILE_FORMAT in system:  # a template start: the whole file, at baseline parity
        return "<<<<<<< FILE mod.rs\nfn solve() { let k = 1; }\n>>>>>>> END FILE\n"
    k = int(re.search(r"let k = (\d+);", user).group(1))
    return f"<<<<<<< SEARCH mod.rs\nlet k = {k};\n=======\nlet k = {k + 1};\n>>>>>>> REPLACE\n"


FAKE_MAINNET = types.SimpleNamespace(
    top_algorithm=lambda ch: ("fake_base", "c003_a000", 1),
    find_algorithm=lambda ch, name: (name, "c003_a001", 0) if name == "fake_other" else None,
    top_hyperparameters=lambda algorithm_id, tracks, fuel: {
        t: TrackHyperparameters({"fake_boost": 1}, "fake-benchmark", "0xfake", 100.0)
        for t in tracks},
    fetch_algorithm_files=lambda ch, name: {"mod.rs": "fn solve() { let k = 1; }\n"},
    fetch_template=lambda ch: "pub fn solve_challenge(",
    fetch_template_cu=lambda ch: "// fake template.cu\n",
    # one stub per name on the challenge's list, so nested names (nn/linear.rs) are exercised
    fetch_challenge_docs=lambda ch: {
        name: "Fake challenge: maximise k.\n" if name == "README.md" else f"fake {ch}/{name}\n"
        for name in TEMPLATE_START_FILES[ch]})


def unpriced_reason(provider: str, model: str) -> str | None:
    """Why Talos cannot measure this job's LLM spend, or None when it can. Then llm_usd stays
    0.00: it must be reported as unpriced, never as a measured $0.00, and --budget-usd cannot
    be enforced against it. A provider that bills for tokens is unpriced when Talos has no
    price for the model (every Completion carries cost_usd=None). codex-cli reports no cost in
    either mode, so it is never measured. claude-cli is: `claude -p` reports total_cost_usd."""
    if provider == "codex-cli":
        return "codex-cli does not report what its calls cost"
    if provider not in UNMETERED and estimate_cost(model, Usage(1, 1)) is None:
        return f"model {model} has no price-table entry"
    return None


def unpriced(provider: str, model: str) -> bool:
    return unpriced_reason(provider, model) is not None


def _llm_spend_text(provider: str, model: str, llm_usd: float, why: bool = False) -> str:
    """The LLM spend as the status line (short) and the run summary (`why=True`) show it."""
    reason = unpriced_reason(provider, model)
    if reason is None:
        return f"${llm_usd:.2f}"
    label = "not tracked" if provider == "codex-cli" else "unpriced"
    return f"{label} ({reason})" if why else label


def _status_line(spec: JobSpec, state: JobState, now: float) -> str:
    best = f"{state.best.delta['mean_rel_delta']:+.3%}" if state.best else "n/a"
    hours = effective_budget(spec.budget, state.budget_extensions).hours
    if hours is None:
        left = "∞"
    else:
        left = f"{max(0.0, hours - (now - state.spend.started_at) / 3600):.1f}h"
    llm = _llm_spend_text(spec.provider, spec.model, state.spend.llm_usd)
    return f"best {best} | llm {llm} | compute ≈${state.spend.compute_usd:.2f} | {left} left"


EVENT_LINE_WIDTH = 100
OUTCOME_TEXT = {
    "failed:score": "no improvement",
    "failed:edit": "no candidate: edit failed",
    "failed:compile": "no candidate: did not compile",
    "failed:leak": "no candidate: broke the causality rule",
    "failed:dead_code": "no candidate: new code never called",
    "failed:runtime": "rejected: error rate over the ceiling",
    "failed:validation": "rejected: failed metered validation",
    "accepted": "kept as the chain's next step",
}


def _one_line(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


def _holdout_text(data: dict) -> str:
    if "holdout" not in data:
        return str(data.get("error", ""))
    h = data["holdout"]
    return (f"held-out {h['mean_rel_delta']:+.3%} vs baseline "
            f"(worst track {h['worst_rel_delta']:+.3%})")


def _noise_text(data: dict) -> str | None:
    """The noise figure of a scored event, or None when the line shows none.
    Reads the recorded noise_state only; an event written before issue #69 has no state."""
    state = data.get("noise_state")
    if state == "ok":
        se, ratio, band = data.get("mean_rel_se"), data.get("noise_ratio"), data.get("noise_band")
        if se is None or ratio is None or band is None:
            return None
        return f"±{se:.3%}, {ratio:.1f}x noise, {band}"
    return NOISE_WORDS.get(state)


def _event_line(kind: str, data: dict, width: int = EVENT_LINE_WIDTH) -> str | None:
    """One loop event as one terminal line, or None for an event the terminal does not show.
    timeline.jsonl keeps every event with every field; this is the summary of it."""
    if kind == "stopped":
        return None  # the summary block printed after the loop reports it
    if kind == "baseline":
        text = data["message"]
    elif kind == "baseline_ready":
        text = "baseline ready"
    elif kind == "base_measuring":
        text = f"scoring base algorithm {data['name']}"
    elif kind == "base_measured":
        text = (f"base algorithm {data['name']}: {data['mean_rel_delta']:+.3%} vs baseline, "
                f"worst track {data['worst_rel_delta']:+.3%}")
    elif kind == "hypothesis":
        tag = f" ({data['strategy_tag']})" if data.get("strategy_tag") else ""
        text = f"trying: {data.get('title', '')}{tag}"
    elif kind == "scored":
        runtime = data.get("runtime_ratio")
        noise = _noise_text(data)
        text = (f"scored {data['mean_rel_delta']:+.3%} vs baseline ("
                + (f"{noise}; " if noise else "")
                + f"worst track {data['worst_rel_delta']:+.3%}, errors {data['error_rate']:.1%}"
                + (f", runtime x{runtime:.2f}" if runtime is not None else "")
                + (f", {data['not_run']} not run" if data.get("not_run") else "") + ")")
    elif kind == "compile_failed":
        # a timeline written before the loop recorded `error` has only the output's tail
        error = data.get("error") or first_error(data.get("output") or "")
        text = f"compile failed: {error}" if error else "compile failed"
    elif kind == "edits_rejected":
        text = f"edit rejected: {len(data['paths'])} path(s) outside the algorithm files"
    elif kind == "dead_code":
        text = "new code never called: " + ", ".join(data["names"])
    elif kind == "leak_rejected":
        text = "causality rule broken: " + data["problem"]
    elif kind == "iteration_done":
        text = OUTCOME_TEXT.get(data["outcome"], data["outcome"])
        if data["outcome"] == "improved":
            text = "new best"
        else:
            text += f" ({data['runs_since_improvement']} in a row)"
    elif kind == "accept":
        score = "n/a" if data["score"] is None else f"{data['score']:+.3%}"
        if data["probability"] is None:
            verdict = ("accepted (improved)" if data["accepted"]
                       else f"rejected ({data.get('outcome', 'not accepted')})")
        else:
            verdict = (f"{'accepted' if data['accepted'] else 'rejected'} "
                       f"(p={data['probability']:.2f}, T={data['temperature']:.3f})")
        text = (f"chain {data['chain']} #{data['node']} <- #{data['parent']} "
                f"[{data['family']}] {score} vs mainnet: {verdict}")
    elif kind == "reheat":
        text = (f"chain {data['chain']} reheats from #{data['node']}, "
                f"T={data['temperature']:.3f}")
    elif kind == "reset":
        text = f"switching strategy to: {data['forced_tag']}"
    elif kind == "distilled":
        text = f"lesson: {data['lesson']}"
    elif kind == "confirming":
        text = "confirming on held-out nonces"
    elif kind == "validating":
        text = "validating on TIG's metered runtime"
    elif kind == "validated":
        text = f"validated: {data['mean_rel_delta']:+.3%} vs baseline on the metered runtime"
    elif kind == "demoted":
        text = f"demoted by metered validation: {data['reason']}"
    elif kind == "calibrated":
        text = "native budgets: " + ", ".join(f"{t} {us / 1e6:.1f}s"
                                              for t, us in sorted(data["budgets_us"].items()))
    elif kind == "calibration_fallback":
        text = f"native scoring unavailable, scoring metered: {data['reason']}"
    elif kind == "won":
        text = f"won: {_holdout_text(data)}"
    elif kind == "false_positive":
        text = f"not confirmed: {_holdout_text(data)}"
    elif kind == "llm_cost_unknown":
        text = f"LLM cost not counted: {data['reason']}"
    elif kind == "rate_limited":
        text = f"rate limited, waiting {data['wait_s']}s"
    elif kind == "resumed_pending":
        text = f"reattached to bench job {data['job_id']}"
    elif kind == "discarded_incomplete":
        text = f"discarded {data['discarded']} unfinished iteration(s) from the previous run"
    else:
        # A kind added to the loop later still reaches the terminal.
        text = f"{kind} " + " ".join(f"{k}={v}" for k, v in data.items())
    return _one_line(text, width)


_event_sleep = time.sleep  # the pause between event lines; tests patch this, not time.sleep


def execute_job(spec: JobSpec, store: JobStore, cfg: Config, resume: bool,
                add: dict[str, float] | None = None, *, fakes: tuple | None = None,
                event_delay: float = 0.0) -> int:
    """Runs a new job, or resumes one. `add` holds the budget a resume adds, by dimension
    (only the dimensions the user gave). With provider "fake", `fakes` is the (provider, bench)
    pair to run in place of the `--fake` one (`talos demo` passes its scripted pair).
    `event_delay` pauses that many seconds after each printed event line."""
    if fakes is not None and cfg.provider != "fake":
        raise ValueError("fakes replace the fake provider's pair; the provider must be fake")
    # spec §7.1: refuse before anything is spent when the credential is missing.
    if cfg.provider not in UNMETERED and resolve_api_key(cfg) is None:
        env = ENV_KEYS.get(cfg.provider, "TALOS_CUSTOM_API_KEY")
        print(f"no API key for provider {cfg.provider}: run `talos setup` or set {env}",
              file=sys.stderr)
        return 2
    from talos.package import build_package
    state = store.load() if resume else JobState.fresh(Spend(started_at=time.time()))
    if resume:
        if state.status == "won" or (state.status == "exhausted" and not add):
            print(f"job {spec.job_id} already {state.status}; nothing to resume")
            flag = ADD_FLAGS.get(state.stop_reason) if state.status == "exhausted" else None
            if flag:
                print(f"to continue it, add budget with {flag}")
            pkg = build_package(spec, state, store)
            print(f"Package: {pkg}")
            _print_submission_hint(state, pkg)
            return 1
        if spec.monorepo_ref != MONOREPO_REF or spec.dev_image_tag != DEV_IMAGE_TAG:
            print(f"job {spec.job_id} was created with a different or unrecorded TIG build "
                  f"(monorepo {spec.monorepo_ref}, dev image {spec.dev_image_tag or 'unknown'}); "
                  f"start a new job with dev image {DEV_IMAGE_TAG} to remeasure the baseline "
                  "and candidates together", file=sys.stderr)
            return 2
        void = job_void_reason(spec, state, store.run_dir)
        if void:
            print(f"job {spec.job_id} is void: {void}; start a new job", file=sys.stderr)
            return 2
        cu_path = _template_cu_path(spec, store)
        if cu_path is not None and not cu_path.is_file():
            print(f"cannot resume {spec.job_id}: {cu_path} is missing; a GPU template start "
                  "builds its algorithm.cu from it", file=sys.stderr)
            return 1
        if spec.start == "base" and read_base_root(store.run_dir) is None:
            print(f"cannot resume {spec.job_id}: {store.run_dir / 'base_algorithm'} is missing; "
                  "every iteration builds on those files", file=sys.stderr)
            return 1
        previous = state.status
        # A hard-killed job (left researching, say) resumes without a shift: it was working
        # until it died, as far as its last save shows.
        stopped = previous in ("cancelled", "failed", "paused", "exhausted")
        if stopped:
            # The wall-clock budget measures time the job was WORKING. started_at is shifted
            # forward by the pause (approximated as the time since the last save), or a job
            # resumed after its window has passed exits "exhausted (hours)" at once, for ever.
            paused_s = max(0.0, time.time() - (store.run_dir / "state.json").stat().st_mtime)
            state.spend.started_at += paused_s
        entry = None
        if add:
            # After the shift, before any save: a refused extension leaves the files untouched.
            try:
                entry = extend(spec.budget, state, add, time.time(), previous)
            except ExtensionRefused as e:
                print(f"job {spec.job_id}: {e}", file=sys.stderr)
                return 2
        if stopped:
            state.status, state.stop_reason = "researching", None
            store.save(state)
            store.event("resumed", previous_status=previous, paused_s=round(paused_s, 1))
        elif entry is not None:
            store.save(state)
        if entry is not None:
            budget = effective_budget(spec.budget, state.budget_extensions).to_dict()
            store.event("budget_extended", add=entry["add"], budget=budget)
            added = ", ".join(f"+{v} {d}" for d, v in entry["add"].items() if v is not None)
            print(f"Budget extended: {added}; budget now {budget}")
    else:
        store.save(state)

    from talos.agentic import attach_agentic
    from talos.bench import BenchCancelled, BenchUnavailable, PendingJobStore
    fake = cfg.provider == "fake"

    def _set_pending(d):
        state.pending_job = d
        store.save(state)

    # The backend records the job it has in flight in the run's own state, so a resumed run
    # reattaches to it instead of abandoning one C3 is still billing for.
    pending = PendingJobStore(get=lambda: state.pending_job, set=_set_pending)
    local = gpu_name = None
    if fake:
        from talos.bench import FakeBench
        from talos.providers.fake import FakeProvider
        provider, bench = (fakes if fakes is not None
                           else (FakeProvider(_fake_script), FakeBench(_fake_scores)))
        cache_dir, mainnet = store.run_dir / "baseline_cache", FAKE_MAINNET
        calibration_dir = store.run_dir / "calibration_cache"
    else:
        provider = make_provider(cfg.provider, cfg.model, api_key=resolve_api_key(cfg),
                                 api_base=cfg.api_base)
        c3_api_key = resolve_c3_api_key(cfg) if cfg.backend == "c3" else None
        if c3_api_key:
            # `talos compile` in the agentic sandbox has no secrets.json to read the key from.
            os.environ["C3_API_KEY"] = c3_api_key
        cache_dir, mainnet = BASELINE_CACHE, None
        calibration_dir = CALIBRATION_CACHE
        # Before the baseline, not after: C3 pulls the image at job start, so a missing tag
        # costs a whole job (and its queue wait) to report a pull failure.
        if cfg.backend == "c3" and not image_available(spec.challenge):
            # A run that stops here is over: left at its initial status `talos status` would
            # list it as live for ever, with no reason recorded.
            state.status, state.stop_reason = "failed", "dev image not on GHCR"
            store.save(state)
            print(f"image {c3_image(spec.challenge)} is not on GHCR (or GHCR is "
                  f"unreachable): {IMAGE_HINT}", file=sys.stderr)
            return 1
        if cfg.backend == "local":
            try:
                local = local_settings(cfg)  # asks Docker only if the config lacks the limits
                if CHALLENGES[spec.challenge].is_gpu and not has_gpu_runtime():
                    state.status, state.stop_reason = "failed", "no NVIDIA container runtime"
                    store.save(state)
                    print(f"{spec.challenge} needs a GPU and Docker reports no NVIDIA runtime; "
                          f"install the NVIDIA container toolkit, or run this challenge on the "
                          f"modal or c3 backend", file=sys.stderr)
                    return 1
                # Before the baseline: the first run per challenge pulls the image and does one
                # clean build, and the user should see that happening rather than a silent wait.
                gpu_name = prepare(spec.challenge)
            except C3CommandError as e:  # the daemon is down, for the GPU check or prepare
                state.status, state.stop_reason = "failed", f"docker: {str(e)[:120]}"
                store.save(state)
                print(f"Docker failed while preparing {spec.challenge}: {e}", file=sys.stderr)
                return 1
        bench = make_bench(cfg.backend, store.run_dir, pending, c3_api_key=c3_api_key,
                           local=local)
    from talos.loop import Loop
    # `talos compile` inside the agentic sandbox runs in a worktree with no talos.config.json.
    os.environ["TALOS_BACKEND"] = "modal" if fake else cfg.backend
    try:
        if cfg.backend == "modal" and not fake:
            # Before any probe: a stale app must not be billed. Through the retry wrapper, so a
            # network blip pauses the job (BenchUnavailable) instead of ending it in a traceback.
            bench._with_retry(bench.check_deploy)
        freeze_hardware(bench, spec, state, store, "modal" if fake else cfg.backend)
    except BenchUnavailable as e:
        # Resumable: the GPUs may free up. A `failed` job could not be resumed at all.
        state.status, state.stop_reason = "paused", f"bench: {e}"
        store.save(state)
        print(f"paused: {e}", file=sys.stderr)
        return 1
    except BenchCancelled as e:
        state.status, state.stop_reason = "cancelled", f"stopped; bench job cancelled: {e}"
        store.save(state)
        return 1
    except BudgetExhausted as e:
        state.status, state.stop_reason = "exhausted", e.dimension
        store.save(state)
        print(f"budget exhausted before the hardware probe: {e.dimension}", file=sys.stderr)
        return 1
    hardware = bench_hardware_class(cfg.backend, spec.challenge, local=local, gpu_name=gpu_name,
                                    hardware=state.hardware)

    def on_event(kind, data):
        # loop._n is the iteration the event belongs to; state.iteration only catches up when an
        # iteration finishes, so it labels iteration 1's events "#0".
        n = getattr(loop, "_n", state.iteration)
        prefix = f"[{time.strftime('%H:%M:%S')}] " + (f"#{n} " if n else "")
        width = shutil.get_terminal_size((EVENT_LINE_WIDTH + len(prefix), 24)).columns
        text = _event_line(kind, data, max(40, width - len(prefix)))
        if text is None:
            return
        if kind == "iteration_done":
            text = f"{text} | {_status_line(spec, state, time.time())}"
        if kind == "hypothesis":
            print(flush=True)  # a blank line between iterations
        print(prefix + text, flush=True)
        if event_delay > 0:
            _event_sleep(event_delay)

    challenge_docs = None
    if spec.start == "template":
        # Only the names on the challenge's list: nothing else in the run directory is a doc.
        docs_dir = store.run_dir / "challenge"
        challenge_docs = {f: (docs_dir / f).read_text(encoding="utf-8")
                          for f in TEMPLATE_START_FILES[spec.challenge]
                          if (docs_dir / f).is_file()}
    cu_path = _template_cu_path(spec, store)
    template_cu = (cu_path.read_bytes().decode("utf-8", errors="replace")
                   if cu_path is not None else None)
    loop = Loop(spec, state, store, provider, bench, template_rs="", on_event=on_event,
                challenge_docs=challenge_docs, template_cu=template_cu)
    if cfg.mode == "agentic":
        attach_agentic(loop, cfg.provider, cfg.model)
    previous_sigint = signal.signal(signal.SIGINT,
                                   lambda *_: (loop.request_stop(), bench.request_stop()))
    try:
        if state.baseline is None:
            loop.measure_baseline(cache_dir, hardware, mainnet=mainnet)
        elif mainnet is not None:
            loop.template_rs = mainnet.fetch_template(spec.challenge)
        else:
            from talos.mainnet import fetch_template
            loop.template_rs = fetch_template(spec.challenge)
        loop.calibrate(calibration_dir, hardware)
        final = loop.run()
    except BenchCancelled as e:
        # measure_baseline or calibrate raised it: run() records its own cancelled outcome.
        # Without this the blanket handler below would call a stopped run a failed one.
        state.status, state.stop_reason = "cancelled", f"stopped; bench job cancelled: {e}"
        store.save(state)
        final = state
    except Exception as e:  # noqa: BLE001 - report, package what we have, exit non-zero
        state.status, state.stop_reason = "failed", str(e)
        store.save(state)
        final = state
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    pkg = build_package(spec, final, store)
    print(f"\nStatus: {final.status} ({final.stop_reason})")
    if final.best is not None:
        print(f"Best delta vs baseline: {final.best.delta['mean_rel_delta']:+.3%}")
    elif final.base_leads:
        print(f"Best delta vs baseline: {final.base_delta['mean_rel_delta']:+.3%} "
              "(the base algorithm; no candidate improved on it)")
    else:
        print("Best delta vs baseline: n/a (no candidate)")
    llm = _llm_spend_text(spec.provider, spec.model, final.spend.llm_usd, why=True)
    print(f"LLM spend: {llm}   Compute spend (estimated): ${final.spend.compute_usd:.2f}")
    print(f"Package: {pkg}")
    _print_submission_hint(final, pkg)
    return 0 if final.status == "won" else 1


def _print_submission_hint(state: JobState, pkg: Path) -> None:
    from talos.package import SUBMISSION_DIR
    # build_package decides whether there is a submission folder; follow what it wrote
    if not (pkg / SUBMISSION_DIR).is_dir():
        return
    if state.status == "won":
        print(f"You can find the code to submit at {pkg / SUBMISSION_DIR}, fill out the template "
              "README.md to make it ready to submit to The Innovation Game!")
        return
    # the best of a job that did not win is unconfirmed or a held-out false positive
    print(f"The best candidate is in {pkg / SUBMISSION_DIR}, but it has not beaten the baseline "
          f"on the held-out nonces. Read {pkg / 'README.md'} before submitting it.")


BACKEND_BILLING = {
    "modal": "billed per container second, no fixed per-job overhead",
    "c3": "one batch job per iteration, about 12 minutes of fixed overhead each "
          "(docs/compute-backends.md#c3-timings)",
    "local": "Docker on this machine, no compute charge",
}


def _new_job_id(root: Path, challenge: str, prefix: str = "") -> str:
    stamp = prefix + time.strftime("%Y%m%d-%H%M%S") + f"-{challenge}"
    job_id, n = stamp, 1
    while (root / "runs" / job_id / "job.json").exists():  # two runs in the same second
        n += 1
        job_id = f"{stamp}-{n}"
    return job_id


def _job_summary(*, job_id: str, challenge: str, scope: str, nonces: int, fuel: int,
                 cfg: Config, budget: Budget, start: str, search: str, chains: int,
                 scoring: str, hp_line: str, backend: str, root: Path,
                 base: dict | None = None) -> str:
    cs = CHALLENGES[challenge]
    warning = status_warning(cs)
    caps = []
    if budget.usd is not None:
        caps.append(f"LLM ${budget.usd:.2f}")
    if budget.hours is not None:
        caps.append(f"{budget.hours:g} h")
    if budget.iterations is not None:
        caps.append(f"{budget.iterations} iterations")
    if budget.compute_usd is not None:
        caps.append(f"compute ${budget.compute_usd:.2f}")
    budget_text = ", ".join(caps)
    if unpriced_reason(cfg.provider, cfg.model):
        label = "not tracked" if cfg.provider == "codex-cli" else "unpriced"
        budget_text += f" (LLM spend {label})"
    billing = ("fake (in-process, no charge)" if cfg.provider == "fake"
               else BACKEND_BILLING[backend])
    lines = [
        "Job summary",
        f"  challenge: {challenge} ({'GPU' if cs.is_gpu else 'CPU'}, {status_text(cs)})"
        + (f" - {warning}" if warning else ""),
        f"  scope:     {scope}, {nonces} nonces per track, fuel {fuel}",
        f"  provider:  {cfg.provider} / {cfg.model}, {cfg.mode} mode",
        f"  search:    start {start}, {search}, {chains} chain(s), scoring {scoring}",
        *([f"  base:      {base['source']} algorithm {base['name']}; the baseline stays "
           "mainnet's top algorithm, as a score to beat"] if base else []),
        *(["             its source goes to your LLM provider and compute backend"]
          if base and base["source"] == "local" else []),
        *([f"  baseline:  never {', '.join(sorted(KNOWN_LEAKING[challenge]))} (hidden_seed leak); "
           "compare only with algorithms rebuilt at the pin, never with mainnet scores"]
          if challenge in KNOWN_LEAKING else []),
        f"  {hp_line}",
        f"  budget:    {budget_text}",
        f"  backend:   {backend} - {billing}" if cfg.provider != "fake" else
        f"  backend:   {billing}",
        f"  results:   {root / 'runs' / job_id}",
    ]
    return "\n".join(lines)


def cmd_run(args, ask) -> int:
    # A `talos run ... &` from a non-interactive shell starts with SIGINT ignored, so without
    # this a `kill -INT` before execute_job installs the loop's handler does nothing (#71).
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return _cmd_run(args, ask)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)


def _template_cu_path(spec: JobSpec, store: JobStore) -> Path | None:
    """runs/<job>/challenge/template.cu for a GPU template start, else None."""
    if spec.start != "template" or not CHALLENGES[spec.challenge].is_gpu:
        return None
    return store.run_dir / "challenge" / "template.cu"


def _run_job(spec: JobSpec, store: JobStore, cfg: Config, **kw) -> int:
    """execute_job, where a Ctrl-C after job.json exists but before the loop installs its own
    handler (the image pull, the capacity probe) leaves the job `cancelled` and resumable
    instead of `queued` for ever."""
    try:
        return execute_job(spec, store, cfg, **kw)
    except KeyboardInterrupt:
        try:
            state = store.load()
        except (OSError, ValueError, KeyError):
            state = JobState.fresh(Spend(started_at=time.time()))
        if state.status in ("won", "exhausted", "failed", "cancelled", "paused"):
            print("interrupted", file=sys.stderr)  # already stopped: keep what it recorded
        else:
            state.status, state.stop_reason = "cancelled", "interrupted before the research loop"
            store.save(state)
            print(f"interrupted; continue with talos run --resume {spec.job_id}",
                  file=sys.stderr)
        return 130


def _cmd_run(args, ask) -> int:
    root = workspace.find_root_for_write() if args.fake else workspace.find_root()
    add = {d: v for d in ADD_DIMENSIONS
           if (v := getattr(args, f"add_{d}")) is not None}  # 0 counts as given; extend refuses
    if add and not args.resume:
        d = next(iter(add))
        print(f"{ADD_FLAGS[d]} only applies with --resume; use {BUDGET_FLAGS[d]} for a new job",
              file=sys.stderr)
        return 2
    # spec §11: --fake needs neither a config file nor network, so it must be resolved before
    # `load(root)` is even attempted; a fresh clone with no talos.config.json still runs it.
    if args.fake:
        root.mkdir(parents=True, exist_ok=True)
        cfg = Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                     backend="modal")  # the fake bench is in-process: no backend to pick
    else:
        try:
            cfg = load(root)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    # A backend that reached the config by hand would otherwise raise out of make_bench, which
    # sits past every early return in execute_job: a traceback rather than a message.
    if cfg.backend not in BACKENDS:
        print(f"unknown backend {cfg.backend!r} in talos.config.json; run `talos setup`",
              file=sys.stderr)
        return 2
    # Before a job directory exists or anything is spent. --fake builds a modal Config but never
    # reaches Modal, so it needs neither extra.
    try:
        if (not args.fake and not args.resume and cfg.provider != "fake"
                and cfg.backend == "modal"):
            extras.require("modal", "the modal backend")
        if not args.fake and not args.resume and cfg.provider == "anthropic":
            extras.require("anthropic", "the anthropic provider")
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.resume:
        run_dir = root / "runs" / args.resume
        # A run that was refused before its first save (a bad credential, say) has job.json but
        # no state.json; loading it would traceback instead of reporting a resumable job.
        missing = [f for f in ("job.json", "state.json") if not (run_dir / f).exists()]
        if missing:
            print(f"no job to resume at {run_dir}: missing {', '.join(missing)}", file=sys.stderr)
            return 2
        store = JobStore(run_dir)
        spec = store.read_spec()
        # job.json records how the job was started, and every iteration so far was produced that
        # way. A resume runs the same provider, model and mode whatever the config says today.
        if args.mode and args.mode != spec.mode:
            print(f"job {spec.job_id} was started in {spec.mode} mode; start a new job to "
                  f"change mode", file=sys.stderr)
            return 2
        if args.track is not None and _track_arg(args.track) != spec.track:
            print(f"job {spec.job_id} was started with track {spec.track or 'all'}; start a new "
                  f"job to change track", file=sys.stderr)
            return 2
        if args.hyperparameters is not None:
            print(f"job {spec.job_id} fixed its hyperparameters at start; start a new job to "
                  f"change them", file=sys.stderr)
            return 2
        stored = spec.training[0].count if spec.training else None
        if args.nonces is not None and args.nonces != stored:
            print(f"job {spec.job_id} was started with {stored} nonces per track; start a new "
                  f"job to change nonces", file=sys.stderr)
            return 2
        if args.scoring is not None and args.scoring != spec.scoring:
            print(f"job {spec.job_id} was started with {spec.scoring} scoring; start a new job "
                  f"to change scoring", file=sys.stderr)
            return 2
        for flag, given, stored in (("start", args.start, spec.start),
                                    ("search", args.search, spec.search),
                                    ("chains", args.chains, spec.chains)):
            if given is not None and given != stored:
                print(f"job {spec.job_id} was started with --{flag} {stored}; start a new job "
                      f"to change {flag}", file=sys.stderr)
                return 2
        if args.base_algorithm is not None or args.base_dir is not None:
            print(f"job {spec.job_id} fixed its starting code at start; start a new job to "
                  "change it", file=sys.stderr)
            return 2
        cfg = replace(cfg, provider=spec.provider, model=spec.model, mode=spec.mode)
        # A resume keeps the provider it started with, whatever the config says today.
        try:
            if not args.fake and spec.provider != "fake" and cfg.backend == "modal":
                extras.require("modal", "the modal backend")
            if spec.provider == "anthropic":
                extras.require("anthropic", "the anthropic provider")
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
        if _codex_refused(cfg, spec.start):
            return 2
        return _run_job(spec, store, cfg, resume=True, add=add)
    if args.mode:
        if args.mode == "agentic" and cfg.provider not in CLI_PROVIDERS:
            print(f"--mode agentic needs a CLI provider ({' or '.join(CLI_PROVIDERS)}), "
                  f"not {cfg.provider!r}", file=sys.stderr)
            return 2
        cfg = replace(cfg, mode=args.mode)
    if args.base_algorithm is not None and args.base_dir is not None:
        print("pass either --base-algorithm or --base-dir, not both", file=sys.stderr)
        return 2
    has_base = args.base_algorithm is not None or args.base_dir is not None
    if has_base and args.start is not None:
        print("--base-algorithm and --base-dir choose the start; do not pass --start with them",
              file=sys.stderr)
        return 2
    start, search = "base" if has_base else args.start or "mainnet", args.search or "greedy"
    chains = 1 if args.chains is None else args.chains
    if chains < 1:
        print(f"--chains must be 1 or more, not {chains}", file=sys.stderr)
        return 2
    if _codex_refused(cfg, start):
        return 2
    challenge = args.challenge
    if not challenge:
        try:
            challenge = ask_choice(ask, "Challenge", _challenge_options(), DEFAULT_CHALLENGE)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    if challenge not in CHALLENGES:
        print(f"unknown challenge {challenge!r}", file=sys.stderr)
        return 2
    warning = status_warning(CHALLENGES[challenge])
    if warning:
        print(warning, file=sys.stderr)
    if start == "template" and challenge not in TEMPLATE_START_FILES:
        print(f"--start template supports {', '.join(TEMPLATE_START_FILES)} only, not "
              f"{challenge}", file=sys.stderr)
        return 2
    base = base_files = None
    if has_base:
        try:
            base, base_files = _load_base(args, FAKE_MAINNET if cfg.provider == "fake"
                                          else mainnet_api, challenge)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
    if args.direction and args.direction_file:
        print("pass either --direction or --direction-file, not both", file=sys.stderr)
        return 2
    direction = args.direction
    if args.direction_file:
        df = Path(args.direction_file)
        if not df.exists():
            print(f"direction file not found: {df}", file=sys.stderr)
            return 2
        direction = df.read_text(encoding="utf-8")
    if not direction:
        direction = ask("Direction for the agent (what to explore)")
    metered = cfg.provider not in CLI_PROVIDERS
    try:
        if not args.yes and args.budget_usd is None and args.budget_hours is None \
                and args.budget_iterations is None:
            if metered:
                args.budget_usd = _ask_number(ask, "LLM budget in USD", "20")
            else:
                args.budget_iterations = _ask_number(ask, "Iteration budget", "50", int)
            args.budget_hours = _ask_number(ask, "Wall-clock budget in hours", "4")
        try:
            budget = _budget_from_args(args)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        # spec §5.2: compute spend is always capped, except on the local backend where it is
        # always zero. The default is applied only after the budget has been validated, so it
        # can never stand in for the LLM/time/iteration cap the run needs.
        if budget.compute_usd is None and cfg.backend != "local":
            if args.yes:
                budget = replace(budget, compute_usd=DEFAULT_COMPUTE_USD)
                print(f"No --budget-compute-usd given; capping compute spend at "
                      f"${DEFAULT_COMPUTE_USD:.2f}.")
            else:
                budget = replace(budget, compute_usd=_ask_number(
                    ask, "Compute budget in USD", str(DEFAULT_COMPUTE_USD)))
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    # spec §5.2 step 4: mode for CLI providers, with the cost warning before the prompt.
    if not args.yes and args.mode is None and cfg.provider in CLI_PROVIDERS:
        try:
            mode = ask_choice(ask, "Mode", MODE_OPTIONS, cfg.mode)
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
        cfg = replace(cfg, mode=mode)
        if _codex_refused(cfg, start):
            return 2
    # A dollar cap on an unpriced model is a cap that can never fire: the provider reports no
    # cost, so llm_usd never rises. Refuse rather than run a job with no effective cap at all.
    reason = unpriced_reason(cfg.provider, cfg.model)
    if reason:
        if budget.hours is None and budget.iterations is None:
            print(f"{reason}, so --budget-usd cannot be enforced; add --budget-hours or "
                  f"--budget-iterations", file=sys.stderr)
            return 2
        print(f"warning: {reason}; LLM spend is not measured and --budget-usd is not enforced "
              f"for this run", file=sys.stderr)
    cs = CHALLENGES[challenge]
    if args.fake:
        info = ChallengeInfo(id=cs.id, name=challenge, is_gpu=cs.is_gpu, tracks=["n=1"], max_fuel=1)
    else:
        try:
            info = fetch_challenge_info(challenge)
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
        # The Modal side scores against the static table: its challenge id goes into the runtime
        # settings and its is_gpu picks the CPU or GPU function. If mainnet has moved, every
        # score would be measured under the wrong challenge, so stop before the job exists.
        if info.id != cs.id or info.is_gpu != cs.is_gpu:
            print(f"challenge table drift: mainnet says {info.id}/{info.is_gpu}, Talos has "
                  f"{cs.id}/{cs.is_gpu}; update talos/challenges.py", file=sys.stderr)
            return 1
    track = _track_arg(args.track)
    if args.track is None and not args.yes:
        try:
            track = _track_arg(ask_choice(
                ask, "Track to optimise",
                [("all", "every active track")] + [(t, "") for t in info.tracks], "all"))
        except ConfigError as e:
            print(str(e), file=sys.stderr)
            return 2
    if track is not None and track not in info.tracks:
        print(f"unknown track {track!r} for {challenge}; active tracks: "
              f"{', '.join(info.tracks)}", file=sys.stderr)
        return 2
    choice = args.hyperparameters
    if choice is None:
        if args.yes:
            choice = "mainnet"
        else:
            try:
                choice = ask_choice(ask, "Hyperparameters", HYPERPARAMETER_OPTIONS, "mainnet")
            except ConfigError as e:
                print(str(e), file=sys.stderr)
                return 2
    algorithm = hyperparameters = hp_source = None
    if choice == "mainnet":
        api = FAKE_MAINNET if cfg.provider == "fake" else mainnet_api
        try:
            algorithm, hyperparameters, hp_source = _mainnet_hyperparameters(api, challenge, info)
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
    nonces = args.nonces
    if nonces is None:
        if args.yes:
            nonces = NONCES_PER_TRACK
        else:
            answer = ask("Nonces per track", str(NONCES_PER_TRACK)).strip()
            try:
                nonces = int(answer) if answer else NONCES_PER_TRACK
            except ValueError:
                print(f"not a number: {answer!r}", file=sys.stderr)
                return 2
    if not 1 <= nonces <= HOLDOUT_START:
        # the training set starts at nonce 0 and the held-out set at HOLDOUT_START: a count past
        # that makes the two overlap, and a count of 0 draws nothing to score
        print(f"nonces per track must be between 1 and {HOLDOUT_START}, not {nonces}",
              file=sys.stderr)
        return 2
    docs = None
    templates: dict[str, str] = {}
    if start == "template":
        # Before the job exists: a failed fetch leaves no run directory behind.
        api = FAKE_MAINNET if cfg.provider == "fake" else mainnet_api
        try:
            docs = api.fetch_challenge_docs(challenge)
            templates["template.rs"] = api.fetch_template(challenge)
            if CHALLENGES[challenge].is_gpu:  # the root's algorithm.cu (talos/template_start.py)
                templates["template.cu"] = api.fetch_template_cu(challenge)
        except MainnetError as e:
            print(f"mainnet unreachable: {e}", file=sys.stderr)
            return 1
    job_id = _new_job_id(root, challenge)
    scope = f"track {track} of {len(info.tracks)} tracks" if track else f"{len(info.tracks)} tracks"
    if hyperparameters is None and algorithm is not None:
        hp_line = (f"hyperparameters: none (no mainnet benchmark of {algorithm['name']} "
                   f"at fuel {info.max_fuel})")
    elif hyperparameters is None:
        hp_line = "hyperparameters: none"
    else:
        used = sum(1 for v in hyperparameters.values() if v is not None)
        hp_line = f"hyperparameters: {used}/{len(info.tracks)} tracks from mainnet"
        if base is not None:
            hp_line += ", on the baseline only"
    scoring = args.scoring or "native"
    print(_job_summary(job_id=job_id, challenge=challenge, scope=scope, nonces=nonces,
                       fuel=info.max_fuel, cfg=cfg, budget=budget, start=start, search=search,
                       chains=chains, scoring=scoring, hp_line=hp_line,
                       backend=cfg.backend, root=root, base=base))
    # Nothing below this point has run yet: no image recommendation, no run directory, no bench.
    if not args.yes and not confirm(ask, "Start this job?", True):
        print("Not started.")
        return 0
    if cfg.provider != "fake":
        recommend_image_update(challenge)
    rand_hash = new_rand_hash()
    training, holdout = draw_nonce_sets(info.tracks, rand_hash, training_count=nonces,
                                        holdout_count=nonces)
    spec = JobSpec(job_id=job_id, challenge=challenge, direction=direction, provider=cfg.provider,
                   model=cfg.model, mode=cfg.mode, budget=budget, rand_hash=rand_hash,
                   tracks=info.tracks, training=training, holdout=holdout, fuel=info.max_fuel,
                   created_at=time.time(), monorepo_ref=MONOREPO_REF, challenge_id=info.id,
                   track=track, baseline_algorithm=algorithm, hyperparameters=hyperparameters,
                   hyperparameters_source=hp_source, dev_image_tag=DEV_IMAGE_TAG,
                   scoring=scoring, start=start, base_algorithm=base, search=search,
                   chains=chains,
                   anneal=AnnealSettings().to_dict() if search == "anneal" else None)
    store = JobStore(root / "runs" / job_id)
    store.write_spec(spec)
    if base_files is not None:
        write_base_root(store.run_dir, base_files)
    if docs is not None:
        docs_dir = store.run_dir / "challenge"
        docs_dir.mkdir()
        for name, text in {**docs, **templates}.items():
            (docs_dir / name).parent.mkdir(parents=True, exist_ok=True)  # nn/linear.rs
            (docs_dir / name).write_text(text, encoding="utf-8", newline="\n")
    (store.run_dir / "tacit.md").write_text(user_tacit(direction),
                                            encoding="utf-8", newline="\n")
    return _run_job(spec, store, cfg, resume=False)


def cmd_demo(args, ask) -> int:
    """A scripted job on the real loop, in-process: a compile failure and its fix, two
    iterations with no gain, a false positive and a win (talos/demo.py)."""
    from talos.bench import FakeBench
    from talos.demo import DemoScript, demo_compile_ok, demo_scores, describe_package
    from talos.providers.fake import FakeProvider

    root = workspace.find_root_for_write()
    root.mkdir(parents=True, exist_ok=True)
    challenge = args.challenge
    cfg = Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                 backend="modal")  # the fake bench is in-process: no backend to pick
    cs = CHALLENGES[challenge]
    info = ChallengeInfo(id=cs.id, name=challenge, is_gpu=cs.is_gpu, tracks=["n=1"], max_fuel=1)
    algorithm, hyperparameters, hp_source = _mainnet_hyperparameters(FAKE_MAINNET, challenge,
                                                                     info)
    direction = "demo: raise k"
    rand_hash = new_rand_hash()
    training, holdout = draw_nonce_sets(info.tracks, rand_hash, training_count=NONCES_PER_TRACK,
                                        holdout_count=NONCES_PER_TRACK)
    job_id = _new_job_id(root, challenge, prefix="demo-")
    budget = Budget(usd=None, hours=None, iterations=6, compute_usd=None)
    spec = JobSpec(job_id=job_id, challenge=challenge, direction=direction, provider=cfg.provider,
                   model=cfg.model, mode=cfg.mode, budget=budget, rand_hash=rand_hash,
                   tracks=info.tracks, training=training, holdout=holdout, fuel=info.max_fuel,
                   created_at=time.time(), monorepo_ref=MONOREPO_REF, challenge_id=info.id,
                   track=None, baseline_algorithm=algorithm,
                   hyperparameters=hyperparameters, hyperparameters_source=hp_source,
                   dev_image_tag=DEV_IMAGE_TAG, scoring="native", start="mainnet",
                   search="greedy", chains=1, anneal=None)
    store = JobStore(root / "runs" / job_id)
    store.write_spec(spec)
    (store.run_dir / "tacit.md").write_text(user_tacit(direction),
                                            encoding="utf-8", newline="\n")
    print(f"This is a scripted demo of a {challenge} job: no account, no network and no spend. "
          f"Results go to {root / 'runs' / job_id}.")
    # Zero prices: the status lines then show $0.00, as the line above promises.
    rc = execute_job(spec, store, cfg, resume=False,
                     fakes=(FakeProvider(DemoScript(), usd_per_call=0.0),
                            FakeBench(demo_scores, compile_ok=demo_compile_ok,
                                      usd_per_nonce=0.0)),
                     event_delay=args.delay)
    print("\nWhat is in the package:")
    for line in describe_package(store.run_dir / "package"):
        print(line)
    print(f"See the run as a page: talos report {job_id}")
    print("Next: talos setup, then talos doctor")
    return rc


def compile_backend(args, root: Path) -> str:
    """--backend, then TALOS_BACKEND (exported by execute_job for the agentic sandbox, whose
    worktree has no config file), then the config in `root`, then modal."""
    backend = getattr(args, "backend", None) or os.environ.get("TALOS_BACKEND")
    if not backend:
        try:
            backend = load(root).backend
        except ConfigError:
            return "modal"
    if backend not in BACKENDS:
        raise ConfigError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    return backend


def cmd_deploy(args, ask) -> int:
    """Redeploys the Modal app from this install's source. Reads no config, and keeps the token
    `talos setup` already set."""
    try:
        extras.require("modal", "talos deploy")
        deploy_bench(None, None)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"Deployed talos-bench ({marker_name()})")
    return 0


def cmd_compile(args, ask) -> int:
    """Compile only: no nonce is scored. On C3 and local the job dir is the fixed
    `.talos/compile/<c3|local>/adhoc`, which every run rewrites, and on local the run removes
    every container of that run dir when it ends, so two `talos compile` runs at once in one
    directory are unsupported."""
    if args.challenge not in CHALLENGES:
        print(f"unknown challenge {args.challenge!r}", file=sys.stderr)
        return 2
    d = Path(args.dir)
    files = read_algorithm_dir(d)
    # An empty file map builds nothing on the far side: refuse here rather than pay for a
    # container that can only report a mystery failure.
    if not files:
        print(f"no .rs/.cu files under {d}", file=sys.stderr)
        return 2
    from talos.bench import EvalRequest, PendingJobStore
    try:
        backend = compile_backend(args, workspace.find_root())
        if backend == "modal":
            extras.require("modal", "the modal backend")
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        cfg = load(workspace.find_root())
    except ConfigError:
        cfg = None  # the agentic sandbox: the key comes from C3_API_KEY
    local = None
    if backend == "local":
        try:
            prepare(args.challenge)
            local = local_settings(cfg)
        except C3CommandError as e:
            print(f"Docker failed while preparing {args.challenge}: {e}", file=sys.stderr)
            return 1
    run_dir = workspace.find_root() / ".talos" / "compile"
    bench = make_bench(backend, run_dir, PendingJobStore.memory(),
                       c3_api_key=resolve_c3_api_key(cfg) if backend == "c3" else None,
                       local=local)
    from talos.bench import BenchUnavailable
    try:
        if backend == "modal":
            bench._with_retry(bench.check_deploy)
        # The job's GPU when run from the agentic sandbox (execute_job exports it); a probe of
        # its own for an ad hoc compile, which has no job to match.
        bench.select_hardware(args.challenge, chosen=os.environ.get("TALOS_HARDWARE") or None)
    except BenchUnavailable as e:
        print(str(e), file=sys.stderr)
        return 1
    try:
        r = bench.evaluate(EvalRequest(args.challenge, files, [], [], 0, None,
                                       CHALLENGES[args.challenge].beat,
                                       mode="native" if args.native else "metered")).compile
    finally:
        if backend == "local":
            remove_compile_containers(run_dir)
    print(r.output[-4000:])
    return 0 if r.ok else 1


def remove_compile_containers(run_dir: Path) -> None:
    """The local containers of one `talos compile`, once its result is in or it was cut short.
    The agentic sandbox's run key is new every iteration, so nothing else would remove them."""
    try:
        DockerTransport().remove_run(run_dir)
    except C3CommandError as e:
        print(f"could not remove the compile containers: {e}", file=sys.stderr)


def parse_which(text: str) -> str | int:
    """--iteration: "best" or a positive iteration number."""
    if text == "best":
        return text
    # isascii: "²".isdigit() is True and int() rejects it.
    if text.isascii() and text.isdigit() and int(text) >= 1:
        return int(text)
    raise argparse.ArgumentTypeError("expected a positive iteration number or 'best'")


def _print_json(obj: dict) -> None:
    print(json.dumps(obj, indent=1))


def _utf8_stdout() -> None:
    """Write stdout as UTF-8. On Windows a piped or redirected stdout uses the ANSI code page
    (cp1252), which cannot encode ∞, LLM-written titles or source diffs."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    if encoding != "utf8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def cmd_status(args, ask) -> int:
    """Lists the jobs under runs/, or one job's detail. Reads runs/ only."""
    from talos import inspect_run

    root = workspace.find_root() / "runs"
    if args.job_id is not None:
        try:
            detail = inspect_run.job_detail(inspect_run.load_run(root, args.job_id), time.time())
        except inspect_run.InspectError as e:
            print(e, file=sys.stderr)
            return 1
        if args.json:
            _print_json(detail)
        else:
            print(inspect_run.format_job_detail(detail))
        return 0
    rows = inspect_run.status_rows(root)
    if args.json:
        _print_json({"schema_version": inspect_run.SCHEMA_VERSION, "jobs": rows})
        return 0
    if not rows:
        print(f"no jobs under {root}")
        return 0
    print(inspect_run.format_status_table(rows))
    for row in rows:
        if row["error"] is not None:
            print(f"{row['job_id']}: {row['error']}", file=sys.stderr)
    return 0


def cmd_show(args, ask) -> int:
    """One iteration's hypothesis, outcome, scores and files. Reads runs/ only."""
    from talos import inspect_run

    try:
        run = inspect_run.load_run(workspace.find_root() / "runs", args.job_id)
        detail = inspect_run.iteration_detail(run, args.iteration)
    except inspect_run.InspectError as e:
        print(e, file=sys.stderr)
        return 1
    if args.json:
        _print_json(detail)
    else:
        print(inspect_run.format_iteration(detail))
    return 0


def cmd_diff(args, ask) -> int:
    """One iteration's unified diff against the baseline or its parent. Reads runs/ only."""
    from talos import inspect_run

    try:
        run = inspect_run.load_run(workspace.find_root() / "runs", args.job_id)
        text = inspect_run.iteration_diff(run, args.iteration, args.against)
    except inspect_run.InspectError as e:
        print(e, file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


def cmd_report(args, ask) -> int:
    """Writes runs/<job_id>/package/report.html. Needs no credentials or config."""
    run_dir = workspace.find_root() / "runs" / args.job_id
    # Checked before JobStore, whose constructor creates the directory.
    if not run_dir.is_dir():
        print(f"no job {args.job_id} under runs/", file=sys.stderr)
        return 1
    if not (run_dir / "state.json").is_file():
        print(f"job {args.job_id} has no state.json", file=sys.stderr)
        return 1
    from talos.report import write_report

    path = write_report(JobStore(run_dir))
    print(f"Report: {path}")
    return 0


def recommend_image_update(challenge: str) -> None:
    """New-job advisory only. A registry outage must not prevent an otherwise valid run."""
    try:
        update = image_updates.check_update(challenge)
    except image_updates.UpdateCheckError:
        return
    if update.available:
        print(f"TIG dev image update available for {challenge}: {update.current} -> "
              f"{update.latest}. Run `talos check-updates` for upgrade guidance. "
              f"This job will use the pinned image {update.current}.", file=sys.stderr)


def cmd_guide(args, ask) -> int:
    """Print the packaged agent guide (#71); read as package data so a wheel install works."""
    from importlib import resources
    print(resources.files("talos.data").joinpath("agent_guide.md").read_text(encoding="utf-8"),
          end="")
    return 0


def cmd_doctor(args, ask) -> int:
    from talos import doctor
    return doctor.run(workspace.find_root(), full=args.full, challenge=args.challenge, yes=args.yes,
                      json_out=args.json, ask=ask)


def cmd_check_updates(args, ask) -> int:
    """Check published versions without Docker, compute credentials or a configured project."""
    from concurrent.futures import ThreadPoolExecutor

    names = [args.challenge] if args.challenge else list(CHALLENGES)

    def check(name):
        try:
            return image_updates.check_update(name)
        except image_updates.UpdateCheckError:
            return None

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        updates = list(pool.map(check, names))
    newer = False
    for name, update in zip(names, updates):
        if update is None:
            print(f"{name}: unable to check GHCR; update status unknown", file=sys.stderr)
            continue
        status = "update available" if update.available else "no newer release"
        print(f"{name}: configured {update.current}, newest published {update.latest} ({status})")
        newer |= update.available
    if newer:
        print("To upgrade, review the TIG release and its matching source revision, then update "
              "DEV_IMAGE_TAG and MONOREPO_REF in talos/challenges.py. The image tag must be "
              "available for all challenges and required architectures. Run `make check`, "
              "redeploy Modal with `talos deploy` if used, then run `talos doctor --full`. "
              "Start new jobs after upgrading. "
              "A newer published tag does not establish source compatibility.")
    return 1 if any(update is None for update in updates) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="talos",
        description="Talos researches better algorithms for a TIG challenge: it asks an LLM for "
                    "edits, scores each candidate on TIG's benchmark and keeps the "
                    "improvements.")
    p.add_argument("--version", action="version", version=f"talos {__version__}")
    p.add_argument("--workspace", metavar="DIR",
                   help="the Talos workspace (default: nearest talos.config.json above the "
                        "current directory, else ~/.talos/workspace)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")
    sub.add_parser("setup", help="choose a backend and LLM provider and save the settings",
                   description="Interactive setup: pick a compute backend (modal, c3 or local) "
                               "and an LLM provider, then save them to the Talos config.")
    sub.add_parser("deploy", help="redeploy the Modal app after upgrading Talos",
                   description="Deploy the talos-bench app to your Modal account from this "
                               "install's source. Run it after upgrading Talos; a run stops "
                               "before spending when the deployed app is older.")
    r = sub.add_parser(
        "run", help="research a challenge, or resume a job",
        description="Start a research job on one challenge, or resume an earlier job with "
                    "--resume. Options left out are asked for interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  talos run --challenge knapsack --direction \"...\" --budget-usd 20 "
               "--budget-hours 4\n"
               "  talos run --resume <job_id> --add-iterations 10\n"
               "  talos run --challenge knapsack --direction-file notes.md "
               "--budget-iterations 30 --yes\n")
    what = r.add_argument_group("what to research")
    what.add_argument("--challenge", choices=list(CHALLENGES), help="challenge to research")
    what.add_argument("--direction", help="what to try, in your own words")
    what.add_argument("--direction-file", help="read the direction from this file")
    what.add_argument("--track", help="one active track to optimise, or all (the default)")
    what.add_argument("--start", choices=["mainnet", "template"],
                      help="start from mainnet's top algorithm (the default) or from the empty "
                           "template")
    what.add_argument("--base-algorithm", metavar="NAME",
                      help="start from this mainnet algorithm; the baseline to beat stays "
                           "mainnet's top algorithm")
    what.add_argument("--base-dir", metavar="DIR",
                      help="start from the algorithm in this local directory; the baseline to "
                           "beat stays mainnet's top algorithm")
    what.add_argument("--hyperparameters", choices=["mainnet", "none"],
                      help="per-track hyperparameters from the baseline algorithm's best "
                           "mainnet benchmark (mainnet, the default) or none")
    budget = r.add_argument_group("budget")
    budget.add_argument("--budget-usd", type=float, help="cap on LLM spend in USD")
    budget.add_argument("--budget-hours", type=float, help="cap on wall-clock hours")
    budget.add_argument("--budget-iterations", type=int, help="cap on research iterations")
    budget.add_argument("--budget-compute-usd", type=float,
                        help="cap on backend compute spend in USD")
    search = r.add_argument_group("search")
    search.add_argument("--mode", choices=["single-shot", "agentic"],
                        help="how the LLM edits: one reply per iteration (single-shot) or an "
                             "agent loop (agentic)")
    search.add_argument("--scoring", choices=["metered", "native"],
                        help="research scoring: native (fast, validated on metered; the "
                             "default) or metered (TIG's runtime for every candidate)")
    search.add_argument("--search", choices=["greedy", "anneal"],
                        help="keep only improvements (greedy, the default) or anneal")
    search.add_argument("--chains", type=int, help="independent search chains (default 1)")
    search.add_argument("--nonces", type=int,
                        help=f"nonces per track in each of the training and held-out sets "
                             f"(default {NONCES_PER_TRACK})")
    resume = r.add_argument_group("resume")
    resume.add_argument("--resume", metavar="JOB_ID", help="resume this job from runs/")
    # Each adds to one cap of a resumed job's budget (talos/extend.py); refused without --resume.
    resume.add_argument("--add-iterations", type=int,
                        help="add this many iterations to the resumed job's cap")
    resume.add_argument("--add-hours", type=float,
                        help="add this many hours to the resumed job's cap")
    resume.add_argument("--add-usd", type=float,
                        help="add this many USD to the resumed job's LLM spend cap")
    resume.add_argument("--add-compute-usd", type=float,
                        help="add this many USD to the resumed job's compute cap")
    r.add_argument("--yes", action="store_true",
                   help="skip every question and the confirmation; take defaults (compute cap "
                        f"${DEFAULT_COMPUTE_USD:.0f} on hosted backends)")
    r.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    d = sub.add_parser("demo", help="watch a scripted job: no account, network or spend",
                       description="Run a scripted job in-process on the real research loop. "
                                   "It shows a compile failure and its fix, iterations with "
                                   "no gain, a false positive on the held-out nonces and a "
                                   "win, then the package a real job writes.")
    d.add_argument("--challenge", choices=list(CHALLENGES), default=DEFAULT_CHALLENGE,
                   help=f"challenge to label the demo with (default {DEFAULT_CHALLENGE})")
    d.add_argument("--delay", type=float, default=0.4,
                   help="seconds to pause after each event line (default 0.4; 0 disables)")
    c = sub.add_parser("compile", help="compile an algorithm directory on the backend",
                       description="Compile a challenge algorithm and print the result. "
                                   "Agents call this from inside a run.")
    c.add_argument("--challenge", required=True, choices=list(CHALLENGES),
                   help="challenge the algorithm belongs to")
    c.add_argument("--dir", default="algorithm",
                   help="directory holding mod.rs (default: algorithm)")
    c.add_argument("--backend", choices=list(BACKENDS),
                   help="compute backend (default: from the Talos config)")
    c.add_argument("--native", action="store_true",
                   help="build the unmetered talos-native runner, as a --scoring native job "
                        "scores it")
    stat = sub.add_parser("status", help="list jobs under runs/, or show one job's detail",
                          description="Without a job id, one row per job under runs/: status, "
                                      "iterations, best delta, spend and the last event. With a "
                                      "job id, that job's settings, best candidate, budget left "
                                      "and current activity. Reads runs/ only.")
    stat.add_argument("job_id", nargs="?", help="job id, the directory name under runs/")
    stat.add_argument("--json", action="store_true",
                      help="print one JSON object (schema: docs/json-output.md)")
    show = sub.add_parser("show", help="show one iteration's hypothesis, scores and files",
                          description="Show one iteration of a job: its hypothesis, outcome, "
                                      "training and held-out scores, and the paths of its "
                                      "files. Reads runs/ only.")
    show.add_argument("job_id", help="job id, the directory name under runs/")
    show.add_argument("--iteration", type=parse_which, default="best", metavar="N|best",
                      help="the iteration to show (default: best)")
    show.add_argument("--json", action="store_true",
                      help="print one JSON object (schema: docs/json-output.md)")
    diff = sub.add_parser("diff", help="print one iteration's diff as a unified patch",
                          description="Print the unified diff of one iteration's files against "
                                      "the baseline (or template) or against its parent in the "
                                      "search tree. Reads runs/ only.")
    diff.add_argument("job_id", help="job id, the directory name under runs/")
    diff.add_argument("--iteration", type=parse_which, default="best", metavar="N|best",
                      help="the iteration to diff (default: best)")
    diff.add_argument("--against", choices=("parent", "baseline"), default="baseline",
                      help="what to diff against: the root the job started from (baseline, or "
                           "the template for --start template) or the iteration's parent "
                           "(default: baseline)")
    rep = sub.add_parser("report", help="write runs/<job_id>/package/report.html from the run's "
                                        "files",
                         description="Write runs/<job_id>/package/report.html from the files "
                                     "of a finished or interrupted run.")
    rep.add_argument("job_id", help="job id, the directory name under runs/")
    updates = sub.add_parser("check-updates", help="check GHCR for newer TIG dev images",
                             description="Compare the configured TIG dev image tag with the "
                                         "newest one published on GHCR.")
    updates.add_argument("--challenge", choices=list(CHALLENGES),
                         help="check one challenge (default: all)")
    doc = sub.add_parser(
        "doctor", help="check the config, credentials, backend, mainnet and GHCR",
        description="Run free health checks: the config, the LLM credential and one test "
                    "call, the compute backend, mainnet and the GHCR dev images. --full also "
                    "compiles and scores one baseline on the backend, after showing its cost "
                    "and asking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  talos doctor\n"
               "  talos doctor --full --challenge knapsack\n"
               "  talos doctor --json\n")
    doc.add_argument("--full", action="store_true",
                     help="also compile and score one baseline on the backend (spends "
                          "compute; asks first)")
    doc.add_argument("--challenge", choices=list(CHALLENGES), default=DEFAULT_CHALLENGE,
                     help=f"challenge for --full (default {DEFAULT_CHALLENGE})")
    doc.add_argument("--yes", action="store_true",
                     help="run --full without asking; with --json, --full needs it")
    doc.add_argument("--json", action="store_true",
                     help="print one JSON document instead of text lines")
    sub.add_parser("guide", help="print the guide for coding agents driving Talos",
                   description="Print the guide a coding agent follows to run Talos for a "
                               "user: what spends money, how to start, watch and stop a job, "
                               "and how to report back. Needs no config or network.")
    return p


COMMANDS = {"setup": cmd_setup, "deploy": cmd_deploy, "run": cmd_run, "compile": cmd_compile,
            "status": cmd_status, "show": cmd_show, "diff": cmd_diff, "report": cmd_report,
            "check-updates": cmd_check_updates, "demo": cmd_demo, "doctor": cmd_doctor,
            "guide": cmd_guide}


def main(argv=None, ask=default_ask) -> int:
    args = build_parser().parse_args(argv)
    if args.workspace:
        os.environ[workspace.ENV_VAR] = str(Path(args.workspace).expanduser().resolve())
    # Every command, not only the inspect ones: `talos run` prints ≈, ∞ and LLM-written
    # titles on every iteration (issue #85).
    _utf8_stdout()
    try:
        return COMMANDS[args.cmd](args, ask)
    except workspace.WorkspaceError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
