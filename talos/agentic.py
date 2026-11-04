"""Agentic mode: one headless claude or codex call per iteration inside a sandboxed worktree.
The agent edits algorithm files, may run `talos compile`, and must write .talos/hypothesis.json.
The loop reads the files back and owns compile, score and publish as in single-shot mode."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from talos.executables import argv0
from talos.providers.claude_cli import NO_COST, session_cost
from talos.prompts import (PromptContext, STRATEGY_TAGS, _rust_rules, causality_rules,
                           challenge_block,
                           focus_sentence, gap_block, hyperparameters_block, recall_block,
                           template_cu_block)


class AgenticError(ValueError):
    """A ValueError so Loop.iterate records it as a failed iteration instead of crashing the run."""


class AgentTimeout(AgenticError):
    """The CLI was killed at its timeout. It ran, so it spent tokens, but it printed no result."""


# codex ignores claude-cli's settings file, and its own `--sandbox workspace-write` restricts writes
# only: the agent can execute arbitrary commands on this machine and read any file it can reach.
# claude-cli enforces the allow/deny list in sandbox_settings; codex cannot, so it is opt-in.
CODEX_AGENTIC_ENV = "TALOS_ALLOW_CODEX_AGENTIC"
CODEX_AGENTIC_REFUSAL = (
    "codex agentic mode runs agent-authored commands on this machine and cannot restrict reads; "
    f"set {CODEX_AGENTIC_ENV}=1 to accept that")


# A template start has no mainnet source to withhold from the agent's shell, and its prompts are
# the only barrier; codex cannot restrict reads, so no opt-in applies.
TEMPLATE_CODEX_REFUSAL = (
    "codex agentic mode is not available with --start template: codex cannot restrict what the "
    "agent reads, and a template start must keep the baseline's source away from it")


def codex_agentic_refused(provider_kind: str, mode: str, start: str = "mainnet") -> bool:
    if mode != "agentic" or provider_kind != "codex-cli":
        return False
    return start == "template" or os.environ.get(CODEX_AGENTIC_ENV) != "1"


# Passed with --settings. Not .claude/settings.json: Claude Code also reads that path as project
# settings, ignores its allow list in an untrusted directory, and warns about it on every run.
# Outside every Edit rule, so the agent cannot rewrite its own permissions.
SETTINGS_REL = Path(".talos") / "claude-settings.json"


def rule_root(path: str) -> str:
    """`path` as the root of an absolute Claude Code permission rule: `//abs/path`, with a
    Windows path in the POSIX form Claude Code matches against (C:\\Users\\a -> //c/Users/a)."""
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        p = "/" + p[0].lower() + p[2:]
    return "/" + p


def sandbox_settings(worktree: Path) -> dict:
    """Deny is evaluated before allow in Claude Code, so never deny a glob that covers an
    allowed path. Unlisted tools are refused by `dontAsk` mode rather than prompted for.

    File rules are absolute. Claude Code resolves a relative rule against the shell's current
    directory, which moves with every `cd`; agents habitually `cd <wt>/algorithm` before editing,
    after which `Edit(algorithm/**)` matched nothing (34 of 37 such sessions on tig-adi had Edit
    denied, 2026-09-26..29). A symlinked worktree (macOS temp dirs) gets rules for both paths.
    Edit rules cover Write, so it needs no rule of its own. CLI 2.1.284 has no Glob or Grep
    tool (agents search with Bash grep/ls), so there are no rules for them either."""
    roots = list(dict.fromkeys([rule_root(str(worktree)), rule_root(str(worktree.resolve()))]))
    read_scope = ["algorithm/**", "CHALLENGE.md", "tacit.md", "AGENTS.md",
                  ".talos/hypothesis.json"]
    allow = [f"Read({r}/{p})" for r in roots for p in read_scope]
    allow += [f"Edit({r}/{p})" for r in roots for p in ("algorithm/**", ".talos/hypothesis.json")]
    allow.append("Bash(talos compile:*)")
    return {"permissions": {
        "allow": allow,
        "deny": ["WebFetch", "WebSearch", "Bash(curl:*)", "Bash(wget:*)", "Bash(git:*)",
                 "Bash(ssh:*)", "Bash(python:*)", "Bash(pip:*)", "Bash(nc:*)", "Bash(rm:*)"],
        "defaultMode": "dontAsk"}}


def claude_md(ctx: PromptContext) -> str:
    scope = f'on track "{ctx.track}"' if ctx.track else "on every active track"
    focus = (focus_sentence(ctx) + "\n\n") if ctx.track else ""
    hp = hyperparameters_block(ctx)
    hp = (hp + "\n\n") if hp else ""
    compile_cmd = (f"talos compile --challenge {ctx.challenge} --dir algorithm"
                   + (" --native" if ctx.scoring == "native" else ""))
    role = ("writing a Rust solver from the empty template for"
            if ctx.start == "template" else "improving a Rust solver for")
    return f"""# Talos agentic iteration: {ctx.challenge}

You are {role} the TIG challenge "{ctx.challenge}". Beat the mainnet
baseline "{ctx.baseline_name}" on TIG's benchmark (higher verifier quality per nonce under a
fixed fuel budget {scope}).

{gap_block(ctx)}

{focus}{hp}Rules:
- Edit ONLY files under `algorithm/`. Do not create new files. Do not touch anything else.
- You may run `{compile_cmd}` to check the build.
  Nothing else may be executed. There is no network.
- Make ONE focused change per iteration that implements a single hypothesis.
- Before you stop, EDIT the existing file `.talos/hypothesis.json` (it starts as `{{}}`) so it
  holds keys "title", "description", "strategy_tag" (one of: {", ".join(STRATEGY_TAGS)}) and
  "design_family" (a short label, at most 80 characters, for the kind of design this solver is).
  Use the Edit tool; creating new files is not permitted.

{causality_rules(ctx.challenge)}The algorithm files are: {", ".join(f"algorithm/{name}" for name in sorted(ctx.files))}.

Direction from the user:
{ctx.direction}

Tacit knowledge so far is in `tacit.md`. The solver contract is in `CHALLENGE.md`.

{_rust_rules()}
"""


def prepare_worktree(ctx: PromptContext, parent: Path | None = None) -> Path:
    """A fresh temporary directory, deliberately NOT under runs/<job>/: the agent gets a shell in
    here (codex) or a sandboxed Read scope rooted here (claude), and a worktree inside the run
    directory would put job.json -- the only file holding the job's rand_hash -- one `..` away.
    The caller owns the returned path and must remove it (attach_agentic does, one iteration
    later, so a failed iteration can still be inspected)."""
    wt = Path(tempfile.mkdtemp(prefix="talos-agentic-", dir=parent))
    (wt / "algorithm").mkdir(parents=True)
    (wt / ".talos").mkdir()
    # pre-created so the agent fills it with Edit, the tool its allow rule names
    (wt / ".talos" / "hypothesis.json").write_text("{}\n", encoding="utf-8", newline="\n")
    for name, text in ctx.files.items():
        p = wt / "algorithm" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    challenge_md = (f"# {ctx.challenge} solver contract (template.rs)\n\n"
                    f"```rust\n{ctx.template_rs}\n```\n")
    cu = template_cu_block(ctx)
    if cu:
        challenge_md += "\n" + cu.rstrip("\n") + "\n"
    docs = challenge_block(ctx)
    if docs:
        challenge_md += "\n" + docs + "\n"
    (wt / "CHALLENGE.md").write_text(challenge_md, encoding="utf-8", newline="\n")
    (wt / "tacit.md").write_text(ctx.tacit, encoding="utf-8", newline="\n")
    md = claude_md(ctx)
    (wt / "CLAUDE.md").write_text(md, encoding="utf-8", newline="\n")
    (wt / "AGENTS.md").write_text(md, encoding="utf-8", newline="\n")
    (wt / SETTINGS_REL).write_text(json.dumps(sandbox_settings(wt), indent=1),
                                   encoding="utf-8", newline="\n")
    return wt


def read_back(worktree: Path, ctx: PromptContext) -> tuple[dict, dict[str, str]]:
    """spec §9: an edit outside the algorithm files fails the iteration in both modes. For codex,
    which ignores claude-cli's settings file, this scope check is the ONLY enforcement of that
    rule."""
    hp = worktree / ".talos" / "hypothesis.json"
    if not hp.exists() or hp.read_text(encoding="utf-8").strip() in ("", "{}"):
        raise AgenticError("agent did not fill in .talos/hypothesis.json")
    try:
        h = json.loads(hp.read_text(encoding="utf-8"))
        tag = h.get("strategy_tag")
        hypothesis = {"title": str(h["title"])[:200], "description": str(h["description"])[:4000],
                      "strategy_tag": tag if tag in STRATEGY_TAGS else "hybrid",
                      "design_family": str(h.get("design_family") or "unlabelled").strip()[:80]
                      or "unlabelled"}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise AgenticError(f"bad hypothesis.json: {e}") from None
    algo_dir = worktree / "algorithm"
    on_disk = {p.relative_to(algo_dir).as_posix() for p in algo_dir.rglob("*") if p.is_file()}
    extra = on_disk - set(ctx.files)
    if extra:
        raise AgenticError(f"agent created files outside the algorithm set: {sorted(extra)}")
    missing = set(ctx.files) - on_disk
    if missing:
        raise AgenticError(f"agent deleted algorithm files: {sorted(missing)}")
    files = {name: (algo_dir / name).read_text(encoding="utf-8") for name in ctx.files}
    if files == ctx.files:
        raise AgenticError("agent changed no algorithm file")
    return hypothesis, files


# spec §7.6/§10: `talos compile` inside the sandbox runs agent-controlled build code, and
# anything left in the environment is exfiltratable through it. So the child gets an allowlist,
# not the scrubbed-copy Prometheus uses (Prometheus's `_scrubbed_env` removes only known secret
# keys/prefixes and passes the rest of os.environ through). We keep: PATH (interpreter's bin dir
# prepended, as before), the login/locale/terminal basics a CLI needs to run at all (HOME, USER,
# LOGNAME, SHELL, TERM, LANG, LC_*, TZ, TMPDIR, XDG_*), and proxy variables — Prometheus's comment
# on `_scrubbed_env` notes it deliberately lets proxy vars pass through so the claude/codex login
# session still works on networks that require an outbound proxy. No provider keys, no Modal
# tokens, no anything else.
# CLAUDE_CONFIG_DIR / CODEX_HOME are where each CLI keeps its logged-in session; dropping them
# makes the child look logged out on any machine that relocates them away from $HOME.
_AGENT_ENV_ALLOWLIST = frozenset({
    "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "CLAUDE_CONFIG_DIR", "CODEX_HOME",
    # the sandbox runs `talos compile`, which has no talos.config.json to read the backend from,
    # and no state.json to read the hardware the job was frozen to
    "TALOS_BACKEND", "TALOS_HARDWARE",
    # ...nor .talos/secrets.json to read a C3 API key from. The key is as reachable as the Modal
    # token in ~/.modal.toml or a `c3 login` session under HOME: claude-cli's permissions let the
    # agent run `talos compile` and read only the worktree; opt-in codex can read either.
    "C3_API_KEY",
    # Windows's equivalents of the basics above. Node, which claude and codex run on, fails to
    # start without SYSTEMROOT; both CLIs keep their login under USERPROFILE/APPDATA; an npm
    # `.cmd` wrapper needs COMSPEC and PATHEXT. None of these is set on Linux or macOS.
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE", "USERNAME",
    "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
})

_TRANSCRIPT_CAP = 200_000


def _agent_env(worktree: Path) -> dict[str, str]:
    """The agent runs `talos compile`; make sure the interpreter that runs Talos is first on PATH so
    the console script resolves even when the venv is not activated in the agent's shell."""
    env = {k: v for k, v in os.environ.items()
           if k in _AGENT_ENV_ALLOWLIST or k.startswith("XDG_")}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    # The sandbox's `talos compile` must not fall through to the user's workspace and read their
    # config and secrets; it behaves as it did before workspaces: no config, state in the worktree.
    env["TALOS_WORKSPACE"] = str(worktree)
    return env


def _run_agent(cmd: list[str], wt: Path, prompt: str, timeout_s: int,
               run) -> subprocess.CompletedProcess:
    """The prompt goes on stdin, not argv: on Windows an npm-installed CLI is a `.cmd` wrapper
    that cmd.exe runs, and cmd.exe cuts an argument at its first newline. The exit code is the
    caller's to check, after it has read the session's cost: a failed session spent tokens too."""
    try:
        r = run(cmd, cwd=wt, input=prompt, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_s, env=_agent_env(wt))
    except FileNotFoundError:
        raise AgenticError(f"{cmd[0]} CLI not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise AgentTimeout(f"{cmd[0]} timed out after {timeout_s}s") from None
    (wt / ".talos" / "agent_stdout.txt").write_text((r.stdout or "")[-_TRANSCRIPT_CAP:],
                                                    encoding="utf-8", newline="\n")
    (wt / ".talos" / "agent_stderr.txt").write_text((r.stderr or "")[-_TRANSCRIPT_CAP:],
                                                    encoding="utf-8", newline="\n")
    return r


def _run_claude(wt: Path, model: str, prompt: str, timeout_s: int,
                run) -> subprocess.CompletedProcess:
    # the JSON result carries the session's cost and lists the tool calls the CLI refused; the
    # text format has neither
    return _run_agent([argv0("claude"), "-p", "--model", model, "--output-format", "json",
                       "--settings", str(wt / SETTINGS_REL),
                       "--permission-mode", "dontAsk"], wt, prompt, timeout_s, run)


def _run_codex(wt: Path, model: str, prompt: str, timeout_s: int,
               run) -> subprocess.CompletedProcess:
    # "-" makes codex read the prompt from stdin
    return _run_agent([argv0("codex"), "exec", "-m", model, "--sandbox", "workspace-write",
                       "--skip-git-repo-check", "-"], wt, prompt, timeout_s, run)


def _charge_session(loop, r: subprocess.CompletedProcess) -> None:
    """Charge one claude session to the job's LLM spend, or record that its cost is unknown so
    an uncounted session never reads as a free one. codex reports no cost and is not charged:
    the CLI reports such a run as "not tracked" (talos/cli.py::unpriced_reason)."""
    cost = session_cost(r.stdout or "")
    if cost is None:
        loop._event("llm_cost_unknown", reason=NO_COST)
    else:
        loop._charge_llm(cost)


def _copy_transcript(wt: Path, loop) -> None:
    """The agentic worktree is removed at the start of the next iteration, so a failed
    iteration's stdout/stderr must be copied into the iteration dir now or it is lost.
    loop._n is only set on a real Loop mid-iterate; the test's SimpleNamespace stub has none."""
    n = getattr(loop, "_n", None)
    if n is None:
        return
    it_dir = loop.store.iteration_dir(n)
    for name in ("agent_stdout.txt", "agent_stderr.txt"):
        src = wt / ".talos" / name
        if src.exists():
            shutil.copy(src, it_dir / name)


def _denial_note(wt: Path) -> str:
    """What the claude CLI refused, for the error of an iteration that produced nothing: without
    it a session whose every Edit was denied reads as the agent doing no work. Empty for codex,
    whose stdout is prose, and for a JSON result cut by _TRANSCRIPT_CAP."""
    try:
        result = json.loads((wt / ".talos" / "agent_stdout.txt").read_text(encoding="utf-8"))
        tools = Counter(str(d["tool_name"]) for d in result["permission_denials"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    if not tools:
        return ""
    n = sum(tools.values())
    return (f"; the CLI refused {n} tool call{'s' if n != 1 else ''} ("
            + ", ".join(f"{tool} x{count}" for tool, count in sorted(tools.items())) + ")")


def attach_agentic(loop, provider_kind: str, model: str, timeout_s: int = 1800,
                   run=subprocess.run) -> None:
    runner = {"claude-cli": _run_claude, "codex-cli": _run_codex}.get(provider_kind)
    if runner is None:
        raise AgenticError(f"agentic mode needs claude-cli or codex-cli, not {provider_kind}")
    start = getattr(getattr(loop, "spec", None), "start", "mainnet")
    if codex_agentic_refused(provider_kind, "agentic", start):
        raise AgenticError(TEMPLATE_CODEX_REFUSAL if start == "template"
                           else CODEX_AGENTIC_REFUSAL)

    def propose_and_edit(ctx: PromptContext) -> tuple[dict, dict[str, str]]:
        previous = getattr(loop, "_agentic_wt", None)
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
        wt = prepare_worktree(ctx)
        loop._agentic_wt = wt
        prompt = ("Read CLAUDE.md (or AGENTS.md), then make one improvement to the solver under "
                  "algorithm/, check it with `talos compile`, and write .talos/hypothesis.json.")
        recall = recall_block(ctx)
        if recall:
            prompt += "\n" + recall
        if ctx.forced_tag:
            prompt += f"\nYou have stagnated: use strategy_tag \"{ctx.forced_tag}\"."
        loop._check_budget()
        try:
            r = runner(wt, model, prompt, timeout_s, run)
        except AgentTimeout as e:
            # The json format prints its result only at the end, so a killed session's cost
            # is lost; say so rather than count it as $0.
            if provider_kind == "claude-cli":
                loop._event("llm_cost_unknown", reason=str(e))
            raise
        finally:
            _copy_transcript(wt, loop)
        if provider_kind == "claude-cli":
            _charge_session(loop, r)
        if r.returncode != 0:
            raise AgenticError(f"{r.args[0]} exited {r.returncode}: {(r.stderr or '')[-500:]}")
        try:
            hypothesis, files = read_back(wt, ctx)
        except AgenticError as e:
            raise AgenticError(f"{e}{_denial_note(wt)}") from None
        loop._event("hypothesis", **hypothesis)
        return hypothesis, files

    loop.propose_and_edit = propose_and_edit
