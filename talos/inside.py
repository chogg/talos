"""Container-side logic for the Modal bench. Pure functions that take the subprocess runner
as a parameter, so they are unit-tested on any machine. Runs inside the TIG dev image,
where `build_algorithm`, `tig-runtime` and `tig-verifier` are on PATH and the monorepo
checkout is the working directory."""
from __future__ import annotations

import glob
import hashlib
import json
import multiprocessing
import os
import re
import platform
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from talos import native_runner
from talos.challenges import CHALLENGES

ALGO_NAME = "talos_cand"
# The challenge crate is compiled with the candidate as its only module (see stage_algorithm);
# part of every artifact id and baseline key, so a .so or a baseline measured beside the
# shipped algorithms is never mistaken for one measured in the pruned crate. Bump on any
# change to what the crate contains at build time.
CRATE_LAYOUT = "pruned-1"
PRISTINE_MOD_RS = "mod.rs.talos-pristine"
PRUNED_MARKER = ("// talos: crate pruned to the candidate; the pinned mod.rs is beside this "
                  "file as " + PRISTINE_MOD_RS)
_QUALITY_RE = re.compile(r"quality:\s*(-?\d+)")
# Exit codes, from tig-runtime/src/main.rs at MONOREPO_REF.
RUNTIME_ERROR_RC = 84  # compute_solution returned Err: the algorithm gave up / no solution
OUT_OF_FUEL_RC = 87    # the algorithm library exits 87 when fuel runs out
RUST_PANIC_RC = 101    # a Rust panic that unwinds to main


def _algo_root(monorepo: Path, challenge: str) -> Path:
    return monorepo / "tig-algorithms" / "src" / challenge


def _check_rel(target: Path, rel: str) -> None:
    """Reject anything that is not a plain relative file path strictly inside `target`.
    An LLM-authored file map is untrusted input: `""`, `"."`, `"/etc/evil.rs"` and
    `"sub/../../evil.rs"` all have to fail here rather than at write_text."""
    bad = f"file path escapes algorithm dir: {rel!r}"
    if not rel or rel == "." or Path(rel).is_absolute():
        raise ValueError(bad)
    if any(seg in ("", ".", "..") for seg in rel.split("/")):
        raise ValueError(bad)
    root = target.resolve()
    p = (target / rel).resolve()
    if p == root or not p.is_relative_to(root):  # a "/" prefix test fails on Windows paths
        raise ValueError(bad)


def stage_algorithm(monorepo: Path, challenge: str, files: dict[str, str], name: str) -> None:
    root = _algo_root(monorepo, challenge)
    target = root / name
    for rel in files:
        _check_rel(target, rel)
    target.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    _prune_crate(root, name)


def _prune_crate(root: Path, name: str) -> None:
    """Make `name` the challenge crate's only module, keeping the pinned mod.rs beside it.

    The build compiles every module the crate lists, on one codegen unit (build_so passes
    `-C codegen-units=1`), so the crate's size sets the build cost whatever the candidate is:
    job_scheduling's 14 shipped algorithms are 345k lines and the full crate ran past 2 h and
    23 GB of rustc before it was stopped, against 629 s and 3.4 GB for the candidate alone
    (MEASURED 2026-09-25). TIG's own CI builds an algorithm from its branch, whose mod.rs
    lists only that algorithm, so the pruned crate is the closer match to mainnet, not the
    further one. The pinned file is kept byte for byte so unstage can put it back; it is not
    re-saved when mod.rs already carries the marker, because a job that died before
    unstaging (the local backend's /app volume outlives it) leaves the pruned file in place."""
    mod_rs = root / "mod.rs"
    pristine = root / PRISTINE_MOD_RS
    current = mod_rs.read_bytes()
    if not pristine.exists() and not current.startswith(PRUNED_MARKER.encode()):
        pristine.write_bytes(current)
    mod_rs.write_text(f"{PRUNED_MARKER}\npub mod {name};\n", encoding="utf-8", newline="\n")


def unstage_algorithm(monorepo: Path, challenge: str, name: str) -> None:
    import shutil
    root = _algo_root(monorepo, challenge)
    shutil.rmtree(root / name, ignore_errors=True)
    mod_rs = root / "mod.rs"
    pristine = root / PRISTINE_MOD_RS
    if pristine.exists():
        mod_rs.write_bytes(pristine.read_bytes())
        pristine.unlink()
        return
    # A crate that was never pruned (or whose pristine copy is gone): drop the line alone.
    line = f"pub mod {name};"
    kept = [ln for ln in mod_rs.read_text(encoding="utf-8").splitlines() if ln != line]
    mod_rs.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")


# A cap on what leaves the container, not a window on the diagnostics: every consumer filters
# with diagnostics.relevant first and truncates after. A 20000-byte window here kept 6 of the
# 8 errors of run 20260916-095103 iteration 5, cut off before anything could drop the other
# algorithms' warnings.
BUILD_OUTPUT_CAP = 1_000_000


def challenge_problem(challenge: str, monorepo: Path) -> str | None:
    """Why this checkout's challenge still leaks its hidden seed, or None. The derive compiles into
    the algorithm's .so from this source, so this file is the code a score depends on, whatever
    the pin says it is."""
    if challenge != "energy_arbitrage":
        return None
    path = Path(monorepo) / "tig-challenges" / "src" / challenge / "mod.rs"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"cannot check {path} for TIG's hidden_seed fix (2703c9af): {e}"
    derive = re.search(r"#\[derive\(([^)]*)\)\]\s*pub struct Challenge\b", source)
    if derive is None or re.search(r"\b(?:Debug|Deserialize)\b", derive.group(1)):
        return f"{path} predates TIG's hidden_seed fix (2703c9af): Challenge can expose it"
    if not re.search(r"#\[serde\(skip_serializing\)\]\s*hidden_seed\s*:", source):
        return f"{path} predates TIG's hidden_seed fix (2703c9af): hidden_seed is not private"
    return None


def build(monorepo: Path, challenge: str, name: str, run=subprocess.run) -> tuple[bool, str]:
    problem = challenge_problem(challenge, monorepo)
    if problem:
        return False, problem
    r = run(["build_algorithm", name], cwd=monorepo, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-BUILD_OUTPUT_CAP:]


# Defined beside the runner templates, since it is part of native_runner.runner_digest.
NATIVE_ENV = native_runner.NATIVE_ENV
NATIVE_PTX_SUFFIX = ".native.ptx"


def _native_ptx(monorepo: Path, challenge: str, name: str) -> Path:
    return monorepo / "tig-algorithms" / "lib" / challenge / "ptx" / f"{name}{NATIVE_PTX_SUFFIX}"


def build_native_ptx(monorepo: Path, challenge: str, name: str,
                     run=subprocess.run) -> tuple[bool, str]:
    """tig-binary/scripts/build_ptx at MONOREPO_REF without inject_fuel_and_runtime_sig: the
    same files in the same order (framework, the challenge's .cu by the same recursive glob,
    the algorithm's), the same nvcc flags, written beside the metered PTX, never over it. An
    empty challenge or algorithm glob fails the build, as build_ptx's FileNotFoundError does."""
    framework = monorepo / "tig-binary" / "src" / "framework.cu"
    challenge_cus = glob.glob(str(monorepo / "tig-challenges" / "src" / challenge / "**" / "*.cu"),
                              recursive=True)
    if not challenge_cus:  # build_ptx raises FileNotFoundError here
        return False, f"no .cu files in tig-challenges/src/{challenge}; is the challenge right?"
    algo_cus = glob.glob(str(_algo_root(monorepo, challenge) / name / "*.cu"))
    if not algo_cus:
        return False, f"no .cu files in the {name} algorithm; a GPU algorithm needs its kernels"
    code = framework.read_text(encoding="utf-8") + "\n"
    for p in challenge_cus:
        code += Path(p).read_text(encoding="utf-8") + "\n"
    for p in algo_cus:
        code += Path(p).read_text(encoding="utf-8") + "\n\n"
    dest = _native_ptx(monorepo, challenge, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        cu = Path(td) / "temp.cu"
        cu.write_text(code, encoding="utf-8", newline="\n")
        r = run(["nvcc", "-ptx", str(cu), "-o", str(dest), "-arch", "compute_70",
                 "-code", "sm_70", "--use_fast_math", "-dopt=on"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def build_native(monorepo: Path, challenge: str, name: str,
                 run=subprocess.run) -> tuple[bool, str]:
    """Stage the talos-native runner, build the unmetered PTX for a GPU challenge, then
    `cargo build` the runner. The algorithm must already be staged (stage_algorithm)."""
    problem = challenge_problem(challenge, monorepo)
    if problem:
        return False, problem
    is_gpu = CHALLENGES[challenge].is_gpu
    native_runner.stage(monorepo, challenge, name, is_gpu)
    out = ""
    if is_gpu:
        ok, ptx_out = build_native_ptx(monorepo, challenge, name, run)
        out += ptx_out
        if not ok:
            return False, out[-BUILD_OUTPUT_CAP:]
    r = run(["cargo", native_runner.TOOLCHAIN, "build", "--release", "-p", native_runner.PACKAGE],
            cwd=monorepo, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, **NATIVE_ENV})
    out += (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out[-BUILD_OUTPUT_CAP:]


def native_artifact_paths(monorepo: Path, challenge: str, name: str) -> tuple[Path, Path | None]:
    binary = monorepo / "target" / "release" / native_runner.PACKAGE
    ptx = _native_ptx(monorepo, challenge, name)
    return binary, (ptx if ptx.exists() else None)


# The TIG build writes lib/<challenge>/<arch>/ using Docker's architecture names,
# not uname's: amd64 on x86_64 hosts, arm64 on aarch64 hosts.
_ARCH_DIR = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


def artifact_paths(monorepo: Path, challenge: str, name: str,
                   machine: str | None = None) -> tuple[Path, Path | None]:
    machine = machine or platform.machine()
    arch = _ARCH_DIR.get(machine, machine)
    lib = monorepo / "tig-algorithms" / "lib" / challenge
    so = lib / arch / f"{name}.so"
    ptx = lib / "ptx" / f"{name}.ptx"
    return so, (ptx if ptx.exists() else None)


def classify(runtime_rc: int, verifier_rc: int, quality: int | None,
             timed_out: bool) -> tuple[bool, str | None]:
    """A run that saved a solution before running out of fuel still counts if it verifies.
    tig-verifier exits 1 for every rejection, so the runtime exit code carries the reason."""
    if timed_out:
        return False, "timeout"
    if verifier_rc == 0 and quality is not None:
        return True, None
    if runtime_rc == OUT_OF_FUEL_RC:
        return False, "out_of_fuel"
    if runtime_rc < 0 or runtime_rc == RUST_PANIC_RC:
        return False, "panic"
    if runtime_rc == RUNTIME_ERROR_RC:
        return False, "no_solution"
    if runtime_rc != 0:
        return False, "panic"
    return False, "invalid"


def _settings(challenge_id: str, track: str) -> str:
    return json.dumps({"algorithm_id": "", "challenge_id": challenge_id, "track_id": track,
                       "block_id": "", "player_id": ""}, separators=(",", ":"))


def _hp_args(hyperparameters: dict | None) -> list[str]:
    return ([] if hyperparameters is None else
            ["--hyperparameters", json.dumps(hyperparameters, separators=(",", ":"))])


def _run_and_verify(cmd: list[str], out_file: Path, settings: str, rand_hash: str, nonce: int,
                    gpu_args: list[str], timeout_s: int, run, clock) -> dict:
    """Runs the solver (`cmd`), then tig-verifier on what it saved. Returns the classified row
    fields plus the saved JSON document (None when nothing was saved)."""
    t0 = clock()
    timed_out = False
    try:
        r1 = run(cmd, capture_output=True, text=True, timeout=timeout_s,
                 encoding="utf-8", errors="replace")
        rt_rc = r1.returncode
    except subprocess.TimeoutExpired:
        timed_out, rt_rc = True, -1
    elapsed = clock() - t0
    quality = None
    ver_rc = 1
    doc = None
    if not timed_out and out_file.exists():
        try:
            doc = json.loads(out_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            doc = None
        # The verifier gets only the time the runtime left. Giving it a fresh timeout_s put
        # the worst case at 2 x timeout_s, past the Modal function timeout, which kills the
        # container and turns a slow nonce into an infrastructure error instead of a result.
        # A verifier timeout must not raise either: TimeoutExpired's str carries the whole
        # argv, rand_hash included, and it would surface in a client-side error message.
        try:
            r2 = run(["tig-verifier", settings, rand_hash, str(nonce), str(out_file)] + gpu_args,
                     capture_output=True, text=True,
                     timeout=max(1, int(timeout_s - elapsed)),
                     encoding="utf-8", errors="replace")
            ver_rc = r2.returncode
            m = _QUALITY_RE.search(r2.stdout or "")
            quality = int(m.group(1)) if m else None
        except subprocess.TimeoutExpired:
            timed_out = True
    ok, err = classify(rt_rc, ver_rc, quality, timed_out)
    return {"ok": ok, "quality": quality if ok else None, "runtime_ms": int(elapsed * 1000),
            "error": err, "doc": doc if isinstance(doc, dict) else None,
            "limit_hit": rt_rc == OUT_OF_FUEL_RC}


def _int_or_none(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def run_nonce(challenge_id: str, track: str, rand_hash: str, nonce: int, so: Path, fuel: int,
              timeout_s: int, ptx: Path | None, run=subprocess.run,
              workdir: Path | None = None, clock=time.monotonic,
              hyperparameters: dict | None = None) -> dict:
    """Mirrors scripts/test_algorithm in the monorepo:
    `tig-runtime SETTINGS RAND_HASH NONCE SO --fuel F --output DIR [--hyperparameters JSON]
    [--ptx P --gpu 0]` writes DIR/<nonce>.json, then
    `tig-verifier SETTINGS RAND_HASH NONCE DIR/<nonce>.json [--ptx P --gpu 0]`
    prints `quality: N` and exits 0 on a valid solution. `{}` is passed as `{}`: the algorithm
    receives Some(empty map), not None. The row carries the fuel_consumed tig-runtime wrote at
    the last save, which calibration pairs with the native solve time."""
    settings = _settings(challenge_id, track)
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        cmd = ["tig-runtime", settings, rand_hash, str(nonce), str(so),
               "--fuel", str(fuel), "--output", td] + _hp_args(hyperparameters) + gpu_args
        v = _run_and_verify(cmd, out_file, settings, rand_hash, nonce, gpu_args, timeout_s,
                            run, clock)
    fuel_consumed = _int_or_none((v["doc"] or {}).get("fuel_consumed"))
    return {"track": track, "nonce": nonce, "ok": v["ok"], "quality": v["quality"],
            "runtime_ms": v["runtime_ms"], "error": v["error"], "fuel_consumed": fuel_consumed,
            "limit_hit": v["limit_hit"]}


def run_nonce_native(challenge_id: str, track: str, rand_hash: str, nonce: int, binary: Path,
                     budget_us: int | None, timeout_s: int, ptx: Path | None,
                     run=subprocess.run, workdir: Path | None = None, clock=time.monotonic,
                     hyperparameters: dict | None = None, *, fuel: int) -> dict:
    """`talos-native SETTINGS RAND_HASH NONCE OUT_FILE --fuel F [--budget-us B]
    [--hyperparameters JSON] [--ptx P]`, then tig-verifier exactly as run_nonce calls it. The
    fuel is only the cap the runner's `__fuel_remaining` starts from, for algorithms that read
    it; the budget is what stops the solve. The runner exits 87 when its
    solve has run for budget_us, so a budget exit is classified as a metered out-of-fuel exit
    is: a saved solution that verifies still counts. `timeout_s` is the outer cap and reports
    `timeout`, as on the metered path."""
    settings = _settings(challenge_id, track)
    gpu_args = ["--ptx", str(ptx), "--gpu", "0"] if ptx else []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        out_file = Path(td) / f"{nonce}.json"
        cmd = [str(binary), settings, rand_hash, str(nonce), str(out_file), "--fuel", str(fuel)]
        if budget_us is not None:
            cmd += ["--budget-us", str(budget_us)]
        cmd += _hp_args(hyperparameters) + (["--ptx", str(ptx)] if ptx else [])
        v = _run_and_verify(cmd, out_file, settings, rand_hash, nonce, gpu_args, timeout_s,
                            run, clock)
    solve_us = _int_or_none((v["doc"] or {}).get("solve_us"))
    return {"track": track, "nonce": nonce, "ok": v["ok"], "quality": v["quality"],
            "runtime_ms": v["runtime_ms"], "error": v["error"], "fuel_consumed": None,
            "solve_us": solve_us,
            "limit_hit": v["limit_hit"]}


NONCE_TIMEOUT_S = 600

# One scoring task, as a positional tuple so it pickles into a worker process unchanged:
# (challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hyperparameters,
#  mode, budget_us), with `so`, `ptx` and `workdir` as strings. For mode "native", `so` is the
# talos-native binary and `fuel` is the cap its __fuel_remaining starts from; budget_us is the
# track's solve budget or None.
NonceTask = tuple


def run_task(task: NonceTask, run=subprocess.run) -> dict:
    (challenge_id, track, rand_hash, nonce, so, fuel, timeout_s, ptx, workdir, hp, mode,
     budget_us) = task
    if mode == "native":
        return run_nonce_native(challenge_id, track, rand_hash, nonce, Path(so), budget_us,
                                timeout_s, Path(ptx) if ptx else None, run=run,
                                workdir=Path(workdir), hyperparameters=hp, fuel=fuel)
    return run_nonce(challenge_id, track, rand_hash, nonce, Path(so), fuel, timeout_s,
                     Path(ptx) if ptx else None, run=run, workdir=Path(workdir),
                     hyperparameters=hp)


def _own_process_group() -> None:
    """Pool initializer: the worker leads a process group of its own, which the solver and
    verifier it starts inherit, so a stop can kill them together (run_nonces)."""
    os.setpgrp()


def _kill_groups(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # that worker and everything it started have already exited


def run_nonces(tasks: list[NonceTask], workers: int, run=subprocess.run,
               pool_factory=None) -> Iterator[dict]:
    """Scores `tasks` on `workers` processes at once and yields each row as it finishes, in no
    particular order. This is the one place a container's cores are spread over: the C3 and
    local jobs and every Modal batch go through it. `run_task` builds its own subprocess calls
    and cannot carry an injected runner into a worker process, so the pool is only used with
    the real subprocess, or with a `pool_factory` a test supplied to drive that branch."""
    if workers > 1 and (pool_factory is not None or run is subprocess.run):
        groups = hasattr(os, "setpgrp")  # the containers are Linux; the tests also run on Windows
        pool = (pool_factory(workers) if pool_factory is not None else
                multiprocessing.Pool(workers, initializer=_own_process_group if groups else None))
        with pool:
            try:
                yield from pool.imap_unordered(run_task, tasks)
            except GeneratorExit:
                # The caller stopped early (talos/c3_job.py::_score). terminate() kills only the
                # workers; the tig-runtime each one started would run on to its own timeout and
                # keep the job alive. Terminate first, so no worker is replaced in between.
                pids = [p.pid for p in multiprocessing.active_children()]
                pool.terminate()
                if groups and pool_factory is None:
                    _kill_groups(pids)
                raise
    else:
        for t in tasks:
            yield run_task(t, run=run)


def content_hash(files: dict[str, str], monorepo_ref: str, dev_image_tag: str,
                 mode: str = "metered") -> str:
    """Artifact cache key. The monorepo pin, the dev image tag and the crate layout are part
    of it: the same sources built against a different monorepo, or beside a different set of
    modules, are a different .so. A native build is a different artifact from a metered one;
    the mode enters the hash only when it is not "metered", so every metered id is unchanged.
    A native id also carries the runner digest: a changed template, toolchain or build
    profile is a different binary."""
    h = hashlib.sha256()
    h.update(monorepo_ref.encode())
    h.update(b"\0")
    h.update(dev_image_tag.encode())
    h.update(b"\0")
    h.update(CRATE_LAYOUT.encode())
    h.update(b"\0")
    if mode == "native":
        h.update(f"mode=native:{native_runner.runner_digest()}".encode())
        h.update(b"\0")
    elif mode != "metered":
        h.update(f"mode={mode}".encode())
        h.update(b"\0")
    for k in sorted(files):
        h.update(k.encode())
        h.update(b"\0")
        h.update(files[k].encode())
        h.update(b"\0")
    return h.hexdigest()[:32]
