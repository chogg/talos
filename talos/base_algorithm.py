"""The root files of a `talos run --base-algorithm/--base-dir` job: the code every chain starts
from, snapshotted into the run directory at job start so a resume never reads the source again.
The baseline stays mainnet's top leak-free algorithm. Loop, package and report all read the root
here."""
from __future__ import annotations

from pathlib import Path

from talos.challenges import CHALLENGES
from talos.leak_check import describe, leak_findings, void_reason

ROOT_DIR = "base_algorithm"


def read_algorithm_dir(d: Path) -> dict[str, str]:
    """{relative_path: contents} of the .rs and .cu files under `d`; empty when it is no
    directory."""
    d = Path(d)
    return ({p.relative_to(d).as_posix(): p.read_text(encoding="utf-8") for p in d.rglob("*")
             if p.is_file() and p.suffix in (".rs", ".cu")} if d.is_dir() else {})


def root_problem(challenge: str, files: dict[str, str]) -> str | None:
    """Why these files cannot be a root, or None. Checked before the job exists: without them
    the build fails, and only after the baseline has been paid for."""
    if "mod.rs" not in files:
        return "the base algorithm has no mod.rs at its top level"
    if CHALLENGES[challenge].is_gpu and not any(name.endswith(".cu") for name in files):
        return f"{challenge} is a GPU challenge: the base algorithm needs a .cu file"
    findings = leak_findings(challenge, files)
    if findings:
        return (f"the base algorithm carries the hidden_seed leak ({describe(findings)}); remove "
                "it first, because every candidate would inherit it")
    return None


def write_base_root(run_dir: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = Path(run_dir) / ROOT_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def read_base_root(run_dir: Path) -> dict[str, str] | None:
    """The snapshot in runs/<job>/base_algorithm/, or None when it is missing. Every file, not
    only .rs and .cu: a mainnet algorithm's root is its whole directory, as a baseline's is."""
    root = Path(run_dir) / ROOT_DIR
    files = {p.relative_to(root).as_posix(): p.read_bytes().decode("utf-8", errors="replace")
             for p in root.rglob("*") if p.is_file()} if root.is_dir() else {}
    return files or None


def job_void_reason(spec, state, run_dir: Path) -> str | None:
    """`talos/leak_check.py::void_reason` with this job's root read from its run directory. A
    mainnet start's root is its baseline, which is checked anyway."""
    root = read_base_root(run_dir) if spec.start == "base" else None
    return void_reason(spec, state, root)


def root_label(start: str) -> str:
    """What a job's node 0 is called in diffs, reports and JSON."""
    return {"template": "template", "base": "base"}.get(start, "baseline")
