"""Read-only views of a run directory for `talos status`, `talos show` and `talos diff`.

Everything here reads runs/<job_id>/ with plain file reads and writes nothing: JobStore is not
used, because its constructor creates the directory. The data functions return plain dicts,
which are the `--json` output (docs/json-output.md); the format_* functions render those same
dicts as text, so the two forms cannot disagree.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from talos.base_algorithm import job_void_reason, root_label
from talos.budget import effective_budget
from talos.package import _diff, score_pairs
from talos.report import ReportNode, holdout_points, read_timeline, report_nodes, run_sources
from talos.scoring import ScoringError, bundle_delta
from talos.state import TERMINAL, JobSpec, JobState

SCHEMA_VERSION = 1

# Statuses whose clock has stopped: hours spent is measured up to the last timeline event.
_STOPPED = TERMINAL | {"paused"}


class InspectError(Exception):
    """A user-facing error; str() is the message printed on stderr."""


@dataclass(frozen=True)
class Run:
    run_dir: Path
    spec: JobSpec
    state: JobState
    timeline: list[dict]

    @property
    def job_id(self) -> str:
        # The directory name, which is what the user typed; job.json's job_id matches it in
        # every run Talos writes.
        return self.run_dir.name


def _read(run_dir: Path, job_id: str, name: str, parse):
    path = run_dir / name
    try:
        if not path.is_file():
            raise InspectError(f"job {job_id} has no {name}")
        return parse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise InspectError(f"job {job_id}: cannot read {name}: {e}") from e


def load_run(runs_root: Path, job_id: str) -> Run:
    """The run under runs_root/<job_id>/. Raises InspectError with the message to print."""
    if job_id in ("", ".", "..") or "/" in job_id or "\\" in job_id:
        raise InspectError(f"no job {job_id} under runs/")
    run_dir = Path(runs_root) / job_id
    if not run_dir.is_dir():
        raise InspectError(f"no job {job_id} under runs/")
    state = _read(run_dir, job_id, "state.json", JobState.from_dict)
    spec = _read(run_dir, job_id, "job.json", JobSpec.from_dict)
    return Run(run_dir, spec, state, read_timeline(run_dir / "timeline.jsonl"))


def holdout_status(state: JobState, iteration: int) -> str:
    if iteration in state.confirmed:
        return "confirmed"
    if iteration in state.false_positives:
        return "false_positive"
    return "not_scored"


def _stamps(timeline: list[dict]) -> list[float]:
    return [row["ts"] for row in timeline
            if isinstance(row.get("ts"), (int, float)) and not isinstance(row["ts"], bool)]


def _last_ts(timeline: list[dict]) -> float | None:
    stamps = _stamps(timeline)
    return float(stamps[-1]) if stamps else None


def hours_spent(run: Run, now: float) -> float | None:
    """Working hours: up to the last event for a job that has stopped, up to `now` otherwise."""
    started = run.state.spend.started_at
    if run.state.status in _STOPPED:
        last = _last_ts(run.timeline)
        return None if last is None else (last - started) / 3600
    return (now - started) / 3600


def last_event(timeline: list[dict]) -> dict | None:
    """The last readable event as {kind, ts, iteration, text}, or None."""
    from talos.cli import _event_line

    for row in reversed(timeline):
        ts, kind = row.get("ts"), row.get("kind")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not isinstance(kind, str):
            continue
        data = {k: v for k, v in row.items() if k not in ("ts", "kind")}
        try:
            text = _event_line(kind, data)
        except (KeyError, TypeError, ValueError):
            text = None
        iteration = row.get("iteration")
        return {"kind": kind, "ts": float(ts),
                "iteration": iteration if isinstance(iteration, int) else None,
                "text": text or kind}
    return None


def _llm(spec: JobSpec, llm_usd: float) -> tuple[float | None, bool, str | None]:
    """(llm_usd or None when unpriced, priced, why it is unpriced)."""
    from talos.cli import unpriced_reason

    reason = unpriced_reason(spec.provider, spec.model)
    return (float(llm_usd) if reason is None else None), reason is None, reason


_ROW_KEYS = ("job_id", "challenge", "provider", "model", "status", "iteration",
             "best_mean_rel_delta", "llm_usd", "llm_priced", "llm_unpriced_reason",
             "compute_usd", "last_event_ts", "void_reason", "error")


def _best_delta(state: JobState) -> float | None:
    return state.best.delta.get("mean_rel_delta") if state.best is not None else None


def status_rows(runs_root: Path) -> list[dict]:
    """One row per directory under runs_root that holds job.json or state.json, by name. A
    directory that cannot be read is an `unreadable` row, so it never hides the others."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    rows = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        row = dict.fromkeys(_ROW_KEYS)
        row["job_id"] = d.name
        try:
            # is_file raises PermissionError on a directory the user cannot read
            if not ((d / "job.json").is_file() or (d / "state.json").is_file()):
                continue
            run = load_run(runs_root, d.name)
        except OSError as e:
            row["status"], row["error"] = "unreadable", f"job {d.name}: cannot read: {e}"
            rows.append(row)
            continue
        except InspectError as e:
            row["status"], row["error"] = "unreadable", str(e)
            rows.append(row)
            continue
        spec, state = run.spec, run.state
        llm_usd, priced, reason = _llm(spec, state.spend.llm_usd)
        row.update(challenge=spec.challenge, provider=spec.provider, model=spec.model,
                   status=state.status, iteration=state.iteration,
                   best_mean_rel_delta=_best_delta(state), llm_usd=llm_usd, llm_priced=priced,
                   llm_unpriced_reason=reason, compute_usd=float(state.spend.compute_usd),
                   last_event_ts=_last_ts(run.timeline),
                   void_reason=job_void_reason(spec, state, run.run_dir))
        rows.append(row)
    return rows


def _num(x, as_float: bool):
    return None if x is None else (float(x) if as_float else x)


def _cap(cap, spent, as_float: bool = True) -> dict:
    """{cap, spent, left}. Money and hours are floats even when job.json or state.json holds
    a whole number, as docs/json-output.md types them; iterations stay ints."""
    left = None if cap is None or spent is None else max(0, cap - spent)
    return {"cap": _num(cap, as_float), "spent": _num(spent, as_float),
            "left": _num(left, as_float)}


def job_detail(run: Run, now: float) -> dict:
    spec, state = run.spec, run.state
    eff = effective_budget(spec.budget, state.budget_extensions)
    llm_usd, priced, reason = _llm(spec, state.spend.llm_usd)
    best = None
    if state.best is not None:
        it = state.best.iteration
        best = {"iteration": it, "title": state.best.hypothesis.get("title", ""),
                "mean_rel_delta": _best_delta(state),
                "holdout_status": holdout_status(state, it),
                "holdout_mean_rel_delta": holdout_points(run.timeline).get(it)}
    pending = None
    if state.pending_job is not None:
        hyp = state.pending_job.get("hypothesis")
        pending = {"purpose": state.pending_job.get("purpose"),
                   "title": hyp.get("title") if isinstance(hyp, dict) else None}
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": run.job_id,
        "job": spec.redacted(),
        "void_reason": job_void_reason(spec, state, run.run_dir),
        "status": state.status,
        "stop_reason": state.stop_reason,
        "iteration": state.iteration,
        "runs_since_improvement": state.runs_since_improvement,
        "hardware": state.hardware,
        "scoring": {"requested": spec.scoring, "actual": state.scoring},
        "best": best,
        "budget": {"iterations": _cap(eff.iterations, state.spend.iterations, as_float=False),
                   "hours": _cap(eff.hours, hours_spent(run, now)),
                   "usd": _cap(eff.usd, llm_usd),
                   "compute_usd": _cap(eff.compute_usd, state.spend.compute_usd)},
        "llm_priced": priced,
        "llm_unpriced_reason": reason,
        "last_event_ts": _last_ts(run.timeline),
        "activity": {"pending": pending, "last_event": last_event(run.timeline)},
    }


# ── one iteration ────────────────────────────────────────────────────

def resolve_node(run: Run, which: str | int) -> ReportNode:
    """`which` is "best" or an iteration number; the node for it, or InspectError."""
    nodes = report_nodes(run.spec, run.state)
    if which == "best":
        if run.state.best is None:
            raise InspectError(f"job {run.job_id} has no best candidate; use --iteration N")
        which = run.state.best.iteration
    for n in nodes:
        if n.id == which:
            return n
    has = f"1..{max(n.id for n in nodes)}" if nodes else "none"
    raise InspectError(f"job {run.job_id} has no iteration {which} (it has {has})")


def _label(run: Run, node_id: int) -> str:
    if node_id == 0:
        return root_label(run.spec.start)
    return f"#{node_id}"


def _rel(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:  # the workspace is not under the current directory
        return path.as_posix()


def _files(run: Run, node_id: int, is_best: bool) -> list[str]:
    dirs = [run.run_dir / "iterations" / f"{node_id:04d}"]
    if is_best:
        dirs.append(run.run_dir / "best")
    return sorted(_rel(p) for d in dirs if d.is_dir() for p in d.rglob("*") if p.is_file())


def _tables(run: Run) -> list[dict]:
    out = []
    for heading, base, cand in score_pairs(run.spec, run.state):
        try:
            d = bundle_delta(base, cand)
        except ScoringError as e:
            out.append({"heading": heading, "tracks": [], "mean_rel_delta": None,
                        "error_rate": None, "error": str(e)})
            continue
        out.append({"heading": heading,
                    "tracks": [{"track": t.track, "base_mean": float(t.base_mean),
                                "cand_mean": float(t.cand_mean),
                                "rel_delta": float(t.rel_delta),
                                "cand_errors": t.cand_errors, "n": t.n} for t in d.tracks],
                    "mean_rel_delta": float(d.mean_rel_delta),
                    "error_rate": float(d.error_rate),
                    "error": None})
    return out


def iteration_detail(run: Run, which: str | int) -> dict:
    node = resolve_node(run, which)
    state = run.state
    record = next((h for h in state.hypotheses if h.get("iteration") == node.id), {})
    stored = next((n for n in state.nodes if n.id == node.id), None)
    is_best = state.best is not None and state.best.iteration == node.id
    # The training mean is the record's, not ReportNode.score: that is the metered score when
    # a native job validated the candidate.
    mean = record.get("mean_rel_delta")
    if mean is None and stored is not None:
        mean = stored.score
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": run.job_id,
        "iteration": node.id,
        "parent": node.parent,
        "parent_label": _label(run, node.parent),
        "chain": node.chain,
        "kind": node.kind,
        "void_reason": job_void_reason(run.spec, run.state, run.run_dir),
        "outcome": node.outcome,
        "error": node.error,
        "hypothesis": {"title": node.title, "description": node.description,
                       "strategy_tag": node.strategy_tag,
                       "design_family": record.get("design_family") or node.family or ""},
        "scores": {
            "mean_rel_delta": mean,
            "worst_track": record.get("worst_track"),
            "worst_rel_delta": record.get("worst_rel_delta"),
            "runtime_ratio": record.get("runtime_ratio"),
            "metered_rel_delta": stored.metered_score if stored is not None else None,
            "tracks": stored.tracks if stored is not None else None,
            "holdout_status": holdout_status(state, node.id),
            "holdout_mean_rel_delta": holdout_points(run.timeline).get(node.id),
            "tables": _tables(run) if is_best and state.baseline is not None else None,
        },
        "files": _files(run, node.id, is_best),
    }


def iteration_diff(run: Run, which: str | int, against: str) -> str:
    """The unified diff of one iteration against the root ("baseline") or its parent,
    untruncated."""
    node = resolve_node(run, which)
    sources = run_sources(run.run_dir, run.spec, run.state)
    cand = sources(node.id)
    if not cand:
        raise InspectError(f"iteration {node.id} saved no files (outcome {node.outcome})")
    base_id = 0 if against == "baseline" else node.parent
    label = _label(run, base_id)
    base = sources(base_id)
    if not base:
        raise InspectError(f"the files of {label} were not saved")
    return _diff(base, cand, label)


# ── human output ─────────────────────────────────────────────────────

def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3%}"


def _utc(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip()
                     for r in [header, *rows])


def format_status_table(rows: list[dict]) -> str:
    from talos.cli import _llm_spend_text

    header = ["JOB", "CHALLENGE", "STATUS", "IT", "BEST", "LLM", "COMPUTE", "LAST EVENT"]
    body = []
    for r in rows:
        if r["error"] is not None:
            body.append([r["job_id"], "-", r["status"], "-", "-", "-", "-", "-"])
            continue
        status = f"{r['status']} (void)" if r["void_reason"] else r["status"]
        body.append([r["job_id"], r["challenge"], status, str(r["iteration"]),
                     _pct(r["best_mean_rel_delta"]),
                     _llm_spend_text(r["provider"], r["model"], r["llm_usd"] or 0.0),
                     f"${r['compute_usd']:.2f}", _utc(r["last_event_ts"])])
    return _table(header, body)


def _amount(dim: str, x) -> str:
    if x is None:
        return "-"
    if dim == "iterations":
        return str(x)
    if dim == "hours":
        return f"{x:.2f}h"
    return f"${x:.2f}"


def format_job_detail(d: dict) -> str:
    job = d["job"]
    lines = [f"job: {d['job_id']}",
             f"challenge: {job['challenge']}",
             f"track focus: {job.get('track') or 'all tracks'}",
             f"provider: {job['provider']} / {job['model']}",
             f"mode: {job['mode']}  search: {job.get('search')}  start: {job.get('start')}",
             f"scoring: {d['scoring']['requested']}"
             + (f" (running {d['scoring']['actual']})"
                if d["scoring"]["actual"] not in (None, d["scoring"]["requested"]) else ""),
             f"created: {_utc(job.get('created_at'))}",
             f"status: {d['status']}",
             f"stop reason: {d['stop_reason'] or '-'}",
             f"iteration: {d['iteration']} ({d['runs_since_improvement']} since improvement)"]
    if d["void_reason"]:
        lines.append(f"VOID: {d['void_reason']}")
    b = d["best"]
    if b is None:
        lines.append("best: none")
    else:
        held = b["holdout_status"].replace("_", " ")
        if b["holdout_mean_rel_delta"] is not None:
            held += f" ({_pct(b['holdout_mean_rel_delta'])})"
        lines.append(f"best: #{b['iteration']} {_pct(b['mean_rel_delta'])} training, "
                     f"held-out {held}: {b['title']}")
    from talos.cli import _llm_spend_text

    lines.append("budget:")
    unpriced = (None if d["llm_priced"]
                else _llm_spend_text(job["provider"], job["model"], 0.0, why=True))
    for dim, label in (("iterations", "iterations"), ("hours", "hours"), ("usd", "llm"),
                       ("compute_usd", "compute")):
        c = d["budget"][dim]
        cap = "∞" if c["cap"] is None else _amount(dim, c["cap"])
        if dim == "usd" and unpriced:
            lines.append(f"  {label}: cap {cap}, spent {unpriced}")
            continue
        lines.append(f"  {label}: cap {cap}, spent {_amount(dim, c['spent'])}, "
                     f"left {'∞' if c['cap'] is None else _amount(dim, c['left'])}")
    act = d["activity"]
    if act["pending"] is not None:
        p = act["pending"]
        what = f"iteration {p['purpose']}" if isinstance(p["purpose"], int) else str(p["purpose"])
        lines.append(f"in flight: {what}" + (f" ({p['title']})" if p["title"] else ""))
    ev = act["last_event"]
    lines.append("last event: -" if ev is None
                 else f"last event: {_utc(ev['ts'])} {ev['text']}")
    lines.append(f"hardware: {d['hardware'] or 'default'}")
    return "\n".join(lines)


def format_iteration(d: dict) -> str:
    h, s = d["hypothesis"], d["scores"]
    lines = [f"job: {d['job_id']}",
             f"iteration: {d['iteration']}",
             f"parent: {d['parent_label']}",
             f"chain: {d['chain']}",
             f"kind: {d['kind']}",
             f"outcome: {d['outcome'] or '-'}"]
    if d["void_reason"]:
        lines.append(f"VOID: {d['void_reason']}")
    if d["error"]:
        lines.append(f"error: {d['error']}")
    lines += [f"title: {h['title']}",
              f"strategy: {h['strategy_tag'] or '-'}",
              f"design family: {h['design_family'] or '-'}",
              "description:",
              *(textwrap.fill(para, 98, initial_indent="  ", subsequent_indent="  ") or ""
                for para in (h["description"] or "-").splitlines()),
              f"training delta: {_pct(s['mean_rel_delta'])}"]
    if s["worst_track"] is not None:
        lines.append(f"worst track: {s['worst_track']} {_pct(s['worst_rel_delta'])}")
    if s["runtime_ratio"] is not None:
        lines.append(f"runtime ratio: x{s['runtime_ratio']:.2f}")
    if s["metered_rel_delta"] is not None:
        lines.append(f"metered delta: {_pct(s['metered_rel_delta'])}")
    if s["tracks"]:
        lines.append("per-track training deltas:")
        lines += [f"  {t} {_pct(v)}" for t, v in sorted(s["tracks"].items())]
    held = s["holdout_status"].replace("_", " ")
    if s["holdout_mean_rel_delta"] is not None:
        held += f" ({_pct(s['holdout_mean_rel_delta'])})"
    lines.append(f"held-out: {held}")
    for t in s["tables"] or []:
        lines.append(f"{t['heading']}:")
        if t["error"] is not None:
            lines.append(f"  no bundle delta: {t['error']}")
            continue
        rows = [[r["track"], f"{r['base_mean']:.6g}", f"{r['cand_mean']:.6g}",
                 _pct(r["rel_delta"]), str(r["cand_errors"]), str(r["n"])] for r in t["tracks"]]
        rows.append(["mean", "", "", _pct(t["mean_rel_delta"]), f"{t['error_rate']:.2%}", ""])
        table = _table(["track", "baseline mean", "best mean", "rel delta", "errors", "n"], rows)
        lines += [f"  {line}" for line in table.splitlines()]
    lines.append("files:")
    lines += [f"  {f}" for f in d["files"]] or ["  (none saved)"]
    return "\n".join(lines)
