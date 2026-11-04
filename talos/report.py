"""The per-run report.html: one self-contained page (inline CSS, SVG and JS) showing a job's
search, built only from the files already in its run directory, so it works on old runs and on
runs still in progress.

The data functions (report_nodes, node_diffs, axis_for, holdout_points, truncate) are pure.
run_sources and read_timeline read the run directory and nothing else.
"""
from __future__ import annotations

import html
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from talos.package import _diff
from talos.state import JobSpec, JobState, JobStore
from talos.base_algorithm import ROOT_DIR, read_base_root, root_label
from talos.template_start import read_template_root, root_names

KINDS = ("best", "false_positive", "demoted", "improved", "accepted", "rejected", "failed")

Sources = Callable[[int], "dict[str, str] | None"]


@dataclass(frozen=True)
class ReportNode:
    id: int
    parent: int
    chain: int
    family: str
    title: str
    description: str
    strategy_tag: str
    outcome: str
    error: str | None
    score: float | None
    tracks: dict[str, float] | None
    metered_score: float | None
    accepted: bool
    probability: float | None
    draw: float | None
    temperature: float | None
    kind: str
    former_best: bool
    confirmed: bool
    false_positive: bool
    se: float | None = None
    noise_ratio: float | None = None
    noise_band: str | None = None
    noise_state: str | None = None


def _kind(n: int, score: float | None, outcome: str, accepted: bool, state: JobState,
          demoted: set[int]) -> str:
    if state.best is not None and state.best.iteration == n:
        return "best"
    if n in state.false_positives:
        return "false_positive"
    if n in demoted:
        return "demoted"
    if score is None:
        return "failed"
    if outcome in ("improved", "won"):
        return "improved"
    return "accepted" if accepted else "rejected"


def report_nodes(spec: JobSpec, state: JobState) -> list[ReportNode]:
    """One node per hypothesis record, sorted by id; the root is not included. A stored lineage
    node with the same id supplies the tree fields; a record from before the lineage tree (or
    from before a resume onto it) supplies them itself."""
    stored = {n.id: n for n in state.nodes}
    demoted = {v.get("iteration") for v in state.validations if v.get("outcome") == "demoted"}
    out = []
    for h in state.hypotheses:
        i = h.get("iteration")
        if not isinstance(i, int):
            continue
        outcome = h.get("outcome", "")
        s = stored.get(i)
        if s is not None:
            parent, chain, family = s.parent, s.chain, s.family
            score, tracks, metered = s.rank_score(), s.tracks, s.metered_score
            accepted, probability, draw, temperature = (s.accepted, s.probability, s.draw,
                                                        s.temperature)
        else:
            parent, chain = h.get("against", 0), h.get("chain", 0)
            family = h.get("design_family") or h.get("strategy_tag", "")
            score, tracks, metered = h.get("mean_rel_delta"), None, None
            accepted = h.get("accepted", outcome in ("improved", "won", "accepted"))
            probability, draw, temperature = (h.get("probability"), h.get("draw"),
                                              h.get("temperature"))
        kind = _kind(i, score, outcome, accepted, state, demoted)
        out.append(ReportNode(
            id=i, parent=parent if isinstance(parent, int) else 0, chain=chain or 0,
            family=family or "", title=h.get("title", ""), description=h.get("description", ""),
            strategy_tag=h.get("strategy_tag", ""), outcome=outcome, error=h.get("error"),
            score=score, tracks=tracks, metered_score=metered, accepted=bool(accepted),
            probability=probability, draw=draw, temperature=temperature, kind=kind,
            former_best=bool(s is not None and s.best and kind != "best"),
            confirmed=i in state.confirmed, false_positive=i in state.false_positives,
            se=h.get("mean_rel_se"),
            noise_ratio=h.get("noise_ratio"), noise_band=h.get("noise_band"),
            noise_state=h.get("noise_state")))
    return sorted(out, key=lambda n: n.id)


def root_location(spec: JobSpec) -> str:
    """Where the root's files are in the run directory, for the truncation notes."""
    if spec.start == "base":
        return ROOT_DIR
    return "challenge/template.rs" if spec.start == "template" else "baseline"


def run_sources(run_dir: Path, spec: JobSpec, state: JobState) -> Sources:
    """node id -> that node's files, or None when they were not saved. The root is the
    baseline's files, the template as mod.rs in a template job, or the snapshot in a base job;
    the current best is
    state.best's files; any other node is what its iteration directory holds of the root's file
    names (edit and compile failures save none)."""
    run_dir = Path(run_dir)
    if spec.start == "template":
        root = read_template_root(run_dir / "challenge", spec.challenge)
        names: list[str] = root_names(spec.challenge)
    elif spec.start == "base":
        root = read_base_root(run_dir)
        names = sorted(root) if root else []
    else:
        root = dict(state.baseline.files) if state.baseline is not None else None
        names = sorted(root) if root else []
    cache: dict[int, dict[str, str] | None] = {}

    def lookup(node_id: int) -> dict[str, str] | None:
        if node_id == 0:
            return root
        if state.best is not None and state.best.iteration == node_id:
            return dict(state.best.files)
        if node_id not in cache:
            # Not JobStore.iteration_dir: it creates the directory.
            it_dir = run_dir / "iterations" / f"{node_id:04d}"
            files = {name: (it_dir / name).read_bytes().decode("utf-8", errors="replace")
                     for name in names if (it_dir / name).is_file()}
            cache[node_id] = files or None
        return cache[node_id]

    return lookup


def truncate(text: str, cap: int, note: str) -> str:
    """text cut to `cap` characters, inside a line if need be, then the note on its own line
    with `{n}` replaced by the number of lines (or rest of a line) left out."""
    if len(text) <= cap:
        return text
    kept = text[:cap]
    n = len(text[cap:].splitlines())
    return kept + ("" if kept.endswith("\n") else "\n") + note.replace("{n}", str(n))


def _location(node_id: int, spec: JobSpec) -> str:
    return root_location(spec) if node_id == 0 else f"iterations/{node_id:04d}"


def node_diffs(nodes: list[ReportNode], sources: Sources, spec: JobSpec,
               total: int = 1_000_000, per_node: int = 64_000) -> dict[int, str]:
    """Each node's diff against its parent, or a message saying why there is none. The diffs
    share `total` characters evenly, each capped at `per_node`."""
    label = root_label(spec.start)
    raw: dict[int, tuple[str, str | None]] = {}
    for n in nodes:
        mine = sources(n.id)
        if mine is None:
            raw[n.id] = ("no files saved", None)
            continue
        theirs = sources(n.parent)
        if theirs is None:
            raw[n.id] = ("parent's files not saved", None)
            continue
        text = _diff(theirs, mine, label if n.parent == 0 else f"#{n.parent}")
        if not text:
            raw[n.id] = ("no change", None)
            continue
        here = _location(n.id, spec)
        if n.parent == 0 and spec.start == "template":
            here += "/mod.rs"  # `diff -u file dir` looks for dir/<file's basename>
        raw[n.id] = (text, f"… truncated ({{n}} more lines); full diff: diff -u "
                           f"{_location(n.parent, spec)} {here}")
    count = sum(1 for _, note in raw.values() if note is not None)
    cap = min(per_node, total // max(1, count))
    return {i: text if note is None else truncate(text, cap, note)
            for i, (text, note) in raw.items()}


@dataclass(frozen=True)
class Axis:
    lo: float
    hi: float
    cut_lo: float | None
    cut_hi: float | None


def axis_for(scores: list[float]) -> Axis:
    """The y range of the search figure. It always includes 0. With at least 8 scores, a score
    more than 3 interquartile ranges (at least 0.01) beyond a quartile is cut: it is drawn on
    the edge and left out of the range, so one outlier cannot squash the other points."""
    cut_lo = cut_hi = None
    if len(scores) >= 8:
        q1, _, q3 = statistics.quantiles(scores, n=4, method="inclusive")
        w = max(q3 - q1, 0.01)
        lo_edge, hi_edge = q1 - 3 * w, q3 + 3 * w
        cut_lo = lo_edge if any(s < lo_edge for s in scores) else None
        cut_hi = hi_edge if any(s > hi_edge for s in scores) else None
    inside = [s for s in scores if (cut_lo is None or s >= cut_lo)
              and (cut_hi is None or s <= cut_hi)]
    lo = min([0.0, *inside])
    hi = max([0.0, *inside])
    if lo == hi:
        hi = lo + 0.01
    return Axis(lo, hi, cut_lo, cut_hi)


def read_timeline(path: Path) -> list[dict]:
    """timeline.jsonl's rows; unreadable lines are skipped, a missing file is empty."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def holdout_points(timeline: list[dict]) -> dict[int, float]:
    """iteration -> held-out mean delta, from `won` and `false_positive` events that carry one."""
    out = {}
    for row in timeline:
        ho = row.get("holdout")
        if (row.get("kind") in ("won", "false_positive") and isinstance(ho, dict)
                and isinstance(ho.get("mean_rel_delta"), (int, float))
                and isinstance(row.get("iteration"), int)):
            out[row["iteration"]] = float(ho["mean_rel_delta"])
    return out


# ── rendering ────────────────────────────────────────────────────────

BEST_DIFF_CAP = 400_000
_W, _H = 1000, 420  # SVG viewBox
_LEFT, _RIGHT, _TOP, _PLOT_BOTTOM = 64, 20, 18, 330
_STRIP_Y = 378  # the failure strip, below the x-axis tick labels
_KIND_LABELS = {"best": "current best", "improved": "improved", "accepted": "accepted",
                "rejected": "rejected (scored)", "demoted": "demoted on metered validation",
                "false_positive": "false positive (held-out)", "failed": "failed, no score"}


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _pct(x: float | None, places: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{places}%}"


def _utc(ts: float | None) -> str:
    if ts is None:
        return "n/a"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    minutes = int(max(0.0, seconds) // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _star(x: float, y: float, r: float = 9.0) -> str:
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append(f"{x + rr * math.cos(a):.1f},{y - rr * math.sin(a):.1f}")
    return f'<polygon class="m" points="{" ".join(pts)}"/>'


def _marker(kind: str, x: float, y: float, cut: str | None) -> str:
    if cut == "lo":
        return f'<path class="m" d="M{x - 6:.1f},{y - 5:.1f}H{x + 6:.1f}L{x:.1f},{y + 6:.1f}Z"/>'
    if cut == "hi":
        return f'<path class="m" d="M{x - 6:.1f},{y + 5:.1f}H{x + 6:.1f}L{x:.1f},{y - 6:.1f}Z"/>'
    if kind == "best":
        return _star(x, y)
    if kind == "failed":
        return (f'<path class="m x" d="M{x - 4.5:.1f},{y - 4.5:.1f}l9,9m-9,0l9,-9"/>')
    return f'<circle class="m" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'


def _ticks(lo: float, hi: float) -> list[float]:
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    return [first + k * step for k in range(int((hi - first) / step + 1e-9) + 1)]


def _tick_label(v: float, step: float) -> str:
    if abs(v) < step / 1000:
        return "0%"
    return f"{v * 100:+.4f}".rstrip("0").rstrip(".") + "%"


def _figure(spec: JobSpec, nodes: list[ReportNode], axis: Axis,
            holdout: dict[int, float], root_score: float | None = None) -> str:
    max_id = max((n.id for n in nodes), default=1) or 1

    def x(i: int) -> float:
        return _LEFT + (i / max_id) * (_W - _LEFT - _RIGHT)

    def y(s: float) -> float:
        s = min(max(s, axis.lo), axis.hi)
        return _TOP + (axis.hi - s) / (axis.hi - axis.lo) * (_PLOT_BOTTOM - _TOP)

    template = spec.start == "template"
    pos: dict[int, tuple[float, float]] = {0: (x(0), _STRIP_Y if template else y(root_score or 0.0))}
    cuts: dict[int, str | None] = {}
    for n in nodes:
        if n.score is None:
            pos[n.id], cuts[n.id] = (x(n.id), _STRIP_Y), None
        elif axis.cut_lo is not None and n.score < axis.cut_lo:
            pos[n.id], cuts[n.id] = (x(n.id), _PLOT_BOTTOM), "lo"
        elif axis.cut_hi is not None and n.score > axis.cut_hi:
            pos[n.id], cuts[n.id] = (x(n.id), _TOP), "hi"
        else:
            pos[n.id], cuts[n.id] = (x(n.id), y(n.score)), None

    out = [f'<svg id="fig" viewBox="0 0 {_W} {_H}" role="img" '
           f'aria-label="Training delta of every iteration, with lineage">']
    # axes, grid and the zero line
    step_ticks = _ticks(axis.lo, axis.hi)
    step = step_ticks[1] - step_ticks[0] if len(step_ticks) > 1 else axis.hi - axis.lo
    for t in step_ticks:
        out.append(f'<line class="grid" x1="{_LEFT}" x2="{_W - _RIGHT}" y1="{y(t):.1f}" '
                   f'y2="{y(t):.1f}"/><text class="tick" x="{_LEFT - 6}" y="{y(t) + 4:.1f}" '
                   f'text-anchor="end">{_esc(_tick_label(t, step))}</text>')
    out.append(f'<line class="zero" x1="{_LEFT}" x2="{_W - _RIGHT}" y1="{y(0.0):.1f}" '
               f'y2="{y(0.0):.1f}"/><text class="zlabel" x="{_W - _RIGHT - 4}" '
               f'y="{y(0.0) + 14:.1f}" text-anchor="end">mainnet baseline</text>')
    out.append(f'<line class="axis" x1="{_LEFT}" x2="{_W - _RIGHT}" y1="{_PLOT_BOTTOM + 8}" '
               f'y2="{_PLOT_BOTTOM + 8}"/>')
    xstep = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 10**9)
                 if max_id / s <= 12)
    for i in range(0, max_id + 1, xstep):
        out.append(f'<text class="tick" x="{x(i):.1f}" y="{_PLOT_BOTTOM + 22}" '
                   f'text-anchor="middle">{i}</text>')
    out.append(f'<rect class="stripbg" x="{_LEFT}" y="{_STRIP_Y - 11}" '
               f'width="{_W - _LEFT - _RIGHT}" height="22" rx="3"/>'
               f'<text class="tick" x="{_LEFT - 6}" y="{_STRIP_Y + 4}" text-anchor="end">'
               f'no score</text>')
    out.append(f'<text class="alabel" x="{(_LEFT + _W - _RIGHT) / 2:.0f}" y="{_H - 8}" '
               f'text-anchor="middle">iteration</text>'
               f'<text class="alabel" transform="translate(14 {(_TOP + _PLOT_BOTTOM) / 2:.0f}) '
               f'rotate(-90)" text-anchor="middle">training delta vs mainnet baseline</text>')
    if axis.cut_lo is not None:
        out.append(f'<text class="cutlabel" x="{_LEFT + 4}" y="{_PLOT_BOTTOM - 4}">'
                   f'▼ below {_esc(_pct(axis.cut_lo, 1))}</text>')
    if axis.cut_hi is not None:
        out.append(f'<text class="cutlabel" x="{_LEFT + 4}" y="{_TOP + 10}">'
                   f'▲ above {_esc(_pct(axis.cut_hi, 1))}</text>')
    # lineage edges
    ids = {n.id for n in nodes}
    for n in nodes:
        if n.parent != 0 and n.parent not in ids:
            continue
        (x1, y1), (x2, y2) = pos[n.parent], pos[n.id]
        cls = "edge solid" if n.accepted or n.outcome in ("improved", "won") else "edge"
        out.append(f'<line data-edge="{n.parent}-{n.id}" class="{cls}" x1="{x1:.1f}" '
                   f'y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    # best so far: the running max over nodes that became the job's best
    best_so_far, points = None, []
    for n in nodes:
        if n.score is None or n.outcome not in ("improved", "won"):
            continue
        if best_so_far is None or n.score > best_so_far:
            best_so_far = n.score
            points.append((x(n.id), y(n.score)))
    if points:
        d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
        for px, py in points[1:]:
            d += f"H{px:.1f}V{py:.1f}"
        d += f"H{x(max_id):.1f}"
        out.append(f'<path class="stepline" d="{d}"/>')
    # the root
    rx, ry = pos[0]
    label = root_label(spec.start)
    out.append(f'<g data-root class="root"><circle cx="{rx:.1f}" cy="{ry:.1f}" r="5"/>'
               f'<text x="{rx + 8:.1f}" y="{ry - 8:.1f}">{label}</text></g>')
    # held-out deltas
    for i, v in sorted(holdout.items()):
        if i not in ids:
            continue
        hx, hy = x(i), y(v)
        out.append(f'<path data-holdout="{i}" class="holdout" '
                   f'd="M{hx:.1f},{hy - 7:.1f}l7,7l-7,7l-7,-7Z"><title>#{i} held-out '
                   f'{_esc(_pct(v))}</title></path>')
    # one standard error either side of a scored node, behind the nodes; clipped to the plot
    for n in nodes:
        if n.noise_state != "ok" or n.score is None or n.se is None or cuts[n.id]:
            continue
        wx = pos[n.id][0]
        out.append(f'<line class="whisker" data-whisker="{n.id}" x1="{wx:.1f}" '
                   f'x2="{wx:.1f}" y1="{y(n.score + n.se):.1f}" '
                   f'y2="{y(n.score - n.se):.1f}"/>')
    # nodes, drawn last so they sit on top
    for n in nodes:
        px, py = pos[n.id]
        cls = f"node k-{n.kind}"
        if cuts[n.id]:
            cls += " cut"
        if n.kind == "failed":
            cls += " strip"
        ring = (f'<circle class="ring" cx="{px:.1f}" cy="{py:.1f}" r="9"/>'
                if n.former_best else "")
        out.append(f'<g data-node="{n.id}" class="{cls}">'
                   f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="10"/>{ring}'
                   f'{_marker(n.kind, px, py, cuts[n.id])}</g>')
    out.append("</svg>")
    return "".join(out)


def _legend() -> str:
    items = "".join(f'<li class="k-{k}"><span class="sw"></span>{_esc(v)}</li>'
                    for k, v in _KIND_LABELS.items())
    return (f'<ul class="legend" id="legend-outcome">{items}'
            '<li class="lh"><span class="sw"></span>held-out delta</li>'
            '<li class="ls"><span class="sw"></span>best so far</li>'
            '<li class="lr"><span class="sw"></span>former best</li></ul>'
            '<ul class="legend" id="legend-group" hidden></ul>')


def _kv(rows: list[tuple[str, str]]) -> str:
    return ('<table class="kv">' + "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>"
                                           for k, v in rows) + "</table>")


def _header(spec: JobSpec, state: JobState, timeline: list[dict],
            holdout: dict[int, float]) -> str:
    from talos.budget import effective_budget
    from talos.cli import _llm_spend_text
    from talos.package import _render_budget

    r = spec.redacted()
    start = r["start"]
    if spec.base_algorithm:
        start += f" ({spec.base_algorithm['source']} {spec.base_algorithm['name']})"
    eff = effective_budget(spec.budget, state.budget_extensions)
    stamps = [row["ts"] for row in timeline if isinstance(row.get("ts"), (int, float))]
    first, last = (stamps[0], stamps[-1]) if stamps else (None, None)
    best = state.best
    if best is None:
        best_text = held = "none"
    else:
        best_text = f"{_pct(best.delta.get('mean_rel_delta'))} (iteration #{best.iteration})"
        ho = holdout.get(best.iteration)
        ho_text = f" (held-out {_pct(ho)})" if ho is not None else ""
        if best.iteration in state.confirmed:
            held = "confirmed" + ho_text
        elif best.iteration in state.false_positives:
            held = "false positive" + (ho_text or " (held-out not scored)")
        else:
            held = "not scored"
    working = None if last is None else last - state.spend.started_at
    hours = ("n/a" if working is None else f"{working / 3600:.2f} h"
             + (f" of {eff.hours:g} h budget" if eff.hours is not None else ""))
    iterations = str(state.spend.iterations) + (f" of {eff.iterations}"
                                                if eff.iterations is not None else "")
    scoring = r["scoring"] + (f" (running {state.scoring})" if state.scoring
                              and state.scoring != r["scoring"] else "")
    budget = _render_budget(spec.budget) or "none"
    if state.budget_extensions:
        budget += f"; after extensions: {_render_budget(eff)}"
    rows = [("Challenge", f"{r['challenge']} ({r['challenge_id']})"), ("Job", r["job_id"]),
            ("Status", state.status), ("Stop reason", state.stop_reason or "n/a"),
            ("Best training delta", best_text), ("Held-out", held),
            ("Baseline", state.baseline.name if state.baseline else "not measured"),
            ("LLM spend", _llm_spend_text(r["provider"], r["model"], state.spend.llm_usd,
                                          why=True)),
            ("Compute spend", f"${state.spend.compute_usd:.2f} (estimate)"),
            ("Iterations", iterations), ("Started", _utc(first)), ("Last event", _utc(last)),
            ("Elapsed", _duration(None if first is None else last - first)),
            ("Working hours", hours)]
    settings = [("Provider", r["provider"]), ("Model", r["model"]), ("Mode", r["mode"]),
                ("Search", r["search"]), ("Chains", str(r["chains"])), ("Start", start),
                ("Scoring", scoring), ("Track focus", r["track"] or "all tracks"),
                ("Fuel", str(r["fuel"])), ("Tracks", ", ".join(r["tracks"])),
                ("Hardware", state.hardware or "default"), ("Budget", budget),
                ("Direction", r["direction"])]
    return (f'<header><h1>{_esc(r["challenge"])} <span class="muted">{_esc(r["job_id"])}'
            f'</span></h1><div class="cols">{_kv(rows)}{_kv(settings)}</div></header>')


def _track_tables(spec: JobSpec, state: JobState) -> str:
    from talos.package import score_pairs
    from talos.challenges import CHALLENGES
    from talos.scoring import NOISE_WORDS, ScoringError, bundle_delta, noise

    out = []
    for heading, base, cand in score_pairs(spec, state):
        out.append(f"<h3>{_esc(heading)}</h3><table class='grid'><tr><th>track</th>"
                   "<th>baseline mean</th><th>best mean</th><th>rel delta</th><th>errors</th>"
                   "<th>n</th><th>SE</th></tr>")
        try:
            d = bundle_delta(base, cand)
        except ScoringError as e:
            out.append(f"<tr><td colspan='7'>no bundle delta: {_esc(e)}</td></tr></table>")
            continue
        ratio, band, nstate = noise(d, CHALLENGES[spec.challenge].beat)
        for t in d.tracks:
            se = ("" if nstate == "over_ceiling"
                  else "n=1" if t.rel_se is None else f"±{t.rel_se:.3%}")
            out.append(f"<tr><td>{_esc(t.track)}</td><td>{t.base_mean:.6g}</td>"
                       f"<td>{t.cand_mean:.6g}</td><td>{_esc(_pct(t.rel_delta))}</td>"
                       f"<td>{t.cand_errors}</td><td>{t.n}</td>"
                       f"<td>{se}</td></tr>")
        text = NOISE_WORDS.get(nstate)
        if nstate == "ok":
            text = f"±{d.mean_rel_se:.3%}, {ratio:.1f}x {band}"
        # colour only a gain: a loss is not good news at any ratio
        cls = (" noise-" + (band if d.mean_rel_delta > 0 else "muted")) if nstate == "ok" else ""
        out.append(f"<tr class='total{cls}'><td>mean</td><td></td><td></td>"
                   f"<td>{_esc(_pct(d.mean_rel_delta))}</td>"
                   f"<td>{_esc(f'{d.error_rate:.2%}')}</td><td></td>"
                   f"<td class='nz'>{_esc(text or '')}</td></tr></table>")
    return "".join(out)


def _best_diff(spec: JobSpec, state: JobState, sources: Sources) -> str:
    root = sources(0)
    label = root_label(spec.start)
    if root is None:
        return "<p>The root's files were not found, so there is no diff.</p>"
    text = _diff(root, state.best.files, label)
    where = ("diff -u challenge/template.rs best/mod.rs" if spec.start == "template"
             else f"diff -u {root_location(spec)} best")
    # Counted before the cut, so a truncated diff still reports the whole change.
    changed = sum(1 for line in text.splitlines()
                  if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    text = truncate(text, BEST_DIFF_CAP, f"… truncated ({{n}} more lines); full diff: {where}")
    return (f'<details id="best-diff"><summary>#{state.best.iteration} vs '
            f'{label}: {changed} lines added or removed'
            f'</summary><pre class="diff">{html.escape(text, quote=False)}</pre></details>')


def _hypotheses(spec: JobSpec, state: JobState, ids: set[int]) -> str:
    from talos.package import _hypothesis_line

    rows = []
    for h in state.hypotheses:
        i = h.get("iteration")
        line = _hypothesis_line(spec, h)[2:]
        goto = f' data-goto="{i}"' if i in ids else ""
        rows.append(f"<li{goto}>{_esc(line)}</li>")
    return '<ol class="hyps">' + "".join(rows) + "</ol>"


def _node_json(nodes: list[ReportNode], diffs: dict[int, str]) -> str:
    ids = {n.id for n in nodes}
    data = []
    for n in nodes:
        d = asdict(n)
        d["diff"] = diffs.get(n.id, "")
        if n.parent != 0 and n.parent not in ids:
            d["parent_note"] = f"parent #{n.parent} not recorded"
        data.append(d)
    # `<` escaped so a "</script>" in any text cannot end the block
    return json.dumps(data).replace("<", "\\u003c")


def render_report(spec: JobSpec, state: JobState, timeline: list[dict],
                  sources: Sources) -> str:
    """The whole report.html. Pure: no I/O, no clock, no randomness."""
    holdout = holdout_points(timeline)
    parts = [_header(spec, state, timeline, holdout)]
    nodes: list[ReportNode] = []
    if state.baseline is None:
        parts.append('<p class="none">No baseline was measured, so there is nothing else to '
                     "show.</p>")
    else:
        nodes = report_nodes(spec, state)
        diffs = node_diffs(nodes, sources, spec)
        root_score = state.base_delta["mean_rel_delta"] if state.base_delta else None
        axis = axis_for([n.score for n in nodes if n.score is not None]
                        + ([] if root_score is None else [root_score]))
        chains = sorted({n.chain for n in nodes})
        families = sorted({n.family for n in nodes})
        options = ['<option value="kind">outcome</option>']
        if len(chains) > 1:
            options.append('<option value="chain">chain</option>')
        if len(families) > 1:
            options.append('<option value="family">design family</option>')
        toggle = (f'<label class="toggle">Colour by <select id="colour">{"".join(options)}'
                  "</select></label>" if len(options) > 1 else "")
        parts.append(
            f'<section><h2>Search</h2><div class="bar">{toggle}{_legend()}</div>'
            f'<div class="figwrap"><div class="fig">{_figure(spec, nodes, axis, holdout, root_score)}</div>'
            '<aside id="panel" hidden><button id="p-close" type="button" '
            'aria-label="Close">×</button><h3 id="p-title"></h3><p id="p-desc"></p>'
            '<dl id="p-fields"></dl><pre class="diff" id="p-diff"></pre></aside></div>'
            '<p class="hint">Click a point for its details and its diff against its parent.'
            "</p></section>")
        if state.best is None:
            parts.append('<section><p class="none">No candidate beat the baseline.</p>'
                         "</section>")
        else:
            parts.append(f"<section><h2>Per-track results, best vs baseline</h2>"
                         f"{_track_tables(spec, state)}</section>")
            parts.append(f"<section><h2>Best diff</h2>{_best_diff(spec, state, sources)}"
                         "</section>")
        parts.append(f"<section><h2>Hypotheses</h2>"
                     f"{_hypotheses(spec, state, {n.id for n in nodes})}</section>")
        parts.append(f'<script type="application/json" id="nodes">{_node_json(nodes, diffs)}'
                     "</script>")
    title = f"Talos report {spec.job_id}"
    body = "".join(parts)
    script = f"<script>{_JS}</script>" if nodes else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
            f'{body}<div id="tip" hidden></div>{script}</body></html>\n')


_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#d9d9dc;--panel:#fff;--code:#f3f3f1;
--best:#c98a00;--improved:#1f9d55;--accepted:#2f6fdb;--rejected:#9a9aa0;--demoted:#e07b00;
--false_positive:#e07b00;--failed:#d93b3b;--holdout:#7a4fd0;--step:#1f9d55;
--add:#e3f5e8;--addfg:#155d32;--del:#fbe6e6;--delfg:#8f1f1f;--hunk:#5a5fb0;--noise-clear:#155d32;
--noise-marginal:#9a6700;--noise-noise:#b42318;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--muted:#9a9aa2;
--line:#34343a;--panel:#1f1f23;--code:#232327;--best:#f2b84b;--improved:#4cc983;
--accepted:#6d9cf0;--rejected:#7c7c84;--demoted:#f59d3d;--false_positive:#f59d3d;
--failed:#f06a6a;--holdout:#b195f2;--step:#4cc983;--add:#173a24;--addfg:#a6e8bf;
--del:#45201f;--delfg:#f3b0ae;--hunk:#a3a7f0;--noise-clear:#4cc983;
--noise-marginal:#f2b84b;--noise-noise:#f06a6a;color-scheme:dark}}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0 auto;max-width:1400px;padding:16px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1{font-size:22px;margin:4px 0 12px}h2{font-size:17px;margin:28px 0 8px}
h3{font-size:14px;margin:16px 0 6px}.muted{color:var(--muted);font-weight:400;font-size:15px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:4px 24px}
table{border-collapse:collapse}.kv th{text-align:left;font-weight:500;color:var(--muted);
padding:2px 12px 2px 0;vertical-align:top;white-space:nowrap}.kv td{padding:2px 0;
vertical-align:top;overflow-wrap:anywhere}
table.grid{min-width:min(100%,560px)}table.grid th,table.grid td{border-bottom:1px solid
var(--line);padding:4px 10px;text-align:right}table.grid th:first-child,table.grid
td:first-child{text-align:left}table.grid tr.total td{font-weight:600}
table.grid tr.noise-clear td.nz{color:var(--noise-clear)}
table.grid tr.noise-marginal td.nz{color:var(--noise-marginal)}
table.grid tr.noise-noise td.nz{color:var(--noise-noise)}
table.grid tr.noise-muted td.nz{color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 20px;margin-bottom:6px}
.toggle select{font:inherit;margin-left:4px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:0;padding:0;list-style:none;
font-size:12px;color:var(--muted)}.legend li{display:flex;align-items:center;gap:5px}
.legend .sw{width:10px;height:10px;border-radius:50%;background:var(--c)}
.legend .k-false_positive .sw{background:none;border:2px solid var(--c)}
.legend .k-best .sw{border-radius:2px;transform:rotate(45deg)}
.legend .lh .sw{background:none;border:2px solid var(--holdout);border-radius:1px;
transform:rotate(45deg)}.legend .ls .sw{height:2px;border-radius:0;background:var(--step)}
.legend .lr .sw{background:none;border:1px solid var(--fg);opacity:.6}
.figwrap{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
@media (min-width:1100px){.figwrap.open{grid-template-columns:minmax(0,1fr) 440px}}
#fig{width:100%;height:auto;display:block}
#panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;
position:relative;min-width:0}#panel h3{margin:0 28px 6px 0;font-size:15px}
#p-close{position:absolute;top:6px;right:8px;border:0;background:none;color:var(--muted);
font-size:20px;cursor:pointer}#p-desc{margin:0 0 8px;white-space:pre-wrap}
#panel dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 10px;margin:0 0 8px;
font-size:13px}#panel dt{color:var(--muted)}#panel dd{margin:0;overflow-wrap:anywhere}
pre.diff{background:var(--code);border-radius:6px;padding:8px;margin:0;max-height:560px;
overflow:auto;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre.diff span{display:block;white-space:pre}pre.diff .add{background:var(--add);color:var(--addfg)}
pre.diff .del{background:var(--del);color:var(--delfg)}pre.diff .hunk{color:var(--hunk)}
pre.diff .note{color:var(--muted);font-style:italic}
details summary{cursor:pointer;margin-bottom:6px}
.hyps{padding-left:0;list-style:none;font-size:13px}.hyps li{padding:3px 6px;border-radius:4px}
.hyps li[data-goto]{cursor:pointer}.hyps li[data-goto]:hover{background:var(--code)}
.none{color:var(--muted);font-style:italic}.hint{color:var(--muted);font-size:12px;margin:4px 0}
#tip{position:fixed;z-index:9;pointer-events:none;background:var(--panel);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;max-width:360px}
svg text{fill:var(--muted);font-size:11px}svg .alabel{font-size:12px}
svg .grid{stroke:var(--line);stroke-width:.6}svg .zero{stroke:var(--fg);stroke-width:1;
stroke-dasharray:5 4;opacity:.7}svg .axis{stroke:var(--line)}svg .stripbg{fill:var(--code)}
svg .cutlabel{font-size:10px}
.k-best{--c:var(--best)}.k-improved{--c:var(--improved)}.k-accepted{--c:var(--accepted)}
.k-rejected{--c:var(--rejected)}.k-demoted{--c:var(--demoted)}
.k-false_positive{--c:var(--false_positive)}.k-failed{--c:var(--failed)}
.node .m{fill:var(--c);stroke:var(--c);stroke-width:1}.node .x{fill:none;stroke-width:2}
.node.k-false_positive .m{fill:none;stroke-width:2}.node.k-best .m{stroke:var(--bg)}
.node .ring{fill:none;stroke:var(--fg);stroke-width:1;opacity:.45}.node .hit{fill:transparent}
.node{cursor:pointer}.node.sel .hit{fill:var(--c);opacity:.25}
.edge{stroke:var(--muted);stroke-width:.8;opacity:.25}.edge.solid{opacity:.7;stroke-width:1.1}
.stepline{fill:none;stroke:var(--step);stroke-width:1.5;opacity:.55}
.holdout{fill:none;stroke:var(--holdout);stroke-width:1.6}
.whisker{stroke:var(--muted);stroke-width:1.2;opacity:.8}
.root circle{fill:var(--bg);stroke:var(--fg);stroke-width:1.5}.root text{fill:var(--fg)}
"""

_JS = r"""
(() => {
  "use strict";
  const nodes = JSON.parse(document.getElementById("nodes").textContent);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const tip = document.getElementById("tip");
  const panel = document.getElementById("panel");
  const wrap = document.querySelector(".figwrap");
  const fig = document.getElementById("fig");
  const PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
                   "#b07aa1", "#ff9da7", "#9c755f", "#8c8c8c"];
  const pct = (x) => x == null ? "n/a" : (x >= 0 ? "+" : "") + (x * 100).toFixed(3) + "%";
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const colourDiff = (pre, text) => {
    pre.textContent = "";
    const frag = document.createDocumentFragment();
    for (const line of text.split("\n")) {
      let cls = "";
      if (line.startsWith("@@")) cls = "hunk";
      else if (line.startsWith("+") && !line.startsWith("+++")) cls = "add";
      else if (line.startsWith("-") && !line.startsWith("---")) cls = "del";
      else if (line.startsWith("…")) cls = "note";
      frag.appendChild(el("span", cls, line || " "));
    }
    pre.appendChild(frag);
  };
  const field = (dl, k, v) => {
    if (v == null || v === "") return;
    dl.appendChild(el("dt", null, k));
    dl.appendChild(el("dd", null, String(v)));
  };
  const noiseText = (n) => {
    if (n.noise_state === "ok") {
      return "\u00b1" + (n.se * 100).toFixed(3) + "%, " + n.noise_ratio.toFixed(1) +
        "x noise, " + n.noise_band;
    }
    return {n1: "n=1", identical: "identical on every nonce",
            no_spread: "\u00b10, no spread"}[n.noise_state] || null;
  };
  let selected = null;
  const show = (id) => {
    const n = byId.get(id);
    if (!n) return;
    panel.hidden = false;
    wrap.classList.add("open");
    document.getElementById("p-title").textContent = "#" + n.id + " " + n.title;
    document.getElementById("p-desc").textContent = n.description;
    const dl = document.getElementById("p-fields");
    dl.textContent = "";
    field(dl, "parent", n.parent_note || (n.parent === 0 ? "root" : "#" + n.parent));
    field(dl, "chain", n.chain);
    field(dl, "family", n.family);
    field(dl, "strategy tag", n.strategy_tag);
    field(dl, "outcome", n.outcome);
    const flags = [];
    if (n.kind === "best") flags.push("current best");
    if (n.former_best) flags.push("former best");
    if (n.confirmed) flags.push("confirmed on held-out");
    if (n.false_positive) flags.push("false positive on held-out");
    if (n.kind === "demoted") flags.push("demoted on metered validation");
    field(dl, "status", flags.join(", "));
    field(dl, "error", n.error);
    field(dl, "training delta", pct(n.score));
    if (n.tracks) {
      field(dl, "per track", Object.keys(n.tracks).sort()
        .map((t) => t + " " + pct(n.tracks[t])).join(", "));
    }
    field(dl, "noise", noiseText(n));
    if (n.metered_score != null) field(dl, "metered", pct(n.metered_score));
    if (n.probability != null) field(dl, "accept probability", n.probability.toFixed(3));
    if (n.draw != null) field(dl, "draw", n.draw.toFixed(3));
    if (n.temperature != null) field(dl, "temperature", n.temperature.toFixed(4));
    colourDiff(document.getElementById("p-diff"), n.diff);
    if (selected) selected.classList.remove("sel");
    selected = fig.querySelector('[data-node="' + id + '"]');
    if (selected) selected.classList.add("sel");
    if (!window.matchMedia("(min-width: 1100px)").matches) {
      panel.scrollIntoView({behavior: "smooth", block: "start"});
    }
  };
  document.getElementById("p-close").addEventListener("click", () => {
    panel.hidden = true;
    wrap.classList.remove("open");
    if (selected) selected.classList.remove("sel");
  });
  fig.addEventListener("click", (e) => {
    const g = e.target.closest("[data-node]");
    if (g) show(Number(g.dataset.node));
  });
  fig.addEventListener("mousemove", (e) => {
    const g = e.target.closest("[data-node]");
    if (!g) { tip.hidden = true; return; }
    const n = byId.get(Number(g.dataset.node));
    const nz = noiseText(n);
    tip.textContent = "#" + n.id + " " + n.title + " (" + pct(n.score) +
      (nz ? ", " + nz : "") + ")";
    tip.hidden = false;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 370) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  fig.addEventListener("mouseleave", () => { tip.hidden = true; });
  for (const li of document.querySelectorAll("[data-goto]")) {
    li.addEventListener("click", () => {
      show(Number(li.dataset.goto));
      fig.scrollIntoView({behavior: "smooth", block: "start"});
    });
  }
  const best = document.getElementById("best-diff");
  if (best) {
    best.addEventListener("toggle", () => {
      const pre = best.querySelector("pre");
      if (best.open && !pre.dataset.done) {
        pre.dataset.done = "1";
        colourDiff(pre, pre.textContent);
      }
    });
  }
  const select = document.getElementById("colour");
  if (select) {
    const group = document.getElementById("legend-group");
    const outcome = document.getElementById("legend-outcome");
    select.addEventListener("change", () => {
      const by = select.value;
      const values = [...new Set(nodes.map((n) => n[by]))].sort((a, b) =>
        typeof a === "number" ? a - b : String(a).localeCompare(String(b)));
      const colour = (v) => PALETTE[values.indexOf(v) % PALETTE.length];
      for (const g of fig.querySelectorAll("[data-node]")) {
        const n = byId.get(Number(g.dataset.node));
        if (by === "kind") g.style.removeProperty("--c");
        else g.style.setProperty("--c", colour(n[by]));
      }
      outcome.hidden = by !== "kind";
      group.hidden = by === "kind";
      group.textContent = "";
      if (by !== "kind") {
        for (const v of values) {
          const li = el("li");
          const sw = el("span", "sw");
          sw.style.setProperty("--c", colour(v));
          li.appendChild(sw);
          li.appendChild(document.createTextNode(
            by === "chain" ? "chain " + v : (v === "" ? "(none)" : v)));
          group.appendChild(li);
        }
      }
    });
  }
})();
"""


def write_report(store: JobStore, spec: JobSpec | None = None,
                 state: JobState | None = None) -> Path:
    """Writes run_dir/package/report.html from the run's files and returns its path. Reads
    job.json and state.json only when spec and state are not given."""
    spec = spec if spec is not None else store.read_spec()
    state = state if state is not None else store.load()
    text = render_report(spec, state, read_timeline(store.run_dir / "timeline.jsonl"),
                         run_sources(store.run_dir, spec, state))
    pkg = store.run_dir / "package"
    pkg.mkdir(exist_ok=True)
    path = pkg / "report.html"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
