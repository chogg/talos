"""Build the hand-back package: files, diff, per-nonce tables, hypothesis log, evidence draft."""
from __future__ import annotations

import difflib
import json
import shutil
import sys
from importlib import resources
from pathlib import Path

from talos.budget import ADD_DIMENSIONS, Budget, effective_budget
from talos.challenges import CHALLENGES, BeatRule
from talos.lineage import Node
from talos.prompts import in_tree, parent_label
from talos.scoring import (
    NOISE_WORDS, BundleDelta, ScoringError, beats, bundle_delta, focus_sets, noise, select)
from talos.state import JobSpec, JobState, JobStore
from talos.base_algorithm import job_void_reason, read_base_root
from talos.leak_check import KNOWN_LEAKING
from talos.template_start import read_template_root
from talos.types import NonceResult


def scores_markdown(baseline: list[NonceResult], candidate: list[NonceResult],
                    rule: BeatRule | None = None) -> str:
    rows = ["| track | nonce | baseline | candidate | note |", "|---|---|---|---|---|"]
    by_b = {(r.track, r.nonce): r for r in baseline}
    for c in sorted(candidate, key=lambda r: (r.track, r.nonce)):
        b = by_b.get((c.track, c.nonce))
        bq = b.quality if b and b.ok else (b.error if b else "missing")
        cq = c.quality if c.ok else c.error
        if not c.ok or (b is not None and not b.ok):
            note = "err"
        elif b and b.ok and c.ok:
            note = "+" if c.quality > b.quality else ("=" if c.quality == b.quality else "-")
        else:
            note = ""
        rows.append(f"| {c.track} | {c.nonce} | {bq} | {cq} | {note} |")
    try:
        d = bundle_delta(baseline, candidate)
        summary = (f"\nmean_rel_delta: {d.mean_rel_delta:+.4%}  worst_rel_delta: "
                   f"{d.worst_rel_delta:+.4%}  error_rate: {d.error_rate:.2%}\n")
        if rule is not None:
            summary += _noise_lines(d, rule)
    except ScoringError as e:
        summary = f"\n(no bundle delta: {e})\n"
    return "\n".join(rows) + "\n" + summary


def _noise_lines(d: BundleDelta, rule: BeatRule) -> str:
    """The SE and noise ratio lines under a table's summary; display only. Nothing when the
    error rate is over the ceiling, as for the other noise displays."""
    ratio, band, state = noise(d, rule)
    if state == "over_ceiling":
        return ""
    if state == "ok":
        head = f"mean_rel_se: ±{d.mean_rel_se:.4%}  noise: {ratio:.1f}x ({band})"
    else:
        head = f"noise: {NOISE_WORDS[state]}"
    # 0.0 is a legitimate SE (no spread), so test for None, not falsiness
    tracks = "; ".join(f"{t.track}: n=1" if t.rel_se is None
                       else f"{t.track}: ±{t.rel_se:.2%} (n={t.n})" for t in d.tracks)
    return f"{head}\nper track: {tracks}\n"


def _diff(base: dict[str, str], cand: dict[str, str], label: str = "baseline") -> str:
    out = []
    for name in sorted(set(base) | set(cand)):
        a = base.get(name, "").splitlines(keepends=True)
        b = cand.get(name, "").splitlines(keepends=True)
        for line in difflib.unified_diff(a, b, fromfile=f"{label}/{name}",
                                         tofile=f"candidate/{name}"):
            # A file's last line with no newline: difflib leaves it unterminated, so the next
            # line runs onto it. diff and git mark it instead (issue #86).
            out.append(line if line.endswith("\n")
                       else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def score_pairs(spec: JobSpec, state: JobState
                ) -> list[tuple[str, list[NonceResult], list[NonceResult]]]:
    """(heading, baseline rows, candidate rows) per scores table. Focused jobs get the track's
    training and held-out rows plus a guard pair for the other tracks; unfocused jobs keep the
    two plain headings."""
    base, best = state.baseline, state.best
    training_sets, holdout_sets = focus_sets(spec.track, spec.training, spec.holdout)
    if spec.track is None:
        out = [("Training nonces", base.training, best.training)]
        if best.holdout:
            out.append(("Held-out nonces", base.holdout, best.holdout))
        return out
    t = spec.track
    out = [(f"Training nonces (track {t})", select(base.training, training_sets),
            best.training)]
    if best.holdout:
        focus_ho = [s for s in holdout_sets if s.track == t]
        guards = [s for s in holdout_sets if s.track != t]
        out.append((f"Held-out nonces (track {t})", select(base.holdout, focus_ho),
                    select(best.holdout, focus_ho)))
        if guards:
            out.append(("Regression guard (other tracks, training nonces)",
                        select(base.training, guards), select(best.holdout, guards)))
    return out


def scores_sections(spec: JobSpec, state: JobState, *, with_noise: bool = False
                    ) -> list[tuple[str, str]]:
    """(heading, table) pairs for scores.md and the evidence appendix. with_noise adds the
    SE and noise ratio lines (scores.md only; the evidence draft stays as it was)."""
    rule = CHALLENGES[spec.challenge].beat if with_noise else None
    return [(h, scores_markdown(b, c, rule)) for h, b, c in score_pairs(spec, state)]


def evidence_draft(spec: JobSpec, state: JobState) -> str:
    template = (resources.files("talos.data").joinpath("evidence_template.md")
                .read_text(encoding="utf-8"))
    # deviation from brief: also include outcome "won" (a winning iteration's hypothesis
    # record carries outcome "won", not "improved" — see JobState.confirmed, Task 11)
    winners = [h for h in state.hypotheses if h.get("outcome") in ("improved", "won")]
    if state.nodes and in_tree(spec.start, spec.search, spec.chains):
        # In a tree, improvements on other chains or branches are not part of the best's method:
        # only the steps from the root down to the best node are. A greedy one-chain mainnet job
        # keeps the list above, which is the same path there and also covers iterations from
        # before nodes were recorded.
        by_iteration = {h.get("iteration"): h for h in state.hypotheses}
        winners = [by_iteration[n.id] for n in _best_path(state) if n.id in by_iteration]
    method = "\n".join(f"- {h['title']}: {h['description']}" for h in winners) or "(none recorded)"
    filled = template.replace(
        "PLEASE IDENTIFY WHICH TIG CHALLENGE THE METHOD ADDRESSES.\n\n> YOUR RESPONSE HERE",
        f"PLEASE IDENTIFY WHICH TIG CHALLENGE THE METHOD ADDRESSES.\n\n> {spec.challenge} "
        f"({spec.challenge_id})", 1)
    filled = filled.replace(
        "PLEASE DESCRIBE THE METHOD THAT YOU HAVE SELECTED FOR ASSESSMENT.\n\n> YOUR RESPONSE HERE",
        f"PLEASE DESCRIBE THE METHOD THAT YOU HAVE SELECTED FOR ASSESSMENT.\n\n> Draft from the "
        f"Talos hypothesis log; rewrite as a discrete method:\n{method}", 1)
    bench = ""
    if state.best and state.baseline:
        focus = f" Optimised for track {spec.track}." if spec.track else ""
        bench = ("\n\n## TALOS BENCHMARK APPENDIX (auto-generated)\n\n"
                 f"Baseline: mainnet `{state.baseline.name}` at monorepo `{spec.monorepo_ref}`, "
                 f"fuel {spec.fuel}, tracks {', '.join(spec.tracks)}.{focus}\n\n"
                 f"{_comparison_note(spec, state)}")
        for heading, table in scores_sections(spec, state):
            bench += f"\n### {heading}\n\n{table}"
    return filled + bench


def _render_budget(b: Budget) -> str:
    return ", ".join(f"{d} {getattr(b, d)}" for d in ("usd", "hours", "iterations", "compute_usd")
                     if getattr(b, d) is not None)


def _budget_section(spec: JobSpec, state: JobState) -> str:
    """The budget the job started with, each `--add-*` extension, and the total it ran on;
    empty for a job never extended. job.json shows only the starting budget."""
    if not state.budget_extensions:
        return ""
    lines = ["## Budget extensions", "", f"- Started with: {_render_budget(spec.budget)}"]
    for e in state.budget_extensions:
        added = ", ".join(f"+{e['add'][d]} {d}" for d in ADD_DIMENSIONS
                          if e["add"].get(d) is not None)
        lines.append(f"- Extended at iteration {e['iteration']} "
                     f"(was {e['previous_status']}): {added}")
    now = effective_budget(spec.budget, state.budget_extensions)
    lines.append(f"- Now: {_render_budget(now)}")
    return "\n".join(lines) + "\n\n"


def _base_note(spec: JobSpec, state: JobState) -> str:
    """Where a base job started, once that code has been scored."""
    base = spec.base_algorithm
    if base is None or state.base_delta is None:
        return ""
    return (f"Started from the {base['source']} algorithm `{base['name']}`, which scored "
            f"{state.base_delta['mean_rel_delta']:+.3%} against the baseline.\n\n")


def _closest_note(spec: JobSpec, state: JobState) -> str:
    """A base job can climb without reaching the baseline. That code is not a result, but it is
    the budget's work, so the hand-back says where it is."""
    if spec.start != "base":
        return ""
    kept = [n for n in state.nodes if n.accepted and n.score is not None]
    if not kept:
        return ""
    top = max(kept, key=lambda n: n.score)
    return (f"The closest candidate was #{top.id}, at {top.score:+.3%} against the baseline. It "
            f"is not a result, but its code is in `iterations/{top.id:04d}/`.\n\n")


def _comparison_note(spec: JobSpec, state: JobState) -> str:
    """Scores from a challenge whose mainnet rankings a leak has distorted mean one thing only."""
    if spec.challenge not in KNOWN_LEAKING or state.baseline is None:
        return ""
    return (f"Every delta here is against `{state.baseline.name}` rebuilt at monorepo "
            f"`{spec.monorepo_ref}` on the same nonces. Do not compare these scores with mainnet "
            "scores: mainnet may still allow the hidden_seed leak.\n\n")


def _readme_void(state: JobState, void: str) -> str:
    return ("# Talos hand-back\n\n"
            f"This job is void: {void}. Its best candidate, from iteration "
            f"{state.best.iteration}, must not be submitted, and its scores are not results. "
            "`report.html` is kept to show what happened.\n\n"
            "There is nothing to submit from this job. Start a new one.\n")


def _readme_no_candidate(spec: JobSpec, state: JobState) -> str:
    baseline = f" Baseline: mainnet `{state.baseline.name}`." if state.baseline else ""
    outcome = ("No candidate improved on the base algorithm" if state.base_leads
               else "No candidate beat the baseline")
    return ("# Talos hand-back\n\n"
            f"{outcome}.{baseline} Status: {state.status}. "
            f"Reason: {state.stop_reason}\n\n"
            + _base_note(spec, state) + _closest_note(spec, state)
            + _comparison_note(spec, state) + _budget_section(spec, state) +
            "There is nothing to submit from this job.\n")


def _won_training(spec: JobSpec, state: JobState) -> bool:
    try:
        training_sets, _ = focus_sets(spec.track, spec.training, spec.holdout)
        return beats(select(state.baseline.training, training_sets), state.best.training,
                     CHALLENGES[spec.challenge].beat)
    except (ScoringError, KeyError):
        return False


def _best_path(state: JobState) -> list[Node]:
    """Nodes from the root's child down to the best node, following parent links."""
    by_id = {n.id: n for n in state.nodes}
    node = by_id.get(state.best.iteration) if state.best else None
    if node is None or not node.best:
        return []
    path = []
    while node is not None and len(path) <= len(by_id):
        path.append(node)
        node = by_id.get(node.parent)
    return path[::-1]


def lineage_markdown(spec: JobSpec, state: JobState) -> str:
    lines = ["# Lineage", "", "## Path from the root to the best node", ""]
    path = _best_path(state)
    for n in path:
        score = n.rank_score()
        shown = "unscored" if score is None else f"{score:+.3%} vs mainnet"
        lines.append(f"- #{n.id} [{n.family}] {n.title}: {shown}")
    if not path:
        lines.append("(no best node)")
    lines += ["", "## Families", "", "| family | nodes | accepted | best score |",
              "|---|---|---|---|"]
    for family in sorted({n.family for n in state.nodes}):
        members = [n for n in state.nodes if n.family == family]
        scores = [s for s in (n.rank_score() for n in members) if s is not None]
        best = f"{max(scores):+.3%}" if scores else "n/a"
        lines.append(f"| {family} | {len(members)} | {sum(n.accepted for n in members)} "
                     f"| {best} |")
    demoted = sum(1 for v in state.validations if v.get("outcome") == "demoted")
    lines += ["", f"Metered validations: {len(state.validations)} ({demoted} demoted)", ""]
    return "\n".join(lines)


def _hyperparameters_section(spec: JobSpec) -> str:
    if spec.hyperparameters is None:
        return ""
    if spec.start != "mainnet":
        lines = [f"- {t}: " + ("none" if spec.hyperparameters.get(t) is None
                               else f"`{json.dumps(spec.hyperparameters[t], sort_keys=True)}`")
                 for t in spec.tracks]
        return ("\n## Hyperparameters\n\nThe baseline ran with these per-track hyperparameters; "
                "the candidate ran with none. They are not shipped as a file, because the "
                "candidate was not tuned for them.\n\n" + "\n".join(lines) + "\n")
    lines = []
    for track in spec.tracks:
        hp = spec.hyperparameters.get(track)
        src = (spec.hyperparameters_source or {}).get(track)
        value = "none" if hp is None else f"`{json.dumps(hp, sort_keys=True)}`"
        origin = (f" (mainnet benchmark {src['benchmark_id']} by {src['player_id']}, "
                  f"mean quality {src['mean_quality']:g})" if src else "")
        lines.append(f"- {track}: {value}{origin}")
    return ("\n## Hyperparameters\n\nThe baseline and every candidate ran with these per-track "
            "hyperparameters, from the best mainnet benchmark of the baseline algorithm. The "
            "measured improvement holds only with them: benchmark with the same values (they are "
            "also in hyperparameters.json).\n\n" + "\n".join(lines) + "\n")


def _readme(spec: JobSpec, state: JobState, template_missing: bool = False) -> str:
    confirmed = state.best.iteration in state.confirmed
    false_positive = state.best.iteration in state.false_positives
    head = ("# Talos hand-back\n\n"
            f"Challenge: {spec.challenge}. Baseline: mainnet `{state.baseline.name}`. "
            f"Status: {state.status}. Reason: {state.stop_reason}\n\n")
    head += _budget_section(spec, state)
    if spec.track:
        head += (f"Optimised for track {spec.track}; the other tracks were re-scored on "
                 "confirmation as a regression guard (see the last section of scores.md).\n\n")
    if confirmed:
        head += ("This candidate beat the baseline on both the training and the held-out nonce "
                 "sets. See scores.md.\n\n")
    elif false_positive:
        head += ("This candidate beat the baseline on the training nonces but NOT on the "
                 "held-out nonces (a false positive); see scores.md.\n\n")
    elif _won_training(spec, state):
        head += ("This candidate beat the baseline on the training nonces but was never scored "
                 "on the held-out nonces: the run stopped before the confirmation. Treat it as "
                 "unconfirmed. See scores.md.\n\n")
    else:
        head += ("This candidate was never scored on the held-out nonces; it did not beat the "
                 "baseline on training.\n\n")
    if spec.start == "template":
        head += ("Written from TIG's empty template; the baseline's code was never shown to the "
                 "LLM.\n\n")
        if template_missing:
            head += "The template copy was not found, so no diff against it is included.\n\n"
    head += _base_note(spec, state) + _comparison_note(spec, state)
    base = spec.base_algorithm
    # The mainnet code this candidate descends from, whose identifier it may carry.
    uai = ""
    if spec.start == "mainnet" or (base and base["source"] == "mainnet"):
        whose, name = (("the base algorithm", base["name"]) if base
                       else ("the baseline", state.baseline.name))
        uai = (f" If {whose}'s own README on mainnet "
               f"(`tig-algorithms/src/{spec.challenge}/{name}/README.md`) lists a "
               "Unique Algorithm Identifier and this candidate still embodies that method, carry "
               "it over.")
    head += ("## Submitting\n\n1. Fill out the template `submission/README.md`: algorithm name, "
             f"copyright, submitter.{uai}\n"
             "2. Copy the contents of `submission/` into "
             f"`tig-algorithms/src/{spec.challenge}/<your_name>/` in a monorepo checkout.\n"
             "3. Follow docs/guides in the monorepo to submit. For Advance Rewards, complete "
             "evidence_draft.md.\n")
    return head + _hyperparameters_section(spec)


def _hypothesis_line(spec: JobSpec, h: dict) -> str:
    against = h.get("against")
    if in_tree(spec.start, spec.search, spec.chains) and isinstance(against, int):
        parent = f"vs {parent_label(against, spec.start, True)}"
    else:
        parent = f"vs best #{h.get('against', '?')}"
    line = (f"- #{h['iteration']} ({parent}) [{h.get('strategy_tag', '')}] "
            f"{h.get('title', '')}: {h.get('description', '')} -> {h.get('outcome', '')}")
    error = h.get("error")
    if error:
        line += f" — {error[:200]}"
    return line


SUBMISSION_DIR = "submission"


def write_submission(spec: JobSpec, files: dict[str, str], dest: Path) -> None:
    """The candidate's code plus TIG's blank submission README, laid out as an algorithm folder
    under tig-algorithms/src/<challenge>/. The algorithm's own README.md is the mainnet author's
    submission header (their copyright), so the template is written over it."""
    for name, text in files.items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text(text, encoding="utf-8", newline="\n")
    template = (resources.files("talos.data").joinpath("submission_readme_template.md")
                .read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(template.replace("{challenge}", spec.challenge),
                                    encoding="utf-8", newline="\n")


def _write_report_safely(spec: JobSpec, state: JobState, store: JobStore) -> None:
    """report.html is a view of the run; failing to write it must not cost the user the
    package, so the error is reported and the zip is still written."""
    try:
        # Imported here, not at the top: talos.report imports this module.
        from talos.report import write_report

        write_report(store, spec, state)
    except Exception as e:
        print(f"report.html not written: {e}", file=sys.stderr)


def build_package(spec: JobSpec, state: JobState, store: JobStore) -> Path:
    pkg = store.run_dir / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir()
    if state.nodes:
        (pkg / "lineage.json").write_text(
            json.dumps({"nodes": [n.to_dict() for n in state.nodes],
                        "chains": [c.to_dict() for c in state.chains]}, indent=1),
            encoding="utf-8", newline="\n")
        (pkg / "lineage.md").write_text(lineage_markdown(spec, state),
                                        encoding="utf-8", newline="\n")
    if state.best is None:
        (pkg / "README.md").write_text(_readme_no_candidate(spec, state),
                                       encoding="utf-8", newline="\n")
        _write_report_safely(spec, state, store)
        shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
        return pkg
    if state.baseline is None:
        (pkg / "README.md").write_text("# Talos hand-back\n\nNo candidate was produced.\n",
                                       encoding="utf-8", newline="\n")
        _write_report_safely(spec, state, store)
        shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
        return pkg
    void = job_void_reason(spec, state, store.run_dir)
    if void:
        (pkg / "README.md").write_text(_readme_void(state, void), encoding="utf-8", newline="\n")
        _write_report_safely(spec, state, store)
        shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
        return pkg
    write_submission(spec, state.best.files, pkg / SUBMISSION_DIR)
    template_missing = False
    if spec.start == "template":
        root = read_template_root(store.run_dir / "challenge", spec.challenge)
        template_missing = root is None
        if not template_missing:
            (pkg / "diff_vs_template.patch").write_text(
                _diff(root, state.best.files, "template"), encoding="utf-8", newline="\n")
    elif spec.start == "base":
        root = read_base_root(store.run_dir)
        if root is not None:
            (pkg / "diff_vs_base.patch").write_text(
                _diff(root, state.best.files, "base"), encoding="utf-8", newline="\n")
    else:
        (pkg / "diff_vs_baseline.patch").write_text(
            _diff(state.baseline.files, state.best.files), encoding="utf-8", newline="\n")
    scores = "\n".join(f"# {heading}\n\n{table}"
                       for heading, table in scores_sections(spec, state, with_noise=True))
    (pkg / "scores.md").write_text(scores, encoding="utf-8", newline="\n")
    hyps = "\n".join(_hypothesis_line(spec, h) for h in state.hypotheses)
    (pkg / "hypotheses.md").write_text("# Hypotheses\n\n" + hyps + "\n",
                                       encoding="utf-8", newline="\n")
    (pkg / "evidence_draft.md").write_text(evidence_draft(spec, state),
                                           encoding="utf-8", newline="\n")
    (pkg / "README.md").write_text(_readme(spec, state, template_missing),
                                   encoding="utf-8", newline="\n")
    if spec.hyperparameters is not None and spec.start == "mainnet":
        (pkg / "hyperparameters.json").write_text(json.dumps(spec.hyperparameters, indent=1),
                                                  encoding="utf-8", newline="\n")
    _write_report_safely(spec, state, store)
    shutil.make_archive(str(store.run_dir / "package"), "zip", pkg)
    return pkg
