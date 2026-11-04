"""talos/inspect_run.py: read-only views of a run directory for status, show and diff."""
import json
import os
import sys

import pytest

from talos.inspect_run import (InspectError, SCHEMA_VERSION, format_iteration,
                               format_job_detail, format_status_table, holdout_status,
                               hours_spent, iteration_detail, iteration_diff, job_detail,
                               last_event, load_run, resolve_node, status_rows)
from tests.run_fixture import T0, make_run


def _run(tmp_path, name="J", **kw):
    make_run(tmp_path / "runs" / name, kw.pop("hyps", [{"iteration": 1}]), **kw)
    return load_run(tmp_path / "runs", name)


# ── loading ──────────────────────────────────────────────────────────

def test_load_run_reads_without_creating(tmp_path):
    # mutation: load_run constructing JobStore(run_dir) creates runs/nope
    with pytest.raises(InspectError, match=r"^no job nope under runs/$"):
        load_run(tmp_path / "runs", "nope")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("bad", ["../J", "a/b", "a\\b", "..", ".", ""])
def test_load_run_rejects_path_like_ids(tmp_path, bad):
    # mutation: dropping the separator check lets "../J" resolve to a sibling directory
    make_run(tmp_path / "J", [{"iteration": 1}])  # a real run OUTSIDE runs/
    (tmp_path / "runs").mkdir()
    with pytest.raises(InspectError, match="under runs/"):
        load_run(tmp_path / "runs", bad)


def test_load_run_names_the_missing_file(tmp_path):
    # mutation: treating a missing state.json as "no job" hides a job that crashed early
    d = tmp_path / "runs" / "J"
    make_run(d, [{"iteration": 1}])
    (d / "state.json").unlink()
    with pytest.raises(InspectError, match=r"^job J has no state.json$"):
        load_run(tmp_path / "runs", "J")
    (d / "state.json").write_text("{not json")
    with pytest.raises(InspectError, match=r"^job J: cannot read state.json"):
        load_run(tmp_path / "runs", "J")


def test_holdout_status_three_ways(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": i} for i in (1, 2, 3)], best=3,
               confirmed=[1], false_positives=[3])
    assert [holdout_status(run.state, i) for i in (1, 3, 2)] == \
        ["confirmed", "false_positive", "not_scored"]


# ── hours and events ─────────────────────────────────────────────────

@pytest.mark.parametrize("status", ["exhausted", "paused"])
def test_hours_spent_stopped_job_uses_last_event(tmp_path, status):
    # mutation: using `now` for a stopped job makes a finished run's hours grow forever;
    # the "paused" case catches dropping `or status == "paused"`
    run = _run(tmp_path, status=status, timeline=[{"kind": "a"}] * 4)  # last ts T0+180
    assert hours_spent(run, now=T0 + 10**6) == pytest.approx(180 / 3600)


def test_hours_spent_running_job_uses_now(tmp_path):
    # mutation: using the last event for a running job under-reports time in a long iteration
    run = _run(tmp_path, status="researching", timeline=[{"kind": "a"}] * 4)
    assert hours_spent(run, now=T0 + 7200) == pytest.approx(2.0)


def test_hours_spent_stopped_job_without_timeline_is_none(tmp_path):
    run = _run(tmp_path, status="failed", timeline=None)
    assert hours_spent(run, now=T0 + 7200) is None


def test_last_event_skips_a_torn_line(tmp_path):
    run_dir = tmp_path / "runs" / "J"
    make_run(run_dir, [{"iteration": 1}],
             timeline=[{"kind": "scored", "iteration": 1, "mean_rel_delta": 0.01,
                        "worst_rel_delta": 0.0, "error_rate": 0.0}])
    with (run_dir / "timeline.jsonl").open("a") as f:
        f.write('{"ts": 1, "kind": "hypo')  # torn write, no newline
    ev = last_event(load_run(tmp_path / "runs", "J").timeline)
    assert ev["kind"] == "scored" and ev["ts"] == T0 and ev["iteration"] == 1
    assert ev["text"].startswith("scored +1.000% vs baseline")


def test_last_event_text_falls_back_to_the_kind(tmp_path):
    # mutation: letting _event_line's KeyError escape on an old row that lacks a field
    ev = last_event([{"ts": T0, "kind": "scored", "iteration": 1}])
    assert ev == {"kind": "scored", "ts": T0, "iteration": 1, "text": "scored"}


# ── status rows ──────────────────────────────────────────────────────

def test_status_rows_one_per_job_including_unreadable(tmp_path):
    # mutation: a bare `except` that skips the bad dir, or iterating only */state.json
    runs = tmp_path / "runs"
    make_run(runs / "A", [{"iteration": 1}], best=1, timeline=[{"kind": "x"}])
    make_run(runs / "B", [{"iteration": 1}])
    (runs / "B" / "state.json").unlink()
    make_run(runs / "C", [{"iteration": 1}])
    (runs / "C" / "state.json").write_text("{")
    (runs / "D").mkdir()  # nothing saved yet: no row
    # replaces PR #82's test_status_without_a_readable_job_json_still_prints_the_job:
    # an incomplete job.json must give a row, not a traceback
    make_run(runs / "E", [{"iteration": 1}])
    (runs / "E" / "job.json").write_text('{"provider": "x"}')
    rows = status_rows(runs)
    assert [r["job_id"] for r in rows] == ["A", "B", "C", "E"]
    a, b, c, e = rows
    assert e["status"] == "unreadable" and e["error"].startswith("job E: cannot read job.json")
    assert a["status"] == "exhausted" and a["best_mean_rel_delta"] == 0.105
    assert a["last_event_ts"] == T0 and a["error"] is None and a["llm_usd"] == 0.0
    assert b["status"] == "unreadable" and b["error"] == "job B has no state.json"
    assert c["status"] == "unreadable" and c["error"].startswith("job C: cannot read state.json")
    assert set(b) == set(a)  # same keys, nulls not missing keys


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                    reason="chmod 000 does not deny access on Windows or to root")
def test_status_rows_survive_a_directory_without_permission(tmp_path):
    # spec anti-goal: one unreadable run directory must not stop the others being listed.
    # mutation: probing job.json outside the try lets PermissionError escape status_rows
    runs = tmp_path / "runs"
    make_run(runs / "A", [{"iteration": 1}])
    make_run(runs / "B", [{"iteration": 1}])
    (runs / "B").chmod(0)
    try:
        rows = status_rows(runs)
    finally:
        (runs / "B").chmod(0o755)
    assert [r["job_id"] for r in rows] == ["A", "B"]
    assert rows[1]["status"] == "unreadable" and "B" in rows[1]["error"]


def test_status_rows_missing_runs_dir_is_empty(tmp_path):
    assert status_rows(tmp_path / "runs") == []


def test_unpriced_spend_is_null_not_zero(tmp_path):
    # mutation: reporting state.spend.llm_usd (0.0) for codex-cli
    make_run(tmp_path / "runs" / "J", [{"iteration": 1}], provider="codex-cli", model="x")
    row, = status_rows(tmp_path / "runs")
    assert row["llm_usd"] is None and row["llm_priced"] is False
    assert row["llm_unpriced_reason"] == "codex-cli does not report what its calls cost"
    d = job_detail(load_run(tmp_path / "runs", "J"), now=T0)
    assert d["budget"]["usd"] == {"cap": 1.0, "spent": None, "left": None}


# ── job detail ───────────────────────────────────────────────────────

def test_job_detail_budget_and_best(tmp_path):
    # Fixture budget: usd 1.0, hours 2.0, iterations 10, compute_usd None.
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}], best=2, confirmed=[2],
               status="won", timeline=[{"kind": "x"}, {"kind": "won", "iteration": 2,
               "holdout": {"mean_rel_delta": 0.04, "worst_rel_delta": 0.01}}])
    d = job_detail(run, now=T0 + 10**6)
    assert d["schema_version"] == SCHEMA_VERSION == 1
    assert d["budget"]["iterations"] == {"cap": 10, "spent": 2, "left": 8}
    assert d["budget"]["hours"] == {"cap": 2.0, "spent": pytest.approx(60 / 3600),
                                    "left": pytest.approx(2.0 - 60 / 3600)}
    assert d["budget"]["compute_usd"]["cap"] is None
    assert d["budget"]["compute_usd"]["left"] is None
    assert d["best"] == {"iteration": 2, "title": "title 2", "mean_rel_delta": 0.105,
                         "holdout_status": "confirmed", "holdout_mean_rel_delta": 0.04}
    assert "rand_hash" not in d["job"]
    assert d["activity"]["pending"] is None and d["activity"]["last_event"]["kind"] == "won"


def test_budget_left_never_negative(tmp_path):
    # mutation: cap - spent without max(0, ...) prints "-3 iterations left"
    run = _run(tmp_path, hyps=[{"iteration": i} for i in range(1, 14)])  # 13 > cap 10
    assert job_detail(run, now=T0)["budget"]["iterations"]["left"] == 0


def test_overrun_hours_left_is_a_float_zero(tmp_path):
    # docs/json-output.md types budget.hours.left as a float; max(0, ...) returned the int 0
    run = _run(tmp_path, status="exhausted", timeline=[{"kind": "a"}] * 200)  # 199 min > 2 h
    left = job_detail(run, now=T0)["budget"]["hours"]["left"]
    assert left == 0 and isinstance(left, float)


def test_money_and_hours_are_floats_even_when_stored_as_ints(tmp_path):
    # docs/json-output.md types these as floats; a job.json/state.json written with whole
    # numbers (hours 2, compute 0) would otherwise come out as JSON ints
    run_dir = tmp_path / "runs" / "J"
    make_run(run_dir, [{"iteration": 1}], timeline=[{"kind": "a"}, {"kind": "b"}])
    spec = json.loads((run_dir / "job.json").read_text())
    spec["budget"] = {"usd": 1, "hours": 2, "iterations": 10, "compute_usd": 5}
    (run_dir / "job.json").write_text(json.dumps(spec))
    st = json.loads((run_dir / "state.json").read_text())
    st["spend"].update(llm_usd=0, compute_usd=0)
    (run_dir / "state.json").write_text(json.dumps(st))
    budget = job_detail(load_run(tmp_path / "runs", "J"), now=T0)["budget"]
    for dim in ("hours", "usd", "compute_usd"):
        assert all(isinstance(v, float) for v in budget[dim].values()), (dim, budget[dim])
    row, = status_rows(tmp_path / "runs")
    assert isinstance(row["llm_usd"], float) and isinstance(row["compute_usd"], float)


def test_budget_counts_extensions(tmp_path):
    # mutation: reading spec.budget instead of effective_budget(...)
    run_dir = tmp_path / "runs" / "J"
    make_run(run_dir, [{"iteration": 1}])
    st = json.loads((run_dir / "state.json").read_text())
    st["budget_extensions"] = [{"add": {"iterations": 5}}]
    (run_dir / "state.json").write_text(json.dumps(st))
    d = job_detail(load_run(tmp_path / "runs", "J"), now=T0)
    assert d["budget"]["iterations"]["cap"] == 15


def test_detail_in_progress_names_the_pending_iteration(tmp_path):
    run = _run(tmp_path, status="researching", timeline=[{"kind": "hypothesis", "iteration": 2,
               "title": "try X"}], pending_job={"purpose": 2, "hypothesis": {"title": "try X"},
               "files": {}})
    assert job_detail(run, now=T0)["activity"]["pending"] == {"purpose": 2, "title": "try X"}


def test_detail_of_a_job_with_no_baseline(tmp_path):
    run = _run(tmp_path, hyps=[], baseline=None, status="failed", timeline=None)
    d = job_detail(run, now=T0)
    assert d["best"] is None and d["activity"]["last_event"] is None
    assert d["last_event_ts"] is None


# ── human formatters ─────────────────────────────────────────────────

def test_status_table_shows_unpriced_and_missing_values(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs / "A", [{"iteration": 1}], best=1, timeline=[{"kind": "x"}])
    make_run(runs / "B", [{"iteration": 1}], provider="codex-cli", model="x", timeline=None)
    lines = format_status_table(status_rows(runs)).splitlines()
    assert lines[0].split()[:4] == ["JOB", "CHALLENGE", "STATUS", "IT"]
    assert "+10.500%" in lines[1] and "2001-09-09 01:46 UTC" in lines[1]  # T0 in UTC
    # COMPUTE is a measured $0.00 on this row; the LLM cell must not be a second one
    assert "not tracked" in lines[2] and lines[2].count("$0.00") == 1
    assert lines[2].rstrip().endswith("-")
    # replaces PR #82's test_status_labels_untracked_and_unpriced_llm_spend (openai half)
    make_run(runs / "C", [{"iteration": 1}], provider="openai", model="gpt-5", timeline=None)
    assert "unpriced" in format_status_table(status_rows(runs)).splitlines()[3]


def test_status_table_unreadable_row(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs / "B", [{"iteration": 1}])
    (runs / "B" / "state.json").unlink()
    line = format_status_table(status_rows(runs)).splitlines()[1]
    assert line.split()[:3] == ["B", "-", "unreadable"]


def test_job_detail_text_prints_infinite_cap(tmp_path):
    # mutation: printing the None cap as "None" (or a number) instead of ∞
    text = format_job_detail(job_detail(_run(tmp_path), now=T0))
    assert "  compute: cap ∞, spent $0.00, left ∞" in text.splitlines()


# ── one iteration: show ──────────────────────────────────────────────

def test_resolve_errors(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}])
    with pytest.raises(InspectError, match=r"^job J has no best candidate; use --iteration N$"):
        resolve_node(run, "best")
    with pytest.raises(InspectError, match=r"^job J has no iteration 3 \(it has 1\.\.2\)$"):
        resolve_node(run, 3)
    empty = _run(tmp_path, name="K", hyps=[])
    with pytest.raises(InspectError, match=r"^job K has no iteration 1 \(it has none\)$"):
        resolve_node(empty, 1)


def test_show_uses_record_mean_not_metered_rank_score(tmp_path):
    # mutation: reading ReportNode.score (the metered score when present) as the training mean
    nodes = [{"id": 1, "chain": 0, "parent": 0, "family": "f", "title": "t", "outcome": "improved",
              "score": 0.02, "tracks": {"t": 0.02}, "metered_score": 0.015}]
    run = _run(tmp_path, hyps=[{"iteration": 1, "mean_rel_delta": 0.02, "worst_track": "t",
               "worst_rel_delta": 0.02, "design_family": "f"}], nodes=nodes,
               iteration_files={1: {"mod.rs": "x\n"}})
    s = iteration_detail(run, 1)["scores"]
    assert s["mean_rel_delta"] == 0.02 and s["metered_rel_delta"] == 0.015
    assert s["tracks"] == {"t": 0.02} and s["tables"] is None


def test_show_pre_lineage_record_has_null_tracks(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1, "mean_rel_delta": -0.01, "worst_track": "t",
               "worst_rel_delta": -0.01, "against": 0}])
    d = iteration_detail(run, 1)
    assert d["scores"]["tracks"] is None and d["scores"]["worst_track"] == "t"
    assert d["parent"] == 0 and d["schema_version"] == 1


def test_show_best_has_tables_holdout_and_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # file paths are relative to the cwd
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}], best=2, false_positives=[2],
               timeline=[{"kind": "false_positive", "iteration": 2,
                          "holdout": {"mean_rel_delta": -0.03, "worst_rel_delta": -0.05}}])
    d = iteration_detail(run, "best")
    assert d["iteration"] == 2 and d["kind"] == "best"
    assert d["scores"]["holdout_status"] == "false_positive"
    assert d["scores"]["holdout_mean_rel_delta"] == -0.03  # the failing value, not null
    headings = [t["heading"] for t in d["scores"]["tables"]]
    assert headings == ["Training nonces", "Held-out nonces"]
    assert d["scores"]["tables"][0]["tracks"][0]["track"] == "t"
    # the fixture's baseline qualities are ints (100, 100): statistics.mean gives the int 100
    row = d["scores"]["tables"][0]["tracks"][0]
    assert all(isinstance(row[k], float) for k in ("base_mean", "cand_mean", "rel_delta"))
    assert "runs/J/iterations/0002/mod.rs" in d["files"]


def test_show_best_lists_the_best_dir(tmp_path, monkeypatch):
    # mutation: listing only iterations/NNNN/ (the spec names best/ for the best)
    monkeypatch.chdir(tmp_path)
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2}], best=2,
               iteration_files={1: {"mod.rs": "one\n"}})
    (run.run_dir / "best").mkdir()
    (run.run_dir / "best" / "mod.rs").write_text("x\n")
    assert "runs/J/best/mod.rs" in iteration_detail(run, "best")["files"]
    # mutation: listing best/ for every iteration
    assert "runs/J/best/mod.rs" not in iteration_detail(run, 1)["files"]


def test_format_iteration_names_parent(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2, "against": 1}], best=2)
    assert "parent: #1" in format_iteration(iteration_detail(run, 2))
    assert "parent: baseline" in format_iteration(iteration_detail(run, 1))


def test_format_iteration_template_parent_and_full_description(tmp_path):
    long = "word " * 200
    run = _run(tmp_path, hyps=[{"iteration": 1, "description": long}], start="template",
               template="fn t() {}\n")
    text = format_iteration(iteration_detail(run, 1))
    assert "parent: template" in text
    # mutation: cutting the description to one line, as the event lines do
    assert text.count("word") == 200
    # the spec: the description is printed full and wrapped
    lines = text.splitlines()
    desc = lines[lines.index("description:") + 1:lines.index("training delta: n/a")]
    assert len(desc) > 1 and max(len(line) for line in desc) <= 100


# ── one iteration: diff ──────────────────────────────────────────────

def test_diff_best_against_baseline_and_parent(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1}, {"iteration": 2, "against": 1}], best=2,
               iteration_files={1: {"mod.rs": "line a\nline b\nline c one\n"}})
    base = iteration_diff(run, "best", "baseline")
    assert base.startswith("--- baseline/mod.rs\n+++ candidate/mod.rs\n")
    assert "+line c best 2" in base and "-line c\n" in base
    par = iteration_diff(run, "best", "parent")
    assert par.startswith("--- #1/mod.rs") and "-line c one" in par


def test_diff_template_start_labels_template(tmp_path):
    run = _run(tmp_path, start="template", template="fn t() {}\n", best=1)
    assert iteration_diff(run, "best", "baseline").startswith("--- template/mod.rs")


def test_diff_against_a_parent_with_no_files(tmp_path):
    # mutation: diffing against {} when the base is None prints a whole-file "addition"
    run = _run(tmp_path, hyps=[{"iteration": 1, "outcome": "failed:compile"},
                               {"iteration": 2, "against": 1}], best=2, iteration_files={1: {}})
    with pytest.raises(InspectError, match=r"^the files of #1 were not saved$"):
        iteration_diff(run, "best", "parent")


def test_diff_of_an_iteration_with_no_files(tmp_path):
    run = _run(tmp_path, hyps=[{"iteration": 1, "outcome": "failed:compile"}],
               iteration_files={1: {}})
    with pytest.raises(InspectError,
                       match=r"^iteration 1 saved no files \(outcome failed:compile\)$"):
        iteration_diff(run, 1, "baseline")


def test_diff_best_uses_state_files_when_the_iteration_dir_is_empty(tmp_path):
    # make_run copies iteration_files[1] into state.best.files, so set those afterwards
    run = _run(tmp_path, best=1, iteration_files={1: {}})  # iterations/0001 has no sources
    st = json.loads((run.run_dir / "state.json").read_text())
    st["best"]["files"] = {"mod.rs": "line a\nline b\nfrom state\n"}
    (run.run_dir / "state.json").write_text(json.dumps(st))
    run = load_run(run.run_dir.parent, "J")
    assert "+from state" in iteration_diff(run, "best", "baseline")


def test_diff_is_not_truncated(tmp_path):
    # mutation: reusing report.node_diffs / truncate caps the patch
    big = "".join(f"line {i}\n" for i in range(200_000))
    run = _run(tmp_path, best=1, iteration_files={1: {"mod.rs": big}})
    assert iteration_diff(run, "best", "baseline").count("\n+line ") == 200_000


def _void_run(tmp_path):
    """A knapsack fixture moved onto the challenge the rules cover, with the leak in its best."""
    make_run(tmp_path / "runs" / "V", [{"iteration": 1}], best=1,
             iteration_files={1: {"mod.rs": "line a\n"}})
    run_dir = tmp_path / "runs" / "V"
    spec = json.loads((run_dir / "job.json").read_text())
    spec.update(challenge="energy_arbitrage", dev_image_tag="0.0.8")
    (run_dir / "job.json").write_text(json.dumps(spec))
    state = json.loads((run_dir / "state.json").read_text())
    state["best"]["files"] = {"mod.rs": "let v = serde_json::to_value(challenge);\n"}
    (run_dir / "state.json").write_text(json.dumps(state))
    return load_run(tmp_path / "runs", "V")


def test_a_void_job_says_so_everywhere_it_is_read(tmp_path):
    # mutation: reporting it only on resume leaves `talos status` showing the score as a result
    run = _void_run(tmp_path)
    rows = status_rows(tmp_path / "runs")
    assert "hidden_seed" in rows[0]["void_reason"]
    assert "(void)" in format_status_table(rows)
    detail = job_detail(run, 0.0)
    assert "hidden_seed" in detail["void_reason"]
    assert "VOID:" in format_job_detail(detail)
    shown = iteration_detail(run, "best")
    assert "hidden_seed" in shown["void_reason"]
    assert "VOID:" in format_iteration(shown)


def test_a_clean_job_has_no_void_marker(tmp_path):
    # mutation: a marker on every row teaches the reader to ignore it
    make_run(tmp_path / "runs" / "C", [{"iteration": 1}], best=1)
    rows = status_rows(tmp_path / "runs")
    assert rows[0]["void_reason"] is None and "(void)" not in format_status_table(rows)
