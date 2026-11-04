"""`talos run --base-algorithm/--base-dir`: the search starts from the chosen code, and the
baseline to beat stays mainnet's top algorithm."""
import json
from dataclasses import replace

import pytest

from talos import cli
from talos.base_algorithm import read_algorithm_dir, read_base_root, root_problem, write_base_root
from talos.bench import BenchUnavailable, FakeBench
from talos.budget import Budget
from talos.challenges import CHALLENGES
from talos.lineage import AnnealSettings
from talos.loop import BaseAlgorithmError, Thresholds
from talos.package import build_package
from talos.prompts import PromptContext, gap_block, parent_label
from talos.scoring import Bar, bundle_delta, improves, validation_failure
from talos.state import JobState
from talos.types import NonceResult
from tests.test_cli import fake_config
from tests.test_loop import edit, hyp, make

BASE = {"source": "local", "name": "mine", "id": None}
SPEC_KW = {"start": "base", "base_algorithm": BASE, "hyperparameters": {"t": {"x": 1}}}


def _k_scores(challenge, files, ns):
    """Level with the baseline at k=10, which scores 100; each k is one point."""
    import re
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [90 + k for _ in ns.nonces()]


def _iterations(n):
    return Budget(usd=None, hours=None, iterations=n, compute_usd=None)


def _job(tmp_path, script, k=15, budget=None, scores=_k_scores, bench=None, **kw):
    write_base_root(tmp_path, {"mod.rs": f"fn solve() {{ let k = {k}; }}\n"})
    return make(tmp_path, script, scores=scores, budget=budget or _iterations(1), bench=bench,
                spec_kw=SPEC_KW, **kw)


def test_root_is_scored_first_under_the_candidates_conditions(tmp_path):
    # mutation: scoring the root with the baseline's hyperparameters, or after the first LLM call
    loop, fp, fb, _ = _job(tmp_path, [hyp("up"), edit(17, frm=15)], scoring="native")
    st = loop.run()
    first = fb.calls[0]
    assert first.files == read_base_root(tmp_path)
    assert first.mode == "metered" and first.hyperparameters is None
    assert st.base_delta["mean_rel_delta"] == pytest.approx(0.05)


# ── the best of the baseline, the base algorithm and the candidate ───────

def test_a_base_over_the_baseline_is_what_candidates_edit(tmp_path):
    # mutation: _root_files returning the baseline's files, or candidates keeping its map
    loop, fp, fb, _ = _job(tmp_path, [hyp("up"), edit(17, frm=15)])
    st = loop.run()
    assert st.base_leads is True
    prompts = "\n".join(str(call) for call in fp.calls)
    assert "let k = 15;" in prompts and "let k = 1;" not in prompts
    assert all(call.hyperparameters is None for call in fb.calls)


def test_a_candidate_over_the_baseline_but_under_the_base_is_not_selected(tmp_path):
    # mutation: improves() passing a strict bar on beating the baseline makes this +2% the best,
    # and scoring wins without improved confirms it as a win below the +5% base
    loop, fp, fb, _ = _job(tmp_path, [hyp("up"), edit(12, frm=15), hyp("again"), edit(16, frm=15)],
                           budget=_iterations(2))
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:score"
    assert st.hypotheses[0]["mean_rel_delta"] == pytest.approx(0.02)
    # The next iteration still edits the base, and the first candidate over it wins.
    assert st.hypotheses[1]["against"] == 0
    assert st.status == "won" and st.best.iteration == 2
    assert st.confirmed == [2] and st.false_positives == []


@pytest.mark.parametrize("chains", [1, 2])
def test_the_under_base_candidate_is_refused_however_many_chains(tmp_path, chains):
    # mutation: the chain's own bar not being strict lets a chain continue from it
    write_base_root(tmp_path, {"mod.rs": "fn solve() { let k = 15; }\n"})
    loop, _, _, _ = make(tmp_path, [hyp("up"), edit(12, frm=15)], scores=_k_scores,
                         budget=_iterations(1), spec_kw={**SPEC_KW, "chains": chains})
    st = loop.run()
    assert st.best is None and st.status == "exhausted" and st.confirmed == []
    assert st.hypotheses[0]["outcome"] == "failed:score"
    assert st.chains[0].best == 0 and st.chains[0].current == 0


def test_a_later_candidate_under_the_best_is_not_selected(tmp_path):
    # mutation: a non-strict best bar lets +2% replace a +7% best that held-out nonces rejected
    def scores(challenge, files, ns):
        k = int(__import__("re").search(r"let k = (\d+);", files["mod.rs"]).group(1))
        held_out = ns.start >= 1_000_000
        return [(80 if held_out and k == 17 else 90 + k) for _ in ns.nonces()]
    loop, _, _, _ = _job(tmp_path, [hyp("up"), edit(17, frm=15), hyp("down"), edit(12, frm=17)],
                         budget=_iterations(2), scores=scores)
    st = loop.run()
    assert st.false_positives == [1] and st.best.iteration == 1 and st.status == "exhausted"
    assert st.hypotheses[1]["outcome"] == "failed:score"


@pytest.mark.parametrize("k", [5, 10])
def test_a_base_that_does_not_beat_the_baseline_is_still_what_candidates_edit(tmp_path, k):
    # mutation: going back to the baseline's code, or giving candidates of the base its map
    loop, fp, fb, _ = _job(tmp_path, [hyp("up"), edit(12, frm=k)], k=k)
    st = loop.run()
    assert st.base_leads is False and st.base_delta is not None
    prompts = "\n".join(str(call) for call in fp.calls)
    assert f"let k = {k};" in prompts and "let k = 1;" not in prompts
    assert all(call.hyperparameters is None for call in fb.calls)
    assert st.status == "won" and st.best.delta["mean_rel_delta"] == pytest.approx(0.02)


def test_a_chain_climbs_from_a_base_under_the_baseline_without_it_being_the_best(tmp_path):
    # mutation: holding the job's best to the base's -5% makes a -3% candidate the result, and
    # holding the chain to the baseline never lets it move off a root under mainnet
    script = [hyp("a"), edit(7, frm=5), hyp("b"), edit(6, frm=7), hyp("c"), edit(9, frm=7),
              hyp("d"), edit(12, frm=9)]
    loop, fp, _, store = _job(tmp_path, script, k=5, budget=_iterations(4))
    st = loop.run()
    # mutation: in_tree leaving out a base start calls #3, which never was the best, "best #3"
    assert "- #4 (vs node #3)" in (build_package(loop.spec, st, store) / "hypotheses.md").read_text()
    outcomes = [h["outcome"] for h in st.hypotheses]
    assert outcomes == ["accepted", "failed:score", "accepted", "won"]
    assert [h["against"] for h in st.hypotheses] == [0, 1, 1, 3]
    assert st.best.iteration == 4 and st.chains[0].best == 4
    prompts = "\n".join(str(call) for call in fp.calls)
    assert "Parent (node #1): -3.000%" in prompts and "Best so far: none yet" in prompts


def test_climbing_towards_the_baseline_is_not_a_stall(tmp_path):
    # mutation: counting only the job's best as progress distills and rotates strategy on a chain
    # that is improving every iteration
    script = [x for k in range(5, 9) for x in (hyp("up"), edit(k + 1, frm=k))]
    thresholds = Thresholds(recall=1, distill=1, reset=1)
    loop, fp, _, _ = _job(tmp_path, script, k=5, budget=_iterations(4), thresholds=thresholds)
    st = loop.run()
    assert st.best is None and st.runs_since_improvement == 0 and st.tacit == ""
    assert len(fp.calls) == 8


def test_annealing_from_a_base_under_the_baseline_never_makes_the_best(tmp_path):
    # mutation: an annealing chain's move being read as a job-wide improvement
    write_base_root(tmp_path, {"mod.rs": "fn solve() { let k = 5; }\n"})
    loop, _, _, _ = make(tmp_path, [hyp("up"), edit(7, frm=5)], scores=_k_scores,
                         budget=_iterations(1),
                         spec_kw={**SPEC_KW, "search": "anneal",
                                  "anneal": AnnealSettings().to_dict()})
    st = loop.run()
    assert st.best is None and st.status == "exhausted"
    assert st.chains[0].current == 1 and st.chains[0].best == 1


def test_a_higher_mean_that_regresses_a_track_does_not_lead_the_baseline(tmp_path):
    # mutation: comparing means alone goes against the rule a candidate is held to (issue #32)
    def scores(challenge, files, ns):
        return [(130 if ns.track == "t" else 90) for _ in ns.nonces()]
    write_base_root(tmp_path, {"mod.rs": "fn solve() { let k = 15; }\n"})
    loop, _, _, _ = make(tmp_path, [hyp("up"), edit(16, frm=15)], scores=scores, two_tracks=True,
                         spec_kw=SPEC_KW, budget=_iterations(1))
    st = loop.run()
    assert st.base_delta["mean_rel_delta"] > 0 and st.base_leads is False


def test_the_strict_bar_also_holds_at_metered_validation():
    # mutation: a native candidate that loses to the base on TIG's runtime stays the best
    rule = CHALLENGES["knapsack"].beat
    rows = [NonceResult("t", n, True, 100, 1) for n in range(4)]
    better = [NonceResult("t", n, True, 102, 1) for n in range(4)]
    delta = bundle_delta(rows, better)
    assert improves(delta, Bar(0.05, 0.05), rule) is True
    assert improves(delta, Bar(0.05, 0.05, strict=True), rule) is False
    assert improves(delta, Bar(0.01, 0.01, strict=True), rule) is True
    reason, _ = validation_failure(True, better, better, rows, Bar(0.05, 0.05, strict=True), rule)
    assert reason == "not_improved"


def test_prompt_states_the_roots_gap_to_mainnet(tmp_path):
    # mutation: _parent_gap returning None for the root, so the LLM is told nothing was scored
    loop, fp, _, _ = _job(tmp_path, [hyp("up"), edit(17, frm=15)])
    loop.run()
    prompts = "\n".join(str(call) for call in fp.calls)
    assert 'Base algorithm, the parent (node #0): +5.000% vs mainnet "base"' in prompts


def test_a_package_says_when_the_base_stayed_the_best(tmp_path):
    # mutation: the README saying no candidate beat the baseline, when one did, under the base
    loop, _, _, store = _job(tmp_path, [hyp("up"), edit(12, frm=15)])
    st = loop.run()
    readme = (build_package(loop.spec, st, store) / "README.md").read_text()
    assert "No candidate improved on the base algorithm" in readme
    assert "Started from the local algorithm `mine`, which scored +5.000%" in readme


@pytest.mark.parametrize("bench", [
    FakeBench(_k_scores, compile_ok=lambda files: False),  # does not build
    FakeBench(lambda ch, files, ns: [None for _ in ns.nonces()]),  # builds, scores nothing
])
def test_a_base_that_cannot_run_stops_before_any_llm_call(tmp_path, bench):
    # mutation: letting the loop start edits code that can never score, until the budget is gone
    loop, fp, _, _ = _job(tmp_path, [], bench=bench)
    with pytest.raises(BaseAlgorithmError, match="mine"):
        loop.run()
    assert fp.calls == [] and loop.state.pending_job is None


def test_a_zero_compute_cap_stops_before_the_root_is_scored(tmp_path):
    # mutation: AGENTS.md invariant 5, a compute call made without the budget check
    budget = Budget(usd=None, hours=None, iterations=None, compute_usd=0.0)
    loop, _, fb, _ = _job(tmp_path, [], budget=budget)
    st = loop.run()
    assert st.status == "exhausted" and fb.calls == []


def test_a_resume_keeps_the_measured_root(tmp_path):
    # mutation: re-scoring the root on every run spends compute and could move the bar
    loop, _, fb, _ = _job(tmp_path, [hyp("up"), edit(17, frm=15)])
    loop.state.base_delta = {"mean_rel_delta": 0.05, "worst_rel_delta": 0.05, "tracks": [
        {"track": "t", "rel_delta": 0.05}]}
    loop.state.base_leads = True
    loop.run()
    assert all("let k = 15;" not in call.files["mod.rs"] for call in fb.calls)


def test_a_bench_outage_while_scoring_the_root_pauses_and_a_resume_measures_it(tmp_path):
    # mutation: a "base" pending job cleared as stale would lose a C3 job still billing
    class Down(FakeBench):
        down = True

        def evaluate(self, request):
            if self.down:
                raise BenchUnavailable("outage")
            return super().evaluate(request)

    bench = Down(_k_scores)
    loop, _, _, _ = _job(tmp_path, [hyp("up"), edit(17, frm=15)], bench=bench)
    st = loop.run()
    assert st.status == "paused" and st.pending_job["purpose"] == "base"
    bench.down = False
    st = loop.run()
    assert st.base_delta is not None and st.pending_job is None and st.best.iteration == 1


def test_a_missing_snapshot_is_an_error_not_the_baselines_files(tmp_path):
    # mutation: falling back to the baseline silently researches different code
    loop, _, _, _ = make(tmp_path, [], spec_kw=SPEC_KW, budget=_iterations(1))
    with pytest.raises(BaseAlgorithmError, match="missing"):
        loop.run()


# ── prompts, state ───────────────────────────────────────────────────────

def _ctx(**kw):
    return PromptContext(challenge="knapsack", template_rs="t", direction="d", tacit="",
                         files={"mod.rs": "x"}, baseline_name="top", best_delta=0.0, **kw)


def test_gap_block_says_the_baseline_code_is_not_the_parent():
    # mutation: the LLM told its parent is mainnet's code, or that it can read the baseline
    gap = {"node": 0, "mean": -0.02, "tracks": {"t": -0.02}}
    text = gap_block(_ctx(start="base", anchor=0, parent_gap=gap))
    assert text.startswith('Base algorithm, the parent (node #0): -2.000% vs mainnet "top"')
    assert text.endswith("The baseline's code is not available to you; only its score is.")
    later = gap_block(_ctx(start="base", anchor=3, parent_gap={**gap, "node": 3}))
    assert later.startswith("Parent (node #3)")


def test_parent_label_names_the_base_algorithm():
    assert parent_label(0, "base", False) == "the base algorithm"
    assert parent_label(0, "mainnet", False) == "the baseline"
    assert parent_label(2, "base", False) == "best #2"


def test_state_round_trips_the_root_delta_and_old_state_has_none():
    # mutation: a field dropped from to_dict is measured again on every resume
    from talos.budget import Spend
    st = JobState.fresh(Spend(started_at=0.0))
    st = replace(st, base_delta={"mean_rel_delta": 0.1}, base_leads=True)
    d = st.to_dict()
    assert JobState.from_dict(d).base_delta == {"mean_rel_delta": 0.1}
    # mutation: dropping base_leads would lower a resumed job's bar to the baseline's
    assert JobState.from_dict(d).base_leads is True
    del d["base_delta"], d["base_leads"]
    assert JobState.from_dict(d).base_delta is None
    assert JobState.from_dict(d).base_leads is False


# ── the files ────────────────────────────────────────────────────────────

def test_read_algorithm_dir_takes_rust_and_kernels_only(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "mod.rs").write_text("a")
    (tmp_path / "sub" / "x.cu").write_text("b")
    (tmp_path / "notes.md").write_text("c")
    assert read_algorithm_dir(tmp_path) == {"mod.rs": "a", "sub/x.cu": "b"}
    assert read_algorithm_dir(tmp_path / "absent") == {}


def test_root_problem_needs_mod_rs_and_a_kernel_on_gpu():
    assert "mod.rs" in root_problem("knapsack", {"x.rs": ""})
    assert root_problem("knapsack", {"mod.rs": ""}) is None
    assert ".cu" in root_problem("hypergraph", {"mod.rs": ""})
    assert root_problem("hypergraph", {"mod.rs": "", "k.cu": ""}) is None


def test_snapshot_round_trips_nested_files(tmp_path):
    assert read_base_root(tmp_path) is None
    write_base_root(tmp_path, {"mod.rs": "a", "sub/ls.rs": "b"})
    assert read_base_root(tmp_path) == {"mod.rs": "a", "sub/ls.rs": "b"}


# ── the command line ─────────────────────────────────────────────────────

def _local_dir(tmp_path, k=0, name="my_algo"):
    d = tmp_path / name
    d.mkdir()
    (d / "mod.rs").write_text(f"fn solve() {{ let k = {k}; }}\n")
    return d


def _run(*flags, challenge="knapsack", iterations="4"):
    return cli.main(["run", "--fake", "--yes", "--challenge", challenge, "--direction", "go",
                     "--budget-iterations", iterations, *flags])


def _only_run(tmp_path):
    return next((tmp_path / "runs").glob("*/job.json")).parent


def test_local_base_job_end_to_end(tmp_path, monkeypatch, capsys):
    # mutation: the baseline replaced by the local code would call k=1 a win over itself
    monkeypatch.chdir(tmp_path)
    assert _run("--base-dir", str(_local_dir(tmp_path, k=2))) == 0
    out = capsys.readouterr().out
    assert "local algorithm my_algo; the baseline stays mainnet's top algorithm" in out
    assert "goes to your LLM provider" in out
    assert "on the baseline only" in out
    run_dir = _only_run(tmp_path)
    spec = json.loads((run_dir / "job.json").read_text())
    assert spec["start"] == "base"
    assert spec["base_algorithm"] == {"source": "local", "name": "my_algo", "id": None}
    state = json.loads((run_dir / "state.json").read_text())
    assert state["baseline"]["name"] == "fake_base"
    assert state["base_delta"]["mean_rel_delta"] == pytest.approx(0.01)
    assert state["base_leads"] is True
    pkg = run_dir / "package"
    assert "let k = 2;" in (pkg / "diff_vs_base.patch").read_text()
    assert not (pkg / "diff_vs_baseline.patch").exists()
    assert not (pkg / "hyperparameters.json").exists()  # mutation: shipping values never used
    readme = (pkg / "README.md").read_text()
    assert "Baseline: mainnet `fake_base`" in readme
    assert "Started from the local algorithm `my_algo`" in readme
    assert "tig-algorithms/src/knapsack/fake_base" not in readme  # not this code's identifier
    report = (pkg / "report.html").read_text()
    assert ">base</text>" in report and "base (local my_algo)" in report
    capsys.readouterr()
    assert cli.main(["diff", run_dir.name]) == 0
    assert capsys.readouterr().out.startswith("--- base/mod.rs")


def test_a_base_under_the_baseline_end_to_end(tmp_path, monkeypatch):
    # mutation: the package describing the baseline as the code this job edited
    monkeypatch.chdir(tmp_path)
    assert _run("--base-dir", str(_local_dir(tmp_path, k=0))) == 0
    run_dir = _only_run(tmp_path)
    state = json.loads((run_dir / "state.json").read_text())
    assert state["base_leads"] is False
    assert [h["outcome"] for h in state["hypotheses"]] == ["accepted", "won"]
    pkg = run_dir / "package"
    assert (pkg / "diff_vs_base.patch").exists() and not (pkg / "hyperparameters.json").exists()
    assert "`my_algo`, which scored -1.000%" in (pkg / "README.md").read_text()
    assert ">base</text>" in (pkg / "report.html").read_text()


def test_a_job_that_never_reaches_the_baseline_says_where_its_work_is(tmp_path):
    # mutation: "nothing to submit" and no pointer loses what the budget paid for
    loop, _, _, store = _job(tmp_path, [hyp("up"), edit(7, frm=5)], k=5)
    st = loop.run()
    readme = (build_package(loop.spec, st, store) / "README.md").read_text()
    assert "No candidate beat the baseline" in readme
    assert "The closest candidate was #1, at -3.000%" in readme and "iterations/0001/" in readme


def test_named_mainnet_base_job(tmp_path, monkeypatch):
    # mutation: --hyperparameters none leaving the pin to resolve_baseline must not drop the base
    monkeypatch.chdir(tmp_path)
    assert _run("--base-algorithm", "fake_other", "--hyperparameters", "none",
                iterations="1") == 0
    run_dir = _only_run(tmp_path)
    spec = json.loads((run_dir / "job.json").read_text())
    assert spec["base_algorithm"] == {"source": "mainnet", "name": "fake_other",
                                      "id": "c003_a001"}
    assert read_base_root(run_dir) == {"mod.rs": "fn solve() { let k = 1; }\n"}
    readme = (run_dir / "package" / "README.md").read_text()
    assert "Started from the mainnet algorithm `fake_other`" in readme
    # mutation: pointing the identifier note at the baseline, whose method this is not
    assert "the base algorithm's own README" in readme
    assert "tig-algorithms/src/knapsack/fake_other/README.md" in readme


def test_the_source_directory_is_not_read_after_the_job_starts(tmp_path, monkeypatch):
    # mutation: reading --base-dir on resume would let its edits change what the job measured
    monkeypatch.chdir(tmp_path)
    d = _local_dir(tmp_path, k=2)
    assert _run("--base-dir", str(d), iterations="1") == 0
    run_dir = _only_run(tmp_path)
    (d / "mod.rs").write_text("not rust at all\n")
    assert read_base_root(run_dir)["mod.rs"] == "fn solve() { let k = 2; }\n"


@pytest.mark.parametrize("make_flags, needle", [
    (lambda d: ["--base-dir", str(d), "--base-algorithm", "fake_other"], "not both"),
    (lambda d: ["--base-dir", str(d), "--start", "template"], "--start"),
    (lambda d: ["--base-dir", str(d), "--start", "mainnet"], "--start"),
    (lambda d: ["--base-dir", str(d / "absent")], "no .rs/.cu files"),
    (lambda d: ["--base-algorithm", "nobody"], "no compiled algorithm 'nobody'"),
])
def test_a_bad_choice_is_refused_before_a_job_exists(tmp_path, monkeypatch, capsys, make_flags,
                                                     needle):
    monkeypatch.chdir(tmp_path)
    assert _run(*make_flags(_local_dir(tmp_path))) == 2
    assert needle in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_a_directory_without_the_files_a_build_needs_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: found only after the baseline is paid for, when the root fails to compile
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "gpu"
    d.mkdir()
    (d / "mod.rs").write_text("fn solve() {}\n")
    assert _run("--base-dir", str(d), challenge="hypergraph") == 2
    assert ".cu" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_a_directory_that_is_not_utf8_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    d = _local_dir(tmp_path)
    (d / "mod.rs").write_bytes(b"\xff\xfe")
    assert _run("--base-dir", str(d)) == 2
    assert "cannot read --base-dir" in capsys.readouterr().err


def test_a_failing_base_ends_the_job_with_a_clear_reason(tmp_path, monkeypatch, capsys):
    # mutation: the error escaping execute_job loses the package and the failed status
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "broken"
    d.mkdir()
    (d / "mod.rs").write_text("fn solve() { let k = nope; }\n")  # _fake_scores cannot parse it
    assert _run("--base-dir", str(d), iterations="1") == 1
    state = json.loads((_only_run(tmp_path) / "state.json").read_text())
    assert state["status"] == "failed" and state["base_delta"] is None
    assert state["spend"]["llm_usd"] == 0


def test_resume_refuses_a_new_base_and_a_missing_snapshot(tmp_path, monkeypatch, capsys):
    # mutation: dropping either check changes or loses the code the job's nodes descend from
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    d = _local_dir(tmp_path, k=2)
    assert _run("--base-dir", str(d), iterations="0") == 1
    run_dir = _only_run(tmp_path)
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name, "--base-dir", str(d)]) == 2
    assert "start a new job" in capsys.readouterr().err
    assert cli.main(["run", "--resume", run_dir.name, "--start", "mainnet"]) == 2
    assert "--start base" in capsys.readouterr().err
    (run_dir / "base_algorithm" / "mod.rs").unlink()
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 1
    assert "base_algorithm is missing" in capsys.readouterr().err


LEAK = "fn solve() { let k = 1; let v = serde_json::to_value(challenge); }\n"


def test_root_problem_refuses_leak_code_only_on_its_challenge():
    # mutation: dropping the scan lets every candidate inherit the leak and ship it
    assert "hidden_seed" in root_problem("energy_arbitrage", {"mod.rs": LEAK})
    assert root_problem("knapsack", {"mod.rs": LEAK}) is None
    assert root_problem("energy_arbitrage", {"mod.rs": "fn solve() { let k = 1; }\n"}) is None


def test_base_dir_with_the_leak_is_refused_before_the_job_exists(tmp_path, monkeypatch, capsys):
    # mutation: the check after write_spec leaves a run directory for a job that cannot run
    monkeypatch.chdir(tmp_path)
    d = _local_dir(tmp_path)
    (d / "mod.rs").write_text(LEAK)
    assert _run("--base-dir", str(d), challenge="energy_arbitrage") != 0
    assert "hidden_seed" in capsys.readouterr().err
    assert not list((tmp_path / "runs").glob("*/job.json"))


def test_base_algorithm_known_to_leak_is_refused_by_name(tmp_path, monkeypatch, capsys):
    # mutation: only scanning would fetch it first, and spend the anonymous GitHub rate limit
    monkeypatch.chdir(tmp_path)
    assert _run("--base-algorithm", "gridflow", challenge="energy_arbitrage") != 0
    assert "hidden_seed" in capsys.readouterr().err
    assert not list((tmp_path / "runs").glob("*/job.json"))


def test_a_copy_with_the_leak_removed_can_be_a_root(tmp_path, monkeypatch, capsys):
    # mutation: refusing by directory name stops the honest fork this route exists for
    # mutation: the leak line between `base:` and its continuation misattributes the source warning
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "gridflow"
    d.mkdir()
    (d / "mod.rs").write_text("fn solve() { let k = 2; }\n")
    assert _run("--base-dir", str(d), challenge="energy_arbitrage", iterations="1") in (0, 1)
    assert list((tmp_path / "runs").glob("*/job.json"))
    lines = capsys.readouterr().out.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("  base:"))
    assert "its source goes to your LLM provider" in lines[at + 1]
