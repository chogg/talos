import json
import math
import re
import types
from dataclasses import replace

import pytest

from talos.budget import Budget, BudgetExhausted, Spend
from talos.bench import FakeBench
from talos.challenges import BeatRule
from talos.extend import extend
from talos.loop import Loop, Thresholds
from talos.prompts import SEARCH_REPLACE_FORMAT, WHOLE_FILE_FORMAT
from talos.providers import (ProviderAuthError, ProviderError, ProviderRateLimited,
                             ProviderTimeout)
from talos.providers.fake import FakeProvider
from talos.state import BaselineRecord, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "ab" * 32
TR = [NonceSet("t", HASH, 0, 4)]
HO = [NonceSet("t", HASH, 1_000_000, 4)]
TR2 = [NonceSet("t", HASH, 0, 4), NonceSet("u", HASH, 0, 4)]
HO2 = [NonceSet("t", HASH, 1_000_000, 4), NonceSet("u", HASH, 1_000_000, 4)]
BASE_FILES = {"mod.rs": "fn solve() { let k = 1; }\n"}


def spec(budget=None, track=None, two_tracks=False, scoring="metered", fuel=1, **spec_kw):
    """Any other JobSpec field (start, search, chains, anneal, hyperparameters) is passed
    through as given."""
    tracks = ["t", "u"] if two_tracks else ["t"]
    return JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot",
                   budget=budget or Budget(usd=None, hours=None, iterations=20, compute_usd=None),
                   rand_hash=HASH, tracks=tracks, training=TR2 if two_tracks else TR,
                   holdout=HO2 if two_tracks else HO, fuel=fuel, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003", track=track, scoring=scoring,
                   **spec_kw)


def baseline(q=100, holdout_n=4, tracks=("t",)):
    tr = [NonceResult(t, n, True, q, 1) for t in tracks for n in range(4)]
    ho = [NonceResult(t, 1_000_000 + n, True, q, 1) for t in tracks for n in range(holdout_n)]
    return BaselineRecord(name="base", adoption=1, artifact_id="base-art", files=BASE_FILES,
                          training=tr, holdout=ho)


def hyp(title, tag="local_search"):
    return f'{{"title": "{title}", "description": "d", "strategy_tag": "{tag}"}}'


def edit(k, frm=1):
    """An edit block turning `let k = {frm};` into `let k = {k};`. The SEARCH text must match the
    files the loop is currently editing (the best so far), not always the baseline."""
    return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {k};\n>>>>>>> REPLACE\n"


def quality_from_files(challenge, files, ns):
    # quality = 100 + k on every nonce; k parsed from the code. "let k = 7" -> 107
    import re
    k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
    return [100 + k - 1 for _ in ns.nonces()]  # k=1 -> 100 (baseline parity)


def make(tmp_path, script, scores=quality_from_files, budget=None, thresholds=None, holdout_n=4,
         track=None, two_tracks=False, scoring="metered", fuel=1, budgets=None, bench=None,
         spec_kw=None, template_rs="pub fn solve_challenge(", challenge_docs=None):
    store = JobStore(tmp_path)
    sp = spec(budget, track=track, two_tracks=two_tracks, scoring=scoring, fuel=fuel,
              **(spec_kw or {}))
    store.write_spec(sp)
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline(holdout_n=holdout_n, tracks=("t", "u") if two_tracks else ("t",))
    st.status = "researching"
    st.best = None
    if scoring == "native":
        st.scoring = "native"
        st.fuel_budgets_us = budgets or {"t": 1_000_000}
    fb = bench if bench is not None else FakeBench(scores)
    fp = FakeProvider(script)
    loop = Loop(sp, st, store, fp, fb, template_rs=template_rs,
                clock=lambda: 0.0, sleep=lambda s: None, thresholds=thresholds or Thresholds(),
                **({"challenge_docs": challenge_docs} if challenge_docs is not None else {}))
    return loop, fp, fb, store


def test_win_requires_training_then_holdout_beat(tmp_path):
    # mutation: skipping the held-out confirmation declares a win after training alone
    loop, fp, fb, store = make(tmp_path, [hyp("bump k"), edit(5)])
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 1
    assert st.best.holdout is not None and st.confirmed == [1]
    assert fb.holdout_runs == 1
    # mutation: stamping events with state.iteration labels iteration 1's events as iteration 0
    raw = (tmp_path / "timeline.jsonl").read_text()
    events = [json.loads(ln) for ln in raw.splitlines()]
    mid = [e for e in events if e["kind"] in ("hypothesis", "scored", "confirming", "won")]
    assert mid and all(e["iteration"] == 1 for e in mid)
    # spec §7.10: the rand_hash reaches neither the timeline nor a prompt
    assert HASH not in raw
    assert all(HASH not in system and HASH not in user for system, user in fp.calls)


def test_any_improvement_over_the_baseline_wins(tmp_path):
    # user decision 2026-10-02: no minimum margin. k=2 over a baseline of 1000 is +0.1%.
    # mutation: restoring the 0.5% default margin leaves this job researching to its budget
    loop, fp, fb, store = make(tmp_path, [hyp("tiny"), edit(2)],
                               scores=lambda ch, files, ns: [
                                   q + 900 for q in quality_from_files(ch, files, ns)])
    loop.state.baseline = baseline(q=1000)
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 1 and st.confirmed == [1]
    assert st.best.delta["mean_rel_delta"] == pytest.approx(0.001)


def test_false_positive_returns_to_research(tmp_path):
    # training says win, held-out says no -> keep researching, record false positive
    # mutation: confirming on held-out but still setting status "won" (or dropping the
    # false_positives record) loses the distinction between a real win and a fluke
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.start >= 1_000_000:
            return [100 for _ in ns.nonces()]  # held-out never improves
        return [100 + k - 1 for _ in ns.nonces()]
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    # iteration 2 edits the iteration-1 best (k=5), so its SEARCH text is `let k = 5;`
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(6, frm=5)], scores, b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "iterations"
    assert st.false_positives == [1, 2] and st.best.iteration == 2


def test_compile_fix_rounds_then_skip(tmp_path):
    # mutation: unlimited fix rounds never terminates on a stubborn error
    def swap(frm, to):
        return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {to};\n>>>>>>> REPLACE\n"

    # each fix APPLIES (so the loop would keep going) but still does not build
    script = [hyp("a"), swap(1, "BUG"), swap("BUG", "BUG2"), swap("BUG2", "BUG3"),
              swap("BUG3", "BUG4"), hyp("b"), edit(5)]

    def compile_ok(files):
        return "BUG" not in files["mod.rs"]

    loop, fp, fb, store = make(tmp_path, script)
    fb._compile_ok = compile_ok
    st = loop.run()
    assert st.status == "won"
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert len(fb.calls) == 4 + 1  # 1 edit + 3 fixes + winning edit


def test_budget_stops_before_llm_call(tmp_path):
    # mutation: checking the budget only between iterations lets a half-iteration overspend
    # (the compile here must be refused, not run)
    b = Budget(usd=0.015, hours=None, iterations=None, compute_usd=None)  # one fake call = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2), hyp("b"), edit(3)], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "usd"
    assert len(fp.calls) == 2  # hypothesis + edit, then the compile's budget check refuses
    assert len(fb.calls) == 0  # the evaluate's budget check refuses before any bench call


def test_over_error_ceiling_is_failed_runtime(tmp_path):
    # spec §9: a candidate over the error ceiling is logged failed:runtime and never becomes best
    # mutation: dropping the ceiling check lets a half-crashing candidate become the best
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if k == 1:
            return [100 for _ in ns.nonces()]
        return [None if n % 2 else 100 + k for n in ns.nonces()]  # half the nonces error out

    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(9)], scores, b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:runtime" and st.best is None
    assert st.status == "exhausted"


def test_stagnation_recall_distill_reset(tmp_path):
    # non-improving edits (k stays 1) drive stagnation through recall -> distill -> reset
    # mutation: never resetting runs_since_improvement thresholds (or anchoring the recall list
    # on the wrong iteration) drops the recall block, the distillation and the forced tag
    script = []
    for i in range(6):
        script += [hyp(f"h{i}"), edit(1)]
    script.insert(6, "LESSON: constants are not the answer.")  # distill call after 3rd failure
    b = Budget(usd=None, hours=None, iterations=6, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=3, reset=5))
    st = loop.run()
    assert "constants are not the answer" in st.tacit
    users = [u for _, u in fp.calls]
    assert any("do not repeat" in u for u in users)          # recall block appeared
    assert any("strategy_tag MUST be" in u for u in users)   # reset forced a tag
    # mutation: distilling on `>=` instead of `==` fires a distill call every stagnant iteration,
    # which eats the scripted hypothesis of the next iteration and derails it into failed:edit.
    # The call count alone does NOT catch this (each extra distill is offset by the provider call
    # a derailed iteration no longer makes), so pin the outcomes too.
    assert len(fp.calls) == 6 * 2 + 1  # 6 hypothesis + 6 edit + exactly one distill
    assert [h["outcome"] for h in st.hypotheses] == ["failed:score"] * 6


def test_distilled_lesson_keeps_the_direction_in_tacit_md_and_out_of_the_tacit_prompt(tmp_path):
    # tacit.md starts as the user's direction (cli.cmd_run writes it); a distilled lesson is
    # added after it. The prompts already carry the direction, so state.tacit holds lessons only.
    # mutation: writing tacit.md from state.tacit alone erases the direction at the first lesson;
    # putting the direction into the prompts' tacit block sends it twice in every hypothesis prompt
    script = []
    for i in range(4):
        script += [hyp(f"h{i}"), edit(1)]
    script.insert(6, "LESSON: constants are not the answer.")  # distill call after 3rd failure
    b = Budget(usd=None, hours=None, iterations=4, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=3, reset=5))
    loop.spec = replace(loop.spec, direction="DIRECTION-SENTINEL")
    st = loop.run()
    assert (store.run_dir / "tacit.md").read_text() == (
        "- USER: DIRECTION-SENTINEL\n- LLM: constants are not the answer.\n")
    assert st.tacit == "- LLM: constants are not the answer.\n"
    last_hypothesis_prompt = fp.calls[-2][1]
    assert "constants are not the answer" in last_hypothesis_prompt
    assert last_hypothesis_prompt.count("DIRECTION-SENTINEL") == 1


def test_resume_discards_incomplete_iteration(tmp_path):
    # mutation: resuming without discarding the half-written iteration dir would leave the
    # stale hypothesis.json in place and re-number the next iteration
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.iteration = 3
    store.iteration_dir(4).joinpath("hypothesis.json").write_text("{}")  # started, never finished
    store.iteration_dir(4).joinpath("stale.rs").write_text("half-written")
    store.save(loop.state)
    loop2 = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                 clock=lambda: 0.0, sleep=lambda s: None)
    st = loop2.run()
    assert st.status == "won" and st.best.iteration == 4
    assert not store.iteration_dir(4).joinpath("hypothesis.json").read_text() == "{}"
    assert not store.iteration_dir(4).joinpath("stale.rs").exists()  # the dir was discarded whole
    # mutation: discarding before the TERMINAL early return deletes a finished job's winning
    # iteration directory (state.iteration lags by one when the kill lands mid-confirmation)
    loop2.state.iteration = 0
    store.iteration_dir(1).joinpath("mod.rs").write_text("winner")
    assert loop2.run().status == "won"
    assert store.iteration_dir(1).joinpath("mod.rs").read_text() == "winner"


def test_request_stop_cancels_at_safe_point(tmp_path):
    # mutation: ignoring the stop flag (or checking it only after the LLM call) burns a
    # provider call and reports a status other than "cancelled"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(1), hyp("b"), edit(5)])
    loop.request_stop()
    st = loop.run()
    assert st.status == "cancelled" and len(fp.calls) == 0


def test_win_with_lower_delta_than_a_false_positive_best(tmp_path):
    # mutation: confirming a candidate without making it the best reports won with the wrong code
    # in state.best
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.start >= 1_000_000:
            return [100 + (2 if k == 3 else 0) for _ in ns.nonces()]  # only k=3 holds up
        return [100 + k - 1 for _ in ns.nonces()]

    # k=5 wins training (+4%) but not held-out; k=3 wins less on training (+2%) yet confirms
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(3, frm=5)], scores)
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 2
    assert "let k = 3;" in st.best.files["mod.rs"]
    assert st.hypotheses[1]["outcome"] == "won"
    assert st.confirmed == [2] and st.false_positives == [1]


def test_distill_auth_error_stops_the_run(tmp_path):
    # mutation: swallowing ProviderAuthError in _distill lets a dead key keep the loop running
    seen = {"n": 0}

    def script(system, user):
        if "distill one reusable lesson" in system:
            raise ProviderAuthError("key expired")
        seen["n"] += 1
        return hyp(f"h{seen['n']}") if seen["n"] % 2 else edit(1)

    loop, fp, fb, store = make(tmp_path, script)  # 3 non-improving iterations reach the distill
    st = loop.run()
    assert st.status == "failed"
    assert st.stop_reason.startswith("provider auth")


def test_holdout_scoring_error_is_a_false_positive(tmp_path):
    # mutation: letting ScoringError escape _confirm strands the job at "confirming"
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    # the baseline holds 3 held-out nonces but the run scores 4 -> bundle_delta cannot pair them
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b, holdout_n=3)
    st = loop.run()
    assert st.false_positives == [1] and st.confirmed == []
    assert st.status == "exhausted" and st.stop_reason == "iterations"


def test_compile_fix_stops_when_repair_applies_nothing(tmp_path):
    # mutation: recompiling after a repair that matched nothing burns a bench call per round on
    # byte-identical files
    broken = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    never_matches = "<<<<<<< SEARCH mod.rs\nlet zzz = 1;\n=======\nlet zzz = 2;\n>>>>>>> REPLACE\n"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), broken, never_matches, hyp("b"), edit(5)])

    def compile_ok(files):
        return "BUG" not in files["mod.rs"]

    fb._compile_ok = compile_ok
    st = loop.run()
    assert st.status == "won"
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert len(fb.calls) == 1 + 1  # the broken edit + the winning edit


def test_recall_starts_exactly_at_threshold(tmp_path):
    # mutation: `> recall` instead of `>= recall` delays the recall block by a whole iteration
    script = []
    for i in range(3):
        script += [hyp(f"h{i}"), edit(1)]
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               thresholds=Thresholds(recall=2, distill=99, reset=99))
    loop.run()
    users = [u for _, u in fp.calls]
    assert "do not repeat" not in users[2]  # 2nd hypothesis: runs_since_improvement == 1
    assert "do not repeat" in users[4]      # 3rd hypothesis: runs_since_improvement == 2


def test_failures_against_an_earlier_best_survive_the_best_changing(tmp_path):
    # issue #33: run 20260925-140701-knapsack retried #18's idea at #26 because the best changed
    # at #23 and the recall list and the distillation only saw failures against the new best.
    # mutation: filtering on `against == anchor` drops "Old idea" from both prompts; dropping the
    # parent label, or putting the old failure under "this exact code", misleads the model
    script = [hyp("Old idea"), edit(1), "NONE",   # 1: failed:score against the baseline
              hyp("Winner"), edit(2),             # 2: improved, below the win margin
              hyp("New idea"), edit(2, frm=2), "NONE",  # 3: failed:score against best #2
              hyp("Next"), edit(2, frm=2)]
    b = Budget(usd=None, hours=None, iterations=4, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               scores=lambda ch, files, ns: [
                                   q + 900 for q in quality_from_files(ch, files, ns)],
                               thresholds=Thresholds(recall=1, distill=1, reset=99))
    loop.state.baseline = baseline(q=1000)  # so k=2 is +0.1%: a new best, not a win
    loop.rule = BeatRule(margin=0.005)  # the default margin of 0 would make it a win
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == [
        "failed:score", "improved", "failed:score", "failed:score"]
    users = [u for _, u in fp.calls]
    distill, proposal = users[7], users[8]
    assert "Old idea" in distill and "New idea" in distill
    current, _, earlier = proposal.partition("against an earlier best")
    assert "New idea" in current and "Old idea" not in current
    assert "Old idea" in earlier and "against the baseline" in earlier


def test_rate_limit_wait_rechecks_the_budget(tmp_path):
    # mutation: not re-checking the budget after a rate-limit sleep ignores a stop request (or an
    # hours cap) for the whole retry storm
    seen = {"n": 0}

    def script(system, user):
        seen["n"] += 1
        if seen["n"] == 1:
            raise ProviderRateLimited("429")
        return hyp("a")

    loop, fp, fb, store = make(tmp_path, script)

    def stop_while_sleeping(seconds):
        loop.request_stop()

    loop.sleep = stop_while_sleeping
    st = loop.run()
    assert st.status == "cancelled"
    assert seen["n"] == 1  # the retry after the sleep never happened


def test_rejected_edit_paths_fail_the_iteration(tmp_path):
    # spec §9: an edit outside the algorithm files fails the iteration in both modes.
    # mutation: applying the in-scope blocks after a rejected one lets an out-of-scope edit go
    # unpunished — the response below would otherwise reach "won" on its second block.
    # mutation: dropping the edits_rejected event hides an LLM trying to write outside the
    # algorithm's own files
    stray = "<<<<<<< SEARCH Cargo.toml\nfoo\n=======\nbar\n>>>>>>> REPLACE\n" + edit(5)
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), stray], budget=b)
    st = loop.run()
    assert st.status == "exhausted" and st.best is None
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 0  # the candidate never reached the bench
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    rejected = [e for e in events if e["kind"] == "edits_rejected"]
    assert rejected and rejected[0]["paths"] == ["Cargo.toml"]


def test_baseline_is_budget_checked_before_the_first_bench_call(tmp_path):
    # mutation: an unchecked baseline spends the whole compute budget before the first check —
    # resolve_baseline compiles and scores both nonce sets in one evaluate of its own, and on a
    # cold cache that single call is the most expensive of the whole job
    b = Budget(usd=None, hours=None, iterations=20, compute_usd=0.0)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.state.baseline = None
    calls_before = len(fb.calls)
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", "base_id", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    with pytest.raises(BudgetExhausted) as ei:
        loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert ei.value.dimension == "compute_usd"
    assert len(fb.calls) == calls_before  # nothing was evaluated
    assert not (tmp_path / "cache").exists()


def test_baseline_compute_cost_is_charged_before_the_next_check(tmp_path):
    # The baseline is now a single evaluate, so "charged per call" and "charged once
    # resolve_baseline returns" cannot be told apart inside measure_baseline itself; what still
    # has to hold is that the charge is on spend, and on disk, before the NEXT budget check.
    # mutation: dropping the charge (or the save) in _BudgetedBench._metered lets the loop's
    # next check pass on a stale zero and burn an iteration past the compute cap
    b = Budget(usd=None, hours=None, iterations=20, compute_usd=0.015)  # one scored nonce = 0.01
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.state.baseline = None
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", "base_id", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    # the evaluate call charged before the next check
    assert loop.state.spend.compute_usd > 0.0
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["compute_usd"] > 0.0
    st = loop.run()
    assert st.status == "exhausted" and st.stop_reason == "compute_usd"
    assert fp.calls == [] and len(fb.calls) == 1  # the baseline evaluate only


def test_baseline_and_best_dirs_are_written(tmp_path):
    # spec §5.4: runs/<job>/baseline/ and best/ hold the current sources on disk.
    # mutation: never writing them leaves two directories the spec promises permanently empty
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.baseline = None
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", "base_id", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert (tmp_path / "baseline" / "mod.rs").read_text() == BASE_FILES["mod.rs"]
    results = json.loads((tmp_path / "baseline" / "results.json").read_text())
    assert len(results["training"]) == 4 and len(results["holdout"]) == 4
    st = loop.run()
    assert st.status == "won"
    assert "let k = 5;" in (tmp_path / "best" / "mod.rs").read_text()


NEVER_MATCHES = "<<<<<<< SEARCH mod.rs\nlet zzz = 1;\n=======\nlet zzz = 2;\n>>>>>>> REPLACE\n"
STRAY = "<<<<<<< SEARCH Cargo.toml\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"


def test_repair_round_keeps_the_blocks_already_applied(tmp_path):
    # One good block and one miss: the good block is applied, the miss goes to a repair round,
    # and the repair misses again. search_replace's contract is "repair, then skip whatever still
    # doesn't match", so the iteration must go on with the block that DID apply.
    # mutation: replacing the outcome with the repair round's own outcome resets `applied` to 0
    # and fails the iteration as "no edit block applied" although one block was applied
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5) + NEVER_MATCHES, NEVER_MATCHES],
                               budget=b)
    st = loop.run()
    assert st.status == "won" and "let k = 5;" in st.best.files["mod.rs"]
    assert len(fp.calls) == 3  # hypothesis, edit, exactly one repair round


def test_rejected_path_survives_a_repair_round(tmp_path):
    # spec §9: the out-of-scope block in the FIRST response must still fail the iteration when a
    # miss in the same response sends the loop through a repair round.
    # mutation: taking `rejected` from the repair round's outcome alone forgets the stray block
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), STRAY + edit(5) + NEVER_MATCHES, NEVER_MATCHES],
                               budget=b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 0  # the candidate never reached the bench


def test_compile_fix_rejects_out_of_scope_edits(tmp_path):
    # spec §9 applies to the compile-fix response too: a fix that touches a file outside the
    # algorithm fails the iteration instead of having its in-scope blocks applied.
    # mutation: ignoring `rejected` in the compile-fix loop lets the fix below reach "won"
    def swap(frm, to):
        return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {to};\n>>>>>>> REPLACE\n"

    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), swap(1, "BUG"), STRAY + swap("BUG", 5)],
                               budget=b)
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "Cargo.toml" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 1  # the broken edit only; the fix never built
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert [e["paths"] for e in events if e["kind"] == "edits_rejected"] == [["Cargo.toml"]]


def test_compile_failed_event_names_the_error_even_when_the_tail_is_all_warnings(tmp_path):
    # The event keeps the last 2000 characters of output; on a real C3 job that tail was 46
    # warnings and cargo's summary, so the terminal could not say what the error was.
    # mutation: reading the error out of the 2000-character tail instead of the whole output
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit("BUG"), "no edit blocks here"], budget=b)
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    real = fb.evaluate
    long_output = ("error[E0382]: borrow of moved value: `order`\n"
                   + "warning: unused variable: `hp`\n" * 100
                   + "error: could not compile `tig-algorithms` (lib) due to 1 previous error\n")
    assert len(long_output) > 2000 + 50

    def evaluate(request):
        res = real(request)
        if not res.compile.ok:
            res.compile.output = long_output
        return res

    fb.evaluate = evaluate
    loop.run()
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["kind"] == "compile_failed"]
    assert [e["error"] for e in failed] == ["error[E0382]: borrow of moved value: `order`"]
    assert "E0382" not in failed[0]["output"] and len(failed[0]["output"]) == 2000


def test_win_is_recorded_before_the_time_cap_bites(tmp_path):
    # mutation: checking the hours cap between the evaluate call and recording the win would
    # strand a proven winner as "exhausted" with best.holdout set but confirmed == []
    b = Budget(usd=None, hours=1.0, iterations=None, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.clock = lambda: 7200.0 if fb.calls else 0.0  # the cap passes during the evaluate
    st = loop.run()
    assert st.status == "won" and st.confirmed == [1]


def test_holdout_not_scored_is_a_false_positive(tmp_path):
    # mutation: treating holdout=None as "won" would confirm a candidate whose held-out run
    # timed out inside a C3 job
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    real = fb.evaluate

    def evaluate(request):
        r = real(request)
        r.holdout, r.holdout_reason = None, "timeout"
        return r
    fb.evaluate = evaluate
    st = loop.run()
    assert st.false_positives == [1] and st.confirmed == [] and st.status == "exhausted"
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    fp_ev = [e for e in events if e["kind"] == "false_positive"]
    assert fp_ev and "timeout" in fp_ev[0]["error"]


def test_pending_job_is_written_before_evaluate_and_cleared_after(tmp_path):
    # mutation: writing pending_job after evaluate returns means a kill during a 20-minute C3
    # job leaves nothing to reattach to; never clearing it makes every later resume "reattach"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    seen = {}
    real = fb.evaluate

    def evaluate(request):
        seen["pending"] = json.loads((tmp_path / "state.json").read_text())["pending_job"]
        return real(request)
    fb.evaluate = evaluate
    st = loop.run()
    assert seen["pending"]["purpose"] == 1
    assert seen["pending"]["files"] == {"mod.rs": "fn solve() { let k = 5; }\n"}
    assert seen["pending"]["hypothesis"]["title"] == "a"
    assert st.pending_job is None
    assert json.loads((tmp_path / "state.json").read_text())["pending_job"] is None


def test_pending_job_files_follow_a_compile_fix(tmp_path):
    # mutation: writing pending_job only in iterate() (and not in _bench_evaluate) leaves the
    # pre-fix sources in the record, so a resume during the fix round's evaluate reattaches to,
    # or rebuilds, a request for code the loop has already thrown away
    broken = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    fix = "<<<<<<< SEARCH mod.rs\nlet k = BUG;\n=======\nlet k = 5;\n>>>>>>> REPLACE\n"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), broken, fix])
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    seen = []
    real = fb.evaluate

    def evaluate(request):
        seen.append(json.loads((tmp_path / "state.json").read_text())["pending_job"]["files"])
        return real(request)
    fb.evaluate = evaluate
    st = loop.run()
    assert st.status == "won"
    assert seen == [{"mod.rs": "fn solve() { let k = BUG; }\n"},
                    {"mod.rs": "fn solve() { let k = 5; }\n"}]


def test_resume_with_a_pending_job_re_evaluates_without_a_new_hypothesis(tmp_path):
    # mutation: ignoring pending_job on resume proposes a fresh hypothesis (fp.calls != [])
    # and abandons the job C3 is still billing for
    store = JobStore(tmp_path)
    store.write_spec(spec(Budget(usd=None, hours=None, iterations=1, compute_usd=None)))
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    files = {"mod.rs": "fn solve() { let k = 5; }\n"}
    st.pending_job = {"purpose": 1, "files": files,
                      "hypothesis": {"title": "a", "description": "d",
                                     "strategy_tag": "local_search"}}
    store.save(st)
    fb = FakeBench(quality_from_files)
    fp = FakeProvider([])  # an empty script raises if the loop asks for a completion
    loop = Loop(store.read_spec(), store.load(), store, fp, fb, template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    result = loop.run()
    assert fp.calls == [] and len(fb.calls) == 1 and fb.calls[0].files == files
    assert result.status == "won" and result.confirmed == [1] and result.iteration == 1
    assert result.hypotheses[0]["title"] == "a" and result.hypotheses[0]["outcome"] == "won"
    assert result.strategy_counts == {"local_search": 1}  # counted once, not on both runs
    assert result.pending_job is None


def test_resume_does_not_discard_the_pending_iteration_dir(tmp_path):
    # mutation: _discard_incomplete_iteration wiping iterations/0001 on a pending resume deletes
    # the hypothesis.json the package later reads
    store = JobStore(tmp_path)
    store.write_spec(spec(Budget(usd=None, hours=None, iterations=1, compute_usd=None)))
    st = JobState.fresh(Spend(started_at=0.0))
    st.baseline = baseline()
    st.status = "researching"
    st.pending_job = {"purpose": 1, "files": {"mod.rs": "fn solve() { let k = 5; }\n"},
                      "hypothesis": {"title": "a", "description": "d",
                                     "strategy_tag": "local_search"}}
    store.save(st)
    (store.iteration_dir(1) / "hypothesis.json").write_text("{}")
    loop = Loop(store.read_spec(), store.load(), store, FakeProvider([]),
                FakeBench(quality_from_files), template_rs="x",
                clock=lambda: 0.0, sleep=lambda s: None)
    loop.run()
    assert (tmp_path / "iterations" / "0001" / "hypothesis.json").exists()


def test_stale_baseline_pending_job_is_cleared(tmp_path):
    # mutation: dropping run()'s `purpose == "baseline"` branch leaves the stale record on disk
    # until iterate() happens to overwrite it — so a run that exhausts its budget (or is stopped)
    # before the first iteration keeps a baseline job_id a C3 backend would reattach to. The
    # end-state assertions alone cannot see that, so snapshot the record before iterating.
    # mutation: a leftover {"purpose": "baseline"} record must also never be taken for a pending
    # iteration (an int purpose), which would send _resume_pending looking for a hypothesis
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    loop.state.pending_job = {"purpose": "baseline", "job_id": "job_old"}
    seen = {}
    real = fp.complete

    def complete(system, user):
        seen.setdefault("pending",
                        json.loads((tmp_path / "state.json").read_text())["pending_job"])
        return real(system, user)
    fp.complete = complete
    st = loop.run()
    assert seen["pending"] is None  # cleared before the first iteration, not by it
    assert st.status == "won" and st.pending_job is None


def test_measure_baseline_keeps_a_stored_baseline_job_record(tmp_path):
    # mutation: overwriting pending_job with a fresh {"purpose": "baseline"} drops the job_id a
    # killed baseline run stored, so the resume submits (and pays for) a second baseline job
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])
    loop.state.baseline = None
    loop.state.pending_job = {"purpose": "baseline", "job_id": "job_b", "request_hash": "h"}
    seen = {}
    real = fb.evaluate

    def evaluate(request):
        seen["pending"] = dict(loop.state.pending_job)
        return real(request)
    fb.evaluate = evaluate
    mainnet = types.SimpleNamespace(
        top_algorithm=lambda ch: ("base", "base_id", 1),
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert seen["pending"]["job_id"] == "job_b" and loop.state.pending_job is None


def test_bench_cancelled_stops_the_run_as_cancelled(tmp_path):
    # mutation: letting BenchCancelled escape run() tracebacks out of the CLI instead of
    # packaging the best so far
    from talos.bench import BenchCancelled
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)])

    def evaluate(request):
        raise BenchCancelled("job_x")
    fb.evaluate = evaluate
    st = loop.run()
    assert st.status == "cancelled" and "job_x" in st.stop_reason


def test_focused_request_narrows_training_and_guards_holdout(tmp_path):
    # mutation: forgetting the guard sends only t's held-out set; sending the whole baseline
    # makes the in-job holdout decision raise a nonce mismatch on every iteration
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    loop.run()
    req = fb.calls[0]
    assert req.training == [TR2[0]]
    assert req.holdout == [HO2[0], TR2[1]]
    assert {(r.track, r.nonce) for r in req.baseline_training} == {("t", n) for n in range(4)}


def test_focused_win_is_confirmed_on_the_track_and_the_guard(tmp_path):
    # mutation: comparing the guard rows against the baseline's held-out rows (wrong nonces)
    # raises a ScoringError and records a false positive instead of the win
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    st = loop.run()
    assert st.status == "won" and st.confirmed == [1]
    assert {r.track for r in st.best.holdout} == {"t", "u"}


def test_focused_guard_regression_is_a_false_positive(tmp_path):
    # mutation: confirming on the focus track alone declares a win that broke track u
    def scores(challenge, files, ns):
        import re
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if ns.track == "u" and k != 1:
            return [90 for _ in ns.nonces()]  # the edit regresses the other track
        return [100 + k - 1 for _ in ns.nonces()]
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], scores, b, track="t",
                               two_tracks=True)
    st = loop.run()
    assert st.status == "exhausted" and st.false_positives == [1] and st.confirmed == []
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    fp_event = next(e for e in events if e["kind"] == "false_positive")
    assert fp_event["holdout"]["worst_rel_delta"] < 0


def test_unfocused_request_is_unchanged(tmp_path):
    # mutation: the focus path leaking into unfocused jobs changes every existing request
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], two_tracks=True)
    loop.run()
    req = fb.calls[0]
    assert req.training == TR2 and req.holdout == HO2
    assert len(req.baseline_training) == 8


def test_focused_context_names_track_and_guards(tmp_path):
    # mutation: dropping either field leaves the model aiming at every track
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], track="t", two_tracks=True)
    ctx = loop._context()
    assert ctx.track == "t" and ctx.guard_tracks == ["u"]
    loop2, *_ = make(tmp_path / "b", [hyp("a"), edit(5)], two_tracks=True)
    ctx2 = loop2._context()
    assert ctx2.track is None and ctx2.guard_tracks == []


DEAD_WARNING = ("warning: function `polish` is never used\n"
                "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:3:4\n")


def add_polish(k, call):
    body = f"let k = {k};" + (" polish();" if call else "")
    return ("<<<<<<< SEARCH mod.rs\nfn solve() { let k = 1; }\n=======\n"
            f"fn polish() {{}}\nfn solve() {{ {body} }}\n>>>>>>> REPLACE\n")


def dead_if_uncalled(files):
    src = files["mod.rs"]
    return DEAD_WARNING if "fn polish" in src and "polish();" not in src else "ok"


def test_dead_new_code_gets_a_fix_round_before_scoring(tmp_path):
    # mutation: treating a dead_code result like a scored one records failed:score for a
    # candidate whose change was never on the solve path
    wire = "<<<<<<< SEARCH mod.rs\nlet k = 2; }\n=======\nlet k = 2; polish(); }\n>>>>>>> REPLACE\n"
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), add_polish(2, call=False), wire], budget=b)
    fb._compile_output = dead_if_uncalled
    st = loop.run()
    assert st.status == "won"
    assert len(fb.calls) == 2
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    dead = [e for e in events if e["kind"] == "dead_code"]
    assert [e["names"] for e in dead] == [["mod.rs: polish"]]
    # the request tells the bench which functions the edited code already had
    assert fb.calls[0].prior_functions == {"mod.rs": ["solve"]}


def test_dead_new_code_after_the_fix_rounds_fails_the_iteration(tmp_path):
    # mutation: falling through to scoring after the rounds run out scores the no-op after all
    still_dead = "<<<<<<< SEARCH mod.rs\nlet k = 2;\n=======\nlet k = 3;\n>>>>>>> REPLACE\n"
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), add_polish(2, call=False), still_dead],
                               budget=b, thresholds=Thresholds(compile_fix_rounds=1))
    fb._compile_output = dead_if_uncalled
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:dead_code"
    assert "polish" in st.hypotheses[0]["error"]
    assert len(fb.calls) == 2


def test_a_scored_record_carries_the_numbers_the_next_prompt_needs(tmp_path):
    # mutation: recording only the outcome leaves the recall block with titles alone
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=b,
                               scores=lambda ch, files, ns: [100] * ns.count)
    fb._runtime_ms = 3
    st = loop.run()
    rec = st.hypotheses[0]
    assert rec["outcome"] == "failed:score"
    assert rec["mean_rel_delta"] == pytest.approx(0.0) and rec["worst_track"] == "t"
    assert rec["worst_rel_delta"] == pytest.approx(0.0)
    assert rec["runtime_ratio"] == pytest.approx(3.0)  # baseline rows run 1 ms


def test_candidate_timeouts_follow_the_baseline_runtime_per_track(tmp_path):
    # iteration 2 of run 20260916-095103 ran 395 s per nonce against a 28 s baseline under a
    # flat 600 s timeout and ate a third of the hours budget. mutation: a flat timeout, or a
    # ratio without the floor, sends 600 or 3 seconds for a 1 ms baseline track
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=b, two_tracks=True)
    loop.state.baseline.training[0].runtime_ms = 100_000  # track "t"; track "u" stays at 1 ms
    loop.run()
    assert fb.calls[0].timeouts == {"t": 300, "u": 60}
    # the ceiling never exceeds the flat timeout the baseline itself ran under
    loop2, fp2, fb2, store2 = make(tmp_path / "b", [hyp("a"), edit(2)], budget=b)
    loop2.state.baseline.training[0].runtime_ms = 10_000_000
    loop2.run()
    assert fb2.calls[0].timeouts == {"t": 600}
    # a zero ceiling disables the guard
    loop3, fp3, fb3, store3 = make(tmp_path / "c", [hyp("a"), edit(2)], budget=b,
                                   thresholds=Thresholds(runtime_ceiling=0.0))
    loop3.run()
    assert fb3.calls[0].timeouts is None


def test_the_spec_hyperparameters_reach_baseline_candidates_and_prompts(tmp_path):
    hp = {"t": {"x": 1}}
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=b)
    loop.spec = replace(loop.spec, hyperparameters=hp,
                        baseline_algorithm={"name": "pinned", "id": "p1", "adoption": 3})
    loop.state.baseline = None

    def no_top(ch):
        raise AssertionError("the pinned algorithm must be used")
    mainnet = types.SimpleNamespace(
        top_algorithm=no_top,
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: "pub fn solve_challenge(")
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    # mutation: measure_baseline not passing algorithm= asks mainnet again
    assert loop.state.baseline.name == "pinned"
    # mutation: not passing hyperparameters= measures the baseline without the map
    assert fb.calls[0].hyperparameters == hp
    loop.run()
    # mutation: _request without hyperparameters= scores candidates without the map
    assert len(fb.calls) >= 2 and all(c.hyperparameters == hp for c in fb.calls)
    # mutation: _context without hyperparameters= leaves the model unaware of the keys
    assert any('track t: {"x":1}' in user for _system, user in fp.calls)


def native(tmp_path, script, **kw):
    kw.setdefault("scoring", "native")
    return make(tmp_path, script, **kw)


def test_native_research_validates_a_new_best_on_the_metered_path(tmp_path):
    loop, fp, fb, store = native(tmp_path, [hyp("bump k"), edit(5)])
    st = loop.run()
    research, validation = fb.calls[0], fb.calls[1]
    assert research.mode == "native" and research.fuel_budgets_us == {"t": 1_000_000}
    # held-out is decided on the metered run, never on native results
    assert research.holdout == [] and research.baseline_training is None
    assert validation.mode == "metered" and validation.holdout == HO
    assert validation.baseline_training is not None and validation.prior_functions is None
    # mutation: best set from the native rows (they carry solve_us, not fuel_consumed)
    assert st.best.iteration == 1 and st.best.training[0].fuel_consumed == 1000
    assert st.validations == [{"iteration": 1, "outcome": "validated", "reason": None}]
    assert st.status == "won" and st.confirmed == [1]
    # mutation: confirming on the native call's (empty) held-out rows instead of the metered
    # ones; FakeBench counts a held-out run for the native call too, so count rows, not runs
    assert len(st.best.holdout) == 4 and all(r.fuel_consumed == 1000 for r in st.best.holdout)


def test_a_native_score_carries_no_runtime_ratio_against_the_metered_baseline(tmp_path):
    loop, fp, fb, store = native(tmp_path, [hyp("lower k"), edit(0)],
                                 budget=Budget(usd=None, hours=None, iterations=1,
                                               compute_usd=None))
    st = loop.run()
    # mutation: a native-over-metered ratio reaches the recall prompt as "runtime 0.4x
    # baseline" for code that is no faster at all
    assert "runtime_ratio" not in st.hypotheses[0]
    scored = [json.loads(x) for x in
              (store.run_dir / "timeline.jsonl").read_text().splitlines()
              if json.loads(x)["kind"] == "scored"]
    assert scored[0]["runtime_ratio"] is None


def test_a_native_loss_is_not_validated(tmp_path):
    # k = 0 scores 99 against the baseline's 100: worse, so nothing to validate
    loop, fp, fb, store = native(tmp_path, [hyp("lower k"), edit(0)],
                                 budget=Budget(usd=None, hours=None, iterations=1,
                                               compute_usd=None))
    loop.run()
    # mutation: validating every scored candidate doubles the compute of every iteration
    assert [c.mode for c in fb.calls] == ["native"]


def test_a_nondeterministic_candidate_is_demoted_and_research_stays_on_the_old_code(tmp_path):
    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [x + 1 for x in q] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5), hyp("b"), edit(3)], bench=fb,
                                budget=Budget(usd=None, hours=None, iterations=2,
                                              compute_usd=None))
    st = loop.run()
    # spec §6: both qualities recorded. k = 5 is 104 natively and 105 metered on every nonce
    assert st.validations[0] == {
        "iteration": 1, "outcome": "demoted", "reason": "nondeterministic",
        "mismatches": [{"track": "t", "nonce": n, "native": 104, "metered": 105}
                       for n in range(4)]}
    assert st.hypotheses[0]["outcome"] == "failed:validation"
    # mutation: building on the demoted candidate; edit(3) searches for "let k = 1;", which
    # only the baseline still has, so iteration 2 would fail to apply its edit
    assert st.best.iteration == 2 and "let k = 3;" in st.best.files["mod.rs"]
    # mutation: a demotion that leaves best pointing at the rejected candidate
    assert not (store.run_dir / "best" / "mod.rs").read_text().count("let k = 5;")


def test_a_metered_build_failure_demotes_with_its_reason(tmp_path):
    fb = FakeBench(quality_from_files, metered_compile_ok=lambda files: False)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb,
                                budget=Budget(usd=None, hours=None, iterations=1,
                                              compute_usd=None))
    st = loop.run()
    assert st.validations[0]["reason"] == "native_metered_build_mismatch" and st.best is None


def _calibrated(loop, tmp_path, ratio=1.0):
    """A fresh native job's first calibrate, from a stored record. make() pre-sets native
    budgets for the loop tests; a fresh job has none, so they are cleared first."""
    from talos import calibration
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files,
                                      loop.spec.hyperparameters)
    calibration.save(tmp_path / "cal", "knapsack", key, calibration.new_record({"t": ratio}))
    loop.calibrate(tmp_path / "cal", "hw")
    return key


def test_resume_keeps_the_jobs_budgets_and_its_pending_job(tmp_path):
    from talos import calibration
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000, budgets={"t": 1_234_567})
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files, None)
    # another job has since tightened the shared record: 0.25 x 1e7 x 0.8 = 2_000_000
    calibration.save(tmp_path / "cal", "knapsack", key, calibration.new_record({"t": 0.25}))
    pend = {"purpose": 1, "hypothesis": {}, "files": {}, "job_id": "job_1",
            "request_hash": "abc"}
    loop.state.pending_job = dict(pend)
    loop.calibrate(tmp_path / "cal", "hw")
    # mutation: re-deriving budgets from the record changes the in-flight request's hash, and
    # the resume submits a second job while the first still bills
    assert loop.state.fuel_budgets_us == {"t": 1_234_567}
    # mutation: re-measuring on resume overwrites pending_job and loses job_1's id
    assert loop.state.pending_job == pend and fb.calls == []
    # demotions still reach the record
    assert loop._calibration == (tmp_path / "cal", key)


def test_three_fuel_misses_lower_the_margin_and_the_budgets(tmp_path):
    from talos import calibration
    def metered(ch, files, ns):
        return ["out_of_fuel"] + quality_from_files(ch, files, ns)[1:]
    fb = FakeBench(quality_from_files, metered_scores=metered)
    script = [hyp("a"), edit(5), hyp("b"), edit(6), hyp("c"), edit(7)]
    loop, fp, _, store = native(tmp_path, script, bench=fb, fuel=10_000_000,
                                budget=Budget(usd=None, hours=None, iterations=3,
                                              compute_usd=None))
    key = _calibrated(loop, tmp_path)
    # 1.0 us/fuel x 10_000_000 fuel x 0.8 = 8_000_000 us, by hand
    assert loop.state.fuel_budgets_us == {"t": 8_000_000}
    st = loop.run()
    assert [v["reason"] for v in st.validations] == ["fuel_proxy_miss"] * 3
    rec = calibration.load(tmp_path / "cal", "knapsack", key)
    assert rec["tracks"]["t"]["margin"] == pytest.approx(0.7)
    assert rec["demotions"] == {"fuel_proxy_miss": 3}
    # mutation: lowering the stored margin without recomputing the job's budgets
    assert st.fuel_budgets_us == {"t": 7_000_000}
    assert fb.calls[-2].fuel_budgets_us == {"t": 8_000_000}  # iteration 3's research call


def test_a_kill_after_the_record_save_does_not_count_the_miss_twice_on_resume(tmp_path):
    # Issue #42 case 2: the calibration record is saved before state.json, which still holds
    # pending_job["native_training"]. The resume validates iteration 1 again.
    from talos import calibration

    class Killed(BaseException):
        pass

    def metered(ch, files, ns):
        return ["out_of_fuel"] + quality_from_files(ch, files, ns)[1:]
    fb = FakeBench(quality_from_files, metered_scores=metered)
    one = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, fuel=10_000_000,
                                budget=one)
    key = _calibrated(loop, tmp_path)

    def kill(*a, **kw):
        raise Killed

    loop._finish_iteration = kill  # the hard kill lands between the two saves
    with pytest.raises(Killed):
        loop.run()
    assert store.load().pending_job.get("native_training")
    assert calibration.load(tmp_path / "cal", "knapsack", key)["tracks"]["t"]["misses"] == 1

    resumed = Loop(store.read_spec(), store.load(), store, FakeProvider([]), fb,
                   template_rs="pub fn solve_challenge(", clock=lambda: 0.0,
                   sleep=lambda s: None, thresholds=Thresholds())
    resumed.calibrate(tmp_path / "cal", "hw")
    st = resumed.run()
    assert [v["reason"] for v in st.validations] == ["fuel_proxy_miss"]
    assert st.pending_job is None and st.iteration == 1
    rec = calibration.load(tmp_path / "cal", "knapsack", key)
    # mutation: no counted-token check gives misses 2 and demotions 2
    assert rec["tracks"]["t"]["misses"] == 1
    assert rec["demotions"] == {"fuel_proxy_miss": 1}


def test_a_kill_after_a_margin_change_does_not_record_the_validation_twice(tmp_path):
    # The record starts one miss short of MISS_LIMIT, so this job's miss lowers the margin. A
    # state.json save there would store the validation while pending_job still holds
    # native_training, and the resume would validate iteration 1 again and append it twice.
    from talos import calibration

    class Killed(BaseException):
        pass

    def metered(ch, files, ns):
        return ["out_of_fuel"] + quality_from_files(ch, files, ns)[1:]
    fb = FakeBench(quality_from_files, metered_scores=metered)
    one = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, fuel=10_000_000,
                                budget=one)
    key = _calibrated(loop, tmp_path)
    rec = calibration.load(tmp_path / "cal", "knapsack", key)
    rec["tracks"]["t"]["misses"] = calibration.MISS_LIMIT - 1
    calibration.save(tmp_path / "cal", "knapsack", key, rec)

    def kill(*a, **kw):
        raise Killed

    loop._finish_iteration = kill
    with pytest.raises(Killed):
        loop.run()
    saved = store.load()
    assert saved.pending_job.get("native_training")
    # mutation: saving state.json in _note_demotion stores the validation before the kill
    assert saved.validations == [] and saved.fuel_budgets_us == {"t": 8_000_000}

    resumed = Loop(store.read_spec(), store.load(), store, FakeProvider([]), fb,
                   template_rs="pub fn solve_challenge(", clock=lambda: 0.0,
                   sleep=lambda s: None, thresholds=Thresholds())
    resumed.calibrate(tmp_path / "cal", "hw")
    st = resumed.run()
    assert st.pending_job is None and st.iteration == 1
    assert [v["iteration"] for v in st.validations] == [1]
    # The repeat counts nothing but returns the first count's margin change, so the resumed
    # job still tightens its budgets: 1.0 us/fuel x 10_000_000 x 0.7.
    # mutation: count_validation returning False for a repeat leaves 8_000_000
    assert st.fuel_budgets_us == {"t": 7_000_000}
    assert store.load().fuel_budgets_us == {"t": 7_000_000}
    rec = calibration.load(tmp_path / "cal", "knapsack", key)
    assert rec["tracks"]["t"]["misses"] == 0 and rec["demotions"] == {"fuel_proxy_miss": 1}


def test_resume_mid_validation_goes_straight_to_the_metered_run(tmp_path):
    loop, fp, fb, store = native(tmp_path, [])  # no LLM reply scripted: none may be asked for
    files = {"mod.rs": "fn solve() { let k = 5; }\n"}
    native_rows = [NonceResult("t", n, True, 104, 1, None, solve_us=500) for n in range(4)]
    loop.state.pending_job = {"purpose": 1, "hypothesis": {"title": "h", "description": "d",
                                                           "strategy_tag": "local_search"},
                              "files": files,
                              "native_training": [r.to_dict() for r in native_rows]}
    loop._save()
    st = loop.run()
    # mutation: re-entering at _score_candidate scores natively again before validating
    assert fb.calls[0].mode == "metered"
    assert st.best is not None and st.best.iteration == 1


def test_calibrate_rescores_metered_fuel_when_the_cached_baseline_has_none(tmp_path):
    from talos import calibration
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    loop.state.scoring = None  # as a fresh native job before calibration
    loop.calibrate(tmp_path / "cal", "hw")
    # the test baseline's rows have no fuel_consumed, like every pre-change cached baseline
    # mutation: falling back to metered research instead of measuring the fuel
    assert [c.mode for c in fb.calls] == ["metered", "native"]
    assert all(c.holdout == [] for c in fb.calls)
    # FakeBench: fuel 1000, solve 500 us -> 0.5 us/fuel; 0.5 x 10_000_000 x 0.8 = 4_000_000
    assert loop.state.scoring == "native" and loop.state.fuel_budgets_us == {"t": 4_000_000}
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files, None)
    assert calibration.load(tmp_path / "cal", "knapsack", key) is not None
    assert loop.state.pending_job is None


def test_calibrate_uses_the_baselines_own_fuel_when_it_has_it(tmp_path):
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    loop.state.scoring = None
    loop.state.baseline.training = [replace(r, fuel_consumed=2000)
                                    for r in loop.state.baseline.training]
    loop.calibrate(tmp_path / "cal", "hw")
    assert [c.mode for c in fb.calls] == ["native"]
    # 500 us / 2000 fuel = 0.25; 0.25 x 10_000_000 x 0.8 = 2_000_000
    assert loop.state.fuel_budgets_us == {"t": 2_000_000}


def test_calibrate_reuses_a_stored_record_without_a_bench_call(tmp_path):
    # _calibrated clears make()'s preset budgets, so this is a fresh job reading the record
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    _calibrated(loop, tmp_path, ratio=0.25)
    assert fb.calls == [] and loop.state.fuel_budgets_us == {"t": 2_000_000}


def test_calibration_without_solve_times_falls_back_to_metered_research(tmp_path):
    fb = FakeBench(quality_from_files, solve_us=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb,
                                budget=Budget(usd=None, hours=None, iterations=1,
                                              compute_usd=None))
    loop.state.scoring = None
    loop.calibrate(tmp_path / "cal", "hw")
    assert loop.state.scoring == "metered"
    events = [json.loads(x)["kind"] for x in
              (store.run_dir / "timeline.jsonl").read_text().splitlines()]
    assert "calibration_fallback" in events
    n_before = len(fb.calls)
    loop.run()
    # mutation: a native run with no budget
    assert all(c.mode == "metered" for c in fb.calls[n_before:])


def test_a_metered_job_never_calibrates(tmp_path):
    loop, fp, fb, store = make(tmp_path, [])
    loop.calibrate(tmp_path / "cal", "hw")
    assert fb.calls == [] and loop.state.scoring is None and loop._scoring() == "metered"


def test_calibration_is_budget_checked_before_its_first_call(tmp_path):
    loop, fp, fb, store = native(tmp_path, [],
                                 budget=Budget(usd=None, hours=None, iterations=5,
                                               compute_usd=0.0))
    loop.state.scoring = None
    # mutation: `if budget:` lets a zero compute cap through (AGENTS.md invariant 5)
    with pytest.raises(BudgetExhausted):
        loop.calibrate(tmp_path / "cal", "hw")
    assert fb.calls == []


def test_a_focused_native_job_needs_a_budget_only_for_its_track(tmp_path):
    from talos import calibration
    loop, fp, fb, store = native(tmp_path, [], track="t", two_tracks=True)
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    key = calibration.calibration_key("knapsack", "hw", loop.state.baseline.files, None)
    calibration.save(tmp_path / "cal", "knapsack", key, calibration.new_record({"t": 1.0}))
    loop.calibrate(tmp_path / "cal", "hw")
    # the guard track u is scored only in the metered validation, so it needs no budget
    # mutation: requiring every track's ratio sends a focused job to the metered fallback
    assert loop.state.scoring == "native"


def test_a_resumed_calibration_reuses_its_stored_metered_rows(tmp_path):
    loop, fp, fb, store = native(tmp_path, [], fuel=10_000_000)
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    # killed during the native call: the metered rows (fuel 2000) were stored before it
    rows = [NonceResult("t", n, True, 100, 1, None, fuel_consumed=2000) for n in range(4)]
    loop.state.pending_job = {"purpose": "calibration",
                              "calibration_metered": [r.to_dict() for r in rows]}
    loop.calibrate(tmp_path / "cal", "hw")
    # mutation: re-running the metered call on resume, whose request hash differs from the
    # in-flight native job's, so C3 submits a new job and the native one is orphaned
    assert [c.mode for c in fb.calls] == ["native"]
    # 500 us / 2000 fuel = 0.25 (the stored rows, not FakeBench's 1000); x 1e7 x 0.8
    assert loop.state.fuel_budgets_us == {"t": 2_000_000}
    assert loop.state.pending_job is None


def test_a_calibration_stores_its_metered_rows_before_the_native_call(tmp_path):
    seen = []

    class Recording(FakeBench):
        def evaluate(self, request):
            if request.mode == "native":
                seen.append(store.load().pending_job)  # what a kill here would leave
            return super().evaluate(request)

    loop, fp, _, store = native(tmp_path, [], fuel=10_000_000,
                                bench=Recording(quality_from_files))
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    loop.calibrate(tmp_path / "cal", "hw")
    # mutation: keeping the metered rows only in a local, so a kill here loses them
    assert seen[0]["purpose"] == "calibration"
    assert [r["fuel_consumed"] for r in seen[0]["calibration_metered"]] == [1000] * 4
    assert store.load().pending_job is None  # cleared once the native call is back


def test_a_native_calibration_build_failure_scores_the_job_metered(tmp_path):
    # The GPU runner template has never been compiled live; if it fails, the job still runs.
    fb = FakeBench(quality_from_files, compile_ok=lambda files: False,
                   metered_compile_ok=lambda files: True)
    loop, fp, _, store = native(tmp_path, [], bench=fb)
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    loop.calibrate(tmp_path / "cal", "hw")
    assert loop.state.scoring == "metered" and loop.state.fuel_budgets_us is None
    assert [c.mode for c in fb.calls][-1] == "native"
    events = [json.loads(x) for x in
              (store.run_dir / "timeline.jsonl").read_text().splitlines()]
    fallback = [e for e in events if e["kind"] == "calibration_fallback"]
    # mutation: dropping the native compile check reports "no native budget" with no build
    # output, so the user cannot tell a broken runner from a baseline without ratios
    assert any("native build" in e["reason"] and "E0308" in e.get("output", "")
               for e in fallback)
    assert list((tmp_path / "cal").rglob("*.json")) == []  # a failed build caches nothing


def test_a_metered_calibration_build_failure_falls_back_with_its_output(tmp_path):
    fb = FakeBench(quality_from_files, metered_compile_ok=lambda files: False)
    loop, fp, _, store = native(tmp_path, [], bench=fb)
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    loop.calibrate(tmp_path / "cal", "hw")
    events = [json.loads(x) for x in
              (store.run_dir / "timeline.jsonl").read_text().splitlines()]
    fallback = [e for e in events if e["kind"] == "calibration_fallback"]
    # mutation: a failed metered build silently becomes "no fuel", and the event hides why
    assert any("metered build" in e["reason"] and "E0308" in e.get("output", "")
               for e in fallback)
    assert loop.state.scoring == "metered" and [c.mode for c in fb.calls] == ["metered"]


def raising(*items):
    """A FakeProvider script that raises each exception instance in `items` and returns the rest."""
    queue = list(items)

    def script(system, user):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item
    return script


def test_provider_timeout_fails_one_iteration(tmp_path):
    # issue #35: a claude CLI timeout ended the whole run. It must fail that iteration only.
    # mutation: not catching ProviderTimeout in iterate() ends the run as "failed"
    loop, fp, fb, store = make(tmp_path, raising(hyp("a"), ProviderTimeout("claude CLI timed out"),
                                                 hyp("b"), edit(5)))
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 2
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "timed out" in st.hypotheses[0]["error"]
    assert st.pending_job is None


def test_other_provider_errors_still_stop_the_run(tmp_path):
    # mutation: catching every ProviderError in iterate() spins a broken provider through
    # iterations that fail instantly instead of stopping with the error
    loop, fp, fb, store = make(tmp_path, raising(ProviderError("claude CLI failed: boom"),
                                                 hyp("b"), edit(5)))
    st = loop.run()
    assert st.status == "failed" and st.stop_reason == "provider: claude CLI failed: boom"
    assert len(fp.calls) == 1


def test_provider_timeout_in_a_repair_round_keeps_the_applied_blocks(tmp_path):
    # mutation: letting the repair round's timeout escape fails an iteration that had an edit
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, raising(hyp("a"), edit(5) + NEVER_MATCHES,
                                                 ProviderTimeout("claude CLI timed out")),
                               budget=b)
    st = loop.run()
    assert st.status == "won" and "let k = 5;" in st.best.files["mod.rs"]
    assert len(fp.calls) == 3


def test_provider_timeout_in_a_compile_fix_is_a_compile_failure(tmp_path):
    # mutation: letting the fix round's timeout escape stops the run with a pending job
    bug = "<<<<<<< SEARCH mod.rs\nlet k = 1;\n=======\nlet k = BUG;\n>>>>>>> REPLACE\n"
    loop, fp, fb, store = make(tmp_path, raising(hyp("a"), bug,
                                                 ProviderTimeout("claude CLI timed out"),
                                                 hyp("b"), edit(5)))
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 2
    assert st.hypotheses[0]["outcome"] == "failed:compile"
    assert len(fb.calls) == 2  # the broken edit once, no re-evaluation, then the winning edit


# ── lineage tree and chains ───────────────────────────────────────────


def heldout_flat(challenge, files, ns):
    """quality_from_files on training; the baseline's 100 on held-out, so no candidate is ever
    confirmed and the job keeps researching."""
    if ns.start >= 1_000_000:
        return [100 for _ in ns.nonces()]
    return quality_from_files(challenge, files, ns)


def test_default_job_is_unchanged_and_records_a_tree(tmp_path):
    # mutation: not appending a Node, or building it with the wrong parent, chain or flags
    loop, fp, fb, store = make(tmp_path, [hyp("bump k"), edit(5)])
    st = loop.run()
    assert st.status == "won" and st.best.iteration == 1 and st.confirmed == [1]
    node = st.nodes[0]
    assert (node.id, node.parent, node.chain) == (1, 0, 0)
    assert node.accepted is True and node.best is True
    # mutation: leaving the node's outcome at "improved" when _mark_won rewrites the record
    assert node.outcome == "won" == st.hypotheses[0]["outcome"]
    assert node.score == pytest.approx(0.04) and node.tracks == {"t": pytest.approx(0.04)}
    assert st.chains[0].best == 1 and st.chains[0].best_score == pytest.approx(0.04)
    rec = st.hypotheses[0]
    assert (rec["chain"], rec["against"], rec["accepted"]) == (0, 0, True)
    assert rec["design_family"] == "unlabelled"  # hyp() gives none; parse_hypothesis fills it


def test_two_greedy_chains_take_turns_and_keep_their_own_parent(tmp_path):
    # mutation: building chain 1's iteration 2 on the job best (k=5) makes its SEARCH text
    # `let k = 1;` miss, and that iteration fails as failed:edit
    # mutation: leaving parent_gap unset drops the parent line from iteration 3's prompt
    # mutation: not resetting a chain's stall on a new best leaves chain 0 at 1
    b = Budget(usd=None, hours=None, iterations=7, compute_usd=None)
    script = [hyp("a"), edit(5),            # 1, chain 0: from the root to k=5
              hyp("b"), edit(3),            # 2, chain 1: from the root to k=3
              hyp("c"), edit(6, frm=5),     # 3, chain 0: from node 1
              hyp("d"), edit(4, frm=3),     # 4, chain 1: from node 2
              hyp("e"), edit(0, frm=6),     # 5, chain 0: worse, failed:score
              hyp("f"), edit(1, frm=4),     # 6, chain 1: the baseline's score, failed:score
              hyp("g"), edit(7, frm=6)]     # 7, chain 0: from node 3, a new best
    loop, fp, fb, store = make(tmp_path, script, scores=heldout_flat, budget=b,
                               spec_kw={"chains": 2})
    st = loop.run()
    assert st.status == "exhausted"
    assert [h["outcome"] for h in st.hypotheses] == (
        ["improved"] * 4 + ["failed:score"] * 2 + ["improved"])
    assert [n.chain for n in st.nodes] == [0, 1, 0, 1, 0, 1, 0]
    assert [n.parent for n in st.nodes] == [0, 0, 1, 2, 3, 4, 3]
    assert [n.id for n in st.nodes] == [1, 2, 3, 4, 5, 6, 7]
    assert [h["against"] for h in st.hypotheses] == [0, 0, 1, 2, 3, 4, 3]
    assert [(c.index, c.best) for c in st.chains] == [(0, 7), (1, 4)]
    assert [c.stall for c in st.chains] == [0, 1]
    system, user = fp.calls[4]  # iteration 3's hypothesis call
    assert "Parent (node #1): +4.000%" in system + user


def test_recall_is_per_chain(tmp_path):
    # mutation: an unfiltered _failed() shows chain 0's failure to chain 1
    # The positive control ("fail-c1" present) catches recall never being shown, which would
    # make the absence check pass on its own.
    b = Budget(usd=None, hours=None, iterations=4, compute_usd=None)
    script = [hyp("fail-c0"), edit(1),      # 1, chain 0: no change, failed:score
              hyp("fail-c1"), edit(1),      # 2, chain 1: no change, failed:score
              hyp("third"), edit(1),        # 3, chain 0
              hyp("fourth"), edit(1)]       # 4, chain 1
    loop, fp, fb, store = make(tmp_path, script, budget=b, spec_kw={"chains": 2},
                               thresholds=Thresholds(recall=2, distill=99))
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == ["failed:score"] * 4
    assert [h["chain"] for h in st.hypotheses] == [0, 1, 0, 1]
    system, user = fp.calls[6]  # iteration 4's hypothesis call
    assert "fail-c1" in user
    assert "fail-c0" not in system + user
    # a failed iteration is never accepted and raises its chain's stall
    assert [n.accepted for n in st.nodes] == [False] * 4
    assert [(c.best, c.stall) for c in st.chains] == [(0, 2), (0, 2)]


def test_job_from_before_lineage_resumes_on_its_best(tmp_path):
    # Review Focus 3: a job saved before the lineage tree has a best but no nodes or chains.
    # mutation: seeding chain 0 at ROOT makes the SEARCH text `let k = 5;` miss against the
    # baseline's `let k = 1;`
    # mutation: reading the best's files only from disk raises FileNotFoundError (there is no
    # iterations/0001/ here)
    from talos.scoring import bundle_delta
    from talos.state import Candidate
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("next"), edit(6, frm=5)], budget=b)
    st = loop.state
    rows = [NonceResult("t", n, True, 104, 1) for n in range(4)]
    st.iteration = 1
    st.best = Candidate(iteration=1, files={"mod.rs": "fn solve() { let k = 5; }\n"},
                        artifact_id="old", training=rows,
                        delta=bundle_delta(st.baseline.training, rows).to_dict(),
                        hypothesis={"title": "old", "description": "d",
                                    "strategy_tag": "local_search"})
    st.hypotheses = [{"iteration": 1, "against": 0, "title": "old", "outcome": "improved"}]
    assert st.nodes == [] and st.chains == []
    assert not (tmp_path / "iterations" / "0001").exists()
    st = loop.run()
    assert st.hypotheses[-1]["outcome"] != "failed:edit"
    assert st.nodes[0].parent == 1 and st.nodes[0].id == 2
    assert st.chains[0].best == 2 and st.best.iteration == 2


def test_parent_files_come_from_disk_after_a_restart(tmp_path):
    # Chain 0's best (node 1) is not the job best (node 2, on chain 1), so after a restart its
    # files can only come from iterations/0001/.
    # mutation: keeping parent files only in memory loses them across the restart
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    script = [hyp("a"), edit(5),            # 1, chain 0: k=5
              hyp("b"), edit(7),            # 2, chain 1: k=7, the job best
              hyp("c"), edit(0, frm=5)]     # 3, chain 0: k=0 is worse, failed:score
    loop, fp, fb, store = make(tmp_path, script, scores=heldout_flat, budget=b,
                               spec_kw={"chains": 2})
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == ["improved", "improved", "failed:score"]
    assert st.best.iteration == 2 and st.chains[0].best == 1
    again = Loop(store.read_spec(), store.load(), store, FakeProvider([]),
                 FakeBench(quality_from_files), template_rs="x",
                 clock=lambda: 0.0, sleep=lambda s: None)
    on_disk = (tmp_path / "iterations" / "0001" / "mod.rs").read_text(encoding="utf-8")
    assert "let k = 5;" in on_disk
    assert again._current_files() == {"mod.rs": on_disk}


def test_native_chain_best_keeps_its_native_score(tmp_path):
    # spec §6: in a native job the accept rule compares native scores only, including against a
    # parent that has since been validated. Node 1 (k=9) scores +8% natively but +6% metered
    # (one metered nonce errors); iteration 3 on the same chain scores +7% natively.
    # mutation: setting chain.best_score from rank_score() (the metered +6%) accepts iteration 3
    # as chain 0's best
    from talos.challenges import CHALLENGES

    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [None] + q[1:] if "let k = 9;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    script = [hyp("a"), edit(9),            # 1, chain 0: +8% native, +6% metered
              hyp("b"), edit(11),           # 2, chain 1: +10% both ways, the job best
              hyp("c"), edit(8, frm=9)]     # 3, chain 0: +7% native
    loop, fp, _, store = native(tmp_path, script, bench=fb, budget=b, spec_kw={"chains": 2})
    # No candidate beats the baseline outright (margin 50%), so a job-wide improvement needs a
    # mean above the best; the error ceiling lets node 1's metered error through validation.
    loop.rule = replace(CHALLENGES["knapsack"].beat, margin=0.5, error_ceiling=0.5)
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == ["improved", "improved", "failed:score"]
    first = st.nodes[0]
    assert first.score == pytest.approx(0.08) and first.metered_score == pytest.approx(0.06)
    assert st.best.iteration == 2
    assert st.chains[0].best == 1 and st.chains[0].best_score == pytest.approx(0.08)
    assert st.nodes[2].accepted is False


def test_a_candidate_failing_native_validation_is_never_accepted(tmp_path):
    # spec §6: a demoted candidate never becomes a chain state.
    # mutation: marking the demoted record accepted moves chain 0 onto node 1
    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [x + 1 for x in q] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, budget=b)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:validation"
    assert st.nodes[0].accepted is False and st.nodes[0].score == pytest.approx(0.04)
    assert st.chains[0].best == 0 and st.chains[0].current == 0
    assert st.chains[0].stall == 1


def test_one_chain_on_a_mainnet_start_moves_only_with_the_job_best(tmp_path):
    # Node 1 (k=9) scores +6% natively (one native nonce errors) and +8% metered, so the job
    # best is +8% while chain 0's best_score is +6%. Iteration 2 scores +7% natively: above the
    # chain's score, but not a job-wide improvement.
    # mutation: chain-only acceptance with one chain on a mainnet start accepts iteration 2
    from talos.challenges import CHALLENGES

    def native_scores(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [None] + q[1:] if "let k = 9;" in files["mod.rs"] else q
    fb = FakeBench(native_scores, metered_scores=quality_from_files)
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(9), hyp("b"), edit(8, frm=9)],
                                bench=fb, budget=b)
    loop.rule = replace(CHALLENGES["knapsack"].beat, margin=0.5, error_ceiling=0.5)
    st = loop.run()
    assert st.best.iteration == 1 and st.best.delta["mean_rel_delta"] == pytest.approx(0.08)
    assert st.chains[0].best_score == pytest.approx(0.06)
    assert st.hypotheses[1]["outcome"] == "failed:score"
    assert st.nodes[1].accepted is False and st.chains[0].best == 1


# ── simulated annealing ──────────────────────────────────────────────


def annealing(tmp_path, script, settings=None, stop_after=None, on_done=None, **kw):
    """An anneal job on heldout_flat (no win is ever confirmed, so the job keeps iterating). It
    stops itself after `stop_after` iterations, so the default 20-iteration budget sets the
    temperature schedule. `on_done(store)` runs on every iteration_done event."""
    from talos.lineage import AnnealSettings
    kw.setdefault("scores", heldout_flat)
    spec_kw = {"search": "anneal", "anneal": (settings or AnnealSettings()).to_dict(),
               **kw.pop("spec_kw", {})}
    loop, fp, fb, store = make(tmp_path, script, spec_kw=spec_kw, **kw)
    done = []

    def on_event(kind, data):
        if kind != "iteration_done":
            return
        done.append(data)
        if on_done is not None:
            on_done(store)
        if stop_after is not None and len(done) == stop_after:
            loop.request_stop()
    loop.on_event = on_event
    return loop, fp, fb, store


def events(store, kind):
    lines = (store.run_dir / "timeline.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["kind"] == kind]


# Iteration 1 moves k=1 to k=6 (+5%, the job best). Iteration 2 moves k=6 back to k=1: 0.0
# against the baseline, so not a win, and 0.05 below its parent.
WORSE_AFTER_BEST = [hyp("up"), edit(6), hyp("back"), edit(1, frm=6)]


def test_anneal_accepts_a_worse_candidate_when_the_draw_is_under_p(tmp_path, monkeypatch):
    # mutation: making an accepted-worse node the job best
    # mutation: computing T from the post-increment iteration count (2/20, T = 0.0455)
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 0.0)
    loop, fp, fb, store = annealing(tmp_path, WORSE_AFTER_BEST, stop_after=2)
    st = loop.run()
    assert st.status == "cancelled"
    node = st.nodes[1]
    assert node.accepted and node.outcome == "accepted" and node.parent == 1
    assert st.chains[0].current == 2
    # spend.iterations == 1 at the decision: T = 0.05 * (1 - 0.9 * 1/20)
    assert node.temperature == pytest.approx(0.04775)
    assert node.probability == pytest.approx(math.exp(-0.05 / node.temperature))
    assert node.draw == 0.0
    assert st.best.iteration == 1
    accept = [e for e in events(store, "accept") if e["node"] == 2]
    assert len(accept) == 1 and accept[0]["accepted"] is True and accept[0]["parent"] == 1
    assert accept[0]["probability"] == pytest.approx(node.probability)


def test_anneal_rejects_when_the_draw_is_over_p(tmp_path, monkeypatch):
    # mutation: accepting every worse candidate
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 0.999999)
    loop, fp, fb, store = annealing(tmp_path, WORSE_AFTER_BEST, stop_after=2)
    st = loop.run()
    assert not st.nodes[1].accepted and st.nodes[1].outcome == "failed:score"
    assert st.nodes[1].probability is not None
    assert st.chains[0].current == 1


def test_accepted_worse_candidate_is_never_confirmed(tmp_path, monkeypatch):
    # mutation: confirming any accepted node. Only iteration 1, which beat the baseline, was.
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 0.0)
    loop, fp, fb, store = annealing(tmp_path, WORSE_AFTER_BEST, stop_after=2)
    st = loop.run()
    assert st.nodes[1].accepted
    assert fb.holdout_runs == 1 and st.false_positives == [1]


def test_anneal_builds_on_the_current_state_not_the_best(tmp_path, monkeypatch):
    # mutation: parent = chain.best (k=6) makes iteration 3's SEARCH text `let k = 1;` miss
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 0.0)
    script = WORSE_AFTER_BEST + [hyp("lower"), edit(0, frm=1)]  # 3: k=0, -1%, not a win
    loop, fp, fb, store = annealing(tmp_path, script, stop_after=3)
    st = loop.run()
    assert st.nodes[2].parent == 2
    assert st.hypotheses[2]["outcome"] != "failed:edit"
    assert st.nodes[2].score == pytest.approx(-0.01)


# Iteration 1 is the job best; iterations 2 and 3 are accepted-worse (draw 0.0), each lower
# than the last, so with stall=2 chain 0 reheats after iteration 3.
STALLING = WORSE_AFTER_BEST + [hyp("lower"), edit(0, frm=1)]


def test_stall_reheats_back_to_the_chain_best(tmp_path, monkeypatch):
    # mutation: `>` instead of `>=` against stall leaves chain 0 on node 3
    # mutation: reheating without resetting the stall count leaves it at 2
    from talos.lineage import AnnealSettings
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 0.0)
    loop, fp, fb, store = annealing(tmp_path, STALLING, settings=AnnealSettings(stall=2),
                                    stop_after=3)
    st = loop.run()
    assert [n.outcome for n in st.nodes] == ["improved", "accepted", "accepted"]
    reheats = events(store, "reheat")
    assert len(reheats) == 1 and reheats[0]["iteration"] == 3
    assert reheats[0]["node"] == 1 and reheats[0]["temperature"] == pytest.approx(0.05)
    chain = st.chains[0]
    assert chain.current == 1 and chain.stall == 0
    assert chain.reheated_at == pytest.approx(3 / 20)  # iterations budget 20


def test_anneal_never_forces_a_strategy_tag(tmp_path):
    # mutation: _forced_tag() still rotating tags under anneal puts "MUST be" in a prompt and
    # emits reset events
    script = []
    for i in range(6):
        script += [hyp(f"h{i}"), "no edit blocks here"]
    loop, fp, fb, store = annealing(tmp_path, script, stop_after=6,
                                    thresholds=Thresholds(reset=5, distill=99))
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == ["failed:edit"] * 6
    assert st.runs_since_improvement == 6  # past the reset threshold, so a greedy job would force
    assert events(store, "reset") == []
    assert not any("MUST be" in system + user for system, user in fp.calls)


def test_native_demoted_candidate_is_not_a_chain_state(tmp_path):
    # mutation: running the anneal accept rule on a demoted candidate; decide(0.0, +0.04)
    # would accept it
    def metered(ch, files, ns):
        return ["out_of_fuel"] + quality_from_files(ch, files, ns)[1:]
    fb = FakeBench(heldout_flat, metered_scores=metered)
    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, _, store = annealing(tmp_path, [hyp("a"), edit(5)], bench=fb, budget=b,
                                   scoring="native")
    st = loop.run()
    assert st.validations[0]["reason"] == "fuel_proxy_miss"
    node = st.nodes[0]
    assert node.accepted is False and node.outcome == "failed:validation"
    assert st.chains[0].current == 0 and st.chains[0].best == 0


# Five iterations decided by the real draw (job "j", chain 0), with stall=2:
#   1: k=6, +5%, the job best
#   2: k=6 -> 1, drop 0.05, T=0.041, p=0.295, u=0.317: rejected
#   3: k=6 -> 0, drop 0.06, T=0.032, p=0.153, u=0.122: accepted; stall 2, reheat to node 1
#   4: k=6 -> 1, drop 0.05, T=0.05 (just reheated), p=0.368, u=0.303: accepted
#   5: k=1 -> 0, drop 0.01, T=0.0275, p=0.695, u=0.818: rejected; stall 2, reheat
# (ESTIMATE, arithmetic; the test compares two runs rather than these numbers.)
RESUME_SCRIPT = [hyp("1"), edit(6), hyp("2"), edit(1, frm=6), hyp("3"), edit(0, frm=6),
                 hyp("4"), edit(1, frm=6), hyp("5"), edit(0, frm=1)]


def test_resumed_anneal_makes_the_same_decisions(tmp_path):
    # Review Focus 1: a job stopped after iteration 3 and resumed decides as one that never
    # stopped.
    # mutation: an unseeded draw
    # mutation: a draw keyed on anything but (job_id, chain, iteration), for example
    # state.iteration read after the increment
    from talos.lineage import AnnealSettings, draw
    settings = AnnealSettings(stall=2)
    b = Budget(usd=None, hours=None, iterations=5, compute_usd=None)
    th = Thresholds(distill=99)  # two provider calls per iteration, no distill call
    loop, *_ = annealing(tmp_path / "straight", RESUME_SCRIPT, settings=settings, budget=b,
                         thresholds=th)
    st = loop.run()
    assert st.status == "exhausted"
    straight = [(n.accepted, n.draw, n.parent) for n in st.nodes]
    decided = [n for n in st.nodes if n.probability is not None]
    assert decided, "no node was decided by a draw; the comparison would test nothing"
    assert [n.draw for n in decided] == [draw("j", 0, n.id) for n in decided]
    assert {n.accepted for n in decided} == {True, False}
    assert [n.outcome for n in st.nodes] == [
        "improved", "failed:score", "accepted", "accepted", "failed:score"]

    loop, _, _, store = annealing(tmp_path / "resumed", RESUME_SCRIPT, settings=settings,
                                  budget=b, thresholds=th, stop_after=3)
    assert loop.run().status == "cancelled"
    again = Loop(store.read_spec(), store.load(), store, FakeProvider(RESUME_SCRIPT[6:]),
                 FakeBench(heldout_flat), template_rs="x", clock=lambda: 0.0,
                 sleep=lambda s: None, thresholds=th)
    again.state.status = "researching"  # as execute_job does on resume
    resumed = again.run()
    assert resumed.status == "exhausted"
    assert [(n.accepted, n.draw, n.parent) for n in resumed.nodes] == straight
    assert resumed.chains == st.chains


def test_tree_and_reheat_are_saved_before_iteration_done(tmp_path, monkeypatch):
    # mutation: appending the node or applying the reheat after _finish_iteration's save, which
    # a hard kill at that point would lose
    from talos.lineage import AnnealSettings
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 0.0)
    seen = []

    def on_done(store):
        saved = store.load()
        seen.append((len(saved.nodes), saved.chains[0].current, saved.chains[0].stall))
    loop, *_ = annealing(tmp_path, STALLING, settings=AnnealSettings(stall=2), stop_after=3,
                         on_done=on_done)
    loop.run()
    assert seen == [(1, 1, 0), (2, 2, 1), (3, 1, 0)]


def test_one_anneal_chain_on_a_mainnet_start_keeps_the_job_best_as_its_best(tmp_path):
    # As test_one_chain_on_a_mainnet_start_moves_only_with_the_job_best, under anneal: node 1
    # is +6% natively (the job best, +8% metered); iteration 2 is +7% natively, no worse than
    # its parent, so annealing accepts it, but it is not a job-wide improvement.
    # mutation: dropping the _chain_scores_apart() gate makes node 2 chain 0's best, so a
    # reheat would return to code that is not the job's best
    from talos.challenges import CHALLENGES

    def native_scores(ch, files, ns):
        q = heldout_flat(ch, files, ns)
        return [None] + q[1:] if "let k = 9;" in files["mod.rs"] else q
    fb = FakeBench(native_scores, metered_scores=heldout_flat)
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    loop, fp, _, store = annealing(tmp_path, [hyp("a"), edit(9), hyp("b"), edit(8, frm=9)],
                                   bench=fb, budget=b, scoring="native")
    loop.rule = replace(CHALLENGES["knapsack"].beat, margin=0.5, error_ceiling=0.5)
    st = loop.run()
    assert st.best.iteration == 1
    assert [n.outcome for n in st.nodes] == ["improved", "accepted"]
    assert st.nodes[1].probability is None  # accepted outright, no draw needed
    assert st.chains[0].current == 2
    assert st.chains[0].best == 1 and st.chains[0].best_score == pytest.approx(0.06)


# ── template start ───────────────────────────────────────────────────

TEMPLATE_RS = "pub fn solve_challenge() { todo!() }\n"
BASELINE_ONLY = "BASELINE-ONLY-7f3a"


def hypf(title, family=None):
    """A hypothesis reply, with a design_family when one is given."""
    d = {"title": title, "description": "d", "strategy_tag": "local_search"}
    if family is not None:
        d["design_family"] = family
    return json.dumps(d)


def whole(k):
    """A whole-file reply writing mod.rs with `let k = {k};`."""
    return f"<<<<<<< FILE mod.rs\nfn solve() {{ let k = {k}; }}\n>>>>>>> END FILE\n"


def make_template(tmp_path, script, iterations=4, spec_kw=None, **kw):
    """A from-scratch anneal job. The root is the empty template as mod.rs. The baseline scores
    200 on every nonce, so k=11 (quality 110) is -45%, and its code carries a marker that must
    never be shown."""
    from talos.lineage import AnnealSettings
    sk = {"start": "template", "search": "anneal", "anneal": AnnealSettings().to_dict(),
          **(spec_kw or {})}
    kw.setdefault("budget", Budget(usd=None, hours=None, iterations=iterations,
                                   compute_usd=None))
    loop, fp, fb, store = make(tmp_path, script, spec_kw=sk, template_rs=TEMPLATE_RS,
                               challenge_docs={"README.md": "doc"}, **kw)
    loop.state.baseline = replace(
        baseline(q=200), files={"mod.rs": f"fn solve() {{ let k = 1; }} // {BASELINE_ONLY}\n"})
    return loop, fp, fb, store


def edit_systems(fp):
    return [system for system, _ in fp.calls if system.startswith("You are editing")]


def test_whole_file_reply_naming_a_new_file_fails_the_iteration(tmp_path):
    # AGENTS.md invariant 4 for whole-file replies: one out-of-scope FILE block fails the whole
    # iteration, and the in-scope mod.rs block in the same reply is never applied or scored.
    # mutation: dropping the `if outcome.rejected` check after apply_whole_file_response scores
    # the mod.rs block and makes it the best
    reply = whole(11) + "<<<<<<< FILE helpers.rs\nfn h() {}\n>>>>>>> END FILE\n"
    loop, fp, fb, store = make_template(tmp_path, [hypf("lp", "lp"), reply], iterations=1)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:edit"
    assert "helpers.rs" in st.hypotheses[0]["error"]
    assert st.best is None and fb.calls == []


def test_template_start_first_candidate_is_written_whole_from_the_template(tmp_path):
    # mutation: _root_files() returning the baseline's files edits the baseline, not the template
    # mutation: whole_file left False at the root asks for diffs against an empty template
    # mutation: _context() without start or challenge_docs drops the challenge definition
    loop, fp, fb, store = make_template(tmp_path, [hypf("lp", "lp"), whole(11)], iterations=1)
    st = loop.run()
    system, user = fp.calls[1]  # the edit call
    assert "pub fn solve_challenge() { todo!() }" in user
    assert WHOLE_FILE_FORMAT in system
    assert "--- tig-challenges/src/knapsack/README.md ---\ndoc" in user
    assert st.best.iteration == 1 and "let k = 11;" in st.best.files["mod.rs"]
    assert set(st.best.files) == {"mod.rs"}


def test_template_start_never_shows_baseline_code(tmp_path, monkeypatch):
    # mutation: _root_files() returning state.baseline.files
    monkeypatch.setattr("talos.loop.draw", lambda job_id, chain, iteration: 1.0)  # never accept

    def swap(frm, to):
        return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {to};\n>>>>>>> REPLACE\n"

    script = [hyp("a"), whole("BUG"), swap("BUG", "BUG2"),  # 1: no build after its fix round
              hyp("b"), whole(11),                          # 2: from the root, the first best
              hyp("c"), edit(5, frm=11),                    # 3: worse, rejected
              hyp("d"), edit(6, frm=11),                    # 4: worse, rejected; then distill
              "LESSON: start from a feasible schedule.",
              hyp("e"), edit(7, frm=11)]                    # 5: lets the distill call run
    # five iterations: with a budget of four, the distill call after iteration 4 is refused
    loop, fp, fb, store = make_template(tmp_path, script, iterations=5,
                                        thresholds=Thresholds(distill=2, compile_fix_rounds=1))
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    st = loop.run()
    assert all(BASELINE_ONLY not in system + user for system, user in fp.calls)
    assert BASELINE_ONLY not in (store.run_dir / "timeline.jsonl").read_text()
    written = [p for p in (store.run_dir / "iterations").rglob("*") if p.is_file()]
    assert written and all(BASELINE_ONLY not in p.read_text() for p in written)
    # the run went through every prompt kind: hypothesis, edit, compile fix and distill
    assert [h["outcome"] for h in st.hypotheses] == [
        "failed:compile", "improved", "failed:score", "failed:score", "failed:score"]
    assert len(fp.calls) == 12
    assert "start from a feasible schedule" in st.tacit


def test_template_candidates_run_without_hyperparameters(tmp_path):
    hp = {"t": {"x": 1}}
    loop, fp, fb, store = make_template(
        tmp_path, [hyp("a"), whole(11), hyp("b"), edit(12, frm=11)], iterations=2,
        spec_kw={"hyperparameters": hp})
    loop.run()
    assert len(fb.calls) == 2
    # mutation: _request passing spec.hyperparameters
    assert all(c.hyperparameters is None for c in fb.calls)
    # mutation: _context() passing spec.hyperparameters shows the mainnet keys to the model
    assert all("Hyperparameters:" not in user for _system, user in fp.calls)


def test_template_baseline_keeps_the_spec_hyperparameters(tmp_path):
    # mutation: measure_baseline passing _candidate_hyperparameters() scores the baseline
    # without the map it is published with
    hp = {"t": {"x": 1}}
    loop, fp, fb, store = make_template(tmp_path, [hyp("a"), whole(11)], iterations=1,
                                        spec_kw={"hyperparameters": hp})
    loop.spec = replace(loop.spec, baseline_algorithm={"name": "pinned", "id": "p1",
                                                       "adoption": 3})
    loop.state.baseline = None
    mainnet = types.SimpleNamespace(
        fetch_algorithm_files=lambda ch, name: BASE_FILES,
        fetch_template=lambda ch: TEMPLATE_RS)
    loop.measure_baseline(tmp_path / "cache", "cpu4-mem8192", mainnet=mainnet)
    assert fb.calls[0].hyperparameters == hp
    loop.run()
    assert len(fb.calls) == 2 and fb.calls[1].hyperparameters is None


def test_template_best_below_mainnet_is_kept_and_shown(tmp_path):
    loop, fp, fb, store = make_template(
        tmp_path, [hyp("a"), whole(11), hyp("b"), edit(5, frm=11)], iterations=2)
    st = loop.run()
    # mutation: _best_delta() returning 0.0 drops every from-scratch best
    assert st.best.iteration == 1
    assert st.best.delta["mean_rel_delta"] == pytest.approx(-0.45)
    _system, user = fp.calls[2]  # iteration 2's hypothesis call
    assert "Best so far (node #1): -45.000%" in user
    # a best below mainnet is never confirmed
    assert fb.holdout_runs == 0 and st.confirmed == [] and st.status == "exhausted"


def test_template_new_family_gets_whole_file_and_same_family_gets_diffs(tmp_path):
    # mutation: whole-file only at the root makes iteration 3's whole-file reply fail to apply
    # mutation: whole-file for every template edit makes iteration 2's diff reply fail
    script = [hypf("a", "lp"), whole(11),          # 1: root
              hypf("b", "lp"), edit(12, frm=11),   # 2: same family as node 1
              hyp("c"), whole(13),                 # 3: unlabelled, new on this chain
              hyp("d"), edit(14, frm=13)]          # 4: unlabelled again
    loop, fp, fb, store = make_template(tmp_path, script, iterations=4)
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == ["improved"] * 4
    systems = edit_systems(fp)
    assert [WHOLE_FILE_FORMAT in s for s in systems] == [True, False, True, False]
    assert [SEARCH_REPLACE_FORMAT in s for s in systems] == [False, True, False, True]
    assert [n.family for n in st.nodes] == ["lp", "lp", "unlabelled", "unlabelled"]


def test_template_greedy_second_chain_keeps_its_own_worse_best(tmp_path):
    # mutation: dropping the greedy chain-only acceptance leaves chain 1 at the root, so
    # iteration 4 asks for a whole file, its diff reply parses as no FILE block, and it fails
    # failed:edit with parent 0
    # mutation: collecting design families from every chain asks iteration 3 for a diff, and
    # its whole-file reply fails failed:edit
    script = [hypf("a", "greedy"), whole(21),        # 1, chain 0: root, -40%, the job best
              hypf("b", "lp"), whole(11),            # 2, chain 1: root, -45%, chain 1's best
              hypf("c", "lp"), whole(20),            # 3, chain 0: new family here, worse
              hypf("d", "lp"), edit(10, frm=11)]     # 4, chain 1: from node 2, worse
    loop, fp, fb, store = make_template(tmp_path, script, iterations=4,
                                        spec_kw={"search": "greedy", "chains": 2})
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == [
        "improved", "accepted", "failed:score", "failed:score"]
    assert st.nodes[1].outcome == "accepted"
    assert st.chains[1].best == 2 and st.chains[1].best_score == pytest.approx(-0.45)
    assert st.nodes[3].parent == 2 and st.nodes[3].outcome != "failed:edit"
    assert st.best.iteration == 1


def test_template_native_first_candidate_validates_below_mainnet(tmp_path):
    # mutation: validation_failure given 0.0 instead of _best_delta() demotes it not_improved
    loop, fp, fb, store = make_template(tmp_path, [hyp("a"), whole(11)], iterations=1,
                                        scoring="native")
    st = loop.run()
    assert [v["outcome"] for v in st.validations] == ["validated"]
    assert st.best.iteration == 1
    assert st.best.delta["mean_rel_delta"] == pytest.approx(-0.45)


def never_compiling_script(system, user):
    """A reply for every prompt kind, so compile-fix rounds and distillation cannot put a fixed
    list out of step."""
    if '"strategy_tag" (one of' in system:
        return hyp("try")
    if WHOLE_FILE_FORMAT in system:
        return whole(11)
    m = re.search(r"let k = (\d+);", user)
    if system.startswith("You are fixing") and m:
        k = int(m.group(1))
        return edit(k + 1, frm=k)
    return "LESSON: make it build first."


def test_template_that_never_compiles(tmp_path):
    # mutation: a reheat that reads chain.best_score or the root's score crashes on None
    from talos.lineage import AnnealSettings
    from talos.package import build_package
    loop, fp, fb, store = make_template(
        tmp_path, never_compiling_script, iterations=5,
        spec_kw={"anneal": AnnealSettings(stall=2).to_dict()})
    fb._compile_ok = lambda files: False
    st = loop.run()
    assert st.status == "exhausted" and st.best is None
    assert [h["outcome"] for h in st.hypotheses] == ["failed:compile"] * 5
    assert len(st.nodes) == 5 and all(n.accepted is False for n in st.nodes)
    assert events(store, "reheat")
    assert st.chains[0].current == 0
    readme = (build_package(loop.spec, st, store) / "README.md").read_text()
    assert "No candidate" in readme


def test_anneal_recall_names_the_parent_node_not_an_earlier_best(tmp_path):
    # mutation: Loop._context not passing search/chains to PromptContext keeps today's
    # "Tried against an earlier best" title in an anneal job, where a parent need not be a best
    from talos.lineage import AnnealSettings
    script = [hyp("Old idea"), "no edit blocks here",  # 1: failed:edit against the root
              hyp("Winner"), edit(2),                   # 2: improved; the chain moves to node 2
              hyp("Next"), edit(3, frm=2)]
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, script, budget=b,
                               spec_kw={"search": "anneal",
                                        "anneal": AnnealSettings().to_dict()},
                               scores=lambda ch, files, ns: [
                                   q + 900 for q in quality_from_files(ch, files, ns)],
                               thresholds=Thresholds(recall=0, distill=99, reset=99))
    loop.state.baseline = baseline(q=1000)  # so k=2 is +0.1%: a new best, not a win
    loop.rule = BeatRule(margin=0.005)
    loop.run()
    proposals = [u for s, u in fp.calls if s.startswith("You are a research engineer")]
    assert len(proposals) == 3
    assert "Tried against other code and did NOT help there" in proposals[2]
    # a failed:edit record carries no title: the edit failed before the hypothesis was stored
    assert "[failed:edit, against the baseline]" in proposals[2]
    assert "earlier best" not in proposals[2]


@pytest.mark.parametrize("failing", ["metered", "native"])
def test_a_template_start_calibration_failure_keeps_baseline_output_out_of_the_timeline(
        tmp_path, failing):
    # mutation: writing res.compile.output into either calibration_fallback event of a template
    # start; the baseline's compiler output quotes its source
    fb = (FakeBench(quality_from_files, metered_compile_ok=lambda files: False)
          if failing == "metered" else
          FakeBench(quality_from_files, compile_ok=lambda files: False,
                    metered_compile_ok=lambda files: True))
    loop, fp, _, store = native(tmp_path, [], bench=fb, spec_kw={"start": "template"})
    loop.state.scoring, loop.state.fuel_budgets_us = None, None
    loop.calibrate(tmp_path / "cal", "hw")
    fallback = events(store, "calibration_fallback")
    assert any(f"{failing} build" in e["reason"] for e in fallback)
    assert all("output" not in e for e in fallback)
    assert "E0308" not in (store.run_dir / "timeline.jsonl").read_text()


@pytest.mark.parametrize("spec_kw, lines", [({"chains": 2}, True), ({}, False)])
def test_greedy_emits_one_accept_line_per_iteration_when_chains_score_apart(
        tmp_path, spec_kw, lines):
    # mutation: emitting the accept event for anneal only (spec §8 wants a line per iteration
    # with chain, parent, family, score and outcome); emitting it for the default job too
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    script = [hyp("a"), edit(5),                  # 1, chain 0: improved
              hyp("b"), "no edit blocks here",    # 2: failed:edit
              hyp("c"), edit(0, frm=5)]           # 3, chain 0: worse, failed:score
    loop, fp, fb, store = make(tmp_path, script, scores=heldout_flat, budget=b, spec_kw=spec_kw)
    loop.run()
    accept = events(store, "accept")
    if not lines:
        assert accept == []
        return
    assert [(e["node"], e["chain"], e["parent"], e["accepted"], e["outcome"]) for e in accept] == [
        (1, 0, 0, True, "improved"), (2, 1, 0, False, "failed:edit"),
        (3, 0, 1, False, "failed:score")]
    assert accept[2]["score"] is not None and accept[1]["score"] is None


def test_loop_checks_and_anneals_against_the_extended_budget(tmp_path):
    # mutation: _spent reading spec.budget gives 1.0; _check_budget reading it raises
    b = Budget(usd=None, hours=None, iterations=4, compute_usd=None)
    loop, *_ = make(tmp_path, [], budget=b)
    loop.state.spend.iterations = 4
    loop.state.budget_extensions = [{"add": {"iterations": 4, "hours": None, "usd": None,
                                             "compute_usd": None}}]
    assert loop._spent() == pytest.approx(0.5)
    loop._check_budget()  # does not raise


def test_an_extended_exhausted_loop_runs_the_added_iterations(tmp_path):
    # mutation: any reader left on spec.budget stops the second run at once with 2 iterations
    # failed edits: each iteration is counted, nothing ever wins, two provider calls each
    script = [r for t in "abcd" for r in (hyp(t), "no edit blocks here")]
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    th = Thresholds(distill=99)  # two provider calls per iteration, no distill call
    loop, _, _, store = make(tmp_path, script[:4], scores=heldout_flat, budget=b,
                             thresholds=th)
    assert loop.run().status == "exhausted"
    st = store.load()
    extend(b, st, {"iterations": 2}, now=0.0, previous_status="exhausted")
    st.status = "researching"  # as execute_job does
    again = Loop(store.read_spec(), st, store, FakeProvider(script[4:]),
                 FakeBench(heldout_flat), template_rs="x", clock=lambda: 0.0,
                 sleep=lambda s: None, thresholds=th)
    final = again.run()
    assert final.status == "exhausted" and final.stop_reason == "iterations"
    assert final.spend.iterations == 4


def test_budget_usd_stops_an_agentic_run_after_the_session_that_crossed_it(tmp_path):
    # issue 34: agentic sessions were never charged, so --budget-usd could not stop the run.
    # Each fake session costs $0.60 against a $1.00 cap: two run (0.60 < 1.00 before the
    # second), the third is refused before it starts.
    # mutation: not charging the session runs all 5 iterations; checking the budget only
    # after the session lets a third one start
    import subprocess
    from talos.agentic import attach_agentic
    sessions = []

    def run(cmd, **kw):
        sessions.append(cmd)
        wt = kw["cwd"]
        (wt / ".talos" / "hypothesis.json").write_text(json.dumps(
            {"title": f"t{len(sessions)}", "description": "d", "strategy_tag": "local_search"}))
        mod = wt / "algorithm" / "mod.rs"
        mod.write_text(mod.read_text() + f"// session {len(sessions)}\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(
            {"type": "result", "result": "done", "total_cost_usd": 0.6}), stderr="")

    b = Budget(usd=1.0, hours=None, iterations=5, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, ["lesson"], budget=b)
    attach_agentic(loop, "claude-cli", "m", timeout_s=1, run=run)
    try:
        st = loop.run()
    finally:
        if getattr(loop, "_agentic_wt", None) is not None:
            import shutil
            shutil.rmtree(loop._agentic_wt, ignore_errors=True)
    assert len(sessions) == 2
    assert st.status == "exhausted" and st.stop_reason == "usd"
    assert st.spend.llm_usd >= 1.2
    # persisted, not only in memory: a resume reads spend back from state.json
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["llm_usd"] >= 1.2


def test_charge_llm_persists_before_the_iteration_ends(tmp_path):
    # an agentic session can run 30 minutes and the process can be killed right after it; a
    # charge held only in memory is lost and a resume under-reports llm spend.
    # mutation: dropping the save in _charge_llm (the iteration-end save hides it elsewhere)
    loop, fp, fb, store = make(tmp_path, ["unused"])
    loop._charge_llm(0.5)
    loop._charge_llm(0.0)
    loop._charge_llm(None)
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["llm_usd"] == 0.5


def _timeline(tmp_path):
    return [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]


def test_failed_call_that_printed_a_cost_is_charged(tmp_path):
    # issue #74: a claude-cli call that exits non-zero still cost money. The run stops on the
    # error, and the cost is in state.json. mutation: charging only on success leaves 0.0
    loop, fp, fb, store = make(tmp_path, raising(ProviderError("claude CLI failed: x",
                                                               cost_usd=0.75)))
    st = loop.run()
    assert st.status == "failed" and st.stop_reason == "provider: claude CLI failed: x"
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["llm_usd"] == 0.75


def test_timed_out_call_records_an_unknown_cost_and_the_run_carries_on(tmp_path):
    # issue #74. mutation: not recording the event on a raised ProviderTimeout leaves no trace
    loop, fp, fb, store = make(tmp_path, raising(
        hyp("a"), ProviderTimeout("claude CLI timed out after 7s",
                                  cost_unknown="claude CLI timed out after 7s"),
        hyp("b"), edit(5)))
    st = loop.run()
    assert st.status == "won"
    unknown = [e for e in _timeline(tmp_path) if e["kind"] == "llm_cost_unknown"]
    assert [e["reason"] for e in unknown] == ["claude CLI timed out after 7s"]


def test_completion_without_a_cost_records_an_unknown_cost(tmp_path):
    # issue #74: exit 0 with no total_cost_usd. mutation: skipping a None cost silently
    from talos.types import Completion, Usage

    def script(system, user):
        return Completion(text="not json", usage=Usage(), cost_unknown="no total_cost_usd")

    class Once(FakeProvider):
        def complete(self, system, user):
            self.calls.append((system, user))
            return script(system, user)

    b = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, ["unused"], budget=b)
    loop.provider = Once([])
    loop.run()
    unknown = [e for e in _timeline(tmp_path) if e["kind"] == "llm_cost_unknown"]
    assert unknown and all(e["reason"] == "no total_cost_usd" for e in unknown)
    assert len(unknown) == len(loop.provider.calls)


def test_provider_error_without_cost_fields_records_nothing(tmp_path):
    # an HTTP provider's error carries neither field. mutation: recording an unknown cost for
    # every error (cost_unknown or "...") puts a spurious event on the timeline
    loop, fp, fb, store = make(tmp_path, raising(ProviderError("boom")))
    loop.run()
    assert not [e for e in _timeline(tmp_path) if e["kind"] == "llm_cost_unknown"]
    assert json.loads((tmp_path / "state.json").read_text())["spend"]["llm_usd"] == 0.0


# ── issue #84: stopping a candidate's scoring at the error ceiling ─────────

def test_only_research_candidates_ask_the_job_to_stop_at_the_error_ceiling(tmp_path):
    one = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path / "m", [hyp("bump k"), edit(5)], budget=one)
    loop.run()
    # mutation: never setting it leaves every rejected candidate running to the end
    assert [c.stop_at_error_ceiling for c in fb.calls] == [True]
    loop, fp, fb, store = native(tmp_path / "n", [hyp("bump k"), edit(5)], budget=one)
    loop.run()
    # mutation: setting it on validation too; a stopped validation could hide the
    # fuel_proxy_miss that lowers the calibration margin
    assert [(c.mode, c.stop_at_error_ceiling) for c in fb.calls] == [("native", True),
                                                                     ("metered", False)]
    loop, fp, fb, store = native(tmp_path / "c", [], fuel=10_000_000)
    loop.state.scoring = None
    loop.calibrate(tmp_path / "cal", "hw")
    # the calibration rows become the native budgets nonce by nonce
    assert [c.stop_at_error_ceiling for c in fb.calls] == [False, False]


class StoppedBench(FakeBench):
    """Answers a candidate as the C3 client does for a job stopped at the ceiling: nonce 0
    panicked after 6 ms, nonce 1 solved in 2 ms, nonces 2 and 3 never ran."""

    def evaluate(self, request):
        from talos.bench import EvalResult
        from talos.types import CompileResult
        self.calls.append(request)
        rows = [NonceResult("t", 0, False, None, 6, "panic"), NonceResult("t", 1, True, 100, 2),
                NonceResult("t", 2, False, None, 0, "not_run"),
                NonceResult("t", 3, False, None, 0, "not_run")]
        return EvalResult(CompileResult(ok=True, artifact_id="a", output="ok"), rows, None,
                          "not_won")


def test_a_candidate_stopped_at_the_ceiling_says_so_in_its_record(tmp_path):
    one = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("bump k"), edit(5)], budget=one,
                               bench=StoppedBench(quality_from_files))
    st = loop.run()
    h = st.hypotheses[0]
    assert h["outcome"] == "failed:runtime" and st.best is None
    # mutation: no note, and the recall line shows the not-run filler as if it were measured
    assert h["error"] == ("stopped early at the error ceiling: 1 of the 2 nonces run "
                          "failed; 2 of 4 not run")
    # mean candidate runtime over the nonces that ran, (6 + 2) / 2 = 4 ms, against the
    # baseline's 1 ms on those same nonces. mutation: counting the 0 ms filler gives 2.0
    assert h["runtime_ratio"] == 4.0
    scored = [json.loads(x) for x in (store.run_dir / "timeline.jsonl").read_text().splitlines()
              if json.loads(x)["kind"] == "scored"]
    assert scored[0]["not_run"] == 2


def test_a_candidate_that_ran_every_nonce_has_no_stop_note(tmp_path):
    one = Budget(usd=None, hours=None, iterations=1, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("bump k"), edit(5)], budget=one)
    st = loop.run()
    # mutation: writing the note (or a not_run count of 0) on every scored candidate
    assert "error" not in st.hypotheses[0]
    scored = [json.loads(x) for x in (store.run_dir / "timeline.jsonl").read_text().splitlines()
              if json.loads(x)["kind"] == "scored"]
    assert "not_run" not in scored[0]


# Issue #32: the best is the code every later iteration edits, so a candidate that buys a higher
# mean by regressing a track must not become it. Qualities per k on tracks t and u.
def per_track(table, holdout=None):
    def scores(challenge, files, ns):
        k = int(re.search(r"let k = (\d+);", files["mod.rs"]).group(1))
        if holdout is not None and ns.start >= 1_000_000:
            return [holdout for _ in ns.nonces()]
        return [table[k][0 if ns.track == "t" else 1] for _ in ns.nonces()]
    return scores


def test_a_higher_mean_that_regresses_a_track_does_not_become_the_best(tmp_path):
    table = {1: (100, 100), 2: (102, 102), 5: (120, 95), 3: (103, 103)}
    b = Budget(usd=None, hours=None, iterations=3, compute_usd=None)
    script = [hyp("a"), edit(2),            # 1: +2% on both, the best (held-out flat: no win)
              hyp("b"), edit(5, frm=2),     # 2: mean +7.5%, but u -5% under the best's +2%
              hyp("c"), edit(3, frm=2)]     # 3: built on iteration 1, +3% on both
    loop, fp, fb, store = make(tmp_path, script, per_track(table, holdout=100), b,
                               two_tracks=True, thresholds=Thresholds(recall=1, distill=99))
    st = loop.run()
    # mutation: the mean-only rule makes iteration 2 the best, and iteration 3's SEARCH text
    # `let k = 2;` then misses its `let k = 5;`, so iteration 3 is failed:edit
    assert [h["outcome"] for h in st.hypotheses] == ["improved", "failed:score", "improved"]
    assert st.best.iteration == 3
    assert "u" in st.hypotheses[1]["error"] and "-5.00%" in st.hypotheses[1]["error"]
    # the next prompt's recall block carries the reason
    assert any("-5.00%" in user for _system, user in fp.calls[4:])


def test_template_stepping_stones_keep_the_worst_track(tmp_path):
    # A template start has no best, so the first scored candidate is the best however far below
    # mainnet it is; after that the worst track may not fall. One chain on a template start
    # also runs the per-chain greedy branch, which must apply the same rule.
    table = {1: (10, 10), 2: (50, 60), 3: (70, 55), 4: (90, 40), 5: (75, 55)}
    b = Budget(usd=None, hours=None, iterations=4, compute_usd=None)
    script = [hyp("a"), whole(2),           # 1: -50%, -40%: the first best
              hyp("b"), edit(3, frm=2),     # 2: -30%, -45%: mean up, worst -45% >= -50%
              hyp("c"), edit(4, frm=3),     # 3: -10%, -60%: mean up, worst -60% < -45%
              hyp("d"), edit(5, frm=3)]     # 4: -25%, -45%, built on iteration 2
    loop, fp, fb, store = make(tmp_path, script, per_track(table), b, two_tracks=True,
                               spec_kw={"start": "template"})
    st = loop.run()
    # mutation: the job-wide rule alone lets the chain branch take iteration 3 as "accepted",
    # and the mean-only rule takes it as "improved"; either way iteration 4 is built on it
    # and its SEARCH text `let k = 3;` misses, so it is failed:edit
    assert [h["outcome"] for h in st.hypotheses] == [
        "improved", "improved", "failed:score", "improved"]
    assert [h["against"] for h in st.hypotheses] == [0, 1, 2, 2]
    assert st.best.iteration == 4 and st.chains[0].best == 4


def test_each_chain_keeps_its_own_worst_track(tmp_path):
    # Chain 0's bar is its own best (node 1: +2%, +2%), not the job best (node 2: +10%, +6%).
    from talos.challenges import CHALLENGES
    table = {1: (100, 100), 2: (102, 102), 3: (110, 106), 4: (105, 103), 9: (100, 100),
             6: (112, 101)}
    b = Budget(usd=None, hours=None, iterations=5, compute_usd=None)
    script = [hyp("a"), edit(2),            # 1, chain 0: the job best
              hyp("b"), edit(3),            # 2, chain 1: the job best
              hyp("c"), edit(4, frm=2),     # 3, chain 0: +4% mean, worst +3% >= +2%
              hyp("d"), edit(9, frm=3),     # 4, chain 1: no gain
              hyp("e"), edit(6, frm=4)]     # 5, chain 0: +6.5% mean, worst +1% < +3%
    loop, fp, fb, store = make(tmp_path, script, per_track(table), b, two_tracks=True,
                               spec_kw={"chains": 2})
    loop.rule = replace(CHALLENGES["knapsack"].beat, margin=0.5)  # nothing wins outright
    st = loop.run()
    # mutation: the job best's bar for chain 0 rejects iteration 3, and iteration 5 is then
    # failed:edit; the mean-only chain rule accepts iteration 5
    assert [h["outcome"] for h in st.hypotheses] == [
        "improved", "improved", "accepted", "failed:score", "failed:score"]
    assert [c.best for c in st.chains] == [3, 2]
    assert "+1.00%" in st.hypotheses[4]["error"]


def test_native_validation_demotes_a_metered_worst_track_regression(tmp_path):
    # k=5 is +10%, +4% natively; on the metered path three of u's nonces fail and u is +1%,
    # under the validated best's worst track of +2%.
    from talos.challenges import CHALLENGES
    table = {1: (100, 100), 2: (102, 102), 5: (110, 104)}

    def metered(ch, files, ns):
        q = per_track(table)(ch, files, ns)
        if "let k = 5;" in files["mod.rs"] and ns.track == "u":
            return [None, None, None] + q[3:]
        return q
    fb = FakeBench(per_track(table), metered_scores=metered)
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(2), hyp("b"), edit(5, frm=2)],
                                bench=fb, budget=b, two_tracks=True,
                                budgets={"t": 1_000_000, "u": 1_000_000})
    # no win outright; the ceiling lets u's three metered errors in 8 nonces through
    loop.rule = replace(CHALLENGES["knapsack"].beat, margin=0.5, error_ceiling=0.5)
    st = loop.run()
    # mutation: validation_failure with the mean-only rule validates iteration 2 (+5.5% mean)
    assert [v["reason"] for v in st.validations] == [None, "not_improved"]
    assert st.best.iteration == 1


def test_an_annealed_candidate_that_regresses_a_track_moves_current_but_not_best(tmp_path):
    # Annealing accepts iteration 2 (its mean is over its parent's), so research goes on from
    # it; a reheat still returns to the chain's best, which must be the one the rule allows.
    # A template start, because with one chain on a mainnet start the chain's best moves only
    # with the job's (Loop._chain_scores_apart).
    from talos.lineage import AnnealSettings
    table = {2: (50, 60), 5: (90, 40)}  # -50%, -40%; then mean -35% but worst -60%
    b = Budget(usd=None, hours=None, iterations=2, compute_usd=None)
    loop, fp, fb, store = make(tmp_path, [hyp("a"), whole(2), hyp("b"), edit(5, frm=2)],
                               per_track(table), b, two_tracks=True,
                               spec_kw={"start": "template", "search": "anneal",
                                        "anneal": AnnealSettings().to_dict()})
    st = loop.run()
    assert [h["outcome"] for h in st.hypotheses] == ["improved", "accepted"]
    # mutation: the mean-only chain rule in _add_node makes node 2 the chain's best
    assert (st.chains[0].current, st.chains[0].best) == (2, 1)


ONE = Budget(usd=None, hours=None, iterations=1, compute_usd=None)


def _scored_events(store):
    rows = [json.loads(x) for x in (store.run_dir / "timeline.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == "scored"]


def test_scored_event_and_record_carry_the_noise_figures(tmp_path):
    # baseline 100 on nonces 0..3; candidate [100, 102, 100, 102]: differences [0, 2, 0, 2],
    # delta 1%, SE sqrt(1/3)%, ratio sqrt(3). mutation: not passing the fields to _event
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=ONE,
                               scores=lambda ch, files, ns: [100 + 2 * (n % 2)
                                                             for n in ns.nonces()])
    st = loop.run()
    ev = _scored_events(store)[0]
    se = math.sqrt(1 / 3) / 100
    assert ev["mean_rel_se"] == pytest.approx(se, rel=1e-9)
    assert ev["noise_ratio"] == pytest.approx(math.sqrt(3), rel=1e-9)
    assert (ev["noise_band"], ev["noise_state"]) == ("marginal", "ok")
    assert ev["track_rel_se"] == {"t": pytest.approx(se, rel=1e-9)}
    rec = st.hypotheses[0]
    assert rec["mean_rel_se"] == pytest.approx(se, rel=1e-9)
    assert (rec["noise_band"], rec["noise_state"]) == ("marginal", "ok")


def test_over_ceiling_scored_event_withholds_the_ratio(tmp_path):
    # one of four nonces fails: 25% > the 5% ceiling. mutation: calling noise() without the
    # rule's ceiling, or computing the band in the loop
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=ONE,
                               scores=lambda ch, files, ns: [None] + [101] * (ns.count - 1))
    loop.run()
    ev = _scored_events(store)[0]
    assert ev["noise_state"] == "over_ceiling"
    assert ev["noise_ratio"] is None and ev["noise_band"] is None
    assert ev["mean_rel_se"] is not None


def test_a_validated_record_carries_the_metered_noise(tmp_path):
    # spec Tests 11. native: 104 on every nonce, so zero spread. metered: [error, 104, 104, 104];
    # the error scores at the worst observed (100), so the differences are [0, 4, 4, 4]: delta 3%,
    # SE 1%, ratio 3, error rate 0.25, which is at the ceiling and not over it.
    # mutation: no overwrite in _validate_and_finish leaves 0.0 / no_spread in the record
    from talos.challenges import CHALLENGES

    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [None] + q[1:] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, budget=ONE)
    loop.rule = replace(CHALLENGES["knapsack"].beat, error_ceiling=0.25)
    st = loop.run()
    assert st.validations[0]["outcome"] == "validated"
    rec = st.hypotheses[0]
    assert rec["mean_rel_se"] == pytest.approx(0.01, rel=1e-9)
    assert rec["noise_ratio"] == pytest.approx(3.0, rel=1e-9)
    assert (rec["noise_band"], rec["noise_state"]) == ("clear", "ok")
    ev = _scored_events(store)[0]
    assert ev["mean_rel_se"] == 0.0 and ev["noise_state"] == "no_spread"


def test_a_demoted_record_keeps_the_native_noise(tmp_path):
    # Review Focus 1. Metered fails one nonce of four: 25% is over the default 5% ceiling, so
    # validation demotes it as "error_ceiling" and, unlike "nondeterministic", returns a
    # metered delta. mutation: overwriting the noise fields from that delta before the
    # demotion return records "over_ceiling" instead of the native "no_spread"
    def metered(ch, files, ns):
        q = quality_from_files(ch, files, ns)
        return [None] + q[1:] if "let k = 5;" in files["mod.rs"] else q
    fb = FakeBench(quality_from_files, metered_scores=metered)
    loop, fp, _, store = native(tmp_path, [hyp("a"), edit(5)], bench=fb, budget=ONE)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] == "failed:validation"
    assert st.validations[0]["reason"] == "error_ceiling"
    ev = _scored_events(store)[0]
    assert ev["noise_state"] == "no_spread"
    for k in ("mean_rel_se", "noise_ratio", "noise_band", "noise_state"):
        assert st.hypotheses[0][k] == ev[k]


def test_focused_scored_event_has_the_focus_track_se_only(tmp_path):
    # Review Focus 3. mutation: computing the SE against the unselected baseline raises a
    # track mismatch, or lists track u
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(2)], budget=ONE, two_tracks=True,
                               track="t")
    loop.run()
    assert set(_scored_events(store)[0]["track_rel_se"]) == {"t"}


def _energy(loop):
    """The loop's job, on the challenge the leak rules apply to."""
    loop.spec = replace(loop.spec, challenge="energy_arbitrage")
    return loop


def _with(code, frm=1, k=5):
    return f"<<<<<<< SEARCH mod.rs\nlet k = {frm};\n=======\nlet k = {k}; {code}\n>>>>>>> REPLACE\n"


ONE_ITERATION = Budget(usd=None, hours=None, iterations=1, compute_usd=None)


@pytest.mark.parametrize("code, why", [
    ("let v = serde_json::to_value(challenge);", "serialises"),
    ("let a = std::env::args();", "process"),
    ("solution.schedule = rewritten;", "rewrites the schedule"),
], ids=["leak", "argv", "hindsight"])
def test_a_candidate_breaking_the_causality_rule_never_reaches_the_bench(tmp_path, code, why):
    # mutation: checking after the bench pays to score it, and could promote it to best
    loop, fp, fb, store = make(tmp_path, [hyp("a"), _with(code)], budget=ONE_ITERATION)
    st = _energy(loop).run()
    assert st.best is None and fb.calls == [] and st.spend.compute_usd == 0
    assert st.hypotheses[0]["outcome"] == "failed:leak" and why in st.hypotheses[0]["error"]
    events = [json.loads(ln) for ln in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert [e for e in events if e["kind"] == "leak_rejected"]


def test_the_rule_and_the_rejection_reach_the_next_hypothesis_prompt(tmp_path):
    # mutation: not recording the reason has the LLM repeat the same idea blind
    loop, fp, fb, store = make(tmp_path, [hyp("a"), _with("let h = hidden_seed;"), hyp("b"),
                                          edit(5)],
                               budget=Budget(usd=None, hours=None, iterations=2, compute_usd=None),
                               thresholds=Thresholds(recall=1, distill=99, reset=99))
    st = _energy(loop).run()
    assert st.hypotheses[0]["outcome"] == "failed:leak"
    assert "reads hidden_seed" in fp.calls[2][1]
    assert "Causality rule" in fp.calls[2][0]


def test_a_compile_fix_cannot_introduce_the_leak(tmp_path):
    # mutation: checking only the first edit lets the fix round carry it to the bench
    fix = _with("let v = serde_json::to_value(challenge);", frm="BUG")
    loop, fp, fb, store = make(tmp_path, [hyp("a"), _with("", k="BUG"), fix], budget=ONE_ITERATION)
    fb._compile_ok = lambda files: "BUG" not in files["mod.rs"]
    st = _energy(loop).run()
    assert st.hypotheses[0]["outcome"] == "failed:leak"
    assert len(fb.calls) == 1  # the broken edit only


def test_inherited_schedule_writes_do_not_fail_a_candidate(tmp_path):
    # mutation: measuring hindsight without the parent refuses every candidate of peakshift
    site = "solution.schedule = rewritten;"
    loop, fp, fb, store = make(tmp_path, [hyp("a"), edit(5)], budget=ONE_ITERATION)
    loop.state.baseline.files = {"mod.rs": f"fn solve() {{ let k = 1;\n{site} }}\n"}
    st = _energy(loop).run()
    assert st.hypotheses[0]["outcome"] != "failed:leak" and len(fb.calls) >= 1


def test_the_rules_do_not_touch_other_challenges(tmp_path):
    # mutation: an unscoped check fails a knapsack candidate over code the rule says nothing about
    loop, fp, fb, store = make(tmp_path, [hyp("a"), _with("let v = serde_json::to_value(x);")],
                               budget=ONE_ITERATION)
    st = loop.run()
    assert st.hypotheses[0]["outcome"] != "failed:leak" and len(fb.calls) >= 1
