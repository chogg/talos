import pytest

from talos.prompts import (PromptContext, SEARCH_REPLACE_FORMAT, STRATEGY_TAGS,
                           WHOLE_FILE_FORMAT, challenge_block, compile_fix_prompts,
                           distill_prompts, describe_attempt, edit_prompts, edit_repair_prompts,
                           gap_block, hypothesis_prompts,
                           parse_distillation, parse_hypothesis)


def ctx(**kw):
    base = dict(challenge="knapsack", template_rs="pub fn solve_challenge(", direction="Try tabu",
                tacit="- lesson one", files={"mod.rs": "fn x(){}"}, baseline_name="qk_v3",
                best_delta=-0.01, failed_hypotheses=[], forced_tag=None, is_gpu=False)
    base.update(kw)
    return PromptContext(**base)


def test_hypothesis_prompt_carries_direction_tacit_and_target():
    # mutation: dropping direction/tacit/baseline_name from the user prompt, or the
    # strategy tag list / template_rs from the system prompt, leaves the model blind
    system, user = hypothesis_prompts(ctx())
    assert "Try tabu" in user and "lesson one" in user and "qk_v3" in user
    assert "solve_challenge" in system
    assert all(t in system for t in STRATEGY_TAGS)


def test_hypothesis_prompt_never_contains_hash():
    # a rand_hash is 64 hex chars; the context has no field for it, so make sure
    # nothing that looks like one is interpolated from files or tacit
    # mutation: adding a rand_hash field to PromptContext and interpolating it would leak
    # a nonce identifier into the prompt, which the design forbids
    system, user = hypothesis_prompts(ctx())
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", system + user)


def test_failed_hypotheses_and_forced_tag_appear_when_given():
    # mutation: dropping the recall block loses the "do not repeat" signal
    c = ctx(failed_hypotheses=[{"title": "Bigger tabu tenure", "outcome": "failed:score"}],
            forced_tag="decomposition")
    _, user = hypothesis_prompts(c)
    assert "Bigger tabu tenure" in user and "decomposition" in user


def test_recall_separates_the_current_code_from_earlier_bests():
    # issue #33. mutation: listing every failure under "this exact code", or dropping the label
    # naming the best it failed against, tells the model an old failure was on today's code
    c = ctx(anchor=5, failed_hypotheses=[
        {"title": "Current fail", "outcome": "failed:score", "against": 5},
        {"title": "Baseline fail", "outcome": "failed:score", "against": 0},
        {"title": "Middle fail", "outcome": "failed:compile", "against": 3},
        {"title": "Unanchored", "outcome": "failed:edit"}])
    _, user = hypothesis_prompts(c)
    current, _, earlier = user.partition("against an earlier best")
    assert "Current fail" in current and "Unanchored" in current
    assert "Baseline fail" not in current and "Middle fail" not in current
    assert "- Baseline fail [failed:score, against the baseline]" in earlier
    assert "- Middle fail [failed:compile, against best #3]" in earlier


def test_distill_prompt_names_the_best_each_attempt_failed_against():
    # mutation: distilling the whole job without the parent label merges attempts on different
    # code into one list with nothing to tell them apart
    _, user = distill_prompts(ctx(anchor=4), [
        {"title": "A", "outcome": "failed:score", "against": 0},
        {"title": "B", "outcome": "failed:score", "against": 4}])
    assert "- A [failed:score, against the baseline]" in user
    assert "- B [failed:score, against best #4]" in user


def test_edit_prompt_shows_files_and_format():
    hyp = {"title": "Bitset tabu", "description": "Use a bitset for the tabu list",
           "strategy_tag": "local_search"}
    system, user = edit_prompts(ctx(), hyp)
    # mutation: dropping the description from the user prompt leaves the coder without the idea
    assert "<<<<<<< SEARCH" in system and "mod.rs" in user and "fn x(){}" in user
    assert "Use a bitset for the tabu list" in user


def test_parse_hypothesis_tolerates_prose_and_validates_tag():
    text = 'Sure.\n{"title": "A", "description": "B", "strategy_tag": "local_search"}\nThanks'
    h = parse_hypothesis(text)
    assert h == {"title": "A", "description": "B", "strategy_tag": "local_search",
                 "design_family": "unlabelled"}
    # mutation: accepting an unknown tag breaks strategy_counts bookkeeping
    bad = '{"title": "A", "description": "B", "strategy_tag": "magic"}'
    assert parse_hypothesis(bad)["strategy_tag"] == "hybrid"
    with pytest.raises(ValueError):
        parse_hypothesis("no json here")


def test_parse_hypothesis_survives_braces_in_description():
    # mutation: the non-greedy brace regex cuts the object at the first `}` inside the
    # description
    text = ('Idea:\n{"title": "A", "description": "Use {x: y} sets for lookups", '
            '"strategy_tag": "data_structure"}')
    h = parse_hypothesis(text)
    assert h == {"title": "A", "description": "Use {x: y} sets for lookups",
                 "strategy_tag": "data_structure", "design_family": "unlabelled"}
    # a leading JSON object without title/description must be skipped in favour of the
    # next, valid one
    text2 = ('{"note": 1}\n{"title": "A", "description": "B", '
             '"strategy_tag": "local_search"}')
    assert parse_hypothesis(text2) == {"title": "A", "description": "B",
                                        "strategy_tag": "local_search",
                                        "design_family": "unlabelled"}


def test_compile_fix_and_repair_prompts_carry_inputs():
    # mutation: dropping the compiler output, file content, SEARCH format, or rust rules
    # from these prompts leaves the fixer without the context it needs
    system, user = compile_fix_prompts(ctx(), {"mod.rs": "fn broken("},
                                       "error[E0308]: mismatched types")
    assert "error[E0308]: mismatched types" in user
    assert "fn broken(" in user
    assert "<<<<<<< SEARCH" in system
    # distinctive substring from talos/data/rust_rules.md, unlikely to appear elsewhere
    assert "RULE 1 - NO DUPLICATE STRUCTS" in system

    system2, user2 = edit_repair_prompts(ctx(), {"mod.rs": "fn x(){}"},
                                         "- [not_found] in mod.rs: ...")
    assert "- [not_found] in mod.rs: ..." in user2
    assert "fn x(){}" in user2
    assert "<<<<<<< SEARCH" in system2


def test_compile_fix_prompts_truncates_compiler_output():
    # mutation: dropping the `[-12000:]` slice blows the prompt budget on long build logs
    # distinct head/tail content (not a repeated char) so substring checks are discriminating
    long_output = "A" * 4000 + "B" * 12000
    _, user = compile_fix_prompts(ctx(), {"mod.rs": "fn x(){}"}, long_output)
    assert long_output[-12000:] in user
    assert "A" * 4000 not in user


def test_distill_roundtrip():
    system, user = distill_prompts(ctx(), [{"title": "Bigger tabu tenure", "outcome": "failed:score"}])
    assert "Bigger tabu tenure" in user  # mutation: dropping the failure list from the prompt
    assert parse_distillation("LESSON: Prefer cheap moves early.") == "Prefer cheap moves early."
    assert parse_distillation("nothing useful") is None


def test_focused_prompts_name_the_track_and_the_guards():
    # mutation: dropping the track from the system prompt leaves the model optimising every
    # track; dropping the guard list hides that the other tracks are re-scored
    c = ctx(track="n_items=5000,budget=25", guard_tracks=["n_items=1000,budget=5"])
    system, user = hypothesis_prompts(c)
    assert "n_items=5000,budget=25" in system and "n_items=1000,budget=5" in system
    assert "regression guard" in system
    system2, user2 = edit_prompts(c, {"title": "t", "description": "d"})
    assert "n_items=5000,budget=25" in (system2 + user2)


def test_unfocused_prompts_do_not_mention_a_guard():
    # mutation: emitting the focus sentence with an empty track changes every existing prompt
    system, user = hypothesis_prompts(ctx())
    assert "regression guard" not in system and "across every active track" in system


def test_compile_fix_prompt_names_the_files_and_filters_foreign_warnings():
    # The model in iteration 3 of run 20260916-095103 addressed the candidate's files by the
    # path the compiler printed, and in iterations 1 and 5 by another algorithm's directory
    # that dominated the warning spam. mutation: dropping the file-name line or the filter
    # brings both back
    foreign = ("warning: unused variable: `x`\n"
               "  --> tig-algorithms/src/knapsack/knap_quality_opt_v11/track1.rs:1:1\n\n")
    err = ("error[E0308]: mismatched types\n"
           "  --> tig-algorithms/src/knapsack/talos_cand/track1.rs:5:5\n")
    _, user = compile_fix_prompts(ctx(), {"track1.rs": "fn x(){}", "mod.rs": ""}, foreign + err)
    assert "knap_quality_opt_v11" not in user
    assert err in user
    assert "mod.rs, track1.rs" in user


def test_failed_attempt_lines_carry_the_numbers_and_the_error():
    # A title plus "failed:score" told the model nothing about iteration 2's +0.14% on one
    # track, -0.37% on another, and 14x runtime. mutation: dropping any field silences it
    scored = {"title": "Dynamic greedy", "outcome": "failed:score", "mean_rel_delta": -0.00049,
              "worst_track": "n_items=5000,budget=10", "worst_rel_delta": -0.00374,
              "runtime_ratio": 14.07}
    line = describe_attempt(scored)
    assert line.startswith("- Dynamic greedy [failed:score]")
    assert "-0.05%" in line and "n_items=5000,budget=10" in line and "-0.37%" in line
    assert "14.1x" in line
    edit = {"title": "Beam", "outcome": "failed:edit",
            "error": "edit outside the algorithm files rejected: ['x/track2.rs']"}
    assert "rejected: ['x/track2.rs']" in describe_attempt(edit)
    _, user = hypothesis_prompts(ctx(failed_hypotheses=[scored, edit]))
    assert "14.1x" in user and "rejected: ['x/track2.rs']" in user


def test_compile_fix_prompt_shows_every_error_of_a_build_the_size_of_iteration_5():
    # iteration 5 of run 20260916-095103 failed with 8 errors; its filtered output would run
    # to about 9 KB, and a 6000-character window shows the fixer only the last few.
    # mutation: a window under the size of eight rustc errors drops the first ones
    errs = "".join(f"error[E0308]: mismatched types (site {i})\n"
                   f"   --> tig-algorithms/src/knapsack/talos_cand/track1.rs:{i}:5\n"
                   "    |\n" + f"{i:<3} | " + "x" * 900 + "\n\n" for i in range(8))
    _, user = compile_fix_prompts(ctx(), {"track1.rs": "fn x(){}"}, errs)
    assert all(f"(site {i})" in user for i in range(8))


HP = {"t": {"b": 2, "a": 1}, "u": None}


def test_hypothesis_and_edit_prompts_show_every_tracks_hyperparameters():
    c = ctx(hyperparameters=HP)
    _, hyp_user = hypothesis_prompts(c)
    _, edit_user = edit_prompts(c, {"title": "t", "description": "d"})
    for user in (hyp_user, edit_user):
        # mutation: leaving the block out of either prompt lets the model rename a key unawares
        assert 'track t: {"a":1,"b":2}' in user
        assert "track u: none (solve_challenge receives None)" in user
        assert "do not rename or remove" in user
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", hyp_user + edit_user)


def test_no_hyperparameters_block_without_a_map():
    _, user = hypothesis_prompts(ctx())
    # mutation: printing the block for None tells the model values are passed when none are
    assert "Hyperparameters:" not in user


def test_a_focused_prompt_shows_only_the_focus_tracks_hyperparameters():
    c = ctx(hyperparameters={"t": {"x": 1}, "u": {"x": 2}}, track="t", guard_tracks=["u"])
    _, user = hypothesis_prompts(c)
    # mutation: listing every track spends the focused prompt on tracks it must not tune for
    assert 'track t: {"x":1}' in user and '{"x":2}' not in user
    assert "guard tracks run with their own values" in user


def test_hypothesis_asks_for_a_design_family():
    # mutation: never requesting the key makes every node "unlabelled"
    system, _ = hypothesis_prompts(ctx())
    assert "design_family" in system
    assert "\"strategy_tag\" (one of" in system


def test_parse_hypothesis_design_family():
    # mutation: dropping the key from the result, or the "unlabelled" default
    text = ('{"title": "A", "description": "B", "strategy_tag": "hybrid", '
            '"design_family": "rolling-horizon LP"}')
    assert parse_hypothesis(text)["design_family"] == "rolling-horizon LP"
    assert parse_hypothesis('{"title": "A", "description": "B"}')["design_family"] == "unlabelled"
    blank = '{"title": "A", "description": "B", "design_family": "  "}'
    assert parse_hypothesis(blank)["design_family"] == "unlabelled"
    long = '{"title": "A", "description": "B", "design_family": "' + "x" * 200 + '"}'
    assert len(parse_hypothesis(long)["design_family"]) == 80


def test_gap_block_shows_parent_and_best_per_track():
    # mutation: unsorted tracks, or the parent and best gaps swapped
    c = ctx(parent_gap={"node": 3, "mean": -0.12, "tracks": {"b": -0.2, "a": -0.04}},
            best_gap={"node": 5, "mean": -0.05, "tracks": {"a": -0.05, "b": -0.05}})
    text = gap_block(c)
    assert 'Parent (node #3): -12.000% vs mainnet "qk_v3"; track a -4.000%; track b -20.000%' \
        in text
    assert 'Best so far (node #5): -5.000% vs mainnet "qk_v3"; track a -5.000%' in text
    none = gap_block(ctx(start="mainnet"))
    assert "Parent: the mainnet baseline itself (+0.000%)" in none
    assert "Best so far: none yet" in none
    assert "not available to you" not in none
    tmpl = gap_block(ctx(start="template"))
    assert "Parent: the empty template (not scored yet)" in tmpl
    assert tmpl.endswith("The baseline's code is not available to you; only its score is.")


def test_template_start_shows_challenge_docs_and_hides_baseline_code():
    # mutation: challenge_block ignoring start, dropping the path header, or omitting it from
    # one of the prompts
    c = ctx(start="template", challenge_docs={"mod.rs": "MODRS-DOC", "README.md": "CHALLENGE-DOC"})
    _, hyp_user = hypothesis_prompts(c)
    _, edit_user = edit_prompts(c, {"title": "t", "description": "d"})
    _, fix_user = compile_fix_prompts(c, c.files, "error")
    for user in (hyp_user, edit_user, fix_user):
        assert "CHALLENGE-DOC" in user and "tig-challenges/src/knapsack/README.md" in user
    block = challenge_block(c)
    assert block.index("README.md ---") < block.index("tig-challenges/src/knapsack/mod.rs ---")
    assert "The baseline's code is not available to you" in hyp_user
    assert challenge_block(ctx(start="mainnet", challenge_docs={"README.md": "X"})) == ""
    _, mainnet_user = hypothesis_prompts(ctx(start="mainnet", challenge_docs={"README.md": "X"}))
    assert "tig-challenges" not in mainnet_user


def test_edit_prompt_switches_to_whole_file_format():
    # mutation: ignoring whole_file, or emitting both formats
    hyp = {"title": "t", "description": "d"}
    whole, _ = edit_prompts(ctx(whole_file=True), hyp)
    assert WHOLE_FILE_FORMAT in whole and SEARCH_REPLACE_FORMAT not in whole
    search, _ = edit_prompts(ctx(whole_file=False), hyp)
    assert SEARCH_REPLACE_FORMAT in search and WHOLE_FILE_FORMAT not in search


def test_hyperparameters_block_absent_for_template_start():
    # mutation: a hard-coded Hyperparameters header regardless of ctx.hyperparameters
    _, user = hypothesis_prompts(ctx(start="template", hyperparameters=None))
    assert "Hyperparameters:" not in user


TREE_FAILURES = [{"title": "Root fail", "outcome": "failed:score", "against": 0},
                 {"title": "Node fail", "outcome": "failed:compile", "against": 3}]


@pytest.mark.parametrize("kw, root", [
    ({"start": "template"}, "the empty template"),
    ({"search": "anneal"}, "the baseline"),
    ({"chains": 2}, "the baseline"),
])
def test_recall_in_a_tree_names_parent_nodes_not_bests(kw, root):
    # mutation: keying the tree wording on start alone (anneal/chains keep "best #3"), labelling
    # node 0 "the baseline" in a template start, or keeping the "earlier best" title
    _, user = hypothesis_prompts(ctx(anchor=5, failed_hypotheses=TREE_FAILURES, **kw))
    assert "earlier best" not in user and "best #" not in user
    current, _, other = user.partition("Tried against other code and did NOT help there; ")
    assert other.startswith("do not repeat unless the current code removes the reason")
    assert f"- Root fail [failed:score, against {root}]" in other
    assert "- Node fail [failed:compile, against node #3]" in other
    _, distill = distill_prompts(ctx(**kw), TREE_FAILURES)
    assert f"against {root}]" in distill and "against node #3]" in distill


def test_gpu_template_hypothesis_prompt_shows_template_cu():
    # mutation: the LLM is not told which file holds its kernels
    c = ctx(challenge="hypergraph", start="template", is_gpu=True, template_cu="CU_MARKER")
    system, _ = hypothesis_prompts(c)
    assert "algorithm.cu" in system and "CU_MARKER" in system
    system, _ = hypothesis_prompts(ctx(challenge="hypergraph", start="template", is_gpu=True))
    assert "algorithm.cu" not in system and "CU_MARKER" not in system


@pytest.mark.parametrize("build", [hypothesis_prompts, lambda c: edit_prompts(
    c, {"title": "t", "description": "d"})], ids=["hypothesis", "edit"])
def test_the_causality_rule_is_stated_only_where_it_applies(build):
    # mutation: not telling the LLM has it spend iterations on candidates the check refuses
    assert "Causality rule" in build(ctx(challenge="energy_arbitrage"))[0]
    assert "Causality rule" not in build(ctx())[0]
