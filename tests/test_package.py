import json
import shutil
import subprocess
import zipfile
from dataclasses import replace

import pytest

from talos.budget import Budget, Spend
from talos.challenges import BeatRule
from talos.lineage import ROOT, Chain, Node
from talos.package import build_package, evidence_draft, scores_markdown
from talos.state import BaselineRecord, Candidate, JobSpec, JobState, JobStore
from talos.types import NonceResult, NonceSet

HASH = "cd" * 32


def make(tmp_path, status="won"):
    spec = JobSpec(job_id="j", challenge="knapsack", direction="go", provider="fake", model="m",
                   mode="single-shot", budget=Budget(usd=1.0, hours=None, iterations=None, compute_usd=None),
                   rand_hash=HASH, tracks=["t"], training=[NonceSet("t", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2)], fuel=1, created_at=0.0,
                   monorepo_ref="r", challenge_id="c003")
    base = BaselineRecord("base", 1, "a", {"mod.rs": "fn a() {}\nlet k = 1;\n"},
                          [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, True, 100, 1)],
                          [NonceResult("t", 1_000_000, True, 100, 1), NonceResult("t", 1_000_001, True, 100, 1)])
    cand = Candidate(3, {"mod.rs": "fn a() {}\nlet k = 9;\n"}, "b",
                     [NonceResult("t", 0, True, 110, 1), NonceResult("t", 1, True, 111, 1)],
                     {"mean_rel_delta": 0.105}, {"title": "Bump k", "description": "d", "strategy_tag": "hybrid"},
                     holdout=[NonceResult("t", 1_000_000, True, 108, 1), NonceResult("t", 1_000_001, True, 109, 1)])
    st = JobState.fresh(Spend(started_at=0.0))
    st.status, st.best, st.baseline = status, cand, base
    st.hypotheses = [{"iteration": 3, "against": 0, "title": "Bump k", "description": "d",
                      "strategy_tag": "hybrid", "outcome": "improved"}]
    store = JobStore(tmp_path)
    store.write_spec(spec)
    store.save(st)
    return spec, st, store


def test_package_contents_and_no_hash(tmp_path):
    spec, st, store = make(tmp_path)
    # so the hash-grep below also covers hyperparameters.json and the README section
    spec = replace(spec, hyperparameters={"t": {"x": 1}},
                   hyperparameters_source={"t": {"benchmark_id": "b1", "player_id": "p1",
                                                 "mean_quality": 1.0}})
    st.hypotheses.append({"iteration": 2, "against": 0, "title": "Bad edit", "description": "x",
                          "strategy_tag": "hybrid", "outcome": "failed:edit",
                          "error": "no edit block applied"})
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert {"submission", "diff_vs_baseline.patch", "scores.md", "hypotheses.md",
            "evidence_draft.md", "README.md", "hyperparameters.json"} <= names
    assert "let k = 9" in (pkg / "diff_vs_baseline.patch").read_text(encoding="utf-8")
    # mutation: dropping the error string from the log hides why an iteration failed
    hyps_text = (pkg / "hypotheses.md").read_text(encoding="utf-8")
    assert "vs best #0" in hyps_text
    assert "no edit block applied" in hyps_text
    # mutation: dumping job.json into the package leaks rand_hash
    for p in pkg.rglob("*"):
        if p.is_file():
            assert HASH not in p.read_text(errors="ignore")
    z = zipfile.ZipFile(tmp_path / "package.zip")
    assert "submission/mod.rs" in z.namelist()
    # mutation: building the zip from run_dir would ship job.json
    for name in z.namelist():
        assert HASH not in z.read(name).decode(errors="ignore")


def test_scores_markdown_has_per_nonce_rows_and_delta():
    # mutation: swapping baseline/candidate columns or dropping the error note
    base = [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, False, None, 1, "panic")]
    cand = [NonceResult("t", 0, True, 120, 1), NonceResult("t", 1, True, 130, 1)]
    md = scores_markdown(base, cand)
    assert "| t | 0 | 100 | 120 |" in md and "panic" in md and "mean_rel_delta" in md
    assert "| panic | 130 | err |" in md


def test_evidence_draft_prefills_challenge_and_method(tmp_path):
    # mutation: leaving "YOUR RESPONSE HERE" unreplaced, or not reading the shipped template
    spec, st, store = make(tmp_path)
    text = evidence_draft(spec, st)
    assert "knapsack" in text and "Bump k" in text and "UNIQUE ALGORITHM IDENTIFIER" in text


def test_package_for_exhausted_job_still_builds(tmp_path):
    # mutation: README always claiming confirmation regardless of state.confirmed
    spec, st, store = make(tmp_path, status="exhausted")
    pkg = build_package(spec, st, store)
    assert "never scored" in (pkg / "README.md").read_text(encoding="utf-8")


def test_package_without_candidate_says_so(tmp_path):
    # mutation: rendering the candidate README with best=None tells the user to submit
    # files that do not exist
    spec, st, store = make(tmp_path, status="exhausted")
    st.best = None
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert names == {"README.md", "report.html"}  # no submission, scores or patch
    text = (pkg / "README.md").read_text(encoding="utf-8")
    assert "No candidate" in text
    assert "scores.md" not in text
    assert "Copy the algorithm files" not in text
    assert (tmp_path / "package.zip").exists()


def test_readme_distinguishes_false_positive_from_untested(tmp_path):
    # mutation: collapsing the two unconfirmed cases hides a measured held-out miss from
    # the submitter
    spec_fp, st_fp, store_fp = make(tmp_path / "fp", status="exhausted")
    st_fp.false_positives = [st_fp.best.iteration]
    pkg_fp = build_package(spec_fp, st_fp, store_fp)
    fp_text = (pkg_fp / "README.md").read_text(encoding="utf-8")

    spec_u, st_u, store_u = make(tmp_path / "untested", status="exhausted")
    pkg_u = build_package(spec_u, st_u, store_u)
    untested_text = (pkg_u / "README.md").read_text(encoding="utf-8")

    assert "false positive" in fp_text
    assert "never scored" in untested_text
    assert "false positive" not in untested_text


def test_readme_does_not_deny_a_training_win_that_was_never_confirmed(tmp_path):
    # A run can stop (budget, Ctrl-C) between the training win and the held-out run. The README
    # must say the candidate beat the baseline on training and was never confirmed, not that it
    # "did not beat the baseline on training".
    # mutation: collapsing every unconfirmed candidate into "did not beat the baseline" misreports
    # a measured training win to the submitter
    spec_w, st_w, store_w = make(tmp_path / "winner", status="exhausted")
    st_w.best.holdout = None  # +10.5% on training, confirmation never ran
    winner_text = (build_package(spec_w, st_w, store_w) / "README.md").read_text(encoding="utf-8")
    assert "beat the baseline on the training nonces" in winner_text
    assert "never" in winner_text and "did not beat the baseline on training" not in winner_text

    spec_l, st_l, store_l = make(tmp_path / "loser", status="exhausted")
    st_l.best.holdout = None
    st_l.best.training = [NonceResult("t", 0, True, 100, 1), NonceResult("t", 1, True, 100, 1)]
    loser_text = (build_package(spec_l, st_l, store_l) / "README.md").read_text(encoding="utf-8")
    assert "did not beat the baseline on training" in loser_text
    assert "beat the baseline on the training nonces" not in loser_text


def make_focused(tmp_path):
    """Two tracks t and u; the job focuses t; the candidate's held-out carries t's held-out rows
    and u's training rows (the guard), exactly as the loop stores them."""
    spec, st, store = make(tmp_path)
    spec = replace(spec, tracks=["t", "u"], track="t",
                   training=[NonceSet("t", HASH, 0, 2), NonceSet("u", HASH, 0, 2)],
                   holdout=[NonceSet("t", HASH, 1_000_000, 2), NonceSet("u", HASH, 1_000_000, 2)])
    st.baseline.training += [NonceResult("u", 0, True, 50, 1), NonceResult("u", 1, True, 50, 1)]
    st.baseline.holdout += [NonceResult("u", 1_000_000, True, 50, 1),
                            NonceResult("u", 1_000_001, True, 50, 1)]
    st.best.holdout += [NonceResult("u", 0, True, 50, 1), NonceResult("u", 1, True, 49, 1)]
    (tmp_path / "job.json").unlink()
    store.write_spec(spec)
    return spec, st, store


def test_focused_package_splits_scores_into_track_and_guard_sections(tmp_path):
    # mutation: comparing the whole baseline against the single-track candidate raises a
    # ScoringError and prints "(no bundle delta" instead of the delta; dropping the guard
    # section hides the u regression the confirmation saw
    spec, st, store = make_focused(tmp_path)
    pkg = build_package(spec, st, store)
    scores = (pkg / "scores.md").read_text(encoding="utf-8")
    assert "# Training nonces (track t)" in scores
    assert "# Held-out nonces (track t)" in scores
    assert "# Regression guard (other tracks, training nonces)" in scores
    assert "(no bundle delta" not in scores
    guard = scores.split("# Regression guard")[1]
    assert "| u | 1 | 50 | 49 | - |" in guard
    readme = (pkg / "README.md").read_text(encoding="utf-8")
    assert "Optimised for track t" in readme and "regression guard" in readme
    evidence = (pkg / "evidence_draft.md").read_text(encoding="utf-8")
    assert "Optimised for track t" in evidence and "Regression guard" in evidence


def test_unfocused_package_keeps_todays_headings(tmp_path):
    # mutation: the focus headings leaking into unfocused packages
    spec, st, store = make(tmp_path)
    scores = (build_package(spec, st, store) / "scores.md").read_text(encoding="utf-8")
    assert "# Training nonces\n" in scores and "(track" not in scores
    assert "Regression guard" not in scores


def test_package_records_the_hyperparameters_and_their_source(tmp_path):
    spec, st, store = make(tmp_path)
    spec = replace(spec, hyperparameters={"t": {"x": 1}},
                   hyperparameters_source={"t": {"benchmark_id": "bm1", "player_id": "0xp",
                                                 "mean_quality": 150.0}})
    pkg = build_package(spec, st, store)
    readme = (pkg / "README.md").read_text(encoding="utf-8")
    # mutation: omitting the section leaves the user submitting without the values they beat with
    assert "## Hyperparameters" in readme
    assert '- t: `{"x": 1}` (mainnet benchmark bm1 by 0xp, mean quality 150)' in readme
    # mutation: not writing the file leaves nothing to paste into a benchmarker config
    import json
    assert json.loads((pkg / "hyperparameters.json").read_text(encoding="utf-8")) == {"t": {"x": 1}}


def test_package_without_hyperparameters_has_no_section_or_file(tmp_path):
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    # mutation: writing the section for None claims values were used when none were
    assert "## Hyperparameters" not in (pkg / "README.md").read_text(encoding="utf-8")
    assert not (pkg / "hyperparameters.json").exists()


def test_submission_folder_holds_the_code_and_the_tig_readme_template(tmp_path):
    spec, st, store = make(tmp_path)
    # the mainnet algorithm folder ships its author's own submission README; a GPU algorithm
    # also has .cu kernels, possibly nested
    st.best.files = {"mod.rs": "fn a() {}\n", "kernels/solve.cu": "__global__ void k() {}\n",
                     "README.md": "* **Copyright:** 2026 NVX\n"}
    pkg = build_package(spec, st, store)
    sub = pkg / "submission"
    assert sorted(str(p.relative_to(sub)).replace("\\", "/") for p in sub.rglob("*")
                  if p.is_file()) == ["README.md", "kernels/solve.cu", "mod.rs"]
    # mutation: filtering on .rs drops the GPU kernels
    assert (sub / "kernels" / "solve.cu").read_text(encoding="utf-8") == "__global__ void k() {}\n"
    # mutation: leaving the code at the top level as well duplicates it next to the reports
    assert not (pkg / "mod.rs").exists()
    readme = (sub / "README.md").read_text(encoding="utf-8")
    # mutation: copying the baseline's README submits under someone else's copyright
    assert "NVX" not in readme
    # mutation: not substituting the challenge leaves the placeholder in the template
    assert "* **Challenge Name:** knapsack\n" in readme and "{challenge}" not in readme
    assert "* **Copyright:** [year work created] [name of copyright owner]" in readme
    assert "TIG Inbound Game License" in readme
    # mutation: writing the template over package/README.md loses the hand-back notes
    assert "Talos hand-back" in (pkg / "README.md").read_text(encoding="utf-8")


def test_package_without_candidate_has_no_submission_folder(tmp_path):
    spec, st, store = make(tmp_path, status="exhausted")
    st.best = None
    # mutation: writing an empty submission/ invites submitting nothing
    assert not (build_package(spec, st, store) / "submission").exists()


def _node(id, parent, family, title, outcome, score, accepted, best=False):
    return Node(id=id, chain=0, parent=parent, family=family, title=title, outcome=outcome,
                score=score, accepted=accepted, best=best)


def test_lineage_files_show_the_best_path_and_families(tmp_path):
    # mutation: listing every node on the path, or counting rejected nodes as accepted
    spec, st, store = make(tmp_path)
    st.nodes = [_node(1, ROOT, "lp", "First", "improved", 0.01, True),
                _node(2, 1, "lp", "Second", "improved", 0.03, True, best=True),
                _node(3, 1, "greedy", "Third", "rejected", -0.02, False)]
    st.chains = [Chain(index=0, current=2, best=2, best_score=0.03)]
    st.best = replace(st.best, iteration=2)
    st.validations = [{"outcome": "validated"}, {"outcome": "demoted"}, {"outcome": "validated"}]
    pkg = build_package(spec, st, store)
    data = json.loads((pkg / "lineage.json").read_text(encoding="utf-8"))
    assert set(data) == {"nodes", "chains"} and len(data["nodes"]) == 3
    md = (pkg / "lineage.md").read_text(encoding="utf-8")
    first = "#1 [lp] First: +1.000% vs mainnet"
    second = "#2 [lp] Second: +3.000% vs mainnet"
    assert first in md and second in md and md.index(first) < md.index(second)
    assert "#3 [greedy]" not in md.split("| family")[0]
    assert "| lp | 2 | 2 |" in md and "| greedy | 1 | 0 |" in md
    assert "Metered validations: 3 (1 demoted)" in md


def test_template_package_diffs_against_the_template_and_drops_the_uai_line(tmp_path):
    # mutation: diffing against the baseline hands the user a patch built from mainnet code
    spec, st, store = make(tmp_path)
    marker = "BASELINE_MARKER"
    st.baseline = replace(st.baseline, files={"mod.rs": f"fn a() {{}}\n// {marker}\n"})
    spec = replace(spec, start="template", hyperparameters={"t": {"x": 1}})
    (store.run_dir / "challenge").mkdir()
    (store.run_dir / "challenge" / "template.rs").write_text("fn a() {}\n", encoding="utf-8")
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert "diff_vs_template.patch" in names
    assert "diff_vs_baseline.patch" not in names and "hyperparameters.json" not in names
    patch = (pkg / "diff_vs_template.patch").read_text(encoding="utf-8")
    assert "template/mod.rs" in patch and marker not in patch
    readme = (pkg / "README.md").read_text(encoding="utf-8")
    assert "Unique Algorithm Identifier" not in readme and "never shown to the LLM" in readme
    assert "candidate ran with none" in readme
    for p in pkg.rglob("*"):
        if p.is_file():
            assert marker not in p.read_text(errors="ignore")


def test_package_without_a_best_still_writes_lineage(tmp_path):
    # mutation: writing lineage.json after the no-best early return drops it
    spec, st, store = make(tmp_path)
    st.best = None
    st.nodes = [_node(1, ROOT, "lp", "First", "rejected", -0.01, False)]
    st.chains = [Chain(index=0)]
    pkg = build_package(spec, st, store)
    assert len(json.loads((pkg / "lineage.json").read_text(encoding="utf-8"))["nodes"]) == 1


def test_template_package_without_a_template_copy_still_hands_back(tmp_path):
    # mutation: falling back to the baseline diff, or reading template.rs unguarded
    spec, st, store = make(tmp_path)
    marker = "BASELINE_MARKER"
    st.baseline = replace(st.baseline, files={"mod.rs": f"fn a() {{}}\n// {marker}\n"})
    spec = replace(spec, start="template")
    pkg = build_package(spec, st, store)
    names = {p.name for p in pkg.iterdir()}
    assert "README.md" in names and (tmp_path / "package.zip").exists()
    assert "diff_vs_baseline.patch" not in names and "diff_vs_template.patch" not in names
    assert "no diff against it is included" in (pkg / "README.md").read_text(encoding="utf-8")
    for p in pkg.rglob("*"):
        if p.is_file():
            assert marker not in p.read_text(errors="ignore")


def _tree_hypotheses(st):
    st.hypotheses = [{"iteration": i, "against": a, "title": t, "description": f"{t} desc",
                      "strategy_tag": "hybrid", "outcome": o, "chain": c}
                     for i, a, t, o, c in [(1, 0, "Root step", "improved", 0),
                                           (2, 0, "Other chain", "improved", 1),
                                           (3, 1, "Best step", "improved", 0)]]
    st.nodes = [_node(1, ROOT, "lp", "Root step", "improved", 0.01, True),
                _node(2, ROOT, "dp", "Other chain", "improved", 0.005, True),
                _node(3, 1, "lp", "Best step", "improved", 0.03, True, best=True)]
    st.nodes[1].chain = 1
    st.chains = [Chain(index=0, current=3, best=3, best_score=0.03),
                 Chain(index=1, current=2, best=2, best_score=0.005)]
    st.best = replace(st.best, iteration=3)


def test_hypothesis_log_names_parent_nodes_in_a_tree(tmp_path):
    # mutation: keeping "vs best #N" for a template start, anneal or several chains, or
    # calling node 0 the baseline in a template start
    spec, st, store = make(tmp_path)
    _tree_hypotheses(st)
    for kw, root in [({"start": "template"}, "vs the empty template"),
                     ({"search": "anneal"}, "vs the baseline"), ({"chains": 2}, "vs the baseline")]:
        text = (build_package(replace(spec, **kw), st, store) / "hypotheses.md").read_text(
            encoding="utf-8")
        assert "vs best #" not in text
        assert f"- #1 ({root}) [hybrid] Root step" in text
        assert "- #3 (vs node #1) [hybrid] Best step" in text


def test_evidence_draft_lists_only_the_best_path_in_a_tree(tmp_path):
    # mutation: listing every improved hypothesis from every chain (today's rule) puts the
    # other chain's method into the method description of the best node
    spec, st, store = make(tmp_path)
    _tree_hypotheses(st)
    text = evidence_draft(replace(spec, chains=2), st)
    assert "- Root step: Root step desc\n- Best step: Best step desc" in text
    assert "Other chain" not in text


def test_readme_reports_budget_extensions(tmp_path):
    # mutation: omitting the section hides that the job ran on more budget than job.json says
    spec, st, store = make(tmp_path, status="exhausted")  # budget usd=1.0
    st.budget_extensions = [{"at": 0.0, "iteration": 3, "previous_status": "exhausted",
                             "add": {"iterations": None, "hours": None, "usd": 0.5,
                                     "compute_usd": None}}]
    text = (build_package(spec, st, store) / "README.md").read_text(encoding="utf-8")
    assert "- Started with: usd 1.0\n" in text
    assert "- Extended at iteration 3 (was exhausted): +0.5 usd\n" in text
    assert "- Now: usd 1.5\n" in text


def test_readme_without_extensions_has_no_budget_section(tmp_path):
    # mutation: always printing the section
    spec, st, store = make(tmp_path, status="exhausted")
    assert "Budget extensions" not in (build_package(spec, st, store) / "README.md").read_text(
        encoding="utf-8")


def test_no_candidate_readme_reports_budget_extensions(tmp_path):
    # mutation: adding the section to the candidate README only
    spec, st, store = make(tmp_path, status="exhausted")
    st.best = None
    st.budget_extensions = [{"at": 0.0, "iteration": 3, "previous_status": "exhausted",
                             "add": {"iterations": None, "hours": None, "usd": 0.5,
                                     "compute_usd": None}}]
    assert "- Now: usd 1.5\n" in (build_package(spec, st, store) / "README.md").read_text(
        encoding="utf-8")


def test_score_pairs_match_scores_sections(tmp_path):
    from talos.package import score_pairs, scores_sections
    spec, st, _ = make(tmp_path)
    assert [h for h, _, _ in score_pairs(spec, st)] == [h for h, _ in scores_sections(spec, st)]
    assert len(score_pairs(spec, st)) == 2  # training and held-out


def test_package_zip_holds_report_with_best(tmp_path):
    spec, st, store = make(tmp_path)
    build_package(spec, st, store)
    assert (tmp_path / "package" / "report.html").is_file()
    assert "report.html" in zipfile.ZipFile(tmp_path / "package.zip").namelist()


def test_package_zip_holds_report_with_no_candidate(tmp_path):
    spec, st, store = make(tmp_path)
    st.best = None
    build_package(spec, st, store)
    assert "report.html" in zipfile.ZipFile(tmp_path / "package.zip").namelist()


def test_package_zip_holds_report_with_best_but_no_baseline(tmp_path):
    spec, st, store = make(tmp_path)
    st.baseline = None
    build_package(spec, st, store)
    assert "report.html" in zipfile.ZipFile(tmp_path / "package.zip").namelist()


def test_report_failure_does_not_stop_package(tmp_path, monkeypatch, capsys):
    spec, st, store = make(tmp_path)
    monkeypatch.setattr("talos.report.render_report", lambda *a, **k: 1 / 0)
    build_package(spec, st, store)
    assert (tmp_path / "package.zip").is_file()
    assert "report.html not written" in capsys.readouterr().err
    assert "report.html" not in zipfile.ZipFile(tmp_path / "package.zip").namelist()


def test_diff_marks_a_last_line_with_no_newline():
    # Issue #86: the '-' and '+' lines ran together ('-b+c') with no end-of-file marker.
    # mutation: dropping the marker, or appending it without the newline before it
    from talos.package import _diff
    assert _diff({"mod.rs": "a\nb"}, {"mod.rs": "a\nc"}) == (
        "--- baseline/mod.rs\n+++ candidate/mod.rs\n@@ -1,2 +1,2 @@\n a\n"
        "-b\n\\ No newline at end of file\n+c\n\\ No newline at end of file\n")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize("base, cand", [("a\nb", "a\nc"),          # changed, neither ends in \n
                                        ("a\nb", "a\nb\n"),        # newline added
                                        ("a\nb\n", "a\nb"),        # newline removed
                                        ("x\na\nb", "y\na\nb"),    # unchanged context last line
                                        ("a\nb\n", "a\nc\n")])     # ordinary change
def test_diff_applies_with_git_and_reproduces_the_candidate(tmp_path, base, cand):
    # mutation: any malformed last line makes git apply refuse the patch or rebuild other bytes
    from talos.package import _diff
    (tmp_path / "mod.rs").write_bytes(base.encode())
    (tmp_path / "p.patch").write_bytes(_diff({"mod.rs": base}, {"mod.rs": cand}).encode())
    # autocrlf off: the Windows runner sets it globally, and git apply would write CRLF
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-p1", "p.patch"],
                       cwd=tmp_path, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "mod.rs").read_bytes() == cand.encode()


T4 = [NonceResult("t", n, True, 100, 1) for n in range(4)]
ALT = [NonceResult("t", n, True, 100 + 2 * (n % 2), 1) for n in range(4)]


def test_scores_markdown_with_a_rule_adds_the_noise_lines():
    # differences [0, 2, 0, 2]: SE sqrt(1/3)% = 0.5774%, ratio 1.7
    md = scores_markdown(T4, ALT, BeatRule())
    assert "mean_rel_se: ±0.5774%  noise: 1.7x (marginal)" in md
    assert "per track: t: ±0.58% (n=4)" in md


def test_scores_markdown_without_a_rule_has_no_noise_lines():
    # mutation: defaulting rule to BeatRule() changes evidence_draft and this table
    assert "noise:" not in scores_markdown(T4, ALT)
    assert "per track:" not in scores_markdown(T4, ALT)


def test_scores_markdown_over_the_ceiling_has_no_noise_lines():
    bad = [NonceResult("t", 0, False, None, 1, "panic")] + ALT[1:]
    md = scores_markdown(T4, bad, BeatRule())
    assert "noise:" not in md and "per track:" not in md and "mean_rel_delta" in md


def test_scores_markdown_names_n1():
    md = scores_markdown(T4[:1], [NonceResult("t", 0, True, 101, 1)], BeatRule())
    assert "noise: n=1" in md and "per track: t: n=1" in md


def test_scores_markdown_zero_se_is_not_n1():
    # mutation: `if t.rel_se:` treats a 0.0 SE as missing and prints "n=1"
    plus = [NonceResult("t", n, True, 101, 1) for n in range(4)]
    md = scores_markdown(T4, plus, BeatRule())
    assert "noise: ±0, no spread" in md and "per track: t: ±0.00% (n=4)" in md
    assert "n=1" not in md


def test_evidence_has_no_noise_but_scores_md_does(tmp_path):
    # spec Tests 12. mutation: build_package not passing with_noise=True, or evidence_draft
    # passing it
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    assert "noise: 21.0x (clear)" in (pkg / "scores.md").read_text(encoding="utf-8")
    text = evidence_draft(spec, st)
    assert "noise:" not in text and "mean_rel_se" not in text


def test_package_builds_from_a_state_without_noise_keys(tmp_path):
    # Review Focus 5: the fixture's best.delta is {"mean_rel_delta": 0.105} only, as in a
    # state.json written before this change
    spec, st, store = make(tmp_path)
    assert "mean_rel_se" not in st.best.delta
    build_package(spec, st, store)


def test_scores_markdown_names_identical_on_every_nonce():
    # mutation: swapping the identical and no_spread words
    md = scores_markdown(T4, list(T4), BeatRule())
    assert "noise: identical on every nonce" in md
    assert "no spread" not in md


GRIDFLOW = "let v = serde_json::to_value(challenge);\n"


def _energy(spec):
    return replace(spec, challenge="energy_arbitrage", dev_image_tag="0.0.8")


def _written(store):
    return {p.relative_to(store.run_dir / "package").as_posix()
            for p in (store.run_dir / "package").rglob("*") if p.is_file()}


def test_a_void_best_is_never_packaged(tmp_path):
    # mutation: writing submission/ hands back code that inflates on a build that still leaks
    spec, st, store = make(tmp_path)
    spec = _energy(spec)
    st.best.files = {"mod.rs": GRIDFLOW}
    pkg = build_package(spec, st, store)
    assert _written(store) <= {"README.md", "report.html"} and "README.md" in _written(store)
    text = (pkg / "README.md").read_text()
    assert "void" in text and "must not be submitted" in text
    with zipfile.ZipFile(store.run_dir / "package.zip") as z:
        assert set(z.namelist()) == {"README.md", "report.html"}
    assert "Void" in (pkg / "report.html").read_text()


@pytest.mark.parametrize("change", [{"dev_image_tag": None},
                                    {"monorepo_ref": "84a5787f5b14a630bdf40f52bccf37887d3d8464"}])
def test_a_score_from_before_the_fix_is_void_however_clean_the_code(tmp_path, change):
    # mutation: checking only the source calls an inflated score from the old build a result
    spec, st, store = make(tmp_path)
    pkg = build_package(replace(_energy(spec), **change), st, store)
    assert not (pkg / "submission").exists() and not (pkg / "scores.md").exists()


def test_a_clean_energy_package_says_what_its_delta_means(tmp_path):
    # mutation: dropping the note invites a comparison with leak-inflated mainnet scores
    spec, st, store = make(tmp_path)
    pkg = build_package(_energy(spec), st, store)
    assert (pkg / "submission" / "mod.rs").is_file()
    for name in ("README.md", "evidence_draft.md"):
        text = (pkg / name).read_text()
        assert "Do not compare these scores with mainnet scores" in text, name
    assert "rebuilt at monorepo `r`" in (pkg / "README.md").read_text()


def test_other_challenges_get_no_leak_note(tmp_path):
    # mutation: an unscoped note warns about a leak knapsack never had
    spec, st, store = make(tmp_path)
    pkg = build_package(spec, st, store)
    assert "hidden_seed" not in (pkg / "README.md").read_text()
