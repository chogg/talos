"""The energy_arbitrage price leak: what the source checks flag and what they must leave alone."""
from types import SimpleNamespace

import pytest

from talos.leak_check import (KNOWN_LEAKING, LeakFinding, describe, hindsight_additions,
                              leak_findings, strip_comments, void_reason)

EA = "energy_arbitrage"
# The shape gridflow, first_energy, super_energy and titan_killer publish.
GRIDFLOW = """\
fn secondary_entropy(challenge: &Challenge) -> Option<[u8; 32]> {
    let value = serde_json::to_value(challenge).ok()?;
    let obj = value.as_object()?;
    None
}
fn expand_price_table(challenge: &Challenge, entropy: [u8; 32]) -> Vec<Vec<f64>> {
    let mut stream = SmallRng::from_seed(StdRng::from_seed(entropy).r#gen());
    vec![challenge.market.generate_rt_prices(&mut stream, 0, &idle)]
}
"""
# What peakshift and the honest algorithms have, each of which a name-based rule would flag.
PEAKSHIFT_STUB = """\
fn exact_price_table(_challenge: &Challenge) -> Option<&'static Vec<Vec<f64>>> {
    None
}
"""
HONEST_SIMULATION = """\
let mut seed = challenge.seed;
let mut rng = SmallRng::from_seed(StdRng::from_seed(challenge.seed).r#gen());
path.push(challenge.market.generate_rt_prices(&mut rng, 0, &vec![false; n]));
state = challenge.take_step(&state, &action, NextRTPrices::Generate(rng.r#gen()))?;
"""
FUEL_READ = """\
extern "C" { static __fuel_remaining: u64; }
fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}
"""
TEMPLATE = """\
pub fn solve_challenge(challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>,
                       hyperparameters: &Option<Map<String, Value>>) -> Result<()> {
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}
"""


def findings(source, path="mod.rs"):
    return leak_findings(EA, {path: source})


def test_gridflow_shape_is_flagged_with_its_line():
    # mutation: dropping the serialise rule leaves only the function names to catch it, and a
    # rename would then pass
    found = findings(GRIDFLOW)
    assert {f.line for f in found if "serialises" in f.why} == {2}
    assert any("published leak" in f.why and f.line == 1 for f in found)
    renamed = GRIDFLOW.replace("secondary_entropy", "a").replace("expand_price_table", "b")
    assert [f.line for f in findings(renamed)] == [2]


@pytest.mark.parametrize("source", [PEAKSHIFT_STUB, HONEST_SIMULATION, FUEL_READ, TEMPLATE,
                                    "let v = prices.to_vec();\nlet s = name.to_string();\n",
                                    "#[derive(Serialize, Deserialize)]\nstruct Hp { a: f64 }\n",
                                    'println!("{:?}", prices);\n'])
def test_honest_code_is_not_flagged(source):
    # mutation: banning generate_rt_prices or from_seed refuses wasserwert and energy_v1; banning
    # unsafe or extern refuses every algorithm that reads the fuel counter; banning by name
    # refuses peakshift's stub
    assert findings(source) == []


@pytest.mark.parametrize("source", [
    "let s = challenge.hidden_seed;\n",
    "use serde_json::to_value;\nlet v = to_value(challenge);\n",
    "let v = to_value(&challenge);\n",
    "let b = bincode::serialize(challenge);\n",
    "let v = challenge.serialize(ser);\n",
    'let t = format!("{:?}", challenge);\n',
    'let t = format!("{challenge:#?}");\n',
    "dbg!(challenge);\n",
    "let b: [u8; 8] = unsafe { std::mem::transmute(x) };\n",
    "let p = challenge as *const Challenge as *const u8;\n",
    "let p = core::ptr::addr_of!(*challenge);\n",
    "let a: Vec<String> = std::env::args().collect();\n",
    "use std::{env, fmt};\n",
    'let c = read("/proc/self/cmdline");\n',
    "let s = std::fs::read(path);\n",
    'extern "C" { fn getenv(k: *const u8) -> *const u8; }\n',
    'extern "C" { static __fuel_remaining: u64; fn getenv(k: i32) -> i32; }\n',
], ids=lambda s: s.split("\n")[0][:40])
def test_each_route_to_the_seed_is_flagged(source):
    # mutation: dropping any one rule fails its case
    assert findings(source)


def test_comments_are_not_code_but_strings_are():
    # mutation: scanning raw text fails an LLM's comment saying it does not read the seed; dropping
    # comments too eagerly hides a quoted path
    assert findings("// never read hidden_seed\nlet a = 1;\n") == []
    assert findings("/* hidden_seed /* nested hidden_seed */ still */\nlet a = 1;\n") == []
    assert findings('let c = \'"\'; // hidden_seed\n') == []
    assert findings('let s = "//"; let h = hidden_seed;\n')
    assert findings('let s = r#"/proc/self/cmdline"#;\n')


def test_comment_stripping_keeps_line_numbers():
    # mutation: dropping a comment's newlines moves every finding above the real line
    source = "/* one\ntwo */\n// three\nlet h = hidden_seed;\n"
    assert strip_comments(source).count("\n") == source.count("\n")
    assert [f.line for f in findings(source)] == [4]


def test_only_rust_is_scanned_and_only_this_challenge():
    # mutation: reading README.md flags a mainnet author's prose; applying the rules everywhere
    # fetches and refuses other challenges' algorithms
    assert leak_findings(EA, {"README.md": "reads hidden_seed"}) == []
    assert leak_findings("knapsack", {"mod.rs": GRIDFLOW}) == []
    assert KNOWN_LEAKING[EA] == {"first_energy", "super_energy", "titan_killer", "gridflow"}


@pytest.mark.parametrize("line", [
    "solution.schedule = rewritten;",
    "solution.schedule[t][b] = x;",
    "solution.schedule[t][b] += x;",
    "polish(&mut solution.schedule);",
    "solution.schedule.push(a);",
    "save_solution(&Solution { schedule: best })?;",
    "save_solution(&Solution { schedule })?;",
    "let mut best = solution.schedule;",
    "let mut best = solution.schedule.clone();",
])
def test_a_new_schedule_write_is_hindsight(line):
    # mutation: dropping a form lets that rewrite through
    assert hindsight_additions(EA, {"mod.rs": TEMPLATE}, {"mod.rs": TEMPLATE + line + "\n"})


def test_hindsight_is_measured_against_the_parent():
    # mutation: counting all writes refuses every candidate of peakshift, whose root has them;
    # comparing per file calls a moved line new
    site = "save_solution(&Solution { schedule: best })?;\n"
    assert hindsight_additions(EA, {"mod.rs": site}, {"mod.rs": site + "let a = 1;\n"}) == []
    assert hindsight_additions(EA, {"a.rs": site}, {"b.rs": site}) == []
    added = hindsight_additions(EA, {"mod.rs": site}, {"mod.rs": site + site})
    assert [(f.path, f.line) for f in added] == [("mod.rs", 2)]


def test_hindsight_allows_the_rollout_and_a_zero_fallback():
    # mutation: treating a read as a write, or the zero fallback as a rewrite, refuses
    # gridflow-style safety code
    allowed = ("let zs = Solution { schedule: vec![vec![0.0; n]; m] };\n"
               "if solution.schedule == other { return Ok(()); }\n"
               "let n = solution.schedule.len();\n")
    assert hindsight_additions(EA, {"mod.rs": TEMPLATE}, {"mod.rs": TEMPLATE + allowed}) == []
    assert hindsight_additions("knapsack", {}, {"mod.rs": "solution.schedule = x;\n"}) == []


def test_describe_caps_what_a_finding_says():
    # mutation: no cap puts every one of gridflow's 25 sites in an error line
    many = [LeakFinding("a.rs", i, "why", "code") for i in range(5)]
    assert describe(many) == ("a.rs:0: why: code; a.rs:1: why: code; a.rs:2: why: code; "
                              "and 2 more")


def _spec(**kw):
    base = dict(challenge=EA, dev_image_tag="0.0.8", monorepo_ref="839c7184",
                baseline_algorithm={"name": "peakshift"})
    return SimpleNamespace(**{**base, **kw})


def _state(baseline="fn a() {}\n", best=None):
    return SimpleNamespace(baseline=SimpleNamespace(files={"mod.rs": baseline}),
                           best=SimpleNamespace(files={"mod.rs": best}) if best else None)


def test_a_clean_job_is_not_void():
    assert void_reason(_spec(), _state(best="let a = 1;\n")) is None
    assert void_reason(_spec(), SimpleNamespace(baseline=None, best=None)) is None


@pytest.mark.parametrize("spec_kw", [
    {"dev_image_tag": None},  # every job older than the image pin
    {"monorepo_ref": "84a5787f5b14a630bdf40f52bccf37887d3d8464"},  # the last pin without the fix
    {"baseline_algorithm": {"name": "gridflow"}},  # chosen, not measured yet
])
def test_a_build_or_a_baseline_from_before_the_fix_is_void(spec_kw):
    # mutation: dropping either build check calls a score from the leaking build a result
    assert void_reason(_spec(**spec_kw), _state())


@pytest.mark.parametrize("state, root", [
    (_state(baseline=GRIDFLOW), None),
    (_state(best=GRIDFLOW), None),
    (_state(), {"mod.rs": GRIDFLOW}),
], ids=["baseline", "best", "root"])
def test_leak_code_in_any_of_the_job_files_is_void(state, root):
    # mutation: checking only the best misses a clean candidate scored against gridflow
    assert "hidden_seed" in void_reason(_spec(), state, root)


def test_other_challenges_are_never_void():
    # mutation: applying this to every challenge voids all pre-pin knapsack jobs
    assert void_reason(_spec(challenge="knapsack", dev_image_tag=None), _state(GRIDFLOW)) is None
