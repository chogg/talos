import json
import types

import pytest

from talos.baseline import BaselineError, cache_key, resolve_baseline
from talos.bench import FakeBench
from talos.challenges import (CHALLENGES, BeatRule, MONOREPO_REF, hardware_class, host_slug,
                              local_hardware_class, local_workers)
from talos.types import NonceSet

TR = [NonceSet("t", "ab" * 32, 0, 2)]
HO = [NonceSet("t", "ab" * 32, 1_000_000, 2)]


def fake_mainnet(top=("algo_x", "algo_x_id", 55)):
    return types.SimpleNamespace(
        top_algorithm=lambda ch, **kw: top,
        fetch_algorithm_files=lambda ch, name, **kw: {"mod.rs": f"// {name}"},
        fetch_template=lambda ch, **kw: "pub fn solve_challenge(",
    )


def test_resolve_compiles_and_scores_both_sets(tmp_path):
    # mutation: swapping training/holdout order, or scoring only one set, or dropping adoption
    fb = FakeBench(lambda ch, files, ns: [10 + n for n in ns.nonces()])
    rec, template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                                     rule=BeatRule(), mainnet=fake_mainnet())
    assert rec.name == "algo_x" and rec.adoption == 55 and rec.artifact_id
    assert [r.quality for r in rec.training] == [10, 11]
    assert [r.nonce for r in rec.holdout] == [1_000_000, 1_000_001]
    assert "solve_challenge" in template
    assert len(fb.calls) == 1 and fb.holdout_runs == 1


def test_baseline_log_prints_adoption_as_a_percentage(tmp_path):
    # mainnet adoption is a fixed-point fraction, 10**18 = 100%; the record keeps the raw integer
    # mutation: logging the raw 18-digit integer, or dividing by the wrong power of ten
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    lines = []
    rec, _ = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                              rule=BeatRule(), log=lines.append,
                              mainnet=fake_mainnet(("knap_lean", "k1", 837530541350171022)))
    assert lines == ["baseline knap_lean (adoption 83.75%): compiling and scoring 1 file(s)"]
    assert rec.adoption == 837530541350171022


def test_cache_hit_skips_bench(tmp_path):
    # mutation: ignoring the cache re-measures and charges the user twice
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                     rule=BeatRule(), mainnet=fake_mainnet())
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                     rule=BeatRule(), mainnet=fake_mainnet())
    assert len(fb.calls) == 1


def test_image_upgrade_remeasures_cached_baseline(tmp_path, monkeypatch):
    # An image-only update must never compare new candidates with old runtime measurements.
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    for tag in ("0.0.7", "0.0.8", "0.0.8"):
        monkeypatch.setattr("talos.baseline.DEV_IMAGE_TAG", tag)
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=fake_mainnet())
    assert len(fb.calls) == 2
    assert fb.holdout_runs == 2


def test_cache_key_changes_with_fuel_and_hardware():
    # mutation: cache_key ignoring fuel or hardware_class would collide keys across
    # different measurement conditions and serve a stale/mismatched cached baseline
    a = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 6, "cpu4-mem8192")
    assert a != cache_key("knapsack", "ref", "algo", TR, HO, 5, "gpu-L40S")


def test_no_top_algorithm_is_an_error(tmp_path):
    # mutation: proceeding with top=None would crash later with an unhelpful traceback
    # instead of a clear BaselineError naming the missing mainnet algorithm
    fb = FakeBench(lambda ch, files, ns: [1])
    with pytest.raises(BaselineError):
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=fake_mainnet(top=None))


def test_baseline_compile_failure_is_an_error_with_output(tmp_path):
    # mutation: swallowing the compile error or omitting the raw compiler output would
    # violate spec §9, which requires the raw output on a baseline compile failure
    fb = FakeBench(lambda ch, files, ns: [1], compile_ok=lambda f: False)
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=fake_mainnet())
    assert "E0308" in str(ei.value)


def test_corrupt_cache_is_remeasured(tmp_path):
    # mutation: letting JSONDecodeError escape aborts the job on a half-written cache file
    key = cache_key("knapsack", MONOREPO_REF, "algo_x", TR, HO, 5, "cpu4-mem8192")
    cache_file = tmp_path / "knapsack" / f"{key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{not json")

    fb = FakeBench(lambda ch, files, ns: [10 + n for n in ns.nonces()])
    rec, _template = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                                      rule=BeatRule(), mainnet=fake_mainnet())
    assert len(fb.calls) == 1 and fb.holdout_runs == 1
    assert [r.quality for r in rec.training] == [10, 11]
    assert json.loads(cache_file.read_text())["name"] == "algo_x"


def test_all_error_baseline_is_refused_and_not_cached(tmp_path):
    # mutation: accepting an all-error baseline burns the budget on unscoreable iterations --
    # every candidate is then compared against nothing and fails at scoring until the cap bites
    fb = FakeBench(lambda ch, files, ns: [None for _ in ns.nonces()])
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=fake_mainnet())
    msg = str(ei.value)
    assert "knapsack" in msg and "fuel 5" in msg and "'t'" in msg and "no_solution" in msg
    assert not list(tmp_path.rglob("*.json"))  # an unusable baseline is never cached


def test_zero_quality_baseline_is_refused(tmp_path):
    # mutation: checking only `any ok` accepts a baseline that scores 0 everywhere, against
    # which every relative delta is a division by zero
    fb = FakeBench(lambda ch, files, ns: [0 for _ in ns.nonces()])
    with pytest.raises(BaselineError):
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=fake_mainnet())


def test_partly_erroring_baseline_is_accepted(tmp_path):
    # mutation: requiring every nonce to score would reject a usable baseline; TIG algorithms
    # legitimately fail some nonces
    fb = FakeBench(lambda ch, files, ns: [None, 10])
    rec, _ = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                              rule=BeatRule(), mainnet=fake_mainnet())
    assert [r.ok for r in rec.training] == [False, True]


def test_baseline_forces_the_holdout_run(tmp_path):
    # mutation: passing the baseline's own training results as baseline_training makes the
    # held-out run conditional on beating itself, which it never does, so holdout comes back None
    fb = FakeBench(lambda ch, files, ns: [10 for _ in ns.nonces()])
    rec, _ = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                              rule=BeatRule(), mainnet=fake_mainnet())
    assert fb.calls[0].baseline_training is None and len(rec.holdout) == 2


def test_cache_key_without_hyperparameters_omits_them_from_the_payload(monkeypatch):
    # Fix the image here so future pin bumps do not require another golden value.
    monkeypatch.setattr("talos.baseline.DEV_IMAGE_TAG", "0.0.8")
    # Baselines cached without the image (10f760b979fa855a984c6498 for these inputs) miss.
    # mutation: always putting "hyperparameters": None in the hashed payload changes this key
    key = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192")
    assert key == "4d81b60c4370ff76982312fb"


def test_cache_key_follows_what_the_hyperparameters_change_at_runtime():
    k = lambda hp: cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192", hp)  # noqa: E731
    plain = k(None)
    # tracks mapped to None run exactly as tracks without an entry, and as no map at all
    assert plain == k({}) == k({"t": None})
    # mutation: ignoring the map serves a baseline measured without it to a job that uses it
    assert k({"t": {"x": 1}}) not in (plain, k({"t": {"x": 2}}))
    # mutation: `if v` instead of `is not None` drops {} and collides it with running flagless
    assert k({"t": {}}) != plain


def test_a_pinned_algorithm_is_measured_instead_of_asking_mainnet(tmp_path):
    def no_top(ch, **kw):
        raise AssertionError("top_algorithm must not be called for a pinned algorithm")
    mn = fake_mainnet()
    mn.top_algorithm = no_top
    fb = FakeBench(lambda ch, files, ns: [10 for _ in ns.nonces()])
    hp = {"t": {"x": 1}}
    rec, _ = resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                              mainnet=mn, algorithm={"name": "pinned", "id": "p1", "adoption": 7},
                              hyperparameters=hp)
    # mutation: calling top_algorithm again measures whatever tops mainnet now, not the algorithm
    # the frozen map belongs to
    assert (rec.name, rec.adoption, rec.files) == ("pinned", 7, {"mod.rs": "// pinned"})
    # mutation: not passing the map measures the baseline without it while candidates use it
    assert fb.calls[0].hyperparameters == hp


def test_an_unscoreable_baseline_with_hyperparameters_suggests_running_without(tmp_path):
    fb = FakeBench(lambda ch, files, ns: [None for _ in ns.nonces()])
    with pytest.raises(BaselineError) as ei:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                         mainnet=fake_mainnet(), hyperparameters={"t": {"x": 1}})
    # mutation: dropping the hint leaves the user guessing whether the map broke the algorithm
    assert "--hyperparameters none" in str(ei.value)
    with pytest.raises(BaselineError) as plain:
        resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                         mainnet=fake_mainnet(), hyperparameters={"t": None})
    # mutation: hinting whenever the argument is not None blames a map that passed nothing
    assert "--hyperparameters none" not in str(plain.value)


def test_hardware_class_separates_cpu_memory_and_gpu():
    # mutation: f"cpu{cpu}" alone lets a memory change reuse a cached baseline measured with a
    # different memory (and a different price per second)
    import dataclasses
    knapsack = CHALLENGES["knapsack"]
    assert hardware_class(knapsack) == f"cpu{knapsack.cpu}-mem{knapsack.memory_mib}-x{knapsack.cpu}"
    bigger = dataclasses.replace(knapsack, memory_mib=knapsack.memory_mib * 2)
    assert hardware_class(bigger) != hardware_class(knapsack)
    assert hardware_class(CHALLENGES["hypergraph"], "L40S") == "gpu-L40S"


def test_modal_workers_fill_the_cpu_container_and_the_class_carries_the_packing():
    from talos.challenges import modal_workers
    knapsack, hypergraph = CHALLENGES["knapsack"], CHALLENGES["hypergraph"]
    # mutation: one worker per 4-core container idles three billed cores per nonce
    assert modal_workers(knapsack) == knapsack.cpu == 4
    # mutation: 4 workers on one GPU serialise on the device and time the batch out
    assert modal_workers(hypergraph) == 1
    # mutation: a class string without the packing lets a baseline measured one nonce per
    # container (before batching) serve candidates scored four to a container
    assert hardware_class(knapsack).endswith("-x4")
    assert hardware_class(hypergraph, "L40S") == "gpu-L40S"  # GPU packing is unchanged


def test_host_slug_is_stable_and_never_empty():
    # mutation: keeping case or dots lets "Box.local" and "box-local" become two cache keys
    assert host_slug("Box.local") == "box-local" == host_slug("box-local")
    assert host_slug("  ") == "host" and host_slug("") == "host"
    assert host_slug("a__b--c") == "a-b-c"


def test_local_hardware_class_separates_limits_host_and_gpu():
    knapsack, hypergraph = CHALLENGES["knapsack"], CHALLENGES["hypergraph"]
    assert local_hardware_class(knapsack, 8, 12, None, "Box") == "local-box-cpu8-mem12"
    # mutation: dropping cpus or memory lets a baseline measured under other limits serve a run
    assert local_hardware_class(knapsack, 4, 12, None, "box") != "local-box-cpu8-mem12"
    assert local_hardware_class(knapsack, 8, 8, None, "box") != "local-box-cpu8-mem12"
    gpu = local_hardware_class(hypergraph, 8, 12, "NVIDIA L40S", "box")
    assert gpu == "local-box-gpu-nvidia-l40s-cpu8-mem12"
    # mutation: a GPU class without the limits (review bot, 2026-09-24) serves a baseline
    # measured under other limits; the build's instrumentation pass runs on the CPUs either way
    assert local_hardware_class(hypergraph, 4, 12, "NVIDIA L40S", "box") != gpu
    assert local_hardware_class(hypergraph, 8, 8, "NVIDIA L40S", "box") != gpu
    assert local_hardware_class(hypergraph, 8, 12, "NVIDIA A100", "box") != gpu
    with pytest.raises(ValueError):
        local_hardware_class(hypergraph, 8, 12, None, "box")
    # mutation: a local class that does not start with "local-" could equal a Modal or C3 one
    for cls in (local_hardware_class(knapsack, 4, 8, None, "x"),
                local_hardware_class(hypergraph, 1, 8, "L40S", "x")):
        assert cls.startswith("local-") and cls != hardware_class(knapsack)


def test_local_workers_use_every_cpu_but_one_gpu():
    # mutation: 8 workers on one GPU serialise on the device and time the job out
    assert local_workers(CHALLENGES["knapsack"], 8) == 8
    assert local_workers(CHALLENGES["hypergraph"], 8) == 1


def test_cache_key_covers_the_crate_layout(monkeypatch):
    # A baseline measured with the candidate built beside every shipped algorithm is not the
    # baseline of the pruned crate. mutation: leaving the layout out of the key keeps every
    # pre-pruning baseline a cache hit
    from talos import inside
    before = cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192")
    monkeypatch.setattr(inside, "CRATE_LAYOUT", "other-layout")
    assert cache_key("knapsack", "ref", "algo", TR, HO, 5, "cpu4-mem8192") != before


LEAK = "let v = serde_json::to_value(challenge);\n"


def _leaky_mainnet(name="copycat"):
    return types.SimpleNamespace(top_algorithm=lambda ch, **kw: (name, "id", 1),
                                 fetch_algorithm_files=lambda ch, name, **kw: {"mod.rs": LEAK},
                                 fetch_template=lambda ch, **kw: "pub fn solve_challenge(")


def test_a_baseline_fetched_with_the_leak_stops_before_any_compute(tmp_path):
    # mutation: checking only at selection misses a branch that moved after job start
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    with pytest.raises(BaselineError, match="hidden_seed"):
        resolve_baseline("energy_arbitrage", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=_leaky_mainnet())
    assert fb.calls == []


def test_a_cached_baseline_with_the_leak_is_refused(tmp_path):
    # mutation: returning a cache hit unread trusts a measurement of leaking code
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    resolve_baseline("knapsack", TR, HO, 5, fb, tmp_path, "cpu4-mem8192", rule=BeatRule(),
                     mainnet=_leaky_mainnet())  # the same files, measured where no rule applies
    src = next((tmp_path / "knapsack").glob("*.json"))
    key = cache_key("energy_arbitrage", MONOREPO_REF, "copycat", TR, HO, 5, "cpu4-mem8192")
    (tmp_path / "energy_arbitrage").mkdir()
    (tmp_path / "energy_arbitrage" / f"{key}.json").write_text(src.read_text())
    with pytest.raises(BaselineError, match="hidden_seed"):
        resolve_baseline("energy_arbitrage", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=_leaky_mainnet())


def test_a_job_pinned_to_a_known_leaking_baseline_is_refused_by_name(tmp_path):
    # mutation: only scanning would still fetch it, and a future stubbed copy on that branch would
    # pass under a name whose mainnet scores are known to be inflated
    fb = FakeBench(lambda ch, files, ns: [1 for _ in ns.nonces()])
    mn = fake_mainnet()
    with pytest.raises(BaselineError, match="gridflow"):
        resolve_baseline("energy_arbitrage", TR, HO, 5, fb, tmp_path, "cpu4-mem8192",
                         rule=BeatRule(), mainnet=mn,
                         algorithm={"name": "gridflow", "id": "x", "adoption": 1})
    assert fb.calls == []
