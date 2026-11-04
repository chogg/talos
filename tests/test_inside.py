import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from talos import inside


def make_monorepo(tmp_path: Path) -> Path:
    mono = tmp_path / "mono"
    (mono / "tig-algorithms" / "src" / "knapsack").mkdir(parents=True)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text("// c003_a001\n")
    return mono


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_stage_writes_files_and_registers_module_once(tmp_path):
    mono = make_monorepo(tmp_path)
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn x(){}", "ls.rs": "fn y(){}"}, "talos_cand")
    d = mono / "tig-algorithms" / "src" / "knapsack" / "talos_cand"
    assert (d / "mod.rs").read_text() == "fn x(){}" and (d / "ls.rs").exists()
    modrs = (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text()
    assert modrs.count("pub mod talos_cand;") == 1
    # mutation: unstage that leaves the line behind breaks the next compile
    inside.unstage_algorithm(mono, "knapsack", "talos_cand")
    assert not d.exists()
    assert (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text() == "// c003_a001\n"


@pytest.mark.parametrize("rel", ["../evil.rs", "", ".", "sub/../../evil.rs", "/etc/evil.rs"])
def test_stage_rejects_path_escape(tmp_path, rel):
    mono = make_monorepo(tmp_path)
    # mutation: a containment check that only compares resolved prefixes lets "", "." and
    # "sub/../../evil.rs" through, and they then raise IsADirectoryError deep inside the
    # container instead of reporting a failed compile
    with pytest.raises(ValueError):
        inside.stage_algorithm(mono, "knapsack", {rel: "x"}, "talos_cand")
    assert not list(tmp_path.rglob("evil.rs"))


def test_build_invokes_build_algorithm_and_reports(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw.get("cwd")))
        return Result(0, "ok", "")

    # mutation: running build_algorithm without cwd=monorepo makes every compile fail
    ok, out = inside.build(tmp_path, "knapsack", "talos_cand", run)
    assert ok and calls[0][0] == ["build_algorithm", "talos_cand"] and calls[0][1] == tmp_path


def test_classify():
    # exit codes from tig-runtime/src/main.rs and tig-verifier/src/main.rs at MONOREPO_REF
    # mutation: keying ok off runtime rc instead of verifier rc + quality
    assert inside.classify(87, 0, 500, False) == (True, None)        # out of fuel but solved
    assert inside.classify(87, 1, None, False) == (False, "out_of_fuel")
    assert inside.classify(84, 1, None, False) == (False, "no_solution")  # "Runtime Error" exit
    assert inside.classify(0, 1, None, False) == (False, "invalid")       # verifier rejected it
    assert inside.classify(-11, 1, None, False) == (False, "panic")       # killed by a signal
    assert inside.classify(101, 1, None, False) == (False, "panic")       # rust panic exit code
    assert inside.classify(0, 0, None, True) == (False, "timeout")
    assert inside.classify(0, 0, 7, False) == (True, None)


def test_run_nonce_builds_commands_and_parses_quality(tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            # tig-runtime's --output names a FOLDER and it writes <nonce>.json inside it.
            # mutation: passing a file path there makes tig-runtime mkdir a folder of that name
            out_dir = Path(cmd[cmd.index("--output") + 1])
            assert out_dir.is_dir(), "--output must name an existing folder"
            (out_dir / f"{cmd[3]}.json").write_text('{"solution": "e30="}')
            return Result(0, "", "")
        return Result(0, "quality: 4242\n", "")

    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None, run, tmp_path)
    assert row["ok"] and row["quality"] == 4242 and row["nonce"] == 7 and row["track"] == "n=1"
    rt, ver = seen
    settings = json.loads(rt[1])
    assert settings == {"algorithm_id": "", "challenge_id": "c003", "track_id": "n=1",
                        "block_id": "", "player_id": ""}
    assert rt[2] == "ab" * 32 and rt[3] == "7" and rt[4] == str(Path("/lib/x.so"))
    assert "--fuel" in rt and rt[rt.index("--fuel") + 1] == "10"
    # tig-verifier takes positional SETTINGS RAND_HASH NONCE SOLUTION_FILE and has no subcommand
    # mutation: inserting a "verify_solution" argv[1] makes clap reject every call
    assert ver[:4] == ["tig-verifier", rt[1], "ab" * 32, "7"] and Path(ver[4]).name == "7.json"
    assert len(ver) == 5
    assert "--ptx" not in rt  # CPU challenge


def test_run_nonce_gpu_passes_ptx_and_gpu_to_both_binaries(tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")

    inside.run_nonce("c005", "k=1", "ab" * 32, 3, Path("/a.so"), 10, 600, Path("/a.ptx"), run, tmp_path)
    rt, ver = seen
    for cmd in (rt, ver):  # mutation: dropping --gpu from the verifier call breaks GPU challenges
        assert cmd[cmd.index("--ptx") + 1] == str(Path("/a.ptx")) and cmd[cmd.index("--gpu") + 1] == "0"


def test_run_nonce_classifies_no_solution(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            # tig-runtime writes an empty solution and exits 84 when the algorithm returns Err
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text('{"solution": ""}')
            return Result(84, "", "Runtime Error: no solution")
        return Result(1, "", "Verification error: Invalid solution")

    # mutation: reporting ok on a non-zero verifier rc would let junk solutions score
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run, tmp_path)
    assert not row["ok"] and row["error"] == "no_solution" and row["quality"] is None


def test_verifier_gets_only_the_time_the_runtime_left(tmp_path):
    # The Modal function timeout is NONCE_TIMEOUT_S + 120, so a runtime and a verifier that each
    # get the full per-nonce timeout can exceed it and have the container killed.
    # mutation: passing timeout_s to the verifier makes the pair's worst case 2 x timeout_s
    seen = []
    now = [0.0]

    def run(cmd, **kw):
        seen.append((cmd[0], kw.get("timeout")))
        if cmd[0] == "tig-runtime":
            now[0] += 500.0  # the runtime took 500s of the 600s budget
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
            return Result(0, "", "")
        return Result(0, "quality: 1\n", "")

    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run,
                           tmp_path, clock=lambda: now[0])
    assert seen[0] == ("tig-runtime", 600)
    assert seen[1] == ("tig-verifier", 100)
    assert row["runtime_ms"] == 500_000
    # mutation: `timeout_s - elapsed` without the floor passes 0 or a negative timeout, which
    # subprocess treats as "expire immediately"
    now[0] = 0.0
    seen.clear()

    def slow(cmd, **kw):
        seen.append((cmd[0], kw.get("timeout")))
        if cmd[0] == "tig-runtime":
            now[0] += 599.9
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")

    inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, slow, tmp_path,
                     clock=lambda: now[0])
    assert seen[1] == ("tig-verifier", 1)


def test_run_nonce_handles_a_verifier_timeout(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
            return Result(0, "", "")
        raise subprocess.TimeoutExpired(cmd, 1)

    # mutation: an unhandled verifier timeout raises through Modal and puts the argv
    # (with rand_hash) in the client error
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 1, Path("/x.so"), 10, 600, None, run, tmp_path)
    assert row["error"] == "timeout" and not row["ok"] and row["quality"] is None


def test_build_keeps_the_diagnostics_past_20k_of_other_algorithms_warnings(tmp_path):
    # In run 20260916-095103 every successful build log came back exactly 20000 bytes, and
    # iteration 5's kept 6 of its 8 errors: the raw cap was applied before anything filtered
    # the other algorithms' warnings out. mutation: restoring `out[-20000:]` drops the error
    # and the candidate's own dead-code warning, which sit before the foreign spam
    from talos.diagnostics import dead_new_functions, relevant
    own_error = ("error[E0308]: mismatched types\n"
                 "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:9:17\n\n")
    own_dead = ("warning: function `polish` is never used\n"
                "   --> tig-algorithms/src/knapsack/talos_cand/mod.rs:3:4\n\n")
    foreign = ("warning: unused variable: `snap`\n"
               "    --> tig-algorithms/src/knapsack/superfast_knap_v1/track5.rs:2069:17\n"
               "     |\n2069 |             let snap = state.clone_solution();\n"
               "     |                 ^^^^ help: prefix it with an underscore\n\n")
    stderr = own_error + own_dead + foreign * 100
    assert len(stderr) > 20000
    _, out = inside.build(tmp_path, "knapsack", "talos_cand",
                          lambda cmd, **kw: Result(1, "", stderr))
    assert own_error in relevant(out)
    assert dead_new_functions(out, {"mod.rs": ["solve"]}) == ["mod.rs: polish"]


def _capture_run(seen):
    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text("{}")
        return Result(0, "quality: 1\n", "")
    return run


@pytest.mark.parametrize("hp, flag", [
    ({"b": [2, 3], "a": 1}, '{"b":[2,3],"a":1}'),
    ({}, "{}"),
])
def test_run_nonce_passes_hyperparameters_to_the_runtime_only(tmp_path, hp, flag):
    seen = []
    inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None,
                     _capture_run(seen), tmp_path, hyperparameters=hp)
    rt, ver = seen
    # mutation: `if hyperparameters:` skips the flag for {}, which the benchmark ran with
    # mutation: json.dumps without compact separators still parses, but pins a different argv
    assert rt[rt.index("--hyperparameters") + 1] == flag
    # mutation: appending the flag to the verifier call makes clap reject every verification
    assert "--hyperparameters" not in ver


def test_run_nonce_without_hyperparameters_passes_no_flag(tmp_path):
    seen = []
    inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/lib/x.so"), 10, 600, None,
                     _capture_run(seen), tmp_path)
    # mutation: always passing the flag sends "null", which tig-runtime rejects as not an object
    assert all("--hyperparameters" not in cmd for cmd in seen)


class FakePool:
    """Stands in for `multiprocessing.Pool`: records the worker count and yields the rows back
    to front, so a caller that needs nonce order has to sort."""
    sizes: list[int] = []

    def __init__(self, workers):
        FakePool.sizes.append(workers)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, fn, tasks):
        return [fn(t) for t in reversed(list(tasks))]


def _task(nonce, workdir, so="/lib/x.so", timeout_s=600, hp=None):
    return ("c003", "t", "ab" * 32, nonce, so, 10, timeout_s, None, str(workdir), hp,
            "metered", None)


def test_run_nonces_spreads_the_tasks_over_a_pool_of_the_given_workers(tmp_path, monkeypatch):
    seen = []

    def stub(task):
        seen.append(task[3])
        return {"track": task[1], "nonce": task[3], "ok": True, "quality": 1, "runtime_ms": 1,
                "error": None}
    monkeypatch.setattr(inside, "run_task", stub)
    FakePool.sizes = []
    rows = list(inside.run_nonces([_task(n, tmp_path) for n in range(6)], 4,
                                  pool_factory=FakePool))
    # mutation: a pool sized to the task count, or to os.cpu_count(), oversubscribes the
    # container's CPU quota; a pool of 1 idles the other cores
    assert FakePool.sizes == [4]
    assert sorted(seen) == list(range(6)) and [r["nonce"] for r in rows] == list(range(5, -1, -1))


def test_run_nonces_with_one_worker_runs_each_task_on_the_injected_runner(tmp_path):
    seen = []
    FakePool.sizes = []
    rows = list(inside.run_nonces([_task(n, tmp_path, hp={"x": n}) for n in range(3)], 1,
                                  run=_capture_run(seen), pool_factory=FakePool))
    # mutation: building a pool for one worker forks a process per nonce for nothing
    assert FakePool.sizes == []
    runtimes = [cmd for cmd in seen if cmd[0] == "tig-runtime"]
    assert [int(cmd[3]) for cmd in runtimes] == [0, 1, 2]
    # mutation: the serial branch dropping a tuple field (here the hyperparameters) runs the
    # nonce with different settings from the pool branch
    assert [cmd[cmd.index("--hyperparameters") + 1] for cmd in runtimes] == [
        '{"x":0}', '{"x":1}', '{"x":2}']
    assert all(r["ok"] and r["quality"] == 1 for r in rows)


def test_run_nonces_never_pools_an_injected_runner_without_an_injected_pool(tmp_path):
    """`run_task` builds its own subprocess calls, so a real pool would ignore a test's fake
    runner and exec a real tig-runtime. Workers above one only take the pool path with the
    real subprocess or a pool the caller supplied."""
    seen = []
    rows = list(inside.run_nonces([_task(n, tmp_path) for n in range(2)], 4,
                                  run=_capture_run(seen)))
    assert len(rows) == 2 and len([c for c in seen if c[0] == "tig-runtime"]) == 2
def test_artifact_paths_follow_the_container_architecture(tmp_path):
    """The TIG build writes lib/<challenge>/<arch>/, amd64 on x86 and arm64 on
    aarch64 hosts; the path must follow the machine the build ran on."""
    from talos import inside

    lib = tmp_path / "tig-algorithms" / "lib" / "knapsack"
    (lib / "arm64").mkdir(parents=True)
    (lib / "amd64").mkdir()
    so, ptx = inside.artifact_paths(tmp_path, "knapsack", "talos_cand", machine="aarch64")
    assert so == lib / "arm64" / "talos_cand.so"
    so, _ = inside.artifact_paths(tmp_path, "knapsack", "talos_cand", machine="x86_64")
    assert so == lib / "amd64" / "talos_cand.so"
    assert ptx is None


# The pinned mod.rs of a challenge crate: one `pub mod` and one algorithm-id alias per shipped
# algorithm. Talos builds the candidate alone, the way TIG's own CI builds an algorithm from
# its branch, whose mod.rs lists only that algorithm.
PINNED_MOD_RS = ("// c003_a001\npub mod knapsplatt;\npub use knapsplatt as c003_a001;\n"
                 "// c003_a002\npub mod knap_supreme;\npub use knap_supreme as c003_a002;\n")


def make_pinned_monorepo(tmp_path: Path) -> Path:
    mono = make_monorepo(tmp_path)
    (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").write_text(PINNED_MOD_RS)
    return mono


def crate_items(mono: Path) -> list[str]:
    text = (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text()
    return [ln for ln in text.splitlines() if ln and not ln.startswith("//")]


def test_stage_prunes_the_crate_to_the_candidate(tmp_path):
    # mutation: a stage that appends its line keeps every shipped algorithm in the build; for
    # job_scheduling that is 14 algorithms, 345k lines, over 2 h and 23 GB of rustc on one
    # codegen unit (MEASURED 2026-09-25), against 629 s and 3.4 GB for the candidate alone
    mono = make_pinned_monorepo(tmp_path)
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn x(){}"}, "talos_cand")
    assert crate_items(mono) == ["pub mod talos_cand;"]
    pristine = mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs.talos-pristine"
    assert pristine.read_text() == PINNED_MOD_RS


def test_unstage_restores_the_pinned_mod_rs_byte_for_byte(tmp_path):
    # mutation: an unstage that only removes its own line leaves the crate pruned for good,
    # aliases included, so the warm-up and every later job see a checkout that is not the pin
    mono = make_pinned_monorepo(tmp_path)
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn x(){}"}, "talos_cand")
    inside.unstage_algorithm(mono, "knapsack", "talos_cand")
    mod_rs = mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs"
    assert mod_rs.read_text() == PINNED_MOD_RS
    assert not (mod_rs.parent / "mod.rs.talos-pristine").exists()


def test_restage_after_a_crashed_job_keeps_the_pristine_copy(tmp_path):
    # The local backend's /app volume outlives a job that died before unstaging.
    # mutation: saving the pristine copy unconditionally captures the pruned file, and the next
    # unstage "restores" a crate with the shipped algorithms gone
    mono = make_pinned_monorepo(tmp_path)
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn x(){}"}, "talos_cand")
    inside.stage_algorithm(mono, "knapsack", {"mod.rs": "fn y(){}"}, "talos_cand")
    pristine = mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs.talos-pristine"
    assert pristine.read_text() == PINNED_MOD_RS
    assert crate_items(mono) == ["pub mod talos_cand;"]
    assert (mono / "tig-algorithms" / "src" / "knapsack" / "talos_cand" / "mod.rs").read_text() == "fn y(){}"
    inside.unstage_algorithm(mono, "knapsack", "talos_cand")
    assert (mono / "tig-algorithms" / "src" / "knapsack" / "mod.rs").read_text() == PINNED_MOD_RS


def test_content_hash_covers_the_crate_layout(monkeypatch):
    # A .so built in the pruned crate is not the .so built beside 13 other algorithms.
    # mutation: leaving CRATE_LAYOUT out of the hash keeps every pre-pruning artifact a cache hit
    files = {"mod.rs": "fn x(){}"}
    before = inside.content_hash(files, "ref", "tag")
    monkeypatch.setattr(inside, "CRATE_LAYOUT", "other-layout")
    assert inside.content_hash(files, "ref", "tag") != before


def test_metered_content_hash_is_unchanged_and_native_differs():
    files = {"mod.rs": "fn x(){}"}
    # recomputed independently from the pre-change definition, not by calling the function
    h = hashlib.sha256()
    for part in ("r", "t", inside.CRATE_LAYOUT, "mod.rs", "fn x(){}"):
        h.update(part.encode())
        h.update(b"\0")
    # mutation: hashing the mode for metered too changes every cached artifact id and every
    # C3 request_hash, so a resumed job orphans its still-billing job
    assert inside.content_hash(files, "r", "t") == h.hexdigest()[:32]
    assert inside.content_hash(files, "r", "t", "metered") == h.hexdigest()[:32]
    # mutation: a native artifact served for a metered request
    assert inside.content_hash(files, "r", "t", "native") != h.hexdigest()[:32]


@pytest.mark.parametrize("attr", ["CARGO_TOML", "_COMMON", "_CPU_RUN", "_GPU_RUN", "_CUDARC",
                                  "TOOLCHAIN"])
def test_native_content_hash_covers_the_runner_and_metered_does_not(monkeypatch, attr):
    # The Modal volume outlives a deploy and _compile_impl serves any artifact it finds, so a
    # runner change has to change the native id.
    # mutation: hashing only `mode=native` keeps serving a binary built from the old runner
    from talos import native_runner
    files = {"mod.rs": "fn x(){}"}
    native, metered = (inside.content_hash(files, "r", "t", "native"),
                       inside.content_hash(files, "r", "t"))
    monkeypatch.setattr(native_runner, attr, getattr(native_runner, attr) + " ")
    assert inside.content_hash(files, "r", "t", "native") != native
    assert inside.content_hash(files, "r", "t") == metered


def test_native_content_hash_covers_the_native_build_env(monkeypatch):
    # mutation: a profile override change (lto, codegen-units) keeps the old binary
    files = {"mod.rs": "fn x(){}"}
    native = inside.content_hash(files, "r", "t", "native")
    monkeypatch.setitem(inside.NATIVE_ENV, "CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1")
    assert inside.content_hash(files, "r", "t", "native") != native


def test_run_nonce_records_the_fuel_tig_runtime_wrote(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text(
                '{"nonce": 7, "runtime_signature": 1, "fuel_consumed": 12345, '
                '"solution": "e30=", "cpu_arch": "AMD64"}')
            return Result(0)
        return Result(0, "quality: 5\n")
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/x.so"), 10, 600, None, run,
                           tmp_path)
    # mutation: leaving fuel_consumed None starves calibration of data
    assert row["fuel_consumed"] == 12345 and row["ok"]
    assert row["limit_hit"] is False


def test_run_nonce_fuel_is_none_when_the_runtime_wrote_nothing(tmp_path):
    def run(cmd, **kw):
        return Result(87 if cmd[0] == "tig-runtime" else 1)
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/x.so"), 10, 600, None, run,
                           tmp_path)
    assert row["fuel_consumed"] is None and row["error"] == "out_of_fuel"
    assert row["limit_hit"] is True


def test_run_nonce_marks_a_verified_out_of_fuel_run_as_limit_hit(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "tig-runtime":
            (Path(cmd[cmd.index("--output") + 1]) / f"{cmd[3]}.json").write_text(
                '{"nonce": 7, "fuel_consumed": 9, "solution": "e30="}')
            return Result(87)
        return Result(0, "quality: 5\n")
    row = inside.run_nonce("c003", "n=1", "ab" * 32, 7, Path("/x.so"), 10, 600, None, run,
                           tmp_path)
    # ok, as tig-runtime counts it, so `error` is None and cannot say the fuel ran out
    # mutation: limit_hit taken from `error == "out_of_fuel"` is False here, and validation
    # then calls a fuel-truncated metered run "nondeterministic"
    assert row["ok"] and row["error"] is None and row["limit_hit"] is True


def _native_run(seen, rc=0, write=True, quality=4242):
    def run(cmd, **kw):
        seen.append(cmd)
        if cmd[0].endswith("talos-native"):
            if write:
                Path(cmd[4]).write_text('{"nonce": 7, "solution": "e30=", "solve_us": 31000}')
            return Result(rc)
        return Result(0, f"quality: {quality}\n") if write else Result(1)
    return run


def test_run_nonce_native_builds_the_runner_and_verifier_commands(tmp_path):
    seen = []
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1_500_000, 600, None, _native_run(seen), tmp_path,
                                  hyperparameters={"a": 1}, fuel=5_000_000_000_000)
    nat, ver = seen
    settings = json.loads(nat[1])
    assert settings["challenge_id"] == "c003" and settings["track_id"] == "n=1"
    assert nat[0] == str(Path("/b/talos-native")) and nat[2:4] == ["ab" * 32, "7"]
    # the runner takes an output FILE (not tig-runtime's folder) and the verifier reads it
    assert Path(nat[4]).name == "7.json" and ver[4] == nat[4]
    # mutation: the fuel limit passed as the time budget, or the budget as the fuel
    assert nat[nat.index("--budget-us") + 1] == "1500000"
    # the cap the runner's __fuel_remaining starts from, for algorithms that read it
    assert nat[nat.index("--fuel") + 1] == "5000000000000"
    assert nat[nat.index("--hyperparameters") + 1] == '{"a":1}'
    assert ver[:4] == ["tig-verifier", nat[1], "ab" * 32, "7"] and len(ver) == 5
    assert row["ok"] and row["quality"] == 4242 and row["solve_us"] == 31000
    assert row["fuel_consumed"] is None and row["limit_hit"] is False


def test_run_nonce_native_without_a_budget_passes_no_flag(tmp_path):
    seen = []
    inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"), None, 600,
                            None, _native_run(seen), tmp_path, fuel=1000)
    # mutation: `--budget-us None` makes the runner's parse fail every calibration nonce
    assert "--budget-us" not in seen[0] and "--hyperparameters" not in seen[0]
    # mutation: the cap tied to the budget leaves calibration's algorithm reading 0 fuel
    assert seen[0][seen[0].index("--fuel") + 1] == "1000"


def test_run_nonce_native_gpu_passes_the_ptx_to_both_and_the_gpu_to_the_verifier(tmp_path):
    seen = []
    inside.run_nonce_native("c005", "k=1", "ab" * 32, 3, Path("/b/talos-native"), None, 600,
                            Path("/a.native.ptx"), _native_run(seen), tmp_path, fuel=1000)
    nat, ver = seen
    assert nat[nat.index("--ptx") + 1] == str(Path("/a.native.ptx")) and "--gpu" not in nat
    assert ver[ver.index("--ptx") + 1] == str(Path("/a.native.ptx"))
    assert ver[ver.index("--gpu") + 1] == "0"


def test_a_budget_exit_with_a_saved_solution_counts_like_metered_out_of_fuel(tmp_path):
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1000, 600, None, _native_run([], rc=87), tmp_path,
                                  fuel=10)
    # mutation: reporting every budget exit as an error is stricter than tig-runtime
    assert row["ok"] and row["error"] is None
    # mutation: limit_hit dropped makes validation compare this truncated quality with the
    # metered one and demote a valid candidate as "nondeterministic"
    assert row["limit_hit"] is True


def test_a_budget_exit_with_nothing_saved_is_out_of_fuel(tmp_path):
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1000, 600, None, _native_run([], rc=87, write=False),
                                  tmp_path, fuel=10)
    # mutation: a budget exit classified as panic or success
    assert not row["ok"] and row["error"] == "out_of_fuel"


def test_the_outer_timeout_on_a_native_nonce_is_timeout(tmp_path):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    row = inside.run_nonce_native("c003", "n=1", "ab" * 32, 7, Path("/b/talos-native"),
                                  1000, 5, None, run, tmp_path, fuel=10)
    assert not row["ok"] and row["error"] == "timeout"


def test_run_task_dispatches_on_the_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(inside, "run_nonce", lambda *a, **k: calls.append(("metered", a)) or {})
    monkeypatch.setattr(inside, "run_nonce_native",
                        lambda *a, **k: calls.append(("native", a, k)) or {})
    base = ("c003", "t", "ab" * 32, 1, "/x", 10, 60, None, str(tmp_path), None)
    inside.run_task(base + ("metered", None))
    inside.run_task(base + ("native", 777))
    # mutation: ignoring the mode runs every native task through tig-runtime
    assert [c[0] for c in calls] == ["metered", "native"]
    assert calls[0][1][5] == 10          # metered gets the fuel
    assert calls[1][1][5] == 777         # native gets the budget in the same slot
    # mutation: native dropping the task's fuel starts __fuel_remaining at no cap
    assert calls[1][2]["fuel"] == 10


GPU_WORKSPACE = "[workspace]\nmembers = [\n    \"tig-algorithms\",\n]\n"


def _gpu_monorepo(tmp_path):
    mono = tmp_path / "mono"
    (mono / "tig-binary" / "src").mkdir(parents=True)
    (mono / "tig-binary" / "src" / "framework.cu").write_text("// framework\n")
    (mono / "tig-challenges" / "src" / "hypergraph").mkdir(parents=True)
    (mono / "tig-challenges" / "src" / "hypergraph" / "kernels.cu").write_text("// challenge\n")
    algo = mono / "tig-algorithms" / "src" / "hypergraph" / "talos_cand"
    algo.mkdir(parents=True)
    (algo / "kernels.cu").write_text("// algorithm\n")
    (mono / "Cargo.toml").write_text(GPU_WORKSPACE)
    return mono


def test_build_native_ptx_concatenates_like_build_ptx_and_skips_fuel_injection(tmp_path):
    mono = _gpu_monorepo(tmp_path)
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        seen.append(Path(cmd[2]).read_text())  # the temp .cu, read before it is deleted
        Path(cmd[cmd.index("-o") + 1]).write_text("// ptx\n")
        return Result(0, "", "")
    ok, _ = inside.build_native_ptx(mono, "hypergraph", "talos_cand", run)
    cmd, code = seen
    assert ok
    # mutation: a different order or missing file compiles a different PTX than TIG's
    assert code.index("// framework") < code.index("// challenge") < code.index("// algorithm")
    # the flags build_ptx passes at MONOREPO_REF
    assert cmd[0] == "nvcc" and cmd[1] == "-ptx"
    assert cmd[cmd.index("-arch") + 1] == "compute_70" and cmd[cmd.index("-code") + 1] == "sm_70"
    assert "--use_fast_math" in cmd and "-dopt=on" in cmd
    dest = Path(cmd[cmd.index("-o") + 1])
    # mutation: writing over the metered <name>.ptx lets a native PTX reach tig-runtime
    assert dest.name == "talos_cand.native.ptx"


def test_build_native_ptx_fails_without_challenge_kernels(tmp_path):
    # build_ptx raises FileNotFoundError when the challenge glob is empty
    mono = _gpu_monorepo(tmp_path)
    (mono / "tig-challenges" / "src" / "hypergraph" / "kernels.cu").unlink()
    calls = []
    ok, out = inside.build_native_ptx(mono, "hypergraph", "talos_cand",
                                      lambda cmd, **kw: calls.append(cmd) or Result(0, "", ""))
    # mutation: building without them compiles a PTX that lacks the challenge's kernels
    assert not ok and "hypergraph" in out and calls == []


def test_build_native_runs_cargo_with_the_pinned_toolchain_and_fast_profile(tmp_path):
    mono = _gpu_monorepo(tmp_path)
    seen = []

    def run(cmd, **kw):
        seen.append((cmd, kw))
        if cmd[0] == "nvcc":
            Path(cmd[cmd.index("-o") + 1]).write_text("// ptx\n")
        return Result(0, "Compiling talos-native\n", "")
    ok, out = inside.build_native(mono, "hypergraph", "talos_cand", run)
    assert ok and "Compiling talos-native" in out
    (nvcc, _), (cargo, kw) = seen
    assert nvcc[0] == "nvcc"  # GPU: the PTX first, the runner loads it at run time
    assert cargo == ["cargo", "+nightly-2025-02-10", "build", "--release", "-p", "talos-native"]
    assert kw["cwd"] == mono
    # mutation: the workspace profile (lto = true, codegen-units = 1) makes the native build
    # as slow as the metered one it replaces
    assert kw["env"]["CARGO_PROFILE_RELEASE_LTO"] == "false"
    assert kw["env"]["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] == "16"
    assert (mono / "talos-native" / "src" / "main.rs").exists()


def test_build_native_stops_at_a_failed_ptx_build(tmp_path):
    mono = _gpu_monorepo(tmp_path)
    seen = []

    def run(cmd, **kw):
        seen.append(cmd[0])
        return Result(1, "", "kernels.cu(3): error: identifier undefined")
    ok, out = inside.build_native(mono, "hypergraph", "talos_cand", run)
    # mutation: carrying on to cargo reports the Rust build and hides the CUDA error
    assert not ok and seen == ["nvcc"] and "identifier undefined" in out


def test_build_native_cpu_builds_no_ptx(tmp_path):
    mono = make_monorepo(tmp_path)
    (mono / "Cargo.toml").write_text(GPU_WORKSPACE)
    seen = []
    inside.build_native(mono, "knapsack", "talos_cand",
                        lambda cmd, **kw: seen.append(cmd[0]) or Result(0))
    assert seen == ["cargo"]


def test_native_artifact_paths(tmp_path):
    binary, ptx = inside.native_artifact_paths(tmp_path, "hypergraph", "talos_cand")
    assert binary == tmp_path / "target" / "release" / "talos-native" and ptx is None
    p = tmp_path / "tig-algorithms" / "lib" / "hypergraph" / "ptx" / "talos_cand.native.ptx"
    p.parent.mkdir(parents=True)
    p.write_text("x")
    assert inside.native_artifact_paths(tmp_path, "hypergraph", "talos_cand")[1] == p


def _alive(pid: int) -> bool:
    """Running or sleeping; a zombie waiting for its reaper has stopped running."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc and fork")
def test_closing_the_pool_stream_early_kills_the_solvers_it_started(tmp_path, monkeypatch):
    # Issue #84. Pool.terminate() kills the workers but not the tig-runtime each one started;
    # those would run on to their own timeout and keep a C3 job alive after the stop.
    import time
    pids = tmp_path / "pids"
    pids.mkdir()
    fake = tmp_path / "bin" / "tig-runtime"
    fake.parent.mkdir()
    # nonce 0 exits at once (an `invalid` row, nothing saved); the others record their pid
    # and sleep well past the test
    fake.write_text('#!/bin/sh\n[ "$3" = 0 ] && exit 0\necho $$ > "' + str(pids)
                    + '/$3"\nexec sleep 120\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    rows = inside.run_nonces([_task(n, tmp_path, timeout_s=300) for n in range(3)], 3)
    assert next(rows)["nonce"] == 0
    deadline = time.monotonic() + 20
    while len(list(pids.iterdir())) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    started = [int(p.read_text()) for p in pids.iterdir()]
    try:
        assert len(started) == 2 and all(_alive(pid) for pid in started)
        rows.close()
        deadline = time.monotonic() + 10
        while any(_alive(pid) for pid in started) and time.monotonic() < deadline:
            time.sleep(0.05)
        # mutation: terminating the pool without killing each worker's process group
        assert not any(_alive(pid) for pid in started)
    finally:
        for pid in started:  # a failing run must not leave the sleeps behind
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


FIXED_CHALLENGE = """\
#[derive(Serialize, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    #[serde(skip_serializing)]
    hidden_seed: [u8; 32],
    pub num_steps: usize,
}
"""
# tig-challenges/src/energy_arbitrage/mod.rs before commit 2703c9af.
LEAKING_CHALLENGE = """\
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    hidden_seed: [u8; 32],
    pub num_steps: usize,
}
"""


def _energy_monorepo(tmp_path, source):
    path = tmp_path / "tig-challenges" / "src" / "energy_arbitrage"
    path.mkdir(parents=True)
    if source is not None:
        (path / "mod.rs").write_text(source)
    return tmp_path


def test_challenge_problem_passes_only_the_fixed_source(tmp_path):
    # mutation: the fix being a comment in challenges.py, as it was, lets a wrong checkout score
    fixed = _energy_monorepo(tmp_path / "a", FIXED_CHALLENGE)
    assert inside.challenge_problem("energy_arbitrage", fixed) is None
    assert inside.challenge_problem("energy_arbitrage",
                                    _energy_monorepo(tmp_path / "b", LEAKING_CHALLENGE))
    assert inside.challenge_problem("knapsack", tmp_path) is None


@pytest.mark.parametrize("source", [
    FIXED_CHALLENGE.replace("    #[serde(skip_serializing)]\n", ""),
    FIXED_CHALLENGE.replace("hidden_seed:", "pub hidden_seed:").replace(
        "#[serde(skip_serializing)]\n    pub", "#[serde(skip_serializing)]\n    pub"),
    FIXED_CHALLENGE.replace("Serialize, Clone", "Serialize, Debug, Clone"),
    FIXED_CHALLENGE.replace("Serialize, Clone", "Serialize, Deserialize, Clone"),
    "pub struct Other {}\n",
    None,
], ids=["serialised", "public", "debug", "deserialize", "no_struct", "no_file"])
def test_challenge_problem_refuses_each_way_the_seed_can_be_exposed(tmp_path, source):
    # mutation: dropping any one test of the source lets that form through
    assert "2703c9af" in inside.challenge_problem("energy_arbitrage",
                                                  _energy_monorepo(tmp_path, source))


@pytest.mark.parametrize("build", [inside.build, inside.build_native], ids=["metered", "native"])
def test_a_build_of_the_leaking_challenge_never_starts(tmp_path, build):
    # mutation: checking only the metered build lets native research score against the old code
    def run(*a, **kw):
        pytest.fail("must not build")
    ok, out = build(_energy_monorepo(tmp_path, LEAKING_CHALLENGE), "energy_arbitrage", "algo",
                    run=run)
    assert not ok and "2703c9af" in out
