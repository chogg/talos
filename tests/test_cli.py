import json
import os
import stat
import time
import types
from dataclasses import replace
from pathlib import Path

import pytest

from talos import cli
from talos.bench import BenchCancelled, EvalResult
from talos.c3_bench import C3CommandError
from talos.challenges import CHALLENGES, DEV_IMAGE_TAG
from talos.config import (Config, ConfigError, load, resolve_api_key, resolve_c3_api_key,
                          save)
from talos.mainnet import MainnetError, TrackHyperparameters
from talos.types import CompileResult


@pytest.fixture(autouse=True)
def _no_external_metadata(monkeypatch):
    """Keep the mainnet baseline lookup and advisory GHCR check offline. Tests that exercise
    these integrations override the corresponding stub."""
    monkeypatch.setattr(cli, "recommend_image_update", lambda challenge: None)
    monkeypatch.setattr("talos.mainnet.top_algorithm", lambda ch: ("fake_base", "c003_a000", 1))
    monkeypatch.setattr("talos.mainnet.top_hyperparameters",
                        lambda algorithm_id, tracks, fuel:
                        {t: TrackHyperparameters(None, None, None, None) for t in tracks})


def scripted(answers):
    it = iter(answers)
    def ask(prompt, default=None, secret=False):
        try:
            v = next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return v if v != "" else (default or "")
    return ask


def test_setup_writes_config_and_0600_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda token_id, token_secret, run=None: calls.append("deploy"))
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0
    cfg = json.loads((tmp_path / "talos.config.json").read_text())
    assert cfg["provider"] == "anthropic" and cfg["model"] == "claude-opus-5"
    sec = tmp_path / ".talos" / "secrets.json"
    assert json.loads(sec.read_text()) == {"api_key": "sk-test"}
    # mutation: os.open creates a NEW file 0600 and chmod repairs a pre-existing one, so only
    # widening the open mode AND dropping the chmod fails this (either alone is covered by the other)
    if os.name != "nt":  # Windows has no owner-only mode bits; chmod there sets read-only alone
        assert stat.S_IMODE(sec.stat().st_mode) == 0o600
    assert calls == ["deploy"]


def test_setup_rejected_key_writes_nothing(tmp_path, monkeypatch):
    # mutation: writing before validation leaves a bad key on disk
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: "credential rejected")
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "bad", "ak", "as"]))
    assert rc != 0
    assert not (tmp_path / "talos.config.json").exists()
    assert not (tmp_path / ".talos").exists()


def test_setup_cli_provider_stores_no_secret(tmp_path, monkeypatch):
    # mutation: prompting for and saving an api_key for a CLI provider would write secrets.json
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    rc = cli.main(["setup"], ask=scripted(["", "claude-cli", "", "", "ak", "as"]))  # mode defaults
    assert rc == 0
    assert not (tmp_path / ".talos" / "secrets.json").exists()
    assert json.loads((tmp_path / "talos.config.json").read_text())["provider"] == "claude-cli"


def test_run_requires_budget_and_creates_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    fake_info = type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                               "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: fake_info)
    started = {}
    def fake_execute(spec, store, cfg, resume):
        started.update(spec=spec)
        return 0
    monkeypatch.setattr(cli, "execute_job", fake_execute)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "3",
                   "--yes"])
    assert rc == 0
    spec = started["spec"]
    assert spec.challenge == "knapsack" and spec.fuel == 7 and spec.tracks == ["n=1"]
    assert spec.budget.iterations == 3
    # mutation: dropping the default leaves Modal spend unbounded
    assert spec.budget.compute_usd == 20.0
    assert len(spec.rand_hash) == 64
    job_files = list((tmp_path / "runs").glob("*/job.json"))
    assert len(job_files) == 1
    # mutation: zero-budget guard using truthiness would accept a job with no cap; applying the
    # Modal default before validation would let it stand in for the missing LLM/time cap
    rc2 = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--yes"])
    assert rc2 == 2


def test_setup_rejects_the_fake_provider(tmp_path, monkeypatch, capsys):
    # mutation: accepting "fake" writes a config whose runs never call an LLM at all — the
    # in-process test double is not a provider a user can set up
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    assert cli.main(["setup"], ask=scripted(["", "fake", "fake", "fake"])) == 2
    assert "not one of the choices: 'fake'" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_wizard_labels_gpu_challenges_asks_mode_and_survives_a_typo(tmp_path, monkeypatch,
                                                                   capsys):
    # spec §5.2: GPU challenges are marked (GPU), with no class or cost figure; CLI providers are
    # asked for a mode, after the 5-20x token warning.
    # mutation: an unlabelled challenge list hides which challenges run on a GPU; dropping the
    # mode prompt means agentic can only be reached by flag
    # mutation: float(ask(...)) on a non-numeric answer tracebacks out of the wizard
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    # No challenge is known broken today; mark one so the menu's issue label is still checked.
    monkeypatch.setitem(cli.CHALLENGES, "job_scheduling",
                        replace(cli.CHALLENGES["job_scheduling"], status="known broken", issue=36))
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec, cfg=cfg) or 0)
    prompts = []
    answers = iter(["knapsack", "go", "abc", "3", "4", "5", "agentic", "", "", "", ""])  # last: Start this job?

    def ask(prompt, default=None, secret=False):
        prompts.append(prompt)
        return next(answers)

    assert cli.main(["run"], ask=ask) == 0
    captured = capsys.readouterr()
    assert "not a number: 'abc'" in captured.err
    assert "hypergraph" in prompts[0] and "GPU, tested" in prompts[0]
    assert "job_scheduling" in prompts[0] and "known broken (#36)" in prompts[0]
    assert "L40S" not in prompts[0]
    assert "$" not in prompts[0] and "estimated" not in prompts[0]
    assert "CPU, tested" in prompts[0]
    assert prompts.count("Iteration budget") == 2  # the typo was re-asked, not fatal
    assert prompts[-5].startswith("Mode:\n") and "roughly 5-20x the tokens" in prompts[-5]
    assert prompts[-4].startswith("Track to optimise:\n") and "n=1" in prompts[-4]
    assert prompts[-3].startswith("Hyperparameters:\n")
    assert prompts[-2] == "Nonces per track"
    assert prompts[-1] == "Start this job?"
    assert seen["spec"].track is None
    assert seen["spec"].budget.iterations == 3 and seen["spec"].budget.hours == 4.0
    assert seen["spec"].budget.compute_usd == 5.0
    assert seen["cfg"].mode == "agentic" and seen["spec"].mode == "agentic"


def test_wizard_gives_up_after_three_non_numbers(tmp_path, monkeypatch, capsys):
    # mutation: an unbounded re-ask loop never terminates on a piped/empty stdin
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    assert cli.main(["run"], ask=scripted(["knapsack", "go", "abc", "def", "ghi"])) == 2
    assert "no number given after 3 tries" in capsys.readouterr().err


def test_challenge_table_drift_stops_before_the_job_exists(tmp_path, monkeypatch, capsys):
    # mutation: not cross-checking mainnet against talos/challenges.py scores every nonce under
    # the wrong challenge id (or on the wrong hardware) and reports the result as if it counted
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)

    def drifted(**over):
        fields = {"id": "c003", "name": "knapsack", "is_gpu": False, "tracks": ["n=1"],
                  "max_fuel": 7}
        fields.update(over)
        return type("I", (), fields)()

    argv = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1",
            "--yes"]
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: drifted(id="c999"))
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "challenge table drift" in err and "c999" in err and "talos/challenges.py" in err
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: drifted(is_gpu=True))
    assert cli.main(argv) == 1
    assert "challenge table drift" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()  # nothing was created either time


def test_compile_refuses_an_empty_or_missing_dir(tmp_path, monkeypatch, capsys):
    # mutation: shipping an empty file map pays Modal for a container that can only fail, and
    # reports the failure as a compiler error
    monkeypatch.chdir(tmp_path)

    def boom(*a, **k):
        raise AssertionError("the bench must not be constructed for an empty file map")

    monkeypatch.setattr(cli, "make_bench", boom)
    assert cli.main(["compile", "--challenge", "knapsack"]) == 2  # no algorithm/ at all
    assert "no .rs/.cu files under algorithm" in capsys.readouterr().err
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "notes.txt").write_text("not a source file\n")
    assert cli.main(["compile", "--challenge", "knapsack", "--dir", "algorithm"]) == 2


def test_resolve_api_key_prefers_file_then_env(tmp_path, monkeypatch):
    # mutation: reading the env var first, or treating "" as a key, fails these three asserts
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), "from-file")
    cfg = load(tmp_path)
    assert resolve_api_key(cfg) == "from-file"
    (tmp_path / ".talos" / "secrets.json").unlink()
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert resolve_api_key(load(tmp_path)) == "from-env"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert resolve_api_key(load(tmp_path)) is None  # empty env is unset, not a key


def fake_config(root):
    """A config whose provider is `fake`: execute_job then drives the whole loop with
    FakeProvider/FakeBench, no network and no Modal."""
    save(root, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)


def knapsack_info():
    return type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                          "tracks": ["n=1"], "max_fuel": 7})()


def fake_run(monkeypatch, extra=()):
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    return cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "5", "--yes", *extra])


def test_run_mode_flag_overrides_config_and_gates_agentic(tmp_path, monkeypatch):
    # mutation: dropping --mode, or letting agentic through for an API provider, fails these
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(cfg=cfg, spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--mode", "agentic",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 0
    assert seen["cfg"].mode == "agentic" and seen["spec"].mode == "agentic"
    # the override is per-run: talos.config.json is not rewritten
    assert json.loads((tmp_path / "talos.config.json").read_text())["mode"] == "single-shot"
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), None)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--mode", "agentic",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 2


def test_execute_job_refuses_without_credential(tmp_path, monkeypatch, capsys):
    # mutation: without the check the loop starts and spends Modal time before the first 401
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no API key for provider openai" in err and "OPENAI_API_KEY" in err
    assert list((tmp_path / "runs").glob("*/job.json"))  # the job dir was written
    assert not list((tmp_path / "runs").glob("*/state.json"))  # but nothing ran


def test_fake_run_end_to_end_wins_and_packages(tmp_path, monkeypatch, capsys):
    # mutation: dropping the delta line, the status line or build_package fails these asserts
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    # metered: the spend below counts the metered path's calls (native adds calibration and
    # validation; test_fake_native_run_validates_its_best_and_wins covers that path)
    rc = fake_run(monkeypatch, ["--scoring", "metered"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Status: won" in out
    assert "Best delta vs baseline: +1.000%" in out
    # one summary line per iteration: the outcome, then best, spend and time left. Compute is
    # FakeBench's $0.01 per nonce over 32 nonces: baseline 8 + 8, candidate 8 + 8.
    assert "#1 new best | best +1.000% | llm $0.02 | compute ≈$0.32 | ∞ left" in out
    assert "[status]" not in out and "iteration_done" not in out and "job=" not in out
    # mutation: stamping the printed prefix with state.iteration labels iteration 1's events #0
    assert "#1 trying: " in out and "#0 trying" not in out
    # mutation: printing raw event fields puts 16-digit floats and key=value pairs on the terminal
    assert "mean_rel_delta" not in out and "strategy_tag=" not in out
    assert "#1 scored +1.000% vs baseline" in out
    assert "#1 won: held-out " in out
    assert "stopped status=" not in out  # the summary block below reports it
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert f"Package: {run_dir / 'package'}" in out
    assert (run_dir / "package" / "scores.md").exists()
    # mutation: dropping the hint leaves the user hunting for the files to submit
    assert (f"You can find the code to submit at {run_dir / 'package' / 'submission'}, fill out "
            "the template README.md to make it ready to submit to The Innovation Game!") in out
    best = json.loads((run_dir / "state.json").read_text())["best"]
    assert "let k = 2;" in best["files"]["mod.rs"]


def test_fake_flag_needs_no_config_and_never_touches_mainnet(tmp_path, monkeypatch, capsys):
    # mutation: reading the config or calling mainnet under --fake makes the flag useless on a
    # fresh clone; a real clone has neither talos.config.json nor Modal/LLM credentials
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "talos.config.json").exists()

    def boom(name):
        raise AssertionError("fetch_challenge_info must not be called under --fake")
    monkeypatch.setattr(cli, "fetch_challenge_info", boom)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "test",
                   "--budget-iterations", "5", "--yes", "--fake"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Status: won" in out
    assert not (tmp_path / "talos.config.json").exists()


def test_resume_restarts_a_cancelled_job(tmp_path, monkeypatch, capsys):
    # mutation: resuming a cancelled job with a terminal status makes run() return immediately
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status="cancelled", stop_reason="user requested stop", best=None, iteration=0,
              confirmed=[], hypotheses=[], nodes=[], chains=[])
    st["spend"]["iterations"] = 0
    (run_dir / "state.json").write_text(json.dumps(st))
    capsys.readouterr()
    rc = cli.main(["run", "--resume", run_dir.name])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out
    events = [json.loads(ln)["kind"]
              for ln in (run_dir / "timeline.jsonl").read_text().splitlines()]
    assert "resumed" in events


def test_update_advisory_runs_only_for_new_real_jobs(tmp_path, monkeypatch):
    from talos.budget import Spend
    from talos.state import JobState

    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None),
         None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = []
    monkeypatch.setattr(cli, "recommend_image_update", seen.append)

    def execute(spec, store, cfg, resume, add=None):
        if not resume:
            store.save(JobState.fresh(Spend(started_at=0.0)))
        return 0

    monkeypatch.setattr(cli, "execute_job", execute)
    args = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1",
            "--yes"]
    assert cli.main(args) == 0
    assert seen == ["knapsack"]
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert cli.main(["run", "--resume", run_dir.name]) == 0
    assert cli.main(args + ["--fake"]) == 0
    assert seen == ["knapsack"]


@pytest.mark.parametrize("changed", [
    {"dev_image_tag": "0.0.7"},
    {"dev_image_tag": None},
    {"monorepo_ref": "old-ref"},
])
def test_resume_refuses_changed_build_before_spending_or_mutating_state(
        tmp_path, monkeypatch, capsys, changed):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    spec = json.loads((run_dir / "job.json").read_text())
    assert spec["dev_image_tag"] == DEV_IMAGE_TAG
    spec.update(changed)
    if spec["dev_image_tag"] is None:
        del spec["dev_image_tag"]  # a real job written before the field existed
    (run_dir / "job.json").write_text(json.dumps(spec))
    state = json.loads((run_dir / "state.json").read_text())
    state["status"] = "paused"
    state["pending_job"] = {"purpose": 1, "job_id": "old-image-job"}
    (run_dir / "state.json").write_text(json.dumps(state))
    before = {name: (run_dir / name).read_bytes()
              for name in ("job.json", "state.json", "timeline.jsonl")}
    monkeypatch.setattr(cli, "make_bench", lambda *a, **kw: pytest.fail("must not start compute"))
    monkeypatch.setattr(cli, "make_provider", lambda *a, **kw: pytest.fail("must not call LLM"))
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "start a new job" in capsys.readouterr().err
    assert before == {name: (run_dir / name).read_bytes() for name in before}


def test_resume_after_the_hours_window_shifts_the_clock(tmp_path, monkeypatch, capsys):
    # mutation: keeping the original started_at makes every late resume exhausted — a job with
    # --budget-hours 4 that is resumed five hours later exits "exhausted (hours)" at once and can
    # never be resumed again
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "5", "--budget-hours", "4", "--yes"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status="cancelled", stop_reason="user requested stop", best=None, iteration=0,
              confirmed=[], hypotheses=[], nodes=[], chains=[])
    st["spend"]["iterations"] = 0
    cancelled_at = time.time() - 5 * 3600  # cancelled five hours ago, an hour past the window
    st["spend"]["started_at"] = cancelled_at - 60
    (run_dir / "state.json").write_text(json.dumps(st))
    os.utime(run_dir / "state.json", (cancelled_at, cancelled_at))
    capsys.readouterr()
    rc = cli.main(["run", "--resume", run_dir.name])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out
    resumed = [json.loads(ln) for ln in (run_dir / "timeline.jsonl").read_text().splitlines()
               if json.loads(ln)["kind"] == "resumed"]
    # the pause is approximated as the time since the last save, and recorded, not silent
    assert resumed and resumed[-1]["paused_s"] >= 4 * 3600


def test_resume_uses_the_spec_not_the_current_config(tmp_path, monkeypatch, capsys):
    # mutation: resuming with today's talos.config.json switches provider/model/mode mid-job, so
    # the iterations before and after the resume were produced by different agents
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    spec = json.loads((run_dir / "job.json").read_text())
    spec.update(provider="anthropic", model="spec-model", mode="single-shot")
    (run_dir / "job.json").write_text(json.dumps(spec))
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status="cancelled", best=None, iteration=0, confirmed=[], hypotheses=[])
    st["spend"]["iterations"] = 0
    (run_dir / "state.json").write_text(json.dumps(st))
    # the config now says a different model
    save(tmp_path, Config(provider="anthropic", model="config-model", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    seen = {}

    def capture(kind, model, api_key=None, api_base=None):
        seen.update(kind=kind, model=model)
        raise RuntimeError("stop before the loop starts")

    monkeypatch.setattr(cli, "make_provider", capture)
    with pytest.raises(RuntimeError):
        cli.main(["run", "--resume", run_dir.name])
    assert seen == {"kind": "anthropic", "model": "spec-model"}
    # mutation: allowing --mode on a resume silently changes how the rest of the job is produced
    assert cli.main(["run", "--resume", run_dir.name, "--mode", "agentic"]) == 2
    assert "start a new job to change mode" in capsys.readouterr().err


def test_resume_without_state_json_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: guarding on job.json alone tracebacks on a run whose credential was refused
    # before the first save (job.json written, state.json never)
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 2)
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "1", "--yes"]) == 2
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert not (run_dir / "state.json").exists()
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "missing state.json" in capsys.readouterr().err


def test_resume_of_finished_job_and_of_missing_job(tmp_path, monkeypatch, capsys):
    # mutation: without the terminal-status check a won job silently re-enters the loop and
    # without the job.json check --resume mkdirs a stray empty run directory
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    out = capsys.readouterr().out
    assert "already won; nothing to resume" in out
    # mutation: the resume path re-packages but forgets to say where the code to submit is
    assert f"You can find the code to submit at {run_dir / 'package' / 'submission'}" in out
    assert cli.main(["run", "--resume", "20990101-000000-knapsack"]) == 2
    assert not (tmp_path / "runs" / "20990101-000000-knapsack").exists()


def test_run_reports_mainnet_failure_and_creates_nothing(tmp_path, monkeypatch, capsys):
    # mutation: creating the job dir before fetch_challenge_info leaves an empty run behind
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    url = "https://mainnet-api.tig.foundation/get-block"
    def boom(name):
        raise MainnetError(f"network error fetching {url}: x")
    monkeypatch.setattr(cli, "fetch_challenge_info", boom)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1
    assert url in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_direction_file_is_read_and_conflicts_are_refused(tmp_path, monkeypatch, capsys):
    # mutation: ignoring --direction-file starts the job with an empty direction and an empty
    # first tacit entry
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    (tmp_path / "dir.md").write_text("explore beam search\n")
    assert cli.main(["run", "--challenge", "knapsack", "--direction-file", "dir.md",
                     "--budget-iterations", "1", "--yes"]) == 0
    assert seen["spec"].direction.strip() == "explore beam search"
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert (run_dir / "tacit.md").read_text() == "- USER: explore beam search\n"
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--direction-file",
                     "dir.md", "--budget-iterations", "1", "--yes"]) == 2
    assert cli.main(["run", "--challenge", "knapsack", "--direction-file", "missing.md",
                     "--budget-iterations", "1", "--yes"]) == 2
    assert "direction file not found" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["exhausted", "cancelled", "failed"])
def test_unconfirmed_candidate_gets_a_caveat_not_the_submit_hint(tmp_path, capsys, status):
    # A job that did not win can still hold a best candidate, e.g. a held-out false positive.
    # mutation: printing the won hint for it tells the user to submit code that lost on held-out
    pkg = tmp_path / "package"
    (pkg / "submission").mkdir(parents=True)
    cli._print_submission_hint(types.SimpleNamespace(status=status, best=object()), pkg)
    out = capsys.readouterr().out
    assert "ready to submit" not in out
    # mutation: printing nothing hides where the unconfirmed candidate's code is
    assert out == (f"The best candidate is in {pkg / 'submission'}, but it has not beaten the "
                   f"baseline on the held-out nonces. Read {pkg / 'README.md'} before "
                   "submitting it.\n")


def test_no_hint_when_the_package_has_no_submission_folder(tmp_path, capsys):
    # build_package writes no submission/ for a best without a baseline
    # mutation: gating on state.best alone points the user at a folder that does not exist
    from talos.package import build_package
    from tests.test_package import make
    spec, st, store = make(tmp_path)
    st.baseline = None
    pkg = build_package(spec, st, store)
    cli._print_submission_hint(st, pkg)
    assert capsys.readouterr().out == ""


def test_compile_ships_sources_and_returns_compiler_status(tmp_path, monkeypatch, capsys):
    # mutation: returning 0 on a failed compile, or shipping non-source files, fails these
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    (tmp_path / "algorithm" / "notes.txt").write_text("ignore me\n")
    (tmp_path / "algorithm" / "sub").mkdir()
    (tmp_path / "algorithm" / "sub" / "helper.rs").write_text("fn help() {}\n")
    seen = {}

    def stub(ok):
        class B:
            def check_deploy(self):
                pass

            def _with_retry(self, call):
                return call()

            def select_hardware(self, challenge, chosen=None):
                return None

            def evaluate(self, request):
                seen.update(challenge=request.challenge, files=request.files,
                            training=request.training, holdout=request.holdout)
                return EvalResult(CompileResult(ok=ok, artifact_id="a1" if ok else None,
                                                output="compiler says"), [], None,
                                  "not_compiled" if not ok else "forced")

        def make(backend, run_dir, pending, c3_api_key=None, local=None):
            seen["backend"] = backend
            return B()
        return make
    monkeypatch.setattr(cli, "make_bench", stub(False))
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    assert cli.main(["compile", "--challenge", "knapsack"]) == 1  # no config here: modal
    assert seen["backend"] == "modal"
    # mutation: str(relative_to) keys a nested file `sub\\helper.rs` on Windows, which the Linux
    # side stages as one file with a backslash in its name
    assert seen["challenge"] == "knapsack" and sorted(seen["files"]) == ["mod.rs", "sub/helper.rs"]
    # mutation: a compile that ships nonce sets pays for scoring the agent did not ask for
    assert seen["training"] == [] and seen["holdout"] == []
    assert "compiler says" in capsys.readouterr().out
    monkeypatch.setenv("TALOS_BACKEND", "c3")
    monkeypatch.setattr(cli, "make_bench", stub(True))
    assert cli.main(["compile", "--challenge", "knapsack"]) == 0
    assert seen["backend"] == "c3"


def test_unpriced_model_refuses_a_dollar_only_budget(tmp_path, monkeypatch, capsys):
    # mutation: a None cost silently disables the dollar cap. estimate_cost has no entry for
    # gpt-5, so every Completion carries cost_usd=None, spend.llm_usd stays 0.00 and a
    # --budget-usd-only job runs forever.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    save(tmp_path, Config(provider="openai", model="gpt-5", mode="single-shot", api_base=None),
         None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = []
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.append(spec) or 0)
    base = ["run", "--challenge", "knapsack", "--direction", "go", "--yes"]
    assert cli.main(base + ["--budget-usd", "5"]) == 2
    err = capsys.readouterr().err
    assert "gpt-5 has no price-table entry" in err
    assert "--budget-hours or --budget-iterations" in err
    assert not seen and not (tmp_path / "runs").exists()  # refused before anything was created
    # another real cap present: the job runs, with the warning
    assert cli.main(base + ["--budget-usd", "5", "--budget-iterations", "3"]) == 0
    assert seen and "no price-table entry" in capsys.readouterr().err
    # mutation: a priced model must NOT be warned about or refused
    save(tmp_path, Config(provider="anthropic", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    assert cli.main(base + ["--budget-usd", "5"]) == 0
    assert "price-table" not in capsys.readouterr().err


def test_status_line_says_unpriced_instead_of_zero_dollars(tmp_path):
    # mutation: printing $0.00 for an unpriced model reports "we measured nothing spent" when
    # the truth is "we cannot measure what is being spent"
    from talos.budget import Budget, Spend
    from talos.state import JobState

    def line(provider, model):
        spec = cli.JobSpec(job_id="j", challenge="knapsack", direction="d", provider=provider,
                           model=model, mode="single-shot",
                           budget=Budget(usd=5.0, hours=None, iterations=None, compute_usd=1.0),
                           rand_hash="ab" * 32, tracks=["t"], training=[], holdout=[], fuel=1,
                           created_at=0.0, monorepo_ref="r", challenge_id="c003")
        return cli._status_line(spec, JobState.fresh(Spend(started_at=0.0)), 0.0)

    assert line("openai", "gpt-5") == "best n/a | llm unpriced | compute ≈$0.00 | ∞ left"
    assert line("anthropic", "claude-opus-5") == "best n/a | llm $0.00 | compute ≈$0.00 | ∞ left"
    # claude-cli reports each call's cost (total_cost_usd), so $0.00 there is a measurement
    assert "llm $0.00" in line("claude-cli", "claude-opus-5")
    # issue 34: codex reports no cost at all. mutation: listing codex as metered prints $0.00
    assert line("codex-cli", "gpt-5.5") == "best n/a | llm not tracked | compute ≈$0.00 | ∞ left"


def test_codex_refuses_a_dollar_only_budget(tmp_path, monkeypatch, capsys):
    # issue 34: codex exec reports no cost in either mode, so llm_usd never rises and a
    # --budget-usd-only job would run uncapped. mutation: treating codex as metered at $0
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="codex-cli", model="gpt-5.5", mode="single-shot",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = []
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.append(spec) or 0)
    base = ["run", "--challenge", "knapsack", "--direction", "go", "--yes"]
    assert cli.main(base + ["--budget-usd", "5"]) == 2
    err = capsys.readouterr().err
    assert "codex-cli does not report what its calls cost" in err
    assert "--budget-hours or --budget-iterations" in err
    assert not seen and not (tmp_path / "runs").exists()
    assert cli.main(base + ["--budget-usd", "5", "--budget-iterations", "3"]) == 0
    assert seen and "warning: codex-cli does not report" in capsys.readouterr().err
    # mutation: claude-cli is metered through total_cost_usd and must not be refused
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)
    assert cli.main(base + ["--budget-usd", "5"]) == 0
    assert "does not report" not in capsys.readouterr().err


def test_unpriced_reason_names_the_cause():
    # the summary line prints the reason; a codex run must not blame the price table
    assert cli.unpriced_reason("codex-cli", "gpt-5.5") == \
        "codex-cli does not report what its calls cost"
    assert cli.unpriced_reason("openai", "gpt-5") == "model gpt-5 has no price-table entry"
    assert cli.unpriced_reason("claude-cli", "anything") is None
    assert cli.unpriced_reason("fake", "fake") is None
    assert cli.unpriced_reason("anthropic", "claude-opus-5") is None


def test_run_summary_labels_codex_spend_not_tracked():
    # README says codex spend "is shown as `not tracked`"; the end-of-run summary must agree
    # with the status line. mutation: labelling every unmeasured run "unpriced" in the summary
    assert cli._llm_spend_text("codex-cli", "gpt-5.5", 0.0, why=True) == \
        "not tracked (codex-cli does not report what its calls cost)"
    assert cli._llm_spend_text("openai", "gpt-5", 0.0, why=True) == \
        "unpriced (model gpt-5 has no price-table entry)"
    assert cli._llm_spend_text("claude-cli", "claude-opus-5", 1.234, why=True) == "$1.23"
    assert cli._llm_spend_text("codex-cli", "gpt-5.5", 0.0) == "not tracked"


def test_event_line_for_an_uncounted_call():
    # issue #74: single-shot calls record this event too, so it must not say "agent session"
    assert cli._event_line("llm_cost_unknown", {"reason": "claude timed out after 1800s"}) == \
        "LLM cost not counted: claude timed out after 1800s"


def test_status_line_reports_hours_left(tmp_path):
    # mutation: dropping the elapsed time prints the whole budget as still left
    from talos.budget import Budget, Spend
    from talos.state import JobState
    spec = cli.JobSpec(job_id="j", challenge="knapsack", direction="d", provider="claude-cli",
                       model="m", mode="single-shot",
                       budget=Budget(usd=None, hours=4.0, iterations=None, compute_usd=1.0),
                       rand_hash="ab" * 32, tracks=["t"], training=[], holdout=[], fuel=1,
                       created_at=0.0, monorepo_ref="r", challenge_id="c003")
    state = JobState.fresh(Spend(started_at=0.0))
    assert cli._status_line(spec, state, 5400.0).endswith("| 2.5h left")


def _budget_spec(challenge, budget):
    return cli.JobSpec(job_id="j", challenge=challenge, direction="d", provider="claude-cli",
                       model="m", mode="single-shot", budget=budget, rand_hash="ab" * 32,
                       tracks=["t"], training=[], holdout=[], fuel=1, created_at=0.0,
                       monorepo_ref="r", challenge_id="c003")


def test_status_line_counts_added_hours(tmp_path):
    # mutation: _status_line reading spec.budget.hours prints "0.0h left" (1h cap, 1.5h used)
    from talos.budget import Budget, Spend
    from talos.state import JobState
    spec = _budget_spec("knapsack", Budget(usd=None, hours=1.0, iterations=None,
                                           compute_usd=1.0))
    state = JobState.fresh(Spend(started_at=0.0))
    state.budget_extensions = [{"add": {"iterations": None, "hours": 2.0, "usd": None,
                                        "compute_usd": None}}]
    assert cli._status_line(spec, state, 5400.0).endswith("| 1.5h left")


def test_freeze_hardware_checks_the_extended_budget(tmp_path, monkeypatch):
    # mutation: the pre-probe check reading spec.budget raises BudgetExhausted on a job whose
    # extension gave it room. hypergraph on modal has options ('L40S', 'A100-80GB', 'H100'),
    # so the pre-probe check runs; with no options it is skipped and this would pass anyway.
    from talos.budget import Budget, Spend
    from talos.state import JobState, JobStore
    # setenv, not delenv(raising=False): delenv on an unset name records nothing, so the L40
    # that freeze_hardware exports would leak into every later test.
    monkeypatch.setenv("TALOS_HARDWARE", "")
    spec = _budget_spec("hypergraph", Budget(usd=None, hours=None, iterations=1,
                                             compute_usd=None))
    state = JobState.fresh(Spend(started_at=0.0))
    state.spend.iterations = 1
    state.budget_extensions = [{"add": {"iterations": 1, "hours": None, "usd": None,
                                        "compute_usd": None}}]
    bench = types.SimpleNamespace(select_hardware=lambda ch, chosen=None: "L40",
                                  cost_mark=lambda: 0.0, cost_usd_since=lambda m: 0.0)
    cli.freeze_hardware(bench, spec, state, JobStore(tmp_path), "modal", log=lambda s: None,
                        clock=lambda: 0.0)
    assert state.hardware == "L40"


# Payloads below are copied from runs/20260916-095103-knapsack/timeline.jsonl (a real C3 job),
# with the expected lines written by hand.
def test_event_line_hypothesis_shows_title_and_tag_only():
    # mutation: printing the description puts a sentence cut at 60 characters on every iteration
    data = {"title": "Beam-Greedy Reconstruction",
            "description": "Replace single-path greedy reconstruction after perturbation",
            "strategy_tag": "hybrid"}
    assert cli._event_line("hypothesis", data) == "trying: Beam-Greedy Reconstruction (hybrid)"
    assert cli._event_line("hypothesis", {"title": "T"}) == "trying: T"


def test_event_line_scored_prints_percentages():
    # mutation: printing the floats as they are gives -0.00019886764555467371
    data = {"mean_rel_delta": -0.00019886764555467371,
            "worst_rel_delta": -0.0009943382277733685, "error_rate": 0.0,
            "runtime_ratio": 1.2345}
    assert cli._event_line("scored", data) == \
        "scored -0.020% vs baseline (worst track -0.099%, errors 0.0%, runtime x1.23)"
    del data["runtime_ratio"]  # timelines written before the field existed
    assert cli._event_line("scored", data) == \
        "scored -0.020% vs baseline (worst track -0.099%, errors 0.0%)"


def test_event_line_scored_says_how_many_nonces_were_not_run():
    data = {"mean_rel_delta": -0.5, "worst_rel_delta": -0.5, "error_rate": 0.95,
            "runtime_ratio": None, "not_run": 37}
    # mutation: without it, "errors 95.0%" reads as 38 measured failures
    assert cli._event_line("scored", data) == \
        "scored -50.000% vs baseline (worst track -50.000%, errors 95.0%, 37 not run)"


SCORED = {"mean_rel_delta": 0.004, "worst_rel_delta": -0.0012, "error_rate": 0.0,
          "runtime_ratio": 1.02}
TAIL = "worst track -0.120%, errors 0.0%, runtime x1.02)"


# Full-length lines exceed EVENT_LINE_WIDTH; talos run passes the terminal width.
def test_event_line_scored_shows_the_noise_figure():
    # mutation: the ratio formatted with :.2f, or the band left out
    data = {**SCORED, "mean_rel_se": 0.0031, "noise_ratio": 1.29, "noise_band": "marginal",
            "noise_state": "ok"}
    assert cli._event_line("scored", data, width=200) == \
        "scored +0.400% vs baseline (±0.310%, 1.3x noise, marginal; " + TAIL


@pytest.mark.parametrize("state,se,text", [
    ("n1", None, "n=1"), ("identical", 0.0, "identical on every nonce"),
    ("no_spread", 0.0, "±0, no spread")])
def test_event_line_scored_names_the_undefined_cases(state, se, text):
    data = {**SCORED, "mean_rel_se": se, "noise_ratio": None, "noise_band": None,
            "noise_state": state}
    assert cli._event_line("scored", data, width=200) == \
        f"scored +0.400% vs baseline ({text}; " + TAIL


@pytest.mark.parametrize("se", [None, 0.0, 0.009])
def test_event_line_scored_over_the_ceiling_has_no_noise(se):
    # spec Tests 10. mutation: switching on the SE value instead of noise_state prints "n=1"
    # or "no spread" for a broken candidate
    data = {**SCORED, "mean_rel_se": se, "noise_ratio": None, "noise_band": None,
            "noise_state": "over_ceiling"}
    assert cli._event_line("scored", data, width=200) == "scored +0.400% vs baseline (" + TAIL


def test_event_line_scored_noise_survives_truncation():
    # Review Focus 2. mutation: appending the noise after the runtime puts it past the cut
    data = {**SCORED, "mean_rel_se": 0.0031, "noise_ratio": 1.29, "noise_band": "marginal",
            "noise_state": "ok"}
    # the noise text with its ";" is 58 characters; width 70 keeps 67 of them, then "..."
    line = cli._event_line("scored", data, width=70)
    assert line.startswith("scored +0.400% vs baseline (±0.310%, 1.3x noise, marginal;")
    assert len(line) == 70 and line.endswith("...")


def test_event_line_compile_failed_is_one_line_with_the_first_error():
    # mutation: printing the output as it is spills rustc's multi-line diagnostic over the
    # terminal; taking the last line instead of the first error prints "error: aborting due to"
    output = ("warning: unused variable: `budget`\n    --> src/track1.rs:12:9\n"
              "error[E0004]: non-exhaustive patterns: `None` not covered\n    --> src/t.rs:1896:5\n"
              "     |\n1896 |     for &(i, score) in &order {\n"
              "error: aborting due to 1 previous error; 1 warning emitted\n")
    line = cli._event_line("compile_failed", {"output": output})
    assert line == "compile failed: error[E0004]: non-exhaustive patterns: `None` not covered"
    # mutation: ignoring the event's `error` field, which the loop reads from the whole output
    assert cli._event_line("compile_failed", {"output": output, "error": "error[E0382]: moved"}) \
        == "compile failed: error[E0382]: moved"
    # timelines written before that field: only the last 2000 characters are there
    tail_only = "     |\n1896 |     for &(i, score) in &order {\n\n"
    assert cli._event_line("compile_failed", {"output": tail_only}) == \
        "compile failed: 1896 | for &(i, score) in &order {"  # whitespace runs collapse
    assert cli._event_line("compile_failed", {"output": ""}) == "compile failed"


def test_event_line_iteration_done_words_each_outcome():
    # mutation: printing outcome=failed:edit runs_since_improvement=3 as it is
    def done(outcome, runs):
        return cli._event_line("iteration_done",
                               {"outcome": outcome, "runs_since_improvement": runs})
    assert done("improved", 0) == "new best"
    assert done("failed:score", 4) == "no improvement (4 in a row)"
    assert done("failed:edit", 1) == "no candidate: edit failed (1 in a row)"
    assert done("failed:compile", 2) == "no candidate: did not compile (2 in a row)"
    assert done("failed:dead_code", 2) == "no candidate: new code never called (2 in a row)"
    assert done("failed:runtime", 3) == "rejected: error rate over the ceiling (3 in a row)"
    assert done("failed:novel", 5) == "failed:novel (5 in a row)"  # an outcome added later


def test_event_line_other_events_are_sentences():
    # mutation: any of these falling back to key=value prints a Python repr on the terminal
    line = cli._event_line
    assert line("baseline", {"message": "baseline knap_lean: cache hit abc"}) == \
        "baseline knap_lean: cache hit abc"
    assert line("baseline_ready", {"name": "knap_lean", "adoption": 860122008192060065}) == \
        "baseline ready"
    assert line("edits_rejected", {"paths": ["a/track1.rs", "a/track2.rs"]}) == \
        "edit rejected: 2 path(s) outside the algorithm files"
    assert line("dead_code", {"names": ["polish", "swap"]}) == \
        "new code never called: polish, swap"
    assert line("reset", {"forced_tag": "construction"}) == "switching strategy to: construction"
    assert line("confirming", {}) == "confirming on held-out nonces"
    assert line("won", {"holdout": {"mean_rel_delta": 0.0125, "worst_rel_delta": 0.001,
                                    "error_rate": 0.0}}) == \
        "won: held-out +1.250% vs baseline (worst track +0.100%)"
    assert line("false_positive", {"holdout": {"mean_rel_delta": -0.002,
                                               "worst_rel_delta": -0.01,
                                               "error_rate": 0.0}}) == \
        "not confirmed: held-out -0.200% vs baseline (worst track -1.000%)"
    assert line("false_positive", {"error": "held-out not scored (timeout)"}) == \
        "not confirmed: held-out not scored (timeout)"
    assert line("rate_limited", {"wait_s": 60}) == "rate limited, waiting 60s"
    assert line("resumed_pending", {"job_id": "job_17_hffwm0"}) == \
        "reattached to bench job job_17_hffwm0"
    assert line("discarded_incomplete", {"discarded": 1}) == \
        "discarded 1 unfinished iteration(s) from the previous run"
    assert line("stopped", {"status": "exhausted", "reason": "hours"}) is None


def test_event_line_cuts_long_text_to_one_line():
    # mutation: an uncut lesson wraps over several terminal rows
    lesson = "When greedy reconstruction and bounded swaps repeatedly fail, " * 5
    out = cli._event_line("distilled", {"lesson": lesson + "\nsecond line"})
    assert out.startswith("lesson: When greedy reconstruction")
    assert "\n" not in out and len(out) == 100 and out.endswith("...")
    assert cli._event_line("distilled", {"lesson": "Short."}) == "lesson: Short."


def test_event_line_unknown_kind_falls_back_to_one_line_of_fields():
    # mutation: returning None for a kind with no formatter hides every event added later
    out = cli._event_line("brand_new", {"a": 1, "b": "x\ny" * 100})
    assert out.startswith("brand_new a=1 b=x y") and "\n" not in out and len(out) == 100


def test_run_refuses_agentic_codex_without_the_opt_in(tmp_path, monkeypatch, capsys):
    # mutation: leaving the refusal to attach_agentic tells the user only after the baseline has
    # been measured, i.e. after Modal has already been paid for
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_ALLOW_CODEX_AGENTIC", raising=False)
    save(tmp_path, Config(provider="codex-cli", model="gpt-5-codex", mode="agentic",
                          api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 0)
    argv = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1",
            "--yes"]
    assert cli.main(argv) == 2
    assert "TALOS_ALLOW_CODEX_AGENTIC=1" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()  # refused before the job dir was created
    monkeypatch.setenv("TALOS_ALLOW_CODEX_AGENTIC", "1")
    assert cli.main(argv) == 0  # the opt-in lets it through


def test_codex_refused_takes_the_start(monkeypatch, capsys):
    # mutation: ignoring start lets the opt-in through for a template start; pure function, no
    # subprocess or network
    monkeypatch.setenv("TALOS_ALLOW_CODEX_AGENTIC", "1")
    cfg = Config(provider="codex-cli", model="m", mode="agentic", api_base=None)
    assert cli._codex_refused(cfg) is False
    assert cli._codex_refused(cfg, "template") is True
    assert "--start template" in capsys.readouterr().err


def test_deploy_bench_failure_never_quotes_the_token_secret():
    # mutation: echoing argv into the error leaks the Modal secret
    def run(cmd, **kw):
        if "deploy" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Error: app deploy failed")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(ConfigError) as excinfo:
        cli.deploy_bench("ak-1", "as-supersecret", run=run)
    assert "as-supersecret" not in str(excinfo.value)
    assert "app deploy failed" in str(excinfo.value)


def _c3_runner(whoami_rc=0, balance="  Credit balance: £9.89\n", balance_rc=0):
    def run(cmd, **kw):
        if cmd[:2] == ["c3", "whoami"]:
            return types.SimpleNamespace(
                returncode=whoami_rc, stdout="Access approved" if whoami_rc == 0 else "",
                stderr="" if whoami_rc == 0 else "Your C3 session has expired")
        if cmd[:2] == ["c3", "balance"]:
            return types.SimpleNamespace(returncode=balance_rc,
                                         stdout="" if balance_rc else balance,
                                         stderr="c3: request failed" if balance_rc else "")
        raise AssertionError(cmd)
    return run


def test_setup_c3_skips_modal_and_writes_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployed")))
    monkeypatch.setattr(cli, "check_c3", lambda run=None, api_key=None: 9.89)
    # prompts: backend, provider, model, api key, C3 key (blank) — no Modal token prompts
    rc = cli.main(["setup"], ask=scripted(["c3", "anthropic", "", "sk-test", ""]))
    assert rc == 0
    cfg = json.loads((tmp_path / "talos.config.json").read_text())
    # mutation: not persisting the backend makes every run go to Modal after a C3 setup
    assert cfg["backend"] == "c3"
    assert load(tmp_path).backend == "c3"


def test_setup_modal_is_the_default_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench",
                        lambda token_id, token_secret, run=None: calls.append("deploy"))
    rc = cli.main(["setup"], ask=scripted(["", "anthropic", "", "sk-test", "ak-1", "as-1"]))
    assert rc == 0 and calls == ["deploy"]
    assert json.loads((tmp_path / "talos.config.json").read_text())["backend"] == "modal"


def test_setup_c3_fails_when_the_session_has_expired(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("C3_API_KEY", raising=False)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_c3_runner(whoami_rc=1)))
    rc = cli.main(["setup"], ask=scripted(["c3", "anthropic", "", "sk-test", ""]))
    # mutation: ignoring whoami's exit code writes a config whose first run fails 20 min later
    assert rc == 1 and "c3 login" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_setup_c3_without_the_c3_binary_reports_it_instead_of_tracebacking(tmp_path,
                                                                           monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("C3_API_KEY", raising=False)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)

    def missing(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory", "c3")
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=missing))
    rc = cli.main(["setup"], ask=scripted(["c3", "anthropic", "", "sk-test", ""]))
    # mutation: cmd_setup catches ConfigError only, so a FileNotFoundError escaping check_c3
    # tracebacks out of the wizard and throws away every answer already typed
    assert rc == 1
    assert "install the `c3` CLI" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_check_c3_parses_the_balance_and_warns_when_low(capsys):
    assert cli.check_c3(run=_c3_runner()) == 9.89
    assert cli.check_c3(run=_c3_runner(balance="Credit balance: £0.40\n")) == 0.40
    # mutation: dropping the warning lets a run start on 40p and pause at the first job
    assert "low" in capsys.readouterr().err.lower()
    assert cli.check_c3(run=_c3_runner(balance_rc=1)) == 0.0
    err = capsys.readouterr().err
    # mutation: reporting an unreadable balance as "£0.00 is low" invents a number `c3` never gave
    assert "could not read" in err and "low" not in err.lower()
    with pytest.raises(ConfigError):
        cli.check_c3(run=_c3_runner(whoami_rc=1))


def test_config_without_backend_loads_as_modal(tmp_path):
    (tmp_path / "talos.config.json").write_text(json.dumps(
        {"provider": "anthropic", "model": "m", "mode": "single-shot", "api_base": None}))
    # mutation: a KeyError here breaks every config written before this change
    assert load(tmp_path).backend == "modal"


def test_config_round_trips_the_local_limits_and_omits_them_when_unset(tmp_path):
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None,
                          backend="local", local_cpus=8, local_memory_gib=12), None)
    cfg = load(tmp_path)
    assert cfg.local_cpus == 8 and cfg.local_memory_gib == 12
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None), None)
    # mutation: writing null keys makes a Modal config say something about a local container
    assert "local_cpus" not in json.loads((tmp_path / "talos.config.json").read_text())
    assert load(tmp_path).local_cpus is None


def test_image_available_checks_the_ghcr_manifest(monkeypatch):
    seen = []

    def fetch(url, headers):
        seen.append((url, headers))
        if url.startswith("https://ghcr.io/token?"):
            return 200, '{"token": "tok123"}'
        # a manifest GET without the pull token is refused, so a check that skips the token
        # step reports every image as missing
        if headers.get("Authorization") != "Bearer tok123":
            return 401, ""
        return (200, "{}") if url.endswith(f"knapsack/dev/manifests/{DEV_IMAGE_TAG}") else (404, "")
    assert cli.image_available("knapsack", fetch=fetch) is True
    token_url, manifest_url = seen[0][0], seen[1][0]
    assert token_url == ("https://ghcr.io/token?scope=repository:tig-foundation/tig-monorepo/"
                         "knapsack/dev:pull")
    # mutation: dropping the tag from the URL degrades the check to "the repo exists", so a
    # DEV_IMAGE_TAG bump with no matching image pays for a job that dies at the pull
    assert manifest_url == ("https://ghcr.io/v2/tig-foundation/tig-monorepo/knapsack/dev/"
                            f"manifests/{DEV_IMAGE_TAG}")
    assert cli.image_available("hypergraph", fetch=fetch) is False


def test_image_available_is_false_when_ghcr_is_unreachable():
    assert cli.image_available("knapsack", fetch=lambda url, headers: (0, "")) is False

    def token_then_dead(url, headers):
        # mutation: treating 0 as "present" on the manifest step alone submits a job that
        # dies at the pull whenever GHCR is down at check time
        return (200, '{"token": "t"}') if url.startswith("https://ghcr.io/token?") else (0, "")
    assert cli.image_available("knapsack", fetch=token_then_dead) is False


def test_image_available_is_false_when_the_socket_fails_outside_urlerror(monkeypatch):
    # urllib wraps only connect failures in URLError: a body read timeout is a bare
    # TimeoutError and a dropped connection a ConnectionResetError. Either escaping the check
    # tracebacks out of `talos run` before the run's outcome is recorded, so `talos status`
    # lists it as live for ever
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise TimeoutError("timed out")
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    assert cli.image_available("knapsack") is False

    def reset(req, timeout=None):
        raise ConnectionResetError(104, "Connection reset by peer")
    monkeypatch.setattr(cli.urllib.request, "urlopen", reset)
    assert cli.image_available("knapsack") is False


def test_run_c3_refuses_a_missing_image_before_the_baseline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None)
    monkeypatch.setattr(cli, "image_available", lambda ch, fetch=None: False)
    # the bench is stubbed as well as the check: the check is what is under test, so the test
    # must not depend on it to stay off a real, billable C3 job
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _refuse_bench("baseline reached"))
    fake_info = type("I", (), {"id": "c003", "name": "knapsack", "is_gpu": False,
                               "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: fake_info)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations",
                   "1", "--yes"])
    err = capsys.readouterr().err
    # mutation: checking the image after the baseline pays for a job that fails at the pull
    assert rc == 1 and "DEV_IMAGE_TAG" in err and "tig-monorepo/knapsack/dev" in err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    # mutation: returning without recording the outcome leaves `talos status` listing the run
    # as live for ever, with no reason
    assert st["status"] == "failed" and st["stop_reason"] == "dev image not on GHCR"


def test_compile_backend_resolution_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(backend=None)
    # no flag, no env, no config: modal (the agentic worktree has no config file)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    assert cli.compile_backend(args, tmp_path) == "modal"
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                          backend="c3"), None)
    assert cli.compile_backend(args, tmp_path) == "c3"
    # mutation: reading the config before the env makes an agentic C3 run compile on Modal
    monkeypatch.setenv("TALOS_BACKEND", "modal")
    assert cli.compile_backend(args, tmp_path) == "modal"
    assert cli.compile_backend(types.SimpleNamespace(backend="c3"), tmp_path) == "c3"


def test_execute_job_exports_the_backend_for_the_agentic_child(tmp_path, monkeypatch):
    # mutation: not exporting TALOS_BACKEND makes every `talos compile` from the agentic
    # sandbox default to Modal on a C3 run
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                          backend="c3"), None)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations",
                   "1", "--yes", "--fake"])
    assert rc == 0 and os.environ.get("TALOS_BACKEND") == "modal"  # --fake forces modal


def test_make_bench_picks_the_backend_and_hardware_class(tmp_path):
    from talos.bench import ModalBench, PendingJobStore
    from talos.c3_bench import C3Bench
    assert isinstance(cli.make_bench("modal", tmp_path, PendingJobStore.memory()), ModalBench)
    assert isinstance(cli.make_bench("c3", tmp_path, PendingJobStore.memory()), C3Bench)
    with pytest.raises(ConfigError):
        cli.make_bench("aws", tmp_path, PendingJobStore.memory())
    # mutation: using the Modal hardware class for C3 lets a Modal baseline serve a C3 run
    assert cli.bench_hardware_class("c3", "knapsack", hardware="cpu-d3-4vcpu-16gb") == \
        "c3-cpu-d3-4vcpu-16gb"
    assert cli.bench_hardware_class("modal", "knapsack") == "cpu4-mem8192-x4"


def test_sigint_stops_the_bench_too(tmp_path, monkeypatch):
    # mutation: only calling loop.request_stop leaves a 20-minute C3 poll running (and billing)
    # until it returns on its own
    import signal
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    stops = []
    from talos.bench import FakeBench
    real_init = FakeBench.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self.request_stop = lambda: stops.append("bench")
    monkeypatch.setattr(FakeBench, "__init__", init)
    handlers = {}
    # Every registration in order: cmd_run's default handler first, then the loop's stop
    # handler, then execute_job's `finally` putting the previous one back.
    monkeypatch.setattr(cli.signal, "signal",
                        lambda sig, h: handlers.setdefault(sig, []).append(h))
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "1", "--yes", "--fake"]) == 0
    assert handlers[signal.SIGINT][0] is signal.default_int_handler
    handlers[signal.SIGINT][1](None, None)
    assert stops == ["bench"]


def _fake_config_run(tmp_path, monkeypatch, at_fetch):
    """`talos run` with a fake config, stopped at the mainnet fetch by `at_fetch`."""
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None), None)
    monkeypatch.setattr(cli, "fetch_challenge_info", at_fetch)
    return cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "1", "--yes"])


def test_run_restores_the_default_sigint_handler_on_entry(tmp_path, monkeypatch):
    # A `talos run ... &` from a non-interactive shell inherits SIGINT ignored, so an agent's
    # `kill -INT` before the loop installs its own handler did nothing (#71).
    # mutation: no reset, or a reset placed after the mainnet fetch, records SIG_IGN
    import signal
    seen = []

    def at_fetch(name):
        seen.append(signal.getsignal(signal.SIGINT))
        raise MainnetError("stop")
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        assert _fake_config_run(tmp_path, monkeypatch, at_fetch) == 1
    finally:
        signal.signal(signal.SIGINT, previous)
    assert seen == [signal.default_int_handler]


def test_run_puts_the_callers_sigint_handler_back(tmp_path, monkeypatch):
    # mutation: no `finally` leaves default_int_handler installed for an in-process caller
    import signal

    def callers_handler(*_):
        pass

    def at_fetch(name):
        raise MainnetError("stop")
    previous = signal.signal(signal.SIGINT, callers_handler)
    try:
        assert _fake_config_run(tmp_path, monkeypatch, at_fetch) == 1
        assert signal.getsignal(signal.SIGINT) is callers_handler
    finally:
        signal.signal(signal.SIGINT, previous)


def test_early_interrupt_exits_130_without_traceback(tmp_path, monkeypatch, capsys):
    # mutation: no catch lets KeyboardInterrupt out of main(); returning 1 or printing the
    # traceback changes the exit code or stderr
    def at_fetch(name):
        raise KeyboardInterrupt
    try:
        rc = _fake_config_run(tmp_path, monkeypatch, at_fetch)
    except KeyboardInterrupt:  # escaping would abort the pytest session instead of failing
        pytest.fail("KeyboardInterrupt escaped talos run")
    assert rc == 130
    assert capsys.readouterr().err == "interrupted\n"
    assert list(tmp_path.glob("runs/*/job.json")) == []


def test_interrupt_after_the_job_exists_leaves_it_cancelled(tmp_path, monkeypatch, capsys):
    # job.json is written before execute_job pulls images and probes capacity, and the loop's
    # own handler comes after both. mutation: not marking the job leaves it `queued`, which
    # `talos status` lists as live for ever; overwriting a stopped status would relabel a job
    # that had already finished
    monkeypatch.chdir(tmp_path)

    def probe(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "freeze_hardware", probe)
    try:
        rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                       "--budget-iterations", "1", "--yes", "--fake"])
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped talos run")
    assert rc == 130
    [run_dir] = [p.parent for p in tmp_path.glob("runs/*/job.json")]
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert (state["status"], state["stop_reason"]) == ("cancelled",
                                                      "interrupted before the research loop")
    err = capsys.readouterr().err
    assert err.endswith(f"interrupted; continue with talos run --resume {run_dir.name}\n"), err


def test_interrupted_resume_of_a_dead_job_leaves_it_cancelled(tmp_path, monkeypatch):
    # A job whose process died stays `researching`; a resume interrupted during the probe
    # must stop it too. mutation: the resume path calling execute_job directly leaves it live
    monkeypatch.chdir(tmp_path)

    def probe(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "freeze_hardware", probe)
    args = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "1",
            "--yes", "--fake"]
    assert cli.main(args) == 130
    [run_dir] = [p.parent for p in tmp_path.glob("runs/*/job.json")]
    path = run_dir / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["status"], state["stop_reason"] = "researching", None
    path.write_text(json.dumps(state), encoding="utf-8")
    assert cli.main(["run", "--resume", run_dir.name, "--yes", "--fake"]) == 130
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"


def test_interrupt_keeps_a_stopped_status(tmp_path, monkeypatch):
    # A Ctrl-C while the package is written after the loop must not relabel the finished job.
    monkeypatch.chdir(tmp_path)

    def late(*a, **k):
        raise KeyboardInterrupt
    import talos.package
    monkeypatch.setattr(talos.package, "build_package", late)
    try:
        rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                       "--budget-iterations", "1", "--yes", "--fake"])
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped talos run")
    assert rc == 130
    [run_dir] = [p.parent for p in tmp_path.glob("runs/*/job.json")]
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "won"


def test_a_cancelled_baseline_is_not_a_failed_run(tmp_path, monkeypatch, capsys):
    # mutation: letting BenchCancelled out of measure_baseline fall into execute_job's blanket
    # `except Exception` records a Ctrl-C during the C3 baseline job as a failed run
    monkeypatch.chdir(tmp_path)
    from talos.bench import BenchCancelled, FakeBench

    def cancelled(self, request):
        raise BenchCancelled("job_x")
    monkeypatch.setattr(FakeBench, "evaluate", cancelled)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations",
                   "1", "--yes", "--fake"])
    assert rc == 1
    run_dir = next((tmp_path / "runs").glob("*/state.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    assert st["status"] == "cancelled" and "job_x" in st["stop_reason"]
    out = capsys.readouterr().out
    assert "Status: cancelled" in out
    # mutation: printing the hint without a candidate points at a folder that does not exist
    assert "You can find the code to submit" not in out


class _RefusingBench:
    """A bench that fails the test instead of reaching a backend: no test may submit a job."""

    def __init__(self, why):
        self._why = why

    def check_deploy(self):
        pass

    def _with_retry(self, call):
        return call()

    def select_hardware(self, challenge, chosen=None):
        return None

    def evaluate(self, request):
        raise AssertionError(self._why)

    def request_stop(self):
        pass

    def cost_mark(self):
        return 0.0

    def cost_usd_since(self, mark):
        return 0.0


def _refuse_bench(why):
    return _RefusingBench(why)


def _stub_mainnet(monkeypatch):
    """Everything resolve_baseline reads from mainnet, so a test never hits the network."""
    monkeypatch.setattr("talos.mainnet.top_algorithm", lambda ch: ("fake_base", "c003_a000", 1))
    monkeypatch.setattr("talos.mainnet.fetch_template", lambda ch: "pub fn solve_challenge(")
    monkeypatch.setattr("talos.mainnet.fetch_algorithm_files",
                        lambda ch, name: {"mod.rs": "fn solve() {}\n"})


def test_execute_job_exports_the_c3_backend_it_actually_runs_on(tmp_path, monkeypatch):
    # mutation: hardcoding "modal" in the export (or reading anything but cfg.backend) sends
    # every `talos compile` from the agentic sandbox to Modal on a C3 run
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None)
    monkeypatch.setattr(cli, "image_available", lambda ch, fetch=None: True)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    seen = {}

    class B(_RefusingBench):
        def select_hardware(self, challenge, chosen=None):
            return chosen or "cpu-d3-4vcpu-16gb"

        def evaluate(self, request):
            seen["backend"] = os.environ.get("TALOS_BACKEND")
            raise BenchCancelled("stop")  # nothing runs past the baseline

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("unreachable"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and seen["backend"] == "c3"


def test_setup_rejects_an_unknown_backend(tmp_path, monkeypatch, capsys):
    # mutation: accepting any answer writes a config whose first run cannot pick a bench at all
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    assert cli.main(["setup"], ask=scripted(["aws", "aws", "aws"])) == 2
    assert "not one of the choices: 'aws'" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_run_refuses_a_config_naming_an_unknown_backend(tmp_path, monkeypatch, capsys):
    # mutation: leaving a hand-edited backend to make_bench raises ConfigError from inside
    # execute_job, past every early return, so `talos run` tracebacks instead of reporting
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="fake", model="fake", mode="single-shot", api_base=None,
                          backend="aws"), None)
    monkeypatch.setattr(cli, "execute_job", lambda *a, **k: _refuse_bench("started").evaluate(None))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 2 and "unknown backend 'aws'" in capsys.readouterr().err


def test_compile_rejects_an_unknown_challenge(tmp_path, monkeypatch, capsys):
    # mutation: indexing CHALLENGES with an unvalidated name tracebacks in the agent's sandbox
    # instead of printing a message and exiting 2
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    monkeypatch.setattr(cli, "make_bench",
                        lambda *a, **k: _refuse_bench("bench built for an unknown challenge"))
    with pytest.raises(SystemExit) as e:
        cli.main(["compile", "--challenge", "knapsak"])
    assert e.value.code == 2
    assert "invalid choice: 'knapsak'" in capsys.readouterr().err


def test_compile_refuses_an_unknown_backend(tmp_path, monkeypatch, capsys):
    # mutation: a stale TALOS_BACKEND in the sandbox env raises ConfigError out of cmd_compile
    # and the agent sees a traceback instead of exit code 2
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    monkeypatch.setenv("TALOS_BACKEND", "aws")
    monkeypatch.setattr(cli, "make_bench",
                        lambda *a, **k: _refuse_bench("bench built for an unknown backend"))
    assert cli.main(["compile", "--challenge", "knapsack"]) == 2
    assert "unknown backend 'aws'" in capsys.readouterr().err


def _recording(answers):
    prompts = []
    inner = scripted(answers)
    def ask(prompt, default=None, secret=False):
        prompts.append((prompt, default))
        return inner(prompt, default, secret)
    return ask, prompts


def test_setup_codex_offers_the_catalog_and_defaults_to_its_first_model(tmp_path, monkeypatch,
                                                                          capsys):
    # mutation: ignoring the catalog keeps the static default; defaulting to the last slug
    # instead of the first picks gpt-5.5
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    monkeypatch.setattr(cli, "list_codex_models", lambda: ["gpt-5.6-sol", "gpt-5.5"])
    ask, prompts = _recording(["", "codex-cli", "", "", "ak", "as"])
    rc = cli.main(["setup"], ask=ask)
    assert rc == 0
    assert json.loads((tmp_path / "talos.config.json").read_text())["model"] == "gpt-5.6-sol"
    out = capsys.readouterr().out
    assert "gpt-5.6-sol" in out and "gpt-5.5" in out
    assert [d for p, d in prompts if p.startswith("Model")] == ["gpt-5.6-sol"]


def test_setup_codex_falls_back_to_the_static_default_without_a_catalog(tmp_path, monkeypatch):
    # mutation: the stale gpt-5-codex default is what the ChatGPT-account codex rejects
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    monkeypatch.setattr(cli, "list_codex_models", lambda: [])
    rc = cli.main(["setup"], ask=scripted(["", "codex-cli", "", "", "ak", "as"]))
    assert rc == 0
    assert json.loads((tmp_path / "talos.config.json").read_text())["model"] == "gpt-5.5"


def test_setup_claude_cli_names_the_model_aliases_in_the_prompt(tmp_path, monkeypatch):
    # mutation: dropping the alias hint leaves the user guessing at full model ids
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    ask, prompts = _recording(["", "claude-cli", "", "", "ak", "as"])
    assert cli.main(["setup"], ask=ask) == 0
    model_prompts = [p for p, d in prompts if p.startswith("Model")]
    assert len(model_prompts) == 1 and "fable" in model_prompts[0] and "sonnet" in model_prompts[0]


def test_run_track_flag_lands_in_the_spec(tmp_path, monkeypatch):
    # mutation: parsing the flag but not storing it makes every focused job an all-tracks job
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--track", "n=1"])
    assert rc == 0 and seen["spec"].track == "n=1"
    # the nonce draw is unchanged: every track is still in the spec, so the baseline cache key is
    assert [s.track for s in seen["spec"].training] == ["n=1"]


def test_run_rejects_a_track_mainnet_does_not_have(tmp_path, monkeypatch, capsys):
    # mutation: skipping validation starts a job whose focus set is empty
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--track", "n=9"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "n=9" in err and "n=1" in err
    assert not (tmp_path / "runs").exists()  # validation runs before the run directory exists


def test_resume_with_a_different_track_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: letting --track through on resume would score a job on sets its baseline
    # comparison was never built for
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, ["--track", "n=1"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert cli.main(["run", "--resume", run_dir.name, "--track", "n=2"]) == 2
    assert "started with track" in capsys.readouterr().err


def test_run_nonces_flag_sizes_both_nonce_sets(tmp_path, monkeypatch):
    # mutation: the flag not reaching draw_nonce_sets, or reaching only one of the two sets,
    # leaves a set at the default
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--nonces", "5"])
    assert rc == 0
    assert [s.count for s in seen["spec"].training] == [5]
    assert [s.count for s in seen["spec"].holdout] == [5]


def test_run_defaults_to_eight_nonces_per_track(tmp_path, monkeypatch):
    # mutation: leaving the draw at 32 makes a GPU baseline ten hours of L40 at hypergraph rates
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes"])
    assert rc == 0
    assert [s.count for s in seen["spec"].training] == [8]
    assert [s.count for s in seen["spec"].holdout] == [8]


@pytest.mark.parametrize("bad", ["0", "1000001"])
def test_run_rejects_a_nonce_count_out_of_range(tmp_path, monkeypatch, capsys, bad):
    # mutation: dropping the lower bound draws an empty set; dropping the upper bound lets the
    # training range overlap the held-out range that starts at HOLDOUT_START
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--nonces", bad])
    assert rc == 2
    assert bad in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_resume_with_a_different_nonce_count_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: letting --nonces through on resume would silently keep the stored sets while
    # the summary claims another size; the same value, or no flag, must still resume
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, ["--nonces", "4"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    resumed = []
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume, add=None: resumed.append(resume) or 0)
    assert cli.main(["run", "--resume", run_dir.name, "--nonces", "5"]) == 2
    assert "started with 4 nonces" in capsys.readouterr().err
    assert resumed == []
    assert cli.main(["run", "--resume", run_dir.name, "--nonces", "4"]) == 0
    assert resumed == [True]


def test_run_summary_names_the_nonce_count(tmp_path, monkeypatch, capsys):
    # mutation: a summary without the count hides what a run was sized to
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--nonces", "3"])
    assert rc == 0
    assert "3 nonces per track" in capsys.readouterr().out


def test_run_wizard_asks_for_nonces_only_when_the_flag_is_absent(tmp_path, monkeypatch):
    # mutation: a wizard that never asks fixes every interactive run at the default; one that
    # asks despite the flag re-answers a question already settled
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    ask, prompts = _recording(["", "", "", "6", ""])  # compute budget, track, hyperparameters, nonces, confirm
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3"], ask=ask)
    assert rc == 0 and [s.count for s in seen["spec"].training] == [6]
    assert prompts[-2] == ("Nonces per track", "8")
    assert prompts[-1] == ("Start this job?", "y")
    ask, prompts = _recording(["", "", "", ""])
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--nonces", "2"], ask=ask)
    assert rc == 0 and [s.count for s in seen["spec"].training] == [2]
    assert not any(p.startswith("Nonces") for p, _ in prompts)


def test_fake_run_with_a_track_wins_and_packages_per_track(tmp_path, monkeypatch, capsys):
    # mutation: the whole focused path; the fake challenge has one track, so there is no guard
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    rc = fake_run(monkeypatch, ["--track", "n=1"])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out
    assert "track n=1 of 1 tracks" in out
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    scores = (run_dir / "package" / "scores.md").read_text()
    assert "# Training nonces (track n=1)" in scores
    assert "# Held-out nonces (track n=1)" in scores
    assert "Regression guard" not in scores


def test_run_track_all_on_the_flag_means_every_track(tmp_path, monkeypatch, capsys):
    # The wizard's answer "all" and the README's "default all" must be spellable on the flag too.
    # mutation: validating "all" against mainnet's tracks refuses it as an unknown track
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    prompts = []
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)

    def ask(prompt, default=None, secret=False):
        prompts.append(prompt)
        return default or ""
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--track", "all"], ask=ask)
    assert rc == 0 and seen["spec"].track is None
    assert not any(p.startswith("Track to optimise") for p in prompts)  # the flag answered it
    assert "1 tracks, 8 nonces per track, fuel 7" in capsys.readouterr().out


def test_resume_of_a_focused_job_with_track_all_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: normalising "all" to None before the resume check lets it through as "no flag"
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, ["--track", "n=1"]) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert cli.main(["run", "--resume", run_dir.name, "--track", "all"]) == 2
    assert "started with track n=1" in capsys.readouterr().err


def _cli_config(tmp_path):
    save(tmp_path, Config(provider="claude-cli", model="claude-opus-5", mode="single-shot",
                          api_base=None), None)


def test_run_pins_the_top_algorithm_and_its_hyperparameters(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr("talos.mainnet.top_algorithm", lambda ch: ("algo", "c003_a7", 9))
    asked = []

    def top_hp(algorithm_id, tracks, fuel):
        asked.append((algorithm_id, tracks, fuel))
        return {"n=1": TrackHyperparameters({"x": 1}, "bm1", "0xp", 150.0)}
    monkeypatch.setattr("talos.mainnet.top_hyperparameters", top_hp)
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes"])
    assert rc == 0
    # mutation: passing the algorithm name, or another fuel, matches no precommit on mainnet
    assert asked == [("c003_a7", ["n=1"], 7)]
    spec = seen["spec"]
    # mutation: not pinning the algorithm lets the baseline measure a different one later
    assert spec.baseline_algorithm == {"name": "algo", "id": "c003_a7", "adoption": 9}
    assert spec.hyperparameters == {"n=1": {"x": 1}}
    assert spec.hyperparameters_source == {"n=1": {"benchmark_id": "bm1", "player_id": "0xp",
                                                   "mean_quality": 150.0}}
    # mutation: a silent default hides from the user that their job runs with mainnet values
    assert "hyperparameters: 1/1 tracks from mainnet" in capsys.readouterr().out


def test_run_hyperparameters_none_reads_nothing_from_mainnet(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    def boom(*a, **k):
        pytest.fail("mainnet must not be read for --hyperparameters none")
    monkeypatch.setattr("talos.mainnet.top_algorithm", boom)
    monkeypatch.setattr("talos.mainnet.top_hyperparameters", boom)
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes", "--hyperparameters", "none"])
    # mutation: ignoring the flag fetches and applies the map anyway
    assert rc == 0
    assert (seen["spec"].baseline_algorithm, seen["spec"].hyperparameters) == (None, None)
    assert "hyperparameters: none" in capsys.readouterr().out


def test_run_with_no_matching_benchmark_pins_the_algorithm_but_no_map(tmp_path, monkeypatch,
                                                                      capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    assert cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                     "--budget-iterations", "3", "--yes"]) == 0
    # mutation: storing {"n=1": None} makes the package claim hyperparameters were used
    assert seen["spec"].hyperparameters is None and seen["spec"].hyperparameters_source is None
    assert seen["spec"].baseline_algorithm == {"name": "fake_base", "id": "c003_a000",
                                               "adoption": 1}
    # mutation: printing the plain "hyperparameters: none" hides that mainnet was asked and had
    # no benchmark of this algorithm at this fuel (spec §7: "say so")
    out = capsys.readouterr().out
    assert "hyperparameters: none (no mainnet benchmark of fake_base at fuel 7)" in out


def test_run_reports_a_hyperparameters_fetch_failure_and_creates_nothing(tmp_path, monkeypatch,
                                                                         capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())

    def down(*a, **k):
        raise MainnetError("HTTP 503 fetching get-benchmarks")
    monkeypatch.setattr("talos.mainnet.top_hyperparameters", down)
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: pytest.fail("must not start a job"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "3", "--yes"])
    # mutation: letting MainnetError escape tracebacks; resolving after write_spec leaves a run dir
    assert rc == 1 and "mainnet unreachable" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_wizard_hyperparameters_answer_none_and_a_bad_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _cli_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    seen = {}
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.update(spec=spec) or 0)
    wizard = ["knapsack", "go", "3", "4", "5", "single-shot", ""]
    assert cli.main(["run"], ask=scripted(wizard + ["none", "", ""])) == 0  # "" = default nonces, then yes
    # mutation: ignoring the wizard answer applies mainnet values the user declined
    assert seen["spec"].hyperparameters is None and seen["spec"].baseline_algorithm is None
    # mutation: accepting any answer starts a job whose choice nobody made
    assert cli.main(["run"], ask=scripted(wizard + ["maybe", "maybe", "maybe"])) == 2
    assert "not one of the choices: 'maybe'" in capsys.readouterr().err


def test_resume_refuses_a_hyperparameters_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    # mutation: letting it through on resume would score a job against a baseline measured
    # with different hyperparameters
    assert cli.main(["run", "--resume", run_dir.name, "--hyperparameters", "none"]) == 2
    assert "fixed its hyperparameters" in capsys.readouterr().err


def test_fake_run_carries_hyperparameters_into_the_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    job = json.loads((run_dir / "job.json").read_text())
    # mutation: the fake provider path skipping FAKE_MAINNET's map leaves the end-to-end run
    # exercising none of this feature
    assert job["hyperparameters"] == {"n=1": {"fake_boost": 1}}
    assert "## Hyperparameters" in (run_dir / "package" / "README.md").read_text()


def test_save_keeps_the_llm_key_and_the_c3_key_side_by_side(tmp_path, monkeypatch):
    monkeypatch.delenv("C3_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None,
                          backend="c3"), "sk-1", c3_api_key="c3_key_1")
    sec = tmp_path / ".talos" / "secrets.json"
    # mutation: writing {"api_key": ...} alone drops the C3 key on every setup
    assert json.loads(sec.read_text()) == {"api_key": "sk-1", "c3_api_key": "c3_key_1"}
    if os.name != "nt":
        assert stat.S_IMODE(sec.stat().st_mode) == 0o600
    cfg = load(tmp_path)
    assert resolve_api_key(cfg) == "sk-1" and resolve_c3_api_key(cfg) == "c3_key_1"


def test_save_writes_a_c3_key_for_a_cli_provider(tmp_path, monkeypatch):
    # mutation: gating the secrets write on the LLM key never stores a claude-cli user's C3 key
    monkeypatch.delenv("C3_API_KEY", raising=False)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None, c3_api_key="c3_key_1")
    assert json.loads((tmp_path / ".talos" / "secrets.json").read_text()) == {
        "c3_api_key": "c3_key_1"}
    assert resolve_c3_api_key(load(tmp_path)) == "c3_key_1"


def test_resolve_c3_api_key_prefers_file_then_env(tmp_path, monkeypatch):
    monkeypatch.setenv("C3_API_KEY", "from-env")
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None,
                          backend="c3"), "sk-1", c3_api_key="from-file")
    # mutation: env first lets a stale shell variable override the key setup just checked
    assert resolve_c3_api_key(load(tmp_path)) == "from-file"
    save(tmp_path, Config(provider="anthropic", model="m", mode="single-shot", api_base=None,
                          backend="c3"), "sk-1")
    assert resolve_c3_api_key(load(tmp_path)) == "from-env"
    assert resolve_c3_api_key(None) == "from-env"  # `talos compile` in a dir with no config
    monkeypatch.setenv("C3_API_KEY", "")
    # mutation: returning the empty string sets C3_API_KEY="" and c3 rejects a login session
    assert resolve_c3_api_key(load(tmp_path)) is None and resolve_c3_api_key(None) is None


def test_check_c3_passes_the_api_key_in_the_environment_not_argv(monkeypatch):
    monkeypatch.setenv("PATH", "/some/bin")
    seen = []
    inner = _c3_runner()

    def run(cmd, **kw):
        seen.append((cmd, kw.get("env")))
        return inner(cmd, **kw)
    cli.check_c3(run=run, api_key="c3_key_secret")
    # mutation: dropping env= leaves the CLI on an expired login session; a bare
    # {"C3_API_KEY": k} without os.environ loses PATH and HOME
    assert [c[0][1] for c in seen] == ["whoami", "balance"]
    assert all(env["C3_API_KEY"] == "c3_key_secret" and env["PATH"] == "/some/bin"
               for _, env in seen)
    assert all("c3_key_secret" not in " ".join(cmd) for cmd, _ in seen)
    seen.clear()
    cli.check_c3(run=run)
    # mutation: always passing an env copy would still work, but setting C3_API_KEY=None
    # or "" breaks a login-session user; with no key the child inherits the environment as is
    assert all(env is None for _, env in seen)


def test_check_c3_failure_names_the_api_key_option():
    with pytest.raises(ConfigError) as excinfo:
        cli.check_c3(run=_c3_runner(whoami_rc=1), api_key="c3_key_secret")
    msg = str(excinfo.value)
    # mutation: the old message sends an API-key user to `c3 login`, which they chose not to use
    assert "API key" in msg and "c3_key_secret" not in msg


def test_setup_c3_checks_and_stores_the_c3_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    checked = []
    monkeypatch.setattr(cli, "check_c3", lambda run=None, api_key=None: checked.append(api_key))
    rc = cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", "c3_key_1"]))
    assert rc == 0
    # mutation: checking the login session instead of the typed key accepts a revoked key
    assert checked == ["c3_key_1"]
    assert json.loads((tmp_path / ".talos" / "secrets.json").read_text()) == {
        "c3_api_key": "c3_key_1"}


def test_setup_c3_rejected_key_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)

    def rejected(run=None, api_key=None):
        raise ConfigError("C3 login check failed: Invalid or revoked API key")
    monkeypatch.setattr(cli, "check_c3", rejected)
    rc = cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", "c3_key_bad"]))
    # mutation: saving before the check leaves a revoked key on disk
    assert rc == 1 and "revoked" in capsys.readouterr().err
    assert not (tmp_path / ".talos").exists()


def test_setup_modal_does_not_ask_for_a_c3_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: None)
    ask, prompts = _recording(["", "anthropic", "", "sk-test", "ak-1", "as-1"])
    assert cli.main(["setup"], ask=ask) == 0
    # mutation: asking on every backend puts a C3 prompt in a Modal user's wizard
    assert not any("C3 API key" in p for p, _ in prompts)


def test_execute_job_gives_the_c3_key_to_the_bench_and_the_sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # setenv before delenv: execute_job writes both into os.environ, and delenv of an unset
    # variable records nothing to undo, so the key would leak into every later test
    for var in ("C3_API_KEY", "TALOS_BACKEND"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None, c3_api_key="c3_key_1")
    monkeypatch.setattr(cli, "image_available", lambda ch, fetch=None: True)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    seen = {}

    class B(_RefusingBench):
        def select_hardware(self, challenge, chosen=None):
            return chosen or "cpu-d3-4vcpu-16gb"

        def evaluate(self, request):
            seen["env"] = os.environ.get("C3_API_KEY")
            raise BenchCancelled("stop")

    def make(backend, run_dir, pending, c3_api_key=None, local=None):
        seen["bench_key"] = c3_api_key
        return B("unreachable")
    monkeypatch.setattr(cli, "make_bench", make)
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    # mutation: not passing the key runs every job on the login session; not exporting it
    # leaves `talos compile` in the agentic sandbox with no credential
    assert rc == 1 and seen == {"bench_key": "c3_key_1", "env": "c3_key_1"}


def test_compile_uses_the_c3_key_from_the_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("C3_API_KEY", raising=False)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend="c3"), None, c3_api_key="c3_key_1")
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn solve() {}\n")
    seen = {}

    class B:
        def select_hardware(self, challenge, chosen=None):
            return None

        def evaluate(self, request):
            return EvalResult(CompileResult(ok=True, artifact_id="a1", output="ok"), [], None,
                              "forced")

    def make(backend, run_dir, pending, c3_api_key=None, local=None):
        seen["key"] = c3_api_key
        return B()
    monkeypatch.setattr(cli, "make_bench", make)
    assert cli.main(["compile", "--challenge", "knapsack"]) == 0
    # mutation: ignoring the config makes a direct `talos compile` fall back to `c3 login`
    assert seen["key"] == "c3_key_1"


def test_make_bench_hands_the_c3_key_to_c3bench(tmp_path, monkeypatch):
    from talos import c3_transport
    from talos.bench import PendingJobStore
    seen = []

    def fake_make(api_key, run=None):
        seen.append(api_key)
        return object()
    monkeypatch.setattr(c3_transport, "make_transport", fake_make)
    cli.make_bench("c3", tmp_path, PendingJobStore.memory(), c3_api_key="c3_key_1")
    # mutation: make_bench accepting the key but not forwarding it
    assert seen == ["c3_key_1"]


def test_setup_with_no_keys_left_removes_the_old_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("C3_API_KEY", raising=False)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    checked = []
    monkeypatch.setattr(cli, "check_c3", lambda run=None, api_key=None: checked.append(api_key))
    assert cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", "c3_old"])) == 0
    assert cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", ""])) == 0
    # mutation: skipping the write when no key was given leaves the old c3_api_key on disk, so
    # a run uses a key the user just chose to drop, after setup checked the login session
    assert checked == ["c3_old", None]
    assert resolve_c3_api_key(load(tmp_path)) is None
    assert not (tmp_path / ".talos" / "secrets.json").exists()


def test_check_c3_failure_with_the_key_from_the_environment_names_the_api_key(monkeypatch):
    monkeypatch.setenv("C3_API_KEY", "c3_key_env")
    with pytest.raises(ConfigError) as excinfo:
        cli.check_c3(run=_c3_runner(whoami_rc=1))
    msg = str(excinfo.value)
    # mutation: keying the hint on the argument alone sends a C3_API_KEY user to `c3 login`
    assert "check the C3 API key" in msg and "c3_key_env" not in msg


def test_check_c3_with_a_key_uses_mcp_and_never_shells_out(monkeypatch, capsys):
    calls = []

    class T:
        name = "mcp"

        def whoami(self):
            calls.append("whoami")
            return {"user_id": "u1"}

        def balance_gbp(self):
            calls.append("balance")
            return 0.5

    def no_subprocess(*a, **kw):
        raise AssertionError("check_c3 must not run a subprocess when a key is configured")

    monkeypatch.setattr(cli.subprocess, "run", no_subprocess)
    made = {}

    def fake_make(api_key, run=None):
        # AUDIT: was `made.setdefault("key", api_key) or T()`, which returns the key string.
        made["key"] = api_key
        return T()
    monkeypatch.setattr(cli, "make_transport", fake_make)
    assert cli.check_c3(api_key="c3_key_" + "a" * 20) == 0.5
    assert calls == ["whoami", "balance"]  # mutation: skipping whoami accepts a revoked key
    assert made["key"] == "c3_key_" + "a" * 20
    # mutation: dropping the warning hides a balance that cannot pay for the next job
    assert "low" in capsys.readouterr().err.lower()


def test_check_c3_reports_a_rejected_key_as_a_key_problem(monkeypatch):
    from talos.c3_mcp import McpAuthError

    class T:
        name = "mcp"

        def whoami(self):
            raise McpAuthError("the C3 API key was rejected")

        def balance_gbp(self):
            return 1.0

    monkeypatch.setattr(cli, "make_transport", lambda api_key, run=None: T())
    with pytest.raises(ConfigError) as ei:
        cli.check_c3(api_key="c3_key_" + "a" * 20)
    msg = str(ei.value)
    # mutation: telling a key user to run `c3 login` sends them down the wrong path
    assert "C3 rejected the API key" in msg
    assert "C3 login check failed" not in msg and "`c3 login`" not in msg


def test_check_c3_without_a_key_still_asks_the_cli_to_log_in(monkeypatch):
    monkeypatch.delenv("C3_API_KEY", raising=False)  # check_c3 reads it to pick the hint

    class T:
        name = "cli"

        def whoami(self):
            raise C3CommandError("c3 whoami could not be run: [Errno 2] no such file")

        def balance_gbp(self):
            return 1.0

    monkeypatch.setattr(cli, "make_transport", lambda api_key, run=None: T())
    with pytest.raises(ConfigError) as ei:
        cli.check_c3()
    msg = str(ei.value)
    # mutation: a CLI user with no session gets no instruction at all
    assert "run `c3 login`" in msg and "apikey list" not in msg


def test_check_c3_unreadable_balance_warns_and_returns_zero(monkeypatch, capsys):
    class T:
        name = "mcp"

        def whoami(self):
            return {}

        def balance_gbp(self):
            return None

    monkeypatch.setattr(cli, "make_transport", lambda api_key, run=None: T())
    # mutation: returning a number the server never reported ("£0.00 is low")
    assert cli.check_c3(api_key="k") == 0.0
    assert "could not read the C3 balance" in capsys.readouterr().err


def test_check_c3_reports_a_key_rejected_at_balance_as_a_key_problem(monkeypatch):
    from talos.c3_mcp import McpAuthError

    class T:
        name = "mcp"

        def whoami(self):
            return {}

        def balance_gbp(self):
            raise McpAuthError("the C3 API key was rejected")

    # mutation: remove the new McpAuthError clause in the balance_gbp try, letting
    # McpAuthError (a C3CommandError subclass) fall into the generic warning-and-0.0 path,
    # which downgrades a revoked key to a warning instead of stopping setup
    with pytest.raises(ConfigError) as ei:
        cli.check_c3(transport=T())
    assert "C3 rejected the API key" in str(ei.value)


def test_check_c3_with_an_injected_run_drives_the_cli_even_with_a_key(monkeypatch):
    def no_mcp(api_key, run=None):
        raise AssertionError("an injected `run` means the CLI; this would be a network call")
    monkeypatch.setattr(cli, "make_transport", no_mcp)
    # mutation: routing run= + api_key= to make_transport sends the existing keyed check_c3
    # tests to the real api.cthree.cloud
    assert cli.check_c3(run=_c3_runner(), api_key="c3_key_secret") == 9.89


def test_setup_checks_c3_with_the_environment_key_and_says_which_transport(tmp_path, monkeypatch,
                                                                            capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("C3_API_KEY", "c3_key_env")
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    checked = []
    monkeypatch.setattr(cli, "check_c3", lambda run=None, api_key=None: checked.append(api_key))
    assert cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", ""])) == 0
    # mutation: checking with the typed key alone makes setup demand the c3 CLI from a user
    # whose runs go over MCP on the C3_API_KEY in their environment
    assert checked == ["c3_key_env"]
    keyed_out = capsys.readouterr().out
    assert "MCP" in keyed_out and "`c3 login`" not in keyed_out
    # mutation: the same "no c3 CLI needed" line for a typed key is wrong for an environment
    # key, since a shell without C3_API_KEY set falls back to the c3 CLI
    assert "C3_API_KEY" in keyed_out
    monkeypatch.delenv("C3_API_KEY")
    assert cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", ""])) == 0
    # mutation: a fixed line tells a CLI user that nothing needs installing
    keyless_out = capsys.readouterr().out
    assert checked[-1] is None
    assert "`c3 login` session" in keyless_out and "MCP" not in keyless_out
    assert cli.main(["setup"], ask=scripted(["c3", "claude-cli", "", "", "c3_key_typed"])) == 0
    # mutation: printing the typed-key line for the environment-key case too (or vice versa)
    # tells a typed-key user that a shell without C3_API_KEY still works over MCP
    typed_out = capsys.readouterr().out
    assert checked[-1] == "c3_key_typed"
    assert "C3_API_KEY" not in typed_out and "MCP" in typed_out


def _local_docker(runtimes=("runc",), fail=False, ncpu=16, mem_gib=30):
    def run(cmd, **kw):
        if fail:
            return types.SimpleNamespace(returncode=1, stdout="",
                                         stderr="Cannot connect to the Docker daemon")
        # argv[0] is a full path on Windows (executables.argv0); match the command word
        assert cmd[1] == "info", cmd
        doc = {"Runtimes": {r: {} for r in runtimes}, "NCPU": ncpu, "MemTotal": mem_gib * 2 ** 30}
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(doc), stderr="")
    return run


def test_setup_local_asks_limits_checks_docker_and_writes_no_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "deploy_bench",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployed")))
    monkeypatch.setattr(cli, "check_c3",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("c3")))
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_local_docker(("runc", "nvidia"))))
    # prompts: backend, provider, model, mode, cpus, memory
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "6", "10"]))
    assert rc == 0
    cfg = load(tmp_path)
    assert cfg.backend == "local" and cfg.local_cpus == 6 and cfg.local_memory_gib == 10
    assert not (tmp_path / ".talos" / "secrets.json").exists()
    assert "GPU challenges: available" in capsys.readouterr().out


def test_setup_local_without_docker_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_local_docker(fail=True)))
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "", ""]))
    # mutation: ignoring the docker check writes a config whose first run fails at the baseline
    assert rc == 1 and "Docker" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_default_local_memory_floors_caps_and_falls_back():
    assert cli.default_local_memory_gib(6) == 4
    assert cli.default_local_memory_gib(30) == 26
    # mutation: a floor above the daemon's total proposes a default setup then refuses
    assert cli.default_local_memory_gib(3) == 3
    assert cli.default_local_memory_gib(None) == cli.LOCAL_DEFAULT_MEMORY_GIB


def test_setup_local_defaults_come_from_the_docker_daemon_not_the_host(tmp_path, monkeypatch):
    # MEASURED 2026-09-24 (Docker 29.1.3): `docker run --cpus` above the daemon's CPU count is
    # refused ("range of CPUs is from 0.01 to 16.00, as there are only 16 CPUs available"), and
    # `--memory` above its total is accepted without a check. On Docker Desktop the daemon
    # is a VM with fewer CPUs and less memory than the host, so host figures are wrong defaults.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 99)
    monkeypatch.setattr(cli, "subprocess",
                        types.SimpleNamespace(run=_local_docker(ncpu=5, mem_gib=12)))
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "", ""]))
    assert rc == 0
    cfg = load(tmp_path)
    # mutation: defaults from os.cpu_count / os.sysconf write 99 CPUs and the host's memory
    assert (cfg.local_cpus, cfg.local_memory_gib) == (5, 8)


def test_setup_local_refuses_limits_above_the_daemons(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "subprocess",
                        types.SimpleNamespace(run=_local_docker(ncpu=4, mem_gib=8)))
    # mutation: accepting 5 CPUs on a 4-CPU daemon fails every deploy with Docker's range error
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "5", "8"]))
    err = capsys.readouterr().err
    assert rc == 2 and "4 CPUs" in err and "8 GiB" in err
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "4", "9"]))
    assert rc == 2 and "8 GiB" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()
    # the daemon's own figures are accepted
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "4", "8"]))
    assert rc == 0 and (load(tmp_path).local_cpus, load(tmp_path).local_memory_gib) == (4, 8)


def test_make_bench_local_is_the_c3_bench_over_docker_and_the_local_class(tmp_path):
    from talos.bench import PendingJobStore
    from talos.c3_bench import C3Bench
    from talos.c3_jobdir import LocalSettings
    from talos.local_transport import DockerTransport
    b = cli.make_bench("local", tmp_path, PendingJobStore.memory(), local=LocalSettings(8, 12))
    assert isinstance(b, C3Bench) and isinstance(b._t, DockerTransport)
    # mutation: a local bench billing C3's hourly rate
    assert b.usd_per_hour == 0.0 and b.subdir == "local"
    cls = cli.bench_hardware_class("local", "knapsack", local=LocalSettings(8, 12), host="Box")
    assert cls == "local-box-cpu8-mem12"
    gpu = cli.bench_hardware_class("local", "hypergraph", local=LocalSettings(8, 12),
                                   gpu_name="NVIDIA L40S", host="box")
    assert gpu == "local-box-gpu-nvidia-l40s-cpu8-mem12"


def test_local_settings_come_from_the_config_or_the_docker_daemon(monkeypatch):
    from talos.local_transport import DockerInfo
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 99)
    calls = []

    def info():
        calls.append(1)
        return DockerInfo(runtimes=["runc"], ncpu=16, mem_total_gib=30)
    monkeypatch.setattr(cli, "docker_info", info)
    s = cli.local_settings(Config(provider="x", model="m", mode="single-shot", api_base=None,
                                  backend="local", local_cpus=6, local_memory_gib=10))
    # a config written by setup never asks Docker
    assert (s.cpus, s.memory_gib) == (6, 10) and calls == []
    # the agentic sandbox's `talos compile` has no config: the daemon's figures, not the host's
    # (mutation: os.cpu_count gives 99, which Docker Desktop's VM would refuse)
    s = cli.local_settings(None)
    assert (s.cpus, s.memory_gib) == (16, 26) and calls == [1]
    # a config without the limits (`talos compile --backend local` beside a modal config) too
    s = cli.local_settings(Config(provider="x", model="m", mode="single-shot", api_base=None))
    assert (s.cpus, s.memory_gib) == (16, 26)


def _local_config(root, cpus=8, memory=12):
    save(root, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                      backend="local", local_cpus=cpus, local_memory_gib=memory), None)


def test_run_local_skips_the_compute_budget_question_and_leaves_the_cap_unset(tmp_path,
                                                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: None)
    monkeypatch.setattr(cli, "has_gpu_runtime", lambda: False)
    seen = {}

    class B(_RefusingBench):
        def evaluate(self, request):
            raise BenchCancelled("stop")

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("unreachable"))
    real = cli.execute_job

    def spy(spec, store, cfg, resume):
        seen["compute"] = spec.budget.compute_usd
        return real(spec, store, cfg, resume)
    monkeypatch.setattr(cli, "execute_job", spy)
    # prompts: direction, iteration budget, hours, [compute: skipped, an extra prompt would
    # raise "unexpected prompt"], mode, track, hyperparameters, nonces (blank = the default
    # for each)
    rc = cli.main(["run", "--challenge", "knapsack"],
                  ask=scripted(["go", "1", "1", "", "", "", "", ""]))  # last: Start this job?
    assert rc == 1 and seen["compute"] is None
    # --yes must not apply DEFAULT_COMPUTE_USD either
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and seen["compute"] is None


def test_run_local_prepares_before_the_baseline_and_exports_the_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    order = []
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: order.append(("prepare", ch)))

    class B(_RefusingBench):
        def evaluate(self, request):
            order.append(("evaluate", os.environ.get("TALOS_BACKEND")))
            raise BenchCancelled("stop")

    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("unreachable"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    # mutation: preparing after the baseline, or not at all
    assert rc == 1 and order == [("prepare", "knapsack"), ("evaluate", "local")]


def test_run_local_reports_a_docker_failure_and_fails_the_job(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: (_ for _ in ()).throw(
        C3CommandError("docker info could not be run: [Errno 2] No such file")))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _RefusingBench("no bench"))
    rc = cli.main(["run", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and "Docker" in capsys.readouterr().err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    # mutation: a run that stops here left at its initial status shows as live for ever
    assert st["status"] == "failed" and "docker" in st["stop_reason"].lower()


def test_run_local_refuses_a_gpu_challenge_without_the_nvidia_runtime(tmp_path, monkeypatch,
                                                                       capsys):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    hypergraph = type("I", (), {"id": "c005", "name": "hypergraph", "is_gpu": True,
                                "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: hypergraph)
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "has_gpu_runtime", lambda: False)
    monkeypatch.setattr(cli, "prepare",
                        lambda ch, **k: (_ for _ in ()).throw(AssertionError("prepared")))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _RefusingBench("no bench"))
    rc = cli.main(["run", "--challenge", "hypergraph", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    err = capsys.readouterr().err
    assert rc == 1 and "NVIDIA" in err and "modal" in err and "c3" in err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    assert st["status"] == "failed"


def test_run_local_gpu_challenge_with_docker_down_fails_the_job_not_the_process(tmp_path,
                                                                                monkeypatch,
                                                                                capsys):
    monkeypatch.chdir(tmp_path)
    _local_config(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    hypergraph = type("I", (), {"id": "c005", "name": "hypergraph", "is_gpu": True,
                                "tracks": ["n=1"], "max_fuel": 7})()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: hypergraph)
    _stub_mainnet(monkeypatch)
    monkeypatch.setattr(cli, "has_gpu_runtime", lambda: (_ for _ in ()).throw(
        C3CommandError("docker info could not be run: Cannot connect to the Docker daemon")))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: _RefusingBench("no bench"))
    # mutation: the GPU check outside the try tracebacks and leaves state.json "initial"
    rc = cli.main(["run", "--challenge", "hypergraph", "--direction", "go",
                   "--budget-iterations", "1", "--yes"])
    assert rc == 1 and "Docker" in capsys.readouterr().err
    st = json.loads(next((tmp_path / "runs").glob("*/state.json")).read_text())
    assert st["status"] == "failed" and "docker" in st["stop_reason"].lower()


def test_setup_local_refuses_zero_cpus_or_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "validate_provider", lambda p: None)
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(run=_local_docker()))
    # mutation: accepting 0 surfaces 20 minutes later as "local Docker deploy failed"
    rc = cli.main(["setup"], ask=scripted(["local", "claude-cli", "", "single-shot", "0", "8"]))
    assert rc == 2 and "at least 1" in capsys.readouterr().err
    assert not (tmp_path / "talos.config.json").exists()


def test_compile_local_prepares_then_evaluates(tmp_path, monkeypatch, capsys):
    from talos.local_transport import DockerInfo
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn x(){}")
    order = []
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: order.append("prepare"))
    # no config here (the agentic sandbox): the limits come from the daemon, never the host
    monkeypatch.setattr(cli, "docker_info", lambda: DockerInfo(["runc"], 4, 8))
    monkeypatch.setattr(cli, "remove_compile_containers", lambda run_dir: None)  # no real docker
    seen = {}

    class B(_RefusingBench):
        def evaluate(self, request):
            order.append("evaluate")
            from talos.bench import EvalResult
            from talos.types import CompileResult
            return EvalResult(CompileResult(ok=True, artifact_id="a", output="ok"), [], None,
                              "not_won")

    def make(*a, local=None, **k):
        seen["local"] = local
        return B("x")
    monkeypatch.setattr(cli, "make_bench", make)
    rc = cli.main(["compile", "--challenge", "knapsack", "--backend", "local"])
    assert rc == 0 and order == ["prepare", "evaluate"]
    assert (seen["local"].cpus, seen["local"].memory_gib) == (4, 4)


def _compile_local(tmp_path, monkeypatch, evaluate, remove_error=None):
    """`talos compile --backend local` with Docker stubbed out. Returns (the return code or the
    exception that escaped, the call order, the run dirs whose containers were removed)."""
    from talos.local_transport import DockerInfo
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn x(){}")
    order, removed = [], []
    monkeypatch.setattr(cli, "prepare", lambda ch, **k: None)
    monkeypatch.setattr(cli, "docker_info", lambda: DockerInfo(["runc"], 4, 8))

    class T:
        def remove_run(self, run_dir):
            order.append("remove")
            removed.append(Path(run_dir))
            if remove_error is not None:
                raise remove_error
    monkeypatch.setattr(cli, "DockerTransport", T)

    class B(_RefusingBench):
        def evaluate(self, request):
            order.append("evaluate")
            return evaluate()
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: B("x"))
    try:
        out = cli.main(["compile", "--challenge", "knapsack", "--backend", "local",
                        "--dir", "algorithm"])
    except BaseException as e:  # noqa: BLE001 - the test inspects what escaped
        out = e
    return out, order, removed


def test_compile_local_removes_its_containers_after_the_result(tmp_path, monkeypatch):
    """The agentic sandbox runs `talos compile` from a new temp dir each iteration, so its run
    key is new each time and no later deploy's prune ever matches its containers."""
    out, order, removed = _compile_local(tmp_path, monkeypatch, lambda: EvalResult(
        CompileResult(ok=True, artifact_id="a", output="ok"), [], None, "not_won"))
    # mutation: no removal leaves one exited container per agentic compile, for ever
    # mutation: removing before evaluate deletes the container before its results are copied
    assert out == 0 and order == ["evaluate", "remove"]
    # mutation: a run dir other than the one make_bench was given carries another key
    assert removed == [tmp_path / ".talos" / "compile"]


def test_compile_local_removes_its_containers_when_evaluate_raises(tmp_path, monkeypatch):
    def interrupted():
        raise KeyboardInterrupt  # Ctrl-C, or a parent tearing the compile down mid-wait
    out, order, removed = _compile_local(tmp_path, monkeypatch, interrupted)
    # mutation: removal after evaluate but outside a finally leaves a running container behind
    assert isinstance(out, KeyboardInterrupt) and order == ["evaluate", "remove"]


def test_compile_local_reports_a_failed_removal_without_changing_the_result(tmp_path, monkeypatch,
                                                                           capsys):
    out, order, _ = _compile_local(
        tmp_path, monkeypatch,
        lambda: EvalResult(CompileResult(ok=True, artifact_id="a", output="ok"), [], None,
                           "not_won"),
        remove_error=C3CommandError("docker ps failed (1): daemon gone"))
    assert order == ["evaluate", "remove"]
    # mutation: letting the command error escape turns a good build into a traceback
    assert out == 0 and "daemon gone" in capsys.readouterr().err


def test_deploy_bench_deploys_the_app_as_a_package_module():
    """The functions are serialized=True. Deployed by file path, Modal imports the file as the
    top-level module `talos_bench`, the pickle refers to that name, and every container dies
    with "the 'talos_bench' module is not available in the remote environment": the image
    only carries the package `talos`."""
    seen = []

    def run(cmd, **kw):
        seen.append((list(cmd), kw))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    cli.deploy_bench(None, None, run=run)
    (cmd, kw), = seen
    # mutation: deploying talos/modal_app/talos_bench.py by path deploys an app no container
    # can load
    assert cmd[-3:] == ["deploy", "-m", "talos.modal_app.talos_bench"]
    # mutation: without the cwd, `-m` cannot import talos from outside the checkout
    assert (Path(kw["cwd"]) / "talos" / "modal_app" / "talos_bench.py").is_file()


def test_deploy_command_deploys_without_a_config(tmp_path, monkeypatch, capsys):
    from talos.deploy_digest import marker_name
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("talos.extras.importlib.util.find_spec", lambda name, *a, **k: object())
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: calls.append(a))
    assert cli.main(["deploy"]) == 0
    # mutation: passing a token re-sets the one `talos setup` stored; reading a config needs one
    assert calls == [(None, None)]
    assert marker_name() in capsys.readouterr().out


def test_deploy_without_the_modal_extra_names_the_install_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("talos.extras.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(cli, "deploy_bench", lambda *a, **k: pytest.fail("deployed"))
    rc = cli.main(["deploy"])
    err = capsys.readouterr().err
    # mutation: skipping require() lets `-m modal` fail with a bare "No module named modal"
    assert rc == 2 and "talos deploy needs the modal extra" in err


def test_deploy_command_reports_a_failed_modal_deploy(tmp_path, monkeypatch, capsys):
    from talos.deploy_digest import marker_name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("talos.extras.importlib.util.find_spec", lambda name, *a, **k: object())

    def fail(*a, **k):
        raise ConfigError("modal deploy failed: boom")
    monkeypatch.setattr(cli, "deploy_bench", fail)
    rc = cli.main(["deploy"])
    out, err = capsys.readouterr()
    # mutation: not catching the failure ends in a traceback; printing "Deployed" anyway hides it
    assert rc == 2 and "modal deploy failed: boom" in err
    assert marker_name() not in out


def test_deploy_uses_the_running_interpreters_modal(monkeypatch):
    import shutil
    import sys
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/modal")
    seen = []

    def run(cmd, **kw):
        seen.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    cli.deploy_bench("ak-1", "as-1", run=run)
    # mutation: a `modal` found on PATH may belong to another environment, without the
    # modal this Talos was installed with
    assert len(seen) == 2
    assert all(c[:3] == [sys.executable, "-m", "modal"] for c in seen)


# ── Hardware fallback ──────────────────────────────────────────────────────
def hypergraph_info():
    return type("I", (), {"id": "c005", "name": "hypergraph", "is_gpu": True,
                          "tracks": ["n=1"], "max_fuel": 7})()


class _ProbingBench(_RefusingBench):
    """Records select_hardware calls; evaluate stops the run before any job."""

    def __init__(self, choose="A100-80GB"):
        super().__init__("no job may be submitted")
        self.choose, self.selections, self.seen_env = choose, [], []

    def select_hardware(self, challenge, chosen=None):
        self.selections.append((challenge, chosen))
        return chosen or self.choose

    def evaluate(self, request):
        self.seen_env.append(os.environ.get("TALOS_HARDWARE"))
        raise BenchCancelled("stop")


def _gpu_run(tmp_path, monkeypatch, bench, backend="modal", extra=(),
             challenge="hypergraph"):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_HARDWARE", raising=False)
    save(tmp_path, Config(provider="claude-cli", model="m", mode="single-shot", api_base=None,
                          backend=backend), None)
    monkeypatch.setattr(cli, "image_available", lambda ch, fetch=None: True)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: object())
    info = knapsack_info() if challenge == "knapsack" else hypergraph_info()
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: info)
    monkeypatch.setattr("talos.mainnet.top_algorithm",
                        lambda ch: ("fake_base", f"{info.id}_a000", 1))
    monkeypatch.setattr("talos.mainnet.fetch_template", lambda ch: "pub fn solve_challenge(")
    monkeypatch.setattr("talos.mainnet.fetch_algorithm_files",
                        lambda ch, name: {"mod.rs": "fn solve() {}\n"})
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: bench)
    return cli.main(["run", "--challenge", challenge, "--direction", "go",
                     "--budget-iterations", "1", "--yes", *extra])


def test_execute_job_freezes_the_probed_gpu_in_state_and_exports_it(tmp_path, monkeypatch):
    from talos.state import JobStore
    b = _ProbingBench("A100-80GB")
    on_disk = {}
    real_event = JobStore.event

    def event(self, kind, **data):
        if kind == "hardware_selected":  # what state.json says when the choice is announced
            on_disk["hardware"] = json.loads(
                (self.run_dir / "state.json").read_text(encoding="utf-8")).get("hardware")
        real_event(self, kind, **data)
    monkeypatch.setattr(JobStore, "event", event)
    rc = _gpu_run(tmp_path, monkeypatch, b)
    assert rc == 1 and b.selections == [("hypergraph", None)]
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    # mutation: not saving the choice at once leaves a crash before the baseline's own save
    # to probe again on resume (and maybe land elsewhere); not exporting it makes the
    # sandbox's `talos compile` probe for its own
    assert st["hardware"] == "A100-80GB" and on_disk == {"hardware": "A100-80GB"}
    assert b.seen_env == ["A100-80GB"]
    # a resume hands the frozen choice back and does not probe
    b2 = _ProbingBench("H100")
    rc = _gpu_run(tmp_path, monkeypatch, b2, extra=["--resume", run_dir.name])
    assert rc == 1 and b2.selections == [("hypergraph", "A100-80GB")]
    assert b2.seen_env == ["A100-80GB"]


def test_execute_job_pauses_on_stale_deploy_before_any_spend(tmp_path, monkeypatch, capsys):
    from talos.bench import BenchUnavailable

    class Stale(_ProbingBench):
        def check_deploy(self):
            raise BenchUnavailable("the deployed talos-bench app was built from different Talos "
                                   "code than this client; run `talos deploy`")

    b = Stale()
    rc = _gpu_run(tmp_path, monkeypatch, b, challenge="knapsack")
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    # mutation: checking after freeze_hardware probes (and bills) a GPU on a stale app; not
    # checking at all runs the whole job on old code
    assert rc == 1 and st["status"] == "paused" and "talos deploy" in st["stop_reason"]
    assert b.selections == [] and b.seen_env == []


def test_execute_job_retries_a_transient_deploy_lookup_instead_of_crashing(tmp_path,
                                                                          monkeypatch):
    from talos.bench import ModalBench
    from tests.test_bench import FakeClock, recording_sleep
    clock, sleeps, lookups = FakeClock(), [], []

    class Flaky(ModalBench):
        def check_deploy(self):
            lookups.append(1)
            if len(lookups) == 1:
                raise ConnectionError("connection reset")

        def select_hardware(self, challenge, chosen=None):
            self.reached = True
            return None

        def evaluate(self, request):
            raise BenchCancelled("stop")

    b = Flaky(clock=clock, sleep=recording_sleep(clock, sleeps))
    b.reached = False
    rc = _gpu_run(tmp_path, monkeypatch, b, challenge="knapsack")
    # mutation: a bare check_deploy lets the ConnectionError end `talos run` in a traceback
    assert len(lookups) == 2 and sleeps and b.reached and rc == 1


def test_execute_job_pauses_when_no_gpu_is_available(tmp_path, monkeypatch, capsys):
    from talos.bench import BenchUnavailable

    class NoGpu(_ProbingBench):
        def select_hardware(self, challenge, chosen=None):
            raise BenchUnavailable("no Modal capacity for any of L40S, A100-80GB, H100")

    rc = _gpu_run(tmp_path, monkeypatch, NoGpu())
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    # mutation: letting BenchUnavailable fall into the blanket handler marks the job failed
    # (not resumable) and never says which GPUs were tried
    assert rc == 1 and st["status"] == "paused" and "L40S" in st["stop_reason"]
    assert st["hardware"] is None
    assert "no Modal capacity" in capsys.readouterr().err


def test_bench_hardware_class_uses_the_frozen_hardware():
    # mutation: ignoring `hardware` keys every GPU baseline as an L40S one, and every C3 CPU
    # baseline as a d3 one
    assert cli.bench_hardware_class("modal", "hypergraph", hardware="H100") == "gpu-H100"
    assert cli.bench_hardware_class("c3", "hypergraph", hardware="a100") == "c3-a100"
    assert cli.bench_hardware_class("c3", "knapsack", hardware="cpu-e2-4vcpu-16gb") == \
        "c3-cpu-e2-4vcpu-16gb"
    assert cli.bench_hardware_class("modal", "knapsack") == "cpu4-mem8192-x4"
    with pytest.raises(ValueError):
        cli.bench_hardware_class("c3", "knapsack")


def test_execute_job_freezes_a_c3_cpu_jobs_profile_and_hands_it_back_on_resume(tmp_path,
                                                                             monkeypatch):
    b = _ProbingBench("cpu-e2-4vcpu-16gb")
    rc = _gpu_run(tmp_path, monkeypatch, b, backend="c3", challenge="knapsack")
    assert rc == 1 and b.selections == [("knapsack", None)]
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert st["hardware"] == "cpu-e2-4vcpu-16gb" and b.seen_env == ["cpu-e2-4vcpu-16gb"]
    b2 = _ProbingBench("cpu-d3-4vcpu-16gb")
    rc = _gpu_run(tmp_path, monkeypatch, b2, backend="c3", challenge="knapsack",
                  extra=["--resume", run_dir.name])
    # mutation: probing again on resume can move the candidates to the other CPU
    assert rc == 1 and b2.selections == [("knapsack", "cpu-e2-4vcpu-16gb")]


def _forge_legacy_job(run_dir):
    """The pre-choice shape of state.json: a measured baseline and no hardware field."""
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    del st["hardware"]
    st["status"], st["baseline"] = "paused", {
        "name": "fake_base", "adoption": 1, "artifact_id": "a", "files": {"mod.rs": ""},
        "training": [], "holdout": []}
    (run_dir / "state.json").write_text(json.dumps(st), encoding="utf-8")


def test_a_c3_cpu_job_from_before_the_choice_is_frozen_to_d3_not_probed(tmp_path, monkeypatch):
    _gpu_run(tmp_path, monkeypatch, _ProbingBench("cpu-e2-4vcpu-16gb"), backend="c3",
             challenge="knapsack")
    run_dir = next((tmp_path / "runs").iterdir())
    _forge_legacy_job(run_dir)
    b2 = _ProbingBench("cpu-e2-4vcpu-16gb")
    _gpu_run(tmp_path, monkeypatch, b2, backend="c3", challenge="knapsack",
             extra=["--resume", run_dir.name])
    # mutation: probing here can land the candidates on e2 under a d3 baseline (keyed
    # c3-cpu-d3-4vcpu-16gb, the only CPU class there was)
    assert b2.selections == [("knapsack", "cpu-d3-4vcpu-16gb")]
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert st["hardware"] == "cpu-d3-4vcpu-16gb"


def test_a_modal_cpu_job_from_before_the_choice_is_frozen_to_nothing(tmp_path, monkeypatch):
    class NoChoice(_ProbingBench):
        def select_hardware(self, challenge, chosen=None):
            self.selections.append((challenge, chosen))
            return chosen  # the real ModalBench: a CPU challenge has no options

    _gpu_run(tmp_path, monkeypatch, NoChoice(), challenge="knapsack")
    run_dir = next((tmp_path / "runs").iterdir())
    _forge_legacy_job(run_dir)
    b2 = NoChoice()
    _gpu_run(tmp_path, monkeypatch, b2, challenge="knapsack", extra=["--resume", run_dir.name])
    # mutation: freezing to "the first option" of an empty list is an IndexError on resume
    assert b2.selections == [("knapsack", None)]
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert st["hardware"] is None
    assert "hardware_selected" not in (run_dir / "timeline.jsonl").read_text(encoding="utf-8")


def test_compile_uses_the_exported_gpu_or_probes_for_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.rs").write_text("fn x(){}")

    class B(_ProbingBench):
        def evaluate(self, request):
            return EvalResult(CompileResult(ok=True, artifact_id="a", output="ok"), [], None,
                              "not_won")

    b = B("L40S")
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: b)
    monkeypatch.setenv("TALOS_HARDWARE", "A100-80GB")
    assert cli.main(["compile", "--challenge", "hypergraph", "--dir", "src",
                     "--backend", "modal"]) == 0
    # mutation: ignoring TALOS_HARDWARE compiles the sandbox's candidate on a GPU of its own
    assert b.selections == [("hypergraph", "A100-80GB")]
    monkeypatch.delenv("TALOS_HARDWARE")
    assert cli.main(["compile", "--challenge", "hypergraph", "--dir", "src",
                     "--backend", "modal"]) == 0
    assert b.selections[-1] == ("hypergraph", None)


def test_compile_on_modal_refuses_a_stale_deploy_before_probing(tmp_path, monkeypatch, capsys):
    from talos.bench import BenchUnavailable
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    monkeypatch.delenv("TALOS_HARDWARE", raising=False)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.rs").write_text("fn x(){}")

    class Stale(_ProbingBench):
        def check_deploy(self):
            raise BenchUnavailable("built from different Talos code; run `talos deploy`")

    b = Stale("L40S")
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: b)
    rc = cli.main(["compile", "--challenge", "hypergraph", "--dir", "src", "--backend", "modal"])
    # mutation: skipping the check in `talos compile` compiles on stale code and bills a probe
    assert rc == 1 and b.selections == [] and "talos deploy" in capsys.readouterr().err


def test_a_job_from_before_the_gpu_choice_is_frozen_to_the_first_option_not_probed(tmp_path,
                                                                                  monkeypatch):
    b = _ProbingBench("H100")
    _gpu_run(tmp_path, monkeypatch, b)
    run_dir = next((tmp_path / "runs").iterdir())
    # forge the pre-change shape: a measured baseline and no gpu field at all
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    del st["hardware"]
    st["status"], st["baseline"] = "paused", {
        "name": "fake_base", "adoption": 1, "artifact_id": "a", "files": {"mod.rs": ""},
        "training": [], "holdout": []}
    (run_dir / "state.json").write_text(json.dumps(st), encoding="utf-8")
    b2 = _ProbingBench("H100")
    _gpu_run(tmp_path, monkeypatch, b2, extra=["--resume", run_dir.name])
    # mutation: probing here can land the candidates on an H100 under an L40S baseline
    assert b2.selections == [("hypergraph", "L40S")]
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert st["hardware"] == "L40S"


def test_a_fake_run_on_a_gpu_challenge_freezes_the_first_gpu_without_probing(tmp_path,
                                                                            monkeypatch):
    # The review of PR #20 found this path crashing: FakeBench.select_hardware returned None for a
    # GPU challenge, and hardware_class(spec, None) raises for one. No probe, no Modal: the
    # fake run keys its baseline under the first option, as a real run did before the choice.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TALOS_HARDWARE", raising=False)
    rc = cli.main(["run", "--challenge", "hypergraph", "--direction", "go",
                   "--budget-iterations", "1", "--yes", "--fake"])
    assert rc in (0, 1)  # won or exhausted, never a traceback
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert st["hardware"] == "L40S" and st["status"] in ("won", "exhausted")
    assert os.environ.get("TALOS_HARDWARE") == "L40S"


def test_the_gpu_probe_is_budget_checked_before_it_runs(tmp_path, monkeypatch, capsys):
    # invariant 5: the probe is a real compute call; a zero compute cap must stop before it
    b = _ProbingBench("A100-80GB")
    rc = _gpu_run(tmp_path, monkeypatch, b, extra=["--budget-compute-usd", "0"])
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    # mutation: dropping the check lets `--budget-compute-usd 0` pay for up to three probes
    assert b.selections == [] and rc == 1
    assert st["status"] == "exhausted" and st["stop_reason"] == "compute_usd"
    assert "compute_usd" in capsys.readouterr().err


def test_the_cpu_probe_on_c3_is_budget_checked_too(tmp_path, monkeypatch, capsys):
    # the C3 CPU probe bills like the GPU one; a zero compute cap must stop before it
    b = _ProbingBench("cpu-e2-4vcpu-16gb")
    rc = _gpu_run(tmp_path, monkeypatch, b, backend="c3", challenge="knapsack",
                  extra=["--budget-compute-usd", "0"])
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    # mutation: keying the check on is_gpu lets `--budget-compute-usd 0` pay for a CPU probe
    assert b.selections == [] and rc == 1
    assert st["status"] == "exhausted" and st["stop_reason"] == "compute_usd"
    assert "compute_usd" in capsys.readouterr().err


def test_the_gpu_probe_is_charged_to_compute_spend(tmp_path, monkeypatch):
    class Billing(_ProbingBench):
        def select_hardware(self, challenge, chosen=None):
            self.charged = 0.05
            return super().select_hardware(challenge, chosen)

        def cost_usd_since(self, mark):
            return getattr(self, "charged", 0.0) - mark

        def cost_mark(self):
            return getattr(self, "charged", 0.0)

    b = Billing("A100-80GB")
    _gpu_run(tmp_path, monkeypatch, b)
    run_dir = next((tmp_path / "runs").iterdir())
    st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    # mutation: not adding the bench's charge leaves the probe out of the printed estimate
    # and out of the next budget check
    assert st["spend"]["compute_usd"] == pytest.approx(0.05)


def test_fake_native_run_validates_its_best_and_wins(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    rc = fake_run(monkeypatch, ["--scoring", "native"])
    out = capsys.readouterr().out
    assert rc == 0 and "Status: won" in out and "scoring native" in out
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert json.loads((run_dir / "job.json").read_text())["scoring"] == "native"
    kinds = [json.loads(x)["kind"] for x in (run_dir / "timeline.jsonl").read_text().splitlines()]
    # mutation: execute_job never calling calibrate leaves state.scoring None, so the native
    # job silently runs metered
    assert "calibrated" in kinds and "validated" in kinds
    # the fake run keeps its calibration inside the run directory, never in ~/.talos
    assert list((run_dir / "calibration_cache").rglob("*.json"))


def test_run_defaults_to_native_scoring_and_metered_is_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    seen = []
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume: seen.append(spec.scoring) or 0)
    fake_run(monkeypatch)
    fake_run(monkeypatch, ["--scoring", "metered"])
    # mutation: a metered default (`args.scoring or "metered"`) gives metered first; ignoring
    # the flag gives native second
    assert seen == ["native", "metered"]


def test_resume_refuses_a_different_scoring_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 0)
    fake_run(monkeypatch)
    job_id = next((tmp_path / "runs").glob("*/job.json")).parent.name
    run_dir = tmp_path / "runs" / job_id
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    rc = cli.main(["run", "--resume", job_id, "--scoring", "metered"])
    assert rc == 2
    # mutation: dropping the resume check lets a native job continue as metered
    assert "started with native scoring; start a new job" in capsys.readouterr().err


def test_compile_is_metered_by_default_and_native_on_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "algorithm").mkdir()
    (tmp_path / "algorithm" / "mod.rs").write_text("fn x(){}")
    modes = []

    class Bench:
        def check_deploy(self):
            pass

        def _with_retry(self, call):
            return call()

        def select_hardware(self, challenge, chosen=None):
            return None

        def evaluate(self, request):
            modes.append(request.mode)
            from talos.bench import EvalResult
            from talos.types import CompileResult
            return EvalResult(CompileResult(True, "a", "ok"), [], [], "forced")
    monkeypatch.setattr(cli, "make_bench", lambda *a, **k: Bench())
    monkeypatch.setenv("TALOS_BACKEND", "modal")
    assert cli.main(["compile", "--challenge", "knapsack"]) == 0
    assert cli.main(["compile", "--challenge", "knapsack", "--native"]) == 0
    # mutation: a native default before the rollout's live checks (spec §8); the flag ignored
    # keeps a native job's agent on the slow path for good
    assert modes == ["metered", "native"]


def test_event_lines_for_validation():
    # mutation: a missing branch prints nothing for the validation events
    assert cli._event_line("validating", {}) == "validating on TIG's metered runtime"
    assert cli._event_line("validated", {"mean_rel_delta": 0.012}) == \
        "validated: +1.200% vs baseline on the metered runtime"
    assert cli._event_line("demoted", {"reason": "nondeterministic"}) == \
        "demoted by metered validation: nondeterministic"
    assert cli._event_line("iteration_done", {"outcome": "failed:validation",
                                               "runs_since_improvement": 2}) == \
        "rejected: failed metered validation (2 in a row)"


# ── --start, --search, --chains ───────────────────────────────────────────────

def _template_job(tmp_path, extra=(), challenge="energy_arbitrage"):
    """A fake template-start job (energy_arbitrage by default); returns (rc, run_dir)."""
    rc = cli.main(["run", "--fake", "--yes", "--challenge", challenge, "--direction",
                   "go", "--start", "template", "--budget-iterations", "1", *extra])
    return rc, next((tmp_path / "runs").glob("*/job.json")).parent


def test_template_start_runs_on_a_cpu_challenge(tmp_path, monkeypatch):
    # mutation: the CLI refusal left in place for knapsack
    monkeypatch.chdir(tmp_path)
    rc, run_dir = _template_job(tmp_path, challenge="knapsack")
    # 1 = exhausted: the fake first candidate is at baseline parity (k=1)
    assert rc == 1
    assert (run_dir / "challenge" / "mod.rs").read_text() == "fake knapsack/mod.rs\n"
    assert (run_dir / "package" / "diff_vs_template.patch").is_file()


def test_template_start_writes_and_quotes_nested_docs(tmp_path, monkeypatch):
    # mutation: writing docs without mkdir(parents=True) raises for nn/linear.rs
    monkeypatch.chdir(tmp_path)
    prompts = []
    script = cli._fake_script
    monkeypatch.setattr(cli, "_fake_script",
                        lambda s, u: prompts.append(s + u) or script(s, u))
    _template_job(tmp_path, challenge="neuralnet_optimizer")
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert (run_dir / "challenge" / "nn" / "linear.rs").read_text() == \
        "fake neuralnet_optimizer/nn/linear.rs\n"
    assert any("tig-challenges/src/neuralnet_optimizer/nn/linear.rs" in p for p in prompts)


def test_gpu_template_job_roots_carry_algorithm_cu(tmp_path, monkeypatch):
    # mutation: cli not writing challenge/template.cu, or the loop root without it
    monkeypatch.chdir(tmp_path)
    rc, run_dir = _template_job(tmp_path, challenge="hypergraph")
    assert (run_dir / "challenge" / "template.cu").read_text() == "// fake template.cu\n"
    # the fake whole-file reply replaces mod.rs only, so algorithm.cu is the template's
    first = run_dir / "iterations" / "0001"
    assert (first / "algorithm.cu").read_text() == "// fake template.cu\n"


def test_gpu_template_package_diff_names_algorithm_cu(tmp_path, monkeypatch):
    # mutation: the package diffing against {"mod.rs": template.rs} only shows algorithm.cu as
    # a new file, with no removed template line
    from talos.prompts import WHOLE_FILE_FORMAT
    monkeypatch.chdir(tmp_path)
    script = cli._fake_script

    def with_cu(system, user):
        reply = script(system, user)
        if WHOLE_FILE_FORMAT in system:
            reply += "<<<<<<< FILE algorithm.cu\n// changed\n>>>>>>> END FILE\n"
        return reply
    monkeypatch.setattr(cli, "_fake_script", with_cu)
    _, run_dir = _template_job(tmp_path, challenge="hypergraph")
    patch = (run_dir / "package" / "diff_vs_template.patch").read_text()
    assert "-// fake template.cu" in patch and "+// changed" in patch


def _stopped_gpu_template_job(tmp_path, monkeypatch):
    """A one-iteration fake hypergraph template job, made resumable for one more iteration."""
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)  # a resume reads the config; it names the fake provider
    rc, run_dir = _template_job(tmp_path, challenge="hypergraph")
    assert rc == 1
    spec = json.loads((run_dir / "job.json").read_text())
    spec["budget"]["iterations"] = 2
    (run_dir / "job.json").write_text(json.dumps(spec))
    st = json.loads((run_dir / "state.json").read_text())
    st["status"] = "cancelled"  # an exhausted job is finished; a cancelled one resumes
    (run_dir / "state.json").write_text(json.dumps(st))
    return run_dir


def test_gpu_template_resume_keeps_algorithm_cu(tmp_path, monkeypatch):
    # mutation: resume not passing template_cu to the Loop (_node_files calls _root_files
    # first, template_root raises, the job fails and iteration 2 writes no algorithm.cu).
    # Iteration 2's parent is node 1 = state.best, whose files come from state.best.files.
    run_dir = _stopped_gpu_template_job(tmp_path, monkeypatch)
    cli.main(["run", "--resume", run_dir.name])
    assert (run_dir / "iterations" / "0002" / "algorithm.cu").read_text() == \
        "// fake template.cu\n"


def test_gpu_template_resume_without_template_cu_fails_clearly(tmp_path, monkeypatch, capsys):
    # mutation: a KeyError/ValueError traceback, or a resume that runs with a root missing
    # the .cu, instead of a message naming the file
    from talos.state import JobStore
    run_dir = _stopped_gpu_template_job(tmp_path, monkeypatch)
    (run_dir / "challenge" / "template.cu").unlink()
    capsys.readouterr()
    rc = cli.main(["run", "--resume", run_dir.name])
    assert rc == 1
    assert "template.cu" in capsys.readouterr().err
    assert JobStore(run_dir).load().iteration == 1


def test_start_template_refuses_an_unsupported_challenge(tmp_path, monkeypatch, capsys):
    # mutation: dropping the TEMPLATE_START_FILES check fetches docs that do not exist and
    # creates a job before failing
    from talos.challenges import TEMPLATE_START_FILES
    monkeypatch.chdir(tmp_path)
    # every challenge has a list now; a challenge dropped from it in future must still refuse
    monkeypatch.delitem(TEMPLATE_START_FILES, "knapsack")
    monkeypatch.setattr(cli.FAKE_MAINNET, "fetch_challenge_docs",
                        lambda ch: pytest.fail("must not fetch for an unsupported challenge"))
    rc = cli.main(["run", "--fake", "--yes", "--start", "template", "--challenge", "knapsack",
                   "--direction", "go", "--budget-iterations", "1"])
    assert rc == 2
    assert "energy_arbitrage" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_chains_must_be_positive(tmp_path, monkeypatch, capsys):
    # mutation: dropping the check starts a job whose chain index divides by zero
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["run", "--fake", "--yes", "--challenge", "knapsack", "--direction", "go",
                   "--budget-iterations", "1", "--chains", "0"])
    assert rc == 2
    assert "--chains" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("flags", [["--start", "template"], ["--search", "anneal"],
                                   ["--chains", "2"]])
def test_resume_refuses_a_different_start_search_or_chains(tmp_path, monkeypatch, capsys, flags):
    # mutation: dropping one frozen-field check lets a resume change how the job searches
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    spec = json.loads((run_dir / "job.json").read_text())
    # mutation: a new job with no flags must keep today's behaviour
    assert (spec["start"], spec["search"], spec["chains"], spec["anneal"]) == \
        ("mainnet", "greedy", 1, None)
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name, *flags]) == 2
    assert "start a new job" in capsys.readouterr().err
    # mutation: refusing any given value; the job's own value is not a change
    same = {"--start": "mainnet", "--search": "greedy", "--chains": "1"}[flags[0]]
    assert cli.main(["run", "--resume", run_dir.name, flags[0], same]) == 1  # already won


def test_resume_without_the_flags_keeps_the_jobs_own(tmp_path, monkeypatch, capsys):
    # mutation: an argparse default of "mainnet" / "greedy" / 1 refuses this resume
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    rc, run_dir = _template_job(tmp_path, ["--search", "anneal", "--chains", "2"])
    assert rc == 1
    spec = json.loads((run_dir / "job.json").read_text())
    assert (spec["start"], spec["search"], spec["chains"]) == ("template", "anneal", 2)
    assert spec["anneal"] == {"t0": 0.05, "floor": 0.1, "stall": 5}
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) != 2
    assert "start a new job" not in capsys.readouterr().err


def test_template_start_refuses_codex_agentic(tmp_path, monkeypatch, capsys):
    # mutation: not passing the start to _codex_refused at any of its three call sites lets the
    # opt-in through for a template start
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TALOS_ALLOW_CODEX_AGENTIC", "1")
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: pytest.fail("fetched"))
    monkeypatch.setattr(cli, "execute_job", lambda *a, **kw: pytest.fail("must not start"))
    monkeypatch.setattr(cli, "make_provider", lambda *a, **kw: pytest.fail("must not call LLM"))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **kw: pytest.fail("must not compute"))
    argv = ["run", "--challenge", "energy_arbitrage", "--direction", "go", "--start", "template",
            "--budget-iterations", "1"]
    # a new job, agentic from the config
    save(tmp_path, Config(provider="codex-cli", model="m", mode="agentic", api_base=None), None)
    assert cli.main(argv + ["--yes"]) == 2
    assert "--start template" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
    # a new job, agentic picked in the wizard (the compute budget prompt comes first)
    save(tmp_path, Config(provider="codex-cli", model="m", mode="single-shot", api_base=None),
         None)
    assert cli.main(argv, ask=scripted(["", "agentic"])) == 2
    assert "--start template" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_template_start_resume_refuses_codex_agentic(tmp_path, monkeypatch, capsys):
    # mutation: the resume path calling _codex_refused without spec.start
    monkeypatch.chdir(tmp_path)
    rc, run_dir = _template_job(tmp_path)
    assert rc == 1
    spec = json.loads((run_dir / "job.json").read_text())
    spec.update(provider="codex-cli", model="m", mode="agentic")
    (run_dir / "job.json").write_text(json.dumps(spec))
    save(tmp_path, Config(provider="codex-cli", model="m", mode="agentic", api_base=None), None)
    monkeypatch.setenv("TALOS_ALLOW_CODEX_AGENTIC", "1")
    monkeypatch.setattr(cli, "execute_job", lambda *a, **kw: pytest.fail("must not start"))
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "--start template" in capsys.readouterr().err


def test_a_greedy_rejection_line_names_its_outcome():
    # mutation: rendering every probability-free accept event as "accepted (improved)", which
    # is what a greedy rejection (no probability, no temperature) printed before
    data = {"chain": 1, "node": 4, "parent": 2, "family": "lp", "score": -0.01,
            "accepted": False, "probability": None, "temperature": None,
            "outcome": "failed:score"}
    assert cli._event_line("accept", data) == \
        "chain 1 #4 <- #2 [lp] -1.000% vs mainnet: rejected (failed:score)"

def test_accept_and_reheat_event_lines():
    # mutation: swapping the improved and p/T branches, or dropping the rejected wording
    base = {"chain": 1, "node": 7, "parent": 3, "family": "local_search", "score": -0.0123,
            "accepted": True, "probability": 0.42, "temperature": 0.031}
    head = "chain 1 #7 <- #3 [local_search] -1.230% vs mainnet: "
    assert cli._event_line("accept", base) == head + "accepted (p=0.42, T=0.031)"
    improved = {**base, "score": 0.004, "probability": None, "temperature": None}
    assert cli._event_line("accept", improved) == \
        "chain 1 #7 <- #3 [local_search] +0.400% vs mainnet: accepted (improved)"
    rejected = {**base, "accepted": False, "probability": 0.05, "temperature": 0.02}
    assert cli._event_line("accept", rejected) == head + "rejected (p=0.05, T=0.020)"
    assert cli._event_line("reheat", {"chain": 2, "node": 5, "temperature": 0.05}) == \
        "chain 2 reheats from #5, T=0.050"
    assert cli._event_line("iteration_done", {"outcome": "accepted",
                                              "runs_since_improvement": 1}) == \
        "kept as the chain's next step (1 in a row)"


def test_fake_template_anneal_run_end_to_end(tmp_path, monkeypatch, capsys):
    # The fake reply is at baseline parity: a job best (a template start's floor is minus
    # infinity) and never a win (the margin is strict). Chains take turns: 0, then 1.
    # mutation: not storing the challenge files, not passing --chains through, or a fake
    # script with no whole-file reply (the edit fails, so no node is scored or accepted)
    from talos.state import JobStore
    monkeypatch.chdir(tmp_path)
    prompts = []
    script = cli._fake_script
    monkeypatch.setattr(cli, "_fake_script",
                        lambda system, user: prompts.append(user) or script(system, user))
    rc = cli.main(["run", "--fake", "--yes", "--challenge", "energy_arbitrage", "--direction",
                   "go", "--start", "template", "--search", "anneal", "--chains", "3",
                   "--budget-iterations", "2"])
    out = capsys.readouterr().out
    assert rc == 1
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    state = JobStore(run_dir).load()
    assert state.status == "exhausted"
    assert (run_dir / "challenge" / "README.md").is_file()
    assert (run_dir / "challenge" / "template.rs").is_file()
    assert [n.chain for n in state.nodes] == [0, 1]
    assert all(n.score == 0.0 and n.accepted for n in state.nodes)
    assert state.chains[2].current == 0 and state.chains[2].best == 0
    assert (run_dir / "package" / "lineage.json").is_file()
    assert "start template, anneal, 3 chain(s)" in out
    assert "chain 0 #1 <- #0 [" in out and "accepted (improved)" in out
    # mutation: not passing the stored docs to the Loop leaves them out of every prompt
    assert any("Fake challenge: maximise k." in u for u in prompts)
    assert not any("let k" in u for u in prompts)  # no byte of the baseline's source


def test_template_start_fetch_failure_creates_no_job(tmp_path, monkeypatch, capsys):
    # mutation: fetching after the run directory exists, or not catching MainnetError, leaves a
    # job behind (or a traceback) for a run that never started
    monkeypatch.chdir(tmp_path)

    def unreachable(ch):
        raise MainnetError("network error fetching template.rs")

    monkeypatch.setattr(cli.FAKE_MAINNET, "fetch_template", unreachable)
    rc = cli.main(["run", "--fake", "--yes", "--challenge", "energy_arbitrage", "--direction",
                   "go", "--start", "template", "--budget-iterations", "1"])
    assert rc == 1
    assert "mainnet unreachable" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_fake_template_anneal_chains_resume_reads_the_stored_challenge(
        tmp_path, monkeypatch, capsys):
    # A fake --start template --search anneal --chains 2 job runs 2 iterations, then resumes
    # with the iteration cap raised to 5. The fake baseline gets a marker comment, because the
    # fake whole-file reply (`let k = 1;`) is a candidate's code from iteration 1 on, so the
    # baseline's own text cannot tell a leak apart. After the first run the docs fetcher returns
    # different text, so the resumed prompts can only show the docs stored under the run
    # directory. (A resume fetches template.rs again at the pinned monorepo commit.)
    # mutation: a resume that refetches or drops the challenge docs, or drops the template, a
    # resume that edits the baseline's files instead of the template root, or one that restarts
    # the iteration count at 1
    from talos.state import JobStore
    marker = "FAKE_BASELINE_ONLY"
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)  # a resume reads the config; it names the fake provider
    monkeypatch.setattr(cli.FAKE_MAINNET, "fetch_algorithm_files",
                        lambda ch, name: {"mod.rs": f"fn solve() {{ let k = 1; }} // {marker}\n"})
    prompts = []
    script = cli._fake_script
    monkeypatch.setattr(cli, "_fake_script",
                        lambda system, user: prompts.append((system, user)) or script(system, user))
    rc = cli.main(["run", "--fake", "--yes", "--challenge", "energy_arbitrage", "--direction",
                   "go", "--start", "template", "--search", "anneal", "--chains", "2",
                   "--budget-iterations", "2"])
    assert rc == 1
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    assert [n.id for n in JobStore(run_dir).load().nodes] == [1, 2]
    monkeypatch.setattr(cli.FAKE_MAINNET, "fetch_challenge_docs",
                        lambda ch: {"README.md": "REFETCHED DOC\n"})
    spec = json.loads((run_dir / "job.json").read_text())
    spec["budget"]["iterations"] = 5
    (run_dir / "job.json").write_text(json.dumps(spec))
    st = json.loads((run_dir / "state.json").read_text())
    st["status"] = "cancelled"  # an exhausted job is finished; a cancelled one resumes
    (run_dir / "state.json").write_text(json.dumps(st))
    first = len(prompts)
    capsys.readouterr()
    rc = cli.main(["run", "--resume", run_dir.name])
    state = JobStore(run_dir).load()
    # Iteration 3 is chain 0's turn, built on node 1 (k=1, baseline parity); its k=2 beats the
    # baseline on training and held-out, so the resumed job wins there.
    assert (rc, state.status, state.iteration) == (0, "won", 3)
    assert [(n.id, n.chain, n.parent) for n in state.nodes] == [(1, 0, 0), (2, 1, 0), (3, 0, 1)]
    resumed = prompts[first:]
    proposals = [(s, u) for s, u in resumed if s.startswith("You are a research engineer")]
    edits = [(s, u) for s, u in resumed if s.startswith("You are editing")]
    assert len(proposals) == 1 and len(edits) == 1
    assert all("Fake challenge: maximise k." in u for s, u in proposals + edits)
    assert "pub fn solve_challenge(" in proposals[0][0]
    for system, user in prompts:
        assert marker not in system + user and "REFETCHED" not in system + user


def stopped_job(tmp_path, monkeypatch, status="exhausted", stop_reason="iterations",
                extra=(), **state_kw):
    """A fake job whose 5-iteration cap is spent, stopped before any candidate; once resumed
    the fake provider wins on its first iteration."""
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, extra) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    st = json.loads((run_dir / "state.json").read_text())
    st.update(status=status, stop_reason=stop_reason, best=None, iteration=5, confirmed=[],
              hypotheses=[], nodes=[], chains=[])
    st["spend"]["iterations"] = 5
    st.update(state_kw)
    (run_dir / "state.json").write_text(json.dumps(st))
    return run_dir


def test_resume_continues_an_exhausted_job_with_added_iterations(tmp_path, monkeypatch, capsys):
    # mutation: the old refusal of exhausted jobs exits 1 "nothing to resume"
    # mutation: rewriting job.json to hold the new budget breaks invariant 1's write-once file
    run_dir = stopped_job(tmp_path, monkeypatch)
    spec_bytes = (run_dir / "job.json").read_bytes()
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "2"]) == 0
    assert "Status: won" in capsys.readouterr().out
    st = json.loads((run_dir / "state.json").read_text())
    assert st["spend"]["iterations"] == 6
    [entry] = st["budget_extensions"]
    assert entry["previous_status"] == "exhausted" and entry["add"]["iterations"] == 2
    ev = [json.loads(ln) for ln in (run_dir / "timeline.jsonl").read_text().splitlines()]
    assert [e["budget"]["iterations"] for e in ev if e["kind"] == "budget_extended"] == [7]
    assert "resumed" in [e["kind"] for e in ev]
    assert (run_dir / "job.json").read_bytes() == spec_bytes


def test_exhausted_resume_without_added_budget_names_the_flag(tmp_path, monkeypatch, capsys):
    # mutation: dropping the hint leaves the user with "nothing to resume" and no way forward
    run_dir = stopped_job(tmp_path, monkeypatch)
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    out = capsys.readouterr().out
    assert "already exhausted; nothing to resume" in out and "--add-iterations" in out


def test_exhausted_resume_with_an_unknown_reason_does_not_crash(tmp_path, monkeypatch, capsys):
    # Review Focus 3. mutation: ADD_FLAGS[state.stop_reason] raises KeyError
    run_dir = stopped_job(tmp_path, monkeypatch, stop_reason="budget")
    assert cli.main(["run", "--resume", run_dir.name]) == 1
    assert "already exhausted; nothing to resume" in capsys.readouterr().out


def test_extension_still_out_of_budget_is_refused_untouched(tmp_path, monkeypatch, capsys):
    # mutation: saving the shift or the entry before extend() checks changes state.json
    # 4 h cap, started 5 h ago, saved just now (no pause to shift): still out on hours
    run_dir = stopped_job(tmp_path, monkeypatch, stop_reason="hours",
                          extra=("--budget-hours", "4"))
    st = json.loads((run_dir / "state.json").read_text())
    st["spend"]["started_at"] = time.time() - 5 * 3600
    (run_dir / "state.json").write_text(json.dumps(st))
    before = {n: (run_dir / n).read_bytes() for n in ("state.json", "timeline.jsonl")}
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "2"]) == 2
    err = capsys.readouterr().err
    assert "still out of budget on hours" in err and "--add-hours" in err
    assert before == {n: (run_dir / n).read_bytes() for n in before}


def test_hours_exhausted_job_continues_with_added_hours(tmp_path, monkeypatch, capsys):
    # Review Focus 5. 4 h cap, started 5 h ago, last saved 1 h ago: the shift gives 4 h used,
    # and +1 h leaves room.
    # mutation: not shifting started_at for an exhausted job counts 5 h, refused with exit 2
    run_dir = stopped_job(tmp_path, monkeypatch, stop_reason="hours",
                          extra=("--budget-hours", "4"))
    st = json.loads((run_dir / "state.json").read_text())
    st["spend"]["iterations"] = 0  # only hours is spent
    st["spend"]["started_at"] = time.time() - 5 * 3600
    (run_dir / "state.json").write_text(json.dumps(st))
    saved = time.time() - 3600
    os.utime(run_dir / "state.json", (saved, saved))
    assert cli.main(["run", "--resume", run_dir.name, "--add-hours", "1"]) == 0


def test_add_flag_for_a_cap_the_job_never_had_is_refused(tmp_path, monkeypatch, capsys):
    # mutation: creating a new hours cap the user never set
    run_dir = stopped_job(tmp_path, monkeypatch)
    before = (run_dir / "state.json").read_bytes()
    assert cli.main(["run", "--resume", run_dir.name, "--add-hours", "1"]) == 2
    assert "--budget-hours" in capsys.readouterr().err
    assert (run_dir / "state.json").read_bytes() == before


def test_add_flags_need_resume(tmp_path, monkeypatch, capsys):
    # mutation: ignoring --add-iterations on a new job silently drops the user's budget
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch, extra=("--add-iterations", "2")) == 2
    assert "only applies with --resume" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_extension_raises_the_cap_of_a_cancelled_job(tmp_path, monkeypatch, capsys):
    # mutation: applying extensions to exhausted jobs only resumes this one at 5/5, exit 1
    run_dir = stopped_job(tmp_path, monkeypatch, status="cancelled",
                          stop_reason="user requested stop")
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 0
    st = json.loads((run_dir / "state.json").read_text())
    assert st["budget_extensions"][0]["previous_status"] == "cancelled"


def test_hard_killed_job_is_extended_without_a_clock_shift(tmp_path, monkeypatch, capsys):
    # mutation: shifting started_at for every status moves a hard-killed job's clock by the
    # hour since its last save
    run_dir = stopped_job(tmp_path, monkeypatch, status="researching", stop_reason=None)
    st = json.loads((run_dir / "state.json").read_text())
    started = st["spend"]["started_at"]
    saved = time.time() - 3600
    os.utime(run_dir / "state.json", (saved, saved))
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 0
    st = json.loads((run_dir / "state.json").read_text())
    assert st["spend"]["started_at"] == started
    assert st["budget_extensions"][0]["previous_status"] == "researching"


def test_job_exhausted_before_its_baseline_measures_it_once(tmp_path, monkeypatch, capsys):
    # mutation: an extension path that skips measure_baseline when the status was exhausted
    run_dir = stopped_job(tmp_path, monkeypatch, baseline=None)
    n_before = len((run_dir / "timeline.jsonl").read_text().splitlines())
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 0
    new = [json.loads(ln)["kind"]
           for ln in (run_dir / "timeline.jsonl").read_text().splitlines()[n_before:]]
    assert new.count("baseline_ready") == 1


def test_add_flags_on_a_won_job_record_nothing(tmp_path, monkeypatch, capsys):
    # Review Focus 2. mutation: extending (and saving) before the won refusal records an entry
    # on a job that never runs again
    # mutation: extending before the won refusal without saving still prints "Budget extended"
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert fake_run(monkeypatch) == 0
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    capsys.readouterr()
    assert cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"]) == 1
    out = capsys.readouterr().out
    assert "already won" in out and "Budget extended" not in out
    assert json.loads((run_dir / "state.json").read_text()).get("budget_extensions", []) == []


def test_hard_killed_job_saves_its_extension_before_the_run_starts(tmp_path, monkeypatch):
    # mutation: saving a non-stopped job's extension only when the loop next saves loses it to
    # a kill during the hardware probe, while timeline.jsonl already says budget_extended
    class Killed(BaseException):
        pass

    run_dir = stopped_job(tmp_path, monkeypatch, status="researching", stop_reason=None)

    def kill(*a, **k):
        raise Killed()

    monkeypatch.setattr(cli, "freeze_hardware", kill)
    with pytest.raises(Killed):
        cli.main(["run", "--resume", run_dir.name, "--add-iterations", "1"])
    st = json.loads((run_dir / "state.json").read_text())
    assert [e["add"]["iterations"] for e in st["budget_extensions"]] == [1]


def test_talos_report_writes_and_prints_path(tmp_path, monkeypatch, capsys):
    from tests.run_fixture import make_run
    monkeypatch.chdir(tmp_path)
    make_run(tmp_path / "runs" / "J", [{"iteration": 1, "against": 0, "mean_rel_delta": 0.01}])
    assert cli.main(["report", "J"]) == 0
    assert (tmp_path / "runs" / "J" / "package" / "report.html").is_file()
    assert "report.html" in capsys.readouterr().out


def test_talos_report_unknown_job_leaves_no_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["report", "NOPE"]) == 1
    assert not (tmp_path / "runs" / "NOPE").exists()
    assert "no job NOPE" in capsys.readouterr().err


def test_talos_report_job_without_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "K").mkdir(parents=True)
    assert cli.main(["report", "K"]) == 1
    assert "has no state.json" in capsys.readouterr().err
    assert list((tmp_path / "runs" / "K").iterdir()) == []


def test_wizard_defaults_to_knapsack(tmp_path, monkeypatch):
    # mutation: the default reverted to vehicle_routing
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "execute_job", lambda spec, store, cfg, resume: 0)  # no bench, no LLM
    seen = []

    def ask(prompt, default=None, secret=False):
        seen.append(prompt)
        if prompt.startswith("Challenge"):
            return ""
        if prompt.startswith("Direction"):
            return "go"
        return default or "1"
    rc = cli.main(["run", "--budget-iterations", "1", "--budget-usd", "1"], ask=ask)
    assert rc == 0
    assert any(p.startswith("Challenge") for p in seen)
    job = json.loads(next((tmp_path / "runs").glob("*/job.json")).read_text())
    assert job["challenge"] == "knapsack"


def test_run_warns_before_an_untested_challenge(tmp_path, monkeypatch, capsys):
    # mutation: the warning removed, or printed for tested challenges
    monkeypatch.chdir(tmp_path)
    base = ["run", "--direction", "go", "--budget-iterations", "1", "--yes", "--fake"]
    assert cli.main([*base, "--challenge", "vehicle_routing"]) == 0
    err = capsys.readouterr().err
    assert "warning: vehicle_routing has never been run end to end with Talos" in err
    assert cli.main([*base, "--challenge", "knapsack"]) == 0
    assert "warning: knapsack" not in capsys.readouterr().err


def _walk(parser):
    import argparse
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helped = {a.dest for a in action._choices_actions if a.help}
            for name, sub in action.choices.items():
                yield f"subcommand {name}", name in helped
                yield from _walk(sub)
        elif action.help is not argparse.SUPPRESS:
            yield f"{parser.prog} {'/'.join(action.option_strings) or action.dest}", bool(action.help)


def test_every_subcommand_and_flag_has_help():
    # mutation: deleting the help= of any one flag or subcommand fails this
    missing = [where for where, ok in _walk(cli.build_parser()) if not ok]
    assert missing == []


def test_run_help_groups_flags_and_shows_examples(capsys):
    # mutation: dropping the argument groups or the epilog fails this
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    out = capsys.readouterr().out
    for title in ("what to research", "budget", "search", "resume"):
        assert title + ":" in out
    assert "Examples:" in out
    assert "--add-iterations" in out.split("Examples:")[1]


def test_challenge_flag_lists_its_choices(capsys):
    # mutation: dropping choices= hides the valid values
    with pytest.raises(SystemExit):
        cli.main(["compile", "--help"])
    assert "{satisfiability,vehicle_routing,knapsack," in capsys.readouterr().out


def anthropic_config(root):
    save(root, Config(provider="anthropic", model="claude-opus-5", mode="single-shot",
                      api_base=None), None)


def stub_run_effects(monkeypatch):
    """Record execute_job and recommend_image_update; neither can reach a bench or Docker."""
    calls = {"execute": [], "image": []}
    monkeypatch.setattr(cli, "fetch_challenge_info", lambda name: knapsack_info())
    monkeypatch.setattr(cli, "recommend_image_update", lambda c: calls["image"].append(c))
    monkeypatch.setattr(cli, "execute_job",
                        lambda spec, store, cfg, resume, add=None:
                        calls["execute"].append(spec.job_id) or 0)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return calls


DECLINE_ARGV = ["run", "--challenge", "knapsack", "--direction", "go", "--budget-iterations", "3",
                "--budget-compute-usd", "5", "--hyperparameters", "none", "--nonces", "2"]


def test_declining_the_summary_writes_nothing_and_builds_no_bench(tmp_path, monkeypatch, capsys):
    # mutation: the confirmation placed after JobStore(...) (runs/ exists), after
    # recommend_image_update (image recorder non-empty), or `return 0` dropped (execute called)
    monkeypatch.chdir(tmp_path)
    anthropic_config(tmp_path)
    calls = stub_run_effects(monkeypatch)
    inner, prompts = scripted(["", "n"]), []

    def ask(prompt, default=None, secret=False):
        prompts.append(prompt)
        return inner(prompt, default, secret)
    rc = cli.main(DECLINE_ARGV, ask=ask)
    assert rc == 0
    assert prompts[-1] == "Start this job?"
    assert "Not started." in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
    assert calls == {"execute": [], "image": []}


def test_eof_at_the_confirmation_is_no(tmp_path, monkeypatch, capsys):
    # mutation: confirm returns the default on EOF
    monkeypatch.chdir(tmp_path)
    anthropic_config(tmp_path)
    calls = stub_run_effects(monkeypatch)
    inner = scripted(["", "n"])

    def ask(prompt, default=None, secret=False):
        if prompt == "Start this job?":
            raise EOFError
        return inner(prompt, default, secret)
    rc = cli.main(DECLINE_ARGV, ask=ask)
    assert rc == 0
    assert "Not started." in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
    assert calls == {"execute": [], "image": []}


def test_accepting_the_summary_starts_the_job(tmp_path, monkeypatch, capsys):
    # mutation: the id re-stamped after the summary, so the printed results path differs from
    # the job that ran; or the job not started on yes
    monkeypatch.chdir(tmp_path)
    anthropic_config(tmp_path)
    calls = stub_run_effects(monkeypatch)
    assert cli.main(DECLINE_ARGV, ask=scripted(["", "y"])) == 0
    assert len(calls["execute"]) == 1 and len(calls["image"]) == 1
    job_id = calls["execute"][0]
    assert (tmp_path / "runs" / job_id / "job.json").exists()
    out = capsys.readouterr().out
    assert f"  results:   {tmp_path.resolve() / 'runs' / job_id}\n" in out


def test_summary_labels_codex_cli_spend_not_tracked(tmp_path, monkeypatch, capsys):
    # mutation: the summary hard-codes "(LLM spend unpriced)" for every unpriced provider, so
    # codex-cli reads "unpriced" here while its status line says "not tracked"
    monkeypatch.chdir(tmp_path)
    save(tmp_path, Config(provider="codex-cli", model="gpt-5.5", mode="single-shot",
                          api_base=None), None)
    calls = stub_run_effects(monkeypatch)
    argv = DECLINE_ARGV + ["--yes"]
    assert cli.main(argv, ask=scripted([])) == 0
    out = capsys.readouterr().out
    assert "(LLM spend not tracked)" in out
    assert "(LLM spend unpriced)" not in out
    assert len(calls["execute"]) == 1


def test_yes_prints_the_summary_and_never_asks(tmp_path, monkeypatch, capsys):
    # mutation: the confirmation not skipped under --yes
    monkeypatch.chdir(tmp_path)
    anthropic_config(tmp_path)
    calls = stub_run_effects(monkeypatch)

    def ask(prompt, default=None, secret=False):
        raise AssertionError(f"--yes must not prompt: {prompt}")
    rc = cli.main([*DECLINE_ARGV, "--yes"], ask=ask)
    assert rc == 0 and len(calls["execute"]) == 1
    out = capsys.readouterr().out
    assert "Job summary" in out and "knapsack (CPU, tested)" in out


def test_summary_lists_each_budget_cap_and_billing(tmp_path, monkeypatch, capsys):
    # mutation: a cap dropped from the summary, or a truthiness check hiding the zero cap
    monkeypatch.chdir(tmp_path)
    anthropic_config(tmp_path)
    stub_run_effects(monkeypatch)
    argv = ["run", "--challenge", "knapsack", "--direction", "go", "--hyperparameters", "none",
            "--nonces", "2", "--budget-usd", "7", "--budget-hours", "2",
            "--budget-iterations", "9", "--yes"]
    assert cli.main([*argv, "--budget-compute-usd", "3"]) == 0
    out = capsys.readouterr().out
    for text in ("LLM $7.00", "2 h", "9 iterations", "compute $3.00",
                 "billed per container second, no fixed per-job overhead"):
        assert text in out
    assert cli.main([*argv, "--budget-compute-usd", "0"]) == 0
    assert "compute $0.00" in capsys.readouterr().out


def test_summary_shows_the_challenge_warning(tmp_path, monkeypatch, capsys):
    # mutation: the warning left out of the summary
    monkeypatch.chdir(tmp_path)
    anthropic_config(tmp_path)
    stub_run_effects(monkeypatch)
    monkeypatch.setattr(cli, "fetch_challenge_info",
                        lambda name: type("I", (), {"id": CHALLENGES[name].id,
                                                    "is_gpu": CHALLENGES[name].is_gpu,
                                                    "tracks": ["n=1"], "max_fuel": 7})())
    assert cli.main(["run", "--challenge", "vehicle_routing", "--direction", "go",
                     "--budget-iterations", "1", "--hyperparameters", "none", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "untested" in out and "never been run end to end" in out


def test_version_flag_prints_the_package_version(capsys):
    import importlib.metadata

    import talos
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"talos {talos.__version__}\n"
    assert importlib.metadata.version("talos-tig") == talos.__version__


@pytest.mark.parametrize("missing", ["mean_rel_se", "noise_ratio", "noise_band"])
def test_event_line_scored_ok_with_a_null_field_has_no_noise(missing):
    # mutation: .get(..., 0) invents "±0.000%, 0.0x noise, None" or raises TypeError on None
    data = {**SCORED, "mean_rel_se": 0.0031, "noise_ratio": 1.29, "noise_band": "marginal",
            "noise_state": "ok", missing: None}
    assert cli._event_line("scored", data, width=200) == "scored +0.400% vs baseline (" + TAIL
    del data[missing]
    assert cli._event_line("scored", data, width=200) == "scored +0.400% vs baseline (" + TAIL


def test_resume_refuses_a_job_carrying_the_leak(tmp_path, monkeypatch, capsys):
    # mutation: a resume keeps scoring candidates against gridflow, on the same void comparison
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert cli.main(["run", "--fake", "--yes", "--challenge", "energy_arbitrage", "--direction",
                     "go", "--budget-iterations", "1"]) in (0, 1)
    run_dir = next((tmp_path / "runs").glob("*/job.json")).parent
    state = json.loads((run_dir / "state.json").read_text())
    state["status"] = "paused"
    state["baseline"]["files"] = {"mod.rs": "let v = serde_json::to_value(challenge);\n"}
    (run_dir / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(cli, "make_bench", lambda *a, **kw: pytest.fail("must not start compute"))
    assert cli.main(["run", "--resume", run_dir.name]) == 2
    assert "is void" in capsys.readouterr().err


def test_job_summary_says_which_algorithms_are_never_the_baseline(tmp_path, monkeypatch, capsys):
    # mutation: silently swapping the baseline leaves the user reading its scores as gridflow's
    monkeypatch.chdir(tmp_path)
    fake_config(tmp_path)
    assert cli.main(["run", "--fake", "--yes", "--challenge", "energy_arbitrage", "--direction",
                     "go", "--budget-iterations", "1"]) in (0, 1)
    out = capsys.readouterr().out
    assert "never first_energy, gridflow, super_energy, titan_killer" in out
    assert "never with mainnet scores" in out


def test_leak_rejection_reads_in_the_run_output(capsys):
    # mutation: no OUTCOME_TEXT entry prints the raw failed:leak
    assert "causality" in cli.OUTCOME_TEXT["failed:leak"]
