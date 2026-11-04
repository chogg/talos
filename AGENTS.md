# Agent guide

Read this first, then follow the links. This file is a router: it says which
document is authoritative, what must not be broken, and what needs a human.
It restates as little as possible, because a second copy of a fact is a copy
that goes stale.

## What this project is

Talos is a command-line tool that runs an LLM-driven research loop against one
TIG challenge. It measures the current top-adoption mainnet algorithm once as
the baseline, then has an LLM propose and edit candidates, compiles and scores
each one with TIG's own `tig-runtime` and `tig-verifier` inside Modal
containers in the user's own account, and stops when a candidate beats the
baseline on training and then held-out nonces, or when the budget runs out.
Its output is a local, submit-ready package under `runs/<job_id>/package/`;
the user submits it to TIG themselves. Talos is not a hosted service and not a
swarm, it never submits on the user's behalf, and LLM-authored code never
executes outside a container. Only the `local` backend runs that container on
the user's own machine, with networking off, capabilities dropped, and CPU and
memory limits (`talos/local_transport.py::run_args`). C3 (cthree.cloud) is the
alternative hosted compute backend chosen at `talos setup`, running the same
compile-and-score work as one batch job per iteration instead of a Modal
container call. The local backend (Docker on the user's machine) is the third,
chosen the same way and running the same batch job through a Docker transport.

## Source of truth, in order

When two documents disagree, the one higher in this list wins.

1. **The code** and its configuration. If a document describes behaviour the
   code does not have, the document is wrong. The per-challenge facts (pinned
   monorepo commit, dev image tag, beat rule, hardware class) are code:
   `talos/challenges.py`.
2. **`README.md`** (quickstart), **`docs/usage.md`** (full command reference),
   **`docs/troubleshooting.md`**, **`docs/windows.md`** and **`docs/maintainers.md`** —
   setup and the commands you run day to day.
3. **`docs/architecture.md`**, **`docs/compute-backends.md`** and
   **`docs/json-output.md`** — the research loop, the guards between build and
   scoring, the agentic sandbox, the baseline cache, how Modal, C3 and local
   Docker are used, C3 and local timings, the dev images, and the versioned
   JSON output of `talos status` and `talos show`.
4. **`docs/ai/`** — *not authoritative*. Design specs and plans written by
   agents during development, kept for the reasoning behind decisions. They
   are not updated as the code changes. See `docs/ai/README.md`. Agents
   writing a new design spec put it in `docs/ai/specs/`; implementation
   plans go in `docs/ai/plans/`; nowhere else.

## Invariants

These are silent until violated. The test suite covers only the parts named
below as tested; the rest is easy to break while believing you are making
progress.

1. **Baseline and candidate are always scored on identical nonces, fuel, and
   hardware class.** A delta across any of those is meaningless and
   `talos/scoring.py::beats` cannot tell. The nonce sets, fuel, tracks and
   monorepo commit are frozen into `job.json` at job start
   (`talos/state.py::JobSpec`), and `job.json` is write-once
   (`talos/state.py::JobStore.write_spec`). The hardware class is part of the
   baseline cache key (`talos/challenges.py::hardware_class`,
   `talos/baseline.py::cache_key`); a cached baseline measured under different
   hardware or fuel is a different key, never a hit. On Modal the class also
   carries how many nonces share one container
   (`talos/challenges.py::modal_workers`): a batch of four on four cores times
   differently from one nonce alone. The local backend's class
   is `talos/challenges.py::local_hardware_class`, which carries the host name
   and the container's CPU and memory limits, so changing either at
   `talos setup` invalidates every local baseline. On Modal and C3 a GPU
   challenge's GPU, and on C3 a CPU challenge's profile, is chosen once, by a
   capacity probe at job start (`talos/cli.py::freeze_hardware`), frozen in
   `state.json` (`talos/state.py::JobState`) and handed back on every resume;
   the fallback order is `talos/challenges.py::MODAL_GPUS`,
   `talos/challenges.py::C3_GPU_CLASSES` and
   `talos/challenges.py::C3_CPU_PROFILES`. A fallback per call, such as Modal's
   own `gpu=[...]` list, would score candidates on hardware the baseline never
   ran on, which is why `talos/modal_app/talos_bench.py::register` pins each function
   to one GPU.
   Both sides also run with identical hyperparameters, except in a `--start template`
   or `--base-algorithm`/`--base-dir` job (the third asymmetry below): the per-track map is
   frozen into `job.json` (`talos/state.py::JobSpec`) together with the
   algorithm it belongs to, every request takes it from there
   (`talos/loop.py::Loop._request`, `talos/baseline.py::resolve_baseline`), and
   `talos/inside.py::run_nonce` passes it to `tig-runtime`.
   With `--track`, the loop slices both sides with the same `NonceSet` lists
   (`talos/scoring.py::select`, `talos/scoring.py::focus_sets`); the guard compares
   the other tracks' training nonces against the cached baseline training results
   for those same nonces, and the confirmation rule is
   `talos/scoring.py::beats_focused`.
   There are three deliberate asymmetries. The first is the per-nonce timeout: the
   baseline runs under the flat `talos/inside.py::NONCE_TIMEOUT_S`, a candidate under the
   tighter per-track cap from `talos/loop.py::Loop._timeouts`. A nonce over the cap is a
   `timeout` error, never a quality, so it can only fail a candidate, never flatter one.
   Likewise, a candidate's C3 or local job stops once its training errors are
   over the error ceiling (`talos/c3_job.py::_score`): the nonces it never started are
   `not_run` errors, so a candidate scored on fewer nonces is always one that had already
   failed. The baseline, calibration and validation runs never stop early.
   With `--scoring native` there is a second, bounded asymmetry. Research candidates run
   natively (`talos/inside.py::run_nonce_native`) under a per-track time budget calibrated
   from the baseline's metered fuel (`talos/calibration.py::budgets_us`), while the baseline
   ran metered. A native score never decides anything by itself: a candidate becomes best
   only after `talos/loop.py::Loop._validate_and_finish` scores it on TIG's metered runtime,
   on the same nonces, fuel and hardware as the baseline, and held-out confirmation runs only
   on those metered results. The calibration key (`talos/calibration.py::calibration_key`)
   carries the hardware class, both pins, the baseline's code, the native runner's digest
   (`talos/native_runner.py::runner_digest`) and the baseline's hyperparameters.
   The third: with `--start template` the candidates run with hyperparameters `None`
   while the baseline keeps the mainnet map
   (`talos/loop.py::Loop._candidate_hyperparameters`), because a solver written from the
   empty template is not tuned for the mainnet algorithm's values. The package then ships
   no `hyperparameters.json`. This changes the conditions a template candidate's delta is
   measured under, and it is a decision of the repo owner (design spec 2026-10-19 §5). A
   `--base-algorithm`/`--base-dir` job (`start == "base"`) has the same asymmetry, for the same
   reason: its candidates and its root run with `None`, the baseline stays mainnet's top
   algorithm with the mainnet map, and the root is scored once, metered, before any LLM call
   (`talos/loop.py::Loop._measure_base`).
2. **The job's `rand_hash` lives in `job.json` and nowhere the agent can
   read.** It seeds every nonce; an LLM that sees it can tune to the exact
   nonces it is scored on. It is stripped from the spec the prompts see
   (`talos/state.py::JobSpec.redacted`), redacted from bench output
   (`talos/bench.py::_redact`), and the agentic worktree is deliberately
   placed outside `runs/` (`talos/agentic.py::prepare_worktree`). Tests grep
   the prompts and timeline (`tests/test_loop.py`), the hand-back package and
   zip (`tests/test_package.py`), the package's report.html
   (`tests/test_report.py`) and the redacted bench output
   (`tests/test_bench.py`) for the hash. Nothing greps the agentic mode's
   copied transcript (`talos/agentic.py::_copy_transcript`) for it.
   With `--start template` the baseline's source never reaches the LLM either: the
   root of every chain is the empty template, and the baseline's files are not read
   to build a prompt, a timeline entry, an iteration directory or an agentic
   worktree (`talos/loop.py::Loop._root_files`; tested in `tests/test_loop.py`).
   The agentic worktree of a template job also holds no baseline code. Codex is
   refused for template starts even with `TALOS_ALLOW_CODEX_AGENTIC=1`
   (`talos/agentic.py::codex_agentic_refused`).
3. **The monorepo commit and dev image tag are pins, and both are part of
   every cache key.** `MONOREPO_REF` and `DEV_IMAGE_TAG` in
   `talos/challenges.py` enter the Modal artifact hash
   (`talos/modal_app/talos_bench.py::content_hash`) and the baseline cache key. The
   exit codes in `talos/inside.py` were read from `tig-runtime` at that
   commit. Bumping either pin silently changes what every cached baseline
   meant, and the Modal app must be redeployed (`talos deploy`) before any run.
   The baseline cache key also includes the hyperparameter map
   (`talos/baseline.py::effective_hyperparameters`); a key computed without one
   is unchanged from before the map existed.
   The calibration key carries both pins too (`talos/calibration.py::calibration_key`).
   The native runner's toolchain is `talos/native_runner.py::TOOLCHAIN`, the one `build_so`
   uses at the pin, and must move with `MONOREPO_REF`. A native artifact id also carries
   `talos/native_runner.py::runner_digest` (templates, toolchain, build profile), so a
   runner change never reuses a binary cached on the Modal volume.
4. **An edit outside the algorithm files fails the whole iteration; its
   in-scope blocks are never applied either.** `talos/edits.py::apply_edit_response`
   (and `talos/edits.py::apply_whole_file_response` for a template start's whole-file
   replies) reports rejected paths and `talos/loop.py::Loop.iterate` fails the iteration
   on any of them, including in a compile-fix round. The only spelling accepted
   besides the bare file name is the candidate's own directory, with or without the
   directories above it (`.../talos_cand/<name>`, `talos_cand/<name>`;
   `talos/edits.py::_resolve`), because that is how the compiler prints it; another
   algorithm's directory with the same basename is still rejected. With `codex-cli` the
   sandbox settings are ignored, so `talos/agentic.py::read_back` is the only
   enforcement, which is why agentic codex is opt-in behind
   `TALOS_ALLOW_CODEX_AGENTIC`.
5. **Every budget dimension is checked before a call, never only after, and
   zero is a real cap.** `talos/budget.py::exhausted` uses `>=`. Two mechanisms
   apply it: `talos/loop.py::_BudgetedBench` wraps the baseline's compute calls,
   and an iteration's own calls go through `talos/loop.py::Loop._bench_evaluate`,
   which checks the budget and then charges the spend inline. The
   capacity probe that runs before the baseline (a GPU, or a C3 CPU profile)
   is a compute call too:
   `talos/cli.py::freeze_hardware` checks the budget before it and charges what
   the bench estimated for it. A job with `--budget-compute-usd 0` must stop
   before the baseline compile, and before the probe. A guard written as
   `if budget:` reintroduces the bug this was fixed for. This covers the
   compute calls a job makes. `talos doctor --full` runs outside any job and
   has no budget: its probe and its one compile-and-score are guarded by the
   `Spend this?` confirmation (default no) or `--yes` instead (the repo owner's
   decision, 2026-10-26).
6. **`state.json` is written atomically and fsynced.**
   `talos/state.py::_atomic_write` is the only way it is written. A resume
   reads it back; a truncated `state.json` is an unresumable job.
7. **Every cost Talos shows is an estimate, never a bill.** Modal spend is
   measured container seconds times list prices shipped in
   `talos/bench.py::_seconds_cost`; C3 spend is measured running seconds times
   `talos/c3_bench.py::GBP_PER_HOUR` converted at
   `talos/c3_bench.py::USD_PER_GBP`; LLM spend is measured tokens times
   `talos/providers/pricing.py::PRICES`, except that claude-cli charges the
   `total_cost_usd` the CLI reports, for single-shot calls and agentic sessions
   alike (`talos/providers/claude_cli.py::session_cost`), including a call that
   failed after printing it. An unknown model is "unpriced"
   (`None`), never zero, and so is codex-cli, which reports no cost
   (`talos/cli.py::unpriced_reason`); a dollar-only budget with either is
   refused rather than allowed to run uncapped. A claude-cli call or agentic
   session killed at its timeout, or ending without a cost, is recorded as
   `llm_cost_unknown` (`talos/loop.py::Loop._charge_call`,
   `talos/agentic.py::_charge_session`), not charged. The one fixed price is the
   local backend's compute, which is zero by design rather than estimated:
   `talos/cli.py::make_bench` passes `usd_per_hour=0.0`, the machine is the
   user's own, and `talos run` neither asks for nor defaults a compute cap
   there (`talos/cli.py::cmd_run`).

## What "done" means

Run from the repo root:

    make check

That is the command CI's `check` job runs (`.github/workflows/ci.yml`). CI also runs a
`wheel` job that builds the wheel, checks its contents with `scripts/check_wheel.py` and
runs it with no extras; `make check` does not cover that. No target uses
`|| true`; a red suite is a failure, not a warning. The `Makefile` is the
executable definition of a valid change.

`make check` is ruff lint, pytest with the `live` marker excluded, and the
agentify contract self-check. `tests/test_live.py` spends real Modal budget, real
C3 credit or LLM tokens and is run by hand (see `docs/maintainers.md#live-smoke-tests`).

## What requires a human

Do not decide these yourself. Raise them and stop.

- Changing `MONOREPO_REF` or `DEV_IMAGE_TAG` in `talos/challenges.py`. It
  redefines every cached baseline and needs a redeploy and a live smoke test.
- Changing a challenge's `BeatRule` (margin, track tolerance, error ceiling)
  or its hardware class. That redefines what "beats the baseline" meant for
  every package already handed back.
- Changing the nonce counts or `HOLDOUT_START` in `talos/nonces.py`. Training
  and held-out sets must stay disjoint under the same hash.
- Adding or repricing a model in `talos/providers/pricing.py`, or a Modal
  per-second rate in `talos/bench.py`. Those tables are the budget.
- Loosening the agentic sandbox (`talos/agentic.py::sandbox_settings`), the
  child-process environment allowlist, or the codex opt-in.
- Loosening the local job container's isolation flags
  (`talos/local_transport.py::run_args`): networking, capabilities, the pids
  limit, the user it runs as. They are the sandbox for LLM-authored code on
  the user's machine.
- Anything that submits to TIG, stores a credential anywhere other than
  `.talos/secrets.json`, or sends data anywhere other than the user's own LLM
  provider and Modal account.
- Running `tests/test_live.py`. It spends the user's Modal budget, C3 credit or, for
  the agentic test, LLM tokens.
- Running `talos doctor --full` against a real backend. It compiles and scores a
  baseline on the user's Modal account, C3 credit or machine, outside any job budget.
- Changing the calibration constants in `talos/calibration.py` (`MARGIN_START`,
  `MARGIN_STEP`, `MARGIN_FLOOR`, `MISS_LIMIT`, `BUDGET_FLOOR_US`), or changing the default
  of `talos run --scoring` (`native` since 2026-09-29, the repo owner's decision). Together
  they decide how close a native budget is to TIG's fuel limit.
- Running the native parity live tests (`tests/test_live.py::test_native_parity`). They
  spend Modal budget or C3 credit.
- Changing the C3 prices in `talos/c3_bench.py::GBP_PER_HOUR`. They are the
  budget for the C3 backend.
- Running the C3 live test (`tests/test_live.py::test_c3_knapsack_job`). It
  spends the user's C3 credit.

To report a bug in this project, file an issue with the `agent-reported`
label:

    gh issue create --label agent-reported --title "..." --body "..."

The label is what routes the issue to a person. An issue without it notifies
nobody.

## Where to look

| Task | Start here |
|---|---|
| Set up and run day-to-day commands | `README.md`, `docs/usage.md` |
| The guide for agents *using* Talos (`talos guide`) | `talos/data/agent_guide.md`; its drift test is `tests/test_guide.py` |
| Run the gate | `Makefile` |
| The research loop, its guards, the agentic sandbox, the baseline cache | `docs/architecture.md` |
| Modal, C3 and local transports, job directories, timings, the dev images | `docs/compute-backends.md` |
| JSON output of talos status / show (the versioned schema) | `docs/json-output.md` |
| `talos status`, `show`, `diff`: reading a run directory | `talos/inspect_run.py` |
| Why a decision was made (non-authoritative) | `docs/ai/specs/` |
| CLI flags and dispatch | `talos/cli.py::build_parser`, `talos/cli.py::COMMANDS` |
| Wizard menus and the yes/no confirmation | `talos/wizard.py`; the `run` wizard and job summary in `talos/cli.py::cmd_run` |
| `talos demo`: scripted provider, scores, package guide | `talos/demo.py`, `talos/cli.py::cmd_demo` |
| `talos doctor`: health checks and `--full` | `talos/doctor.py` |
| Where config, secrets and runs live (the workspace); `--workspace`; optional extras | `talos/workspace.py`, `talos/extras.py`, `docs/usage.md#workspace` |
| Per-challenge pins, beat rules, hardware | `talos/challenges.py::CHALLENGES` |
| One iteration of the research loop | `talos/loop.py::Loop.iterate`, confirmation in `talos/loop.py::Loop._confirm` |
| Lineage tree, chains, annealing accept rule and temperature | `talos/lineage.py::decide`, `talos/lineage.py::temperature`, `talos/loop.py::Loop._anneal_step` |
| How a candidate is compared to the baseline | `talos/scoring.py::bundle_delta`, `talos/scoring.py::beats` |
| What a job persists, and the run directory layout | `talos/state.py::JobStore`, `docs/usage.md#where-results-land` |
| Baseline resolution and its cache | `talos/baseline.py::resolve_baseline` |
| Modal app: image, compile and score functions | `talos/modal_app/talos_bench.py`; container-side logic in `talos/inside.py` |
| Modal client: retries, pause window, cost estimate | `talos/bench.py::ModalBench` |
| Stale-deploy check before spend; `talos deploy` | `talos/deploy_digest.py`, `talos/bench.py::ModalBench.check_deploy`, `talos/cli.py::cmd_deploy` |
| C3 client: deploy, poll, pull, cost estimate | `talos/c3_bench.py::C3Bench` |
| C3 job directory: .c3 config, job.sh, payload.json | `talos/c3_jobdir.py::write_job_dir` |
| C3 in-container runner: build, score, write results (the local backend runs it too) | `talos/c3_job.py::main` |
| Local backend: Docker transport, container flags | `talos/local_transport.py::DockerTransport`, `talos/local_transport.py::run_args` |
| Local backend: image pull, volumes, clone, warm build | `talos/local_transport.py::prepare` |
| LLM providers and prices | `talos/providers/__init__.py`, `talos/providers/pricing.py::PRICES` |
| Agentic mode: sandbox and scope check | `talos/agentic.py::sandbox_settings`, `talos/agentic.py::read_back` |
| Native research scoring: runner, calibration, validation | `talos/native_runner.py::render`, `talos/calibration.py::budgets_us`, `talos/scoring.py::validation_failure`, `talos/loop.py::Loop.calibrate` |
| Budget rules | `talos/budget.py::exhausted`, `docs/usage.md#budget` |
| Hand-back package contents | `talos/package.py`, `docs/usage.md#where-results-land` |
| Tests | `tests/`, one file per module; the manual live smoke test is `tests/test_live.py` |
