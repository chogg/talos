# Using Talos

This is the full reference for every Talos command. The [README](../README.md) is the
quickstart.

## Contents

- [Workspace](#workspace)
- [Running a job](#running-a-job)
- [Command reference](#command-reference)
- [Agentic mode](#agentic-mode)
- [Where results land](#where-results-land)
- [Budget](#budget)

## Workspace

Talos keeps `talos.config.json`, `.talos/secrets.json` and `runs/` (and `.talos/` caches) in
one workspace directory. It picks the directory in this order:

1. `TALOS_WORKSPACE`, if set. Except for the three commands named below, it must name an
   existing directory, else the command exits 2 with `TALOS_WORKSPACE=<value> is not a
   directory`.
2. The nearest directory at or above the current directory that holds `talos.config.json`.
3. `~/.talos/workspace`.

The global flag `talos --workspace DIR <command>` sets `TALOS_WORKSPACE` for that command, so it
wins over the variable and the other rules. `talos setup`, `talos demo` and `talos run --fake`
create the workspace if it does not exist; every other command does not, and one that finds no
`talos.config.json` there says to run `talos setup`. `talos setup` prints `Workspace: <path>`, and
`talos run` and `talos demo` print absolute paths for the results.

Paths you pass as `--dir` or `--direction-file` are relative to the current directory, not the
workspace. To keep state in a clone, as earlier versions did, run `talos --workspace . setup`
there; from then on, any directory inside the clone finds it by rule 2.

## Running a job

### Start

Interactively, answering each prompt:

```bash
talos run
```

Or with every answer given as a flag:

```bash
talos run --challenge knapsack \
  --direction "Try a tighter upper bound in the branch-and-bound pruning" \
  --budget-usd 20 --budget-hours 4 --budget-compute-usd 10 --yes
```

To start from notes you already have (what has been tried, what failed, what to avoid), write
them to a file and pass `--direction-file notes.md` instead of `--direction`. The whole file
is sent to the LLM as the direction on every iteration.

Before the job starts, Talos fetches the challenge's active tracks and fuel from mainnet,
draws the nonces, and prints a job summary such as this one (from a `--fake` job; the figures
differ per run):

```
Job summary
  challenge: knapsack (CPU, tested)
  scope:     1 tracks, 8 nonces per track, fuel 1
  provider:  fake / fake, single-shot mode
  search:    start mainnet, greedy, 1 chain(s), scoring native
  hyperparameters: 1/1 tracks from mainnet
  budget:    1 iterations, compute $20.00
  backend:   fake (in-process, no charge)
  results:   /home/you/.talos/workspace/runs/20261026-142231-knapsack
```

It then asks `Start this job?` (default yes). Answering no prints `Not started.`, exits 0 and
leaves nothing under `runs/`. `--yes` skips the question but still prints the summary. The job
id is the start time plus the challenge name.

### Watch

The terminal prints one line per event, prefixed with the time and the iteration number.
Each iteration ends with a summary line: the outcome, then the best delta so far, the spend
and the time left.

```
[10:41:02] #1 trying: bump k (local_search)
[10:49:12] #1 scored +1.000% vs baseline (±0.250%, 4.0x noise, clear; worst track +1.000%, errors 0.0%, runtime x1.00)
[10:49:12] #1 new best | best +1.000% | llm $0.02 | compute ≈$1.28 | 3.9h left
[10:49:12] #1 confirming on held-out nonces
[10:49:12] #1 won: held-out +1.000% vs baseline (worst track +1.000%)
```

The `±` figure after a scored delta is the paired standard error of the mean delta over the
nonces. `4.0x noise` is the size (absolute value) of the delta divided by that error, and the
word after it is a band: `clear` for 2 or more, `marginal` for 1 to under 2, `noise` for under 1.
The band says only how far the delta is from zero. It does not say whether the delta is a gain,
so a regression can read `clear`. Three cases have no ratio. `n=1` means a track has one nonce,
so no error can be computed. `identical on every nonce` means the candidate scored the same as
the baseline on every nonce. `±0, no spread` means every nonce moved by the same amount. Over
the error ceiling, nothing is shown. The noise text comes first on the line. A line cut to the
terminal width loses the worst-track, error and runtime figures before the noise figure.

The standard error measures nonce-to-nonce variation for one candidate. It does not account
for the best being chosen out of many candidates, which is how a candidate improves on training
and then fails held-out. A ratio of 3x on the best of 7 tries is weaker evidence than 3x on a
single try. Whether the ratio predicts held-out results is not yet known.

`best` is the best candidate's mean relative delta versus the baseline on training nonces.
`compute ≈` is an estimate (see [Budget](#budget)). The last field is the wall-clock budget
remaining, or `∞ left` without `--budget-hours`. An iteration that did not improve reads
`no improvement (4 in a row)` or `no candidate: did not compile (2 in a row)`; a failed
compile prints the first `error` line of the compiler output. Lines are cut to the terminal
width. `runs/<job_id>/timeline.jsonl` records every event with every field in full,
including hypothesis descriptions and the compiler output (except the baseline's own compiler
output in a `--start template` job, which is left out so its source cannot leak).

With `--search anneal`, or with several chains or a template start, each scored candidate
also prints an `accept` line (chain, node, parent, design family, gap to mainnet, and the
decision), and a chain that stalls prints a `reheat` line. A candidate kept as its chain's
next step without improving the job's best reads `kept as the chain's next step`:

```
[11:02:40] #4 chain 1 #4 <- #2 [rolling-horizon LP] -12.000% vs mainnet: accepted (p=0.42, T=0.031)
[11:09:15] #6 chain 1 #6 <- #4 [greedy dispatch] -20.000% vs mainnet: rejected (p=0.05, T=0.029)
[11:31:03] #9 chain 1 reheats from #3, T=0.050
```

### Stop

Press `Ctrl-C` once. Talos stops at the next safe point, packages the best candidate found
so far, and prints the final status. The job can be resumed later.

For a `talos run` started in the background from a script, send one SIGINT instead:
`kill -INT <pid>`. A plain `kill` (SIGTERM) or `kill -9` ends it without writing the package.

### Resume

```bash
talos status                              # find the job id
talos run --resume 20260917-101006-knapsack
```

A resumed job keeps the provider, model, mode, track, hyperparameters and nonces it started
with, whatever `talos.config.json` says now. A C3 job that was still running when Talos
stopped is reattached rather than paid for twice.

An `exhausted` job continues when you add budget to it:

```bash
talos run --resume 20260917-101006-knapsack --add-iterations 20
```

Nothing is re-measured: the job keeps its baseline, nonces, hardware, calibration, best
candidate and lineage. An annealing job continues at the temperature its spend gives on the
new total budget, rather than reheating. If another cap is still spent (say `--add-iterations`
on a job out of hours), the extension is refused and names the flag to add. The `--add-*`
flags also work on any job `--resume` accepts. A `won` job cannot be continued.

A job started before a TIG dev image upgrade may not resume after it; see
[Upgrading the TIG dev image](maintainers.md#upgrading-the-tig-dev-image).

### Submit

When a job ends with `Status: won`, the code to submit is in `runs/<job_id>/package/submission/`,
next to a copy of TIG's submission `README.md` template. Fill out the template (algorithm name,
copyright, submitter), then copy the folder's contents into
`tig-algorithms/src/<challenge>/<your_name>/` in a monorepo checkout.
`runs/<job_id>/package/README.md` lists the steps. `evidence_draft.md` in the same directory
is a partly filled-in advance-evidence template. You submit it yourself; Talos does not.

A job that ends any other way still writes a package with its best candidate in
`runs/<job_id>/package/submission/`, but that candidate has not beaten the baseline on
held-out nonces, and the package README says so. The last line of the run's output says the
same instead of telling you to submit it.

## Command reference

| Command | What it does |
|---|---|
| `talos setup` | One-time configuration: backend, LLM provider, credentials. Validates them and deploys the Modal app. Prints the workspace it uses. |
| `talos deploy` | Redeploys the Modal app from the installed Talos code. Run it after upgrading Talos. |
| `talos run` | Starts or resumes a research job and writes its package when it stops. |
| `talos compile` | Compiles a directory of algorithm files on the configured backend and prints the compiler output. Scores nothing. |
| `talos status [JOB_ID]` | Lists every job under `runs/`, or one job's detail with `JOB_ID`; `--json` for scripts. |
| `talos show JOB_ID` | Shows one iteration (the best by default): hypothesis, outcome, scores and file paths; `--json` for scripts. |
| `talos diff JOB_ID` | Prints one iteration's unified diff against the baseline or its parent. |
| `talos report JOB_ID` | Writes `runs/<job_id>/package/report.html`, a one-page view of the job's search, from the files already in the run directory. |
| `talos demo` | Runs a scripted job with no account, network or spend, and prints its package guide. |
| `talos doctor` | Runs health checks on the config, credential, provider (one small LLM test call), backend, mainnet and dev images, with no compute spend; `--full` also scores one real baseline after asking. |
| `talos check-updates` | Checks GHCR for newer published TIG dev image versions and prints upgrade guidance. |
| `talos guide` | Prints the guide for a coding agent that drives Talos for you. |

### `talos setup`

```bash
talos setup
```

No flags. It asks, in order:

1. **Compute backend**: `modal`, `c3` or `local`.
2. **Provider**: one of the kinds in the [provider table](../README.md#pick-an-llm-provider).
3. **Model**: press Enter for the default. For `codex-cli` it first prints the models your
   login accepts; for `claude-cli` an alias such as `fable`, `opus` or `sonnet` works.
4. **API base URL**: `custom` provider only.
5. **API key**: API providers only. Each character you type or paste shows as `*`, so you can
   see that a paste arrived; the key itself is never shown.
6. **Mode** (`single-shot` or `agentic`): CLI providers only. This is the default; `talos
   run --mode` overrides it per job.
7. **Modal token id and secret**: `modal` backend only; the secret shows as `*`.
8. **C3 API key**: `c3` backend only; shows as `*`. Leave it blank to use your `c3 login`
   session. With it blank, a `C3_API_KEY` environment variable is used if set.
9. **CPUs and memory for the local container**: `local` backend only. The defaults are what
   Docker reports: every CPU the daemon has and its memory minus 4 GiB. On Docker Desktop
   those are the VM's figures (Settings > Resources), not the machine's, and values above them
   are refused, because Docker rejects a container with more CPUs than the daemon has. Both
   are part of the local baseline cache key: change them and local baselines are measured
   again.

It then checks everything before writing anything:

- the provider, with one cheap call (for CLI providers: that the binary is on `PATH` and a
  trivial call succeeds);
- on `modal`: sets your Modal token and deploys the benchmark app to your account;
- on `c3`: runs `c3 whoami` and `c3 balance` (with the API key, if you gave one), and warns
  if the balance is below £1;
- on `local`: `docker info` is run before the limits are asked (its CPU and memory figures
  are their defaults and ceilings), and the summary says whether the NVIDIA runtime is present.

On success it prints ``Setup complete. Run `talos run` to start a job.`` and writes:

- `talos.config.json`;
- `.talos/secrets.json` (mode 0600), holding only the LLM API key and the C3 API key, if you
  gave either. If you gave neither, it deletes any `.talos/secrets.json` left by an earlier
  setup.

If any check fails it prints the reason, exits non-zero, and writes neither file.

Run `talos setup` again to change provider, model or backend. After upgrading Talos on the
Modal backend, run [`talos deploy`](#talos-deploy) instead. The C3 backend ships its code with
each job, so it needs no redeploy.

Exit code 0 on success, 1 when a credential or backend check fails, 2 for an unknown
backend or provider, or for a backend or provider whose extra is not installed.

### `talos deploy`

```bash
talos deploy
```

No flags. It deploys the `talos-bench` app to your Modal account from the installed Talos
code and prints `Deployed talos-bench (deployed_<12 hex digits>)`. It needs the `modal` extra,
reads no config, and keeps the Modal token that `talos setup` set. The function name ends in
a digest of the code the app was built from. With the Modal backend, `talos run` pauses before
any spend, and `talos compile` refuses, when the deployed digest differs from the client's;
the message says `the deployed talos-bench app was built from different Talos code than this
client; run `talos deploy``. `talos doctor` reports the same as `backend: FAIL`. Every Talos
version needs its own deploy, so run `talos deploy` after each upgrade. The C3 and local
backends need no deploy.

Exit code 0 on success. Exit code 2 when the `modal` extra is missing, and also when
`modal deploy` itself fails (the last part of its output is printed to stderr).

### `talos demo`

```bash
talos demo                         # a knapsack-labelled demo
talos demo --challenge hypergraph  # the same script, labelled with another challenge
talos demo --delay 0               # no pause between event lines
```

| Flag | Meaning |
|---|---|
| `--challenge NAME` | The challenge to label the demo with (default `knapsack`). The script and scores do not change. |
| `--delay SECONDS` | Pause after each event line (default 0.4; 0 disables). |

Needs no config, account, network or compute backend, and spends nothing: every figure is
scripted and priced at $0.00. It writes a job to `runs/demo-<timestamp>-<challenge>/` in the
[workspace](#workspace), runs the real research loop on it, and prints the event stream, the final status (`won`), the
package path, a one-line description of each package file, `talos report <job_id>` and
`Next: talos setup, then talos doctor`. Exit code 0 on success.

### `talos doctor`

```bash
talos doctor                                    # no compute; one small LLM test call
talos doctor --full --challenge knapsack        # also score one baseline, after asking
talos doctor --json                             # one JSON document instead of text lines
```

| Flag | Meaning |
|---|---|
| `--full` | After the free checks, compile and score mainnet's top algorithm for the challenge on 2 training nonces of its first track on your backend. It prints the cost, then asks `Spend this?` (default no; a closed stdin counts as no). |
| `--challenge NAME` | Challenge for `--full` (default `knapsack`). |
| `--yes` | Run `--full` without asking. `--json` works without it, but `--full --json` needs it. |
| `--json` | Print `{"checks": [{"name", "status", "detail"}, ...], "ok": true or false}` and nothing else on stdout. With `--full` and no `--yes` the `full` check is `skip` with detail `needs --yes`. |

The free checks run in this order and print one line each, `name: ok (detail)` or
`name: STATUS: detail`, where STATUS is `warn`, `FAIL` or `skip`:

- `config`: `talos.config.json` loads. With none in the directory, `credential`, `provider`
  and `backend` print `skip` and the other checks still run.
- `credential`: the provider's API key, or its CLI (`claude`, `codex`) on `PATH`.
- `provider`: one test call to the LLM.
- `backend`: Modal has every function of the `talos-bench` app deployed; C3 is logged in and
  has credit (`warn` under £1); Docker answers on a local backend.
- `mainnet`: mainnet is reachable, and each challenge's id and CPU/GPU flag match
  `talos/challenges.py`.
- `images`: the pinned dev image is on GHCR for every challenge.

Most `FAIL` lines name the command that fixes it (such as `talos setup` or `talos deploy`); the rest say what to check, such as your network, or pass on the backend's own error text. With `--full` a last line, `full`, reports
the compile and the nonces scored. If a free check failed, `full` is `skip` and does not run.
The cost it prints is MEASURED 2026-09-15: £0.02 and 12 min 20 s for knapsack on C3;
for every other challenge and backend it is labelled `ESTIMATE (unverified)` and says
how many minutes it assumes. The local backend has no compute charge but pulls a 13 GB image
the first time.

Exit code 0 when no check is `FAIL`. Exit code 1 when any check is `FAIL`, and also when
`--full` was requested and did not run and pass (declined, skipped, or failed).

### `talos check-updates`

```bash
talos check-updates                       # all eight challenges
talos check-updates --challenge knapsack  # one challenge
```

Shows the configured image version and the newest published stable `X.Y.Z` tag for each
challenge. It reads only registry metadata: Docker, sudo, project configuration and
compute credentials are not needed. Aliases, prereleases and architecture-specific tags
are excluded. A newer tag is an upgrade candidate; source compatibility still needs review.
Exit code 0 means every lookup succeeded (including when updates are available), 1 means
at least one lookup failed and its update status is unknown, and 2 means invalid arguments.

Each new real job also checks its challenge and prints a recommendation if a newer image
is available. Each check has a five-second request budget; a failed automatic check is
silently skipped. Resumes, compile calls and fake runs do not make this advisory check.

To apply an upgrade, see
[Upgrading the TIG dev image](maintainers.md#upgrading-the-tig-dev-image).

### `talos run`

Prompts for anything not given as a flag, unless `--yes` is passed. Before it writes anything
or spends anything, it prints a job summary and asks `Start this job?`; `--yes` skips that
question. Answering no, or a closed stdin, ends the command with nothing under `runs/`.

The [challenge status table](../README.md#challenges) in the README gives each challenge's
status. `talos run` repeats the status as a warning before it starts a job on a challenge
that is not tested.

| Flag | Meaning |
|---|---|
| `--challenge NAME` | The challenge to work on, from the [challenge status table](../README.md#challenges). See **`--challenge`** below for the hardware each runs on. |
| `--direction TEXT` | What to explore, in free text. It becomes the first entry in the job's `tacit.md`. |
| `--direction-file PATH` | The same, read from a file. Pass one of `--direction` or `--direction-file`, not both. |
| `--track NAME` | Optimise one active track of the challenge instead of all of them (`all` is the default; the prompt lists the tracks). See **`--track`** below. |
| `--nonces N` | Nonces per track in each of the training and held-out sets (8 is the default; the prompt asks). It sets the cost of every evaluation; see **`--nonces`** below. |
| `--hyperparameters mainnet\|none` | `mainnet` (the default) runs the baseline and every candidate with the baseline algorithm's mainnet hyperparameters per track, fixed at job start; with `--start template` only the baseline uses them. `none` runs every nonce without hyperparameters. See **`--hyperparameters`** below. |
| `--mode single-shot\|agentic` | Overrides the configured mode for this job. `agentic` needs a CLI provider and uses roughly 5 to 20 times the tokens of `single-shot`. |
| `--scoring native\|metered` | `native` (the default) scores research candidates with a faster unmetered runner and scores an improvement again on the metered runtime before it can become the best. `metered` scores every candidate with TIG's own `tig-runtime`. See **`--scoring`** below. |
| `--start mainnet\|template` | `mainnet` (the default) starts from the top-adoption mainnet algorithm. `template` starts from TIG's empty template, on any challenge. See **`--start`** below. |
| `--base-algorithm NAME` | Starts from this compiled mainnet algorithm instead. The baseline to beat stays the top-adoption algorithm. Not with `--start` or `--base-dir`. See **`--base-algorithm` and `--base-dir`** below. |
| `--base-dir DIR` | Starts from the `.rs` and `.cu` files in this local directory instead. The baseline to beat stays the top-adoption algorithm. Not with `--start` or `--base-algorithm`. See **`--base-algorithm` and `--base-dir`** below. |
| `--search greedy\|anneal` | `greedy` (the default) edits the best candidate so far. `anneal` edits the chain's current candidate and sometimes accepts a worse one so the search can leave a local optimum. The annealing settings (start temperature 0.05, final temperature 0.1 of that, reheat after 5 iterations without a new chain best) are estimates with no measurement behind them. |
| `--chains N` | Independent search chains (default 1; a value below 1 is refused). Chains take turns: iteration n runs on chain (n-1) mod N, one at a time; running them concurrently comes in a later change. |
| `--budget-usd N` | LLM spend cap in USD. |
| `--budget-hours N` | Wall-clock cap in hours. |
| `--budget-iterations N` | Iteration cap. |
| `--budget-compute-usd N` | Estimated compute spend cap in USD. |
| `--resume JOB_ID` | Continue a job that was interrupted, cancelled, paused or failed, or an `exhausted` job given an `--add-*` flag. Cannot change its mode, track, hyperparameters, nonces, start, base algorithm, search or chains. |
| `--add-iterations N` | Adds N to the job's iteration cap. Only with `--resume`; refused for a cap the job was started without. |
| `--add-hours N` | Adds N to the job's wall-clock cap. Only with `--resume`; refused for a cap the job was started without. |
| `--add-usd N` | Adds N to the job's LLM spend cap. Only with `--resume`; refused for a cap the job was started without. |
| `--add-compute-usd N` | Adds N to the job's compute spend cap. Only with `--resume`; refused for a cap the job was started without. |
| `--yes` | Accept defaults instead of prompting, and skip the `Start this job?` confirmation. At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` must still be given; compute defaults to $20 (not asked and not set on the `local` backend, where it is always zero). |

**`--challenge`.** One of `satisfiability`, `vehicle_routing`, `knapsack`, `job_scheduling`,
`energy_arbitrage` (CPU), or `vector_search`, `hypergraph`, `neuralnet_optimizer` (GPU: an
L40S, or when none is free the next of A100-80GB and H100). On C3 a CPU challenge likewise
runs on `cpu-d3-4vcpu-16gb`, or `cpu-e2-4vcpu-16gb` when that is out of stock; see
[docs/compute-backends.md](compute-backends.md#hardware-fallback).

**`--track`.** Training scores that track only. When a candidate wins on training, the
confirmation scores that track's held-out nonces plus every other track's training nonces,
and no other track may get worse. The LLM still sees and may edit every file.

**`--nonces`.** Every nonce is scored for the baseline on both sets and for every candidate on
training, then again on held-out when it wins on training, so the count sets the cost of
every evaluation. On C3 and local, a candidate's training stops early once its errors put it
over the error ceiling, so a failing candidate costs less. On hypergraph an L40 spends about 10 minutes per nonce-per-track on the
baseline's two sets at the current fuel (measured 2026-09-25); 32 made that baseline 11 hours
against the 6-hour C3 job limit. A resume keeps the sets the job was drawn with.

**`--hyperparameters`.** With `mainnet`, the values are the per-track hyperparameters of the
baseline algorithm's best-quality mainnet benchmark at the job's fuel. A track with no such
benchmark runs without any. The package lists the values and the benchmark they came from.
With `--start template`, only the baseline runs with these values: candidates run with none,
and the package has no `hyperparameters.json` (its README lists the baseline's values).

**`--scoring`.** `native` scores research candidates with an unmetered runner under a
per-track time budget calibrated from the baseline, which is faster; a candidate that
improves natively is then scored again on the metered runtime and only becomes the best if it
holds up there ([how it works](architecture.md#native-research-scoring)). `metered` scores
every candidate with TIG's own `tig-runtime` under the job's fuel limit. The live parity test
has not been run: the CPU runner is measured on knapsack only, and the GPU runner has never
been compiled; a native build that fails at calibration makes the job score metered. A job
keeps its scoring on resume; `--resume` with a different `--scoring` is refused. A job whose
`job.json` predates the `scoring` field resumes metered. The calibration is cached at
`~/.talos/calibration/<challenge>/<key>.json`, keyed by
`talos/calibration.py::calibration_key` (challenge, hardware class, both pins, the baseline's
code, the native runner and the baseline's hyperparameters). If it cannot be made, the job
scores metered and says so in the timeline.

**`--start`.** With `template`, the LLM sees the challenge's `README.md` and its non-baseline
source files from the pinned monorepo (`talos/challenges.py::TEMPLATE_START_FILES`, not the
baseline algorithms' directory) and the template, and never the baseline's code. The baseline
is still measured and every candidate is scored against it. Candidates run without
hyperparameters while the baseline keeps the mainnet ones. hypergraph's challenge files include
TIG's greedy partitioner, which sets the quality zero point. On a GPU challenge
(`hypergraph`, `vector_search`, `neuralnet_optimizer`) the starting candidate also has
`algorithm.cu`, TIG's `template.cu`, for its CUDA kernels. Every new best of a template job is
scored again on the metered runtime (with `--scoring native`, the default), and a template
job's first scored candidate is always a new best, so expect more of these runs than on a
mainnet start; they cost more on a GPU challenge. `neuralnet_optimizer` has nine challenge
files, so its prompts are long (ESTIMATE (unverified) ~25k tokens each). Refused with
`codex-cli` in agentic mode, even with `TALOS_ALLOW_CODEX_AGENTIC=1`. See
[From-scratch start and annealing search](architecture.md#from-scratch-start-and-annealing-search).

**`--base-algorithm` and `--base-dir`.** The search edits this code, whatever it scores. The
top-adoption mainnet algorithm is never edited: it is a score to beat. The job's best, and any win,
must clear the higher of the two scores. A win also needs to beat the mainnet algorithm on held-out
nonces.
`--base-dir` needs `mod.rs` at the top of the directory, and a `.cu` file on a GPU challenge; its
source goes to your LLM provider and compute backend, as a mainnet algorithm's does. The files
are copied to `runs/<job_id>/base_algorithm/` at job start, so later edits to the directory do not
affect the job, and a resume refuses a job whose copy is missing.

After the baseline, the job scores the base algorithm once on the training nonces with TIG's
metered runtime, before any LLM call. If it does not compile, or fails on more nonces than the
challenge's error ceiling allows, the job ends `failed`.

If the base algorithm beats the baseline, a candidate under it is rejected even if it beats the
baseline. If no candidate clears it, the base algorithm stays the best and the package has nothing
to submit.

If the base algorithm does not beat the baseline, including a tie, or a higher mean that regresses
a track past the challenge's tolerance, the search still works on it. A candidate that improves on
its chain's best is `accepted` and the next iteration edits it, but nothing is the job's best until
a candidate beats the baseline. A job that runs out of budget before then has nothing to submit,
and the package's `README.md` names the closest candidate and where its code is.

The base algorithm and every candidate run without hyperparameters, and the package has no
`hyperparameters.json`. `--hyperparameters` applies only to the baseline. An algorithm that needs
hyperparameters to run or to score well will show that in its score against the baseline. The
package has `diff_vs_base.patch` instead of `diff_vs_baseline.patch`.

Annealing and several chains keep their own rules: an annealing chain can accept a worse candidate
and a chain edits its own best, but neither can make that candidate the job's best or its win.

Without flags, the wizard asks for challenge, direction, an LLM budget (USD for metered
providers, default 20; iterations for CLI providers, default 50), a wall-clock budget
(default 4 hours), a compute budget (default $20), the mode (CLI providers only), the
track, the hyperparameters (`mainnet` or `none`), and the nonces per track (default 8).

Talos refuses to start, before spending anything, when: there is no API key for the
provider; the model has no entry in the price table, or the provider is `codex-cli` (which
reports no cost), and the only budget is `--budget-usd` (a dollar cap it could not enforce);
the mainnet challenge id no longer matches `talos/challenges.py`; mainnet gives no fuel
budget for the challenge; or, on C3, the challenge's dev image tag is not on GHCR.

Exit code 0 when the job ends `won`, or when you decline the confirmation; 1 when it ends any other way, or when the mainnet or
dev image check stops it; 2 for invalid arguments, a missing config, or a missing API key. 130 when an interrupt lands outside the research
loop. Before the loop, a job already written under `runs/` is left `cancelled` and `--resume`
continues it; after the loop (while the package is written) the job keeps the status it stopped
with. An interrupt during the loop stops it as described under [Stop](#stop) and exits 1.

### `talos compile`

```bash
talos compile --challenge knapsack --dir algorithm
```

| Flag | Meaning |
|---|---|
| `--challenge NAME` | Required. One of the names in the [challenge status table](../README.md#challenges); any other name exits 2 with a usage error. |
| `--dir PATH` | Directory whose `.rs` and `.cu` files are compiled (default `algorithm`). |
| `--backend modal\|c3\|local` | Backend to use. Without it: `TALOS_BACKEND`, then `talos.config.json`, then `modal`. |
| `--native` | Build the unmetered `talos-native` runner instead of TIG's metered build. This is the build a `--scoring native` job (the default) scores with, and its agent is told to pass this flag. Without it the build is TIG's metered one, the build a submission uses. |

A candidate that compiles with `--native` can still fail TIG's metered build. A native job's
validation catches that and demotes the candidate as `native_metered_build_mismatch`.

Prints the last 4000 characters of compiler output. Exit code 0 if the build succeeded, 1 if
it failed, 2 if the directory has no `.rs`/`.cu` files. Agentic mode uses this command to
check its own edits.

On Modal it is one function call. On C3 it is one batch job
([about 12 minutes](compute-backends.md#c3-timings)) whose job directory is
`.talos/compile/`. On `local` it is one Docker container on this machine, after the same
prepare step as `talos run` (image pull, clone and warm-up build on the first run per
challenge; [about 3 minutes for knapsack](compute-backends.md#local-timings) per build after that),
with its job directory under `.talos/compile/` too; the container is removed when the command
returns or is interrupted. Each run overwrites that directory, so do not run two
`talos compile` commands at once from the same directory.

### `talos status`

```bash
talos status [JOB_ID] [--json]
```

Without a job id, one row per job under `runs/` (the row below is illustrative, not a real
run):

```
JOB                             CHALLENGE       STATUS     IT  BEST     LLM          COMPUTE  LAST EVENT
20260917-101006-knapsack        knapsack        won        1   +0.412%  $0.02        $1.28    2026-09-17 10:41 UTC
20261026-120733-job_scheduling  job_scheduling  exhausted  6   n/a      not tracked  $12.40   2026-10-26 21:14 UTC
```

BEST is the best candidate's mean training delta, `n/a` with no candidate. LLM reads
`not tracked` for a `codex-cli` job and `unpriced` for a model with no price-table entry, as in
the status line during a run. LAST EVENT is the time of the last event in `timeline.jsonl`. A
directory whose `state.json` or `job.json` is missing or cannot be parsed is listed as
`unreadable`, with the reason on stderr; it does not stop the other jobs from being listed.

With a job id, that job's settings, status and stop reason, its best candidate (training delta
and held-out result), its budget (each cap after `--add-*` extensions, what was spent and what
is left), the compute job in flight and the last event. Hours spent run to the last event for
a job that has stopped (`won`, `exhausted`, `failed`, `cancelled` or `paused`) and to now for
any other. Talos writes no lock file, so a `researching` job whose last event is old may belong
to a process that has died.

`--json` prints one JSON object instead; its schema is in
[docs/json-output.md](json-output.md). The command reads `runs/` only: no config,
credentials or network. Exit code 0, or 1 with a message on stderr for an unknown job id.

Status is one of `queued`, `measuring_baseline`, `researching`, `confirming`, `won`,
`exhausted` (a budget ran out; resumable with `--add-*`), `cancelled` (Ctrl-C or one SIGINT before
the job had stopped; resumable), `paused`
(the compute backend was unreachable) or `failed` (the reason is in `state.json` and the last
lines of `timeline.jsonl`).

### `talos show`

```bash
talos show JOB_ID [--iteration N|best] [--json]
```

One iteration of a job: its parent in the search tree, outcome, hypothesis (title, strategy,
design family and the full description), training delta with the worst track and per-track
deltas, the held-out result, and the paths of its saved files. For the best candidate it also
prints the per-track tables, baseline against best, on the training and held-out nonces.
`--iteration` defaults to `best`. `--json` prints one JSON object; see
[docs/json-output.md](json-output.md).

Exit code 0; 1 with a message on stderr when the job, the iteration or a best candidate does
not exist (an iteration still in flight has no record yet); 2 for a bad flag.

### `talos diff`

```bash
talos diff JOB_ID [--iteration N|best] [--against baseline|parent]
```

Prints the unified diff of one iteration's files to stdout, untruncated. `--iteration`
defaults to `best`. `--against baseline` (the default) diffs against the root the job started
from: the mainnet baseline, the template for a `--start template` job, or the base algorithm for
a `--base-algorithm` or `--base-dir` job. `--against parent`
diffs against the iteration it was built on. An empty diff prints nothing.

Exit code 0; 1 with a message on stderr when the job, the iteration or a best candidate does
not exist, or when the iteration saved no files (an edit or compile failure); 2 for a bad flag.

### `talos report`

```bash
talos report 20260917-101006-knapsack
```

Writes `runs/<job_id>/package/report.html` and prints its path. The page is one file with no
network access: a header (status, best delta, held-out result, spend, times, settings), a plot of
every iteration's training delta against the mainnet baseline with lines from each node to its
parent, a panel with any node's details and its diff against its parent, per-track tables (with a
standard error column) for the best candidate, the best diff, and the hypothesis list. It is built
only from files already in the run directory, so it works on old runs and on a job that is still
running; it needs no credentials and writes nothing else. The package written when a job stops
includes it too.
Diffs are shortened to keep the page under about 2 MB; a shortened diff ends with the `diff -u`
command that prints it in full.

### `talos guide`

```bash
talos guide
```

Prints the guide for a coding agent (Claude Code, Codex, Cursor or any other) that runs Talos
on your behalf: which commands spend money and need your agreement, how to start a job with
every answer as a flag, how to watch it with `talos status JOB_ID --json` and stop it with one
SIGINT, how to read the results, and a template for reporting back. Tell your agent "Run
`talos guide` and follow it." The text is `talos/data/agent_guide.md`, shipped in the package,
so it needs no config, network or checkout. Exit code 0.

## Agentic mode

`--mode agentic` (CLI providers only) hands each iteration to a headless `claude` or `codex`
session in a throwaway worktree outside `runs/`, instead of asking an API for one edit.

With `claude-cli`, the session runs under a sandbox that the CLI enforces: it can read only
inside its worktree and edit only the algorithm files and its notes, the only commands it can
run are `talos compile` and read-only ones such as `ls` and `grep`, it has no network tools,
and its environment holds no LLM keys or Modal tokens. The exact rules are in [docs/architecture.md](architecture.md#the-agentic-sandbox).

`codex-cli` has no equivalent sandbox: under it the agent can execute arbitrary
agent-authored commands on your machine and read any file you can read. Talos therefore
refuses to start an agentic codex run unless you opt in:

```bash
export TALOS_ALLOW_CODEX_AGENTIC=1
```

With either CLI, an edit outside the algorithm files fails the iteration.

On the C3 backend, the sandbox has no `talos.config.json` to read, so the configured backend
is passed to it as `TALOS_BACKEND`, and a C3 API key, if you use one, as `C3_API_KEY`. Each
`talos compile` the agent runs from the sandbox is one C3 job
([about 12 minutes](compute-backends.md#c3-timings)). If the agent's 30-minute timeout
kills a sandbox compile while its C3 job is still running, the job is not cancelled: it runs
on to its own time limit and bills for it, which bounds the cost but does not avoid it.

## Where results land

Each job gets a directory under `runs/` in the workspace:

| Path | Contents |
|---|---|
| `runs/<job_id>/job.json` | The job's fixed inputs: challenge, direction, provider, model, mode, budget, nonces, fuel, track, baseline algorithm, hyperparameters, scoring, start, search, chains and the annealing settings. Written once. Its budget is the starting budget; a job resumed with `--add-*` runs on that budget plus the extensions in `state.json`. |
| `runs/<job_id>/state.json` | Progress: status, iteration, spend, the best candidate, the hypothesis log, and the lineage tree (`nodes`, one per iteration, and `chains`, each chain's current node, best node and stall count). A job resumed with `--add-*` records each extension in `budget_extensions` (when, at which iteration, the status it was resumed from, and what it added). A `--scoring native` job also records its scoring, its per-track time budgets and a `validations` list: one entry per candidate that improved natively, with its iteration, its outcome (`validated` or `demoted`) and the reason it was demoted. |
| `runs/<job_id>/timeline.jsonl` | One JSON event per line; the same events the terminal prints. A `scored` event carries `mean_rel_se`, `noise_ratio`, `noise_band`, `noise_state` and `track_rel_se` (the standard error of each track's delta). A resume with `--add-*` adds `budget_extended` (the additions and the resulting budget). A native job adds `calibrated` (the budgets), `validating`, `validated` and `demoted` (with its reason); `calibration_fallback` says the job scores metered and why. A tree job (anneal, several chains or a template start) adds `accept` (one per scored candidate: chain, node, parent, family, score, decision, probability, temperature) and `reheat`. |
| `runs/<job_id>/tacit.md` | Your direction, plus lessons the LLM distilled from failed attempts. |
| `runs/<job_id>/baseline/` | The baseline algorithm's files and its per-nonce results. |
| `runs/<job_id>/challenge/` | A `--start template` job only: the challenge files and the template the LLM was shown (`template.rs`, plus `template.cu` on a GPU challenge). |
| `runs/<job_id>/base_algorithm/` | A `--base-algorithm` or `--base-dir` job only: the copy of the code the job started from. |
| `runs/<job_id>/iterations/<n>/` | Each candidate's files and its hypothesis. |
| `runs/<job_id>/best/` | The best candidate's files so far. |
| `runs/<job_id>/package/` | The hand-back package, written when the job stops. |
| `runs/<job_id>/package.zip` | The same package, zipped. |

The package directory holds:

| File | Contents |
|---|---|
| `submission` | A folder with the candidate to submit: the best algorithm's `.rs` and `.cu` files and TIG's submission `README.md` template, to fill out. Absent when the job produced no candidate. |
| `diff_vs_baseline.patch` | The candidate as a patch against the baseline. A `--start template` job writes `diff_vs_template.patch` instead, against the empty template, and a `--base-algorithm` or `--base-dir` job writes `diff_vs_base.patch`, against the code it started from. |
| lineage.json, lineage.md | The search tree: every iteration's node and each chain's state (lineage.json), and the path from the root to the best node with a per-design-family table (lineage.md). Written when the job recorded any node. |
| `scores.md` | Per-nonce tables for baseline and candidate on training and held-out nonces. A focused job (`--track`) adds a regression-guard table for the other tracks. Under each table's summary it adds `mean_rel_se: ±…  noise: N.Nx (band)` and a `per track:` line; the evidence draft for the TIG submission does not have them. |
| `hypotheses.md` | The full hypothesis log with outcomes. |
| `report.html` | The search on one page: a plot of every iteration with its lineage (a ±1 standard error whisker on each scored node that has a nonzero standard error under the error ceiling), each node's diff against its parent, per-track tables and the hypothesis log. Open it in a browser; it loads nothing from the network. Written on every package, including one with no candidate. If the page fails to render, the package prints `report.html not written: …` and is written without it; `talos report` rewrites it. |
| `evidence_draft.md` | A partly filled-in TIG advance-evidence template. |
| `hyperparameters.json` | The per-track hyperparameters, when the job used them. Not written for a `--start template`, `--base-algorithm` or `--base-dir` job, whose candidate ran without them. |
| `README.md` | The submission steps, with a Hyperparameters section when the job used them. |

Benchmark the submitted algorithm with the same hyperparameters: the measured improvement
holds only with them. A `--start template` candidate was scored without hyperparameters, so
benchmark it without any.

On the C3 backend each iteration also gets a `runs/<job_id>/c3/<n>/` directory, described in
[docs/compute-backends.md](compute-backends.md#c3-job-directories). The measured baseline
is cached outside the run directory, in `~/.talos/baselines/`; see
[docs/architecture.md](architecture.md#the-baseline-cache).

## Budget

- At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` must be set,
  directly or through the wizard. Zero is a valid, real cap, not "unset".
- Compute spend is always capped separately by `--budget-compute-usd`, which defaults to $20
  when `--yes` is passed without it. It is checked before every compute call, whatever the
  other caps are. The one exception is the `local` backend, where compute costs nothing: the
  question is skipped, no default is applied, and the cap stays unset.
- The wall-clock budget (`--budget-hours`) counts elapsed time from the job's start,
  including time spent resumed.
- `talos run --resume JOB_ID --add-*` adds to the job's caps (see [Resume](#resume)). The time
  between a stop and the continuation is not counted against `--budget-hours`.

Every spend figure in the status lines and the final report is an **estimate**, not a billed
amount. Compute spend is measured running seconds times list prices in a table shipped with
Talos. LLM spend is measured tokens times a price table. With `claude-cli`, in both modes, it
is the `total_cost_usd` the CLI reports for each call or agentic session, the CLI's own estimate
at API prices, whatever your subscription actually charges. A model with no entry in that
table is shown as `unpriced`, and `--budget-usd` cannot be enforced for it: add
`--budget-hours` or `--budget-iterations`. The same applies to `codex-cli`, which reports no
cost at all and is shown as `not tracked`.

`--budget-usd` is checked before each LLM call or agentic session, so the call that crosses
the cap still finishes; the run stops before the next one. A `claude-cli` call that fails is
still charged the cost it printed. A call or agentic session killed at its 30-minute timeout
prints no cost, so it is not counted; the timeline records it, and any other call that ends
without a cost, as an `llm_cost_unknown` event.

Compute spent by `talos compile` from the agentic sandbox is not counted against
`--budget-compute-usd`. On C3 each of those is one job with the fixed overhead described
under [C3 timings](compute-backends.md#c3-timings), billed at the profile's rate. A
focused job's confirmation scores the other tracks' training nonces too, about a minute more
per winning iteration on C3 (ESTIMATE, unverified).
