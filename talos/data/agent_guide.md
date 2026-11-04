# Driving Talos from a coding agent

This guide is for a coding agent (Claude Code, Codex, Cursor, Gemini CLI, Aider or any other)
that runs Talos on a user's behalf. It is printed by `talos guide`. Follow it in order.

## 1. What Talos does and does not do

- Talos runs an LLM research loop against one TIG challenge. It measures the current top mainnet
  algorithm as the baseline, asks an LLM for edits, compiles and scores each candidate on the
  user's compute backend (Modal, C3 or local Docker), and stops when a candidate beats the
  baseline on training and then held-out nonces, or when a budget runs out.
- Talos never submits anything to TIG. The user does that from the package Talos writes.
- Talos spends the user's money: LLM calls on their provider account or CLI subscription, and
  compute on their Modal or C3 account. Every figure it shows is an estimate, not a bill.
- Talos keeps `talos.config.json`, `.talos/secrets.json` and `runs/` in one workspace directory:
  the one given by `talos --workspace DIR <command>` or `TALOS_WORKSPACE`, else the nearest
  directory at or above the current one that holds `talos.config.json`, else
  `~/.talos/workspace`. `talos setup` prints `Workspace: <path>`, and `talos run` and
  `talos demo` print absolute result paths. Use one workspace for all commands of a job; if you
  are unsure which one applies, pass `--workspace` on every command.
- `talos --version` prints the installed version.

## 2. Spending rules

| Spends nothing | `talos guide`, `talos demo`, `talos status`, `talos show`, `talos diff`, `talos report`, `talos check-updates` (reads GHCR over the network), `talos compile --backend local` |
|---|---|
| **No compute, one small LLM call** | `talos doctor` |
| **Spends money** | `talos run`, `talos doctor --full`, `talos compile` on the modal or c3 backend |
| **Changes the user's Modal account, no compute** | `talos deploy`: redeploys the Modal app; run it, after asking, when Talos or `talos doctor` says the deployed app was built from different Talos code |
| **The user runs it, not you** | `talos setup`: it is interactive and stores credentials |

Rules:

1. Before any command in the "spends money" row, tell the user which budgets you will pass and
   wait for them to agree. Do not pick budgets on your own.
2. Never pass an `--add-*` flag (`--add-iterations`, `--add-hours`, `--add-usd`,
   `--add-compute-usd`) without the user's agreement. They raise a job's budget.
3. Never run `talos doctor --full --yes` without asking: it compiles and scores a real baseline
   outside any job budget.
4. If there is no `talos.config.json`, stop and ask the user to run `talos setup`. Then run
   `talos doctor` and fix what it reports before starting a job.
5. To try Talos with no account and no spend, run `talos demo --delay 0`. Its LLM and scores
   are scripted: its `won` status and package show what a real job produces, and there is
   nothing in it to submit. Say so when you report a demo.

## 3. Starting a job

Ask the user for the challenge, the direction and the budgets. Then run, with every answer as a
flag so nothing prompts:

```
talos run --challenge knapsack --direction-file direction.md \
  --budget-iterations 5 --budget-hours 2 --budget-compute-usd 5 --yes
```

- `--yes` skips every prompt and the `Start this job?` question. It still prints the job
  summary, whose `results:` line names the job directory, `runs/<job_id>/`.
- At least one of `--budget-usd`, `--budget-hours` or `--budget-iterations` is required.
  `--budget-compute-usd` defaults to $20 under `--yes` on Modal and C3. On the local backend
  compute is free and no compute cap applies.
- With `codex-cli`, or a model with no price-table entry, LLM spend is not measured, so
  `--budget-usd` alone is refused. Pass `--budget-iterations` or `--budget-hours`.
- Choosing a challenge: the status table at
  https://github.com/FibonAdithya/talos/blob/main/README.md#challenges says which challenges are
  tested. `talos run` warns before it starts an untested one. Prefer a tested challenge unless the
  user asks otherwise.
- Exit codes: 0 when the job ends `won`; 1 when it ends any other way, or a mainnet or dev-image
  check stops it; 2 for bad arguments, a missing config or a missing API key; 130 when an
  interrupt lands outside the research loop (before it starts, or while the package is written).
  An exit code alone does not say how the job ended: read `status` from
  `talos status JOB_ID --json`.

### Writing the direction file

The whole file is sent to the LLM on every iteration. Write what a researcher would want to know:
what has been tried and how it did, constraints, and ideas worth trying. Keep it short. Example:

```
Goal: better solution quality on knapsack without slowing the solver down.
Tried before: raising the local-search iteration count (no gain, 2x slower).
Constraints: keep the greedy construction; no new dependencies.
Ideas:
- A tighter upper bound in the branch-and-bound pruning.
- Re-rank items by value density after each fixed item.
Avoid: changes that only help one track.
```

Pass it with `--direction-file`, or put a one-line direction in `--direction "..."`.

### Budgets to propose

Propose these as a first job and let the user change them. The per-backend starting points are
ESTIMATE (unverified); the figures they come from are labelled.

| Backend | Time per iteration | Starting point to propose |
|---|---|---|
| `local` | about 3 minutes for one knapsack build and 4 nonces (MEASURED 2026-09-25 on a 16-core, 30 GB machine) | `--budget-iterations 5 --budget-hours 1` (ESTIMATE (unverified)) |
| `c3` | about 12 minutes of fixed overhead per iteration plus the nonces (MEASURED 2026-09-14); one knapsack baseline cost £0.02 (MEASURED 2026-09-15) | `--budget-iterations 5 --budget-hours 3 --budget-compute-usd 5` (ESTIMATE (unverified)) |
| `modal` | not measured | `--budget-iterations 5 --budget-hours 2 --budget-compute-usd 5` (ESTIMATE (unverified)) |

GPU challenges cost far more per nonce: on hypergraph an L40 spends about 10 minutes per
nonce-per-track on the baseline (MEASURED 2026-09-25). Keep `--nonces` at its default of 8 or
lower there, and say so to the user.

## 4. Monitoring a job

A job runs for minutes to hours. Run it in the background with its output in a log file, and
record the process id. `$!` is only set in the shell that started the job, so print it in the
same command:

```
talos run ... --yes > talos-run.log 2>&1 & echo $!
```

Then poll, at most every minute or two:

```
talos status JOB_ID --json
```

Read `status`, `stop_reason`, `iteration`, `best`, `budget` (each cap's `spent` and `left`) and
`last_event_ts`. The full schema is at
https://github.com/FibonAdithya/talos/blob/main/docs/json-output.md. `talos status --json` lists
every job, if you lost the id.

| Status | Meaning |
|---|---|
| `queued`, `measuring_baseline` | starting; the baseline is measured once per job |
| `researching` | the loop is running iterations |
| `confirming` | a candidate beat the baseline on training; it is being scored on held-out nonces |
| `won` | a candidate beat the baseline on training and held-out nonces; the job has stopped |
| `exhausted` | a budget ran out first; the job has stopped |
| `cancelled` | stopped by the user (one SIGINT); resumable |
| `paused` | the compute backend was unreachable; resumable |
| `failed` | an error stopped it; `stop_reason` says why |

Talos writes no lock file. A `researching` job whose `last_event_ts` is old, and whose process is
gone, has died. Tell the user, and resume it only if they agree (resuming spends money).

### Stopping

Send one SIGINT to the process id you recorded: `kill -INT <pid>` on Linux and macOS. Talos stops
at the next safe point, writes the package with status `cancelled` and exits. On C3 and local it
cancels the compute job in flight; on Modal, and for an LLM call in flight, it waits for that
call to finish, which can take minutes. Send it once and wait for the process to exit. Never
use `kill -9` or a plain `kill` (SIGTERM): the package is then not written. A SIGINT before the
research loop starts (during the image pull or the capacity probe, say) exits with code 130. If
the job summary had already been printed, the job is left `cancelled` and
`talos run --resume JOB_ID` continues it; before that, nothing is written. A SIGINT while the
package is being written also exits 130, and the job keeps the status it stopped with. On
Windows there is no tested way to interrupt a background job; run it in the foreground, or rely
on the budgets to stop it.

### Resuming

`talos run --resume JOB_ID` continues a `cancelled`, `paused` or `failed` job, or a job whose
process died. It keeps the job's settings and spend, and it spends money, so ask the user first
(section 2). An `exhausted` job continues only with an `--add-*` flag, and those need the user's
agreement too. A `won` job cannot be continued.

## 5. Reading results

- `talos show JOB_ID --json`: the best candidate's hypothesis, training delta, worst track,
  per-track deltas and held-out result. `--iteration N` shows another iteration.
- `talos diff JOB_ID`: the best candidate as a unified diff against the baseline.
- `runs/<job_id>/package/`: `README.md` (the submission steps), `scores.md` (per-nonce tables),
  `hypotheses.md` (every attempt and its outcome), `report.html` (the search on one page), and
  `submission/` (the code to submit). `talos report JOB_ID` rewrites `report.html`.

Deltas in the JSON output are fractions: `0.00797` is +0.797%.

What counts:

- Only a `won` job's candidate has beaten the baseline on held-out nonces. Its held-out status is
  `confirmed`.
- An `exhausted` job still has a package, but its best candidate has not passed held-out. If its
  held-out status is `false_positive`, it beat the baseline on training and lost on held-out
  nonces: tell the user it is not a candidate to submit.
- A job with `void_reason` set (the `energy_arbitrage` `hidden_seed` leak) has no result: its package
  has no submission and it cannot be resumed. Tell the user to start a new job.
- Compare `energy_arbitrage` scores only with algorithms rebuilt at the same pin on the same nonces,
  never with mainnet scores.
- Look at the worst track as well as the mean. A positive mean with a negative worst track helps
  one track at another's expense.

## 6. Reporting back to the user

When the job stops, report in this shape. Take the values from `talos status JOB_ID --json` and
`talos show JOB_ID --json`.

```
Outcome: won | exhausted | cancelled | paused | failed  (stop_reason)
Best candidate: iteration N, "<title>", training delta +X.XXX% (worst track <name> +Y.YYY%)
Held-out: confirmed +Z.ZZZ% | false_positive +Z.ZZZ% | not scored
Spend: LLM $A.AA (measured | not measured: <reason>), compute ≈$B.BB (estimate)
Package: runs/<job_id>/package/
Next step: <one suggestion, such as submitting, or resuming with more budget if the user agrees>
```

Do not tell the user to submit anything unless the outcome is `won` on a real job (not a demo).
After a demo, the next step is for the user to run `talos setup` themselves (it asks for
credentials interactively), then for you to run `talos doctor`.

## 7. Troubleshooting

Run `talos doctor` first: it checks the config, the LLM credential, the backend, mainnet and the
dev images, and says what to fix. Known failures, by the message Talos prints, are in
https://github.com/FibonAdithya/talos/blob/main/docs/troubleshooting.md. If something there does
not cover it, show the user the last lines of the log and of `runs/<job_id>/timeline.jsonl`.
