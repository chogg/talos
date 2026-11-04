# JSON output of `talos status` and `talos show`

`talos status --json`, `talos status JOB_ID --json` and `talos show JOB_ID --json` print one
JSON object to stdout and nothing else. They are for scripts and coding agents. All three read
`runs/<job_id>/` only: no config, credentials, network or Docker, and they write nothing.

## Stability

- Every object has `"schema_version": 1` as a top-level key.
- Adding a key is a compatible change and keeps the version. Removing or renaming a key, or
  changing what a value means, bumps `schema_version`.
- Deltas are raw fractions: a +0.797% improvement is `0.00797`. Times are Unix seconds (a
  float, UTC). Money is US dollars.
- An absent value is `null`; a documented key is never missing. A whole object can be `null`
  (for example `best` on a job with no candidate); its keys are then absent with it.
- Errors are not JSON. On an error the command prints a message on stderr, nothing on stdout,
  and exits 1. A bad flag exits 2.
- LLM spend that Talos cannot measure (a `codex-cli` job, or a model with no price-table
  entry) is `null`, never `0.0`; `llm_priced` is `false` and `llm_unpriced_reason` says why.

In the tables, a key is a dotted path from the top of the object. `[]` means each element of a
list. A map keyed by track name is documented down to the map itself.

`tests/test_inspect_cli.py` checks every key path below against the commands' output.

## talos status --json

One row per directory under `runs/` that holds a `job.json` or a `state.json`, sorted by job
id. A directory that cannot be read is listed with `status` `"unreadable"` and `error` set.
With no jobs, `jobs` is `[]`.

| key | type | meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `jobs` | list | one object per job |
| `jobs[].job_id` | string | the directory name under `runs/` |
| `jobs[].challenge` | string or null | the challenge, e.g. `knapsack` |
| `jobs[].provider` | string or null | the LLM provider, e.g. `claude-cli` |
| `jobs[].model` | string or null | the LLM model |
| `jobs[].status` | string | one of the job statuses listed below, or `unreadable` |
| `jobs[].iteration` | int or null | the highest iteration started |
| `jobs[].best_mean_rel_delta` | float or null | the best candidate's mean training delta vs the baseline; null with no candidate |
| `jobs[].llm_usd` | float or null | LLM spend so far; null when it is not measured |
| `jobs[].llm_priced` | bool or null | whether `llm_usd` is a measurement |
| `jobs[].llm_unpriced_reason` | string or null | why LLM spend is not measured |
| `jobs[].compute_usd` | float or null | estimated compute spend so far |
| `jobs[].last_event_ts` | float or null | time of the last readable `timeline.jsonl` event |
| `jobs[].error` | string or null | why the job could not be read; null unless `status` is `unreadable` |

Statuses: `queued`, `measuring_baseline`, `researching`, `confirming`, `paused`, `won`,
`exhausted`, `failed`, `cancelled`. A `researching` job may belong to a process that has died:
Talos writes no lock file, so compare `last_event_ts` with the current time.

## talos status JOB_ID --json

| key | type | meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `job_id` | string | the directory name under `runs/` |
| `job` | object | the job's settings from `job.json` (`JobSpec.redacted()`: without the nonce seed). Its fields follow `JobSpec` in `talos/state.py` and are outside this versioned contract |
| `status` | string | the job status |
| `stop_reason` | string or null | why the job stopped, e.g. `budget exhausted: iterations` |
| `iteration` | int | the highest iteration started |
| `runs_since_improvement` | int | iterations since the best last changed |
| `hardware` | string or null | the GPU or CPU profile the job is frozen to; null when there is no choice |
| `scoring` | object | research scoring |
| `scoring.requested` | string | `metered` or `native`, as `talos run --scoring` set it |
| `scoring.actual` | string or null | what research scoring runs; null before calibration |
| `best` | object or null | the current best candidate; null with no candidate |
| `best.iteration` | int | its iteration |
| `best.title` | string | its hypothesis title |
| `best.mean_rel_delta` | float or null | its mean training delta vs the baseline; null only for a state.json whose best has no recorded delta (Talos always writes one) |
| `best.holdout_status` | string | `confirmed`, `false_positive` (its held-out run failed; the best is not cleared) or `not_scored` |
| `best.holdout_mean_rel_delta` | float or null | its held-out mean delta, whether it passed or failed; null when not scored |
| `budget` | object | the caps after any `--add-*` extensions, and spend |
| `budget.iterations.cap` | int or null | null means no cap |
| `budget.iterations.spent` | int | iterations run |
| `budget.iterations.left` | int or null | `max(0, cap - spent)`; null with no cap |
| `budget.hours.cap` | float or null | null means no cap |
| `budget.hours.spent` | float or null | working hours: to the last event for a stopped (`won`, `exhausted`, `failed`, `cancelled`, `paused`) job, to now otherwise; null for a stopped job with no events |
| `budget.hours.left` | float or null | `max(0, cap - spent)`; null with no cap or no spent value |
| `budget.usd.cap` | float or null | LLM dollar cap; null means no cap |
| `budget.usd.spent` | float or null | LLM spend; null when it is not measured |
| `budget.usd.left` | float or null | null with no cap or no measured spend |
| `budget.compute_usd.cap` | float or null | compute dollar cap; null means no cap |
| `budget.compute_usd.spent` | float | estimated compute spend |
| `budget.compute_usd.left` | float or null | null with no cap |
| `llm_priced` | bool | whether LLM spend is a measurement |
| `llm_unpriced_reason` | string or null | why it is not |
| `last_event_ts` | float or null | time of the last readable timeline event |
| `activity` | object | what the job is doing |
| `activity.pending` | object or null | the compute job in flight, from `state.json`; null when none |
| `activity.pending.purpose` | int or string | the iteration number, or `baseline` and similar |
| `activity.pending.title` | string or null | the hypothesis title, when it has one |
| `activity.last_event` | object or null | the last readable timeline event; null when none |
| `activity.last_event.kind` | string | the event kind, e.g. `scored`, `hypothesis`, `won` |
| `activity.last_event.ts` | float | its time |
| `activity.last_event.iteration` | int or null | its iteration, when it has one |
| `activity.last_event.text` | string | the one-line summary `talos run` prints for it, or the kind |

## talos show --json

`talos show JOB_ID [--iteration N|best]`; the default is `best`.

| key | type | meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `job_id` | string | the directory name under `runs/` |
| `iteration` | int | the iteration shown |
| `parent` | int | the iteration it was built on; `0` is the root (the baseline, the template, or the base algorithm) |
| `parent_label` | string | `baseline`, `template`, `base` or `#N` |
| `chain` | int | its annealing chain; `0` for a greedy job |
| `kind` | string | `best`, `false_positive`, `demoted`, `improved`, `accepted`, `rejected` or `failed` |
| `outcome` | string | the loop's outcome, e.g. `improved`, `failed:score`, `failed:compile` |
| `error` | string or null | the error text for a failed iteration |
| `hypothesis` | object | what the LLM proposed |
| `hypothesis.title` | string | its title |
| `hypothesis.description` | string | its full description |
| `hypothesis.strategy_tag` | string | its strategy tag |
| `hypothesis.design_family` | string | its design family; `""` when none |
| `scores` | object | how it scored |
| `scores.mean_rel_delta` | float or null | mean training delta vs the baseline; null when it did not score |
| `scores.worst_track` | string or null | the track with the lowest delta |
| `scores.worst_rel_delta` | float or null | that track's delta |
| `scores.runtime_ratio` | float or null | candidate runtime over baseline runtime, when recorded |
| `scores.metered_rel_delta` | float or null | the metered delta, when a native-scoring job validated it |
| `scores.tracks` | object or null | track name to training delta; null for runs from before the lineage tree |
| `scores.holdout_status` | string | `confirmed`, `false_positive` or `not_scored` |
| `scores.holdout_mean_rel_delta` | float or null | held-out mean delta, whether it passed or failed |
| `scores.tables` | list or null | the best's per-track tables, baseline vs best; null for any other iteration |
| `scores.tables[].heading` | string | e.g. `Training nonces`, `Held-out nonces` |
| `scores.tables[].tracks` | list | one row per track; empty when `error` is set |
| `scores.tables[].tracks[].track` | string | track name |
| `scores.tables[].tracks[].base_mean` | float | baseline mean quality |
| `scores.tables[].tracks[].cand_mean` | float | best mean quality |
| `scores.tables[].tracks[].rel_delta` | float | relative delta |
| `scores.tables[].tracks[].cand_errors` | int | nonces the best failed |
| `scores.tables[].tracks[].n` | int | nonces scored |
| `scores.tables[].mean_rel_delta` | float or null | mean delta over the tracks |
| `scores.tables[].error_rate` | float or null | fraction of nonces the best failed |
| `scores.tables[].error` | string or null | why the table could not be computed |
| `files` | list | paths of the iteration's saved files under `runs/<job_id>/iterations/NNNN/`, plus `runs/<job_id>/best/` for the best, relative to the current directory when the workspace is under it and absolute otherwise; empty when nothing was saved |
