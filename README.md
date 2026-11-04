# Talos

> **Are you an AI agent?** Go to the
> [agent guide](https://github.com/FibonAdithya/talos/blob/main/talos/data/agent_guide.md) to
> set up and run Talos. It says which commands spend money, which ones the user must run
> themselves, and how to start, watch and stop a job. Once Talos is
> [installed](https://github.com/FibonAdithya/talos#install), `talos guide` prints the same guide.

Single-user autoresearch for [TIG](https://tig.foundation). You pick a TIG challenge and
describe what to explore. An LLM then repeatedly edits the current top mainnet algorithm for
that challenge, and each candidate is compiled and scored with TIG's own `tig-runtime` and
`tig-verifier` on your own Modal or C3 account, or in Docker on your own machine. The run
stops when a candidate beats the
baseline on both training and held-out nonces, or when the budget runs out. The best
candidate is written to a local, submit-ready package.

Talos never submits to TIG for you, and LLM-authored code never runs outside a container.
Only the `local` backend runs that container on your machine, with networking off,
capabilities dropped, and CPU and memory limits; the one opt-in exception is
[agentic codex](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#agentic-mode).

Driving Talos from a coding agent? Point it at this README, or tell it: "Run `talos guide`
and follow it."

## How it works

`talos run` measures the current top-adoption mainnet algorithm once as the baseline (on
`energy_arbitrage`, the top one free of the `hidden_seed` leak). Then,
until the budget runs out, it asks the LLM for an edit, compiles it on the compute backend,
and scores it on training nonces. A candidate that beats the baseline on training is scored
again on held-out nonces, and wins only if it still beats the baseline there. Baseline and
candidates run on the same nonces, fuel and hardware class. The LLM never sees the
`rand_hash` that seeds the nonces. [docs/architecture.md](https://github.com/FibonAdithya/talos/blob/main/docs/architecture.md) has the full
flowchart and the guards; [docs/compute-backends.md](https://github.com/FibonAdithya/talos/blob/main/docs/compute-backends.md) covers the
three backends.

## Install

You need Linux, macOS or Windows and [uv](https://docs.astral.sh/uv/). Nothing Rust- or
CUDA-related is needed locally: compiling and scoring happen on the compute backend.

```bash
uv tool install --python 3.12 'talos-tig[modal,anthropic]'
talos --version
```

To install the current `main` instead of the latest release:

```bash
uv tool install --python 3.12 'talos-tig[modal,anthropic] @ git+https://github.com/FibonAdithya/talos'
```

The git form was MEASURED 2026-10-27 with a local `git+file://` URL (`[modal]` only; `import modal, talos` worked in the tool environment). The GitHub URL itself has not been run.

The extras are `modal` (the Modal backend) and `anthropic` (the `anthropic` provider). The
`c3` and `local` backends and every other provider need none. Name every extra you use in one
command: MEASURED 2026-10-27, `uv tool install 'talos-tig[anthropic]'` over an install made
with `[modal]` removed the Modal client. A command that needs a missing extra exits with code
2 and prints the install command. `pip install 'talos-tig[modal,anthropic]'` is expected to work
in a virtual environment (Python 3.10 or newer), but it has not been run against a PyPI release that has the extras; `pipx` has not been tried. On Windows, see [docs/windows.md](https://github.com/FibonAdithya/talos/blob/main/docs/windows.md).

### Workspace

Talos keeps `talos.config.json`, `.talos/secrets.json` and `runs/` in one workspace
directory: the one given by `talos --workspace DIR <command>` or `TALOS_WORKSPACE`, else the
nearest directory at or above the current one that holds `talos.config.json`, else
`~/.talos/workspace`. `talos setup` prints `Workspace: <path>`. In a
fresh clone, `talos --workspace . setup` keeps everything in the clone, as before. See
[Workspace](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#workspace).

### Upgrading

Upgrade with `uv tool upgrade talos-tig` (not yet verified against a PyPI release), then run
`talos deploy` if you use Modal: every new version needs a redeploy, and until then `talos
run` pauses before spending and `talos doctor` fails. uv records the installed extras in the
tool's receipt (MEASURED 2026-10-27, uv 0.11.26, local wheel). If `talos deploy` or `talos
run` still reports a missing extra, reinstall naming every extra you use, as under Install.

## Try it with no accounts

`talos demo` runs a scripted job end to end before you create any account: no config file,
no network, no credentials and no spend. It uses the real research loop with a scripted LLM
and an in-process stand-in for the compute backend.

```bash
talos demo
```

It takes a few seconds (`--delay 0` removes the pause after each line) and prints the same
event stream a real run does: a compile failure and its fix, two iterations with no gain, a
false positive (a candidate that wins on training and loses on the held-out nonces) and a
win. It ends with:

```
Status: won (beat baseline on training and held-out nonces)
Best delta vs baseline: +4.000%
LLM spend: $0.00   Compute spend (estimated): $0.00
Package: .../runs/demo-<timestamp>-knapsack/package
You can find the code to submit at .../runs/demo-<timestamp>-knapsack/package/submission, fill out the template README.md to make it ready to submit to The Innovation Game!
```

followed by a guide to the files in the package. The demo job is written to
`runs/demo-<timestamp>-<challenge>/` in the workspace; delete that folder if you do not want
`talos status`
to list it.

## Pick a compute backend

| Backend | What you need before setup | Cost per iteration |
|---|---|---|
| `modal` (default) | A [Modal](https://modal.com) account (the free tier works) and an API token from modal.com/settings/tokens. | Container seconds only; no fixed per-job overhead. |
| `c3` | A C3 API key (`c3 apikey create`), or the `c3` CLI ([cthree.cloud](https://cthree.cloud)) 0.20.0 or newer, logged in with `c3 login` (older releases refuse the GHCR dev image before it reaches C3; check with `c3 version`). Credit on the account (`c3 topup`). | One batch job with about 12 minutes of fixed overhead (MEASURED 2026-09-14). |
| `local` | Docker running on this machine. The first run per challenge pulls a 13 GB image. GPU challenges need the NVIDIA container toolkit and are unverified. | Nothing. |

Timings and details per backend are in [docs/compute-backends.md](https://github.com/FibonAdithya/talos/blob/main/docs/compute-backends.md).

## Pick an LLM provider

| Provider | Credential | Environment variable used if `.talos/secrets.json` has no key | Default model |
|---|---|---|---|
| `anthropic` | API key | `ANTHROPIC_API_KEY` | `claude-opus-5` |
| `openai` | API key | `OPENAI_API_KEY` | `gpt-5` |
| `google` | API key | `GEMINI_API_KEY` | `gemini-2.5-pro` |
| `openrouter` | API key | `OPENROUTER_API_KEY` | `anthropic/claude-opus-5` |
| `custom` | API key and base URL of an OpenAI-compatible endpoint | `TALOS_CUSTOM_API_KEY` | none |
| `claude-cli` | A logged-in `claude` CLI on `PATH` | none | `claude-opus-5` |
| `codex-cli` | A logged-in `codex` CLI on `PATH` | none | the first model `codex debug models` lists |

Only the CLI providers support [agentic mode](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#agentic-mode). They bill through
your CLI subscription rather than per API call, so `talos run` asks them for an iteration
budget instead of a dollar budget.

## Set up and check

```bash
talos setup
talos doctor
```

`talos setup` asks for the backend, provider, model and credentials, checks each one (on Modal
it also deploys the benchmark app to your account), and only then writes `talos.config.json`
and `.talos/secrets.json` in the workspace. Run it again to change provider, model or backend;
on Modal, `talos deploy` redeploys after upgrading Talos. Every prompt and check is listed
under [`talos setup`](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#talos-setup).

`talos doctor` spends no compute, but its provider check makes one small LLM test call. It
checks the config, the LLM credential, the backend, mainnet and the dev images, and says what
to fix for each failure. Then run `talos doctor --full` once: it compiles and scores one real
mainnet algorithm on your backend, and shows the cost and asks `Spend this?` (default no)
before it spends anything. The one measured figure is for knapsack on C3 (MEASURED
2026-09-15: £0.02 and 12 min 20 s); every other figure is labelled `ESTIMATE (unverified)`.
See [`talos doctor`](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#talos-doctor).

## Your first real run

```bash
talos run --challenge knapsack \
  --direction "Try a tighter upper bound in the branch-and-bound pruning" \
  --budget-usd 20 --budget-hours 4 --budget-compute-usd 10 --yes
```

`talos run` with no flags asks for each answer instead. Before it spends anything it prints a
job summary. Press `Ctrl-C` once to stop: Talos packages the best candidate so far, and
`talos run --resume <job_id>` continues later. `talos status` lists your jobs. When the job
stops, open `runs/<job_id>/package/report.html` in a browser to see how the search evolved
(see [Where results land](https://github.com/FibonAdithya/talos#where-results-land)). Watching,
stopping, resuming and submitting are covered in [Running a job](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#running-a-job).

## What to expect

| Backend | Time per iteration | Compute cost per iteration |
|---|---|---|
| `c3` | About 12 minutes of fixed overhead before the first nonce is scored, plus the nonces (MEASURED 2026-09-14; see [C3 timings](https://github.com/FibonAdithya/talos/blob/main/docs/compute-backends.md#c3-timings)). | Not measured per iteration. The closest measured figure is £0.02 for one knapsack baseline on 2 training and 2 held-out nonces (MEASURED 2026-09-15). |
| `local` | About 3 minutes for one knapsack build and 4 nonces (MEASURED 2026-09-25 on a 16-core, 30 GB machine; see [Local timings](https://github.com/FibonAdithya/talos/blob/main/docs/compute-backends.md#local-timings)). The first run per challenge also pulls the image, clones the pin and does a warm-up build: 90 s for knapsack with the image already pulled (MEASURED 2026-09-25, same machine). | £0: it runs on your machine. |
| `modal` | not measured | not measured |

Most jobs end `exhausted`: a budget ran out before a candidate beat the baseline on both
training and held-out nonces. An `exhausted` job's package is not a candidate to submit, and
`talos run --resume <job_id> --add-iterations N` (or another `--add-*` flag) continues it.

## Challenges

"Tested" means at least one Talos job got past its baseline on some backend. `talos run`
warns before it starts a job on a challenge that is not tested.

| Challenge | Hardware | Status |
|---|---|---|
| `knapsack` | CPU | tested |
| `energy_arbitrage` | CPU | tested |
| `job_scheduling` | CPU | tested |
| `hypergraph` | GPU | tested |
| `satisfiability` | CPU | untested |
| `vehicle_routing` | CPU | untested |
| `vector_search` | GPU | untested |
| `neuralnet_optimizer` | GPU | untested |

## Where results land

Each job writes to `runs/<job_id>/` in the workspace. When it stops, `runs/<job_id>/package/` holds the
hand-back package: the code to submit in its `submission` folder, per-nonce `scores.md`, the
`hypotheses.md` log, a one-page `report.html` and a `README.md` with the submission steps.
You submit it yourself.

`report.html` shows how the search evolved over the run: a plot of every iteration's training
score against the baseline, with a line from each candidate to the candidate it was edited
from. Selecting a candidate shows its diff against that parent. Below the plot are per-track
scores for the best candidate against the baseline, the best candidate's diff and the
hypothesis log. It is one file that loads nothing from the network. `talos report <job_id>`
writes it again from the run directory. The full layout is in
[Where results land](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md#where-results-land).

## Documentation

- [docs/usage.md](https://github.com/FibonAdithya/talos/blob/main/docs/usage.md): every command and flag, agentic mode, results and budget.
- [docs/troubleshooting.md](https://github.com/FibonAdithya/talos/blob/main/docs/troubleshooting.md): known failures, by the message Talos prints.
- [docs/windows.md](https://github.com/FibonAdithya/talos/blob/main/docs/windows.md): installing and running on Windows.
- [docs/maintainers.md](https://github.com/FibonAdithya/talos/blob/main/docs/maintainers.md): development, live smoke tests, dev image upgrades.
- [docs/architecture.md](https://github.com/FibonAdithya/talos/blob/main/docs/architecture.md): the research loop, its guards and the agentic sandbox.
- [docs/compute-backends.md](https://github.com/FibonAdithya/talos/blob/main/docs/compute-backends.md): how Modal, C3 and local Docker are used, with timings.
- [docs/json-output.md](https://github.com/FibonAdithya/talos/blob/main/docs/json-output.md): the JSON schema of `talos status --json` and `talos show --json`.

## Licence

GPLv3 (see `LICENSE`). `talos/search_replace.py` is lifted from
[tig-foundation/prometheus-swarm](https://github.com/tig-foundation/prometheus-swarm)
(its file scripts/search_replace.py), also GPLv3; `talos/agentic.py`'s sandbox settings mirror
Prometheus's `_build_sandbox_settings` design without copying its code.
