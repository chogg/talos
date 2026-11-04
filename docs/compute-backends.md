# Compute backends

How Talos uses Modal, C3 and local Docker, what a C3 job and a local job cost in time, and
what maintainers do to keep the C3 and local backends working. For choosing and configuring a backend, see
[README.md](../README.md#pick-a-compute-backend).

## Transports

On Modal, `talos setup` (or, after an upgrade, `talos deploy`) deploys the benchmark app to your
account, and Talos calls its compile and score functions. The app carries a marker function,
`deployed_<12 hex digits>`, named after a digest of the Talos code it was built from. `talos run`
and `talos compile` look the marker up, and stop with a message naming `talos deploy` when the client's
digest is not deployed; `talos doctor` reports the same.

On the C3 backend Talos uses C3's hosted MCP endpoint (`https://api.cthree.cloud/mcp`) when
a C3 API key is configured, and the `c3` CLI when it is not. The key path needs no C3
install, which is what makes the C3 backend usable on Windows, and it uploads job.sh already
marked executable. Both paths submit the same job directory, written to
`runs/<job_id>/c3/<n>/`.

On the local backend Talos writes the same job directory in a local flavour and runs it in a
Docker container on this machine through `talos/local_transport.py::DockerTransport`, which
satisfies the C3 transport protocol (deploy, status, cancel, fetch). The container's name is
the job id. See [Local backend](#local-backend).

## Scoring workers

Every backend scores a CPU challenge's nonces on all of its container's cores at once, through
one worker pool, `talos/inside.py::run_nonces`. A GPU challenge always scores one nonce at a
time: nonces would serialise on the device, and a batch of them would outlive its timeout.

| Backend | Workers | Where the count is fixed |
|---|---|---|
| Modal | The container's CPU count, `talos/challenges.py::modal_workers` (4). The client sends one `score_batch_<challenge>` call per `modal_workers` nonces, so a container scores its batch on every core and the batch's wall time is its slowest nonce's, which keeps the function timeout at one nonce plus slack. The container returns its own wall seconds, and the client charges those at the 4-core rate. | `talos/challenges.py::ChallengeSpec.cpu` |
| C3 | `talos/challenges.py::c3_workers` (4, the CPU profile's vCPUs) | `talos/challenges.py::C3_CPU_WORKERS` |
| Local | The container's `--cpus` limit, `talos/challenges.py::local_workers` | `talos setup` |

The Modal packing is part of the hardware class, `cpu4-mem8192-x4`
(`talos/challenges.py::hardware_class`): four nonces sharing a container's cores and memory
time differently from one nonce alone, so a Modal CPU baseline cached before batching is
never a hit and is re-measured on the first run after upgrading. Upgrading also needs
`talos deploy`, because the per-nonce `score_nonce_<challenge>` function is replaced. GPU
classes are unchanged. Each nonce in a Modal batch has a quarter of the container's 8 GiB,
against 4 GiB on C3's profile; a memory bump is a change to `ChallengeSpec.memory_mib`, and
therefore to the class.

MEASURED 2026-09-24 on the development machine with a stand-in `tig-runtime` that sleeps for
one second, so these are the pool's overheads and nothing about real nonces: the C3 job runner
scored 16 nonces in 16.2 s with one worker, 4.1 s with four and 1.1 s with sixteen, and a
Modal batch of 4 took 4.04 s with one worker and 1.04 s with four.

MEASURED 2026-09-24 on Modal, live, with the redeployed app: the live test
(`tests/test_live.py::test_baseline_compiles_and_scores`, knapsack, cached compile) passed in
12.3 s with both nonces in one batch, and a follow-up of 8 nonces of the top knapsack
algorithm ran as two batches of 4, each container reporting 1.9 s and 1.8 s of wall time
against 4.7 s and 4.4 s of summed per-nonce runtime, so four nonces did share a container at
once; the client's estimate for the 8 nonces was $0.0004.

Every backend runs a nonce as one 12-field task tuple, defined above
`talos/inside.py::run_task`: challenge id, track, rand hash, nonce, the artifact path (`so`),
fuel, the per-nonce timeout, the PTX path, the work directory, the hyperparameters, the mode
and the native time budget in microseconds. `run_task` dispatches on the mode: `metered`
calls `run_nonce` (`tig-runtime` under the fuel), `native` calls
`talos/inside.py::run_nonce_native` (the `talos-native` runner under the budget, then
`tig-verifier`). In native mode `so` is the runner binary.

On Modal, an artifact is stored under `/artifacts/<challenge>/<artifact id>/`. A native build's
hash includes its mode, and it is stored there as `talos-native` beside `algo.ptx`, never over
the metered `algo.so`. A Modal app deployed before native scoring keeps working for metered
jobs and refuses a native one as a stale deploy (`talos/bench.py::_stale_deploy`); run
`talos deploy` to redeploy.

## Hardware fallback

A job runs on the first option in a preference list that has capacity when the job starts,
and stays on it for the whole job. For GPU challenges the lists are
`talos/challenges.py::MODAL_GPUS` (`L40S`, `A100-80GB`, `H100`) and
`talos/challenges.py::C3_GPU_CLASSES` (`l40`, `a100`, `h100`): only GPUs with at least the
L40S's 48 GB of VRAM, cheapest first, because a smaller card could OOM a candidate the
baseline fit. C3's `l40` class spans both the L40 and the L40S. For CPU challenges on C3 the
list is `talos/challenges.py::C3_CPU_PROFILES` (`cpu-d3-4vcpu-16gb`, `cpu-e2-4vcpu-16gb`):
the two 4 vCPU / 16 GB profiles C3 offers (AMD EPYC Genoa, then Intel Ice Lake, both £0.11/h
per `c3 list -al` on 2026-09-24), the profile every existing CPU baseline was measured on
first. The 4 GB `cpu-n1` profile is under the 8 GB the challenge spec declares and is left
out for the same reason as the smaller GPUs. On Modal a CPU challenge has one class
(`cpu=4`, 8 GB) and nothing to fall back to, so nothing is probed or frozen there.

The choice is made once and frozen, never per call, because the baseline and every candidate
must score on the same hardware class (AGENTS.md invariant 1) and the class is in the baseline
cache key. Modal's own `gpu=["L40S", "A100-80GB"]` fallback list is deliberately not used: it
picks a GPU per container. Instead the Modal deploy (`talos setup` or `talos deploy`) creates one `compile_<challenge>_<gpu>` and
`score_batch_<challenge>_<gpu>` pair per GPU, plus one `probe_<gpu>` function on a stock image,
and `talos/cli.py::freeze_hardware` fixes the hardware before the baseline:

| Backend | Probe | "No capacity" means | Cost of one probe |
|---|---|---|---|
| Modal | `probe_<gpu>` is spawned; the call must return within `ModalBench.probe_window_s` (120 s). A call still queued then is cancelled, or it would run and bill whenever that GPU freed up. | Every probe timed out. | The wait at the GPU's rate. MEASURED 2026-09-24: the L40S probe returned in 33 s from a cold start and was charged $0.018. |
| C3 | A two-minute job whose script is `true`, on `ubuntu:24.04`, written by `talos/c3_jobdir.py::write_probe_dir` to `runs/<job_id>/c3/probe-<option>/`, asking for CUDA only on a GPU class. It must leave the queue within the capacity window (`C3Bench.pending_timeout_s`, 30 min); once RUNNING it is cancelled. A job still queued at the window's end is cancelled and the next option tried. A deploy C3 refuses outright as out of stock (HTTP 409 `GPU_OUT_OF_STOCK`, which the CPU profile returned on 2026-09-23) moves to the next option at once; any other deploy failure pauses the run. | Every option stayed queued for the whole window or was refused, so up to 90 minutes for a GPU challenge and 60 for a CPU one before the run pauses. | The probe's two-minute walltime at the option's rate, the most it can bill. MEASURED 2026-09-24 over MCP: the `l40` probe left the queue in 6 s, was cancelled, and was charged $0.067. |

The choice is `hardware` in `state.json`, a `hardware_selected` event in the timeline, and
`TALOS_HARDWARE` in the agentic sandbox's environment so `talos compile` there builds on the
same hardware. A resume hands the saved choice back and probes nothing. A job that was created
before this choice existed (a measured baseline and no `hardware` field) is frozen to the
first option, which is the only one there was; a `state.json` from the release where the field
was named `gpu` is read as well. If the frozen option runs out of capacity later, the run
pauses as it does for any other capacity shortage and resumes on the same option; it never
switches. When no option answers, the run is paused with the list tried, and
`talos run --resume` tries again.

The probe is a compute call like any other: the budget is checked before it, so a job with
`--budget-compute-usd 0` stops before probing, and its estimated cost goes on compute spend.
Modal charges the wait for a successful probe at the GPU's rate (queue time included, so never
below what the container billed); C3 charges the probe's whole two-minute walltime at the
option's rate, the most it can bill. A probe that never started charges nothing.

The local backend has one hardware class and no probe, and so does a CPU challenge on Modal.
The Modal app must be redeployed (`talos deploy`) after upgrading to a version with this table:
the old deploy has no per-GPU functions, and a run against it stops with "the deployed
talos-bench app was built from different Talos code than this client; run `talos deploy`".

Both GPU probes were run against live accounts on 2026-09-24 (`tests/test_live.py`, the
`c3_gpu_probe` and `modal_gpu_probe` tests): C3 accepted the `ubuntu:24.04` probe with
`requires_accelerator: cuda`, and the redeployed Modal app scheduled the `probe_l40s` call.
The C3 CPU probe ran live the same day at the start of `test_c3_knapsack_job`, over the `c3`
CLI: it chose `cpu-d3-4vcpu-16gb` in about a minute, the probe job having finished before a
poll saw it running. The out-of-stock deploy path has not been exercised live; its match is
on the wording seen on 2026-09-23.

## C3 job directories

On the C3 backend, each iteration gets `runs/<job_id>/c3/<n>/` in the
[run directory](usage.md#where-results-land), and the baseline measurement gets
`runs/<job_id>/c3/baseline/`. Each holds the files submitted to C3 for that job (a .c3
config, a job.sh entrypoint, a payload.json, and copies of the modules the container needs)
and the pulled artifacts, under `<c3-job-id>/artifacts/`. `<c3-job-id>` is C3's own job id,
distinct from the Talos `<job_id>`. The payload carries the job's `rand_hash` and is
uploaded to C3's workspace store as part of the job directory.

## C3 timings

On C3, one iteration is one batch job with about 12 minutes of fixed overhead before any
nonce is scored — MEASURED 2026-09-14 on a spike run (knapsack, 4 vCPU): about 4 minutes
to script start, 466 seconds to build the candidate, then 1 to 2 seconds per nonce at
mainnet fuel. The release smoke test on 2026-09-15 (MEASURED, knapsack, 4 vCPU, two
training and two held-out nonces) took 12 min 20 s from submission to result: about 2 minutes
queued, 470 seconds to build, 1.3 to 1.7 seconds per nonce; the client's cost estimate was
$0.027 and the account balance fell by £0.02. Modal has no equivalent per-job overhead.

## C3 dev images

C3 pulls the official TIG dev image straight from GHCR,
`ghcr.io/tig-foundation/tig-monorepo/<challenge>/dev:<DEV_IMAGE_TAG>`, the same reference
Modal builds from, so a candidate is compiled in one image whichever backend runs it. There
is no mirror to maintain: a `DEV_IMAGE_TAG` bump takes effect on the next job. `talos run` on
C3 checks that the tag exists on GHCR (an anonymous pull-scoped token, then a manifest GET)
before the baseline is measured, and fails with a `DEV_IMAGE_TAG` hint if it is missing. The
local backend pulls the same image with `docker pull` the first time a challenge is run.

The current pin is `0.0.8`, which includes TIG's [seed-obfuscation fix](https://github.com/tig-foundation/tig-monorepo/commit/2703c9af12df835fcf08112722b915e34cd1c035).
The monorepo pin includes the same fix: compiling against the old energy-arbitrage
sources would still expose its hidden seed through serialization and debug formatting.
`talos/inside.py::challenge_problem` reads the checkout's `mod.rs` before every energy_arbitrage
build and fails the build if the fix is missing. Nothing ties the image tag to a commit, so the
image's part of that claim is unverified. Keep both pins fixed for a run; a moving `latest` tag would let image contents change
without invalidating cached artifacts or measurements. An upgrade gets new baseline
cache keys and local volumes; existing runs with different or unrecorded pins must be
restarted as new jobs. Redeploy Modal with `talos deploy` and run `talos doctor --full`
after upgrading.

`talos check-updates` compares the configured tag with the newest published stable numeric
tag for each challenge, using GHCR metadata without pulling image layers. New real jobs
make the same advisory check for their challenge. It never changes either pin or redeploys
anything; newer tags still need a source compatibility review. Failed automatic lookups
do not stop jobs, and the manual command explicitly reports an unknown update status.

## Local backend

The local backend is the C3 bench with a Docker transport. One evaluate call is one detached
`docker run` of the challenge's dev image, polled with `docker inspect` until it exits, with
`results.json` and `build.log` copied out of the stopped container by `docker cp`. Nothing is billed; the compute figure on the status line
reads `$0.00`.

### Volumes and the prepare step

Two named Docker volumes per challenge, keyed by the first 12 hex characters of
`sha256(MONOREPO_REF + "\0" + DEV_IMAGE_TAG)`, so a pin bump gets fresh volumes and never
reuses a target directory built against another monorepo:

| Volume | Mounted at | Holds |
|---|---|---|
| `talos-app-<challenge>-<key>` | `/app` | The monorepo checkout at `MONOREPO_REF` and its cargo target directory. |
| `talos-cargo-<challenge>-<key>` | the image's cargo home (`/root/.cargo` in the 0.0.8 images) | The crate registry, so builds work with networking off. |

What the volumes buy is the clone, the dependency compile and `--network none`; they do not
make a build incremental. The 0.0.8 image's `build_so` runs an LLVM fuel-instrumentation pass
(`opt`, `llc`, `clang`) over every dependency's IR, standard library included, on every
build, and that pass is most of the build time (see [Local timings](#local-timings)).

`talos/local_transport.py::prepare` runs before the baseline on every `talos run` (and before
`talos compile`). Each step is skipped when its marker is present, and one line is printed per
step performed:

1. `docker pull` of the dev image if `docker image inspect` does not find it (about 13 GB).
2. Create the cargo volume, labelled with the image's cargo and rustup home paths.
3. Create the app volume.
4. Clone the monorepo tarball at the pin into `/app` (marker `/app/.talos-ready`).
5. Warm-up build: `build_algorithm` of the first algorithm the pinned monorepo ships for the
   challenge, alone in its crate (see [The crate is pruned to the candidate](#the-crate-is-pruned-to-the-candidate)),
   with networking on, so the registry and the dependency compile are populated by code that
   is not LLM-authored (marker `/app/.talos-warm`).
6. For a GPU challenge, read the GPU name with `nvidia-smi` for the hardware class.

The warm-up also runs `cargo fetch`, so the crates the `talos-native` runner needs are in the
cargo volume and its build works with networking off. Its marker is `.talos-warm-2`
(`talos/local_transport.py::WARM_MARKER`); a volume warmed before the fetch was added has
only `.talos-warm`, and is warmed again on the next run.

Both volumes are shared by every job of the challenge on this machine, and the runner stages
the candidate into the checkout on `/app` (`talos/c3_job.py::main`), so jobs take turns: the
local job script (`talos/c3_jobdir.py::local_job_sh_text`) runs the runner under
`flock /app/.talos-lock`, and the clone and warm-up scripts hold the same lock. Two runs of one challenge at once, or a `talos compile` beside a
run, therefore serialise on the build instead of compiling each other's files; a waiting job's
time limit still counts from its start.

To reclaim space after a pin bump, list and remove the old volumes by hand:

```bash
docker volume ls --filter name=talos-
docker volume rm <old volume names>
```

### The job container

Every flag is in `talos/local_transport.py::run_args`; loosening them is a human decision
(`AGENTS.md`, "What requires a human").

| Flag | Why |
|---|---|
| `-v <job dir>:/work:ro` | The payload and the talos modules; the runner only reads them. |
| (no mount for `/artifacts`) | `results.json` and `build.log` stay inside the container; Talos copies them out with `docker cp` once the job has stopped. Nothing writable on the host is mounted, so root in the container cannot leave a file (setuid or otherwise) on the host. |
| `-v talos-app-…:/app`, `-v talos-cargo-…:<cargo home>` | The volumes above. |
| `-e CARGO_HOME=… -e RUSTUP_HOME=…` | Cargo and rustup are told where their files are, so the mounted registry is the one used whatever the image's profile does. |
| (no `--user`) | The job runs as root inside the container: the image keeps cargo and rustup under `/root`, mode 700, so a non-root user cannot build. With every capability dropped, no network, and no writable host mount, what root can reach is the container's own filesystem and the two volumes. Files under `runs/` are written by Talos itself, so they belong to the user. |
| `--cpus N --memory Mg` | From `talos setup`, which defaults them to what `docker info` reports and refuses more: Docker rejects a container with more CPUs than the daemon has (MEASURED 2026-09-24, Docker 29.1.3: "range of CPUs is from 0.01 to 16.00"), and accepts one with more memory than the daemon has (MEASURED: `--memory 999g` on a 30 GiB daemon), which would leave the limit meaning nothing. On Docker Desktop both figures are the VM's (Settings > Resources), not the machine's. Part of the hardware class. |
| `--network none` | The checkout and the registry are on the volumes; the job never needs the network. |
| `CARGO_NET_OFFLINE=true` (in the job script, `talos/c3_jobdir.py::local_job_sh_text`) | Cargo fails at once instead of retrying a network it cannot reach. The native build's crates come from the `cargo fetch` warm-up. MEASURED 2026-09-29 on knapsack: the CPU runner compiled offline on the first attempt. |
| `--cap-drop ALL --security-opt no-new-privileges --pids-limit 4096` | Nothing in the build or the runtime needs a capability; a fork bomb fails the container, not the machine. |
| `--gpus all` | GPU challenges only. |

Job directories land at `runs/<job_id>/local/<purpose>/`, and the container's artifacts under
`<job dir>/<container name>/artifacts/` (copied out of the container). The container name is
`talos-<8 hex of the run dir>-<purpose>-<request hash>`; the request hash is derived from a
hash of the rand hash, never the rand hash itself. Docker has no wall-clock limit of its own,
so the transport stores the job's limit as a container label and kills the container when it
is exceeded, reporting `TIMED_OUT` exactly as C3 would. A stopped container is kept until the
next deploy of the same run, because its artifacts are read from it; exited containers of
earlier iterations are removed then. `talos compile` removes its own containers when it
returns or is interrupted: the agentic sandbox runs it from a new temporary directory each
iteration, so its run key never recurs and no later deploy would remove them. The local build
allowance in the time limit is one hour (`talos/c3_jobdir.py::LOCAL_BUILD_ALLOWANCE_S`),
against 20 minutes on C3, because a
candidate build took about 15 minutes on 16 cores before the crate was pruned to the
candidate (about 3 minutes since; see [Local timings](#local-timings)) and longer on fewer.
The allowance is kept for the smaller machines.

The hardware class is `local-<host>-cpu<N>-mem<M>` (or `local-<host>-gpu-<name>-cpu<N>-mem<M>`), so a local
baseline never matches a Modal or C3 one, and a changed CPU or memory setting is a re-measure.

### Local timings

MEASURED 2026-09-25 on a 16-core, 30 GB machine (`tests/test_live.py::test_local_knapsack_job`,
knapsack, two training and two held-out nonces), with the crate pruned to the candidate
(see [below](#the-crate-is-pruned-to-the-candidate)):

| Step | Time |
|---|---|
| Live run, fresh volumes: `prepare` (image present; create volumes, clone, warm-up build) | 90 s (`prepare_s: 90`) |
| Same run: one job, build plus 4 nonces (1.3 to 1.4 s per nonce) | 2m57s (`job_s: 177`; `1 passed` in 4m33s) |

The same test for job_scheduling is in the pruning section below: `prepare_s: 103`,
`job_s: 990`, with nonces of 150 to 165 s each.

Before the crate was pruned, MEASURED 2026-09-23 on the same machine, same test, when the
build compiled the 13 shipped knapsack algorithms beside the candidate:

| Step | Time |
|---|---|
| Spike: clean `build_algorithm` of a shipped algorithm, no cache | 12m0s |
| Live run 1: warm-up build in `prepare` (clone done, registry warm) | 12m (14:31 to 14:43 UTC) |
| Live run 1: the job's `build_algorithm` of the candidate, cargo cache warm | 14m44s (container start to exit) |
| Live run 2: `prepare` with every marker present | 1 s (`prepare_s: 1`) |
| Live run 2: one job, build plus 4 nonces (1.3 to 1.7 s per nonce) | 14m4s (`job_s: 844`; `1 passed` in 14m7s) |
| Live run 3, the shipped copy-out transport: prepare with markers present / one job | 1 s / 14m59s (`job_s: 899`; `1 passed` in 15m3s) |

Those 2026-09-23 runs showed the job's build no faster than the warm-up build, which was
read at the time as the instrumentation pass being the cost. The 2026-09-25 runs disprove
that: with the crate pruned to the candidate the same job fell from 844 s to 177 s, so the
compile of the shipped algorithms was most of it. The instrumentation pass still runs over
every dependency's IR on every build; caching the instrumented objects per IR file would
remove that and is out of scope here.

### The crate is pruned to the candidate

`build_algorithm` compiles every module the challenge crate's `mod.rs` lists, on one codegen
unit (`build_so` passes `-C codegen-units=1 -C opt-level=3`), so the crate's size sets the
build time and memory whatever the candidate is, and core count does not help. The knapsack
crate at the pin is 13 algorithms and 2.1 MB of Rust. The job_scheduling crate is 13
algorithms and 15.0 MB, four of them 2.4 to 3.2 MB each.

MEASURED 2026-09-25 on the 16-core, 30 GB machine, `build_algorithm ember_stromboli` in the
job_scheduling dev image with no memory cap:

| Crate at build time | Result |
|---|---|
| The pinned crate plus the candidate (14 algorithms, 345k lines) | stopped after 133 min still inside LLVM optimisation, rustc at 23.4 GB and the host swapping; one thread busy throughout |
| The candidate alone | `.so` in 629 s, rustc peak 3.4 GB |

Through Talos itself (MEASURED 2026-09-25, `TALOS_LIVE_CHALLENGE=job_scheduling` with
`tests/test_live.py::test_local_knapsack_job`, same machine, job memory limit 8 GiB): `prepare`
from fresh volumes, clone and pruned warm-up included, 103 s; one job building
ember_stromboli plus 4 nonces, 990 s, of which the nonces took 150 to 165 s each.

The same build on C3's 16 GB CPU profile was killed on 2026-09-23 once rustc passed the
limit. `talos/inside.py::stage_algorithm` therefore rewrites the crate's `mod.rs` to list
the candidate alone, keeping the pinned file beside it as `mod.rs.talos-pristine`, and
`unstage_algorithm` puts the pinned file back byte for byte. TIG's own CI builds an
algorithm from its branch, whose `mod.rs` lists only that algorithm, so the pruned crate is
the closer match to the mainnet build. The crate layout is part of every artifact id and
baseline key (`inside.CRATE_LAYOUT`), so artifacts and baselines from before the pruning are
re-measured once rather than compared against it.

### Local GPU support (unverified)

The GPU branch is the `--gpus all` flag, one scoring worker, the `nvidia-smi` name query in
`prepare`, and the GPU hardware class. All four have unit tests. The live test has not been run
on a GPU host as of 2026-09-23. To verify it, on a machine with the NVIDIA container toolkit:

```bash
TALOS_LIVE_BACKEND=local TALOS_LIVE_CHALLENGE=vector_search .venv/bin/pytest -m live tests/test_live.py -k local -s
```

Record the printed `gpu`, `prepare_s` and `job_s` here when it passes.
