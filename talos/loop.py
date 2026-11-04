"""The research loop. One Loop per job; all state lives in JobState and is saved after every
step so the run can be resumed. No network code here beyond calling the provider and bench."""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from talos import calibration
from talos.base_algorithm import read_base_root
from talos.baseline import resolve_baseline
from talos.bench import BenchCancelled, BenchUnavailable, EvalRequest, EvalResult
from talos.budget import BudgetExhausted, effective_budget, exhausted
from talos.challenges import CHALLENGES
from talos.edits import EditError, EditOutcome, apply_edit_response, apply_whole_file_response
from talos.diagnostics import dead_new_functions, defined_functions, first_error
from talos.inside import NONCE_TIMEOUT_S
from talos.lineage import (ROOT, AnnealSettings, Chain, Node, chain_for, decide, draw,
                           spent_fraction, temperature)
from talos.prompts import (PromptContext, STRATEGY_TAGS, compile_fix_prompts,
                           dead_code_fix_prompts, distill_prompts, edit_prompts,
                           edit_repair_prompts, hypothesis_prompts,
                           parse_distillation, parse_hypothesis)
from talos.providers import (ProviderAuthError, ProviderError, ProviderRateLimited,
                             ProviderTimeout)
from talos.scoring import (Bar, ScoringError, beats, beats_focused, bundle_delta, clears,
                           focus_sets, fuel_proxy_misses, improves, noise, quality_mismatches,
                           runtime_ratio, select, validation_failure, worst_track_shortfall)
from talos.search_replace import format_misses
from talos.state import Candidate, JobSpec, JobState, JobStore, TERMINAL, user_tacit
from talos.template_start import template_root
from talos.types import Completion, NonceResult


@dataclass
class Thresholds:
    recall: int = 2
    distill: int = 3
    reset: int = 5
    compile_fix_rounds: int = 3
    edit_repair_rounds: int = 1
    rate_limit_wait_s: int = 30
    rate_limit_max_waits: int = 20
    # A candidate nonce may run this many times the baseline's slowest nonce on its track,
    # but never less than runtime_floor_s and never more than the flat NONCE_TIMEOUT_S the
    # baseline ran under. 0 disables the guard. TIG itself caps fuel, not seconds; this is a
    # research-economy cap: one 14x-slower candidate ate a third of a 4 h budget.
    runtime_ceiling: float = 3.0
    runtime_floor_s: int = 60


class Interrupted(Exception):
    pass


class BaseAlgorithmError(RuntimeError):
    pass


class _BudgetedBench:
    """The bench, with the loop's budget check and cost accounting around every call.

    resolve_baseline compiles and scores both nonce sets in one evaluate of its own, and on a
    cold cache that is the most expensive compute call of the whole job. Handing it the raw bench
    let a baseline run to completion with --budget-compute-usd 0, and charged its cost only after
    the call had already happened."""

    def __init__(self, loop: "Loop"):
        self._loop = loop
        self._bench = loop.bench

    def _metered(self, call):
        loop = self._loop
        loop._check_budget()
        mark = self._bench.cost_mark()
        try:
            return call()
        finally:
            loop.state.spend.compute_usd += self._bench.cost_usd_since(mark)
            loop._save()

    def evaluate(self, request):
        return self._metered(lambda: self._bench.evaluate(request))

    def request_stop(self) -> None:
        self._bench.request_stop()

    def cost_mark(self) -> float:
        return self._bench.cost_mark()

    def cost_usd_since(self, mark: float) -> float:
        return self._bench.cost_usd_since(mark)


class Loop:
    def __init__(self, spec: JobSpec, state: JobState, store: JobStore, provider, bench,
                 template_rs: str, clock=time.time, sleep=time.sleep,
                 on_event: Callable[[str, dict], None] | None = None,
                 thresholds: Thresholds | None = None,
                 challenge_docs: dict[str, str] | None = None,
                 template_cu: str | None = None):
        self.spec, self.state, self.store = spec, state, store
        self.provider, self.bench = provider, bench
        self.template_rs = template_rs
        # A GPU template start's kernel template (runs/<job>/challenge/template.cu); None else.
        self.template_cu = template_cu
        # The challenge definition files shown to a template start (prompts.challenge_block).
        self.challenge_docs = dict(challenge_docs or {})
        self.clock, self.sleep = clock, sleep
        self.on_event = on_event or (lambda kind, data: None)
        self.t = thresholds or Thresholds()
        self.rule = CHALLENGES[spec.challenge].beat
        self._stop = False
        # The iteration an event belongs to. state.iteration only catches up when the iteration
        # finishes, so events raised mid-iteration would otherwise carry the previous number.
        self._n = state.iteration
        # The (cache dir, key) of the calibration record in use, set by calibrate(); demotions
        # are counted in it.
        self._calibration: tuple[Path, str] | None = None
        self.propose_and_edit = self.single_shot_propose_and_edit

    # ── plumbing ──────────────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

    def _event(self, kind: str, **data) -> None:
        self.store.event(kind, iteration=self._n, **data)
        self.on_event(kind, data)

    def _save(self) -> None:
        self.store.save(self.state)

    def _check_budget(self) -> None:
        dim = exhausted(self._budget(), self.state.spend, self.clock())
        if dim:
            raise BudgetExhausted(dim)
        if self._stop:
            raise Interrupted()

    def _llm(self, system: str, user: str) -> Completion:
        self._check_budget()
        waits = 0
        while True:
            try:
                c = self.provider.complete(system, user)
                break
            except ProviderRateLimited as e:
                waits += 1
                if waits > self.t.rate_limit_max_waits:
                    raise ProviderError(f"gave up after {waits} rate-limit waits: {e}")
                self._event("rate_limited", wait_s=self.t.rate_limit_wait_s)
                self.sleep(self.t.rate_limit_wait_s)
                self._check_budget()  # a stop or an hours cap must land during a rate-limit storm
            except ProviderError as e:
                # A failed CLI call can still have spent money; count it before the error goes
                # on to whichever caller fails the iteration or stops the run.
                self._charge_call(e.cost_usd, e.cost_unknown)
                raise
        self._charge_call(c.usage.cost_usd, c.cost_unknown)
        return c

    def _charge_call(self, cost_usd: float | None, cost_unknown: str | None) -> None:
        """Charge one single-shot call, or record that a call which may have cost something
        has no figure, so an uncounted call never reads as a free one."""
        if cost_unknown is not None:
            self._event("llm_cost_unknown", reason=cost_unknown)
        self._charge_llm(cost_usd)

    def _charge_llm(self, cost_usd: float | None) -> None:
        """Add one LLM call's cost to the job's spend and persist it. Single-shot calls come
        through _llm; an agentic session's cost comes from talos/agentic.py.
        `is not None`, not truthiness: a measured cost of exactly 0.0 is a measurement, and
        None means "this model has no price-table entry" — which the CLI reports as unpriced
        rather than as $0.00, because a dollar cap cannot be enforced against it."""
        if cost_usd is not None:
            self.state.spend.llm_usd += cost_usd
        self._save()

    def _focus(self) -> tuple[list, list]:
        return focus_sets(self.spec.track, self.spec.training, self.spec.holdout)

    def _request(self, files: dict[str, str], baseline_training, scoring: str = "metered",
                 validation: bool = False) -> EvalRequest:
        training, holdout = self._focus()
        base = select(baseline_training, training) if baseline_training is not None else None
        prior = ({name: defined_functions(text) for name, text in self._current_files().items()}
                 if not validation else None)  # the native build already passed the check
        req = EvalRequest(challenge=self.spec.challenge, files=files, training=training,
                          holdout=holdout, fuel=self.spec.fuel, baseline_training=base,
                          rule=self.rule, prior_functions=prior,
                          timeouts=self._timeouts(baseline_training),
                          hyperparameters=self._candidate_hyperparameters(),
                          # a validation's rows decide a demotion reason nonce by nonce
                          stop_at_error_ceiling=not validation)
        if scoring == "native":
            # Held-out is decided on the metered validation run, never on native results.
            req.holdout, req.baseline_training = [], None
            req.mode, req.fuel_budgets_us = "native", self.state.fuel_budgets_us
        return req

    def _candidate_hyperparameters(self) -> dict | None:
        """What a candidate runs with. A template or base start is not the mainnet algorithm, so
        that algorithm's hyperparameters mean nothing to it; the baseline keeps them."""
        return self.spec.hyperparameters if self.spec.start == "mainnet" else None

    def _timeouts(self, baseline_training) -> dict[str, int] | None:
        if self.t.runtime_ceiling <= 0 or not baseline_training:
            return None
        slowest: dict[str, int] = {}
        for r in baseline_training:
            slowest[r.track] = max(slowest.get(r.track, 0), r.runtime_ms)
        return {track: min(NONCE_TIMEOUT_S,
                           max(self.t.runtime_floor_s,
                               math.ceil(self.t.runtime_ceiling * ms / 1000)))
                for track, ms in slowest.items()}

    def _scoring(self) -> str:
        return "native" if self.state.scoring == "native" else "metered"

    def _bench_evaluate(self, files: dict[str, str], scoring: str = "metered",
                        validation: bool = False) -> EvalResult:
        """pending_job carries the files BEFORE the call: a C3 job outlives this process, and a
        resume must be able to rebuild the exact request and reattach to it."""
        self._check_budget()
        self.state.pending_job = {**(self.state.pending_job or {}), "files": files}
        self._save()
        mark = self.bench.cost_mark()
        try:
            return self.bench.evaluate(self._request(files, self.state.baseline.training,
                                                     scoring, validation))
        finally:
            self.state.spend.compute_usd += self.bench.cost_usd_since(mark)
            self._save()

    # ── baseline ──────────────────────────────────────────────────────

    def measure_baseline(self, cache_dir, hardware_class: str, mainnet=None) -> None:
        self._check_budget()  # a zero (or already spent) compute cap must bite before the call
        self.state.status = "measuring_baseline"
        self._save()
        pend = self.state.pending_job
        if not (pend and pend.get("purpose") == "baseline"):
            pend = {"purpose": "baseline"}
        self.state.pending_job = pend  # keep a stored job_id: a resumed baseline reattaches
        self._save()
        kw = {"mainnet": mainnet} if mainnet else {}
        rec, template = resolve_baseline(self.spec.challenge, self.spec.training,
                                         self.spec.holdout, self.spec.fuel, _BudgetedBench(self),
                                         cache_dir, hardware_class, rule=self.rule,
                                         log=lambda m: self._event("baseline", message=m),
                                         algorithm=self.spec.baseline_algorithm,
                                         hyperparameters=self.spec.hyperparameters, **kw)
        self.state.pending_job = None
        self.state.baseline = rec
        self.template_rs = template
        self.state.status = "researching"
        self._save()
        self._write_files(self.store.run_dir / "baseline", rec.files)
        (self.store.run_dir / "baseline" / "results.json").write_text(json.dumps(
            {"training": [r.to_dict() for r in rec.training],
             "holdout": [r.to_dict() for r in rec.holdout]}, indent=1),
            encoding="utf-8", newline="\n")
        self._event("baseline_ready", name=rec.name, adoption=rec.adoption)

    def _measure_base(self) -> None:
        """A base start's root, scored once before any LLM call and under the candidates'
        conditions. Metered even in a native job, like every score a best is held to
        (_validate_and_finish). A base that cannot run ends the job here, not after a budget of
        edits to code that was never going to score. It stays the root whatever it scores; the
        rule a candidate replaces the best under (scoring.clears) says whether it or the baseline
        is the score to beat, so a tie leaves the baseline."""
        if self.spec.start != "base" or self.state.base_delta is not None:
            return
        name = self.spec.base_algorithm["name"]
        pend = self.state.pending_job
        # keep a stored job_id: a resumed measurement reattaches
        self.state.pending_job = pend if pend and pend.get("purpose") == "base" else {
            "purpose": "base"}
        self._event("base_measuring", name=name)
        res = self._bench_evaluate(self._root_files(), "metered", validation=True)
        self.state.pending_job = None
        self._save()
        if not res.compile.ok:
            raise BaseAlgorithmError(f"base algorithm {name} failed to compile at "
                                     f"{self.spec.monorepo_ref}.\n{res.compile.output[-4000:]}")
        try:
            delta = bundle_delta(select(self.state.baseline.training, self._focus()[0]),
                                 res.training)
        except ScoringError as e:
            raise BaseAlgorithmError(f"base algorithm {name} could not be scored: {e}") from None
        if delta.error_rate > self.rule.error_ceiling:
            raise BaseAlgorithmError(f"base algorithm {name} failed on {delta.error_rate:.0%} of "
                                     "its nonces, without hyperparameters, so it cannot be "
                                     "compared with the baseline")
        self.state.base_delta = delta.to_dict()
        self.state.base_leads = clears(delta.mean_rel_delta, delta.worst_rel_delta, Bar(0.0, 0.0),
                                       self.rule)
        self._save()
        self._event("base_measured", name=name, mean_rel_delta=delta.mean_rel_delta,
                    worst_rel_delta=delta.worst_rel_delta)

    # ── calibration (native scoring) ──────────────────────────────────

    def calibrate(self, cache_dir, hardware_class: str) -> None:
        """Native scoring only. Finds this job's per-track budgets: a stored record for its key,
        or one measured now from the baseline on the training nonces in both modes. A job the
        record cannot give every research track a budget runs metered for good, and says so.
        Called on every start and resume, after the baseline; a stored decision is kept."""
        if self.spec.scoring != "native" or self.state.scoring == "metered":
            return
        key = calibration.calibration_key(self.spec.challenge, hardware_class,
                                          self.state.baseline.files, self.spec.hyperparameters)
        if self.state.scoring == "native" and self.state.fuel_budgets_us is not None:
            # A resume keeps the job's own budgets. Re-deriving them from the record (which
            # another job may have tightened since) changes an in-flight native request, so the
            # C3 request_hash no longer matches and the still-billing job is orphaned; and a
            # re-measure would overwrite pending_job, losing that job's id altogether.
            self._calibration = (Path(cache_dir), key)
            return
        rec = calibration.load(cache_dir, self.spec.challenge, key)
        if rec is None:
            rec = self._measure_calibration()
            if rec is not None:
                # Another job may have stored one while this one measured; that one is kept.
                rec = calibration.create(cache_dir, self.spec.challenge, key, rec)
        needed = {s.track for s in self._focus()[0]}
        missing = sorted(needed - set((rec or {}).get("tracks", {})))
        if rec is None or missing:
            self.state.scoring, self.state.fuel_budgets_us = "metered", None
            self._save()
            self._event("calibration_fallback",
                        reason=f"no native budget for track(s) {missing or sorted(needed)}; "
                               f"scoring this job metered")
            return
        self._calibration = (Path(cache_dir), key)
        self.state.scoring = "native"
        self.state.fuel_budgets_us = calibration.budgets_us(rec, self.spec.fuel)
        self._save()
        self._event("calibrated", budgets_us=self.state.fuel_budgets_us)

    def _baseline_output(self, res: EvalResult) -> dict:
        """The tail of a baseline build's compiler output for an event. Empty for a template
        start: the output quotes the baseline's source, which must not reach the timeline."""
        return {} if self.spec.start == "template" else {"output": res.compile.output[-2000:]}

    def _measure_calibration(self) -> dict | None:
        """The baseline on the training nonces: metered only if its stored rows carry no fuel
        (every baseline cached before fuel was recorded), then native with no budget. Both go
        through the budgeted bench. The metered rows are kept in pending_job until the native
        call returns. None when either build fails or no track yields a ratio."""
        bench = _BudgetedBench(self)
        pend = self.state.pending_job
        self.state.pending_job = (pend if pend and pend.get("purpose") == "calibration"
                                  else {"purpose": "calibration"})
        self._save()

        def score(mode: str):
            return bench.evaluate(EvalRequest(
                challenge=self.spec.challenge, files=self.state.baseline.files,
                training=self.spec.training, holdout=[], fuel=self.spec.fuel,
                baseline_training=None, rule=self.rule,
                hyperparameters=self.spec.hyperparameters, mode=mode))

        metered = self.state.baseline.training
        stored = self.state.pending_job.get("calibration_metered")
        if stored is not None:
            # A resume after a kill during the native call. Re-running the metered call would
            # not match the native job's request_hash, so C3 would submit a new job and orphan
            # the native one, still billing.
            metered = [NonceResult.from_dict(r) for r in stored]
        elif all(r.fuel_consumed is None for r in metered):
            res = score("metered")
            if not res.compile.ok:
                self.state.pending_job = None
                self._save()
                self._event("calibration_fallback",
                            reason="the baseline failed the metered build",
                            **self._baseline_output(res))
                return None
            metered = res.training
            # Saved before the native call, so a kill during it resumes at the native call.
            # The metered job is collected, so its reattach keys are dropped.
            self.state.pending_job = {"purpose": "calibration",
                                      "calibration_metered": [r.to_dict() for r in metered]}
            self._save()
        res = score("native")
        self.state.pending_job = None
        self._save()
        if not res.compile.ok:
            self._event("calibration_fallback", reason="the baseline failed the native build",
                        **self._baseline_output(res))
            return None
        ratios = calibration.track_ratios(metered, res.training)
        return calibration.new_record(ratios) if ratios else None

    @staticmethod
    def _write_files(directory, files: dict[str, str]) -> None:
        """spec §5.4: `baseline/` and `best/` in the run directory hold the current sources as
        plain files, replaced whole so they never mix two candidates' files."""
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for name, text in files.items():
            p = directory / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")

    # ── context ───────────────────────────────────────────────────────

    def _current_files(self) -> dict[str, str]:
        """The files of the parent that iteration self._n edits."""
        return self._node_files(self._parent_id(self._chain(self._n)))

    def _current_training(self):
        return self.state.best.training if self.state.best else self.state.baseline.training

    def _best_delta(self) -> float:
        """The best's mean delta: the mean a candidate must exceed to become the best."""
        return self._best_bar().mean

    def _best_bar(self) -> Bar:
        """What a candidate must clear to become the best (talos/scoring.py::improves). With no
        best yet that is the baseline against itself on a mainnet start, the better of the
        baseline and the base algorithm on a base start, and minus infinity on a template start,
        so the first scored candidate becomes the best however far below mainnet it is."""
        if self.state.best:
            return Bar.of(self.state.best.delta, self._strict())
        if self.spec.start == "base" and not self.state.base_leads:
            return Bar(0.0, 0.0, strict=True)
        return self._root_bar()

    def _root_bar(self) -> Bar:
        if self.spec.start == "template":
            return Bar(-math.inf, -math.inf)
        if self.spec.start == "base":
            return Bar.of(self.state.base_delta, strict=True)
        return Bar(0.0, 0.0)

    def _strict(self) -> bool:
        """A base job's best is the best of the baseline, the base algorithm and its candidates,
        so beating the baseline does not excuse falling short of the best."""
        return self.spec.start == "base"

    def _chain_bar(self, chain: Chain) -> Bar:
        """What a candidate must clear to become its chain's best: that node's score and worst
        track, as it was produced (native in a native job)."""
        if chain.best == ROOT:
            return self._root_bar()
        node = self._node(chain.best)
        if node is None or node.score is None or not node.tracks:
            return self._best_bar()  # a best kept from before the lineage tree has no Node
        return Bar(node.score, min(node.tracks.values()), self._strict())

    # ── lineage tree ──────────────────────────────────────────────────

    def _chain(self, n: int) -> Chain:
        """The chain iteration n belongs to. The chains are created on first use."""
        if not self.state.chains:
            self.state.chains = [Chain(index=i, best_score=self._root_score())
                                 for i in range(self.spec.chains)]
            if self.state.best is not None:
                # A job saved before the lineage tree has a best but no chains. Chain 0 continues
                # from that best; starting it at the root would edit the baseline again.
                first = self.state.chains[0]
                first.current = first.best = self.state.best.iteration
                first.best_score = self._best_delta()
        return self.state.chains[chain_for(n, self.spec.chains)]

    def _parent_id(self, chain: Chain) -> int:
        return chain.current if self.spec.search == "anneal" else chain.best

    def _root_files(self) -> dict[str, str]:
        """A template start's root is TIG's empty template as mod.rs (plus template.cu as
        algorithm.cu on a GPU challenge). The baseline's code is never read here for it, so it
        reaches no prompt, timeline entry or iteration directory."""
        if self.spec.start == "template":
            return template_root(self.spec.challenge, self.template_rs, self.template_cu)
        if self.spec.start == "base":
            root = read_base_root(self.store.run_dir)
            if root is None:
                raise BaseAlgorithmError(f"the base algorithm's files are missing from "
                                         f"{self.store.run_dir}")
            return root
        return dict(self.state.baseline.files)

    def _root_score(self) -> float | None:
        """The mainnet baseline scores 0 against itself, a base algorithm its measured delta; the
        empty template is not scored."""
        if self.spec.start == "base":
            return self.state.base_delta["mean_rel_delta"]
        return None if self.spec.start == "template" else 0.0

    def _node(self, node_id: int) -> Node | None:
        return next((node for node in self.state.nodes if node.id == node_id), None)

    def _node_files(self, node_id: int) -> dict[str, str]:
        """A node's sources: the root's files, the job best's, or the files saved in the
        node's iteration directory. A best kept from before the lineage tree has no directory
        of its own guaranteed, so state.best is checked before the disk."""
        root = self._root_files()
        if node_id == ROOT:
            return root
        if self.state.best is not None and self.state.best.iteration == node_id:
            return dict(self.state.best.files)
        # Not store.iteration_dir: it creates the directory.
        it_dir = self.store.run_dir / "iterations" / f"{node_id:04d}"
        return {name: (it_dir / name).read_bytes().decode("utf-8") for name in root}

    @staticmethod
    def _candidate_gap(cand: Candidate) -> dict:
        return {"node": cand.iteration, "mean": cand.delta["mean_rel_delta"],
                "tracks": {t["track"]: t["rel_delta"] for t in cand.delta["tracks"]}}

    def _parent_gap(self, parent: int) -> dict | None:
        if parent == ROOT:
            if self.spec.start != "base":
                return None
            return {"node": ROOT, "mean": self.state.base_delta["mean_rel_delta"],
                    "tracks": {t["track"]: t["rel_delta"]
                               for t in self.state.base_delta["tracks"]}}
        node = self._node(parent)
        if node is not None:
            return {"node": parent, "mean": node.rank_score(), "tracks": node.tracks or {}}
        best = self.state.best  # a best kept from before the lineage tree has no Node
        return self._candidate_gap(best) if best and best.iteration == parent else None

    def _best_gap(self) -> dict | None:
        return self._candidate_gap(self.state.best) if self.state.best else None

    def _node_score(self, node_id: int) -> float | None:
        """The score a node was produced with (native in a native job), which the accept rule
        compares against. A best kept from before the lineage tree has no Node."""
        if node_id == ROOT:
            return self._root_score()
        node = self._node(node_id)
        if node is not None:
            return node.score
        best = self.state.best
        return self._best_delta() if best is not None and best.iteration == node_id else None

    def _anneal_settings(self) -> AnnealSettings:
        return (AnnealSettings.from_dict(self.spec.anneal) if self.spec.anneal
                else AnnealSettings())

    def _budget(self):
        """The job's budget with every `--add-*` extension added (talos/extend.py)."""
        return effective_budget(self.spec.budget, self.state.budget_extensions)

    def _spent(self) -> float:
        return spent_fraction(self._budget(), self.state.spend, self.clock())

    def _anneal_step(self, n: int, record: dict, score: float) -> None:
        """Accepts or rejects a scored candidate that is not a job-wide improvement: always when
        it is no worse than its parent, otherwise with probability exp(-drop / T). The draw is
        keyed on the job, the chain and the iteration, so a resumed job decides the same way."""
        chain = self._chain(n)
        t = temperature(self._anneal_settings(), self._spent(), chain.reheated_at)
        u = draw(self.spec.job_id, chain.index, n)
        accepted, p = decide(self._node_score(record["against"]), score, t, u)
        record.update(outcome="accepted" if accepted else "failed:score", accepted=accepted,
                      probability=p, draw=u, temperature=t)

    def _chain_scores_apart(self) -> bool:
        """Whether a chain's best can differ from the job's. With one chain on a mainnet start
        the chain's best moves only with state.best, as before chains existed. A base start's
        chain climbs from a root that can be under the baseline, before anything is the best."""
        return self.spec.chains > 1 or self.spec.start in ("template", "base")

    def _lineage_fields(self, chain: Chain) -> dict:
        """The lineage part of a new iteration record."""
        return {"chain": chain.index, "against": self._parent_id(chain), "design_family": "",
                "accepted": False, "probability": None, "draw": None, "temperature": None}

    def _noise_fields(self, d) -> dict:
        """Display-only noise figures for a delta; the rules live in scoring.noise."""
        ratio, band, state = noise(d, self.rule)
        return {"mean_rel_se": d.mean_rel_se, "noise_ratio": ratio, "noise_band": band,
                "noise_state": state}

    def _add_node(self, n: int, record: dict, improved: bool, scored: dict | None,
                  metered: dict | None) -> None:
        """Appends iteration n's Node and moves its chain. A job-wide improvement always becomes
        the chain's best, so with one chain the chain's best stays the job's best. A chain's
        best_score is the score its best was produced with (native in a native job), never the
        metered score of a validation: spec §6 compares native scores only."""
        chain = self._chain(n)
        node = Node(id=n, chain=chain.index, parent=record["against"],
                    family=record.get("design_family", ""), title=record.get("title", ""),
                    outcome=record["outcome"],
                    score=scored["mean_rel_delta"] if scored else None,
                    tracks=({t["track"]: t["rel_delta"] for t in scored["tracks"]}
                            if scored else None),
                    metered_score=metered["mean_rel_delta"] if metered else None,
                    accepted=record["accepted"], probability=record["probability"],
                    draw=record["draw"], temperature=record["temperature"],
                    best=self.state.best is not None and self.state.best.iteration == n)
        self.state.nodes.append(node)
        if node.accepted:
            chain.current = n
            score = node.score
            if improved or (self._chain_scores_apart() and score is not None and node.tracks
                            and clears(score, min(node.tracks.values()),
                                       self._chain_bar(chain), self.rule)):
                chain.best, chain.best_score, chain.stall = n, score, 0
                return
        chain.stall += 1

    def _failed(self) -> list[dict]:
        """Every failure of the job, whichever best it was tried against (issue #33)."""
        return [h for h in self.state.hypotheses if h.get("outcome", "").startswith("failed")]

    def _forced_tag(self) -> str | None:
        """Annealing never forces a strategy tag: reheating takes the place of the rotation."""
        if self.spec.search == "anneal" or self.state.runs_since_improvement < self.t.reset:
            return None
        counts = self.state.strategy_counts
        return min(STRATEGY_TAGS, key=lambda tag: counts.get(tag, 0))

    def _context(self) -> PromptContext:
        chain = self._chain(self._n)
        parent = self._parent_id(chain)
        recall = self.state.runs_since_improvement >= self.t.recall
        # Each chain recalls only its own failures; a record from before chains is chain 0's.
        failed = ([h for h in self._failed() if h.get("chain", 0) == chain.index]
                  if recall else [])
        return PromptContext(challenge=self.spec.challenge, template_rs=self.template_rs,
                             direction=self.spec.direction, tacit=self.state.tacit,
                             files=self._node_files(parent),
                             baseline_name=self.state.baseline.name,
                             best_delta=self._best_delta(), failed_hypotheses=failed,
                             anchor=parent, parent_gap=self._parent_gap(parent),
                             best_gap=self._best_gap(),
                             forced_tag=self._forced_tag(),
                             is_gpu=CHALLENGES[self.spec.challenge].is_gpu,
                             track=self.spec.track,
                             guard_tracks=([t for t in self.spec.tracks if t != self.spec.track]
                                           if self.spec.track else []),
                             hyperparameters=self._candidate_hyperparameters(),
                             scoring=self._scoring(), start=self.spec.start,
                             challenge_docs=self.challenge_docs, search=self.spec.search,
                             chains=self.spec.chains, template_cu=self.template_cu)

    # ── single-shot propose + edit ────────────────────────────────────

    def single_shot_propose_and_edit(self, ctx: PromptContext) -> tuple[dict, dict[str, str]]:
        system, user = hypothesis_prompts(ctx)
        hypothesis = parse_hypothesis(self._llm(system, user).text)
        self._event("hypothesis", **hypothesis)
        whole_file = self._whole_file(ctx, hypothesis)
        # A copy: _score_candidate reuses ctx for the compile-fix prompts, which stay diffs.
        system, user = edit_prompts(replace(ctx, whole_file=whole_file), hypothesis)
        text = self._llm(system, user).text
        if whole_file:
            outcome = apply_whole_file_response(ctx.files, text)
        else:
            outcome = apply_edit_response(ctx.files, text)
        for _ in range(self.t.edit_repair_rounds):
            if not outcome.misses:
                break
            system, user = edit_repair_prompts(ctx, outcome.files, format_misses(outcome.misses))
            try:
                repaired = apply_edit_response(outcome.files, self._llm(system, user).text)
            except (EditError, ProviderTimeout):
                break  # no usable repair reply: keep what was applied, skip the misses
            # Accumulate, never replace: the repair round re-emits only the failed blocks, so its
            # own `applied` says nothing about the first response's blocks, and its `rejected`
            # would forget an out-of-scope block that the first response carried.
            outcome = EditOutcome(files=repaired.files, applied=outcome.applied + repaired.applied,
                                  misses=repaired.misses,
                                  rejected=outcome.rejected + repaired.rejected)
        if outcome.rejected:
            # spec §9: an edit outside the algorithm files fails the iteration in BOTH modes.
            # Applying the in-scope blocks of the same response and carrying on would let the
            # out-of-scope block go unpunished and reward a response that tried it.
            self._event("edits_rejected", paths=outcome.rejected)
            raise EditError("edit outside the algorithm files "
                            f"rejected: {sorted(outcome.rejected)}")
        if outcome.applied == 0:
            raise EditError("no edit block applied")
        return hypothesis, outcome.files

    def _whole_file(self, ctx: PromptContext, hypothesis: dict) -> bool:
        """Template starts only: the edit is a whole-file write from the root, or when the
        hypothesis names a design family no node on this chain has used. A mainnet start always
        edits with search/replace."""
        if self.spec.start != "template":
            return False
        chain = self._chain(self._n)
        families = {node.family for node in self.state.nodes if node.chain == chain.index}
        return ctx.anchor == ROOT or hypothesis["design_family"] not in families

    # ── one iteration ─────────────────────────────────────────────────

    def iterate(self) -> None:
        n = self.state.iteration + 1
        self._n = n
        it_dir = self.store.iteration_dir(n)
        ctx = self._context()
        record = {"iteration": n, **self._lineage_fields(self._chain(n)), "title": "",
                  "description": "", "strategy_tag": "", "outcome": "started"}
        try:
            hypothesis, files = self.propose_and_edit(ctx)
        # A CLI timeout fails this iteration only, as the agentic path's AgenticError does; every
        # other ProviderError still stops the run in run().
        except (EditError, ValueError, ProviderTimeout) as e:
            record.update(outcome="failed:edit", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        record.update(hypothesis)
        (it_dir / "hypothesis.json").write_text(json.dumps(hypothesis),
                                                encoding="utf-8", newline="\n")
        self.state.pending_job = {"purpose": n, "hypothesis": hypothesis, "files": files}
        self._save()
        self._score_candidate(n, ctx, record, hypothesis, files)

    def _score_candidate(self, n: int, ctx: PromptContext, record: dict, hypothesis: dict,
                         files: dict[str, str]) -> None:
        """Everything after the LLM has produced files: compile-fix rounds, scoring, confirmation.
        Entered from iterate() and from a resume with a pending job."""
        it_dir = self.store.iteration_dir(n)
        res = self._bench_evaluate(files, self._scoring())
        for _ in range(self.t.compile_fix_rounds):
            dead = self._dead_code(res)
            if res.compile.ok and not dead:
                break
            if dead:
                self._event("dead_code", names=dead)
                system, user = dead_code_fix_prompts(ctx, files, dead)
            else:
                # `error` is read from the whole output: the tail kept here is often all warnings
                self._event("compile_failed", error=first_error(res.compile.output),
                            output=res.compile.output[-2000:])
                system, user = compile_fix_prompts(ctx, files, res.compile.output)
            try:
                fixed = apply_edit_response(files, self._llm(system, user).text)
            except (EditError, ProviderTimeout):
                break
            if fixed.rejected:
                # spec §9 holds for the fix response too: never apply its in-scope blocks either.
                self._event("edits_rejected", paths=fixed.rejected)
                record.update(outcome="failed:edit", error="edit outside the algorithm files "
                              f"rejected: {sorted(fixed.rejected)}")
                self._finish_iteration(n, record, improved=False)
                return
            if fixed.applied == 0:
                break  # byte-identical files; re-evaluating them would repeat the same error
            files = fixed.files
            res = self._bench_evaluate(files, self._scoring())
        if not res.compile.ok:
            record.update(outcome="failed:compile")
            self._finish_iteration(n, record, improved=False)
            return
        dead = self._dead_code(res)
        if dead:
            record.update(outcome="failed:dead_code", error=f"never called: {dead}")
            self._finish_iteration(n, record, improved=False)
            return
        for name, text in files.items():
            (it_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (it_dir / name).write_text(text, encoding="utf-8", newline="\n")
        results = res.training
        base_tr = select(self.state.baseline.training, self._focus()[0])
        try:
            delta = bundle_delta(base_tr, results)
        except ScoringError as e:
            record.update(outcome="failed:score", error=str(e))
            self._finish_iteration(n, record, improved=False)
            return
        cand = Candidate(iteration=n, files=files, artifact_id=res.compile.artifact_id,
                         training=results, delta=delta.to_dict(), hypothesis=hypothesis)
        worst = min(delta.tracks, key=lambda t: t.rel_delta)
        # What the next prompt's recall block says about this attempt, beyond its outcome.
        record.update(mean_rel_delta=delta.mean_rel_delta, worst_track=worst.track,
                      worst_rel_delta=delta.worst_rel_delta)
        # A job stopped at the error ceiling (issue #84) never started these. They count as
        # errors at the worst quality in the delta, but have no runtime to compare.
        ran = [r for r in results if r.error != "not_run"]
        not_run = len(results) - len(ran)
        if self._scoring() == "metered":
            # Native runtimes are not comparable with the baseline's metered ones; the
            # validated record gets the metered ratio instead (_validate_and_finish).
            keys = {(r.track, r.nonce) for r in ran}
            record["runtime_ratio"] = runtime_ratio(
                [r for r in base_tr if (r.track, r.nonce) in keys], ran)
        fields = self._noise_fields(delta)
        record.update(**fields)
        self._event("scored", mean_rel_delta=delta.mean_rel_delta,
                    worst_rel_delta=delta.worst_rel_delta, error_rate=delta.error_rate,
                    runtime_ratio=record.get("runtime_ratio"), **fields,
                    track_rel_se={t.track: t.rel_se for t in delta.tracks},
                    **({"not_run": not_run} if not_run else {}))
        if delta.error_rate > self.rule.error_ceiling:  # spec §9: over the ceiling is a failure
            record.update(outcome="failed:runtime", error_rate=delta.error_rate)
            if not_run:
                failed = sum(1 for r in ran if not r.ok)
                record["error"] = (f"stopped early at the error ceiling: {failed} of the "
                                   f"{len(ran)} nonces run failed; {not_run} of "
                                   f"{len(results)} not run")
            self._finish_iteration(n, record, improved=False, scored=cand.delta)
            return
        # spec §7.7: a candidate that beats the baseline on training becomes the best, whether or
        # not the held-out set goes on to confirm it. Confirming a candidate that is not best would
        # report "won" with different code in state.best, so a base job's candidate that beats the
        # baseline but not the best does not win.
        improved = improves(delta, self._best_bar(), self.rule)
        wins = improved and beats(base_tr, results, self.rule)
        if improved and self._scoring() == "native":
            self._validate_and_finish(n, record, cand, base_tr)
            return
        chain = self._chain(n)
        if improved:
            self.state.best = cand
            self._write_files(self.store.run_dir / "best", cand.files)
            record.update(outcome="improved", accepted=True)
        elif self.spec.search == "anneal":
            self._anneal_step(n, record, delta.mean_rel_delta)
        elif self._chain_scores_apart() and improves(delta, self._chain_bar(chain), self.rule):
            # Greedy, per chain: not a job-wide improvement, but better than this chain's best.
            record.update(outcome="accepted", accepted=True)
        else:
            record["outcome"] = "failed:score"
            bar = self._chain_bar(chain) if self._chain_scores_apart() else self._best_bar()
            shortfall = worst_track_shortfall(delta, bar, self.rule)
            if shortfall:
                record["error"] = shortfall
        # The record and the iteration bump are persisted BEFORE the confirmation is read out of
        # the result, so a kill between the two never loses the iteration.
        self._finish_iteration(n, record, improved=improved, scored=cand.delta)
        if wins:
            self._confirm(cand, res.holdout, res.holdout_reason)

    def _dead_code(self, res: EvalResult) -> list[str]:
        """The bench stopped after the build because the edit's new functions are never called.
        The names come back out of the compile output, which every backend keeps."""
        if not res.compile.ok or res.holdout_reason != "dead_code":
            return []
        prior = {name: defined_functions(text) for name, text in self._current_files().items()}
        return dead_new_functions(res.compile.output, prior) or ["<unnamed>"]

    def _validate_and_finish(self, n: int, record: dict, cand: Candidate, base_tr) -> None:
        """A native candidate that would become the best is scored again on TIG's metered
        runtime, on the job's own training and held-out nonces, before it counts. It becomes
        best only on a pass; held-out confirmation then runs on the metered results. On a fail
        research stays on the previous best. The native rows go into pending_job first, so a
        resume re-enters here rather than scoring natively again."""
        self.state.pending_job = {**(self.state.pending_job or {}), "files": cand.files,
                                  "native_training": [r.to_dict() for r in cand.training]}
        self._save()
        self._event("validating")
        res = self._bench_evaluate(cand.files, "metered", validation=True)
        reason, d = validation_failure(res.compile.ok, res.training, cand.training, base_tr,
                                       self._best_bar(), self.rule)
        entry = {"iteration": n, "outcome": "demoted" if reason else "validated",
                 "reason": reason}
        if reason == "nondeterministic":
            entry["mismatches"] = quality_mismatches(res.training, cand.training)
        self.state.validations.append(entry)
        if reason:
            self._note_demotion(n, reason, res.training, cand.training)
            record.update(outcome="failed:validation", error=reason)
            self._event("demoted", reason=reason)
            self._finish_iteration(n, record, improved=False, scored=cand.delta)
            return
        best = Candidate(iteration=n, files=cand.files, artifact_id=res.compile.artifact_id,
                         training=res.training, delta=d.to_dict(), hypothesis=cand.hypothesis)
        self.state.best = best
        self._write_files(self.store.run_dir / "best", best.files)
        record.update(outcome="improved", accepted=True, mean_rel_delta=d.mean_rel_delta,
                      worst_rel_delta=d.worst_rel_delta,
                      runtime_ratio=runtime_ratio(base_tr, res.training),
                      **self._noise_fields(d))
        self._event("validated", mean_rel_delta=d.mean_rel_delta)
        # The node keeps the native score it was produced with beside the metered one.
        self._finish_iteration(n, record, improved=True, scored=cand.delta, metered=best.delta)
        if beats(base_tr, res.training, self.rule):
            self._confirm(best, res.holdout, res.holdout_reason)

    def _note_demotion(self, n: int, reason: str, metered, native) -> None:
        """Counts the demotion in the calibration record; a fuel-proxy miss also counts against
        each track with a miss (talos/scoring.py::fuel_proxy_misses), and a lowered margin
        tightens this job's budgets. The count is locked against other jobs with the same key
        and made once per iteration, even when a resume validates it again
        (talos/calibration.py::count_validation)."""
        if self._calibration is None:
            return
        cache_dir, key = self._calibration
        missed = (sorted({r.track for r in fuel_proxy_misses(metered, native)})
                  if reason == "fuel_proxy_miss" else [])
        # created_at as well: job ids are unique only within one runs/ directory, and two
        # checkouts share the record. Never the rand hash, which must stay inside the job.
        token = f"{self.spec.job_id}@{self.spec.created_at!r}:{n}"
        counted = calibration.count_validation(cache_dir, self.spec.challenge, key, token,
                                               reason, missed)
        if counted is None:
            return
        rec, changed = counted
        if changed:
            # No save here: _finish_iteration saves next. A save now would store the validation
            # while pending_job still holds native_training, and a resume after a kill would
            # validate the iteration again and record it twice. That resume recomputes these
            # budgets from the repeat's result instead.
            self.state.fuel_budgets_us = calibration.budgets_us(rec, self.spec.fuel)

    def _confirm(self, cand: Candidate, ho, reason: str) -> None:
        """The held-out results come back from the same evaluate call that scored training: the
        bench (or the C3 job) applies the beat rule itself and scores held-out only on a win."""
        self.state.status = "confirming"
        self._save()
        self._event("confirming")
        cand.holdout = ho
        if ho is None:
            # The job said "won on training" but produced no held-out results (a timeout inside
            # the container, say). That is not a win, and it must not strand the job at
            # "confirming".
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._save()
            self._event("false_positive", error=f"held-out not scored ({reason})")
            return
        _, holdout_sets = self._focus()
        # The guard sets are TRAINING sets of the other tracks, so their baseline rows live in
        # baseline.training; select() picks the right rows from both lists by (track, nonce).
        base_ho = select(self.state.baseline.holdout + self.state.baseline.training, holdout_sets)
        try:
            if self.spec.track is None:
                won = beats(base_ho, ho, self.rule)
            else:
                won = beats_focused(base_ho, ho, self.rule, self.spec.track)
            holdout_delta = bundle_delta(base_ho, ho).to_dict()
        except ScoringError as e:
            # An unscoreable held-out run is not a win; it must not leave the job at "confirming".
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._save()
            self._event("false_positive", error=str(e))
            return
        if won:
            self.state.confirmed.append(cand.iteration)
            self.state.status = "won"
            self.state.stop_reason = "beat baseline on training and held-out nonces"
            self._mark_won(cand.iteration)
            self._event("won", holdout=holdout_delta)
        else:
            self.state.false_positives.append(cand.iteration)
            self.state.status = "researching"
            self._event("false_positive", holdout=holdout_delta)
        self._save()

    def _mark_won(self, iteration: int) -> None:
        """The record for this iteration was appended by _finish_iteration just before the
        held-out run, so it is the last one — but check rather than assume."""
        if self.state.hypotheses and self.state.hypotheses[-1].get("iteration") == iteration:
            self.state.hypotheses[-1]["outcome"] = "won"
        if self.state.nodes and self.state.nodes[-1].id == iteration:
            self.state.nodes[-1].outcome = "won"

    def _finish_iteration(self, n: int, record: dict, improved: bool,
                          scored: dict | None = None, metered: dict | None = None) -> None:
        """`scored` is the delta the candidate was scored with (native in a native job) and
        `metered` its validated metered delta; both None for an iteration with no score."""
        self.state.iteration = n
        self.state.spend.iterations += 1
        self.state.hypotheses.append(record)
        self._n = n
        # Before the save below, so an event handler's stop or an interrupted distill never
        # loses a chain's move.
        self._add_node(n, record, improved, scored, metered)
        reheated = self._reheat(n)
        # A base job climbing towards the baseline is progress before anything is the best: it
        # must not read as a stall that forces a new strategy on a line of work that is improving.
        progressed = improved or (self._strict() and self._chain(n).best == n)
        if progressed:
            self.state.runs_since_improvement = 0
        else:
            self.state.runs_since_improvement += 1
        # Counted here, not where the hypothesis is parsed: a resumed pending iteration re-enters
        # at _score_candidate, and counting at propose time would count that iteration twice.
        if record.get("strategy_tag"):
            self.state.strategy_counts[record["strategy_tag"]] = (
                self.state.strategy_counts.get(record["strategy_tag"], 0) + 1)
        self.state.pending_job = None
        self._save()
        node = self.state.nodes[-1]
        # Spec §8: one line per iteration in a tree. Greedy shows it whenever chains score apart
        # (several chains or a template start); the default job's events stay as they were.
        if self.spec.search == "anneal":
            line = node.accepted or node.temperature is not None
        else:
            line = self._chain_scores_apart()
        if line:
            self._event("accept", chain=node.chain, node=node.id, parent=node.parent,
                        family=node.family, score=node.score, accepted=node.accepted,
                        probability=node.probability, temperature=node.temperature,
                        outcome=node.outcome)
        if reheated is not None:
            self._event("reheat", **reheated)
        self._event("iteration_done", outcome=record["outcome"],
                    runs_since_improvement=self.state.runs_since_improvement)
        if not progressed and self.state.runs_since_improvement == self.t.distill:
            self._distill()
        if (self.spec.search != "anneal" and not progressed
                and self.state.runs_since_improvement >= self.t.reset):
            self._event("reset", forced_tag=self._forced_tag())

    def _reheat(self, n: int) -> dict | None:
        """Anneal only. After `stall` iterations with no new chain best, the chain goes back to
        its best and cooling restarts over the budget left. Runs after the iteration is counted,
        so the spent fraction includes it. Returns the reheat event's fields, or None."""
        if self.spec.search != "anneal":
            return None
        settings = self._anneal_settings()
        chain = self._chain(n)
        if chain.stall < settings.stall:
            return None
        chain.current = chain.best
        chain.reheated_at = self._spent()
        chain.stall = 0
        return {"chain": chain.index, "node": chain.current,
                "temperature": temperature(settings, chain.reheated_at, chain.reheated_at)}

    def _distill(self) -> None:
        failed = self._failed()
        if not failed:
            return
        system, user = distill_prompts(self._context(), failed)
        try:
            text = self._llm(system, user).text
        except ProviderAuthError:
            raise  # spec §9: a dead key or a billing failure stops the run at once
        except ProviderError:
            return
        lesson = parse_distillation(text)
        if lesson:
            self.state.tacit = (self.state.tacit.rstrip() + f"\n- LLM: {lesson}\n").lstrip()
            (self.store.run_dir / "tacit.md").write_text(
                user_tacit(self.spec.direction) + self.state.tacit,
                encoding="utf-8", newline="\n")
            self._save()
            self._event("distilled", lesson=lesson)

    # ── driver ────────────────────────────────────────────────────────

    def _discard_incomplete_iteration(self) -> None:
        n = self.state.iteration + 1
        if (self.state.pending_job or {}).get("purpose") == n:
            return  # that iteration is not abandoned: its job is still in flight and resumable
        d = self.store.run_dir / "iterations" / f"{n:04d}"
        if d.exists():
            shutil.rmtree(d)
            self._event("discarded_incomplete", discarded=n)

    def _resume_pending(self, pending: dict) -> None:
        n = pending["purpose"]
        self._n = n
        self._event("resumed_pending", job_id=pending.get("job_id"))
        hypothesis = pending["hypothesis"]
        record = {"iteration": n, **self._lineage_fields(self._chain(n)), **hypothesis,
                  "outcome": "started"}
        if pending.get("native_training"):
            files = pending["files"]
            native = [NonceResult.from_dict(r) for r in pending["native_training"]]
            base_tr = select(self.state.baseline.training, self._focus()[0])
            cand = Candidate(iteration=n, files=files, artifact_id="", training=native,
                             delta=bundle_delta(base_tr, native).to_dict(),
                             hypothesis=hypothesis)
            self._validate_and_finish(n, record, cand, base_tr)
            return
        self._score_candidate(n, self._context(), record, hypothesis, pending["files"])

    def run(self) -> JobState:
        if self.state.status in TERMINAL:
            return self.state  # before the discard: a finished job keeps its winning directory
        self._discard_incomplete_iteration()
        try:
            self.state.status = "researching"
            self._save()
            pending = self.state.pending_job
            if pending is not None and pending.get("purpose") in ("baseline", "calibration"):
                self.state.pending_job = None  # the baseline is recorded; the record is stale
                self._save()
            elif pending is not None and isinstance(pending.get("purpose"), int):
                # An evaluate killed mid-flight (a stop, a bench outage, a hard kill): re-enter
                # that iteration with its own files, so the backend can reattach to the job
                # instead of abandoning one C3 is still billing for.
                self._resume_pending(pending)
            self._measure_base()
            while self.state.status not in TERMINAL:
                self._check_budget()
                self.iterate()
        except BudgetExhausted as e:
            self.state.status, self.state.stop_reason = "exhausted", e.dimension
        except Interrupted:
            self.state.status, self.state.stop_reason = "cancelled", "user requested stop"
        except BenchCancelled as e:
            self.state.status, self.state.stop_reason = ("cancelled",
                                                         f"stopped; bench job cancelled: {e}")
        except ProviderAuthError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider auth: {e}"
        except ProviderError as e:
            self.state.status, self.state.stop_reason = "failed", f"provider: {e}"
        except BenchUnavailable as e:
            self.state.status, self.state.stop_reason = "paused", f"bench: {e}"
        self._save()
        self._event("stopped", status=self.state.status, reason=self.state.stop_reason)
        return self.state
