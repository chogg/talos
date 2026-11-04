"""Everything a job persists. JSON on disk under runs/<job_id>/; writes are atomic."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from talos.budget import Budget, Spend
from talos.lineage import Chain, Node
from talos.types import NonceResult, NonceSet

STATUSES = ("queued", "measuring_baseline", "researching", "confirming", "paused",
            "won", "exhausted", "failed", "cancelled")
TERMINAL = {"won", "exhausted", "failed", "cancelled"}


@dataclass
class JobSpec:
    job_id: str
    challenge: str
    direction: str
    provider: str
    model: str
    mode: str
    budget: Budget
    rand_hash: str
    tracks: list[str]
    training: list[NonceSet]
    holdout: list[NonceSet]
    fuel: int
    created_at: float
    monorepo_ref: str
    challenge_id: str
    track: str | None = None  # one active track to optimise; None = all tracks
    # Pinned at job start by `talos run --hyperparameters mainnet` (the default). The map
    # belongs to this algorithm's code: resolve_baseline measures this algorithm rather than
    # whatever tops mainnet adoption by then. None on all three = no hyperparameters, as
    # before.
    baseline_algorithm: dict | None = None  # {"name", "id", "adoption"}
    hyperparameters: dict[str, dict | None] | None = None  # track -> map passed to tig-runtime
    hyperparameters_source: dict[str, dict] | None = None  # track -> benchmark it came from
    # None identifies jobs written before the image pin was recorded. Their saved scores
    # cannot be reused with a newer runtime, even when the monorepo pin is unchanged.
    dev_image_tag: str | None = None
    scoring: str = "metered"  # research scoring, `talos run --scoring`; frozen with the job
    start: str = "mainnet"  # "mainnet" baseline, "template" from scratch, or "base" (below)
    # `talos run --base-algorithm/--base-dir`: the code chains start from when start == "base".
    # Its files are in runs/<job>/base_algorithm/ (talos/base_algorithm.py).
    base_algorithm: dict | None = None  # {"source": "mainnet" or "local", "name", "id"}
    search: str = "greedy"  # `talos run --search`: "greedy" or "anneal"
    chains: int = 1  # `talos run --chains`: independent annealing chains
    anneal: dict | None = None  # AnnealSettings.to_dict(); None unless search == "anneal"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["budget"] = self.budget.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobSpec":
        d = dict(d)
        d["budget"] = Budget(**d["budget"])
        d["training"] = [NonceSet(**n) for n in d["training"]]
        d["holdout"] = [NonceSet(**n) for n in d["holdout"]]
        return cls(**d)

    def redacted(self) -> dict:
        d = self.to_dict()
        d.pop("rand_hash")
        for key in ("training", "holdout"):
            d[key] = [{"track": n["track"], "start": n["start"], "count": n["count"]}
                      for n in d[key]]
        return d


@dataclass
class BaselineRecord:
    name: str
    adoption: int
    artifact_id: str
    files: dict[str, str]
    training: list[NonceResult]
    holdout: list[NonceResult]

    def to_dict(self) -> dict:
        return {"name": self.name, "adoption": self.adoption, "artifact_id": self.artifact_id,
                "files": self.files, "training": [r.to_dict() for r in self.training],
                "holdout": [r.to_dict() for r in self.holdout]}

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineRecord":
        return cls(name=d["name"], adoption=d["adoption"], artifact_id=d["artifact_id"],
                   files=d["files"], training=[NonceResult.from_dict(r) for r in d["training"]],
                   holdout=[NonceResult.from_dict(r) for r in d["holdout"]])


@dataclass
class Candidate:
    iteration: int
    files: dict[str, str]
    artifact_id: str
    training: list[NonceResult]
    delta: dict
    hypothesis: dict
    holdout: list[NonceResult] | None = None

    def to_dict(self) -> dict:
        return {"iteration": self.iteration, "files": self.files, "artifact_id": self.artifact_id,
                "training": [r.to_dict() for r in self.training], "delta": self.delta,
                "hypothesis": self.hypothesis,
                "holdout": [r.to_dict() for r in self.holdout] if self.holdout else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(iteration=d["iteration"], files=d["files"], artifact_id=d["artifact_id"],
                   training=[NonceResult.from_dict(r) for r in d["training"]], delta=d["delta"],
                   hypothesis=d["hypothesis"],
                   holdout=([NonceResult.from_dict(r) for r in d["holdout"]]
                            if d.get("holdout") else None))


def user_tacit(direction: str) -> str:
    """The first line of tacit.md. It is not kept in JobState.tacit: the prompts already carry
    the direction, and repeating it in the tacit block would send it twice."""
    return f"- USER: {direction.strip()}\n"


@dataclass
class JobState:
    status: str
    iteration: int
    best: Candidate | None
    baseline: BaselineRecord | None
    runs_since_improvement: int
    hypotheses: list[dict]
    spend: Spend
    confirmed: list[int] = field(default_factory=list)
    false_positives: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    tacit: str = ""
    strategy_counts: dict[str, int] = field(default_factory=dict)
    # The compute job in flight, if any: written before the call and cleared after it, so a
    # resume can rebuild the request and reattach to a job that outlived this process.
    pending_job: dict | None = None
    # The hardware (a Modal GPU, or a C3 GPU class or CPU profile) this job was frozen to by
    # the first run's capacity probe; None until then, and always None where there is nothing
    # to choose (a CPU challenge on Modal, the local backend). A resume hands it back to the
    # bench instead of probing again, so the candidates score on the hardware the baseline was
    # measured on (AGENTS.md invariant 1).
    hardware: str | None = None
    # What research scoring actually runs: None until Loop.calibrate has run, "native" once a
    # calibration record gave every research track a budget, "metered" when the job asked for
    # native and calibration could not give one (the fallback is kept for the job's life).
    scoring: str | None = None
    fuel_budgets_us: dict[str, int] | None = None  # track -> native solve budget, microseconds
    # One entry per metered validation of a would-be best: iteration, outcome, reason.
    validations: list[dict] = field(default_factory=list)
    # The lineage tree and one cursor per chain; empty for a job that does not anneal.
    nodes: list[Node] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)
    # One entry per `talos run --resume --add-*`: when, at which iteration, the status it was
    # resumed from, and what it added. job.json keeps the budget the job started with;
    # talos/budget.py::effective_budget adds these to it.
    budget_extensions: list[dict] = field(default_factory=list)
    # A base start's root against the baseline, metered (Loop._measure_base); None until then.
    base_delta: dict | None = None
    # Whether that root beat the baseline, so it and not the baseline is the job's score to beat.
    base_leads: bool = False

    @classmethod
    def fresh(cls, spend: Spend) -> "JobState":
        return cls(status="queued", iteration=0, best=None, baseline=None,
                   runs_since_improvement=0, hypotheses=[], spend=spend)

    def to_dict(self) -> dict:
        return {"status": self.status, "iteration": self.iteration,
                "best": self.best.to_dict() if self.best else None,
                "baseline": self.baseline.to_dict() if self.baseline else None,
                "runs_since_improvement": self.runs_since_improvement,
                "hypotheses": self.hypotheses, "spend": self.spend.to_dict(),
                "confirmed": self.confirmed, "false_positives": self.false_positives,
                "stop_reason": self.stop_reason, "tacit": self.tacit,
                "strategy_counts": self.strategy_counts, "pending_job": self.pending_job,
                "hardware": self.hardware, "scoring": self.scoring,
                "fuel_budgets_us": self.fuel_budgets_us, "validations": self.validations,
                "nodes": [n.to_dict() for n in self.nodes],
                "chains": [c.to_dict() for c in self.chains],
                "budget_extensions": self.budget_extensions, "base_delta": self.base_delta,
                "base_leads": self.base_leads}

    @classmethod
    def from_dict(cls, d: dict) -> "JobState":
        return cls(status=d["status"], iteration=d["iteration"],
                   best=Candidate.from_dict(d["best"]) if d.get("best") else None,
                   baseline=BaselineRecord.from_dict(d["baseline"]) if d.get("baseline") else None,
                   runs_since_improvement=d["runs_since_improvement"],
                   hypotheses=d["hypotheses"], spend=Spend(**d["spend"]),
                   confirmed=d.get("confirmed", []), false_positives=d.get("false_positives", []),
                   stop_reason=d.get("stop_reason"), tacit=d.get("tacit", ""),
                   strategy_counts=d.get("strategy_counts", {}),
                   pending_job=d.get("pending_job"),
                   hardware=d.get("hardware", d.get("gpu")),
                   scoring=d.get("scoring"), fuel_budgets_us=d.get("fuel_budgets_us"),
                   validations=d.get("validations", []),
                   nodes=[Node.from_dict(n) for n in d.get("nodes", [])],
                   chains=[Chain.from_dict(c) for c in d.get("chains", [])],
                   budget_extensions=d.get("budget_extensions", []),
                   base_delta=d.get("base_delta"), base_leads=d.get("base_leads", False))


def _atomic_write(path: Path, text: str) -> None:
    """os.replace is atomic, but only against the file contents that actually reached the disk:
    without the fsync a crash can leave the renamed file empty or truncated, which for
    state.json is an unresumable job."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class JobStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_spec(self, spec: JobSpec) -> None:
        path = self.run_dir / "job.json"
        if path.exists():
            raise FileExistsError(f"{path} already exists; job.json is immutable")
        _atomic_write(path, json.dumps(spec.to_dict(), indent=1))

    def read_spec(self) -> JobSpec:
        text = (self.run_dir / "job.json").read_text(encoding="utf-8")
        return JobSpec.from_dict(json.loads(text))

    def save(self, state: JobState) -> None:
        _atomic_write(self.run_dir / "state.json", json.dumps(state.to_dict(), indent=1))

    def load(self) -> JobState:
        text = (self.run_dir / "state.json").read_text(encoding="utf-8")
        return JobState.from_dict(json.loads(text))

    def event(self, kind: str, **data) -> None:
        row = {"ts": time.time(), "kind": kind, **data}
        with (self.run_dir / "timeline.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")

    def iteration_dir(self, n: int) -> Path:
        d = self.run_dir / "iterations" / f"{n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d
