"""Resolve the mainnet top algorithm, compile and score it once, cache the result."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from talos import inside
from talos import mainnet as _mainnet
from talos.bench import EvalRequest
from talos.challenges import DEV_IMAGE_TAG, MONOREPO_REF
from talos.leak_check import KNOWN_LEAKING, describe, leak_findings
from talos.state import BaselineRecord, _atomic_write
from talos.types import NonceResult, NonceSet


class BaselineError(RuntimeError):
    pass


def effective_hyperparameters(hp: dict[str, dict | None] | None) -> dict[str, dict] | None:
    """The per-track map as it changes a run. A track mapped to None runs exactly as a track with
    no entry, so both normalise away; a map with nothing left is None. A track's {} is kept: it is
    passed to tig-runtime and is not the same input as no flag."""
    if hp is None:
        return None
    kept = {track: v for track, v in hp.items() if v is not None}
    return kept or None  # the whole map, not a track's value: empty means "no flag anywhere"


def cache_key(challenge: str, monorepo_ref: str, name: str, training: list[NonceSet],
              holdout: list[NonceSet], fuel: int, hardware_class: str,
              hyperparameters: dict[str, dict | None] | None = None) -> str:
    h = hashlib.sha256()
    payload = {"challenge": challenge, "ref": monorepo_ref, "name": name, "fuel": fuel,
               "hw": hardware_class, "layout": inside.CRATE_LAYOUT, "image": DEV_IMAGE_TAG,
               "training": [(n.track, n.rand_hash, n.start, n.count) for n in training],
               "holdout": [(n.track, n.rand_hash, n.start, n.count) for n in holdout]}
    effective = effective_hyperparameters(hyperparameters)
    if effective is not None:
        # Only when present, so a key computed before hyperparameters existed is unchanged.
        payload["hyperparameters"] = effective
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:24]


def _require_scoreable(challenge: str, fuel: int, label: str, nonce_sets: list[NonceSet],
                      results: list[NonceResult]) -> None:
    """A baseline that scores nothing on a track cannot be compared against, let alone beaten:
    every iteration would fail at scoring until the budget was gone. Fail here, once, naming the
    error kinds, rather than once per iteration for the rest of the run."""
    by_track: dict[str, list[NonceResult]] = {ns.track: [] for ns in nonce_sets}
    for r in results:
        by_track.setdefault(r.track, []).append(r)
    for track, rows in sorted(by_track.items()):
        ok = [r for r in rows if r.ok and r.quality is not None]
        mean_quality = sum(r.quality for r in ok) / len(ok) if ok else 0.0
        if not ok or mean_quality <= 0:
            kinds = dict(Counter(r.error for r in rows))
            raise BaselineError(
                f"baseline for {challenge} at fuel {fuel} is unscoreable on {label} track "
                f"{track!r}: {len(ok)}/{len(rows)} nonces scored, mean quality {mean_quality:g}, "
                f"errors {kinds}")


def _require_leak_free(challenge: str, name: str, files: dict[str, str]) -> None:
    """Checked on the files themselves: the branch they come from can move after job start."""
    findings = leak_findings(challenge, files)
    if findings:
        raise BaselineError(f"baseline {name} carries the hidden_seed leak: {describe(findings)}")


def resolve_baseline(challenge: str, training: list[NonceSet], holdout: list[NonceSet],
                     fuel: int, bench, cache_dir: Path, hardware_class: str, rule,
                     mainnet=_mainnet, log=lambda msg: None, algorithm: dict | None = None,
                     hyperparameters: dict[str, dict | None] | None = None,
                     ) -> tuple[BaselineRecord, str]:
    if algorithm is not None:
        # Pinned at job start with the hyperparameters, which belong to this algorithm's code.
        name, adoption = algorithm["name"], algorithm["adoption"]
    else:
        top = mainnet.top_algorithm(challenge)
        if top is None:
            raise BaselineError(f"no adopted, compiled, leak-free algorithm found on mainnet for "
                                f"{challenge}")
        name, _algorithm_id, adoption = top
    if name in KNOWN_LEAKING.get(challenge, ()):
        raise BaselineError(f"baseline {name} carries the hidden_seed leak; start a new job")
    template = mainnet.fetch_template(challenge)
    key = cache_key(challenge, MONOREPO_REF, name, training, holdout, fuel, hardware_class,
                    hyperparameters)
    cache_file = Path(cache_dir) / challenge / f"{key}.json"
    if cache_file.exists():
        try:
            rec = BaselineRecord.from_dict(json.loads(cache_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log(f"baseline {name}: cache file {cache_file} is corrupt; re-measuring")
        else:
            _require_leak_free(challenge, name, rec.files)
            log(f"baseline {name}: cache hit {key}")
            return rec, template
    files = mainnet.fetch_algorithm_files(challenge, name)
    _require_leak_free(challenge, name, files)
    # mainnet adoption is a fixed-point fraction of the challenge's benchmarker usage, 10**18 = 100%
    log(f"baseline {name} (adoption {adoption / 10**18:.2%}): "
        f"compiling and scoring {len(files)} file(s)")
    r = bench.evaluate(EvalRequest(challenge=challenge, files=files, training=training,
                                   holdout=holdout, fuel=fuel, baseline_training=None, rule=rule,
                                   hyperparameters=hyperparameters))
    if not r.compile.ok:
        raise BaselineError(f"baseline {name} failed to compile; likely dev-image drift at "
                            f"{MONOREPO_REF}.\n{r.compile.output[-4000:]}")
    if r.holdout is None:
        raise BaselineError(f"baseline {name}: held-out set was not scored ({r.holdout_reason})")
    tr, ho = r.training, r.holdout
    try:
        _require_scoreable(challenge, fuel, "training", training, tr)
        _require_scoreable(challenge, fuel, "held-out", holdout, ho)
    except BaselineError as e:
        if effective_hyperparameters(hyperparameters) is None:
            raise
        raise BaselineError(f"{e}. The baseline ran with mainnet hyperparameters; start a new job "
                            f"with --hyperparameters none to rule them out") from None
    rec = BaselineRecord(name=name, adoption=adoption,
                         artifact_id=r.compile.artifact_id, files=files,
                         training=tr, holdout=ho)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(cache_file, json.dumps(rec.to_dict(), indent=1))
    return rec, template
