"""Read-only mainnet and monorepo access. Every function takes its HTTP getter as a
parameter so tests never touch the network."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import mean

from talos.challenges import MONOREPO_REF, TEMPLATE_START_FILES
from talos.leak_check import KNOWN_LEAKING, leak_findings

MAINNET_API = "https://mainnet-api.tig.foundation"
GH_REPO = "tig-foundation/tig-monorepo"
GH_API = f"https://api.github.com/repos/{GH_REPO}"
GH_RAW = f"https://raw.githubusercontent.com/{GH_REPO}"
HTTP_TIMEOUT = 15
UA = "talos-tig"


class MainnetError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _get(url: str, accept: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise MainnetError(f"HTTP {e.code} fetching {url}", status=e.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise MainnetError(f"network error fetching {url}: {e}", status=None) from None


def _get_json(url: str):
    return json.loads(_get(url, "application/json"))


def _get_text(url: str) -> str:
    return _get(url, "text/plain").decode("utf-8")


@dataclass(frozen=True)
class ChallengeInfo:
    id: str
    name: str
    is_gpu: bool
    tracks: list[str]
    max_fuel: int


def _block_id(get_json) -> str:
    return get_json(f"{MAINNET_API}/get-block")["block"]["id"]


def fetch_challenges(get_json=_get_json) -> list[ChallengeInfo]:
    """Every named challenge on mainnet, from one block lookup and one `get-challenges` call.
    A challenge may have no active tracks; `max_fuel` is 0 when mainnet gives no budget."""
    block_id = _block_id(get_json)
    resp = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    out = []
    for c in resp["challenges"]:
        cfg = c.get("config") or {}
        name = cfg.get("name")
        if not name:
            continue
        out.append(ChallengeInfo(id=c["id"], name=name, is_gpu=cfg.get("type") == "gpu",
                                 tracks=sorted((cfg.get("active_tracks") or {}).keys()),
                                 max_fuel=int(cfg.get("max_fuel_budget") or 0)))
    return out


def fetch_challenge_info(name: str, get_json=_get_json) -> ChallengeInfo:
    for info in fetch_challenges(get_json):
        if info.name != name:
            continue
        if not info.tracks:
            raise MainnetError(f"challenge {name} has no active tracks on mainnet")
        if not info.max_fuel:
            # was a KeyError before fetch_challenges defaulted it; a job needs a real budget
            raise MainnetError(f"challenge {name} has no max_fuel_budget on mainnet")
        return info
    raise MainnetError(f"challenge {name!r} not found on mainnet")


def _compiled_algorithms(name: str, get_json) -> list[tuple[str, str, int]]:
    """(algorithm_name, algorithm_id, adoption) of every named, compiled algorithm of a challenge.
    The id is what a benchmark's precommit names its algorithm by."""
    block_id = _block_id(get_json)
    challenges = get_json(f"{MAINNET_API}/get-challenges?block_id={block_id}")
    algos = get_json(f"{MAINNET_API}/get-algorithms?block_id={block_id}")
    cid = next((c["id"] for c in challenges["challenges"]
                if (c.get("config") or {}).get("name") == name), None)
    if cid is None:
        raise MainnetError(f"challenge {name!r} not found on mainnet")
    compiled = {b["algorithm_id"]: bool((b.get("details") or {}).get("compile_success"))
                for b in algos.get("binarys", [])}
    out: list[tuple[str, str, int]] = []
    for algo in algos.get("codes", []):
        details = algo.get("details") or {}
        if details.get("challenge_id") != cid or not compiled.get(algo["id"]):
            continue
        try:
            adoption = int((algo.get("block_data") or {}).get("adoption") or 0)
        except (TypeError, ValueError):
            adoption = 0
        if details.get("name"):
            out.append((details["name"], algo["id"], adoption))
    return out


def top_algorithm(name: str, get_json=_get_json, get_text=_get_text
                  ) -> tuple[str, str, int] | None:
    """(algorithm_name, algorithm_id, adoption) of the highest-adoption compiled algorithm free of
    a price leak (talos/leak_check.py), or None. A challenge with no leak rules fetches nothing;
    on one with rules, a fetch that fails raises rather than choosing an algorithm unread."""
    leaking = KNOWN_LEAKING.get(name)
    adopted = [found for found in _compiled_algorithms(name, get_json) if found[2] > 0]
    for found in sorted(adopted, key=lambda f: -f[2]):  # stable: the first listed wins a tie
        if leaking is None:
            return found
        if found[0] not in leaking and not leak_findings(
                name, fetch_algorithm_files(name, found[0], get_text, get_json)):
            return found
    return None


def find_algorithm(name: str, algorithm: str, get_json=_get_json) -> tuple[str, str, int] | None:
    """The compiled algorithm of this name, at any adoption, or None."""
    return next((found for found in _compiled_algorithms(name, get_json)
                 if found[0] == algorithm), None)


@dataclass(frozen=True)
class TrackHyperparameters:
    """One track's best mainnet benchmark for an algorithm. Every field is None when no benchmark
    qualified; `hyperparameters` alone is None when that benchmark ran without any."""
    hyperparameters: dict | None
    benchmark_id: str | None
    player_id: str | None
    mean_quality: float | None

    def source(self) -> dict:
        return {"benchmark_id": self.benchmark_id, "player_id": self.player_id,
                "mean_quality": self.mean_quality}


def top_hyperparameters(algorithm_id: str, tracks: list[str], fuel: int,
                        get_json=_get_json) -> dict[str, TrackHyperparameters]:
    """Per track, the hyperparameters of the highest mean-quality benchmark that ran
    `algorithm_id` at exactly `fuel`, among every player's benchmarks in mainnet's recent window.
    Frauds and benchmarks without a quality are skipped; ties go to the lower benchmark id. One
    player that cannot be read raises: the choice is never made from a partial view. An
    unexpected shape in the response (a missing key, a row that is not a dict) raises
    MainnetError too, rather than a bare KeyError/TypeError out of `talos run`."""
    block_id = _block_id(get_json)
    try:
        opow = get_json(f"{MAINNET_API}/get-opow?block_id={block_id}")
        best: dict[str, tuple[float, str, dict]] = {}
        for player in [row["player_id"] for row in opow.get("opow", [])]:
            data = get_json(f"{MAINNET_API}/get-benchmarks?block_id={block_id}&player_id={player}")
            frauds = {f["benchmark_id"] for f in data.get("frauds") or []}
            quality = {b["id"]: (b.get("details") or {}).get("average_quality_by_bundle")
                       for b in data.get("benchmarks") or []}
            for pre in data.get("precommits") or []:
                bid = pre["benchmark_id"]
                settings, details = pre.get("settings") or {}, pre.get("details") or {}
                bundles = quality.get(bid)
                if (settings.get("algorithm_id") != algorithm_id
                        or settings.get("track_id") not in tracks
                        or details.get("fuel_budget") != fuel
                        or bid in frauds or not bundles):  # None/[]: no quality to rank
                    continue
                score = mean(bundles)
                held = best.get(settings["track_id"])
                if held is None or score > held[0] or (score == held[0] and bid < held[1]):
                    best[settings["track_id"]] = (score, bid, pre)
        out: dict[str, TrackHyperparameters] = {}
        for track in tracks:
            if track not in best:
                out[track] = TrackHyperparameters(None, None, None, None)
                continue
            score, bid, pre = best[track]
            out[track] = TrackHyperparameters(pre["details"].get("hyperparameters"), bid,
                                              pre["settings"].get("player_id"), float(score))
        return out
    except (KeyError, TypeError, ValueError) as e:
        # get_json's own MainnetError (a RuntimeError) passes through this clause untouched;
        # only a malformed-shape failure while parsing the response is converted.
        raise MainnetError(f"unexpected /get-benchmarks data: {e!r}") from None


def _walk(path: str, ref: str, get_json) -> list[str]:
    entries = get_json(f"{GH_API}/contents/{path}?ref={ref}")
    if isinstance(entries, dict):  # a single file, not a directory
        return [entries["path"]]
    out: list[str] = []
    for e in entries:
        if e["type"] == "dir":
            out.extend(_walk(e["path"], ref, get_json))
        elif e["type"] == "file":
            out.append(e["path"])
    return out


def fetch_algorithm_files(name: str, algorithm: str, get_text=_get_text,
                          get_json=_get_json) -> dict[str, str]:
    """{relative_path: contents}. A single-file algorithm comes back as {"mod.rs": ...}."""
    ref = f"{name}/{algorithm}"
    base_dir = f"tig-algorithms/src/{name}/{algorithm}"
    try:
        paths = _walk(base_dir, ref, get_json)
    except MainnetError as e:
        if e.status != 404:
            raise
        single = f"tig-algorithms/src/{name}/{algorithm}.rs"
        return {"mod.rs": get_text(f"{GH_RAW}/{ref}/{single}")}
    files: dict[str, str] = {}
    for p in paths:
        rel = p[len(base_dir) + 1:] if p.startswith(base_dir + "/") else p.rsplit("/", 1)[-1]
        files[rel] = get_text(f"{GH_RAW}/{ref}/{p}")
    if not files:
        raise MainnetError(f"no files found for {name}/{algorithm}")
    return files


def fetch_template(name: str, get_text=_get_text) -> str:
    return get_text(f"{GH_RAW}/{MONOREPO_REF}/tig-algorithms/src/{name}/template.rs")


def fetch_template_cu(name: str, get_text=_get_text) -> str:
    """A GPU challenge's kernel template, the template root's algorithm.cu."""
    return get_text(f"{GH_RAW}/{MONOREPO_REF}/tig-algorithms/src/{name}/template.cu")


def fetch_challenge_docs(name: str, get_text=_get_text) -> dict[str, str]:
    """The challenge's own files at the pinned monorepo commit, for a template start. The
    reference-algorithm policy does not apply: nothing here is an algorithm."""
    if name not in TEMPLATE_START_FILES:
        raise MainnetError(f"challenge {name!r} has no file list for a template start")
    base = f"{GH_RAW}/{MONOREPO_REF}/tig-challenges/src/{name}"
    return {f: get_text(f"{base}/{f}") for f in TEMPLATE_START_FILES[name]}
