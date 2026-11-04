"""Source checks for the energy_arbitrage price leak. Before TIG commit 2703c9af the Challenge
serialised its private hidden_seed, which fixes every future real-time price. The pin builds
the fixed challenge, where that code reads nothing, but it would still inflate on an older build
and it must not be handed back as submit-ready. Pure: no network, no I/O.

A static scan cannot prove that code is causal, only that it has none of the routes below."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Named so a baseline search never spends a GitHub fetch on them, and so a mainnet name stays
# refused however its source changes. A local fork with the leak removed passes on its source.
KNOWN_LEAKING: dict[str, frozenset[str]] = {
    "energy_arbitrage": frozenset({"first_energy", "super_energy", "titan_killer", "gridflow"}),
}
# The pin before the fix, for a job that recorded it beside an image tag. Jobs older than the
# image pin have no tag at all (JobSpec.dev_image_tag).
PRE_FIX_REFS = frozenset({"84a5787f5b14a630bdf40f52bccf37887d3d8464"})

RULES: dict[str, str] = {
    "energy_arbitrage": (
        "Causality rule (a candidate that breaks it is rejected, and any score it got is void): "
        "the action at a step may depend only on public instance data and on the real-time "
        "prices up to and including that step. Do not read hidden_seed; do not serialise or "
        "debug-format the Challenge; do not read process arguments, the environment, files or "
        "raw memory; do not reconstruct future real-time prices. Save exactly the schedule "
        "grid_optimize returns: recording prices during the rollout and then rewriting the "
        "schedule is hindsight, so do not add such a pass or enable one already in the code. "
        "Sampling price scenarios from your own randomness or from the public challenge.seed "
        "is allowed."),
}

_NAME = r"(?:self\.)?(?:challenge|instance|ch|c)"
_FUEL_BLOCK = (r"\s*(?:#\[[^\]]*\]\s*)*(?:pub\s+)?(?:safe\s+)?static\s+(?:mut\s+)?"
               r"__fuel_remaining\s*:\s*u64\s*;\s*\}")

# Each matches how the route is reached, not what a function is called: peakshift still has a
# stubbed exact_price_table that returns None. generate_rt_prices is not here, because sampling
# scenarios from the public challenge.seed is honest and wasserwert does it.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (why, re.compile(pattern)) for why, pattern in (
        ("reads hidden_seed", r"\bhidden_seed\b"),
        ("serialises data, the route to hidden_seed before the fix",
         r"\bserde_json::to_\w+|serde_json::\{[^}]*\bto_\w+|\bbincode\b|\.serialize\s*\("
         r"|\bSerialize::serialize\b"
         rf"|\bto_(?:value|vec|string|string_pretty|writer|writer_pretty)\s*\(\s*&?\s*\*?\s*"
         rf"{_NAME}\b(?!\.)"),
        ("debug-formats the challenge",
         rf"\"[^\"\n]*\{{:#?\?\}}[^\"\n]*\"\s*,[^;\n]*&?\b{_NAME}\b(?![.\w])"
         rf"|\{{{_NAME}:#?\?\}}|\bdbg!\(\s*&?\s*{_NAME}\s*\)"),
        ("carries the published leak functions", r"\b(?:secondary_entropy|expand_price_table)\b"),
        ("reads raw memory",
         r"\btransmute(?:_copy)?\b|\bfrom_raw_parts(?:_mut)?\b|\bas\s+\*(?:const|mut)\b"
         r"|&raw\s+(?:const|mut)\b|\*(?:const|mut)\s+(?:Challenge|u8)\b|\basm!"
         rf"|\baddr_of(?:_mut)?!\s*\(\s*\*?\s*{_NAME}\b"),
        ("reads the process or the file system, where rand_hash and the nonce are",
         r"\bstd::(?:env|fs|process|net)\b|\buse\s+std::\{[^}]*\b(?:env|fs|process|net)\b"
         r"|/proc/|\blibc\b|\bdlopen\b|\binclude_(?:bytes|str)!"),
        ("declares a foreign function", rf"\bextern\s+\"C\"\s*\{{(?!{_FUEL_BLOCK})"),
    ))

# A write to a Solution's schedule. grid_optimize is what makes a schedule causal, and the
# verifier accepts any feasible one. A zero-filled fallback saved before the rollout is allowed.
_HINDSIGHT = re.compile(
    r"\.schedule\s*=[^=]|\.schedule\s*\[[^\n]*\]\s*[-+*/]?=[^=]"
    r"|&mut\s+[\w.]+\.schedule\b"
    r"|\.schedule\s*\.(?:iter_mut|last_mut|first_mut|get_mut|push|insert|clear|swap|truncate)\b"
    r"|\bSolution\s*\{\s*schedule\b(?!\s*:\s*vec!\[\s*vec!\[\s*0\.0)"
    r"|=\s*[\w.]+\.schedule(?:\.clone\(\))?\s*;")

_TOKEN = re.compile(r"//|/\*|\bb?r(#*)\"|\"|'")
_CHAR = re.compile(r"'(?:\\.[^']*|[^\\'])'")
_BLOCK_EDGE = re.compile(r"/\*|\*/")


@dataclass(frozen=True)
class LeakFinding:
    path: str
    line: int
    why: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.why}: {self.text[:80]}"


def _blank(text: str) -> str:
    """Spaces for everything but newlines, so a finding's line number stays right."""
    return re.sub(r"[^\n]", " ", text)


def strip_comments(source: str) -> str:
    """Rust source with comments blanked. A rule may name hidden_seed in a comment saying not to
    read it; strings stay, because "/proc/self/cmdline" lives in one."""
    out, i = [], 0
    while (m := _TOKEN.search(source, i)) is not None:
        out.append(source[i:m.start()])
        i, token = m.end(), m.group()
        if token == "//":
            end = source.find("\n", i)
            end = len(source) if end < 0 else end
            out.append(_blank(source[m.start():end]))
            i = end
        elif token == "/*":
            depth, end = 1, i
            while depth and (edge := _BLOCK_EDGE.search(source, end)) is not None:
                depth += 1 if edge.group() == "/*" else -1
                end = edge.end()
            end = len(source) if depth else end
            out.append(_blank(source[m.start():end]))
            i = end
        elif token == "'":
            # A char literal, so '"' opens no string; anything else is a lifetime.
            char = _CHAR.match(source, m.start())
            i = char.end() if char else i
            out.append(source[m.start():i])
        else:
            if token == '"':
                end = i
                while end < len(source) and source[end] != '"':
                    end += 2 if source[end] == "\\" else 1
                end = min(end + 1, len(source))
            else:
                close = '"' + m.group(1)
                found = source.find(close, i)
                end = len(source) if found < 0 else found + len(close)
            out.append(source[m.start():end])
            i = end
    out.append(source[i:])
    return "".join(out)


def _sites(path: str, source: str, pattern: re.Pattern[str], why: str) -> list[LeakFinding]:
    matches = list(pattern.finditer(source))
    if not matches:
        return []
    lines = source.split("\n")
    return [LeakFinding(path, n + 1, why, lines[n].strip())
            for n in (source.count("\n", 0, m.start()) for m in matches)]


def _rust_files(files: dict[str, str]) -> list[tuple[str, str]]:
    return [(path, strip_comments(text)) for path, text in sorted(files.items())
            if path.endswith(".rs")]


def leak_findings(challenge: str, files: dict[str, str]) -> list[LeakFinding]:
    """Every route to hidden_seed or a future price in these files; empty for a challenge without
    rules, which costs nothing."""
    if challenge not in KNOWN_LEAKING:
        return []
    return [finding for path, source in _rust_files(files) for why, pattern in _RULES
            for finding in _sites(path, source, pattern, why)]


def hindsight_additions(challenge: str, parent: dict[str, str],
                        files: dict[str, str]) -> list[LeakFinding]:
    """The schedule writes in `files` that `parent` does not have. Measured against the parent
    because peakshift's root has inactive ones: track 3's sits behind its None stub and track 4's
    behind a flag that is off. So this cannot see a candidate turning one of those on. A line
    moved between files is not an addition."""
    if challenge not in KNOWN_LEAKING:
        return []
    why = "rewrites the schedule grid_optimize committed to"
    held = Counter(f.text for path, source in _rust_files(parent)
                   for f in _sites(path, source, _HINDSIGHT, why))
    added = []
    for path, source in _rust_files(files):
        for finding in _sites(path, source, _HINDSIGHT, why):
            if held[finding.text] > 0:
                held[finding.text] -= 1
            else:
                added.append(finding)
    return added


def describe(findings: list[LeakFinding], limit: int = 3) -> str:
    shown = "; ".join(str(f) for f in findings[:limit])
    more = len(findings) - limit
    return shown + (f"; and {more} more" if more > 0 else "")


def void_reason(spec, state, root_files: dict[str, str] | None = None) -> str | None:
    """Why this job's scores cannot be trusted, or None. Rule: a score counts only from the fixed
    challenge, against a baseline free of the leak, from code free of it."""
    leaking = KNOWN_LEAKING.get(spec.challenge)
    if leaking is None:
        return None
    if spec.dev_image_tag is None or spec.monorepo_ref in PRE_FIX_REFS:
        return (f"built at monorepo {spec.monorepo_ref}, before TIG's hidden_seed fix "
                "(2703c9af)")
    named = spec.baseline_algorithm["name"] if spec.baseline_algorithm else None
    if named in leaking:
        return f"its baseline {named} carries the hidden_seed leak"
    best = state.best.files if state.best else None
    baseline = state.baseline.files if state.baseline else None
    for label, files in (("baseline", baseline), ("root", root_files), ("best candidate", best)):
        findings = leak_findings(spec.challenge, files) if files else []
        if findings:
            return f"its {label} carries the hidden_seed leak: {describe(findings, limit=1)}"
    return None
