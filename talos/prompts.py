"""Prompt builders and parsers. Pure functions over PromptContext; no I/O beyond reading the
packaged Rust rules once."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from talos.diagnostics import relevant
from talos.leak_check import RULES

STRATEGY_TAGS = ["construction", "local_search", "metaheuristic", "constraint_relaxation",
                 "decomposition", "hybrid", "data_structure", "parameter_tuning"]

SEARCH_REPLACE_FORMAT = """\
Respond ONLY with one or more edit blocks in exactly this format:

<<<<<<< SEARCH path/to/file.rs
<lines to find, copied exactly>
=======
<replacement lines>
>>>>>>> REPLACE

Rules: the SEARCH text must match the current file exactly once; keep each block small;
emit several blocks for several places; never emit whole-file rewrites; never touch files
other than the algorithm files shown."""

WHOLE_FILE_FORMAT = """\
Respond ONLY with one block per file you change, in exactly this format:

<<<<<<< FILE path/to/file.rs
<the complete new contents of the file>
>>>>>>> END FILE

Rules: give each file in full, never a diff or an excerpt; a file you do not give is left
unchanged; touch only the algorithm files shown."""


@lru_cache(maxsize=1)
def _rust_rules() -> str:
    return resources.files("talos.data").joinpath("rust_rules.md").read_text(encoding="utf-8")


def causality_rules(challenge: str) -> str:
    """What this challenge's source check rejects, so the LLM is told before it is refused."""
    rules = RULES.get(challenge)
    return f"{rules}\n\n" if rules else ""


@dataclass
class PromptContext:
    challenge: str
    template_rs: str
    direction: str
    tacit: str
    files: dict[str, str]
    baseline_name: str
    best_delta: float
    failed_hypotheses: list[dict] = field(default_factory=list)
    anchor: int = 0  # the parent node id; 0 is the root
    forced_tag: str | None = None
    is_gpu: bool = False
    track: str | None = None
    guard_tracks: list[str] = field(default_factory=list)
    hyperparameters: dict[str, dict | None] | None = None
    scoring: str = "metered"
    start: str = "mainnet"
    challenge_docs: dict[str, str] = field(default_factory=dict)
    # {"node": int, "mean": float, "tracks": {track: delta}}; None means not scored (the
    # template root, or no best yet)
    parent_gap: dict | None = None
    best_gap: dict | None = None
    whole_file: bool = False
    search: str = "greedy"
    chains: int = 1
    # A GPU template start's template.cu, the root's algorithm.cu; None otherwise.
    template_cu: str | None = None


def _files_block(files: dict[str, str]) -> str:
    return "\n\n".join(f"--- {name} ---\n{text}" for name, text in sorted(files.items()))


def _gap_line(prefix: str, gap: dict, baseline_name: str) -> str:
    line = f"{prefix} (node #{gap['node']}): {gap['mean']:+.3%} vs mainnet \"{baseline_name}\""
    for track, delta in sorted(gap["tracks"].items()):
        line += f"; track {track} {delta:+.3%}"
    return line


def gap_block(ctx: PromptContext) -> str:
    """Where the parent and the best node stand against the mainnet baseline."""
    if ctx.parent_gap is not None:
        prefix = ("Base algorithm, the parent" if ctx.start == "base" and ctx.anchor == 0
                  else "Parent")
        parent = _gap_line(prefix, ctx.parent_gap, ctx.baseline_name)
    elif ctx.start == "template":
        parent = "Parent: the empty template (not scored yet)"
    else:
        parent = "Parent: the mainnet baseline itself (+0.000%)"
    best = ("Best so far: none yet" if ctx.best_gap is None
            else _gap_line("Best so far", ctx.best_gap, ctx.baseline_name))
    text = f"{parent}\n{best}"
    if ctx.start != "mainnet":
        text += "\nThe baseline's code is not available to you; only its score is."
    return text


def template_cu_block(ctx: PromptContext) -> str:
    """For a GPU template start: which file holds the kernels, and its starting text."""
    if ctx.start != "template" or ctx.template_cu is None:
        return ""
    return ("algorithm.cu holds your CUDA kernels; it starts as TIG's template.cu:\n"
            f"```cuda\n{ctx.template_cu}\n```\n\n")


def challenge_block(ctx: PromptContext) -> str:
    """The challenge definition files of a template start; empty otherwise."""
    if ctx.start != "template":
        return ""
    names = sorted(ctx.challenge_docs, key=lambda n: (n != "README.md", n))
    sections = [f"--- tig-challenges/src/{ctx.challenge}/{n} ---\n{ctx.challenge_docs[n]}"
                for n in names]
    return ("The challenge definition (from the pinned TIG monorepo; read it, do not edit it):"
            "\n\n" + "\n\n".join(sections))


def focus_sentence(ctx: PromptContext) -> str:
    """The target sentence for a focused job; empty when the job optimises every track."""
    if ctx.track is None:
        return ""
    guards = ", ".join(ctx.guard_tracks) or "none"
    return (f"Optimise for track \"{ctx.track}\" only. The other active tracks ({guards}) are "
            f"re-scored as a regression guard when a candidate wins, and none of them may get "
            f"worse: confine changes to the code path that serves \"{ctx.track}\".")


def hyperparameters_block(ctx: PromptContext) -> str:
    """What every run passes to solve_challenge, so edits keep those keys readable. Empty when the
    job runs without hyperparameters. A focused job shows only its own track."""
    if ctx.hyperparameters is None:
        return ""
    shown = [ctx.track] if ctx.track else sorted(ctx.hyperparameters)
    lines = []
    for track in shown:
        hp = ctx.hyperparameters.get(track)
        value = ("none (solve_challenge receives None)" if hp is None
                 else json.dumps(hp, sort_keys=True, separators=(",", ":")))
        lines.append(f"track {track}: {value}")
    text = ("Hyperparameters: every run of this code, the baseline and your candidate alike, "
            "passes these values to solve_challenge as `hyperparameters`, per track. They came "
            "from the best mainnet benchmark of this algorithm. Keep every key readable: you may "
            "add keys, but do not rename or remove existing ones.\n" + "\n".join(lines))
    if ctx.track and len(ctx.hyperparameters) > 1:
        text += "\nThe guard tracks run with their own values too."
    return text


def in_tree(start: str, search: str, chains: int) -> bool:
    """Whether an attempt's parent can be a node that never was the job's best: a template or
    base start, annealing or several chains. Otherwise every parent is the baseline or a best."""
    return start in ("template", "base") or search == "anneal" or chains > 1


def parent_label(against: int, start: str, tree: bool) -> str:
    """The code an attempt was built on: node 0 is the baseline, the empty template in a template
    start or the base algorithm in a base start; any other node is a best, or just a node in a
    tree."""
    if against == 0:
        return {"template": "the empty template",
                "base": "the base algorithm"}.get(start, "the baseline")
    return f"node #{against}" if tree else f"best #{against}"


def _parent(h: dict, ctx: PromptContext | None) -> str:
    if ctx is None:
        return parent_label(h["against"], "mainnet", False)
    return parent_label(h["against"], ctx.start, in_tree(ctx.start, ctx.search, ctx.chains))


def describe_attempt(h: dict, labelled: bool = False, ctx: PromptContext | None = None) -> str:
    """One line per failed attempt, with what the run measured: a title plus the outcome told
    the model nothing about a +0.14% track, a -0.37% track and a 14x runtime. `labelled` names
    the code the attempt failed against, in `ctx`'s wording."""
    tag = h.get("outcome", "")
    if labelled and "against" in h:
        tag += f", against {_parent(h, ctx)}"
    line = f"- {h.get('title', '')} [{tag}]"
    if "mean_rel_delta" in h:
        line += (f" mean {h['mean_rel_delta']:+.2%}, worst track {h.get('worst_track', '?')} "
                 f"{h.get('worst_rel_delta', 0.0):+.2%}")
        if "runtime_ratio" in h:
            line += f", runtime {h['runtime_ratio']:.1f}x baseline"
    if h.get("error"):
        line += f": {str(h['error'])[:200]}"
    return line


def recall_block(ctx: PromptContext) -> str:
    """The failed attempts of the whole job. Issue #33: listing only those against the current
    best let run 20260925-140701-knapsack retry an idea that failed before the best changed."""
    current = [h for h in ctx.failed_hypotheses if h.get("against", ctx.anchor) == ctx.anchor]
    earlier = [h for h in ctx.failed_hypotheses if h.get("against", ctx.anchor) != ctx.anchor]
    parts = []
    if current:
        parts.append("Already tried against this exact code and did NOT help; do not repeat:\n"
                     + "\n".join(describe_attempt(h) for h in current))
    if earlier:
        other = ("other code" if in_tree(ctx.start, ctx.search, ctx.chains)
                 else "an earlier best")
        parts.append(f"Tried against {other} and did NOT help there; do not repeat "
                     "unless the current code removes the reason it failed:\n"
                     + "\n".join(describe_attempt(h, labelled=True, ctx=ctx) for h in earlier))
    return "\n\n".join(parts)


def hypothesis_prompts(ctx: PromptContext) -> tuple[str, str]:
    scope = f"on track \"{ctx.track}\"" if ctx.track else "across every active track"
    focus = (focus_sentence(ctx) + "\n\n") if ctx.track else ""
    role = ("writing a Rust solver from the empty template for"
            if ctx.start == "template" else "improving a Rust solver for")
    system = (
        f"You are a research engineer {role} the TIG challenge "
        f"\"{ctx.challenge}\". The goal is to beat the current mainnet state of the art on "
        f"TIG's own benchmark: higher verifier quality per nonce under a fixed fuel budget, "
        f"{scope}.\n\n{focus}"
        f"The solver must keep this contract (template.rs):\n```rust\n{ctx.template_rs}\n```\n\n"
        f"{template_cu_block(ctx)}"
        f"{causality_rules(ctx.challenge)}"
        f"Propose ONE specific change. Reply with a JSON object with keys \"title\" "
        f"(short), \"description\" (what to change and why it should raise quality), "
        f"\"design_family\" (a short name for the overall approach, such as "
        f"\"rolling-horizon LP\"), and "
        f"\"strategy_tag\" (one of: {', '.join(STRATEGY_TAGS)}). No other text."
    )
    parts = [f"Baseline to beat: mainnet algorithm \"{ctx.baseline_name}\".\n{gap_block(ctx)}",
             f"Direction from the user:\n{ctx.direction}"]
    if ctx.tacit.strip():
        parts.append(f"Tacit knowledge (lessons so far):\n{ctx.tacit}")
    recall = recall_block(ctx)
    if recall:
        parts.append(recall)
    if ctx.forced_tag:
        parts.append(f"You have stagnated. Your strategy_tag MUST be \"{ctx.forced_tag}\" "
                     f"this time; change the approach, not the constants.")
    hp = hyperparameters_block(ctx)
    if hp:
        parts.append(hp)
    challenge = challenge_block(ctx)
    if challenge:
        parts.append(challenge)
    parts.append("Current algorithm source:\n" + _files_block(ctx.files))
    return system, "\n\n".join(parts)


def edit_prompts(ctx: PromptContext, hypothesis: dict) -> tuple[str, str]:
    system = (
        f"You are editing a Rust solver for the TIG challenge \"{ctx.challenge}\".\n\n"
        f"{causality_rules(ctx.challenge)}"
        f"{WHOLE_FILE_FORMAT if ctx.whole_file else SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}"
    )
    challenge = challenge_block(ctx)
    focus = focus_sentence(ctx)
    hp = hyperparameters_block(ctx)
    user = (f"Implement this hypothesis:\nTitle: {hypothesis['title']}\n"
            f"Description: {hypothesis['description']}\n\n"
            + (focus + "\n\n" if focus else "")
            + (hp + "\n\n" if hp else "")
            + (challenge + "\n\n" if challenge else "")
            + f"Current algorithm source files:\n{_files_block(ctx.files)}")
    return system, user


def compile_fix_prompts(ctx: PromptContext, files: dict[str, str],
                        compiler_output: str) -> tuple[str, str]:
    system = (f"You are fixing a Rust compile error in a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}")
    # 12000 holds the eight errors of run 20260916-095103 iteration 5; 6000 showed the last few.
    challenge = challenge_block(ctx)
    user = (f"The build failed with:\n```\n{relevant(compiler_output)[-12000:]}\n```\n\n"
            f"{_file_names_line(files)}\n\n"
            + (challenge + "\n\n" if challenge else "")
            + f"Current files:\n{_files_block(files)}\n\nEmit edit blocks that fix the build "
            f"without abandoning the intended change.")
    return system, user


def dead_code_fix_prompts(ctx: PromptContext, files: dict[str, str],
                          names: list[str]) -> tuple[str, str]:
    """The build succeeded but the functions the edit added are never called, so the change is
    not on the solve path. Same contract as compile_fix_prompts."""
    system = (f"You are completing an unfinished edit to a Rust solver for the TIG "
              f"\"{ctx.challenge}\" solver.\n\n{SEARCH_REPLACE_FORMAT}\n\n{_rust_rules()}")
    listed = "\n".join(f"- {n}" for n in names)
    user = (f"The build succeeded, but these functions the edit added are never called, so the "
            f"change cannot affect the score:\n{listed}\n\n{_file_names_line(files)}\n\n"
            f"Current files:\n{_files_block(files)}\n\nEmit edit blocks that call them from "
            f"the solve path as the hypothesis intended (or remove them if the hypothesis is "
            f"already implemented without them).")
    return system, user


def _file_names_line(files: dict[str, str]) -> str:
    return ("The only files you may edit, and the names to use in SEARCH headers: "
            + ", ".join(sorted(files)) + ". Other algorithms in the build output are not yours.")


def edit_repair_prompts(ctx: PromptContext, files: dict[str, str],
                        misses_text: str) -> tuple[str, str]:
    system = (f"You are repairing edit blocks for a TIG \"{ctx.challenge}\" solver.\n\n"
              f"{SEARCH_REPLACE_FORMAT}")
    user = (f"These blocks did not match the files exactly once:\n{misses_text}\n\n"
            f"Current files:\n{_files_block(files)}\n\nRe-emit only the failed blocks with "
            f"SEARCH text copied exactly from the files above.")
    return system, user


def distill_prompts(ctx: PromptContext, failed: list[dict]) -> tuple[str, str]:
    system = ("You distill one reusable lesson from failed optimisation attempts. Reply with "
              "exactly one line starting with \"LESSON: \" or the single word NONE.")
    lines = "\n".join(f"{describe_attempt(h, labelled=True, ctx=ctx)}\n"
                      f"  {h.get('description', '')}"
                      for h in failed)
    user = (f"Challenge: {ctx.challenge}. Direction: {ctx.direction}\n\nFailed attempts:\n"
            f"{lines}\n\nWhat general lesson, independent of these exact constants, should "
            f"guide the next attempts?")
    return system, user


def parse_hypothesis(text: str) -> dict:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            d, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "title" in d and "description" in d:
            tag = d.get("strategy_tag")
            return {"title": str(d["title"]), "description": str(d["description"]),
                    "strategy_tag": tag if tag in STRATEGY_TAGS else "hybrid",
                    "design_family":
                        str(d.get("design_family") or "unlabelled").strip()[:80] or "unlabelled"}
    raise ValueError("no hypothesis JSON object found in response")


def parse_distillation(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith("LESSON:"):
            lesson = line.split("LESSON:", 1)[1].strip()
            return lesson or None
    return None
