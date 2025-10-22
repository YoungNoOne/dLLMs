import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd


@dataclass
class DetectionConfig:
    """
    Tunable thresholds for spotting global attention perturbations.

    Attributes:
        metric: Which ratio column to analyse (e.g. 'ratio_last').
        std_multiplier: Number of standard deviations above the mean
            required for the global delta statistic to qualify as an event.
        sign_threshold: Fraction of positions that must share the same
            sign after the delta step.
        min_positions: Require at least this many valid positions to
            avoid reacting to sparse slices (e.g. near sequence end).
        min_abs_delta: Optional guard-rail that enforces an absolute
            mean delta threshold in addition to the z-score rule.
    """

    metric: str = "ratio_last"
    std_multiplier: float = 2.0
    sign_threshold: float = 0.75
    min_positions: int = 4
    min_abs_delta: float = 0.0


@dataclass
class TriggerEvent:
    """
    Container describing an attention perturbation candidate.
    """

    step: int
    direction: int
    mean_delta: float
    median_delta: float
    sign_agreement: float
    valid_positions: int
    prompt_mean: float


@dataclass
class TriggerToken:
    """
    Record that ties a trigger event to a concrete decoded token.
    """

    step: int
    direction: int
    impact_score: float
    prompt_shift: float
    token_pos: int
    token_id: int
    token_str: str
    step_decoded: int
    conf_decoded: float
    categories: Sequence[str] = field(default_factory=list)


# -----------------------
# Context reconstruction
# -----------------------

SENTENCE_PUNCT = {".", "!", "?", "。", "！", "？"}
CLAUSE_PUNCT = {",", ";", ":", "—", "–", "-", "…", "...", "—", "—", "—"}
BRACKETS = {"(", ")", "[", "]", "{", "}", '"', "'", "“", "”", "‘", "’"}
SPECIAL_TOKENS = {"</s>", "<s>", "<eos>", "<EOS>", "<0x0A>", "\n", "<|endoftext|>", "<|eot_id|>", "Ċ", "ċ"}
OTHER_PUNCT = {"/", "\\", "|"}


def load_ratio_table(path: Path) -> pd.DataFrame:
    """
    Parse the attn_ratio.csv artefact and ensure expected columns exist.
    """

    df = pd.read_csv(path)
    required = {"step", "pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing mandatory columns: {sorted(missing)}")
    ratio_cols = [c for c in df.columns if c.startswith("ratio")]
    if not ratio_cols:
        raise ValueError("No ratio_* columns present; did attn_ratio.py run with probe=True?")
    df["step"] = df["step"].astype(int)
    df["pos"] = df["pos"].astype(int)
    return df


def load_decode_events(path: Path) -> pd.DataFrame:
    """
    Load decode event logs; tolerate absence by returning an empty table.
    """

    if not path.exists():
        return pd.DataFrame(columns=["pos", "step_decoded", "conf_decoded", "token_id", "token_str"])
    df = pd.read_csv(path)
    for col in ["pos", "step_decoded", "token_id"]:
        df[col] = df[col].astype(int)
    if "conf_decoded" in df.columns:
        df["conf_decoded"] = df["conf_decoded"].astype(float)
    else:
        df["conf_decoded"] = np.nan
    return df


def build_ratio_matrices(
    df: pd.DataFrame,
    metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert long-form ratios to (step x pos) matrices and their deltas.
    """

    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not found. Available: {sorted(c for c in df.columns if c.startswith('ratio_'))}")
    ratio_wide = df.pivot_table(index="step", columns="pos", values=metric, aggfunc="mean").sort_index()
    ratio_diff = ratio_wide.diff()
    return ratio_wide, ratio_diff


def _majority_sign(series: pd.Series) -> float:
    """
    Fraction of finite values that share the prevailing sign.
    """

    values = series.values
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return 0.0
    signs = np.sign(values)
    pos_share = (signs > 0).sum() / values.size
    neg_share = (signs < 0).sum() / values.size
    return float(max(pos_share, neg_share))


def compute_step_statistics(ratio_wide: pd.DataFrame, ratio_diff: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-step metrics used by the detector.
    """

    stats = pd.DataFrame(index=ratio_wide.index)
    stats["valid_positions"] = ratio_wide.notna().sum(axis=1)
    stats["mean_delta"] = ratio_diff.abs().mean(axis=1, skipna=True)
    stats["median_delta"] = ratio_diff.abs().median(axis=1, skipna=True)
    stats["sign_agreement"] = ratio_diff.apply(_majority_sign, axis=1)
    stats["prompt_mean"] = ratio_wide.mean(axis=1, skipna=True)
    stats["signed_mean_delta"] = ratio_diff.mean(axis=1, skipna=True)
    return stats


def detect_triggers(stats: pd.DataFrame, cfg: DetectionConfig) -> List[TriggerEvent]:
    """
    Flag steps whose deltas satisfy the configured criteria.
    """

    baseline = stats[stats["valid_positions"] >= cfg.min_positions]
    if baseline.empty:
        return []
    delta_mean = baseline["mean_delta"]
    mu = delta_mean.mean()
    sigma = delta_mean.std()
    threshold = mu + cfg.std_multiplier * sigma if sigma > 0 else mu
    threshold = max(threshold, cfg.min_abs_delta)
    events: List[TriggerEvent] = []

    for step, row in baseline.iterrows():
        if row["mean_delta"] < threshold:
            continue
        if row["sign_agreement"] < cfg.sign_threshold:
            continue
        direction = int(math.copysign(1, row["signed_mean_delta"])) if row["signed_mean_delta"] != 0 else 0
        events.append(
            TriggerEvent(
                step=int(step),
                direction=direction,
                mean_delta=float(row["mean_delta"]),
                median_delta=float(row["median_delta"]),
                sign_agreement=float(row["sign_agreement"]),
                valid_positions=int(row["valid_positions"]),
                prompt_mean=float(row["prompt_mean"]),
            )
        )
    return events


def _token_to_text_piece(tok: str) -> str:
    """
    Convert a single tokenizer piece into display text, preserving common
    whitespace markers seen in BPE-like tokenizers (e.g. 'Ġ', '▁', 'Ċ').
    """
    if tok in {"Ċ", "<0x0A>", "\n"}:
        return "\n"
    # Preserve leading-space markers as actual spaces
    if tok.startswith("Ġ") or tok.startswith("▁"):
        return " " + tok[1:]
    return tok


def detokenise(tokens: Sequence[str]) -> str:
    """
    Lightweight detokeniser that converts a list of token strings into
    a readable snippet. Heuristics only; avoids external dependencies.
    """
    def _classify_base(tok: str) -> str:
        # Remove leading space markers and trailing annotation like ^(+3)
        if tok.startswith("Ġ") or tok.startswith("▁"):
            tok = tok[1:]
        if "^(" in tok and tok.endswith(")"):
            tok = tok.split("^(", 1)[0]
        return tok

    out: List[str] = []
    for tok in tokens:
        piece = _token_to_text_piece(tok)
        if not out:
            out.append(piece)
            continue
        # If current piece starts with a space or newline, just append.
        if piece.startswith(" ") or piece.startswith("\n"):
            out.append(piece)
            continue
        # No extra space before closing punctuation; otherwise append directly.
        base_tok = _classify_base(tok)
        if base_tok in SENTENCE_PUNCT or base_tok in CLAUSE_PUNCT or base_tok in BRACKETS or base_tok in OTHER_PUNCT:
            out.append(piece)
        else:
            # Default: add a space between words if previous char is alnum.
            prev = out[-1]
            if prev and prev[-1].isalnum():
                out.append(" ")
            out.append(piece)
    return "".join(out)


def _format_offset(val: int) -> str:
    return f"{val:+d}"


def annotate_tokens_for_display(tokens: Sequence[str], offsets: Sequence[int]) -> str:
    """
    Render tokens with relative step offsets inline, e.g. token^(+2).
    Uses the same spacing heuristics as detokenise by injecting the
    annotation into the token piece before spacing decisions.
    """
    assert len(tokens) == len(offsets)
    annotated: List[str] = []
    for tok, off in zip(tokens, offsets):
        annotated.append(f"{tok}^({_format_offset(int(off))})")
    # Reuse detokenise to place spaces/newlines
    return detokenise(annotated)

DISCOURSE_MARKERS: Dict[str, set] = {
    "contrast_marker": {
        "but",
        "however",
        "though",
        "although",
        "yet",
        "nevertheless",
        "nonetheless",
        "still",
        "whereas",
        "instead",
        "conversely",
    },
    "cause_marker": {
        "because",
        "since",
        "as",
        "therefore",
        "thus",
        "hence",
        "thereby",
        "so",
        "consequently",
        "accordingly",
    },
    "addition_marker": {
        "and",
        "also",
        "furthermore",
        "moreover",
        "besides",
        "additionally",
        "plus",
        "in_addition",
    },
    "temporal_marker": {
        "then",
        "after",
        "afterward",
        "afterwards",
        "before",
        "meanwhile",
        "later",
        "eventually",
        "finally",
        "subsequently",
        "earlier",
    },
    "condition_marker": {
        "if",
        "unless",
        "provided",
        "whenever",
        "once",
        "in_case",
    },
    "summary_marker": {
        "overall",
        "in_sum",
        "in_summary",
        "to_sum_up",
        "therefore",
        "thus",
        "ultimately",
    },
}

FILLER_MARKERS = {
    "well",
    "um",
    "uh",
    "like",
    "maybe",
    "perhaps",
}

FUNCTION_WORDS = {
    # Articles / determiners
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    # Pronouns
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "hers",
    "our",
    "their",
    "mine",
    "yours",
    "ours",
    "theirs",
    "oneself",
    # Auxiliary / modal / be verbs
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    # Conjunctions / particles
    "and",
    "or",
    "nor",
    "yet",
    "so",
    "for",
    "but",
    "if",
    "while",
    "as",
    "than",
    "that",
    "because",
    "when",
    "where",
    "after",
    "before",
    "until",
    "though",
    "although",
    "since",
    "unless",
    # Adverbs / quantifiers commonly treated as function words
    "very",
    "too",
    "also",
    "either",
    "neither",
    "not",
    "only",
    "just",
    "even",
    "ever",
    "never",
    "always",
    "often",
    "usually",
    "sometimes",
    "here",
    "there",
    "then",
    "now",
    "again",
    "maybe",
}


def normalise_token_str(token: str) -> str:
    """
    Canonicalise token strings for matching; keeping simple heuristics.
    """

    token = token.strip()
    # Hugging Face tokenizers often prepend a space; record stripped variant for comparisons.
    if token.startswith("▁"):
        token = token[1:]
    if token.startswith("Ġ"):
        token = token[1:]
    return token.strip()


def categorise_token(token: str) -> List[str]:
    """
    Assign broad semantic categories used in the analysis.
    """

    token_norm = normalise_token_str(token).lower()
    categories: List[str] = []

    nonlexical = False

    if not token_norm:
        categories.append("empty")
    if token_norm in SENTENCE_PUNCT:
        categories.append("punctuation")
        categories.append("sentence_boundary")
        nonlexical = True
    if token_norm in CLAUSE_PUNCT:
        categories.append("punctuation")
        categories.append("clause_boundary")
        nonlexical = True
    if token_norm in BRACKETS:
        categories.append("punctuation")
        categories.append("bracket")
        nonlexical = True
    if token_norm in SPECIAL_TOKENS:
        categories.append("special_token")
        nonlexical = True
    if token_norm in OTHER_PUNCT:
        categories.append("punctuation")
        nonlexical = True

    marker_key = token_norm.replace(" ", "_")
    for cat_name, vocab in DISCOURSE_MARKERS.items():
        if token_norm in vocab or marker_key in vocab:
            categories.append(cat_name)
    if token_norm in FILLER_MARKERS:
        categories.append("filler_marker")

    if (not nonlexical) and token_norm.isalpha() and token_norm not in FUNCTION_WORDS:
        categories.append("content_heavy")
    if not categories:
        categories.append("other")
    # Deduplicate while maintaining order.
    seen = set()
    deduped: List[str] = []
    for cat in categories:
        if cat not in seen:
            deduped.append(cat)
            seen.add(cat)
    return deduped


def build_token_map(decode_df: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    """
    Map absolute position -> {token_str, token_id, step_decoded, conf_decoded} using
    the decode-events CSV (one row per position decoded). Assumes positions are
    unique (first decode fixes the value for the rest of the process).
    """
    token_map: Dict[int, Dict[str, object]] = {}
    for _, row in decode_df.iterrows():
        pos = int(row["pos"])  # type: ignore[arg-type]
        # Prefer the earliest decode if duplicates exist; keep first occurrence.
        if pos not in token_map:
            token_map[pos] = {
                "token_str": str(row.get("token_str", "")),
                "token_id": int(row.get("token_id", -1)),
                "step_decoded": int(row.get("step_decoded", -1)),
                "conf_decoded": float(row.get("conf_decoded", float("nan"))),
            }
    return token_map


def build_trigger_contexts(
    tokens: Sequence["TriggerToken"],
    decode_df: pd.DataFrame,
    window: int = 6,
    include_categories: Optional[Sequence[str]] = ("content_heavy",),
) -> List[Dict[str, object]]:
    """
    Build windowed contexts around each trigger token using the final decoded
    sequence (no step gating). Returns a list of row dicts for CSV/TXT export.
    """
    if not tokens or decode_df.empty:
        return []
    token_map = build_token_map(decode_df)
    positions_sorted = sorted(token_map.keys())
    pos_min = positions_sorted[0] if positions_sorted else 0
    pos_max = positions_sorted[-1] if positions_sorted else -1

    rows: List[Dict[str, object]] = []
    for t in tokens:
        # Category filter
        if include_categories:
            if not any(cat in include_categories for cat in t.categories):
                continue
        center = int(t.token_pos)
        # Clamp to available decoded positions (prompt positions are absent)
        left_positions = [p for p in range(center - window, center) if p in token_map]
        right_positions = [p for p in range(center + 1, center + 1 + window) if p in token_map]
        left_tokens = [str(token_map[p]["token_str"]) for p in left_positions]
        right_tokens = [str(token_map[p]["token_str"]) for p in right_positions]

        # Timing (step) information
        center_step = int(token_map.get(center, {}).get("step_decoded", -1))
        left_steps = [int(token_map[p]["step_decoded"]) for p in left_positions]
        right_steps = [int(token_map[p]["step_decoded"]) for p in right_positions]
        left_offsets = [s - center_step for s in left_steps]
        right_offsets = [s - center_step for s in right_steps]
        earlier_count = sum(1 for d in left_offsets + right_offsets if d < 0)
        later_count = sum(1 for d in left_offsets + right_offsets if d > 0)
        same_count = sum(1 for d in left_offsets + right_offsets if d == 0)

        left_text = detokenise(left_tokens)
        right_text = detokenise(right_tokens)
        left_text_annot = annotate_tokens_for_display(left_tokens, left_offsets) if left_tokens else ""
        right_text_annot = annotate_tokens_for_display(right_tokens, right_offsets) if right_tokens else ""
        center_annot = f"{t.token_str}^({_format_offset(0)})"

        row = {
            # Use unified naming: only expose generation step in outputs.
            "step_decoded": center_step,
            "token_pos": t.token_pos,
            "token_id": t.token_id,
            "token_str": t.token_str,
            "token_norm": normalise_token_str(t.token_str),
            "direction": t.direction,
            "impact_score": t.impact_score,
            "prompt_shift": t.prompt_shift,
            "window": int(window),
            "categories": ";".join(t.categories),
            "left_text": left_text,
            "center_token": t.token_str,
            "right_text": right_text,
            "center_step_decoded": center_step,
            "earlier_count": earlier_count,
            "later_count": later_count,
            "same_count": same_count,
            "left_steps_json": json.dumps(left_steps),
            "right_steps_json": json.dumps(right_steps),
            "left_offsets_json": json.dumps(left_offsets),
            "right_offsets_json": json.dumps(right_offsets),
            "left_text_annot": left_text_annot,
            "center_annot": center_annot,
            "right_text_annot": right_text_annot,
            "left_tokens_json": json.dumps(left_tokens, ensure_ascii=False),
            "right_tokens_json": json.dumps(right_tokens, ensure_ascii=False),
        }
        rows.append(row)
    return rows


def attach_tokens_to_events(
    events: Sequence[TriggerEvent],
    ratio_wide: pd.DataFrame,
    ratio_diff: pd.DataFrame,
    decode_df: pd.DataFrame,
) -> List[TriggerToken]:
    """
    Expand trigger events to the token level using decode metadata.
    """

    if decode_df.empty:
        return []

    ratio_mean = ratio_wide.mean(axis=1, skipna=True)
    ratio_prev = ratio_mean.shift(1)
    tokens: List[TriggerToken] = []

    for event in events:
        # Primary alignment: tokens decoded in the previous step (attention measured before the new update).
        prev_step = event.step - 1
        candidates = decode_df[decode_df["step_decoded"] == prev_step] if prev_step >= 0 else pd.DataFrame()
        # Fallback: if no records exist (e.g. event happens at step 0 or logs missing), look at the same step.
        if candidates.empty:
            candidates = decode_df[decode_df["step_decoded"] == event.step]
        if candidates.empty:
            continue
        prompt_shift = float(ratio_mean.get(event.step, np.nan) - ratio_prev.get(event.step, np.nan))
        for _, row in candidates.iterrows():
            categories = categorise_token(row.get("token_str", ""))
            tokens.append(
                TriggerToken(
                    step=event.step,
                    direction=event.direction,
                    impact_score=event.mean_delta,
                    prompt_shift=prompt_shift,
                    token_pos=int(row["pos"]),
                    token_id=int(row["token_id"]),
                    token_str=str(row.get("token_str", "")),
                    step_decoded=int(row["step_decoded"]),
                    conf_decoded=float(row.get("conf_decoded", np.nan)),
                    categories=categories,
                )
            )
    return tokens


def summarise_by_category(tokens: Sequence[TriggerToken]) -> pd.DataFrame:
    """
    Produce per-category aggregates for reporting.
    """

    if not tokens:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for token in tokens:
        for cat in token.categories:
            rows.append(
                {
                    "category": cat,
                    "step": token.step,
                    "impact_score": token.impact_score,
                    "prompt_shift": token.prompt_shift,
                    "direction": token.direction,
                    "token_str": token.token_str,
                }
            )
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("category")
        .agg(
            triggers=("step", "nunique"),
            samples=("step", "size"),
            mean_impact=("impact_score", "mean"),
            median_impact=("impact_score", "median"),
            mean_prompt_shift=("prompt_shift", "mean"),
        )
        .sort_values("mean_impact", ascending=False)
    )
    return grouped


def make_background_samples(
    ratio_stats: pd.DataFrame,
    events: Sequence[TriggerEvent],
    rng: np.random.Generator,
    repeats: int = 200,
) -> pd.DataFrame:
    """
    Draw bootstrap samples of non-event steps to quantify background behaviour.
    """

    event_steps = {e.step for e in events}
    candidate_steps = [s for s in ratio_stats.index if s not in event_steps]
    if not candidate_steps or not events:
        return pd.DataFrame()

    sample_size = min(len(events), len(candidate_steps))
    records: List[Dict[str, float]] = []

    for _ in range(repeats):
        selection = rng.choice(candidate_steps, size=sample_size, replace=False)
        slice_stats = ratio_stats.loc[selection]
        records.append(
            {
                "mean_delta": slice_stats["mean_delta"].mean(),
                "median_delta": slice_stats["median_delta"].mean(),
                "sign_agreement": slice_stats["sign_agreement"].mean(),
                "prompt_mean": slice_stats["prompt_mean"].mean(),
            }
        )
    return pd.DataFrame(records)


def export_results(
    out_dir: Path,
    events: Sequence[TriggerEvent],
    tokens: Sequence[TriggerToken],
    category_summary: pd.DataFrame,
    background_stats: pd.DataFrame,
    contexts: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    """
    Persist intermediate artefacts for downstream notebooks or plotting.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "triggers.json"
    categories_path = out_dir / "category_summary.csv"
    background_path = out_dir / "background_stats.csv"

    # Skip standalone trigger_events.json and trigger_tokens.json; only write merged file.

    # Write merged triggers file with duplicates removed at token level.
    # Structure: list of events with nested tokens; per-token fields drop
    # those that are event-constant (direction, impact_score, prompt_shift).
    if events:
        by_step = {int(e.step): e for e in events}
        tokens_by_step: Dict[int, List[Dict[str, object]]] = {}
        # Build event-level prompt_shift from the first token if available.
        event_prompt_shift: Dict[int, float] = {}
        for t in tokens:
            step = int(t.step)
            tokens_by_step.setdefault(step, []).append(
                {
                    "step_decoded": int(t.step_decoded),
                    "token_pos": int(t.token_pos),
                    "token_id": int(t.token_id),
                    "token_str": str(t.token_str),
                    "conf_decoded": float(t.conf_decoded),
                    "categories": list(t.categories),
                }
            )
            if step not in event_prompt_shift:
                try:
                    event_prompt_shift[step] = float(t.prompt_shift)
                except Exception:
                    pass

        merged_records: List[Dict[str, object]] = []
        for step, e in sorted(by_step.items(), key=lambda kv: kv[0]):
            gen_step = max(0, int(e.step) - 1)
            rec = {
                "step_decoded": gen_step,
                "direction": int(e.direction),
                "mean_delta": float(e.mean_delta),
                "median_delta": float(e.median_delta),
                "sign_agreement": float(e.sign_agreement),
                "valid_positions": int(e.valid_positions),
                "prompt_mean": float(e.prompt_mean),
            }
            if step in event_prompt_shift:
                rec["prompt_shift"] = event_prompt_shift[step]
            rec["tokens"] = tokens_by_step.get(step, [])
            merged_records.append(rec)

        with merged_path.open("w", encoding="utf-8") as f:
            json.dump(merged_records, f, indent=2)

    if not category_summary.empty:
        category_summary.to_csv(categories_path)

    if not background_stats.empty:
        background_stats.to_csv(background_path, index=False)

    # Optional contexts export
    if contexts:
        ctx_csv = out_dir / "trigger_contexts.csv"
        ctx_txt = out_dir / "trigger_contexts.txt"

        # CSV
        import csv as _csv
        headers = [
            "step_decoded",
            "token_pos",
            "token_id",
            "token_str",
            "token_norm",
            "direction",
            "impact_score",
            "prompt_shift",
            "window",
            "categories",
            "left_text",
            "center_token",
            "right_text",
            "center_step_decoded",
            "earlier_count",
            "later_count",
            "same_count",
            "left_steps_json",
            "right_steps_json",
            "left_offsets_json",
            "right_offsets_json",
            "left_text_annot",
            "center_annot",
            "right_text_annot",
            "left_tokens_json",
            "right_tokens_json",
        ]
        with ctx_csv.open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row in contexts:
                w.writerow({k: row.get(k, "") for k in headers})

        # TXT (readable)
        with ctx_txt.open("w", encoding="utf-8") as f:
            for row in contexts:
                cats = row.get("categories", "")
                dir_sym = "+" if int(row.get("direction", 0)) >= 0 else "-"
                f.write(
                    f"step_decoded={row.get('step_decoded')} pos={row.get('token_pos')} dir={dir_sym} "
                    f"impact={row.get('impact_score'):.6f} shift={row.get('prompt_shift'):.6f} cats={cats} "
                    f"earlier={row.get('earlier_count')} "
                    f"later={row.get('later_count')} same={row.get('same_count')}\n"
                )
                left = str(row.get("left_text_annot", row.get("left_text", "")))
                center = str(row.get("center_annot", row.get("center_token", "")))
                right = str(row.get("right_text_annot", row.get("right_text", "")))
                f.write(f"{left} [{center}] {right}\n\n")


def run_pipeline(
    ratio_csv: Path,
    decode_csv: Path,
    out_dir: Path,
    cfg: DetectionConfig,
    rng_seed: int = 2025,
    context_window: int = 6,
    context_categories: Optional[Sequence[str]] = ("content_heavy",),
    enable_background: bool = False,
) -> Dict[str, object]:
    """
    Execute the full analysis pipeline and return key artefacts.
    """

    ratio_df = load_ratio_table(ratio_csv)
    decode_df = load_decode_events(decode_csv)
    ratio_wide, ratio_diff = build_ratio_matrices(ratio_df, cfg.metric)
    stats = compute_step_statistics(ratio_wide, ratio_diff)
    events = detect_triggers(stats, cfg)
    tokens = attach_tokens_to_events(events, ratio_wide, ratio_diff, decode_df)
    cat_summary = summarise_by_category(tokens)
    background = (
        make_background_samples(stats, events, np.random.default_rng(rng_seed))
        if enable_background
        else pd.DataFrame()
    )
    contexts = build_trigger_contexts(tokens, decode_df, window=context_window, include_categories=context_categories)
    export_results(out_dir, events, tokens, cat_summary, background, contexts=contexts)
    return {
        "events": events,
        "tokens": tokens,
        "category_summary": cat_summary,
        "background_stats": background,
        "step_stats": stats,
        "contexts": contexts,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Detect and analyse attention-ratio perturbation events.")
    parser.add_argument("--ratio-csv", type=Path, default=Path("attn_ratio.csv"), help="Path to attn_ratio.csv.")
    parser.add_argument("--decode-csv", type=Path, default=Path("attn_decode_events.csv"), help="Path to decode events CSV.")
    parser.add_argument("--out-dir", type=Path, default=Path("attn_analysis"), help="Directory to store analysis artefacts.")
    parser.add_argument("--metric", type=str, default="ratio_last", help="Which ratio column to analyse.")
    parser.add_argument("--std-multiplier", type=float, default=2.0, help="Standard deviation multiplier for event detection.")
    parser.add_argument("--sign-threshold", type=float, default=0.75, help="Required majority sign agreement.")
    parser.add_argument("--min-positions", type=int, default=16, help="Minimum valid positions per step.")
    parser.add_argument("--min-abs-delta", type=float, default=0.0, help="Absolute mean delta floor for events.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for background sampling.")
    parser.add_argument("--context-window", type=int, default=16, help="Window size for context extraction around triggers.")
    parser.add_argument(
        "--context-categories",
        type=str,
        default="content_heavy",
        help="Semicolon- or comma-separated category names to include (e.g. 'content_heavy,other').",
    )
    parser.add_argument(
        "--enable-background",
        action="store_true",
        help="Enable bootstrap sampling of background (non-event) steps.",
    )
    args = parser.parse_args(argv)

    cfg = DetectionConfig(
        metric=args.metric,
        std_multiplier=args.std_multiplier,
        sign_threshold=args.sign_threshold,
        min_positions=args.min_positions,
        min_abs_delta=args.min_abs_delta,
    )

    # Parse categories list
    raw_cats = (args.context_categories or "").replace(";", ",")
    cats = tuple(c.strip() for c in raw_cats.split(",") if c.strip()) or None

    outputs = run_pipeline(
        args.ratio_csv,
        args.decode_csv,
        args.out_dir,
        cfg,
        rng_seed=args.seed,
        context_window=args.context_window,
        context_categories=cats,
        enable_background=bool(args.enable_background),
    )
    events = outputs["events"]
    tokens = outputs["tokens"]
    cat_summary = outputs["category_summary"]

    print(f"Detected {len(events)} trigger steps.")
    if events:
        print("All events by mean_delta:")
        for event in sorted(events, key=lambda e: e.mean_delta, reverse=True):
            gen_step = max(0, event.step - 1)
            print(
                f"  step_decoded={gen_step:3d} mean_delta={event.mean_delta:.4f} "
                f"direction={'+' if event.direction >= 0 else '-'} "
                f"sign_agreement={event.sign_agreement:.2f} valid_positions={event.valid_positions}"
            )
    print(f"Annotated {len(tokens)} tokens linked to trigger steps.")
    if not cat_summary.empty:
        print("Category summary:")
        print(cat_summary.head())
    ctx_count = len(outputs.get("contexts", []) or [])
    print(f"Exported {ctx_count} trigger contexts.")


if __name__ == "__main__":
    main()
