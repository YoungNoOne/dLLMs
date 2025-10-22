"""
Batch runner that ties together attn_ratio probing and attention-event analysis.

Usage example:
    python batch_attn_analysis.py --config experiments.yaml \
        --model-path /path/to/LLaDA-8B-Instruct \
        --out-dir runs

    python batch_attn_analysis.py \
    --config experiments.yaml \
    --model-path /mnt/users/wuyuyang-20250915/models/LLaDA-8B-Instruct \
    --out-dir runs \
    --device cuda:0
    
The script expects attn_ratio.py and analyze_attention_events.py to be present
in the workspace and reuses their helpers directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import textwrap
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from transformers import AutoModel, AutoTokenizer

from attn_ratio import generate as generate_with_probe
from attn_ratio import visualize_attn_ratio
from analyze_attention_events import DetectionConfig, run_pipeline, normalise_token_str

try:  # Optional dependency: PyYAML for human-readable configs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML missing.
    yaml = None

LOGGER = logging.getLogger("batch_attn_analysis")


@dataclass
class RunConfig:
    """User-provided generation settings."""

    name: str
    prompt: str
    use_chat_template: bool = True
    steps: int = 256
    gen_length: int = 256
    block_length: int = 8
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    save_visuals: bool = True
    extra: Dict[str, object] = field(default_factory=dict)


def _load_config(path: Path) -> List[RunConfig]:
    """Read configuration from YAML or JSON file."""

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs.")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))

    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must define non-empty 'runs' list.")

    seen = set()
    configs: List[RunConfig] = []
    for item in runs:
        if "name" not in item or "prompt" not in item:
            raise ValueError(f"Each run must include 'name' and 'prompt'. Got: {item}")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"Duplicate run name detected: {name}")
        seen.add(name)
        cfg = RunConfig(
            name=name,
            prompt=str(item["prompt"]),
            use_chat_template=bool(item.get("use_chat_template", True)),
            steps=int(item.get("steps", 256)),
            gen_length=int(item.get("gen_length", 256)),
            block_length=int(item.get("block_length", 16)),
            temperature=float(item.get("temperature", 0.0)),
            cfg_scale=float(item.get("cfg_scale", 0.0)),
            remasking=str(item.get("remasking", "low_confidence")),
            mask_id=int(item.get("mask_id", 126336)),
            save_visuals=bool(item.get("save_visuals", False)),
            extra={k: v for k, v in item.items() if k not in {
                "name", "prompt", "use_chat_template", "steps", "gen_length",
                "block_length", "temperature", "cfg_scale", "remasking", "mask_id",
                "save_visuals"}},
        )
        configs.append(cfg)
    return configs


def _prepare_prompt(tokenizer: AutoTokenizer, prompt: str, use_chat_template: bool) -> torch.Tensor:
    """
    Convert raw prompt string into token IDs, optionally using the chat template.
    """

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt_rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt_rendered = prompt
    ids = tokenizer(prompt_rendered)["input_ids"]
    return torch.tensor(ids).unsqueeze(0)


def _load_model(model_path: Path, device: torch.device) -> AutoModel:
    """
    Load the diffusion LM onto the desired device.
    """

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model = model.to(device).eval()
    return model


def _run_single(
    base_dir: Path,
    cfg: RunConfig,
    model,
    tokenizer,
    device: torch.device,
    detection_cfg: DetectionConfig,
    viz_steps_total: Optional[int],
    context_window: int,
    context_categories: Optional[Sequence[str]],
) -> Dict[str, object]:
    """
    Execute one experiment run: generation + analysis.
    """

    run_dir = base_dir / cfg.name
    analysis_dir = run_dir / "analysis"
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running %s", cfg.name)
    prompt_ids = _prepare_prompt(tokenizer, cfg.prompt, cfg.use_chat_template).to(device)

    # Run generation; attn_ratio.generate handles CSV emission.
    ratio_csv = run_dir / "attn_ratio.csv"
    decode_csv = run_dir / "attn_decode_events.csv"
    output_tokens = generate_with_probe(
        model=model,
        prompt=prompt_ids,
        tokenizer=tokenizer,
        steps=cfg.steps,
        gen_length=cfg.gen_length,
        block_length=cfg.block_length,
        temperature=cfg.temperature,
        cfg_scale=cfg.cfg_scale,
        remasking=cfg.remasking,
        mask_id=cfg.mask_id,
        probe=True,
        save_csv=str(ratio_csv),
        save_decode_csv=str(decode_csv),
    )

    decoded = tokenizer.batch_decode(output_tokens[:, prompt_ids.shape[1]:], skip_special_tokens=True)[0]
    LOGGER.debug("Decoded output: %s", decoded)

    # Optional visualisation (writes PNGs beside CSVs).
    if cfg.save_visuals:
        visualize_attn_ratio(
            csv_path=str(ratio_csv),
            steps_total=cfg.steps if viz_steps_total is None else viz_steps_total,
            out_prefix=str(run_dir / "attn"),
            decode_csv=str(decode_csv),
        )

    # Run attention-event analysis.
    outputs = run_pipeline(
        ratio_csv=ratio_csv,
        decode_csv=decode_csv,
        out_dir=analysis_dir,
        cfg=detection_cfg,
        context_window=context_window,
        context_categories=context_categories,
    )

    events = outputs["events"]
    tokens = outputs["tokens"]
    cat_summary = outputs["category_summary"]

    # Load merged triggers to build concise metadata based on final structure.
    triggers_path = analysis_dir / "triggers.json"
    triggers_data: List[Dict[str, object]] = []
    if triggers_path.exists():
        try:
            triggers_data = json.loads(triggers_path.read_text(encoding="utf-8"))
        except Exception:
            triggers_data = []

    category_rows: List[Dict[str, object]] = []
    if not cat_summary.empty:
        category_rows = cat_summary.reset_index().rename(columns={"index": "category"}).to_dict(orient="records")

    # Metadata events summary (lightweight, derived from triggers.json)
    events_summary: List[Dict[str, object]] = []
    total_tokens_from_triggers = 0
    for ev in triggers_data:
        token_list = ev.get("tokens", []) if isinstance(ev, dict) else []
        total_tokens_from_triggers += len(token_list)
        events_summary.append(
            {
                "step_decoded": ev.get("step_decoded"),
                "direction": ev.get("direction"),
                "mean_delta": ev.get("mean_delta"),
                "sign_agreement": ev.get("sign_agreement"),
                "tokens": len(token_list),
            }
        )

    summary = {
        "name": cfg.name,
        "prompt": cfg.prompt,
        "decoded": decoded,
        "events_count": len(events_summary) if triggers_data else len(events),
        "max_mean_delta": max((e.mean_delta for e in events), default=0.0),
        "max_sign_agreement": max((e.sign_agreement for e in events), default=0.0),
        "token_count": total_tokens_from_triggers if triggers_data else len(tokens),
        "categories": category_rows,
        "events": events_summary,  # concise list; tokens per event not inlined to avoid redundancy
        "extra": cfg.extra,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary, tokens


def _choose_device(user_choice: Optional[str]) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch driver for attn_ratio probes and attention-event analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON file describing experiment runs.")
    parser.add_argument("--model-path", type=Path, required=True, help="Local path or HF repo id for the model.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"), help="Directory to store run artefacts.")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device string, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--metric", type=str, default="ratio_last", help="Ratio column for analysis.")
    parser.add_argument("--std-multiplier", type=float, default=2.0, help="Standard deviation multiplier for events.")
    parser.add_argument("--sign-threshold", type=float, default=0.75, help="Sign agreement threshold.")
    parser.add_argument("--min-positions", type=int, default=8, help="Minimum positions per step to consider.")
    parser.add_argument("--min-abs-delta", type=float, default=0.0, help="Absolute delta floor for events.")
    parser.add_argument("--viz-steps-total", type=int, default=None, help="Override x-axis length for visualisations.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--context-window", type=int, default=16, help="Window size for trigger contexts.")
    parser.add_argument(
        "--context-categories",
        type=str,
        default="content_heavy",
        help="Semicolon- or comma-separated categories to include in contexts (e.g. 'content_heavy,other').",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    run_cfgs = _load_config(args.config)
    device = _choose_device(args.device)

    LOGGER.info("Loading model from %s onto %s", args.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = _load_model(args.model_path, device)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_cfg = DetectionConfig(
        metric=args.metric,
        std_multiplier=args.std_multiplier,
        sign_threshold=args.sign_threshold,
        min_positions=args.min_positions,
        min_abs_delta=args.min_abs_delta,
    )

    summaries: List[Dict[str, object]] = []
    aggregate = defaultdict(lambda: {
        "triggers": 0,
        "samples": 0,
        "impact_weighted_sum": 0.0,
        "impact_weight": 0,
        "prompt_shift_weighted_sum": 0.0,
        "prompt_shift_weight": 0,
    })
    token_aggregate = defaultdict(lambda: {
        "token_str": "",
        "token_id": None,
        "token_norm": "",
        "categories": set(),
        "count": 0,
        "runs": set(),
        "occurrences": [],
        "dir_pos": 0,
        "dir_neg": 0,
        "dir_zero": 0,
    })
    category_dir_counts = defaultdict(lambda: {"dir_pos": 0, "dir_neg": 0, "dir_zero": 0})

    # Parse categories
    raw_cats = (args.context_categories or "").replace(";", ",")
    cats: Optional[Sequence[str]] = [c.strip() for c in raw_cats.split(",") if c.strip()] or None

    for cfg in run_cfgs:
        summary, tokens = _run_single(
            base_dir=out_dir,
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            device=device,
            detection_cfg=detection_cfg,
            viz_steps_total=args.viz_steps_total,
            context_window=args.context_window,
            context_categories=cats,
        )
        summaries.append(summary)
        for token in tokens:
            key = (token.token_id, token.token_str)
            entry = token_aggregate[key]
            entry["token_str"] = token.token_str
            entry["token_id"] = token.token_id
            entry["token_norm"] = normalise_token_str(token.token_str)
            entry["categories"].update(token.categories)
            entry["count"] += 1
            entry["runs"].add(summary["name"])
            entry["occurrences"].append({
                "run": summary["name"],
                "step_decoded": int(token.step_decoded),
            })
            if token.direction > 0:
                entry["dir_pos"] += 1
            elif token.direction < 0:
                entry["dir_neg"] += 1
            else:
                entry["dir_zero"] += 1
            # Category-level direction counts
            for cat in token.categories:
                c = category_dir_counts[cat]
                if token.direction > 0:
                    c["dir_pos"] += 1
                elif token.direction < 0:
                    c["dir_neg"] += 1
                else:
                    c["dir_zero"] += 1
        for cat in summary.get("categories", []):
            category = str(cat["category"])
            triggers = int(cat.get("triggers", 0))
            samples = int(cat.get("samples", 0))
            mean_impact = float(cat.get("mean_impact", 0.0))
            mean_prompt_shift = float(cat.get("mean_prompt_shift", 0.0))
            aggregate_entry = aggregate[category]
            aggregate_entry["triggers"] += triggers
            aggregate_entry["samples"] += samples
            aggregate_entry["impact_weighted_sum"] += mean_impact * max(triggers, 1)
            aggregate_entry["impact_weight"] += max(triggers, 1)
            aggregate_entry["prompt_shift_weighted_sum"] += mean_prompt_shift * max(triggers, 1)
            aggregate_entry["prompt_shift_weight"] += max(triggers, 1)

    aggregate_rows: List[Dict[str, object]] = []
    for category, metrics in aggregate.items():
        impact_weight = metrics["impact_weight"] or 1
        prompt_weight = metrics["prompt_shift_weight"] or 1
        dirs = category_dir_counts.get(category, {"dir_pos": 0, "dir_neg": 0, "dir_zero": 0})
        aggregate_rows.append(
            {
                "category": category,
                "triggers": metrics["triggers"],
                "samples": metrics["samples"],
                "mean_impact": metrics["impact_weighted_sum"] / impact_weight,
                "mean_prompt_shift": metrics["prompt_shift_weighted_sum"] / prompt_weight,
                "dir_pos": dirs["dir_pos"],
                "dir_neg": dirs["dir_neg"],
                "dir_zero": dirs["dir_zero"],
            }
        )

    aggregate_rows_sorted = sorted(aggregate_rows, key=lambda x: x["triggers"], reverse=True)
    aggregate_path = out_dir / "category_aggregate.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate_rows_sorted, f, indent=2)

    if aggregate_rows_sorted:
        csv_path = out_dir / "category_aggregate.csv"
        headers = ["category", "triggers", "samples", "mean_impact", "mean_prompt_shift", "dir_pos", "dir_neg", "dir_zero"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in aggregate_rows_sorted:
                writer.writerow(
                    [
                        row["category"],
                        row["triggers"],
                        row["samples"],
                        f"{row['mean_impact']:.6f}",
                        f"{row['mean_prompt_shift']:.6f}",
                        row["dir_pos"],
                        row["dir_neg"],
                        row["dir_zero"],
                    ]
                )

    token_rows_sorted: List[Dict[str, object]] = []
    for entry in token_aggregate.values():
        token_rows_sorted.append(
            {
                "token_id": entry["token_id"],
                "token_str": entry["token_str"],
                "token_norm": entry["token_norm"],
                "count": entry["count"],
                "runs": sorted(entry["runs"]),
                "occurrences": entry["occurrences"],
                "categories": sorted(entry["categories"]),
            }
        )
    token_rows_sorted.sort(key=lambda x: x["count"], reverse=True)

    token_json_path = out_dir / "trigger_token_aggregate.json"
    with token_json_path.open("w", encoding="utf-8") as f:
        json.dump(token_rows_sorted, f, indent=2)

    if token_rows_sorted:
        token_csv_path = out_dir / "trigger_token_aggregate.csv"
        with token_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["token_id", "token_str", "token_norm", "count", "unique_runs", "dir_pos", "dir_neg", "dir_zero", "categories"])
            for row in token_rows_sorted:
                categories = ";".join(row["categories"])
                writer.writerow(
                    [
                        row["token_id"],
                        row["token_str"],
                        row["token_norm"],
                        row["count"],
                        len(row["runs"]),
                        row.get("dir_pos", 0),
                        row.get("dir_neg", 0),
                        row.get("dir_zero", 0),
                        categories,
                    ]
                )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    LOGGER.info("Completed %d runs. Summary saved to %s", len(summaries), summary_path)

    # Aggregate trigger contexts across runs
    import csv as _csv
    context_rows: List[Dict[str, object]] = []
    for cfg in run_cfgs:
        ctx_csv = out_dir / cfg.name / "analysis" / "trigger_contexts.csv"
        if not ctx_csv.exists():
            continue
        with ctx_csv.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                row_copy = dict(row)
                row_copy["run"] = cfg.name
                row_copy["name"] = cfg.name
                context_rows.append(row_copy)

    if context_rows:
        # Write global CSV
        global_csv = out_dir / "trigger_contexts.csv"
        fieldnames = [
            "run",
            "name",
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
        with global_csv.open("w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in context_rows:
                # Ensure all fields exist
                out_row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(out_row)

        # Write global TXT
        global_txt = out_dir / "trigger_contexts.txt"
        with global_txt.open("w", encoding="utf-8") as f:
            for r in context_rows:
                dir_sym = "+" if int(r.get("direction", 0)) >= 0 else "-"
                cats = r.get("categories", "")
                f.write(
                    f"run={r.get('run')} step_decoded={r.get('step_decoded')} pos={r.get('token_pos')} dir={dir_sym} "
                    f"impact={r.get('impact_score')} shift={r.get('prompt_shift')} cats={cats} "
                    f"earlier={r.get('earlier_count')} "
                    f"later={r.get('later_count')} same={r.get('same_count')}\n"
                )
                left = r.get("left_text_annot", r.get("left_text", ""))
                center = r.get("center_annot", r.get("center_token", ""))
                right = r.get("right_text_annot", r.get("right_text", ""))
                f.write(f"{left} [{center}] {right}\n\n")


if __name__ == "__main__":  # pragma: no cover
    main()
