import json
import csv
import math

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境下可保存图片
import matplotlib.pyplot as plt
import numpy as np


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, prompt, tokenizer, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, probe=True, save_csv="attn_ratio.csv", save_decode_csv="attn_decode_events.csv"):
    """
    说明：
    - CSV1: 每一步“仍为 mask 的位置”的观测：step,pos,ratio_mean,ratio_last4,ratio_last,conf
    - CSV2: 每个位置“被解码的时刻”的单行记录：pos,step_decoded,conf_decoded,token_id,token_str
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    records = []  # -> attn_ratio.csv
    decode_events = []  # -> attn_decode_events.csv

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)

            # ---- CHANGED: always compute with output_attentions ----
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                out = model(x_, output_attentions=True)
                logits_all = out.logits
                logits, un_logits = torch.chunk(logits_all, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                # attentions: take conditional (first batch slice)
                attentions = [A[:1] for A in out.attentions]  # list of (1, nH, T, S)
            else:
                out = model(x, output_attentions=True)
                logits = out.logits
                attentions = out.attentions  # list of (1, nH, T, S)

            # --- original decode/confidence logic (kept) ---
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # --- NEW: compute attention ratio & Δ_i^P for current masked positions ---
            if probe:
                b0 = 0
                T = x.shape[1]
                # key sets
                key_prompt = prompt_index[b0]                            # boolean (T,)
                key_ctx    = (~mask_index[b0]) & (~prompt_index[b0])     # 已解出的非prompt
                key_valid  = key_prompt | key_ctx                        # 分母只看这些
                # masked query indices:
                q_idx = torch.nonzero(mask_index[b0], as_tuple=False).squeeze(-1).tolist()

                # per-layer averaged-over-head ratio for all masked queries
                layer_ratios = []
                for A in attentions:  # A: (1, nH, T, S)
                    Ah = A[b0]        # (nH, T, S)
                    # sum over keys in prompt / valid, keep queries only at masked positions
                    # shapes: (nH, |Q|)
                    prompt_mass = Ah[:, q_idx][:, :, key_prompt].sum(dim=-1)
                    valid_mass  = Ah[:, q_idx][:, :, key_valid ].sum(dim=-1) + 1e-8
                    ratio = (prompt_mass / valid_mass).mean(dim=0)  # mean over heads -> (|Q|,)
                    layer_ratios.append(ratio)                      # list of (|Q|,)

                ratios = torch.stack(layer_ratios, dim=0)  # (L_layers, |Q|)
                ratio_mean   = ratios.mean(dim=0)                          # (|Q|,)
                ratio_last4  = ratios[-4:].mean(dim=0) if ratios.size(0) >= 4 else ratio_mean
                ratio_last = ratios[-1]

                for j, pos in enumerate(q_idx):
                    records.append({
                        "step": num_block * steps + i,
                        "pos": int(pos),
                        "ratio_mean": float(ratio_mean[j].item()),
                        "ratio_last4": float(ratio_last4[j].item()),
                        "ratio_last": float(ratio_last[j].item()),
                        "conf": float(confidence[b0, pos].item())
                    })

            # --- original transfer (kept) ---
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for b in range(confidence.shape[0]):  # 通常就是 1 个 batch
                k = int(num_transfer_tokens[b, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[b], k=k)
                    transfer_index[b, select_index] = True
                    # 记录这些位置在“当前 step 被解码”
                    for pos in select_index.tolist():
                        tok_id = int(x0[b, pos].item())
                        # 更稳妥的 token 文本：prefer convert_ids_to_tokens，其次 decode 单 token
                        try:
                            tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
                        except Exception:
                            tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
                        decode_events.append({
                            "pos": int(pos),
                            "step_decoded": int(num_block * steps + i),
                            "conf_decoded": float(confidence[b, pos].item()),
                            "token_id": tok_id,
                            "token_str": tok_str
                        })
            x[transfer_index] = x0[transfer_index]

    # --- NEW: dump CSV  ---
    if probe and len(records) > 0:
        with open(save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader(); w.writerows(records)
        print(f" -> saved: {save_csv}")

    if len(decode_events) > 0:
        with open(save_decode_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pos", "step_decoded", "conf_decoded", "token_id", "token_str"])
            w.writeheader();
            w.writerows(decode_events)
        print(f" -> saved: {save_decode_csv}")
    return x

# === NEW: 可视化函数 ===
def _read_csv_by_pos(csv_path):
    """读取 CSV，按 pos 分组；返回 positions(升序) 与 {pos: {metric: [(step, y), ...], ...}}"""
    by_pos = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row["pos"])
            step = int(row["step"])
            rl   = float(row["ratio_last"])
            rl4  = float(row["ratio_last4"])
            rm   = float(row["ratio_mean"])
            d = by_pos.setdefault(pos, {"ratio_last": [], "ratio_last4": [], "ratio_mean": []})
            d["ratio_last"].append((step, rl))
            d["ratio_last4"].append((step, rl4))
            d["ratio_mean"].append((step, rm))
    positions = sorted(by_pos.keys())
    return positions, by_pos


def _make_grid(positions, by_pos, metric_name, out_path, steps_total, ncols=8,
               y_range=(0.0, 1.0), y_ticks=None, decimals=2,
               decode_events=None):
    """每个 pos 一张小图，x 轴统一到 [0, steps_total-1]；在解码 step 处画竖线并标注 token/step/conf。"""
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    decode_events = decode_events or {}

    n = len(positions)
    if n == 0:
        print(f"[warn] no data to plot for {metric_name}")
        return

    # 统一 y 轴范围
    if y_range is None:
        all_ys = []
        for pos in positions:
            all_ys.extend([y for _, y in by_pos[pos][metric_name]])
        y_min = float(np.nanmin(all_ys))
        y_max = float(np.nanmax(all_ys))
        if y_min == y_max:
            eps = 1e-3
            y_min -= eps; y_max += eps
    else:
        y_min, y_max = y_range

    # 统一 y 刻度
    if y_ticks is None:
        y_ticks = np.linspace(y_min, y_max, 5).tolist()

    ncols_eff = min(ncols, max(1, n))
    nrows = math.ceil(n / ncols_eff)
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(ncols_eff*3.2, nrows*2.4), sharex=True)
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    # 统一 x 刻度
    xticks = np.linspace(0, steps_total-1, 5, dtype=int).tolist() if steps_total > 1 else [0]

    # y 轴格式
    yfmt = ("%." + str(decimals) + "f") if decimals is not None else None
    yformatter = FormatStrFormatter(yfmt) if yfmt else None

    for idx, pos in enumerate(positions):
        ax = axes[idx]
        data = sorted(by_pos[pos][metric_name], key=lambda t: t[0])
        xs = [s for s, _ in data]
        ys = [y for _, y in data]

        ax.plot(xs, ys, linewidth=1.0)
        ax.set_title(f"pos {pos}", fontsize=9)
        ax.set_xlim(0, steps_total-1)
        ax.set_xticks(xticks)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        if idx % ncols_eff == 0:
            ax.set_ylabel(metric_name)

        # —— 在“解码 step”做标注 —— #
        ev = decode_events.get(pos)
        if ev is not None:
            s = ev["step_decoded"]
            c = ev["conf_decoded"]
            tok = ev.get("token_str", "")

            # y 值：用该 metric 在“解码 step”的观测（若缺失则用曲线末值）
            y_at_s = None
            for s_i, y_i in data:
                if s_i == s:
                    y_at_s = y_i; break
            if y_at_s is None:
                y_at_s = ys[-1] if ys else (y_min + y_max) / 2

            # 竖线 + 标记点
            ax.axvline(s, linestyle="--", linewidth=0.8)
            ax.scatter([s], [y_at_s], s=10)

            # 文本框位置：略偏右上，避免压住点；并在 y 轴范围内裁剪
            x_text = min(s + max(1, steps_total // 32), steps_total - 1)
            y_text = min(max(y_at_s + 0.05*(y_max - y_min), y_min), y_max)

            label = f"{tok}\nstep={s}, conf={c:.2f}"
            ax.text(x_text, y_text, label, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.15, lw=0.5))

    # 隐藏多余子图
    for j in range(len(positions), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{metric_name} over steps (0..{steps_total-1})", fontsize=12)
    fig.supxlabel("step")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f" -> saved figure: {out_path}")


def visualize_attn_ratio(csv_path="attn_ratio.csv", steps_total=64, out_prefix="attn",
                         ncols=8, lock_unit=True, decimals=2,
                         decode_csv="attn_decode_events.csv"):
    """
    读取 CSV 并输出三张大图（ratio_last / ratio_last4 / ratio_mean）：
    - 每张大图内部：x/y 轴刻度统一；
    - 在每个小图上标注该 pos 的解码 token、step、conf（来自 decode_csv）。
    """
    positions, by_pos = _read_csv_by_pos(csv_path)
    if not positions:
        print(f"[warn] {csv_path} has no rows.")
        return

    decode_events = _read_decode_events(decode_csv)

    if lock_unit:
        y_range = (0.0, 1.0)
        y_ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
    else:
        y_range = None
        y_ticks = None  # 自动用该 metric 的全局 min/max

    _make_grid(positions, by_pos, "ratio_last",
               f"{out_prefix}_ratio_last.png", steps_total, ncols=ncols,
               y_range=y_range, y_ticks=y_ticks, decimals=decimals,
               decode_events=decode_events)

    _make_grid(positions, by_pos, "ratio_last4",
               f"{out_prefix}_ratio_last4.png", steps_total, ncols=ncols,
               y_range=y_range, y_ticks=y_ticks, decimals=decimals,
               decode_events=decode_events)

    _make_grid(positions, by_pos, "ratio_mean",
               f"{out_prefix}_ratio_mean.png", steps_total, ncols=ncols,
               y_range=y_range, y_ticks=y_ticks, decimals=decimals,
               decode_events=decode_events)

def _read_decode_events(csv_path="attn_decode_events.csv"):
    """读取每个 pos 的解码事件（若有多个，取最早/唯一一次）。返回 dict: pos -> {step_decoded, conf_decoded, token_str}"""
    import csv, os
    if not os.path.exists(csv_path):
        print(f"[warn] decode-events CSV not found: {csv_path} (将退化为仅标注 step/conf，且无 token 文本)")
        return {}
    events = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row["pos"])
            step_decoded = int(row["step_decoded"])
            if pos not in events or step_decoded < events[pos]["step_decoded"]:
                events[pos] = {
                    "step_decoded": step_decoded,
                    "conf_decoded": float(row["conf_decoded"]),
                    "token_str": row.get("token_str", "")
                }
    return events

def main():
    device = 'cuda'

    # model_path = "/Users/young/Documents/models/LLaDA-8B-Instruct"
    model_path = "/mnt/users/wuyuyang-20250915/models/LLaDA-8B-Instruct"

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Please don't use latex format."
    # prompt = "Hana sold 4/7 of her stamp collection for $28. How much would she have earned from selling the entire collection?"
    # prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    # prompt = "A storm dropped 5 inches of rain in the first thirty minutes. In the next 30 minutes, the hurricane dropped half that amount of rain. It then dropped 1/2 inch of rain for the next hour. What was the average rainfall total for the duration of the storm?"
    # prompt = "Describe how a person can be very talented and still struggle to find success."
    # prompt = "Talk about how social media connects people and at the same time creates distance."
    # prompt = "Talk about a rule that makes sense in theory but not always in real life."
    # prompt = "Describe why people sometimes avoid what they truly want."
    # prompt = "Explain why two people can hear the same words but understand them differently."
    prompt = "Begin by defending the idea that lying is sometimes necessary for compassion and social harmony. After establishing your case, reverse your stance and argue that every lie, no matter how small, corrodes trust and moral integrity at its foundation."

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # === 设定总步数（横轴统一用这个值） ===
    steps_total = 256

    out = generate(model, input_ids, tokenizer, steps=steps_total, gen_length=256, block_length=16, temperature=0., cfg_scale=0., remasking='low_confidence', probe=True, save_csv="attn_ratio.csv",
    save_decode_csv="attn_decode_events.csv")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

    # === 生成三张大图（每个 pos 一张小图）===
    visualize_attn_ratio(csv_path="attn_ratio.csv", steps_total=steps_total, out_prefix="attn", ncols=8, lock_unit=False, decode_csv="attn_decode_events.csv")

if __name__ == '__main__':
    main()
