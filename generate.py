import json

import torch
import numpy as np
import csv
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


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
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, probe=True, topk_kl=100, save_csv="probe_metrics.csv"):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # --- NEW: prompt-only baseline p^P (one-time) ---
    if probe:
        outputs_P = model(x, output_attentions=False)
        pP = F.softmax(outputs_P.logits.float(), dim=-1)[0].to(torch.float16).cpu()  # (L, V)
        del outputs_P

        # 预计算每个位置的 prompt-only 熵
        eps = 1e-8
        pP_f32 = pP.to(torch.float32)
        entropy_pP_all = (-pP_f32 * torch.log(pP_f32 + eps)).sum(dim=-1)  # (T,)
    records = []  # to CSV

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

                # Δ_i^P (top-K KL) on masked queries
                p_full = F.softmax(logits[b0], dim=-1)   # (T, V)
                eps = 1e-8
                V = p_full.shape[-1]
                for j, pos in enumerate(q_idx):
                    # topK of current full distribution
                    topv, topk_idx = torch.topk(p_full[pos], k=min(topk_kl, p_full.shape[-1]-1), dim=-1)
                    # gather p^P on CPU for these indices
                    qv = pP[pos, topk_idx.cpu()].to(topv.device, dtype=torch.float32)
                    delta_kl = torch.sum(topv.to(torch.float32) * (torch.log(topv + eps) - torch.log(qv + eps))).item()

                    # prompt-only Top-5 (降序)
                    k5 = min(5, V)
                    pP_top5_vals, _ = torch.topk(pP_f32[pos], k=k5, dim=-1)
                    pP_top5_list = [round(float(x), 6) for x in pP_top5_vals.tolist()]  # e.g., [0.3, 0.2, ...]

                    entropy_pP = float(entropy_pP_all[pos].item())

                    records.append({
                        "step": num_block * steps + i,
                        "pos": int(pos),
                        "pP_top5": json.dumps(pP_top5_list),
                        "entropy_pP": entropy_pP,
                        "ratio_mean": float(ratio_mean[j].item()),
                        "ratio_last4": float(ratio_last4[j].item()),
                        "ratio_last": float(ratio_last[j].item()),
                        "delta_kl_topk": float(delta_kl),
                        "conf": float(confidence[b0, pos].item())
                    })

            # --- original transfer (kept) ---
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    # --- NEW: dump CSV & quick correlation readout ---
    # if probe and len(records) > 0:
    #     with open(save_csv, "w", newline="") as f:
    #         w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    #         w.writeheader(); w.writerows(records)
    #
    #     arr_k = np.array([r["delta_kl_topk"] for r in records], dtype=np.float64)
    #
    #     # last4 layers
    #     arr_r4 = np.array([r["ratio_last4"] for r in records], dtype=np.float64)
    #     pear4 = np.corrcoef(arr_r4, arr_k)[0, 1]
    #     sr4 = np.corrcoef(np.argsort(np.argsort(arr_r4)), np.argsort(np.argsort(arr_k)))[0, 1]
    #     # last layer
    #     arr_rl = np.array([r["ratio_last"] for r in records], dtype=np.float64)
    #     pearl = np.corrcoef(arr_rl, arr_k)[0, 1]
    #     srl = np.corrcoef(np.argsort(np.argsort(arr_rl)), np.argsort(np.argsort(arr_k)))[0, 1]
    #
    #     print(f"[Probe] N={len(records)} Pearson(last4_ratio, Δ_KL)={pear4:.4f}  Spearman={sr4:.4f}")
    #     print(f"[Probe] N={len(records)} Pearson(last_ratio,  Δ_KL)={pearl:.4f}  Spearman={srl:.4f}")
    #     print(f" -> saved: {save_csv}")
    return x

# @ torch.no_grad()
# def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()
#
#     prompt_index = (x != mask_id)
#
#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length
#
#     assert steps % num_blocks == 0
#     steps = steps // num_blocks
#
#     for num_block in range(num_blocks):
#         block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         for i in range(steps):
#             mask_index = (x == mask_id)
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 logits = model(x_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits
#
#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#
#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)
#
#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
#
#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)
#
#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]
#
#     return x


def main():
    device = 'cuda'

    # model_path = "/Users/young/Documents/models/LLaDA-8B-Instruct"
    model_path = "/data/wuyuyang-20250915/models/LLaDA-8B-Instruct"

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    # prompt = "Hana sold 4/7 of her stamp collection for $28. How much would she have earned from selling the entire collection?"
    # prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

    prompt = "A storm dropped 5 inches of rain in the first thirty minutes. In the next 30 minutes, the hurricane dropped half that amount of rain. It then dropped 1/2 inch of rain for the next hour. What was the average rainfall total for the duration of the storm?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=64, gen_length=64, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence', probe=True)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
