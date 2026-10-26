"""Scenario definitions: prompt, executable reference solution, traps, rubric.

Reference answers are *computed*, never hardcoded, so a scenario's parameters
can be changed (to mint variants and defeat memorisation) without hand
re-deriving the key.

Each solver returns (answers, steps, traps):
  answers - {key: value} the grader checks against
  steps   - human-readable derivation, used as the reference write-up
  traps   - {label: {key: value}} answers produced by a *plausible wrong*
            method; the grader uses these to diagnose the specific mistake
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .hardware import GPUS, MAMBA, MODELS

Solved = Tuple[Dict[str, float], List[str], Dict[str, Dict[str, float]]]


@dataclass
class Rubric:
    """A qualitative check. Satisfied if ANY regex matches (case-insensitive)."""
    name: str
    weight: float
    any_of: List[str]
    pitfall: bool = False  # if True, a match *subtracts* weight


@dataclass
class Scenario:
    id: str
    title: str
    difficulty: int  # 1-5
    topics: List[str]
    prompt: str
    solve: Callable[[], Solved]
    tolerances: Dict[str, float]  # relative tolerance per answer key
    rubric: List[Rubric] = field(default_factory=list)


ANSWER_FORMAT = (
    "\n\nShow your reasoning, then finish with one line per requested quantity in "
    "exactly this form (plain number, no units):\n"
    "ANSWER: <key> = <value>\n"
)


# ---------------------------------------------------------------- 1. KV cache
def _kv_capacity() -> Solved:
    m, g = MODELS["llama3-70b"], GPUS["h100-sxm"]
    tp, w_bytes, kv_bytes, util, reserve_gb, ctx = 4, 1.0, 2.0, 0.90, 6.0, 8192
    kv_tok = m.kv_bytes_per_token(kv_bytes)
    kv_tok_gpu = kv_tok / tp
    weights_gpu = m.params * w_bytes / tp
    pool_gpu = g.mem_bytes * util - weights_gpu - reserve_gb * 1e9
    tokens = math.floor(pool_gpu / kv_tok_gpu)
    seqs = math.floor(tokens / ctx)
    # Trap: uses attention heads instead of KV heads (ignores GQA)
    mha_tok = 2 * m.layers * m.heads * m.head_dim * kv_bytes
    mha_seqs = math.floor(math.floor(pool_gpu / (mha_tok / tp)) / ctx)
    steps = [
        f"KV bytes/token = 2 (K,V) x {m.layers} layers x {m.kv_heads} KV heads x {m.head_dim} x {kv_bytes:g} B = {kv_tok:,.0f} B (GQA: use KV heads, not {m.heads} query heads).",
        f"TP={tp} shards KV heads ({m.kv_heads}/{tp} = {m.kv_heads // tp} per rank): {kv_tok_gpu:,.0f} B/token/GPU.",
        f"Weights/GPU = {m.params / 1e9:.1f}B x {w_bytes:g} B / {tp} = {weights_gpu / 1e9:.2f} GB.",
        f"KV pool/GPU = {g.mem_gb:g} x {util} - {weights_gpu / 1e9:.2f} - {reserve_gb:g} = {pool_gpu / 1e9:.2f} GB.",
        f"Pool tokens = {pool_gpu:,.0f} / {kv_tok_gpu:,.0f} = {tokens:,}; concurrent {ctx}-token sequences = floor({tokens:,}/{ctx}) = {seqs}.",
    ]
    return (
        {"kv_bytes_per_token": kv_tok, "max_concurrent_seqs": seqs},
        steps,
        {"ignored GQA (used query heads)": {"kv_bytes_per_token": mha_tok, "max_concurrent_seqs": mha_seqs}},
    )


KV_PROMPT = (
    "You are sizing a vLLM deployment of Llama-3-70B (80 layers, 64 query heads, 8 KV heads, head_dim 128) "
    "on 4x H100 80GB with tensor parallelism TP=4. Weights are FP8 (1 byte/param, 70.6e9 params, sharded evenly). "
    "The KV cache is FP16. gpu_memory_utilization=0.90 (of 80 GB, decimal), and each GPU additionally reserves 6 GB "
    "for activations/CUDA graphs that is NOT part of the KV pool. Ignore block-size fragmentation.\n\n"
    "Compute (a) kv_bytes_per_token: total KV bytes per token across all ranks, and "
    "(b) max_concurrent_seqs: how many 8192-token sequences fit in the KV pool."
) + ANSWER_FORMAT


# ------------------------------------------------------- 2. decode roofline
def _decode_roofline() -> Solved:
    m, g = MODELS["llama3-70b"], GPUS["h100-sxm"]
    tp, w_bytes = 2, 2.0
    per_gpu = m.params * w_bytes / tp
    t_ms = per_gpu / g.hbm_bps * 1e3
    ridge = g.ridge_flops_per_byte
    # each weight (w_bytes) does 2 FLOPs per sequence in the batch
    crossover = ridge * w_bytes / 2
    steps = [
        f"Batch-1 decode streams every weight once: {per_gpu / 1e9:.1f} GB/GPU at {g.hbm_tbps} TB/s = {t_ms:.2f} ms (lower bound; ignores KV reads and comm).",
        f"Ridge point = {g.fp16_tflops} TFLOP/s / {g.hbm_tbps} TB/s = {ridge:.0f} FLOP/B.",
        f"GEMM intensity at batch B = 2B FLOPs per {w_bytes:g} B weight = B FLOP/B (FP16), so compute-bound once B >~ {crossover:.0f}.",
    ]
    wrong = {"min_decode_ms": m.params * w_bytes / g.hbm_bps * 1e3, "crossover_batch": ridge * w_bytes}
    return (
        {"min_decode_ms": t_ms, "crossover_batch": crossover},
        steps,
        {"forgot TP sharding / 2x FLOP-per-weight": wrong},
    )


DECODE_PROMPT = (
    "Llama-3-70B in FP16 (70.6e9 params) is served on 2x H100 SXM (3.35 TB/s HBM each, 989 TFLOP/s dense FP16) "
    "with TP=2. Treat decode as purely weight-streaming: ignore KV-cache reads, attention FLOPs and communication.\n\n"
    "Compute (a) min_decode_ms: the lower-bound time in ms for one decode step at batch size 1, and "
    "(b) crossover_batch: the batch size at which the linear layers shift from memory-bound to compute-bound "
    "(an FP16 weight costs 2 bytes and does 2 FLOPs per sequence in the batch)."
) + ANSWER_FORMAT


# -------------------------------------------------------------- 3. quant
def _quant() -> Solved:
    m, g = MODELS["llama3-8b"], GPUS["l40s"]
    group = 128
    bits_int4 = 4 + 16 / group  # fp16 scale per group
    fp16_ms = m.params * 2 / g.hbm_bps * 1e3
    int4_bytes = m.params * bits_int4 / 8
    int4_ms = int4_bytes / g.hbm_bps * 1e3
    speed = fp16_ms / int4_ms
    steps = [
        f"Effective bits/weight = 4 + 16/{group} = {bits_int4:.3f}.",
        f"INT4 weights = {m.params / 1e9:.2f}B x {bits_int4:.3f}/8 = {int4_bytes / 1e9:.2f} GB (FP16: {m.params * 2 / 1e9:.2f} GB).",
        f"Bandwidth-bound batch-1 decode: FP16 {fp16_ms:.2f} ms, INT4 {int4_ms:.2f} ms -> speedup {speed:.2f}x (not 4x: scale overhead).",
    ]
    naive = {"int4_weights_gb": m.params * 0.5 / 1e9, "int4_speedup": 4.0, "int4_decode_ms": fp16_ms / 4}
    return (
        {"int4_weights_gb": int4_bytes / 1e9, "int4_decode_ms": int4_ms, "int4_speedup": speed},
        steps,
        {"ignored scale/group overhead (exactly 4 bits)": naive},
    )


QUANT_PROMPT = (
    "Llama-3-8B (8.03e9 params) runs on one L40S (864 GB/s HBM). You quantise weights to INT4 (W4A16) with "
    "group size 128 and one FP16 scale per group (no zero-point). Assume batch-1 decode is purely "
    "weight-bandwidth-bound and dequantisation is free.\n\n"
    "Compute (a) int4_weights_gb: weight footprint in GB, (b) int4_decode_ms: batch-1 decode time per token in ms, "
    "(c) int4_speedup: speedup over the FP16 baseline. Then explain in 3-5 sentences when W4A16 stops helping as batch size grows."
) + ANSWER_FORMAT


# ------------------------------------------------------- 4. TP all-reduce
def _tp_allreduce() -> Solved:
    m, g = MODELS["llama3-70b"], GPUS["h100-sxm"]
    n, batch, alpha_us = 8, 64, 3.0
    msg = batch * m.hidden * 2
    wire = 2 * (n - 1) / n * msg
    bw_us = wire / g.link_bps * 1e6
    lat_us = 2 * (n - 1) * alpha_us
    ar_us = bw_us + lat_us
    step_ms = 2 * m.layers * ar_us / 1e3
    steps = [
        f"Message = {batch} x {m.hidden} x 2 B = {msg:,} B.",
        f"Ring all-reduce moves 2(N-1)/N x M = {wire:,.0f} B/GPU -> {bw_us:.2f} us at {g.link_gbps:g} GB/s.",
        f"Latency term = 2(N-1) hops x {alpha_us} us = {lat_us:.0f} us -> one all-reduce = {ar_us:.1f} us (latency dominates {lat_us / ar_us:.0%}).",
        f"2 all-reduces/layer (attn out, MLP down) x {m.layers} layers = {2 * m.layers} per step -> {step_ms:.2f} ms/step.",
    ]
    wrong_ar = msg / g.link_bps * 1e6
    traps = {"bandwidth term only, no latency, 1 all-reduce/layer": {
        "allreduce_us": wrong_ar, "step_comm_ms": m.layers * wrong_ar / 1e3}}
    return {"allreduce_us": ar_us, "step_comm_ms": step_ms}, steps, traps


TP_PROMPT = (
    "Llama-3-70B (80 layers, hidden 8192) decodes at batch 64 with TP=8 on one NVLink H100 node "
    "(450 GB/s unidirectional per GPU). Activations are FP16. Model each all-reduce as a ring: "
    "time = 2(N-1)/N x M / BW  +  2(N-1) x alpha, with alpha = 3 us per hop. "
    "Megatron-style TP does two all-reduces per transformer layer.\n\n"
    "Compute (a) allreduce_us: time of one all-reduce in us, and (b) step_comm_ms: total all-reduce time per decode step in ms. "
    "Then state which term dominates and name two techniques that reduce it."
) + ANSWER_FORMAT


# -------------------------------------------------------------- 5. Mamba2
def _mamba_state() -> Solved:
    mb, tr = MAMBA["mamba2-2.7b"], MODELS["llama3-8b"]
    dt = 2.0
    state = mb.layers * mb.state_elems_per_layer() * dt
    kv_tok = tr.kv_bytes_per_token(dt)
    cross = state / kv_tok
    kv128 = kv_tok * 131072
    steps = [
        f"Mamba2 d_inner={mb.d_inner}, nheads={mb.nheads}. SSM state/layer = {mb.nheads}x{mb.headdim}x{mb.d_state} = {mb.nheads * mb.headdim * mb.d_state:,} elems.",
        f"Conv state/layer = (d_inner + 2*ngroups*d_state) x (d_conv-1) = {(mb.d_inner + 2 * mb.ngroups * mb.d_state) * (mb.d_conv - 1):,} elems.",
        f"Total = {mb.layers} layers x {mb.state_elems_per_layer():,} x {dt:g} B = {state / 1e6:.2f} MB, independent of context length.",
        f"Llama-3-8B KV = {kv_tok:,.0f} B/token -> equal at {cross:.0f} tokens; at 131,072 tokens KV = {kv128 / 1e9:.2f} GB.",
    ]
    traps = {"omitted conv state": {
        "mamba_state_mb": mb.layers * mb.nheads * mb.headdim * mb.d_state * dt / 1e6,
        "crossover_tokens": mb.layers * mb.nheads * mb.headdim * mb.d_state * dt / kv_tok}}
    return (
        {"mamba_state_mb": state / 1e6, "crossover_tokens": cross, "kv_at_128k_gb": kv128 / 1e9},
        steps,
        traps,
    )


MAMBA_PROMPT = (
    "Compare per-sequence inference memory for Mamba2-2.7B and Llama-3-8B (32 layers, 8 KV heads, head_dim 128). "
    "Mamba2-2.7B: 64 layers, d_model 2560, expand 2, headdim 64, d_state 128, ngroups 1, d_conv 4. "
    "Per layer the recurrent cache is the SSM state (nheads x headdim x d_state) plus the causal-conv state "
    "((d_inner + 2*ngroups*d_state) x (d_conv - 1)). Store everything in FP16.\n\n"
    "Compute (a) mamba_state_mb: total Mamba2 state per sequence in MB (1e6 B), (b) crossover_tokens: the context "
    "length at which Llama-3-8B's KV cache equals that state, and (c) kv_at_128k_gb: Llama-3-8B KV size in GB at 131,072 tokens. "
    "Then explain why prefix caching is harder for SSM layers than for attention layers."
) + ANSWER_FORMAT


# ------------------------------------------------- 6. continuous batching
def _lengths(n: int = 256, seed: int = 0) -> List[int]:
    r = random.Random(seed)
    return [max(1, min(1024, int(r.lognormvariate(4.5, 1.0)))) for _ in range(n)]


def simulate_static(lengths: List[int], slots: int) -> int:
    return sum(max(lengths[i:i + slots]) for i in range(0, len(lengths), slots))


def simulate_continuous(lengths: List[int], slots: int) -> int:
    queue, active, steps = list(lengths), [], 0
    while queue or active:
        while queue and len(active) < slots:
            active.append(queue.pop(0))
        steps += 1
        active = [x - 1 for x in active if x - 1 > 0]
    return steps


def _batching() -> Solved:
    ls, slots = _lengths(), 32
    s, c = simulate_static(ls, slots), simulate_continuous(ls, slots)
    ideal = math.ceil(sum(ls) / slots)
    steps = [
        f"{len(ls)} requests, mean output {sum(ls) / len(ls):.1f} tokens, max {max(ls)}, {slots} slots.",
        f"Static: each group of {slots} runs max(len) steps -> {s:,} steps (slots idle after a sequence finishes).",
        f"Continuous: a slot refills the moment its sequence ends -> {c:,} steps (work lower bound {ideal:,}).",
        f"Speedup = {s / c:.2f}x. Ignores prefill and scheduling overhead.",
    ]
    traps = {"used mean length instead of per-batch max": {
        "static_steps": math.ceil(len(ls) / slots) * (sum(ls) / len(ls)), "speedup": 1.0}}
    return {"static_steps": s, "continuous_steps": c, "speedup": s / c}, steps, traps


BATCH_PROMPT = (
    "A server has 32 decode slots; each step every active slot emits one token. Compare static batching "
    "(take 32 requests in arrival order, run until the longest finishes, then take the next 32) against "
    "continuous batching (a finished sequence's slot is refilled from the FIFO queue on the next step). "
    "Ignore prefill cost and scheduling overhead; one step = one unit.\n\n"
    "Output lengths:\n"
    "```python\nimport random\nr = random.Random(0)\n"
    "lengths = [max(1, min(1024, int(r.lognormvariate(4.5, 1.0)))) for _ in range(256)]\n```\n"
    "Compute (a) static_steps, (b) continuous_steps, (c) speedup = static/continuous. You may run code."
) + ANSWER_FORMAT


# ---------------------------------------------- 7. fused kernel bandwidth
def _fused_norm() -> Solved:
    g = GPUS["h100-sxm"]
    tokens, hidden, eff = 8192, 8192, 0.90
    t = tokens * hidden * 2
    unfused, fused = 6 * t, 4 * t
    fused_us = fused / (eff * g.hbm_bps) * 1e6
    steps = [
        f"One activation tensor = {tokens}x{hidden}x2 B = {t / 1e6:.1f} MB.",
        "Unfused: add (read x, read res, write h = 3) + reduce (read h = 1) + normalise (read h, write y = 2) = 6 tensor passes.",
        "Fused: read x, read res, write h, write y, row held in registers/SMEM = 4 passes. Kernel is memory-bound (~5 FLOP/elem).",
        f"Bytes: unfused {unfused / 1e6:.1f} MB, fused {fused / 1e6:.1f} MB -> bound {unfused / fused:.2f}x. Fused time at {eff:.0%} of peak BW = {fused_us:.1f} us.",
    ]
    traps = {"counted only 3 passes for fused (skipped residual write)": {
        "fused_bytes_mb": 3 * t / 1e6, "fused_time_us": 3 * t / (eff * g.hbm_bps) * 1e6}}
    return {"unfused_bytes_mb": unfused / 1e6, "fused_bytes_mb": fused / 1e6, "fused_time_us": fused_us}, steps, traps


NORM_PROMPT = (
    "You are writing a fused residual-add + RMSNorm CUDA kernel for prefill. Input: tokens=8192, hidden=8192, FP16, "
    "on an H100 SXM (3.35 TB/s). The fused kernel must output both the updated residual h = x + res and y = RMSNorm(h) * w "
    "(ignore the tiny weight vector w). The eager baseline runs three kernels: add (x,res -> h); a reduction (h -> rstd); "
    "and a normalise (h, rstd -> y), where rstd is negligible in size. Assume the fused kernel keeps each row in "
    "registers/shared memory and achieves 90% of peak bandwidth.\n\n"
    "Compute (a) unfused_bytes_mb, (b) fused_bytes_mb (HBM traffic in 1e6 B), and (c) fused_time_us. "
    "Then explain why raising occupancy would not help much, and what actually limits a hidden=8192 row-per-block design."
) + ANSWER_FORMAT


SCENARIOS: Dict[str, Scenario] = {s.id: s for s in [
    Scenario("kv-capacity", "KV-cache sizing under GQA + TP", 2, ["vllm", "kv-cache", "tensor-parallel"],
             KV_PROMPT, _kv_capacity, {"kv_bytes_per_token": 0.001, "max_concurrent_seqs": 0.01},
             [Rubric("mentions GQA / KV heads", 0.5, [r"\bGQA\b", r"grouped[- ]query", r"KV heads?"]),
              Rubric("KV sharded across TP ranks", 0.3, [r"shard", r"per (rank|gpu)", r"/\s*4"]),
              Rubric("notes paged/block fragmentation caveat", 0.2, [r"fragment", r"block size", r"paged"])]),
    Scenario("decode-roofline", "Decode roofline and batch crossover", 2, ["roofline", "gpu-perf"],
             DECODE_PROMPT, _decode_roofline, {"min_decode_ms": 0.01, "crossover_batch": 0.02},
             [Rubric("identifies memory-bound", 0.5, [r"memory[- ]bound", r"bandwidth[- ]bound"]),
              Rubric("mentions ridge / arithmetic intensity", 0.5, [r"ridge", r"arithmetic intensity", r"FLOP/?B"])]),
    Scenario("quant-w4a16", "W4A16 footprint, speedup and limits", 3, ["quantization", "bandwidth"],
             QUANT_PROMPT, _quant, {"int4_weights_gb": 0.01, "int4_decode_ms": 0.02, "int4_speedup": 0.02},
             [Rubric("group-scale overhead", 0.3, [r"scale", r"group"]),
              Rubric("large batch becomes compute-bound", 0.4, [r"compute[- ]bound", r"large(r)? batch", r"dequant"]),
              Rubric("KV cache / activations not shrunk", 0.3, [r"KV", r"activation"]),
              Rubric("claims exactly 4x with no caveat", 0.3, [r"exactly 4x"], pitfall=True)]),
    Scenario("tp-allreduce", "Tensor-parallel all-reduce cost at decode", 4, ["tensor-parallel", "nccl", "latency"],
             TP_PROMPT, _tp_allreduce, {"allreduce_us": 0.02, "step_comm_ms": 0.02},
             [Rubric("latency term dominates", 0.4, [r"latency[- ](bound|dominat)", r"alpha", r"small message"]),
              Rubric("names mitigation (custom/one-shot AR, NVLS, overlap, fusion)", 0.6,
                     [r"one[- ]shot", r"custom all[- ]?reduce", r"NVLS|multimem", r"overlap", r"fus(e|ion)", r"two[- ]shot"])]),
    Scenario("mamba2-state", "Mamba2 state vs. attention KV cache", 4, ["mamba", "kv-cache", "ssm"],
             MAMBA_PROMPT, _mamba_state, {"mamba_state_mb": 0.005, "crossover_tokens": 0.01, "kv_at_128k_gb": 0.005},
             [Rubric("state is constant in sequence length", 0.3, [r"constant", r"independent of (context|sequence)"]),
              Rubric("prefix caching needs state snapshots/checkpoints", 0.5,
                     [r"checkpoint", r"snapshot", r"cannot (be )?(truncat|slice)", r"not (truncat|slice)"]),
              Rubric("state precision (fp32 SSM state) matters", 0.2, [r"fp32|float32|precision"])]),
    Scenario("continuous-batching", "Static vs. continuous batching", 3, ["scheduling", "continuous-batching"],
             BATCH_PROMPT, _batching, {"static_steps": 0.0, "continuous_steps": 0.0, "speedup": 0.005}, []),
    Scenario("fused-rmsnorm", "Fused residual+RMSNorm traffic model", 3, ["cuda", "kernel-fusion", "roofline"],
             NORM_PROMPT, _fused_norm, {"unfused_bytes_mb": 0.005, "fused_bytes_mb": 0.005, "fused_time_us": 0.02},
             [Rubric("memory-bound, not occupancy-bound", 0.4, [r"memory[- ]bound", r"bandwidth"]),
              Rubric("register/SMEM pressure for 8192-wide row", 0.4,
                     [r"register", r"shared memory|smem", r"vector(ized)? (load|access)|128[- ]bit|float4"]),
              Rubric("warp-shuffle / block reduction", 0.2, [r"shuffle", r"warp", r"block[- ]reduc"])]),
]}


# Reference commentary for the qualitative part of each prompt (graded by rubric).
DISCUSSION: Dict[str, str] = {
    "kv-capacity": (
        "Because of GQA only the 8 KV heads are cached, shrinking the cache 8x versus MHA. KV heads shard across TP ranks "
        "(2 per rank). Real vLLM adds block-size fragmentation (paged blocks), so treat the figure as an upper bound."),
    "decode-roofline": (
        "Batch-1 decode is memory-bound: arithmetic intensity is far below the ridge point (FLOP/B). Throughput scales "
        "almost linearly with batch until the ridge, which is why continuous batching is so effective."),
    "quant-w4a16": (
        "The speedup is below 4x because each group carries an FP16 scale. At large batch the GEMM becomes compute-bound "
        "and dequant work sits on the critical path, so W4A16 gains vanish; KV cache and activations are not shrunk by "
        "weight-only quantisation, so long-context workloads see less benefit."),
    "tp-allreduce": (
        "Messages are small (1 MB), so each all-reduce is latency-bound (alpha term dominates). Mitigations: a custom "
        "one-shot all-reduce for small messages, NVLS/multimem in-switch reduction, overlap with compute, or fusing the "
        "all-reduce with the following RMSNorm."),
    "mamba2-state": (
        "The recurrent state is constant in sequence length, unlike KV. Prefix caching is harder because the SSM state is "
        "a lossy summary: you cannot truncate it to a shorter prefix, so the server must store state checkpoints at chosen "
        "block boundaries. FP32 state precision is often required for stability, doubling the figure."),
    "continuous-batching": "",
    "fused-rmsnorm": (
        "The kernel is memory-bound, so occupancy beyond saturating HBM does not help. An 8192-wide FP16 row is 16 KB: "
        "keeping it in registers or shared memory limits blocks per SM, so use vectorised 128-bit loads (float4) and a "
        "warp-shuffle plus block reduction to compute the variance."),
}
