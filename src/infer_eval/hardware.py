"""Hardware and model specs used by the scenarios.

All bandwidths are decimal (1 GB = 1e9 B) so that scenario prompts and
reference solutions use identical units.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float
    hbm_tbps: float       # HBM bandwidth, TB/s
    fp16_tflops: float    # dense tensor-core FP16/BF16
    link_gbps: float      # per-GPU unidirectional interconnect bandwidth, GB/s

    @property
    def mem_bytes(self) -> float:
        return self.mem_gb * 1e9

    @property
    def hbm_bps(self) -> float:
        return self.hbm_tbps * 1e12

    @property
    def link_bps(self) -> float:
        return self.link_gbps * 1e9

    @property
    def ridge_flops_per_byte(self) -> float:
        return self.fp16_tflops * 1e12 / self.hbm_bps


@dataclass(frozen=True)
class Transformer:
    name: str
    params: float
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int

    def kv_bytes_per_token(self, kv_dtype_bytes: float) -> float:
        """Total across all TP ranks: K and V, every layer, KV heads only (GQA)."""
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_dtype_bytes


@dataclass(frozen=True)
class Mamba2:
    name: str
    layers: int
    d_model: int
    expand: int
    headdim: int
    d_state: int
    ngroups: int
    d_conv: int

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim

    def state_elems_per_layer(self) -> int:
        ssm = self.nheads * self.headdim * self.d_state
        conv = (self.d_inner + 2 * self.ngroups * self.d_state) * (self.d_conv - 1)
        return ssm + conv


GPUS = {
    "h100-sxm": GPU("H100 SXM 80GB", 80, 3.35, 989, 450),
    "a100-80g": GPU("A100 SXM 80GB", 80, 2.039, 312, 300),
    "l40s": GPU("L40S 48GB", 48, 0.864, 362, 25),  # PCIe Gen4 x16 effective
}

MODELS = {
    "llama3-70b": Transformer("Llama-3-70B", 70.6e9, 80, 8192, 64, 8, 128),
    "llama3-8b": Transformer("Llama-3-8B", 8.03e9, 32, 4096, 32, 8, 128),
}

MAMBA = {
    "mamba2-2.7b": Mamba2("Mamba2-2.7B", 64, 2560, 2, 64, 128, 1, 4),
}
