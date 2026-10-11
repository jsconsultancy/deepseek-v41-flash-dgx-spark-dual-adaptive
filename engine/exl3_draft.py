"""DSV41_EXL3_DRAFT_DENSE=1: the DSpark draft blocks' attention and shared-expert matrices in EXL3.

The draft reads its own dense weights once (attention per block) or five times (the Markov head)
per draft pass, and it has no quality obligation: the verifier accepts exactly the target's
argmax under any drafter, so a wrong draft costs throughput and never a token
(docs/decode-dynamic-depth.md, "How"). That makes it the one place in this engine where precision
is purely a speed question -- and the checkpoint's draft matrices are all 4-bit in the EXL3 pack
against 8 for the FP8 ones the draft reads today, so the draft's dense bytes halve.

Measured (2026-10-07 rank-0 trace, results/round-breakdown-20261007/, 12 draft passes): the draft
graph spent 10.40 ms a pass, of which 2.52 ms was FP8 dense projections (the 3 blocks' wq_a, wkv,
wq_b, wo_b, shared) and 4.51 ms was BF16 reads of the drafter's own LM head (2.95 ms for one
66 MB... 662 MB call) and Markov head (1.56 ms for five 33 MB calls). This module covers the 2.52;
engine/exl3_draft.py does not touch the two heads, which are separate switches
(DSV41_DRAFT_HEAD_FMT / DSV41_TP_DRAFT_HEAD and MTPWeights.markov_head).

Opt-in, pinned in the boot guard, and off by default: acceptance moves both ways and this is a
different-weights comparison, not a kernel-order one. The pack is built by tools/pack_exl3_draft.py
(~195 MB a rank) and is deliberately separate from the backbone dense pack, so enabling the draft
matrices cannot change the backbone's numerics or force its 2.1 GB to be resident.

Scope: the decode graph's draft passes (engine/fastdecode.py `_draft`, `_final`). Prefill does not
run the drafter. The draft's routed experts stay native FP4 (the `dspark_arena`), as before.
"""
from __future__ import annotations

import json
import os

import torch

FORMAT = "dsv41-exl3-draft-v1"
N_BLOCKS = 3
# Split-K per shape. The backbone table (engine/exl3_dense.SPLITS) was tuned at 4 rows, L2-cold;
# these are called at T = T_DRAFT (5 or 6) rows and are a starting point, not a measurement --
# tools/bench_exl3_dense_decode.py is the instrument.
SPLITS = {"wq_a": 8, "wkv": 16, "wq_b": 1, "wo_a": 2, "wo_b": 4, "w13": 4, "w2": 4, "main_proj": 8}


def enabled() -> bool:
    v = os.environ.get("DSV41_EXL3_DRAFT_DENSE", "0")
    if v not in ("0", "1"):
        raise ValueError("DSV41_EXL3_DRAFT_DENSE must be 0 or 1")
    return v == "1"


def pack_path(model_dir: str, rank: int, world: int) -> str:
    env = os.environ.get("DSV41_EXL3_DRAFT_PACK")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(model_dir)), "exl3-packs",
                        f"exl3-draft-r{rank}of{world}.safetensors")


class SharedExpert:
    """The draft block's shared expert: w1||w3 as one G=2 Exl3Matrix plus the down projection.

    G=2 groups read the same input rows (`_Members`), one input rotation each, and produce
    [T, 2N] in the w1-then-w3 row order `fp8_linear.concat_rows` uses -- so `FastDecoder._shared_ffn`
    splits it at `n` exactly as it splits the merged FP8 weight. Replicated, not TP-split: the
    draft's shared expert is added on both ranks (`_draft`'s comment), and an OutputParallel gather
    here would double its contribution."""

    def __init__(self, w13, w2, n: int):
        self.w13, self.w2, self.n = w13, w2, n

    @property
    def bytes(self) -> int:
        return self.w13.bytes + self.w2.bytes


class DraftLayer:
    """One draft block's EXL3 matrices. The five attention names match
    `engine/exl3_dense.Exl3Layer`, so `FastDecoder._attention` reads it unchanged; `shared` and
    `main_proj` are the draft-only extras. `main_proj` exists on block 0 only."""

    def __init__(self, wq_a, wkv, wq_b, wo_a, wo_b, shared, main_proj=None):
        self.wq_a, self.wkv, self.wq_b, self.wo_a, self.wo_b = wq_a, wkv, wq_b, wo_a, wo_b
        self.shared, self.main_proj = shared, main_proj


class Exl3Draft:
    """Everything tools/pack_exl3_draft.py holds, on the GPU: three draft blocks' matrices."""

    def __init__(self, path: str, device, rank: int, world: int):
        from safetensors import safe_open
        from engine.exl3_dense import Exl3Matrix, Exl3OutputParallel, _Members
        self.path = path
        f = safe_open(path, "pt")
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not an EXL3 draft pack")
        if (int(meta["rank"]), int(meta["world"])) != (rank, world):
            raise ValueError(f"{path}: pack is rank {meta['rank']}/{meta['world']}, engine is {rank}/{world}")
        self.meta = meta
        self.source_sha256 = meta["source_sha256"]
        self.bits = json.loads(meta["bits"])
        members = _Members(device)

        def get(k):
            return f.get_tensor(k).to(device)

        def mat(keys, split):
            return Exl3Matrix([get(k + ".trellis") for k in keys],
                              torch.stack([get(k + ".suh") for k in keys]),
                              torch.stack([get(k + ".svh") for k in keys]),
                              [int(self.bits[k]) for k in keys], split, members)

        self.layers = []
        for k in range(N_BLOCKS):
            p = f"mtp.{k}"
            shared = SharedExpert(
                mat([f"{p}.shared.w1", f"{p}.shared.w3"], SPLITS["w13"]),
                mat([f"{p}.shared.w2"], SPLITS["w2"]),
                int(f.get_tensor(f"{p}.shared.w1.svh").numel()))
            main_proj = (mat([f"{p}.main_proj"], SPLITS["main_proj"])
                         if f"{p}.main_proj.trellis" in f.keys() else None)
            self.layers.append(DraftLayer(
                mat([f"{p}.attn.wq_a"], SPLITS["wq_a"]), mat([f"{p}.attn.wkv"], SPLITS["wkv"]),
                mat([f"{p}.attn.wq_b"], SPLITS["wq_b"]),
                mat([f"{p}.attn.wo_a.{g}" for g in range(4)], SPLITS["wo_a"]),
                Exl3OutputParallel(mat([f"{p}.attn.wo_b"], SPLITS["wo_b"]), world),
                shared, main_proj))
        self.bytes = sum(m.bytes for l in self.layers for m in
                         (l.wq_a, l.wkv, l.wq_b, l.wo_a, l.wo_b, l.main_proj) if m is not None) \
            + sum(l.shared.bytes for l in self.layers)

    def attach(self, mtp):
        """`_x3` for `_attention`/`_final`, `_x3sh` for `_shared_ffn`."""
        for k, w in enumerate(mtp):
            w._x3 = self.layers[k]
            w._x3sh = self.layers[k].shared
