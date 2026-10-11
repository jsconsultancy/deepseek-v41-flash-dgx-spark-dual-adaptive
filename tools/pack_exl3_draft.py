"""Offline: one TP rank's EXL3 DSpark draft matrices (the three `mtp.*` blocks).

    python tools/pack_exl3_draft.py --source ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \\
        --rank 0 --world 2 --out ~/models/exl3-packs/exl3-draft-r0of2.safetensors

`engine/exl3_draft.py` reads these instead of the native FP8/FP4 draft weights
(`DSV41_EXL3_DRAFT_DENSE`). This is a separate, small pack (~195 MB a rank) on purpose: the
draft's EXL3 weights are a speed-only lever, so enabling them must not require loading the
2.1 GB backbone dense pack or changing the backbone's numerics.

Every draft matrix in this checkpoint is 4-bit, against 8 for the FP8 weights the draft reads
today, so the draft's dense bytes halve. The draft proposes and the verify path accepts exactly
the target's argmax, so a worse drafter costs throughput, never the final token: this is the one
place in the engine where precision is not a quality question. It is still opt-in, because
acceptance moves both ways (docs/decode-dynamic-depth.md's FP8 draft head swung +0.9% to +11.5%
between workloads).

Slices follow `engine/tensor_parallel.shard_attention` exactly, which is what
`DSV41_TP_DRAFT_ATTN=1` applies to `mtp.*`:

  attn.wq_a, attn.wkv   replicated (whole)
  attn.wq_b             output rows = heads: N tiles [rank*N/2/16, +N/2/16), svh slice, suh whole
  attn.wo_a             this rank's groups: g in [rank*4, rank*4 + 4), whole
  attn.wo_b             OutputParallel output rows of 5120 (K 8192), svh slice, suh whole
  shared_experts w1/w3  **replicated** (whole, 2304) -- unlike the backbone, the draft's shared
                        expert is not TP-split (engine/fastdecode.py `_draft`: another collective
                        there would double its contribution)
  shared_experts w2     replicated (whole, 5120)
  mtp.0.main_proj       replicated (K 15360 -> N 5120)

No abliteration term: the overlay (`DSV41_ABLIT_WOB`) replaces wo_b in backbone layers 10-35 only.

EXL3 rotates in blocks of 128 along K and N. `wq_b`'s per-rank 16384 and `wo_b`'s 2560 output
rows are both multiples of 128, so a rank's matrix is an exact slice (the routed experts'
argument, docs/exl3-plan.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

N_BLOCKS = 3
O_GROUPS = 8
FORMAT = "dsv41-exl3-draft-v1"


def bits_of(trellis_shape) -> int:
    return int(trellis_shape[-1]) // 16


def slice_matrix(src: SourceCheckpoint, name: str, n_lo: int, n_cols: int | None):
    """trellis [KT, NT, 16*bits] int16 -> the N columns [n_lo, n_lo + n_cols) (None = whole)."""
    shape = src.shape(f"{name}.trellis")
    mul = int(src.array(f"{name}.mul1").reshape(-1)[0]) & 0xFFFFFFFF
    if mul != R.MUL1_MUL:
        raise SystemExit(f"{name}: mul1 marker {mul:#x}, kernel decodes {R.MUL1_MUL:#x} only")
    if n_cols is None:
        t = src.array(f"{name}.trellis")
        svh = src.array(f"{name}.svh")
    else:
        assert n_lo % 128 == 0 and n_cols % 128 == 0, (name, n_lo, n_cols)
        t = src.array(f"{name}.trellis", (slice(None), slice(n_lo // 16, (n_lo + n_cols) // 16), slice(None)))
        svh = src.array(f"{name}.svh", (slice(n_lo, n_lo + n_cols),))
    suh = src.array(f"{name}.suh")
    return {"trellis": t, "suh": suh, "svh": svh}, bits_of(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.world == 2 and 0 <= a.rank < 2, "the engine's draft attention TP is two ranks"
    from safetensors.numpy import save_file
    src = SourceCheckpoint(a.source)
    r, w = a.rank, a.world
    tensors, meta_bits = {}, {}

    def put(key, parts, bits):
        for k, v in parts.items():
            tensors[f"{key}.{k}"] = np.ascontiguousarray(v)
        meta_bits[key] = bits

    for k in range(N_BLOCKS):
        p = f"mtp.{k}"
        for name in ("attn.wq_a", "attn.wkv"):
            put(f"{p}.{name}", *slice_matrix(src, f"{p}.{name}", 0, None))
        n = src.shape(f"{p}.attn.wq_b.svh")[0] // w
        put(f"{p}.attn.wq_b", *slice_matrix(src, f"{p}.attn.wq_b", r * n, n))
        for gi, g in enumerate(range(r * O_GROUPS // w, (r + 1) * O_GROUPS // w)):
            put(f"{p}.attn.wo_a.{gi}", *slice_matrix(src, f"{p}.attn.wo_a.slice.{g}", 0, None))
        n = src.shape(f"{p}.attn.wo_b.svh")[0] // w
        put(f"{p}.attn.wo_b", *slice_matrix(src, f"{p}.attn.wo_b", r * n, n))
        for name in ("w1", "w3", "w2"):
            put(f"{p}.shared.{name}", *slice_matrix(src, f"{p}.ffn.shared_experts.{name}", 0, None))
        if k == 0:
            put(f"{p}.main_proj", *slice_matrix(src, f"{p}.main_proj", 0, None))
        print(f"block {k}: {sum(v.nbytes for kk, v in tensors.items() if kk.startswith(p + '.')) / 1e6:.1f} MB",
              flush=True)
    meta = {"format": FORMAT, "rank": str(r), "world": str(w),
            "source_sha256": R.manifest_sha256(a.source), "bits": json.dumps(meta_bits)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, a.out)
    total = sum(v.nbytes for v in tensors.values())
    print(f"wrote {a.out}: {len(tensors)} tensors, {total / 1e6:.1f} MB, source {meta['source_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
