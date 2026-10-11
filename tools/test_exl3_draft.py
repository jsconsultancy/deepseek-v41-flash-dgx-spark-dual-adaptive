"""engine/exl3_draft.py and tools/pack_exl3_draft.py against the reference decoder and the source.

Three separate claims, because they fail in different places:

  kernel   -- Exl3Matrix(x) vs x @ exl3_ref.dequantize(pack) in fp64. The kernel rounds the rotated
              input to fp16, so ~1e-3 relative is expected; the bar is the routed tests' 2e-2.
  slicing  -- the packed rank slice equals the rank slice of the FULL source matrix's decode. This
              is the packer's claim (`engine/tensor_parallel.shard_attention`'s boundaries) and the
              one a wrong N offset would break silently, with a plausible-looking drafter.
  layout   -- the shared expert's w1||w3 pair returns w1's columns first, at the same values a
              w1-only call returns. `FastDecoder._shared_ffn` splits at `n` on that assumption.

Plus row invariance: a row's bits must not depend on how many rows share the call, because the
draft graph is captured once and replayed at a fixed T.

    python3 tools/test_exl3_draft.py [--pack /models/exl3-packs/exl3-draft-r0of2.safetensors] \\
        [--source /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import exl3_ref as R  # noqa: E402
from engine.exl3_dense import MAX_ROWS  # noqa: E402
from engine.exl3_draft import Exl3Draft  # noqa: E402

fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def deq(f, key, bits, cols=None):
    """[K, N] fp64 decode of a packed matrix, optionally the N columns [cols]."""
    t, suh, svh = f.get_tensor(key + ".trellis").cuda(), f.get_tensor(key + ".suh").cuda(), \
        f.get_tensor(key + ".svh").cuda()
    if cols is not None:
        lo, n = cols
        t = t[:, lo // 16:(lo + n) // 16, :].contiguous()
        svh = svh[lo:lo + n].contiguous()
    return R.dequantize(t, suh, svh, float(bits), "mul1").double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/models/exl3-packs/exl3-draft-r0of2.safetensors")
    ap.add_argument("--source", default="/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw")
    a = ap.parse_args()
    from safetensors import safe_open
    f = safe_open(a.pack, "pt")
    meta = f.metadata()
    bits = json.loads(meta["bits"])
    rank, world = int(meta["rank"]), int(meta["world"])
    d = Exl3Draft(a.pack, "cuda", rank, world)
    print(f"pack rank {rank}/{world}, {d.bytes / 1e6:.1f} MB on the GPU, bits "
          f"{sorted(set(bits.values()))}")

    g = torch.Generator(device="cuda").manual_seed(0)
    for k, layer in enumerate(d.layers):
        p = f"mtp.{k}"
        for name, m, key, G in (("wq_a", layer.wq_a, f"{p}.attn.wq_a", 1),
                                ("wkv", layer.wkv, f"{p}.attn.wkv", 1),
                                ("wq_b", layer.wq_b, f"{p}.attn.wq_b", 1),
                                ("wo_a", layer.wo_a, f"{p}.attn.wo_a", 4),
                                ("wo_b", layer.wo_b.local, f"{p}.attn.wo_b", 1)):
            T = 5
            x = (torch.randn(T, G, m.K, generator=g, device="cuda") * 0.5).bfloat16()
            got = m(x).double().view(T, G, m.N)
            keys = [key] if G == 1 else [f"{key}.{i}" for i in range(G)]
            ref = torch.stack([x[:, i].double() @ deq(f, kk, bits[kk]) for i, kk in enumerate(keys)], 1)
            rel = float((got - ref).norm() / ref.norm())
            ok(rel < 2e-2, f"block {k} {name} ({bits[keys[0]]}-bit, K={m.K} N={m.N} G={G}) rel L2 {rel:.2e}")
            one, many = m(x[:1]), m(torch.cat([x[:1], x[1:]]))[:1]
            wide = m(torch.cat([x, x, x, x])[:MAX_ROWS])[:1]
            ok(torch.equal(one, many) and torch.equal(one, wide),
               f"block {k} {name} row-invariant (T=1, 5, {MAX_ROWS})")
        if layer.main_proj is not None:
            m = layer.main_proj
            x = (torch.randn(5, m.K, generator=g, device="cuda") * 0.5).bfloat16()
            got = m(x).double()
            ref = x.double() @ deq(f, f"{p}.main_proj", bits[f"{p}.main_proj"])
            rel = float((got - ref).norm() / ref.norm())
            ok(rel < 2e-2, f"block {k} main_proj (K={m.K} N={m.N}) rel L2 {rel:.2e}")
        else:
            ok(k > 0, f"block {k} has no main_proj (only mtp.0 does)")
        sh = layer.shared
        x = (torch.randn(5, sh.w13.K, generator=g, device="cuda") * 0.5).bfloat16()
        sh_got = sh.w13(x[:, None, :].expand(-1, 2, -1)).double().view(5, 2, sh.n)
        for gi, wn in enumerate(("w1", "w3")):
            ref = x.double() @ deq(f, f"{p}.shared.{wn}", bits[f"{p}.shared.{wn}"])
            rel = float((sh_got[:, gi] - ref).norm() / ref.norm())
            ok(rel < 2e-2, f"block {k} shared.{wn} in group {gi} rel L2 {rel:.2e}")

    # --- slicing: packed rank slice vs the rank slice of the full source decode -------------
    # The source is 39 shards; open the one each key lives in.
    fmap = json.load(open(os.path.join(a.source, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def src(key):
        if key + ".trellis" not in fmap:
            return None
        shard = fmap[key + ".trellis"]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(a.source, shard), "pt")
        h = handles[shard]
        b = int(h.get_slice(key + ".trellis").get_shape()[-1]) // 16
        return R.dequantize(h.get_tensor(key + ".trellis").cuda(), h.get_tensor(key + ".suh").cuda(),
                            h.get_tensor(key + ".svh").cuda(), float(b), "mul1").double()

    # The two sliced matrices (N split by shard_attention) and two replicated ones.
    for key, packed, sliced in (("mtp.0.attn.wq_b", "mtp.0.attn.wq_b", True),
                                ("mtp.0.attn.wo_b", "mtp.0.attn.wo_b", True),
                                ("mtp.0.attn.wq_a", "mtp.0.attn.wq_a", False),
                                ("mtp.0.ffn.shared_experts.w1", "mtp.0.shared.w1", False),
                                ("mtp.0.ffn.shared_experts.w2", "mtp.0.shared.w2", False)):
        W = src(key)
        if W is None:
            ok(False, f"source is missing {key}")
            continue
        N = W.shape[1]
        want = W[:, rank * (N // world):(rank + 1) * (N // world)] if sliced else W
        got = deq(f, packed, bits[packed])
        ok(tuple(got.shape) == tuple(want.shape),
           f"{packed}: packed {tuple(got.shape)} == source "
           f"{'slice' if sliced else 'whole'} {tuple(want.shape)}")
        if got.shape == want.shape:
            rel = float((got - want).norm() / want.norm())
            ok(rel < 2e-3, f"{packed}: equals the source decode (rel L2 {rel:.2e})")
        del W, want, got

    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
