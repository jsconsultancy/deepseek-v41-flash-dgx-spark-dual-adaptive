"""DSV41_EXPERT_TOPP: the pruning rule, its one-launch kernel, and the EXL3 grouping's skip.

    python3 -m unittest engine.test_expert_topp      (the kernel cases need CUDA)
"""
import os
import sys
import unittest
from unittest.mock import patch

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tools'), os.path.join(HERE, '..')]
from engine import expert_topp as ETP  # noqa: E402

K, SENT = 6, 384


def _routes(T=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(T, K, generator=g) ** 3 + 1e-3
    w = w / w.sum(-1, keepdim=True) * 2.5
    w[0] = 2.5 / K                                   # a flat row: top-p keeps all six
    idx = torch.stack([torch.randperm(384, generator=g)[:K] for _ in range(T)])
    return idx, w


class Rule(unittest.TestCase):
    def test_config(self):
        with patch.dict(os.environ, {'DSV41_EXPERT_TOPP': ''}):
            self.assertIsNone(ETP.config())
        with patch.dict(os.environ, {'DSV41_EXPERT_TOPP': '0.90'}):
            self.assertEqual(ETP.config(), (0.9, 3))
        for bad in ('1.0', '-0.1', 'x'):
            with patch.dict(os.environ, {'DSV41_EXPERT_TOPP': bad}), self.assertRaises(ValueError):
                ETP.config()

    def test_reference(self):
        idx, w = _routes()
        i2, w2 = ETP.reference(idx, w, 0.9, 3, SENT)
        keep = i2 != SENT
        self.assertTrue(torch.equal(w2 == 0, ~keep))
        self.assertTrue(torch.equal(i2[keep], idx[keep]) and torch.equal(w2[keep], w[keep]))
        self.assertTrue(bool((keep.sum(-1) >= 3).all()))
        self.assertTrue(bool(keep[0].all()))
        f = w / w.sum(-1, keepdim=True)
        for t in range(idx.shape[0]):
            kept = f[t][keep[t]]
            if (~keep[t]).any() and int(keep[t].sum()) > 3:   # minimal above min_k: one fewer misses 0.9
                self.assertGreaterEqual(float(kept.sum()), 0.9 - 1e-6)
                self.assertLess(float(kept.sum() - kept.min()), 0.9 + 1e-6)
                self.assertLessEqual(float(f[t][~keep[t]].max()), float(kept.min()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
    def test_kernel_matches_reference(self):
        for seed in range(4):
            idx, w = _routes(T=6 + seed, seed=seed)
            want_i, want_w = ETP.reference(idx, w, 0.9, 3, SENT)
            gi, gw = idx.cuda().contiguous(), w.cuda().contiguous()
            ETP.apply_(gi, gw, 0.9, 3, SENT)
            self.assertTrue(torch.equal(gi.cpu(), want_i), seed)
            self.assertTrue(torch.equal(gw.cpu(), want_w), seed)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
class Exl3Grouping(unittest.TestCase):
    def test_negative_picks_join_no_group(self):
        try:
            import exl3_moe_cuda as XC
            XC._lib()
        except Exception as e:  # no nvcc here
            self.skipTest(f'EXL3 kernel unavailable: {e}')
        arena = type('A', (), {'slots': 1000})()
        T = 6
        pick = torch.tensor([[5, 7, 9, -1, -1, 11], [7, 5, 12, 13, -1, -1], [5, 9, 7, 14, 15, 16],
                             [20, 21, 22, -1, -1, -1], [5, 7, 9, 11, 12, 13], [30, 5, 7, -1, 31, -1]],
                            dtype=torch.int32, device='cuda')
        uids, ucount, members = XC.decode_group(pick.reshape(-1).contiguous(), T, K, arena)
        n = int(ucount.item())
        got = sorted(uids[:n].tolist())
        want = sorted({int(e) for e in pick.reshape(-1).tolist() if e >= 0})
        self.assertEqual(got, want)
        flat = pick.reshape(-1).tolist()
        for g in range(n):
            ms = [m for m in members[g].tolist() if m >= 0]
            for m in ms:
                self.assertEqual(flat[(m // 32) * K + m % 32], int(uids[g]))
            self.assertEqual(len(ms), flat.count(int(uids[g])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
