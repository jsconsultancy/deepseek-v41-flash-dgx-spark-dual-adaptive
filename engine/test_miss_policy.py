"""Model._apply_miss_policy on CPU: rows without a miss are untouched, and each policy changes only
what it claims to (tools/bench_miss_policy_tp.py measures what that does to held-out NLL).

    python3 -m unittest engine.test_miss_policy
"""
import os
import sys
import unittest
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tools'), os.path.join(HERE, '..')]
from engine.model import Model  # noqa: E402

E, K, T, SCALE = 64, 6, 40, 2.5


def _route(seed=0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(T, E, generator=g) + 0.01
    logits = scores + 0.1 * torch.randn(E, generator=g)
    keep = torch.rand(E, generator=g) < 0.6
    masked = logits.masked_fill(~keep, float('-inf'))
    idx = masked.topk(K, dim=-1)[1]
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * SCALE
    return logits, scores, keep, idx, w


def _apply(policy, seed=0):
    logits, scores, keep, idx, w = _route(seed)
    me = SimpleNamespace(args=SimpleNamespace(route_scale=SCALE), MISS_POLICIES=Model.MISS_POLICIES,
                         miss_gen=torch.Generator().manual_seed(7), miss_stats=torch.zeros(2, dtype=torch.int64))
    i2, w2 = Model._apply_miss_policy(me, policy, logits, scores, idx, w, keep, K)
    want = logits.topk(K, dim=-1)[1]
    sub = ~(idx[:, :, None] == want[:, None, :]).any(-1)
    return dict(logits=logits, scores=scores, keep=keep, idx=idx, w=w, i2=i2, w2=w2, want=want, sub=sub,
                stats=me.miss_stats)


class MissPolicy(unittest.TestCase):
    def test_fixture_has_misses_and_clean_rows(self):
        r = _apply('skip')
        rows = r['sub'].any(-1)
        self.assertTrue(bool(rows.any()) and bool((~rows).any()))
        self.assertEqual(r['stats'].tolist(), [int(r['sub'].sum()), T * K])

    def test_clean_rows_bit_identical(self):
        for policy in ('skip', 'skip_raw', 'random'):
            r = _apply(policy)
            clean = ~r['sub'].any(-1)
            self.assertTrue(torch.equal(r['i2'][clean], r['idx'][clean]), policy)
            self.assertTrue(torch.equal(r['w2'][clean], r['w'][clean]), policy)

    def test_skip_zeroes_substitutes_and_renormalizes(self):
        r = _apply('skip')
        self.assertTrue(torch.equal(r['i2'], r['idx']))  # ids stay resident: the slot table is valid
        self.assertTrue(bool((r['w2'][r['sub']] == 0).all()))
        some = (~r['sub']).any(-1)
        torch.testing.assert_close(r['w2'][some].sum(-1), torch.full((int(some.sum()),), SCALE))

    def test_skip_raw_keeps_unpruned_weights(self):
        r = _apply('skip_raw')
        full = r['scores'].gather(1, r['want'])
        full = full / full.sum(-1, keepdim=True) * SCALE
        for t in range(T):
            for j in range(K):
                if r['sub'][t, j]:
                    self.assertEqual(float(r['w2'][t, j]), 0.0)
                else:
                    e = int(r['idx'][t, j])
                    want = full[t][r['want'][t] == e]
                    torch.testing.assert_close(r['w2'][t, j], want[0])

    def test_random_replaces_only_substitutes_with_other_residents(self):
        r = _apply('random')
        self.assertTrue(torch.equal(r['w2'], r['w']))
        self.assertTrue(torch.equal(r['i2'][~r['sub']], r['idx'][~r['sub']]))
        for t in range(T):
            row = r['i2'][t].tolist()
            self.assertEqual(len(set(row)), K)
            for j in range(K):
                if r['sub'][t, j]:
                    e = row[j]
                    self.assertTrue(bool(r['keep'][e]))
                    self.assertNotIn(e, r['idx'][t].tolist())
        self.assertFalse(torch.equal(r['i2'], r['idx']))

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):
            _apply('nearest')


class ExpertTopP(unittest.TestCase):
    def _run(self, weights, rule):
        me = SimpleNamespace(args=SimpleNamespace(route_scale=SCALE), topp_stats=torch.zeros(2, dtype=torch.int64))
        return Model._apply_expert_topp(me, rule, weights), me.topp_stats

    def test_prefix_mass_and_min_k(self):
        # shares 0.5, 0.3, 0.1, 0.05, 0.03, 0.02 (slot order shuffled): mass before the 3rd is 0.8 < 0.85,
        # before the 4th 0.9 -> keep 3 at top-p 0.85; at 0.95 keep 4; min_k 4 forces 4 at 0.85
        share = torch.tensor([[0.1, 0.5, 0.02, 0.3, 0.05, 0.03]])
        w = share * SCALE
        out, st = self._run(w, (0.85, 3, 'orig'))
        self.assertEqual((out[0] > 0).tolist(), [True, True, False, True, False, False])
        self.assertTrue(torch.equal(out[0][out[0] > 0], w[0][w[0] * (out[0] > 0) > 0]))
        self.assertEqual(st.tolist(), [3, 6])
        out, _ = self._run(w, (0.95, 3, 'orig'))
        self.assertEqual(int((out[0] > 0).sum()), 4)
        out, _ = self._run(w, (0.85, 4, 'orig'))
        self.assertEqual(int((out[0] > 0).sum()), 4)

    def test_kept_renormalizes_and_full_rows_untouched(self):
        g = torch.Generator().manual_seed(1)
        w = torch.rand(64, K, generator=g) ** 4 + 1e-3
        w = w / w.sum(-1, keepdim=True) * SCALE
        flat = torch.full((1, K), SCALE / K)          # every share 1/6: top-p 0.85 keeps all six
        w = torch.cat([w, flat])
        out, _ = self._run(w, (0.85, 3, 'kept'))
        torch.testing.assert_close(out.sum(-1), torch.full((65,), SCALE))
        self.assertTrue(torch.equal(out[-1], w[-1]))
        orig, _ = self._run(w, (0.85, 3, 'orig'))
        self.assertTrue(bool((orig.sum(-1) <= SCALE + 1e-6).all()))
        self.assertTrue(torch.equal(orig[-1], w[-1]))
        self.assertTrue(torch.equal((out > 0), (orig > 0)))   # same kept set either way


if __name__ == '__main__':
    unittest.main(verbosity=2)
