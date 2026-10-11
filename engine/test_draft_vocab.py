"""DSV41_DRAFT_VOCAB: the listed draft head's id mapping, on CPU with a simulated two-rank gather.

    python3 -m unittest engine.test_draft_vocab

What is pinned here is the part that fails silently: which token id a rank's column stands for.
The two ranks run in threads and exchange through a fake `all_gather_fast` with a barrier, so the
split chain's five exchanges happen in lockstep as they do across the pair. The reference chain is
built from the same per-rank logits and Markov products the head computes, so the comparison is
exact and tests the mapping and the tie rule, not GEMM rounding. Whether a listed vocabulary keeps
acceptance is the two-node A/B's question (tools/bench_accept_ab_tp.py `vocab_*`).
"""
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tools'), os.path.join(HERE, '..')]
import v41_ref as R  # noqa: E402
from engine import draft_vocab  # noqa: E402
from engine.tensor_parallel import ListedVocabHead, VocabParallelHead  # noqa: E402

V, K, RK, WORLD, ROWS = 512, 64, 16, 2, 128


class Pair:
    """A two-thread stand-in for comm.all_gather_fast: rank-major concatenation along dim 0."""

    def __init__(self):
        self.barrier = threading.Barrier(WORLD)
        self.sent = [None] * WORLD
        self.rank_of = {}

    def gather(self, out, send, group=None):
        rank = self.rank_of[threading.get_ident()]
        self.sent[rank] = send.detach().clone()
        self.barrier.wait()
        out.copy_(torch.cat(self.sent, 0).reshape(out.shape))
        self.barrier.wait()

    def run(self, fn):
        results, errors = [None] * WORLD, []

        def body(rank):
            self.rank_of[threading.get_ident()] = rank
            try:
                results[rank] = fn(rank)
            except BaseException as e:  # surface a rank's failure instead of hanging the peer
                errors.append(e)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(WORLD)]
        with patch('engine.tensor_parallel.comm.all_gather_fast', self.gather), \
                patch('engine.tensor_parallel.collective_group', return_value=None):
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        if errors:
            raise errors[0]
        return results


def _weights(seed=3):
    g = torch.Generator().manual_seed(seed)
    head = (torch.randn(V, K, generator=g) * 0.1).bfloat16()
    embed = (torch.randn(V, RK, generator=g) * 0.5).bfloat16()
    markov = (torch.randn(V, RK, generator=g) * 0.5).bfloat16()
    x = torch.randn(5, K, generator=g).bfloat16()
    return head, embed, markov, x


def _listed(seed=5):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(V, generator=g).tolist()
    return draft_vocab.select(order, ROWS, WORLD, V)


def _linear(e, w):
    return F.linear(e, w)


def _reference_chain(x, embed, parts, anchor):
    """Greedy chain over the listed ids, from each rank's own (head rows, Markov rows, ids)."""
    base = torch.cat([R.head_logits(x, h) for h, _, _ in parts], dim=-1)
    ids = torch.cat([i for _, _, i in parts])
    prev, out = int(anchor), []
    for i in range(x.shape[0]):
        e = embed[torch.tensor([prev])]
        lg = base[i] + torch.cat([_linear(e, m).float()[0] for _, m, _ in parts])
        top = lg.max()
        prev = int(ids[(lg == top).nonzero()[:, 0]].min())   # ties: the lowest id
        out.append(prev)
    return out


class Selection(unittest.TestCase):
    def test_runs_are_contiguous_and_ascending(self):
        ids = _listed()
        self.assertEqual(ids.numel(), WORLD * ROWS)
        self.assertTrue(bool((ids[1:] > ids[:-1]).all()))
        r0, r1 = (draft_vocab.rank_ids(ids, r, WORLD) for r in range(WORLD))
        self.assertLess(int(r0.max()), int(r1.min()))

    def test_short_ranking_fills_lowest_unlisted(self):
        ids = draft_vocab.select([500, 7, 500, 9], 4, 2, V)
        self.assertEqual(ids.tolist(), [0, 1, 2, 3, 4, 7, 9, 500])

    def test_ranking_file_rules(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as fh:
            fh.write('# header\n5 3 5  # dup\n\n9\n')
            name = fh.name
        try:
            self.assertEqual(draft_vocab.ranking(name, V), [5, 3, 9])
            with open(name, 'a') as fh:
                fh.write(f'{V}\n')
            with self.assertRaisesRegex(ValueError, 'outside the vocabulary'):
                draft_vocab.ranking(name, V)
        finally:
            os.unlink(name)

    def test_digest_depends_on_list(self):
        a, b = _listed(5), _listed(6)
        self.assertEqual(draft_vocab.digest(a), draft_vocab.digest(a.clone()))
        self.assertNotEqual(draft_vocab.digest(a), draft_vocab.digest(b))

    def test_rows_setting(self):
        with patch.dict(os.environ, {'DSV41_DRAFT_VOCAB_ROWS': '100'}):
            with self.assertRaisesRegex(ValueError, 'multiple of'):
                draft_vocab.rows()
        with patch.dict(os.environ, {'DSV41_DRAFT_VOCAB_ROWS': ''}):
            self.assertEqual(draft_vocab.rows(), draft_vocab.DEFAULT_ROWS)


class ListedChain(unittest.TestCase):
    def setUp(self):
        self.head, self.embed, self.markov, self.x = _weights()
        self.ids = _listed()
        self.anchor = torch.tensor([17])

    def _parts(self):
        return [(self.head[r], self.markov[r], r) for r in
                (draft_vocab.rank_ids(self.ids, k, WORLD) for k in range(WORLD))]

    def _heads(self):
        return [ListedVocabHead(h, WORLD, self.ids, V, k) for k, (h, _, _) in enumerate(self._parts())]

    def test_split_chain_matches_listed_reference(self):
        heads, parts = self._heads(), self._parts()
        want = _reference_chain(self.x, self.embed, parts, self.anchor)
        got = Pair().run(lambda r: heads[r].tp_markov_greedy(
            self.x, self.embed, parts[r][1], self.anchor, _linear, r)[0].tolist())
        self.assertEqual(got[0], got[1])
        self.assertEqual(got[0], want)
        listed = set(self.ids.tolist())
        self.assertTrue(all(t in listed for t in got[0]))

    def test_tie_goes_to_lower_id(self):
        r0, r1 = (draft_vocab.rank_ids(self.ids, k, WORLD) for k in range(WORLD))
        lo, mid, hi = int(r0[3]), int(r0[90]), int(r1[40])
        x = self.x[:1]
        # make three listed ids score identically and higher than anything else at position 0
        for t in (lo, mid, hi):
            self.head[t] = x[0] * 4
            self.markov[t] = 0
        parts = self._parts()
        heads = [ListedVocabHead(h, WORLD, self.ids, V, k) for k, (h, _, _) in enumerate(parts)]
        got = Pair().run(lambda r: heads[r].tp_markov_greedy(
            x, self.embed, parts[r][1], self.anchor, _linear, r)[0].tolist())
        self.assertEqual(got, [[lo], [lo]])
        # without the rank-0 pair, the cross-rank tie must still pick rank 0's lower id
        self.head[lo] = 0
        parts = self._parts()
        heads = [ListedVocabHead(h, WORLD, self.ids, V, k) for k, (h, _, _) in enumerate(parts)]
        got = Pair().run(lambda r: heads[r].tp_markov_greedy(
            x, self.embed, parts[r][1], self.anchor, _linear, r)[0].tolist())
        self.assertEqual(got, [[mid], [mid]])

    def test_logits_scatter_to_full_vocabulary(self):
        heads, parts = self._heads(), self._parts()
        got = Pair().run(lambda r: heads[r].tp_logits(self.x))
        self.assertTrue(torch.equal(got[0], got[1]))
        self.assertEqual(tuple(got[0].shape), (5, V))
        listed = torch.zeros(V, dtype=torch.bool)
        listed[self.ids] = True
        self.assertTrue(bool(torch.isneginf(got[0][:, ~listed]).all()))
        for h, _, ids in parts:
            self.assertTrue(torch.equal(got[0][:, ids], R.head_logits(self.x, h)))

    def test_candidates_carry_token_ids(self):
        heads, parts = self._heads(), self._parts()
        got = Pair().run(lambda r: heads[r].tp_candidates(self.x, 8))
        ids, vals = got[0]
        self.assertTrue(torch.equal(ids, got[1][0]) and torch.equal(vals, got[1][1]))
        full = torch.full((5, V), float('-inf'))
        for h, _, i in parts:
            full[:, i] = R.head_logits(self.x, h)
        self.assertTrue(torch.equal(vals, full.gather(1, ids)))
        for row in range(5):
            # each rank's own top 8, as token ids: the union is not the global top 16
            want = set()
            for h, _, i in parts:
                want |= set(i[R.head_logits(self.x, h)[row].topk(8).indices].tolist())
            self.assertEqual(set(ids[row].tolist()), want)


class FullVocabularyUnchanged(unittest.TestCase):
    """The refactor behind the listed head: the plain split chain still maps column + offset."""

    def test_plain_split_chain(self):
        head, embed, markov, x = _weights(9)
        anchor = torch.tensor([3])
        halves = [torch.arange(r * V // 2, (r + 1) * V // 2) for r in range(WORLD)]
        parts = [(head[h], markov[h], h) for h in halves]
        heads = [VocabParallelHead(h, WORLD) for h, _, _ in parts]
        want = _reference_chain(x, embed, parts, anchor)
        got = Pair().run(lambda r: heads[r].tp_markov_greedy(
            x, embed, parts[r][1], anchor, _linear, r)[0].tolist())
        self.assertEqual(got, [want, want])


if __name__ == '__main__':
    unittest.main(verbosity=2)
