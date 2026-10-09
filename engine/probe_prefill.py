"""Compute routing on excerpts or a full short suffix, without a history lookup.

Excerpts approximate context; full suffix discovery uses the restored prefix.
Both use current residents: discovering an unavailable expert does not fetch it.
Only the existing global planner may subsequently promote it.
"""
from contextlib import contextmanager
import copy
import hashlib
import math
import os
import time

import numpy as np
import torch

VERSION = 2


def select_spans(length, prefix, limit):
    """Keep the instruction head, two suffix windows and the suffix tail, in order.

    Half the budget goes to the tail (usually the latest question). Cached head
    rows supply approximate context but do not vote in the routing histogram.
    Offsets are into the original token sequence. Never select the entire suffix.
    """
    if not 0 <= prefix <= length or limit < 0:
        raise ValueError('invalid probe boundary or limit')
    limit = min(limit, max(0, length - prefix - 1))
    if limit < 4:
        return []
    head = max(1, limit // 8)
    tail = limit // 2
    middle = limit - head - tail
    stop = length - tail
    spans = [(0, head)]
    for i, size in enumerate((middle // 2, middle - middle // 2)):
        if not size:
            continue
        lo = max(prefix, head)
        start = lo + (stop - lo - size) * i // 2
        spans.append((start, start + size))
    spans.append((stop, length))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class ProbeDemand:
    """Separate counters; graph-captured request/history buffers stay untouched."""
    def __init__(self, layers, experts, rows, device):
        self.counts = torch.zeros((layers, experts), dtype=torch.float64, device=device)
        self.mass = torch.zeros_like(self.counts)
        self.rows = None if rows is None else torch.tensor(rows, dtype=torch.bool, device=device)

    def record(self, logits, scores, layer, k, start):
        want = logits.topk(k, dim=-1)[1]
        # Boolean indexing here is fine: this path is eager and never captured.
        if self.rows is not None:
            want = want[self.rows[start:start + len(want)]]
            scores = scores[self.rows[start:start + len(scores)]]
        score = scores.gather(1, want)
        flat = want.reshape(-1)
        self.counts[layer].scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
        self.mass[layer].scatter_add_(0, flat, score.double().reshape(-1))


_MISSING = object()


@contextmanager
def isolated_model(model, caches, demand, *, reset_cache=True):
    """Swap objects, not tensor contents: preserve live KV, hash history and graphs.

    Prefill FFN graphs retain the live demand-buffer addresses, so disable replay
    in this probe. Do not invalidate or recapture those graphs. Read-only expert
    weights and resident route tables are shared, as are the Engram row caches.
    """
    overrides = dict(c=caches, stats=dict(model.stats), _probe_demand=demand,
                     _prefix_replay_only=False, tap=None, prefill_graphs=None,
                     prefill_moe_timing=None, user_prompt=None, critical_stream=None,
                     prefill_replica_counts=None, prefill_replica_lut=None,
                     prefill_replica_routes=None)
    overwritten = ('_rep', '_rep_end', 'last_h', 'last_pre_mix', 'image_mask',
                   'engram_mask', '_moe_start')
    saved = {name: getattr(model, name, _MISSING) for name in (*overrides, *overwritten)}
    from engine import model as M
    marks = len(M._PHASE_MARKS)
    try:
        for name, value in overrides.items():
            setattr(model, name, value)
        if reset_cache:
            caches.len = 0
            caches.pending = dict.fromkeys(caches.pending)
            caches._chunk_inputs.clear()
            model.begin_prompt()
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                if hasattr(model, name):
                    delattr(model, name)
            else:
                setattr(model, name, value)
        # Do not charge diagnostic attention events to the real request.
        del M._PHASE_MARKS[marks:]


def suffix_checkpoint(model, prefix):
    """Bounded ring backup; compressed prefix rows are append-only.

    Suffix forwards append compressed/index rows at or beyond prefix//ratio.
    The unpaired compressor input is copied separately. Restoring the valid
    window, pending state and length makes appended rows unreachable; the real
    pass overwrites them before use. No complete KV-cache copy is needed.
    """
    from engine.model import RING
    c = model.c
    if c.len != prefix:
        raise ValueError('suffix discovery requires the restored prefix boundary')
    slots = model._positions[max(0, prefix-model.args.window_size):prefix] % RING
    return dict(slots=slots, win=[w[slots].clone() for w in c.win],
                pending={L:None if v is None else tuple(t.clone() for t in v)
                         for L,v in c.pending.items()})


@contextmanager
def isolated_suffix(model, prefix, demand, checkpoint):
    """Discover using the real cached context, then restore its valid boundary."""
    c = model.c
    pending, chunks = c.pending, c._chunk_inputs
    try:
        with isolated_model(model, c, demand, reset_cache=False):
            c.pending = checkpoint['pending']
            c._chunk_inputs = {}
            if prefix:
                # _rep_keep appends/pops lists; preserve the original containers.
                model._rep = {k:list(v) for k,v in model._rep.items()}
            else:
                model.begin_prompt()
            try:
                yield
            finally:
                for w, values in zip(c.win, checkpoint['win']):
                    w.index_copy_(0, checkpoint['slots'], values)
    finally:
        c.len, c.pending, c._chunk_inputs = prefix, pending, chunks


def excerpt_hashes(hash_state, tokens, spans, device):
    """Preserve original n-grams at excerpt boundaries without hashing old context.

    Hash each excerpt with its original short lookback. Joining excerpts must not
    invent Engram n-grams between otherwise unrelated source positions. All
    immutable hash buffers are shared; only the tiny cache belongs to this copy.
    """
    if hash_state is None:
        return None
    state = copy.copy(hash_state)
    state._buffers = dict(hash_state._buffers)
    lookback = state.layout.max_ngram_size - 1
    size = max(end - start for start, end in spans) + lookback
    state.cache = torch.empty((1, size), dtype=torch.int64, device=device)
    parts = []
    for start, end in spans:
        lo = max(0, start - lookback)
        ids = torch.tensor(tokens[lo:end], dtype=torch.long, device=device)
        parts.append(state(ids[None], 0)[0, start-lo:].clone())
    return torch.cat(parts)


class SmallProbe:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.max_tokens = int(env.get('DSV41_PROBE_TOKENS', '256'))
        self.fraction = float(env.get('DSV41_PROBE_FRACTION', '.05'))
        self.min_tokens = int(env.get('DSV41_PROBE_MIN_TOKENS', '32'))
        self.target_ratio = float(env.get('DSV41_PROBE_TARGET_RATIO', '.1'))
        self.two_pass_below = int(env.get('DSV41_PROBE_TWO_PASS_BELOW', '0'))
        if (not 4 <= self.min_tokens <= self.max_tokens <= 1024
                or not math.isfinite(self.fraction) or not 0 < self.fraction <= .1
                or not math.isfinite(self.target_ratio) or not 0 < self.target_ratio <= 1
                or not 0 <= self.two_pass_below <= 1024):
            raise ValueError('invalid small-probe token or time budget')
        self.caches = None

    def boot_fields(self):
        return dict(predictive_probe_version=VERSION, predictive_probe_tokens=self.max_tokens,
                    predictive_probe_fraction=self.fraction, predictive_probe_min_tokens=self.min_tokens,
                    predictive_probe_target_ratio=self.target_ratio,
                    predictive_probe_two_pass_below=self.two_pass_below)

    def predict(self, engine, tokens, prefix):
        """All ranks enter, including skips. Agree on readiness before any forward.

        A failure *during* the distributed model pass is fatal, just as a partial
        expert swap is: attempting another collective could wedge the peer.
        """
        from engine.v41_engine import ADAPT
        started = time.perf_counter()
        packet = None
        if engine.ep.rank == 0:
            new = len(tokens) - prefix
            status = 'probe'
            spans = []
            kind = 'excerpt'
            if getattr(engine, '_images', None) is not None:
                status = 'skip_vision'
            elif 0 < new < self.two_pass_below:
                kind, status, spans = 'full_suffix', 'two_pass_suffix', [(prefix,len(tokens))]
            elif new < ADAPT.swap_prefill_min:
                status = 'skip_short_or_cached'
            else:
                limit = min(self.max_tokens, int(new * self.fraction))
                if limit < self.min_tokens:
                    status = 'skip_probe_budget'
                else:
                    spans = select_spans(len(tokens), prefix, limit)
            packet = dict(status=status, kind=kind, spans=spans,
                          ids=[t for start, end in spans for t in tokens[start:end]],
                          rows=[i >= prefix for start, end in spans for i in range(start, end)])
        packet = engine.ep.broadcast_obj(packet)
        info = dict(status=packet['status'], probe_tokens=len(packet['ids']),
                    probe_vote_tokens=sum(packet['rows']), probe_spans=packet['spans'],
                    prediction_kind=packet['kind'], two_pass_below=self.two_pass_below,
                    probe_target_ratio=self.target_ratio)
        if not packet['ids']:
            return None, info
        prepared, error = None, None
        try:
            from engine.model import Caches
            full = packet['kind'] == 'full_suffix'
            checkpoint = suffix_checkpoint(engine.model, prefix) if full else None
            if not full and (self.caches is None or self.caches.max_seq < self.max_tokens + 8):
                self.caches = Caches(engine.model.args, self.max_tokens + 8, engine.device)
            ids = torch.tensor(packet['ids'], dtype=torch.long, device=engine.device)
            hashes = excerpt_hashes(engine.model.hash_state, tokens, packet['spans'], engine.device)
            demand = ProbeDemand(engine.args.n_layers, engine.args.n_routed_experts,
                                 None if full else packet['rows'], engine.device)
            prepared = ids, hashes, demand, full, checkpoint
        except Exception as exc:
            error = type(exc).__name__
        errors = engine.ep.gather_objects(error)
        if any(errors):
            info.update(status='probe_prepare_error', probe_errors=errors)
            return None, info
        try:
            from engine.model import MAX_CHUNK
            ids, hashes, demand, full, checkpoint = prepared
            context = (isolated_suffix(engine.model, prefix, demand, checkpoint) if full else
                       isolated_model(engine.model, self.caches, demand))
            spans = engine._prefill_spans(len(tokens),prefix) if full else [(s,min(s+MAX_CHUNK,len(ids)))
                                                                       for s in range(0,len(ids),MAX_CHUNK)]
            origin = prefix if full else 0
            with context:
                for start,end in spans:
                    a,b = start-origin,end-origin
                    engine.model.forward(ids[a:b], start, prefill=True, need_logits=False,
                                         encoder_only=engine.swa_replay,
                                         hashes=None if hashes is None else hashes[a:b])
                if engine.swa_replay:
                    engine.model.decoder_replay(need_logits=False)
                    info['probe_replay_tokens'] = min(engine.args.window_size, len(tokens) if full else len(ids))
                torch.cuda.synchronize(engine.device)
                arrays = tuple(a.cpu().numpy().copy() for a in (demand.counts, demand.mass))
            signature = hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
            elapsed = 1000 * (time.perf_counter() - started)
        except Exception:
            import traceback
            traceback.print_exc()
            os._exit(1)
        # Reached by both ranks unconditionally after every completed probe.
        reports = engine.ep.gather_objects((signature, elapsed))
        info['probe_ms'] = round(max(ms for _, ms in reports), 3)
        if len({sig for sig, _ in reports}) != 1:
            info['status'] = 'probe_rank_disagreement'
            return None, info
        if any(not np.isfinite(a).all() or (a < 0).any() or (a.sum(axis=1) <= 0).any()
               for a in arrays):
            info['status'] = 'probe_invalid_demand'
            return None, info
        return arrays, info
