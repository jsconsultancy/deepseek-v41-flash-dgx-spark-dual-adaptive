"""Disposable TP2 CS/Python/news -> medical switch, with matched starting state.

Run with serving stopped and an isolated copied demand DB. Normal post-prefill
and decode correction remain enabled. Every measured arm restores exact expert
slot ownership, masks, LUT and adaptive counters from the warmed mixed baseline.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING',
            'DSV41_STEP_TIMING', 'DSV41_PREFILL_GRAPHS', 'DSV41_ATTN_TIMING'):
    os.environ[key] = '0'
os.environ.pop('DSV41_PROBE_PREFILL', None)  # This driver explicitly selects off/shadow/apply.
os.environ['DSV41_PREDICTIVE_PREFILL'] = 'shadow'
os.environ['DSV41_PREDICTIVE_BACKEND'] = 'probe'
os.environ['DSV41_PROBE_TOKENS'] = '256'
os.environ['DSV41_PROBE_FRACTION'] = '.1'
os.environ['DSV41_PREFIX_CACHE'] = '1'

import numpy as np
import torch
import engine.v41_engine as V
from engine.expert_profiles import mask_digest
from engine.predictive_prefill import normalize
from server.app import Tok, load_encoding_module, build_chat_prompt
from prefill_expert_probes import grade
from topic_shift_probes import warm_probe, medical_probe, python_control


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--warm-rounds', type=int, default=4)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    e = V.V41Engine(os.environ['MODEL_DIR'], max_seq=65536, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    assert e.dynamic_experts and e.ep.tensor_parallel
    tok, enc = Tok(os.environ['MODEL_DIR']), load_encoding_module(os.environ['MODEL_DIR'])
    eos = tok.token_to_id(enc.eos_token)
    e.depth_policy = e.confidence_depth_policy = None
    render = lambda body: build_chat_prompt(body, enc, tok, False, 75, e)[1]
    os.makedirs(args.out, exist_ok=True)
    owner = os.stat('/app/results')
    result = dict(config=e.config(), warm_domains=['cs', 'python', 'news'], rows=[])
    path = f'{args.out}/rank{e.ep.rank}.json'
    def save():
        with open(path, 'w') as f:
            json.dump(result, f, indent=1, ensure_ascii=False, default=str)
        os.chown(path, owner.st_uid, owner.st_gid)
    plans, observation = [], {}
    phase = ['warm']
    apply_original, begin_original, finish_original = e.apply_swaps, e.predictive.begin, e.predictive.finish
    def apply(swaps):
        started = time.perf_counter()
        n = apply_original(swaps)
        if swaps:
            plans.append((phase[0], list(swaps), 1000 * (time.perf_counter() - started)))
        return n
    e.apply_swaps = apply
    def begin(engine, tokens, prefix):
        torch.cuda.synchronize()
        observation.clear()
        observation['start'] = time.perf_counter()
        phase[0] = 'early'
        begin_original(engine, tokens, prefix)
        torch.cuda.synchronize()
        observation['real_start'] = time.perf_counter()
        observation['mask'] = np.stack([engine.model_prune_mask[L].cpu().numpy()
                                       for L in range(engine.args.n_layers)])
        phase[0] = 'real_prefill'
    def finish(engine):
        counts, mass = [a.cpu().numpy().copy() for a in (engine.model._req_counts, engine.model._req_mass)]
        now = time.perf_counter()
        mask = observation['mask']
        observation.update(counts=counts, mass=mass,
            demand_sha256=hashlib.sha256(counts.tobytes()+mass.tobytes()).hexdigest(),
            first_prefill_miss=float((counts * ~mask).sum() / max(counts.sum(), 1)),
            first_prefill_score_miss=float((mass * ~mask).sum() / max(mass.sum(), 1e-20)),
            ordinary_prefill_ms=1000 * (now - observation['real_start']),
            before_correction_ms=1000 * (now - observation['start']))
        finish_original(engine)
        phase[0] = 'post_prefill_or_decode'
    e.predictive.begin, e.predictive.finish = begin, finish

    def run(p, ids, mode, width, label):
        # Every arm and request must agree before entering model collectives.
        assert len(set(e.ep.gather_objects(digest([p, ids, mode, width, label])))) == 1
        e.predictive.mode, e.predictive.pending = mode, None
        if mode == 'off':
            e.predictive.last = dict(mode='off', backend='probe', status='off')
        e.predictive.probe.max_tokens = width
        e._prefix_cache = None
        e._prefix_snapshots.clear()
        plans.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = []
        for burst in e.generate(ids, max_tokens=p['limit'], temperature=0, seed=42, stop_token_ids={eos}):
            output.extend(burst)
        torch.cuda.synchronize()
        text = tok.decode(output[:output.index(eos)] if eos in output else output)
        row = dict(name=p['name'], label=label, mode=mode, width=width,
                   prompt_tokens=len(ids), prefix_cached_tokens=e.last_stats['prefix_cached_tokens'],
                   output=output, text=text, grade=grade(p, text, enc),
                   wall_s=time.perf_counter()-started, prefill_s=e.last_stats['prefill_s'],
                   predictive=dict(e.predictive.last), swaps_by_phase=[dict(phase=k, count=len(s), ms=t)
                                                                      for k, s, t in plans],
                   **{k:observation[k] for k in ('demand_sha256','first_prefill_miss',
                       'first_prefill_score_miss','ordinary_prefill_ms','before_correction_ms')})
        assert row['prefix_cached_tokens'] == 0
        parity = digest([output, row['demand_sha256'], mask_digest(e.model_prune_mask)])
        assert len(set(e.ep.gather_objects(parity))) == 1, 'rank numerical disagreement'
        result['rows'].append(row)
        save()
        if e.ep.rank == 0:
            print('SHIFT_ROW ' + json.dumps({k:row[k] for k in row if k not in ('output','text')}), flush=True)
        return row, (observation['counts'].copy(), observation['mass'].copy(), observation['mask'].copy())

    # Warm both probe shapes without adapting or adding their data to the mixed
    # history. First allocation/JIT cost is separately visible in these rows.
    inline = e.maintain_inline
    e.maintain_inline = lambda *a: 0
    history = [a.clone() for a in (e.model._want_counts,e.model._want_mass)]
    p, ids = warm_probe('python', 99100, 8192, render)
    for width in (64, 256):
        run(p, ids, 'shadow', width, 'jit-warmup')
        for live, saved in zip((e.model._want_counts,e.model._want_mass),history):
            live.copy_(saved)
    e.maintain_inline = inline
    for round_ in range(args.warm_rounds):
        for j, domain in enumerate(result['warm_domains']):
            p, ids = warm_probe(domain, 99200+round_*3+j, 8192, render)
            run(p, ids, 'off', 64, 'mixed-residency-training')
    # Finish fitting the warmed history, still under the ordinary gain/cap/floor
    # policy. All plans are authoritative on rank 0 and broadcast unconditionally.
    settle = []
    for _ in range(16):
        swaps = e.ep.broadcast_obj(e.plan_swaps(max_swaps=V.ADAPT.swap_max) if e.ep.rank == 0 else None)
        if not swaps:
            break
        settle.append(apply_original(swaps))
    else:
        raise AssertionError('mixed residency did not settle in bounded passes')
    arrays = [e.model._want_counts,e.model._want_mass,e.model._miss_tot,e.model._miss_mass_tot,
              e.model._req_counts,e.model._req_mass,e.model._miss_phase,*e.model._want_phase]
    baseline = [a.clone() for a in arrays]
    scalars = {k:getattr(e,k) for k in ('_swap_baseline','_requests_since_plan','expert_generation')}
    directory, slot_keys, lut = list(e.store.lru.items()), dict(e.store.slot_key), e.fast.lut.clone()
    warmed_mask = mask_digest(e.model_prune_mask)
    history_hash = hashlib.sha256(b''.join(a.cpu().numpy().tobytes() for a in baseline[:2])).hexdigest()
    result.update(warmed_mask=warmed_mask, warmed_history_sha256=history_hash, settle_swaps=settle)
    assert len(set(e.ep.gather_objects(digest([warmed_mask,history_hash])))) == 1
    if e.ep.rank == 0:
        warm_path = f'{args.out}/warmed-state.npz'
        np.savez(warm_path, counts=baseline[0].cpu().numpy(), mass=baseline[1].cpu().numpy(),
                 mask=np.stack([e.model_prune_mask[L].cpu().numpy() for L in range(e.args.n_layers)]))
        os.chown(warm_path, owner.st_uid, owner.st_gid)
    save()

    def restore():
        # Undo complete batches in reverse order, including normal correction.
        # Reversing a global batch also reverses its floor-safe transfer order.
        for _, swaps, _ in reversed(plans):
            inverse = [(Li,ei,Lo,eo,g) for Lo,eo,Li,ei,g in reversed(swaps)]
            assert len(set(e.ep.gather_objects(digest(inverse)))) == 1
            apply_original(inverse)
        for live, saved in zip(arrays, baseline):
            live.copy_(saved)
        for k, value in scalars.items():
            setattr(e, k, value)
        e.store.lru.clear()
        e.store.lru.update(directory)
        assert e.store.slot_key == slot_keys, 'expert sector placement was not restored'
        assert torch.equal(e.fast.lut, lut), 'decode LUT was not restored'
        assert mask_digest(e.model_prune_mask) == warmed_mask, 'warmed mask was not restored'
        assert all(torch.equal(a,b) for a,b in zip(arrays,baseline)), 'adaptive history was not restored'
        e._prefix_cache = None
        e._prefix_snapshots.clear()
        assert all(e.ep.gather_objects(True))

    cases = [python_control(8192, render)]
    cases += [medical_probe(kind, 99300+i, 8192, render) for i,kind in enumerate(('chart','trial','trend'))]
    cases += [medical_probe('trial', 99400, 32768, render)]
    comparisons = []
    for i,(p,ids) in enumerate(cases):
        arms = [('off',64),('shadow',64),('apply',64),('apply',256)]
        arms = arms[i % len(arms):] + arms[:i % len(arms)]
        rows, demands, masks = {}, {}, {}
        for mode,width in arms:
            row, (counts,mass,mask) = run(p,ids,mode,width,'first-request-after-mixed-training')
            key = f'{mode}-{width}'
            rows[key], demands[key], masks[key] = row, (counts,mass), mask
            restore()
        off, shadow = rows['off-64'], rows['shadow-64']
        assert off['output'] == shadow['output'], 'shadow changed answer'
        assert off['demand_sha256'] == shadow['demand_sha256'], 'shadow changed actual demand'
        reference = normalize(baseline[1].cpu().numpy())
        actual = normalize(demands['off-64'][1])
        comparison = dict(name=p['name'], prompt_tokens=len(ids),
                          history_demand_tv=float(np.abs(actual-reference).sum(axis=1).mean()/2),
                          arms={})
        for key,row in rows.items():
            # Fixed off-arm demand under the early-applied map distinguishes
            # resident coverage from changes to hidden states caused by swapping.
            fixed_counts, fixed_mass = demands['off-64']
            comparison['arms'][key] = dict(
                miss=row['first_prefill_miss'], score_miss=row['first_prefill_score_miss'],
                fixed_demand_miss=float((fixed_counts*~masks[key]).sum()/fixed_counts.sum()),
                fixed_demand_score_miss=float((fixed_mass*~masks[key]).sum()/fixed_mass.sum()),
                prefill_s=row['prefill_s'], before_correction_ms=row['before_correction_ms'],
                prediction_ms=row['predictive'].get('prediction_ms',0),
                load_ms=row['predictive'].get('load_ms',0),
                early_swaps=row['predictive'].get('swaps',0), correct=row['grade']['pass_'])
        comparisons.append(comparison)
        result['comparisons'] = comparisons
        save()
        if e.ep.rank == 0:
            print('SHIFT_COMPARISON ' + json.dumps(comparison), flush=True)
    summary = {}
    medical = [c for c in comparisons if c['name'].startswith('medical') and c['prompt_tokens'] < 10000]
    for arm in ('off-64','shadow-64','apply-64','apply-256'):
        rows = [c['arms'][arm] for c in medical]
        summary[arm] = dict(n=len(rows), correct=sum(r['correct'] for r in rows),
            **{k:statistics.median(r[k] for r in rows) for k in
               ('miss','score_miss','fixed_demand_miss','fixed_demand_score_miss',
                'prefill_s','before_correction_ms','prediction_ms','load_ms','early_swaps')})
    result['medical_8k_summary'] = summary
    save()
    if e.ep.rank == 0:
        print('SHIFT_SUMMARY ' + json.dumps(summary), flush=True)
    assert all(e.ep.gather_objects(True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
