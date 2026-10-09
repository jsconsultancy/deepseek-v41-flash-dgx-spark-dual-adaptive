"""Disposable TP2 probe qualification; frozen history/map, no production DB writes.

First compare off vs computed shadow on the same prompt, checking full-prefill
demand and generated tokens for exact equality (including a cached extension).
Then exercise early apply on an isolated history. Run only with serving stopped.
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
os.environ['DSV41_PROBE_TOKENS'] = '1024'
os.environ['DSV41_PROBE_FRACTION'] = '.1'
os.environ['DSV41_PREFIX_CACHE'] = '1'

import torch
import engine.v41_engine as V
from engine.expert_profiles import mask_digest
from server.app import Tok, load_encoding_module, build_chat_prompt
from prefill_expert_probes import grade, retrieval_probe


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--widths', default='128,256,512,1024')
    ap.add_argument('--max-seq', type=int, default=65536)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    widths = [int(n) for n in args.widths.split(',')]
    assert all(32 <= n <= 1024 for n in widths)
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=args.max_seq, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    initial_mask = mask_digest(e.model_prune_mask)
    initial_history = tuple(a.clone() for a in (e.model._want_counts,e.model._want_mass))
    # Keep the normal planner, but freeze automatic adaptations for the matched
    # shadow/off comparison. The optional apply arm still calls apply_swaps.
    e.maintain_inline = lambda *a: 0
    e.maintain_decode = lambda *a: 0
    e.maintain_demand = lambda *a: False
    e.depth_policy = e.confidence_depth_policy = None
    demand = {}
    original_finish = e.predictive.finish
    def finish(engine):
        demand['sha256'] = hashlib.sha256(b''.join(a.cpu().numpy().tobytes()
            for a in (engine.model._req_counts,engine.model._req_mass))).hexdigest()
        original_finish(engine)
    e.predictive.finish = finish
    os.makedirs(args.out,exist_ok=True)
    owner = os.stat('/app/results')
    path = f'{args.out}/rank{e.ep.rank}.json'
    result = dict(config=e.config(), initial_mask=initial_mask, rows=[])
    def save():
        with open(path,'w') as f:json.dump(result,f,indent=1,ensure_ascii=False,default=str)
        os.chown(path,owner.st_uid,owner.st_gid)
    def render(body):return build_chat_prompt(body,enc,tok,False,75,e)[1]
    def chat(text):return render({'messages':[{'role':'user','content':text}]})
    def run(name,ids,mode,width=256,p=None,warm=False,cache=False):
        assert len(set(e.ep.gather_objects(digest([name,ids,mode,width,cache]))))==1
        e.predictive.mode=mode;e.predictive.pending=None
        if mode=='off':e.predictive.last=dict(mode='off',backend='probe',status='off')
        e.predictive.probe.max_tokens=width
        for live,baseline in zip((e.model._want_counts,e.model._want_mass),initial_history):
            live.copy_(baseline)
        if not cache:
            e._prefix_cache=None;e._prefix_snapshots.clear()
        output=[]
        torch.cuda.synchronize()
        started=time.perf_counter()
        for burst in e.generate(ids,max_tokens=p['limit'] if p else 16,temperature=0,seed=42,
                                stop_token_ids={eos},ignore_eos=p is None):output.extend(burst)
        torch.cuda.synchronize()
        text=tok.decode(output[:output.index(eos)] if eos in output else output)
        row=dict(name=name,mode=mode,width=width,warm=warm,prompt_tokens=len(ids),
                 output=output,text=text,wall_s=time.perf_counter()-started,
                 demand_sha256=demand['sha256'],mask_sha256=mask_digest(e.model_prune_mask),
                 prefill_s=e.last_stats['prefill_s'],prefix_cached_tokens=e.last_stats['prefix_cached_tokens'],
                 predictive=dict(e.predictive.last),grade=grade(p,text,enc) if p else None)
        assert len(set(e.ep.gather_objects(digest(output))))==1,'rank output disagreement'
        if mode!='apply':assert row['mask_sha256']==initial_mask,'frozen map changed'
        result['rows'].append(row);save()
        if e.ep.rank==0:print('PROBE_ROW '+json.dumps({k:row[k] for k in
            ('name','mode','width','prefill_s','prefix_cached_tokens','predictive','grade')}),flush=True)
        return row
    def equal(a,b):
        assert a['output']==b['output'],'shadow probe changed generation'
        assert a['demand_sha256']==b['demand_sha256'],'shadow probe changed real routing demand'
        assert a['prefix_cached_tokens']==b['prefix_cached_tokens'],'probe changed cache reuse'
    # Warm each probe shape before measuring; the first allocation/JIT is logged
    # separately, and cannot be represented as steady-state prediction cost.
    notes='\n'.join(f'Entry {i}: inventory quantity {i%17}, status pending.' for i in range(900))
    ids=chat(notes+'\nReply with READY only.')
    for width in widths:run('warm',ids,'shadow',width,warm=True)
    prompts=[]
    for i,pos in enumerate((.1,.5,.9)):
        p,ids=retrieval_probe(8192,pos,919100+i,render)
        prompts.append((p['name'],ids,p))
    p,ids=retrieval_probe(32768,.5,919200,render);prompts.append((p['name'],ids,p))
    code='\n'.join(f'def item_{i}(x):\n    return x + {i}' for i in range(650))
    prompts.append(('code',chat('Read this module:\n'+code+
        '\nWrite only Python source defining add(a, b) that returns their sum.'),
        dict(family='code',function='add',cases=[([2,3],5),([-4,1],-3)],limit=128)))
    for i,(name,ids,p) in enumerate(prompts):
        rows=[]
        arms=[('off',256)]+[('shadow',n) for n in widths]
        arms=arms[i%len(arms):]+arms[:i%len(arms)]
        for mode,width in arms:rows.append(run(name,ids,mode,width,p))
        baseline=next(r for r in rows if r['mode']=='off')
        for row in rows:equal(baseline,row)
    # Same cached prefix and new suffix on both arms. Probe state must not
    # overwrite the prefix, including the ring and decoder replay tail.
    base=chat(notes+'\nReply with READY only.')
    extension=base+chat('\n'.join(f'Update {i}: quantity {i%11}.' for i in range(120)))
    cached=[]
    for mode in ('off','shadow'):
        run('cache-seed',base,'off',warm=True)
        cached.append(run('cached-extension',extension,mode,widths[0],cache=True))
        assert cached[-1]['prefix_cached_tokens']==len(base)
    equal(*cached)
    # A full cache hit must still follow the same broadcast path, and skip.
    hit=run('full-cache-hit',extension,'shadow',widths[0],cache=True)
    assert hit['prefix_cached_tokens']==len(extension)
    assert hit['predictive']['status']=='skip_short_or_cached'
    if args.apply:
        p,ids=retrieval_probe(8192,.5,919300,render)
        row=run('apply-integration',ids,'apply',widths[0],p)
        assert row['predictive']['planned_swaps']>0,'apply path did not exercise swaps'
        assert row['predictive']['swaps']>0,'predicted swaps were not applied'
    result['summary']={}
    for n in widths:
        rows=[r for r in result['rows'] if not r['warm'] and r['mode']=='shadow'
              and r['width']==n and r['predictive'].get('probe_ms') is not None]
        summary=dict(n=len(rows))
        for label,key in (('median_prediction_ms','prediction_ms'),('median_ratio','prediction_prefill_ratio'),
                          ('median_precision','promotion_precision'),('median_recall','promotion_recall')):
            values=[r['predictive'][key] for r in rows if r['predictive'].get(key) is not None]
            summary[label]=statistics.median(values) if values else None
        result['summary'][str(n)]=summary
    save()
    if e.ep.rank==0:print('PROBE_SUMMARY '+json.dumps(result['summary']),flush=True)
    assert all(e.ep.gather_objects(True))
    sys.stdout.flush();os._exit(0)


if __name__=='__main__':main()
