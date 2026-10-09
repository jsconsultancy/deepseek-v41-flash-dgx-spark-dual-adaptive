"""Disposable TP2 two-pass suffix gate; copied history, serving must be stopped."""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path[:0] = ['/app','/app/tools']
for key in ('DSV41_PREFIX_DISK','DSV41_PREFIX_RESPONSE','DSV41_GPU_TIMING',
            'DSV41_STEP_TIMING','DSV41_PREFILL_GRAPHS','DSV41_ATTN_TIMING'):
    os.environ[key]='0'
os.environ.pop('DSV41_PROBE_PREFILL', None)  # This driver explicitly selects off/shadow/apply.
os.environ.update(DSV41_PREDICTIVE_PREFILL='shadow',DSV41_PREDICTIVE_BACKEND='probe',
                 DSV41_PROBE_TOKENS='64',DSV41_PROBE_FRACTION='.1',
                 DSV41_PROBE_TWO_PASS_BELOW='500',DSV41_PREFIX_CACHE='1')

import numpy as np
import torch
import engine.v41_engine as V
from engine.expert_profiles import mask_digest
from server.app import Tok,load_encoding_module,build_chat_prompt
from prefill_expert_probes import grade
from topic_shift_probes import warm_probe


def signature(items):
    return hashlib.sha256(b''.join(a.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                 for a in items)).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True)
    args=ap.parse_args();V.save_prune_db=lambda *a,**k:None
    e=V.V41Engine(os.environ['MODEL_DIR'],max_seq=65536,arena_gb=90.2,
        trace_stats='/app/results/trace-union/stats/coverage.json',spec=True,prune_keep=.61,
        transient_slots=8,keep_free_gb=6)
    tok,enc=Tok(os.environ['MODEL_DIR']),load_encoding_module(os.environ['MODEL_DIR'])
    eos=tok.token_to_id(enc.eos_token)
    render=lambda body:build_chat_prompt(body,enc,tok,False,75,e)[1]
    chat=lambda text:render({'messages':[{'role':'user','content':text}]})
    history=[a.clone() for a in (e.model._want_counts,e.model._want_mass)]
    initial_mask=mask_digest(e.model_prune_mask)
    inline=e.maintain_inline
    e.maintain_inline=lambda *a:0;e.maintain_decode=lambda *a:0;e.maintain_demand=lambda *a:False
    e.depth_policy=e.confidence_depth_policy=None
    result=dict(config=e.config(),rows=[]);os.makedirs(args.out,exist_ok=True)
    owner=os.stat('/app/results');path=f'{args.out}/rank{e.ep.rank}.json'
    def save():
        with open(path,'w') as f:json.dump(result,f,indent=1,default=str)
        os.chown(path,owner.st_uid,owner.st_gid)
    measured={};finish=e.predictive.finish;begin=e.predictive.begin
    def before(engine,tokens,prefix):
        # Check every valid cached prefix row, compressor input and replay tail.
        # Discovery can append rows after the boundary; those are unreachable.
        if engine.predictive.mode!='off' and 0<len(tokens)-prefix<500:
            c,m=engine.caches,engine.model
            slots=m._positions[max(0,prefix-engine.args.window_size):prefix]%V.M_.RING
            live=[w[slots] for w in c.win]
            for L in c.ckv:
                n=prefix//engine.args.compress_ratios[L]
                live.extend((c.ckv[L][:n],c.ik[L][:n]))
            live.extend(t for v in c.pending.values() if v is not None for t in v)
            if prefix:live.extend(t for values in m._rep.values() for t in values if t is not None)
            old=signature(live)
            pending=c.pending;rep=m._rep;chunks=c._chunk_inputs
            begin(engine,tokens,prefix)
            restored=[w[slots] for w in c.win]
            for L in c.ckv:
                n=prefix//engine.args.compress_ratios[L]
                restored.extend((c.ckv[L][:n],c.ik[L][:n]))
            restored.extend(t for v in c.pending.values() if v is not None for t in v)
            if prefix:restored.extend(t for values in m._rep.values() for t in values if t is not None)
            assert signature(restored)==old,'discovery changed valid prefix state'
            assert c.len==prefix and c.pending is pending and m._rep is rep and c._chunk_inputs is chunks
        else:begin(engine,tokens,prefix)
    def after(engine):
        measured['demand']=signature((engine.model._req_counts,engine.model._req_mass))
        finish(engine)
    e.predictive.begin=before;e.predictive.finish=after
    def run(name,ids,mode,cached=0,p=None):
        packet=json.dumps([name,ids,mode,cached],sort_keys=True)
        assert len(set(e.ep.gather_objects(hashlib.sha256(packet.encode()).hexdigest())))==1
        e.predictive.mode=mode;e.predictive.pending=None
        if mode=='off':e.predictive.last=dict(mode='off',status='off')
        for a,b in zip((e.model._want_counts,e.model._want_mass),history):a.copy_(b)
        if not cached:e._prefix_cache=None;e._prefix_snapshots.clear()
        output=[];torch.cuda.synchronize();start=time.perf_counter()
        for burst in e.generate(ids,max_tokens=p['limit'] if p else 8,temperature=0,seed=42,
            stop_token_ids={eos},ignore_eos=p is None):output.extend(burst)
        torch.cuda.synchronize()
        text=tok.decode(output[:output.index(eos)] if eos in output else output)
        row=dict(name=name,mode=mode,prompt_tokens=len(ids),new_tokens=len(ids)-cached,
            prefix_cached_tokens=e.last_stats['prefix_cached_tokens'],prefill_s=e.last_stats['prefill_s'],
            wall_s=time.perf_counter()-start,output=output,text=text,
            demand_sha256=measured['demand'],history_sha256=signature((e.model._want_counts,e.model._want_mass)),
            predictive=dict(e.predictive.last),grade=grade(p,text,enc) if p else None)
        assert row['prefix_cached_tokens']==cached
        assert len(set(e.ep.gather_objects(json.dumps([output,row['demand_sha256'],row['history_sha256']]))))==1
        if mode!='apply':assert mask_digest(e.model_prune_mask)==initial_mask
        result['rows'].append(row);save()
        if e.ep.rank==0:print('TWO_PASS_ROW '+json.dumps({k:v for k,v in row.items() if k not in ('output','text')}),flush=True)
        return row
    def equal(off,shadow):
        for k in ('output','demand_sha256','history_sha256','prefix_cached_tokens'):
            assert off[k]==shadow[k],f'shadow changed {k}'
        if 0<shadow['new_tokens']<500:
            # finish's evaluation is rank-0-only; both ranks receive it before
            # assertions, so a missing telemetry key cannot kill only the peer.
            report=e.ep.broadcast_obj(shadow['predictive'] if e.ep.rank==0 else None)
            assert report['prediction_kind']=='full_suffix'
            assert report['probe_tokens']==shadow['new_tokens']
            assert report['mean_total_variation']<1e-12,'full discovery does not match real demand'
            if report['planned_swaps']:
                assert report['promotion_precision']==report['promotion_recall']==1
                assert report['exact_swap_overlap']==report['planned_swaps']
    # Odd/even compressor boundary and exact threshold, at >98% prefix reuse.
    p,base=warm_probe('python',99500,32768,render)
    seed_p=dict(family='json',limit=32,expected={'ready':True})
    run('initial-seed',base,'off',p=seed_p)
    seed_snapshot=e._prefix_cache
    for n in (1,31,128,499,500):
        prefix=base if n%2 else base+[tok.encode(' x')[0]]
        suffix=(tok.encode(' Continue reviewing the module. ') * (n+1))[:n]
        rows=[]
        for mode in ('off','shadow'):
            e._prefix_cache=seed_snapshot;e._prefix_snapshots.clear()
            run(f'seed-{n}',prefix,'off',len(base))
            rows.append(run(f'cached-{n}',prefix+suffix,mode,len(prefix)))
        equal(*rows)
        if n==500:assert rows[-1]['predictive']['prediction_kind']=='excerpt'
    # Short cold prompt, zero-new-token hit, and graded medical conversation turn.
    med=dict(family='json',limit=96,expected={'age':47,'assay_z':11,'change':-7})
    medtext=('Synthetic medical record: patient A age 47; fictional assay Z at visit 1 is 18 '
             'and at visit 2 is 11. Return only JSON with age, assay_z (visit 2 value), '
             'and change (visit 2 minus visit 1).')
    ids=chat(medtext)
    cold=[run('cold-medical',ids,mode,p=med) for mode in ('off','shadow')];equal(*cold)
    hit=run('full-hit',ids,'shadow',len(ids),med)
    assert hit['predictive']['probe_tokens']==0
    # Serialized chat extension, including the actual assistant reply, as served.
    long_body=p['body']['messages'][0]['content']
    # Cold short requests above replaced the compressed backing at its start.
    # Rebuild it once, then restore the saved input boundary for each arm.
    run('conversation-fresh-seed',base,'off',p=seed_p)
    conversation_snapshot=e._prefix_cache
    conversation=[]
    for mode in ('off','shadow','apply'):
        e._prefix_cache=conversation_snapshot;e._prefix_snapshots.clear()
        seed=run('conversation-seed',base,'off',len(base),seed_p)
        body={'messages':[{'role':'user','content':long_body},
                          {'role':'assistant','content':seed['text']},
                          {'role':'user','content':medtext}]}
        turn=render(body)
        assert turn[:len(base)]==base,'serialized chat did not extend seed prefix'
        assert 0<len(turn)-len(base)<500
        if mode=='apply':e.maintain_inline=inline
        conversation.append(run('medical-conversation',turn,mode,len(base),med))
    equal(*conversation[:2])
    assert conversation[-1]['predictive']['swaps']>0,'early loading was not exercised'
    assert all(r['grade']['pass_'] for r in result['rows'] if r['grade'])
    assert all(e.ep.gather_objects(True));save();sys.stdout.flush();os._exit(0)


if __name__=='__main__':main()
