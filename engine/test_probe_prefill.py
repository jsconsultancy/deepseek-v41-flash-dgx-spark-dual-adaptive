"""CPU checks of probe isolation, unmasked demand and distributed admission."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine import model as M, v41_engine as V
from engine.adapt_config import resolve
from engine.predictive_prefill import PredictivePrefill
from engine.probe_prefill import (SmallProbe, ProbeDemand, select_spans, isolated_model,
                                 excerpt_hashes, suffix_checkpoint, isolated_suffix)
from engine import test_predictive_prefill as P


class HashState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layout = NS(max_ngram_size=3)
        self.register_buffer('cache', torch.full((1, 3000), -999))

    def forward(self, ids, start):
        self.cache[:, start:start+ids.size(1)] = ids
        pos = torch.arange(start, start+ids.size(1))
        return torch.stack([torch.where(pos >= k, self.cache[:, (pos-k).clamp_min(0)], 0)
                            for k in range(3)], dim=-1)


class ProbePrefillTest(unittest.TestCase):
    def test_simple_switch_applies_computed_probe_and_ignores_legacy_settings(self):
        b=PredictivePrefill(2,16,{'DSV41_PROBE_PREFILL':'1',
            'DSV41_PREDICTIVE_PREFILL':'invalid','DSV41_PREDICTIVE_BACKEND':'invalid',
            'DSV41_PREDICTIVE_MIN_SAMPLES':'invalid','DSV41_PREDICTIVE_DB':'do-not-open',
            'DSV41_PROBE_TOKENS':'64','DSV41_PROBE_TWO_PASS_BELOW':'500'})
        self.assertEqual((b.mode,b.backend),('apply','probe'))
        self.assertEqual(b.boot_fields()['predictive_mode'],'apply')
        self.assertEqual(b.boot_fields()['predictive_probe_two_pass_below'],500)
        b.configure({},'does-not-exist','does-not-exist')

    def test_simple_switch_off_never_runs_probe_or_broadcast_even_with_legacy_apply(self):
        e,_,_=self.engine()
        b=PredictivePrefill(2,16,{'DSV41_PROBE_PREFILL':'0',
            'DSV41_PREDICTIVE_PREFILL':'apply','DSV41_PREDICTIVE_BACKEND':'nearest'})
        b.probe.predict=lambda *a:self.fail('disabled probe ran')
        e.ep.broadcast_obj=lambda *a:self.fail('disabled predictor broadcast')
        b.begin(e,list(range(2048)),0)
        self.assertEqual(b.boot_fields()['predictive_mode'],'off')

    def test_simple_switch_rejects_ambiguous_values(self):
        for value in ('','true','shadow','apply'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError,'DSV41_PROBE_PREFILL must be 0 or 1'):
                    PredictivePrefill(2,16,{'DSV41_PROBE_PREFILL':value})

    def test_spans_bounded_ordered_tail_retained_and_never_full_suffix(self):
        for length, prefix in ((100,0),(10000,0),(200000,198000),(200000,199900)):
            for limit in (4,32,256,512,1024,100000):
                spans = select_spans(length,prefix,limit)
                self.assertEqual(spans[-1][1],length)
                self.assertLess(sum(b-a for a,b in spans),length-prefix)
                self.assertLessEqual(sum(b-a for a,b in spans),limit)
                self.assertEqual(spans[0][0],0)
                for i,(a,b) in enumerate(spans):
                    self.assertTrue(0 <= a < b <= length)
                    if i:self.assertGreater(a,spans[i-1][1])
                    if a:self.assertGreaterEqual(a,prefix)
        self.assertEqual(select_spans(100,100,256),[])
        with self.assertRaises(ValueError):select_spans(100,101,10)

    def test_probe_observes_missing_experts_and_excludes_cached_context_rows(self):
        demand=ProbeDemand(2,4,[False,True,True,False], 'cpu')
        m=NS(_probe_demand=demand,_moe_start=0)
        logits=torch.tensor([[20.,0.,0.,0.],[0.,0.,0.,20.],[0.,0.,0.,30.],[20.,0.,0.,0.]])
        scores=logits+1
        # No live counters/allocator are supplied: reaching the normal recorder fails.
        M.Model._record_prune_miss(m,logits,scores,torch.tensor([True,True,False,False]),0,1)
        self.assertEqual(demand.counts[0].tolist(),[0.,0.,0.,2.])
        self.assertEqual(demand.mass[0,3].item(),52.)
        m._moe_start=2
        M.Model._record_prune_miss(m,logits[2:],scores[2:],None,1,1)
        self.assertEqual(demand.counts[1].tolist(),[0.,0.,0.,1.])

    def test_original_ngrams_preserved_and_live_hash_cache_untouched(self):
        state=HashState();before=state.cache.clone()
        tokens=list(range(1,1001));spans=[(0,20),(300,330),(900,1000)]
        actual=excerpt_hashes(state,tokens,spans,'cpu')
        expected=HashState()(torch.tensor(tokens)[None],0)[0]
        expected=torch.cat([expected[a:b] for a,b in spans])
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(state.cache,before)
        self.assertIsNone(excerpt_hashes(None,tokens,spans,'cpu'))

    @staticmethod
    def engine():
        e,db,req=P.PredictivePrefillTest.engine()
        e.device='cpu';e.args=NS(n_layers=2,n_routed_experts=16,window_size=4);e.swa_replay=False
        e.model.args=e.args;e.model.stats={'tokens':99};e.model.hash_state=None
        e.model.c=NS(len=2000);e.model._rep={'old':'prefix'};e.model._rep_end=2000
        e.model.prefill_graphs=object();e.model.last_h=torch.tensor([42.])
        e.model.begin_prompt=lambda:setattr(e.model,'_rep',{})
        e.ep.broadcast_obj=lambda packet:packet
        e.ep.gather_objects=lambda value:[value,value]
        def forward(ids,start,**kw):
            e.model.c.len=start+len(ids)
            e.model.stats['tokens']+=len(ids)
            e.model._moe_start=start
            logits=torch.zeros((len(ids),16));logits[:,12]=10
            for layer in range(2):M.Model._record_prune_miss(e.model,logits,logits+1,None,layer,1)
        e.model.forward=forward
        return e,db,req

    def test_isolation_restores_objects_on_success_and_failure(self):
        for fail in (False,True):
            e,_,_=self.engine();m=e.model
            original={name:getattr(m,name) for name in ('c','stats','_rep','_rep_end','last_h','prefill_graphs','_req_counts')}
            scratch=NS(len=0,pending={0:None},_chunk_inputs={})
            def run():
                with isolated_model(m,scratch,object()):
                    self.assertIsNone(m.prefill_graphs)
                    m.stats['tokens']=1000;m._rep['new']=1;m.last_h=None;m.last_pre_mix=None
                    M._PHASE_MARKS.append(('probe',None))
                    if fail:raise RuntimeError('test')
            if fail:
                with self.assertRaises(RuntimeError):run()
            else:run()
            for name,value in original.items():self.assertIs(getattr(m,name),value)
            self.assertFalse(hasattr(m,'last_pre_mix'));self.assertFalse(hasattr(m,'_probe_demand'))
            self.assertFalse(M._PHASE_MARKS)

    @patch('torch.cuda.synchronize')
    @patch('engine.model.Caches',side_effect=lambda *a:NS(len=0,max_seq=a[1],pending={},_chunk_inputs={}))
    def test_computed_probe_never_records_live_history(self,caches,sync):
        e,_,req=self.engine();live=e.model._req_counts.clone();c=e.model.c
        p=SmallProbe({'DSV41_PROBE_TOKENS':'64'})
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            demand,info=p.predict(e,list(range(2048)),0)
        self.assertEqual(info['probe_tokens'],64)
        np.testing.assert_array_equal(demand[0][:,12],[64,64])
        torch.testing.assert_close(e.model._req_counts,live)
        self.assertIs(e.model.c,c);self.assertEqual(e.model.stats['tokens'],99)
        self.assertIsNone(e.model.hash_state)

    @patch('torch.cuda.synchronize')
    @patch('engine.model.Caches',side_effect=lambda *a:NS(len=0,max_seq=a[1],pending={},_chunk_inputs={}))
    def test_peer_prepare_failure_prevents_both_model_passes(self,caches,sync):
        e,_,_=self.engine();e.model.forward=lambda *a,**k:self.fail('peer was not ready')
        e.ep.gather_objects=lambda value:[None,'OutOfMemoryError']
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            demand,info=SmallProbe().predict(e,list(range(2048)),0)
        self.assertIsNone(demand);self.assertEqual(info['status'],'probe_prepare_error')

    @patch('torch.cuda.synchronize')
    @patch('engine.model.Caches',side_effect=lambda *a:NS(len=0,max_seq=a[1],pending={},_chunk_inputs={}))
    def test_rank_disagreement_discards_probe_without_planning(self,caches,sync):
        e,_,_=self.engine()
        e.ep.gather_objects=lambda value:([None,None] if value is None else [('a',1.),('b',2.)])
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            demand,info=SmallProbe().predict(e,list(range(2048)),0)
        self.assertIsNone(demand);self.assertEqual(info['status'],'probe_rank_disagreement')

    @patch('traceback.print_exc')
    @patch('os._exit',side_effect=SystemExit(1))
    @patch('engine.model.Caches',side_effect=lambda *a:NS(len=0,max_seq=a[1],pending={},_chunk_inputs={}))
    def test_forward_failure_is_fatal_and_does_not_attempt_another_collective(self,caches,exit_,trace):
        e,_,_=self.engine();c=e.model.c;calls=[]
        e.ep.gather_objects=lambda value:calls.append(value) or [value,value]
        def fail(*a,**kw):raise RuntimeError('failed collective')
        e.model.forward=fail
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            with self.assertRaises(SystemExit):SmallProbe().predict(e,list(range(2048)),0)
        self.assertEqual(calls,[None]);self.assertIs(e.model.c,c)
        self.assertFalse(hasattr(e.model,'_probe_demand'));exit_.assert_called_once_with(1)

    def test_skips_are_broadcast_and_no_full_short_second_pass(self):
        for tokens,prefix,vision,status in ((100,0,None,'skip_probe_budget'),
                (2000,2000,None,'skip_short_or_cached'),(2000,0,[1],'skip_vision')):
            e,_,_=self.engine();e._images=vision;packets=[]
            e.ep.broadcast_obj=lambda p:packets.append(p) or p
            with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
                demand,info=SmallProbe().predict(e,list(range(tokens)),prefix)
            self.assertIsNone(demand);self.assertEqual(info['status'],status)
            self.assertEqual(len(packets),1)

    @staticmethod
    def suffix_engine(prefix):
        e,db,req=ProbePrefillTest.engine()
        e.swa_replay=True
        c=e.model.c=NS(len=prefix,win=[torch.arange(M.RING,dtype=torch.float32)[:,None] for _ in range(2)],
            pending={0:(torch.tensor([3.]),torch.tensor([7.]))},_chunk_inputs={'old':object()})
        e.model._positions=torch.arange(prefix+600)
        e.model._rep={'h':[torch.tensor([1.])],'pre_mix':[],'topk':[],'cand':[]}
        e.model._rep_end=prefix
        e.model.begin_prompt=lambda:M.Model.begin_prompt(e.model)
        e._prefill_spans=lambda end,start:[(start,end)]
        original_forward=e.model.forward
        def forward(ids,start,**kw):
            original_forward(ids,start,**kw)
            c.win[0][start%M.RING]=-1
            c.pending[0][0].fill_(100)
            c._chunk_inputs['new']=1
            e.model._rep['h'].append(torch.tensor([2.]))
        e.model.forward=forward
        def replay(**kw):
            end=c.len;n=min(end,e.args.window_size)
            e.model._moe_start=end-n
            logits=torch.zeros((n,16));logits[:,12]=10
            for L in range(2):
                c.win[L][(end-n)%M.RING]=-2
                M.Model._record_prune_miss(e.model,logits,logits+1,None,L,1)
        e.model.decoder_replay=replay
        return e,db,req

    @patch('torch.cuda.synchronize')
    @patch('engine.model.Caches',side_effect=AssertionError('full discovery must reuse cached context'))
    def test_full_suffix_restores_cached_context_and_counts_actual_replay(self,caches,sync):
        e,_,_=self.suffix_engine(2000);m=e.model;c=m.c
        old={k:getattr(m,k) for k in ('_rep','_rep_end','stats','last_h','prefill_graphs')}
        pending,chunks=c.pending,c._chunk_inputs
        rings=[w.clone() for w in c.win];live=m._req_counts.clone()
        b=SmallProbe({'DSV41_PROBE_TWO_PASS_BELOW':'500','DSV41_PROBE_TOKENS':'64'})
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            demand,info=b.predict(e,list(range(2013)),2000)
        self.assertEqual(info['prediction_kind'],'full_suffix')
        self.assertEqual(info['probe_tokens'],13)
        self.assertEqual(info['probe_spans'],[(2000,2013)])
        np.testing.assert_array_equal(demand[0][:,12],[17,17])
        self.assertEqual(c.len,2000);self.assertIs(c.pending,pending);self.assertIs(c._chunk_inputs,chunks)
        self.assertEqual(pending[0][0].item(),3.)
        self.assertEqual(len(old['_rep']['h']),1)
        for k,v in old.items():self.assertIs(getattr(m,k),v)
        for before,after in zip(rings,c.win):torch.testing.assert_close(before[1996:2000],after[1996:2000])
        torch.testing.assert_close(m._req_counts,live)
        self.assertIsNone(b.caches)

    @patch('torch.cuda.synchronize')
    @patch('engine.model.Caches',side_effect=lambda *a:NS(len=0,max_seq=a[1],pending={},_chunk_inputs={}))
    def test_two_pass_threshold_is_strict_and_bypasses_short_probe_minimum(self,caches,sync):
        for new,kind in ((1,'full_suffix'),(31,'full_suffix'),(499,'full_suffix'),(500,'excerpt')):
            e,_,_=self.suffix_engine(0) if kind=='full_suffix' else self.engine()
            b=SmallProbe({'DSV41_PROBE_TWO_PASS_BELOW':'500','DSV41_PROBE_TOKENS':'64',
                          'DSV41_PROBE_FRACTION':'.1'})
            with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
                demand,info=b.predict(e,list(range(new)),0)
            self.assertIsNotNone(demand);self.assertEqual(info['prediction_kind'],kind)
            if kind=='full_suffix':self.assertEqual(info['probe_tokens'],new)
            else:self.assertLess(info['probe_tokens'],new)
        e,_,_=self.suffix_engine(20)
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            demand,info=b.predict(e,list(range(20)),20)
        self.assertIsNone(demand);self.assertEqual(info['probe_tokens'],0)

    def test_suffix_boundary_is_restored_after_failure(self):
        e,_,_=self.suffix_engine(2000);m=e.model;c=m.c
        original=m._rep;pending=c.pending;chunks=c._chunk_inputs
        backup=suffix_checkpoint(m,2000)
        with self.assertRaises(RuntimeError):
            with isolated_suffix(m,2000,object(),backup):
                c.win[0][1999]=0;c.len=2050;c.pending[0][0].zero_()
                m._rep['h'].append(torch.tensor([9.]))
                raise RuntimeError('test')
        self.assertEqual(c.win[0][1999].item(),1999.)
        self.assertEqual(c.len,2000);self.assertIs(c.pending,pending);self.assertIs(c._chunk_inputs,chunks)
        self.assertEqual(c.pending[0][0].item(),3.)
        self.assertIs(m._rep,original);self.assertEqual(len(original['h']),1)
        self.assertFalse(hasattr(m,'_probe_demand'))

    def test_probe_planning_uses_normal_policy_and_never_opens_or_writes_bank(self):
        e,db,req=self.engine();applied=[]
        e.apply_swaps=lambda p:applied.extend(p) or len(p)
        b=PredictivePrefill(2,16,{'DSV41_PROBE_PREFILL':'1'})
        b.configure({},'does-not-exist','does-not-exist')
        b.probe.predict=lambda *a:((req,req),{'status':'probe','probe_ms':1.})
        b.observe=b.save=lambda *a:self.fail('computed probes must not train a lookup bank')
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high','DSV41_PRUNE_METRIC':'score'})):
            expected=e.plan_swaps(max_swaps=V.ADAPT.swap_max,request_demand=(req,req))
            b.begin(e,list(range(2048)),0);b.finish(e)
        self.assertEqual(applied,expected);self.assertIn('prediction_prefill_ratio',b.last)
        np.testing.assert_array_equal(e.model.prune_miss_report()[1],db)
        np.testing.assert_array_equal(e.model._req_counts,req)

    def test_probe_configuration_is_guarded_and_invalid_budgets_rejected(self):
        b=PredictivePrefill(2,16,{'DSV41_PREDICTIVE_BACKEND':'probe','DSV41_PROBE_TOKENS':'512'})
        self.assertEqual(b.boot_fields()['predictive_backend'],'probe')
        self.assertEqual(b.boot_fields()['predictive_probe_tokens'],512)
        self.assertEqual(SmallProbe({'DSV41_PROBE_TWO_PASS_BELOW':'500'}).boot_fields()
                         ['predictive_probe_two_pass_below'],500)
        for env in ({'DSV41_PROBE_TOKENS':'2048'}, {'DSV41_PROBE_FRACTION':'nan'},
                    {'DSV41_PROBE_FRACTION':'.5'},{'DSV41_PROBE_TARGET_RATIO':'inf'},
                    {'DSV41_PROBE_TWO_PASS_BELOW':'-1'},{'DSV41_PROBE_TWO_PASS_BELOW':'1025'}):
            with self.assertRaises(ValueError):SmallProbe(env)


if __name__=='__main__':unittest.main()
