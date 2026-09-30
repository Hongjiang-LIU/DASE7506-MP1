"""Synthetic CPU initialization only; no real weights, text, or evaluator."""
import ast
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from common import PROTOCOL
from push150_expand import ExpandedSignedCache, projection_modules
import push152_init as subject


SOURCE = dict(qkv=6, proj=0, gate=4, value=4, out=4)


def tables():
    result = {'unigram': torch.full((2048,), 1/2048)}
    for order in range(2,6):
        result.update({f'contexts_{order}':torch.tensor([1],dtype=torch.int64),
            f'offsets_{order}':torch.tensor([0,1],dtype=torch.int32),
            f'words_{order}':torch.tensor([7],dtype=torch.int16),
            f'masses_{order}':torch.tensor([.25]), f'backoffs_{order}':torch.tensor([.75])})
    return result


def parent_fixture():
    arrays=tables()
    config=dict(vocab=2048,width=16,heads=4,depth=2,context=256,ffn_hidden=32,
        rope=True,dropout=0.,lowrank_rank=4,push150_ranks=SOURCE.copy(),
        temperature=1.3,cache_weight=.05,cache_theta=15.,pointer_rank=4,pointer_max_weight=.5,
        mkn_shapes={name:list(value.shape) for name,value in arrays.items()},mkn_order=5,mkn_weight=.15,discount_scale=.7)
    model=ExpandedSignedCache(config).eval()
    model.ngram.load_state_dict(arrays)
    with torch.no_grad():
        model.cache_recency_strength.fill_(-.4)
        model.pointer_key.weight.normal_(0,.02)
    parent=dict(protocol=PROTOCOL,implementation='push150_expand',config=copy.deepcopy(model.config),
        model=copy.deepcopy(model.state_dict()),train_tokens=638976000,
        ancestry={'old_shared_targets':638976000},push151_combination={'synthetic_origin':True})
    return model,parent


def plan_fixture():
    return dict(parent=dict(path='runs/push151-combination/checkpoint.pt',sha256=subject.PARENT_SHA256,
        implementation='push150_expand',implementation_sha256=subject.IMPLEMENTATION_SHA256,
        selected_lineage_targets=subject.LINEAGE),
        expansion=dict(source_layout=subject.SOURCE_LAYOUT.copy(),target_layout=subject.TARGET_LAYOUT.copy(),
            seed=193,expected_parameter_increase=681984,run_dir='runs/push152-expanded-init'),
        shared=dict(mkn_weight=.15,mkn_order=5,discount_scale=.7,validation_temperature=1.3))


class FFNInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(193)
        self.source,self.parent=parent_fixture()
        self.ids=torch.randint(2048,(2,11))

    def init(self,parent=None,seed=193):
        model,report=subject.initialize_from_parent(self.parent if parent is None else parent,target_ffn_rank=6,seed=seed)
        return model.eval(),report

    def test_only_ffn_grows_and_all_retained_values_are_exact(self):
        model,report=self.init()
        for prefix,role,module in projection_modules(model):
            if role in subject.FFN_ROLES:
                self.assertTrue(torch.equal(module.left.weight[:4],self.parent['model'][prefix+'.left.weight']))
                self.assertTrue(torch.equal(module.right.weight[:,:4],self.parent['model'][prefix+'.right.weight']))
                self.assertEqual(module.right.weight[:,4:].count_nonzero().item(),0)
                self.assertGreater(module.left.weight[4:].count_nonzero().item(),0)
            else:
                for name,value in module.state_dict().items():
                    self.assertTrue(torch.equal(value,self.parent['model'][prefix+'.'+name]))
        self.assertEqual(report['source_layout'],SOURCE)
        self.assertEqual(report['target_layout'],dict(SOURCE,gate=6,value=6,out=6))
        self.assertTrue(report['dense_projection_copied_without_folding'])

    def test_unchanged_full_non_ffn_state_and_all_twenty_one_tables(self):
        model,report=self.init()
        ffn=tuple(prefix+'.' for prefix,role,_ in projection_modules(model) if role in subject.FFN_ROLES)
        for name,value in model.state_dict().items():
            if not name.startswith(ffn):
                self.assertEqual(value.dtype,self.parent['model'][name].dtype)
                self.assertTrue(torch.equal(value,self.parent['model'][name]),name)
        self.assertEqual(report['statistics_buffers_preserved_bitwise'],21)
        self.assertIs(model.head.weight,model.token.weight)

    def test_parameter_delta_formula_and_no_new_targets(self):
        model,report=self.init()
        expected=2*3*(6-4)*(16+32)
        self.assertEqual(report['parameter_delta'],expected)
        self.assertEqual(report['parameters']-report['parent_parameters'],expected)
        self.assertEqual(report['parameter_payload_delta_bytes'],4*expected)
        self.assertEqual(report['parent_parameters'],sum(value.numel() for value in self.source.parameters()))
        self.assertFalse(report['prediction_or_scoring_performed'])
        self.assertEqual(report['new_training_targets'],0)

    def test_predictor_and_features_equal_with_fp32_roundoff(self):
        model,_=self.init()
        with torch.no_grad():
            torch.testing.assert_close(model.features(self.ids),self.source.features(self.ids),atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(model.predict_log_probs(self.ids),self.source.predict_log_probs(self.ids),atol=1e-5,rtol=1e-5)

    def test_causal_short_prefix_normalization_and_reset(self):
        model,_=self.init();logp=model.predict_log_probs(self.ids)
        changed=self.ids.clone();changed[:,5:]=7
        torch.testing.assert_close(logp[:,:5],model.predict_log_probs(changed)[:,:5],atol=4e-6,rtol=1e-6)
        for length in (1,3,7):
            torch.testing.assert_close(logp[:,:length],model.predict_log_probs(self.ids[:,:length]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(logp.logsumexp(-1),torch.zeros(2,11),atol=1e-6,rtol=0)
        model.predict_log_probs(torch.zeros(2,3,dtype=torch.long))
        self.assertTrue(torch.equal(logp,model.predict_log_probs(self.ids)))

    def test_new_right_columns_receive_gradients_then_left_rows_learn(self):
        model,_=self.init();model.train()
        with patch.object(model.ngram,'probabilities',side_effect=AssertionError('training must not use counts')):
            loss=-model.training_log_probs(self.ids).gather(-1,self.ids.roll(-1,1)[...,None]).mean()
            loss.backward()
        for prefix,role,module in projection_modules(model):
            if role in subject.FFN_ROLES:
                self.assertGreater(module.right.weight.grad[:,4:].abs().sum().item(),0,prefix)
                self.assertEqual(module.left.weight.grad[4:].count_nonzero().item(),0)
        with torch.no_grad():
            for value in model.parameters():
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                value.add_(value.grad,alpha=-.1)
        model.zero_grad(set_to_none=True)
        (-model.training_log_probs(self.ids).gather(-1,self.ids.roll(-1,1)[...,None]).mean()).backward()
        for prefix,role,module in projection_modules(model):
            if role in subject.FFN_ROLES:
                self.assertGreater(module.left.weight.grad[4:].abs().sum().item(),0,prefix)

    def test_parent_and_global_rng_not_mutated(self):
        before=copy.deepcopy(self.parent);rng=torch.random.get_rng_state().clone()
        self.init()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(before['config'],self.parent['config'])
        for name,value in before['model'].items():self.assertTrue(torch.equal(value,self.parent['model'][name]))

    def test_seed_reproducible_changes_only_new_left_rows(self):
        first,_=self.init(seed=193);same,_=self.init(seed=193);other,_=self.init(seed=194)
        for name,value in first.state_dict().items():self.assertTrue(torch.equal(value,same.state_dict()[name]))
        left=first.blocks[0].mlp.out.left.weight;changed=other.blocks[0].mlp.out.left.weight
        self.assertTrue(torch.equal(left[:4],changed[:4]));self.assertFalse(torch.equal(left[4:],changed[4:]))

    def test_serialization_tying_ancestry_and_no_false_selected_metadata(self):
        checkpoint,report=subject.initialized_checkpoint(self.parent,'a'*64,target_ffn_rank=6)
        self.assertEqual(checkpoint['train_tokens'],638976000)
        self.assertEqual(checkpoint['ancestry']['shared_inherited_targets_counted_once'],638976000)
        self.assertEqual(checkpoint['ancestry']['prior_ancestry'],self.parent['ancestry'])
        self.assertEqual(checkpoint['push151_combination'],self.parent['push151_combination'])
        self.assertIsNone(checkpoint['selected_step'])
        buffer=io.BytesIO();torch.save(checkpoint,buffer);buffer.seek(0)
        restored=torch.load(buffer,weights_only=True)
        self.assertEqual(restored['model']['head.weight'].data_ptr(),restored['model']['token.weight'].data_ptr())
        model=ExpandedSignedCache(restored['config']);model.load_state_dict(restored['model'],strict=True)
        self.assertIs(model.head.weight,model.token.weight)

    def test_unknown_missing_and_broken_tied_state_rejected(self):
        parent=copy.deepcopy(self.parent);parent['model']['unaccounted']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'Unaccounted'):self.init(parent)
        parent=copy.deepcopy(self.parent);parent['model'].pop('blocks.0.proj.weight')
        with self.assertRaisesRegex(ValueError,'Missing'):self.init(parent)
        parent=copy.deepcopy(self.parent);parent['model']['head.weight']=parent['model']['head.weight']+1
        with self.assertRaisesRegex(ValueError,'tied embedding'):self.init(parent)

    def test_dtype_shape_and_nonfinite_sources_rejected(self):
        for name in ('blocks.0.mlp.out.left.weight','blocks.0.proj.weight','ngram.masses_3','token.weight'):
            for defect in ('dtype','shape','nan'):
                parent=copy.deepcopy(self.parent);value=parent['model'][name]
                parent['model'][name]=value.double() if defect=='dtype' else value.unsqueeze(0) if defect=='shape' else torch.full_like(value,float('nan'))
                with self.subTest(name=name,defect=defect),self.assertRaises(ValueError):self.init(parent)

    def test_wrong_architecture_and_layout_refused(self):
        parent=copy.deepcopy(self.parent);parent['implementation']='push147_signed_cache'
        with self.assertRaises(ValueError):self.init(parent)
        for updates in ({'proj':4},{'gate':0},{'gate':True},{'gate':17},{'qkv':0},{'out':3}):
            parent=copy.deepcopy(self.parent);parent['config']['push150_ranks'].update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.init(parent)
        for rank in (4,3,True,17):
            with self.assertRaises(ValueError):subject.initialize_from_parent(self.parent,target_ffn_rank=rank)

    def test_protocol_lineage_and_hash_must_be_explicit(self):
        for key,value in [('protocol','wrong'),('train_tokens',None),('train_tokens',True),('train_tokens',-1)]:
            parent=copy.deepcopy(self.parent);parent[key]=value
            with self.assertRaises(ValueError):subject.initialized_checkpoint(parent,'a'*64,target_ffn_rank=6)
        for digest in ('a','Z'*64,None):
            with self.assertRaises(ValueError):subject.initialized_checkpoint(self.parent,digest,target_ffn_rank=6)

    def test_fixed_plan_identity_and_capacity(self):
        subject.validate_plan(plan_fixture())
        for group,key,value in [('parent','sha256','x'*64),('parent','selected_lineage_targets',0),
                                ('expansion','seed',194),('expansion','expected_parameter_increase',1),
                                ('shared','mkn_weight',.2)]:
            plan=plan_fixture();plan[group][key]=value
            with self.assertRaises(ValueError):subject.validate_plan(plan)

    def test_no_corpus_io_and_exclusive_writes(self):
        with self.assertRaises(ValueError):subject.sha('WikiText_test.txt')
        with self.assertRaises(ValueError):subject.read('wikitext_test.txt')
        with tempfile.TemporaryDirectory() as folder:
            json=Path(folder)/'record.json';cp=Path(folder)/'synthetic.pt'
            subject.write_new(json,{'synthetic':True});subject.save_new(cp,{'synthetic':torch.ones(1)})
            with self.assertRaises(FileExistsError):subject.write_new(json,{})
            with self.assertRaises(FileExistsError):subject.save_new(cp,{})

    def test_recursive_local_source_pins_and_source_change_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'push152_init.py').write_text('from helper import value\n',encoding='utf-8')
            (root/'push150_expand.py').write_text('import helper\n',encoding='utf-8')
            (root/'helper.py').write_text('import nested\nvalue=1\n',encoding='utf-8')
            (root/'nested.py').write_text('value=2\n',encoding='utf-8')
            pins=subject.source_hashes(root)
            self.assertEqual(set(pins),{'push152_init.py','push150_expand.py','helper.py','nested.py'})
            subject.verify_sources(pins,root)
            (root/'nested.py').write_text('value=3\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'source changed'):subject.verify_sources(pins,root)

    def test_initializer_contains_no_scoring_or_training_invocations(self):
        tree=ast.parse(Path(subject.__file__).read_text(encoding='utf-8'))
        calls={ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)}
        self.assertFalse({'score','common.load_data','load_data','model.predict_log_probs','optimizer.step','loss.backward'} & calls)


if __name__=='__main__':
    unittest.main()
