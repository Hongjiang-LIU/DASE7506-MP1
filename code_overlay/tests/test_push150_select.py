"""Pure synthetic metadata/checkpoint checks; no real data, score or GPU."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import push150_select as selector


def measurement(bpb):
    return dict(bpb=bpb,nll_nats=bpb*math.log(2)*1148007,targets=376599,utf8_bytes=1148007,seconds=1.)


def metadata_fixture():
    parent = dict(implementation='push150_expand',train_tokens=500,config={'temperature':1.3,'discount_scale':.55})
    train = dict(split='train',tokens=selector.bc.TRAIN_TOKENS,utf8_bytes=selector.bc.TRAIN_BYTES,
        text_sha256=selector.a.DATA_SHA['data/wikitext_train.txt'],tokenizer_sha256=selector.a.DATA_SHA['data/tokenizer.json'],
        manifest_sha256=selector.a.DATA_SHA['data/manifest.json'])
    build = dict(train=train,full_table={'path':'full.pt'},folds=[])
    folds = []
    for index,spec in enumerate(selector.bc.fold_specifications()):
        start,end = spec['excluded_token_interval']
        table = dict(spec,path=f'tables/fold-{index:02d}.pt',sha256=str(index)*64,bytes=100)
        build['folds'].append(table)
        folds.append(dict(fold=index,start=start,end=end,valid_random_starts=end-start-256,
            sentinel_input_position=end-1,table_path=table['path'],table_sha256=table['sha256'],table_bytes=100,
            target_probability_queries=4*(end-start-1),queried_label_positions=end-start-1,
            parity=dict(passes=True,window_starts_local=[0,(end-start-257)//2,end-start-257],prefix_lengths=[1,2,3,256],
                maximum_orders=[2,3,4,5],comparisons=48,full_distribution_forward_target_occurrences=3144,
                max_absolute_difference=1e-8,atol=1e-7,rtol=1e-5)))
    metadata = dict(status='completed',test_used=False,plan_sha256=selector.D_PLAN_SHA,parent_sha256='a'*64,
        parent_lineage_targets=500,parent_implementation='push150_expand',parent_config=parent['config'],phase_a_selection_sha256='b'*64,
        train=train,full_train_table=build['full_table'],fold_table_sha256=[row['sha256'] for row in build['folds']],
        training_intermediate_only=True,include_in_inference_assets=False,no_validation_or_test_arrays=True,
        only_count_component_is_out_of_fold=True,neural_parent_saw_all_train=True,
        array=dict(path='train_oof_probabilities.npy',shape=[selector.bc.TRAIN_TOKENS,4],dtype='float32',maximum_orders=[2,3,4,5]),
        window_context=256,window_target_span=257,window_columns='column=min(within_window_position,3); maximum_order=min(5,t+2)',
        folds=folds,valid_random_starts=selector.bc.TRAIN_TOKENS-2560,valid_label_positions=selector.bc.TRAIN_TOKENS-10,
        invalid_label_positions=[row['end']-1 for row in folds],
        costs=dict(target_probability_queries=4*(selector.bc.TRAIN_TOKENS-10),parity_forward_target_occurrences=31440,
            new_count_scan_token_occurrences=0,reused_table_scan_cost_counted_in_phase_b_only=True,
            new_neural_training_targets=0,new_gate_training_targets=0,seconds_total=1.),
        parity=dict(passes=True,fold_checks=10,atol=1e-7,rtol=1e-5,maximum_absolute_difference=1e-8))
    return metadata,parent,build


def history_fixture(objective='control'):
    validation = [dict(step=0,weights='raw',session=0,**measurement(1.5))]
    validation += [dict(step=step,weights=kind,session=0,**measurement(1.5-step/1e6-(.001 if kind=='ema' else 0)))
                   for step in range(2000,12001,2000) for kind in ('raw','ema')]
    history = [dict(step=step,session=0,count_block_kept=True,
                   count_block_coins=dict(drawn=step,kept=step//2,dropped=step//2)) for step in range(400,12001,400)]
    coins = dict(drawn=12000,kept=6000,dropped=6000)
    return dict(status='completed',completed_steps=12000,resume_history=[],discarded_completed_steps=0,
        validation_history=validation,best=dict(validation[-1],selected_lineage_targets=500+12000*8192),
        count_block_coins=coins,discarded_count_block_coins=dict(drawn=0,kept=0,dropped=0),
        count_block_coins_including_discarded=coins,cached_target_probability_lookups=12000*8192,
        joint_count_kept_target_occurrences=6000*8192 if objective=='joint' else 0,
        validation_choices_in_retained_history=13,history=history)


def paired_fixture(objective):
    return dict(parent_sha256='a'*64,source_sha256={'runtime.py':'b'*64},
        recipe=selector.expected_recipe({'temperature':1.3,'mkn_weight':.15},objective),
        count_block_coins=dict(drawn=12000,kept=6000,dropped=6000),trajectory=[{'synthetic':True}],
        sampler_state=torch.tensor([1,2,3],dtype=torch.uint8),cuda_rng=[torch.tensor([4,5],dtype=torch.uint8)])


class FinalSelectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_prepared_metadata_and_shared_scan_zero_cost(self):
        metadata,parent,build = metadata_fixture()
        selector.verify_prepared_schema(metadata,parent,'a'*64,'b'*64,build)
        self.assertEqual(metadata['costs']['new_count_scan_token_occurrences'],0)
        self.assertGreater(metadata['costs']['target_probability_queries'],0)

    def test_prepared_pins_include_preparation_only_source_graph(self):
        # The frozen loader pins five preparation artifacts plus every source
        # used to construct them, not just the trainer's own import graph.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root/'runs'/name for name in ('metadata.json','identity.json','array.npy','selection.json','plan.json')]
            sources = {'prepare_only.py':'a'*64,'transitive_helper.py':'b'*64}
            with patch.object(selector,'sha',return_value='c'*64) as digest:
                pins = selector.prepared_file_pins(root,paths,sources)
            expected = {str(path.resolve()):'c'*64 for path in paths}
            expected.update({str((root/name).resolve()):value for name,value in sources.items()})
            self.assertEqual(pins,expected)
            self.assertEqual(len(pins),7)
            self.assertEqual(digest.call_count,5)
            self.assertNotEqual(pins,{str(path.resolve()):'c'*64 for path in paths})

    def test_prepared_source_pins_remain_bound_to_verified_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = 'preparation_dependency.py'
            before = selector.prepared_file_pins(root,[],{source:'a'*64})
            after = selector.prepared_file_pins(root,[],{source:'b'*64})
            self.assertNotEqual(before,after)
            self.assertEqual(before[str((root/source).resolve())],'a'*64)
            with self.assertRaisesRegex(ValueError,'Unsafe|escapes'):
                selector.prepared_file_pins(root,[],{'../outside.py':'a'*64})

    def test_prepared_must_use_a_parent_not_bc_winner(self):
        metadata,parent,build = metadata_fixture(); metadata['parent_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError,'Phase A'):
            selector.verify_prepared_schema(metadata,parent,'a'*64,'b'*64,build)

    def test_preparation_rejects_double_counting_reused_tables(self):
        metadata,parent,build = metadata_fixture(); metadata['costs']['new_count_scan_token_occurrences'] = selector.bc.TRAIN_TOKENS
        with self.assertRaisesRegex(ValueError,'double-count'):
            selector.verify_prepared_schema(metadata,parent,'a'*64,'b'*64,build)

    def test_preparation_requires_all_train_parity_and_correct_window_reset(self):
        metadata,parent,build = metadata_fixture(); metadata['folds'][3]['parity']['comparisons'] = 47
        with self.assertRaisesRegex(ValueError,'parity'):
            selector.verify_prepared_schema(metadata,parent,'a'*64,'b'*64,build)
        metadata,parent,build = metadata_fixture(); metadata['window_columns'] = 'always column 3'
        with self.assertRaisesRegex(ValueError,'reset'):
            selector.verify_prepared_schema(metadata,parent,'a'*64,'b'*64,build)

    def test_preparation_cannot_deploy_array_or_hide_neural_in_sample_bias(self):
        for key,value in (('include_in_inference_assets',True),('neural_parent_saw_all_train',False)):
            metadata,parent,build = metadata_fixture(); metadata[key] = value
            with self.assertRaisesRegex(ValueError,'training-only'):
                selector.verify_prepared_schema(metadata,parent,'a'*64,'b'*64,build)

    def test_two_recipes_only_differ_in_objective(self):
        control = selector.expected_recipe({'mkn_weight':.15,'temperature':1.3},'control')
        joint = selector.expected_recipe({'mkn_weight':.15,'temperature':1.3},'joint')
        self.assertEqual((joint['steps'],joint['seed'],joint['eval_every'],joint['warmup']), (12000,211,2000,200))
        for value in (control,joint):
            value.pop('objective'); value.pop('objective_description')
        self.assertEqual(control,joint)

    def test_thirteen_history_choices_and_true_coin_lookup_costs(self):
        for objective in ('control','joint'):
            metrics = history_fixture(objective)
            self.assertEqual(selector.verify_training_history(metrics,500,objective)['step'],12000)
            self.assertEqual(metrics['cached_target_probability_lookups'],98304000)
        self.assertEqual(history_fixture('control')['joint_count_kept_target_occurrences'],0)

    def test_history_rejects_missing_choices_failed_and_resume(self):
        for mutate in (lambda m:m['validation_history'].pop(),lambda m:m.update(status='failed'),
                       lambda m:m.update(resume_history=[{'session':1}])):
            metrics = history_fixture(); mutate(metrics)
            with self.assertRaises(ValueError):
                selector.verify_training_history(metrics,500,'control')

    def test_historical_coin_counts_and_final_cost_must_agree(self):
        metrics = history_fixture('joint'); metrics['history'][-1]['count_block_coins']['kept'] += 1
        with self.assertRaisesRegex(ValueError,'trajectory'):
            selector.verify_training_history(metrics,500,'joint')
        metrics = history_fixture('joint'); metrics['joint_count_kept_target_occurrences'] = 0
        with self.assertRaisesRegex(ValueError,'costs'):
            selector.verify_training_history(metrics,500,'joint')

    def test_paired_sampler_and_dropout_final_states_match(self):
        control,joint = paired_fixture('control'),paired_fixture('joint')
        self.assertTrue(selector.verify_pair(control,joint)['passed'])
        joint['sampler_state'][0] = 8
        with self.assertRaisesRegex(ValueError,'RNG'):
            selector.verify_pair(control,joint)

    def test_pair_rejects_changed_schedule_or_coin_trajectory(self):
        control,joint = paired_fixture('control'),paired_fixture('joint')
        joint['recipe']['muon_lr'] *= 2
        with self.assertRaisesRegex(ValueError,'recipes'):
            selector.verify_pair(control,joint)
        control,joint = paired_fixture('control'),paired_fixture('joint')
        joint['trajectory'] = []
        with self.assertRaisesRegex(ValueError,'streams'):
            selector.verify_pair(control,joint)

    def test_old_parent_retained_and_improvement_gate_separate(self):
        rows = [dict(name='a',selection_eligible=True,validation_bpb_gpu_fp32=1.447),
                dict(name='d',selection_eligible=True,validation_bpb_gpu_fp32=1.4468)]
        result = selector.select(rows,1.447,[])
        self.assertEqual(result['winner']['name'],'d'); self.assertFalse(result['passes_predeclared_improvement_gate'])
        self.assertTrue(result['all_declared_phases_accounted'])
        result = selector.select(rows,1.447,[{'phase':'failed-c'}])
        self.assertIsNone(result['winner']); self.assertFalse(result['all_declared_phases_accounted'])

    def test_npy_header_checked_without_decoding_values(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'push150-oof-data'; directory.mkdir()
            path = directory/'train_oof_probabilities.npy'; np.save(path,np.ones((10,4),dtype=np.float32))
            record = dict(bytes=path.stat().st_size,sha256=selector.sha(path),shape=[10,4])
            with patch.object(np,'load',side_effect=AssertionError('Must not decode training array')):
                selector.verify_array_header(path,record)
            record['shape'] = [10,3]
            with self.assertRaisesRegex(ValueError,'header'):
                selector.verify_array_header(path,record)

    def test_no_corpus_hash_or_unknown_npy_path(self):
        with patch.object(Path,'open',side_effect=AssertionError('No I/O')):
            for path in ('data/wikitext_test.txt','data/wikitext_train.txt','somewhere/train_oof_probabilities.npy'):
                with self.assertRaises(ValueError):
                    selector.sha(Path(path))

    def test_latest_unit_record_must_include_new_selector(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'tests').mkdir()
            (root/'push150_select.py').write_text('pass\n'); (root/'push150_old.py').write_text('pass\n')
            (root/'tests/test_push150_select.py').write_text('pass\n')
            def write_unit(complete):
                sources = {path.name:selector.sha(path) for path in root.glob('push150*.py')}
                if not complete:
                    sources.pop('push150_select.py')
                record = dict(successful=True,failures=[],errors=[],skipped=[],test_text_loaded=False,tests_run=4,
                    source_sha256=sources,unit_source_sha256={'test_push150_select.py':selector.sha(root/'tests/test_push150_select.py')})
                path = root/('complete.json' if complete else 'old.json'); path.write_text(json.dumps(record)); return path
            old,new = write_unit(False),write_unit(True)
            with self.assertRaisesRegex(ValueError,'cover all'):
                selector.verify_latest_units(root,[old])
            unit,history = selector.verify_latest_units(root,[old,new])
            self.assertEqual(unit,dict(path='complete.json',sha256=selector.sha(new),executions=4))
            self.assertEqual(history['execution_count'],8)

    def test_new_phase_d_frozen_source_graph_is_closed(self):
        # Python source metadata only; no training module is imported/executed.
        for implementation in ('push150_expand','push147_signed_cache'):
            sources = selector.graph(selector.ROOT,['push150_oof_train.py',implementation+'.py','push148_muon.py','push150_oof_prepare.py'])
            self.assertTrue({'push150_oof_train.py','push150_oof_prepare.py','evaluate.py'} <= set(sources))
        prepare = selector.graph(selector.ROOT,['push150_oof_prepare.py','push150_phase_a_report.py',
                                                 'push150_expand.py',*selector.bc.GATE_SOURCE_NAMES])
        self.assertIn('push150_phase_a_report.py',prepare)

    def test_missing_bc_preserves_completed_failed_d_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'runs/push150-oof-joint'; directory.mkdir(parents=True)
            metrics = dict(status='failed',completed_steps=3000,discarded_completed_steps=0,
                incremental_search_targets=3000*8192,incremental_sampled_label_targets=3000*8192,
                lineage_new_targets=3000*8192,seconds=42.,failure_type='Synthetic',failure_message='fixture',validation_history=[])
            (directory/'metrics.json').write_text(json.dumps(metrics))
            result = selector.collect_report(root,[root/'missing_units.json'])
            self.assertEqual(result['status'],'refused'); self.assertIsNone(result['winner'])
            self.assertEqual(result['cost_summary']['phase_d_new_label_occurrences'],3000*8192)
            self.assertFalse(result['cost_summary']['accounting_verified'])
            self.assertTrue(result['final_resource_audit_pending'])

    def test_partial_calibration_json_is_preserved_as_error_not_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'runs/push150-oof-joint-calibrated'; directory.mkdir(parents=True)
            (directory/'metrics.json').write_text('{"partial":')
            result = selector.collect_report(root,[root/'missing_units.json'])
            self.assertEqual(result['status'],'refused')
            evidence = result['observed_phase_d']['runs'][1]['calibration_records']['metrics.json']
            self.assertIn('observation_error',evidence)

    def test_complete_synthetic_training_state_and_scoring_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'runs/push150-oof-control'; directory.mkdir(parents=True)
            parent_sha = 'a'*64
            parent = dict(protocol=selector.a.PROTOCOL,implementation='push150_expand',train_tokens=500,
                config={'temperature':1.3,'mkn_weight':.15,'dropout':.2,'discount_scale':.55},
                model={'token.weight':torch.ones(2),'head.weight':torch.ones(2),'ngram.unigram':torch.ones(2)},ancestry={'fixed':True})
            recipe = selector.expected_recipe(parent['config'],'control')
            sources = {'push150_expand.py':selector.FROZEN_SOURCES['push150_expand.py']}
            preparation,pins = {'synthetic':'metadata'},{'synthetic':'b'*64}
            identity = dict(parent_sha256=parent_sha,implementation='push150_expand',recipe=recipe,
                source_sha256=sources,data_sha256=selector.a.DATA_SHA,oof_preparation=preparation,
                prepared_file_sha256=pins,runtime=selector.a.RUNTIME)
            metrics = history_fixture()
            metrics.update(parent_sha256=parent_sha,parent_train_targets=500,training_implementation='push150_expand',
                inference_implementation='push150_expand',inference_architecture_unchanged_from_parent=True,
                recipe=recipe,identity=identity,training_config=recipe['training_config'],validation_config=recipe['validation_config'],
                seed=211,steps=12000,eval_every=2000,batch_size=32,dropout_views_per_sample=1,
                processed_view_targets_per_step=8192,sampled_label_targets_per_step=8192,optimizer='muon',training_precision='bf16',
                ema_decay=.99,no_statistical_tables_in_training=True,source_splits=['train','validation'],test_used=False,
                prior_test_exposure_disclosed=True,source_sha256=sources,data_sha256=selector.a.DATA_SHA,
                implementation_sha256=sources['push150_expand.py'],oof_preparation=preparation,objective='control',
                optimizer_initialization='Fresh optimizers and EMA from scoring parent; old optimizer states unavailable',
                incremental_search_targets=12000*8192,incremental_sampled_label_targets=12000*8192,
                lineage_new_targets=12000*8192,seconds=10.)
            def cp(step,label):
                return dict(protocol=selector.a.PROTOCOL,implementation='push150_expand',config=recipe['validation_config'],
                    model=copy.deepcopy(parent['model']),seed=211,parent_sha256=parent_sha,train_tokens=500+step*8192,
                    selected_step=step,weights=label,training_config=recipe['training_config'],training_recipe=recipe,
                    ancestry=parent['ancestry'],selection='synthetic',inference_architecture_unchanged_from_parent=True)
            trained = cp(12000,'ema')
            torch.save(trained,directory/'checkpoint.pt'); torch.save(trained,directory/'best.pt')
            metrics['best'].update(checkpoint_file='best.pt',checkpoint_sha256=selector.sha(directory/'best.pt'),
                                   scoring_checkpoint_sha256=selector.sha(directory/'checkpoint.pt'))
            metrics['tail_snapshots'] = []
            for step in (10000,12000):
                for label in ('raw','ema'):
                    path = directory/f'tail-{step}-{label}.pt'; torch.save(cp(step,label),path)
                    metrics['tail_snapshots'].append(dict(step=step,weights=label,path=path.name,sha256=selector.sha(path)))
            (directory/'metrics.json').write_text(json.dumps(metrics))
            neural = {key:value for key,value in parent['model'].items() if not key.startswith('ngram.')}
            state = dict(format_version=1,identity=identity,step=12000,log=metrics,raw=neural,ema=neural,
                optimizers=[{'state':{0:'synthetic'}},{'state':{1:'synthetic'}}],
                rng=dict(python=(),numpy={},torch_cpu=torch.zeros(1,dtype=torch.uint8),sampler=torch.ones(1,dtype=torch.uint8),
                         torch_cuda=[torch.ones(1,dtype=torch.uint8)]))
            torch.save(state,directory/'train_state.pt')
            with patch.object(selector,'graph',return_value=sources),patch.object(selector.bc,'verify_calibration'):
                row = {}; paired = selector.verify_run(root,'oof-control',parent,parent_sha,preparation,pins,row)
                self.assertTrue(row['accounting_verified']); self.assertEqual(row['selected_step'],12000)
                self.assertTrue(torch.equal(paired['sampler_state'],torch.ones(1,dtype=torch.uint8)))
                trained['training_probability_cache'] = torch.zeros(1)
                torch.save(trained,directory/'checkpoint.pt')
                metrics['best']['scoring_checkpoint_sha256'] = selector.sha(directory/'checkpoint.pt')
                (directory/'metrics.json').write_text(json.dumps(metrics))
                with self.assertRaisesRegex(ValueError,'undeclared training payload'):
                    selector.verify_run(root,'oof-control',parent,parent_sha,preparation,pins,{})

    def test_evidence_manifest_includes_failure_skip_plans_not_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'runs/push150-fast-dense').mkdir(parents=True)
            (root/'runs/push150-oof-joint').mkdir(); (root/'data').mkdir(); (root/'tests').mkdir()
            (root/'runs/push150-fast-dense/training_decision.json').write_text('{}')
            (root/'runs/push150-oof-joint/preflight-failure-001.json').write_text('{}')
            (root/'runs/push150_phase_d_oof_plan.json').write_text('{}')
            for name in selector.HISTORICAL_METADATA_EVIDENCE:
                (root/'runs'/name).write_text('{"historical":true}')
            (root/'data/wikitext_test.txt').write_text('never read')
            with patch.object(selector,'FROZEN_SOURCES',{}):
                result = selector.evidence_manifest(root,[],[])
            self.assertIn('runs/push150-fast-dense/training_decision.json',result)
            self.assertIn('runs/push150-oof-joint/preflight-failure-001.json',result)
            for name in selector.HISTORICAL_METADATA_EVIDENCE:
                self.assertIn('runs/'+name,result)
            self.assertIn('Historical pre-repair',selector.HISTORICAL_METADATA_EVIDENCE['push150_unit_tests_final.json'])
            self.assertFalse(any('wikitext' in path for path in result))

    def test_exclusive_report_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'selection.json'; selector.a.write_report(path,{'status':'refused'})
            with self.assertRaises(FileExistsError):
                selector.a.write_report(path,{'status':'complete'})


if __name__ == '__main__':
    unittest.main()
