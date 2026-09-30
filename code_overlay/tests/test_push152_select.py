"""Synthetic metadata tests only; no weights, corpus, torch or real selection."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import push152_select as select


def synthetic_plan():
    return dict(parent=dict(validation_bpb_gpu_fp32=1.44),phase_a=dict(
        learning_rates=dict(backbone=7.5e-7,pointer=7.5e-6,scalar=2.5e-5,muon=5e-5),
        variants={name:dict(run_dir='runs/push152-'+name) for name in select.NAMES[:2]}),
        phase_b=dict(variants={name:dict(run_dir='runs/push152-'+name) for name in select.NAMES[2:]}))


def fixture(name='optimizer-keep'):
    phase = 'A' if name in select.NAMES[:2] else 'B'
    plan = synthetic_plan()
    config = dict(vocab=2048,width=320,heads=8,depth=6,context=256,ffn_hidden=864,
        dropout=.2,push150_ranks=select.ranks(name),temperature=1.15)
    evaluation = dict(config,temperature=1.3,mkn_weight=.15,mkn_order=5,discount_scale=.7)
    recipe = dict(optimizer='muon',batch_size=32,context=256,training_precision='bf16',
        training_temperature=1.15,validation_temperature=1.3,validation_mkn_weight=.15,dropout=.2,
        gradient_clip=1.,training_config=config,validation_config=evaluation)
    if phase == 'A':
        recipe.update(phase='A',phase_name=name,steps=8000,eval_every=2000,schedule='constant',
            learning_rates=plan['phase_a']['learning_rates'],optimizer_state_kept=name=='optimizer-keep',
            ema_decays=dict(ema99=.99,ema999=.999),validation_precision='fp32',
            validation_choices_per_completed_run=14,additional_calibration_choices=0,
            omitted_unused_oof_target_lookup=True,objective='pure neural-plus-pointer target NLL')
    else:
        recipe.update(branch=name,steps=20000,eval_every=4000,seed=193,schedule='cosine',warmup=200,
            decay_steps=4000,lr_floor=.1,backbone_lr=7.5e-6,head_lr=7.5e-5,scalar_lr=.00025,
            muon_lr=.0005,ema_decay=.99,validation_choices=11,all_validation_snapshots_saved=True,
            auxiliary_weight=.1 if name=='auxiliary' else 0.,dropout_views_per_sample=1,
            adamw_betas=[.9,.95],backbone_weight_decay=.1,pointer_weight_decay=.01,
            scalar_weight_decay=0.,muon_momentum=.95,muon_ns_steps=5)
    parent = 'e'*64 if name=='expanded' else select.PARENT_SHA
    sources = {select.DRIVERS[phase][0]:select.DRIVERS[phase][1]}
    identity = dict(plan_sha256=select.PLAN_SHA,recipe=recipe,implementation='push150_expand',
        source_sha256=sources,data_sha256=copy.deepcopy(select.DATA_SHA),
        runtime=dict(cpu_threads=4,device='cuda:0',cuda_device_count=1))
    if phase=='A':
        identity.update(phase_name=name,fork_training_state_sha256=select.OLD_STATE_SHA,
            statistics_parent_sha256=select.PARENT_SHA)
    else:
        identity.update(branch=name,parent_sha256=parent,statistical_reference_sha256=select.PARENT_SHA)
    steps = recipe['steps']; targets=steps*select.TARGETS_PER_STEP
    record = dict(status='completed',identity=identity,recipe=recipe,plan_sha256=select.PLAN_SHA,
        parent_sha256=parent,parent_train_targets=select.LINEAGE,test_used=False,
        source_splits=['train','validation'],training_implementation='push150_expand',
        inference_implementation='push150_expand',implementation_sha256=select.IMPLEMENTATION_SHA,
        no_statistical_tables_in_training=True,source_sha256=sources,data_sha256=identity['data_sha256'],
        parameters=5597331+(681984 if name=='expanded' else 0),processed_view_targets_per_step=8192,
        sampled_label_targets_per_step=8192,completed_steps=steps,discarded_completed_steps=0,
        incremental_search_targets=targets,incremental_sampled_label_targets=targets,lineage_new_targets=targets,
        seconds=123.,validation_history=[])
    snapshots=[]
    for index,(step,weight) in enumerate(select.menu(phase)):
        bpb=1.439-index*.00015
        prefix='session' if phase=='A' else 'validation-session'
        filename=f'{prefix}000-step{step:08d}-{weight}.pt'
        row=dict(step=step,weights=weight,session=0,checkpoint_file=filename,
            checkpoint_sha256=f'{index+1:064x}',bpb=bpb,nll_nats=bpb*math.log(2)*1148007,
            targets=376599,utf8_bytes=1148007,token_ppl=20.,seconds=1.)
        if phase=='A':
            row.update(selected_lineage_targets=select.LINEAGE+step*8192,protocol=select.PROTOCOL,
                split='validation',precision='fp32',device='cuda:0',test_used=False,
                implementation_sha256=select.IMPLEMENTATION_SHA,evaluator_sha256=select.SCORER_SHA,
                tokenizer_sha256=select.DATA_SHA['data/tokenizer.json'],window_nll_sha256='a'*64)
        record['validation_history'].append(row)
        snapshots.append(dict(step=step,weights=weight,path=filename,sha256=row['checkpoint_sha256']))
    record['best']=dict(record['validation_history'][-1],selected_lineage_targets=select.LINEAGE+steps*8192,
        scoring_checkpoint_sha256='b'*64)
    if phase=='A':
        record.update(snapshots=snapshots,fork_training_state_sha256=select.OLD_STATE_SHA,
            cached_target_probability_lookups=0,new_statistics_scans=0,same_ancestor_not_summed=True,
            count_block_coins=dict(drawn=steps,kept=steps//2,dropped=steps//2),
            count_block_coins_including_discarded=dict(drawn=steps,kept=steps//2,dropped=steps//2),
            omitted_unused_oof_target_lookups=targets,discarded_validation_passes=0,
            validation_completed_records=14,validation_attempts_recorded=14)
    else:
        record.update(branch=name,validation_snapshots=snapshots,discarded_validation_history=[],
            completed_validation_calls=11,mixed_nll_target_exposures=targets,
            auxiliary_neural_nll_target_exposures=targets if name=='auxiliary' else 0)
    return record,phase,name,plan,parent


class FixedMenuTests(unittest.TestCase):
    def test_all_five_complete_costs_and_61_choices(self):
        allrows=[]; costs=[]
        for name in select.NAMES:
            rows,cost=select.check_metrics(*fixture(name)); allrows+=rows; costs.append(cost)
        self.assertEqual(len(allrows),61)
        self.assertEqual(sum(cost['incremental_search_targets'] for cost in costs),622592000)
        self.assertEqual(sum(cost['auxiliary_neural_nll_target_exposures'] for cost in costs),163840000)
        self.assertEqual(max(row['selected_lineage_targets'] for row in allrows),802816000)

    def test_two_ema_variants_are_not_extra_training(self):
        rows,cost=select.check_metrics(*fixture())
        self.assertEqual(cost['incremental_search_targets'],65536000)
        self.assertEqual(len(rows),14)

    def test_unplanned_step_zero_long_ema_rejected(self):
        args=fixture();args[0]['validation_history'][1]['weights']='ema999'
        with self.assertRaisesRegex(ValueError,'menu'):
            select.check_metrics(*args)

    def test_incomplete_completed_menu_rejected(self):
        args=fixture();args[0]['validation_history'].pop()
        with self.assertRaisesRegex(ValueError,'entire fixed menu'):
            select.check_metrics(*args)

    def test_reordered_b_menu_rejected(self):
        args=fixture('control');history=args[0]['validation_history'];history[1],history[2]=history[2],history[1]
        with self.assertRaisesRegex(ValueError,'menu'):
            select.check_metrics(*args)

    def test_running_not_selectable(self):
        args=fixture();args[0]['status']='running'
        with self.assertRaisesRegex(ValueError,'pending/running'):
            select.check_metrics(*args)

    def test_wrong_training_source_rejected(self):
        args=fixture();args[0]['source_sha256']['push152_resume.py']='f'*64
        with self.assertRaisesRegex(ValueError,'source/data'):
            select.check_metrics(*args)

    def test_changed_validation_temperature_rejected(self):
        args=fixture();args[0]['recipe']['validation_config']['temperature']=1.31
        with self.assertRaisesRegex(ValueError,'calibration'):
            select.check_metrics(*args)

    def test_changed_auxiliary_coefficient_rejected(self):
        args=fixture('auxiliary');args[0]['recipe']['auxiliary_weight']=.2
        with self.assertRaisesRegex(ValueError,'branch recipe'):
            select.check_metrics(*args)

    def test_undeclared_expansion_rejected(self):
        args=fixture('expanded');args[0]['recipe']['training_config']['push150_ranks']['out']=192
        with self.assertRaisesRegex(ValueError,'architecture'):
            select.check_metrics(*args)

    def test_wrong_full_validation_size_rejected(self):
        args=fixture();args[0]['validation_history'][0]['targets']-=1
        with self.assertRaisesRegex(ValueError,'Incomplete validation'):
            select.check_metrics(*args)

    def test_wrong_bpb_denominator_and_nonfinite_rejected(self):
        for key,value in [('nll_nats',1.),('bpb',float('nan'))]:
            args=fixture();args[0]['validation_history'][0][key]=value
            with self.assertRaises(ValueError):select.check_metrics(*args)

    def test_best_must_match_true_minimum(self):
        args=fixture();args[0]['best']['step']=0
        with self.assertRaisesRegex(ValueError,'Best checkpoint'):
            select.check_metrics(*args)

    def test_no_doubled_shared_ancestor_or_auxiliary_targets(self):
        for name in ('optimizer-keep','auxiliary'):
            args=fixture(name);args[0]['incremental_search_targets']*=2
            with self.assertRaisesRegex(ValueError,'Training cost'):
                select.check_metrics(*args)

    def test_failed_prefix_excluded_but_replay_cost_retained(self):
        args=fixture();record=args[0]
        record.update(status='failed',failure_type='RuntimeError',failure_message='',completed_steps=2000,
            discarded_completed_steps=200,incremental_search_targets=2200*8192,
            incremental_sampled_label_targets=2200*8192,lineage_new_targets=2000*8192)
        record['validation_history']=record['validation_history'][:5]
        record['snapshots']=record['snapshots'][:5]
        record['best']=dict(record['validation_history'][-1],scoring_checkpoint_sha256='b'*64)
        record['count_block_coins_including_discarded']=dict(drawn=2200,kept=1100,dropped=1100)
        record['omitted_unused_oof_target_lookups']=2200*8192
        rows,cost=select.check_metrics(*args)
        self.assertEqual(rows,[])
        self.assertEqual(cost['incremental_search_targets'],18022400)
        self.assertTrue(cost['partial_failed_work_unquantified'])


class DispositionAndRankTests(unittest.TestCase):
    def terminal(self):
        return dict(status='skipped',phase='B',name='expanded',plan_sha256=select.PLAN_SHA,
            driver_sha256=select.DRIVERS['B'][1],test_used=False,no_running_processes=True,
            reason='preflight resource limit',completed_steps=0,discarded_completed_steps=0,
            incremental_search_targets=0,lineage_new_targets=0,metrics_sha256=None,
            partial_failed_work_unquantified=False)

    def test_only_unauthorized_expansion_can_skip(self):
        select.check_terminal(self.terminal(),'B','expanded',dict(training_authorized=False))
        with self.assertRaises(ValueError):
            select.check_terminal(self.terminal(),'B','expanded',dict(training_authorized=True))
        terminal=self.terminal();terminal['name']='control'
        with self.assertRaises(ValueError):
            select.check_terminal(terminal,'B','control',dict(training_authorized=False))

    def test_skip_cannot_hide_completed_work(self):
        terminal=self.terminal();terminal.update(completed_steps=1,incremental_search_targets=8192,lineage_new_targets=8192)
        with self.assertRaises(ValueError):
            select.check_terminal(terminal,'B','expanded',dict(training_authorized=False))

    def test_failed_missing_metrics_must_disclose_unknown_partial_work(self):
        terminal=self.terminal();terminal.update(status='failed',failure_type='AssertionError',failure_message='',
            partial_failed_work_unquantified=True)
        select.check_terminal(terminal,'B','expanded',dict(training_authorized=True))
        terminal['partial_failed_work_unquantified']=False
        with self.assertRaises(ValueError):
            select.check_terminal(terminal,'B','expanded',dict(training_authorized=True))

    def test_deterministic_ties_and_top_three(self):
        rows=[]
        for name in reversed(select.NAMES):
            for step,weight in [(8000,'ema999'),(8000,'raw'),(2000,'raw')]:
                rows.append(dict(name=name,step=step,weights=weight,validation_bpb_gpu_fp32=1.438,
                    passes_validation_improvement_gate=True))
        ranked=select.rank_candidates(rows)
        self.assertEqual([(r['name'],r['step'],r['weights']) for r in ranked[:3]],
            [('optimizer-keep',2000,'raw'),('optimizer-keep',8000,'raw'),('optimizer-keep',8000,'ema999')])
        rows[0]['passes_validation_improvement_gate']=False
        self.assertEqual(len(select.rank_candidates(rows)),14)

    def test_endpoint_comparison_does_not_choose_better_early_snapshot(self):
        rows=[]
        for name in select.NAMES:
            found,_=select.check_metrics(*fixture(name));rows.extend(found)
        rows[0]['validation_bpb_gpu_fp32']=.5
        comparison=select.endpoint_comparisons(rows)
        self.assertGreater(comparison['phase_a']['raw']['keep_bpb'],1.)
        self.assertEqual(comparison['phase_b']['expanded-ema']['candidate_minus_control_bpb'],0.)


class PathAndLaunchTests(unittest.TestCase):
    def test_unsafe_path_corpus_and_final_test_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('../outside','/absolute','C:/absolute','a//b','a/./b'):
                with self.assertRaises(ValueError):select.child(directory,name)
            for name in ('wikitext_test.txt','wikitext_train.txt','push152-final-test/x.json'):
                with self.assertRaisesRegex(ValueError,'prohibited'):select.sha(Path(directory)/name)
            with self.assertRaises(ValueError):select.read(Path(directory)/'weights.pt')

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'output.json';select.write_new(path,dict(synthetic=True))
            with self.assertRaises(FileExistsError):select.write_new(path,{})

    def test_explicit_utc_required(self):
        select.utc('2026-10-01T00:00:00+00:00')
        for stamp in ('2026-10-01T00:00:00','2026-10-01T00:00:00+08:00'):
            with self.assertRaises(ValueError):select.utc(stamp)

    def make_launch(self,root,name='optimizer-keep'):
        phase='A' if name in select.NAMES[:2] else 'B'
        directory=root/'runs/push152-launches';directory.mkdir(parents=True)
        base=directory/name;driver,digest=select.DRIVERS[phase]
        argv=[name,'--plan',select.PLAN] if phase=='A' else ['--branch',name,'--plan',select.PLAN]
        started=dict(phase=name,command=[str(root/'.venv-gpu/Scripts/python.exe'),'-B',str(root/driver),*argv],
            cwd=str(root),plan_sha256=select.PLAN_SHA,script_sha256=digest,launcher_sha256=select.LAUNCHER_SHA,
            started_at_utc='2026-10-01T00:00:00+00:00',test_used=False)
        select.write_new(str(base)+'.started.json',started)
        process=dict(phase=name,started_record_sha256=select.sha(str(base)+'.started.json'),
            script_unchanged=True,test_used=False,seconds_total=1.,completed_at_utc='2026-10-01T00:00:01+00:00',
            returncode=0,failure=None,output_sha256=None)
        select.write_new(str(base)+'.process.json',process)
        return phase,started,process,base

    def test_launch_accepts_exact_command_and_exit(self):
        for name in ('optimizer-keep','auxiliary'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();phase,_,_,_=self.make_launch(root,name)
                self.assertEqual(select.check_launch(root,phase,name,'completed',{}),1.)

    def test_launch_rejects_failed_exit_with_completed_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();phase,_,process,base=self.make_launch(root)
            process['returncode']=1
            Path(str(base)+'.process.json').write_text(json.dumps(process),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'successful process exit'):
                select.check_launch(root,phase,'optimizer-keep','completed',{})

    def test_launch_rejects_command_extra_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();phase,started,_,base=self.make_launch(root)
            started['command'].append('--calibrate')
            Path(str(base)+'.started.json').write_text(json.dumps(started),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'launch command'):
                select.check_launch(root,phase,'optimizer-keep','completed',{})


if __name__=='__main__':
    unittest.main()
