"""Read-only evidence selection for the declared push150 phase A.

Never instantiates a predictor, scores a split, or reads/hashes corpus text.
Checkpoint deserialization is CPU-only, for metadata and weight-equality checks.
An exclusive JSON record is written even when evidence refuses selection.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent
PROTOCOL='7506-mp1-wt2-v2'
PLAN_SHA='55d9473934d9b528278b6a297a3e4ef34bd40982674ddc2d810cfc99fb803afe'
PARENT_SHA='97696b7fe5e0e0f5c7573f24efd43e86579c48a29a9e995b8c03829cbaf26867'
PARENT_TARGETS=376832000
BASELINE_SHA='b7fdb67c8ef14b5166c0de1f17cf306f34800b41dfe97da856c9812cdb1f9701'
BASELINE_IMPL_SHA='6cce142a31d396f017319543f50d70f30fa6da740729af01c8baed13c6f671f9'
RUNS=('control-cosine','control-wsd','expanded-cosine')
DATA_SHA={
    'data/manifest.json':'f4790bb599d9fcd04e48263e685335e2f62d9bca0324b08061a10c38984f4fe4',
    'data/tokenizer.json':'020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e',
    'data/wikitext_train.txt':'c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8',
    'data/wikitext_validation.txt':'714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307',
}
TRAIN_SOURCES={
    'common.py':'d40dc1fe6578b5993a563bdb4d9e5317ddc9fae8afe48822b66571c7042e84db',
    'efficient.py':'bd2171ee6d11be80c94054c0d565f656e99e471ebb049f6a8018d28467278ab0',
    'evaluate.py':'128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d',
    'exact_refined.py':'c48317dbed64fd34b9e145484a9c1a969ff44c301b972166a7609f589df22528',
    'fast_refined.py':'4968314ff059330a1d27f7cba7abe3e0f6d9de9eee9bd5d1af2f5491647d9335',
    'lowrank.py':'757d8df342a328d9da8c623a617acae100f32a4ba13a9326782137bb66cd1902',
    'ngram_hybrid.py':'babff6003d79381798825c1be47dbd0903487964477303fe0baa52a0c7a5caf3',
    'push145_cache.py':'684fdfa1e80f8c629736abc4843f9f6d5c6985faa35730c6e0bc9260070743ca',
    'push145_cache_hybrid.py':'1270ae1db1b0923b65c7d47ab9eb43c55c1e9b302ced20f38f887d43f759ac67',
    'push145_mkn.py':'afb3a47dfd40738c544173b4d97a6bc9478807a87fddb00d700a9df3b97cd289',
    'push147_cache.py':'956f94727ddeb1d690c039c669631c4c20b08e56bf885a6c54cb86022c45e86d',
    'push147_signed_cache.py':'c20fc5411f63010cb9b3f3a9952f0c8aac851d55a8ee1c0ccc2ab5680ed901a3',
    'push147_train.py':'bbdeb09f4d8e850156cc4f9765c55addaa88760f76990b8bcf04f40f582ed1e6',
    'push148_losses.py':'dd7c9c46b3886b4eedcb0895998b123155e22794476e968f969bbd15cad67aed',
    'push148_muon.py':'e7c05976ea395799275f162162359d4c77263a085f8bb9de41891983ae56ce56',
    'push148_train.py':'e28dca9ad752a00632226f35eadfa2831d280029072ed548e95fdb537507f643',
    'push150_expand.py':'d0340092e2a2882113533f05f4668c48b721ee4416842910c2ba2ba1758e1d45',
    'push150_train.py':'dd0ea441a6aef739b862c2258bde3baea34c83b67109a191ac2135f1a7f3f6b6',
    'refined.py':'6efc8af8cde267430af6c3e81426f2c1cedd6d04112b4d79a8917598a1e5bb7b',
    'student.py':'d7230b25ef86bd2544a071ea36e26fa9a2a19d4ece02e669113be8f89ba8cc8a',
}
INIT_SHA='7c3788118d8e18027753e3e7d510a4c504bfc1a18697320cc5693307fa3c07b3'
CALIBRATE_SHA='a75c4c2f40ce5da5d0254fdce14a82ac9f8dfa768967feb87d0587b7b025f99a'
MKN_SOURCES={name:TRAIN_SOURCES[name] for name in
    ('push145_mkn.py','push147_signed_cache.py','push147_cache.py','push145_cache_hybrid.py',
     'push145_cache.py','lowrank.py','refined.py','student.py','efficient.py','ngram_hybrid.py','common.py','evaluate.py')}
MKN_SOURCES.update({
    'push150_mkn_sweep.py':'feba4e7b310e8310554865fa6749cfe7ad7f9853fe1ea9ffc4e3ecbe6e3f93a6',
    'push145_mkn_build.py':'449eab742cda10c1f476e01210184a9cbb93a800d6fa3c430fc1456e2b5f010e',
    'push145_mkn_select.py':'18374312aa5b5c61139d4eda9aa107355418e556796d91dd030d37a64c6aa535'})
RUNTIME=dict(torch='2.7.1+cu128',numpy='2.5.3',python='3.12.14',cuda='12.8',device='cuda:0',
             device_name='NVIDIA GeForce RTX 5080 Laptop GPU',cuda_device_count=1,cpu_threads=4)
GRID={(t,w) for t in (1.2,1.25,1.3,1.35,1.4) for w in (.1,.15,.2)}
MKN_GRID={(s,w) for s in (.4,.55,.7) for w in (.1,.15,.2)}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def natural(value,label):
    require(isinstance(value,int) and not isinstance(value,bool) and value>=0,'Invalid '+label)
    return value


def finite(value,label):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),'Invalid '+label)
    return float(value)


def allowed(path):
    path=Path(path)
    require(path.suffix.lower() in ('.json','.pt','.py','.ps1'),'Refusing non-evidence file: '+str(path))
    return path


def sha(path):
    digest=hashlib.sha256()
    with allowed(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    path=allowed(path)
    require(path.suffix=='.json','JSON required')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def checkpoint(path):
    path=allowed(path)
    require(path.suffix=='.pt','Checkpoint required')
    value=torch.load(path,map_location='cpu',weights_only=True)
    require(value.get('protocol')==PROTOCOL,'Checkpoint protocol mismatch')
    return value


def child(root,name):
    result=(Path(root)/str(name).replace('\\','/')).resolve()
    require(result.is_relative_to(Path(root).resolve()),'Evidence path escapes its directory')
    return result


def verify_sources(root,recorded,expected):
    require(recorded==expected,'Recorded source graph differs from frozen graph')
    for name,digest in expected.items():
        require(sha(child(root,name))==digest,'Changed source: '+name)


def verify_measurement(row):
    require(row['targets']==376599 and row['utf8_bytes']==1148007,'Validation target/byte count mismatch')
    bpb=finite(row['bpb'],'validation BPB')
    require(bpb>0 and finite(row['seconds'],'scoring seconds')>0,'Nonpositive measurement')
    require(abs(finite(row['nll_nats'],'NLL')/math.log(2)/1148007-bpb)<1e-10,'NLL/BPB mismatch')
    return bpb


def same_tensors(first,second,prefix=None):
    keys={key for key in first if prefix is None or key.startswith(prefix)}
    require(keys=={key for key in second if prefix is None or key.startswith(prefix)},'Tensor keys changed')
    for key in keys:
        a,b=first[key],second[key]
        require(a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),'Tensor changed: '+key)


def training_accounting(training):
    step=natural(training['completed_steps'],'completed steps')
    extra=natural(training['discarded_completed_steps'],'discarded completed steps')
    require(step<=20000,'Completed steps exceed phase cap')
    total=(step+extra)*8192
    require(training['incremental_search_targets']==total and
            training['incremental_sampled_label_targets']==total,'Completed search accounting mismatch')
    require(training['lineage_new_targets']==step*8192,'Lineage accounting mismatch')
    seconds=finite(training['seconds'],'training seconds')
    require(seconds>=0,'Negative training time')
    return dict(completed_steps=step,discarded_completed_steps=extra,
        new_search_processed_view_targets=total,new_search_sampled_label_occurrences=total,
        training_loop_seconds=seconds,recorded_status=training.get('status'),
        failure_type=training.get('failure_type'),failure_message=training.get('failure_message'),
        failed_attempt_cost_note=training.get('failed_attempt_cost_note'),
        raw_ema_validation_choices=len(training.get('validation_history',[])))


def verify_history(training):
    require(not training.get('resume_history') and training['discarded_completed_steps']==0,
            'Resumed run requires manual merged-history/search accounting; automatic selection refused')
    history=training['validation_history']
    expected=[(0,'raw')]+[(step,kind) for step in range(2000,20001,2000) for kind in ('raw','ema')]
    actual=[(row['step'],row['weights']) for row in history]
    require(actual==expected[:len(actual)],'Validation choices are duplicated, reordered or undeclared')
    require(bool(actual),'Missing step-zero validation')
    require(all(step<=training['completed_steps'] for step,_ in actual),'Validation exceeds completed training')
    if training['status']=='completed':
        require(training['completed_steps']==20000 and actual==expected,'Completed run lacks all 21 choices')
    for row in history:
        require(row.get('session')==0,'Unexpected resumed validation session')
        verify_measurement(row)
    chosen=min(history,key=lambda row:row['bpb'])
    best=training['best']
    require(all(best[key]==chosen[key] for key in ('step','weights','bpb')),'Best snapshot is not the recorded validation minimum')
    require(best['selected_lineage_targets']==PARENT_TARGETS+best['step']*8192,'Selected lineage mismatch')
    return chosen


def expected_recipe(declared,initial_config):
    train={k:copy.deepcopy(v) for k,v in initial_config.items()
           if not k.startswith(('ngram_','mkn_')) and k!='discount_scale'}
    train.update(temperature=1.15,dropout=.2)
    validation=copy.deepcopy(initial_config); validation.update(temperature=1.25,mkn_weight=.15,dropout=.2)
    return dict(steps=20000,eval_every=2000,seed=173,optimizer='muon',schedule=declared['schedule'],
        warmup=200,decay_steps=4000,lr_floor=.1,backbone_lr=.0000075,head_lr=.000075,
        scalar_lr=.00025,muon_lr=.0005,batch_size=32,context=256,dropout_views_per_sample=1,
        dropout=.2,training_temperature=1.15,validation_temperature=1.25,validation_mkn_weight=.15,
        ema_decay=.99,gradient_clip=1.,adamw_betas=[.9,.95],backbone_weight_decay=.1,
        pointer_weight_decay=.01,scalar_weight_decay=0.,muon_momentum=.95,muon_ns_steps=5,
        training_precision='bf16',training_config=train,validation_config=validation,
        tail_snapshot_start_step=16000,objective='train-only neural+pointer NLL; no statistical table loss')


def verify_training_metadata(root,training,declared,init):
    recipe=expected_recipe(declared,init['checkpoint']['config'])
    require(training['recipe']==recipe,'Actual recipe differs from declared phase A')
    expected=dict(parent_sha256=init['report']['checkpoint_sha256'],parent_train_targets=PARENT_TARGETS,
        training_implementation='push150_expand',inference_implementation='push150_expand',
        inference_architecture_unchanged_from_parent=True,no_statistical_tables_in_training=True,
        source_splits=['train','validation'],test_used=False,seed=173,steps=20000,eval_every=2000,
        batch_size=32,dropout_views_per_sample=1,processed_view_targets_per_step=8192,
        sampled_label_targets_per_step=8192,optimizer='muon',training_precision='bf16',ema_decay=.99)
    for key,value in expected.items():
        require(training.get(key)==value,'Training field changed: '+key)
    require(training['training_config']==recipe['training_config'] and
            training['validation_config']==recipe['validation_config'],'Config differs from recipe')
    require(training['data_sha256']==DATA_SHA,'Data provenance differs from fixed data')
    verify_sources(root,training['source_sha256'],TRAIN_SOURCES)
    expected_identity=dict(parent_sha256=init['report']['checkpoint_sha256'],implementation='push150_expand',
        recipe=recipe,source_sha256=TRAIN_SOURCES,data_sha256=DATA_SHA,runtime=RUNTIME)
    require(training['identity']==expected_identity,'Runtime/resume identity differs')
    require(training['implementation_sha256']==TRAIN_SOURCES['push150_expand.py'],'Implementation SHA differs')


def verify_calibration(calibration,parent_sha,implementation='push150_expand'):
    require(calibration['parent_sha256']==parent_sha,'Calibration attached to another training snapshot')
    require(calibration['test_used'] is False and calibration['new_training_targets']==0,'Calibration data/cost changed')
    require(calibration['implementation']==implementation and
            calibration['implementation_sha256']==TRAIN_SOURCES[implementation+'.py'],'Calibration implementation differs')
    require(calibration['script_sha256']==CALIBRATE_SHA and
            calibration['evaluator_sha256']==TRAIN_SOURCES['evaluate.py'] and
            calibration['tokenizer_sha256']==DATA_SHA['data/tokenizer.json'] and
            calibration['validation_sha256']==DATA_SHA['data/wikitext_validation.txt'],'Calibration hashes differ')
    require(calibration['precision']=='fp32' and calibration['device']=='cuda:0','Calibration device/precision differs')
    trials=calibration['trials']
    require(len(trials)==15 and {(r['temperature'],r['mkn_weight']) for r in trials}==GRID,'Calibration is not the declared 15-choice grid')
    require(calibration['best']==min(trials,key=lambda r:finite(r['validation_bpb'],'trial BPB')),'Calibration best differs from grid minimum')
    full=verify_measurement(calibration['full_validation'])
    difference=abs(full-calibration['best']['validation_bpb'])
    require(difference<1e-6 and calibration['shortcut_matches_full_predictor'] is True and
            abs(difference-calibration['shortcut_full_bpb_absolute_difference'])<1e-12,'Full prediction did not confirm shortcut')
    require(calibration['validation_target_occurrences']==376599 and
            calibration['validation_utf8_bytes']==1148007,'Calibration split counts differ')
    return full


def collect_training(root,declared,init,feasibility,row):
    directory=root/'runs'/('push150-'+declared['name'])
    caldir=directory.with_name(directory.name+'-calibrated')
    eligible=feasibility['candidates'][declared['initialization']]['training_authorized_by_phase_a']
    if not (directory/'metrics.json').exists():
        require(not eligible and not directory.exists(),'Authorized run has not completed, or unexplained partial directory exists')
        row.update(status='skipped_cpu_precheck',skip_evidence='push150-feasibility/summary.json',
                   completed_steps=0,new_search_processed_view_targets=0,new_search_sampled_label_occurrences=0,
                   training_loop_seconds=0.,raw_ema_validation_choices=0)
        return
    training=read(directory/'metrics.json')
    row.update(training_accounting(training),training_metrics_sha256=sha(directory/'metrics.json'))
    row['resume_evidence']=[dict(path=path.name,sha256=sha(path)) for path in sorted(directory.glob('metrics-before-resume-*.json'))]
    row['resume_history']=training.get('resume_history',[])
    require(eligible,'Training occurred despite failed CPU feasibility gate')
    verify_training_metadata(root,training,declared,init)
    require(not row['resume_evidence'],'Archived resume evidence requires manual merged-history accounting')
    verify_history(training)
    status=training['status']
    if status=='failed':
        require(bool(training.get('failure_type')) and bool(training.get('failed_attempt_cost_note')),'Failure lacks cost/status evidence')
    elif status!='completed':
        stop=read(directory/'termination.json')
        require(stop.get('status') in ('stopped','cancelled','partial') and bool(stop.get('reason')) and
                stop['completed_steps']==training['completed_steps'] and
                stop['metrics_sha256']==row['training_metrics_sha256'],'Run is still active or lacks bound terminal evidence')
        row['termination']=stop
    best=training['best']
    trained_path=directory/'checkpoint.pt'
    trained_sha=sha(trained_path)
    require(trained_sha==best['scoring_checkpoint_sha256'],'Scoring checkpoint SHA differs (archive SHA is a distinct field)')
    archive=child(directory,best['checkpoint_file'])
    require(sha(archive)==best['checkpoint_sha256'],'Immutable best archive SHA differs')
    trained=checkpoint(trained_path)
    archived=checkpoint(archive)
    same_tensors(trained['model'],archived['model'])
    require(trained['config']==archived['config'],'Canonical best and archived config differ')
    require(trained['implementation']=='push150_expand' and trained['parent_sha256']==init['report']['checkpoint_sha256'] and
            trained['selected_step']==best['step'] and trained['weights']==best['weights'] and
            trained['train_tokens']==best['selected_lineage_targets'] and
            trained['training_recipe']==training['recipe'],'Training checkpoint metadata differs')
    same_tensors(trained['model'],init['checkpoint']['model'],'ngram.')
    if status=='completed':
        state_path=directory/'train_state.pt'
        state=torch.load(allowed(state_path),map_location='cpu',weights_only=True)
        require(state['format_version']==1 and state['identity']==training['identity'] and state['step']==20000 and
                state['log']['completed_steps']==20000 and state['log']['best']==best,
                'Completed training state differs from final metrics')
        row['train_state_sha256']=sha(state_path)
        del state
    row.update(training_checkpoint_sha256=trained_sha,best_archive_sha256=best['checkpoint_sha256'],
        selected_step=best['step'],selected_weights=best['weights'],selected_lineage_targets=trained['train_tokens'],
        best_training_snapshot_scalars=best.get('scalars',{}),runtime=training['identity']['runtime'])
    if not (caldir/'metrics.json').exists():
        if (caldir/'trials.json').exists():
            row['calibration_choices']=len(read(caldir/'trials.json'))
            row['calibration_partial_evidence_sha256']=sha(caldir/'trials.json')
        require(status!='completed' and not caldir.exists(),'Completed run lacks calibration confirmation, or partial calibration needs manual accounting')
        row['status']='failed_or_stopped_without_calibration'
        return
    calibration=read(caldir/'metrics.json')
    row.update(calibration_choices=len(calibration.get('trials',[])),
               calibration_seconds=finite(calibration['seconds_total'],'calibration seconds'))
    full=verify_calibration(calibration,trained_sha)
    cal_sha=sha(caldir/'checkpoint.pt')
    require(cal_sha==calibration['checkpoint_sha256'],'Calibrated checkpoint SHA differs')
    cal=checkpoint(caldir/'checkpoint.pt')
    require(cal['calibration_parent_sha256']==trained_sha and cal['train_tokens']==trained['train_tokens'] and
            cal['implementation']==trained['implementation'],'Calibrated checkpoint lineage differs')
    same_tensors(cal['model'],trained['model'])
    expected_config=copy.deepcopy(trained['config']); expected_config.update(
        temperature=calibration['best']['temperature'],mkn_weight=calibration['best']['mkn_weight'])
    require(cal['config']==expected_config and calibration['parent_config']==trained['config'],'Calibration changed undeclared config')
    cplan=read(caldir/'experiment_plan.json')
    require(cplan['parent_sha256']==trained_sha and cplan['trials']==15 and cplan['test_used'] is False and
            {(t,w) for t in cplan['temperatures'] for w in cplan['mkn_weights']}==GRID,'Calibration plan differs')
    row.update(status='calibrated',selection_eligible=True,selected_run=caldir.name,checkpoint_sha256=cal_sha,
        validation_bpb_gpu_fp32=full,temperature=cal['config']['temperature'],mkn_weight=cal['config']['mkn_weight'],
        implementation=cal['implementation'],discount_scale=cal['config']['discount_scale'],
        calibration_metrics_sha256=sha(caldir/'metrics.json'))


def collect_initializers(root,plan,parent):
    initializers={}
    for layout in ('control','expanded','dense'):
        directory=root/'runs'/('push150-init-'+layout)
        report=read(directory/'initialization.json')
        require(report['layout']==layout and report['seed']==173 and report['new_training_targets']==0 and
                report['parent_sha256']==PARENT_SHA and report['inherited_processed_view_targets']==PARENT_TARGETS and
                report['shared_inherited_targets_counted_once']==PARENT_TARGETS,'Initializer lineage/recipe differs')
        require(report['implementation']=='push150_expand' and report['initializer_sha256']==INIT_SHA and
                report['implementation_sha256']==TRAIN_SOURCES['push150_expand.py'],'Initializer source differs')
        require(report['test_used'] is False and report['data_text_loaded'] is False and report['source_splits']==[],
                'Initializer used undeclared data')
        path=directory/'checkpoint.pt'
        require(sha(path)==report['checkpoint_sha256'] and path.stat().st_size==report['checkpoint_bytes'],'Initializer checkpoint changed')
        cp=checkpoint(path)
        require(cp['train_tokens']==PARENT_TARGETS and cp['parent_sha256']==PARENT_SHA and
                cp['implementation']=='push150_expand' and cp['initialization_layout']==layout,'Initializer checkpoint metadata differs')
        expected=copy.deepcopy(parent['config']); expected['push150_ranks']=plan['initialization']['layouts'][layout]
        require(cp['config']==expected and report['push150_ranks']==expected['push150_ranks'],'Initializer config differs')
        same_tensors(cp['model'],parent['model'],'ngram.')
        initializers[layout]=dict(report=report,checkpoint=cp,report_sha256=sha(directory/'initialization.json'))
    return initializers


def verify_feasibility(root,record,initializers):
    require(record['status']=='completed' and record['phase_plan_sha256']==PLAN_SHA and record['test_used'] is False,
            'CPU feasibility lacks completed bound evidence')
    require((record['device'],record['precision'],record['threads'],record['affinity_mask'],record['split'])==
            ('cpu','fp32',4,15,'validation'),'CPU feasibility protocol differs')
    measurements=record['measurements']
    require([m['name'] for m in measurements]==['baseline-before','control','expanded','dense','baseline-after'],
            'CPU feasibility passes missing, duplicated or reordered')
    baseline_sha=sha(root/'runs/baseline-s17/checkpoint.pt')
    require(baseline_sha==BASELINE_SHA and sha(root/'model.py')==BASELINE_IMPL_SHA,'Official baseline changed')
    for row in measurements:
        require(len(row['repeats'])==1 and row['split']=='validation' and row['threads']==4 and
                row['cpu_affinity_mask']==15 and row['evaluator_sha256']==TRAIN_SOURCES['evaluate.py'],'CPU pass protocol differs')
        verify_measurement(row['repeats'][0])
        require(row['median_scoring_seconds']==row['repeats'][0]['seconds'],'CPU seconds mismatch')
        expected=baseline_sha if row['name'].startswith('baseline') else initializers[row['name']]['report']['checkpoint_sha256']
        require(row['checkpoint_sha256']==expected,'CPU pass checkpoint mismatch')
        expected_impl=BASELINE_IMPL_SHA if row['name'].startswith('baseline') else TRAIN_SOURCES['push150_expand.py']
        require(row['implementation_sha256']==expected_impl,'CPU pass implementation mismatch')
    mean=(measurements[0]['median_scoring_seconds']+measurements[-1]['median_scoring_seconds'])/2
    require(abs(record['baseline_mean_seconds']-mean)<1e-9,'CPU baseline mean mismatch')
    for row in measurements[1:4]:
        ratio=row['median_scoring_seconds']/mean
        saved=record['candidates'][row['name']]
        difference=abs(row['repeats'][0]['bpb']-1.4472849220037163)
        require(abs(saved['parent_bpb_absolute_difference']-difference)<1e-10,
                'Initializer parent-score difference was recorded incorrectly')
        ram=natural(row['peak_process_working_set_bytes'],'precheck RAM bytes')
        cp_bytes=natural(row['checkpoint_bytes'],'precheck checkpoint bytes')
        require(ram>0 and cp_bytes>0,'Missing precheck RAM/checkpoint size')
        feasible=ratio<=5 and ram<=4*2**30 and cp_bytes<64*2**20 and difference<1e-5
        require(abs(saved['cpu_ratio']-ratio)<1e-9 and saved['preliminary_feasible']==feasible,
                'CPU feasibility result mismatch')
        require(saved['training_authorized_by_phase_a']==(feasible and row['name']!='dense'),'CPU training eligibility mismatch')


def collect_mkn(root,parent,row):
    directory=root/'runs/push150-mkn'
    build=read(directory/'build.json'); metrics=read(directory/'metrics.json')
    plan=read(directory/'experiment_plan.json'); provenance=metrics['provenance']
    row.update(calibration_choices=len(metrics.get('trials',[])),
        table_build_seconds=finite(build['seconds_total'],'MKN build seconds'),
        calibration_seconds=finite(metrics['seconds_total'],'MKN score seconds'),
        count_scan_token_occurrences=natural(build['count_scan_token_occurrences'],'count scan occurrences'),
        new_search_processed_view_targets=0,new_search_sampled_label_occurrences=0,
        metrics_sha256=sha(directory/'metrics.json'),build_sha256=sha(directory/'build.json'))
    require(build['status']=='completed' and metrics['status']=='confirmed','Incomplete MKN experiment')
    require(all(record['parent_sha256']==PARENT_SHA and record['test_used'] is False for record in (build,metrics,plan)),
            'MKN parent/data provenance differs')
    require(build['inherited_train_targets']==PARENT_TARGETS and build['new_neural_training_targets']==0 and
            metrics['new_training_targets']==0 and metrics['neural_weights_unchanged'] is True,'MKN lineage/neural cost differs')
    verify_sources(root,build['source_sha256'],MKN_SOURCES)
    require(plan['source_sha256']==MKN_SOURCES and provenance['source_sha256']==MKN_SOURCES,'MKN source records disagree')
    require(plan['scales']==[.4,.55,.7] and plan['mkn_weights']==[.1,.15,.2] and plan['temperature']==1.25 and
            plan['total_choices']==9 and plan['mkn_order']==5 and plan['selection_split']=='validation' and
            plan['minimum_count_by_order']=={'2':1,'3':1,'4':2,'5':2},'MKN predeclared grid/pruning differs')
    require(provenance['build_sha256']==row['build_sha256'] and
            provenance['experiment_plan_sha256']==sha(directory/'experiment_plan.json'),'MKN bound evidence changed')
    for split,expected_tokens,expected_bytes in (('train',3613343,10951562),('validation',376600,1148007)):
        data=provenance[split]
        require(data['split']==split and data['tokens']==expected_tokens and data['utf8_bytes']==expected_bytes and
                data['text_sha256']==DATA_SHA['data/wikitext_'+split+'.txt'] and
                data['tokenizer_sha256']==DATA_SHA['data/tokenizer.json'] and
                data['manifest_sha256']==DATA_SHA['data/manifest.json'],'MKN corpus metadata differs')
    require(build['train']==provenance['train'] and row['count_scan_token_occurrences']==3613343,'MKN scan cost differs')
    trials=metrics['trials']
    require(len(trials)==9 and {(r['discount_scale'],r['mkn_weight']) for r in trials}==MKN_GRID and
            all(r['temperature']==1.25 and r['mkn_order']==5 for r in trials),'MKN grid differs')
    require(metrics['best']==min(trials,key=lambda r:finite(r['validation_bpb'],'MKN trial BPB')),'MKN best differs')
    require(provenance['choices']==trials and provenance['full_validation']==metrics['full_validation'],'MKN provenance choices differ')
    full=verify_measurement(metrics['full_validation'])
    require(abs(full-metrics['best']['validation_bpb'])<1e-6 and metrics['shortcut_matches_full_predictor'] is True and
            metrics['saved_candidate_exactly_matches_full_scored_model'] is True,'MKN full confirmation absent')
    require(metrics['precision']=='fp32' and metrics['device']=='cuda:0' and
            metrics['implementation']=='push147_signed_cache' and
            metrics['implementation_sha256']==TRAIN_SOURCES['push147_signed_cache.py'],'MKN scored protocol differs')
    require(build['control_scale_parity']==metrics['control_scale_parity'] and
            build['control_scale_parity']['shapes_dtypes_values_bitwise_equal'] is True,'MKN .70 parity absent')
    require(build['variants']==provenance['all_table_variants'] and
            [r['scale'] for r in build['variants']]==[.4,.55,.7],'MKN variants differ')
    for variant in build['variants']:
        path=child(directory,variant['path'])
        require(sha(path)==variant['sha256'] and path.stat().st_size==variant['bytes'],'MKN table asset changed')
    cp_path=directory/'checkpoint.pt'; cp_sha=sha(cp_path); cp=checkpoint(cp_path)
    require(cp_sha==metrics['checkpoint_sha256'] and cp_path.stat().st_size==metrics['checkpoint_bytes'],'MKN selected checkpoint changed')
    require(cp['train_tokens']==PARENT_TARGETS and cp['implementation']=='push147_signed_cache','MKN selected lineage differs')
    chosen_table=next(v for v in build['variants'] if v['scale']==metrics['best']['discount_scale'])
    require(cp['mkn_sweep_parent_sha256']==PARENT_SHA and cp['mkn_train_tables_sha256']==chosen_table['sha256'] and
            cp['mkn_sweep_provenance']==provenance,'MKN checkpoint provenance differs from scored record')
    neural={k:v for k,v in parent['model'].items() if not k.startswith('ngram.')}
    same_tensors(neural,{k:v for k,v in cp['model'].items() if not k.startswith('ngram.')})
    require(cp['config']['discount_scale']==metrics['best']['discount_scale'] and
            cp['config']['mkn_weight']==metrics['best']['mkn_weight'] and cp['config']['temperature']==1.25,'MKN chosen config differs')
    expected_config=copy.deepcopy(parent['config'])
    expected_config.update(discount_scale=metrics['best']['discount_scale'],mkn_weight=metrics['best']['mkn_weight'],temperature=1.25)
    require(cp['config']==expected_config,'MKN changed an undeclared non-calibration setting')
    row.update(status='confirmed',selection_eligible=True,selected_run=directory.name,checkpoint_sha256=cp_sha,
        validation_bpb_gpu_fp32=full,selected_lineage_targets=PARENT_TARGETS,
        temperature=1.25,mkn_weight=cp['config']['mkn_weight'],discount_scale=cp['config']['discount_scale'],
        implementation=cp['implementation'],selected_step=None,selected_weights=None)


def collect_report(root,unit_test_executions):
    root=Path(root); natural(unit_test_executions,'unit test executions')
    require(unit_test_executions>0,'Supply measured nonzero unit test executions')
    result=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),phase='A',selection_split='validation',
        test_used=False,no_raw_text_loaded=True,prior_test_exposure_disclosed=True,
        status='refused',winner=None,candidates=[],initializers={},evidence_errors=[],
        minimum_validation_improvement=.0005,passes_predeclared_improvement_gate=False,
        preliminary_phase_choice_only=True,final_cpu_audit_completed=False,test_authorized_by_this_report=False,
        complete_submission=False,unit_test_executions=unit_test_executions,
        report_script_sha256=sha(__file__),required_training_source_sha256=TRAIN_SOURCES,required_training_runtime=RUNTIME)
    # Keep recorded work even if an unrelated global hash check fails. These
    # counts are explicitly unverified until the full evidence chain passes.
    for name in RUNS:
        row=dict(name=name,kind='training',selection_eligible=False,evidence_errors=[],
                 calibration_choices=0,calibration_seconds=None,accounting_verified=False)
        result['candidates'].append(row)
        metrics_path=root/'runs'/('push150-'+name)/'metrics.json'
        if metrics_path.exists():
            try:
                row.update(training_accounting(read(metrics_path)),training_metrics_sha256=sha(metrics_path))
            except (OSError,ValueError,KeyError,TypeError) as error:
                row['preliminary_accounting_error']=str(error)
    try:
        plan_path=root/'runs/push150_phase_a_plan.json'; plan=read(plan_path)
        result['plan_sha256']=sha(plan_path)
        require(result['plan_sha256']==PLAN_SHA,'Phase A plan changed since declaration')
        require(tuple(r['name'] for r in plan['runs'])==RUNS and plan['parent_sha256']==PARENT_SHA and
                plan['parent_train_targets']==PARENT_TARGETS,'Plan parent/runs differ')
        verify_sources(root,TRAIN_SOURCES,TRAIN_SOURCES)
        require(sha(root/'push150_init.py')==INIT_SHA and sha(root/'push146_calibrate.py')==CALIBRATE_SHA,'Init/calibration source differs')
        for name in ('data/tokenizer.json','data/manifest.json'):
            require(sha(root/name)==DATA_SHA[name],'Fixed metadata changed: '+name)
        parent_path=root/'runs/push149-muon-low-calibrated/checkpoint.pt'
        require(sha(parent_path)==PARENT_SHA,'Original parent checkpoint changed')
        parent=checkpoint(parent_path)
        require(parent['train_tokens']==PARENT_TARGETS,'Original parent target count differs')
        pmetrics=read(parent_path.parent/'metrics.json')
        require(pmetrics['checkpoint_sha256']==PARENT_SHA and pmetrics['test_used'] is False,'Original parent validation record differs')
        parent_bpb=verify_measurement(pmetrics['full_validation'])
        require(parent_bpb==plan['parent_validation_bpb_gpu_fp32'],'Parent GPU validation score differs')
        result.update(parent_checkpoint_sha256=PARENT_SHA,parent_validation_bpb_gpu_fp32=parent_bpb,
                      parent_lineage_targets=PARENT_TARGETS,parent_metrics_sha256=sha(parent_path.parent/'metrics.json'))
        initializers=collect_initializers(root,plan,parent)
        result['initializers']={name:dict(report=item['report'],report_sha256=item['report_sha256']) for name,item in initializers.items()}
        feasibility=read(root/'runs/push150-feasibility/summary.json')
        verify_feasibility(root,feasibility,initializers)
        result.update(initial_cpu_feasibility=feasibility,initial_cpu_feasibility_sha256=sha(root/'runs/push150-feasibility/summary.json'))
        for declared in plan['runs']:
            row=next(item for item in result['candidates'] if item['name']==declared['name'])
            try:
                collect_training(root,declared,initializers[declared['initialization']],feasibility,row)
                row['accounting_verified']=True
            except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
                row['evidence_errors'].append(str(error))
                result['evidence_errors'].append(dict(name=row['name'],error=str(error)))
        row=dict(name='mkn-discount',kind='statistics',selection_eligible=False,evidence_errors=[],calibration_choices=0,calibration_seconds=None)
        result['candidates'].append(row)
        try:
            collect_mkn(root,parent,row)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
            row['evidence_errors'].append(str(error)); result['evidence_errors'].append(dict(name=row['name'],error=str(error)))
        result['candidates'].append(dict(name='unchanged-parent',kind='parent',selection_eligible=True,
            selected_run=parent_path.parent.name,checkpoint_sha256=PARENT_SHA,validation_bpb_gpu_fp32=parent_bpb,
            selected_lineage_targets=PARENT_TARGETS,temperature=parent['config']['temperature'],
            mkn_weight=parent['config']['mkn_weight'],discount_scale=parent['config']['discount_scale'],
            implementation=parent['implementation'],new_search_processed_view_targets=0,new_search_sampled_label_occurrences=0,
            calibration_choices=0,calibration_seconds=0.))
        if not result['evidence_errors']:
            winner=min((row for row in result['candidates'] if row['selection_eligible']),key=lambda row:row['validation_bpb_gpu_fp32'])
            improvement=parent_bpb-winner['validation_bpb_gpu_fp32']
            result.update(status='phase_a_choice_complete',winner=copy.deepcopy(winner),improvement_vs_parent=improvement,
                          passes_predeclared_improvement_gate=improvement>=.0005)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        result['evidence_errors'].append(dict(name='global_evidence',error=str(error)))
    training=[r for r in result['candidates'] if r.get('kind')=='training']
    result.update(all_training_accounting_present=len(training)==3 and all('new_search_processed_view_targets' in r for r in training),
        all_training_accounting_verified=len(training)==3 and all(r.get('accounting_verified') for r in training),
        all_new_search_processed_view_targets=sum(r.get('new_search_processed_view_targets',0) for r in training),
        all_new_search_sampled_label_occurrences=sum(r.get('new_search_sampled_label_occurrences',0) for r in training),
        total_recorded_training_loop_seconds=sum(r.get('training_loop_seconds',0) for r in training),
        total_recorded_raw_ema_validation_choices=sum(r.get('raw_ema_validation_choices',0) for r in training),
        total_recorded_calibration_choices=sum(r.get('calibration_choices',0) for r in result['candidates']),
        total_recorded_calibration_seconds=sum(r.get('calibration_seconds') or 0 for r in result['candidates']),
        accounting_scope='Counts are occurrences, not unique corpus tokens. Unknown partial attempt costs are not invented. Missing accounting makes totals partial. MKN count scans are reported separately from gradient/view targets.',
        interpretation='Same-parent cosine/WSD compares schedules. Expanded changes capacity and projection factorization and is not a schedule-only comparison. All full-validation searches count; no test-based selection.',
        next_action='A later declared phase may follow. No inference timing/resource guarantee or final submission approval: audit final actual weights before any freeze/test.')
    return result


def write_report(path,result):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-test-executions',required=True,type=int)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/push150_phase_a_selection.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing phase A evidence; choose a new output path')
    result=collect_report(ROOT,args.unit_test_executions)
    write_report(args.output,result)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    if result['status']=='refused':
        raise SystemExit(2)


if __name__=='__main__':
    main()
