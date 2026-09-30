"""Read-only post-test evidence report; never loads a model or corpus.

Only JSON is decoded. Source/checkpoint/evidence/archive bytes may be hashed;
training probability arrays are never decoded or deployed. Failed, incomplete,
changed or resource-ineligible results do not produce a successful RESULTS.
"""
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
import re
import statistics

from push150_final_test import (ROOT, BASELINE_SHA, BASELINE_MODEL_SHA, SCORER_SHA,
    TOKENIZER_SHA, require, read, sha, child, write_new, check_measurement, verify_freeze)
from push150_freeze import check_timing


SELECTOR_SHA = 'a6e44b8b7e46506193ea51d265e9b6222735c4347134f29081030d36c004e457'
PRESERVATION_SHA = 'd7c78ffe615ca37e0d6f71538105f6a09f3f7827488144c3d2ef1ae4b2ccb7c9'
ORIGINAL_PARENT_TARGETS = 376832000
PREVIOUS_TEST_BPB = 1.4635140025181628
PASS_NAMES = ('baseline-before', 'candidate', 'baseline-after')
TRAIN_RUNS = ('control-cosine','control-wsd','expanded-cosine',
              'suffix-control','suffix-joint','oof-control','oof-joint')
ZIP_NAMES = {'MP1_FINAL_1.55249.zip','MP1_leaderboard_candidate_validation.zip',
    'MP1_push145_tested_candidate.zip','MP1_push146_tested_candidate.zip',
    'MP1_push147_tested_candidate.zip','MP1_push148_tested_candidate.zip',
    'MP1_push149_tested_candidate.zip','MP1_submission.zip'}


def natural(value, label):
    require(isinstance(value,int) and not isinstance(value,bool) and value >= 0, 'Invalid '+label)
    return value


def finite(value, label, positive=False):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid '+label)
    return float(value)


def near(left, right, label, tolerance=1e-10):
    require(math.isfinite(left) and math.isfinite(right) and abs(left-right) <= tolerance, 'Mismatch: '+label)


def timestamp(value):
    parsed=datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'Evidence timestamp needs an explicit time zone')
    return parsed.astimezone(timezone.utc)


def verified(path, expected):
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected), 'Invalid recorded SHA')
    require(sha(path)==expected, 'Changed artifact: '+str(path))
    return expected


def verify_ledger_records(launch, records, summary, freeze, winner):
    """Pure metadata cross-check, used by the file reader and synthetic tests."""
    require(summary.get('status')=='completed' and summary.get('limits_pass') is True,
            'Formal test failed, is incomplete, or did not satisfy limits')
    require(summary.get('candidate_scoring_process_launches')==1 and
            summary.get('baseline_scoring_process_launches')==2 and
            summary.get('no_post_test_selection') is True, 'Unexpected formal launch count/policy')
    require(launch.get('candidate_pass_cap')==1 and launch.get('device')=='cpu' and
            launch.get('precision')=='fp32' and launch.get('threads')==4 and
            launch.get('affinity_mask')==15, 'Changed official CPU protocol')
    require(launch.get('candidate_sha256')==winner['checkpoint_sha256'] and
            launch.get('baseline_sha256')==BASELINE_SHA and launch.get('evaluator_sha256')==SCORER_SHA and
            launch.get('tokenizer_sha256')==TOKENIZER_SHA, 'Formal launch identity differs')
    require([row['name'] for row in launch['planned_passes']]==list(PASS_NAMES) and
            [row['name'] for row in summary['measurements']]==list(PASS_NAMES) and
            list(records)==list(PASS_NAMES), 'Need exactly the ordered baseline/candidate/baseline passes')
    previous=timestamp(freeze['frozen_at_utc'])
    require(previous <= timestamp(launch['created_at_utc']), 'Test launch predates freeze')
    previous=timestamp(launch['created_at_utc'])
    reconstructed=[]
    for index,name in enumerate(PASS_NAMES):
        item=records[name]; started,process,measured,resource=(item[key] for key in ('started','process','measured','resource'))
        digest=winner['checkpoint_sha256'] if name=='candidate' else BASELINE_SHA
        implementation=freeze['implementation_sha256'] if name=='candidate' else BASELINE_MODEL_SHA
        require(started['index']==index and started['name']==name and started['checkpoint_sha256']==digest
                and launch['planned_passes'][index]['sha256']==digest, 'Started-pass identity differs')
        begin,end=timestamp(started['started_at_utc']),timestamp(process['completed_at_utc'])
        require(previous <= begin <= end, 'Formal passes overlapped or chronological evidence differs')
        previous=end
        require(process.get('returncode')==0, 'A formal scoring process failed')
        check_measurement(measured,resource,digest,implementation)
        ram=natural(resource['peak_process_working_set_bytes'],'peak working-set bytes')/2**30
        row=dict(name=name,output=name+'.json',output_sha256=item['output_sha256'],
                 resources_sha256=item['resources_sha256'],**measured,peak_ram_gib=ram)
        require(summary['measurements'][index]==row, 'Official summary differs from independent result/resources')
        reconstructed.append(row)
    require(previous <= timestamp(summary['completed_at_utc']), 'Summary predates a scoring exit')
    candidate=reconstructed[1]; baseline=[reconstructed[0]['seconds'],reconstructed[2]['seconds']]
    mean_ratio=candidate['seconds']/statistics.mean(baseline)
    faster_ratio=candidate['seconds']/min(baseline)
    near(summary['test_bpb'],candidate['bpb'],'test BPB')
    near(summary['cpu_ratio_to_baseline_mean'],mean_ratio,'mean CPU ratio')
    near(summary['cpu_ratio_to_faster_baseline'],faster_ratio,'faster-baseline CPU ratio')
    near(summary['peak_ram_gib'],candidate['peak_ram_gib'],'official RAM')
    near(summary['inference_assets_mib'],freeze['inference_assets_mib'],'official inference assets')
    require(mean_ratio <= 5 and 0 < candidate['peak_ram_gib'] <= 4 and
            0 < freeze['inference_assets_mib'] <= 64, 'Recomputed official resource limits failed')
    return dict(measurements=reconstructed,baseline_seconds=baseline,
        candidate_seconds=candidate['seconds'],ratio_to_baseline_mean=mean_ratio,
        ratio_to_faster_baseline=faster_ratio,individual_baseline_ratios=[candidate['seconds']/s for s in baseline],
        test_bpb=candidate['bpb'],peak_ram_gib=candidate['peak_ram_gib'],limits_pass=True,
        candidate_scoring_passes=1,baseline_scoring_passes=2)


def summarize_costs(selection, training, calibration, gate_phases, mkn_build, mkn_score, preparation):
    """Each child counted once; shared ancestry/launch wall times stay separate."""
    require(set(training)==set(calibration)==set(TRAIN_RUNS), 'Incomplete A/C/D cost ledger')
    rows=[]
    for name in TRAIN_RUNS:
        record,cal=training[name],calibration[name]
        require(record['status']=='completed' and record['test_used'] is False and cal['test_used'] is False,
                'Unfinished/failed training or calibration cannot be presented as complete')
        completed=natural(record['completed_steps'],'completed steps')
        discarded=natural(record.get('discarded_completed_steps',0),'discarded steps')
        require(completed==record['steps'], 'Completed training has a partial step count')
        batch=natural(record['batch_size'],'batch size'); views=natural(record['dropout_views_per_sample'],'views')
        targets=(completed+discarded)*batch*256
        require(record['incremental_search_targets']==targets*views and
                record['incremental_sampled_label_targets']==targets, 'Training occurrence accounting differs')
        best=record['best']; chosen=natural(best['step'],'selected step')
        require(chosen <= completed and best['selected_lineage_targets']==record['parent_train_targets']+chosen*batch*views*256,
                'Selected lineage differs from actual best step')
        trials=cal['trials']; history=record['validation_history']
        require(len(trials)==15 and cal['new_training_targets']==0, 'Expected fixed15 zero-training calibration')
        require(abs(cal['full_validation']['bpb']-cal['best']['validation_bpb']) < 1e-6,
                'Calibration shortcut not confirmed by full prediction')
        rows.append(dict(name=name,completed_steps=completed,discarded_completed_steps=discarded,
            processed_view_targets=targets*views,sampled_label_occurrences=targets,
            selected_step=chosen,selected_weights=best['weights'],selected_lineage_targets=best['selected_lineage_targets'],
            validation_bpb=cal['full_validation']['bpb'],raw_ema_choices=len(history),calibration_choices=len(trials),
            training_seconds=finite(record['seconds'],'training seconds'),
            calibration_seconds=finite(cal['seconds_total'],'calibration seconds')))
    gate_train=gate_phases['train']; gate_labels=natural(gate_train['gate_target_occurrences'],'gate targets')
    require(all(item['status']=='completed' and item['test_used'] is False for item in gate_phases.values()),
            'Incomplete gate phase')
    require(gate_train['new_neural_training_targets']==0 and preparation['costs']['new_count_scan_token_occurrences']==0,
            'Gate invented neural work or D duplicated B scans')
    phase_a=sum(row['sampled_label_occurrences'] for row in rows[:3])
    phase_c=sum(row['sampled_label_occurrences'] for row in rows[3:5])
    phase_d=sum(row['sampled_label_occurrences'] for row in rows[5:])
    bc=selection['phase_bc']['record']['cost_summary']; overall=selection['cost_summary']
    require(bc['accounting_verified'] is True and overall['accounting_verified'] is True, 'Unverified overall costs')
    require(bc['phase_a_new_neural_view_targets']==phase_a and bc['phase_c_new_label_occurrences']==phase_c and
            bc['gate_label_occurrences']==gate_labels and
            bc['all_recorded_new_gradient_label_occurrences']==phase_a+phase_c+gate_labels, 'ABC costs double counted or differ')
    require(overall['phase_abc_recorded_new_gradient_label_occurrences']==phase_a+phase_c+gate_labels and
            overall['phase_d_new_label_occurrences']==phase_d and
            overall['all_recorded_new_gradient_label_occurrences']==phase_a+phase_c+phase_d+gate_labels and
            overall['shared_B_count_scan_token_occurrences_added_again_in_D']==0 and
            overall['shared_original149_ancestry_targets']==ORIGINAL_PARENT_TARGETS, 'D/shared ancestry costs differ')
    require(overall['missing_or_partial_work_remains_unquantified'] is False, 'Partial work cannot be labelled complete')
    gate_build=gate_phases['build']
    return dict(training_runs=rows,phase_a_neural_label_occurrences=phase_a,phase_c_neural_label_occurrences=phase_c,
        phase_d_neural_label_occurrences=phase_d,new_neural_processed_view_targets=sum(row['processed_view_targets'] for row in rows),
        new_neural_sampled_label_occurrences=phase_a+phase_c+phase_d,gate_label_occurrences=gate_labels,
        all_new_gradient_label_occurrences=phase_a+phase_c+phase_d+gate_labels,
        shared_original149_ancestry_targets_counted_once=ORIGINAL_PARENT_TARGETS,
        selected_lineage_targets=selection['winner']['selected_lineage_targets'],
        selected_lineage_is_not_total_search_work=True,occurrences_are_not_unique_tokens=True,
        raw_ema_validation_choices=sum(row['raw_ema_choices'] for row in rows),
        neural_calibration_choices=sum(row['calibration_choices'] for row in rows),
        mkn_choices=len(mkn_score['trials']),gate_choices=len(gate_phases['score']['choices']),
        training_loop_seconds=sum(row['training_seconds'] for row in rows),
        calibration_seconds=sum(row['calibration_seconds'] for row in rows),
        mkn_build_seconds=mkn_build['seconds_total'],mkn_score_seconds=mkn_score['seconds_total'],
        gate_phase_seconds={phase:item['seconds_total'] for phase,item in gate_phases.items()},
        oof_preparation_costs=copy.deepcopy(preparation['costs']),
        count_scan_token_occurrences=dict(mkn=mkn_build['count_scan_token_occurrences'],
            gate_full_parity=gate_build['parity_count_scan_token_occurrences'],
            gate_tenfold=gate_build['ten_crossfit_count_scan_token_occurrences'],phase_d_additional=0),
        extraction_target_occurrences=gate_phases['extract']['neural_extraction_target_occurrences'],
        initialization_parity_target_occurrences=gate_phases['extract']['initialization_parity_neural_forward_target_occurrences'],
        followon_launch_seconds_inclusive_not_added_again=selection['followon_launches']['seconds_total_inclusive_not_added_to_child_costs'],
        source_cost_summaries=dict(ABC=copy.deepcopy(bc),ABCD=copy.deepcopy(overall)),
        all_cpu_screens_and_timing_drift_retained=True,
        accounting_note='Optimizer loops include their validation work; do not add their inclusive child-launch durations again. Reused B statistics scans and original149 ancestry are counted once. Earlier-round full search is not reconstructed.')


def comparison(rows, method, control, caveat):
    index={row['name']:row for row in rows}; left,right=index[method],index[control]
    return dict(method=method,control=control,validation_improvement=right['validation_bpb']-left['validation_bpb'],
        processed_targets_match=left['processed_view_targets']==right['processed_view_targets'],
        selected_step_method=left['selected_step'],selected_step_control=right['selected_step'],
        selected_weights_method=left['selected_weights'],selected_weights_control=right['selected_weights'],
        single_seed=True,independent_snapshot_and_calibration_selection=True,caveat=caveat)


def collect_results(root=ROOT, project=None):
    root=Path(root).resolve(); project=Path(project).resolve() if project else root.parents[2]
    selection_path=root/'runs/push150_selection.json'
    evidence=verify_freeze(root,selection_path)
    selection,freeze,audit,winner=(evidence['selection'],evidence['freeze'],evidence['audit'],evidence['selection']['winner'])
    require(selection['status']=='validation_choice_complete' and selection['selection_integrity_errors']==[], 'Selection not complete')
    verified(root/'push150_select.py',SELECTOR_SHA)
    require(selection['report_script_sha256']==SELECTOR_SHA and selection['evidence_sha256'], 'Unpinned final selection')
    for name,digest in selection['evidence_sha256'].items():
        verified(child(root,name),digest)
    verified(root/'push150_freeze.py',freeze['freeze_script_sha256'])
    require(freeze['implementation']==winner['implementation'] and freeze['selection_split']=='validation', 'Dynamic inference identity differs')
    eligible=[row for row in selection['candidates'] if row.get('selection_eligible')]
    require(winner==min(eligible,key=lambda row:row['validation_bpb_gpu_fp32']), 'Winner is not the validation minimum')
    near(selection['improvement_vs_original149'],selection['original149_validation_bpb_gpu_fp32']-winner['validation_bpb_gpu_fp32'],'improvement')
    require(selection['improvement_vs_original149'] >= selection['minimum_validation_improvement']==.0005,
            'Predeclared improvement gate failed')
    require(timestamp(selection['recorded_at_utc']) <= timestamp(freeze['frozen_at_utc']), 'Selection postdates freeze')
    checkpoint=evidence['run_dir']/'checkpoint.pt'
    require(checkpoint.stat().st_size==freeze['checkpoint_bytes'], 'Frozen checkpoint byte size differs')
    require(audit['parameters']==freeze['parameters'] and natural(audit['parameters'],'parameters')>0 and
            audit['train_targets_on_checkpoint']==freeze['train_processed_targets']==winner['selected_lineage_targets'], 'Parameters/lineage differ')
    near(audit['inference_assets_bytes']/2**20,audit['inference_assets_mib'],'asset MiB')
    near(audit['inference_assets_mib'],freeze['inference_assets_mib'],'frozen assets')
    near(audit['peak_process_working_set_bytes']/2**30,audit['peak_ram_gib'],'audit RAM')
    require(0 < audit['peak_ram_gib'] <= 4, 'Audit RAM over limit')
    near(audit['measurement']['bpb'],freeze['validation_bpb_cpu_fp32'],'frozen validation BPB')
    require(abs(audit['measurement']['bpb']-winner['validation_bpb_gpu_fp32']) < 1e-5, 'CPU/GPU validation differs')
    unit=selection['unit_tests']; units=read(child(root,unit['path']))
    verified(child(root,unit['path']),unit['sha256'])
    require(units['successful'] is True and not units['errors'] and not units['failures'] and
            unit['executions']==units['tests_run']==freeze['unit_test_executions_passed']>0, 'Final unit evidence failed')
    for group,base in (('source_sha256',root),('unit_source_sha256',root/'tests')):
        for name,digest in units[group].items(): verified(child(base,name),digest)
    require('push150_results.py' in units['source_sha256'] and 'test_push150_results.py' in units['unit_source_sha256'],
            'Final complete synthetic suite must include this result verifier')
    timing=read(child(root,freeze['timing_record'])); verified(child(root,freeze['timing_record']),freeze['timing_record_sha256'])
    require(check_timing(timing,winner)==freeze['validation_timing'], 'Frozen timing summary differs')
    verified(root/'push145_fresh_benchmark.py',timing['script_sha256'])
    require(all(abs(row['bpb']-audit['measurement']['bpb']) < 1e-10 for row in timing['measurements'] if row['name']=='candidate'),
            'Actual timed predictor differs from audit')
    directory=root/'runs/push150-final-test'; launch_path=directory/'launch_plan.json'
    launch,summary=read(launch_path),read(directory/'summary.json')
    allowed={'launch_plan.json','summary.json'}|{name+suffix for name in PASS_NAMES for suffix in
                                              ('.started.json','.process.json','.json','.resources.json')}
    require({path.name for path in directory.glob('*.json')}==allowed, 'Extra/missing official launch, failure or output record')
    for key,path in (('selection_sha256',selection_path),('freeze_sha256',evidence['freeze_path']),
                     ('wrapper_sha256',root/'push150_final_test.py'),('official_wrapper_sha256',root/'official_resource_run.py')):
        verified(path,launch[key])
    require(summary['launch_plan_sha256']==sha(launch_path), 'Official summary launch binding differs')
    records={}; test_evidence={}
    for index,name in enumerate(PASS_NAMES):
        paths={key:directory/(name+suffix) for key,suffix in
               (('started','.started.json'),('process','.process.json'),('measured','.json'),('resource','.resources.json'))}
        item={key:read(path) for key,path in paths.items()}
        require(item['started']['launch_plan_sha256']==sha(launch_path), 'Started launch plan binding differs')
        cp=checkpoint if name=='candidate' else evidence['baseline']
        command=item['started']['command']
        require(len(command)==9 and command[1:]==['-B',str(root/'official_resource_run.py'),'--checkpoint',str(cp),
                '--split','test','--output',str(paths['measured'])], 'Official command differs from fixed scorer wrapper')
        require(Path(launch['planned_passes'][index]['checkpoint']).resolve()==cp.resolve(), 'Planned checkpoint path differs')
        require(Path(item['resource']['evaluator_output']).resolve()==paths['measured'].resolve() and
                item['resource']['includes']=='entire evaluator process including all fixed dataset loading', 'Resource scope differs')
        item.update(output_sha256=sha(paths['measured']),resources_sha256=sha(paths['resource']))
        records[name]=item
        for path in paths.values(): test_evidence[path.name]=sha(path)
    actual=verify_ledger_records(launch,records,summary,freeze,winner)
    preserve_path=root/'runs/push150_preserved_packages.json'; verified(preserve_path,PRESERVATION_SHA)
    preserved=read(preserve_path)['archives']
    require(len(preserved)==8 and {row['name'] for row in preserved}==ZIP_NAMES, 'Incomplete eight-archive preservation baseline')
    for row in preserved:
        path=child(project/'outputs',row['name']); verified(path,row['sha256'])
        require(path.stat().st_size==row['bytes'], 'Previous archive byte size differs')
    training={name:read(root/'runs'/('push150-'+name)/'metrics.json') for name in TRAIN_RUNS}
    calibration={name:read(root/'runs'/('push150-'+name+'-calibrated')/'metrics.json') for name in TRAIN_RUNS}
    gate={phase:read(root/'runs/push150-gate'/(phase+'.json')) for phase in ('build','extract','train','score')}
    costs=summarize_costs(selection,training,calibration,gate,read(root/'runs/push150-mkn/build.json'),
        read(root/'runs/push150-mkn/metrics.json'),read(root/'runs/push150-oof-data/metadata.json'))
    rows=costs['training_runs']
    comparisons=[comparison(rows,'control-wsd','control-cosine','Schedule comparison, seed173; not multiple-seed evidence.'),
        comparison(rows,'expanded-cosine','control-cosine','Capacity and projection parameterization both change; not a pure rank or optimizer theorem.'),
        comparison(rows,'suffix-joint','suffix-control','Same Phase-A parent, seed191 and windows; compare against continuation control, not only parent.'),
        comparison(rows,'oof-joint','oof-control','Same Phase-A parent, seed211, legal starts and count coins; only count expert is out of fold.')]
    return dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),status='verified_post_freeze_test_result',
        selected_run=winner['selected_run'],checkpoint_sha256=winner['checkpoint_sha256'],implementation=winner['implementation'],
        implementation_sha256=freeze['implementation_sha256'],test_bpb=actual['test_bpb'],official_test_bpb_cpu_fp32=actual['test_bpb'],
        validation_bpb_cpu_fp32=audit['measurement']['bpb'],validation_bpb_gpu_fp32=winner['validation_bpb_gpu_fp32'],
        validation_timing=freeze['validation_timing'],official_test=actual,baseline_cpu_seconds=actual['baseline_seconds'],
        official_cpu_ratio_to_baseline_mean=actual['ratio_to_baseline_mean'],official_cpu_ratio_to_faster_baseline=actual['ratio_to_faster_baseline'],
        peak_ram_gib=actual['peak_ram_gib'],inference_assets_mib=audit['inference_assets_mib'],
        inference_assets_bytes=audit['inference_assets_bytes'],parameters=audit['parameters'],
        selected_lineage_targets=winner['selected_lineage_targets'],selected_validation_record=copy.deepcopy(winner),
        all_search_costs=costs,method_comparisons=comparisons,all_validation_candidates=copy.deepcopy(selection['candidates']),
        phase_d_matched_evidence=selection['phase_d_matched_comparison'],
        phase_bc_report_reference=dict(path=selection['phase_bc']['path'],sha256=selection['phase_bc']['sha256']),
        all_candidate_failure_and_timing_drift_evidence_retained=True,limits_pass=True,unit_tests=unit,
        frozen_at_utc=freeze['frozen_at_utc'],test_used_for_tuning=False,no_post_test_selection=True,
        test_used_for_final_scoring_after_freeze=True,test_candidate_scoring_passes=1,
        prior_test_exposure_disclosed=True,prior_test_history_is_exhaustive=False,
        previous_push149_test_bpb_disclosure_only=PREVIOUS_TEST_BPB,
        observed_test_improvement_vs_previous_push149=PREVIOUS_TEST_BPB-actual['test_bpb'],
        test_comparison_note='Descriptive post-freeze result only. It was not used to select, tune or retry the new predictor.',
        limitations=['One seed per declared comparison; no robust universal algorithm claim.',
            'Neural parent saw all train; only statistical expert is cross-fitted.',
            'All CPU records retained, including initial timing drift and rejected fast-dense variants.',
            'Resource figures apply to this recorded machine/runtime; no universal speed guarantee.'],
        complete_from_scratch_training_reproduction_bundle=False,
        scoring_reproduction_with_original_course_data_supported=True,original_course_raw_text_required=True,
        historical_parent_weights_included=False,complete_final_report_and_public_artifact_links_pending=True,
        uploaded_or_submitted_by_this_script=False,training_caches_excluded_from_inference=True,
        preserved_previous_packages=preserved,preservation_record_sha256=PRESERVATION_SHA,
        selection_sha256=sha(selection_path),freeze_sha256=sha(evidence['freeze_path']),audit_sha256=freeze['audit_sha256'],
        timing_sha256=freeze['timing_record_sha256'],official_launch_plan_sha256=sha(launch_path),
        official_summary_sha256=sha(directory/'summary.json'),official_evidence_sha256=test_evidence,
        selection_evidence_sha256=copy.deepcopy(selection['evidence_sha256']),results_script_sha256=sha(__file__),
        ai_assistance_acknowledged=True,muon_license='THIRD_PARTY_MUON_LICENSE.txt (MIT)',
        references=dict(mixture='https://aclanthology.org/D16-1124/',rdrop='https://arxiv.org/abs/2106.14448',
                        muon='https://github.com/KellerJordan/Muon'))


def main():
    destination=ROOT.parents[2]/'outputs/MP1_push150_tested_candidate/RESULTS.json'
    require(not destination.exists(), 'Preserve the existing formal RESULTS record')
    result=collect_results()
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_new(destination,result)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
