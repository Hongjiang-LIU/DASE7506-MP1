"""Account for fixed A/B choices and at most one predeclared combination.

Validation-only screening. This is not a test runner or resource-compliance
override. A later frozen audit is required before any official test.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import torch
from common import ROOT, setup, sha
from push150_mkn_sweep import load_one_split
from push151_statistics import (PLAN_SHA, PLAN_PATH, PARENT_SHA, FIXED_CONFIG, read,
    child, require, write_new, save_new, load_plan, load_parent, measure_checkpoint)


COMBO = ROOT/'runs/push151-combination'


def verify_measurement(measured):
    require(measured['targets'] == 376599 and measured['utf8_bytes'] == 1148007 and
            measured['precision'] == 'fp32' and measured['split'] == 'validation' and
            math.isfinite(measured['bpb']) and measured['bpb'] > 0 and measured['test_used'] is False,
            'Incomplete validation measurement')


def phase_records():
    plan = load_plan()
    records = {}
    for name, directory in [('A', ROOT/'runs/push151-average'), ('B', ROOT/'runs/push151-statistics')]:
        build, scored = read(directory/'build.json'), read(directory/'score.json')
        require(build['status'] == scored['status'] == 'completed' and
                build['plan_sha256'] == scored['plan_sha256'] == PLAN_SHA and
                scored['build_sha256'] == sha(directory/'build.json') and
                scored['test_used'] is False, 'Phase incomplete or identity differs: '+name)
        require(build['source_sha256'] == scored['source_sha256'], 'Build/score sources differ')
        for relative, digest in scored['source_sha256'].items():
            require(sha(child(relative)) == digest, 'A scored source changed: '+relative)
        expected_names = (['control','tail-ema-average','last-raw-ema-average'] if name == 'A' else
                          ['control','information-equal','information-plus8mib'])
        require([row['name'] for row in build['candidates']] == expected_names and
                [row['name'] for row in scored['candidates']] == expected_names,
                'Unexpected, duplicated or missing declared candidates')
        by_name = {row['name']:row for row in build['candidates']}
        rows = []
        for measured in scored['candidates']:
            built = by_name[measured['name']]
            path = child(built['checkpoint'])
            require(measured['checkpoint_sha256'] == built['checkpoint_sha256'] == sha(path), 'Candidate changed')
            verify_measurement(measured)
            rows.append(dict(phase=name, name=measured['name'], checkpoint=built['checkpoint'],
                checkpoint_sha256=built['checkpoint_sha256'], validation_bpb_gpu_fp32=measured['bpb'],
                measured=measured))
        require(rows[0]['name'] == 'control' and rows[0]['checkpoint_sha256'] == PARENT_SHA,
                'Original parent control missing')
        require(abs(rows[0]['validation_bpb_gpu_fp32']-plan['parent']['validation_bpb_gpu_fp32']) < 1e-6,
                'Full parent score differs')
        records[name] = dict(rows=rows, build=build, score=scored, directory=directory)
    require(abs(records['A']['rows'][0]['validation_bpb_gpu_fp32']-
                records['B']['rows'][0]['validation_bpb_gpu_fp32']) < 1e-8, 'Phase parent scores disagree')
    return plan, records


def combination_spec(records):
    a = min(records['A']['rows'], key=lambda row:row['validation_bpb_gpu_fp32'])
    b = min(records['B']['rows'], key=lambda row:row['validation_bpb_gpu_fp32'])
    active = all(row['name'] != 'control' and row['validation_bpb_gpu_fp32'] <
                 records[phase]['rows'][0]['validation_bpb_gpu_fp32'] for phase,row in [('A',a),('B',b)])
    return active, a, b


def combine_states(average, statistics):
    require(average['implementation'] == statistics['implementation'] == 'push150_expand' and
            average['train_tokens'] == statistics['train_tokens'] == 638976000,
            'Combination identity/ancestry mismatch')
    require(set(average['model']) == set(statistics['model']), 'Combination state keys differ')
    before = {key:value for key,value in average['config'].items() if key != 'mkn_shapes'}
    after = {key:value for key,value in statistics['config'].items() if key != 'mkn_shapes'}
    require(before == after and all(before.get(key) == value for key,value in FIXED_CONFIG.items()),
            'Combination would change calibration or architecture')
    result = copy.deepcopy(average)
    for name, value in statistics['model'].items():
        if name.startswith('ngram.'):
            result['model'][name] = value.clone()
    result['config']['mkn_shapes'] = copy.deepcopy(statistics['config']['mkn_shapes'])
    for name,value in average['model'].items():
        if not name.startswith('ngram.'):
            require(torch.equal(value,result['model'][name]), 'Combination altered averaged neural weights')
    return result


def run_combination():
    plan, records = phase_records()
    active, a, b = combination_spec(records)
    COMBO.mkdir(exist_ok=False)
    pin = dict(plan_sha256=PLAN_SHA, script_sha256=sha(__file__),
        phase_a_score_sha256=sha(records['A']['directory']/'score.json'),
        phase_b_score_sha256=sha(records['B']['directory']/'score.json'))
    started = time.perf_counter()
    write_new(COMBO/'started.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv], cwd=str(Path.cwd()), test_used=False, **pin))
    if not active:
        result = dict(status='skipped', reason='At least one phase did not beat its unchanged parent',
            best_a=a, best_b=b, test_used=False, new_gradient_targets=0,
            seconds_total=time.perf_counter()-started, **pin)
        write_new(COMBO/'result.json', result)
        return result
    try:
        average = torch.load(child(a['checkpoint']), map_location='cpu', weights_only=True)
        statistics = torch.load(child(b['checkpoint']), map_location='cpu', weights_only=True)
        candidate = combine_states(average,statistics)
        candidate['push151_combination'] = dict(average_sha256=a['checkpoint_sha256'],
            statistics_sha256=b['checkpoint_sha256'],shared_ancestor_targets=638976000,
            new_gradient_targets=0, duplicate_ancestors_not_summed=True,test_used=False,**pin)
        candidate['selection'] = 'Only predeclared A/B validation winner combination; fixed T/w'
        save_new(COMBO/'checkpoint.pt',candidate)
        device,_ = setup('cuda','fp32',2)
        ids,validation = load_one_split('validation')
        measured = measure_checkpoint(COMBO/'checkpoint.pt',sha(COMBO/'checkpoint.pt'),torch.from_numpy(ids),
                                      validation['utf8_bytes'],device,COMBO/'validation.json')
        phase_records()
        require(sha(__file__) == pin['script_sha256'], 'Combination implementation changed')
        result = dict(status='completed',phase='C',name='combined',measurement=measured,
            source_a=a,source_b=b,new_gradient_targets=0,shared_ancestor_targets=638976000,
            test_used=False,seconds_total=time.perf_counter()-started,**pin)
        write_new(COMBO/'result.json',result)
        return result
    except BaseException as error:
        write_new(COMBO/'failure.json',dict(status='failed',failure_type=type(error).__name__,
            failure_message=str(error),test_used=False,no_automatic_retry=True,**pin))
        raise


def report():
    plan,records = phase_records()
    combo = read(COMBO/'result.json')
    active,a,b = combination_spec(records)
    require(combo['plan_sha256'] == PLAN_SHA and combo['test_used'] is False and
            combo['script_sha256'] == sha(__file__) and
            combo['phase_a_score_sha256'] == sha(records['A']['directory']/'score.json') and
            combo['phase_b_score_sha256'] == sha(records['B']['directory']/'score.json'), 'Combination evidence changed')
    require(combo['status'] == ('completed' if active else 'skipped'), 'Combination not accounted for')
    rows = [records['A']['rows'][0],*records['A']['rows'][1:],*records['B']['rows'][1:]]
    if active:
        measured = combo['measurement']
        verify_measurement(measured)
        require(combo['source_a'] == a and combo['source_b'] == b and
                measured == read(COMBO/'validation.json') and
                measured['checkpoint'] == (COMBO/'checkpoint.pt').relative_to(ROOT).as_posix(),
                'Combination score/sources differ from the declared winners')
        require(sha(child(measured['checkpoint'])) == measured['checkpoint_sha256'], 'Combined checkpoint changed')
        rows.append(dict(phase='C',name='combined',checkpoint=measured['checkpoint'],
            checkpoint_sha256=measured['checkpoint_sha256'],validation_bpb_gpu_fp32=measured['bpb'],measured=measured))
    original = rows[0]['validation_bpb_gpu_fp32']
    for row in rows:
        row['improvement_vs_parent'] = original-row['validation_bpb_gpu_fp32']
        row['passes_validation_improvement_gate'] = row['improvement_vs_parent'] >= plan['selection']['minimum_improvement']
        row['cpu_resource_status'] = 'historical_only' if row['name'] == 'control' else 'not_yet_audited'
    ranked = sorted((row for row in rows if row['passes_validation_improvement_gate']),
                    key=lambda row:row['validation_bpb_gpu_fp32'])
    result = dict(status='validation_qualified_candidates_need_resource_audit' if ranked else 'no_new_candidate_meets_validation_gate',
        created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,script_sha256=sha(__file__),
        selection_split='validation',test_used=False,prior_test_exposure_disclosed=True,
        all_declared_A_B_C_accounted=True,phase_c_status=combo['status'],minimum_improvement=.0005,
        candidates=rows,validation_qualified_in_rank_order=ranked,new_neural_gradient_targets=0,
        selected_shared_ancestor_targets=638976000,statistical_train_tokens_scanned=records['B']['build']['count_source_token_occurrences'],
        construction_seconds=dict(A=records['A']['build']['seconds_total'],B=records['B']['build']['seconds_total']),
        development_score_seconds=dict(A=records['A']['score']['seconds_total'],B=records['B']['score']['seconds_total']),
        combination_seconds_including_build_and_score=combo['seconds_total'],
        full_validation_passes=6+int(active),final_cpu_audit_and_freeze_required=bool(ranked),
        evidence_sha256={str(path.relative_to(ROOT).as_posix()):sha(path) for path in
            [PLAN_PATH,records['A']['directory']/'build.json',records['A']['directory']/'score.json',
             records['B']['directory']/'build.json',records['B']['directory']/'score.json',COMBO/'result.json']})
    write_new(ROOT/'runs/push151_validation_selection.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('combine','report'))
    args=parser.parse_args()
    result=run_combination() if args.phase=='combine' else report()
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
