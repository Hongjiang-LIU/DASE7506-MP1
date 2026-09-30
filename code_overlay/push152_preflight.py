"""Fixed FFN expansion preflight; no training or test-corpus access.

audit: synthetic model/FP64 checks and complete inference assets.
authorize: verify six original full-validation timing records; no scoring.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PLAN_SHA = 'd15cd6276ace1894a37db7c005453c61acd2e5acb7e934cb6790f3860aa06b51'
HELPER_SHA = '92f033e1cc7b3532924d2d6bbde6eee1d0a8fa73c1c076f32e340f63201ebe25'
TIMING_SHA = '5e0a5173dd00a1b3519877172d1dcb3d195a2a106a17cd6256d7b55cb78bb99e'
SCORER_SHA = '128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d'
BASELINE_SHA = 'b7fdb67c8ef14b5166c0de1f17cf306f34800b41dfe97da856c9812cdb1f9701'
from push151_launch import sha, read, write_new, preserve


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inputs():
    plan_path = ROOT/'runs/push152_plan.json'
    require(sha(plan_path) == PLAN_SHA, 'Fixed plan changed')
    plan = read(plan_path)
    preserve()
    archive = ROOT.parents[2]/plan['protected_previous_archives']['tenth']
    require(archive.stat().st_size == plan['protected_previous_archives']['tenth_bytes'] and
        sha(archive) == plan['protected_previous_archives']['tenth_sha256'], 'Tenth protected ZIP changed')
    parent_path = ROOT/plan['parent']['path']
    checkpoint = ROOT/plan['expansion']['run_dir']/'checkpoint.pt'
    initialization = read(checkpoint.parent/'initialization.json')
    require(sha(parent_path) == plan['parent']['sha256'] and initialization['checkpoint_sha256'] == sha(checkpoint)
        and initialization['plan_sha256'] == PLAN_SHA, 'Initializer or parent identity differs')
    require(initialization['status'] == 'completed' and initialization['test_used'] is False and
        initialization['data_text_loaded'] is False, 'Initializer was not the completed fixed procedure')
    for name, digest in initialization['source_sha256'].items():
        source = (ROOT/name).resolve()
        require(source.is_relative_to(ROOT) and source.suffix == '.py' and sha(source) == digest, 'Initializer source changed')
    require(sha(ROOT/'push150_expand.py') == plan['parent']['implementation_sha256'], 'Inference source changed')
    return plan, parent_path, checkpoint, initialization


def audit():
    plan, parent_path, checkpoint, initialization = inputs()
    directory = ROOT/plan['expansion']['preflight_dir']
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    script_sha = sha(__file__)
    write_new(directory/'audit.started.json', dict(plan_sha256=PLAN_SHA, script_sha256=script_sha,
        checkpoint_sha256=sha(checkpoint), test_used=False))
    try:
        import torch
        from common import make_model, setup
        from measure_validation import peak_memory
        helper_path = WORK/'push150_numerical_diagnostic.py'
        require(sha(helper_path) == HELPER_SHA, 'Independent diagnostic helper changed')
        spec = importlib.util.spec_from_file_location('_push152_fp64_helper', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.cpu_setup()
        device, precision = setup('cpu', 'fp32', 4)
        before = torch.load(parent_path, map_location='cpu', weights_only=True)
        after = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(after['implementation'] == before['implementation'] == 'push150_expand' and
            after['train_tokens'] == before['train_tokens'] == 638976000, 'Initializer changed lineage/implementation')
        for key, value in before['config'].items():
            if key != 'push150_ranks':
                require(after['config'].get(key) == value, 'Unplanned config change: '+key)
        require(after['config']['push150_ranks'] == plan['expansion']['target_layout'], 'Wrong expanded layout')
        statistics_count = 0
        for key, value in before['model'].items():
            is_expanded = key.startswith('blocks.') and any('.mlp.'+name+'.' in key for name in ('gate','value','out'))
            if not is_expanded:
                require(key in after['model'] and value.dtype == after['model'][key].dtype and
                    torch.equal(value, after['model'][key]), 'A non-FFN tensor changed: '+key)
            if key.startswith('ngram.'):
                statistics_count += 1
        require(statistics_count == 21, 'Expected21 unchanged statistical buffers')
        parent_model, _ = make_model('push150_expand', before['config'], device)
        child_model, _ = make_model('push150_expand', after['config'], device)
        parent_model.load_state_dict(before['model'], strict=True)
        child_model.load_state_dict(after['model'], strict=True)
        parent_model.eval(); child_model.eval()
        require(sum(p.numel() for p in child_model.parameters())-sum(p.numel() for p in parent_model.parameters()) ==
            plan['expansion']['expected_parameter_increase'], 'Parameter increase differs')
        del before, after
        torch.manual_seed(127)
        ids = torch.randint(2048, (2,256))
        with torch.no_grad():
            previous, current = parent_model.predict_log_probs(ids), child_model.predict_log_probs(ids)
            require(torch.isfinite(current).all().item() and current.dtype == torch.float32 and
                current.logsumexp(-1).abs().max().item() < 1e-3, 'Invalid/unnormalized child probabilities')
            require(torch.allclose(previous, current, atol=1e-5, rtol=1e-5), 'Initialization changes prediction beyond fixed tolerance')
            parity_max = (previous-current).abs().max().item()
            reference = helper.Float64Reference(child_model)
            batch = helper.diagnose_batch(child_model, reference, ids, 'random_seed127', 0)
            summary = helper.summarize([batch])
            require(summary['predeclared_supplemental_diagnostic_pass'] is True, 'Synthetic causal/FP64 diagnostic failed')
        original_audit_path = ROOT/plan['parent']['audit']
        require(sha(original_audit_path) == plan['parent']['audit_sha256'], 'Parent asset manifest changed')
        assets = read(original_audit_path)['assets']
        for name, entry in assets.items():
            source = parent_path if name == 'checkpoint.pt' else ROOT/name
            require(sha(source) == entry['sha256'] and source.stat().st_size == entry['bytes'], 'Inherited inference asset differs')
        assets['checkpoint.pt'] = dict(bytes=checkpoint.stat().st_size, sha256=sha(checkpoint))
        total = sum(row['bytes'] for row in assets.values())
        peak = peak_memory()
        require(total <= 64*2**20 and peak <= 4*2**30, 'Preflight memory/asset limit exceeded')
        require(sha(__file__) == script_sha and sha(ROOT/'runs/push152_plan.json') == PLAN_SHA and
            sha(checkpoint) == initialization['checkpoint_sha256'], 'Preflight input changed')
        report = dict(status='completed', plan_sha256=PLAN_SHA, script_sha256=script_sha,
            checkpoint_sha256=sha(checkpoint), initialization_sha256=sha(checkpoint.parent/'initialization.json'),
            helper_sha256=HELPER_SHA, parent_sha256=sha(parent_path), test_used=False, any_corpus_loaded=False,
            normalization_finite_pass=True, parent_prediction_equivalence_pass=True,
            maximum_parent_log_difference=parity_max, untouched_statistical_buffers=21,
            diagnostic_summary=summary, diagnostic_batch=batch, assets=assets,
            inference_assets_bytes=total, peak_process_working_set_bytes=peak,
            asset_limit_pass=True, ram_limit_pass=True,
            synthetic_preflight_only_not_final_trained_model_audit=True, limits_except_timing_pass=True,
            seconds_total=time.perf_counter()-started)
        write_new(directory/'audit.json', report)
        print(json.dumps({key:report[key] for key in ('status','maximum_parent_log_difference','inference_assets_bytes','seconds_total')}))
    except BaseException as error:
        write_new(directory/'audit.failure.json', dict(status='failed', failure_type=type(error).__name__,
            failure_message=str(error), test_used=False, seconds_total=time.perf_counter()-started, no_automatic_retry=True))
        raise


def authorize():
    plan, _, checkpoint, _ = inputs()
    directory = ROOT/plan['expansion']['preflight_dir']
    audit_path, timing_path = directory/'audit.json', directory/'timing/summary.json'
    checked, timing = read(audit_path), read(timing_path)
    cp_sha = sha(checkpoint)
    require(checked['status'] == 'completed' and checked['limits_except_timing_pass'] is True and
        checked['plan_sha256'] == PLAN_SHA and checked['checkpoint_sha256'] == cp_sha and
        checked['script_sha256'] == sha(__file__), 'Preflight audit identity differs')
    for name, entry in checked['assets'].items():
        asset = checkpoint if name == 'checkpoint.pt' else ROOT/name
        require(sha(asset) == entry['sha256'] and asset.stat().st_size == entry['bytes'], 'Audited asset changed')
    require(sum(row['bytes'] for row in checked['assets'].values()) == checked['inference_assets_bytes'], 'Asset sum differs')
    require(timing['candidate_sha256'] == cp_sha and timing['test_used'] is False and timing['split'] == 'validation' and
        timing['device'] == 'cpu' and timing['precision'] == 'fp32' and timing['threads'] == 4 and timing['affinity_mask'] == 15 and
        timing['baseline_sha256'] == sha(ROOT/'runs/baseline-s17/checkpoint.pt') == BASELINE_SHA and
        timing['script_sha256'] == sha(ROOT/'push145_fresh_benchmark.py') == TIMING_SHA and
        timing['evaluator_sha256'] == sha(ROOT/'evaluate.py') == SCORER_SHA, 'Full-validation timing identity differs')
    order = [(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
    require([(row['repeat'],row['name']) for row in timing['measurements']] == order, 'Incomplete alternating pairs')
    seconds = {'baseline':[], 'candidate':[]}
    for row in timing['measurements']:
        raw = read(timing_path.parent/(row['name']+'-'+str(row['repeat'])+'.json'))
        expected_sha = cp_sha if row['name'] == 'candidate' else timing['baseline_sha256']
        require(raw['checkpoint_sha256'] == expected_sha and raw['threads'] == 4 and raw['cpu_affinity_mask'] == 15 and
            raw['split'] == 'validation' and len(raw['repeats']) == 1, 'Original timing identity differs')
        require(row == dict(name=row['name'],repeat=row['repeat'],**raw['repeats'][0],
            peak_process_working_set_bytes=raw['peak_process_working_set_bytes']), 'Timing summary differs from raw record')
        require(row['targets'] == 376599 and row['utf8_bytes'] == 1148007 and math.isfinite(row['seconds']) and row['seconds'] > 0,
            'Invalid/incomplete validation')
        require(math.isfinite(row['bpb']) and row['bpb'] > 0 and
            type(row['peak_process_working_set_bytes']) is int and row['peak_process_working_set_bytes'] > 0,
            'Invalid BPB/memory measurement')
        require(abs(row['bpb']-row['nll_nats']/math.log(2)/row['utf8_bytes']) < 1e-10, 'BPB denominator differs')
        if row['name'] == 'candidate':
            require(abs(row['bpb']-plan['parent']['validation_bpb_cpu_fp32']) <= 1e-5, 'Initialized full validation differs')
        seconds[row['name']].append(row['seconds'])
    ratio = statistics.median(seconds['candidate'])/statistics.median(seconds['baseline'])
    peak = max(row['peak_process_working_set_bytes'] for row in timing['measurements'])
    allowed = ratio <= 5 and peak <= 4*2**30 and checked['inference_assets_bytes'] <= 64*2**20
    report = dict(status='completed', plan_sha256=PLAN_SHA, initializer_checkpoint_sha256=cp_sha,
        timing_summary=str(timing_path.relative_to(ROOT)).replace('\\','/'), timing_summary_sha256=sha(timing_path),
        audit=str(audit_path.relative_to(ROOT)).replace('\\','/'), audit_sha256=sha(audit_path),
        training_authorized=allowed, cpu_ratio=ratio, peak_process_working_set_bytes=peak,
        inference_assets_bytes=checked['inference_assets_bytes'], all_limits_pass=allowed, test_used=False,
        script_sha256=sha(__file__), no_layout_fallback=True)
    write_new(ROOT/plan['expansion']['authorization'], report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit','authorize'))
    args = parser.parse_args()
    {'audit':audit, 'authorize':authorize}[args.phase]()
