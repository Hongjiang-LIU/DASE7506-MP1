"""Fixed 15-choice, validation-only temperature / count-mixture calibration.

The grid is fixed by push150_phase_c_suffix_plan.json before the experiment,
not by any test result. No neural training or test-file I/O occurs here.
The actual target-free predictor is checked with the unchanged full scorer.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from common import PROTOCOL, ROOT, setup, make_model, sha, windows
from evaluate import score
from push145_mkn_select import target_probabilities
from push150_suffix_train import source_digests


TEMPERATURES = (1.20, 1.25, 1.30, 1.35, 1.40)
MIXTURE_WEIGHTS = (.10, .15, .20)


@torch.no_grad()
def neural_cache_targets(model, ids, targets):
    """Scoring-only gather; inference model never receives target tokens."""
    features = model.features(ids)
    logits = model.head(features).float()
    attention, gate = model.cache_attention_and_gate(features, ids)
    word_ids = torch.cat([ids[:, 1:], ids.new_zeros(ids.shape[0], 1)], dim=1)
    memory = torch.zeros_like(logits)
    memory.scatter_add_(-1, word_ids[:, None, :].expand(-1, ids.shape[1], -1), attention)
    target = targets.clamp_min(0).unsqueeze(-1)
    cache_target = memory.gather(-1, target).squeeze(-1)
    cache_gate = gate.squeeze(-1)
    result = {}
    for temperature in TEMPERATURES:
        neural = (logits/temperature).softmax(-1).gather(-1, target).squeeze(-1)
        result[temperature] = neural*(1-cache_gate)+cache_target*cache_gate
    return result


def score_grid(neural_by_temperature, count_targets, targets, byte_count):
    if byte_count <= 0:
        raise ValueError('The byte count must be positive.')
    valid = targets != -100
    if not valid.any():
        raise ValueError('No scored targets.')
    trials = []
    for temperature, weight in itertools.product(TEMPERATURES, MIXTURE_WEIGHTS):
        neural = neural_by_temperature[temperature]
        mixed = (1-weight)*neural+weight*count_targets
        nll = float(-np.log(mixed[valid].astype(np.float64).clip(1e-300)).sum())
        trials.append(dict(temperature=temperature, mkn_weight=weight,
                           validation_bpb=nll/math.log(2)/byte_count,
                           validation_nll_nats=nll))
    # Ties follow the declared grid order. All original trials remain recorded.
    return trials, min(trials, key=lambda row: row['validation_bpb'])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent_sha = sha(args.parent)
    pinned_sources = source_digests('push150_suffix')
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                parent_path=str(args.parent), parent_sha256=parent_sha,
                temperatures=list(TEMPERATURES), mkn_weights=list(MIXTURE_WEIGHTS),
                trials=15, split='validation', precision='fp32', device='cuda',
                rationale='The fixed Phase-C 15-choice grid; suffix matching receives only input ids.',
                frozen_other_config=True, new_training_targets=0,
                test_used=False, prior_test_exposure_disclosed=True,
                script_sha256=sha(__file__), source_sha256=pinned_sources)
    (args.run_dir/'experiment_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    device, _ = setup('cuda', 'fp32', 2)
    checkpoint = torch.load(args.parent, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError('Checkpoint has the wrong protocol.')
    if checkpoint.get('implementation') != 'push150_suffix':
        raise ValueError('Expected the declared Phase-C suffix implementation.')
    if not checkpoint['config'].get('mkn_shapes'):
        raise ValueError('Expected compact train-derived n-gram tables in parent.')
    model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    if not callable(getattr(model, 'cache_attention_and_gate', None)):
        raise ValueError('Parent must expose a frozen learned-cache predictor.')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    path = ROOT/'data/wikitext_validation.txt'
    tokenizer_path = ROOT/'data/tokenizer.json'
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    if sha(path) != manifest['sha256'][path.name] or sha(tokenizer_path) != manifest['sha256'][tokenizer_path.name]:
        raise ValueError('Validation data or tokenizer was changed.')
    raw = path.read_bytes()
    tokens = torch.tensor(Tokenizer.from_file(str(tokenizer_path)).encode(raw.decode('utf-8')).ids)
    xs, ys = [], []
    values = {temperature: [] for temperature in TEMPERATURES}
    for x, y in windows(tokens):
        gathered = neural_cache_targets(model, x.to(device), y.to(device))
        for temperature in TEMPERATURES:
            values[temperature].append(gathered[temperature].cpu().numpy())
        xs.append(x.numpy()); ys.append(y.numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    neural = {temperature: np.concatenate(parts) for temperature, parts in values.items()}
    arrays = {k.removeprefix('ngram.'): v.cpu().numpy()
              for k, v in checkpoint['model'].items() if k.startswith('ngram.')}
    count_targets = target_probabilities(arrays, x, y, maximum=checkpoint['config'].get('mkn_order', 5))
    trials, best = score_grid(neural, count_targets, y, len(raw))
    # Store the entire screen before full confirmation, so failures remain visible.
    (args.run_dir/'trials.json').write_text(json.dumps(trials, indent=2)+'\n')
    parent_config = dict(checkpoint['config'])
    checkpoint['config'].update(temperature=best['temperature'], mkn_weight=best['mkn_weight'])
    checkpoint['calibration_parent_sha256'] = parent_sha
    checkpoint['selection'] = '15 declared validation-only temperature/mixture settings; all weights frozen'
    destination = args.run_dir/'checkpoint.pt'
    torch.save(checkpoint, destination)
    # Reload saved output rather than relying on the mutable search model.
    frozen = torch.load(destination, map_location='cpu', weights_only=True)
    confirmed_model, _ = make_model(frozen['implementation'], frozen['config'], device)
    confirmed_model.load_state_dict(frozen['model'])
    confirmed_model.eval()
    measured = score(confirmed_model, tokens, len(raw), device, 'fp32')
    losses = measured.pop('window_nll_nats')
    np.save(args.run_dir/'validation.window-nll.npy', np.asarray(losses, dtype=np.float64))
    difference = abs(measured['bpb']-best['validation_bpb'])
    unchanged_config = {k: v for k, v in checkpoint['config'].items() if k not in ('temperature', 'mkn_weight')}
    expected_config = {k: v for k, v in parent_config.items() if k not in ('temperature', 'mkn_weight')}
    if unchanged_config != expected_config:
        raise AssertionError('An undeclared configuration changed.')
    for name, tensor in checkpoint['model'].items():
        if not torch.equal(tensor, frozen['model'][name]):
            raise AssertionError(f'Saved parameter changed: {name}')
    if source_digests('push150_suffix') != pinned_sources or sha(args.parent) != parent_sha:
        raise ValueError('Pinned source or parent changed during calibration.')
    result = dict(best=best, trials=trials, full_validation=measured,
                  shortcut_full_bpb_absolute_difference=difference,
                  shortcut_matches_full_predictor=difference < 1e-6,
                  device=str(device), precision='fp32', threads=2,
                  checkpoint_sha256=sha(destination), checkpoint_bytes=destination.stat().st_size,
                  parent_sha256=parent_sha, parent_config=parent_config,
                  implementation=checkpoint['implementation'], implementation_sha256=implementation_sha,
                  new_training_targets=0, test_used=False, prior_test_exposure_disclosed=True,
                  inherited_processed_view_targets=checkpoint['train_tokens'], source_sha256=pinned_sources,
                  script_sha256=sha(__file__), evaluator_sha256=sha(ROOT/'evaluate.py'),
                  validation_sha256=sha(path), tokenizer_sha256=sha(tokenizer_path),
                  validation_target_occurrences=int((y != -100).sum()),
                  validation_utf8_bytes=len(raw), seconds_total=time.perf_counter()-started)
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if difference >= 1e-6:
        raise AssertionError('Scoring shortcut differs from the actual full predictor.')


if __name__ == '__main__':
    main()
