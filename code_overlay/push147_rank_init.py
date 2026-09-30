"""CPU SVD initialization from owned train-derived dense and pointer weights.

Reads checkpoint files only, never any training/validation/test text. Two-parent
costs are explicit. Unless an overlap is explicitly supplied and documented,
train_tokens is a conservative sum of both parent chains, not their maximum.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from common import PROTOCOL, sha
from lowrank import LowRankLinear
from push147_rank import DEFAULT_RANK_LAYOUT, RankLayoutCache, projection_modules


STATISTICS_CONFIG_PREFIXES = ("mkn_", "ngram_", "discount_", "statistics_", "table_", "count_")


def verified_shared_evidence(dense_sha, pointer_sha, shared_targets, *, code_root=None):
    """Verify the known local parent graph from JSON logs, never dataset text.

    A positive deduction is accepted only for this reproducible metadata chain.
    Unrecognized parent combinations must use the conservative zero deduction.
    """
    if shared_targets == 0:
        return dict(status="no_overlap_deducted", shared_ancestor_targets=0,
                    note="Both parent costs are summed conservatively; no unique-cost claim")
    if shared_targets != 98304000:
        raise ValueError("This initializer can verify only the known 98,304,000-target shared ancestor")
    root = Path(__file__).resolve().parent if code_root is None else Path(code_root)
    documents = []

    def read(relative):
        path = root / relative
        record = json.loads(path.read_text())
        documents.append(dict(path=relative, sha256=sha(path)))
        return record

    initial = read("runs/gpu-large-s31/metrics.json")
    ancestor_sha = initial["best"]["checkpoint_sha256"]
    if initial["total_search_targets"] != shared_targets:
        raise ValueError("Dense ancestor training-target count does not match metadata")
    dense_refinement = read("runs/refine-large-fine/metrics.json")
    if dense_refinement["parent_sha256"] != ancestor_sha or dense_sha not in (
            ancestor_sha, dense_refinement["checkpoint_sha256"]):
        raise ValueError("Dense input is not the documented shared ancestor or its score-only calibration")
    # source path, predecessor field, successor field, selected incremental SGD targets
    specifications = [
        ("runs/large-svd128/compression.json", "parent_sha256", "checkpoint_sha256", 0),
        ("runs/lowrank128-s71/metrics.json", "parent_sha256", "best.checkpoint_sha256", 49152000),
        ("runs/refine-lowrank128-s71/metrics.json", "parent_sha256", "checkpoint_sha256", 0),
        ("runs/push145-cache-head-s103/metrics.json", "parent_sha256", "best.checkpoint_sha256", 16384000),
        ("runs/push145-cache-joint-s107/metrics.json", "parent_sha256", "best.checkpoint_sha256", 16384000),
        ("runs/push145-cache-joint-mkn/validation.json", "neural_parent_sha256", "checkpoint_sha256", 0),
        ("runs/push145-cache-calibrated/metrics.json", "parent_sha256", "checkpoint_sha256", 0),
        ("runs/push146-continue-drop20/metrics.json", "parent_sha256", "best.checkpoint_sha256", 32768000),
        ("runs/push146-drop20-calibrated/metrics.json", "parent_sha256", "checkpoint_sha256", 0),
    ]
    edges, previous, total = [], ancestor_sha, shared_targets
    for relative, parent_field, child_field, increment in specifications:
        record = read(relative)
        child = record
        for component in child_field.split("."):
            child = child[component]
        if record[parent_field] != previous:
            raise ValueError(f"Broken shared-ancestor hash link in {relative}")
        if increment and record.get("incremental_search_targets") != increment:
            raise ValueError(f"Unexpected training-target increment in {relative}")
        total += increment
        edges.append(dict(source=relative, parent_sha256=previous, checkpoint_sha256=child,
                          selected_incremental_sgd_targets=increment, cumulative_targets=total))
        previous = child
    if previous != pointer_sha:
        raise ValueError("Pointer input is not the endpoint of the verified shared-ancestor chain")
    return dict(status="verified_local_metadata_chain", dense_parent_sha256=dense_sha,
                pointer_parent_sha256=pointer_sha, shared_ancestor_sha256=ancestor_sha,
                shared_ancestor_targets=shared_targets, unique_inherited_sgd_targets=total,
                explanation="Dense 98,304,000 targets are already included in pointer 212,992,000; calibration/SVD/count attachment add no SGD targets",
                dense_calibration_source="runs/refine-large-fine/metrics.json",
                metadata_documents=documents, pointer_ancestry_edges=edges,
                note="Count-table scans and all off-line search costs are separate disclosures; this is the selected SGD ancestry only")


def lineage_targets(dense_targets, pointer_targets=0, shared_targets=0):
    values = (dense_targets, pointer_targets, shared_targets)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
        raise ValueError("Ancestry target counts must be nonnegative integers")
    if shared_targets > min(dense_targets, pointer_targets):
        raise ValueError("Shared ancestor targets cannot exceed either parent chain")
    return dense_targets + pointer_targets - shared_targets


@torch.no_grad()
def initialize_from_dense(dense_checkpoint, pointer_checkpoint=None, *, rank_layout=None,
                          include_statistics=False, seed=147):
    """Return an unselected model and SVD diagnostics, with no data evaluation.

    The optional learned pointer came from a different feature representation:
    copied weights are an initialization, not a claim of preserved predictions.
    Every dense source matrix is required explicitly; an already factorized
    checkpoint is rejected rather than silently inventing additional rank.
    """
    torch.manual_seed(seed)
    config = dict(dense_checkpoint["config"])
    for name in ("ngram_shapes", "mkn_shapes", "rank_layout"):
        config.pop(name, None)
    config["rank_layout"] = dict(DEFAULT_RANK_LAYOUT if rank_layout is None else rank_layout)
    config.setdefault("pointer_rank", 32)
    config.setdefault("pointer_max_weight", .5)
    config.setdefault("cache_weight", .05)
    config.setdefault("cache_theta", 15.)
    config.setdefault("temperature", 1.15)
    if pointer_checkpoint is not None:
        parent_config = pointer_checkpoint["config"]
        if parent_config["width"] != config["width"]:
            raise ValueError("Pointer parent width must match the dense backbone width")
        if parent_config["context"] != config["context"]:
            raise ValueError("Pointer parent context must match the dense backbone context")
        for name in ("pointer_rank", "pointer_max_weight", "cache_weight", "cache_theta", "temperature"):
            if name in parent_config:
                config[name] = parent_config[name]
        if include_statistics:
            # Include provenance-only options such as discount_scale as well as
            # operational table shapes/weights; do not silently lose the recipe.
            for name, value in parent_config.items():
                if name.startswith(STATISTICS_CONFIG_PREFIXES):
                    config[name] = copy.deepcopy(value)
    elif include_statistics:
        raise ValueError("--include-statistics requires a pointer parent checkpoint")
    model = RankLayoutCache(config)
    state = model.state_dict()
    original = dense_checkpoint["model"]
    projection_prefixes = tuple(name + "." for name, _, _ in projection_modules(model))
    for name, target in state.items():
        if name.startswith(("pointer_", "ngram.")) or name.startswith(projection_prefixes):
            continue
        source = original.get(name)
        if source is None or source.shape != target.shape:
            raise ValueError(f"Missing or incompatible dense backbone tensor: {name}")
        state[name] = source.detach().cpu().to(target.dtype).clone()
    errors = []
    for name, semantic_name, module in projection_modules(model):
        source = original.get(name + ".weight")
        if source is None:
            raise ValueError(f"Dense source required at {name}.weight; factorized parents are not accepted")
        in_features = module.left.in_features if isinstance(module, LowRankLinear) else module.in_features
        out_features = module.right.out_features if isinstance(module, LowRankLinear) else module.out_features
        if tuple(source.shape) != (out_features, in_features):
            raise ValueError(f"Dense source shape mismatch at {name}: {tuple(source.shape)}")
        weight = source.detach().cpu().float()
        if not torch.isfinite(weight).all():
            raise ValueError(f"Non-finite dense source at {name}")
        rank = config["rank_layout"][semantic_name]
        if rank == 0:
            state[name + ".weight"] = weight.clone()
            discarded = 0.
        else:
            u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
            root = singular[:rank].sqrt()
            state[name + ".left.weight"] = vh[:rank] * root[:, None]
            state[name + ".right.weight"] = u[:, :rank] * root[None, :]
            discarded = (singular[rank:].square().sum() /
                         singular.square().sum().clamp_min(1e-30)).item()
        errors.append(dict(layer=name, rank=rank, dense_shape=list(weight.shape),
                           discarded_energy_fraction=discarded))
    if pointer_checkpoint is not None:
        pointer_state = pointer_checkpoint["model"]
        for name, target in state.items():
            if not name.startswith(("pointer_", "ngram.")):
                continue
            source = pointer_state.get(name)
            if source is None or source.shape != target.shape:
                raise ValueError(f"Missing or incompatible pointer/statistical tensor: {name}")
            state[name] = source.detach().cpu().to(target.dtype).clone()
    state["head.weight"] = state["token.weight"]
    model.load_state_dict(state, strict=True)
    return model, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-parent", required=True, type=Path)
    parser.add_argument("--pointer-parent", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rank-layout", default=json.dumps(DEFAULT_RANK_LAYOUT),
                        help="JSON object containing qkv/proj/gate/value/out ranks; zero means dense")
    parser.add_argument("--shared-ancestor-targets", type=int, default=0,
                        help="Explicitly verified parent-chain overlap; default sums both conservatively")
    parser.add_argument("--include-statistics", action="store_true")
    parser.add_argument("--seed", type=int, default=147)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    dense = torch.load(args.dense_parent, map_location="cpu", weights_only=True)
    pointer = (torch.load(args.pointer_parent, map_location="cpu", weights_only=True)
               if args.pointer_parent else None)
    if dense.get("protocol") != PROTOCOL or (pointer is not None and pointer.get("protocol") != PROTOCOL):
        raise ValueError("Parent checkpoint belongs to a different course protocol")
    dense_targets = int(dense["train_tokens"])
    pointer_targets = int(pointer["train_tokens"]) if pointer is not None else 0
    accounted_targets = lineage_targets(dense_targets, pointer_targets, args.shared_ancestor_targets)
    model, errors = initialize_from_dense(dense, pointer, rank_layout=json.loads(args.rank_layout),
                                         include_statistics=args.include_statistics, seed=args.seed)
    parents = [dict(role="dense backbone", path=str(args.dense_parent.resolve()),
                    sha256=sha(args.dense_parent), train_targets=dense_targets)]
    if pointer is not None:
        parents.append(dict(role="learned pointer" + (" and statistics" if args.include_statistics else ""),
                            path=str(args.pointer_parent.resolve()), sha256=sha(args.pointer_parent),
                            train_targets=pointer_targets))
    evidence = verified_shared_evidence(parents[0]["sha256"],
        parents[1]["sha256"] if pointer is not None else None, args.shared_ancestor_targets)
    if evidence.get("unique_inherited_sgd_targets", accounted_targets) != accounted_targets:
        raise ValueError("Checkpoint train_tokens do not match the verified metadata-chain costs")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = args.run_dir / "shared_ancestor_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    evidence_reference = dict(path=evidence_path.name, sha256=sha(evidence_path), status=evidence["status"])
    ancestry = dict(parents=parents, sum_parent_targets_before_overlap=dense_targets + pointer_targets,
                    shared_ancestor_targets=args.shared_ancestor_targets,
                    shared_ancestor_policy="Explicit operator-supplied overlap; zero means conservative sum",
                    shared_ancestor_evidence=evidence_reference,
                    accounted_lineage_targets=accounted_targets,
                    incremental_training_targets=0,
                    warning="Full search costs for both parent branches remain reportable separately")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state["head.weight"] = state["token.weight"]
    checkpoint = dict(protocol=PROTOCOL, implementation="push147_rank", config=model.config,
                      model=state, seed=args.seed, train_tokens=accounted_targets,
                      weights="dense SVD plus optional copied pointer; not yet recovered",
                      selection="unselected initialization; no held-out scoring performed",
                      initialization_ancestry=ancestry, ancestry=ancestry, source_text_read=False,
                      parent_sha256=parents[0]["sha256"],
                      pointer_parent_sha256=parents[1]["sha256"] if pointer is not None else None)
    torch.save(checkpoint, args.run_dir / "checkpoint.pt")
    info = dict(config=model.config, initialization_ancestry=ancestry,
                shared_ancestor_evidence=evidence_reference, errors=errors,
                parameters=sum(p.numel() for p in model.parameters()),
                checkpoint_bytes=(args.run_dir / "checkpoint.pt").stat().st_size,
                checkpoint_sha256=sha(args.run_dir / "checkpoint.pt"),
                script_sha256=sha(Path(__file__)),
                implementation_sha256=sha(Path(__file__).with_name("push147_rank.py")),
                seconds=time.perf_counter() - started, source_text_read=False,
                train_used=False, validation_used=False, test_used=False,
                pointer_transfer_note="Dense and pointer parents have different learned feature coordinates; recovery training is required")
    (args.run_dir / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")
    (args.run_dir / "initialization.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "errors"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
