"""Train-statistics-only, KL-inspired budget allocation for the existing MKN.

No file, tokenizer, checkpoint, evaluator or dataset loader is used here. The
caller is responsible for supplying only course TRAIN statistics. An entry's
score is the exact KL of its ISOLATED interpolated correction versus the lower
distribution, not the KL of deleting it from a complete interpolated model.
Static amortized-density ranking is a heuristic, not optimal knapsack or exact
global entropy pruning. All inference buffers retain the existing FP32 format.
"""
import math
import numpy as np
import torch

from push145_mkn_build import discounts, make_tables


ORDERS = (3, 4, 5)
FIELDS = ('contexts', 'offsets', 'words', 'masses', 'backoffs')
FIXED_NAMES = ('unigram',) + tuple(name+'_2' for name in FIELDS)
TABLE_NAMES = frozenset(('unigram',) + tuple(name+'_'+str(order) for order in range(2, 6) for name in FIELDS))
DEFAULT_EXTRA_BUDGETS = (0, 8*1024*1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer_vector(values, name, *, positive=False):
    array = np.asarray(values)
    require(array.ndim == 1 and np.issubdtype(array.dtype, np.integer), name+' must be a 1D integer array')
    if array.size:
        require(int(array.min()) >= (1 if positive else 0) and int(array.max()) <= np.iinfo(np.int64).max,
                name+' has invalid integer values')
    return array.astype(np.int64, copy=True)


def tensor_bytes(tables):
    require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' for value in tables.values()),
            'Tables must be CPU tensors')
    return sum(value.numel()*value.element_size() for value in tables.values())


def raw_context_counts_from_ids(ids):
    """Raw counts of histories with a successor; accepts TRAIN ids in memory.

    For order n the denominator is exactly len(ids)-n+1. The final history
    without a target is deliberately excluded. No fragment/fold concatenation
    or document-boundary policy is introduced by this helper.
    """
    ids = integer_vector(ids, 'train ids')
    require(len(ids) >= 5 and int(ids.max()) < 2048, 'Need at least five valid fixed-vocabulary train ids')
    result = {}
    for order in ORDERS:
        size = len(ids)-order+1
        packed = np.zeros(size, dtype=np.int64)
        for offset in range(order-1):
            packed = (packed << 11) | ids[offset:offset+size]
        result[order] = np.unique(packed, return_counts=True)
    return result


def isolated_kl_proxy(mass, lower_probability, context_probability):
    """P_train(h) * KL((1-m)Q + m delta_w || Q), using a binary reduction.

    Other outcomes all rescale by 1-m, making this isolated-entry KL exact.
    It is only a PROXY when multiple entries and lower orders are also pruned.
    With q and context probability fixed, it is nondecreasing in m >= 0.
    No claim is made that frequency orders entries having different q values.
    """
    mass, q, context = np.broadcast_arrays(np.asarray(mass, dtype=np.float64),
        np.asarray(lower_probability, dtype=np.float64), np.asarray(context_probability, dtype=np.float64))
    require(np.isfinite(mass).all() and np.isfinite(q).all() and np.isfinite(context).all(), 'Proxy inputs must be finite')
    require(((mass >= 0) & (mass < 1)).all(), 'Explicit mass must be in [0,1)')
    require(((q > 0) & (q < 1)).all(), 'Lower probability must be strictly between zero and one')
    require(((context >= 0) & (context <= 1)).all(), 'Context probability must be in [0,1]')
    increase = mass*(1-q)
    p = q+increase
    kl = p*np.log1p(increase/q)+(1-p)*np.log1p(-mass)
    # Roundoff around m=0 may produce a tiny negative cancellation residual.
    require((kl >= -1e-12).all(), 'Unexpected negative isolated KL')
    return context*np.maximum(kl, 0.)


def _effective_copy(effective):
    require(set(effective) == set(range(1, 6)), 'Need unpruned effective counts for orders 1..5')
    result = {}
    for order in range(1, 6):
        keys, counts = effective[order]
        keys = integer_vector(keys, f'order{order} keys')
        counts = integer_vector(counts, f'order{order} counts', positive=True)
        require(len(keys) == len(counts) and len(keys) > 0, 'Each effective order must be nonempty and aligned')
        require(np.all(keys[1:] > keys[:-1]) and int(keys[-1]) < (1 << (11*order)), 'Packed keys must be exact, sorted and unique')
        require(int(counts.max()) < 2**53, 'Counts exceed exact FP64 integer accumulation range')
        result[order] = keys, counts
    return result


def _table_numpy(tables):
    return {name: value.detach().numpy() for name, value in tables.items()}


def lower_target_probabilities(reference_tables, packed_contexts, words, maximum_order):
    """Sparse recursive lookup of Q(w|suffix(h)); never creates vocab^N data.

    Reference tables are the UNPRUNED train lower orders. For each query only
    the requested word is gathered. This is for offline scoring, not inference.
    """
    require(1 <= maximum_order <= 4, 'Proxy references only lower orders 1..4')
    contexts = integer_vector(packed_contexts, 'query contexts')
    words = integer_vector(words, 'query words')
    require(len(contexts) == len(words) and (not len(words) or int(words.max()) < 2048), 'Invalid aligned target queries')
    arrays = _table_numpy(reference_tables)
    probabilities = arrays['unigram'][words].astype(np.float64, copy=True)
    for order in range(2, maximum_order+1):
        known = arrays[f'contexts_{order}']
        if not len(known):
            continue
        suffix = contexts & ((1 << (11*(order-1)))-1)
        positions = np.searchsorted(known, suffix)
        safe = np.minimum(positions, len(known)-1)
        exists = (positions < len(known)) & (known[safe] == suffix)
        probabilities[exists] *= arrays[f'backoffs_{order}'][safe[exists]].astype(np.float64)
        offsets = arrays[f'offsets_{order}']
        exact_keys = (np.repeat(known, np.diff(offsets).astype(np.int64)) << 11) | arrays[f'words_{order}'].astype(np.int64)
        if not len(exact_keys):
            continue
        query = (suffix << 11) | words
        indices = np.searchsorted(exact_keys, query)
        bounded = np.minimum(indices, len(exact_keys)-1)
        matched = (indices < len(exact_keys)) & (exact_keys[bounded] == query)
        probabilities[matched] += arrays[f'masses_{order}'][bounded[matched]].astype(np.float64)
    require(np.isfinite(probabilities).all() and (probabilities > 0).all() and (probabilities < 1).all(),
            'Lower-order training reference is not a valid positive target probability')
    return probabilities


def deterministic_budget_mask(priority, score, counts, orders, packed_keys, *, budget_bytes, fixed_bytes):
    """Static order, exact marginal accounting: six bytes plus new-context16.

    Stable declared tie-break: density DESC, score DESC, count DESC, order ASC,
    packed key ASC. Entries too expensive at their turn are skipped; this is
    not a global knapsack optimum or a dynamic-density re-ranking algorithm.
    """
    require(isinstance(budget_bytes, int) and not isinstance(budget_bytes, bool) and
            isinstance(fixed_bytes, int) and not isinstance(fixed_bytes, bool) and
            0 <= fixed_bytes <= budget_bytes, 'Invalid budget or insufficient fixed-buffer budget')
    priorities = np.asarray(priority, dtype=np.float64)
    scores = np.asarray(score, dtype=np.float64)
    counts = integer_vector(counts, 'priority counts', positive=True)
    orders = integer_vector(orders, 'priority orders')
    keys = integer_vector(packed_keys, 'priority keys')
    require(priorities.ndim == scores.ndim == 1 and
            len(priorities) == len(scores) == len(counts) == len(orders) == len(keys), 'Priority vectors must be aligned')
    require(np.isfinite(priorities).all() and np.isfinite(scores).all() and
            (priorities >= 0).all() and (scores >= 0).all() and np.isin(orders, ORDERS).all(), 'Invalid priority values')
    require(all(int(key) < (1 << (11*int(order))) for key,order in zip(keys,orders)), 'Priority packed key overflow')
    # Four-token history uses at most 44 bits; the order occupies disjoint bits.
    context_ids, inverse = np.unique((orders << 44) | (keys >> 11), return_inverse=True)
    ordered = np.lexsort((keys, orders, -counts, -scores, -priorities))
    active = np.zeros(len(context_ids), dtype=bool)
    keep = np.zeros(len(keys), dtype=bool)
    used = fixed_bytes
    for index in ordered:
        context = inverse[index]
        cost = 6 if active[context] else 22
        if used+cost <= budget_bytes:
            keep[index] = True
            active[context] = True
            used += cost
    return keep, dict(used_tensor_bytes=used,unused_budget_bytes=budget_bytes-used,
        selected_entries=int(keep.sum()),selected_contexts=int(active.sum()),
        selected_proxy_score=float(scores[keep].sum(dtype=np.float64)))


def _materialize_order(keys, masses, probability_dtype=np.float32):
    contexts, starts = np.unique(keys >> 11, return_index=True)
    require(len(keys) <= np.iinfo(np.int32).max, 'Offsets would overflow int32')
    offsets = np.r_[starts, len(keys)].astype(np.int32)
    masses = masses.astype(probability_dtype)
    backoffs = ((1-np.add.reduceat(masses.astype(np.float64), starts)).astype(probability_dtype)
                if len(starts) else np.zeros(0, dtype=probability_dtype))
    require(np.all(backoffs >= -1e-6) and np.all(backoffs <= 1+1e-6), 'Invalid pruned backoff mass')
    backoffs = np.clip(backoffs, 0, 1)
    return dict(contexts=torch.from_numpy(contexts), offsets=torch.from_numpy(offsets),
        words=torch.from_numpy((keys & 2047).astype(np.int16)), masses=torch.from_numpy(masses),
        backoffs=torch.from_numpy(backoffs))


def unpruned_reference_tables(effective, parent_arrays, scale=.7):
    """Build-time Q: fixed FP32 base, directly constructed FP64 orders 3/4.

    The fixed base is intentionally not recomputed in a different precision.
    Upper-order explicit masses and backoffs are never rounded through FP32.
    No fifth-order reference is needed to value a fifth-order correction.
    """
    reference = {name:parent_arrays[name].detach().clone() for name in FIXED_NAMES}
    for order in (3, 4):
        keys,counts = effective[order]
        _,starts,lengths = np.unique(keys >> 11, return_index=True, return_counts=True)
        totals = np.add.reduceat(counts.astype(np.float64), starts)
        base,_ = discounts(counts)
        chosen = np.clip(base*scale,.01,[0.99,1.99,2.99])
        masses = (counts-chosen[np.minimum(counts,3)-1])/np.repeat(totals,lengths)
        for field,value in _materialize_order(keys,masses,np.float64).items():
            reference[field+'_'+str(order)] = value
    return reference


def build_pruned_candidates(effective, raw_context_counts, parent_arrays,
                            extra_budgets=DEFAULT_EXTRA_BUDGETS, scale=.7):
    """Return [(21-buffer tables, JSON-safe report)] for fixed extra budgets.

    Scores are prepared ONCE and shared across the requested budget variants.
    The parent must exactly match the original minimum2/trigram-minimum1 MKN
    rebuild. Budgets concern uncompressed tensor payload, not torch ZIP bytes;
    the caller must separately audit total inference assets and CPU runtime.
    """
    require(isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale > 0,
            'Discount scale must be finite and positive')
    require(set(parent_arrays) == TABLE_NAMES and set(raw_context_counts) == set(ORDERS), 'Incorrect table or raw-context schema')
    parent_bytes = tensor_bytes(parent_arrays)
    effective = _effective_copy(effective)
    rebuilt, parent_stats = make_tables(effective, scale=scale, minimum=2, trigram_minimum=1)
    for name in TABLE_NAMES:
        require(parent_arrays[name].dtype == rebuilt[name].dtype and parent_arrays[name].shape == rebuilt[name].shape and
                torch.equal(parent_arrays[name], rebuilt[name]), 'Original MKN bitwise rebuild mismatch: '+name)
    reference = unpruned_reference_tables(effective,parent_arrays,scale)
    for name in FIXED_NAMES:
        require(torch.equal(reference[name], parent_arrays[name]), 'Unpruned reference changed fixed low-order table')
    raw, inferred_lengths = {}, []
    for order in ORDERS:
        contexts, counts = raw_context_counts[order]
        contexts = integer_vector(contexts, f'raw{order} contexts')
        counts = integer_vector(counts, f'raw{order} counts', positive=True)
        require(len(contexts) == len(counts) and len(contexts) > 0 and np.all(contexts[1:] > contexts[:-1]) and
                int(contexts[-1]) < (1 << (11*(order-1))), 'Raw context keys must be exact, sorted, unique and aligned')
        total = sum(map(int, counts))
        inferred_lengths.append(total+order-1)
        raw[order] = contexts, counts, total
    require(len(set(inferred_lengths)) == 1 and raw[5][2] == sum(map(int, effective[5][1])),
            'Raw denominators must be len(train_ids)-order+1 and agree with highest-order raw counts')
    all_keys, all_counts, all_orders, all_scores, all_priorities = [], [], [], [], []
    order_data, score_stats = {}, []
    for order in ORDERS:
        keys, counts = effective[order]
        contexts, starts, lengths = np.unique(keys >> 11, return_index=True, return_counts=True)
        totals = np.add.reduceat(counts.astype(np.float64), starts)
        base, _ = discounts(counts)
        chosen = np.clip(base*scale, .01, [0.99, 1.99, 2.99])
        masses = (counts-chosen[np.minimum(counts, 3)-1])/np.repeat(totals, lengths)
        raw_keys, raw_counts, raw_total = raw[order]
        indices = np.searchsorted(raw_keys, contexts)
        require(np.all(indices < len(raw_keys)) and np.array_equal(raw_keys[np.minimum(indices,len(raw_keys)-1)],contexts),
                'An effective context is missing from raw TRAIN context counts')
        frequencies = np.repeat(raw_counts[indices].astype(np.float64)/raw_total, lengths)
        lower = lower_target_probabilities(reference, keys >> 11, keys & 2047, order-1)
        scores = isolated_kl_proxy(masses, lower, frequencies)
        priorities = scores/(6.+16./np.repeat(lengths, lengths))
        all_keys.append(keys); all_counts.append(counts); all_orders.append(np.full(len(keys),order,dtype=np.int64))
        all_scores.append(scores); all_priorities.append(priorities)
        order_data[order] = dict(keys=keys,masses=masses,length=len(keys))
        score_stats.append(dict(order=order,candidate_entries=len(keys),candidate_contexts=len(contexts),
            raw_context_occurrences=raw_total,original_effective_count_total=int(counts.sum()),
            applied_discounts=chosen.tolist(),minimum_lower_probability=float(lower.min()),
            maximum_lower_probability=float(lower.max()),proxy_score_sum=float(scores.sum(dtype=np.float64))))
    keys, counts, orders, scores, priorities = map(np.concatenate,
        (all_keys, all_counts, all_orders, all_scores, all_priorities))
    fixed_bytes = tensor_bytes({name:parent_arrays[name] for name in FIXED_NAMES})+4*len(ORDERS)
    extras = tuple(extra_budgets)
    require(len(extras) > 0 and len(set(extras)) == len(extras) and
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in extras),
            'Extra budgets must be distinct nonnegative integer byte counts')
    results = []
    for extra in extras:
        budget = parent_bytes+extra
        keep, accounting = deterministic_budget_mask(priorities,scores,counts,orders,keys,
            budget_bytes=budget,fixed_bytes=fixed_bytes)
        tables = {name:parent_arrays[name].detach().clone() for name in FIXED_NAMES}
        cursor = 0; kept_stats = []
        for order in ORDERS:
            data = order_data[order]; selected = keep[cursor:cursor+data['length']]
            selected_keys = data['keys'][selected]
            values = _materialize_order(selected_keys,data['masses'][selected])
            for field,value in values.items():
                tables[field+'_'+str(order)] = value
            kept_stats.append(dict(order=order,entries=int(selected.sum()),contexts=values['contexts'].numel(),
                tensor_bytes=tensor_bytes(values),retained_effective_count_mass=float(effective[order][1][selected].sum()/effective[order][1].sum())))
            cursor += data['length']
        actual = tensor_bytes(tables)
        require(set(tables) == TABLE_NAMES and actual == accounting['used_tensor_bytes'] <= budget, 'Exact serialized tensor budget disagrees')
        for name in FIXED_NAMES:
            require(tables[name].dtype == parent_arrays[name].dtype and torch.equal(tables[name],parent_arrays[name]),
                    'Fixed unigram/bigram changed')
        report = dict(algorithm='isolated-entry KL proxy, static amortized-density ordering with exact marginal budget',
            proxy_formula='P_train_raw(h) * KL((1-m)Q + m*delta_w || Q); m=(c-D_c)/unpruned_effective_total(h)',
            lower_reference='unpruned TRAIN lower-order interpolation; fixed parent unigram and bigram',
            reference_precision='Fixed parent order1/2 FP32 values promoted in recursion; order3/4 masses/backoffs constructed directly FP64; no order5 reference',
            recursive_lookup_and_proxy_precision='FP64',
            density_formula='score/(6+16/n_h), n_h=all unpruned entries in this context',
            tie_break='density descending, score descending, effective_count descending, order ascending, packed_key ascending',
            not_exact_complete_model_deletion_kl=True,not_optimal_knapsack=True,
            original_parent_tables_rebuilt_bitwise=True,low_orders_fixed_bitwise=True,
            original_denominators_and_discounts_preserved=True,removed_probability_mass_backed_off=True,
            table_buffers=21,inference_implementation_unchanged=True,all_probability_buffers_fp32=True,
            source_contract='Caller supplied in-memory TRAIN-only counts; helper performs no file I/O or scoring',
            train_tokens_inferred_from_raw_context_counts=inferred_lengths[0],
            parent_tensor_bytes=parent_bytes,additional_budget_bytes=extra,budget_tensor_bytes=budget,
            fixed_tensor_bytes_including_three_offset_sentinels=fixed_bytes,
            marginal_entry_bytes=6,new_context_extra_bytes=16,discount_scale=float(scale),
            score_preparations_shared_across_budgets=1,unpruned_score_statistics=score_stats,
            parent_frequency_pruned_statistics=parent_stats,selected_statistics=kept_stats,**accounting)
        results.append((tables,report))
    return results
