"""Raw-key union cross-fitting and complete, within-region training windows."""
import numpy as np
import torch


CONTEXT = 256
FOLDS = 10


def contiguous_folds(length, folds=FOLDS, context=CONTEXT):
    blocks = length//context
    if folds < 2 or blocks < folds*2:
        raise ValueError('Every fold needs at least one complete 257-token window.')
    boundaries = [(index*blocks//folds)*context for index in range(folds)]+[length]
    return [(boundaries[index], boundaries[index+1]) for index in range(folds)]


def effective_counts_segments(segments, maximum=5):
    """Do not concatenate text or sum precomputed continuation counts.

Only already encoded n-grams are concatenated. Globally unique (n+1)-gram
types determine lower-order distinct predecessor counts across ALL segments.
"""
    if not 2 <= maximum <= 5:
        raise ValueError('Lossless 64-bit keys support maximum order 2..5.')
    segments = [np.asarray(segment, dtype=np.int64) for segment in segments]
    if not segments or any(segment.ndim != 1 for segment in segments):
        raise ValueError('Expected nonempty list of one-dimensional segments.')
    if any(np.any((segment < 0) | (segment >= 2048)) for segment in segments):
        raise ValueError('Unexpected token id.')
    keys, counts = {}, {}
    for order in range(1, maximum+1):
        parts = []
        for segment in segments:
            size = len(segment)-order+1
            if size <= 0:
                continue
            packed = np.zeros(size, dtype=np.int64)
            for offset in range(order):
                packed = (packed << 11) | segment[offset:offset+size]
            parts.append(packed)
        if not parts:
            raise ValueError('No observations at the maximum order.')
        keys[order], counts[order] = np.unique(np.concatenate(parts), return_counts=True)
    effective = {}
    for order in range(1, maximum):
        effective[order] = np.unique(keys[order+1] & ((1 << (11*order))-1), return_counts=True)
    effective[maximum] = keys[maximum], counts[maximum]
    return effective


def complete_window_starts(start, end, context=CONTEXT):
    # Input AND next-token targets span context+1 tokens, all inside the fold.
    return np.arange(start, end-context, context, dtype=np.int64)


def fold_windows(ids, bounds, batch_size=32, context=CONTEXT):
    ids = np.asarray(ids, dtype=np.int64)
    start, end = bounds
    if not 0 <= start < end <= len(ids) or batch_size <= 0:
        raise ValueError('Invalid fold/window bounds.')
    starts = complete_window_starts(start, end, context)
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset+batch_size]
        positions = selected[:, None]+np.arange(context+1, dtype=np.int64)[None, :]
        if positions.min() < start or positions.max() >= end:
            raise AssertionError('Training window crossed its held-out fold boundary.')
        chunks = ids[positions]
        yield torch.from_numpy(chunks[:, :-1].copy()), torch.from_numpy(chunks[:, 1:].copy())
