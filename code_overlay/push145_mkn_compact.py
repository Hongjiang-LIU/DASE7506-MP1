"""Losslessly encoded old absolute-discount model, low-rank neural parent.

Uses the proven compact inference kernel; this is NOT the new MKN smoother.
"""
from push145_mkn import MKNLowRank


def build_model(config):
    return MKNLowRank(config)
