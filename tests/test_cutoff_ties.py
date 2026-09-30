from __future__ import annotations

import numpy as np

from scripts.check_cutoff_ties import MACHINE_TOLERANCE_MULTIPLIER


def test_machine_scale_cutoff_tolerance_is_explicit_and_small():
    scale = max(1.0, 0.25, 0.25 + 10 * np.finfo(float).eps)
    tolerance = MACHINE_TOLERANCE_MULTIPLIER * np.finfo(float).eps * scale
    assert tolerance > 0
    assert tolerance < 1e-12


def test_distinct_cutoff_gap_is_not_a_tie():
    d_k, d_next = 0.25, 0.250001
    scale = max(1.0, abs(d_k), abs(d_next))
    tolerance = MACHINE_TOLERANCE_MULTIPLIER * np.finfo(float).eps * scale
    assert d_next != d_k
    assert abs(d_next - d_k) > tolerance
