#!/usr/bin/env python3
"""
rq1-2_BPI17_parallel.py  —  RQ1.2  FDR Surface over (n, rho): BPI Challenge 2017
==================================================================================

PURPOSE
-------
Estimate the function  FDR_emp(n, rho)  for all four methods (Fisher-Storey,
BH-Fisher, Cecconi, Tusher) across a feasibility-constrained grid of
(n_eff, rho) pairs derived from the BPI Challenge 2017 loan-application log.
Fit a smooth curve over log(n_eff) per method and identify the minimum
effective sample size n*_eff at which each method achieves FDR control.

DATASET
-------
BPI Challenge 2017 (Teinemaa et al. TKDE 2019 — bpic2017_accepted definition):
    Class 1 (Not-Accepted / Deviant): 18,747 cases  (59.5 %)
    Class 0 (Accepted   / Normal):    12,762 cases  (40.5 %)
    Total:                            31,509 cases
    Imbalance ratio (maj/min):         1.469

    Natural rho (fraction of class 1) ≈ 0.595
    Pool sizes: n_pos_pool = 18,747  (class 1),  n_neg_pool = 12,762  (class 0)

SCIENTIFIC ESTIMAND
-------------------
FDR_emp : (n, rho) -> [0, 1]  where  n_eff = 2*n*rho*(1-rho)

    FDR_emp(n, rho) = E_b[ |S_b| / max(R_obs, 1) ]

where:
    |S_b|   = rejection count in doubly-null replicate b  (FP by construction)
    R_obs   = rejection count on the real sub-log  (denominator / power reference)
    b       = 1, ..., B_null  (held-out permutation replicates)

Every |S_b| counts only true false positives: the double-null protocol
(sigma_label ∘ sigma_trace) destroys both discriminative and structural signal
simultaneously.  Therefore FDR_emp <= alpha is a valid empirical test of the
framework's FDR guarantee.

GRID DESIGN
-----------
Three rho levels span the natural balance to severe imbalance:

    rho = 0.595 (natural):  n_eff in {100, 200, 400, 800, 1600, 3200, 6400, 12800}  [8 cells]
    rho = 0.400:            n_eff in {100, 200, 400, 800, 1600, 3200, 6400}          [7 cells]
    rho = 0.200:            n_eff in {100, 200, 400, 800, 1600, 3200}                [6 cells]

Total: 21 feasible cells.

For each (n_eff, rho) pair:
    n       = n_eff / (2 * rho * (1 - rho))    (exact)
    n_pos   = round(n * rho),   n_neg = n - n_pos

Feasibility constraints:
    n_pos >= 50, n_neg >= 50              (Phipson-Smyth resolution floor)
    n_pos <= 18,747 (pool size),  n_neg <= 12,762 (pool size)

The binding constraint at high n_eff:
    rho = 0.595:  n_pos becomes binding above n_eff ≈ 15,200  →  max feasible 12,800
    rho = 0.400:  n_neg becomes binding above n_eff ≈ 10,200  →  max feasible 6,400
    rho = 0.200:  n_neg becomes binding above n_eff ≈ 5,100   →  max feasible 3,200

ISO-n_eff PAIRS (same n_eff, different rho): for n_eff in {800, 3200}, all three
rho values are feasible.  These pairs test whether FDR_emp depends purely on
n_eff or also on balance independently.

SUBSAMPLING PROTOCOL
--------------------
For each cell (n_k, rho_j) and replicate r in {1, ..., R}:
    1. Stratified SRS without replacement from the full case pool:
           C_sub = SRS_WOR(C_0, n_neg, seed) ∪ SRS_WOR(C_1, n_pos, seed)
    2. TV divergence check: TV(P_sub, P_full) < 0.10
       (activity marginal representativeness across BPI 2017 lifecycle namespaces)
    3. Recompute holds on the sub-log using the fixed candidate pool
       from the full-log run.

DOUBLE-NULL PROTOCOL (per sub-log)
-----------------------------------
Identical to rq1_BPI_17_parallel.py Section 6:
    sigma_trace:  shuffle activities within each trace  -> p_struct ~ U(0,1)
    sigma_label:  permute class labels                  -> p_disc   ~ U(0,1)
    -> T_Fisher = -2(ln p_struct + ln p_disc) ~ chi2(4) exactly

Under this joint null, every rejection is a false positive on both axes.

SURFACE ESTIMATION
------------------
After collecting FDR_emp at all 21 cells, fit a smooth cubic spline over
log(n_eff) per method.  Report:
    n*_eff(method) = min n_eff such that fitted FDR_emp(n_eff) <= alpha

This collapses the FDR surface to a single interpretable number per method.
The iso-n_eff diagnostic tests whether the 1D log(n_eff) summary is adequate.

COMPUTATIONAL BUDGET
--------------------
    R         = 10 sub-sampling replicates per cell
    B_null    = 100 doubly-null replicates per sub-log
    B1_sub    = 1,000 (label permutation budget per replicate)
    B2_sub    = 300  (structural permutation budget per replicate)

    Estimated wall time per inner replicate:
        n =    200: < 1 min
        n =  1,700: ~2 min
        n = 26,000: ~8 min

    Total (21 cells × 10 reps × 100 null): ~90–130 HPC hours (8-core node).
    Dominant cost: the 3 largest-n cells at rho = 0.595.

OUTPUT FILES
------------
    rq1_2_cell_results.csv      Per-cell FDR_emp, R_obs, pi0, m'' (Table data).
    rq1_2_null_counts.csv       All raw rejection counts (B_null rows per cell).
    rq1_2_surface_fit.csv       Fitted curve per method over n_eff grid.
    rq1_2_results.json          Full results for paper generation.
    rq1_2_diagnostics.csv       TV divergence, m'', pi0 per cell.

Version : 1.0
Author  : Ahmed Nour Abdesselam
Date    : April 2026
"""

import sys
import os
import copy
import io
import contextlib
import time
import json
import argparse
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ═══════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from Experiments.P1_SDSM.p1_BPI_17_parallel import (
    load_and_preprocess_data,
    generate_candidate_patterns,
    split_by_class,
    compute_prevalence_from_holds,
    compute_holds_by_case_batch,
    precompute_activity_index,
    run_label_permutation_test,
    run_structural_permutation_test,
    fisher_conjunction_pvalue,
    adaptive_storey_pi0,
    storey_qvalue,
    benjamini_hochberg,
    execute_pipeline,
    CaseInfo,
    PatternTestResult,
    CONFIG as P1_CONFIG,
    INPUT_FILE as P1_INPUT_FILE,
    OUTPUT_DIR as P1_OUTPUT_DIR,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
)

from eval_utils import (
    run_cecconi_baseline,
    run_tusher_flat_null,
)

# ── BCa bootstrap CI (self-contained) ─────────────────────────────────────

def _bca_ci(data: np.ndarray, alpha: float = 0.05, n_boot: int = 999,
            seed: int = 42) -> tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) bootstrap 95% CI for the mean.

    Efron & Tibshirani (1993), Algorithm 14.2.
    """
    n = len(data)
    if n < 2:
        m = float(np.mean(data))
        return m, m

    rng = np.random.RandomState(seed)
    theta_obs = float(np.mean(data))

    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    z0 = stats.norm.ppf(np.mean(boot_means < theta_obs) + 1e-10)

    jack_means = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jack_bar   = np.mean(jack_means)
    num = np.sum((jack_bar - jack_means) ** 3)
    den = 6.0 * (np.sum((jack_bar - jack_means) ** 2) ** 1.5)
    a   = num / den if abs(den) > 1e-15 else 0.0

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1.0 - alpha / 2)

    def adj_q(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))

    q_lo = np.clip(adj_q(z_lo), 0.001, 0.999)
    q_hi = np.clip(adj_q(z_hi), 0.001, 0.999)

    return float(np.quantile(boot_means, q_lo)), float(np.quantile(boot_means, q_hi))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH    = P1_INPUT_FILE
PHASE0_JSON = P1_SPEC_FILE
OUTPUT_DIR  = "RQ1_2_BPI17"

# Permutation budgets for sub-log runs
# B1=1000 → Phipson-Smyth resolution 1/1001 ≈ 1e-3, well below alpha=0.05
B1_SUB = 1_000
B2_SUB = 300

# Full-log pipeline budget (candidate generation + tf_null_matrix)
B1_FULL = 2_000
B2_FULL = 1_000

# Sub-sampling replicates per cell
R_REPS = 10

# Doubly-null held-out replicates per sub-log
B_NULL = 100

# FDR target level
ALPHA = 0.05

# RNG base seed (cell seeds derived from this)
BASE_SEED = 20260322

# Parallelism: outer loop over (cell, rep) pairs
N_JOBS = -1

# Activity marginal TV divergence threshold for sub-log validity check
TV_THRESHOLD = 0.10

# Minimum n+, n- for Phipson-Smyth resolution floor
N_MIN_CLASS = 50

# BPI Challenge 2017 pool sizes
# Class 1 (Not-Accepted / Deviant): 18,747 cases — the majority class
# Class 0 (Accepted    / Normal):   12,762 cases — the minority class
# Natural rho (fraction of class 1) ≈ 0.595
N_POS_POOL = 18_747   # class 1 (Not-Accepted / Deviant)
N_NEG_POOL = 12_762   # class 0 (Accepted    / Normal)

# Method name constants
METHOD_FISHER_STOREY = "Fisher-Storey"
METHOD_BH_FISHER     = "BH-Fisher"
METHOD_CECCONI       = "Cecconi_ChiSq_BH"
METHOD_TUSHER        = "Tusher_FlatNull"
ALL_METHODS          = [METHOD_FISHER_STOREY, METHOD_BH_FISHER, METHOD_CECCONI, METHOD_TUSHER]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — GRID DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

def build_design_grid(
    n_pos_pool: int = N_POS_POOL,
    n_neg_pool: int = N_NEG_POOL,
    n_min_class: int = N_MIN_CLASS,
) -> list[dict]:
    """
    Build the feasibility-constrained (n_eff, rho) design grid for BPI 2017.

    rho values:
        0.595 — natural fraction of class 1 (Not-Accepted) in the full log
        0.400 — moderate imbalance (class 1 minority relative to natural)
        0.200 — severe imbalance

    For each (n_eff, rho) pair:
        n     = n_eff / (2 * rho * (1 - rho))
        n_pos = round(n * rho),  n_neg = n - n_pos

    Feasibility:
        n_pos >= n_min_class, n_neg >= n_min_class
        n_pos <= n_pos_pool,  n_neg <= n_neg_pool

    Binding constraints (BPI 2017):
        rho = 0.595:  n_pos pool (18,747) limits n_eff to ≤ ~15,200  → max 12,800
        rho = 0.400:  n_neg pool (12,762) limits n_eff to ≤ ~10,200  → max  6,400
        rho = 0.200:  n_neg pool (12,762) limits n_eff to ≤  ~5,100  → max  3,200

    Returns:
        List of dicts with keys:
            cell_id, rho, neff_target, neff_actual, n, n_pos, n_neg
    """
    rho_values  = [0.595, 0.500	, 0.250, 0.100]
    # neff_levels = [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600]
    neff_levels = [100,200,400,800,1_600,3_200,6_400,12_800]

    grid = []
    for rho in rho_values:
        for neff in neff_levels:
            n_float = neff / (2.0 * rho * (1.0 - rho))
            n_pos   = int(round(n_float * rho))
            n_neg   = int(round(n_float * (1.0 - rho)))
            n_total = n_pos + n_neg
            neff_actual = 2.0 * n_pos * n_neg / n_total if n_total > 0 else 0.0

            feasible = (
                n_pos >= n_min_class and n_neg >= n_min_class
                and n_pos <= n_pos_pool and n_neg <= n_neg_pool
            )
            if feasible:
                cell_id = f"neff{neff:05d}_rho{rho:.3f}"
                grid.append({
                    'cell_id':     cell_id,
                    'rho':         rho,
                    'neff_target': neff,
                    'neff_actual': round(neff_actual, 2),
                    'n':           n_total,
                    'n_pos':       n_pos,
                    'n_neg':       n_neg,
                })

    return grid


def print_grid(grid: list[dict]) -> None:
    print(f"\n  Design grid ({len(grid)} feasible cells):")
    print(f"  {'cell_id':32s} {'n':>8s} {'n+':>7s} {'n-':>7s} {'n_eff':>8s}")
    print(f"  {'─'*67}")
    for cell in grid:
        print(f"  {cell['cell_id']:32s} {cell['n']:>8d} {cell['n_pos']:>7d} "
              f"{cell['n_neg']:>7d} {cell['neff_actual']:>8.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FULL-LOG PIPELINE  (run once; candidates_all fixed for all cells)
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(n_workers: int = 1) -> dict:
    """
    Run p1_BPI_17_parallel's execute_pipeline on the complete log.

    Two purposes:
        1. Establish the fixed candidate pool (candidates_all) for all cells.
           Using a fixed pool ensures pattern prevalences are comparable
           across sub-logs of different (n, rho).
        2. Obtain tf_null_matrix for reference (not applied to sub-logs;
           sub-log replicates use analytic chi2_4 — oracle under double-null).

    Returns:
        dict: case_data, candidates_all, case_ids_sorted, labels,
              tf_null_matrix, activity_pmf, wall_seconds.
    """
    print("\n" + "=" * 90)
    print("SECTION 2: FULL-LOG PIPELINE (candidate generation + BPI 2017 case pool)")
    print(f"  B1={B1_FULL}, B2={B2_FULL}, alpha={ALPHA}, n_workers={n_workers}")
    print("=" * 90)

    t0 = time.time()
    cfg = P1_CONFIG.copy()
    cfg['B_label']      = B1_FULL
    cfg['B_trace']      = B2_FULL
    cfg['fdr_alpha']    = ALPHA
    cfg['random_state'] = 42
    cfg['n_workers']    = n_workers

    output = execute_pipeline(input_file=CSV_PATH, config=cfg)

    case_data       = output['case_data']
    candidates_all  = output['candidates_all']
    tf_null_mat     = output['tf_null_matrix']
    case_ids_sorted = sorted(case_data.keys())
    labels = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    # Activity marginal distribution for TV divergence checks
    activity_counts: dict[str, int] = {}
    for ci in case_data.values():
        for act in ci.trace:
            activity_counts[act] = activity_counts.get(act, 0) + 1
    total_events = sum(activity_counts.values())
    activity_pmf = {a: c / total_events for a, c in activity_counts.items()}

    wall = time.time() - t0
    n, n1 = len(labels), int(labels.sum())
    print(f"\n  Pipeline complete:")
    print(f"    n = {n:,}  (n1/Not-Accepted = {n1:,},  n0/Accepted = {n - n1:,})")
    print(f"    Candidates: {len(candidates_all):,}  |  Activities: {len(activity_pmf)}")
    print(f"    Wall: {wall:.1f}s")

    return {
        'case_data':        case_data,
        'candidates_all':   candidates_all,
        'case_ids_sorted':  case_ids_sorted,
        'labels':           labels,
        'tf_null_matrix':   tf_null_mat,
        'activity_pmf':     activity_pmf,
        'wall_seconds':     wall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — STRATIFIED SUBSAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def stratified_subsample(
    case_ids_pos: list,
    case_ids_neg: list,
    n_pos: int,
    n_neg: int,
    seed: int,
) -> list:
    """
    Draw a stratified subsample of size (n_pos + n_neg) without replacement.

    Sampling is done independently within each class to preserve the
    requested rho = n_pos / (n_pos + n_neg).

    Raises:
        ValueError if n_pos > pool or n_neg > pool.
    """
    if n_pos > len(case_ids_pos):
        raise ValueError(f"Requested n_pos={n_pos} > pool size {len(case_ids_pos)}")
    if n_neg > len(case_ids_neg):
        raise ValueError(f"Requested n_neg={n_neg} > pool size {len(case_ids_neg)}")

    rng     = np.random.RandomState(seed)
    arr_pos = np.array(case_ids_pos)
    arr_neg = np.array(case_ids_neg)

    idx_pos = rng.choice(len(arr_pos), size=n_pos, replace=False)
    idx_neg = rng.choice(len(arr_neg), size=n_neg, replace=False)

    return list(arr_pos[idx_pos]) + list(arr_neg[idx_neg])


def compute_tv_divergence(
    sub_case_ids: list,
    case_data: dict,
    activity_pmf_full: dict[str, float],
) -> float:
    """
    TV distance between sub-log and full-log activity marginal distributions.

    TV(P_sub, P_full) = 0.5 * Σ_a |P_sub(a) - P_full(a)|

    Captures representativeness across the three BPI 2017 activity namespaces
    (A_ application lifecycle, O_ offer lifecycle, W_ workflow items).
    """
    counts_sub: dict[str, int] = {}
    for cid in sub_case_ids:
        for act in case_data[cid].trace:
            counts_sub[act] = counts_sub.get(act, 0) + 1

    total = sum(counts_sub.values())
    if total == 0:
        return 1.0

    pmf_sub = {a: c / total for a, c in counts_sub.items()}
    all_acts = set(activity_pmf_full) | set(pmf_sub)
    return 0.5 * sum(
        abs(pmf_sub.get(a, 0.0) - activity_pmf_full.get(a, 0.0))
        for a in all_acts
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPER FUNCTIONS (shared across real and null runs)
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout/stderr during inner permutation runs."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def _build_doubly_nullified_sublog(
    sub_case_data: dict,
    sub_case_ids_sorted: list,
    permuted_labels: np.ndarray,
    random_state: int,
) -> dict:
    """
    Apply sigma_trace ∘ sigma_label to a sub-log dict.

    sigma_trace:  Shuffle activities within each trace.
                  Preserves trace length and activity multiset.
                  Destroys temporal ordering → p_struct ~ U(0,1).

    sigma_label:  Override outcomes with permuted_labels.
                  Preserves marginal class counts.
                  Destroys class-activity association → p_disc ~ U(0,1).

    The original sub_case_data is never mutated.
    """
    rng = np.random.RandomState(random_state)
    nullified = {}
    for i, cid in enumerate(sub_case_ids_sorted):
        ci_orig = sub_case_data[cid]
        ci = copy.copy(ci_orig)
        ci.outcome = int(permuted_labels[i])
        shuffled_trace = ci_orig.trace.copy()
        rng.shuffle(shuffled_trace)
        ci.trace = shuffled_trace
        ci.activity_index = precompute_activity_index(shuffled_trace, case_id=cid)
        nullified[cid] = ci
    return nullified


def _run_analysis_on_case_data(
    case_data_sub: dict,
    candidates_all: list,
    B1: int,
    B2: int,
    alpha: float,
    random_state: int,
) -> dict:
    """
    Run the full four-method analysis on a case_data dict.

    Faithfully reproduces execute_three_hypothesis_protocol from
    p1_BPI_17_parallel.py v8.0 at the given (n, rho) budget:

        Step 1:  Recompute holds on the provided case_data.
        Step 2:  Label permutation test (H0d) → p_disc, null_delta_matrix.
        Step 3:  Structural permutation test (H0s) → p_struct screen/test per class.
        Step 4:  Dominant class from sub-log prevalences.
        Step 5:  Fisher conjunction  T_F = -2(ln p_struct_dom + ln p_disc).
                 Analytic chi2_4 p-value used throughout sub-log runs.
                 Under the double-null, analytic chi2_4 IS the oracle p-value.
                 Under real sub-log distributions, analytic chi2_4 is slightly
                 anti-conservative (Brown 1975), making FDR_emp estimates
                 slightly liberal — a safe direction for the FDR control claim.
        Step 6:  Structural scope filter m'' = {i : min(p_screen_c0, p_screen_c1) ≤ α}.
        Step 7:  Fisher-Storey: adaptive_storey_pi0 → storey_qvalue.
                 Gate: q_Fisher ≤ α  AND  p_struct_dom_test ≤ α  ("Both").
        Step 8:  BH-Fisher on m'' analytic chi2_4 p-values.
        Step 9:  Cecconi chi-square + BH on holds.
        Step 10: Tusher flat-null SAM on null_delta_matrix.

    Returns:
        dict with method_counts, m_prime, pi0_disc, delta_obs,
        null_delta_matrix, p_disc, p_struct_dom_test.
    """
    m  = len(candidates_all)
    rs = random_state

    # Step 1: holds
    with _suppress_output():
        holds = compute_holds_by_case_batch(case_data_sub, candidates_all)

    # Step 2: discriminative test
    with _suppress_output():
        disc_results = run_label_permutation_test(
            case_data_sub, candidates_all, holds, B1, rs,
        )
    null_delta_mat = disc_results.pop('__null_delta_matrix__')
    p_disc    = np.array([disc_results[spec]['p_two_sided'] for spec in candidates_all])
    delta_obs = np.array([disc_results[spec]['delta_obs']   for spec in candidates_all])

    # Step 3: structural tests on both classes
    D_0, D_1 = split_by_class(case_data_sub)
    cid_set_0, cid_set_1 = set(D_0.keys()), set(D_1.keys())

    with _suppress_output():
        struct_0 = run_structural_permutation_test(
            D_0, candidates_all, class_label=0, B2=B2,
            random_state=rs + 1, n_workers=1,
        )
        struct_1 = run_structural_permutation_test(
            D_1, candidates_all, class_label=1, B2=B2,
            random_state=rs + 2, n_workers=1,
        )

    p_struct_screen_c0 = np.array([
        struct_0[spec]['p_structural_screen'] if spec in struct_0 else 1.0
        for spec in candidates_all
    ])
    p_struct_screen_c1 = np.array([
        struct_1[spec]['p_structural_screen'] if spec in struct_1 else 1.0
        for spec in candidates_all
    ])
    p_struct_test_c0 = np.array([
        struct_0[spec]['p_structural_test'] if spec in struct_0 else 1.0
        for spec in candidates_all
    ])
    p_struct_test_c1 = np.array([
        struct_1[spec]['p_structural_test'] if spec in struct_1 else 1.0
        for spec in candidates_all
    ])

    # Step 4: dominant class from sub-log prevalences
    prev0, prev1 = np.zeros(m), np.zeros(m)
    for i, spec in enumerate(candidates_all):
        h = holds[spec]
        p0, _, _ = compute_prevalence_from_holds(h, cid_set_0)
        p1, _, _ = compute_prevalence_from_holds(h, cid_set_1)
        prev0[i], prev1[i] = p0, p1
    dominant = np.where(prev1 >= prev0, 1, 0)
    p_struct_dom_test = np.where(dominant == 1, p_struct_test_c1, p_struct_test_c0)

    # Step 5: Fisher conjunction (analytic chi2_4 — oracle under double-null)
    _eps = 1e-300
    _ps  = np.clip(p_struct_dom_test, _eps, 1.0)
    _pd  = np.clip(p_disc,            _eps, 1.0)
    tf   = -2.0 * (np.log(_ps) + np.log(_pd))
    p_fisher = stats.chi2.sf(tf, df=4)

    # Step 6: structural scope filter (sample-split screen p-values)
    structural_idx = [
        i for i in range(m)
        if min(p_struct_screen_c0[i], p_struct_screen_c1[i]) <= alpha
    ]
    m_prime = len(structural_idx)
    p_fisher_mp = p_fisher[structural_idx] if m_prime > 0 else np.array([])

    # Step 7: Fisher-Storey — conjunctive gate (q_Fisher AND p_struct_dom)
    if m_prime > 0:
        pi0_f, _ = adaptive_storey_pi0(p_fisher_mp, q=alpha)
        q_fisher  = storey_qvalue(p_fisher_mp, pi0_f)
        p_struct_mp = p_struct_dom_test[structural_idx]
        n_fs = int(np.sum((q_fisher <= alpha) & (p_struct_mp <= alpha)))
    else:
        pi0_f, n_fs = 1.0, 0

    # Step 8: BH-Fisher on m'' analytic chi2_4 p-values
    if m_prime > 0:
        rej_bh, _, _ = benjamini_hochberg(p_fisher_mp, alpha)
        n_bh = int(np.sum(rej_bh))
    else:
        n_bh = 0

    # Step 9: Cecconi chi-square + BH
    cecconi_res = run_cecconi_baseline(holds, cid_set_0, cid_set_1, alpha)
    n_cecconi   = cecconi_res.n_rejected

    # Step 10: Tusher flat-null SAM
    tusher_res = run_tusher_flat_null(null_delta_mat, delta_obs, alpha, pi0_hat=1.0)
    n_tusher   = tusher_res['k_star']

    # Discriminative pi0 for diagnostics
    pi0_disc_diag = 1.0
    if len(p_disc) > 10:
        try:
            pi0_disc_diag, _ = adaptive_storey_pi0(p_disc, q=alpha)
        except Exception:
            pass

    return {
        'method_counts': {
            METHOD_FISHER_STOREY: n_fs,
            METHOD_BH_FISHER:     n_bh,
            METHOD_CECCONI:       n_cecconi,
            METHOD_TUSHER:        n_tusher,
        },
        'm_prime':           m_prime,
        'pi0_disc':          pi0_disc_diag,
        'delta_obs':         delta_obs,
        'null_delta_matrix': null_delta_mat,
        'p_disc':            p_disc,
        'p_struct_dom_test': p_struct_dom_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — CELL WORKER
# ═══════════════════════════════════════════════════════════════════════════

def run_cell_replicate(
    cell: dict,
    rep_idx: int,
    case_data: dict,
    candidates_all: list,
    case_ids_pos: list,
    case_ids_neg: list,
    activity_pmf_full: dict,
) -> dict:
    """
    Execute one (cell, replicate) unit of work for BPI Challenge 2017.

    Workflow:
        1. Stratified subsample (n_pos Not-Accepted + n_neg Accepted).
        2. TV divergence check across A_/O_/W_ activity namespaces.
        3. Real sub-log run → R_obs per method.
        4. B_null doubly-null replicates → |S_b| per method.
        5. FDR_emp per method = mean(|S_b|) / max(R_obs, 1).

    Seed architecture (three independent layers, non-overlapping offset bands):
        _slot + 0        subsampling
        _slot + 1000     real pipeline
        _slot + 2000+b   held-out label permutations (b = 0..B_NULL-1)
        _slot + 10000+b*1000  doubly-null inner analysis

    Returns:
        dict with cell metadata, tv_divergence, R_obs, null_counts, fdr_emp,
        m_prime_real, pi0_disc_real.
    """
    cell_id = cell['cell_id']
    n_pos   = cell['n_pos']
    n_neg   = cell['n_neg']

    cell_ord = abs(hash(cell_id)) % 10_000
    _slot = BASE_SEED + cell_ord * 10_000_000 + rep_idx * 200_000

    # ── Step 1: stratified subsample ─────────────────────────────────────
    sub_ids = stratified_subsample(
        case_ids_pos, case_ids_neg, n_pos, n_neg, seed=_slot
    )
    sub_case_data       = {cid: case_data[cid] for cid in sub_ids}
    sub_case_ids_sorted = sorted(sub_ids)
    sub_labels = np.array([sub_case_data[cid].outcome for cid in sub_case_ids_sorted])

    # ── Step 2: TV divergence check ──────────────────────────────────────
    tv         = compute_tv_divergence(sub_ids, case_data, activity_pmf_full)
    tv_flagged = (tv >= TV_THRESHOLD)

    # ── Step 3: real sub-log → R_obs ─────────────────────────────────────
    real_out = _run_analysis_on_case_data(
        sub_case_data, candidates_all,
        B1=B1_SUB, B2=B2_SUB, alpha=ALPHA, random_state=_slot + 1_000,
    )
    R_obs         = real_out['method_counts']
    m_prime_real  = real_out['m_prime']
    pi0_disc_real = real_out['pi0_disc']

    # ── Step 4: B_null doubly-null replicates → |S_b| ────────────────────
    null_counts = {m_name: np.zeros(B_NULL, dtype=int) for m_name in ALL_METHODS}

    for b in range(B_NULL):
        perm_rng    = np.random.RandomState(_slot + 2_000 + b)
        perm_labels = perm_rng.permutation(sub_labels)

        null_rs = _slot + 10_000 + b * 1_000
        null_cd = _build_doubly_nullified_sublog(
            sub_case_data, sub_case_ids_sorted, perm_labels,
            random_state=null_rs + 500,
        )
        null_out = _run_analysis_on_case_data(
            null_cd, candidates_all,
            B1=B1_SUB, B2=B2_SUB, alpha=ALPHA, random_state=null_rs,
        )
        for m_name in ALL_METHODS:
            null_counts[m_name][b] = null_out['method_counts'][m_name]

    # ── Step 5: FDR_emp per method ────────────────────────────────────────
    fdr_emp = {
        m_name: float(np.mean(null_counts[m_name]) / max(R_obs[m_name], 1))
        for m_name in ALL_METHODS
    }

    return {
        'cell_id':       cell_id,
        'rho':           cell['rho'],
        'neff_target':   cell['neff_target'],
        'neff_actual':   cell['neff_actual'],
        'n':             cell['n'],
        'n_pos':         n_pos,
        'n_neg':         n_neg,
        'rep_idx':       rep_idx,
        'tv_divergence': tv,
        'tv_flagged':    tv_flagged,
        'R_obs':         R_obs,
        'null_counts':   null_counts,
        'fdr_emp':       fdr_emp,
        'm_prime_real':  m_prime_real,
        'pi0_disc_real': pi0_disc_real,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PARALLEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def _cell_rep_worker(cell, rep_idx,
                     case_data, candidates_all,
                     case_ids_pos, case_ids_neg,
                     activity_pmf_full):
    """Joblib top-level worker: one cell × one replicate."""
    return run_cell_replicate(
        cell, rep_idx, case_data, candidates_all,
        case_ids_pos, case_ids_neg, activity_pmf_full,
    )


def run_surface_experiment(
    grid: list[dict],
    case_data: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels: np.ndarray,
    activity_pmf: dict,
    n_jobs: int = N_JOBS,
) -> list[dict]:
    """
    Run all (cell, replicate) pairs in parallel.

    21 cells × R_REPS replicates = 210 tasks.
    n_workers=1 is enforced inside _run_analysis_on_case_data to avoid
    nested loky process spawning.

    Returns:
        List of result dicts, one per (cell, rep) pair.
    """
    print("\n" + "=" * 90)
    print("SECTION 6: PARALLEL SURFACE EXPERIMENT  (BPI Challenge 2017)")
    print(f"  {len(grid)} cells × {R_REPS} reps = {len(grid) * R_REPS} tasks")
    print(f"  B_null={B_NULL}, B1_sub={B1_SUB}, B2_sub={B2_SUB}")
    print(f"  n_jobs={n_jobs}")
    print("=" * 90)

    case_ids_pos = [cid for cid in case_ids_sorted if case_data[cid].outcome == 1]
    case_ids_neg = [cid for cid in case_ids_sorted if case_data[cid].outcome == 0]
    print(f"\n  Pool: n_pos (Not-Accepted)={len(case_ids_pos):,},  "
          f"n_neg (Accepted)={len(case_ids_neg):,}")

    tasks = [(cell, rep) for cell in grid for rep in range(R_REPS)]
    print(f"  Total tasks: {len(tasks)}")

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5, backend='loky')(
        delayed(_cell_rep_worker)(
            cell, rep, case_data, candidates_all,
            case_ids_pos, case_ids_neg, activity_pmf,
        )
        for cell, rep in tasks
    )
    wall = time.time() - t0
    print(f"\n  All tasks complete.  Wall: {wall:.1f}s ({wall/3600:.2f}h)")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — AGGREGATE RESULTS PER CELL
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_cell_results(all_results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-(cell, rep) results into per-cell summaries.

    For each cell (averaging over R_REPS replicates):
        FDR_emp_mean  = pooled estimator E[S_bar] / E[R_obs]
                        (avoids Jensen's inequality bias from E[S_bar/R_obs])
        FDR_emp_std   = std of per-replicate FDR_emp values
        FDR_CI_lo/hi  = BCa 95% CI across reps
        R_obs_mean    = mean R_obs
        m_prime_mean  = mean m''
        pi0_disc_mean = mean pi0 (discriminative axis)
        tv_mean       = mean TV divergence
        tv_flag_any   = True if any rep had TV ≥ threshold

    Returns:
        DataFrame with one row per (cell_id, method) combination.
    """
    from collections import defaultdict

    cell_groups: dict[str, list[dict]] = defaultdict(list)
    for res in all_results:
        cell_groups[res['cell_id']].append(res)

    rows = []
    for cell_id, reps in cell_groups.items():
        r0   = reps[0]
        base = {
            'cell_id':      cell_id,
            'rho':          r0['rho'],
            'neff_target':  r0['neff_target'],
            'neff_actual':  r0['neff_actual'],
            'n':            r0['n'],
            'n_pos':        r0['n_pos'],
            'n_neg':        r0['n_neg'],
            'tv_mean':      float(np.mean([r['tv_divergence'] for r in reps])),
            'tv_flag_any':  any(r['tv_flagged'] for r in reps),
            'm_prime_mean': float(np.mean([r['m_prime_real']  for r in reps])),
            'pi0_disc_mean': float(np.mean([r['pi0_disc_real'] for r in reps])),
            'n_reps':       len(reps),
        }

        for m_name in ALL_METHODS:
            s_bar_vals = np.array([np.mean(r['null_counts'][m_name]) for r in reps])
            r_obs_vals = np.array([r['R_obs'][m_name]                for r in reps])
            fdr_vals   = np.array([r['fdr_emp'][m_name]              for r in reps])

            fdr_pooled = float(np.mean(s_bar_vals)) / max(float(np.mean(r_obs_vals)), 1.0)
            ci_lo, ci_hi = _bca_ci(fdr_vals, alpha=0.05, n_boot=999, seed=42)

            rows.append({
                **base,
                'method':       m_name,
                'fdr_emp_mean': fdr_pooled,
                'fdr_emp_std':  float(np.std(fdr_vals, ddof=1)),
                'fdr_ci_lo':    float(ci_lo),
                'fdr_ci_hi':    float(ci_hi),
                'R_obs_mean':   float(np.mean(r_obs_vals)),
                'R_obs_std':    float(np.std(r_obs_vals, ddof=1)),
                'controls_fdr': fdr_pooled <= ALPHA,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — SURFACE ESTIMATION AND ROOT FINDING
# ═══════════════════════════════════════════════════════════════════════════

def fit_fdr_surface(agg_df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """
    Fit a smooth cubic spline to FDR_emp over log10(n_eff) per method.

    1D spline: average FDR_emp across rho values per n_eff level, then
    fit UnivariateSpline(log10(n_eff_actual), FDR_emp_mean).

    Root n*_eff = min n_eff s.t. fitted FDR_emp(n_eff) ≤ alpha, found
    via scipy.optimize.brentq.

    Iso-n_eff diagnostic: compare FDR_emp across rho values for
    n_eff ∈ {800, 3200} where all three rho levels are feasible.

    Returns:
        dict with surface_df, nstar, nstar_ci, iso_neff_df.
    """
    print("\n" + "=" * 90)
    print("SECTION 8: SURFACE ESTIMATION AND ROOT FINDING")
    print("=" * 90)

    surface_rows = []
    nstar    = {}
    nstar_ci = {}

    neff_min = agg_df['neff_actual'].min()
    neff_max = agg_df['neff_actual'].max()
    log_grid = np.linspace(np.log10(neff_min), np.log10(neff_max), 500)
    neff_dense = 10.0 ** log_grid

    for m_name in ALL_METHODS:
        sub = agg_df[agg_df['method'] == m_name].copy()

        sub_avg = (sub.groupby('neff_target')
                      .agg(fdr_emp_mean=('fdr_emp_mean', 'mean'),
                           neff_actual  =('neff_actual',  'mean'))
                      .reset_index()
                      .sort_values('neff_actual'))

        x = np.log10(sub_avg['neff_actual'].values.astype(float))
        y = sub_avg['fdr_emp_mean'].values.astype(float)

        if len(x) < 4:
            print(f"  WARNING: {m_name} has only {len(x)} data points — skipping spline.")
            nstar[m_name]    = np.nan
            nstar_ci[m_name] = (np.nan, np.nan)
            continue

        try:
            spline = UnivariateSpline(x, y, k=3, s=len(x) * 0.1, ext=3)
        except Exception as e:
            print(f"  WARNING: spline fit failed for {m_name}: {e}")
            nstar[m_name]    = np.nan
            nstar_ci[m_name] = (np.nan, np.nan)
            continue

        y_fit = spline(log_grid)
        for neff_v, y_v in zip(neff_dense, y_fit):
            surface_rows.append({'method': m_name, 'neff': neff_v, 'fdr_fit': float(y_v)})

        f    = lambda lx: float(spline(lx)) - alpha
        y_lo = float(spline(log_grid[0]))
        y_hi = float(spline(log_grid[-1]))

        if y_hi > alpha:
            nstar[m_name]    = np.inf
            nstar_ci[m_name] = (np.inf, np.inf)
            print(f"  {m_name:25s}: FDR never ≤ alpha in observed range.")
        elif y_lo <= alpha:
            nstar[m_name]    = 10.0 ** log_grid[0]
            nstar_ci[m_name] = (10.0 ** log_grid[0], 10.0 ** log_grid[0])
            print(f"  {m_name:25s}: FDR controlled from n_eff={nstar[m_name]:.0f}.")
        else:
            try:
                lx_star = brentq(f, log_grid[0], log_grid[-1], xtol=1e-4)
                nstar[m_name] = 10.0 ** lx_star
            except ValueError:
                nstar[m_name] = np.nan

            sub_std_avg = (agg_df[agg_df['method'] == m_name]
                           .groupby('neff_target')
                           .agg(fdr_emp_std=('fdr_emp_std', 'mean'),
                                neff_actual=('neff_actual', 'mean'))
                           .reset_index()
                           .sort_values('neff_actual'))
            se_vals  = sub_std_avg['fdr_emp_std'].values / np.sqrt(R_REPS)
            y_lo_ci  = np.clip(y - se_vals, 0, 1)
            y_hi_ci  = np.clip(y + se_vals, 0, 1)
            try:
                sp_lo = UnivariateSpline(x, y_lo_ci, k=3, s=len(x)*0.1, ext=3)
                sp_hi = UnivariateSpline(x, y_hi_ci, k=3, s=len(x)*0.1, ext=3)
                def _root_or_nan(sp):
                    if float(sp(log_grid[0])) > alpha and float(sp(log_grid[-1])) < alpha:
                        try:
                            return brentq(lambda lx: float(sp(lx)) - alpha,
                                          log_grid[0], log_grid[-1])
                        except ValueError:
                            return np.nan
                    return np.nan
                lo_r = _root_or_nan(sp_lo)
                hi_r = _root_or_nan(sp_hi)
                nstar_ci[m_name] = (
                    10.0**lo_r if not np.isnan(lo_r) else np.nan,
                    10.0**hi_r if not np.isnan(hi_r) else np.nan,
                )
            except Exception:
                nstar_ci[m_name] = (np.nan, np.nan)

            print(f"  {m_name:25s}: n*_eff = {nstar[m_name]:.1f}  "
                  f"CI=[{nstar_ci[m_name][0]:.1f}, {nstar_ci[m_name][1]:.1f}]")

    surface_df = pd.DataFrame(surface_rows)

    # Iso-n_eff diagnostic: n_eff ∈ {800, 3200} are feasible at all 3 rho levels
    iso_cells = agg_df[agg_df['neff_target'].isin([800, 3200])].copy()
    print(f"\n  Iso-n_eff diagnostic  (n_eff ∈ {{800, 3200}}, all rho, BPI 2017):")
    for m_name in ALL_METHODS:
        sub_iso = iso_cells[iso_cells['method'] == m_name][
            ['neff_target', 'rho', 'fdr_emp_mean', 'fdr_emp_std']
        ].sort_values(['neff_target', 'rho'])
        print(f"\n    {m_name}:")
        print(sub_iso.to_string(index=False))

    return {
        'surface_df':  surface_df,
        'nstar':       nstar,
        'nstar_ci':    nstar_ci,
        'iso_neff_df': iso_cells,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_results(
    agg_df: pd.DataFrame,
    surface_dict: dict,
    all_results: list[dict],
    full_wall: float,
    exp_wall: float,
    grid: list[dict],
) -> None:
    """
    Save all output files for the paper (BPI Challenge 2017 version).

    FILE MANIFEST:
        rq1_2_cell_results.csv   — per-cell FDR_emp, R_obs, diagnostics (Table S1)
        rq1_2_null_counts.csv    — raw |S_b| for every (cell, rep) pair
        rq1_2_surface_fit.csv    — fitted FDR surface on dense n_eff grid (Figure 1)
        rq1_2_results.json       — full machine-readable results
        rq1_2_diagnostics.csv    — TV divergence, m'', pi0 per cell (Figure S1)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FILE 1: per-cell summary
    cell_path = os.path.join(OUTPUT_DIR, "rq1_2_cell_results.csv")
    agg_df.to_csv(cell_path, index=False)
    print(f"\n  Saved: {cell_path}")

    # FILE 2: raw null counts
    raw_rows = []
    for res in all_results:
        for b in range(B_NULL):
            row = {'cell_id': res['cell_id'], 'rep_idx': res['rep_idx'], 'b': b}
            for m_name in ALL_METHODS:
                row[m_name] = int(res['null_counts'][m_name][b])
            raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)
    null_path = os.path.join(OUTPUT_DIR, "rq1_2_null_counts.csv")
    raw_df.to_csv(null_path, index=False)
    print(f"  Saved: {null_path}")

    # FILE 3: surface fit
    surf_path = os.path.join(OUTPUT_DIR, "rq1_2_surface_fit.csv")
    surface_dict['surface_df'].to_csv(surf_path, index=False)
    print(f"  Saved: {surf_path}")

    # FILE 4: diagnostics
    diag_rows = []
    for res in all_results:
        diag_rows.append({
            'cell_id':       res['cell_id'],
            'rep_idx':       res['rep_idx'],
            'rho':           res['rho'],
            'neff_target':   res['neff_target'],
            'n':             res['n'],
            'tv_divergence': res['tv_divergence'],
            'tv_flagged':    res['tv_flagged'],
            'm_prime_real':  res['m_prime_real'],
            'pi0_disc_real': res['pi0_disc_real'],
        })
    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(OUTPUT_DIR, "rq1_2_diagnostics.csv")
    diag_df.to_csv(diag_path, index=False)
    print(f"  Saved: {diag_path}")

    # FILE 5: JSON
    json_out = {
        'experiment': 'RQ1.2 — FDR Surface over (n, rho): BPI Challenge 2017',
        'timestamp':  datetime.now().isoformat(),
        'dataset': {
            'name':          'BPI Challenge 2017',
            'n_total':       N_POS_POOL + N_NEG_POOL,
            'n_pos_pool':    N_POS_POOL,
            'n_neg_pool':    N_NEG_POOL,
            'class_1_label': 'Not-Accepted (Deviant)',
            'class_0_label': 'Accepted (Normal)',
            'rho_natural':   round(N_POS_POOL / (N_POS_POOL + N_NEG_POOL), 4),
            'ir_natural':    round(max(N_POS_POOL, N_NEG_POOL) /
                                   min(N_POS_POOL, N_NEG_POOL), 4),
            'label_source':  'Teinemaa et al. TKDE 2019 — bpic2017_accepted',
        },
        'config': {
            'R_REPS':       R_REPS,
            'B_NULL':       B_NULL,
            'B1_SUB':       B1_SUB,
            'B2_SUB':       B2_SUB,
            'ALPHA':        ALPHA,
            'BASE_SEED':    BASE_SEED,
            'N_POS_POOL':   N_POS_POOL,
            'N_NEG_POOL':   N_NEG_POOL,
            'TV_THRESHOLD': TV_THRESHOLD,
            'n_cells':      len(grid),
        },
        'grid': grid,
        'n_star': {
            m: float(v) if not np.isnan(v) and not np.isinf(v) else str(v)
            for m, v in surface_dict['nstar'].items()
        },
        'n_star_ci': {
            m: [
                float(v[0]) if not np.isnan(v[0]) else None,
                float(v[1]) if not np.isnan(v[1]) else None,
            ]
            for m, v in surface_dict['nstar_ci'].items()
        },
        'cell_summary': agg_df.to_dict(orient='records'),
        'timing': {
            'full_pipeline_seconds': full_wall,
            'experiment_seconds':    exp_wall,
            'total_seconds':         full_wall + exp_wall,
        },
        'method_comparison': {
            m: {
                'n_star': float(surface_dict['nstar'].get(m, float('nan'))),
                'mean_fdr_all_cells': float(
                    agg_df[agg_df['method'] == m]['fdr_emp_mean'].mean()
                ),
                'n_cells_controlled': int(
                    (agg_df[agg_df['method'] == m]['fdr_emp_mean'] <= ALPHA).sum()
                ),
                'n_cells_total': int(len(agg_df[agg_df['method'] == m])),
            }
            for m in ALL_METHODS
        },
        'double_null_protocol': {
            'sigma_trace': (
                'Within each trace, randomly permute the activity sequence. '
                'Preserves trace length and activity multiset. '
                'Destroys temporal ordering → p_struct ~ U(0,1).'
            ),
            'sigma_label': (
                'Permute class labels across sub-log cases, preserving marginals. '
                'Destroys class-activity association → p_disc ~ U(0,1).'
            ),
            'joint_null': (
                'T_Fisher = -2(ln p_struct + ln p_disc) ~ chi2(4) exactly. '
                'Every rejection in a doubly-nullified replicate is a false positive.'
            ),
            'analytic_vs_empirical': (
                'Sub-log runs use analytic chi2_4 (oracle under double-null). '
                'Full-log tf_null_matrix is not applicable to sub-logs (different n, rho). '
                'Analytic chi2_4 is slightly anti-conservative under real data (Brown 1975), '
                'making FDR_emp estimates slightly liberal — safe for the FDR control claim.'
            ),
        },
        'iso_neff_claim': (
            'If FDR_emp depends purely on n_eff and not on (n, rho) independently, '
            'the iso-n_eff pairs (n_eff ∈ {800, 3200}) should produce equal FDR_emp '
            'within confidence interval width. See rq1_2_cell_results.csv.'
        ),
        'grid_notes': {
            'binding_constraint_rho_0.595': (
                'n_pos (class 1) pool = 18,747 limits n_eff ≤ ~15,200 → max feasible 12,800'
            ),
            'binding_constraint_rho_0.400': (
                'n_neg (class 0) pool = 12,762 limits n_eff ≤ ~10,200 → max feasible 6,400'
            ),
            'binding_constraint_rho_0.200': (
                'n_neg (class 0) pool = 12,762 limits n_eff ≤ ~5,100  → max feasible 3,200'
            ),
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "rq1_2_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# Colorblind-friendly palette (Wong 2011)
_M_COLORS = {
    METHOD_FISHER_STOREY: '#0072B2',
    METHOD_BH_FISHER:     '#009E73',
    METHOD_CECCONI:       '#D55E00',
    METHOD_TUSHER:        '#CC79A7',
}
_M_MARKERS = {
    METHOD_FISHER_STOREY: 'o',
    METHOD_BH_FISHER:     's',
    METHOD_CECCONI:       '^',
    METHOD_TUSHER:        'D',
}
_M_LABELS = {
    METHOD_FISHER_STOREY: 'Fisher-Storey',
    METHOD_BH_FISHER:     'BH-Fisher',
    METHOD_CECCONI:       'Cecconi χ²+BH',
    METHOD_TUSHER:        'Tusher Flat-Null',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def _save_fig(fig, name: str, out_dir: str = OUTPUT_DIR) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_figures(
    agg_df: pd.DataFrame,
    surface_dict: dict,
    out_dir: str = OUTPUT_DIR,
) -> None:
    """
    Generate three publication figures for RQ1.2 (BPI Challenge 2017).

    Figure 1 (rq1_2_fig1_fdr_surface.pdf):
        Smooth FDR_emp vs. log(n_eff) per method with BCa ribbons,
        α reference line, and vertical n*_eff markers.

    Figure 2 (rq1_2_fig2_heatmap.pdf):
        FDR_emp on the (log n, ρ) design space per method (2×2 subplots),
        colored by FDR level, α=0.05 iso-contour, BPI 2017 natural point.

    Figure 3 (rq1_2_fig3_fdr_power.pdf):
        Dual-axis: FDR_emp (left) and R_obs (right) vs. log(n_eff) per
        method (2×2 subplots), showing the FDR–power trade-off.
    """
    os.makedirs(out_dir, exist_ok=True)
    surface_df = surface_dict['surface_df']
    nstar      = surface_dict['nstar']

    def _avg_by_neff(method_name: str) -> pd.DataFrame:
        return (
            agg_df[agg_df['method'] == method_name]
            .groupby('neff_target', as_index=False)
            .agg(
                fdr_emp_mean=('fdr_emp_mean', 'mean'),
                fdr_ci_lo   =('fdr_ci_lo',    'mean'),
                fdr_ci_hi   =('fdr_ci_hi',    'mean'),
                R_obs_mean  =('R_obs_mean',   'mean'),
                neff_actual =('neff_actual',  'mean'),
            )
            .sort_values('neff_actual')
        )

    # ── Figure 1: FDR Surface ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6))

    for m in ALL_METHODS:
        c, mk, lbl = _M_COLORS[m], _M_MARKERS[m], _M_LABELS[m]
        avg = _avg_by_neff(m)
        x, y = avg['neff_actual'].values, avg['fdr_emp_mean'].values
        lo, hi = avg['fdr_ci_lo'].values, avg['fdr_ci_hi'].values

        ax1.fill_between(x, lo, hi, alpha=0.14, color=c)
        s_m = surface_df[surface_df['method'] == m].sort_values('neff')
        if not s_m.empty:
            ax1.plot(s_m['neff'], s_m['fdr_fit'], '-', color=c,
                     linewidth=2.2, label=lbl)
        ax1.scatter(x, y, color=c, marker=mk, s=55,
                    edgecolors='black', linewidths=0.7, zorder=5)
        ns = nstar.get(m, float('nan'))
        if not (np.isnan(ns) or np.isinf(ns)):
            ax1.axvline(ns, color=c, linestyle=':', linewidth=1.4, alpha=0.75)

    ax1.axhline(ALPHA, color='black', linestyle='--', linewidth=1.6,
                label=f'$\\alpha = {ALPHA}$')
    ax1.set_xscale('log')
    ax1.set_xlabel('Effective sample size $n_{\\mathrm{eff}}$', fontweight='bold')
    ax1.set_ylabel('$\\widehat{\\mathrm{FDR}}_{\\mathrm{emp}}$', fontweight='bold')
    ax1.set_title(
        'FDR Control Surface over $n_{\\mathrm{eff}}$  '
        '(BPI Challenge 2017, double-null $\\sigma_\\emptyset$)',
        fontweight='bold',
    )
    ax1.set_ylim(bottom=-0.01)
    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    for sp in ax1.spines.values():
        sp.set_visible(True)
    plt.tight_layout()
    _save_fig(fig1, 'rq1_2_fig1_fdr_surface.pdf', out_dir)

    # ── Figure 2: 2D Heatmap on (log n, ρ) ───────────────────────────────
    cmap_fdr = plt.cm.RdYlBu_r
    norm_fdr = Normalize(vmin=0.0, vmax=0.20)

    # BPI 2017 natural operating point
    bpi17_n   = N_POS_POOL + N_NEG_POOL           # 31,509
    bpi17_rho = N_POS_POOL / bpi17_n              # ≈ 0.595

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
    for ax_idx, m in enumerate(ALL_METHODS):
        ax  = axes2.flat[ax_idx]
        sub = agg_df[agg_df['method'] == m]
        log_n = np.log10(sub['n'].values.astype(float))
        rho_v = sub['rho'].values.astype(float)
        fdr_v = np.clip(sub['fdr_emp_mean'].values.astype(float), 0.0, 0.30)

        sc = ax.scatter(log_n, rho_v, c=fdr_v, cmap=cmap_fdr, norm=norm_fdr,
                        s=120, edgecolors='black', linewidths=0.8, zorder=5)

        # Green ring for FDR-controlled cells
        ctrl = fdr_v <= ALPHA
        if ctrl.any():
            ax.scatter(log_n[ctrl], rho_v[ctrl], s=170, marker='o',
                       facecolors='none', edgecolors='#009E73',
                       linewidths=2.5, zorder=6)

        # α=0.05 iso-contour
        if len(log_n) >= 4:
            try:
                from scipy.interpolate import griddata as _gd
                xi = np.linspace(log_n.min(), log_n.max(), 120)
                yi = np.linspace(rho_v.min(), rho_v.max(), 80)
                XI, YI = np.meshgrid(xi, yi)
                ZI = _gd((log_n, rho_v), fdr_v, (XI, YI), method='linear')
                if ZI is not None and not np.all(np.isnan(ZI)):
                    cs = ax.contour(XI, YI, ZI, levels=[ALPHA],
                                    colors='black', linewidths=2.0,
                                    linestyles='--')
                    ax.clabel(cs, fmt=f'$\\alpha$={ALPHA}', fontsize=8)
            except Exception:
                pass

        # BPI 2017 natural operating point
        ax.scatter([np.log10(bpi17_n)], [bpi17_rho], s=220, marker='*',
                   c='gold', edgecolors='black', linewidths=1.0, zorder=10,
                   label='BPI 2017 full log')

        ax.set_xlabel('$\\log_{10}(n)$', fontweight='bold')
        ax.set_ylabel('$\\rho = n^+/n$', fontweight='bold')
        ax.set_title(_M_LABELS[m], fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, frameon=True,
                  fancybox=False, edgecolor='black')
        for sp in ax.spines.values():
            sp.set_visible(True)

    sm = ScalarMappable(cmap=cmap_fdr, norm=norm_fdr)
    sm.set_array([])
    cb = fig2.colorbar(sm, ax=axes2, shrink=0.55, pad=0.03)
    cb.set_label('$\\widehat{\\mathrm{FDR}}_{\\mathrm{emp}}$', fontweight='bold')
    cb.ax.axhline(ALPHA, color='black', linestyle='--', linewidth=1.5)
    fig2.suptitle(
        'FDR Control over $(\\log n,\\,\\rho)$ Design Space — BPI Challenge 2017',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    _save_fig(fig2, 'rq1_2_fig2_heatmap.pdf', out_dir)

    # ── Figure 3: FDR–Power Trade-off (dual y-axis, 2×2) ─────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    for ax_idx, m in enumerate(ALL_METHODS):
        ax_l = axes3.flat[ax_idx]
        ax_r = ax_l.twinx()
        c, lbl = _M_COLORS[m], _M_LABELS[m]
        avg = _avg_by_neff(m)
        x  = avg['neff_actual'].values
        yf = avg['fdr_emp_mean'].values
        yr = avg['R_obs_mean'].values
        lo = avg['fdr_ci_lo'].values
        hi = avg['fdr_ci_hi'].values

        ax_l.fill_between(x, lo, hi, alpha=0.15, color=c)
        ax_l.plot(x, yf, '-o', color=c, linewidth=2.2, markersize=6,
                  markeredgecolor='black', markeredgewidth=0.6,
                  label='$\\widehat{\\mathrm{FDR}}_{\\mathrm{emp}}$')
        ax_l.axhline(ALPHA, color='black', linestyle='--', linewidth=1.2)
        ax_r.plot(x, yr, '--D', color=c, linewidth=1.6, markersize=5,
                  alpha=0.65, markeredgecolor='black', markeredgewidth=0.5,
                  label='$R_{\\mathrm{obs}}$')

        ns = nstar.get(m, float('nan'))
        if not (np.isnan(ns) or np.isinf(ns)):
            ax_l.axvline(ns, color=c, linestyle=':', linewidth=1.5, alpha=0.8)

        ax_l.set_xscale('log')
        ax_l.set_ylim(-0.01, max(0.22, float(np.nanmax(hi)) + 0.02))
        ax_l.set_xlabel('$n_{\\mathrm{eff}}$', fontweight='bold')
        ax_l.set_ylabel('$\\widehat{\\mathrm{FDR}}_{\\mathrm{emp}}$',
                        color=c, fontweight='bold')
        ax_r.set_ylabel('$R_{\\mathrm{obs}}$ (rejections)',
                        color=c, alpha=0.7, fontweight='bold')
        ax_l.set_title(lbl, fontweight='bold')

        h1, l1 = ax_l.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax_l.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8,
                    frameon=True, fancybox=False, edgecolor='black')
        for sp in ax_l.spines.values():
            sp.set_visible(True)

    fig3.suptitle(
        'FDR–Power Trade-off over $n_{\\mathrm{eff}}$ — BPI Challenge 2017',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()
    _save_fig(fig3, 'rq1_2_fig3_fdr_power.pdf', out_dir)

    print(f"\n  ✓ 3 figures saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 90)
    print("RQ1.2 — FDR SURFACE OVER (n, rho): BPI CHALLENGE 2017")
    print("  Double-Null Protocol: sigma_label ∘ sigma_trace")
    print("  Dataset: BPI Challenge 2017  (Teinemaa et al. TKDE 2019)")
    print(f"  Class 1 Not-Accepted (Deviant): {N_POS_POOL:,}  "
          f"(rho_nat ≈ {N_POS_POOL/(N_POS_POOL+N_NEG_POOL):.3f})")
    print(f"  Class 0 Accepted    (Normal):   {N_NEG_POOL:,}")
    print("=" * 90)
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  R_REPS={R_REPS}, B_null={B_NULL}, B1_sub={B1_SUB}, B2_sub={B2_SUB}")
    print(f"  alpha={ALPHA}, TV_threshold={TV_THRESHOLD}")
    print(f"  Output: {OUTPUT_DIR}")

    t_total = time.time()

    # ── Section 1: design grid ────────────────────────────────────────────
    grid = build_design_grid()
    print_grid(grid)

    # ── Section 2: full-log pipeline ─────────────────────────────────────
    full_out        = run_full_pipeline(n_workers=N_JOBS)
    case_data       = full_out['case_data']
    candidates_all  = full_out['candidates_all']
    case_ids_sorted = full_out['case_ids_sorted']
    labels          = full_out['labels']
    activity_pmf    = full_out['activity_pmf']
    full_wall       = full_out['wall_seconds']

    # ── Section 6: parallel surface experiment ────────────────────────────
    all_results = run_surface_experiment(
        grid, case_data, candidates_all,
        case_ids_sorted, labels, activity_pmf,
        n_jobs=N_JOBS,
    )
    exp_wall = time.time() - t_total - full_wall

    # ── Section 7: aggregate per cell ────────────────────────────────────
    agg_df = aggregate_cell_results(all_results)

    # ── Section 8: surface estimation ────────────────────────────────────
    surface_dict = fit_fdr_surface(agg_df)

    # ── Section 9: save outputs ───────────────────────────────────────────
    save_results(agg_df, surface_dict, all_results, full_wall, exp_wall, grid)

    # ── Section 10: publication figures ───────────────────────────────────
    print("\n" + "=" * 90)
    print("SECTION 10: PUBLICATION FIGURES")
    print("=" * 90)
    generate_figures(agg_df, surface_dict)

    # ── Final summary ─────────────────────────────────────────────────────
    total_wall = time.time() - t_total
    print(f"\n{'='*90}")
    print("RQ1.2 — BPI CHALLENGE 2017 COMPLETE")
    print(f"{'='*90}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f}h)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\n  n*_eff per method (min effective sample size for FDR control):")
    for m_name in ALL_METHODS:
        ns  = surface_dict['nstar'].get(m_name, float('nan'))
        ci  = surface_dict['nstar_ci'].get(m_name, (float('nan'), float('nan')))
        print(f"    {m_name:25s}: n*_eff = {ns:.1f}  "
              f"CI=[{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"\n  Cells where FDR_emp ≤ alpha (of {len(grid)} total):")
    for m_name in ALL_METHODS:
        n_ctrl = (agg_df[agg_df['method'] == m_name]['fdr_emp_mean'] <= ALPHA).sum()
        print(f"    {m_name:25s}: {n_ctrl}/{len(grid)} cells")
    print(f"\n  TV divergence (max across all reps): "
          f"{max(r['tv_divergence'] for r in all_results):.4f}")
    print(f"  TV-flagged reps: {sum(r['tv_flagged'] for r in all_results)}"
          f" of {len(all_results)}")
    print(f"{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1.2 FDR Surface Study — BPI Challenge 2017 (Double-Null Protocol)"
    )
    parser.add_argument(
        '--r-reps', type=int, default=R_REPS,
        help=f'Sub-sampling replicates per cell (default: {R_REPS})'
    )
    parser.add_argument(
        '--b-null', type=int, default=B_NULL,
        help=f'Doubly-null held-out replicates per sub-log (default: {B_NULL})'
    )
    parser.add_argument(
        '--b1-sub', type=int, default=B1_SUB,
        help=f'Label permutation budget per replicate (default: {B1_SUB})'
    )
    parser.add_argument(
        '--b2-sub', type=int, default=B2_SUB,
        help=f'Structural permutation budget per replicate (default: {B2_SUB})'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Parallel jobs (-1 = all cores, default: {N_JOBS})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run with R_REPS=2, B_null=5, B1=50, B2=20 on a 3-cell subset'
    )
    args = parser.parse_args()

    if args.dry_run:
        R_REPS = 2
        B_NULL = 5
        B1_SUB = 50
        B2_SUB = 20
        print("*** DRY RUN: R_REPS=2, B_NULL=5, B1=50, B2=20, 3-cell subset ***")
        _orig_build = build_design_grid
        def build_design_grid(**kw):  # noqa: F811
            return _orig_build(**kw)[:3]
    else:
        R_REPS  = args.r_reps
        B_NULL  = args.b_null
        B1_SUB  = args.b1_sub
        B2_SUB  = args.b2_sub

    N_JOBS = args.n_jobs
    main()
