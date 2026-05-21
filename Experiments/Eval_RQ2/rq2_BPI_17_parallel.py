#!/usr/bin/env python3
"""
rq2_BPI_17_parallel.py  —  RQ2 Specification Quality Degradation Analysis
==========================================================================
Block B.1: Label Noise Perturbation
Block B.2: Structural Noise Perturbation

Three methods on shared M_all: P1 (Hou-Storey) | DRVA | DeclareMiner

RESEARCH QUESTION
-----------------
RQ2: As signal is progressively corrupted, how do P1, DRVA, and
DeclareMiner differ in:
  (a) the size and composition of their discovered specifications,
  (b) the consistency of those specifications relative to the original
      (Jaccard stability, Gained, Lost pattern counts), and
  (c) the empirical FDR estimates under the doubly-null protocol?

The key distinction from RQ1 is that RQ2 operates under PARTIAL signal
regimes — controlled noise erodes signal gradually — to test graceful
degradation rather than total-null validity.

PERTURBATION OPERATORS
-----------------------
B.1 — Label noise  N_label(ε):
    Flip outcome label of each case independently with probability ε.
    Destroys discriminative signal proportionally to ε.
    Structural signal (trace ordering) is unaffected.
    Grid: ε ∈ {0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50}

B.2 — Structural noise  N_struct(ρ):
    For each trace, scan adjacent pairs and swap each with probability ρ.
    Destroys local temporal ordering proportionally to ρ.
    Discriminative signal (label assignment) is unaffected.
    Grid: ρ ∈ {0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00}

METRICS PER (METHOD, LEVEL)
----------------------------
    R(ℓ)       Discovery count (real-data run on L_ℓ).
    J(ℓ)       Jaccard similarity to the original specification S_orig.
    Gained(ℓ)  |S_ℓ setminus S_orig|  — new patterns under perturbation.
    Lost(ℓ)    |S_orig setminus S_ℓ|  — original patterns dropped under perturbation.
    FDR_emp(ℓ) Empirical FDR from B_null doubly-null replicates on L_ℓ.

SPEED OPTIMISATIONS  (from RQ2 design document)
-------------------------------------------------
  S1. Precompute holds_all(L_ℓ) once per level for real-data runs.
      Reuse the same holds for DM null-replicate confidence recomputation.
  S2. Analytic Hou p-values (no B_null calibration) for P1 real-data runs on
      perturbed levels — valid for trend tracking; avoids 200-replicate
      calibration overhead per level.
  S3. Reduced internal budgets for null replicates:
        B1_NULL = 150  (label perm inside each replicate)
        B2_NULL = 75   (structural perm inside each replicate)
        PI_DRVA_NULL = 100  (DRVA shuffleLog iterations)
  S4. Reduced B_null for intermediate levels (50), anchor levels use 100.
  S5. Level-level parallelism: each (operator, level) pair is an independent
      joblib job. Inner null replicates run sequentially to avoid nesting.
  S6. Vectorised Δconf computation for DRVA and DM null replicates using the
      precomputed holds matrix.

NULL REPLICATE PROTOCOL
------------------------
For each replicate b on L_ℓ:
  1. Apply σ_trace to L_ℓ: fully shuffle activities within each trace.
     → p_struct^(b) ~ U(0,1)
  2. Apply σ_label to the σ_trace(L_ℓ): permute class labels.
     → p_disc^(b) ~ U(0,1)
  3. Recompute holds on doubly-null log.
  4. Run all three methods → V_b^(m,ℓ).
     Every rejection is a false positive by construction.

P1 ORACLE IN NULL REPLICATES
------------------------------
Under the double-null (ρ_sd = 0):
    T_Hou ~ c·χ²_f,  c = W_STRUCT² + W_DISC² = 0.52,  f = 2/c ≈ 3.846
    p_Hou^oracle = chi2.sf(T_Hou / c, f)
Single gate: q_Hou ≤ α (matches p1 is_significant_final).

OUTPUT FILES
-------------
    rq2_label_noise_metrics.csv       Degradation table for B.1.
    rq2_structural_noise_metrics.csv  Degradation table for B.2.
    rq2_cross_jaccard.csv             Cross-method Jaccard per level.
    rq2_null_counts.json              Raw V_b arrays per (op, level, method).
    rq2_results.json                  Full structured results for paper.

Version : 1.0
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Hou (2005). Stat. Prob. Lett. 73:179-187.
Storey (2002). JRSS-B 64(3):479-498.
Gao (2023). arXiv:2310.06357.
Phipson & Smyth (2010). Stat. Appl. Genet. Mol. Biol. 9(1):Art.39.
Cecconi, Augusto & Di Ciccio (2021). BPM Forum 2021, LNBIP 427, pp.73-91.
Pellegrina & Vandin (2018/2020). KDD 2018 / TKDD 2020.
"""

import sys
import os
import copy
import io
import contextlib
import time
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

# ═══════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Phase 1 (Hou-Storey framework) ───────────────────────────────────────
from P1_SDSM.p1_BPI_17_hou import (
    load_and_preprocess_data,
    generate_candidate_patterns,
    split_by_class,
    compute_prevalence_from_holds,
    compute_holds_by_case_batch,
    precompute_activity_index,
    run_label_permutation_test,
    run_structural_permutation_test,
    hou_combination_statistic,
    hou_satterthwaite_params,
    estimate_rho_sd,
    W_DISC,
    W_STRUCT,
    adaptive_storey_pi0,
    storey_qvalue,
    benjamini_hochberg,
    execute_pipeline,
    generate_outputs,
    CaseInfo,
    PatternTestResult,
    CONFIG       as P1_CONFIG,
    INPUT_FILE   as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
)

# ── DRVA baseline ─────────────────────────────────────────────────────────
from BaselinesRQ1.DRVA_BPI_17 import (
    run_drva,
    run_drva_on_doubly_null_log,
    DRVA_CONFIG,
)

# ── DeclareMiner baseline ─────────────────────────────────────────────────
from BaselinesRQ1.DeclareMiner_BPI_17 import (
    run_declareminer,
    run_declareminer_on_doubly_null_log,
    DM_CONFIG,
    apply_threshold_decision as dm_apply_threshold,
    compute_support_from_holds as dm_compute_support_from_holds,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH       = P1_INPUT_FILE
PHASE0_JSON    = P1_SPEC_FILE
RQ2_OUTPUT_DIR = "RQ2_BPI_17"

# Perturbation grids
LABEL_NOISE_LEVELS  = [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
STRUCT_NOISE_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

# FDR level
ALPHA      = 0.05
ALPHA_DRVA = 0.01

# Original P1 run budget (full)
B1_FULL      = 1_500
B2_FULL      = 2_000
B_NULL_FULL  = 200
B1_NULL_FULL = 75
B2_NULL_FULL = 75

# Real-data P1 run budget on perturbed logs (reduced; no calibration)
# Analytic Hou p-values used → no B_null overhead.
B1_REAL = 500    # label perm for real-data perturbed run
B2_REAL = 200    # structural perm for real-data perturbed run

# DRVA full run budget for real-data perturbed levels (reduced from 1000)
PI_DRVA_REAL = 500

# Null replicate budgets (reduced for RQ2 feasibility)
B1_NULL      = 150   # label perm per null replicate
B2_NULL      = 75    # structural perm per null replicate
PI_DRVA_NULL = 100   # DRVA shuffleLog per null replicate

# B_null per level
B_NULL_ANCHOR       = 100   # ℓ=0 and ℓ=max (anchor levels)
B_NULL_INTERMEDIATE = 50    # intermediate levels

# Hou oracle under double-null (rho_sd=0)
_C_NULL, _F_NULL = hou_satterthwaite_params(W_STRUCT, W_DISC, rho_sd=0.0)

# Base seed for RQ2 (offset from RQ1's 20260521)
BASE_SEED = 20260601

# Parallelism (over perturbation levels)
N_JOBS = -1

# Method constants
METHOD_P1   = "P1_HouStorey"
METHOD_DRVA = "DRVA"
METHOD_DM   = "DeclareMiner"
ALL_METHODS = [METHOD_P1, METHOD_DRVA, METHOD_DM]


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress():
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# PERTURBATION OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

def apply_label_noise(
    case_data: dict,
    epsilon: float,
    seed: int,
) -> dict:
    """
    Label noise operator N_label(ε).

    Flip each case's outcome label independently with probability ε.
    ε = 0.0 → original log unchanged.
    ε = 0.5 → labels completely randomised (discriminative signal destroyed).

    Traces are NOT modified; structural signal is unaffected.

    Args:
        case_data: Original case data dict.
        epsilon:   Flip probability ∈ [0, 1].
        seed:      RNG seed for reproducibility.

    Returns:
        New dict with shallow-copied CaseInfo objects (only outcome may differ).
    """
    if epsilon == 0.0:
        return case_data   # no copy needed — nothing changes

    rng    = np.random.RandomState(seed)
    result = {}
    n_flipped = 0

    for cid, case in case_data.items():
        if rng.random() < epsilon:
            ci         = copy.copy(case)
            ci.outcome = 1 - case.outcome
            result[cid] = ci
            n_flipped += 1
        else:
            result[cid] = case   # no copy: unchanged

    return result


def apply_structural_noise(
    case_data: dict,
    rho: float,
    seed: int,
) -> dict:
    """
    Structural noise operator N_struct(ρ).

    For each trace, uniformly sample round(ρ × (n-1)) non-overlapping
    adjacent positions without replacement and swap each selected pair.
    This gives exact control over perturbation magnitude with no positional
    bias — matches the proposal: "select ⌊|trace|/2⌋ random adjacent pairs".

    At ρ = 0.0: traces unchanged (0 pairs selected).
    At ρ = 1.0: all (n-1) adjacent pairs selected and swapped.

    Labels are NOT modified; discriminative signal is unaffected.

    Args:
        case_data: Original case data dict.
        rho:       Fraction of adjacent pairs to swap ∈ [0, 1].
        seed:      RNG seed for reproducibility.

    Returns:
        New dict with shallow-copied CaseInfo objects having modified traces.
    """
    if rho == 0.0:
        return case_data

    rng    = np.random.RandomState(seed)
    result = {}

    for cid, case in case_data.items():
        trace = case.trace.copy()
        n     = len(trace)
        if n > 1:
            n_pairs   = n - 1
            n_to_swap = int(round(rho * n_pairs))
            if n_to_swap > 0:
                pair_idxs = rng.choice(n_pairs, size=n_to_swap, replace=False)
                for i in sorted(pair_idxs):
                    trace[i], trace[i + 1] = trace[i + 1], trace[i]
        if trace != case.trace:
            ci                = copy.copy(case)
            ci.trace          = trace
            ci.activity_index = precompute_activity_index(trace, case_id=cid)
            result[cid]       = ci
        else:
            result[cid] = case

    return result


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLY-NULL LOG BUILDER  (reused from RQ1 design)
# ═══════════════════════════════════════════════════════════════════════════

def _build_doubly_null_log(
    case_data_perturbed: dict,
    case_ids_sorted: list,
    permuted_labels: np.ndarray,
    trace_seed: int,
) -> dict:
    """
    Apply σ_label ∘ σ_trace to a PERTURBED log.

    σ_trace: fully shuffle activities within each trace of L_ℓ.
             → p_struct^(b) ~ U(0,1) regardless of residual structural signal.
    σ_label: replace outcomes with permuted_labels.
             → p_disc^(b) ~ U(0,1).

    Under the double-null T_Hou ~ c·χ²_f (c=0.52, f≈3.846, rho_sd=0).
    """
    rng  = np.random.RandomState(trace_seed)
    null = {}

    for i, cid in enumerate(case_ids_sorted):
        ci_orig = case_data_perturbed[cid]
        ci      = copy.copy(ci_orig)
        ci.outcome = int(permuted_labels[i])

        shuffled = ci_orig.trace.copy()
        rng.shuffle(shuffled)
        ci.trace         = shuffled
        ci.activity_index = precompute_activity_index(shuffled, case_id=cid)
        null[cid] = ci

    return null


# ═══════════════════════════════════════════════════════════════════════════
# P1 SIMPLIFIED RUN ON PERTURBED LOG
# (Analytic Hou p-values; no B_null empirical calibration)
# ═══════════════════════════════════════════════════════════════════════════

def run_p1_on_perturbed_log(
    case_data_perturbed: dict,
    candidates_all: list,
    B1: int,
    B2: int,
    alpha: float,
    seed: int,
    holds_precomputed: dict = None,
) -> dict:
    """
    Simplified P1 decision on a perturbed log.

    Uses ANALYTIC Hou Satterthwaite p-values (not empirical p̃_Hou) to avoid
    the B_null=200 calibration overhead at each perturbation level.  The
    Satterthwaite approximation is valid for trend tracking across ε/ρ grids;
    the ranking of patterns by T_Hou is identical under both calibrations
    (monotone transformation).

    Gate: single q_Hou ≤ α (matches p1 is_significant_final).

    Args:
        case_data_perturbed: Log at perturbation level ℓ.
        candidates_all:      Fixed M_all candidate pool.
        B1:                  Label permutation resamples.
        B2:                  Structural permutation resamples per class.
        alpha:               FDR level.
        seed:                RNG seed.
        holds_precomputed:   If provided, skip holds recomputation (S1 optimisation).

    Returns:
        dict with:
            is_significant  (m,) bool array
            S_set           frozenset of significant (ct, a, b) specs
            R_obs           int
            p_disc          (m,) array
            p_struct_dom    (m,) array
            p_hou_analytic  (m,) array
            q_hou           (m,) array (1.0 for patterns outside scope)
            holds           dict (possibly precomputed)
            null_delta_mat  (B1, m) array
            m_prime         int (scope-filter size)
            structural_idx  list[int]
    """
    m = len(candidates_all)

    # Step 1: holds (precomputed or freshly computed)
    if holds_precomputed is not None:
        holds = holds_precomputed
    else:
        with _suppress():
            holds = compute_holds_by_case_batch(
                case_data_perturbed, candidates_all
            )

    # Step 2: label permutation test → p_disc + null_delta_mat
    with _suppress():
        disc = run_label_permutation_test(
            case_data_perturbed, candidates_all, holds, B1, seed
        )
    null_delta_mat = disc.pop('__null_delta_matrix__')
    p_disc = np.array([disc[spec]['p_two_sided'] for spec in candidates_all])

    # Step 3: structural permutation tests → p_struct screen + test
    D_0, D_1 = split_by_class(case_data_perturbed)

    with _suppress():
        st0 = run_structural_permutation_test(
            D_0, candidates_all, 0, B2, seed + 1, n_workers=1
        )
        st1 = run_structural_permutation_test(
            D_1, candidates_all, 1, B2, seed + 2, n_workers=1
        )

    p_screen_c0 = np.array([
        st0[spec]['p_structural_screen'] if spec in st0 else 1.0
        for spec in candidates_all
    ])
    p_screen_c1 = np.array([
        st1[spec]['p_structural_screen'] if spec in st1 else 1.0
        for spec in candidates_all
    ])
    p_test_c0 = np.array([
        st0[spec]['p_structural_test'] if spec in st0 else 1.0
        for spec in candidates_all
    ])
    p_test_c1 = np.array([
        st1[spec]['p_structural_test'] if spec in st1 else 1.0
        for spec in candidates_all
    ])

    # Step 4: dominant class from label-permutation delta_obs
    delta_obs    = np.array([disc[spec]['delta_obs'] for spec in candidates_all])
    dominant     = np.where(delta_obs >= 0.0, 1, 0)
    p_struct_dom = np.where(dominant == 1, p_test_c1, p_test_c0)

    # Step 5: Hou analytic p-values (Satterthwaite, global rho_sd)
    rho_sd       = estimate_rho_sd(p_struct_dom, p_disc)
    c_h, f_h     = hou_satterthwaite_params(W_STRUCT, W_DISC, rho_sd)
    tf_obs       = hou_combination_statistic(p_struct_dom, p_disc, W_STRUCT, W_DISC)
    p_hou        = np.clip(stats.chi2.sf(tf_obs / c_h, df=f_h), 1e-300, 1.0)

    # Step 6: sample-split scope filter
    structural_idx = [
        i for i in range(m)
        if min(p_screen_c0[i], p_screen_c1[i]) <= alpha
    ]
    m_prime = len(structural_idx)

    # Step 7: Adaptive Storey + single gate
    is_sig  = np.zeros(m, dtype=bool)
    q_hou   = np.ones(m)

    if m_prime > 0:
        p_mp        = p_hou[structural_idx]
        pi0_b, _    = adaptive_storey_pi0(p_mp, q=alpha)
        q_mp        = storey_qvalue(p_mp, pi0_b)
        for rank, orig_i in enumerate(structural_idx):
            q_hou[orig_i] = q_mp[rank]
            is_sig[orig_i] = bool(q_mp[rank] <= alpha)

    return {
        'is_significant':  is_sig,
        'S_set':           frozenset(
                               candidates_all[i] for i in range(m) if is_sig[i]
                           ),
        'R_obs':           int(is_sig.sum()),
        'p_disc':          p_disc,
        'p_struct_dom':    p_struct_dom,
        'p_hou_analytic':  p_hou,
        'q_hou':           q_hou,
        'holds':           holds,
        'null_delta_mat':  null_delta_mat,
        'm_prime':         m_prime,
        'structural_idx':  structural_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DRVA RUN ON PERTURBED LOG
# ═══════════════════════════════════════════════════════════════════════════

def run_drva_on_perturbed_log(
    case_data_perturbed: dict,
    candidates_all: list,
    alpha_drva: float,
    pi: int,
) -> dict:
    """
    Run DRVA on a perturbed log with reduced π.

    Hierarchical simplification disabled; M_tested = M_all.

    Returns:
        dict with S_set (frozenset), R_obs (int), conf_A, conf_B, ediff arrays.
    """
    cfg = DRVA_CONFIG.copy()
    cfg['alpha']                = alpha_drva
    cfg['pi']                   = pi
    cfg['hierarchical_pruning'] = False
    cfg['mmin']                 = 0.0
    cfg['mdiff_min']            = 0.0

    with _suppress():
        drva_out = run_drva(
            config         = cfg,
            case_data      = case_data_perturbed,
            candidates_all = candidates_all,
        )

    rejected = np.array([r['is_significant_cecconi'] for r in drva_out['results']])

    # Build spec set from ordered results
    S_set = frozenset(
        (r['constraint_type'], r['activity_a'], r['activity_b'])
        for r in drva_out['results']
        if r['is_significant_cecconi']
    )

    return {
        'S_set': S_set,
        'R_obs': drva_out['n_rejected_cecconi'],
        'drva_out': drva_out,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DECLAREMINER RUN ON PERTURBED LOG
# ═══════════════════════════════════════════════════════════════════════════

def run_dm_on_perturbed_log(
    case_data_perturbed: dict,
    candidates_all: list,
    tau_star: float,
    tau_min: float,
    holds_precomputed: dict = None,
) -> dict:
    """
    Apply fixed threshold τ* to the perturbed log.

    Uses precomputed holds for vectorised Δconf computation (S6 optimisation).
    τ* is NOT recalibrated — we use the value calibrated on the original data.
    This intentionally exposes how a fixed threshold degrades under noise.

    Returns:
        dict with S_set, R_obs, delta_conf, conf0, conf1.
    """
    m = len(candidates_all)

    # Compute holds on perturbed log if not precomputed
    if holds_precomputed is not None:
        holds = holds_precomputed
    else:
        with _suppress():
            holds = compute_holds_by_case_batch(
                case_data_perturbed, candidates_all
            )

    D_0, D_1   = split_by_class(case_data_perturbed)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    # Vectorised support & confidence from holds (S6 optimisation)
    supp0 = np.zeros(m, dtype=np.float64)
    supp1 = np.zeros(m, dtype=np.float64)
    conf0 = np.zeros(m, dtype=np.float64)
    conf1 = np.zeros(m, dtype=np.float64)

    for i, spec in enumerate(candidates_all):
        h = holds.get(spec, {})
        nsat0 = napp0 = nsat1 = napp1 = 0
        for cid, val in h.items():
            if cid in ids_class0:
                napp0 += 1
                if val == 1:
                    nsat0 += 1
            elif cid in ids_class1:
                napp1 += 1
                if val == 1:
                    nsat1 += 1
        supp0[i] = nsat0 / n0    if n0    > 0 else 0.0
        conf0[i] = nsat0 / napp0 if napp0 > 0 else 0.0
        supp1[i] = nsat1 / n1    if n1    > 0 else 0.0
        conf1[i] = nsat1 / napp1 if napp1 > 0 else 0.0

    # Apply fixed threshold decision
    rejected = dm_apply_threshold(
        conf0, conf1, supp0, supp1,
        tau_delta_conf = tau_star,
        tau_min        = tau_min,
    )
    n_rejected = int(rejected.sum())

    S_set = frozenset(
        candidates_all[i]
        for i in range(m)
        if rejected[i]
    )

    return {
        'S_set':       S_set,
        'R_obs':       n_rejected,
        'delta_conf':  np.abs(conf1 - conf0),
        'conf0':       conf0,
        'conf1':       conf1,
        'rejected':    rejected,
    }


# ═══════════════════════════════════════════════════════════════════════════
# JACCARD METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_jaccard_metrics(S_ref: frozenset, S_perturbed: frozenset) -> dict:
    """
    Compute Jaccard stability and asymmetric change counts.

    J = |S_ref ∩ S_perturbed| / |S_ref ∪ S_perturbed|
    Gained = |S_perturbed setminus S_ref|   (new patterns introduced by perturbation)
    Lost   = |S_ref setminus S_perturbed|   (original patterns dropped by perturbation)

    J = 1.0 for perfectly stable specification.
    J = 0.0 for completely disjoint specifications.

    Args:
        S_ref:       Original discovery set (at ℓ = 0).
        S_perturbed: Discovery set at perturbation level ℓ.

    Returns:
        dict with Jaccard, Gained, Lost, |S_ref|, |S_perturbed|, |intersection|.
    """
    inter  = len(S_ref & S_perturbed)
    union  = len(S_ref | S_perturbed)
    jaccard = inter / union if union > 0 else 1.0

    return {
        'Jaccard':       jaccard,
        'Gained':        len(S_perturbed - S_ref),
        'Lost':          len(S_ref - S_perturbed),
        'n_ref':         len(S_ref),
        'n_perturbed':   len(S_perturbed),
        'n_intersection': inter,
    }


def compute_cross_jaccard(
    S_p1: frozenset,
    S_drva: frozenset,
    S_dm: frozenset,
) -> dict:
    """
    Cross-method Jaccard similarity at a given perturbation level.

    J(P1, DRVA) = |S_P1 ∩ S_DRVA| / |S_P1 ∪ S_DRVA|

    A low cross-Jaccard means methods disagree on which patterns are deviant,
    indicating that the testing procedure drives outcomes, not just signal.
    """
    def _j(A, B):
        u = len(A | B)
        return len(A & B) / u if u > 0 else 1.0

    return {
        'J_P1_DRVA': _j(S_p1, S_drva),
        'J_P1_DM':   _j(S_p1, S_dm),
        'J_DRVA_DM': _j(S_drva, S_dm),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP BCa CI (inline, no eval_utils dependency)
# ═══════════════════════════════════════════════════════════════════════════

def _bca_ci(data: np.ndarray, B: int = 800, seed: int = 42) -> tuple:
    """BCa 95% CI for the mean of data."""
    rng   = np.random.RandomState(seed)
    n     = len(data)
    theta = float(np.mean(data))
    boot  = np.array([np.mean(rng.choice(data, n, replace=True)) for _ in range(B)])
    z0    = stats.norm.ppf(float(np.mean(boot < theta)) + 1e-10)

    jack  = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jm    = jack.mean()
    num   = float(np.sum((jm - jack) ** 3))
    den   = float(6.0 * (np.sum((jm - jack) ** 2) ** 1.5))
    a     = num / den if abs(den) > 1e-15 else 0.0

    def _adj(z_):
        return stats.norm.cdf(z0 + (z0 + z_) / (1.0 - a * (z0 + z_)))

    lo = float(np.clip(_adj(stats.norm.ppf(0.025)), 0.001, 0.999))
    hi = float(np.clip(_adj(stats.norm.ppf(0.975)), 0.001, 0.999))
    return float(np.percentile(boot, lo * 100)), float(np.percentile(boot, hi * 100))


def _compute_fdr_from_null(
    null_counts: np.ndarray,
    R_obs: int,
    alpha_nominal: float,
) -> dict:
    ev  = float(np.mean(null_counts))
    den = max(R_obs, 1)
    fdr = ev / den
    arr = null_counts.astype(float) / den
    try:
        lo, hi = _bca_ci(arr)
    except Exception:
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        'FDR_emp':      fdr,
        'FDR_CI_lower': lo,
        'FDR_CI_upper': hi,
        'E_V_b':        ev,
        'FWER_emp':     float(np.mean(null_counts > 0)),
        'controls_FDR': bool(fdr <= alpha_nominal),
    }


# ═══════════════════════════════════════════════════════════════════════════
# NULL REPLICATE RUNNER FOR ONE LEVEL
# ═══════════════════════════════════════════════════════════════════════════

def run_null_replicates_for_level(
    case_data_perturbed: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels_perturbed: np.ndarray,
    B_null: int,
    B1_null: int,
    B2_null: int,
    pi_drva_null: int,
    alpha: float,
    alpha_drva: float,
    tau_star_dm: float,
    tau_min_dm: float,
    base_seed: int,
) -> dict:
    """
    Run B_null doubly-null replicates on a PERTURBED log.

    Each replicate applies σ_label ∘ σ_trace to case_data_perturbed:
      - σ_trace: fully shuffle trace activities (destroys all temporal order)
      - σ_label: permute class labels of the perturbed log (marginals preserved)

    Every rejection is a false positive by construction, because both the
    structural and discriminative axes are simultaneously nullified.

    P1 oracle: analytic chi2.sf(T_Hou/c, f), c=0.52, f≈3.846, rho_sd=0.
    DRVA:      run_drva_on_doubly_null_log with pi_drva_null iterations.
    DM:        run_declareminer_on_doubly_null_log with fixed τ*.

    Args:
        case_data_perturbed: Log at perturbation level ℓ (already perturbed).
        candidates_all:      Fixed M_all candidate pool.
        case_ids_sorted:     Lexicographic case-ID ordering.
        labels_perturbed:    (n,) original label vector of the perturbed log
                             (before doubly-null permutation — preserves marginals).
        B_null:              Number of null replicates.
        B1_null, B2_null:    Reduced permutation budgets.
        pi_drva_null:        Reduced DRVA shuffleLog iterations.
        alpha:               FDR level for P1.
        alpha_drva:          Per-rule level for DRVA.
        tau_star_dm:         Fixed DM threshold.
        tau_min_dm:          DM interestingness guard.
        base_seed:           Seed base; replicate b uses base_seed + b.

    Returns:
        dict: {METHOD_P1: (B,) int, METHOD_DRVA: (B,) int, METHOD_DM: (B,) int}
    """
    m      = len(candidates_all)
    n      = len(case_ids_sorted)
    counts = {mth: np.zeros(B_null, dtype=int) for mth in ALL_METHODS}

    # Precompute label permutations (preserving marginals of perturbed log)
    rng_outer = np.random.RandomState(base_seed)
    perm_labels_all = np.stack([
        rng_outer.permutation(labels_perturbed).astype(np.int8)
        for _ in range(B_null)
    ], axis=0)

    drva_cfg_null = DRVA_CONFIG.copy()
    drva_cfg_null.update({
        'pi':                   pi_drva_null,
        'alpha':                alpha_drva,
        'hierarchical_pruning': False,
        'mmin':                 0.0,
        'mdiff_min':            0.0,
    })

    for b in range(B_null):
        rs_trace = base_seed + 100_000 + b    # σ_trace seed
        rs_p1    = base_seed + 200_000 + b    # P1 internal seed
        rs_drva  = base_seed + 300_000 + b    # DRVA internal seed

        # ── Build doubly-null log ─────────────────────────────────────────
        null_cd = _build_doubly_null_log(
            case_data_perturbed,
            case_ids_sorted,
            perm_labels_all[b],
            trace_seed = rs_trace,
        )

        # ── P1: fresh holds + structural + label perm → oracle Hou gate ──
        with _suppress():
            holds_null = compute_holds_by_case_batch(null_cd, candidates_all)

        with _suppress():
            disc_b = run_label_permutation_test(
                null_cd, candidates_all, holds_null, B1_null, rs_p1
            )
        disc_b.pop('__null_delta_matrix__', None)
        p_disc_b = np.array([
            disc_b[spec]['p_two_sided'] for spec in candidates_all
        ])

        D0_b, D1_b = split_by_class(null_cd)

        with _suppress():
            st0_b = run_structural_permutation_test(
                D0_b, candidates_all, 0, B2_null, rs_p1 + 1, n_workers=1
            )
            st1_b = run_structural_permutation_test(
                D1_b, candidates_all, 1, B2_null, rs_p1 + 2, n_workers=1
            )

        p_sc0 = np.array([
            st0_b[spec]['p_structural_screen'] if spec in st0_b else 1.0
            for spec in candidates_all
        ])
        p_sc1 = np.array([
            st1_b[spec]['p_structural_screen'] if spec in st1_b else 1.0
            for spec in candidates_all
        ])
        p_t0 = np.array([
            st0_b[spec]['p_structural_test'] if spec in st0_b else 1.0
            for spec in candidates_all
        ])
        p_t1 = np.array([
            st1_b[spec]['p_structural_test'] if spec in st1_b else 1.0
            for spec in candidates_all
        ])

        delta_obs_b  = np.array([disc_b[spec]['delta_obs'] for spec in candidates_all])
        dom_b        = np.where(delta_obs_b >= 0.0, 1, 0)
        p_struct_b   = np.where(dom_b == 1, p_t1, p_t0)

        # Oracle Hou analytic p-value (exact under double-null)
        tf_b      = hou_combination_statistic(p_struct_b, p_disc_b, W_STRUCT, W_DISC)
        p_hou_b   = np.clip(stats.chi2.sf(tf_b / _C_NULL, df=_F_NULL), 1e-300, 1.0)

        # Scope filter + single gate
        sidx_b = [
            i for i in range(m)
            if min(p_sc0[i], p_sc1[i]) <= alpha
        ]
        n_p1_b = 0
        if sidx_b:
            p_mp     = p_hou_b[sidx_b]
            pi0_b, _ = adaptive_storey_pi0(p_mp, q=alpha)
            q_b      = storey_qvalue(p_mp, pi0_b)
            n_p1_b   = int(np.sum(q_b <= alpha))
        counts[METHOD_P1][b] = n_p1_b

        # ── DRVA ─────────────────────────────────────────────────────────
        with _suppress():
            n_drva_b = run_drva_on_doubly_null_log(
                null_case_data  = null_cd,
                candidates_all  = candidates_all,
                alpha           = alpha_drva,
                replicate_seed  = rs_drva,
                config          = drva_cfg_null,
                holds_all       = holds_null,   # fast path: skip re-evaluation
            )
        counts[METHOD_DRVA][b] = n_drva_b

        # ── DeclareMiner ──────────────────────────────────────────────────
        with _suppress():
            n_dm_b = run_declareminer_on_doubly_null_log(
                null_case_data = null_cd,
                candidates_all = candidates_all,
                tau_star       = tau_star_dm,
                tau_min        = tau_min_dm,
                holds_all      = holds_null,   # fast path
            )
        counts[METHOD_DM][b] = n_dm_b

    return counts


# ═══════════════════════════════════════════════════════════════════════════
# PER-LEVEL ANALYSIS FUNCTION  (parallelised over the grid)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_level(
    operator: str,          # 'label_noise' or 'structural_noise'
    level: float,           # ε or ρ value
    level_idx: int,         # position in grid (for seed offset)
    case_data_orig: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels_orig: np.ndarray,
    S_orig: dict,           # {METHOD: frozenset of (ct, a, b)}
    tau_star_dm: float,
    tau_min_dm: float,
    alpha: float,
    alpha_drva: float,
) -> dict:
    """
    Full analysis for one (operator, level) combination.

    Designed to run as an independent joblib job: computes real-data
    discoveries, Jaccard metrics, cross-method overlap, and B_null
    doubly-null null replicates to estimate FDR_emp at this level.

    Args:
        operator:        'label_noise' or 'structural_noise'.
        level:           Perturbation level ε or ρ.
        level_idx:       Index in the grid (used for seed offsets).
        case_data_orig:  Original (unperturbed) case data.
        candidates_all:  Fixed M_all.
        case_ids_sorted: Lexicographic case-ID ordering.
        labels_orig:     (n,) original binary labels.
        S_orig:          Reference specification sets at ℓ=0.
        tau_star_dm:     Fixed DM threshold calibrated on original data.
        tau_min_dm:      DM interestingness guard.
        alpha:           FDR level for P1.
        alpha_drva:      Per-rule level for DRVA.

    Returns:
        dict with all metrics for this (operator, level).
    """
    print(f"  [{operator}  ℓ={level:.2f}]  Starting analysis...", flush=True)
    t0 = time.time()

    # Seed for this level
    base_seed_level = BASE_SEED + 10_000 * (level_idx + 1) + {
        'label_noise': 0,
        'structural_noise': 100_000,
    }[operator]

    # ── Apply perturbation ────────────────────────────────────────────────
    pert_seed = base_seed_level + 1
    if operator == 'label_noise':
        case_data_pert = apply_label_noise(case_data_orig, level, pert_seed)
    else:
        case_data_pert = apply_structural_noise(case_data_orig, level, pert_seed)

    # Extract perturbed labels for null replicate marginal preservation
    labels_pert = np.array([
        case_data_pert[cid].outcome for cid in case_ids_sorted
    ])

    # ── Step S1: precompute holds once for this level ─────────────────────
    print(f"  [{operator}  ℓ={level:.2f}]  Precomputing holds...", flush=True)
    with _suppress():
        holds_pert = compute_holds_by_case_batch(case_data_pert, candidates_all)

    # ── Real-data run: P1 ─────────────────────────────────────────────────
    print(f"  [{operator}  ℓ={level:.2f}]  Running P1...", flush=True)
    p1_res = run_p1_on_perturbed_log(
        case_data_pert, candidates_all,
        B1=B1_REAL, B2=B2_REAL,
        alpha=alpha, seed=base_seed_level + 2,
        holds_precomputed=holds_pert,
    )

    # ── Real-data run: DRVA ───────────────────────────────────────────────
    print(f"  [{operator}  ℓ={level:.2f}]  Running DRVA...", flush=True)
    drva_res = run_drva_on_perturbed_log(
        case_data_pert, candidates_all, alpha_drva, PI_DRVA_REAL
    )

    # ── Real-data run: DM ─────────────────────────────────────────────────
    dm_res = run_dm_on_perturbed_log(
        case_data_pert, candidates_all, tau_star_dm, tau_min_dm,
        holds_precomputed=holds_pert,
    )

    S_pert = {
        METHOD_P1:   p1_res['S_set'],
        METHOD_DRVA: drva_res['S_set'],
        METHOD_DM:   dm_res['S_set'],
    }

    R_obs = {
        METHOD_P1:   p1_res['R_obs'],
        METHOD_DRVA: drva_res['R_obs'],
        METHOD_DM:   dm_res['R_obs'],
    }

    # ── Jaccard metrics per method ────────────────────────────────────────
    jaccard_metrics = {
        mth: compute_jaccard_metrics(S_orig[mth], S_pert[mth])
        for mth in ALL_METHODS
    }

    # ── Cross-method Jaccard ──────────────────────────────────────────────
    cross_j = compute_cross_jaccard(
        S_pert[METHOD_P1], S_pert[METHOD_DRVA], S_pert[METHOD_DM]
    )

    # ── B_null determination: more replicates for anchor levels ──────────
    is_anchor = (level == 0.0) or (
        (operator == 'label_noise'  and level == LABEL_NOISE_LEVELS[-1]) or
        (operator == 'structural_noise' and level == STRUCT_NOISE_LEVELS[-1])
    )
    B_null_this = B_NULL_ANCHOR if is_anchor else B_NULL_INTERMEDIATE

    # ── Null replicates ───────────────────────────────────────────────────
    print(
        f"  [{operator}  ℓ={level:.2f}]  "
        f"Running {B_null_this} null replicates...", flush=True
    )
    null_counts = run_null_replicates_for_level(
        case_data_perturbed = case_data_pert,
        candidates_all      = candidates_all,
        case_ids_sorted     = case_ids_sorted,
        labels_perturbed    = labels_pert,
        B_null              = B_null_this,
        B1_null             = B1_NULL,
        B2_null             = B2_NULL,
        pi_drva_null        = PI_DRVA_NULL,
        alpha               = alpha,
        alpha_drva          = alpha_drva,
        tau_star_dm         = tau_star_dm,
        tau_min_dm          = tau_min_dm,
        base_seed           = base_seed_level + 50_000,
    )

    # ── FDR metrics ───────────────────────────────────────────────────────
    alpha_vals = {METHOD_P1: alpha, METHOD_DRVA: alpha_drva, METHOD_DM: alpha}
    fdr_metrics = {
        mth: _compute_fdr_from_null(
            null_counts[mth], R_obs[mth], alpha_vals[mth]
        )
        for mth in ALL_METHODS
    }

    wall = time.time() - t0
    print(
        f"  [{operator}  ℓ={level:.2f}]  Done in {wall:.1f}s  "
        f"R_obs=[P1:{R_obs[METHOD_P1]}, DRVA:{R_obs[METHOD_DRVA]}, "
        f"DM:{R_obs[METHOD_DM]}]", flush=True
    )

    return {
        'operator':       operator,
        'level':          level,
        'R_obs':          R_obs,
        'S_pert':         {mth: list(S_pert[mth]) for mth in ALL_METHODS},
        'jaccard':        jaccard_metrics,
        'cross_jaccard':  cross_j,
        'fdr_metrics':    fdr_metrics,
        'null_counts':    {mth: null_counts[mth].tolist() for mth in ALL_METHODS},
        'B_null':         B_null_this,
        'wall_seconds':   wall,
        'm_prime_P1':     p1_res['m_prime'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ORIGINAL DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_original_data(n_workers: int = 1) -> dict:
    """
    Run all three methods on the original (unperturbed) BPI_17 log.

    P1: full execute_pipeline (with empirical Phipson-Smyth calibration).
    DRVA: full π=1000 run.
    DM: calibrated to match R_obs^P1.

    Returns the original specification sets and metadata used throughout
    the RQ2 grid.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — ORIGINAL DATA RUN (P1 + DRVA + DeclareMiner)")
    print("=" * 100)

    # ── P1 full pipeline ──────────────────────────────────────────────────
    t0  = time.time()
    cfg = P1_CONFIG.copy()
    cfg.update({
        'B_label':      B1_FULL,
        'B_trace':      B2_FULL,
        'B_null':       B_NULL_FULL,
        'B1_null':      B1_NULL_FULL,
        'B2_null':      B2_NULL_FULL,
        'fdr_alpha':    ALPHA,
        'random_state': 42,
        'n_workers':    n_workers,
        'n_jobs':       n_workers,
    })
    output = execute_pipeline(input_file=CSV_PATH, config=cfg)
    generate_outputs(output['pattern_results'], output['case_data'], output['timing'])

    case_data       = output['case_data']
    candidates_all  = output['candidates_all']
    pattern_results = output['pattern_results']
    case_ids_sorted = sorted(case_data.keys())
    labels_orig     = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    S_orig_p1 = frozenset(
        (r.constraint_type, r.activity_a, r.activity_b)
        for r in pattern_results if r.is_significant_final
    )
    R_orig_p1 = len(S_orig_p1)

    p1_wall = time.time() - t0
    print(f"\n  P1 complete: {p1_wall:.1f}s  k*={R_orig_p1}")

    # ── DRVA full run ─────────────────────────────────────────────────────
    t0 = time.time()
    drva_cfg = DRVA_CONFIG.copy()
    drva_cfg.update({
        'alpha': ALPHA_DRVA, 'hierarchical_pruning': False,
        'mmin': 0.0, 'mdiff_min': 0.0,
    })
    with _suppress():
        drva_orig = run_drva(config=drva_cfg, case_data=case_data,
                             candidates_all=candidates_all)
    S_orig_drva = frozenset(
        (r['constraint_type'], r['activity_a'], r['activity_b'])
        for r in drva_orig['results'] if r['is_significant_cecconi']
    )
    drva_wall = time.time() - t0
    print(f"  DRVA complete: {drva_wall:.1f}s  R_obs={len(S_orig_drva)}")

    # ── DeclareMiner calibrated to P1 ────────────────────────────────────
    t0 = time.time()
    dm_cfg = DM_CONFIG.copy()
    dm_cfg['R_obs_target'] = R_orig_p1
    with _suppress():
        dm_orig = run_declareminer(
            config=dm_cfg, case_data=case_data,
            candidates_all=candidates_all, R_obs_target=R_orig_p1,
        )
    S_orig_dm = frozenset(
        (r['constraint_type'], r['activity_a'], r['activity_b'])
        for r in dm_orig['results_all'] if r['is_significant']
    )
    tau_star = float(dm_orig['tau_star'])
    tau_min  = float(dm_orig['config'].get('tau_min', DM_CONFIG['tau_min']))
    dm_wall  = time.time() - t0
    print(f"  DM complete: {dm_wall:.1f}s  R_obs={len(S_orig_dm)}  τ*={tau_star:.4f}")

    S_orig = {
        METHOD_P1:   S_orig_p1,
        METHOD_DRVA: S_orig_drva,
        METHOD_DM:   S_orig_dm,
    }
    R_orig = {
        METHOD_P1:   R_orig_p1,
        METHOD_DRVA: len(S_orig_drva),
        METHOD_DM:   len(S_orig_dm),
    }

    print(f"\n  Original R_obs: "
          f"P1={R_orig[METHOD_P1]}  "
          f"DRVA={R_orig[METHOD_DRVA]}  "
          f"DM={R_orig[METHOD_DM]}")

    return {
        'case_data':       case_data,
        'candidates_all':  candidates_all,
        'case_ids_sorted': case_ids_sorted,
        'labels_orig':     labels_orig,
        'pattern_results': pattern_results,
        'S_orig':          S_orig,
        'R_orig':          R_orig,
        'tau_star':        tau_star,
        'tau_min':         tau_min,
        'holds_all':       output['holds_all'],
        'null_delta_mat':  output['null_delta_matrix'],
        'delta_obs':       output['delta_obs'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — RQ2 PERTURBATION GRID
# ═══════════════════════════════════════════════════════════════════════════

def _level_worker(
    operator, level, level_idx,
    case_data_orig, candidates_all, case_ids_sorted, labels_orig,
    S_orig, tau_star, tau_min, alpha, alpha_drva,
):
    """Top-level loky-safe worker for one (operator, level) combination."""
    return analyze_level(
        operator        = operator,
        level           = level,
        level_idx       = level_idx,
        case_data_orig  = case_data_orig,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        labels_orig     = labels_orig,
        S_orig          = S_orig,
        tau_star_dm     = tau_star,
        tau_min_dm      = tau_min,
        alpha           = alpha,
        alpha_drva      = alpha_drva,
    )


def run_perturbation_grid(
    case_data_orig: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels_orig: np.ndarray,
    S_orig: dict,
    tau_star: float,
    tau_min: float,
    n_jobs: int = N_JOBS,
) -> dict:
    """
    Run the full B.1 + B.2 perturbation grid in parallel.

    Each (operator, level) pair is a fully independent job (no shared
    state), making level-level parallelism safe via joblib loky backend.

    Returns:
        dict: {
            'label_noise': [level_result_0, level_result_1, ...],
            'structural_noise': [level_result_0, ...],
        }
    """
    # Build the full job list
    jobs = []
    for li, lv in enumerate(LABEL_NOISE_LEVELS):
        jobs.append(('label_noise',      lv, li))
    for ri, rv in enumerate(STRUCT_NOISE_LEVELS):
        jobs.append(('structural_noise', rv, ri + len(LABEL_NOISE_LEVELS)))

    print("\n" + "=" * 100)
    print(f"SECTION 2 — PERTURBATION GRID  ({len(jobs)} (operator, level) jobs)")
    print(f"  B.1 label noise:     {LABEL_NOISE_LEVELS}")
    print(f"  B.2 structural noise:{STRUCT_NOISE_LEVELS}")
    print(f"  B_null (intermediate)={B_NULL_INTERMEDIATE}  "
          f"B_null (anchor)={B_NULL_ANCHOR}")
    print(f"  B1_null={B1_NULL}  B2_null={B2_NULL}  "
          f"PI_DRVA_null={PI_DRVA_NULL}")
    print(f"  n_jobs={n_jobs}")
    print("=" * 100)

    t0 = time.time()

    results_flat = Parallel(n_jobs=n_jobs, verbose=5, backend='loky')(
        delayed(_level_worker)(
            op, lv, li,
            case_data_orig, candidates_all, case_ids_sorted, labels_orig,
            S_orig, tau_star, tau_min, ALPHA, ALPHA_DRVA,
        )
        for op, lv, li in jobs
    )

    wall = time.time() - t0
    print(f"\n  Grid complete. Total wall time: {wall:.1f}s ({wall/60:.1f} min)")

    # Organise by operator
    grid_results = {'label_noise': [], 'structural_noise': []}
    for res in results_flat:
        grid_results[res['operator']].append(res)

    # Sort by level within each operator
    for op in grid_results:
        grid_results[op].sort(key=lambda x: x['level'])

    return grid_results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_metrics_table(
    level_results: list,
    operator_label: str,
) -> pd.DataFrame:
    """
    Build a long-format metrics table for one perturbation operator.

    Columns:
        operator, level, method,
        R_obs, Jaccard, Gained, Lost, n_ref, n_perturbed,
        FDR_emp, FDR_CI_lower, FDR_CI_upper, FWER_emp, E_V_b, controls_FDR
    """
    rows = []
    for res in level_results:
        level = res['level']
        for mth in ALL_METHODS:
            jm  = res['jaccard'][mth]
            fdr = res['fdr_metrics'][mth]
            rows.append({
                'operator':      operator_label,
                'level':         level,
                'method':        mth,
                'R_obs':         res['R_obs'][mth],
                'Jaccard':       jm['Jaccard'],
                'Gained':        jm['Gained'],
                'Lost':          jm['Lost'],
                'n_ref':         jm['n_ref'],
                'n_perturbed':   jm['n_perturbed'],
                'FDR_emp':       fdr['FDR_emp'],
                'FDR_CI_lower':  fdr['FDR_CI_lower'],
                'FDR_CI_upper':  fdr['FDR_CI_upper'],
                'FWER_emp':      fdr['FWER_emp'],
                'E_V_b':         fdr['E_V_b'],
                'controls_FDR':  fdr['controls_FDR'],
                'B_null':        res['B_null'],
                'm_prime_P1':    res['m_prime_P1'],
            })
    return pd.DataFrame(rows)


def build_cross_jaccard_table(level_results: list, operator_label: str) -> pd.DataFrame:
    """Build cross-method Jaccard table per level."""
    rows = []
    for res in level_results:
        row = {
            'operator': operator_label,
            'level':    res['level'],
        }
        row.update(res['cross_jaccard'])
        rows.append(row)
    return pd.DataFrame(rows)


def save_all_outputs(
    grid_results: dict,
    orig_data: dict,
    total_wall: float,
) -> None:
    """
    Save all RQ2 output files.

    Files:
        rq2_label_noise_metrics.csv
        rq2_structural_noise_metrics.csv
        rq2_cross_jaccard.csv
        rq2_null_counts.json
        rq2_results.json
    """
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    # ── CSV: per-operator metrics tables ─────────────────────────────────
    for op_key, op_label, fname in [
        ('label_noise',      'B.1_LabelNoise',     'rq2_label_noise_metrics.csv'),
        ('structural_noise', 'B.2_StructuralNoise','rq2_structural_noise_metrics.csv'),
    ]:
        df = build_metrics_table(grid_results[op_key], op_label)
        path = os.path.join(RQ2_OUTPUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # ── CSV: cross-method Jaccard ─────────────────────────────────────────
    cj_all = pd.concat([
        build_cross_jaccard_table(grid_results['label_noise'],      'B.1_LabelNoise'),
        build_cross_jaccard_table(grid_results['structural_noise'], 'B.2_StructuralNoise'),
    ], ignore_index=True)
    cj_path = os.path.join(RQ2_OUTPUT_DIR, "rq2_cross_jaccard.csv")
    cj_all.to_csv(cj_path, index=False)
    print(f"  Saved: {cj_path}")

    # ── JSON: raw null counts ─────────────────────────────────────────────
    null_counts_json = {}
    for op_key in ['label_noise', 'structural_noise']:
        null_counts_json[op_key] = {}
        for res in grid_results[op_key]:
            lv_key = f"level_{res['level']:.4f}"
            null_counts_json[op_key][lv_key] = res['null_counts']
    nc_path = os.path.join(RQ2_OUTPUT_DIR, "rq2_null_counts.json")
    with open(nc_path, 'w', encoding='utf-8') as f:
        json.dump(null_counts_json, f, indent=2)
    print(f"  Saved: {nc_path}")

    # ── JSON: full results ────────────────────────────────────────────────
    R_orig = orig_data['R_orig']
    full_json = {
        'rq2_version': '1.0',
        'log_name':    'BPI_17',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'block':          'B.1 (Label Noise) + B.2 (Structural Noise)',
            'methods':        ALL_METHODS,
            'shared_pool':    'M_all from Phase 0 DECLARE spec (fixed)',
            'B1_label_noise': {
                'operator':   'Flip ε fraction of labels independently',
                'grid':       LABEL_NOISE_LEVELS,
                'target':     'Discriminative signal (structural unaffected)',
            },
            'B2_structural_noise': {
                'operator':   'Local adjacent swaps at rate ρ per trace',
                'grid':       STRUCT_NOISE_LEVELS,
                'target':     'Temporal ordering signal (labels unaffected)',
            },
            'null_protocol':  'Doubly-null: σ_label ∘ σ_trace on L_ℓ',
            'P1_gate':        'Single gate: q_Hou ≤ α (analytic oracle in null replicates)',
            'DRVA_gate':      f'Per-rule p_Cecconi ≤ {ALPHA_DRVA} (no FDR correction)',
            'DM_gate':        f'|Δconf| ≥ τ* (calibrated, fixed across all levels)',
        },

        'config': {
            'ALPHA':          ALPHA,
            'ALPHA_DRVA':     ALPHA_DRVA,
            'B1_FULL':        B1_FULL,
            'B2_FULL':        B2_FULL,
            'B1_REAL':        B1_REAL,
            'B2_REAL':        B2_REAL,
            'B1_NULL':        B1_NULL,
            'B2_NULL':        B2_NULL,
            'PI_DRVA_REAL':   PI_DRVA_REAL,
            'PI_DRVA_NULL':   PI_DRVA_NULL,
            'B_NULL_ANCHOR':       B_NULL_ANCHOR,
            'B_NULL_INTERMEDIATE': B_NULL_INTERMEDIATE,
            'BASE_SEED':      BASE_SEED,
            'tau_star_DM':    orig_data['tau_star'],
            'tau_min_DM':     orig_data['tau_min'],
            'c_null':         float(_C_NULL),
            'f_null':         float(_F_NULL),
            'W_DISC':         W_DISC,
            'W_STRUCT':       W_STRUCT,
        },

        'original_data': {
            'R_obs': {mth: int(R_orig[mth]) for mth in ALL_METHODS},
        },

        'speed_optimisations_applied': {
            'S1_holds_precomputed_once':   True,
            'S2_analytic_Hou_for_perturbed': True,
            'S3_reduced_B_null_intermediate': B_NULL_INTERMEDIATE,
            'S4_reduced_B1_B2_null':       f'B1={B1_NULL}, B2={B2_NULL}',
            'S5_level_parallelism':        f'n_jobs={N_JOBS}',
            'S6_vectorised_delta_conf':    True,
        },

        'timing': {'total_seconds': total_wall},

        'label_noise_results':      [
            {k: v for k, v in r.items() if k != 'S_pert'}
            for r in grid_results['label_noise']
        ],
        'structural_noise_results': [
            {k: v for k, v in r.items() if k != 'S_pert'}
            for r in grid_results['structural_noise']
        ],
    }

    json_path = os.path.join(RQ2_OUTPUT_DIR, "rq2_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — SUMMARY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def _print_degradation_table(level_results: list, operator_label: str) -> None:
    """Print a compact degradation summary for one operator."""
    print(f"\n  {'─'*110}")
    print(f"  {operator_label}")
    print(f"  {'Level':>6s}  "
          f"{'Method':20s}  "
          f"{'R_obs':>6s}  {'Jaccard':>7s}  {'Gained':>6s}  {'Lost':>6s}  "
          f"{'FDR_emp':>8s}  {'CI':>22s}  {'Pass?':>6s}")
    print(f"  {'─'*110}")
    for res in level_results:
        lv = res['level']
        for mth in ALL_METHODS:
            jm  = res['jaccard'][mth]
            fdr = res['fdr_metrics'][mth]
            ci  = f"[{fdr['FDR_CI_lower']:.3f},{fdr['FDR_CI_upper']:.3f}]"
            vrd = "PASS" if fdr['controls_FDR'] else "FAIL"
            print(
                f"  {lv:>6.2f}  {mth:20s}  "
                f"{res['R_obs'][mth]:>6d}  {jm['Jaccard']:>7.4f}  "
                f"{jm['Gained']:>6d}  {jm['Lost']:>6d}  "
                f"{fdr['FDR_emp']:>8.4f}  {ci:>22s}  {vrd:>6s}"
            )
    print(f"  {'─'*110}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ2 — SPECIFICATION QUALITY DEGRADATION: BPI CHALLENGE 2017")
    print("Block B.1 (Label Noise) + B.2 (Structural Noise)")
    print("Three methods: P1 (Hou-Storey) | DRVA | DeclareMiner")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  B.1 ε grid: {LABEL_NOISE_LEVELS}")
    print(f"  B.2 ρ grid: {STRUCT_NOISE_LEVELS}")
    print(f"  α(P1)={ALPHA}  α_DRVA={ALPHA_DRVA}")
    print(f"  B_null(anchor)={B_NULL_ANCHOR}  B_null(intermediate)={B_NULL_INTERMEDIATE}")
    print(f"  B1_null={B1_NULL}  B2_null={B2_NULL}  PI_DRVA_null={PI_DRVA_NULL}")
    print(f"  Oracle: c={_C_NULL:.3f}, f={_F_NULL:.3f}  (rho_sd=0 under double-null)")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")
    print(f"\n  Speed optimisations enabled:")
    print(f"    S1: holds precomputed once per level")
    print(f"    S2: analytic Hou p-values for perturbed-log P1 runs (no calibration)")
    print(f"    S3: B_null={B_NULL_INTERMEDIATE} for intermediate levels")
    print(f"    S4: reduced B1={B1_NULL}, B2={B2_NULL}, PI_DRVA={PI_DRVA_NULL}")
    print(f"    S5: level-level parallelism (n_jobs={N_JOBS})")
    print(f"    S6: vectorised Δconf from holds (DM + DRVA null replicates)")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    # ── Section 1: original data ──────────────────────────────────────────
    orig_data = run_original_data(n_workers=N_JOBS)

    # ── Section 2: perturbation grid ─────────────────────────────────────
    grid_results = run_perturbation_grid(
        case_data_orig  = orig_data['case_data'],
        candidates_all  = orig_data['candidates_all'],
        case_ids_sorted = orig_data['case_ids_sorted'],
        labels_orig     = orig_data['labels_orig'],
        S_orig          = orig_data['S_orig'],
        tau_star        = orig_data['tau_star'],
        tau_min         = orig_data['tau_min'],
        n_jobs          = N_JOBS,
    )

    # ── Section 3: save outputs ───────────────────────────────────────────
    total_wall = time.time() - t_total

    print("\n" + "=" * 100)
    print("SECTION 3 — SAVING OUTPUTS")
    print("=" * 100)
    save_all_outputs(grid_results, orig_data, total_wall)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("RQ2 — BPI_17 COMPLETE")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")
    print(f"    rq2_label_noise_metrics.csv")
    print(f"    rq2_structural_noise_metrics.csv")
    print(f"    rq2_cross_jaccard.csv")
    print(f"    rq2_null_counts.json")
    print(f"    rq2_results.json")

    _print_degradation_table(grid_results['label_noise'],      "B.1 — Label Noise  (ε)")
    _print_degradation_table(grid_results['structural_noise'], "B.2 — Structural Noise  (ρ)")

    print(f"\n{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 Specification Degradation — BPI Challenge 2017  "
            "(B.1 Label Noise + B.2 Structural Noise)"
        )
    )
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Parallel workers for level grid (-1=all cores, default:{N_JOBS})',
    )
    parser.add_argument(
        '--alpha', type=float, default=ALPHA,
        help=f'P1 FDR level (default:{ALPHA})',
    )
    parser.add_argument(
        '--alpha-drva', type=float, default=ALPHA_DRVA,
        help=f'DRVA per-rule level (default:{ALPHA_DRVA})',
    )
    parser.add_argument(
        '--b-null-anchor', type=int, default=B_NULL_ANCHOR,
        help=f'Null replicates for anchor levels (default:{B_NULL_ANCHOR})',
    )
    parser.add_argument(
        '--b-null-intermediate', type=int, default=B_NULL_INTERMEDIATE,
        help=f'Null replicates for intermediate levels (default:{B_NULL_INTERMEDIATE})',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Smoke test: 3 levels only, B_null=2, minimal budgets',
    )
    args = parser.parse_args()

    if args.dry_run:
        LABEL_NOISE_LEVELS  = [0.0, 0.10, 0.50]
        STRUCT_NOISE_LEVELS = [0.0, 0.30, 1.00]
        B_NULL_ANCHOR       = 2
        B_NULL_INTERMEDIATE = 2
        B1_NULL             = 30
        B2_NULL             = 20
        PI_DRVA_NULL        = 20
        B1_REAL             = 100
        B2_REAL             = 50
        PI_DRVA_REAL        = 100
        B_NULL_FULL         = 5
        B1_NULL_FULL        = 10
        B2_NULL_FULL        = 10
        print("*** DRY-RUN MODE: 3 levels, B_null=2, minimal budgets ***")
    else:
        B_NULL_ANCHOR       = args.b_null_anchor
        B_NULL_INTERMEDIATE = args.b_null_intermediate

    N_JOBS     = args.n_jobs
    ALPHA      = args.alpha
    ALPHA_DRVA = args.alpha_drva

    main()