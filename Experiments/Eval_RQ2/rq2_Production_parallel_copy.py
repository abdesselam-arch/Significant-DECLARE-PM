#!/usr/bin/env python3
"""
rq2_Production_parallel.py  —  RQ2 Specification Quality Degradation Analysis
==========================================================================
Block B.3: Joint Label × Structural Noise Perturbation (2-D Cartesian grid)

Three methods on shared M_all: P1 (Hou-Storey) | DRVA | DeclareMiner

RESEARCH QUESTION
-----------------
RQ2: As signal is progressively corrupted along BOTH noise axes simultaneously,
how do P1, DRVA, and DeclareMiner differ in:
  (a) the size and composition of their discovered specifications,
  (b) the consistency of those specifications relative to the ground truth
      (FDR_ref, Precision, Recall, F1; Jaccard_rq2 procedure stability), and
  (c) the empirical FDR estimates under the doubly-null protocol (anchor cells)?

JOINT PERTURBATION OPERATOR
-----------------------------
B.3 — Joint noise  N_label(ε) ∘ N_struct(ρ):

    The two operators act on orthogonal signal components of the same log.
    Their composition is applied as:
        L_{ε,ρ} = N_struct(ρ) ∘ N_label(ε)(L)

    Because label and trace order are independent case attributes, the order
    of composition does not affect holds computation or statistical analysis.

    N_label(ε): Flip outcome label of each case independently with prob ε.
                Destroys discriminative signal (p_disc channel).
    N_struct(ρ): For each trace, select round(ρ·(n−1)) adjacent pairs without
                 replacement and swap each. Destroys temporal ordering (p_struct).

    2-D grid G = {(ε_i, ρ_j)}:
        ε ∈ {0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50}  (7 levels)
        ρ ∈ {0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00}  (7 levels)
        → 49 cells; (0,0) is established as anchor bootstrap (S_orig_rq2).

    Anchor cells (full doubly-null + S_orig_rq2 baseline): four corners:
        (0.00, 0.00), (0.50, 0.00), (0.00, 1.00), (0.50, 1.00)

HOLDS REUSE STRATEGY
---------------------
Holds depend only on trace structure, not on class labels.
    ρ = 0.00 : reuse holds_orig (no structural perturbation).
    ρ > 0.00 : precompute holds once per ρ row, shared across all ε in that row.
This reduces holds computation from 49 to 1 (ρ=0) + 6 (ρ>0) = 7 total.

REFERENCE SETS
--------------
  S_orig_full   — Full P1 pipeline, empirical p̃_Hou, B_null=200.
                  Ground truth. Fixed across all 49 cells.
                  Defines TP/FP/FN for FDR_ref and Recall.

  S_orig_rq2    — RQ2 empirical Phipson-Smyth baseline at (ε=0, ρ=0).
                  Single baseline for all 49 cells (not per-operator).
                  Jaccard_rq2 = 1.0 at (0,0) by construction.

  S_pert[ε,ρ]  — RQ2 empirical Phipson-Smyth discovery at grid cell (ε,ρ).

PRIMARY METRICS PER (METHOD, CELL)
------------------------------------
    FDR_ref       FP / R_obs             (reference-anchored false discovery rate)
    Precision     TP / R_obs  = 1 − FDR_ref
    Recall        TP / R_full             (power under noise)
    F1            Harmonic mean of Precision and Recall.
    Jaccard_rq2   |S_pert ∩ S_orig_rq2| / |S_pert ∪ S_orig_rq2|
                  Procedure stability relative to (0,0) baseline.

SECONDARY METRICS PER (METHOD, CELL)
--------------------------------------
    FP_over_Rfull  FP / R_full   (oracle-normalised overcall burden; ≠ FDR)
    FN_over_Rfull  FN / R_full   = 1 − Recall  (oracle-normalised miss rate)
    Jaccard_full   |S_pert ∩ S_orig_full| / |S_pert ∪ S_orig_full|
    FDR_emp        Doubly-null empirical FDR (anchor cells only).

SCIENTIFIC INTERPRETATION OF FDR METRICS
------------------------------------------
    FDR_ref = FP / R_obs  answers: "Among my discoveries, what fraction is spurious?"
    FP_over_Rfull = FP / R_full  answers: "How large is the spurious set vs. oracle?"
    These measure different failure modes; FP_over_Rfull is NOT called FDR_ref.

HEATMAP OUTPUT
---------------
The primary scientific deliverable is a 7×7 FDR_ref heatmap per method.
Iso-contour lines C_α = {(ε,ρ) : FDR_ref(ε,ρ) = α} define the noise
robustness boundary at FDR level α. Robustness score per method:
    RobustnessScore = Area{(ε,ρ) : FDR_ref(ε,ρ) ≤ 0.05}

INTERACTION SURFACE
--------------------
Define Δ_int(ε,ρ) = FDR_ref(ε,ρ) − [FDR_ref(ε,0) + FDR_ref(0,ρ) − FDR_ref(0,0)].
Δ_int ≈ 0 → additive (null hypothesis; consistent with Hou weighted combination).
Δ_int > 0 → superadditive (simultaneous loss is disproportionate).
Δ_int < 0 → subadditive (fragile patterns already excluded by one axis).

WEIGHT FAITHFULNESS
-------------------
P1 (full): W_DISC = B_label/(B_label+B2_test) = 1500/2500 = 0.60
           W_STRUCT = B2_test/(B_label+B2_test) = 1000/2500 = 0.40
RQ2 real:  W_DISC = B1_REAL/(B1_REAL+B2_REAL//2) = 1500/2500 = 0.60
           W_STRUCT = (B2_REAL//2)/(B1_REAL+B2_REAL//2) = 1000/2500 = 0.40
Budgets are IDENTICAL to P1 v9.0: T_Hou statistic, weights, and calibration
are fully faithful. (0,0) baseline recovers S_orig_full (Recall≈1.0).

NULL REPLICATE PROTOCOL (ANCHOR CELLS ONLY)
--------------------------------------------
For each replicate b on L_{ε,ρ}:
  1. Apply σ_trace to L_{ε,ρ}: fully shuffle activities within each trace.
  2. Apply σ_label to σ_trace(L_{ε,ρ}): permute class labels.
  3. Recompute holds on doubly-null log.
  4. Run all three methods → V_b^(m,ε,ρ).

P1 ORACLE IN NULL REPLICATES
------------------------------
Under the double-null (ρ_sd = 0), using P1-faithful budget weights:
    c = W_STRUCT_RQ2² + W_DISC_RQ2² = 0.40² + 0.60² = 0.52
    f = 8/c ≈ 3.846  (Satterthwaite; c·f = E[T_Hou] under H₀ᶜ)

OUTPUT FILES
-------------
    rq2_joint_metrics.csv                Long-format table (all 49 cells × methods).
    rq2_joint_fdrref_pivot_<method>.csv  7×7 FDR_ref heatmap matrix per method.
    rq2_joint_<metric>_pivot_<method>.csv  Heatmap pivots for Recall, F1, etc.
    rq2_joint_cross_jaccard.csv          Cross-method Jaccard per cell.
    rq2_joint_null_counts.json           Raw V_b arrays per (ε,ρ,method).
    rq2_joint_results.json               Full structured results for paper.

Version : 3.0 (joint noise design; scientifically faithful to P1 v9.0)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Hou (2005). Stat. Prob. Lett. 73:179-187.
Storey (2002). JRSS-B 64(3):479-498.
Gao (2023). arXiv:2310.06357.
Phipson & Smyth (2010). Stat. Appl. Genet. Mol. Biol. 9(1):Art.39.
Pepe & Fleming (1989). Biometrics 45:497-507.
Donoho & Jin (2004). Ann. Stat. 32(3):962-994.
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
from P1_SDSM.p1_Production_hou import (
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
    empirical_fisher_pvalue,
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
from BaselinesRQ1.DRVA_Production import (
    run_drva,
    run_drva_on_doubly_null_log,
    DRVA_CONFIG,
)

# ── DeclareMiner baseline ─────────────────────────────────────────────────
from BaselinesRQ1.DeclareMiner_Production import (
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
RQ2_OUTPUT_DIR = "RQ2_Production_Joint"

# ── 2-D Joint Perturbation Grid ───────────────────────────────────────────
# B.3: Joint noise N_label(ε) ∘ N_struct(ρ)
# Two orthogonal signal axes combined into one 7×7 Cartesian grid.
# Row axis (ε): destroys discriminative signal (p_disc channel).
# Column axis (ρ): destroys temporal ordering signal (p_struct channel).
JOINT_LABEL_LEVELS  = [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
JOINT_STRUCT_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

JOINT_GRID = [
    (eps, rho)
    for eps in JOINT_LABEL_LEVELS
    for rho in JOINT_STRUCT_LEVELS
]   # 49 cells

# Anchor cells: full doubly-null FDR_emp + S_orig_rq2 bootstrap.
# (0,0) is the origin anchor (is_anchor_bootstrap=True).
# Remaining three corners run doubly-null but not bootstrap.
ANCHOR_CELLS = frozenset({
    (0.00, 0.00),
    (JOINT_LABEL_LEVELS[-1], 0.00),
    (0.00, JOINT_STRUCT_LEVELS[-1]),
    (JOINT_LABEL_LEVELS[-1], JOINT_STRUCT_LEVELS[-1]),
})

# Backward-compat aliases used by legacy helpers kept in the file.
LABEL_NOISE_LEVELS  = JOINT_LABEL_LEVELS
STRUCT_NOISE_LEVELS = JOINT_STRUCT_LEVELS

# FDR level
ALPHA      = 0.05
ALPHA_DRVA = 0.01

# Original P1 run budget (full)
B1_FULL      = 1_500
B2_FULL      = 2_000
B_NULL_FULL  = 200
B1_NULL_FULL = 75
B2_NULL_FULL = 75

# Real-data P1 run budget — FAITHFULLY MATCHING P1 v9.0 (p1_Production_hou.py)
# B1_REAL = P1's CONFIG['B_label'] = 1500
# B2_REAL = P1's CONFIG['B_trace'] = 2000
# All weights, T_Hou statistic, and calibration procedure now identical to P1.
B1_REAL = 1500   # label perm — matches P1's CONFIG['B_label']
B2_REAL = 2000   # structural perm — matches P1's CONFIG['B_trace']

# ── Budget-correct weights — now identical to full P1 ────────────────────
# Weights are precision-proportional (Pepe & Fleming 1989; Hou 2005 §4).
# With B1_REAL=1500, B2_REAL=2000 → B2_test=1000:
#   W_DISC   = 1500/2500 = 0.60  (matches P1)
#   W_STRUCT = 1000/2500 = 0.40  (matches P1)
_B2_TEST_RQ2  = B2_REAL // 2                                        # 1000
W_DISC_RQ2    = B1_REAL / (B1_REAL + _B2_TEST_RQ2)                  # 1500/2500 = 0.60
W_STRUCT_RQ2  = _B2_TEST_RQ2 / (B1_REAL + _B2_TEST_RQ2)            # 1000/2500 = 0.40

# Oracle parameters under double-null using P1-faithful weights (ρ_sd = 0)
# V_X = 4(0.40² + 0.60²) = 2.08 → c = 0.52, f = 8/2.08 ≈ 3.846
_C_NULL, _F_NULL = hou_satterthwaite_params(W_STRUCT_RQ2, W_DISC_RQ2, rho_sd=0.0)

# DRVA full run budget for real-data perturbed levels — matching original full run
PI_DRVA_REAL = 1000   # matches P1 full DRVA run (was 500)

# Null replicate inner budgets — matching P1's CONFIG['B1_null'/'B2_null']
B1_NULL      = 75    # label perm per null replicate — matches P1's B1_null
B2_NULL      = 75    # structural perm per null replicate — matches P1's B2_null
PI_DRVA_NULL = 100   # DRVA shuffleLog per null replicate

# B_null per level: doubly-null is secondary diagnostic, anchor levels only
# Production m_prime ≈ 195: theoretical minimum B_null ≈ 3120; cap at 200 (full P1 budget)
B_NULL_ANCHOR       = 200   # ℓ=0 and ℓ=max (anchor levels)
B_NULL_INTERMEDIATE = 0     # intermediate levels: skip doubly-null (FDR_ref is primary)

# Empirical calibration budget — matching P1's CONFIG['B_null'/'B1_null'/'B2_null']
# Resolution at B_NULL_REAL=200: 1/201 ≈ 0.005 ≪ α=0.05.  ✓
B_NULL_REAL  = 200   # matches P1's CONFIG['B_null']
B1_NULL_REAL = 75    # matches P1's CONFIG['B1_null']
B2_NULL_REAL = 75    # matches P1's CONFIG['B2_null'] (was 30)

# Base seed for RQ2 (offset from RQ1's 20260521)
BASE_SEED = 20260601

# Parallelism (over perturbation levels)
N_JOBS = -1
# Inner parallelism for calibration/null loops inside a joblib worker.
# Set to 1 when running under the outer parallel grid (avoids nested loky).
# Set to -1 for standalone SLURM array jobs (one (operator, level) per task).
INNER_N_JOBS = 1

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
    """
    if epsilon == 0.0:
        return case_data

    rng    = np.random.RandomState(seed)
    result = {}

    for cid, case in case_data.items():
        if rng.random() < epsilon:
            ci         = copy.copy(case)
            ci.outcome = 1 - case.outcome
            result[cid] = ci
        else:
            result[cid] = case

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

    At ρ = 0.0: traces unchanged (0 pairs selected).
    At ρ = 1.0: all (n-1) adjacent pairs selected and swapped.

    Labels are NOT modified; discriminative signal is unaffected.
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
# JOINT PERTURBATION OPERATOR
# ═══════════════════════════════════════════════════════════════════════════

def apply_joint_perturbation(
    struct_log: dict,
    eps: float,
    eps_seed: int,
) -> dict:
    """
    Joint operator N_label(ε) ∘ N_struct(ρ).

    struct_log is already structurally perturbed (ρ applied with a fixed seed
    per ρ row in the precomputation step of run_joint_grid).  This function
    applies only the label-noise layer (ε) on top.

    ε = 0.0 → returns struct_log unchanged (no copy).
    ε > 0.0 → flip each case label independently with probability ε.

    Because N_label acts only on outcome attributes (not on trace order),
    holds computed from struct_log remain valid for the joint log.
    """
    return apply_label_noise(struct_log, eps, eps_seed)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLY-NULL LOG BUILDER
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

    Under the double-null T_Hou ~ c·χ²_f using RQ2 budget weights.
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
# EMPIRICAL CALIBRATION — DOUBLY-NULL T_Hou MATRIX ON PERTURBED LOG
# ═══════════════════════════════════════════════════════════════════════════

def _one_calibration_replicate(
    b:                   int,
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    perm_labels_b:       np.ndarray,   # (n,) permuted labels for replicate b
    B1_null:             int,
    B2_null:             int,
    w_struct:            float,
    w_disc:              float,
    BASE:                int,
) -> np.ndarray:
    """One doubly-null calibration replicate → T_Hou^(b) vector shape (m,).

    Top-level (loky-safe). Called by _compute_doubly_null_tf_perturbed via Parallel.
    """
    rs  = BASE + 100_000 * b
    eps = 1e-300

    rng_trace = np.random.RandomState(rs + 200_000)
    null_cd   = {}
    for i, cid in enumerate(case_ids_sorted):
        ci_orig = case_data_perturbed[cid]
        ci      = copy.copy(ci_orig)
        ci.outcome = int(perm_labels_b[i])
        shuffled   = ci_orig.trace.copy()
        rng_trace.shuffle(shuffled)
        ci.trace          = shuffled
        ci.activity_index = precompute_activity_index(shuffled, case_id=cid)
        null_cd[cid]      = ci

    with _suppress():
        holds_null = compute_holds_by_case_batch(null_cd, candidates_all)

    with _suppress():
        disc_b = run_label_permutation_test(null_cd, candidates_all, holds_null, B1_null, rs)
    disc_b.pop('__null_delta_matrix__', None)
    p_disc_b = np.array([disc_b[spec]['p_two_sided'] for spec in candidates_all])

    D0_b, D1_b = split_by_class(null_cd)
    cid_set0   = set(D0_b.keys())
    cid_set1   = set(D1_b.keys())

    # Guard: degenerate split (extremely rare but possible at high ε with small n)
    if len(D0_b) < 5 or len(D1_b) < 5:
        # Return E[T_Hou] under H₀ᶜ — conservative fallback; does not introduce selection bias
        c_null, f_null = hou_satterthwaite_params(w_struct, w_disc, rho_sd=0.0)
        return np.full(len(candidates_all), c_null * f_null)

    # n_workers=1: intentional — this function runs inside Parallel (replicate-level);
    # nested loky processes are unsafe. Outer loop parallelises across replicates.
    with _suppress():
        st0_b = run_structural_permutation_test(
            D0_b, candidates_all, 0, B2_null, rs + 1, n_workers=1
        )
        st1_b = run_structural_permutation_test(
            D1_b, candidates_all, 1, B2_null, rs + 2, n_workers=1
        )

    p_t0 = np.array([
        st0_b[spec]['p_structural_test'] if spec in st0_b else 1.0
        for spec in candidates_all
    ])
    p_t1 = np.array([
        st1_b[spec]['p_structural_test'] if spec in st1_b else 1.0
        for spec in candidates_all
    ])

    m     = len(candidates_all)
    prev0 = np.zeros(m)
    prev1 = np.zeros(m)
    for i, spec in enumerate(candidates_all):
        h = holds_null[spec]
        p0, _, _ = compute_prevalence_from_holds(h, cid_set0)
        p1, _, _ = compute_prevalence_from_holds(h, cid_set1)
        prev0[i] = p0
        prev1[i] = p1
    dominant  = np.where(prev1 >= prev0, 1, 0)
    p_s_dom_b = np.where(dominant == 1, p_t1, p_t0)

    ps = np.clip(p_s_dom_b, eps, 1.0)
    pd = np.clip(p_disc_b,  eps, 1.0)
    return -2.0 * (w_struct * np.log(ps) + w_disc * np.log(pd))


def _compute_doubly_null_tf_perturbed(
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    labels_perturbed:    np.ndarray,   # (n,) labels of the PERTURBED log
    B_null:              int,
    B1_null:             int,
    B2_null:             int,
    w_struct:            float,
    w_disc:              float,
    seed:                int,
    n_jobs:              int = 1,
) -> np.ndarray:
    """
    Run B_null doubly-null replicates on a PERTURBED log and record T_Hou^(b).

    Applies σ_trace ∘ σ_label to case_data_perturbed (not the original log),
    producing empirical null T_Hou values calibrated for the residual structural
    signal in the perturbed data. This is the correct calibration target for
    measuring FDR_ref relative to S_orig_full.

    Protocol per replicate b:
    1. σ_label: permute labels_perturbed (marginals preserved).
    2. σ_trace: fully shuffle activities within each trace of L_ℓ.
       Together these ensure p_s^(b) ~ U(0,1) and p_d^(b) ~ U(0,1)
       regardless of residual signal in the perturbed log.
    3. Recompute holds on doubly-null log.
    4. Run label perm (B1_null resamples) → p_disc^(b).
    5. Run structural perm (B2_null per class) → p_struct_dom^(b).
    6. T_Hou^(b)(i) = -2[w_struct·ln p_s_dom^(b)(i) + w_disc·ln p_d^(b)(i)]
       Uses THE SAME weights as the observed T_Hou_obs, so the empirical
       p̃_Hou(i) = (1 + #{b: T_Hou^(b)(i) ≥ T_Hou_obs(i)}) / (B_null + 1)
       is stochastically super-uniform by exchangeability (Phipson & Smyth 2010).

    n_jobs=1 (INNER_N_JOBS default) avoids nested parallelism under the outer
    joblib grid. Set n_jobs=-1 for standalone SLURM array jobs.

    Returns:
        tf_null_matrix: (B_null, m) array of null T_Hou values.
    """
    BASE = seed + 500_000

    rng_label = np.random.RandomState(BASE)
    perm_labels_all = np.stack(
        [rng_label.permutation(labels_perturbed) for _ in range(B_null)], axis=0
    )   # (B_null, n)

    rows = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_calibration_replicate)(
            b, case_data_perturbed, candidates_all, case_ids_sorted,
            perm_labels_all[b],
            B1_null, B2_null, w_struct, w_disc, BASE,
        )
        for b in range(B_null)
    )
    return np.stack(rows, axis=0)   # (B_null, m)


# ═══════════════════════════════════════════════════════════════════════════
# P1 EMPIRICAL RUN ON PERTURBED LOG
# (Empirical p̃_Hou via doubly-null calibration; budget-correct weights)
# ═══════════════════════════════════════════════════════════════════════════

def run_p1_on_perturbed_log(
    case_data_perturbed: dict,
    candidates_all:      list,
    B1:                  int,
    B2:                  int,
    alpha:               float,
    seed:                int,
    case_ids_sorted:     list,
    labels_perturbed:    np.ndarray,
    B_null:              int  = B_NULL_REAL,
    B1_null_real:        int  = B1_NULL_REAL,
    B2_null_real:        int  = B2_NULL_REAL,
    holds_precomputed:   dict = None,
) -> dict:
    """
    Full P1 pipeline on a perturbed log with empirical p̃_Hou calibration.

    WEIGHT FAITHFULNESS: Weights are recomputed from the actual B1, B2 budgets
    (precision-proportional, Pepe & Fleming 1989; Hou 2005 §4). With B1=1500,
    B2=2000 (identical to P1 v9.0), W_DISC=0.60 and W_STRUCT=0.40, so T_Hou
    is identical to the full P1 statistic.

    EMPIRICAL CALIBRATION: This function runs B_null=B_NULL_REAL (=200)
    doubly-null replicates on the perturbed log to compute empirical p̃_Hou(i)
    via Phipson-Smyth (2010):
        p̃_Hou(i) = (1 + #{b: T_Hou^(b)(i) ≥ T_Hou_obs(i)}) / (B_null + 1)
    This is the same gate used by the full P1 pipeline (both B_null=200; budgets match).
    Gate type (empirical, not analytic) matches exactly, ensuring the ε=0 baseline
    is faithful (Recall≈1 by construction).

    Gate: single q̃_Hou ≤ α applied to p̃_Hou over scope-filtered patterns.
    Analytic p_hou is computed and stored for reference only.
    """
    m = len(candidates_all)

    # Budget-correct precision-proportional weights for THIS run
    B2_test  = B2 // 2
    w_disc   = B1 / (B1 + B2_test)
    w_struct = B2_test / (B1 + B2_test)

    # Step 1: holds
    if holds_precomputed is not None:
        holds = holds_precomputed
    else:
        with _suppress():
            holds = compute_holds_by_case_batch(
                case_data_perturbed, candidates_all
            )

    # Step 2: label permutation test → p_disc
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

    # Step 5a: T_Hou_obs with budget-correct weights
    rho_sd = estimate_rho_sd(p_struct_dom, p_disc)
    c_h, f_h = hou_satterthwaite_params(w_struct, w_disc, rho_sd)
    tf_obs   = hou_combination_statistic(p_struct_dom, p_disc, w_struct, w_disc)

    # Step 5b: Analytic p_Hou — stored for reference; NOT used as primary gate
    p_hou_analytic = np.clip(stats.chi2.sf(tf_obs / c_h, df=f_h), 1e-300, 1.0)

    # Step 5c: Empirical p̃_Hou via B_null doubly-null replicates on the perturbed log.
    # This is the primary gate input — same gate type as full P1 (Phipson-Smyth 2010).
    tf_null_matrix = _compute_doubly_null_tf_perturbed(
        case_data_perturbed = case_data_perturbed,
        candidates_all      = candidates_all,
        case_ids_sorted     = case_ids_sorted,
        labels_perturbed    = labels_perturbed,
        B_null              = B_null,
        B1_null             = B1_null_real,
        B2_null             = B2_null_real,
        w_struct            = w_struct,
        w_disc              = w_disc,
        seed                = seed + 10_000,
        n_jobs              = INNER_N_JOBS,
    )
    p_tilde_hou = empirical_fisher_pvalue(tf_obs, tf_null_matrix)

    # Step 6: sample-split scope filter
    structural_idx = [
        i for i in range(m)
        if min(p_screen_c0[i], p_screen_c1[i]) <= alpha
    ]
    m_prime = len(structural_idx)

    # Step 7: Adaptive Storey + single gate on empirical p̃_Hou (primary)
    is_sig = np.zeros(m, dtype=bool)
    q_hou  = np.ones(m)

    if m_prime > 0:
        p_mp     = p_tilde_hou[structural_idx]
        pi0_b, _ = adaptive_storey_pi0(p_mp, q=alpha)
        q_mp     = storey_qvalue(p_mp, pi0_b)
        for rank, orig_i in enumerate(structural_idx):
            q_hou[orig_i]  = q_mp[rank]
            is_sig[orig_i] = bool(q_mp[rank] <= alpha)

    return {
        'is_significant':  is_sig,
        'S_set':           frozenset(
                               candidates_all[i] for i in range(m) if is_sig[i]
                           ),
        'R_obs':           int(is_sig.sum()),
        'p_disc':          p_disc,
        'p_struct_dom':    p_struct_dom,
        'p_tilde_hou':     p_tilde_hou,      # primary gate input (empirical)
        'p_hou_analytic':  p_hou_analytic,   # reference only
        'q_hou':           q_hou,
        'holds':           holds,
        'null_delta_mat':  null_delta_mat,
        'm_prime':         m_prime,
        'structural_idx':  structural_idx,
        'w_disc':          w_disc,
        'w_struct':        w_struct,
        'rho_sd':          rho_sd,
        'B_null_used':     B_null,
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

    Uses precomputed holds for vectorised Δconf computation.
    τ* is NOT recalibrated — we use the value calibrated on the original data.
    """
    m = len(candidates_all)

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

    rejected = dm_apply_threshold(
        conf0, conf1, supp0, supp1,
        tau_delta_conf = tau_star,
        tau_min        = tau_min,
    )

    S_set = frozenset(
        candidates_all[i]
        for i in range(m)
        if rejected[i]
    )

    return {
        'S_set':       S_set,
        'R_obs':       int(rejected.sum()),
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
    Gained = |S_perturbed setminus S_ref|
    Lost   = |S_ref setminus S_perturbed|
    """
    inter  = len(S_ref & S_perturbed)
    union  = len(S_ref | S_perturbed)
    jaccard = inter / union if union > 0 else 1.0

    return {
        'Jaccard':        jaccard,
        'Gained':         len(S_perturbed - S_ref),
        'Lost':           len(S_ref - S_perturbed),
        'n_ref':          len(S_ref),
        'n_perturbed':    len(S_perturbed),
        'n_intersection': inter,
    }


def compute_cross_jaccard(
    S_p1: frozenset,
    S_drva: frozenset,
    S_dm: frozenset,
) -> dict:
    """Cross-method Jaccard similarity at a given perturbation level."""
    def _j(A, B):
        u = len(A | B)
        return len(A & B) / u if u > 0 else 1.0

    return {
        'J_P1_DRVA': _j(S_p1, S_drva),
        'J_P1_DM':   _j(S_p1, S_dm),
        'J_DRVA_DM': _j(S_drva, S_dm),
    }


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE-ANCHORED METRICS  (primary RQ2 metrics)
# ═══════════════════════════════════════════════════════════════════════════

def compute_reference_metrics(
    S_pert:      frozenset,   # discoveries at level ℓ
    S_orig_full: frozenset,   # ground truth (full P1 pipeline, empirical p̃_Hou)
    S_orig_rq2:  frozenset,   # RQ2 procedure baseline at ε=0/ρ=0
) -> dict:
    """
    Reference-anchored performance metrics for RQ2 degradation analysis.

    Under the partial-signal regime, doubly-null FDR_emp overestimates true FDR
    because it counts patterns with residual real-data signal as false positives.
    Instead we use S_orig_full (full P1 pipeline, B_null=200) as ground truth:

        TP = |S_pert ∩ S_orig_full|   (correct recoveries)
        FP = |S_pert \ S_orig_full|   (spurious discoveries)
        FN = |S_orig_full \ S_pert|   (missed recoveries)

    FDR_ref = FP / R_obs           (complement of Precision)
    Recall  = TP / |S_orig_full|   (power under noise)

    INTERPRETABILITY NOTE: These metrics are interpretable AS degradation metrics
    (i.e., the ε=0/ρ=0 baseline is approximately FDR_ref≈0, Recall≈1) only when
    S_pert is produced by the empirical P1 pipeline (same gate type as S_orig_full).
    If S_pert were produced by the analytic oracle, FDR_ref(ε=0) ≈ 0.07 and
    Recall(ε=0) ≈ 0.07 would reflect oracle approximation error, not noise effects.
    The empirical calibration in run_p1_on_perturbed_log (B_null=B_NULL_REAL (=200)) ensures the
    gate type matches and the ε=0 baseline is meaningful.

    Jaccard_rq2 measures stability of the RQ2 procedure relative to its own
    ε=0/ρ=0 baseline (S_orig_rq2). Jaccard_rq2 = 1.0 at ε=0/ρ=0 by construction.

    Edge cases:
        R_obs = 0: Precision = NaN, FDR_ref = NaN (undefined, NOT 0 or 1).
        |S_orig_full| = 0: Recall = NaN (degenerate experiment).
    """
    R_obs  = len(S_pert)
    R_full = len(S_orig_full)
    R_rq2  = len(S_orig_rq2)

    TP = len(S_pert & S_orig_full)
    FP = len(S_pert - S_orig_full)
    FN = len(S_orig_full - S_pert)

    # Precision / FDR_ref
    if R_obs == 0:
        precision = float('nan')
        fdr_ref   = float('nan')
        estimable = False
    else:
        precision = TP / R_obs
        fdr_ref   = FP / R_obs
        estimable = True

    # Recall / Power
    recall = TP / R_full if R_full > 0 else float('nan')

    # F1
    if (not estimable) or (precision != precision) or (recall != recall):
        f1 = float('nan')
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    # Jaccard_rq2 (procedure stability vs own ε=0/ρ=0 baseline)
    union_rq2  = len(S_pert | S_orig_rq2)
    inter_rq2  = len(S_pert & S_orig_rq2)
    jaccard_rq2 = inter_rq2 / union_rq2 if union_rq2 > 0 else 1.0

    # Jaccard_full (combined ground-truth similarity)
    union_full   = len(S_pert | S_orig_full)
    jaccard_full = TP / union_full if union_full > 0 else 1.0

    # Secondary oracle-normalised burden metrics.
    # FP_over_Rfull: spurious discoveries relative to oracle size.
    #   Answers "how large is the spurious mass vs. ground truth?" (not FDR).
    # FN_over_Rfull: = 1 − Recall (oracle-normalised miss rate).
    fp_over_rfull = FP / R_full if R_full > 0 else float('nan')
    fn_over_rfull = FN / R_full if R_full > 0 else float('nan')

    return {
        # Primary reference-anchored metrics
        'FDR_ref':       fdr_ref,
        'Precision':     precision,
        'Recall':        recall,
        'F1':            f1,
        'TP':            TP,
        'FP':            FP,
        'FN':            FN,
        'R_obs':         R_obs,
        'R_full':        R_full,
        'Jaccard_rq2':   jaccard_rq2,
        'Jaccard_full':  jaccard_full,
        'Gained_rq2':    len(S_pert - S_orig_rq2),
        'Lost_rq2':      len(S_orig_rq2 - S_pert),
        'estimable':     estimable,
        'reliable':      estimable and (R_obs >= 10),
        # Secondary oracle-normalised burden metrics
        'FP_over_Rfull': fp_over_rfull,
        'FN_over_Rfull': fn_over_rfull,
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


# Theoretical minimum B_null for Storey q-value gate power (documentation only):
#   From q_(1) = π̂₀ · m' · p̃_min / 1 ≤ α  and  p̃_min = 1/(B_null+1):
#       B_null_min = ⌈π̂₀ · m' / α⌉ − 1
#   Production (m'≈195, π̂₀=0.8, α=0.05): min ≈ 3119 — infeasible per level.
#   → B_NULL_REAL=200 matches P1's B_null; resolution 1/201 ≈ 0.005 ≪ α.  ✓
#   → B_NULL_ANCHOR=200 is a fixed constant; no adaptive formula is applied.


def _null_fdr_skipped(reason: str) -> dict:
    return {
        'FDR_emp':       float('nan'),
        'FDR_CI_lower':  float('nan'),
        'FDR_CI_upper':  float('nan'),
        'E_V_b':         float('nan'),
        'FWER_emp':      float('nan'),
        'controls_FDR':  None,
        'estimable':     False,
        'skipped_reason': reason,
    }


def _compute_fdr_from_null(
    null_counts: np.ndarray,
    R_obs: int,
    alpha_nominal: float,
) -> dict:
    ev = float(np.mean(null_counts))
    if R_obs == 0:
        # FDR_emp = E[V_b]/R_obs is undefined when R_obs=0.
        # Do NOT substitute max(R_obs,1) — that would produce FDR_emp > 1
        # (e.g., 1.76) which is numerically nonsensical and misleads readers.
        return {
            'FDR_emp':       float('nan'),
            'FDR_CI_lower':  float('nan'),
            'FDR_CI_upper':  float('nan'),
            'E_V_b':         ev,
            'FWER_emp':      float(np.mean(null_counts > 0)),
            'controls_FDR':  None,
            'estimable':     False,
            'skipped_reason': 'R_obs=0: FDR_emp undefined',
        }
    arr = null_counts.astype(float) / R_obs
    fdr = float(np.mean(arr))
    try:
        lo, hi = _bca_ci(arr)
    except Exception:
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        'FDR_emp':       fdr,
        'FDR_CI_lower':  lo,
        'FDR_CI_upper':  hi,
        'E_V_b':         ev,
        'FWER_emp':      float(np.mean(null_counts > 0)),
        'controls_FDR':  bool(fdr <= alpha_nominal),
        'estimable':     True,
        'skipped_reason': None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NULL REPLICATE RUNNER FOR ONE LEVEL  (anchor levels only)
# ═══════════════════════════════════════════════════════════════════════════

def _one_anchor_replicate(
    b:                   int,
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    perm_labels_b:       np.ndarray,   # (n,) permuted labels for replicate b
    B1_null:             int,
    B2_null:             int,
    alpha:               float,
    alpha_drva:          float,
    tau_star_dm:         float,
    tau_min_dm:          float,
    base_seed:           int,
    drva_cfg_null:       dict,
) -> dict:
    """One doubly-null anchor replicate → {METHOD: int} false positive counts.

    Top-level (loky-safe). Called by run_null_replicates_for_level via Parallel.
    P1 oracle uses P1-faithful budget weights (W_DISC_RQ2=0.60, W_STRUCT_RQ2=0.40)
    and the corresponding _C_NULL/_F_NULL oracle parameters under double-null rho_sd=0.

    GATE NOTE: Real-data run uses empirical p̃_Hou gate (Phipson-Smyth, B_null=200).
    This function uses the analytic Satterthwaite oracle chi2.sf(T/c, f) with
    ρ_sd=0 (double-null assumption). This introduces a small conservative bias in
    FDR_emp relative to the empirical gate: the oracle slightly over-rejects,
    inflating FDR_emp by O(1/B_null). At B_null=200 this is ≈ 0.005 — within
    the BCa CI width reported by _compute_fdr_from_null.
    FDR_emp is the SECONDARY metric; FDR_ref is the primary degradation measure.
    """
    m        = len(candidates_all)
    rs_trace = base_seed + 100_000 + b
    rs_p1    = base_seed + 200_000 + b
    rs_drva  = base_seed + 300_000 + b

    null_cd = _build_doubly_null_log(
        case_data_perturbed, case_ids_sorted, perm_labels_b, trace_seed=rs_trace,
    )

    # Guard: degenerate split (extremely rare but possible at high ε with small n)
    D0_b, D1_b = split_by_class(null_cd)
    if len(D0_b) < 5 or len(D1_b) < 5:
        # Return zero false positives — conservative choice; does not inflate FDR_emp
        return {METHOD_P1: 0, METHOD_DRVA: 0, METHOD_DM: 0}

    with _suppress():
        holds_null = compute_holds_by_case_batch(null_cd, candidates_all)

    with _suppress():
        disc_b = run_label_permutation_test(
            null_cd, candidates_all, holds_null, B1_null, rs_p1
        )
    disc_b.pop('__null_delta_matrix__', None)
    p_disc_b = np.array([disc_b[spec]['p_two_sided'] for spec in candidates_all])

    D0_b, D1_b = split_by_class(null_cd)
    # n_workers=1: intentional — this function runs inside Parallel (replicate-level);
    # nested loky processes are unsafe. Outer loop in run_null_replicates_for_level parallelises.
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

    delta_obs_b = np.array([disc_b[spec]['delta_obs'] for spec in candidates_all])
    dom_b       = np.where(delta_obs_b >= 0.0, 1, 0)
    p_struct_b  = np.where(dom_b == 1, p_t1, p_t0)

    tf_b    = hou_combination_statistic(p_struct_b, p_disc_b, W_STRUCT_RQ2, W_DISC_RQ2)
    p_hou_b = np.clip(stats.chi2.sf(tf_b / _C_NULL, df=_F_NULL), 1e-300, 1.0)

    sidx_b = [i for i in range(m) if min(p_sc0[i], p_sc1[i]) <= alpha]
    n_p1_b = 0
    if sidx_b:
        p_mp     = p_hou_b[sidx_b]
        pi0_b, _ = adaptive_storey_pi0(p_mp, q=alpha)
        q_b      = storey_qvalue(p_mp, pi0_b)
        n_p1_b   = int(np.sum(q_b <= alpha))

    with _suppress():
        n_drva_b = run_drva_on_doubly_null_log(
            null_case_data  = null_cd,
            candidates_all  = candidates_all,
            alpha           = alpha_drva,
            replicate_seed  = rs_drva,
            config          = drva_cfg_null,
            holds_all       = holds_null,
        )
    with _suppress():
        n_dm_b = run_declareminer_on_doubly_null_log(
            null_case_data = null_cd,
            candidates_all = candidates_all,
            tau_star       = tau_star_dm,
            tau_min        = tau_min_dm,
            holds_all      = holds_null,
        )

    return {METHOD_P1: n_p1_b, METHOD_DRVA: n_drva_b, METHOD_DM: n_dm_b}


def run_null_replicates_for_level(
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    labels_perturbed:    np.ndarray,
    B_null:              int,
    B1_null:             int,
    B2_null:             int,
    pi_drva_null:        int,
    alpha:               float,
    alpha_drva:          float,
    tau_star_dm:         float,
    tau_min_dm:          float,
    base_seed:           int,
    n_jobs:              int = 1,
) -> dict:
    """
    Run B_null doubly-null replicates on a PERTURBED log.

    Each replicate applies σ_label ∘ σ_trace to case_data_perturbed.
    Every rejection is a false positive by construction.

    P1 oracle uses RQ2 budget weights (W_DISC_RQ2, W_STRUCT_RQ2) and the
    corresponding _C_NULL/_F_NULL oracle parameters — NOT the full P1 weights.

    This estimates FDR of the RQ2 empirical Phipson-Smyth procedure under complete null.
    Valid for null validity assessment at anchor levels.
    NOT valid as degradation metric under partial signal (use FDR_ref instead).

    n_jobs=1 (INNER_N_JOBS default) avoids nested parallelism under the outer
    joblib grid. Set n_jobs=-1 for standalone SLURM array jobs.

    Returns:
        dict: {METHOD_P1: (B,) int, METHOD_DRVA: (B,) int, METHOD_DM: (B,) int}
    """
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

    reps = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_anchor_replicate)(
            b, case_data_perturbed, candidates_all, case_ids_sorted,
            perm_labels_all[b],
            B1_null, B2_null, alpha, alpha_drva, tau_star_dm, tau_min_dm,
            base_seed, drva_cfg_null,
        )
        for b in range(B_null)
    )

    counts = {mth: np.zeros(B_null, dtype=int) for mth in ALL_METHODS}
    for b, r in enumerate(reps):
        for mth in ALL_METHODS:
            counts[mth][b] = r[mth]
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# JOINT DRAW RUNNER  (called by analyze_joint_cell with retry logic)
# ═══════════════════════════════════════════════════════════════════════════

def _run_methods_on_joint_draw(
    eps:          float,
    rho:          float,
    eps_seed:     int,
    p1_seed:      int,
    struct_log:   dict,       # structurally pre-perturbed log for this ρ row
    holds_for_rho: dict,      # holds precomputed for this ρ row (reused across ε)
    candidates_all:   list,
    case_ids_sorted:  list,
    alpha:            float,
    alpha_drva:       float,
    tau_star_dm:      float,
    tau_min_dm:       float,
) -> dict:
    """
    Apply joint perturbation N_label(ε) ∘ N_struct(ρ) and run all three methods.

    struct_log already has ρ applied (precomputed per ρ row in run_joint_grid).
    This function applies only the label-noise layer (ε_seed) on top, then
    runs P1 / DRVA / DM with the shared holds_for_rho (valid because label
    noise does not change trace structure or DECLARE satisfaction).

    Called by analyze_joint_cell with retry logic on R_obs=0: retrying only
    changes eps_seed (structural perturbation is fixed per ρ row).
    """
    case_data_pert = apply_joint_perturbation(struct_log, eps, eps_seed)

    labels_pert = np.array([
        case_data_pert[cid].outcome for cid in case_ids_sorted
    ], dtype=np.int8)

    holds_pert = holds_for_rho   # reuse — valid for all ε at fixed ρ

    p1_res = run_p1_on_perturbed_log(
        case_data_perturbed = case_data_pert,
        candidates_all      = candidates_all,
        B1                  = B1_REAL,
        B2                  = B2_REAL,
        alpha               = alpha,
        seed                = p1_seed,
        case_ids_sorted     = case_ids_sorted,
        labels_perturbed    = labels_pert,
        B_null              = B_NULL_REAL,
        B1_null_real        = B1_NULL_REAL,
        B2_null_real        = B2_NULL_REAL,
        holds_precomputed   = holds_pert,
    )

    drva_res = run_drva_on_perturbed_log(
        case_data_pert, candidates_all, alpha_drva, PI_DRVA_REAL
    )

    dm_res = run_dm_on_perturbed_log(
        case_data_pert, candidates_all, tau_star_dm, tau_min_dm,
        holds_precomputed=holds_pert,
    )

    S_pert = {
        METHOD_P1:   p1_res['S_set'],
        METHOD_DRVA: drva_res['S_set'],
        METHOD_DM:   dm_res['S_set'],
    }
    R_obs = {mth: len(S_pert[mth]) for mth in ALL_METHODS}

    return {
        'case_data_pert':  case_data_pert,
        'labels_pert':     labels_pert,
        'holds_pert':      holds_pert,
        'S_pert':          S_pert,
        'R_obs':           R_obs,
        'p1_res':          p1_res,
        'drva_res':        drva_res,
        'dm_res':          dm_res,
        'eps_seed_used':   eps_seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PER-CELL ANALYSIS FUNCTION  (parallelised over the 7×7 joint grid)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_joint_cell(
    eps:           float,
    rho:           float,
    eps_idx:       int,
    rho_idx:       int,
    struct_log:    dict,    # ρ already applied; shared across ε in this row
    holds_for_rho: dict,    # holds for struct_log; reused across ε
    candidates_all:  list,
    case_ids_sorted: list,
    S_orig_full:     dict,  # {METHOD: frozenset} — full P1 ground truth
    S_orig_rq2:      dict,  # {METHOD: frozenset} — (0,0) baseline; None for bootstrap
    tau_star_dm:     float,
    tau_min_dm:      float,
    alpha:           float,
    alpha_drva:      float,
    is_anchor_bootstrap: bool = False,
) -> dict:
    """
    Full analysis for one joint noise cell (ε, ρ).

    Seed scheme: BASE_SEED + eps_idx * 1000 + rho_idx
    Ensures reproducibility and uniqueness for each of the 49 grid cells.

    Retry logic: if any method returns R_obs=0, redraw with a new eps_seed
    (structural log is fixed — only the label-flip draw changes).

    Anchor cells (ε,ρ) ∈ ANCHOR_CELLS: run doubly-null FDR_emp replicates.
    (0,0) anchor (is_anchor_bootstrap=True): S_orig_rq2 = S_pert.
    All other cells: skip doubly-null (FDR_ref is primary).
    """
    print(f"  [ε={eps:.2f}  ρ={rho:.2f}]  Starting analysis...", flush=True)
    t0 = time.time()

    # Deterministic seed unique per (eps_idx, rho_idx) cell
    base_seed_cell = BASE_SEED + eps_idx * 1000 + rho_idx

    MAX_RETRIES = 5
    RETRY_PRIME = 999983

    draw      = None
    n_retries = 0

    for retry in range(MAX_RETRIES + 1):
        eps_seed_try = base_seed_cell + 1 + retry * RETRY_PRIME
        p1_seed_try  = base_seed_cell + 2 + retry * RETRY_PRIME

        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
            f"Draw {retry} (eps_seed={eps_seed_try})...",
            flush=True,
        )

        draw = _run_methods_on_joint_draw(
            eps           = eps,
            rho           = rho,
            eps_seed      = eps_seed_try,
            p1_seed       = p1_seed_try,
            struct_log    = struct_log,
            holds_for_rho = holds_for_rho,
            candidates_all   = candidates_all,
            case_ids_sorted  = case_ids_sorted,
            alpha            = alpha,
            alpha_drva       = alpha_drva,
            tau_star_dm      = tau_star_dm,
            tau_min_dm       = tau_min_dm,
        )

        zero_methods = [m for m in ALL_METHODS if draw['R_obs'][m] == 0]

        if not zero_methods:
            if retry > 0:
                print(
                    f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
                    f"R_obs>0 for all methods after "
                    f"{retry} retr{'y' if retry == 1 else 'ies'}.",
                    flush=True,
                )
            n_retries = retry
            break

        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
            f"R_obs=0 for {zero_methods} on draw {retry} — ",
            end="", flush=True,
        )
        if retry < MAX_RETRIES:
            print("retrying...", flush=True)
        else:
            print(
                f"MAX_RETRIES={MAX_RETRIES} exhausted. "
                f"R_obs=0 for {zero_methods} is a genuine signal at "
                f"(ε={eps}, ρ={rho}).",
                flush=True,
            )
            n_retries = retry

    # Unpack accepted draw
    case_data_pert = draw['case_data_pert']
    labels_pert    = draw['labels_pert']
    holds_pert     = draw['holds_pert']
    S_pert         = draw['S_pert']
    R_obs          = draw['R_obs']
    p1_res         = draw['p1_res']
    eps_seed_used  = draw['eps_seed_used']

    # S_orig_rq2: single (0,0) baseline for all cells
    if is_anchor_bootstrap or S_orig_rq2 is None:
        _S_rq2 = S_pert   # bootstrap: own output is the baseline
    else:
        _S_rq2 = S_orig_rq2

    # Primary: reference-anchored metrics
    ref_metrics = {
        mth: compute_reference_metrics(S_pert[mth], S_orig_full[mth], _S_rq2[mth])
        for mth in ALL_METHODS
    }

    cross_j = compute_cross_jaccard(
        S_pert[METHOD_P1], S_pert[METHOD_DRVA], S_pert[METHOD_DM]
    )

    m_prime_p1 = p1_res['m_prime']

    is_anchor = is_anchor_bootstrap or (eps, rho) in ANCHOR_CELLS

    methods_need_null = [mth for mth in ALL_METHODS if R_obs[mth] > 0]
    methods_zero_r    = [mth for mth in ALL_METHODS if R_obs[mth] == 0]

    null_counts = {mth: np.zeros(0, dtype=int) for mth in ALL_METHODS}
    B_null_this = B_NULL_ANCHOR if (is_anchor and methods_need_null) else 0

    fdr_null_metrics = {
        mth: _null_fdr_skipped('R_obs=0: FDR_emp undefined')
        for mth in methods_zero_r
    }
    if B_null_this == 0:
        fdr_null_metrics.update({
            mth: _null_fdr_skipped('non-anchor cell')
            for mth in methods_need_null
        })

    if B_null_this > 0:
        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
            f"Running {B_null_this} null replicates "
            f"(m_prime={m_prime_p1}, {len(methods_need_null)} methods with R_obs>0)...",
            flush=True,
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
            base_seed           = base_seed_cell + 50_000,
            n_jobs              = INNER_N_JOBS,
        )
        alpha_vals = {METHOD_P1: alpha, METHOD_DRVA: alpha_drva, METHOD_DM: alpha}
        for mth in methods_need_null:
            fdr_null_metrics[mth] = _compute_fdr_from_null(
                null_counts[mth], R_obs[mth], alpha_vals[mth]
            )

    wall = time.time() - t0
    print(
        f"  [ε={eps:.2f}  ρ={rho:.2f}]  Done in {wall:.1f}s  "
        f"R_obs=[P1:{R_obs[METHOD_P1]}, DRVA:{R_obs[METHOD_DRVA]}, "
        f"DM:{R_obs[METHOD_DM]}]",
        flush=True,
    )

    result = {
        'eps':           eps,
        'rho':           rho,
        'eps_idx':       eps_idx,
        'rho_idx':       rho_idx,
        'R_obs':         R_obs,
        'S_pert':        {mth: list(S_pert[mth]) for mth in ALL_METHODS},
        'ref_metrics':   ref_metrics,
        'fdr_null':      fdr_null_metrics,
        'cross_jaccard': cross_j,
        'null_counts':   {mth: null_counts[mth].tolist() for mth in ALL_METHODS},
        'B_null':        B_null_this,
        'wall_seconds':  wall,
        'm_prime_P1':    m_prime_p1,
        'w_disc_rq2':    p1_res['w_disc'],
        'w_struct_rq2':  p1_res['w_struct'],
        'is_anchor':     is_anchor,
        'n_retries':     n_retries,
        'eps_seed_used': eps_seed_used,
    }

    if is_anchor_bootstrap:
        result['S_pert_raw'] = S_pert

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ORIGINAL DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_original_data(n_workers: int = 1) -> dict:
    """
    Run all three methods on the original (unperturbed) Production log.

    P1: full execute_pipeline (with empirical Phipson-Smyth calibration, B_null=200).
        Returns S_orig_full — the ground truth for FDR_ref.
    DRVA: full π=1000 run.
    DM: calibrated to match R_obs^P1.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — ORIGINAL DATA RUN (P1 + DRVA + DeclareMiner)")
    print("=" * 100)

    # P1 full pipeline (empirical calibration — produces S_orig_full ground truth)
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

    S_orig_full_p1 = frozenset(
        (r.constraint_type, r.activity_a, r.activity_b)
        for r in pattern_results if r.is_significant_final
    )
    R_orig_p1 = len(S_orig_full_p1)

    p1_wall = time.time() - t0
    print(f"\n  P1 complete: {p1_wall:.1f}s  k*={R_orig_p1}")

    # DRVA full run
    t0 = time.time()
    drva_cfg = DRVA_CONFIG.copy()
    drva_cfg.update({
        'alpha': ALPHA_DRVA, 'hierarchical_pruning': False,
        'mmin': 0.0, 'mdiff_min': 0.0,
    })
    with _suppress():
        drva_orig = run_drva(config=drva_cfg, case_data=case_data,
                             candidates_all=candidates_all)
    S_orig_full_drva = frozenset(
        (r['constraint_type'], r['activity_a'], r['activity_b'])
        for r in drva_orig['results'] if r['is_significant_cecconi']
    )
    drva_wall = time.time() - t0
    print(f"  DRVA complete: {drva_wall:.1f}s  R_obs={len(S_orig_full_drva)}")

    # DeclareMiner calibrated to P1
    t0 = time.time()
    dm_cfg = DM_CONFIG.copy()
    dm_cfg['R_obs_target'] = R_orig_p1
    with _suppress():
        dm_orig = run_declareminer(
            config=dm_cfg, case_data=case_data,
            candidates_all=candidates_all, R_obs_target=R_orig_p1,
        )
    S_orig_full_dm = frozenset(
        (r['constraint_type'], r['activity_a'], r['activity_b'])
        for r in dm_orig['results_all'] if r['is_significant']
    )
    tau_star = float(dm_orig['tau_star'])
    tau_min  = float(dm_orig['config'].get('tau_min', DM_CONFIG['tau_min']))
    dm_wall  = time.time() - t0
    print(f"  DM complete: {dm_wall:.1f}s  R_obs={len(S_orig_full_dm)}  τ*={tau_star:.4f}")

    # S_orig_full = ground truth for FDR_ref (full P1 empirical calibration)
    S_orig_full = {
        METHOD_P1:   S_orig_full_p1,
        METHOD_DRVA: S_orig_full_drva,
        METHOD_DM:   S_orig_full_dm,
    }
    R_orig = {
        METHOD_P1:   R_orig_p1,
        METHOD_DRVA: len(S_orig_full_drva),
        METHOD_DM:   len(S_orig_full_dm),
    }

    print(f"\n  S_orig_full (ground truth) R_obs: "
          f"P1={R_orig[METHOD_P1]}  "
          f"DRVA={R_orig[METHOD_DRVA]}  "
          f"DM={R_orig[METHOD_DM]}")

    return {
        'case_data':       case_data,
        'candidates_all':  candidates_all,
        'case_ids_sorted': case_ids_sorted,
        'labels_orig':     labels_orig,
        'pattern_results': pattern_results,
        'S_orig_full':     S_orig_full,   # ground truth (renamed from S_orig)
        'R_orig':          R_orig,
        'tau_star':        tau_star,
        'tau_min':         tau_min,
        'holds_all':       output['holds_all'],
        'null_delta_mat':  output['null_delta_matrix'],
        'delta_obs':       output['delta_obs'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — RQ2 JOINT PERTURBATION GRID
# ═══════════════════════════════════════════════════════════════════════════

def _joint_cell_worker(
    eps, rho, eps_idx, rho_idx,
    struct_log, holds_for_rho,
    candidates_all, case_ids_sorted,
    S_orig_full, S_orig_rq2,
    tau_star, tau_min, alpha, alpha_drva,
):
    """Top-level loky-safe worker for one (ε, ρ) joint grid cell."""
    return analyze_joint_cell(
        eps            = eps,
        rho            = rho,
        eps_idx        = eps_idx,
        rho_idx        = rho_idx,
        struct_log     = struct_log,
        holds_for_rho  = holds_for_rho,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        S_orig_full    = S_orig_full,
        S_orig_rq2     = S_orig_rq2,
        tau_star_dm    = tau_star,
        tau_min_dm     = tau_min,
        alpha          = alpha,
        alpha_drva     = alpha_drva,
        is_anchor_bootstrap = False,
    )


def run_joint_grid(
    case_data_orig:  dict,
    candidates_all:  list,
    case_ids_sorted: list,
    S_orig_full:     dict,
    tau_star:        float,
    tau_min:         float,
    n_jobs:          int  = N_JOBS,
    holds_orig:      dict = None,
) -> tuple:
    """
    Run the full 7×7 joint noise grid in parallel.

    Architecture
    ------------
    Step 0 — Precompute structural logs and holds per ρ row.
        ρ = 0: reuse case_data_orig and holds_orig (no structural change).
        ρ > 0: apply N_struct(ρ) once per ρ row with a fixed seed, then
               compute holds once. All 7 ε cells in that row share the same
               structural log and holds (label noise does not affect traces).
        Total holds computations: 1 (ρ=0, reused) + 6 (ρ>0) = 7.

    Step 1 — Run (0,0) anchor SEQUENTIALLY to establish S_orig_rq2.
        S_orig_rq2 is the single joint baseline for all 49 cells.

    Step 2 — Run remaining 48 cells in parallel (n_jobs workers).
        Each cell worker receives its row's pre-computed struct_log and
        holds_for_rho; it applies only the label noise layer (ε).

    Returns
    -------
    (all_results, S_orig_rq2)
        all_results : list of 49 dicts, sorted by (eps, rho).
        S_orig_rq2  : {METHOD: frozenset} — (0,0) discovery set.
    """
    print("\n" + "=" * 100)
    print("SECTION 2 — JOINT PERTURBATION GRID  (B.3: Label × Structural)")
    print(f"  ε levels: {JOINT_LABEL_LEVELS}")
    print(f"  ρ levels: {JOINT_STRUCT_LEVELS}")
    print(f"  Grid size: {len(JOINT_GRID)} cells  |  Anchor cells: {sorted(ANCHOR_CELLS)}")
    print(f"  B_null (anchor)={B_NULL_ANCHOR}  (non-anchor=0, FDR_ref is primary)")
    print(f"  B1_null={B1_NULL}  B2_null={B2_NULL}  PI_DRVA_null={PI_DRVA_NULL}")
    print(f"  n_jobs={n_jobs}")
    print(f"  W_DISC_RQ2={W_DISC_RQ2:.4f}  W_STRUCT_RQ2={W_STRUCT_RQ2:.4f}")
    print(f"  Oracle: c={_C_NULL:.4f}, f={_F_NULL:.4f}  (rho_sd=0)")
    print("=" * 100)

    t0 = time.time()

    # ── Step 0: Precompute one structural log + holds per ρ row ──────────
    print("\n  [Step 0] Precomputing structural logs and holds per ρ row...", flush=True)
    struct_logs  = {}
    holds_cache  = {}
    for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS):
        if rho == 0.0:
            struct_logs[rho] = case_data_orig
            holds_cache[rho] = holds_orig
            print(f"    ρ={rho:.2f}  → reusing original holds", flush=True)
        else:
            # Fixed seed per ρ row: independent of eps_idx, unique across ρ rows.
            rho_struct_seed = BASE_SEED + rho_idx * 13
            t_rho = time.time()
            sl = apply_structural_noise(case_data_orig, rho, rho_struct_seed)
            struct_logs[rho] = sl
            with _suppress():
                holds_cache[rho] = compute_holds_by_case_batch(sl, candidates_all)
            print(
                f"    ρ={rho:.2f}  → structural log + holds computed "
                f"in {time.time()-t_rho:.1f}s",
                flush=True,
            )

    # ── Step 1: Run (0,0) anchor to establish S_orig_rq2 ─────────────────
    print("\n  [Step 1] Running (ε=0, ρ=0) anchor to establish S_orig_rq2...",
          flush=True)

    anchor_result = analyze_joint_cell(
        eps=0.0, rho=0.0, eps_idx=0, rho_idx=0,
        struct_log    = struct_logs[0.0],
        holds_for_rho = holds_cache[0.0],
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        S_orig_full     = S_orig_full,
        S_orig_rq2      = None,
        tau_star_dm     = tau_star,
        tau_min_dm      = tau_min,
        alpha           = ALPHA,
        alpha_drva      = ALPHA_DRVA,
        is_anchor_bootstrap = True,
    )

    S_orig_rq2 = {
        mth: frozenset(anchor_result['S_pert_raw'][mth])
        for mth in ALL_METHODS
    }

    print(f"\n  S_orig_rq2 established (single (0,0) baseline for all 49 cells):")
    for mth in ALL_METHODS:
        n_rq2  = len(S_orig_rq2[mth])
        n_full = len(S_orig_full[mth])
        print(f"    [{mth}]: {n_rq2} patterns  "
              f"(vs {n_full} in S_orig_full; oracle approx. gap)")

    # ── Step 2: Run remaining 48 cells in parallel ────────────────────────
    remaining_jobs = [
        (eps, rho, eps_idx, rho_idx)
        for eps_idx, eps in enumerate(JOINT_LABEL_LEVELS)
        for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS)
        if not (eps == 0.0 and rho == 0.0)
    ]

    print(f"\n  [Step 2] Running {len(remaining_jobs)} remaining cells in parallel "
          f"(n_jobs={n_jobs})...", flush=True)

    results_flat = Parallel(n_jobs=n_jobs, verbose=5, backend='loky')(
        delayed(_joint_cell_worker)(
            eps, rho, eps_idx, rho_idx,
            struct_logs[rho], holds_cache[rho],
            candidates_all, case_ids_sorted,
            S_orig_full, S_orig_rq2,
            tau_star, tau_min, ALPHA, ALPHA_DRVA,
        )
        for eps, rho, eps_idx, rho_idx in remaining_jobs
    )

    wall = time.time() - t0
    print(f"\n  Grid complete. Total wall time: {wall:.1f}s ({wall/3600:.2f} h)")

    all_results = [anchor_result] + list(results_flat)
    all_results.sort(key=lambda x: (x['eps'], x['rho']))

    return all_results, S_orig_rq2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_joint_metrics_table(all_results: list) -> pd.DataFrame:
    """
    Build a long-format metrics table for the 7×7 joint noise grid.

    One row per (ε, ρ, method).  Columns:
      eps, rho — noise levels.
      Primary   : FDR_ref, Precision, Recall, F1, TP, FP, FN, R_full,
                  Jaccard_rq2, Jaccard_full, Gained_rq2, Lost_rq2, reliable.
      Secondary : FP_over_Rfull, FN_over_Rfull.
      Doubly-null (anchor cells only):
                  FDR_emp, FDR_CI_lower, FDR_CI_upper, FWER_emp, E_V_b,
                  controls_FDR, estimable, skipped_reason.
      Provenance: B_null, m_prime_P1, is_anchor, n_retries, eps_seed_used.
    """
    rows = []
    for res in all_results:
        eps = res['eps']
        rho = res['rho']
        for mth in ALL_METHODS:
            rm  = res['ref_metrics'][mth]
            fdr = res['fdr_null'][mth]
            rows.append({
                'eps':             eps,
                'rho':             rho,
                'method':          mth,
                'R_obs':           res['R_obs'][mth],
                # ── Primary reference-anchored metrics ──────────────────
                'FDR_ref':         rm['FDR_ref'],
                'Precision':       rm['Precision'],
                'Recall':          rm['Recall'],
                'F1':              rm['F1'],
                'TP':              rm['TP'],
                'FP':              rm['FP'],
                'FN':              rm['FN'],
                'R_full':          rm['R_full'],
                'Jaccard_rq2':     rm['Jaccard_rq2'],
                'Jaccard_full':    rm['Jaccard_full'],
                'Gained_rq2':      rm['Gained_rq2'],
                'Lost_rq2':        rm['Lost_rq2'],
                'reliable':        rm['reliable'],
                # ── Secondary oracle-normalised burden metrics ──────────
                'FP_over_Rfull':   rm['FP_over_Rfull'],
                'FN_over_Rfull':   rm['FN_over_Rfull'],
                # ── Doubly-null (anchor cells only) ─────────────────────
                'FDR_emp':         fdr['FDR_emp'],
                'FDR_CI_lower':    fdr.get('FDR_CI_lower',  float('nan')),
                'FDR_CI_upper':    fdr.get('FDR_CI_upper',  float('nan')),
                'FWER_emp':        fdr.get('FWER_emp',      float('nan')),
                'E_V_b':           fdr.get('E_V_b',         float('nan')),
                'controls_FDR':    fdr.get('controls_FDR',  None),
                'estimable':       fdr.get('estimable',     False),
                'skipped_reason':  fdr.get('skipped_reason', None),
                # ── Provenance ──────────────────────────────────────────
                'B_null':          res['B_null'],
                'm_prime_P1':      res['m_prime_P1'],
                'is_anchor':       res.get('is_anchor',     False),
                'n_retries':       res.get('n_retries',     0),
                'eps_seed_used':   res.get('eps_seed_used', None),
            })
    return pd.DataFrame(rows)


def save_joint_outputs(
    all_results: list,
    orig_data:   dict,
    S_orig_rq2:  dict,
    total_wall:  float,
) -> None:
    """
    Save all joint-design RQ2 output files.

    Files written to RQ2_OUTPUT_DIR:
        rq2_joint_metrics.csv                 Long-format table (49 cells × 3 methods).
        rq2_joint_<metric>_<method>.csv       7×7 pivot heatmap per metric per method.
        rq2_joint_cross_jaccard.csv           Cross-method Jaccard per cell.
        rq2_joint_null_counts.json            Raw V_b arrays per (ε, ρ, method).
        rq2_joint_results.json                Full structured results for paper.
    """
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)
    R_orig      = orig_data['R_orig']
    S_orig_full = orig_data['S_orig_full']

    # ── Long-format CSV ───────────────────────────────────────────────────
    df = build_joint_metrics_table(all_results)
    lf_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_joint_metrics.csv')
    df.to_csv(lf_path, index=False)
    print(f"  Saved: {lf_path}")

    # ── Heatmap pivot CSVs per metric × method ───────────────────────────
    pivot_metrics = ['FDR_ref', 'Recall', 'F1', 'Jaccard_rq2',
                     'FP_over_Rfull', 'FN_over_Rfull', 'Jaccard_full']
    for mth in ALL_METHODS:
        df_m     = df[df['method'] == mth].copy()
        mth_slug = mth.replace(' ', '_').replace('/', '_')
        for metric in pivot_metrics:
            if metric not in df_m.columns:
                continue
            pivot = df_m.pivot(index='eps', columns='rho', values=metric)
            pivot.index.name   = 'eps \\ rho'
            pivot_path = os.path.join(
                RQ2_OUTPUT_DIR,
                f'rq2_joint_{metric.lower()}_{mth_slug}.csv',
            )
            pivot.to_csv(pivot_path)
            print(f"  Saved: {pivot_path}")

    # ── Cross-method Jaccard ──────────────────────────────────────────────
    cj_rows = []
    for res in all_results:
        row = {'eps': res['eps'], 'rho': res['rho']}
        row.update(res['cross_jaccard'])
        cj_rows.append(row)
    cj_df   = pd.DataFrame(cj_rows)
    cj_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_joint_cross_jaccard.csv')
    cj_df.to_csv(cj_path, index=False)
    print(f"  Saved: {cj_path}")

    # ── Null counts JSON ──────────────────────────────────────────────────
    null_counts_json = {}
    for res in all_results:
        key = f"eps_{res['eps']:.4f}_rho_{res['rho']:.4f}"
        null_counts_json[key] = res['null_counts']
    nc_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_joint_null_counts.json')
    with open(nc_path, 'w', encoding='utf-8') as f:
        json.dump(null_counts_json, f, indent=2)
    print(f"  Saved: {nc_path}")

    # ── Full results JSON ─────────────────────────────────────────────────
    full_json = {
        'rq2_version': '3.0',
        'log_name':    'Production',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'block':   'B.3 Joint Label × Structural Noise',
            'methods': ALL_METHODS,
            'shared_pool': 'M_all from Phase 0 DECLARE spec (fixed)',
            'joint_operator': (
                'N_label(ε) ∘ N_struct(ρ): structural log precomputed per ρ row '
                '(fixed seed per row); label noise applied per ε cell.'
            ),
            'joint_label_levels':  JOINT_LABEL_LEVELS,
            'joint_struct_levels': JOINT_STRUCT_LEVELS,
            'n_cells':             len(JOINT_GRID),
            'anchor_cells':        [list(c) for c in sorted(ANCHOR_CELLS)],
            'holds_reuse':         (
                'ρ=0: holds_orig reused. '
                'ρ>0: holds computed once per ρ row, shared across all ε.'
            ),
            'null_protocol':   'Doubly-null: σ_label ∘ σ_trace on L_{ε,ρ} (anchor cells only)',
            'P1_gate':         f'q̃_Hou ≤ α (empirical Phipson-Smyth, B_null={B_NULL_REAL})',
            'DRVA_gate':       f'p_Cecconi ≤ {ALPHA_DRVA}',
            'DM_gate':         '|Δconf| ≥ τ* (calibrated, fixed across all cells)',
            'primary_metrics': [
                'FDR_ref = FP/R_obs  (reference-anchored false discovery rate)',
                'Precision = TP/R_obs = 1 − FDR_ref',
                'Recall = TP/R_full  (power under noise)',
                'F1 = harmonic mean(Precision, Recall)',
                'Jaccard_rq2 = |S_pert ∩ S_orig_rq2| / |S_pert ∪ S_orig_rq2|',
            ],
            'secondary_metrics': [
                'FP_over_Rfull = FP/R_full  (oracle-normalised overcall; ≠ FDR)',
                'FN_over_Rfull = FN/R_full = 1 − Recall',
                'Jaccard_full = |S_pert ∩ S_orig_full| / |S_pert ∪ S_orig_full|',
                'FDR_emp (doubly-null, anchor cells only)',
            ],
        },

        'reference_sets': {
            'S_orig_full': {
                mth: {
                    'R': int(len(S_orig_full[mth])),
                    'description': 'Full P1 empirical calibration (B_null=200); ground truth',
                }
                for mth in ALL_METHODS
            },
            'S_orig_rq2': {
                mth: {
                    'R': int(len(S_orig_rq2[mth])),
                    'description': (
                        f'Joint RQ2 baseline at (ε=0, ρ=0), '
                        f'empirical Phipson-Smyth B_null={B_NULL_REAL}'
                    ),
                }
                for mth in ALL_METHODS
            },
        },

        'config': {
            'ALPHA':               ALPHA,
            'ALPHA_DRVA':          ALPHA_DRVA,
            'B1_FULL':             B1_FULL,
            'B2_FULL':             B2_FULL,
            'B1_REAL':             B1_REAL,
            'B2_REAL':             B2_REAL,
            'B1_NULL':             B1_NULL,
            'B2_NULL':             B2_NULL,
            'PI_DRVA_REAL':        PI_DRVA_REAL,
            'PI_DRVA_NULL':        PI_DRVA_NULL,
            'B_NULL_REAL':         B_NULL_REAL,
            'B1_NULL_REAL':        B1_NULL_REAL,
            'B2_NULL_REAL':        B2_NULL_REAL,
            'B_NULL_ANCHOR':       B_NULL_ANCHOR,
            'B_NULL_INTERMEDIATE': 0,
            'BASE_SEED':           BASE_SEED,
            'tau_star_DM':         orig_data['tau_star'],
            'tau_min_DM':          orig_data['tau_min'],
            'W_DISC_RQ2':          float(W_DISC_RQ2),
            'W_STRUCT_RQ2':        float(W_STRUCT_RQ2),
            'c_null_rq2':          float(_C_NULL),
            'f_null_rq2':          float(_F_NULL),
            'weight_note': (
                'W_DISC=0.60, W_STRUCT=0.40. Identical to full P1 v9.0 '
                '(Pepe & Fleming 1989; Hou 2005 §4).'
            ),
        },

        'original_data': {
            'R_obs': {mth: int(R_orig[mth]) for mth in ALL_METHODS},
        },

        'speed_optimisations_applied': {
            'S1_holds_per_rho_row':          '7 holds computations total (not 49)',
            'S2_empirical_pHou_budget_correct': f'B_null={B_NULL_REAL}, B1={B1_NULL_REAL}, B2={B2_NULL_REAL}',
            'S3_null_anchor_only':           'FDR_emp at 4 corner cells only',
            'S4_reduced_B1_B2_null':         f'B1={B1_NULL}, B2={B2_NULL}',
            'S5_cell_parallelism':           f'n_jobs={N_JOBS}',
            'S6_vectorised_delta_conf':      True,
        },

        'timing': {'total_seconds': total_wall},

        'joint_results': [
            {k: v for k, v in r.items() if k not in ('S_pert', 'S_pert_raw')}
            for r in all_results
        ],
    }

    json_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_joint_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — SUMMARY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt='.4f') -> str:
    """Format a float, returning 'NaN' for nan values."""
    return f"{v:{fmt}}" if v == v else '  NaN'


def _print_joint_summary(all_results: list) -> None:
    """Print compact FDR_ref heatmap tables per method and a full degradation table."""
    # ── Per-method FDR_ref heatmap ────────────────────────────────────────
    for mth in ALL_METHODS:
        print(f"\n  FDR_ref heatmap — {mth}")
        _col_hdr = 'ε \\ ρ'
        print(f"  {_col_hdr:>8s}", end='')
        for rho in JOINT_STRUCT_LEVELS:
            print(f"  {rho:>6.2f}", end='')
        print()
        print(f"  {'─'*70}")
        for eps in JOINT_LABEL_LEVELS:
            print(f"  {eps:>8.2f}", end='')
            for rho in JOINT_STRUCT_LEVELS:
                res = next(
                    (r for r in all_results if r['eps'] == eps and r['rho'] == rho),
                    None,
                )
                if res is None:
                    print(f"  {'?':>6s}", end='')
                else:
                    v = res['ref_metrics'][mth]['FDR_ref']
                    print(f"  {_fmt(v, '.4f'):>6s}", end='')
            print()
        print(f"  {'─'*70}")

    # ── Full long-format table (all cells, primary metrics) ───────────────
    print(f"\n  {'─'*130}")
    print(f"  {'ε':>5s}  {'ρ':>5s}  {'Method':20s}  "
          f"{'R_obs':>6s}  {'FDR_ref':>8s}  {'Recall':>7s}  "
          f"{'F1':>6s}  {'J_rq2':>7s}  {'FP/Rf':>7s}  "
          f"{'FDR_emp':>8s}  {'anchor':>6s}")
    print(f"  {'─'*130}")
    for res in sorted(all_results, key=lambda x: (x['eps'], x['rho'])):
        eps = res['eps']
        rho = res['rho']
        for mth in ALL_METHODS:
            rm  = res['ref_metrics'][mth]
            fdr = res['fdr_null'][mth]
            fe  = fdr['FDR_emp']
            print(
                f"  {eps:>5.2f}  {rho:>5.2f}  {mth:20s}  "
                f"{res['R_obs'][mth]:>6d}  "
                f"{_fmt(rm['FDR_ref']):>8s}  "
                f"{_fmt(rm['Recall']):>7s}  "
                f"{_fmt(rm['F1']):>6s}  "
                f"{rm['Jaccard_rq2']:>7.4f}  "
                f"{_fmt(rm['FP_over_Rfull']):>7s}  "
                f"{'  n/a  ' if fe != fe else f'{fe:.4f}':>8s}  "
                f"{'YES' if res.get('is_anchor') else '-':>6s}"
            )
    print(f"  {'─'*130}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ2 — SPECIFICATION QUALITY DEGRADATION: PRODUCTION  (v3.0)")
    print("Block B.3: Joint Label × Structural Noise  (7×7 = 49 cells)")
    print("Three methods: P1 (Hou-Storey) | DRVA | DeclareMiner")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ε grid:  {JOINT_LABEL_LEVELS}")
    print(f"  ρ grid:  {JOINT_STRUCT_LEVELS}")
    print(f"  Grid:    {len(JOINT_GRID)} cells  |  Anchors: {sorted(ANCHOR_CELLS)}")
    print(f"  α(P1)={ALPHA}  α_DRVA={ALPHA_DRVA}")
    print(f"  W_DISC_RQ2={W_DISC_RQ2:.4f}  W_STRUCT_RQ2={W_STRUCT_RQ2:.4f}")
    print(f"    (budget-correct: B1_REAL={B1_REAL}, B2_REAL//2={_B2_TEST_RQ2})")
    print(f"  Oracle (rho_sd=0): c={_C_NULL:.4f}, f={_F_NULL:.4f}")
    print(f"  Empirical calibration per cell: B_null={B_NULL_REAL}, B1={B1_NULL_REAL}, B2={B2_NULL_REAL}")
    print(f"  B_null(anchor FDR_emp)={B_NULL_ANCHOR}  (non-anchor=0, FDR_ref is primary)")
    print(f"  B1_null={B1_NULL}  B2_null={B2_NULL}  PI_DRVA_null={PI_DRVA_NULL}")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")
    print(f"\n  Primary metrics:   FDR_ref=FP/R_obs, Recall=TP/R_full, F1, Jaccard_rq2")
    print(f"  Secondary metrics: FP_over_Rfull=FP/R_full, FN_over_Rfull, Jaccard_full")
    print(f"    FDR_ref: fraction of discoveries outside S_orig_full")
    print(f"    FP_over_Rfull: spurious mass relative to oracle size (≠ FDR)")
    print(f"\n  Reference sets:")
    print(f"    S_orig_full — full P1 empirical calibration (B_null=200); ground truth")
    print(f"    S_orig_rq2  — single (0,0) baseline; Jaccard_rq2=1 at origin by construction")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    # Section 1: original data run
    orig_data = run_original_data(n_workers=N_JOBS)

    # Section 2: joint perturbation grid
    all_results, S_orig_rq2 = run_joint_grid(
        case_data_orig  = orig_data['case_data'],
        candidates_all  = orig_data['candidates_all'],
        case_ids_sorted = orig_data['case_ids_sorted'],
        S_orig_full     = orig_data['S_orig_full'],
        tau_star        = orig_data['tau_star'],
        tau_min         = orig_data['tau_min'],
        n_jobs          = N_JOBS,
        holds_orig      = orig_data['holds_all'],
    )

    # Section 3: save outputs
    total_wall = time.time() - t_total

    print("\n" + "=" * 100)
    print("SECTION 3 — SAVING OUTPUTS")
    print("=" * 100)
    save_joint_outputs(all_results, orig_data, S_orig_rq2, total_wall)

    # Section 4: summary
    print(f"\n{'='*100}")
    print("RQ2 — PRODUCTION COMPLETE  (v3.0 Joint Noise)")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")

    _print_joint_summary(all_results)

    print(f"\n{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 Specification Degradation — Production v3.0  "
            "(B.3 Joint Label × Structural Noise, 7×7 grid)"
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
        '--dry-run', action='store_true',
        help='Smoke test: 3 levels only, B_null=2, minimal budgets',
    )
    args = parser.parse_args()

    if args.dry_run:
        LABEL_NOISE_LEVELS  = [0.0, 0.10, 0.50]
        STRUCT_NOISE_LEVELS = [0.0, 0.30, 1.00]
        B_NULL_ANCHOR       = 2
        B1_NULL             = 30
        B2_NULL             = 20
        PI_DRVA_NULL        = 20
        B1_REAL             = 100
        B2_REAL             = 50
        PI_DRVA_REAL        = 100
        B_NULL_FULL         = 5
        B1_NULL_FULL        = 10
        B2_NULL_FULL        = 10
        # Recompute budget-correct weights for dry-run budgets
        _B2_TEST_RQ2 = B2_REAL // 2
        W_DISC_RQ2   = B1_REAL / (B1_REAL + _B2_TEST_RQ2)
        W_STRUCT_RQ2 = _B2_TEST_RQ2 / (B1_REAL + _B2_TEST_RQ2)
        _C_NULL, _F_NULL = hou_satterthwaite_params(W_STRUCT_RQ2, W_DISC_RQ2, rho_sd=0.0)
        print("*** DRY-RUN MODE: 3 levels, B_null=2, minimal budgets ***")
    else:
        B_NULL_ANCHOR = args.b_null_anchor

    N_JOBS     = args.n_jobs
    ALPHA      = args.alpha
    ALPHA_DRVA = args.alpha_drva

    main()
