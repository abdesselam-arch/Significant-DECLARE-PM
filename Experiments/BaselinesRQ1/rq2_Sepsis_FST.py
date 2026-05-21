#!/usr/bin/env python3
"""
rq2_Sepsis_FST.py  —  RQ2 Specification Quality Degradation: Sepsis  [DeclareMiner FST / Option A]
====================================================================================================
Block B.3: Joint Label × Structural Noise Perturbation (2-D Cartesian grid)
DeclareMiner FST — Per-Constraint Z-Test Threshold Baseline

RESEARCH QUESTION
-----------------
RQ2 (DM-FST): As signal is progressively corrupted along BOTH noise axes
simultaneously, how does DeclareMiner FST degrade in:
  (a) the size and composition of its discovered specification,
  (b) the consistency relative to the ground truth S_orig_full
      (FDR_ref, Precision, Recall, F1, Jaccard_rq2), and
  (c) the empirical FDR estimates under the doubly-null protocol (anchor cells)?

JOINT PERTURBATION OPERATOR
-----------------------------
B.3 — Joint noise  N_label(ε) ∘ N_struct(ρ):

    L_{ε,ρ} = N_struct(ρ) ∘ N_label(ε)(L)

    N_label(ε): Flip outcome label of each case independently with prob ε.
    N_struct(ρ): For each trace, select round(ρ·(n−1)) adjacent pairs without
                 replacement and swap each.

    2-D grid G = {(ε_i, ρ_j)}:
        ε ∈ {0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50}  (7 levels)
        ρ ∈ {0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00}  (7 levels)
        → 49 cells; (0,0) is established as anchor bootstrap (S_orig_rq2).

    Anchor cells (full doubly-null + S_orig_rq2 baseline): four corners:
        (0.00, 0.00), (0.50, 0.00), (0.00, 1.00), (0.50, 1.00)

DeclareMiner FST ON PERTURBED LOGS
------------------------------------
τ_DM*(c) is computed from the original data ONCE and held FIXED across all
49 grid cells — never recalibrated on perturbed logs.

For each cell L_{ε,ρ}:
    1. Reuse holds (computed once per ρ row — label noise does not change
       trace structure, so holds remain valid).
    2. Compute conf0_{ε,ρ}(c), conf1_{ε,ρ}(c) from holds for all c ∈ M_all.
    3. S^DM_{ε,ρ} = {c : |Δconf̂_{ε,ρ}(c)| ≥ τ_DM*(c)
                          AND (conf0 ≥ τ_min OR conf1 ≥ τ_min)}.

No internal permutation budget is required per cell. DM-FST evaluation at
each of the 49 cells is O(m) — fully deterministic.

HOLDS REUSE STRATEGY
---------------------
Holds depend only on trace structure, not on class labels.
    ρ = 0.00 : reuse holds_orig (no structural perturbation).
    ρ > 0.00 : precompute holds once per ρ row, shared across all ε in that row.
This reduces holds computation from 49 to 1 (ρ=0) + 6 (ρ>0) = 7 total.

REFERENCE SETS
--------------
  S_orig_full   — DM-FST acceptance set on the original (unperturbed) data.
                  Ground truth. Fixed across all 49 cells.
                  Defines TP/FP/FN for FDR_ref and Recall.
                  = {c : |Δconf̂(c)| ≥ τ_DM*(c)} on the original log.

  S_orig_rq2    — DM-FST output at (ε=0, ρ=0) with the same frozen τ_DM*(c).
                  Single baseline for all 49 cells. Jaccard_rq2=1.0 at (0,0)
                  by construction.

PRIMARY METRICS PER CELL
--------------------------
    FDR_ref       FP / R_obs             (reference-anchored false discovery rate)
    Precision     TP / R_obs  = 1 − FDR_ref
    Recall        TP / R_full             (power under noise)
    F1            Harmonic mean of Precision and Recall.
    Jaccard_rq2   |S_pert ∩ S_orig_rq2| / |S_pert ∪ S_orig_rq2|

SECONDARY METRICS
-----------------
    FP_over_Rfull  FP / R_full   (oracle-normalised overcall burden)
    FN_over_Rfull  FN / R_full   = 1 − Recall
    Jaccard_full   |S_pert ∩ S_orig_full| / |S_pert ∪ S_orig_full|
    FDR_emp        Doubly-null empirical FDR (anchor cells only).

NULL REPLICATE BUDGET (ANCHOR CELLS)
--------------------------------------
DM-FST null replicates are FULLY DETERMINISTIC given the doubly-null log:
    1. Build doubly-null log from L_{ε,ρ}: σ_trace ∘ σ_label.
    2. Recompute holds on null log.
    3. Apply FROZEN τ_DM*(c): V_b = #{c: |Δconf̂_b(c)| ≥ τ_DM*(c)}.
Zero internal permutation budget. Contrast with P1 (B1_null=75, B2_null=75 per rep).

MECHANISTIC PREDICTION
-----------------------
At (0,0): FDR_ref ≈ 0 (no noise → discovers original patterns).
As ε or ρ increase: Δconf̂ degrades because label noise erodes class
separation and structural noise destroys temporal ordering of traces.
DM-FST has no principled noise boundary — it cannot distinguish genuine
signal loss from threshold exceedance due to noise.

OUTPUT FILES
-------------
    rq2_dm_fst_joint_metrics.csv                Long-format table (49 cells × 1 method).
    rq2_dm_fst_joint_<metric>_DM_FST.csv        7×7 pivot heatmap per metric.
    rq2_dm_fst_joint_null_counts.json           Raw V_b arrays per (ε, ρ).
    rq2_dm_fst_joint_results.json               Full structured results for paper.

Version : 2.0  (DeclareMiner FST / Option A; joint noise design)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Cecconi, Augusto & Di Ciccio (2021). BPM Forum 2021, LNBIP 427, pp. 73–91.
Di Francescomarino, Donadello, Ghidini, Maggi, Puura (2025). BISE 67(6):877–894.
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

# ── DeclareMiner FST baseline — all primitives resolved through this module ─
from DeclareMiner_Sepsis import (
    run_declareminer,
    run_declareminer_on_doubly_null_log,
    DM_CONFIG,
    compute_support_from_holds,
    apply_threshold_decision,
    compute_holds_by_case_batch,
    precompute_activity_index,
    split_by_class,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

RQ2_OUTPUT_DIR = "RQ2_Sepsis_FST"

# ── 2-D Joint Perturbation Grid ───────────────────────────────────────────
JOINT_LABEL_LEVELS  = [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
JOINT_STRUCT_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

JOINT_GRID = [
    (eps, rho)
    for eps in JOINT_LABEL_LEVELS
    for rho in JOINT_STRUCT_LEVELS
]   # 49 cells

# Anchor cells: full doubly-null FDR_emp + S_orig_rq2 bootstrap.
ANCHOR_CELLS = frozenset({
    (0.00, 0.00),
    (JOINT_LABEL_LEVELS[-1], 0.00),
    (0.00, JOINT_STRUCT_LEVELS[-1]),
    (JOINT_LABEL_LEVELS[-1], JOINT_STRUCT_LEVELS[-1]),
})

# FDR nominal level (DM-FST does not target this; used for pass/fail verdict only)
ALPHA = 0.05

# Doubly-null replicates at anchor cells (DM-FST null reps are O(m) — deterministic)
B_NULL_ANCHOR = 200

# Base seed for RQ2
BASE_SEED = 20260601

# Outer parallelism (over perturbation cells)
N_JOBS = -1
# Inner parallelism for null loops inside a joblib worker (1 = no nested loky)
INNER_N_JOBS = 1

# Method identifier
METHOD_DM   = "DeclareMiner_FST"
ALL_METHODS = [METHOD_DM]


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

    At ρ = 0.0: traces unchanged.
    At ρ = 1.0: all (n-1) adjacent pairs selected and swapped.

    Labels are NOT modified; discriminative signal is unaffected.
    Holds must be recomputed after applying this operator.
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

    σ_trace: fully shuffle activities within each trace of L_{ε,ρ}.
             → p_struct^(b) ~ U(0,1) regardless of residual structural signal.
    σ_label: replace outcomes with permuted_labels.
             → p_disc^(b) ~ U(0,1).

    Every acceptance on this doubly-null log is a false positive.
    """
    rng  = np.random.RandomState(trace_seed)
    null = {}

    for i, cid in enumerate(case_ids_sorted):
        ci_orig  = case_data_perturbed[cid]
        ci       = copy.copy(ci_orig)
        ci.outcome = int(permuted_labels[i])

        shuffled         = ci_orig.trace.copy()
        rng.shuffle(shuffled)
        ci.trace         = shuffled
        ci.activity_index = precompute_activity_index(shuffled, case_id=cid)
        null[cid] = ci

    return null


# ═══════════════════════════════════════════════════════════════════════════
# DeclareMiner FST ON PERTURBED LOG
# ═══════════════════════════════════════════════════════════════════════════

def run_dm_on_perturbed_log(
    case_data_perturbed: dict,
    candidates_all: list,
    tau_c: np.ndarray,
    tau_min: float,
    eligible: np.ndarray = None,
    holds_precomputed: dict = None,
) -> dict:
    """
    Apply the FROZEN τ_DM*(c) thresholds to a perturbed log L_{ε,ρ}.

    τ_DM*(c) is computed ONCE on the original data and held fixed across all
    49 grid cells.  The holds are reused from the ρ-row precomputation step
    (label noise does not change trace structure, so holds remain valid).

    Steps:
        1. Obtain holds (precomputed fast path or fresh computation).
        2. Split by class → ids_class0, ids_class1.
        3. Compute conf0, conf1 from holds (n_app counts implicit in holds).
        4. Apply FROZEN τ_DM*(c):  S^DM_{ε,ρ} = {c : |Δconf̂(c)| ≥ τ_DM*(c)
                                                   AND interestingness guard}.

    No internal permutation budget.  O(m) per cell — deterministic.
    τ_DM*(c) is NOT recomputed from the perturbed-log confidences.

    Parameters
    ----------
    case_data_perturbed : L_{ε,ρ} with label noise applied.
    candidates_all      : Fixed M_all candidate pool.
    tau_c               : (m,) constraint-specific thresholds from the original log.
                          FROZEN — not recomputed here.
    tau_min             : Minimum confidence interestingness guard.
    holds_precomputed   : Holds from the ρ-row precomputation (fast path).
                          If None, holds are computed from case_data_perturbed.

    Returns
    -------
    dict with keys:
        S_set   : frozenset — accepted constraints.
        R_obs   : int — |S^DM_{ε,ρ}|.
        conf0   : (m,) float64 — class-0 confidences on perturbed log.
        conf1   : (m,) float64 — class-1 confidences on perturbed log.
    """
    m = len(candidates_all)

    if holds_precomputed is not None:
        holds = holds_precomputed
    else:
        with _suppress():
            holds = compute_holds_by_case_batch(case_data_perturbed, candidates_all)

    D_0, D_1   = split_by_class(case_data_perturbed)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    if n0 == 0 or n1 == 0:
        return {
            'S_set': frozenset(),
            'R_obs': 0,
            'conf0': np.zeros(m, dtype=np.float64),
            'conf1': np.zeros(m, dtype=np.float64),
        }

    _, _, conf0, conf1 = compute_support_from_holds(
        holds, candidates_all, ids_class0, ids_class1, n0, n1
    )

    # Apply FROZEN original-log thresholds — tau_c and eligible are NOT recomputed here.
    rejected = apply_threshold_decision(conf0, conf1, tau_c=tau_c, tau_min=tau_min, eligible=eligible)
    S_set    = frozenset(candidates_all[i] for i in range(m) if rejected[i])

    return {
        'S_set': S_set,
        'R_obs': int(rejected.sum()),
        'conf0': conf0,
        'conf1': conf1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE-ANCHORED METRICS  (primary RQ2 metrics)
# ═══════════════════════════════════════════════════════════════════════════

def compute_reference_metrics(
    S_pert:      frozenset,
    S_orig_full: frozenset,
    S_orig_rq2:  frozenset,
) -> dict:
    """
    Reference-anchored performance metrics for RQ2 degradation analysis.

    S_orig_full = DM-FST acceptance set on original data ({c: |Δconf̂| ≥ τ_DM*(c)}).
    S_orig_rq2  = DM-FST output at (ε=0, ρ=0) using frozen τ_DM*(c).

        TP = |S_pert ∩ S_orig_full|
        FP = |S_pert \ S_orig_full|
        FN = |S_orig_full \ S_pert|

    FDR_ref = FP / R_obs
    Recall  = TP / R_full
    Jaccard_rq2 = |S_pert ∩ S_orig_rq2| / |S_pert ∪ S_orig_rq2|

    Edge cases:
        R_obs = 0: Precision = NaN, FDR_ref = NaN (undefined).
        R_full = 0: Recall = NaN (degenerate experiment).
    """
    R_obs  = len(S_pert)
    R_full = len(S_orig_full)

    TP = len(S_pert & S_orig_full)
    FP = len(S_pert - S_orig_full)
    FN = len(S_orig_full - S_pert)

    if R_obs == 0:
        precision = float('nan')
        fdr_ref   = float('nan')
        estimable = False
    else:
        precision = TP / R_obs
        fdr_ref   = FP / R_obs
        estimable = True

    recall = TP / R_full if R_full > 0 else float('nan')

    if (not estimable) or (precision != precision) or (recall != recall):
        f1 = float('nan')
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    union_rq2   = len(S_pert | S_orig_rq2)
    inter_rq2   = len(S_pert & S_orig_rq2)
    jaccard_rq2 = inter_rq2 / union_rq2 if union_rq2 > 0 else 1.0

    union_full   = len(S_pert | S_orig_full)
    jaccard_full = TP / union_full if union_full > 0 else 1.0

    fp_over_rfull = FP / R_full if R_full > 0 else float('nan')
    fn_over_rfull = FN / R_full if R_full > 0 else float('nan')

    return {
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
        'FP_over_Rfull': fp_over_rfull,
        'FN_over_Rfull': fn_over_rfull,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FDR HELPERS
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


def _null_fdr_skipped(reason: str) -> dict:
    return {
        'FDR_emp':        float('nan'),
        'FDR_CI_lower':   float('nan'),
        'FDR_CI_upper':   float('nan'),
        'E_V_b':          float('nan'),
        'FWER_emp':       float('nan'),
        'controls_FDR':   None,
        'estimable':      False,
        'skipped_reason': reason,
    }


def _compute_fdr_from_null(
    null_counts: np.ndarray,
    R_obs: int,
    alpha_nominal: float,
) -> dict:
    ev = float(np.mean(null_counts))
    if R_obs == 0:
        return {
            'FDR_emp':        float('nan'),
            'FDR_CI_lower':   float('nan'),
            'FDR_CI_upper':   float('nan'),
            'E_V_b':          ev,
            'FWER_emp':       float(np.mean(null_counts > 0)),
            'controls_FDR':   None,
            'estimable':      False,
            'skipped_reason': 'R_obs=0: FDR_emp undefined',
        }
    arr = null_counts.astype(float) / R_obs
    fdr = float(np.mean(arr))
    try:
        lo, hi = _bca_ci(arr)
    except Exception:
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        'FDR_emp':        fdr,
        'FDR_CI_lower':   lo,
        'FDR_CI_upper':   hi,
        'E_V_b':          ev,
        'FWER_emp':       float(np.mean(null_counts > 0)),
        'controls_FDR':   bool(fdr <= alpha_nominal),
        'estimable':      True,
        'skipped_reason': None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NULL REPLICATE RUNNER FOR ANCHOR CELLS (DeclareMiner FST)
# ═══════════════════════════════════════════════════════════════════════════

def _one_anchor_replicate_dm(
    b:                   int,
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    perm_labels_b:       np.ndarray,
    tau_c:               np.ndarray,
    tau_min:             float,
    base_seed:           int,
    eligible:            np.ndarray = None,
) -> dict:
    """
    One doubly-null anchor replicate → {METHOD_DM: int} false positive count.

    Top-level (loky-safe). Called by run_null_replicates_for_level via Parallel.

    DM-FST null replicate is fully deterministic given the doubly-null log:
        1. Build σ_label ∘ σ_trace(L_{ε,ρ}).
        2. Recompute holds_null.
        3. Apply FROZEN τ_DM*(c): V_b = #{c: |Δconf̂_b(c)| ≥ τ_DM*(c)}.

    τ_DM*(c) is passed in unchanged from the original-log computation and
    NOT recomputed from the null-log confidences.

    No internal permutation budget. Contrast with P1 null reps (B1_null=75,
    B2_null=75 per rep). DM-FST null reps are ~100× faster per replicate.
    """
    rs_trace = base_seed + 100_000 + b

    null_cd = _build_doubly_null_log(
        case_data_perturbed, case_ids_sorted, perm_labels_b, trace_seed=rs_trace,
    )

    D0_b, D1_b = split_by_class(null_cd)
    if len(D0_b) < 5 or len(D1_b) < 5:
        return {METHOD_DM: 0}

    with _suppress():
        holds_null = compute_holds_by_case_batch(null_cd, candidates_all)

    # Apply FROZEN tau_c and eligible — NOT recomputed on null log.
    n_dm = run_declareminer_on_doubly_null_log(
        null_case_data = null_cd,
        candidates_all = candidates_all,
        tau_c          = tau_c,
        tau_min        = tau_min,
        eligible       = eligible,
        holds_all      = holds_null,
    )

    return {METHOD_DM: n_dm}


def run_null_replicates_for_level(
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    labels_perturbed:    np.ndarray,
    B_null:              int,
    tau_c:               np.ndarray,
    tau_min:             float,
    base_seed:           int,
    eligible:            np.ndarray = None,
    n_jobs:              int = 1,
) -> dict:
    """
    Run B_null doubly-null replicates on a PERTURBED log (anchor cells only).

    Each replicate applies σ_label ∘ σ_trace to case_data_perturbed.
    Every DM-FST acceptance is a false positive by construction.

    The FROZEN τ_DM*(c) from the original log is applied to the null-log
    confidences.  τ_DM*(c) is never recomputed per replicate.

    DM-FST null reps have zero internal budget — holds → conf → count.

    n_jobs=1 (INNER_N_JOBS default) avoids nested parallelism under the outer
    joblib grid.  Set n_jobs=-1 for standalone SLURM array jobs.

    Returns:
        dict: {METHOD_DM: (B_null,) int array}
    """
    rng_outer = np.random.RandomState(base_seed)
    perm_labels_all = np.stack([
        rng_outer.permutation(labels_perturbed).astype(np.int8)
        for _ in range(B_null)
    ], axis=0)

    reps = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_anchor_replicate_dm)(
            b, case_data_perturbed, candidates_all, case_ids_sorted,
            perm_labels_all[b], tau_c, tau_min, base_seed, eligible,
        )
        for b in range(B_null)
    )

    counts = {METHOD_DM: np.zeros(B_null, dtype=int)}
    for b, r in enumerate(reps):
        counts[METHOD_DM][b] = r[METHOD_DM]
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# JOINT DRAW RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def _run_dm_on_joint_draw(
    eps:             float,
    eps_seed:        int,
    struct_log:      dict,
    holds_for_rho:   dict,
    candidates_all:  list,
    case_ids_sorted: list,
    tau_c:           np.ndarray,
    tau_min:         float,
    eligible:        np.ndarray = None,
) -> dict:
    """
    Apply N_label(ε) to struct_log and run DM-FST with FROZEN τ_DM*(c).

    struct_log already has ρ applied (precomputed per ρ row).
    This function applies only the label-noise layer (ε) on top, then calls
    run_dm_on_perturbed_log with the reused holds_for_rho.

    τ_DM*(c) is passed in from the original-log computation and not
    recomputed from the perturbed-log confidences.

    ε = 0.0 → struct_log unchanged (holds remain valid by identity).
    ε > 0.0 → flip each case label independently with probability ε.
              Holds remain valid (label noise does not change trace structure).
    """
    case_data_pert = apply_joint_perturbation(struct_log, eps, eps_seed)

    labels_pert = np.array([
        case_data_pert[cid].outcome for cid in case_ids_sorted
    ], dtype=np.int8)

    dm_res = run_dm_on_perturbed_log(
        case_data_perturbed = case_data_pert,
        candidates_all      = candidates_all,
        tau_c               = tau_c,
        tau_min             = tau_min,
        eligible            = eligible,
        holds_precomputed   = holds_for_rho,
    )

    return {
        'case_data_pert': case_data_pert,
        'labels_pert':    labels_pert,
        'S_pert':         dm_res['S_set'],
        'R_obs':          dm_res['R_obs'],
        'dm_res':         dm_res,
        'eps_seed_used':  eps_seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PER-CELL ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def analyze_joint_cell(
    eps:             float,
    rho:             float,
    eps_idx:         int,
    rho_idx:         int,
    struct_log:      dict,
    holds_for_rho:   dict,
    candidates_all:  list,
    case_ids_sorted: list,
    S_orig_full:     frozenset,
    S_orig_rq2:      frozenset,
    tau_c:           np.ndarray,
    tau_min:         float,
    eligible:        np.ndarray = None,
    is_anchor_bootstrap: bool = False,
) -> dict:
    """
    Full analysis for one joint noise cell (ε, ρ), DeclareMiner FST only.

    Seed scheme: BASE_SEED + eps_idx * 1000 + rho_idx (reproducible per cell).

    Retry logic: if DM-FST returns R_obs=0, redraw with a new eps_seed
    (structural log is fixed — only the label-flip draw changes).

    Anchor cells: run B_NULL_ANCHOR doubly-null replicates for FDR_emp.
    (0,0) anchor (is_anchor_bootstrap=True): S_orig_rq2 = S_pert (bootstrap).
    All other cells: skip doubly-null (FDR_ref is primary).
    """
    print(f"  [ε={eps:.2f}  ρ={rho:.2f}]  Starting DM-FST analysis...", flush=True)
    t0 = time.time()

    base_seed_cell = BASE_SEED + eps_idx * 1000 + rho_idx

    MAX_RETRIES = 5
    RETRY_PRIME = 999983

    draw      = None
    n_retries = 0

    for retry in range(MAX_RETRIES + 1):
        eps_seed_try = base_seed_cell + 1 + retry * RETRY_PRIME

        draw = _run_dm_on_joint_draw(
            eps             = eps,
            eps_seed        = eps_seed_try,
            struct_log      = struct_log,
            holds_for_rho   = holds_for_rho,
            candidates_all  = candidates_all,
            case_ids_sorted = case_ids_sorted,
            tau_c           = tau_c,
            tau_min         = tau_min,
        )

        if draw['R_obs'] > 0:
            if retry > 0:
                print(
                    f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
                    f"R_obs>0 after {retry} retr{'y' if retry==1 else 'ies'}.",
                    flush=True,
                )
            n_retries = retry
            break

        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
            f"R_obs=0 on draw {retry} — ",
            end="", flush=True,
        )
        if retry < MAX_RETRIES:
            print("retrying...", flush=True)
        else:
            print(
                f"MAX_RETRIES={MAX_RETRIES} exhausted. "
                f"R_obs=0 is genuine signal at (ε={eps}, ρ={rho}).",
                flush=True,
            )
            n_retries = retry

    case_data_pert = draw['case_data_pert']
    labels_pert    = draw['labels_pert']
    S_pert         = draw['S_pert']
    R_obs          = draw['R_obs']
    eps_seed_used  = draw['eps_seed_used']

    # S_orig_rq2: at (0,0) anchor, own output IS the baseline
    _S_rq2 = S_pert if (is_anchor_bootstrap or S_orig_rq2 is None) else S_orig_rq2

    ref_metrics = compute_reference_metrics(S_pert, S_orig_full, _S_rq2)

    is_anchor = is_anchor_bootstrap or (eps, rho) in ANCHOR_CELLS

    null_counts  = np.zeros(0, dtype=int)
    B_null_this  = B_NULL_ANCHOR if (is_anchor and R_obs > 0) else 0

    if R_obs == 0:
        fdr_null_metrics = _null_fdr_skipped('R_obs=0: FDR_emp undefined')
    elif B_null_this == 0:
        fdr_null_metrics = _null_fdr_skipped('non-anchor cell')
    else:
        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
            f"Running {B_null_this} DM-FST null replicates "
            f"(deterministic, no internal budget)...",
            flush=True,
        )
        counts_dict = run_null_replicates_for_level(
            case_data_perturbed = case_data_pert,
            candidates_all      = candidates_all,
            case_ids_sorted     = case_ids_sorted,
            labels_perturbed    = labels_pert,
            B_null              = B_null_this,
            tau_c               = tau_c,
            tau_min             = tau_min,
            base_seed           = base_seed_cell + 50_000,
            n_jobs              = INNER_N_JOBS,
        )
        null_counts      = counts_dict[METHOD_DM]
        fdr_null_metrics = _compute_fdr_from_null(null_counts, R_obs, ALPHA)

    wall = time.time() - t0
    _fdr_emp_str = 'n/a' if B_null_this == 0 else f"{fdr_null_metrics['FDR_emp']:.4f}"
    print(
        f"  [ε={eps:.2f}  ρ={rho:.2f}]  Done in {wall:.1f}s  "
        f"R_obs={R_obs}  FDR_ref={ref_metrics['FDR_ref']:.4f}  "
        f"FDR_emp={_fdr_emp_str}",
        flush=True,
    )

    result = {
        'eps':           eps,
        'rho':           rho,
        'eps_idx':       eps_idx,
        'rho_idx':       rho_idx,
        'R_obs':         R_obs,
        'S_pert':        list(S_pert),
        'ref_metrics':   ref_metrics,
        'fdr_null':      fdr_null_metrics,
        'null_counts':   null_counts.tolist(),
        'B_null':        B_null_this,
        'wall_seconds':  wall,
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

def run_original_data_dm() -> dict:
    """
    Run DeclareMiner on the original (unperturbed) Sepsis ICU log.

    Delegates to run_declareminer() from DeclareMiner_Sepsis.py.

    S_orig_full = {c : |Δconf̂(c)| ≥ τ_delta_conf} on the original log.
    This serves as the ground truth for FDR_ref and Recall across all 49 cells.

    Also computes holds_all on the original data for reuse at ρ=0 row.

    Returns
    -------
    dict with:
        case_data, candidates_all, case_ids_sorted, labels_orig,
        S_orig_full (frozenset), tau_c (m,) array, tau_min (float),
        tau_delta_conf (float), holds_all (dict).
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — ORIGINAL DATA RUN (DeclareMiner / Uniform Fixed Threshold)")
    print(f"  τ_DM*(c) = τ_delta_conf = {DM_CONFIG['tau_delta_conf']}  [uniform fixed threshold, no sample-size scaling]")
    print(f"  τ_effect={DM_CONFIG['tau_effect']}  n_ref={DM_CONFIG['n_ref']}  n_floor={DM_CONFIG['n_floor']}")
    print(f"  τ_DM*(c) and eligibility mask fixed from original log — never recalibrated.")
    print("=" * 100)

    t0     = time.time()
    dm_out = run_declareminer(config=DM_CONFIG.copy())
    wall   = time.time() - t0

    case_data      = dm_out['case_data']
    candidates_all = dm_out['candidates_all']
    tau_c          = dm_out['tau_c']
    eligible       = dm_out['eligible']
    tau_effect     = float(dm_out['tau_effect'])
    n_ref          = float(dm_out['n_ref'])
    n_floor        = int(dm_out['n_floor'])
    tau_min        = float(dm_out['config']['tau_min'])

    S_orig_full = frozenset(
        candidates_all[i]
        for i in range(dm_out['m_total'])
        if dm_out['rejected'][i]
    )
    case_ids_sorted = sorted(case_data.keys())
    labels_orig     = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    print(f"\n  DM-FST complete: {wall:.1f}s")
    print(f"  M_all={dm_out['m_total']:,}  |  τ_effect={tau_effect}  |  n_ref={n_ref}  "
          f"|  n_floor={n_floor}  |  τ_min={tau_min}  |  R_obs^DM={dm_out['n_rejected']:,}")
    print(f"  τ_DM*(c): mean={tau_c.mean():.4f}  median={np.median(tau_c):.4f}  "
          f"min={tau_c.min():.6f}  max={tau_c.max():.4f}")
    print(f"  eligible: {int(eligible.sum()):,} / {len(eligible):,} constraints pass sparsity floor")
    print(f"  |S_orig_full| = {len(S_orig_full):,}")

    # Compute holds_all on original data for reuse at ρ=0 row
    print("\n  Computing holds_all on original data (ρ=0 row reuse)...", flush=True)
    t0_holds = time.time()
    with _suppress():
        holds_all = compute_holds_by_case_batch(case_data, candidates_all)
    print(f"  holds_all computed in {time.time()-t0_holds:.1f}s")

    return {
        'case_data':       case_data,
        'candidates_all':  candidates_all,
        'case_ids_sorted': case_ids_sorted,
        'labels_orig':     labels_orig,
        'S_orig_full':     S_orig_full,
        'tau_c':           tau_c,
        'eligible':        eligible,
        'tau_effect':      tau_effect,
        'n_ref':           n_ref,
        'n_floor':         n_floor,
        'tau_min':         tau_min,
        'holds_all':       holds_all,
        'dm_out':          dm_out,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — RQ2 JOINT PERTURBATION GRID
# ═══════════════════════════════════════════════════════════════════════════

def _joint_cell_worker(
    eps, rho, eps_idx, rho_idx,
    struct_log, holds_for_rho,
    candidates_all, case_ids_sorted,
    S_orig_full, S_orig_rq2, tau_c, tau_min,
):
    """Top-level loky-safe worker for one (ε, ρ) joint grid cell."""
    return analyze_joint_cell(
        eps             = eps,
        rho             = rho,
        eps_idx         = eps_idx,
        rho_idx         = rho_idx,
        struct_log      = struct_log,
        holds_for_rho   = holds_for_rho,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        S_orig_full     = S_orig_full,
        S_orig_rq2      = S_orig_rq2,
        tau_c           = tau_c,
        tau_min         = tau_min,
        is_anchor_bootstrap = False,
    )


def run_joint_grid(
    case_data_orig:  dict,
    candidates_all:  list,
    case_ids_sorted: list,
    S_orig_full:     frozenset,
    tau_c:           np.ndarray,
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
               compute holds once.  All 7 ε cells in that row share the same
               structural log and holds.
        Total holds computations: 1 (ρ=0) + 6 (ρ>0) = 7.

    Step 1 — Run (0,0) anchor SEQUENTIALLY to establish S_orig_rq2.
        S_orig_rq2 = DM-FST output at (ε=0, ρ=0) using frozen τ_DM*(c).
        Jaccard_rq2 = 1.0 at (0,0) by construction.

    Step 2 — Run remaining 48 cells in parallel (n_jobs workers).
        Each cell worker: apply label noise → DM-FST with frozen τ_DM*(c).
        No recalibration step.  O(m) per cell.

    Returns
    -------
    (all_results, S_orig_rq2)
    """
    print("\n" + "=" * 100)
    print("SECTION 2 — JOINT PERTURBATION GRID  (B.3: Label × Structural, DeclareMiner FST)")
    print(f"  ε levels: {JOINT_LABEL_LEVELS}")
    print(f"  ρ levels: {JOINT_STRUCT_LEVELS}")
    print(f"  Grid size: {len(JOINT_GRID)} cells  |  Anchor cells: {sorted(ANCHOR_CELLS)}")
    print(f"  B_null (anchor)={B_NULL_ANCHOR}  (non-anchor=0, FDR_ref is primary)")
    print(f"  DM-FST null reps: deterministic — no internal permutation budget")
    print(f"  τ_DM*(c): frozen from original log, applied to all 49 cells")
    print(f"  n_jobs={n_jobs}")
    print("=" * 100)

    t0 = time.time()

    # ── Step 0: Precompute one structural log + holds per ρ row ──────────
    print("\n  [Step 0] Precomputing structural logs and holds per ρ row...", flush=True)
    struct_logs = {}
    holds_cache = {}
    for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS):
        if rho == 0.0:
            struct_logs[rho] = case_data_orig
            holds_cache[rho] = holds_orig
            print(f"    ρ={rho:.2f}  → reusing original holds", flush=True)
        else:
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
        struct_log      = struct_logs[0.0],
        holds_for_rho   = holds_cache[0.0],
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        S_orig_full     = S_orig_full,
        S_orig_rq2      = None,
        tau_c           = tau_c,
        tau_min         = tau_min,
        is_anchor_bootstrap = True,
    )

    S_orig_rq2 = frozenset(anchor_result['S_pert_raw'])

    print(f"\n  S_orig_rq2 established (single (0,0) baseline for all 49 cells):")
    print(f"    [DM-FST]: {len(S_orig_rq2):,} patterns  "
          f"(vs {len(S_orig_full):,} in S_orig_full; "
          f"both use frozen τ_DM*(c) on original holds)")

    # ── Step 2: Run remaining 48 cells in parallel ─────────────────────
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
            S_orig_full, S_orig_rq2, tau_c, tau_min,
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
    Build a long-format metrics table for the 7×7 joint noise grid (DM-FST only).

    One row per (ε, ρ).  Columns:
      eps, rho — noise levels.
      Primary   : FDR_ref, Precision, Recall, F1, TP, FP, FN, R_full,
                  Jaccard_rq2, Jaccard_full, Gained_rq2, Lost_rq2, reliable.
      Secondary : FP_over_Rfull, FN_over_Rfull.
      Doubly-null (anchor cells only):
                  FDR_emp, FDR_CI_lower, FDR_CI_upper, FWER_emp, E_V_b,
                  controls_FDR, estimable, skipped_reason.
      Provenance: B_null, is_anchor, n_retries, eps_seed_used.
    """
    rows = []
    for res in all_results:
        rm  = res['ref_metrics']
        fdr = res['fdr_null']
        rows.append({
            'eps':             res['eps'],
            'rho':             res['rho'],
            'method':          METHOD_DM,
            'R_obs':           res['R_obs'],
            # ── Primary reference-anchored metrics ──────────────────────
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
            # ── Secondary oracle-normalised burden metrics ───────────────
            'FP_over_Rfull':   rm['FP_over_Rfull'],
            'FN_over_Rfull':   rm['FN_over_Rfull'],
            # ── Doubly-null (anchor cells only) ──────────────────────────
            'FDR_emp':         fdr['FDR_emp'],
            'FDR_CI_lower':    fdr.get('FDR_CI_lower',  float('nan')),
            'FDR_CI_upper':    fdr.get('FDR_CI_upper',  float('nan')),
            'FWER_emp':        fdr.get('FWER_emp',      float('nan')),
            'E_V_b':           fdr.get('E_V_b',         float('nan')),
            'controls_FDR':    fdr.get('controls_FDR',  None),
            'estimable':       fdr.get('estimable',     False),
            'skipped_reason':  fdr.get('skipped_reason', None),
            # ── Provenance ───────────────────────────────────────────────
            'B_null':          res['B_null'],
            'is_anchor':       res.get('is_anchor',     False),
            'n_retries':       res.get('n_retries',     0),
            'eps_seed_used':   res.get('eps_seed_used', None),
        })
    return pd.DataFrame(rows)


def save_joint_outputs(
    all_results: list,
    orig_data:   dict,
    S_orig_rq2:  frozenset,
    total_wall:  float,
) -> None:
    """
    Save all DM-FST joint-design RQ2 output files.

    Files written to RQ2_OUTPUT_DIR:
        rq2_dm_fst_joint_metrics.csv           Long-format table (49 cells).
        rq2_dm_fst_joint_<metric>_DM_FST.csv  7×7 pivot heatmap per metric.
        rq2_dm_fst_joint_null_counts.json      Raw V_b arrays per (ε, ρ).
        rq2_dm_fst_joint_results.json          Full structured results for paper.
    """
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)
    S_orig_full    = orig_data['S_orig_full']
    tau_c          = orig_data['tau_c']
    tau_delta_conf = float(tau_c[0]) if len(tau_c) > 0 else float(DM_CONFIG['tau_delta_conf'])
    tau_min        = orig_data['tau_min']
    m_total        = orig_data['dm_out']['m_total']

    # ── Long-format CSV ───────────────────────────────────────────────────
    df      = build_joint_metrics_table(all_results)
    lf_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_dm_fst_joint_metrics.csv')
    df.to_csv(lf_path, index=False)
    print(f"  Saved: {lf_path}")

    # ── Heatmap pivot CSVs per metric ─────────────────────────────────────
    pivot_metrics = ['FDR_ref', 'Recall', 'F1', 'Jaccard_rq2',
                     'FP_over_Rfull', 'FN_over_Rfull', 'Jaccard_full', 'R_obs']
    for metric in pivot_metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot(index='eps', columns='rho', values=metric)
        pivot.index.name = 'eps \\ rho'
        pivot_path = os.path.join(
            RQ2_OUTPUT_DIR,
            f'rq2_dm_fst_joint_{metric.lower()}_DM_FST.csv',
        )
        pivot.to_csv(pivot_path)
        print(f"  Saved: {pivot_path}")

    # ── Null counts JSON ──────────────────────────────────────────────────
    null_counts_json = {}
    for res in all_results:
        key = f"eps_{res['eps']:.4f}_rho_{res['rho']:.4f}"
        null_counts_json[key] = res['null_counts']
    nc_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_dm_fst_joint_null_counts.json')
    with open(nc_path, 'w', encoding='utf-8') as f:
        json.dump(null_counts_json, f, indent=2)
    print(f"  Saved: {nc_path}")

    # ── Full results JSON ─────────────────────────────────────────────────
    full_json = {
        'rq2_version': '2.0',
        'method':      METHOD_DM,
        'log_name':    'Sepsis',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'block':        'B.3 Joint Label × Structural Noise',
            'method':       METHOD_DM,
            'shared_pool':  'M_all from Phase 0 DECLARE spec (fixed; same as P1 and DRVA)',
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
            'null_protocol':  (
                'Doubly-null: σ_label ∘ σ_trace on L_{ε,ρ} (anchor cells only). '
                'FROZEN τ_DM*(c) applied to null-log confidences.'
            ),
            'DM_FST_gate':    (
                f'|Δconf̂(c)| >= τ_delta_conf = {tau_delta_conf:.4f}  AND (conf0 >= {tau_min} OR conf1 >= {tau_min}). '
                'Uniform fixed threshold — no sample-size scaling. '
                'τ_delta_conf fixed from original data. No statistical test. No FDR correction.'
            ),
            'DM_FST_null_budget': (
                'Zero internal permutation iterations per null replicate. '
                'DM-FST null reps are fully deterministic given the doubly-null log.'
            ),
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
                'R':          int(len(S_orig_full)),
                'description': (
                    'DM-FST acceptance set on original data: '
                    '{c : |Δconf̂(c)| ≥ τ_DM*(c) AND interestingness guard}. '
                    'Ground truth for FDR_ref and Recall.'
                ),
            },
            'S_orig_rq2': {
                'R':          int(len(S_orig_rq2)),
                'description': (
                    f'DM-FST output at (ε=0, ρ=0) using frozen τ_DM*(c). '
                    'Jaccard_rq2=1.0 at (0,0) by construction.'
                ),
            },
        },

        'config': {
            'ALPHA':              ALPHA,
            'B_NULL_ANCHOR':      B_NULL_ANCHOR,
            'BASE_SEED':          BASE_SEED,
            'tau_delta_conf':     float(tau_delta_conf),
            'tau_min':            float(tau_min),
            'tau_c_mean':         float(tau_c.mean()),
            'tau_c_median':       float(np.median(tau_c)),
            'tau_c_min':          float(tau_c.min()),
            'tau_c_max':          float(tau_c.max()),
            'R_obs_dm_orig':      int(len(S_orig_full)),
            'm_total':            int(m_total),
        },

        'original_data': {
            'R_obs_orig':     int(len(S_orig_full)),
            'tau_delta_conf': float(tau_delta_conf),
            'tau_c_mean':     float(tau_c.mean()),
        },

        'speed_optimisations_applied': {
            'S1_holds_per_rho_row':        '7 holds computations total (not 49)',
            'S2_no_calibration_per_cell':  'DM-FST is deterministic — O(m) per cell',
            'S3_null_anchor_only':         f'FDR_emp at {len(ANCHOR_CELLS)} corner cells only',
            'S4_null_reps_deterministic':  'Zero internal permutation budget per null rep',
            'S5_cell_parallelism':         f'n_jobs={N_JOBS}',
        },

        'timing': {'total_seconds': total_wall},

        'joint_results': [
            {k: v for k, v in r.items() if k not in ('S_pert', 'S_pert_raw')}
            for r in all_results
        ],
    }

    json_path = os.path.join(RQ2_OUTPUT_DIR, 'rq2_dm_fst_joint_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — SUMMARY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt='.4f') -> str:
    return f"{v:{fmt}}" if v == v else '  NaN'


def _print_joint_summary(all_results: list) -> None:
    """Print compact FDR_ref heatmap and a full degradation table (DM-FST)."""
    print(f"\n  FDR_ref heatmap — {METHOD_DM}")
    print(f"  {'ε \\ ρ':>8s}", end='')
    for rho in JOINT_STRUCT_LEVELS:
        print(f"  {rho:>6.2f}", end='')
    print()
    print(f"  {'─'*62}")
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
                v = res['ref_metrics']['FDR_ref']
                print(f"  {_fmt(v, '.4f'):>6s}", end='')
        print()
    print(f"  {'─'*62}")

    print(f"\n  {'─'*110}")
    print(f"  {'ε':>5s}  {'ρ':>5s}  {'R_obs':>6s}  "
          f"{'FDR_ref':>8s}  {'Recall':>7s}  {'F1':>6s}  "
          f"{'J_rq2':>7s}  {'FP/Rf':>7s}  "
          f"{'FDR_emp':>8s}  {'anchor':>6s}")
    print(f"  {'─'*110}")
    for res in sorted(all_results, key=lambda x: (x['eps'], x['rho'])):
        rm  = res['ref_metrics']
        fdr = res['fdr_null']
        fe  = fdr['FDR_emp']
        _fe_str = '  n/a  ' if fe != fe else f'{fe:.4f}'
        print(
            f"  {res['eps']:>5.2f}  {res['rho']:>5.2f}  {res['R_obs']:>6d}  "
            f"{_fmt(rm['FDR_ref']):>8s}  "
            f"{_fmt(rm['Recall']):>7s}  "
            f"{_fmt(rm['F1']):>6s}  "
            f"{rm['Jaccard_rq2']:>7.4f}  "
            f"{_fmt(rm['FP_over_Rfull']):>7s}  "
            f"{_fe_str:>8s}  "
            f"{'YES' if res.get('is_anchor') else '-':>6s}"
        )
    print(f"  {'─'*110}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ2 — SPECIFICATION QUALITY DEGRADATION: SEPSIS  [DeclareMiner FST / Option A]")
    print("Block B.3: Joint Label × Structural Noise  (7×7 = 49 cells)")
    print("Per-Constraint Z-Test Threshold Baseline")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ε grid:  {JOINT_LABEL_LEVELS}")
    print(f"  ρ grid:  {JOINT_STRUCT_LEVELS}")
    print(f"  Grid:    {len(JOINT_GRID)} cells  |  Anchors: {sorted(ANCHOR_CELLS)}")
    print(f"  τ_delta_conf={DM_CONFIG['tau_delta_conf']}  (uniform fixed threshold — no per-comparison alpha)")
    print(f"  α(nominal FDR)={ALPHA}  "
          f"(DM-FST does not control FDR; for comparison only)")
    print(f"  B_null(anchor FDR_emp)={B_NULL_ANCHOR}  "
          f"(non-anchor=0, FDR_ref is primary)")
    print(f"  DM-FST null reps: O(m) deterministic — no internal permutation budget")
    print(f"  τ_DM*(c) frozen from original log — applied to all 49 perturbed cells")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")
    print(f"\n  Primary metrics:   FDR_ref=FP/R_obs, Recall=TP/R_full, F1, Jaccard_rq2")
    print(f"  Secondary metrics: FP_over_Rfull=FP/R_full, FN_over_Rfull, Jaccard_full")
    print(f"\n  Reference sets:")
    print(f"    S_orig_full — DM-FST acceptance on original data; ground truth")
    print(f"    S_orig_rq2  — single (0,0) baseline; Jaccard_rq2=1 at origin")
    print(f"\n  Mechanistic prediction:")
    print(f"    (0,0): FDR_ref≈0, Recall≈1 (no noise)")
    print(f"    ε→0.5: conf1-conf0→0 → |Δconf̂|→0 → R_obs→0 → FDR_ref undefined")
    print(f"    ρ→1.0: conf differences collapse → Recall drops")
    print(f"    DM-FST has no principled noise boundary — degrades without early warning")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    # Section 1: DM-FST original-data run
    orig_data = run_original_data_dm()

    # Section 2: joint perturbation grid
    all_results, S_orig_rq2 = run_joint_grid(
        case_data_orig  = orig_data['case_data'],
        candidates_all  = orig_data['candidates_all'],
        case_ids_sorted = orig_data['case_ids_sorted'],
        S_orig_full     = orig_data['S_orig_full'],
        tau_c           = orig_data['tau_c'],
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
    print("RQ2 — SEPSIS DeclareMiner FST COMPLETE  (Joint Noise, single method)")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  τ_DM*(c) mean={orig_data['tau_c'].mean():.4f}  "
          f"|S_orig_full|={len(orig_data['S_orig_full']):,}")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")

    _print_joint_summary(all_results)

    print(f"\n{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 Specification Degradation — Sepsis (DeclareMiner FST / Option A baseline)  "
            "B.3 Joint Label × Structural Noise, 7×7 grid"
        )
    )
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Parallel workers for cell grid (-1=all cores, default:{N_JOBS})',
    )
    parser.add_argument(
        '--alpha', type=float, default=ALPHA,
        help=f'Nominal FDR level for pass/fail verdict (default:{ALPHA})',
    )
    parser.add_argument(
        '--b-null-anchor', type=int, default=B_NULL_ANCHOR,
        help=f'Null replicates for anchor cells (default:{B_NULL_ANCHOR})',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Smoke test: 3×3 subgrid only, B_null_anchor=2',
    )
    args = parser.parse_args()

    if args.dry_run:
        JOINT_LABEL_LEVELS  = [0.00, 0.10, 0.50]
        JOINT_STRUCT_LEVELS = [0.00, 0.10, 1.00]
        JOINT_GRID[:] = [
            (eps, rho)
            for eps in JOINT_LABEL_LEVELS
            for rho in JOINT_STRUCT_LEVELS
        ]
        ANCHOR_CELLS = frozenset({
            (0.00, 0.00),
            (0.50, 0.00),
            (0.00, 1.00),
            (0.50, 1.00),
        })
        B_NULL_ANCHOR = 2
        print("*** DRY-RUN MODE: 3×3 grid, B_null_anchor=2 ***")
    else:
        B_NULL_ANCHOR = args.b_null_anchor

    N_JOBS = args.n_jobs
    ALPHA  = args.alpha

    main()
