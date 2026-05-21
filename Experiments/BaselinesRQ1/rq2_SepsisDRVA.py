#!/usr/bin/env python3
"""
rq2_SepsisDRVA.py  —  RQ2 Specification Quality Degradation: DRVA Baseline — Sepsis
=====================================================================================
Block B.3: Joint Label × Structural Noise Perturbation (2-D Cartesian grid)
DRVA baseline only (Cecconi et al. BPM Forum 2021)

RESEARCH QUESTION
-----------------
RQ2: As signal is progressively corrupted along BOTH noise axes simultaneously,
how does DRVA differ in:
  (a) the size of its discovered specifications,
  (b) the consistency of those specifications relative to the clean-data DRVA run
      (FDR_ref, Precision, Recall, F1; Jaccard_rq2 procedure stability), and
  (c) the empirical FDR estimates under the doubly-null protocol (anchor cells)?

METHOD
------
DRVA performs a permutation test on ΔConfidence (shuffleLog = label permutation
on pre-cached trace evaluations) per candidate rule r ∈ M_all.
No FDR correction is applied — each rule is evaluated at the per-rule raw α
threshold (α = 0.05).

WHAT DRVA'S NULL NULLIFIES
---------------------------
shuffleLog nullifies H₀ᵈ only (discriminative axis).
H₀ˢ (structural axis / within-trace ordering) is never nullified.
→ FDR_emp > α expected under doubly-null protocol.

JOINT PERTURBATION OPERATOR
-----------------------------
B.3 — Joint noise  N_label(ε) ∘ N_struct(ρ):

    L_{ε,ρ} = N_struct(ρ) ∘ N_label(ε)(L)

    N_label(ε): Flip outcome label of each case independently with prob ε.
    N_struct(ρ): For each trace, select round(ρ·(n−1)) adjacent pairs without
                 replacement and swap each. Destroys temporal ordering.

    2-D grid G = {(ε_i, ρ_j)}:
        ε ∈ {0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50}  (7 levels)
        ρ ∈ {0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00}  (7 levels)
        → 49 cells; (0,0) is the clean-data anchor.

REFERENCE SET
-------------
  S_orig_drva  — DRVA at (ε=0, ρ=0) on the original Sepsis log.
                 Ground truth for FDR_ref, Precision, Recall, F1.
                 Also the Jaccard_rq2 baseline (Jaccard_rq2 = 1.0 at (0,0)).

PRIMARY METRICS PER CELL
--------------------------
    FDR_ref     FP / R_obs             (reference-anchored false discovery rate)
    Precision   TP / R_obs  = 1 − FDR_ref
    Recall      TP / |S_orig_drva|     (power under noise)
    F1          Harmonic mean of Precision and Recall.
    Jaccard_rq2 |S_pert ∩ S_orig_drva| / |S_pert ∪ S_orig_drva|

ANCHOR CELLS (doubly-null FDR_emp)
------------------------------------
    (0.00, 0.00), (0.50, 0.00), (0.00, 1.00), (0.50, 1.00)

NULL REPLICATE PROTOCOL (anchor cells only)
--------------------------------------------
For each replicate b on L_{ε,ρ}:
  1. Apply σ_trace to L_{ε,ρ}: fully shuffle activities within each trace.
  2. Apply σ_label to σ_trace(L_{ε,ρ}): permute class labels.
  3. Recompute holds on doubly-null log.
  4. Run DRVA → V_b.

DRVA PERMUTATION BUDGET
------------------------
  DRVA_CONFIG (from DRVA_Sepsis):
      pi                   = 2000   (§3.3 paper default)
      alpha                = 0.05
      mmin                 = 0.0    (no-op → M_tested = M_all)
      mdiff_min            = 0.0    (no-op → M_tested = M_all)
      hierarchical_pruning = False  (DISABLED — preserves shared M_all)

  RQ2 overrides:
      PI_DRVA_REAL = 1_000  (π for real-data perturbed cells; 49 cells × 1000 iters)
      PI_DRVA_NULL = 100    (π per null replicate; speed — B_NULL_ANCHOR × 100 iters)

OUTPUT FILES
-------------
    rq2_drva_metrics.csv                  Long-format table (49 cells).
    rq2_drva_fdrref_pivot.csv             7×7 FDR_ref heatmap matrix.
    rq2_drva_<metric>_pivot.csv           Heatmap pivots for Recall, F1, etc.
    rq2_drva_null_counts.json             Raw V_b arrays per anchor cell.
    rq2_drva_results.json                 Full structured results for paper.

Version : 1.0  (DRVA-only; joint noise; α=0.05)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Cecconi, Augusto & Di Ciccio (2021). BPM Forum 2021, LNBIP 427, pp.73-91.
Pellegrina & Vandin (2018/2020). KDD 2018 / TKDD 2020.
Phipson & Smyth (2010). Stat. Appl. Genet. Mol. Biol. 9(1):Art.39.
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

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

# ── Data loading and trace utilities ─────────────────────────────────────
from P1_SDSM.p1_Sepsis_hou import (
    load_and_preprocess_data,
    generate_candidate_patterns,
    compute_holds_by_case_batch,
    precompute_activity_index,
    CaseInfo,
    INPUT_FILE        as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
)

# ── DRVA baseline ─────────────────────────────────────────────────────────
from BaselinesRQ1.DRVA_Sepsis import (
    run_drva,
    run_drva_on_doubly_null_log,
    DRVA_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH        = P1_INPUT_FILE
PHASE0_JSON     = P1_SPEC_FILE
RQ2_OUTPUT_DIR  = "RQ2_Sepsis_DRVA"

# ── 2-D Joint Perturbation Grid ───────────────────────────────────────────
JOINT_LABEL_LEVELS  = [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
JOINT_STRUCT_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

JOINT_GRID = [
    (eps, rho)
    for eps in JOINT_LABEL_LEVELS
    for rho in JOINT_STRUCT_LEVELS
]   # 49 cells

ANCHOR_CELLS = frozenset({
    (0.00, 0.00),
    (JOINT_LABEL_LEVELS[-1], 0.00),
    (0.00, JOINT_STRUCT_LEVELS[-1]),
    (JOINT_LABEL_LEVELS[-1], JOINT_STRUCT_LEVELS[-1]),
})

# ── FDR level ──────────────────────────────────────────────────────────────
ALPHA_DRVA = 0.05    # per-rule significance level (no FDR correction)

# ── DRVA permutation budgets ───────────────────────────────────────────────
# DRVA_CONFIG['pi'] = 2000 (paper default); overridden here for RQ2 speed.
PI_DRVA_REAL = 2_000   # π for real-data perturbed cells
PI_DRVA_NULL = 100     # π per null replicate (speed)

# ── Null replicate counts ──────────────────────────────────────────────────
B_NULL_ANCHOR       = 200   # doubly-null replicates at anchor cells
B_NULL_INTERMEDIATE = 0     # non-anchor: skip (FDR_ref is primary)

# ── Base seed ─────────────────────────────────────────────────────────────
BASE_SEED = 20260602

# ── Parallelism ────────────────────────────────────────────────────────────
N_JOBS       = -1
INNER_N_JOBS = 1   # 1 inside outer Parallel to avoid nested loky


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

def apply_label_noise(case_data: dict, epsilon: float, seed: int) -> dict:
    """
    Label noise operator N_label(ε).
    Flip each case's outcome label independently with probability ε.
    ε = 0.0 → original log unchanged.
    """
    if epsilon == 0.0:
        return case_data
    rng    = np.random.RandomState(seed)
    result = {}
    for cid, case in case_data.items():
        if rng.random() < epsilon:
            ci          = copy.copy(case)
            ci.outcome  = 1 - case.outcome
            result[cid] = ci
        else:
            result[cid] = case
    return result


def apply_structural_noise(case_data: dict, rho: float, seed: int) -> dict:
    """
    Structural noise operator N_struct(ρ).
    For each trace, swap round(ρ × (n-1)) randomly selected adjacent pairs.
    ρ = 0.0 → traces unchanged.
    """
    if rho == 0.0:
        return case_data
    rng    = np.random.RandomState(seed)
    result = {}
    for cid, case in case_data.items():
        trace = case.trace.copy()
        n     = len(trace)
        if n > 1:
            n_to_swap = int(round(rho * (n - 1)))
            if n_to_swap > 0:
                pair_idxs = rng.choice(n - 1, size=n_to_swap, replace=False)
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


def apply_joint_perturbation(struct_log: dict, eps: float, eps_seed: int) -> dict:
    """Apply label noise on top of an already structurally-perturbed log."""
    return apply_label_noise(struct_log, eps, eps_seed)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLY-NULL LOG BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _build_doubly_null_log(
    case_data_perturbed: dict,
    case_ids_sorted:     list,
    permuted_labels:     np.ndarray,
    trace_seed:          int,
) -> dict:
    """
    Apply σ_label ∘ σ_trace to a perturbed log.

    σ_trace: fully shuffle activities within each trace.
    σ_label: replace outcomes with permuted_labels.
    Every rejection on this log is a false positive by construction.
    """
    rng  = np.random.RandomState(trace_seed)
    null = {}
    for i, cid in enumerate(case_ids_sorted):
        ci_orig    = case_data_perturbed[cid]
        ci         = copy.copy(ci_orig)
        ci.outcome = int(permuted_labels[i])
        shuffled   = ci_orig.trace.copy()
        rng.shuffle(shuffled)
        ci.trace          = shuffled
        ci.activity_index = precompute_activity_index(shuffled, case_id=cid)
        null[cid] = ci
    return null


# ═══════════════════════════════════════════════════════════════════════════
# DRVA RUN ON PERTURBED LOG
# ═══════════════════════════════════════════════════════════════════════════

def run_drva_on_perturbed_log(
    case_data_perturbed: dict,
    candidates_all:      list,
    alpha_drva:          float,
    pi:                  int,
) -> dict:
    """
    Run DRVA on a perturbed log.
    Hierarchical simplification disabled; M_tested = M_all.
    Returns {'S_set': frozenset, 'R_obs': int}.
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
    }


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE-ANCHORED METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_reference_metrics(
    S_pert:      frozenset,
    S_orig_drva: frozenset,
) -> dict:
    """
    Reference-anchored performance metrics for DRVA degradation analysis.

        TP = |S_pert ∩ S_orig_drva|
        FP = |S_pert \\ S_orig_drva|
        FN = |S_orig_drva \\ S_pert|

    FDR_ref     = FP / R_obs
    Recall      = TP / |S_orig_drva|
    Jaccard_rq2 = |S_pert ∩ S_orig_drva| / |S_pert ∪ S_orig_drva|
                  = 1.0 at (0,0) by construction.
    """
    R_obs  = len(S_pert)
    R_full = len(S_orig_drva)

    TP = len(S_pert & S_orig_drva)
    FP = len(S_pert - S_orig_drva)
    FN = len(S_orig_drva - S_pert)

    if R_obs == 0:
        precision = float('nan')
        fdr_ref   = float('nan')
        estimable = False
    else:
        precision = TP / R_obs
        fdr_ref   = FP / R_obs
        estimable = True

    recall = TP / R_full if R_full > 0 else float('nan')

    if (not estimable) or np.isnan(precision) or np.isnan(recall):
        f1 = float('nan')
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    union_rq2   = len(S_pert | S_orig_drva)
    jaccard_rq2 = len(S_pert & S_orig_drva) / union_rq2 if union_rq2 > 0 else 1.0

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
        'Gained':        len(S_pert - S_orig_drva),
        'Lost':          len(S_orig_drva - S_pert),
        'estimable':     estimable,
        'reliable':      estimable and (R_obs >= 10),
        'FP_over_Rfull': fp_over_rfull,
        'FN_over_Rfull': fn_over_rfull,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FDR FROM NULL COUNTS
# ═══════════════════════════════════════════════════════════════════════════

def _bca_ci(data: np.ndarray, B: int = 800, seed: int = 42) -> tuple:
    """BCa 95% CI for the mean."""
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


def _compute_fdr_from_null(null_counts: np.ndarray, R_obs: int, alpha: float) -> dict:
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
        'controls_FDR':   bool(fdr <= alpha),
        'estimable':      True,
        'skipped_reason': None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NULL REPLICATE RUNNER FOR ANCHOR CELLS
# ═══════════════════════════════════════════════════════════════════════════

def _one_anchor_replicate(
    b:                   int,
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    perm_labels_b:       np.ndarray,
    alpha_drva:          float,
    base_seed:           int,
    drva_cfg_null:       dict,
) -> int:
    """One doubly-null anchor replicate → DRVA false-positive count."""
    rs_trace = base_seed + 100_000 + b
    rs_drva  = base_seed + 300_000 + b

    null_cd = _build_doubly_null_log(
        case_data_perturbed, case_ids_sorted, perm_labels_b, trace_seed=rs_trace,
    )

    with _suppress():
        holds_null = compute_holds_by_case_batch(null_cd, candidates_all)

    with _suppress():
        n_drva = run_drva_on_doubly_null_log(
            null_case_data = null_cd,
            candidates_all = candidates_all,
            alpha          = alpha_drva,
            replicate_seed = rs_drva,
            config         = drva_cfg_null,
            holds_all      = holds_null,
        )

    return n_drva


def run_null_replicates_for_level(
    case_data_perturbed: dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    labels_perturbed:    np.ndarray,
    B_null:              int,
    pi_drva_null:        int,
    alpha_drva:          float,
    base_seed:           int,
    n_jobs:              int = 1,
) -> np.ndarray:
    """
    Run B_null doubly-null replicates on a perturbed log (DRVA only).
    Returns (B_null,) int array of false-positive counts.
    """
    rng_outer       = np.random.RandomState(base_seed)
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

    counts_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_anchor_replicate)(
            b, case_data_perturbed, candidates_all, case_ids_sorted,
            perm_labels_all[b], alpha_drva, base_seed, drva_cfg_null,
        )
        for b in range(B_null)
    )

    return np.array(counts_list, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
# PER-CELL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_joint_cell(
    eps:                 float,
    rho:                 float,
    eps_idx:             int,
    rho_idx:             int,
    struct_log:          dict,
    candidates_all:      list,
    case_ids_sorted:     list,
    S_orig_drva:         frozenset,
    is_anchor_bootstrap: bool = False,
) -> dict:
    """
    Full DRVA analysis for one joint noise cell (ε, ρ).

    Anchor cells run doubly-null FDR_emp replicates.
    (0,0) cell (is_anchor_bootstrap=True): reference metrics computed against
      own output (Jaccard_rq2 = 1.0 by construction).
    Non-anchor cells: FDR_ref is the primary metric, doubly-null is skipped.
    """
    print(f"  [ε={eps:.2f}  ρ={rho:.2f}]  Starting DRVA analysis...", flush=True)
    t0 = time.time()

    base_seed_cell = BASE_SEED + eps_idx * 1000 + rho_idx

    MAX_RETRIES = 5
    RETRY_PRIME = 999983

    eps_seed_used = None
    R_obs_drva    = 0
    S_pert_drva   = frozenset()

    for retry in range(MAX_RETRIES + 1):
        eps_seed_try = base_seed_cell + 1 + retry * RETRY_PRIME

        case_data_pert = apply_joint_perturbation(struct_log, eps, eps_seed_try)
        labels_pert    = np.array([
            case_data_pert[cid].outcome for cid in case_ids_sorted
        ], dtype=np.int8)

        drva_res      = run_drva_on_perturbed_log(
            case_data_pert, candidates_all, ALPHA_DRVA, PI_DRVA_REAL,
        )
        S_pert_drva   = drva_res['S_set']
        R_obs_drva    = drva_res['R_obs']
        eps_seed_used = eps_seed_try

        if R_obs_drva > 0 or retry == MAX_RETRIES:
            if retry > 0 and R_obs_drva > 0:
                print(
                    f"  [ε={eps:.2f}  ρ={rho:.2f}]  R_obs>0 after {retry} "
                    f"retr{'y' if retry == 1 else 'ies'}.",
                    flush=True,
                )
            break
        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  R_obs=0 on draw {retry} — retrying...",
            flush=True,
        )

    # Reference metrics
    _S_ref      = S_pert_drva if is_anchor_bootstrap else S_orig_drva
    ref_metrics = compute_reference_metrics(S_pert_drva, _S_ref)

    # Doubly-null FDR_emp (anchor cells only)
    is_anchor   = is_anchor_bootstrap or (eps, rho) in ANCHOR_CELLS
    B_null_this = B_NULL_ANCHOR if (is_anchor and R_obs_drva > 0) else 0

    if B_null_this == 0:
        reason      = 'R_obs=0: FDR_emp undefined' if R_obs_drva == 0 else 'non-anchor cell'
        fdr_null    = _null_fdr_skipped(reason)
        null_counts = np.zeros(0, dtype=int)
    else:
        print(
            f"  [ε={eps:.2f}  ρ={rho:.2f}]  "
            f"Running {B_null_this} null replicates (π={PI_DRVA_NULL} each)...",
            flush=True,
        )
        null_counts = run_null_replicates_for_level(
            case_data_perturbed = case_data_pert,
            candidates_all      = candidates_all,
            case_ids_sorted     = case_ids_sorted,
            labels_perturbed    = labels_pert,
            B_null              = B_null_this,
            pi_drva_null        = PI_DRVA_NULL,
            alpha_drva          = ALPHA_DRVA,
            base_seed           = base_seed_cell + 50_000,
            n_jobs              = INNER_N_JOBS,
        )
        fdr_null = _compute_fdr_from_null(null_counts, R_obs_drva, ALPHA_DRVA)

    wall = time.time() - t0
    print(
        f"  [ε={eps:.2f}  ρ={rho:.2f}]  Done in {wall:.1f}s  "
        f"R_obs={R_obs_drva}  FDR_ref={ref_metrics['FDR_ref']}",
        flush=True,
    )

    result = {
        'eps':           eps,
        'rho':           rho,
        'eps_idx':       eps_idx,
        'rho_idx':       rho_idx,
        'R_obs':         R_obs_drva,
        'S_pert':        list(S_pert_drva),
        'ref_metrics':   ref_metrics,
        'fdr_null':      fdr_null,
        'null_counts':   null_counts.tolist(),
        'B_null':        B_null_this,
        'wall_seconds':  wall,
        'is_anchor':     is_anchor,
        'eps_seed_used': eps_seed_used,
    }

    if is_anchor_bootstrap:
        result['S_pert_raw'] = S_pert_drva

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — ORIGINAL DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_original_data() -> dict:
    """
    Load data and run DRVA on the original (unperturbed) Sepsis log.
    Returns case_data, candidates_all, case_ids_sorted, labels_orig,
    S_orig_drva.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — ORIGINAL DATA RUN  (DRVA on Sepsis)")
    print(f"  π={PI_DRVA_REAL}  α_DRVA={ALPHA_DRVA}")
    print(f"  DRVA_CONFIG: pi={DRVA_CONFIG['pi']}  alpha={DRVA_CONFIG['alpha']}"
          f"  mmin={DRVA_CONFIG['mmin']}  mdiff_min={DRVA_CONFIG['mdiff_min']}"
          f"  hierarchical_pruning={DRVA_CONFIG['hierarchical_pruning']}")
    print("=" * 100)

    case_data = load_and_preprocess_data(CSV_PATH)

    candidates_pos, candidates_neg = generate_candidate_patterns(case_data)
    pos_set        = set(candidates_pos)
    candidates_all = list(candidates_pos) + [
        p for p in candidates_neg if p not in pos_set
    ]

    case_ids_sorted = sorted(case_data.keys())
    labels_orig     = np.array([case_data[cid].outcome for cid in case_ids_sorted])
    n  = len(labels_orig)
    n1 = int(labels_orig.sum())
    print(f"\n  n={n:,}  (n1={n1:,}, n0={n-n1:,})  M_all={len(candidates_all):,}")

    # DRVA on original data → S_orig_drva (ground truth for FDR_ref)
    t0 = time.time()
    drva_cfg = DRVA_CONFIG.copy()
    drva_cfg.update({
        'pi':                   PI_DRVA_REAL,
        'alpha':                ALPHA_DRVA,
        'hierarchical_pruning': False,
        'mmin':                 0.0,
        'mdiff_min':            0.0,
    })
    with _suppress():
        drva_orig = run_drva(
            config         = drva_cfg,
            case_data      = case_data,
            candidates_all = candidates_all,
        )

    S_orig_drva = frozenset(
        (r['constraint_type'], r['activity_a'], r['activity_b'])
        for r in drva_orig['results']
        if r['is_significant_cecconi']
    )
    wall = time.time() - t0
    print(f"  DRVA complete: {wall:.1f}s  R_obs={len(S_orig_drva):,}  (= S_orig_drva)")

    return {
        'case_data':       case_data,
        'candidates_all':  candidates_all,
        'case_ids_sorted': case_ids_sorted,
        'labels_orig':     labels_orig,
        'S_orig_drva':     S_orig_drva,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — RQ2 JOINT PERTURBATION GRID
# ═══════════════════════════════════════════════════════════════════════════

def _joint_cell_worker(
    eps, rho, eps_idx, rho_idx,
    struct_log,
    candidates_all, case_ids_sorted,
    S_orig_drva,
):
    """Top-level loky-safe worker for one (ε, ρ) joint grid cell."""
    return analyze_joint_cell(
        eps                 = eps,
        rho                 = rho,
        eps_idx             = eps_idx,
        rho_idx             = rho_idx,
        struct_log          = struct_log,
        candidates_all      = candidates_all,
        case_ids_sorted     = case_ids_sorted,
        S_orig_drva         = S_orig_drva,
        is_anchor_bootstrap = False,
    )


def run_joint_grid(
    case_data_orig:  dict,
    candidates_all:  list,
    case_ids_sorted: list,
    S_orig_drva:     frozenset,
    n_jobs:          int = N_JOBS,
) -> list:
    """
    Run the full 7×7 joint noise grid in parallel (DRVA only).

    Architecture:
    Step 0 — Precompute one structural log per ρ row (7 total, not 49).
             DRVA encodes traces internally; no holds cache needed.
    Step 1 — Run (0,0) anchor sequentially to confirm S_orig_drva.
    Step 2 — Run remaining 48 cells in parallel.
    """
    print("\n" + "=" * 100)
    print("SECTION 2 — JOINT PERTURBATION GRID  (DRVA, B.3: Label × Structural)")
    print(f"  ε levels: {JOINT_LABEL_LEVELS}")
    print(f"  ρ levels: {JOINT_STRUCT_LEVELS}")
    print(f"  Grid size: {len(JOINT_GRID)} cells  |  Anchors: {sorted(ANCHOR_CELLS)}")
    print(f"  B_null (anchor)={B_NULL_ANCHOR}  (non-anchor=0, FDR_ref is primary)")
    print(f"  PI_DRVA_REAL={PI_DRVA_REAL}  PI_DRVA_NULL={PI_DRVA_NULL}  α_DRVA={ALPHA_DRVA}")
    print(f"  n_jobs={n_jobs}")
    print("=" * 100)

    t0 = time.time()

    # ── Step 0: Precompute one structural log per ρ row ──────────────────
    # DRVA re-encodes traces internally; no holds cache needed here.
    print("\n  [Step 0] Precomputing structural logs per ρ row...", flush=True)
    struct_logs = {}
    for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS):
        if rho == 0.0:
            struct_logs[rho] = case_data_orig
            print(f"    ρ={rho:.2f}  → reusing original log", flush=True)
        else:
            rho_struct_seed  = BASE_SEED + rho_idx * 13
            t_rho            = time.time()
            struct_logs[rho] = apply_structural_noise(case_data_orig, rho, rho_struct_seed)
            print(
                f"    ρ={rho:.2f}  → structural log in {time.time()-t_rho:.1f}s",
                flush=True,
            )

    # ── Step 1: Run (0,0) anchor to confirm S_orig_drva ──────────────────
    print("\n  [Step 1] Running (ε=0, ρ=0) anchor...", flush=True)
    anchor_result = analyze_joint_cell(
        eps=0.0, rho=0.0, eps_idx=0, rho_idx=0,
        struct_log          = struct_logs[0.0],
        candidates_all      = candidates_all,
        case_ids_sorted     = case_ids_sorted,
        S_orig_drva         = S_orig_drva,
        is_anchor_bootstrap = True,
    )
    all_results = [anchor_result]
    print(
        f"  [Step 1] (0,0) anchor done. "
        f"R_obs={anchor_result['R_obs']}  "
        f"Jaccard_rq2={anchor_result['ref_metrics']['Jaccard_rq2']:.4f}",
        flush=True,
    )

    # ── Step 2: Remaining 48 cells in parallel ────────────────────────────
    remaining = [
        (eps, rho, eps_idx, rho_idx)
        for eps_idx, eps in enumerate(JOINT_LABEL_LEVELS)
        for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS)
        if not (eps == 0.0 and rho == 0.0)
    ]

    print(f"\n  [Step 2] Running {len(remaining)} remaining cells (n_jobs={n_jobs})...",
          flush=True)

    other_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_joint_cell_worker)(
            eps, rho, eps_idx, rho_idx,
            struct_logs[rho],
            candidates_all, case_ids_sorted,
            S_orig_drva,
        )
        for eps, rho, eps_idx, rho_idx in remaining
    )

    all_results.extend(other_results)
    all_results.sort(key=lambda r: (r['eps'], r['rho']))

    print(
        f"\n  Grid complete: {len(all_results)} cells  |  wall={time.time()-t0:.1f}s",
        flush=True,
    )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def save_results(all_results: list, S_orig_drva: frozenset) -> None:
    """Save CSV long-format, pivot heatmaps, null counts JSON, and full JSON."""
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    # ── Long-format CSV ────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        rm = r['ref_metrics']
        fn = r['fdr_null']
        rows.append({
            'eps':           r['eps'],
            'rho':           r['rho'],
            'R_obs':         r['R_obs'],
            'FDR_ref':       rm['FDR_ref'],
            'Precision':     rm['Precision'],
            'Recall':        rm['Recall'],
            'F1':            rm['F1'],
            'TP':            rm['TP'],
            'FP':            rm['FP'],
            'FN':            rm['FN'],
            'Jaccard_rq2':   rm['Jaccard_rq2'],
            'FP_over_Rfull': rm['FP_over_Rfull'],
            'FN_over_Rfull': rm['FN_over_Rfull'],
            'FDR_emp':       fn['FDR_emp'],
            'FDR_CI_lower':  fn['FDR_CI_lower'],
            'FDR_CI_upper':  fn['FDR_CI_upper'],
            'E_V_b':         fn['E_V_b'],
            'FWER_emp':      fn['FWER_emp'],
            'B_null':        r['B_null'],
            'is_anchor':     r['is_anchor'],
            'wall_seconds':  r['wall_seconds'],
        })

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(RQ2_OUTPUT_DIR, "rq2_drva_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Pivot heatmaps ─────────────────────────────────────────────────────
    pivot_metrics = ['FDR_ref', 'Recall', 'F1', 'Jaccard_rq2', 'R_obs', 'FDR_emp']
    for metric in pivot_metrics:
        try:
            piv      = df.pivot(index='eps', columns='rho', values=metric)
            piv_path = os.path.join(RQ2_OUTPUT_DIR, f"rq2_drva_{metric.lower()}_pivot.csv")
            piv.to_csv(piv_path)
            print(f"  Saved: {piv_path}")
        except Exception as e:
            print(f"  WARNING: pivot for {metric} failed: {e}")

    # ── Null counts JSON ───────────────────────────────────────────────────
    null_json = {}
    for r in all_results:
        if r['B_null'] > 0:
            key            = f"eps={r['eps']:.2f}_rho={r['rho']:.2f}"
            null_json[key] = r['null_counts']
    null_path = os.path.join(RQ2_OUTPUT_DIR, "rq2_drva_null_counts.json")
    with open(null_path, 'w', encoding='utf-8') as f:
        json.dump(null_json, f, indent=2)
    print(f"  Saved: {null_path}")

    # ── Full results JSON ──────────────────────────────────────────────────
    full_json = {
        'rq2_version': '1.0',
        'method':      'DRVA',
        'log_name':    'Sepsis',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'method':        'DRVA — Cecconi et al. BPM Forum 2021',
            'grid':          'B.3: Joint Label × Structural noise (7×7 = 49 cells)',
            'reference':     'S_orig_drva = DRVA at (ε=0, ρ=0)',
            'null_protocol': 'Double-null: σ_label ∘ σ_trace (anchor cells only)',
            'DRVA_gate':     f'Per-rule p_Cecconi ≤ {ALPHA_DRVA} (no FDR correction)',
        },

        'config': {
            'ALPHA_DRVA':              ALPHA_DRVA,
            'PI_DRVA_REAL':            PI_DRVA_REAL,
            'PI_DRVA_NULL':            PI_DRVA_NULL,
            'B_NULL_ANCHOR':           B_NULL_ANCHOR,
            'DRVA_CONFIG_pi':          DRVA_CONFIG['pi'],
            'DRVA_CONFIG_alpha':       DRVA_CONFIG['alpha'],
            'DRVA_CONFIG_mmin':        DRVA_CONFIG['mmin'],
            'DRVA_CONFIG_mdiff_min':   DRVA_CONFIG['mdiff_min'],
            'DRVA_CONFIG_hier_pruning': DRVA_CONFIG['hierarchical_pruning'],
            'JOINT_LABEL_LEVELS':      JOINT_LABEL_LEVELS,
            'JOINT_STRUCT_LEVELS':     JOINT_STRUCT_LEVELS,
            'BASE_SEED':               BASE_SEED,
            'N_JOBS':                  N_JOBS,
            'R_orig_drva':             len(S_orig_drva),
        },

        'cells': [
            {
                'eps':          r['eps'],
                'rho':          r['rho'],
                'R_obs':        r['R_obs'],
                'ref_metrics':  r['ref_metrics'],
                'fdr_null':     r['fdr_null'],
                'B_null':       r['B_null'],
                'is_anchor':    r['is_anchor'],
                'wall_seconds': r['wall_seconds'],
            }
            for r in all_results
        ],
    }
    json_path = os.path.join(RQ2_OUTPUT_DIR, "rq2_drva_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ2 — SPECIFICATION QUALITY DEGRADATION: DRVA BASELINE — SEPSIS")
    print("Method: DRVA (Cecconi et al. BPM Forum 2021)  |  Joint noise B.3")
    print("=" * 100)
    print(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  α_DRVA={ALPHA_DRVA}  PI_DRVA_REAL={PI_DRVA_REAL}  PI_DRVA_NULL={PI_DRVA_NULL}")
    print(f"  B_null(anchor)={B_NULL_ANCHOR}  Grid: {len(JOINT_GRID)} cells")
    print(f"  DRVA_CONFIG: pi={DRVA_CONFIG['pi']}  alpha={DRVA_CONFIG['alpha']}"
          f"  hier_pruning={DRVA_CONFIG['hierarchical_pruning']}")
    print(f"  BASE_SEED={BASE_SEED}  Output: {RQ2_OUTPUT_DIR}/")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    # ── Section 1: load data + DRVA on original log ───────────────────────
    orig = run_original_data()

    # ── Section 2: 7×7 joint perturbation grid ────────────────────────────
    all_results = run_joint_grid(
        case_data_orig  = orig['case_data'],
        candidates_all  = orig['candidates_all'],
        case_ids_sorted = orig['case_ids_sorted'],
        S_orig_drva     = orig['S_orig_drva'],
        n_jobs          = N_JOBS,
    )

    # ── Section 3: save results ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SECTION 3 — SAVING RESULTS")
    print("=" * 100)
    save_results(all_results, orig['S_orig_drva'])

    # ── Final summary ─────────────────────────────────────────────────────
    total_wall   = time.time() - t_total
    anchor_cells = [r for r in all_results if r['is_anchor']]

    print(f"\n{'='*100}")
    print("RQ2 — DRVA SEPSIS COMPLETE")
    print(f"{'='*100}")
    print(f"  Total wall time:  {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  S_orig_drva size: {len(orig['S_orig_drva']):,}")
    print(f"  Output: {RQ2_OUTPUT_DIR}/")
    print(f"\n  ANCHOR CELL RESULTS  (FDR_ref primary, FDR_emp secondary):")
    print(f"  {'(ε, ρ)':15s} {'R_obs':>6s} {'FDR_ref':>8s} {'Recall':>8s} "
          f"{'FDR_emp':>8s}")
    print(f"  {'─'*50}")
    for r in sorted(anchor_cells, key=lambda x: (x['eps'], x['rho'])):
        rm    = r['ref_metrics']
        fn    = r['fdr_null']
        fdr_r = f"{rm['FDR_ref']:.4f}" if not np.isnan(rm['FDR_ref']) else '   N/A'
        rec   = f"{rm['Recall']:.4f}"  if not np.isnan(rm['Recall'])  else '   N/A'
        fdr_e = f"{fn['FDR_emp']:.4f}" if not np.isnan(fn['FDR_emp']) else '   N/A'
        print(f"  ({r['eps']:.2f}, {r['rho']:.2f})       "
              f"{r['R_obs']:>6d} {fdr_r:>8s} {rec:>8s} {fdr_e:>8s}")
    print(f"{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ2 Degradation — DRVA Baseline, Sepsis (joint noise B.3)"
    )
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Parallel workers (-1 = all cores, default: {N_JOBS})',
    )
    parser.add_argument(
        '--alpha-drva', type=float, default=ALPHA_DRVA,
        help=f'DRVA per-rule significance level (default: {ALPHA_DRVA})',
    )
    parser.add_argument(
        '--b-null-anchor', type=int, default=B_NULL_ANCHOR,
        help=f'Null replicates at anchor cells (default: {B_NULL_ANCHOR})',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Smoke test: 2×2 grid, B_null=2, π=10',
    )
    args = parser.parse_args()

    if args.dry_run:
        JOINT_LABEL_LEVELS  = [0.00, 0.50]
        JOINT_STRUCT_LEVELS = [0.00, 1.00]
        JOINT_GRID          = [(e, r) for e in JOINT_LABEL_LEVELS for r in JOINT_STRUCT_LEVELS]
        ANCHOR_CELLS        = frozenset(JOINT_GRID)
        B_NULL_ANCHOR       = 2
        PI_DRVA_REAL        = 10
        PI_DRVA_NULL        = 10
        print("*** DRY-RUN MODE: 2×2 grid, B_null=2, π=10 ***")
    else:
        B_NULL_ANCHOR = args.b_null_anchor

    N_JOBS     = args.n_jobs
    ALPHA_DRVA = args.alpha_drva

    main()
