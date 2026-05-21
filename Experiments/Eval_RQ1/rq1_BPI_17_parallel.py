#!/usr/bin/env python3
"""
rq1_BPI_17_parallel.py  —  RQ1 FDR Control Validity: BPI Challenge 2017
================================================================
Block A: Doubly-Null Empirical FDR Comparison Across Three Methods

METHODS COMPARED (shared candidate pool M_all)
-----------------------------------------------
    1.  P1  (Ours)       Hou-Storey conjunction + Adaptive Storey FDR gate.
                         T_Hou = -2[W_STRUCT·ln p_s + W_DISC·ln p_d],
                         W_DISC=0.60, W_STRUCT=0.40 (precision-proportional,
                         B_label=1500, B2_test=1000).
                         Oracle null: chi2.sf(T_Hou/c, f), c=0.52, f≈3.846,
                         rho_sd=0 (exact under double-null independence).
                         Single gate: q_Hou ≤ α (matches p1 is_significant_final).
                         Aligned with p1_BPI_17_hou.py v9.0-HOU-DOUBLY-NULL.

    2.  DRVA             Cecconi et al. (BPM Forum 2021).
                         Permutation test on ΔConfidence (shuffleLog = label perm
                         on pre-cached trace evaluations).
                         No FDR correction — per-rule raw α threshold.
                         Imported from drva_BPI_17_parallel.py.

    3.  DeclareMiner     Differential confidence threshold baseline.
                         Δconf(r) ≥ τ* (calibrated to match R_obs^P1).
                         No permutation test, no FDR correction.
                         Imported from declareminer_BPI_17_parallel.py.

EXPERIMENTAL DESIGN PRINCIPLE
-------------------------------
All three methods operate on the SAME fixed candidate pool M_all derived
from Phase 0 DECLARE specifications. This removes candidate-scope confounding
so that FDR differences reflect only the testing procedure:

    ΔFDR = (different test) NOT (different candidate scope)

DOUBLY-NULL PROTOCOL  (Pellegrina & Vandin 2018, adapted)
----------------------------------------------------------
Each held-out replicate b applies TWO independent operations:

    Null_b = σ_label ∘ σ_trace

    1. σ_trace:  Randomly permute activity sequence within each trace.
                 Preserves trace length and activity multiset per case.
                 Destroys all temporal ordering.
                 → p_struct^(b) ~ U(0,1) by Fisher randomisation.

    2. σ_label:  Permute class labels across cases (marginals preserved).
                 Destroys any class–trace association.
                 → p_disc^(b) ~ U(0,1) by Fisher randomisation.

    Every rejection on L^(b) is a false positive by construction.

EMPIRICAL FDR ESTIMATOR  (Pellegrina & Vandin 2018)
----------------------------------------------------
    FDR_emp(method) = E[V_b^(method)] / max(R_obs^(method), 1)

where V_b = rejections in null replicate b (all false positives).

MECHANISTIC PREDICTION
-----------------------
    P1:           FDR_emp ≈ α = 0.05. Storey π̂₀ correction + scope filter
                  ensure valid FDR control at the nominal level.

    DRVA:         FDR_emp >> α_DRVA = 0.01. No multiple-testing correction:
                  under the null, DRVA rejects ≈ α_DRVA × m patterns per
                  replicate. If R_obs^DRVA is not proportionally larger,
                  FDR_emp = E[V_b] / R_obs >> α_DRVA.

    DeclareMiner: FDR_emp depends on the null distribution of Δconf; small
                  random confidence fluctuations exceed τ* by chance,
                  producing uncontrolled FDR with no principled bound.

NULL REPLICATE BUDGETS
-----------------------
    P1  (null):   B1_VALID label perms, B2_VALID structural perms per replicate.
    DRVA (null):  PI_DRVA_VALID internal shuffleLog iterations per replicate.
    DM  (null):   Deterministic threshold application (zero extra iterations).

    Conservative bias direction for P1:
        B1_VALID < B1_FULL → coarser p_disc resolution → slightly smaller
        T_Hou → fewer null rejections → FDR_emp slightly underestimated.
        This is safe for the FDR control claim (conservative direction).

OUTPUT FILES
-------------
    rq1_fdr_metrics.csv            One row per method (Table data for paper).
    rq1_null_counts.csv            B_null rows × 3 methods (raw V_b counts).
    rq1_m_prime_distribution.csv   P1 scope-filter sizes per null replicate.
    rq1_results.json               Full results for paper generation.
    rq1_pattern_arrays.npz         Per-pattern arrays from original P1 run.

Version : 1.0  (Hou-Storey + DRVA + DeclareMiner; single gate; aligned with
               p1_BPI_17_hou.py v9.0-HOU-DOUBLY-NULL)
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
Benjamini & Hochberg (1995). JRSS-B 57(1):289-300.
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

# ── Phase 1 (Hou-Storey conjunction framework) ────────────────────────────
# File: p1_BPI_17_hou.py  (v9.0-HOU-DOUBLY-NULL)
from P1_SDSM.p1_BPI_17_hou import (
    # Data loading & preprocessing
    load_and_preprocess_data,
    generate_candidate_patterns,
    split_by_class,
    compute_prevalence_from_holds,
    compute_holds_by_case_batch,
    precompute_activity_index,
    # Permutation tests
    run_label_permutation_test,
    run_structural_permutation_test,
    # Hou (2005) statistical machinery
    hou_combination_statistic,
    hou_satterthwaite_params,
    W_DISC,
    W_STRUCT,
    # FDR machinery
    adaptive_storey_pi0,
    storey_qvalue,
    benjamini_hochberg,
    # Pipeline
    execute_pipeline,
    generate_outputs,
    # Data structures
    CaseInfo,
    PatternTestResult,
    # Global config / paths
    CONFIG as P1_CONFIG,
    INPUT_FILE as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
)

# ── DRVA baseline ─────────────────────────────────────────────────────────
# File: drva_BPI_17_parallel.py  (v2.1)
from BaselinesRQ1.DRVA_BPI_17 import (
    run_drva,
    run_drva_on_doubly_null_log,
    DRVA_CONFIG,
)

# ── DeclareMiner baseline ─────────────────────────────────────────────────
# File: declareminer_BPI_17_parallel.py  (v2.0)
from BaselinesRQ1.DeclareMiner_BPI_17 import (
    run_declareminer,
    run_declareminer_on_doubly_null_log,
    DM_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH       = P1_INPUT_FILE
PHASE0_JSON    = P1_SPEC_FILE
RQ1_OUTPUT_DIR = "RQ1_BPI_17"

# ── Permutation budgets ────────────────────────────────────────────────────
# Original P1 run: matches p1_BPI_17_hou CONFIG exactly.
B1_FULL      = 1_500   # label permutations          (P1_CONFIG['B_label'])
B2_FULL      = 2_000   # structural permutations      (P1_CONFIG['B_trace'])
B_NULL_FULL  = 200     # P1 double-null calibration   (P1_CONFIG['B_null'])
B1_NULL_FULL = 75      # label perms per P1 calib rep (P1_CONFIG['B1_null'])
B2_NULL_FULL = 75      # struct perms per P1 calib rep(P1_CONFIG['B2_null'])

# Null replicate budgets for RQ1 held-out permutations (reduced for speed).
# Conservative bias: smaller budget → smaller T_Hou → fewer null rejections
# → FDR_emp slightly underestimated → safe for the FDR control claim.
B1_VALID      = 500    # label perm budget per held-out replicate (P1)
B2_VALID      = 200    # structural perm budget per held-out replicate (P1)
PI_DRVA_VALID = 200    # DRVA shuffleLog iterations per held-out replicate

# Held-out null replicates for FDR estimation
B_NULL = 200

# FDR target level
ALPHA = 0.05

# DRVA's own per-rule significance level (no FDR correction)
ALPHA_DRVA = 0.01    # Cecconi et al. 2021, §3.5 default

# Seed architecture (three non-overlapping layers; safe for B_NULL < 100,000)
#   Held-out σ_label:  BASE_SEED + b
#   P1 internal:       BASE_SEED + 100_000 + b
#   σ_trace shuffle:   BASE_SEED + 100_000 + b + 200_000
#   DRVA internal:     BASE_SEED + 100_000 + b + 50_000
BASE_SEED = 20260521

# Parallelism
N_JOBS = -1

# Hou oracle parameters under the double-null (rho_sd = 0, independence)
# c = W_STRUCT² + W_DISC² = 0.40² + 0.60² = 0.52
# f = 2/c ≈ 3.846
_C_NULL, _F_NULL = hou_satterthwaite_params(W_STRUCT, W_DISC, rho_sd=0.0)

# Method name constants
METHOD_P1   = "P1_HouStorey"
METHOD_DRVA = "DRVA"
METHOD_DM   = "DeclareMiner"

ALL_METHODS = [METHOD_P1, METHOD_DRVA, METHOD_DM]

assert B_NULL < 100_000, (
    f"B_NULL={B_NULL} ≥ 100,000 would cause seed-layer overlap. "
    "Three seed layers use offsets 0, 100_000, and 200_000 from BASE_SEED+b."
)


# ═══════════════════════════════════════════════════════════════════════════
# INLINE HELPERS  (self-contained; no eval_utils dependency)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_heldout_permutation_batch(
    labels: np.ndarray,
    B: int,
    base_seed: int,
) -> np.ndarray:
    """
    Generate B held-out label permutations, preserving marginal class counts.

    Replicate b uses seed (base_seed + b).  Seeds are offset from the Phase 1
    internal permutation seeds (which start at base_seed + 100_000), so the
    held-out permutations are independent of the Phase 1 null distributions.

    Returns:
        (B, n) int8 array of permuted label vectors.
    """
    n   = len(labels)
    out = np.empty((B, n), dtype=np.int8)
    for b in range(B):
        rng   = np.random.RandomState(base_seed + b)
        out[b] = rng.permutation(labels).astype(np.int8)
    return out


def _bootstrap_bca_ci(
    data: np.ndarray,
    stat_fn,
    B_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    BCa (bias-corrected accelerated) bootstrap confidence interval.

    Args:
        data:    1-D array of observed values (e.g. V_b / R_obs per replicate).
        stat_fn: Function mapping array → scalar (e.g. np.mean).
        B_boot:  Bootstrap resamples.
        ci:      Confidence level.
        seed:    RNG seed.

    Returns:
        (lower, upper) BCa CI bounds.
    """
    rng   = np.random.RandomState(seed)
    n     = len(data)
    theta = stat_fn(data)

    boot   = np.array([stat_fn(rng.choice(data, n, replace=True)) for _ in range(B_boot)])
    z0     = stats.norm.ppf(float(np.mean(boot < theta)) + 1e-10)

    # Vectorized O(n) jackknife for the mean statistic (avoids O(n²) loop).
    total  = np.sum(data)
    jack   = (total - data) / (n - 1)
    jack_m = jack.mean()
    num    = float(np.sum((jack_m - jack) ** 3))
    den    = float(6.0 * (np.sum((jack_m - jack) ** 2) ** 1.5))
    a_hat  = num / den if abs(den) > 1e-15 else 0.0

    alpha_tail = (1.0 - ci) / 2.0

    def _adj(z_):
        return stats.norm.cdf(z0 + (z0 + z_) / (1.0 - a_hat * (z0 + z_)))

    lo_pct = float(np.clip(_adj(stats.norm.ppf(alpha_tail)),      0.001, 0.999))
    hi_pct = float(np.clip(_adj(stats.norm.ppf(1.0 - alpha_tail)), 0.001, 0.999))

    lower = float(np.percentile(boot, lo_pct * 100))
    upper = float(np.percentile(boot, hi_pct * 100))
    return lower, upper


def _compute_fdr_metrics(
    null_counts: np.ndarray,
    R_obs: int,
    m_total: int,
    alpha_nominal: float,
) -> dict:
    """
    Compute FDR_emp, PCER_emp, FWER_emp with BCa 95% CI.

    FDR_emp  = E[V_b] / max(R_obs, 1)        Pellegrina & Vandin (2018)
    PCER_emp = E[V_b] / m_total               per-comparison error rate
    FWER_emp = Pr[V_b > 0]                    family-wise error rate

    Args:
        null_counts:   (B,) array of false-positive counts per replicate.
        R_obs:         Rejection count on the real data.
        m_total:       Total patterns in M_all (FDR denominator anchor).
        alpha_nominal: Nominal level for pass/fail verdict.

    Returns:
        dict with FDR_emp, PCER_emp, FWER_emp, BCa CI, controls_FDR.
    """
    B     = len(null_counts)
    ev    = float(np.mean(null_counts))
    denom = max(R_obs, 1)

    fdr_emp  = ev / denom
    pcer_emp = ev / max(m_total, 1)
    fwer_emp = float(np.mean(null_counts > 0))

    fdr_arr = null_counts.astype(float) / denom
    try:
        ci_lo, ci_hi = _bootstrap_bca_ci(fdr_arr, np.mean, B_boot=1000, seed=42)
    except Exception:
        ci_lo = float(np.percentile(fdr_arr, 2.5))
        ci_hi = float(np.percentile(fdr_arr, 97.5))

    return {
        'R_obs':        R_obs,
        'E_V_b':        ev,
        'FDR_emp':      fdr_emp,
        'PCER_emp':     pcer_emp,
        'FWER_emp':     fwer_emp,
        'FDR_CI_lower': ci_lo,
        'FDR_CI_upper': ci_hi,
        'B_null':       B,
        'm_total':      m_total,
        'alpha':        alpha_nominal,
        'controls_FDR': bool(fdr_emp <= alpha_nominal),
    }


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT SUPPRESSION HELPER
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout/stderr to /dev/null during null replicates."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — P1 ORIGINAL-DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_p1_original(n_workers: int = 1) -> dict:
    """
    Run Phase 1 (execute_pipeline) at full computational budget.

    The full budget matches p1_BPI_17_hou CONFIG:
        B_label=1500, B_trace=2000, B_null=200, B1_null=75, B2_null=75.

    Calls generate_outputs to write the Phase 1 discovery files (tables,
    visualisations) once, independently of RQ1.

    Returns a dict containing all fields needed by subsequent sections:
    case_data, candidates_all, pattern_results, null_delta_matrix,
    holds_all, delta_obs, tf_null_matrix, case_ids_sorted, labels,
    timing, wall_seconds.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — P1 ORIGINAL-DATA RUN (Hou-Storey conjunction)")
    print(f"  B1_label={B1_FULL}, B2_trace={B2_FULL}")
    print(f"  B_null={B_NULL_FULL}, B1_null={B1_NULL_FULL}, B2_null={B2_NULL_FULL}")
    print(f"  alpha={ALPHA}, n_workers={n_workers}")
    print("=" * 100)

    t0  = time.time()
    cfg = P1_CONFIG.copy()
    cfg['B_label']      = B1_FULL
    cfg['B_trace']      = B2_FULL
    cfg['B_null']       = B_NULL_FULL
    cfg['B1_null']      = B1_NULL_FULL
    cfg['B2_null']      = B2_NULL_FULL
    cfg['fdr_alpha']    = ALPHA
    cfg['random_state'] = 42
    cfg['n_workers']    = n_workers
    cfg['n_jobs']       = n_workers

    output = execute_pipeline(input_file=CSV_PATH, config=cfg)

    # Write Phase 1 output files (independent of RQ1)
    generate_outputs(
        output['pattern_results'],
        output['case_data'],
        output['timing'],
    )

    case_data       = output['case_data']
    pattern_results = output['pattern_results']
    candidates_all  = output['candidates_all']
    case_ids_sorted = sorted(case_data.keys())
    labels          = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    wall = time.time() - t0
    n    = len(labels)
    n1   = int(labels.sum())
    m    = len(candidates_all)

    sig_final = sum(1 for r in pattern_results if r.is_significant_final)
    print(f"\n  P1 complete: {wall:.1f}s  n={n:,} (n1={n1:,}, n0={n-n1:,})  m={m:,}")
    print(f"  Hou-Storey k* = {sig_final:,}  (q_Hou ≤ {ALPHA})")

    return {
        'case_data':         case_data,
        'candidates_all':    candidates_all,
        'candidates_pos':    output['candidates_pos'],
        'candidates_neg':    output['candidates_neg'],
        'pattern_results':   pattern_results,
        'null_delta_matrix': output['null_delta_matrix'],
        'holds_all':         output['holds_all'],
        'delta_obs':         output['delta_obs'],
        'tf_null_matrix':    output['tf_null_matrix'],
        'case_ids_sorted':   case_ids_sorted,
        'labels':            labels,
        'timing':            output['timing'],
        'wall_seconds':      wall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DRVA ORIGINAL-DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_drva_original(
    case_data: dict,
    candidates_all: list,
) -> dict:
    """
    Run DRVA on the real BPI_17 data with the shared M_all candidate pool.

    Hierarchical simplification is DISABLED (hierarchical_pruning=False)
    and both pre-processing thresholds are set to zero (mmin=0, mdiff_min=0)
    so that M_tested = M_all exactly. This keeps the FDR denominator
    consistent with P1 and DeclareMiner.

    Returns the full drva_out dict from run_drva plus wall_seconds.
    """
    print("\n" + "=" * 100)
    print("SECTION 2 — DRVA ORIGINAL-DATA RUN")
    print(f"  π={DRVA_CONFIG['pi']:,}  α_DRVA={ALPHA_DRVA}"
          f"  hierarchical_pruning=False  mmin=0  mdiff_min=0")
    print("=" * 100)

    t0  = time.time()
    cfg = DRVA_CONFIG.copy()
    cfg['alpha']                = ALPHA_DRVA
    cfg['hierarchical_pruning'] = False
    cfg['mmin']                 = 0.0
    cfg['mdiff_min']            = 0.0

    drva_out = run_drva(
        config         = cfg,
        case_data      = case_data,
        candidates_all = candidates_all,
    )

    wall = time.time() - t0
    print(f"\n  DRVA complete: {wall:.1f}s")
    print(f"  M_all={drva_out['m_all']:,}  M_tested={drva_out['m_tested']:,}  "
          f"R_obs(Cecconi p≤{ALPHA_DRVA})={drva_out['n_rejected_cecconi']:,}")

    drva_out['wall_seconds'] = wall
    return drva_out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DECLAREMINER ORIGINAL-DATA RUN (CALIBRATED)
# ═══════════════════════════════════════════════════════════════════════════

def run_declareminer_original(
    case_data: dict,
    candidates_all: list,
    R_obs_p1: int,
) -> dict:
    """
    Run DeclareMiner with τ_Δconf calibrated so R_obs^DM ≥ R_obs^P1.

    Calibration strategy (conservative): τ* = smallest τ on the grid such
    that R_obs(τ) ≥ R_obs_target.  This guarantees DeclareMiner never
    under-discovers relative to P1, making the FDR comparison informative:
    both methods find at least as many patterns, but only P1 controls FDR.

    Returns the full dm_out dict from run_declareminer plus wall_seconds.
    """
    print("\n" + "=" * 100)
    print("SECTION 3 — DeclareMiner ORIGINAL-DATA RUN (calibrated)")
    print(f"  R_obs_target = R_obs^P1 = {R_obs_p1:,}  (conservative: τ* minimised)")
    print(f"  Primary measure: Δconf (matches DRVA Ediff)")
    print(f"  tau_min = {DM_CONFIG['tau_min']:.4f}")
    print("=" * 100)

    t0  = time.time()
    cfg = DM_CONFIG.copy()
    cfg['R_obs_target'] = R_obs_p1
    cfg['random_state'] = 42

    dm_out = run_declareminer(
        config         = cfg,
        case_data      = case_data,
        candidates_all = candidates_all,
        R_obs_target   = R_obs_p1,
    )

    wall = time.time() - t0
    print(f"\n  DeclareMiner complete: {wall:.1f}s")
    print(f"  τ* = {dm_out['tau_star']:.4f}  "
          f"R_obs^DM = {dm_out['n_rejected']:,}  "
          f"(target was ≥ {R_obs_p1:,})")

    dm_out['wall_seconds'] = wall
    return dm_out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — COLLECT R_obs FOR ALL METHODS
# ═══════════════════════════════════════════════════════════════════════════

def collect_R_obs(
    pattern_results: list,
    drva_out: dict,
    dm_out: dict,
) -> dict:
    """
    Collect the real-data rejection count for each method.

    R_obs^P1   = patterns with is_significant_final = True (single gate q_Hou ≤ α).
    R_obs^DRVA = rules rejected by Cecconi p ≤ α_DRVA (raw per-rule, no correction).
    R_obs^DM   = rules with |Δconf| ≥ τ* and conf_max ≥ τ_min (no stat test).
    """
    R_p1   = int(sum(1 for r in pattern_results if r.is_significant_final))
    R_drva = int(drva_out['n_rejected_cecconi'])
    R_dm   = int(dm_out['n_rejected'])

    print("\n" + "=" * 100)
    print("SECTION 4 — R_obs (REAL-DATA REJECTIONS PER METHOD)")
    print("=" * 100)
    print(f"\n  {METHOD_P1:20s}: {R_p1:,}  (Hou-Storey q_Hou ≤ {ALPHA})")
    print(f"  {METHOD_DRVA:20s}: {R_drva:,}  "
          f"(DRVA p_Cecconi ≤ {ALPHA_DRVA}, no FDR correction)")
    print(f"  {METHOD_DM:20s}: {R_dm:,}  "
          f"(|Δconf| ≥ τ*={dm_out['tau_star']:.4f}, no stat test)")

    return {METHOD_P1: R_p1, METHOD_DRVA: R_drva, METHOD_DM: R_dm}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DIAGNOSTICS (π̂₀ AND SIGNAL DENSITY)
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnostics(pattern_results: list) -> dict:
    """
    Compute π̂₀ estimates on the discriminative and structural axes.

    π̂₀_disc quantifies how many patterns are null on the discriminative axis
    and determines Storey's power gain over BH:  1/π̂₀_disc ≈ power ratio.

    Also reports the structural scope-filter size m'', computed from
    the sample-split screen p-values (independent of the test p-values
    that enter T_Hou).
    """
    print("\n" + "=" * 100)
    print("SECTION 5 — DIAGNOSTICS (π̂₀ AND STRUCTURAL SIGNAL DENSITY)")
    print("=" * 100)

    m    = len(pattern_results)
    pdis = np.array([r.p_discriminative            for r in pattern_results])
    ps0  = np.array([r.p_structural_class0         for r in pattern_results])
    ps1  = np.array([r.p_structural_class1         for r in pattern_results])
    psc0 = np.array([r.p_structural_screen_class0  for r in pattern_results])
    psc1 = np.array([r.p_structural_screen_class1  for r in pattern_results])

    pi0_disc,    _ = adaptive_storey_pi0(pdis, q=ALPHA)
    pi0_struct0, _ = adaptive_storey_pi0(ps0,  q=ALPHA)
    pi0_struct1, _ = adaptive_storey_pi0(ps1,  q=ALPHA)

    # Sensitivity: fixed-λ grid for the paper
    pi0_sens = {
        lam: float(np.clip(np.mean(pdis > lam) / (1.0 - lam), 0.0, 1.0))
        for lam in [0.3, 0.4, 0.5, 0.6, 0.7]
    }

    # Structural scope filter size (screen p-values, independent of T_Hou)
    m_prime = sum(
        1 for i in range(m)
        if min(psc0[i], psc1[i]) <= ALPHA
    )

    print(f"\n  m = {m:,}  |  m'' (scope-filtered) = {m_prime:,}  "
          f"({m - m_prime:,} excluded by screen p > {ALPHA})")
    print(f"\n  π̂₀ estimates (Adaptive Storey, Gao 2023):")
    print(f"    Discriminative  : {pi0_disc:.4f}  "
          f"→ Storey power gain ≈ {1.0/max(pi0_disc, 0.01):.2f}× over BH")
    print(f"    Structural (c0) : {pi0_struct0:.4f}")
    print(f"    Structural (c1) : {pi0_struct1:.4f}")
    print(f"\n  π̂₀_disc sensitivity (fixed λ):")
    for lam, v in pi0_sens.items():
        print(f"    λ={lam:.1f} : {v:.4f}")

    return {
        'pi0_disc':              pi0_disc,
        'pi0_struct_c0':         pi0_struct0,
        'pi0_struct_c1':         pi0_struct1,
        'pi0_disc_sensitivity':  pi0_sens,
        'm':                     m,
        'm_prime':               m_prime,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DOUBLE-NULL LOG BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _build_doubly_nullified_log(
    case_data_orig: dict,
    case_ids_sorted: list,
    permuted_labels: np.ndarray,
    random_state: int,
) -> dict:
    """
    Build a doubly-nullified log: σ_label ∘ σ_trace.

    σ_label: replace each case's outcome with permuted_labels[i].
             → destroys class–trace association.
             → p_disc^(b) ~ U(0,1) by Fisher randomisation.

    σ_trace: randomly permute activities within each trace (in-place on a copy).
             Preserves trace length and activity multiset per case.
             Destroys all temporal ordering.
             → p_struct^(b) ~ U(0,1) because the shuffled trace IS a draw
               from the structural null distribution used by
               run_structural_permutation_test internally.

    Shallow copies of CaseInfo are created; case_data_orig is never mutated.

    Args:
        case_data_orig:  Original case data dict.
        case_ids_sorted: Lexicographic case-ID ordering (fixes alignment with
                         the permuted_labels array).
        permuted_labels: (n,) permuted binary label vector for this replicate.
        random_state:    RNG seed for trace shuffling (independent offset from
                         the label/structural permutation seeds in P1's tests).

    Returns:
        Dict[case_id → CaseInfo] with shuffled traces and permuted labels.
    """
    rng      = np.random.RandomState(random_state)
    nullified = {}

    for i, cid in enumerate(case_ids_sorted):
        ci_orig = case_data_orig[cid]
        ci      = copy.copy(ci_orig)          # shallow copy

        # σ_label: override outcome
        ci.outcome = int(permuted_labels[i])

        # σ_trace: shuffle activity sequence (preserves multiset and length)
        shuffled_trace  = ci_orig.trace.copy()
        rng.shuffle(shuffled_trace)
        ci.trace         = shuffled_trace
        ci.activity_index = precompute_activity_index(
            shuffled_trace, case_id=cid
        )

        nullified[cid] = ci

    return nullified


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DOUBLY-NULL REPLICATE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_doubly_null_replicate(
    permuted_labels: np.ndarray,
    case_data_orig: dict,
    candidates_all: list,
    case_ids_sorted: list,
    B1_internal: int,
    B2_internal: int,
    alpha: float,
    alpha_drva: float,
    tau_star_dm: float,
    tau_min_dm: float,
    random_state: int,
    pi_drva_valid: int = PI_DRVA_VALID,
    n_workers: int = 1,
) -> dict:
    """
    Run all three methods on a single doubly-nullified held-out replicate.

    Under the double-null (σ_trace ∘ σ_label), every rejection is a false
    positive by construction, because L^(b) has no discriminative OR
    structural signal on any axis.

    ── P1 (Hou-Storey) ──────────────────────────────────────────────────────
    Steps 1–9 faithfully reproduce execute_three_hypothesis_protocol on L^(b):

    1. Build L^(b) via _build_doubly_nullified_log.
    2. Recompute holds_all on shuffled traces.
    3. Label permutation test (H₀ᵈ), B1_internal resamples → fresh p_disc.
    4. Structural permutation test (H₀ˢ), B2_internal per class → fresh p_struct.
    5. Determine dominant class from null prevalences.
    6. Hou oracle p-value (exact under double-null):
           T_Hou(i) = -2[W_STRUCT·ln p_s(i) + W_DISC·ln p_d(i)]
           p_Hou^oracle(i) = chi2.sf(T_Hou(i) / c_null, f_null)
           c_null = 0.52, f_null ≈ 3.846, rho_sd = 0 (independence).
       The real-data tf_null_matrix is NOT used here: it captures T_Hou
       under the real distribution (with genuine signal), creating a
       distributional mismatch that would inflate FDR_emp upward.
       The oracle is exact under double-null by the Satterthwaite argument.
    7. Sample-split scope filter: m'' = {i: min(p_screen_c0, p_screen_c1) ≤ α}.
    8. Adaptive Storey π̂₀ on m'' oracle p-values.
    9. Storey q-values → SINGLE GATE: q_Hou ≤ α.
       Matches p1 is_significant_final = is_significant_discriminative.
       Structural evidence is embedded in T_Hou via the W_STRUCT weight;
       a second hard structural gate would double-penalise patterns.

    Monotonicity: p_Hou^oracle is strictly decreasing in T_Hou, so the
    rejection set {i: q_Hou^oracle ≤ α} equals the set that p̃_Hou (from
    a proper double-null reference) would produce. FDR_emp correctly
    characterises the real Hou-Storey procedure.

    ── DRVA ─────────────────────────────────────────────────────────────────
    Calls run_drva_on_doubly_null_log with reduced π = pi_drva_valid.
    This re-encodes L^(b) (from shuffled traces — no temporal structure),
    then runs DRVA's internal shuffleLog test (label-only permutation of
    cached trace evaluations) and applies the per-rule raw α threshold.
    Seed is replicate-specific so null distributions are independent.

    Expected: E[V_b^DRVA] ≈ α_DRVA × m under the double-null (no correction),
    so FDR_emp^DRVA = E[V_b] / R_obs^DRVA. Since R_obs^DRVA is computed on
    the real data (where structural patterns inflate the count), this ratio
    may still exceed α_DRVA, demonstrating DRVA's lack of FDR control.

    ── DeclareMiner ─────────────────────────────────────────────────────────
    Calls run_declareminer_on_doubly_null_log with the precomputed holds_null.
    Recomputes confidence from holds_null, applies the fixed τ* threshold
    calibrated on the real data (zero additional iterations).

    Args:
        permuted_labels: (n,) labels for σ_label.
        case_data_orig:  Original (unpermuted, unshuffled) case data.
        candidates_all:  Fixed M_all candidate pool (shared across methods).
        case_ids_sorted: Lexicographic case-ID ordering.
        B1_internal:     Label perm budget for P1 (null replicate).
        B2_internal:     Structural perm budget for P1 (null replicate).
        alpha:           FDR level for P1 gate and DM τ_min guard.
        alpha_drva:      Per-rule significance level for DRVA.
        tau_star_dm:     Calibrated |Δconf| threshold for DeclareMiner.
        tau_min_dm:      Minimum confidence interestingness guard for DM.
        random_state:    Unique seed for this replicate.
        pi_drva_valid:   Reduced DRVA permutation budget.
        n_workers:       MUST be 1 when called from within Parallel.

    Returns:
        dict: {METHOD_P1: int, METHOD_DRVA: int, METHOD_DM: int,
               '__m_prime__': int}
    """
    m  = len(candidates_all)
    rs = random_state

    # ── Steps 1–2: build null log + recompute holds ───────────────────────
    null_case_data = _build_doubly_nullified_log(
        case_data_orig, case_ids_sorted, permuted_labels,
        random_state=rs + 200_000,
    )
    with _suppress_output():
        holds_null = compute_holds_by_case_batch(
            null_case_data, candidates_all
        )

    # ── Step 3: label permutation test (H₀ᵈ) — fresh p_disc ─────────────
    with _suppress_output():
        disc_results = run_label_permutation_test(
            null_case_data, candidates_all, holds_null,
            B1_internal, rs,
        )
    disc_results.pop('__null_delta_matrix__', None)
    p_disc = np.array([
        disc_results[spec]['p_two_sided'] for spec in candidates_all
    ])

    # ── Step 4: structural permutation test (H₀ˢ) — fresh p_struct ──────
    D_0, D_1   = split_by_class(null_case_data)
    cid_set_0  = set(D_0.keys())
    cid_set_1  = set(D_1.keys())

    with _suppress_output():
        struct0 = run_structural_permutation_test(
            D_0, candidates_all, class_label=0,
            B2=B2_internal, random_state=rs + 1, n_workers=n_workers,
        )
        struct1 = run_structural_permutation_test(
            D_1, candidates_all, class_label=1,
            B2=B2_internal, random_state=rs + 2, n_workers=n_workers,
        )

    p_screen_c0 = np.array([
        struct0[spec]['p_structural_screen'] if spec in struct0 else 1.0
        for spec in candidates_all
    ])
    p_screen_c1 = np.array([
        struct1[spec]['p_structural_screen'] if spec in struct1 else 1.0
        for spec in candidates_all
    ])
    p_test_c0 = np.array([
        struct0[spec]['p_structural_test'] if spec in struct0 else 1.0
        for spec in candidates_all
    ])
    p_test_c1 = np.array([
        struct1[spec]['p_structural_test'] if spec in struct1 else 1.0
        for spec in candidates_all
    ])

    # ── Step 5: dominant class from shuffled prevalences ─────────────────
    # delta_obs = P̂₁ − P̂₀ from the label permutation test; sign gives dominant.
    # Avoids a Python loop over compute_prevalence_from_holds per pattern.
    delta_obs_b  = np.array([disc_results[spec]['delta_obs'] for spec in candidates_all])
    dominant     = np.where(delta_obs_b >= 0.0, 1, 0)
    p_struct_dom = np.where(dominant == 1, p_test_c1, p_test_c0)

    # ── Step 6: Hou oracle analytic p-value (exact under double-null) ────
    # T_Hou = -2[W_STRUCT·ln p_s + W_DISC·ln p_d]
    # Under double-null: p_s ~ U(0,1), p_d ~ U(0,1) independently
    # → T_Hou ~ c·χ²_f  (c=_C_NULL≈0.52, f=_F_NULL≈3.846, rho_sd=0)
    tf_obs_b   = hou_combination_statistic(
        p_struct_dom, p_disc, w_s=W_STRUCT, w_d=W_DISC
    )
    p_hou_oracle = np.clip(
        stats.chi2.sf(tf_obs_b / _C_NULL, df=_F_NULL),
        1e-300, 1.0,
    )

    # ── Step 7: sample-split scope filter ────────────────────────────────
    structural_idx = [
        i for i in range(m)
        if min(p_screen_c0[i], p_screen_c1[i]) <= alpha
    ]
    m_prime = len(structural_idx)

    # ── Steps 8–9: Adaptive Storey + single gate q_Hou ≤ α ───────────────
    if m_prime > 0:
        p_hou_mp = p_hou_oracle[structural_idx]
        pi0_b, _ = adaptive_storey_pi0(p_hou_mp, q=alpha)
        q_hou_b  = storey_qvalue(p_hou_mp, pi0_b)
        n_p1     = int(np.sum(q_hou_b <= alpha))
    else:
        n_p1 = 0

    # ── DRVA: internal shuffleLog permutation test on null log ────────────
    drva_cfg_null = DRVA_CONFIG.copy()
    drva_cfg_null['pi']                 = pi_drva_valid
    drva_cfg_null['alpha']              = alpha_drva
    drva_cfg_null['hierarchical_pruning'] = False
    drva_cfg_null['mmin']               = 0.0
    drva_cfg_null['mdiff_min']          = 0.0

    with _suppress_output():
        n_drva = run_drva_on_doubly_null_log(
            null_case_data = null_case_data,
            candidates_all = candidates_all,
            alpha          = alpha_drva,
            replicate_seed = rs + 50_000,     # independent of P1 layers
            config         = drva_cfg_null,
            holds_all      = holds_null,      # fast path: skip re-evaluation
        )

    # ── DeclareMiner: apply fixed τ* to null confidence differences ───────
    with _suppress_output():
        n_dm = run_declareminer_on_doubly_null_log(
            null_case_data = null_case_data,
            candidates_all = candidates_all,
            tau_star       = tau_star_dm,
            tau_min        = tau_min_dm,
            holds_all      = holds_null,     # fast path: skip re-evaluation
        )

    return {
        METHOD_P1:     n_p1,
        METHOD_DRVA:   n_drva,
        METHOD_DM:     n_dm,
        '__m_prime__': m_prime,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — PARALLEL HELD-OUT NULL PERMUTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _null_worker(
    b,
    permuted_labels_b,
    case_data_orig,
    candidates_all,
    case_ids_sorted,
    B1_internal,
    B2_internal,
    pi_drva_valid,
    alpha,
    alpha_drva,
    tau_star_dm,
    tau_min_dm,
):
    """Joblib top-level worker for one doubly-null replicate (loky-safe).

    All budget parameters are passed explicitly so that loky worker processes
    (which inherit module state from import time, not from dry-run overrides)
    always use the correct values.
    """
    rs = BASE_SEED + 100_000 + b
    return run_doubly_null_replicate(
        permuted_labels = permuted_labels_b,
        case_data_orig  = case_data_orig,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        B1_internal     = B1_internal,
        B2_internal     = B2_internal,
        alpha           = alpha,
        alpha_drva      = alpha_drva,
        tau_star_dm     = tau_star_dm,
        tau_min_dm      = tau_min_dm,
        random_state    = rs,
        pi_drva_valid   = pi_drva_valid,
        n_workers       = 1,   # already inside Parallel — no nested spawning
    )


def run_null_permutations(
    case_data: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels: np.ndarray,
    tau_star_dm: float,
    tau_min_dm: float,
    b1_valid: int = B1_VALID,
    b2_valid: int = B2_VALID,
    pi_drva_valid: int = PI_DRVA_VALID,
    n_jobs: int = N_JOBS,
) -> dict:
    """
    Run B_NULL doubly-nullified held-out replicates in parallel.

    Budget parameters (b1_valid, b2_valid, pi_drva_valid) are passed explicitly
    rather than read from module globals inside workers. Under the loky backend,
    each worker process inherits module state from import time — before any
    dry-run overrides applied in __main__. Explicit parameters are serialized
    with the delayed call and always carry the correct values.

    Seed architecture (three non-overlapping layers, safe for B_NULL < 100,000):
        BASE_SEED + b                    held-out σ_label permutation
        BASE_SEED + 100_000 + b          P1 internal label/structural seeds
        BASE_SEED + 100_000 + b + 200_000  σ_trace activity shuffling
        BASE_SEED + 100_000 + b + 50_000   DRVA internal shuffleLog

    Returns:
        dict with keys:
            null_counts          — {method: (B,) int array}
            wall_seconds         — float
            m_prime_distribution — (B,) int array of P1 scope-filter sizes
    """
    print("\n" + "=" * 100)
    print("SECTION 8 — PARALLEL DOUBLY-NULL HELD-OUT PERMUTATIONS")
    print(f"  B_null={B_NULL}, B1_valid={b1_valid}, B2_valid={b2_valid}")
    print(f"  PI_DRVA_valid={pi_drva_valid}, n_jobs={n_jobs}")
    print(f"\n  Per-replicate protocol:")
    print(f"    1. σ_trace: shuffle activities within each trace")
    print(f"       → p_struct^(b) ~ U(0,1)")
    print(f"    2. σ_label: permute class labels (marginals preserved)")
    print(f"       → p_disc^(b) ~ U(0,1)")
    print(f"\n  P1:  fresh holds + p_struct (B2={b2_valid}) + p_disc (B1={b1_valid})")
    print(f"       oracle T_Hou/c ~ χ²_f  "
          f"(c={_C_NULL:.3f}, f={_F_NULL:.3f}, rho_sd=0)")
    print(f"       single gate: q_Hou ≤ {ALPHA}")
    print(f"  DRVA: holds fast path + shuffleLog (π={pi_drva_valid})")
    print(f"       per-rule p_Cecconi ≤ {ALPHA_DRVA}  (no FDR correction)")
    print(f"  DM:  fixed τ*={tau_star_dm:.4f} on null Δconf  (zero iterations)")
    print("=" * 100)

    t0 = time.time()

    print(f"\n  Generating {B_NULL} held-out label permutations (seed={BASE_SEED})...")
    permuted_labels_all = _generate_heldout_permutation_batch(
        labels, B_NULL, BASE_SEED
    )
    for i in range(min(5, B_NULL)):
        assert int(permuted_labels_all[i].sum()) == int(labels.sum()), \
            f"Replicate {i}: class marginals not preserved"
    print(f"  Marginal check passed  (n+={int(labels.sum()):,} preserved)")

    est_min = B_NULL * 25 / max(abs(n_jobs) if n_jobs != -1 else 8, 1)
    print(f"\n  Estimated wall time: ~{est_min:.0f} min  ({est_min/60:.1f} h)")
    print(f"  Launching {B_NULL} parallel workers (n_jobs={n_jobs})...\n")

    replicate_results = Parallel(
        n_jobs  = n_jobs,
        verbose = 10,
        backend = 'loky',
    )(
        delayed(_null_worker)(
            b,
            permuted_labels_all[b],
            case_data,
            candidates_all,
            case_ids_sorted,
            b1_valid,
            b2_valid,
            pi_drva_valid,
            ALPHA,
            ALPHA_DRVA,
            tau_star_dm,
            tau_min_dm,
        )
        for b in range(B_NULL)
    )

    null_counts     = {m_name: np.zeros(B_NULL, dtype=int) for m_name in ALL_METHODS}
    m_prime_per_rep = np.zeros(B_NULL, dtype=int)

    for b, res in enumerate(replicate_results):
        m_prime_per_rep[b] = res.pop('__m_prime__', -1)
        for method in ALL_METHODS:
            null_counts[method][b] = res[method]

    wall      = time.time() - t0
    n_zero_mp = int(np.sum(m_prime_per_rep == 0))

    print(f"\n  All {B_NULL} replicates complete  |  "
          f"wall={wall:.1f}s ({wall/60:.1f} min)")
    print(f"\n  P1 m'' distribution across {B_NULL} null replicates:")
    print(f"    mean={m_prime_per_rep.mean():.1f}  std={m_prime_per_rep.std():.1f}"
          f"  min={m_prime_per_rep.min()}  max={m_prime_per_rep.max()}"
          f"  zeros={n_zero_mp} ({n_zero_mp/B_NULL*100:.1f}%)")
    # Under σ_trace, p_screen ~ U(0,1), so the expected fraction passing
    # min(p_screen_c0, p_screen_c1) ≤ α is 1−(1−α)² ≈ 9.75% for α=0.05.
    # Consistently zero m'' indicates the scope filter is too aggressive for
    # the log size or that m is very small — FDR_emp becomes 0/0 (resolved to 0).
    if n_zero_mp / B_NULL > 0.10:
        print("  WARNING: >10% zero m'' replicates — P1 FDR_emp may be "
              "confounded by structural scope zeroing under the null. "
              f"Expected ~{100*(1-(1-ALPHA)**2):.1f}% of m to pass scope filter "
              f"under σ_trace (1−(1−α)²); consistently 0 suggests m is very "
              "small or temporal structure is atypically weak.")

    print(f"\n  Null rejection counts V_b per method  "
          f"(mean ± std over {B_NULL} replicates):")
    for method in ALL_METHODS:
        arr = null_counts[method]
        print(f"    {method:20s}: mean={arr.mean():.2f}  std={arr.std():.2f}"
              f"  max={arr.max()}  zeros={np.sum(arr==0)}")

    return {
        'null_counts':          null_counts,
        'wall_seconds':         wall,
        'm_prime_distribution': m_prime_per_rep,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — FDR METRICS AND OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def compute_and_save_metrics(
    null_counts: dict,
    R_obs: dict,
    dm_out: dict,
    diagnostics: dict,
    original_wall: float,
    perm_wall: float,
    m_prime_distribution: np.ndarray,
    pattern_results: list,
    null_delta_matrix: np.ndarray,
    delta_obs: np.ndarray,
) -> tuple:
    """
    Compute FDR_emp / PCER_emp / FWER_emp with BCa 95% CIs for all
    three methods. Save CSV, JSON, and NPZ output files.

    Nominal α for each method:
        P1:          ALPHA = 0.05  (Hou-Storey FDR gate)
        DRVA:        ALPHA_DRVA = 0.01  (per-rule, no correction)
        DeclareMiner:ALPHA = 0.05  (τ* calibrated to match P1's R_obs)

    Note: tf_null_matrix (the T_Hou null reference from the real-data P1 run)
    is intentionally NOT passed here. It was built under the real data
    distribution (with genuine signal); using it in null replicates would
    create a distributional mismatch that inflates FDR_emp.  The oracle
    chi2.sf is exact under double-null (rho_sd=0) and is used instead.

    Returns: (results_df, fdr_tests)
    """
    print("\n" + "=" * 100)
    print("SECTION 9 — FDR METRICS AND OUTPUT")
    print("=" * 100)

    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    m_total    = diagnostics['m']
    alpha_vals = {
        METHOD_P1:   ALPHA,
        METHOD_DRVA: ALPHA_DRVA,
        METHOD_DM:   ALPHA,
    }

    # ── Compute FDR metrics for each method ───────────────────────────────
    rows      = []
    fdr_tests = {}
    for method in ALL_METHODS:
        metrics = _compute_fdr_metrics(
            null_counts[method],
            R_obs[method],
            m_total,
            alpha_vals[method],
        )
        rows.append({'method': method, **metrics})
        fdr_tests[method] = {
            'fdr_emp':         metrics['FDR_emp'],
            'controls_FDR':    metrics['controls_FDR'],
            'FDR_CI_lower':    metrics['FDR_CI_lower'],
            'FDR_CI_upper':    metrics['FDR_CI_upper'],
        }

    results_df = pd.DataFrame(rows)

    print(f"\n  FDR Results (BPI_17, B_null={B_NULL}, double-null protocol):")
    print(f"  {'─'*90}")
    print(f"  {'Method':20s} {'α':>5s} {'R_obs':>6s} {'E[V_b]':>8s} "
          f"{'FDR_emp':>8s} {'95% CI':>22s} {'FWER':>7s} {'Pass?':>7s}")
    print(f"  {'─'*90}")
    for _, row in results_df.iterrows():
        ci_str  = f"[{row['FDR_CI_lower']:.4f}, {row['FDR_CI_upper']:.4f}]"
        verdict = "  PASS" if row['controls_FDR'] else "  FAIL"
        a       = alpha_vals[row['method']]
        print(
            f"  {row['method']:20s} {a:>5.3f} {row['R_obs']:>6d} "
            f"{row['E_V_b']:>8.2f} {row['FDR_emp']:>8.4f} "
            f"{ci_str:>22s} {row['FWER_emp']:>7.4f}{verdict}"
        )
    print(f"  {'─'*90}")

    # ── FILE 1: rq1_fdr_metrics.csv ───────────────────────────────────────
    csv_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_fdr_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── FILE 2: rq1_null_counts.csv ───────────────────────────────────────
    null_df              = pd.DataFrame({m: null_counts[m] for m in ALL_METHODS})
    null_df.index.name   = 'replicate_b'
    null_path            = os.path.join(RQ1_OUTPUT_DIR, "rq1_null_counts.csv")
    null_df.to_csv(null_path)
    print(f"  Saved: {null_path}")

    # ── FILE 2b: rq1_m_prime_distribution.csv ────────────────────────────
    mp    = m_prime_distribution
    mp_df = pd.DataFrame({'replicate_b': np.arange(B_NULL), 'm_prime': mp})
    mp_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_m_prime_distribution.csv")
    mp_df.to_csv(mp_path, index=False)
    n_zero = int(np.sum(mp == 0))
    print(f"  Saved: {mp_path}  "
          f"(mean={mp.mean():.1f}, zeros={n_zero}/{B_NULL}={n_zero/B_NULL*100:.1f}%)")

    # ── FILE 3: rq1_results.json ──────────────────────────────────────────
    p_disc_arr   = np.array([r.p_discriminative            for r in pattern_results])
    p_conj_arr   = np.array([r.p_conjunction               for r in pattern_results])
    p_conj_emp   = np.array([r.p_conjunction_empirical     for r in pattern_results])
    p_struct_c0  = np.array([r.p_structural_class0         for r in pattern_results])
    p_struct_c1  = np.array([r.p_structural_class1         for r in pattern_results])
    psc0_arr     = np.array([r.p_structural_screen_class0  for r in pattern_results])
    psc1_arr     = np.array([r.p_structural_screen_class1  for r in pattern_results])
    q_hou_arr    = np.array([r.q_value_sam                 for r in pattern_results])
    is_sig_arr   = np.array([r.is_significant_final        for r in pattern_results])

    full_json = {
        'rq1_version': '1.0',
        'log_name':    'BPI_17',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'methods':        ALL_METHODS,
            'shared_pool':    'M_all from Phase 0 DECLARE spec (fixed across methods)',
            'null_protocol':  (
                'Double-null: σ_label (permute class labels) ∘ σ_trace '
                '(shuffle activities within each trace). '
                'Under the double-null, p_struct ~ U(0,1) and p_disc ~ U(0,1) '
                'independently → every rejection is a false positive on BOTH axes.'
            ),
            'P1_gate':  'Single gate: q_Hou ≤ α (p1 is_significant_final)',
            'DRVA_gate': f'Per-rule p_Cecconi ≤ {ALPHA_DRVA} (no FDR correction)',
            'DM_gate':   f'|Δconf| ≥ τ*={dm_out["tau_star"]:.4f} (no stat test)',
        },

        'config': {
            'B_NULL':         B_NULL,
            'B1_FULL':        B1_FULL,
            'B2_FULL':        B2_FULL,
            'B1_VALID':       B1_VALID,
            'B2_VALID':       B2_VALID,
            'PI_DRVA_VALID':  PI_DRVA_VALID,
            'ALPHA':          ALPHA,
            'ALPHA_DRVA':     ALPHA_DRVA,
            'BASE_SEED':      BASE_SEED,
            'N_JOBS':         N_JOBS,
            'W_DISC':         W_DISC,
            'W_STRUCT':       W_STRUCT,
            'c_null':         float(_C_NULL),
            'f_null':         float(_F_NULL),
            'm_total':        diagnostics['m'],
            'm_prime_original': diagnostics['m_prime'],
            'tau_star_DM':    float(dm_out['tau_star']),
            'tau_min_DM':     float(dm_out.get('config', {}).get(
                                  'tau_min', DM_CONFIG['tau_min'])),
        },

        'R_obs': {m: int(R_obs[m]) for m in ALL_METHODS},

        'empirical_fdr_table': results_df.to_dict(orient='records'),

        'null_replicate_summary': {
            m: {
                'mean_V_b':   float(null_counts[m].mean()),
                'std_V_b':    float(null_counts[m].std()),
                'max_V_b':    int(null_counts[m].max()),
                'n_zero_V_b': int(np.sum(null_counts[m] == 0)),
            }
            for m in ALL_METHODS
        },

        'P1_m_prime_distribution': {
            'mean':   float(mp.mean()),
            'std':    float(mp.std()),
            'min':    int(mp.min()),
            'max':    int(mp.max()),
            'n_zero': n_zero,
        },

        'diagnostics': {
            'pi0_disc':                  diagnostics['pi0_disc'],
            'pi0_struct_c0':             diagnostics['pi0_struct_c0'],
            'pi0_struct_c1':             diagnostics['pi0_struct_c1'],
            'pi0_disc_sensitivity':      diagnostics['pi0_disc_sensitivity'],
            'storey_power_gain_over_bh': float(
                1.0 / max(diagnostics['pi0_disc'], 0.01)
            ),
        },

        'alignment_with_p1': {
            'conjunction': (
                f'T_Hou = -2[W_STRUCT·ln p_s + W_DISC·ln p_d], '
                f'W_STRUCT={W_STRUCT}, W_DISC={W_DISC} (Hou 2005). '
                f'Oracle: chi2.sf(T_Hou/{_C_NULL:.3f}, {_F_NULL:.3f}), rho_sd=0.'
            ),
            'single_gate': (
                'is_significant_final = q_Hou ≤ α only. '
                'Matches p1_BPI_17_hou.py v9.0-HOU-DOUBLY-NULL. '
                'Structural evidence embedded in T_Hou via W_STRUCT weight.'
            ),
            'why_oracle_not_tf_null_matrix': (
                'tf_null_matrix was built under the real data distribution '
                '(with genuine signal); using it in null replicates creates '
                'a distributional mismatch that inflates FDR_emp. '
                'Oracle chi2.sf is exact under double-null (rho_sd=0).'
            ),
        },

        'timing': {
            'original_P1_seconds':       original_wall,
            'null_permutations_seconds': perm_wall,
            'total_seconds':             original_wall + perm_wall,
        },

        'validation_checks': {
            'disc_p_res_original':  float(1.0 / (B1_FULL + 1)),
            'disc_p_res_valid':     float(1.0 / (B1_VALID + 1)),
            'conservative_bias':    'FDR_emp slightly underestimated (safe)',
            'marginals_preserved':  True,
            'seed_layers': {
                'held_out_label': 'BASE_SEED + b',
                'P1_internal':    'BASE_SEED + 100_000 + b',
                'trace_shuffle':  'BASE_SEED + 100_000 + b + 200_000',
                'DRVA_internal':  'BASE_SEED + 100_000 + b + 50_000',
                'no_overlap':     f'B_NULL={B_NULL} < 100_000',
            },
        },
    }

    json_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # ── FILE 4: rq1_pattern_arrays.npz ───────────────────────────────────
    struct_idx_arr = np.array([
        i for i in range(diagnostics['m'])
        if min(psc0_arr[i], psc1_arr[i]) <= ALPHA
    ])
    sigma_null_arr = np.std(null_delta_matrix, axis=0)

    arrays_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_pattern_arrays.npz")
    np.savez_compressed(
        arrays_path,
        p_discriminative        = p_disc_arr,
        p_conjunction           = p_conj_arr,
        p_conjunction_empirical = p_conj_emp,
        p_structural_c0         = p_struct_c0,
        p_structural_c1         = p_struct_c1,
        p_structural_dom        = np.array([
            r.p_structural_dominant for r in pattern_results
        ]),
        q_hou                   = q_hou_arr,
        q_structural_dom        = np.array([
            r.q_structural_dominant for r in pattern_results
        ]),
        delta_obs               = delta_obs,
        sigma_null              = sigma_null_arr,
        is_significant          = is_sig_arr,
        constraint_types        = np.array([
            r.constraint_type for r in pattern_results
        ]),
        structural_idx          = struct_idx_arr,
    )
    print(f"  Saved: {arrays_path}  ({diagnostics['m']:,} patterns)")

    return results_df, fdr_tests


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ1 — FDR CONTROL VALIDITY: BPI CHALLENGE 2017")
    print("Three methods on shared M_all: P1 (Hou-Storey) | DRVA | DeclareMiner")
    print("Double-null protocol: σ_label ∘ σ_trace")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  B_null={B_NULL}  B1_full={B1_FULL}  B2_full={B2_FULL}")
    print(f"  B1_valid={B1_VALID}  B2_valid={B2_VALID}  "
          f"PI_DRVA_valid={PI_DRVA_VALID}")
    print(f"  α(P1)={ALPHA}  α_DRVA={ALPHA_DRVA}")
    print(f"  Oracle: c={_C_NULL:.3f}, f={_F_NULL:.3f}  (rho_sd=0 under double-null)")
    print(f"  Output: {RQ1_OUTPUT_DIR}/")
    print(f"\n  Mechanistic prediction under doubly-null:")
    print(f"    P1:  FDR_emp ≈ {ALPHA}  (Adaptive Storey + scope filter control)")
    print(f"    DRVA: FDR_emp potentially >> {ALPHA_DRVA}  (no FDR correction; "
          f"E[V_b] ≈ {ALPHA_DRVA}×m under null)")
    print(f"    DM:  FDR_emp uncontrolled  (τ* has no principled link to α)")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    # ── Section 1: P1 full-budget run ─────────────────────────────────────
    p1_orig = run_p1_original(n_workers=N_JOBS)

    case_data         = p1_orig['case_data']
    candidates_all    = p1_orig['candidates_all']
    pattern_results   = p1_orig['pattern_results']
    null_delta_matrix = p1_orig['null_delta_matrix']
    delta_obs         = p1_orig['delta_obs']
    case_ids_sorted   = p1_orig['case_ids_sorted']
    labels            = p1_orig['labels']
    original_wall     = p1_orig['wall_seconds']

    # ── Section 2: DRVA original-data run ────────────────────────────────
    drva_orig = run_drva_original(case_data, candidates_all)

    # ── Section 3: DeclareMiner original-data run (calibrated to P1) ─────
    R_obs_p1 = sum(1 for r in pattern_results if r.is_significant_final)
    dm_orig  = run_declareminer_original(case_data, candidates_all, R_obs_p1)

    # ── Section 4: R_obs for all methods ─────────────────────────────────
    R_obs = collect_R_obs(pattern_results, drva_orig, dm_orig)

    # ── Section 5: diagnostics ────────────────────────────────────────────
    diagnostics = run_diagnostics(pattern_results)

    # ── Section 8: parallel doubly-null permutations ─────────────────────
    perm_out = run_null_permutations(
        case_data       = case_data,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        labels          = labels,
        tau_star_dm     = float(dm_orig['tau_star']),
        tau_min_dm      = float(dm_orig.get('config', {}).get(
                              'tau_min', DM_CONFIG['tau_min'])),
        b1_valid        = B1_VALID,
        b2_valid        = B2_VALID,
        pi_drva_valid   = PI_DRVA_VALID,
        n_jobs          = N_JOBS,
    )
    null_counts  = perm_out['null_counts']
    perm_wall    = perm_out['wall_seconds']
    m_prime_dist = perm_out['m_prime_distribution']

    # ── Section 9: FDR metrics and output ────────────────────────────────
    results_df, fdr_tests = compute_and_save_metrics(
        null_counts          = null_counts,
        R_obs                = R_obs,
        dm_out               = dm_orig,
        diagnostics          = diagnostics,
        original_wall        = original_wall,
        perm_wall            = perm_wall,
        m_prime_distribution = m_prime_dist,
        pattern_results      = pattern_results,
        null_delta_matrix    = null_delta_matrix,
        delta_obs            = delta_obs,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    total_wall = time.time() - t_total

    alpha_vals = {METHOD_P1: ALPHA, METHOD_DRVA: ALPHA_DRVA, METHOD_DM: ALPHA}

    print(f"\n{'='*100}")
    print("RQ1 — BPI_17 COMPLETE  (double-null, three-method comparison)")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  Output: {RQ1_OUTPUT_DIR}/")
    print(f"    rq1_fdr_metrics.csv             — Table (one row per method)")
    print(f"    rq1_null_counts.csv             — V_b ({B_NULL} rows × 3 methods)")
    print(f"    rq1_m_prime_distribution.csv    — P1 scope-filter sizes")
    print(f"    rq1_results.json                — full results for paper")
    print(f"    rq1_pattern_arrays.npz          — per-pattern arrays from P1")
    print(f"\n  KEY RESULTS:")
    print(f"  {'Method':20s} {'α':>5s} {'R_obs':>6s} "
          f"{'E[V_b]':>8s} {'FDR_emp':>8s} {'Pass?':>7s}")
    print(f"  {'─'*60}")
    for method in ALL_METHODS:
        fdr = fdr_tests[method]['fdr_emp']
        a   = alpha_vals[method]
        ev  = float(null_counts[method].mean())
        verd = "PASS" if fdr <= a else "FAIL"
        print(f"  {method:20s} {a:>5.3f} {R_obs[method]:>6d} "
              f"{ev:>8.2f} {fdr:>8.4f} {verd:>7s}")
    print(f"\n  Signal density: π̂₀_disc={diagnostics['pi0_disc']:.4f}"
          f"  → Storey power gain ≈ {1.0/max(diagnostics['pi0_disc'],0.01):.2f}× over BH")
    mp = m_prime_dist
    print(f"  P1 null m'': mean={mp.mean():.1f}  "
          f"zeros={int(np.sum(mp==0))}/{B_NULL}")
    print(f"{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RQ1 FDR Validity — BPI Challenge 2017  "
            "(P1 vs DRVA vs DeclareMiner, double-null protocol)"
        )
    )
    parser.add_argument(
        '--b-null', type=int, default=B_NULL,
        help=f'Held-out null replicates (default: {B_NULL})',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Parallel workers (-1 = all cores, default: {N_JOBS})',
    )
    parser.add_argument(
        '--alpha', type=float, default=ALPHA,
        help=f'P1 and DM FDR level (default: {ALPHA})',
    )
    parser.add_argument(
        '--alpha-drva', type=float, default=ALPHA_DRVA,
        help=f'DRVA per-rule significance level (default: {ALPHA_DRVA})',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Smoke test: B_null=2, reduced budgets',
    )
    args = parser.parse_args()

    if args.dry_run:
        B_NULL        = 2
        B1_VALID      = 50
        B2_VALID      = 30
        PI_DRVA_VALID = 20
        B_NULL_FULL   = 5
        B1_NULL_FULL  = 20
        B2_NULL_FULL  = 20
        print("*** DRY-RUN MODE: B_null=2, reduced budgets ***")
    else:
        B_NULL = args.b_null

    assert B_NULL < 100_000, (
        f"B_NULL={B_NULL} ≥ 100,000 would cause seed-layer overlap."
    )

    N_JOBS     = args.n_jobs
    ALPHA      = args.alpha
    ALPHA_DRVA = args.alpha_drva

    main()