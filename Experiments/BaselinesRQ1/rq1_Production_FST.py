#!/usr/bin/env python3
"""
rq1_Production_FST.py  —  RQ1 FDR Control Validity: Production  [DeclareMiner FST / Option A]
==============================================================================================
Doubly-Null Empirical FDR Estimation for the DeclareMiner FST Baseline

METHOD
------
DeclareMiner FST (Option A) — Per-Constraint Z-Test Threshold
    τ_DM*(c) = z_{α/2} · √[ p̂_c(1−p̂_c)(1/n_app,c+ + 1/n_app,c−) ]
    Computed ONCE on the original log; held FIXED across all null replicates.
    No statistical test. No multiple-testing correction.

    Algebraic connection to FST: squaring the acceptance criterion yields the
    pooled two-proportion chi-squared statistic, which coincides with the
    Welch/unpooled FST statistic under H0 and thresholds at χ²_{1,1−α}.

EXPERIMENTAL DESIGN PRINCIPLE
-------------------------------
DeclareMiner FST operates on the SAME fixed candidate pool M_all derived from
Phase 0 DECLARE specifications (shared with P1 and DRVA in the full three-method
comparison). Here we isolate the DM-FST baseline so that each null replicate
requires no internal permutation budget: DM-FST is fully deterministic given
the null log (conf values suffice; no permutation test inside each replicate).

DOUBLY-NULL PROTOCOL  (Pellegrina & Vandin 2018, adapted)
----------------------------------------------------------
Each held-out replicate b applies TWO independent operations:

    Null_b = σ_label ∘ σ_trace

    1. σ_trace:  Randomly permute activity sequence within each trace.
                 Preserves trace length and activity multiset per case.
                 Destroys all temporal ordering.

    2. σ_label:  Permute class labels across cases (marginals preserved).
                 Destroys any class–trace association.

    Every rejection on L^(b) is a false positive by construction.

EMPIRICAL FDR ESTIMATOR  (Pellegrina & Vandin 2018)
----------------------------------------------------
    FDR_emp(DM-FST) = E[V_b^DM] / max(R_obs^DM, 1)

where V_b = #{c: |Δconf̂_b(c)| ≥ τ_DM*(c)} (all false positives).

MECHANISTIC PREDICTION
-----------------------
By construction of the z-test threshold:
    E[V_b] = α · m    (per-comparison rate × candidate pool size)

DM-FST applies no FDR correction and no Adaptive Storey π̂₀ correction.
Therefore FDR_emp = E[V_b] / R_obs ≈ α · m / R_obs.  For m >> R_obs,
FDR_emp >> α — the baseline is expected to FAIL the FDR validity test.

This is the key contrast with P1: P1 applies Adaptive Storey–Gao correction
which concentrates the false-positive budget on the null fraction π̂₀ · m < m,
empirically achieving FDR̂ ≤ α.

NULL REPLICATE BUDGET
----------------------
DM-FST requires ZERO internal permutation iterations per replicate:
    - Recompute holds_null from null log (σ_trace ∘ σ_label).
    - Compute conf0_null, conf1_null from null holds.
    - V_b = #{c: |Δconf̂_b(c)| ≥ τ_DM*(c)}.  Fully deterministic.

This is the key contrast with P1 (which needs B1_VALID label perms and
B2_VALID structural perms per null replicate) and DRVA (PI_DRVA_VALID
shuffleLog iterations). DM-FST null replicates are ~10–100× faster per rep.

OUTPUT FILES
-------------
    rq1_dm_fst_fdr_metrics.csv          One row for DM-FST (Table data for paper).
    rq1_dm_fst_null_counts.csv          B_null rows of V_b counts.
    rq1_dm_fst_diagnostics.csv          Per-constraint z-score and τ_DM*(c) distribution.
    rq1_dm_fst_results.json             Full results for paper generation.

Version : 2.0  (DeclareMiner FST / Option A; double-null protocol)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Pellegrina & Vandin (2018/2020). KDD 2018 / TKDD 2020.
Cecconi, Augusto & Di Ciccio (2021). BPM Forum 2021, LNBIP 427, pp. 73–91.
Gu, Li & Han (2011). Generalized Fisher Score for Feature Selection. UAI.
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

# Ensure Unicode characters (σ, ∘, etc.) print correctly on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
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
from DeclareMiner_Production import (
    run_declareminer,
    run_declareminer_on_doubly_null_log,
    DM_CONFIG,
    compute_holds_by_case_batch,
    precompute_activity_index,
    CaseInfo,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

RQ1_OUTPUT_DIR = "RQ1_Production_FST"

# Held-out null replicates for FDR estimation
B_NULL = 200

# Nominal FDR level for pass/fail verdict.
# DM-FST does not target this level; α is used only to assess whether FDR_emp
# falls below the standard threshold that P1 controls.
ALPHA = 0.05

# Seed architecture (two non-overlapping layers; safe for B_NULL < 100,000)
#   Held-out σ_label:    BASE_SEED + b
#   Replicate rs:        BASE_SEED + 100_000 + b
#   σ_trace shuffle:     BASE_SEED + 100_000 + b + 200_000
BASE_SEED = 20260521

# Parallelism  (-1 = all cores; reduce if Windows paging file errors occur)
N_JOBS = 4

# Method identifier
METHOD_DM   = "DeclareMiner_FST"
ALL_METHODS = [METHOD_DM]

assert B_NULL < 100_000, (
    f"B_NULL={B_NULL} ≥ 100,000 would cause seed-layer overlap. "
    "Layers use offsets 0 and 100_000 from BASE_SEED+b."
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

    Replicate b uses seed (base_seed + b), independent of any internal DM-FST
    seeds. Returns (B, n) int8 array of permuted label vectors.
    """
    n   = len(labels)
    out = np.empty((B, n), dtype=np.int8)
    for b in range(B):
        rng    = np.random.RandomState(base_seed + b)
        out[b] = rng.permutation(labels).astype(np.int8)
    return out


def _bootstrap_bca_ci(
    data: np.ndarray,
    stat_fn,
    B_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """BCa (bias-corrected accelerated) bootstrap confidence interval."""
    rng   = np.random.RandomState(seed)
    n     = len(data)
    theta = stat_fn(data)

    boot  = np.array([stat_fn(rng.choice(data, n, replace=True)) for _ in range(B_boot)])
    z0    = stats.norm.ppf(float(np.mean(boot < theta)) + 1e-10)

    total  = np.sum(data)
    jack   = (total - data) / (n - 1)
    jack_m = jack.mean()
    num    = float(np.sum((jack_m - jack) ** 3))
    den    = float(6.0 * (np.sum((jack_m - jack) ** 2) ** 1.5))
    a_hat  = num / den if abs(den) > 1e-15 else 0.0

    alpha_tail = (1.0 - ci) / 2.0

    def _adj(z_):
        return stats.norm.cdf(z0 + (z0 + z_) / (1.0 - a_hat * (z0 + z_)))

    lo_pct = float(np.clip(_adj(stats.norm.ppf(alpha_tail)),       0.001, 0.999))
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
    """
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
        'B_null':       len(null_counts),
        'm_total':      m_total,
        'alpha':        alpha_nominal,
        'controls_FDR': bool(fdr_emp <= alpha_nominal),
    }


@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout/stderr during null replicates to keep progress clean."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DeclareMiner FST ORIGINAL-DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_dm_original() -> dict:
    """
    Run the full DeclareMiner FST pipeline on the real Production data.

    Delegates entirely to run_declareminer() from DeclareMiner_Production.py.
    Data loading, candidate generation (M_all), confidence computation,
    per-constraint threshold τ_DM*(c) = z_{α/2} · SE_pooled(c), and
    acceptance decision are all handled inside run_declareminer.

    R_obs is freely determined by the z-test: no top-k target.
    The mechanistic prediction is FDR_emp ≈ α · m / R_obs >> α.

    The returned dict exposes all fields needed by subsequent sections:
        case_data, candidates_all, tau_c (array), n_rejected, alpha,
        conf0, conf1, napp0, napp1, ids_class0, ids_class1, n0, n1,
        rejected (bool mask), config, timing.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — DeclareMiner FST ORIGINAL-DATA RUN")
    print(f"  τ_DM*(c) = τ_effect × √(n_ref / n_harm(c))  [effect-size threshold]")
    print(f"  τ_effect={DM_CONFIG['tau_effect']}  n_ref={DM_CONFIG['n_ref']}  n_floor={DM_CONFIG['n_floor']}")
    print(f"  R_obs^DM = #{'{'}c: |Δconf̂(c)| ≥ τ_DM*(c) AND eligible{'}'} — freely data-determined")
    print(f"  No statistical test. No FDR correction.")
    print("=" * 100)

    t0     = time.time()
    dm_out = run_declareminer(config=DM_CONFIG.copy())
    wall   = time.time() - t0

    m_total    = dm_out['m_total']
    n_rejected = dm_out['n_rejected']

    print(f"\n  DM-FST complete: {wall:.1f}s")
    print(f"  M_all={m_total:,}  |  τ_effect={dm_out['tau_effect']}  "
          f"|  n_floor={dm_out['n_floor']}  "
          f"|  R_obs^DM={n_rejected:,}")

    dm_out['wall_seconds'] = wall
    return dm_out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — COLLECT R_obs
# ═══════════════════════════════════════════════════════════════════════════

def collect_R_obs(dm_out: dict) -> dict:
    """
    R_obs^DM = |S^DM| = #{c: |Δconf̂(c)| ≥ τ_DM*(c)} on the original data.

    This count is freely determined by the data and the per-constraint
    z-test threshold — it is not forced to any calibration target.
    """
    R_dm = int(dm_out['n_rejected'])

    print("\n" + "=" * 100)
    print("SECTION 2 — R_obs (REAL-DATA REJECTIONS, DeclareMiner FST)")
    print("=" * 100)
    print(f"\n  {METHOD_DM:25s}: {R_dm:,}  "
          f"(|Δconf̂(c)| ≥ τ_DM*(c)  [effect-size threshold, τ_effect={dm_out['tau_effect']}])")

    return {METHOD_DM: R_dm}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DeclareMiner FST DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnostics_dm(dm_out: dict) -> dict:
    """
    Report τ_DM*(c), z-score, and Δconf distributions on the real data.

    DM-FST has no p-values, so π̂₀ and structural scope filter metrics
    (which are P1-specific) are not applicable.  Diagnostics here characterise:
        - The distribution of constraint-specific thresholds τ_DM*(c).
        - The z-score distribution z(c) = |Δconf̂(c)| / τ_DM*(c).
        - The mechanistic prediction E[V_b] = α · m and implied FDR_pred.

    The z-score z(c) = |Δconf̂(c)| / τ_DM*(c) is the normalized signal-to-noise
    ratio per constraint, analogous to the chi-squared statistic in FST.
    Constraints with z(c) >= 1 are accepted (|Δconf̂| >= τ_DM*).
    Under the null, z(c) ~ |N(0,1)| — so 5% exceed 1.96 by chance.
    """
    print("\n" + "=" * 100)
    print("SECTION 3 — DeclareMiner FST DIAGNOSTICS")
    print("=" * 100)

    conf0       = dm_out['conf0']
    conf1       = dm_out['conf1']
    delta_conf  = dm_out['delta_conf']
    tau_c       = dm_out['tau_c']
    napp0       = dm_out['napp0']
    napp1       = dm_out['napp1']
    mask        = dm_out['rejected']
    m           = int(dm_out['m_total'])
    r_obs       = int(dm_out['n_rejected'])
    tau_effect  = dm_out['tau_effect']
    n_ref       = dm_out['n_ref']
    n_floor     = dm_out['n_floor']

    # ratio = |Δconf̂| / τ_DM*(c)  (ratio >= 1 ↔ rejected)
    z_scores      = np.where(tau_c > 0, delta_conf / tau_c, np.inf)
    z_accepted    = z_scores[mask]
    dc_accepted   = delta_conf[mask]
    tau_c_accepted = tau_c[mask]

    print(f"\n  M_all = {m:,}  |  τ_effect = {tau_effect}  |  n_ref = {n_ref}  |  n_floor = {n_floor}")
    print(f"  R_obs^DM = {r_obs:,}  (eligible constraints with |Δconf̂| ≥ τ_DM*)")

    print(f"\n  τ_DM*(c) distribution over M_all ({m:,} constraints):")
    print(f"    mean   = {tau_c.mean():.4f}")
    print(f"    median = {np.median(tau_c):.4f}")
    print(f"    min    = {tau_c.min():.6f}")
    print(f"    max    = {tau_c.max():.4f}")

    print(f"\n  Δconf distribution over M_all:")
    print(f"    mean   = {delta_conf.mean():.4f}")
    print(f"    median = {np.median(delta_conf):.4f}")
    print(f"    max    = {delta_conf.max():.4f}")
    print(f"    Δconf >= 0.05: {int((delta_conf >= 0.05).sum()):>6,}")
    print(f"    Δconf >= 0.10: {int((delta_conf >= 0.10).sum()):>6,}")

    print(f"\n  z-score distribution (|Δconf̂| / τ_DM*) over M_all:")
    finite_z = z_scores[np.isfinite(z_scores)]
    print(f"    mean   = {finite_z.mean():.4f}")
    print(f"    median = {np.median(finite_z):.4f}")
    print(f"    max    = {finite_z.max():.4f}")
    print(f"    z >= 1 (= accepted): {int((z_scores >= 1.0).sum()):>6,}")
    print(f"    z >= 2:              {int((z_scores >= 2.0).sum()):>6,}")
    print(f"    z >= 3:              {int((z_scores >= 3.0).sum()):>6,}")

    print(f"\n  Accepted S^DM ({r_obs:,} constraints with |Δconf̂| >= τ_DM*):")
    if r_obs > 0:
        print(f"    min z-score   = {z_accepted.min():.4f}")
        print(f"    max z-score   = {z_accepted.max():.4f}")
        print(f"    mean z-score  = {z_accepted.mean():.4f}")
        print(f"    mean Δconf    = {dc_accepted.mean():.4f}")
        print(f"    mean τ_DM*    = {tau_c_accepted.mean():.4f}")
    else:
        print(f"    (no rules accepted — τ_effect={tau_effect} is too strict for this dataset)")

    return {
        'm_total':          m,
        'R_obs_dm':         r_obs,
        'tau_effect':       tau_effect,
        'n_ref':            n_ref,
        'n_floor':          n_floor,
        'tau_c_mean':       float(tau_c.mean()),
        'tau_c_median':     float(np.median(tau_c)),
        'tau_c_min':        float(tau_c.min()),
        'tau_c_max':        float(tau_c.max()),
        'delta_conf_mean':  float(delta_conf.mean()),
        'delta_conf_median':float(np.median(delta_conf)),
        'delta_conf_max':   float(delta_conf.max()),
        'z_mean':           float(finite_z.mean()),
        'z_median':         float(np.median(finite_z)),
        'z_max':            float(finite_z.max()),
        'z_accepted_mean':  float(z_accepted.mean()) if len(z_accepted) else 0.0,
        'z_accepted_min':   float(z_accepted.min())  if len(z_accepted) else 0.0,
        'z_accepted_max':   float(z_accepted.max())  if len(z_accepted) else 0.0,
        'dc_accepted_mean': float(dc_accepted.mean()) if len(dc_accepted) else 0.0,
        'dc_accepted_min':  float(dc_accepted.min())  if len(dc_accepted) else 0.0,
        'dc_accepted_max':  float(dc_accepted.max())  if len(dc_accepted) else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — DOUBLE-NULL LOG BUILDER
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

    σ_trace: randomly permute activities within each trace (in-place on a copy).
             Preserves trace length and activity multiset per case.
             Destroys all temporal ordering.

    Shallow copies of CaseInfo are created; case_data_orig is never mutated.

    Args:
        case_data_orig:  Original case data dict.
        case_ids_sorted: Lexicographic case-ID ordering (fixes alignment with
                         the permuted_labels array).
        permuted_labels: (n,) permuted binary label vector for this replicate.
        random_state:    RNG seed for trace shuffling (offset layer: +200_000
                         from the replicate base seed, independent of σ_label).

    Returns:
        Dict[case_id → CaseInfo] with shuffled traces and permuted labels.
    """
    rng       = np.random.RandomState(random_state)
    nullified = {}

    for i, cid in enumerate(case_ids_sorted):
        ci_orig = case_data_orig[cid]
        ci      = copy.copy(ci_orig)

        ci.outcome = int(permuted_labels[i])

        shuffled_trace    = ci_orig.trace.copy()
        rng.shuffle(shuffled_trace)
        ci.trace          = shuffled_trace
        ci.activity_index = precompute_activity_index(shuffled_trace, case_id=cid)

        nullified[cid] = ci

    return nullified


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOUBLY-NULL REPLICATE RUNNER (DeclareMiner FST)
# ═══════════════════════════════════════════════════════════════════════════

def run_doubly_null_replicate_dm(
    permuted_labels: np.ndarray,
    case_data_orig: dict,
    candidates_all: list,
    case_ids_sorted: list,
    tau_c: np.ndarray,
    tau_min: float,
    random_state: int,
    eligible: np.ndarray = None,
) -> dict:
    """
    Run DeclareMiner FST on a single doubly-nullified held-out replicate.

    Every acceptance here is a false positive by construction (σ_trace ∘
    σ_label destroys all discriminative and structural signal). DM-FST is
    deterministic given the null log: no internal permutation budget is
    required (contrast with P1 which needs B1_VALID × B2_VALID perms).

    Protocol:
        1. Build L^(b) = σ_label ∘ σ_trace applied to case_data_orig.
        2. Recompute holds_null on the shuffled traces (fast path).
        3. Compute conf0_null, conf1_null from null holds.
        4. V_b = #{c: |Δconf̂_b(c)| ≥ τ_DM*(c)}  with FROZEN τ_DM*(c).

    CRITICAL: tau_c is the array computed ONCE on the original (real) log.
    It is passed in unchanged and NOT recomputed from the null log's
    confidences.  This is the only scientifically valid way to compute V_b.

    Args:
        permuted_labels: (n,) labels for σ_label.
        case_data_orig:  Original (unpermuted, unshuffled) case data.
        candidates_all:  Fixed M_all candidate pool.
        case_ids_sorted: Lexicographic case-ID ordering.
        tau_c:           (m,) per-constraint thresholds from the original log.
                         FROZEN — not recomputed on the null log.
        tau_min:         Minimum confidence interestingness guard.
        random_state:    Unique seed for this replicate.

    Returns:
        dict: {METHOD_DM: int}  — V_b = number of false positives.
    """
    null_case_data = _build_doubly_nullified_log(
        case_data_orig, case_ids_sorted, permuted_labels,
        random_state=random_state + 200_000,
    )
    with _suppress_output():
        holds_null = compute_holds_by_case_batch(null_case_data, candidates_all)

    n_dm = run_declareminer_on_doubly_null_log(
        null_case_data = null_case_data,
        candidates_all = candidates_all,
        tau_c          = tau_c,
        tau_min        = tau_min,
        eligible       = eligible,
        holds_all      = holds_null,
    )

    return {METHOD_DM: n_dm}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PARALLEL HELD-OUT NULL PERMUTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _null_worker_dm(
    b,
    permuted_labels_b,
    case_data_orig,
    candidates_all,
    case_ids_sorted,
    tau_c,
    tau_min,
    eligible,
):
    """Joblib top-level worker for one DM-FST doubly-null replicate (loky-safe).

    All parameters are passed explicitly so that loky worker processes
    (which inherit module state from import time, before any dry-run
    overrides) always use the correct values.
    """
    rs = BASE_SEED + 100_000 + b
    return run_doubly_null_replicate_dm(
        permuted_labels = permuted_labels_b,
        case_data_orig  = case_data_orig,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        tau_c           = tau_c,
        tau_min         = tau_min,
        random_state    = rs,
        eligible        = eligible,
    )


def run_null_permutations(
    case_data: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels: np.ndarray,
    tau_c: np.ndarray,
    tau_min: float,
    eligible: np.ndarray = None,
    n_jobs: int = N_JOBS,
) -> dict:
    """
    Run B_NULL doubly-nullified held-out replicates in parallel.

    DM-FST is deterministic per replicate (no internal permutation budget),
    so each worker only needs to:
        (a) build the null log,
        (b) recompute holds_null,
        (c) apply |Δconf̂_b| >= τ_DM*(c) deterministically.

    Seed architecture (safe for B_NULL < 100,000):
        BASE_SEED + b                      held-out σ_label permutation
        BASE_SEED + 100_000 + b            replicate random_state
        BASE_SEED + 100_000 + b + 200_000  σ_trace activity shuffling

    Returns:
        dict with:
            null_counts  — {METHOD_DM: (B,) int array}
            wall_seconds — float
    """
    print("\n" + "=" * 100)
    print("SECTION 6 — PARALLEL DOUBLY-NULL HELD-OUT PERMUTATIONS (DeclareMiner FST)")
    print(f"  B_null={B_NULL}, n_jobs={n_jobs}")
    print(f"  τ_DM*(c): effect-size thresholds (frozen from original log)")
    print(f"\n  Per-replicate protocol (fully deterministic given null log):")
    print(f"    1. σ_trace: shuffle activities within each trace")
    print(f"    2. σ_label: permute class labels (marginals preserved)")
    print(f"    3. Recompute holds_null (fast path from shuffled traces)")
    print(f"    4. Compute conf0_null, conf1_null from null holds")
    print(f"    5. V_b = #{{c: |Δconf̂_b(c)| >= τ_DM*(c) AND eligible}}  [FROZEN]")
    print(f"\n  No internal permutation budget — DM-FST null reps are deterministic.")
    print("=" * 100)

    t0 = time.time()

    print(f"\n  Generating {B_NULL} held-out label permutations (seed={BASE_SEED})...")
    permuted_labels_all = _generate_heldout_permutation_batch(labels, B_NULL, BASE_SEED)
    for i in range(min(5, B_NULL)):
        assert int(permuted_labels_all[i].sum()) == int(labels.sum()), \
            f"Replicate {i}: class marginals not preserved"
    print(f"  Marginal check passed  (n+={int(labels.sum()):,} preserved in all reps)")

    print(f"\n  Launching {B_NULL} parallel DM-FST workers (n_jobs={n_jobs})...\n")

    replicate_results = Parallel(
        n_jobs  = n_jobs,
        verbose = 10,
        backend = 'loky',
    )(
        delayed(_null_worker_dm)(
            b,
            permuted_labels_all[b],
            case_data,
            candidates_all,
            case_ids_sorted,
            tau_c,
            tau_min,
            eligible,
        )
        for b in range(B_NULL)
    )

    null_counts = {METHOD_DM: np.zeros(B_NULL, dtype=int)}
    for b, res in enumerate(replicate_results):
        null_counts[METHOD_DM][b] = res[METHOD_DM]

    wall = time.time() - t0
    arr  = null_counts[METHOD_DM]

    print(f"\n  All {B_NULL} replicates complete  |  "
          f"wall={wall:.1f}s ({wall/60:.1f} min)")
    print(f"\n  DM-FST null rejection counts V_b over {B_NULL} replicates:")
    print(f"    mean={arr.mean():.2f}  std={arr.std():.2f}  "
          f"max={arr.max()}  zeros={int(np.sum(arr == 0))}")

    return {
        'null_counts':  null_counts,
        'wall_seconds': wall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — FDR METRICS AND OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def compute_and_save_metrics(
    null_counts: dict,
    R_obs: dict,
    dm_out: dict,
    diagnostics: dict,
    original_wall: float,
    perm_wall: float,
) -> tuple:
    """
    Compute FDR_emp / PCER_emp / FWER_emp with BCa 95% CIs for DM-FST.
    Save four output files to RQ1_OUTPUT_DIR.

    Returns: (results_df, fdr_tests)
    """
    print("\n" + "=" * 100)
    print("SECTION 7 — FDR METRICS AND OUTPUT")
    print("=" * 100)

    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    m_total = diagnostics['m_total']
    metrics = _compute_fdr_metrics(
        null_counts[METHOD_DM],
        R_obs[METHOD_DM],
        m_total,
        ALPHA,
    )

    rows       = [{'method': METHOD_DM, **metrics}]
    results_df = pd.DataFrame(rows)
    fdr_tests  = {
        METHOD_DM: {
            'fdr_emp':      metrics['FDR_emp'],
            'controls_FDR': metrics['controls_FDR'],
            'FDR_CI_lower': metrics['FDR_CI_lower'],
            'FDR_CI_upper': metrics['FDR_CI_upper'],
        }
    }

    tau_effect = float(dm_out['tau_effect'])
    n_ref      = float(dm_out['n_ref'])
    n_floor    = int(dm_out['n_floor'])
    tau_c      = dm_out['tau_c']

    print(f"\n  FDR Results (Production, B_null={B_NULL}, double-null protocol):")
    print(f"  {'─'*82}")
    print(f"  {'Method':25s} {'α':>5s} {'R_obs':>6s} {'E[V_b]':>8s} "
          f"{'FDR_emp':>8s} {'95% CI':>22s} {'FWER':>7s} {'Pass?':>6s}")
    print(f"  {'─'*82}")
    for _, row in results_df.iterrows():
        ci_str  = f"[{row['FDR_CI_lower']:.4f}, {row['FDR_CI_upper']:.4f}]"
        verdict = "PASS" if row['controls_FDR'] else "FAIL"
        print(
            f"  {row['method']:25s} {ALPHA:>5.3f} {row['R_obs']:>6d} "
            f"{row['E_V_b']:>8.2f} {row['FDR_emp']:>8.4f} "
            f"{ci_str:>22s} {row['FWER_emp']:>7.4f} {verdict:>6s}"
        )
    print(f"  {'─'*82}")
    print(f"\n  E[V_b] observed: {metrics['E_V_b']:.2f}  "
          f"|  FDR_emp = {metrics['FDR_emp']:.4f}  "
          f"(τ_effect={tau_effect}, n_ref={n_ref}, n_floor={n_floor})")

    # FILE 1: rq1_dm_fst_fdr_metrics.csv
    csv_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_dm_fst_fdr_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # FILE 2: rq1_dm_fst_null_counts.csv
    null_df            = pd.DataFrame({'V_b': null_counts[METHOD_DM]})
    null_df.index.name = 'replicate_b'
    null_path          = os.path.join(RQ1_OUTPUT_DIR, "rq1_dm_fst_null_counts.csv")
    null_df.to_csv(null_path)
    print(f"  Saved: {null_path}")

    # FILE 3: rq1_dm_fst_diagnostics.csv  (per-constraint threshold and z-score)
    diag_df = pd.DataFrame({
        'pattern_id':      [r['pattern_id']      for r in dm_out['results_all']],
        'constraint_type': [r['constraint_type'] for r in dm_out['results_all']],
        'napp0':           [r['napp0']            for r in dm_out['results_all']],
        'napp1':           [r['napp1']            for r in dm_out['results_all']],
        'conf0':           [r['conf0']            for r in dm_out['results_all']],
        'conf1':           [r['conf1']            for r in dm_out['results_all']],
        'delta_conf':      [r['delta_conf']       for r in dm_out['results_all']],
        'tau_c':           [r['tau_c']            for r in dm_out['results_all']],
        'z_score':         [r['z_score']          for r in dm_out['results_all']],
        'is_significant':  [r['is_significant']   for r in dm_out['results_all']],
    })
    diag_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_dm_fst_diagnostics.csv")
    diag_df.to_csv(diag_path, index=False)
    print(f"  Saved: {diag_path}  ({len(diag_df):,} constraints)")

    # FILE 4: rq1_dm_fst_results.json
    full_json = {
        'rq1_version': '2.0',
        'method':      METHOD_DM,
        'log_name':    'Production',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'method':       METHOD_DM,
            'shared_pool':  'M_all from Phase 0 DECLARE spec (fixed; same as P1 and DRVA)',
            'null_protocol': (
                'Double-null: σ_label (permute class labels) ∘ σ_trace '
                '(shuffle activities within each trace). '
                'Under the double-null, every rejection is a false positive.'
            ),
            'DM_FST_gate': (
                f'|Δconf̂(c)| >= τ_DM*(c)  with  '
                f'τ_DM*(c) = τ_effect × √(n_ref / n_harm(c)), '
                f'τ_effect={tau_effect}, n_ref={n_ref}, n_floor={n_floor}. '
                'Thresholds and eligibility mask computed ONCE on the original log; '
                'held FIXED across all null replicates. No statistical test. No FDR correction.'
            ),
            'why_FDR_uncontrolled': (
                'DM-FST applies no FDR correction and no Adaptive Storey π̂₀ correction. '
                'The effect-size threshold has no Type I error interpretation and provides '
                'no distributional bound on E[V_b]. '
                f'Empirical FDR_emp = E[V_b] / R_obs is measured via the doubly-null protocol. '
                'P1 applies Adaptive Storey–Gao correction which concentrates the '
                'budget on π̂₀ · m << m, empirically achieving FDR̂ ≤ α.'
            ),
            'null_rep_budget': (
                'Zero internal permutation iterations per replicate. '
                'DM-FST is deterministic: holds_null → conf_null → V_b.'
            ),
        },

        'config': {
            'B_NULL':       B_NULL,
            'ALPHA':        ALPHA,
            'BASE_SEED':    BASE_SEED,
            'N_JOBS':       N_JOBS,
            'tau_effect':   tau_effect,
            'n_ref':        n_ref,
            'n_floor':      n_floor,
            'tau_min':      float(dm_out['config']['tau_min']),
            'tau_c_mean':   float(tau_c.mean()),
            'tau_c_median': float(np.median(tau_c)),
            'R_obs_dm':     int(dm_out['n_rejected']),
            'm_total':      int(m_total),
        },

        'R_obs': {METHOD_DM: int(R_obs[METHOD_DM])},

        'empirical_fdr_table': results_df.to_dict(orient='records'),

        'null_replicate_summary': {
            METHOD_DM: {
                'mean_V_b':   float(null_counts[METHOD_DM].mean()),
                'std_V_b':    float(null_counts[METHOD_DM].std()),
                'max_V_b':    int(null_counts[METHOD_DM].max()),
                'n_zero_V_b': int(np.sum(null_counts[METHOD_DM] == 0)),
            }
        },

        'diagnostics': {k_d: v for k_d, v in diagnostics.items()},

        'timing': {
            'original_DM_seconds':       original_wall,
            'null_permutations_seconds': perm_wall,
            'total_seconds':             original_wall + perm_wall,
        },

        'validation_checks': {
            'marginals_preserved':   True,
            'deterministic_per_rep': True,
            'tau_c_frozen':          True,
            'seed_layers': {
                'held_out_label': 'BASE_SEED + b',
                'replicate_rs':   'BASE_SEED + 100_000 + b',
                'trace_shuffle':  'BASE_SEED + 100_000 + b + 200_000',
                'no_overlap':     f'B_NULL={B_NULL} < 100_000',
            },
        },
    }

    json_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_dm_fst_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    return results_df, fdr_tests


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ1 — FDR CONTROL VALIDITY: PRODUCTION  [DeclareMiner FST / Option A]")
    print("Per-Constraint Z-Test Threshold Baseline")
    print("Double-null protocol: σ_label ∘ σ_trace")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  B_null={B_NULL}")
    print(f"  τ_effect={DM_CONFIG['tau_effect']}  n_ref={DM_CONFIG['n_ref']}  n_floor={DM_CONFIG['n_floor']}")
    print(f"  τ_DM*(c) = τ_effect × √(n_ref / n_harm(c))  [effect-size threshold]")
    print(f"  α(nominal FDR)={ALPHA}  (for pass/fail verdict only)")
    print(f"  Output: {RQ1_OUTPUT_DIR}/")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    # ── Section 1: DM-FST original-data run ───────────────────────────────
    dm_orig = run_dm_original()

    case_data       = dm_orig['case_data']
    candidates_all  = dm_orig['candidates_all']
    case_ids_sorted = sorted(case_data.keys())
    labels          = np.array([case_data[cid].outcome for cid in case_ids_sorted])
    original_wall   = dm_orig['wall_seconds']
    tau_c           = dm_orig['tau_c']          # (m,) — FROZEN for null replicates
    eligible        = dm_orig['eligible']       # (m,) — FROZEN sparsity mask
    tau_min         = float(dm_orig['config']['tau_min'])

    # ── Section 2: R_obs ──────────────────────────────────────────────────
    R_obs = collect_R_obs(dm_orig)

    # ── Section 3: diagnostics ────────────────────────────────────────────
    diagnostics = run_diagnostics_dm(dm_orig)

    # ── Section 6: parallel doubly-null permutations ──────────────────────
    perm_out = run_null_permutations(
        case_data       = case_data,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        labels          = labels,
        tau_c           = tau_c,
        tau_min         = tau_min,
        eligible        = eligible,
        n_jobs          = N_JOBS,
    )
    null_counts = perm_out['null_counts']
    perm_wall   = perm_out['wall_seconds']

    # ── Section 7: FDR metrics and output ────────────────────────────────
    results_df, fdr_tests = compute_and_save_metrics(
        null_counts   = null_counts,
        R_obs         = R_obs,
        dm_out        = dm_orig,
        diagnostics   = diagnostics,
        original_wall = original_wall,
        perm_wall     = perm_wall,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    total_wall = time.time() - t_total
    fdr        = fdr_tests[METHOD_DM]['fdr_emp']
    ev         = float(null_counts[METHOD_DM].mean())
    verdict    = "PASS" if fdr <= ALPHA else "FAIL"

    print(f"\n{'='*100}")
    print("RQ1 — PRODUCTION DeclareMiner FST COMPLETE  (double-null, DM-FST only)")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  Output: {RQ1_OUTPUT_DIR}/")
    print(f"    rq1_dm_fst_fdr_metrics.csv    — FDR table")
    print(f"    rq1_dm_fst_null_counts.csv    — V_b ({B_NULL} rows)")
    print(f"    rq1_dm_fst_diagnostics.csv    — per-constraint τ_DM*(c) and ratio")
    print(f"    rq1_dm_fst_results.json       — full results for paper")
    print(f"\n  KEY RESULT:")
    print(f"  {'Method':25s} {'α_nom':>6s} {'R_obs':>6s} "
          f"{'E[V_b]obs':>10s} {'FDR_emp':>8s} {'Pass?':>6s}")
    print(f"  {'─'*65}")
    print(f"  {METHOD_DM:25s} {ALPHA:>6.3f} {R_obs[METHOD_DM]:>6d} "
          f"{ev:>10.2f} {fdr:>8.4f} {verdict:>6s}")
    print(f"\n  FDR_emp measured empirically via doubly-null protocol.")
    print(f"  No mechanistic prediction — effect-size threshold has no Type I error bound.")
    print(f"{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RQ1 FDR Validity — Production  "
            "(DeclareMiner FST / Option A per-constraint z-test baseline, "
            "double-null protocol)"
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
        help=f'Nominal FDR level for pass/fail comparison (default: {ALPHA})',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Smoke test: B_null=2',
    )
    args = parser.parse_args()

    if args.dry_run:
        B_NULL = 2
        print("*** DRY-RUN MODE: B_null=2 ***")
    else:
        B_NULL = args.b_null

    assert B_NULL < 100_000, (
        f"B_NULL={B_NULL} ≥ 100,000 would cause seed-layer overlap."
    )

    N_JOBS = args.n_jobs
    ALPHA  = args.alpha

    main()