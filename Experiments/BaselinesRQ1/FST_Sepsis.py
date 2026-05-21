"""
FST_Sepsis_parallel.py  —  Fisher Score Threshold (FST) Non-Statistical Baseline: Sepsis
===================================================================================================
Replaces the DeclareMiner differential confidence threshold baseline with the
Fisher Score Threshold (FST) criterion, grounded in the feature selection
component of Di Francescomarino et al. (2025).

PURPOSE
-------
FST is the non-statistical baseline for the RQ1 and RQ2 ablation study
alongside P1 (Hou-Storey conjunction) and DRVA.  It uses the same fixed
candidate pool M_all as P1 and DRVA, removing candidate-pool confounding so
that all differences in rejection counts reflect only the testing procedure,
not the candidate scope.

The three-method hierarchy is a clean one-variable-at-a-time ablation:

    Method   | Discriminative signal              | Structural | Multiple-testing
    ---------|------------------------------------|------------|------------------
    FST       | F(c) >= τ_F* (Fisher score thresh) | None       | None
    DRVA      | Perm. p-value on Δconf <= α_DRVA   | None       | None (raw α)
    P1 (Ours) | Hou T statistic on p_disc           | p_struct   | Adaptive Storey

FST CRITERION
-------------
For each constraint c ∈ M_all, compute the Fisher score:

    F(c) = (conf₁(c) − conf₀(c))² / (conf₁(c)(1−conf₁(c))/napp₁(c)
                                     + conf₀(c)(1−conf₀(c))/napp₀(c))

where:
    conf_y(c) = #{t ∈ L_y : t non-vacuously satisfies c} / #{t ∈ L_y : t activates c}
              = n_sat_y / napp_y
    napp_y(c) = number of non-vacuously applicable cases in class y

This is the squared Welch t-statistic for comparing two independent
Bernoulli proportions. The numerator is the squared confidence difference.
The denominator is the sum of the Bernoulli sampling variances of the two
class-conditional confidence estimates, weighted by 1/napp_y. Using
1/napp_y (not 1/n_y) is correct because conf_y is estimated from napp_y
non-vacuous cases, and its sampling variance is conf_y(1-conf_y)/napp_y.

RELATIONSHIP TO DECLAREMINER AND BISE 2025
------------------------------------------
The Fisher score is the feature selection criterion in Di Francescomarino
et al. (2025), Step 7.3 of the DvM pipeline (generalized Fisher score,
Gu et al. 2011). In that pipeline, it ranks features before classifier
training. Here we extract this single component and apply it directly to
M_all as a per-constraint acceptance criterion, without any classifier
training. FST is therefore the discriminative selection component of the
DvM pipeline applied directly, in its purest form.

FST strictly generalises the raw |Δconf| threshold used in the original
DeclareMiner baseline: sorting by F(c) and sorting by |conf₁ - conf₀|
produce different orderings whenever constraints differ in within-class
variance. F(c) properly deflates the score for constraints whose confidence
estimates are noisy (small napp, or conf near 0.5). Using FST as the
non-statistical baseline means that even the most principled deterministic
discriminative score threshold fails to control FDR, strengthening the
argument for our full statistical framework.

THRESHOLD SELECTION — FIXED CHI-SQUARED CRITICAL VALUE
-------------------------------------------------------
τ_F* = χ²_{1−α_F}(df=1) = 3.8415 at α_F = 0.05:

    Step 1. Compute F(c) for all c ∈ M_all on the original data.
    Step 2. Accept c if F(c) >= τ_F*.
    Step 3. S^FST_orig = {c : F(c) >= τ_F*}.
            R_obs^FST = |S^FST_orig| is freely data-determined.
            NOT forced to match K_P1_REJECTIONS.

Application to any log L':
    Recompute F_{L'}(c) from L's data for every c ∈ M_all.
    Accept c if F_{L'}(c) >= τ_F*.
    On null/perturbed logs |S^FST_{L'}| varies freely —
    this variation is what the FDR estimator measures.

WHY FST PROVABLY FAILS TO CONTROL FDR
--------------------------------------
Under σ∅ = σ_trace ∘ σ_label, both conf₀(c) and conf₁(c) are estimated
from shuffled traces with permuted labels.  Their difference is zero-mean
under the null, and F_null(c) ≈ chi-squared(1)/n_app (scaled).  For any
fixed positive τ_F*, some fraction of constraints exceed the threshold
purely by chance.  Unlike our method, FST has no Adaptive Storey π̂₀
correction and no doubly-null calibration of the null distribution per
constraint.  FDR_emp in RQ1 is uncontrolled: E[V_b]/R_obs has no
principled bound.  The denominator of F(c) provides some implicit variance
regularization — FST may fail less severely than raw |Δconf| threshold —
but it still fails without any formal guarantee.

DOUBLY-NULL FDR ESTIMATION (RQ1 / RQ2 INTEGRATION)
----------------------------------------------------
For each held-out null replicate b (σ_trace ∘ σ_label already applied):
    1. Recompute holds on null log.
    2. Recompute conf₀^(b)(c), conf₁^(b)(c), napp₀^(b)(c), napp₁^(b)(c).
    3. Recompute F_null^(b)(c) for every c ∈ M_all.
    4. Apply fixed τ_F*: V_b = #{c: F_null^(b)(c) >= τ_F*}.
    Fully deterministic — zero additional random state required.

INTERFACE COMPATIBILITY NOTES
------------------------------
The exported names retain the original DeclareMiner interface so that
rq1_Sepsis_parallel.py imports continue to work without change:
    run_declareminer               = run_fst
    run_declareminer_on_doubly_null_log = run_fst_on_doubly_null_log
    DM_CONFIG                      = FST_CONFIG (same keys, updated values)

The function apply_threshold_decision now takes (conf0, conf1, napp0,
napp1, tau_f_star) instead of (conf0, conf1, supp0, supp1, tau_delta_conf,
tau_min).  The rq2 run_dm_on_perturbed_log must be updated to:
    (a) store napp0_arr / napp1_arr as (m,) arrays when computing confidences,
    (b) call dm_apply_threshold(conf0, conf1, napp0_arr, napp1_arr,
        tau_f_star=tau_star).
compute_support_from_holds now returns (supp0, supp1, conf0, conf1,
napp0, napp1) — a 6-tuple instead of 4-tuple.

OUTPUT FILES
------------
    fst_results.json                All rules with Fisher scores and decisions.
    fst_significant_patterns.json  Accepted rules only.
    fst_report.txt                  Ranked text output.
    fst_score_distribution.csv      F(c) values for all m rules (paper figure).

Version : 3.0  (FST — Top-k with k=248; replaces DeclareMiner v2.0)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Di Francescomarino, Donadello, Ghidini, Maggi, Puura (2025). Business
    Process Deviance Mining with Sequential and Declarative Patterns.
    Bus Inf Syst Eng 67(6):877–894.
Gu, Li & Han (2011). Generalized Fisher Score for Feature Selection. UAI.
Welch (1947). The generalisation of student's problem. Biometrika 34:28–35.
Cecconi, Augusto & Di Ciccio (2021). BPM Forum 2021, LNBIP 427, pp.73–91.
"""

import os
import sys
import json
import time
import argparse
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ─── PATH SETUP ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from P1_SDSM.p1_Sepsis_hou import (
    load_and_preprocess_data,
    generate_candidate_patterns,
    split_by_class,
    compute_holds_by_case_batch,
    evaluate_pattern_fast,
    precompute_activity_index,
    compute_prevalence_from_holds,
    CaseInfo,
    INPUT_FILE        as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
    ALL_CONSTRAINT_TYPES,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CSV_PATH          = P1_INPUT_FILE
DECLARE_SPEC_FILE = P1_SPEC_FILE
OUTPUT_DIR        = "FST_Sepsis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── R_obs^P1 on Sepsis ICU — retained for gap reporting only ─────────────────
# This is the exact rejection count from p1_Sepsis_parallel.py v9.0-HOU-DOUBLY-NULL.
# NOT used to set τ_F* (threshold is now the chi-squared critical value).
K_P1_REJECTIONS: int = 248

# ── Fisher score denominator protection ───────────────────────────────────────
# Prevents division by zero when both class confidences are exactly 0 or 1
# (zero Bernoulli variance). Value chosen small enough to have negligible
# effect on scores with genuine variance (> 1/napp_y for any reasonable napp_y).
_FISHER_EPS: float = 1e-10

# ── Chi-squared threshold (fixed, not data-adaptive) ─────────────────────────
# F(c) ~ chi2(1) asymptotically under H0: conf0(c) = conf1(c).
# τ_F* = chi2.ppf(1 − α_F, df=1): the standard per-comparison critical value.
# At α_F = 0.05: τ_F* = 3.8415 — universally recognised significance boundary.
# This is NOT calibrated to match R_obs^P1; R_obs^FST is freely data-determined.
# Under the null, E[V_b] ≈ α_F × m — uncontrolled relative to R_obs^FST.
# The gap (K_P1_REJECTIONS − R_obs^FST) is a clean scientific finding: P1's
# structural axis and adaptive Storey correction recover constraints beyond FST.
ALPHA_F: float = 0.05
TAU_F_STAR: float = float(scipy_stats.chi2.ppf(1.0 - ALPHA_F, df=1))
# = 3.8414588206941285  (standard chi2(1) 95th percentile)

FST_CONFIG: dict = {
    # alpha_f: per-comparison significance level for chi2(1) threshold.
    'alpha_f':      ALPHA_F,
    # tau_f_star: fixed chi-squared critical value. NOT data-dependent.
    'tau_f_star':   TAU_F_STAR,
    # fisher_eps: denominator floor for F(c) computation.
    'fisher_eps':   _FISHER_EPS,
    # tau_min: retained for backward compatibility; not used in FST computation.
    'tau_min':      0.0,
    'random_state': 42,
}

# Backward-compatible alias: rq1 and rq2 import DM_CONFIG by name.
DM_CONFIG: dict = FST_CONFIG


# ─── SECTION 1: CONFIDENCE COMPUTATION ───────────────────────────────────────

def compute_confidences_from_case_data(
    case_data: Dict[str, 'CaseInfo'],
    candidates: List[Tuple],
    ids_class0: set,
    ids_class1: set,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Compute class-conditional confidence and support for every constraint
    in candidates by evaluating each trace directly.

    Used on the original log when holds are not yet precomputed.

    Returns
    -------
    supp0, supp1 : (m,) float64 — support = n_sat / n_total_class.
    conf0, conf1 : (m,) float64 — confidence = n_sat / napp (PRIMARY).
    napp0, napp1 : (m,) int64  — applicable (non-vacuous) case counts.
    """
    m  = len(candidates)
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    supp0 = np.zeros(m, dtype=np.float64)
    supp1 = np.zeros(m, dtype=np.float64)
    conf0 = np.zeros(m, dtype=np.float64)
    conf1 = np.zeros(m, dtype=np.float64)
    napp0 = np.zeros(m, dtype=np.int64)
    napp1 = np.zeros(m, dtype=np.int64)

    cases0 = [case_data[cid] for cid in ids_class0 if cid in case_data]
    cases1 = [case_data[cid] for cid in ids_class1 if cid in case_data]

    print(f"   Computing confidences for {m:,} constraints over "
          f"n0={n0:,} + n1={n1:,} cases...")

    for r_idx, (ct, a, b) in enumerate(candidates):
        if r_idx % 50 == 0:
            print(f"     [{r_idx:>5d}/{m:>5d}]", end='\r', flush=True)

        # Class 0
        nsat0_ = napp0_ = 0
        for case in cases0:
            result = evaluate_pattern_fast(ct, a, b, case.trace,
                                           case.activity_index)
            if result is not None:
                napp0_ += 1
                if result == 1:
                    nsat0_ += 1
        napp0[r_idx] = napp0_
        supp0[r_idx] = nsat0_ / n0      if n0     > 0 else 0.0
        conf0[r_idx] = nsat0_ / napp0_  if napp0_ > 0 else 0.0

        # Class 1
        nsat1_ = napp1_ = 0
        for case in cases1:
            result = evaluate_pattern_fast(ct, a, b, case.trace,
                                           case.activity_index)
            if result is not None:
                napp1_ += 1
                if result == 1:
                    nsat1_ += 1
        napp1[r_idx] = napp1_
        supp1[r_idx] = nsat1_ / n1      if n1     > 0 else 0.0
        conf1[r_idx] = nsat1_ / napp1_  if napp1_ > 0 else 0.0

    print(f"     [{m:>5d}/{m:>5d}]  done.")
    return supp0, supp1, conf0, conf1, napp0, napp1


def compute_support_from_holds(
    holds_all: Dict[Tuple, Dict[str, int]],
    candidates: List[Tuple],
    ids_class0: set,
    ids_class1: set,
    n0: int,
    n1: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Fast confidence, support, and applicable-count computation from a
    precomputed holds matrix.

    Used in the doubly-null replicate loop and on perturbed logs where
    holds_all is already available.

    Convention: holds_all[spec] maps case_id → {0, 1} for non-vacuous cases
    only. Vacuous cases are absent from the dict. n0/n1 are total case counts
    (including vacuous), used for the support denominator.

    Returns
    -------
    supp0, supp1 : (m,) float64 — support = n_sat / n_total_class.
    conf0, conf1 : (m,) float64 — confidence = n_sat / napp (PRIMARY).
    napp0, napp1 : (m,) int64  — applicable (non-vacuous) case counts.
                   Required for Fisher score computation.

    NOTE: This is a 6-tuple return. Callers that previously used the
    4-tuple (supp0, supp1, conf0, conf1) must be updated to unpack all six.
    """
    m = len(candidates)

    supp0 = np.zeros(m, dtype=np.float64)
    supp1 = np.zeros(m, dtype=np.float64)
    conf0 = np.zeros(m, dtype=np.float64)
    conf1 = np.zeros(m, dtype=np.float64)
    napp0 = np.zeros(m, dtype=np.int64)
    napp1 = np.zeros(m, dtype=np.int64)

    for r_idx, spec in enumerate(candidates):
        holds = holds_all.get(spec, {})

        nsat0_ = napp0_ = 0
        nsat1_ = napp1_ = 0

        for cid, val in holds.items():
            if cid in ids_class0:
                napp0_ += 1
                if val == 1:
                    nsat0_ += 1
            elif cid in ids_class1:
                napp1_ += 1
                if val == 1:
                    nsat1_ += 1

        napp0[r_idx] = napp0_
        napp1[r_idx] = napp1_
        supp0[r_idx] = nsat0_ / n0      if n0     > 0 else 0.0
        conf0[r_idx] = nsat0_ / napp0_  if napp0_ > 0 else 0.0
        supp1[r_idx] = nsat1_ / n1      if n1     > 0 else 0.0
        conf1[r_idx] = nsat1_ / napp1_  if napp1_ > 0 else 0.0

    return supp0, supp1, conf0, conf1, napp0, napp1


# ─── SECTION 2: FISHER SCORE COMPUTATION ─────────────────────────────────────

def compute_fisher_scores(
    conf0: np.ndarray,
    conf1: np.ndarray,
    napp0: np.ndarray,
    napp1: np.ndarray,
    eps: float = _FISHER_EPS,
) -> np.ndarray:
    """
    Compute the Fisher score for every constraint in M_all.

    Formula (squared Welch t-statistic for proportions):

        F(c) = (conf₁(c) − conf₀(c))²
               ─────────────────────────────────────────────────────
               conf₁(c)(1−conf₁(c))/napp₁(c) + conf₀(c)(1−conf₀(c))/napp₀(c)

    Denominator interpretation:
        The sampling variance of conf_y(c) is conf_y(1-conf_y)/napp_y —
        the Bernoulli variance of a proportion estimated from napp_y
        non-vacuous cases. Constraints with small napp_y get a large
        denominator contribution, correctly deflating F(c) for estimates
        based on little data.

    Edge-case handling:
        napp_y = 0: conf_y = 0 by convention; var contribution = 0.
                    (Constraint never activated in that class; no information
                    from it. The denominator uses only the other class.)
        numerator = 0 AND denominator = 0:
                    Both classes have identical confidence and zero variance
                    (e.g., both conf = 0 or both conf = 1). F(c) = 0.
        numerator > 0 AND denominator < eps:
                    Perfect or near-perfect discrimination with near-zero
                    variance (e.g., conf₀=0, conf₁=1 both from many cases).
                    F(c) = numerator / eps (very large, correctly dominant).

    Parameters
    ----------
    conf0, conf1 : (m,) float64 — class-conditional confidences.
    napp0, napp1 : (m,) int64  — applicable case counts per class.
    eps          : float — denominator floor (default: _FISHER_EPS = 1e-10).

    Returns
    -------
    fisher : (m,) float64 — Fisher score per constraint, >= 0.
    """
    numerator = (conf1 - conf0) ** 2

    # Bernoulli variance per class: conf_y(1-conf_y) / napp_y.
    # When napp_y = 0: contribution is 0.0 (no applicable cases → no variance).
    # np.maximum(napp_y, 1) in divisor is safe because the np.where masks napp=0.
    var0 = np.where(
        napp0 > 0,
        conf0 * (1.0 - conf0) / np.maximum(napp0, 1).astype(np.float64),
        0.0,
    )
    var1 = np.where(
        napp1 > 0,
        conf1 * (1.0 - conf1) / np.maximum(napp1, 1).astype(np.float64),
        0.0,
    )
    denominator = var0 + var1

    # Three cases:
    #   denom > eps:   normal division.
    #   denom <= eps and numerator <= eps: both near-zero → F = 0.
    #   denom <= eps and numerator > eps:  near-perfect discrimination → F = num/eps.
    fisher = np.where(
        denominator > eps,
        numerator / denominator,
        np.where(numerator <= eps, 0.0, numerator / eps),
    )
    return fisher


# ─── SECTION 3: CHI-SQUARED THRESHOLD ───────────────────────────────────────

def get_fst_threshold(alpha_f: float = ALPHA_F) -> float:
    """
    Return τ_F* = χ²_{1−α_F}(df=1): the chi-squared critical value at
    per-comparison significance level α_F.

    At α_F = 0.05: τ_F* = 3.8415.
    At α_F = 0.01: τ_F* = 6.6349.

    This is NOT data-dependent. It is the same value for every log,
    every perturbation level, and every null replicate.

    Under H0: conf0(c) = conf1(c), F(c) ~ chi2(1) asymptotically.
    Pr[F(c) >= τ_F*] = α_F per constraint.
    With m constraints tested simultaneously and no multiple-testing
    correction, E[V_b] ≈ α_F × m — uncontrolled relative to R_obs.
    """
    return float(scipy_stats.chi2.ppf(1.0 - alpha_f, df=1))


# ─── SECTION 4: FST DECISION RULE ────────────────────────────────────────────

def apply_fst_decision(
    fisher_scores: np.ndarray,
    tau_f_star: float,
) -> np.ndarray:
    """
    Accept constraint c if F(c) >= τ_F*.

    On the original data (where τ_F* was set by top-k), exactly k constraints
    satisfy this condition by construction.  On new logs (perturbed or null),
    |{c: F_new(c) >= τ_F*}| varies freely — this variation is what the FDR
    estimator and degradation metrics measure.

    Parameters
    ----------
    fisher_scores : (m,) float64 — Fisher scores computed on the target log.
    tau_f_star    : float — fixed chi-squared threshold from get_fst_threshold.

    Returns
    -------
    rejected : (m,) bool — True if constraint is accepted into S^FST.
    """
    return fisher_scores >= tau_f_star


def apply_threshold_decision(
    conf0: np.ndarray,
    conf1: np.ndarray,
    napp0: np.ndarray,
    napp1: np.ndarray,
    tau_f_star: float,
    fisher_eps: float = _FISHER_EPS,
    **_legacy_kwargs,
) -> np.ndarray:
    """
    FST decision rule expressed in terms of raw confidence and applicable
    count inputs.  This is the primary compatibility shim for callers
    (rq2's run_dm_on_perturbed_log) that compute conf0, conf1, napp0, napp1
    and then call dm_apply_threshold.

    Internally computes F(c) from conf0, conf1, napp0, napp1, then applies
    the threshold.

    Parameters
    ----------
    conf0, conf1 : (m,) float64 — class-conditional confidences.
    napp0, napp1 : (m,) int64  — applicable case counts.
    tau_f_star   : float — fixed chi-squared threshold (get_fst_threshold on original data).
                   (Previously named tau_delta_conf in DeclareMiner v2.0.)
    fisher_eps   : float — denominator floor for F(c).
    **_legacy_kwargs : Absorbs deprecated parameters from DeclareMiner v2.0
                       (tau_min, tau_delta_conf, tau_delta_supp, supp0, supp1).
                       All are silently ignored; FST does not use them.

    Returns
    -------
    rejected : (m,) bool — True if F(c) >= tau_f_star.

    NOTE FOR rq2 UPDATE
    -------------------
    The call in run_dm_on_perturbed_log must change from:
        dm_apply_threshold(conf0, conf1, supp0, supp1,
                           tau_delta_conf=tau_star, tau_min=tau_min)
    to:
        dm_apply_threshold(conf0, conf1, napp0_arr, napp1_arr,
                           tau_f_star=tau_star)
    where napp0_arr and napp1_arr are (m,) int arrays of applicable counts
    (already computed in the per-constraint loop in run_dm_on_perturbed_log).
    """
    fisher = compute_fisher_scores(conf0, conf1, napp0, napp1, eps=fisher_eps)
    return apply_fst_decision(fisher, tau_f_star)


# ─── SECTION 5: MAIN FST PIPELINE ────────────────────────────────────────────

def run_fst(
    config: Optional[dict] = None,
    case_data: Optional[Dict[str, 'CaseInfo']] = None,
    candidates_all: Optional[List[Tuple]] = None,
    R_obs_target: Optional[int] = None,
) -> dict:
    """
    Execute the full Fisher Score Threshold pipeline on Sepsis.

    The pipeline has five steps:
        0. Data loading (skipped if case_data is provided).
        1. Candidate pool M_all (reused from P1 if provided).
        2. Confidence and applicable-count computation.
        3. Fisher score computation for all m constraints.
        4. Chi-squared threshold: τ_F* = χ²_{1−α_F}(df=1). Fixed, not data-adaptive.
        5. Decision: S^FST = {c : F(c) >= τ_F*}.

    Parameters
    ----------
    config         : Optional config overrides (see FST_CONFIG).
    case_data      : Pre-loaded case data (skips reload for RQ1 integration).
    candidates_all : Fixed candidate pool M_all (shared with P1 and DRVA).
    R_obs_target   : Ignored in FST. Retained for signature compatibility.

    Returns
    -------
    dict with keys required by rq1 and rq2:
        results_all     : List[dict] — one record per constraint in M_all.
        rejected        : (m,) bool array — FST acceptance decisions.
        n_rejected      : int — |S^FST| = #{c: F(c) >= τ_F*}. Freely data-determined.
        tau_star        : float — τ_F* = χ²_{1−α_F}(df=1). Fixed.
        alpha_f         : float — per-comparison significance level.
        candidates_all  : List[Tuple] — M_all (passthrough).
        m_total         : int — |M_all|.
        conf0, conf1    : (m,) float64 — class-conditional confidences.
        napp0, napp1    : (m,) int64  — applicable case counts.
        supp0, supp1    : (m,) float64 — support values (diagnostic).
        fisher_scores   : (m,) float64 — F(c) for all constraints.
        score_df        : pd.DataFrame — F(c) distribution for paper figure.
        config          : dict — effective config.
        timing          : dict — wall times per step.
        case_data       : dict — passthrough for downstream use.
        ids_class0      : set — class-0 case IDs.
        ids_class1      : set — class-1 case IDs.
        n0, n1          : int — total class sizes.
    """
    cfg = {**FST_CONFIG, **(config or {})}
    k   = int(cfg.get('k', K_P1_REJECTIONS))
    eps = float(cfg.get('fisher_eps', _FISHER_EPS))

    timing   = {}
    t0_total = time.time()

    # ── Step 0: Load data ─────────────────────────────────────────────────
    if case_data is None:
        print("\n" + "=" * 100)
        print("FST — STEP 0: DATA LOADING")
        print("=" * 100)
        t0 = time.time()
        case_data = load_and_preprocess_data(CSV_PATH)
        timing['data_loading'] = time.time() - t0

    # ── Step 1: Candidate pool ────────────────────────────────────────────
    if candidates_all is None:
        print("\n" + "=" * 100)
        print("FST — STEP 1: CANDIDATE GENERATION FROM PHASE 0 SPEC")
        print("=" * 100)
        t0 = time.time()
        candidates_pos, candidates_neg = generate_candidate_patterns(case_data)
        pos_set = set(candidates_pos)
        candidates_all = list(candidates_pos) + [
            p for p in candidates_neg if p not in pos_set
        ]
        timing['candidate_generation'] = time.time() - t0

    m_total = len(candidates_all)

    print("\n" + "=" * 100)
    print("FST — Fisher Score Threshold Baseline (Di Francescomarino et al. 2025)")
    print(f"  k = K_P1_REJECTIONS = {k}  (Top-k selection; τ_F* = F(c_(k)))")
    print(f"  M_all = {m_total:,} constraints")
    print(f"  Fisher score: F(c) = (conf₁−conf₀)² / (σ₁²/napp₁ + σ₀²/napp₀)")
    print(f"  No statistical test.  No FDR correction.")
    print("=" * 100)

    # ── Step 2: Variant logs + confidence computation ─────────────────────
    print("\n" + "=" * 100)
    print("FST — STEP 2: CONFIDENCE AND APPLICABLE-COUNT COMPUTATION")
    print("=" * 100)

    D_0, D_1   = split_by_class(case_data)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    print(f"   L_0 (Normal,  class 0): {n0:,} traces")
    print(f"   L_1 (Deviant, class 1): {n1:,} traces")

    t0 = time.time()
    supp0, supp1, conf0, conf1, napp0, napp1 = compute_confidences_from_case_data(
        case_data, candidates_all, ids_class0, ids_class1
    )
    timing['confidence_computation'] = time.time() - t0

    delta_conf = np.abs(conf1 - conf0)
    print(f"\n   Δconf summary (all {m_total:,} constraints):")
    print(f"     mean={delta_conf.mean():.4f}  median={np.median(delta_conf):.4f}"
          f"  max={delta_conf.max():.4f}  min={delta_conf.min():.4f}")
    print(f"   napp₀ summary: mean={napp0.mean():.1f}  min={napp0.min()}  max={napp0.max()}"
          f"  n_zero={int((napp0 == 0).sum())}")
    print(f"   napp₁ summary: mean={napp1.mean():.1f}  min={napp1.min()}  max={napp1.max()}"
          f"  n_zero={int((napp1 == 0).sum())}")

    # ── Step 3: Fisher score computation ─────────────────────────────────
    print("\n" + "=" * 100)
    print("FST — STEP 3: FISHER SCORE COMPUTATION")
    print("=" * 100)
    print(f"   F(c) = (conf₁−conf₀)² / (conf₁(1−conf₁)/napp₁ + conf₀(1−conf₀)/napp₀)")
    print(f"   Denominator floor: eps = {eps:.2e}")

    t0 = time.time()
    fisher_scores = compute_fisher_scores(conf0, conf1, napp0, napp1, eps=eps)
    timing['fisher_computation'] = time.time() - t0

    n_zero_f = int((fisher_scores == 0.0).sum())
    f_max     = float(fisher_scores.max())
    f_median  = float(np.median(fisher_scores))
    f_mean    = float(fisher_scores.mean())

    print(f"\n   Fisher score summary (all {m_total:,} constraints):")
    print(f"     mean={f_mean:.4f}  median={f_median:.4f}"
          f"  max={f_max:.4f}  n_zero={n_zero_f}")
    print(f"     F >= 1.0  : {int((fisher_scores >= 1.0).sum()):,}")
    print(f"     F >= 5.0  : {int((fisher_scores >= 5.0).sum()):,}")
    print(f"     F >= 10.0 : {int((fisher_scores >= 10.0).sum()):,}")

    # ── Step 4: Chi-squared threshold (fixed, not data-adaptive) ─────────
    print("\n" + "=" * 100)
    print("FST — STEP 4: THRESHOLD  (fixed chi-squared critical value)")
    print("=" * 100)

    alpha_f    = float(cfg.get('alpha_f', ALPHA_F))
    tau_f_star = float(cfg.get('tau_f_star', get_fst_threshold(alpha_f)))

    n_above = int((fisher_scores >= tau_f_star).sum())
    n_below = m_total - n_above

    print(f"\n   α_F (per-comparison):  {alpha_f}")
    print(f"   τ_F* = χ²_{{1−α_F}}(1): {tau_f_star:.8f}")
    print(f"   Derivation: scipy.stats.chi2.ppf({1.0 - alpha_f}, df=1)")
    print(f"   |{{c: F(c) >= τ_F*}}|: {n_above}  (these enter S^FST)")
    print(f"   |{{c: F(c) <  τ_F*}}|: {n_below}  (excluded)")
    print(f"\n   Expected false positives under null:")
    print(f"     E[V_b] ≈ α_F × m = {alpha_f} × {m_total} = {alpha_f * m_total:.1f}")
    print(f"     FDR_emp = E[V_b] / R_obs ≈ "
          f"{alpha_f * m_total / max(n_above, 1):.4f}")
    print(f"     (uncontrolled — no multiple-testing correction applied)")

    # ── Step 5: Decision ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"FST — STEP 5: ACCEPTANCE DECISION  (F(c) >= {tau_f_star:.6f})")
    print("=" * 100)

    t0 = time.time()
    rejected_mask = apply_fst_decision(fisher_scores, tau_f_star)
    n_rejected    = int(rejected_mask.sum())
    timing['decision'] = time.time() - t0

    # n_rejected is now data-determined; it equals #{c: F(c) >= τ_F*}.
    # On original data this is NOT forced to match K_P1_REJECTIONS.
    print(f"\n   R_obs^FST = {n_rejected}  "
          f"(vs R_obs^P1 = {K_P1_REJECTIONS}; gap = {K_P1_REJECTIONS - n_rejected})")
    print(f"   Gap interpretation: P1's structural axis and higher power recover "
          f"{K_P1_REJECTIONS - n_rejected} additional constraints beyond FST's reach.")

    ct_counts = Counter(
        candidates_all[i][0] for i in range(m_total) if rejected_mask[i]
    )
    direction = np.where(conf1 >= conf0, "Positive", "Negative")

    for ct in ALL_CONSTRAINT_TYPES:
        if ct in ct_counts:
            print(f"     {ct:<30s}: {ct_counts[ct]:,}")
    n_pos = int((rejected_mask & (conf1 >= conf0)).sum())
    n_neg = n_rejected - n_pos
    print(f"\n   Direction: Positive (conf₁ > conf₀): {n_pos:,}  "
          f"Negative (conf₀ > conf₁): {n_neg:,}")

    # ── Assemble per-constraint result records ────────────────────────────
    results_all = []
    for r_idx, (ct, a, b) in enumerate(candidates_all):
        pid = f"{ct}_{a}" + (f"_{b}" if b else "")
        results_all.append({
            'pattern_id':       pid,
            'constraint_type':  ct,
            'activity_a':       a,
            'activity_b':       b,
            'direction':        direction[r_idx],
            'conf0':            float(conf0[r_idx]),
            'conf1':            float(conf1[r_idx]),
            'delta_conf':       float(delta_conf[r_idx]),
            'napp0':            int(napp0[r_idx]),
            'napp1':            int(napp1[r_idx]),
            'supp0':            float(supp0[r_idx]),
            'supp1':            float(supp1[r_idx]),
            'fisher_score':     float(fisher_scores[r_idx]),
            'is_significant':   bool(rejected_mask[r_idx]),
        })

    # Sort by Fisher score descending for output readability.
    results_all.sort(key=lambda x: (-x['fisher_score'], x['constraint_type'],
                                     x['activity_a'] or '', x['activity_b'] or ''))

    # Build F(c) distribution dataframe for paper figure.
    score_df = pd.DataFrame({
        'constraint':    [f"{c[0]}_{c[1]}{'_' + c[2] if c[2] else ''}"
                          for c in candidates_all],
        'fisher_score':  fisher_scores,
        'conf0':         conf0,
        'conf1':         conf1,
        'napp0':         napp0,
        'napp1':         napp1,
        'is_accepted':   rejected_mask,
    }).sort_values('fisher_score', ascending=False).reset_index(drop=True)

    timing['total'] = time.time() - t0_total

    print(f"\n   Timing summary:")
    for k_t, v in timing.items():
        print(f"     {k_t:30s}: {v:.2f}s")
    print(f"\n   FST pipeline complete.")
    print(f"     M_all = {m_total:,}  |  τ_F* = {tau_f_star:.8f}  "
          f"|  R_obs^FST = {n_rejected}  |  gap vs P1 = {K_P1_REJECTIONS - n_rejected}")

    return {
        'results_all':    results_all,
        'rejected':       rejected_mask,
        'n_rejected':     n_rejected,       # #{c: F(c) >= τ_F*} — freely data-determined
        'tau_star':       tau_f_star,        # fixed chi2(1) critical value
        'alpha_f':        alpha_f,           # per-comparison significance level
        'candidates_all': candidates_all,
        'm_total':        m_total,
        'conf0':          conf0,
        'conf1':          conf1,
        'napp0':          napp0,
        'napp1':          napp1,
        'supp0':          supp0,
        'supp1':          supp1,
        'fisher_scores':  fisher_scores,
        'score_df':       score_df,
        'config':         cfg,
        'timing':         timing,
        'case_data':      case_data,
        'ids_class0':     ids_class0,
        'ids_class1':     ids_class1,
        'n0':             n0,
        'n1':             n1,
    }


# ─── SECTION 6: OUTPUT GENERATION ────────────────────────────────────────────

def save_outputs(fst_out: dict) -> None:
    """
    Save JSON results, significant-only JSON, text report, and score CSV.

    Files written to OUTPUT_DIR:
        fst_results.json                Full per-constraint records.
        fst_significant_patterns.json   Accepted constraints only.
        fst_report.txt                  Ranked text output.
        fst_score_distribution.csv      F(c) values for all m constraints.
    """
    cfg         = fst_out['config']
    results_all = fst_out['results_all']
    m_total     = fst_out['m_total']
    n_rejected  = fst_out['n_rejected']
    tau_f_star  = fst_out['tau_star']
    alpha_f     = fst_out['alpha_f']
    timing      = fst_out['timing']
    score_df    = fst_out['score_df']
    case_data   = fst_out['case_data']
    n0          = fst_out['n0']
    n1          = fst_out['n1']

    sig_results = [r for r in results_all if r['is_significant']]
    n_pos = sum(1 for r in sig_results if r['direction'] == 'Positive')
    n_neg = sum(1 for r in sig_results if r['direction'] == 'Negative')
    gap   = K_P1_REJECTIONS - n_rejected

    # ── JSON (full) ───────────────────────────────────────────────────────
    full_json = {
        'framework': 'Fisher Score Threshold (FST) — Non-Statistical Baseline',
        'version':   '3.1',
        'timestamp': datetime.now().isoformat(),
        'grounding': (
            'Fisher score criterion from Di Francescomarino et al. (2025), '
            'Step 7.3 of the DvM pipeline (Gu et al. 2011 generalized Fisher '
            'score), applied directly to M_all without classifier training. '
            'Threshold: fixed chi-squared critical value τ_F* = χ²_{1−α_F}(df=1).'
        ),
        'description': {
            'criterion': (
                f'F(c) = (conf₁−conf₀)² / (conf₁(1−conf₁)/napp₁ + conf₀(1−conf₀)/napp₀). '
                f'Accept c if F(c) >= τ_F* = {tau_f_star:.6f}. '
                f'No statistical test. No FDR correction.'
            ),
            'threshold_rule': (
                f'Fixed chi-squared critical value: τ_F* = χ²_{{1−{alpha_f}}}(df=1) = {tau_f_star:.6f}. '
                f'Under H0: conf0=conf1, F(c) ~ chi2(1). '
                f'Pr[F(c) >= τ_F*] = {alpha_f} per constraint. '
                f'R_obs^FST = {n_rejected} is freely data-determined (NOT forced to match K_P1_REJECTIONS).'
            ),
            'why_fdr_fails': (
                'Under σ∅ = σ_trace ∘ σ_label, F_null(c) ~ chi2(1)/n_app (approx). '
                f'E[V_b] ≈ {alpha_f} × {m_total} = {alpha_f * m_total:.1f} false positives per rep. '
                'No Adaptive Storey π̂₀ correction and no empirical doubly-null '
                'calibration: FDR_emp = E[V_b]/R_obs is uncontrolled.'
            ),
            'gap_interpretation': (
                f'R_obs^FST = {n_rejected}  vs  R_obs^P1 = {K_P1_REJECTIONS}. '
                f'Gap = {gap}: P1\'s structural axis and adaptive Storey correction '
                f'recover {gap} constraints beyond FST\'s chi-squared threshold.'
            ),
        },
        'config': {k_c: (v if not isinstance(v, np.integer) else int(v))
                   for k_c, v in cfg.items()},
        'dataset': {
            'n_total':   len(case_data),
            'n_deviant': n1,
            'n_normal':  n0,
        },
        'summary': {
            'm_all':               m_total,
            'R_obs_fst':           n_rejected,
            'R_obs_p1':            K_P1_REJECTIONS,
            'gap_fst_vs_p1':       gap,
            'alpha_f':             alpha_f,
            'tau_f_star':          tau_f_star,
            'E_Vb_under_null':     alpha_f * m_total,
            'FDR_emp_predicted':   alpha_f * m_total / max(n_rejected, 1),
            'acceptance_rate':     n_rejected / max(m_total, 1),
            'n_accepted_positive': n_pos,
            'n_accepted_negative': n_neg,
        },
        'timing':             timing,
        'all_constraints':    results_all,
        'accepted_constraints': sig_results,
    }

    path_full = os.path.join(OUTPUT_DIR, 'fst_results.json')
    with open(path_full, 'w', encoding='utf-8') as fh:
        json.dump(full_json, fh, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON (full):          {path_full}")

    path_sig = os.path.join(OUTPUT_DIR, 'fst_significant_patterns.json')
    sig_json = {k_j: v for k_j, v in full_json.items()
                if k_j != 'all_constraints'}
    with open(path_sig, 'w', encoding='utf-8') as fh:
        json.dump(sig_json, fh, indent=2, ensure_ascii=False)
    print(f"✓ JSON (accepted):      {path_sig}")

    # ── Score distribution CSV ────────────────────────────────────────────
    path_csv = os.path.join(OUTPUT_DIR, 'fst_score_distribution.csv')
    score_df.to_csv(path_csv, index=False)
    print(f"✓ Score distribution:   {path_csv}")

    # ── Text report ───────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 120)
    lines.append("FISHER SCORE THRESHOLD (FST) BASELINE")
    lines.append("Sepsis ICU — Deviant (Return ER) vs. Normal (No Return ER)")
    lines.append("Grounded in: Di Francescomarino et al. (2025), Gu et al. (2011)")
    lines.append("=" * 120)
    lines.append(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"α_F = {alpha_f}  |  τ_F* = {tau_f_star:.6f}  |  M_all = {m_total:,}  |  R_obs^FST = {n_rejected}")
    lines.append("")
    lines.append("Criterion: F(c) = (conf₁−conf₀)² / (conf₁(1−conf₁)/napp₁ + conf₀(1−conf₀)/napp₀)")
    lines.append(f"Threshold: τ_F* = χ²_{{1−{alpha_f}}}(df=1) = {tau_f_star:.6f}  (fixed chi-squared critical value)")
    lines.append("No statistical test.  No FDR correction.")
    lines.append("")
    lines.append(f"Accepted:  {n_rejected:,} constraints  "
                 f"(Positive: {n_pos}, Negative: {n_neg})")
    lines.append("")
    lines.append("=" * 120)
    lines.append(f"TOP {min(50, n_rejected)} ACCEPTED CONSTRAINTS  "
                 f"(ranked by Fisher score descending)")
    lines.append("=" * 120)
    lines.append("")

    for rank, r in enumerate(sig_results[:50], 1):
        lines.append(f"Rank {rank:3d} | {r['pattern_id']}")
        lines.append(f"         F(c)       = {r['fisher_score']:.4f}")
        lines.append(f"         Δconf      = {r['delta_conf']:.4f}")
        lines.append(f"         conf_L0    = {r['conf0']:.4f}  "
                     f"conf_L1 = {r['conf1']:.4f}")
        lines.append(f"         napp_L0    = {r['napp0']:,}  "
                     f"napp_L1 = {r['napp1']:,}")
        lines.append(f"         Direction  = {r['direction']}")
        lines.append("")

    lines.append("=" * 120)
    lines.append("TIMING")
    lines.append("=" * 120)
    for k_t, v in timing.items():
        lines.append(f"  {k_t:30s}: {v:.2f}s")

    path_rpt = os.path.join(OUTPUT_DIR, 'fst_report.txt')
    with open(path_rpt, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"✓ Text report:          {path_rpt}")


# ─── SECTION 7: NULL REPLICATE RUNNER ────────────────────────────────────────

def run_fst_on_doubly_null_log(
    null_case_data: Dict[str, 'CaseInfo'],
    candidates_all: List[Tuple],
    tau_star: float,
    tau_min: float = 0.0,
    holds_all: Optional[Dict] = None,
    fisher_eps: float = _FISHER_EPS,
) -> int:
    """
    Apply the FST decision rule to a pre-built doubly-nullified log.

    Called from the RQ1 null-replicate loop.  Every acceptance here is a false
    positive by construction (σ_trace ∘ σ_label already applied to null_case_data).

    Protocol:
        1. If holds_all is provided, compute conf₀^(b), conf₁^(b), napp₀^(b),
           napp₁^(b) from it (fast path). Otherwise compute from null_case_data.
        2. Compute F_null^(b)(c) for all c ∈ M_all using the null confidences.
        3. V_b = #{c: F_null^(b)(c) >= tau_star}.
        Fully deterministic — no random state, no permutations, no iterations.

    Parameters
    ----------
    null_case_data : Dict[case_id -> CaseInfo] after double nullification.
    candidates_all : Fixed candidate pool M_all (shared across methods).
    tau_star       : τ_F* from the original-data FST run. Named tau_star
                     for backward compatibility with rq1 caller interface.
    tau_min        : Accepted but not used. Retained for rq1 compatibility.
    holds_all      : Precomputed holds on null log (fast path).
                     If None, holds are computed from null_case_data.
    fisher_eps     : Denominator floor for F_null computation.

    Returns
    -------
    n_rejected : int — V_b = number of false positives in this null replicate.
    """
    D_0, D_1   = split_by_class(null_case_data)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    if n0 == 0 or n1 == 0:
        # Degenerate split: no false positives possible (conservative).
        return 0

    if holds_all is not None:
        _, _, conf0_b, conf1_b, napp0_b, napp1_b = compute_support_from_holds(
            holds_all, candidates_all, ids_class0, ids_class1, n0, n1
        )
    else:
        _, _, conf0_b, conf1_b, napp0_b, napp1_b = compute_confidences_from_case_data(
            null_case_data, candidates_all, ids_class0, ids_class1
        )

    fisher_null = compute_fisher_scores(conf0_b, conf1_b, napp0_b, napp1_b,
                                        eps=fisher_eps)
    rejected_b  = apply_fst_decision(fisher_null, tau_f_star=tau_star)
    return int(rejected_b.sum())


# ─── BACKWARD-COMPATIBLE ALIASES ─────────────────────────────────────────────
# rq1_Sepsis_parallel.py and rq2_Sepsis_parallel.py import by the original names.
# These aliases ensure zero changes needed in rq1.
# rq2's run_dm_on_perturbed_log must update its call to apply_threshold_decision
# (see the NOTE in apply_threshold_decision's docstring).

run_declareminer = run_fst

run_declareminer_on_doubly_null_log = run_fst_on_doubly_null_log


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fisher Score Threshold (FST) Baseline — Sepsis ICU  "
            "(Di Francescomarino et al. 2025 / Gu et al. 2011)"
        )
    )
    parser.add_argument(
        '--alpha-f', type=float, default=ALPHA_F,
        help=f'Per-comparison significance level for chi2(1) threshold (default: {ALPHA_F})',
    )
    parser.add_argument(
        '--fisher-eps', type=float, default=_FISHER_EPS,
        help=f'Denominator floor for Fisher score (default: {_FISHER_EPS:.2e})',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Smoke test: run pipeline but skip saving large JSON files.',
    )
    args = parser.parse_args()

    config = {
        'alpha_f':      args.alpha_f,
        'tau_f_star':   get_fst_threshold(args.alpha_f),
        'fisher_eps':   args.fisher_eps,
        'tau_min':      0.0,
        'random_state': 42,
    }

    print("\n" + "=" * 100)
    print("FISHER SCORE THRESHOLD (FST) BASELINE")
    print("Sepsis ICU: Deviant (Return ER) vs. Normal (No Return ER)")
    print("Grounded in: Di Francescomarino et al. (2025); Gu et al. (2011)")
    print("=" * 100)
    print(f"  α_F = {config['alpha_f']}  →  τ_F* = χ²_{{1−α_F}}(df=1) = {config['tau_f_star']:.8f}")
    print(f"  Fisher eps = {config['fisher_eps']:.2e}")
    print(f"  K_P1_REJECTIONS = {K_P1_REJECTIONS}  (P1 result; used for gap reporting only)")
    print("  No statistical test.  No FDR correction.")
    print("=" * 100)

    fst_out = run_fst(config=config)

    if not args.dry_run:
        save_outputs(fst_out)

    print(f"\n{'='*100}")
    print("FST BASELINE COMPLETE")
    print(f"{'='*100}")
    print(f"  M_all:         {fst_out['m_total']:,}")
    print(f"  α_F:           {fst_out['alpha_f']}")
    print(f"  τ_F*:          {fst_out['tau_star']:.8f}  (fixed chi2(1) critical value)")
    print(f"  R_obs^FST:     {fst_out['n_rejected']:,}")
    print(f"  R_obs^P1:      {K_P1_REJECTIONS}  (gap = {K_P1_REJECTIONS - fst_out['n_rejected']})")
    print(f"  Total time:    {fst_out['timing']['total']:.1f}s")
    print(f"  Output:        {OUTPUT_DIR}/")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()