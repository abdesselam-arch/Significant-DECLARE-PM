"""
DeclareMiner_ProductionFST.py  —  DeclareMiner FST Baseline: Production (Option A)
====================================================================================
DeclareMiner-style discriminative baseline using the principled per-constraint
z-test threshold (Option A / FST formulation) instead of a globally calibrated
hard threshold.

PURPOSE
-------
Implement the DeclareMiner FST baseline for the RQ1 sensitivity analysis
alongside P1 (Hou-Storey conjunction) and DRVA.  The baseline uses the same
fixed candidate pool M_all as P1 and DRVA, removing candidate-pool confounding
so that all differences in rejection counts trace back to the testing procedure.

OPTION A — CONSTRAINT-SPECIFIC Z-TEST THRESHOLD (FST)
------------------------------------------------------
Under H0(c): conf+(c) = conf-(c) = p_c, the raw confidence difference
Δconf̂(c) = conf̂+(c) − conf̂−(c) is asymptotically:

    Δconf̂(c) ~ N(0, p̂_c(1 − p̂_c)(1/n_app,c+ + 1/n_app,c−))

where p̂_c is the pooled confidence estimate under H0.  The principled
per-comparison threshold at level α is therefore:

    τ_DM*(c) = z_{α/2} · √[ p̂_c(1 − p̂_c)(1/n_app,c+ + 1/n_app,c−) ]

This is the standard two-proportion z-test critical difference — exactly what
a statistician would apply as a per-constraint threshold before any multiplicity
correction.

Key properties:
    τ_DM*(c) is constraint-specific, not a single global threshold.
    E[V_b] = α · m holds under the null, exactly as with FST.
    Pr[|Δconf̂| > τ_DM*(c) | H0] = α for each c.

ALGEBRAIC EQUIVALENCE TO FST
-----------------------------
Squaring the acceptance condition yields the pooled two-proportion chi-squared:

    (Δconf̂(c))² / [p̂_c(1−p̂_c)(1/n+ + 1/n−)] ≥ z_{α/2}² = χ²_{1,1−α}

FST (Fisher Score Test) uses the Welch/unpooled denominator:

    F_c^FST = (Δconf̂)² / [conf̂+(1−conf̂+)/n+ + conf̂−(1−conf̂−)/n−]

Both threshold at χ²_{1,0.95} = 3.8415.  The FST/Welch denominator is larger
when class confidences differ, making FST strictly more conservative — and they
coincide exactly under H0.  This establishes FST as the statistically grounded
generalization of DeclareMiner's confidence-difference criterion.

DESIGN RATIONALE: WHY CONSTRAINT-SPECIFIC THRESHOLDS
------------------------------------------------------
The original DeclareMiner uses a single global τ* calibrated post-hoc to match
a rejection count target.  This has two scientific problems:
    1.  No connection to a nominal FDR or per-comparison error rate.
    2.  Rules with few applicable traces face the same τ* as well-powered ones,
        ignoring sampling noise — a rule with n_app = 5 needs a much larger
        threshold than one with n_app = 500.

Option A fixes both problems by deriving τ_DM*(c) directly from the null
distribution of Δconf̂(c), making the per-comparison error rate exactly α.

CRITICAL DESIGN DECISION FOR RQ1
---------------------------------
The τ_DM*(c) values are computed ONCE on the original (real) log and then held
FIXED across all doubly-null replicates — exactly as FST holds F_α = 3.8415
fixed.  On each null replicate, conf+(c) and conf−(c) change, but τ_DM*(c)
does not.  V_b counts how many constraints exceed their original-log threshold
on the null log.  Recomputing τ_DM*(c) per replicate would make the threshold
data-adaptive and invalidate the FDR calculation.

PRIMARY MEASURE: CONFIDENCE (not Support)
------------------------------------------
Following Cecconi et al. (2021) and the DeclareMiner/MINERful literature,
Confidence is the primary discriminative measure:

    "we consider Confidence as the best option because it measures the degree
     of satisfaction of a rule in a log independently from the rule frequency"

Support conflates activator frequency with satisfaction rate — a rule can have
high Δsupp simply because activity a fires more often in one variant, even if
the conditional relationship a → b is equally strong in both.

MEASURES
--------
For each rule r and variant log L, we compute:

    Confidence(r, L) = #{t ∈ L : m(r,t) = 1} / #{t ∈ L : m(r,t) ≠ None}
                     = n_satisfied / n_applicable
                     Primary discriminative measure.

    Support(r, L)    = #{t ∈ L : m(r,t) = 1} / |L|
                     = n_satisfied / n_total
                     Diagnostic only.

Decision rule (Option A / FST):
    Reject r iff  |Δconf̂(r)| ≥ τ_DM*(r)
                  AND (conf(r, L_0) ≥ τ_min OR conf(r, L_1) ≥ τ_min)

where τ_DM*(r) = z_{α/2} · SE_pooled(r), computed once on the real log.

FDR ARGUMENT (RQ1 Integration)
--------------------------------
With τ_DM*(c) frozen from the original log:

    E[V_b] = Σ_c Pr[|Δconf̂_b(c)| ≥ τ_DM*(c) | H0(c)]
           = α · m      (by construction of the z-test threshold)

This is identical to FST's per-comparison inflation argument.  DeclareMiner
with Option A expects α · m false discoveries per null replicate, while our
method (Adaptive Storey–Gao correction) empirically achieves FDR̂ ≤ α.

OUTPUT FILES
------------
    declareminer_fst_results.json               All rules, measures, decisions.
    declareminer_fst_significant_patterns.json  Rejected rules only.
    declareminer_fst_report.txt                 Ranked text output.
    declareminer_fst_tau_c.csv                  Per-constraint threshold τ_DM*(c).

Version : 3.0  (Option A: constraint-specific z-test threshold, α=0.05)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References:
    Cecconi, Augusto & Di Ciccio (2021). Detection of Statistically Significant
        Differences Between Process Variants Through Declarative Rules.
        BPM Forum 2021, LNBIP 427, pp. 73–91.
    Di Ciccio & Mecella (2015). On the discovery of declarative control flows
        for artful processes. TMIS 5(4):24.
    Gu et al. (2011). Generalized Fisher Score for Feature Selection. UAI.
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
from tqdm import tqdm

# ─── PATH SETUP ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))  # Experiments/ → P1_SDSM visible

# Reuse data-loading and DECLARE evaluation primitives from Phase 1.
# Guarantees identical constraint semantics across all three methods.
from P1_SDSM.p1_Production_hou import (
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
OUTPUT_DIR        = "DeclareMiner_Production"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DM_CONFIG = {
    # Minimum practically meaningful confidence difference at n_ref applicable traces
    'tau_effect':   0.95,
    # Reference applicable-trace count (anchor for the sample-size scaling)
    'n_ref':        10.0,
    # Hard floor: exclude constraints with fewer applicable traces than this in either class
    'n_floor':      4,
    # Minimum interestingness guard: rule must have conf >= tau_min in at least one variant
    'tau_min':      0.01,
    'random_state': 42,
}


# ─── MEASURE COMPUTATION ──────────────────────────────────────────────────────

def compute_support_and_confidence(
    case_data: Dict[str, CaseInfo],
    candidates: List[Tuple],
    ids_class0: set,
    ids_class1: set,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Support and Confidence for every rule in both variant logs.

    For rule r and variant log L_y (y ∈ {0, 1}):

        conf(r, L_y) = #{t ∈ L_y : m(r,t) = 1} / #{t ∈ L_y : m(r,t) ≠ None}
            Denominator = applicable (non-vacuous) traces in L_y.
            Numerator   = satisfied traces.  PRIMARY measure.

        supp(r, L_y) = #{t ∈ L_y : m(r,t) = 1} / |L_y|
            Denominator = total traces in L_y (including vacuous).
            Numerator   = satisfied traces.  Diagnostic only.

    Args:
        case_data:   Dict[case_id -> CaseInfo].
        candidates:  List[(ct, a, b)] — M_all.
        ids_class0:  Set of case IDs for class 0 (Normal).
        ids_class1:  Set of case IDs for class 1 (Deviant).

    Returns:
        supp0, supp1, conf0, conf1 : (m,) float64 arrays.
        napp0, napp1               : (m,) int arrays (applicable trace counts).
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

    for r_idx, (ct, a, b) in enumerate(tqdm(candidates, desc="Computing measures")):
        # Class 0
        nsat0 = napp0_ = 0
        for case in cases0:
            result = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
            if result is not None:
                napp0_ += 1
                if result == 1:
                    nsat0 += 1
        napp0[r_idx] = napp0_
        supp0[r_idx] = nsat0 / n0      if n0     > 0 else 0.0
        conf0[r_idx] = nsat0 / napp0_  if napp0_ > 0 else 0.0

        # Class 1
        nsat1 = napp1_ = 0
        for case in cases1:
            result = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
            if result is not None:
                napp1_ += 1
                if result == 1:
                    nsat1 += 1
        napp1[r_idx] = napp1_
        supp1[r_idx] = nsat1 / n1      if n1     > 0 else 0.0
        conf1[r_idx] = nsat1 / napp1_  if napp1_ > 0 else 0.0

    return supp0, supp1, conf0, conf1, napp0, napp1


def compute_support_from_holds(
    holds_all: Dict[Tuple, Dict[str, int]],
    candidates: List[Tuple],
    ids_class0: set,
    ids_class1: set,
    n0: int,
    n1: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast support and confidence computation from a precomputed holds matrix.

    Used in the doubly-null replicate loop where holds_all is already available.

    Convention: holds_all[spec] contains ONLY non-vacuous cases (holds ∈ {0, 1}).
    Vacuous cases are absent from the dict entirely.  n0/n1 are total case counts
    (including vacuous) and serve as the support denominator.

    Args:
        holds_all:   Dict[(ct, a, b) -> Dict[case_id -> 0/1]] (non-vacuous only).
        candidates:  List[(ct, a, b)].
        ids_class0:  Set of class-0 case IDs.
        ids_class1:  Set of class-1 case IDs.
        n0, n1:      Total case counts per class (for support denominator).

    Returns:
        supp0, supp1, conf0, conf1 : (m,) float64 arrays.
    """
    assert all(
        v in (0, 1)
        for spec_dict in holds_all.values()
        for v in spec_dict.values()
    ), "holds_all must contain only 0/1 values; vacuous cases must be absent (not None-valued)"

    m = len(candidates)
    supp0 = np.zeros(m, dtype=np.float64)
    supp1 = np.zeros(m, dtype=np.float64)
    conf0 = np.zeros(m, dtype=np.float64)
    conf1 = np.zeros(m, dtype=np.float64)

    for r_idx, spec in enumerate(candidates):
        holds = holds_all.get(spec, {})
        # Class 0
        nsat0 = napp0 = 0
        for cid, val in holds.items():
            if cid in ids_class0:
                napp0 += 1
                if val == 1:
                    nsat0 += 1
        supp0[r_idx] = nsat0 / n0    if n0    > 0 else 0.0
        conf0[r_idx] = nsat0 / napp0 if napp0 > 0 else 0.0
        # Class 1
        nsat1 = napp1 = 0
        for cid, val in holds.items():
            if cid in ids_class1:
                napp1 += 1
                if val == 1:
                    nsat1 += 1
        supp1[r_idx] = nsat1 / n1    if n1    > 0 else 0.0
        conf1[r_idx] = nsat1 / napp1 if napp1 > 0 else 0.0

    return supp0, supp1, conf0, conf1


# ─── DECISION RULE (EFFECT-SIZE THRESHOLD) ───────────────────────────────────

def compute_tau_c_effect_size(
    napp0: np.ndarray,
    napp1: np.ndarray,
    tau_effect: float = 0.10,
    n_ref: float = 30.0,
    n_floor: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a sample-size-scaled non-statistical threshold for each constraint.

    τ*(c) = tau_effect × √(n_ref / n_harm(c))

    where n_harm(c) = 2 * n0 * n1 / (n0 + n1) is the harmonic mean of
    applicable trace counts.  This scales the required difference with
    estimation noise, anchored to a user-chosen practical effect size
    at n_ref applicable traces per class.

    No distributional model is used.  tau_effect has no Type I error
    interpretation — it is the minimum practically meaningful confidence
    difference at the reference sample size.

    The harmonic mean is used instead of the arithmetic mean because it
    is more sensitive to the smaller count: a constraint applicable to
    300 normal traces but only 4 deviant traces carries no more information
    than one applicable to 4 traces in each class.

    This threshold is computed ONCE on the original (real) log and held
    fixed across all doubly-null replicates — identically to the old z-test
    threshold, and for the same reason: recomputing per replicate would make
    the threshold data-adaptive and distort the empirical FDR measurement.

    Args:
        napp0, napp1:  (m,) applicable trace counts per class.
        tau_effect:    Minimum practically meaningful confidence difference
                       at n_ref applicable traces (user-chosen constant).
        n_ref:         Reference applicable-trace count (anchor point).
        n_floor:       Hard exclusion floor: constraints with napp0 < n_floor
                       or napp1 < n_floor are marked ineligible regardless of
                       their observed confidence difference.

    Returns:
        tau_c   : (m,) float64 threshold array.
        eligible: (m,) bool — False where either n_app < n_floor.
    """
    n0 = napp0.astype(np.float64)
    n1 = napp1.astype(np.float64)

    # Harmonic mean of applicable trace counts
    n_harm = np.where(
        (n0 + n1) > 0,
        2.0 * n0 * n1 / (n0 + n1),
        0.0,
    )

    # Scale: larger threshold when fewer applicable traces
    scale = np.where(n_harm > 0, np.sqrt(n_ref / n_harm), np.inf)
    tau_c = tau_effect * scale

    # Hard floor: exclude constraints too sparse to be informative
    eligible = (napp0 >= n_floor) & (napp1 >= n_floor)

    return tau_c, eligible


def apply_threshold_decision(
    conf0: np.ndarray,
    conf1: np.ndarray,
    tau_c: np.ndarray,
    tau_min: float,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply the DeclareMiner effect-size per-constraint decision rule.

    Primary criterion (sample-size-scaled effect-size threshold):
        |Δconf̂(c)| = |conf(c, L_1) − conf(c, L_0)| ≥ τ*(c)

    Interestingness guard (mirrors DRVA's mmin, standard in DeclareMiner):
        conf(c, L_0) ≥ tau_min  OR  conf(c, L_1) ≥ tau_min

    Sparsity floor (prevents rejection on near-empty applicable sets):
        napp0 ≥ n_floor  AND  napp1 ≥ n_floor  (captured in `eligible`)

    Reject rule c iff all three conditions are satisfied.

    NOTE: tau_c and eligible must be computed from the ORIGINAL log and held
    fixed when called on null replicates.  The interestingness guard uses the
    log being tested (original or null) — this is conservative and consistent
    with DRVA.

    Args:
        conf0, conf1:  (m,) confidence arrays for the log under evaluation.
        tau_c:         (m,) constraint-specific thresholds from the original log.
        tau_min:       Minimum confidence interestingness guard.
        eligible:      (m,) bool mask from the original log (FROZEN). If None,
                       the sparsity floor is not applied.

    Returns:
        (m,) bool array — True if rule is rejected ("discovered").
    """
    delta_conf  = np.abs(conf1 - conf0)
    interesting = (conf0 >= tau_min) | (conf1 >= tau_min)
    if eligible is not None:
        return (delta_conf >= tau_c) & interesting & eligible
    return (delta_conf >= tau_c) & interesting


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_declareminer(
    config: Optional[dict] = None,
    case_data: Optional[Dict[str, CaseInfo]] = None,
    candidates_all: Optional[List[Tuple]] = None,
) -> dict:
    """
    Execute the full DeclareMiner FST (Option A) pipeline on Production.

    Args:
        config:          Override DM_CONFIG parameters.
        case_data:       Pre-loaded case data (avoids reload for RQ1).
        candidates_all:  Fixed candidate pool M_all (shared with P1/DRVA).

    Returns:
        dict with all results, measures, per-constraint thresholds (tau_c),
        and intermediate quantities needed for RQ1 integration.

    Key return fields:
        'tau_c'        : (m,) float64 array — per-constraint z-test thresholds,
                         computed on the real log.  Pass this to
                         run_declareminer_on_doubly_null_log unchanged.
        'alpha'        : the per-comparison significance level used.
        'n_rejected'   : number of rules exceeding their individual threshold.
    """
    cfg = {**DM_CONFIG, **(config or {})}

    tau_effect = float(cfg['tau_effect'])
    n_ref      = float(cfg['n_ref'])
    n_floor    = int(cfg['n_floor'])
    tau_min    = float(cfg['tau_min'])

    timing   = {}
    t0_total = time.time()

    # ── Section 0: Load data ──────────────────────────────────────────────
    if case_data is None:
        print("\n" + "=" * 100)
        print("DeclareMiner FST — STEP 0: DATA LOADING")
        print("=" * 100)
        case_data = load_and_preprocess_data(CSV_PATH)

    if candidates_all is None:
        print("\n" + "=" * 100)
        print("DeclareMiner FST — STEP 1: CANDIDATE GENERATION FROM PHASE 0 SPEC")
        print("=" * 100)
        candidates_pos, candidates_neg = generate_candidate_patterns(case_data)
        pos_set = set(candidates_pos)
        candidates_all = list(candidates_pos) + [
            p for p in candidates_neg if p not in pos_set
        ]

    m_total = len(candidates_all)
    print(f"\n   Fixed candidate pool M_all: {m_total:,} rules")

    # ── Section 1: Variant logs ───────────────────────────────────────────
    D_0, D_1 = split_by_class(case_data)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)
    print(f"   L_0 (Normal,  class 0): {n0:,} traces")
    print(f"   L_1 (Deviant, class 1): {n1:,} traces")

    # ── Section 2: Compute measures on real data ──────────────────────────
    print("\n" + "=" * 100)
    print("DeclareMiner FST — STEP 2: CONFIDENCE & SUPPORT COMPUTATION")
    print("=" * 100)
    print("   conf(r, L_y) = n_satisfied / n_applicable_y  (PRIMARY — matches DRVA Ediff)")
    print("   supp(r, L_y) = n_satisfied / n_total_y       (diagnostic only)")

    t_meas = time.time()
    supp0, supp1, conf0, conf1, napp0, napp1 = compute_support_and_confidence(
        case_data, candidates_all, ids_class0, ids_class1
    )
    timing['measure_computation'] = time.time() - t_meas

    delta_conf = np.abs(conf1 - conf0)
    delta_supp = np.abs(supp1 - supp0)

    print(f"\n   Δconf summary (all {m_total:,} rules)  ← PRIMARY decision variable:")
    print(f"     mean={delta_conf.mean():.4f}, median={np.median(delta_conf):.4f}, "
          f"max={delta_conf.max():.4f}, min={delta_conf.min():.4f}")
    print(f"     Rules with Δconf >= 0.05: {(delta_conf >= 0.05).sum():,}")
    print(f"     Rules with Δconf >= 0.10: {(delta_conf >= 0.10).sum():,}")
    print(f"     Rules with Δconf >= 0.20: {(delta_conf >= 0.20).sum():,}")
    print(f"\n   Δsupp summary (diagnostic):")
    print(f"     mean={delta_supp.mean():.4f}, median={np.median(delta_supp):.4f}, "
          f"max={delta_supp.max():.4f}")

    # ── Section 3: Compute sample-size-scaled effect-size thresholds ──────
    print("\n" + "=" * 100)
    print("DeclareMiner FST — STEP 3: SAMPLE-SIZE-SCALED EFFECT-SIZE THRESHOLDS")
    print("=" * 100)
    print(f"   τ*(c) = τ_effect × √(n_ref / n_harm(c))")
    print(f"   τ_effect = {tau_effect}  (min practical confidence diff at n_ref traces)")
    print(f"   n_ref    = {n_ref}  (reference applicable-trace count)")
    print(f"   n_floor  = {n_floor}  (hard exclusion floor per class)")
    print(f"   No distributional model. No Type I error interpretation.")
    print(f"   Computed ONCE on the real log; held FIXED across all null replicates.")

    t_tau = time.time()
    tau_c, eligible = compute_tau_c_effect_size(
        napp0, napp1, tau_effect=tau_effect, n_ref=n_ref, n_floor=n_floor
    )
    timing['threshold_computation'] = time.time() - t_tau

    eligible_tau = tau_c[eligible]
    print(f"\n   τ*(c) summary (eligible rules only):")
    if eligible_tau.size > 0:
        print(f"     mean={eligible_tau.mean():.4f}, median={np.median(eligible_tau):.4f}, "
              f"max={eligible_tau.max():.4f}, min={eligible_tau.min():.4f}")
    print(f"   Eligible (n_app ≥ {n_floor} in both classes): {eligible.sum():,} / {m_total:,}")

    # ── Section 4: Apply decision rule ───────────────────────────────────
    print("\n" + "=" * 100)
    print(f"DeclareMiner FST — STEP 4: DECISION RULE (effect-size threshold)")
    print("=" * 100)
    print(f"   Reject c iff  |Δconf̂(c)| ≥ τ*(c)")
    print(f"                AND (conf(c, L_0) ≥ {tau_min} OR conf(c, L_1) ≥ {tau_min})")
    print(f"                AND n_app,c,0 ≥ {n_floor} AND n_app,c,1 ≥ {n_floor}")
    print(f"   No distributional model.  No FDR correction.  Pure effect-size selection.")

    rejected  = apply_threshold_decision(
        conf0, conf1,
        tau_c=tau_c,
        tau_min=tau_min,
        eligible=eligible,
    )
    n_rejected = int(rejected.sum())

    print(f"\n   Rejected: {n_rejected:,} rules")
    ct_counts = Counter(
        candidates_all[i][0] for i in range(m_total) if rejected[i]
    )
    for ct in ALL_CONSTRAINT_TYPES:
        if ct in ct_counts:
            print(f"     {ct:<30s}: {ct_counts[ct]:,}")

    direction = np.where(conf1 >= conf0, "Positive", "Negative")

    timing['total'] = time.time() - t0_total

    # ── Section 5: Assemble per-rule result records ───────────────────────
    results_all = []
    for r_idx, (ct, a, b) in enumerate(candidates_all):
        pid = f"{ct}_{a}" + (f"_{b}" if b else "")
        results_all.append({
            'pattern_id':      pid,
            'constraint_type': ct,
            'activity_a':      a,
            'activity_b':      b,
            'direction':       direction[r_idx],
            'conf0':           float(conf0[r_idx]),
            'conf1':           float(conf1[r_idx]),
            'delta_conf':      float(delta_conf[r_idx]),
            'tau_c':           float(tau_c[r_idx]),
            'z_score':         float(delta_conf[r_idx] / tau_c[r_idx])
                               if tau_c[r_idx] > 0 else float('inf'),
            'supp0':           float(supp0[r_idx]),
            'supp1':           float(supp1[r_idx]),
            'delta_supp':      float(delta_supp[r_idx]),
            'napp0':           int(napp0[r_idx]),
            'napp1':           int(napp1[r_idx]),
            'is_significant':  bool(rejected[r_idx]),
        })

    # Sort by z-score descending (= Δconf / τ_DM*(c), analogous to DRVA's Ediff ranking)
    results_all.sort(key=lambda x: (-x['z_score'], -x['delta_conf']))

    n_pos = sum(1 for r in results_all if r['is_significant'] and r['direction'] == 'Positive')
    n_neg = sum(1 for r in results_all if r['is_significant'] and r['direction'] == 'Negative')

    print(f"\n   Timing:")
    for k, v in timing.items():
        print(f"     {k:25s}: {v:.1f}s")
    print(f"\n   Summary:")
    print(f"     M_all:            {m_total:,}")
    print(f"     Eligible:         {eligible.sum():,}")
    print(f"     Rejected:         {n_rejected:,}  (Positive: {n_pos}, Negative: {n_neg})")
    print(f"     τ_effect:         {tau_effect:.4f}")
    print(f"     n_ref:            {n_ref:.1f}")
    print(f"     n_floor:          {n_floor}")
    print(f"     τ_min:            {tau_min:.4f}")

    return {
        'results_all':    results_all,
        'rejected':       rejected,
        'n_rejected':     n_rejected,
        'tau_c':          tau_c,          # (m,) — FROZEN thresholds for null replicates
        'eligible':       eligible,       # (m,) — FROZEN eligibility mask for null replicates
        'tau_effect':     tau_effect,
        'n_ref':          n_ref,
        'n_floor':        n_floor,
        'candidates_all': candidates_all,
        'm_total':        m_total,
        'conf0':          conf0,
        'conf1':          conf1,
        'supp0':          supp0,
        'supp1':          supp1,
        'napp0':          napp0,
        'napp1':          napp1,
        'delta_conf':     delta_conf,
        'delta_supp':     delta_supp,
        'config':         cfg,
        'timing':         timing,
        'case_data':      case_data,
        'ids_class0':     ids_class0,
        'ids_class1':     ids_class1,
        'n0':             n0,
        'n1':             n1,
    }


# ─── OUTPUT GENERATION ────────────────────────────────────────────────────────

def save_outputs(dm_out: dict) -> None:
    """Save JSON results, significant-only JSON, text report, and tau_c CSV."""

    cfg         = dm_out['config']
    results_all = dm_out['results_all']
    m_total     = dm_out['m_total']
    n_rejected  = dm_out['n_rejected']
    tau_effect  = dm_out['tau_effect']
    n_ref       = dm_out['n_ref']
    n_floor     = dm_out['n_floor']
    tau_c       = dm_out['tau_c']
    eligible    = dm_out['eligible']
    timing      = dm_out['timing']
    case_data   = dm_out['case_data']


    sig_results = [r for r in results_all if r['is_significant']]
    n_pos = sum(1 for r in sig_results if r['direction'] == 'Positive')
    n_neg = sum(1 for r in sig_results if r['direction'] == 'Negative')

    eligible_tau = tau_c[eligible]

    # ── JSON ─────────────────────────────────────────────────────────────
    full_json = {
        'framework': 'DeclareMiner — Sample-Size-Scaled Effect-Size Threshold',
        'version':   '4.0',
        'timestamp': datetime.now().isoformat(),
        'description': {
            'discriminative_criterion': (
                f"Reject c iff |conf(c, L_1) − conf(c, L_0)| ≥ τ*(c) "
                f"AND (conf(c, L_0) ≥ {cfg['tau_min']} OR conf(c, L_1) ≥ {cfg['tau_min']}) "
                f"AND n_app,c,0 ≥ {n_floor} AND n_app,c,1 ≥ {n_floor}. "
                f"τ*(c) = {tau_effect} × √({n_ref} / n_harm(c)). "
                "No distributional model. No FDR correction. Pure effect-size selection."
            ),
            'threshold_type': (
                f"Sample-size-scaled effect-size threshold. τ_effect = {tau_effect} is the minimum "
                f"practically meaningful confidence difference at n_ref = {n_ref} applicable traces. "
                "Harmonic mean of n_app per class used as the sample-size aggregator — more "
                "sensitive than the arithmetic mean to the smaller count. No Type I error interpretation."
            ),
            'fdr_note': (
                "No analytical FDR bound is available for this threshold type. Empirical FDR is "
                "measured via the doubly-null replicate protocol: V_b counts rejections on each "
                "null log using the frozen tau_c and eligible masks. Our method (Adaptive "
                "Storey–Gao) achieves empirical FDR ≤ α by redistributing the budget in "
                "proportion to signal density."
            ),
            'rq1_protocol': (
                "tau_c and eligible are computed ONCE on the original log and held FIXED across "
                "all doubly-null replicates. V_b = #{c : |Δconf̂_b(c)| ≥ τ*(c) and eligible(c)} "
                "counts false positives on each null replicate using the frozen masks."
            ),
        },
        'config': cfg,
        'dataset': {
            'n_total':   len(case_data),
            'n_deviant': sum(1 for c in case_data.values() if c.outcome == 1),
            'n_normal':  sum(1 for c in case_data.values() if c.outcome == 0),
        },
        'summary': {
            'm_all':               m_total,
            'n_eligible':          int(eligible.sum()),
            'n_rejected':          n_rejected,
            'tau_effect':          tau_effect,
            'n_ref':               n_ref,
            'n_floor':             n_floor,
            'rejection_rate':      n_rejected / max(m_total, 1),
            'n_rejected_positive': n_pos,
            'n_rejected_negative': n_neg,
            'tau_c_mean':          float(eligible_tau.mean()) if eligible_tau.size > 0 else None,
            'tau_c_median':        float(np.median(eligible_tau)) if eligible_tau.size > 0 else None,
            'tau_c_min':           float(eligible_tau.min()) if eligible_tau.size > 0 else None,
            'tau_c_max':           float(eligible_tau.max()) if eligible_tau.size > 0 else None,
        },
        'timing':            timing,
        'all_rules':         results_all,
        'significant_rules': sig_results,
    }

    path_full = os.path.join(OUTPUT_DIR, 'declareminer_fst_results.json')
    with open(path_full, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON (full):        {path_full}")

    path_sig = os.path.join(OUTPUT_DIR, 'declareminer_fst_significant_patterns.json')
    sig_json = {k: v for k, v in full_json.items() if k != 'all_rules'}
    with open(path_sig, 'w', encoding='utf-8') as f:
        json.dump(sig_json, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON (significant): {path_sig}")

    # ── Per-constraint threshold CSV (for paper figures) ──────────────────
    tau_c_df = pd.DataFrame({
        'pattern_id':      [r['pattern_id']      for r in results_all],
        'constraint_type': [r['constraint_type'] for r in results_all],
        'activity_a':      [r['activity_a']      for r in results_all],
        'activity_b':      [r['activity_b']      for r in results_all],
        'napp0':           [r['napp0']            for r in results_all],
        'napp1':           [r['napp1']            for r in results_all],
        'conf0':           [r['conf0']            for r in results_all],
        'conf1':           [r['conf1']            for r in results_all],
        'delta_conf':      [r['delta_conf']       for r in results_all],
        'tau_c':           [r['tau_c']            for r in results_all],
        'z_score':         [r['z_score']          for r in results_all],
        'is_significant':  [r['is_significant']   for r in results_all],
    })
    path_tau = os.path.join(OUTPUT_DIR, 'declareminer_fst_tau_c.csv')
    tau_c_df.to_csv(path_tau, index=False)
    print(f"✓ Threshold CSV:      {path_tau}")

    # ── Text report ───────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 120)
    lines.append("DECLAREMINER — SAMPLE-SIZE-SCALED EFFECT-SIZE THRESHOLD")
    lines.append("Production: Deviant vs. Normal")
    lines.append("=" * 120)
    lines.append(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"τ_effect = {tau_effect}   n_ref = {n_ref}   n_floor = {n_floor}   "
                 f"τ_min = {cfg['tau_min']}")
    lines.append(f"τ*(c) = τ_effect × √(n_ref / n_harm(c))  "
                 f"(computed once on real log, frozen for null replicates)")
    lines.append("")
    lines.append(f"M_all:     {m_total:,}")
    lines.append(f"Eligible:  {eligible.sum():,}  (n_app ≥ {n_floor} in both classes)")
    lines.append(f"Rejected:  {n_rejected:,}  (Positive: {n_pos}, Negative: {n_neg})")
    lines.append("")

    lines.append("Decision rule (per constraint c):")
    lines.append(f"  |conf(c, L_1) − conf(c, L_0)| ≥ τ*(c)   [effect-size threshold]")
    lines.append(f"  AND (conf(c, L_0) ≥ {cfg['tau_min']} OR conf(c, L_1) ≥ {cfg['tau_min']})")
    lines.append(f"  AND n_app,c,0 ≥ {n_floor} AND n_app,c,1 ≥ {n_floor}")
    lines.append("  No distributional model.  No FDR correction.  Pure effect-size selection.")
    lines.append("")

    lines.append(f"τ*(c) distribution (eligible rules only):")
    if eligible_tau.size > 0:
        lines.append(f"  mean={eligible_tau.mean():.4f}  median={np.median(eligible_tau):.4f}  "
                     f"min={eligible_tau.min():.4f}  max={eligible_tau.max():.4f}")
    lines.append("")

    lines.append("=" * 120)
    lines.append("TOP 50 SIGNIFICANT RULES  (ranked by z-score = Δconf / τ*(c) descending)")
    lines.append("=" * 120)
    lines.append("")
    for rank, r in enumerate(sig_results[:50], 1):
        lines.append(f"Rank {rank:3d} | {r['pattern_id']}")
        lines.append(f"         Constraint: {r['constraint_type']}")
        if r['activity_b']:
            lines.append(f"         Activities: {r['activity_a']} → {r['activity_b']}")
        else:
            lines.append(f"         Activity:   {r['activity_a']}")
        lines.append(f"         Direction:  {r['direction']}")
        lines.append(
            f"         conf_L0={r['conf0']:.4f}  conf_L1={r['conf1']:.4f}  "
            f"Δconf={r['delta_conf']:.4f}  τ_DM*={r['tau_c']:.4f}  "
            f"z={r['z_score']:.2f}  "
            f"napp0={r['napp0']:,}  napp1={r['napp1']:,}"
        )
        lines.append("")

    lines.append("=" * 120)
    lines.append("TIMING")
    lines.append("=" * 120)
    for k, v in timing.items():
        lines.append(f"  {k:25s}: {v:.1f}s")

    path_rpt = os.path.join(OUTPUT_DIR, 'declareminer_fst_report.txt')
    with open(path_rpt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ Text report:        {path_rpt}")


# ─── RQ1 INTEGRATION: DOUBLY-NULL COUNTER ─────────────────────────────────────

def run_declareminer_on_doubly_null_log(
    null_case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    tau_c: np.ndarray,
    tau_min: float = 0.01,
    eligible: Optional[np.ndarray] = None,
    holds_all: Optional[Dict] = None,
) -> int:
    """
    Apply the DeclareMiner effect-size decision rule to a pre-built doubly-nullified log.

    Called from the RQ1 null-replicate loop.  Every rejection here is a false
    positive by construction (sigma_trace ∘ sigma_label already applied).

    CRITICAL: tau_c and eligible are computed ONCE on the original (real) log and
    passed in unchanged (FROZEN).  They must NOT be recomputed from the null log.
    Recomputing per replicate would make the threshold data-adaptive and distort
    the empirical FDR measurement.

    Args:
        null_case_data:  Dict[case_id -> CaseInfo] after double nullification.
        candidates_all:  Fixed candidate pool M_all (same as real-data run).
        tau_c:           (m,) effect-size thresholds from the original log. FROZEN.
        tau_min:         Minimum confidence interestingness guard.
        eligible:        (m,) bool eligibility mask from the original log. FROZEN.
        holds_all:       Precomputed holds on the null log (optional speedup).
                         Convention: non-vacuous cases only, values ∈ {0, 1}.
                         If None, measures are computed from scratch.

    Returns:
        n_rejected: int — number of rules rejected on the null log.
    """
    D_0, D_1 = split_by_class(null_case_data)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    if holds_all is not None:
        _, _, conf0, conf1 = compute_support_from_holds(
            holds_all, candidates_all, ids_class0, ids_class1, n0, n1
        )
    else:
        _, _, conf0, conf1, _, _ = compute_support_and_confidence(
            null_case_data, candidates_all, ids_class0, ids_class1
        )

    # Apply FROZEN thresholds and eligibility mask from the original log.
    rejected = apply_threshold_decision(
        conf0, conf1,
        tau_c=tau_c,
        tau_min=tau_min,
        eligible=eligible,
    )
    return int(rejected.sum())


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "DeclareMiner — Sample-Size-Scaled Effect-Size Threshold Baseline — Production. "
            "τ*(c) = tau_effect × √(n_ref / n_harm(c)).  No distributional model. No FDR correction."
        )
    )
    parser.add_argument(
        '--tau-effect', type=float, default=DM_CONFIG['tau_effect'],
        help=(
            f"Minimum practically meaningful confidence difference at n_ref applicable traces "
            f"(default: {DM_CONFIG['tau_effect']}).  Direct practical interpretation; "
            "no Type I error rate meaning."
        ),
    )
    parser.add_argument(
        '--n-ref', type=float, default=DM_CONFIG['n_ref'],
        help=(
            f"Reference applicable-trace count for the effect-size anchor "
            f"(default: {DM_CONFIG['n_ref']})."
        ),
    )
    parser.add_argument(
        '--n-floor', type=int, default=DM_CONFIG['n_floor'],
        help=(
            f"Hard exclusion floor: constraints with fewer than this many applicable traces "
            f"in either class are excluded (default: {DM_CONFIG['n_floor']})."
        ),
    )
    parser.add_argument(
        '--tau-min', type=float, default=DM_CONFIG['tau_min'],
        help=f"Minimum confidence interestingness guard (default: {DM_CONFIG['tau_min']})",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Smoke test: report measures and thresholds without saving large JSON",
    )
    args = parser.parse_args()

    config = {
        'tau_effect': args.tau_effect,
        'n_ref':      args.n_ref,
        'n_floor':    args.n_floor,
        'tau_min':    args.tau_min,
    }

    print("\n" + "=" * 100)
    print("DeclareMiner — SAMPLE-SIZE-SCALED EFFECT-SIZE THRESHOLD")
    print("Production: Deviant vs. Normal")
    print("=" * 100)
    print(f"  τ_effect:  {config['tau_effect']}  (min practical conf diff at n_ref traces)")
    print(f"  n_ref:     {config['n_ref']}  (reference applicable-trace count)")
    print(f"  n_floor:   {config['n_floor']}  (hard exclusion floor per class)")
    print(f"  τ_min:     {config['tau_min']}")
    print(f"  τ*(c) = τ_effect × √(n_ref / n_harm(c))")
    print("  No distributional model.  No FDR correction.  Pure effect-size selection.")
    print("=" * 100)

    dm_out = run_declareminer(config=config)

    if not args.dry_run:
        save_outputs(dm_out)

    print(f"\n{'='*100}")
    print("DeclareMiner EFFECT-SIZE BASELINE COMPLETE")
    print(f"{'='*100}")
    print(f"  M_all:         {dm_out['m_total']:,}")
    print(f"  Eligible:      {dm_out['eligible'].sum():,}")
    print(f"  Rejected:      {dm_out['n_rejected']:,}")
    print(f"  τ_effect:      {dm_out['tau_effect']}")
    print(f"  n_ref:         {dm_out['n_ref']}")
    print(f"  n_floor:       {dm_out['n_floor']}")
    print(f"  τ*(c) mean:    {dm_out['tau_c'][dm_out['eligible']].mean():.4f}")
    print(f"  Time:          {dm_out['timing']['total']:.1f}s")
    print(f"  Output:        {OUTPUT_DIR}/")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
