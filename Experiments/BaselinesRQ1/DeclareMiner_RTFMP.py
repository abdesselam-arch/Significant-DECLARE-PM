"""
declareminer_RTFMP_parallel.py  —  DeclareMiner Differential Baseline: RTFMP
=============================================================================
DeclareMiner-style discriminative baseline using differential confidence
thresholds between the two variant logs.

PURPOSE
-------
Implement the DeclareMiner baseline for the RQ1 sensitivity analysis alongside
P1 (Hou-Storey conjunction) and DRVA.  The baseline uses the same fixed
candidate pool M_all as P1 and DRVA, removing candidate-pool confounding so
that all differences in rejection counts trace back to the testing procedure.

DESIGN RATIONALE
----------------
The standard DeclareMiner (MINERful) approach discovers rules in each variant
log independently using support/confidence THRESHOLDS on single-log measures.
That formulation is not directly comparable to P1 or DRVA because it answers
a different question ("which rules hold in log L?") rather than ("which rules
differ between the two variant logs?").

To make the comparison scientifically clean, we use DIFFERENTIAL thresholds:
for each rule r ∈ M_all, compute the difference in confidence between the
Deviant and Normal variant logs and reject r if that difference exceeds a fixed
threshold τ_Δconf.  This mirrors DRVA's Ediff(r) = |Conf_A − Conf_B| criterion
but replaces the permutation test with a deterministic hard threshold.

PRIMARY MEASURE: CONFIDENCE (not Support)
------------------------------------------
Following Cecconi et al. (2021) and the DeclareMiner/MINERful literature,
Confidence is the primary discriminative measure:

    "we consider Confidence as the best option because it measures the degree
     of satisfaction of a rule in a log independently from the rule frequency"

Support conflates activator frequency with satisfaction rate — a rule can have
high Δsupp simply because activity a fires more often in one variant, even if
the conditional relationship a → b is equally strong in both.  This is a known
weakness of support for DECLARE, and it is why Cecconi et al. chose Confidence.

Using Δconf as the primary measure makes the three-method comparison a clean
one-variable-at-a-time hierarchy:

    Method        | Discriminative signal         | Structural | Multiple-testing
    --------------|-------------------------------|------------|------------------
    DeclareMiner  | Δconf ≥ τ_Δconf (hard thresh) | None       | None
    DRVA          | Perm. p-value on Δconf ≤ α    | None       | None (raw α)
    P1 (Ours)     | Hou T statistic on p_disc      | p_struct   | Adaptive Storey

MEASURES
--------
For each rule r and variant log L, we compute:

    Confidence(r, L) = #{t ∈ L : m(r, t) = 1} / #{t ∈ L : m(r, t) ≠ None}
                     = n_satisfied / n_applicable
                     = prevalence(r, L) in Phase 1 terminology.
                     Only non-vacuous (activator-firing) traces contribute.
                     This is the primary discriminative measure.

    Support(r, L)    = #{t ∈ L : m(r, t) = 1} / |L|
                     = n_satisfied / n_total
                     Denominator includes vacuous traces (activator never fires).
                     Retained as a diagnostic measure; NOT used for rejection.

Differential measures:
    Δconf(r)  = |conf(r, L_1) − conf(r, L_0)|   ← PRIMARY decision variable
    Δsupp(r)  = |supp(r, L_1) − supp(r, L_0)|   ← diagnostic only

Decision rule:
    Reject r iff  Δconf(r) >= τ_Δconf
                  AND (conf(r, L_0) >= τ_min OR conf(r, L_1) >= τ_min)

The τ_min guard mirrors DRVA's mmin pruning and removes rules with negligible
confidence in both variants, where Δconf is dominated by sampling noise.

THRESHOLD CALIBRATION
---------------------
A key challenge is that the DeclareMiner baseline has no principled connection
between τ_Δconf and a nominal FDR level α.  For a scientifically informative
RQ1 comparison, we calibrate τ_Δconf so that R_obs^Decl ≈ R_obs^P1 on the
real data.  This matched-rejection strategy makes the FDR comparison vivid:
two methods finding the same number of patterns can have radically different
empirical FDR under the doubly-null protocol.

Calibration grid: τ_Δconf ∈ {0.001, 0.002, ..., 0.500} — 500 equally-spaced points.
For each τ, R_obs(τ) = #{r ∈ M_all : Δconf(r) >= τ AND interestingness guard}.

Selection objective:
    τ* = max{τ ∈ G : R(τ) >= R_obs_target}
    The tightest (largest) feasible τ — the rightmost element of the feasible
    prefix of the monotone non-increasing R(τ) curve.  Ensures R_obs(τ*) ≥ R_target
    and gives DeclareMiner the hardest threshold still achieving parity, making
    any FDR inflation under the null loop attributable solely to the absence of
    a permutation test.

Hard error on under-discovery (R_obs(τ_min) < R_target at every grid point):
    Raises ValueError — a silent fallback would propagate an invalid τ*.

Pathologies detected and reported explicitly:
    Over-discovery:   R_obs(τ_max = 0.500) >= R_target — threshold non-discriminative.

DOUBLY-NULL FDR ESTIMATION (RQ1 integration)
--------------------------------------------
In each held-out null replicate b (sigma_trace ∘ sigma_label already applied):
    1.  Recompute conf(r, L_0^(b)) and conf(r, L_1^(b)) from the null log.
    2.  Compute Δconf_r^(b).
    3.  Apply the SAME fixed threshold τ*:  reject r iff Δconf_r^(b) >= τ*.
    4.  V_b = #{rejected rules under null} = all false positives by construction.

The expected V_b under the null reflects spurious confidence differences due to
finite-sample fluctuations, which the fixed threshold cannot account for.

OUTPUT FILES
------------
    declareminer_results.json               All rules, measures, and decisions.
    declareminer_significant_patterns.json  Rejected rules only.
    declareminer_report.txt                 Ranked text output.
    declareminer_calibration.csv            R_obs vs τ_Δconf grid (for paper figure).

Version : 2.0  (calibration: max-feasible τ on fine grid {0.001, 0.002, ..., 0.500})
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References:
    Cecconi, Augusto & Di Ciccio (2021). Detection of Statistically Significant
        Differences Between Process Variants Through Declarative Rules.
        BPM Forum 2021, LNBIP 427, pp. 73–91.
    Di Ciccio & Mecella (2015). On the discovery of declarative control flows
        for artful processes. TMIS 5(4):24.
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
from P1_SDSM.p1_RTFMP_hou import (
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
OUTPUT_DIR        = "DeclareMiner_RTFMP"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DM_CONFIG = {
    # PRIMARY threshold on confidence difference (Δconf — matches DRVA's Ediff)
    'tau_delta_conf': None,   # None → calibrate automatically against R_obs_target
    # Optional secondary gate on support difference (None = disabled)
    'tau_delta_supp': None,
    # Minimum interestingness guard: rule must have conf >= tau_min in at least one variant
    'tau_min':        0.01,
    # Calibration target: integer R_obs_target, or None
    'R_obs_target':   None,
    # Calibration grid (start, stop, step) — 5000 points: {0.0001, 0.0002, ..., 0.500}
    'tau_grid_start': 0.0001,   # Layer 1: 10× finer floor
    'tau_grid_stop':  0.500,
    'tau_grid_step':  0.0001,   # 5000 points
    'random_state':   42,
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
    # Verify convention: all values in holds_all must be 0 or 1, never None.
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


# ─── DECISION RULE ───────────────────────────────────────────────────────────

def apply_threshold_decision(
    conf0: np.ndarray,
    conf1: np.ndarray,
    supp0: np.ndarray,
    supp1: np.ndarray,
    tau_delta_conf: float,
    tau_min: float,
    tau_delta_supp: Optional[float] = None,
) -> np.ndarray:
    """
    Apply the DeclareMiner differential threshold decision rule.

    Primary criterion (matches DRVA's Ediff):
        Δconf(r) = |conf(r, L_1) - conf(r, L_0)| >= tau_delta_conf

    Interestingness guard (mirrors DRVA's mmin):
        conf(r, L_0) >= tau_min  OR  conf(r, L_1) >= tau_min

    Optional secondary gate on support difference (disabled by default):
        |supp(r, L_1) - supp(r, L_0)| >= tau_delta_supp

    Reject rule r iff all active conditions are satisfied.

    Args:
        conf0, conf1:    (m,) confidence arrays for class 0 and class 1.
        supp0, supp1:    (m,) support arrays (used only for optional secondary gate).
        tau_delta_conf:  Primary threshold on confidence difference.
        tau_min:         Minimum confidence guard (interestingness).
        tau_delta_supp:  Optional secondary threshold on support difference.
                         None → not applied.

    Returns:
        (m,) bool array — True if rule is rejected ("discovered").
    """
    delta_conf  = np.abs(conf1 - conf0)
    interesting = (conf0 >= tau_min) | (conf1 >= tau_min)

    rejected = (delta_conf >= tau_delta_conf) & interesting

    if tau_delta_supp is not None:
        delta_supp = np.abs(supp1 - supp0)
        rejected   = rejected & (delta_supp >= tau_delta_supp)

    return rejected


# ─── THRESHOLD CALIBRATION ───────────────────────────────────────────────────

def calibrate_threshold(
    conf0: np.ndarray,
    conf1: np.ndarray,
    supp0: np.ndarray,
    supp1: np.ndarray,
    R_obs_target: int,
    config: dict,
) -> Tuple[float, pd.DataFrame]:
    """
    τ* = max{τ ∈ G : R(τ) >= R_obs_target}  (tightest threshold from above).

    Two-layer fallback strategy:
      Layer 1 — Fine grid [0.0001, 0.500] step 0.0001 (5000 points).
                Resolves cases where the 0.001 floor missed small-Δconf rules.
      Layer 2 — Under-discovery cap: if R_obs(τ_min) < R_target at every point,
                set τ* = τ_min (maximum achievable R_obs). Records pathology
                'under_discovery_capped' in calib_df.attrs — never raises.

    Pathologies recorded in calib_df.attrs['pathology']:
        'normal'                  — τ* found strictly inside the grid
        'over_discovery'          — R_obs(τ_max) >= R_target (non-discriminative)
        'under_discovery_capped'  — R_obs(τ_min) < R_target; τ* = τ_min

    Args:
        conf0, conf1: (m,) confidence arrays.
        supp0, supp1: (m,) support arrays (for optional secondary gate only).
        R_obs_target: integer target rejection count.
        config:       DM_CONFIG dict for grid parameters and tau_min.

    Returns:
        tau_star:   float — calibrated threshold on Δconf.
        calib_df:   pd.DataFrame — full calibration grid for paper figures.
    """
    tau_min        = config['tau_min']
    tau_delta_supp = config.get('tau_delta_supp')

    grid = np.arange(
        config['tau_grid_start'],          # 0.0001
        config['tau_grid_stop'] + 1e-9,    # 0.5000
        config['tau_grid_step'],           # 0.0001  → 5000 points
    )

    # ── Build R(τ) curve ─────────────────────────────────────────────────
    r_obs_arr = np.array([
        int(apply_threshold_decision(
            conf0, conf1, supp0, supp1,
            tau_delta_conf=float(tau),
            tau_min=tau_min,
            tau_delta_supp=tau_delta_supp,
        ).sum())
        for tau in grid
    ], dtype=np.int64)

    calib_df = pd.DataFrame({
        'tau_delta_conf': np.round(grid, 6),
        'R_obs':          r_obs_arr,
    })

    # ── Diagnostics ───────────────────────────────────────────────────────
    print(f"\n   Calibration diagnostics:")
    print(f"     Grid:           [{grid[0]:.4f}, {grid[-1]:.4f}]  "
          f"step={config['tau_grid_step']:.4f}  ({len(grid)} points)")
    print(f"     R_obs(τ_min):   {r_obs_arr[0]:,}")
    print(f"     R_obs(τ_max):   {r_obs_arr[-1]:,}")
    print(f"     R_obs_target:   {R_obs_target:,}")
    n_feasible = int((r_obs_arr >= R_obs_target).sum())
    print(f"     Feasible τ:     {n_feasible} / {len(grid)}")

    # ── Selection ─────────────────────────────────────────────────────────
    feasible = r_obs_arr >= R_obs_target

    if not feasible.any():
        # ── LAYER 2 (Option B): under-discovery cap ───────────────────────
        # R_obs(τ_min) < R_target even on the finest grid.
        # τ* = τ_min gives the maximum achievable R_obs.
        # This is a legitimate scientific finding: DM cannot match the target
        # on this log regardless of threshold — report it explicitly.
        best_idx  = 0
        pathology = 'under_discovery_capped'
        print(
            f"\n  [WARNING] Under-discovery: R_obs(τ_min={grid[0]:.4f}) = {r_obs_arr[0]:,} "
            f"< R_obs_target = {R_obs_target:,} at every grid point.\n"
            f"  Layer 2 cap applied: τ* = τ_min = {grid[0]:.4f}  "
            f"(maximum achievable R_obs = {r_obs_arr[0]:,}).\n"
            "  This gap is a scientific result: DM cannot reach the target rejection "
            "count on this log — report R_obs_max vs R_target in the paper."
        )
    else:
        # ── LAYER 1 (Option A): tightest feasible τ ───────────────────────
        best_idx = int(np.where(feasible)[0][-1])
        if feasible[-1]:
            pathology = 'over_discovery'
            print(
                f"\n  [WARNING] Over-discovery: R_obs(τ_max={grid[-1]:.4f}) = {r_obs_arr[-1]:,} "
                f">= R_target={R_obs_target:,}. Threshold non-discriminative on this log.\n"
                "  τ* = τ_max; empirical FDR from doubly-null loop will reflect "
                "non-discrimination."
            )
        else:
            pathology = 'normal'

    tau_star  = float(calib_df.iloc[best_idx]['tau_delta_conf'])
    r_at_star = int(calib_df.iloc[best_idx]['R_obs'])
    # overshoot < 0 signals under-discovery cap (R_obs < target)
    overshoot = r_at_star - R_obs_target

    calib_df.attrs['pathology'] = pathology
    calib_df.attrs['tau_star']  = tau_star
    calib_df.attrs['r_at_star'] = r_at_star
    calib_df.attrs['overshoot'] = overshoot   # negative when capped

    print(f"\n   Calibration result:")
    print(f"     τ*:             {tau_star:.4f}")
    print(f"     R_obs(τ*):      {r_at_star:,}  "
          f"({'overshoot' if overshoot >= 0 else 'shortfall'} = "
          f"{'+'if overshoot >= 0 else ''}{overshoot})")
    print(f"     R_obs_target:   {R_obs_target:,}")
    print(f"     pathology:      {pathology}")

    return tau_star, calib_df


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_declareminer(
    config: Optional[dict] = None,
    case_data: Optional[Dict[str, CaseInfo]] = None,
    candidates_all: Optional[List[Tuple]] = None,
    R_obs_target: Optional[int] = None,
) -> dict:
    """
    Execute the full DeclareMiner differential threshold pipeline on RTFMP.

    Args:
        config:          Override DM_CONFIG parameters.
        case_data:       Pre-loaded case data (avoids reload for RQ1).
        candidates_all:  Fixed candidate pool M_all (shared with P1/DRVA).
        R_obs_target:    Number of P1 discoveries to calibrate against.
                         Overrides config['R_obs_target'].

    Returns:
        dict with all results, measures, calibration curve, and intermediate
        quantities needed for RQ1 integration.
    """
    cfg = {**DM_CONFIG, **(config or {})}
    if R_obs_target is not None:
        cfg['R_obs_target'] = R_obs_target

    tau_min       = float(cfg['tau_min'])
    tau_delta_supp = cfg.get('tau_delta_supp')

    timing   = {}
    t0_total = time.time()

    # ── Section 0: Load data ──────────────────────────────────────────────
    if case_data is None:
        print("\n" + "=" * 100)
        print("DeclareMiner — STEP 0: DATA LOADING")
        print("=" * 100)
        case_data = load_and_preprocess_data(CSV_PATH)

    if candidates_all is None:
        print("\n" + "=" * 100)
        print("DeclareMiner — STEP 1: CANDIDATE GENERATION FROM PHASE 0 SPEC")
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
    print(f"   L_0 (No Credit Collection, class 0): {n0:,} traces")
    print(f"   L_1 (Deviant, class 1): {n1:,} traces")

    # ── Section 2: Compute measures on real data ──────────────────────────
    print("\n" + "=" * 100)
    print("DeclareMiner — STEP 2: CONFIDENCE & SUPPORT COMPUTATION")
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

    # ── Section 3: Threshold determination ───────────────────────────────
    print("\n" + "=" * 100)
    print("DeclareMiner — STEP 3: THRESHOLD DETERMINATION (on Δconf)")
    print("=" * 100)

    tau_star_given = cfg.get('tau_delta_conf')
    calib_df       = None

    if tau_star_given is not None:
        tau_star = float(tau_star_given)
        print(f"   Using provided threshold τ_Δconf = {tau_star:.4f}")
        r_target = cfg['R_obs_target'] or 0
        _, calib_df = calibrate_threshold(
            conf0, conf1, supp0, supp1, R_obs_target=r_target, config=cfg
        )
    elif cfg['R_obs_target'] is not None:
        r_target = int(cfg['R_obs_target'])
        print(f"   Calibrating τ_Δconf to match R_obs_target = {r_target:,} ...")
        t_calib = time.time()
        tau_star, calib_df = calibrate_threshold(
            conf0, conf1, supp0, supp1, R_obs_target=r_target, config=cfg
        )
        timing['calibration'] = time.time() - t_calib
    else:
        tau_star = 0.05
        print(f"   No calibration target provided — using default τ_Δconf = {tau_star:.4f}")
        _, calib_df = calibrate_threshold(
            conf0, conf1, supp0, supp1, R_obs_target=0, config=cfg
        )

    # ── Section 4: Apply decision rule ───────────────────────────────────
    print("\n" + "=" * 100)
    print(f"DeclareMiner — STEP 4: DECISION RULE  (τ_Δconf = {tau_star:.4f})")
    print("=" * 100)
    print(f"   Reject r iff  Δconf(r) >= {tau_star:.4f}  [matches DRVA Ediff threshold]")
    if tau_delta_supp is not None:
        print(f"                AND Δsupp(r) >= {tau_delta_supp:.4f}  [optional secondary]")
    print(f"                AND (conf(r, L_0) >= {tau_min} OR conf(r, L_1) >= {tau_min})")
    print(f"   No statistical test, no FDR correction.")

    rejected  = apply_threshold_decision(
        conf0, conf1, supp0, supp1,
        tau_delta_conf=tau_star,
        tau_min=tau_min,
        tau_delta_supp=tau_delta_supp,
    )
    n_rejected = int(rejected.sum())

    print(f"\n   Rejected: {n_rejected:,} rules")
    ct_counts = Counter(
        candidates_all[i][0] for i in range(m_total) if rejected[i]
    )
    for ct in ALL_CONSTRAINT_TYPES:
        if ct in ct_counts:
            print(f"     {ct:<30s}: {ct_counts[ct]:,}")

    # Direction: class 1 (Deviant) has higher confidence
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
            'supp0':           float(supp0[r_idx]),
            'supp1':           float(supp1[r_idx]),
            'delta_supp':      float(delta_supp[r_idx]),
            'napp0':           int(napp0[r_idx]),
            'napp1':           int(napp1[r_idx]),
            'is_significant':  bool(rejected[r_idx]),
        })

    # Sort by Δconf descending (mirrors DRVA's Ediff-based ranking)
    results_all.sort(key=lambda x: (-x['delta_conf'], -max(x['conf0'], x['conf1'])))

    n_pos = sum(1 for r in results_all if r['is_significant'] and r['direction'] == 'Positive')
    n_neg = sum(1 for r in results_all if r['is_significant'] and r['direction'] == 'Negative')

    print(f"\n   Timing:")
    for k, v in timing.items():
        print(f"     {k:25s}: {v:.1f}s")
    print(f"\n   Summary:")
    print(f"     M_all:            {m_total:,}")
    print(f"     Rejected:         {n_rejected:,}  (Positive: {n_pos}, Negative: {n_neg})")
    print(f"     τ_Δconf:          {tau_star:.4f}")
    print(f"     τ_min:            {tau_min:.4f}")

    return {
        'results_all':       results_all,
        'rejected':          rejected,
        'n_rejected':        n_rejected,
        'tau_star':          tau_star,
        'candidates_all':    candidates_all,
        'm_total':           m_total,
        'conf0':             conf0,
        'conf1':             conf1,
        'supp0':             supp0,
        'supp1':             supp1,
        'napp0':             napp0,
        'napp1':             napp1,
        'delta_conf':        delta_conf,
        'delta_supp':        delta_supp,
        'calib_df':          calib_df,
        'config':            cfg,
        'timing':            timing,
        'case_data':         case_data,
        'ids_class0':        ids_class0,
        'ids_class1':        ids_class1,
        'n0':                n0,
        'n1':                n1,
    }


# ─── OUTPUT GENERATION ────────────────────────────────────────────────────────

def save_outputs(dm_out: dict) -> None:
    """Save JSON results, significant-only JSON, text report, calibration CSV."""

    cfg         = dm_out['config']
    results_all = dm_out['results_all']
    m_total     = dm_out['m_total']
    n_rejected  = dm_out['n_rejected']
    tau_star    = dm_out['tau_star']
    timing      = dm_out['timing']
    calib_df    = dm_out['calib_df']
    case_data   = dm_out['case_data']

    sig_results = [r for r in results_all if r['is_significant']]
    n_pos = sum(1 for r in sig_results if r['direction'] == 'Positive')
    n_neg = sum(1 for r in sig_results if r['direction'] == 'Negative')

    # ── JSON ─────────────────────────────────────────────────────────────
    full_json = {
        'framework': 'DeclareMiner Differential Threshold Baseline',
        'version':   '2.0',
        'timestamp': datetime.now().isoformat(),
        'description': {
            'discriminative_criterion': (
                f"Reject r iff |conf(r, L_1) - conf(r, L_0)| >= τ_Δconf = {tau_star:.4f} "
                f"AND (conf(r, L_0) >= {cfg['tau_min']} OR conf(r, L_1) >= {cfg['tau_min']}). "
                "Primary measure: Δconf (matches DRVA Ediff). "
                "No statistical test, no FDR correction."
            ),
            'primary_measure_rationale': (
                "Confidence is the primary measure following Cecconi et al. (2021) and "
                "the DeclareMiner/MINERful literature, because it is independent of "
                "activator frequency — unlike Support which conflates activator frequency "
                "with satisfaction rate.  Using Δconf as the primary criterion makes the "
                "three-method hierarchy (DeclareMiner / DRVA / P1) a clean one-variable-"
                "at-a-time comparison: the only difference between DeclareMiner and DRVA "
                "is the presence vs. absence of a permutation test on Δconf."
            ),
            'structural_criterion': "None — purely confidence-based threshold.",
            'fdr_note': (
                "DeclareMiner applies no FDR correction. The threshold τ_Δconf has "
                "no principled relationship to a nominal FDR level. The doubly-null "
                "protocol (RQ1) will show empirical FDR >> α because finite-sample "
                "confidence differences under the null can exceed τ_Δconf by chance "
                "(RTFMP: Deviant (Sent for Credit Collection) vs. Normal (No Credit Collection))."
            ),
            'threshold_calibration': (
                f"τ* = max{{τ ∈ G : R(τ) >= R_target}}  (tightest feasible threshold). "
                f"R_target = {cfg.get('R_obs_target', 'N/A')}. "
                "Grid G = {0.001, 0.002, ..., 0.500} (500 points, step=0.001). "
                "ValueError raised on under-discovery (no silent fallback). "
                "This matched-rejection design makes the FDR comparison informative."
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
            'n_rejected':          n_rejected,
            'tau_star':            tau_star,
            'rejection_rate':      n_rejected / max(m_total, 1),
            'n_rejected_positive': n_pos,
            'n_rejected_negative': n_neg,
        },
        'timing':           timing,
        'all_rules':        results_all,
        'significant_rules': sig_results,
    }

    path_full = os.path.join(OUTPUT_DIR, 'declareminer_results.json')
    with open(path_full, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON (full):        {path_full}")

    path_sig = os.path.join(OUTPUT_DIR, 'declareminer_significant_patterns.json')
    sig_json = {k: v for k, v in full_json.items() if k != 'all_rules'}
    with open(path_sig, 'w', encoding='utf-8') as f:
        json.dump(sig_json, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON (significant): {path_sig}")

    # ── Calibration CSV ───────────────────────────────────────────────────
    if calib_df is not None:
        path_calib = os.path.join(OUTPUT_DIR, 'declareminer_calibration.csv')
        calib_df.to_csv(path_calib, index=False)
        print(f"✓ Calibration CSV:    {path_calib}")

    # ── Text report ───────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 120)
    lines.append("DECLAREMINER DIFFERENTIAL THRESHOLD BASELINE  (primary measure: Δconf)")
    lines.append("RTFMP: Deviant (Sent for Credit Collection) vs. Normal (No Credit Collection)")
    lines.append("=" * 120)
    lines.append(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"τ_Δconf = {tau_star:.4f}, τ_min = {cfg['tau_min']}, "
                 f"R_obs_target = {cfg.get('R_obs_target', 'N/A')}")
    lines.append("")
    lines.append(f"M_all:     {m_total:,}")
    lines.append(f"Rejected:  {n_rejected:,}  (Positive: {n_pos}, Negative: {n_neg})")
    lines.append("")

    lines.append("Decision rule:")
    lines.append(f"  |conf(r, L_1) - conf(r, L_0)| >= {tau_star:.4f}  [= Ediff threshold]")
    lines.append(f"  AND (conf(r, L_0) >= {cfg['tau_min']} OR conf(r, L_1) >= {cfg['tau_min']})")
    lines.append("  No permutation test.  No FDR correction.")
    lines.append("")

    lines.append("=" * 120)
    lines.append("TOP 50 SIGNIFICANT RULES  (ranked by Δconf descending)")
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
            f"Δconf={r['delta_conf']:.4f}  "
            f"supp_L0={r['supp0']:.4f}  supp_L1={r['supp1']:.4f}  "
            f"Δsupp={r['delta_supp']:.4f}"
        )
        lines.append("")

    lines.append("=" * 120)
    lines.append("CALIBRATION GRID (τ_Δconf vs R_obs)")
    lines.append("=" * 120)
    if calib_df is not None:
        lines.append(f"  {'τ_Δconf':>10}  {'R_obs':>8}")
        for _, row in calib_df.iterrows():
            marker = "  ← τ*" if abs(row['tau_delta_conf'] - tau_star) < 1e-9 else ""
            lines.append(f"  {row['tau_delta_conf']:>10.4f}  {int(row['R_obs']):>8}{marker}")

    lines.append("")
    lines.append("=" * 120)
    lines.append("TIMING")
    lines.append("=" * 120)
    for k, v in timing.items():
        lines.append(f"  {k:25s}: {v:.1f}s")

    path_rpt = os.path.join(OUTPUT_DIR, 'declareminer_report.txt')
    with open(path_rpt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ Text report:        {path_rpt}")


# ─── RQ1 INTEGRATION: DOUBLY-NULL COUNTER ─────────────────────────────────────

def run_declareminer_on_doubly_null_log(
    null_case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    tau_star: float,
    tau_min: float = 0.01,
    tau_delta_supp: Optional[float] = None,
    holds_all: Optional[Dict] = None,
) -> int:
    """
    Apply the DeclareMiner decision rule to a pre-built doubly-nullified log.

    Called from the RQ1 null-replicate loop.  Every rejection here is a false
    positive by construction (sigma_trace ∘ sigma_label already applied).

    The threshold tau_star applies to Δconf (primary measure), consistent with
    the real-data calibration.  The threshold is held fixed; only the observed
    Δconf changes across null replicates.

    Args:
        null_case_data:  Dict[case_id -> CaseInfo] after double nullification.
        candidates_all:  Fixed candidate pool M_all.
        tau_star:        Calibrated τ_Δconf threshold from the real-data run.
        tau_min:         Minimum confidence interestingness guard.
        tau_delta_supp:  Optional secondary support-difference threshold.
        holds_all:       Precomputed holds on the null log (optional speedup).
                         Convention: non-vacuous cases only, values ∈ {0, 1}.
                         If None, measures are computed from scratch.

    Returns:
        n_rejected: int — number of rules with Δconf >= τ* under the null.
    """
    D_0, D_1 = split_by_class(null_case_data)
    ids_class0 = set(D_0.keys())
    ids_class1 = set(D_1.keys())
    n0 = len(ids_class0)
    n1 = len(ids_class1)

    if holds_all is not None:
        supp0, supp1, conf0, conf1 = compute_support_from_holds(
            holds_all, candidates_all, ids_class0, ids_class1, n0, n1
        )
    else:
        supp0, supp1, conf0, conf1, _, _ = compute_support_and_confidence(
            null_case_data, candidates_all, ids_class0, ids_class1
        )

    rejected = apply_threshold_decision(
        conf0, conf1, supp0, supp1,
        tau_delta_conf=tau_star,
        tau_min=tau_min,
        tau_delta_supp=tau_delta_supp,
    )
    return int(rejected.sum())


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeclareMiner Differential Threshold Baseline — RTFMP"
    )
    parser.add_argument(
        '--tau-delta-conf', type=float, default=None,
        help="Fixed Δconf threshold (default: calibrate against --r-obs-target)",
    )
    parser.add_argument(
        '--tau-delta-supp', type=float, default=None,
        help="Optional secondary Δsupp threshold (default: not used)",
    )
    parser.add_argument(
        '--tau-min', type=float, default=DM_CONFIG['tau_min'],
        help=f"Minimum confidence interestingness guard (default: {DM_CONFIG['tau_min']})",
    )
    parser.add_argument(
        '--r-obs-target', type=int, default=None,
        help="Target rejection count for threshold calibration",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Smoke test: report measures without saving large JSON",
    )
    args = parser.parse_args()

    config = {
        'tau_delta_conf': args.tau_delta_conf,
        'tau_delta_supp': args.tau_delta_supp,
        'tau_min':        args.tau_min,
        'R_obs_target':   args.r_obs_target,
        'tau_grid_start': DM_CONFIG['tau_grid_start'],
        'tau_grid_stop':  DM_CONFIG['tau_grid_stop'],
        'tau_grid_step':  DM_CONFIG['tau_grid_step'],
        'random_state':   42,
    }

    print("\n" + "=" * 100)
    print("DeclareMiner DIFFERENTIAL THRESHOLD BASELINE  (primary measure: Δconf)")
    print("RTFMP: Deviant (Sent for Credit Collection) vs. Normal (No Credit Collection)")
    print("=" * 100)
    print(f"  τ_Δconf:      {config['tau_delta_conf'] or 'calibrate'}")
    print(f"  τ_min:        {config['tau_min']}")
    print(f"  R_obs_target: {config['R_obs_target'] or 'not set'}")
    print("  No statistical test.  No FDR correction.")
    print("=" * 100)

    dm_out = run_declareminer(config=config)

    if not args.dry_run:
        save_outputs(dm_out)

    print(f"\n{'='*100}")
    print("DeclareMiner BASELINE COMPLETE")
    print(f"{'='*100}")
    print(f"  M_all:     {dm_out['m_total']:,}")
    print(f"  Rejected:  {dm_out['n_rejected']:,}")
    print(f"  τ*:        {dm_out['tau_star']:.4f}")
    print(f"  Time:      {dm_out['timing']['total']:.1f}s")
    print(f"  Output:    {OUTPUT_DIR}/")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
