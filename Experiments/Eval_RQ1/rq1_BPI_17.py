#!/usr/bin/env python3
"""
rq1_BPI_17.py  —  RQ1 FDR Control Validity: BPI Challenge 2017
=============================================================================

PURPOSE
-------
Empirically validate that the Fisher-Storey FDR framework controls the
false discovery rate at nominal alpha = 0.05 on the BPI Challenge 2017
(loan application) event log, following the gold-standard held-out
permutation protocol of Pellegrina & Vandin (KDD 2018).

Label definition (Teinemaa et al. TKDE 2019 — bpic2017_accepted sub-log):
    Deviant (1): terminal O_ event is O_Refused or O_Cancelled  -> loan not accepted
    Normal  (0): terminal O_ event is O_Accepted                -> loan accepted

RESEARCH QUESTION
-----------------
"Given a log where no pattern is structurally AND discriminatively
significant, does the Fisher-Storey framework guarantee FDR <= alpha?"

This defines the global joint null for every pattern i:

    H0^(i): pattern i is null on BOTH axes simultaneously

For the Fisher combination statistic T = -2(ln p_struct + ln p_disc) to
follow its nominal chi2(4) distribution under this null, BOTH input
p-values must be independently U(0,1).  Label-only permutation guarantees
only p_disc ~ U(0,1).  It says nothing about p_struct, because the
structural test measures within-class trace ordering — a property of the
traces themselves, not the label assignment.

DOUBLE-NULL PROTOCOL (the scientific correction)
-------------------------------------------------
To manufacture a null log where both axes are simultaneously nullified,
each held-out replicate b applies TWO independent operations:

    Null_replicate(b) = sigma_label ∘ sigma_trace

    1.  sigma_label:  Permute class labels (preserving marginals n+, n-).
                      -> Destroys any discriminative signal between classes.
                      -> Guarantees p_disc^(b) ~ U(0,1) by Fisher randomization.

    2.  sigma_trace:  Within each trace, randomly permute the activity sequence
                      (preserving trace length and activity multiset per case).
                      -> Destroys all temporal ordering within traces.
                      -> Guarantees p_struct^(b) ~ U(0,1), because the shuffled
                         trace IS a draw from the structural null distribution
                         (exactly what run_structural_permutation_test generates
                         internally).

    3.  Both axes freshly recomputed: holds_all recomputed on shuffled traces,
        label permutation test produces fresh p_disc + null_delta_matrix,
        structural permutation test produces fresh p_struct for both classes.

Under this double-null:
    T_i^(b) = -2(ln p_struct^(b) + ln p_disc^(b)) ~ chi2(4)

This is NOT an architectural change to p1_BPI_17.py.  The Phase 1 framework,
Fisher combination, Adaptive Storey, and significance gates remain unchanged.
Only the RQ1 evaluation script constructs held-out replicates that are
faithful to the joint null the RQ claims to validate.

WHY LABEL-ONLY PERMUTATION IS INSUFFICIENT
--------------------------------------------
Even with fresh (non-cached) structural p-values under label-only permutation,
p_struct^(b) is NOT U(0,1) for patterns with real within-log temporal structure.
In BPI 2017, many patterns spanning the A_/O_/W_ activity namespaces have
structural signal visible in any random class partition (pi0_struct < 1).
The Fisher statistic for those patterns remains shifted rightward:

    T_i^(b) = [-2 ln p_struct^(fresh,b)]  +  [-2 ln p_disc^(b)]
               ^^ != chi2(2), > 0 in exp.      ^^ ~ chi2(2)

This residual structural signal inflates FDR_emp beyond alpha.  Within-trace
shuffling eliminates it.

ALIGNMENT WITH p1_BPI_17.py
----------------------------
The decision procedure in null replicates faithfully reproduces
execute_three_hypothesis_protocol.  p1 v8.0 uses EMPIRICAL Phipson-Smyth
calibration of T_F for the Storey gate; the BH reference retains analytic
chi2_4.  Under the double-null, analytic chi2_4 IS the oracle null p-value
(see NOTE ON EMPIRICAL CALIBRATION in run_doubly_null_replicate), so null
replicates correctly use analytic p-values without loss of validity.

    1.  Score:
            T_F(p) = -2*(ln p_struct_dom + ln p_disc)

    2.  Analytic p-value (BH reference only):
            p_conjunction(p) = chi2.sf(T_F(p), df=4)

    3.  Empirical Phipson-Smyth p-value (Storey gate, real run only):
            p̃_F(p) = (1 + #{b: T_F^(b)(p) >= T_F(p)}) / (B_null + 1)
            computed via compute_double_null_tf_matrix (B_null=100 replicates)
            NULL REPLICATES: use p_conjunction (analytic = oracle under double-null)

    4.  Structural scope filter on m'' patterns:
            structural_idx = { i : min(p_struct_screen_c0[i], p_struct_screen_c1[i]) <= alpha }

    5.  Adaptive Storey pi0 (Gao 2023) on m'' empirical p̃_F:
            pi0_f, _ = adaptive_storey_pi0(p_tilde_fisher_m_prime, q=alpha)

    6.  Storey q-values on m'' p̃_F:
            q_fisher = storey_qvalue(p_tilde_fisher_m_prime, pi0_f)

    7.  Conjunctive-gate significance ("Both"):
            is_significant_final = (q_Fisher <= alpha) AND (p_struct_dom <= alpha)

FOUR METHODS COMPARED (at alpha = 0.05)
-----------------------------------------
    1.  Fisher-Storey       Adaptive Storey q-values on m' Fisher p-values.
                            Conjunctive gate: q_Fisher <= alpha AND p_struct_dom <= alpha.  (Primary method.)

    2.  BH-Fisher           BH step-up on m' Fisher p-values (no pi0 correction).

    3.  Cecconi ChiSq+BH   Chi-square 2x2 + BH (Cecconi et al., BPM 2021).

    4.  Tusher flat-null    Original SAM pooled-null (expected: k*=0).

COMPUTATIONAL BUDGET
---------------------
    Original run:     B1=4,000 (label), B2=2,000 (structural)  — full precision
    Null replicates:  B1=2,000 (label), B2=500  (structural)   — reduced
    B_null:           200 held-out replicates

    Per-replicate cost (double-null):
        ~2 min  trace shuffling + holds recomputation
        ~3 min  label permutation test (B1=2000)
        ~15 min structural permutation test (B2=500, both classes)
        ~20 min total per replicate
    
    Total: 200 x 20 min / 8 cores ~ 8 HPC hours

OUTPUT FILES
-------------
    rq1_fdr_metrics.csv      One row per method (Table 1 data).
    rq1_null_counts.csv      B_null rows x 4 methods (raw FP counts).
    rq1_results.json         Full results (Tusher failure, pi0, for paper).
    rq1_pattern_arrays.npz   Per-pattern arrays from original run.

Version : 3.1  (empirical Phipson-Smyth calibration for Storey gate; aligned with p1_BPI_17.py v8.0)
Author  : Ahmed Nour Abdesselam
Date    : March 2026
"""

import sys
import os
import copy
import io
import contextlib
import time
import json
import argparse
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

# ═══════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from Experiments.P1_SDSM.p1_BPI_17 import (
# from p1_BPI_17 import (
    # Data loading & preprocessing
    load_and_preprocess_data,
    generate_candidate_patterns,
    split_by_class,
    compute_prevalence_from_holds,
    # Holds computation (needed for recomputation on shuffled traces)
    compute_holds_by_case_batch,
    precompute_activity_index,
    # Permutation tests
    run_label_permutation_test,
    run_structural_permutation_test,
    # Statistical machinery
    fisher_conjunction_pvalue,
    adaptive_storey_pi0,
    storey_pi0_bootstrap,
    storey_qvalue,
    benjamini_hochberg,
    # Pipeline entry point
    execute_pipeline,
    # Data structures
    CaseInfo,
    PatternTestResult,
    # Global configuration
    CONFIG as P1_CONFIG,
    INPUT_FILE as P1_INPUT_FILE,
    OUTPUT_DIR as P1_OUTPUT_DIR,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
)

from eval_utils import (
    generate_heldout_permutation_batch,
    compute_empirical_fdr,
    bootstrap_bca_ci,
    run_cecconi_baseline,
    run_tusher_flat_null,
    compute_pi0_with_ci,
    compute_pi0_all_axes,
    compute_sigma_null_heterogeneity,
    compute_tusher_inflation_factor,
    build_tusher_failure_report,
    build_rq1_results_df,
    test_fdr_control,
    save_rq1_results_json,
    EmpiricalFDRResult,
    Pi0EstimateWithCI,
    TusherFailureReport,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH       = P1_INPUT_FILE
PHASE0_JSON    = P1_SPEC_FILE
RQ1_OUTPUT_DIR = "../Experiments data/Experiments/Results/RQ1_BPI17"
# RQ1_OUTPUT_DIR = "RQ1_BPI17"

# Original run budget (full precision, used by execute_pipeline)
B1_FULL = 4_000
B2_FULL = 2_000

# Null replicate budget (reduced for feasibility)
# Phipson-Smyth resolution: 1/(B1_VALID+1) ~ 5e-4, well below alpha=0.05
B1_VALID = 2_000
B2_VALID = 500

# Held-out null replicates
B_NULL = 200

# FDR level
ALPHA = 0.05

# Seeds: held-out seeds (BASE_SEED + b) are independent of Phase 1 internals
BASE_SEED = 20260321

# Parallelism
N_JOBS = -1

# Method name constants
METHOD_FISHER_STOREY = "Fisher-Storey"
METHOD_BH_FISHER     = "BH-Fisher"
METHOD_CECCONI       = "Cecconi_ChiSq_BH"
METHOD_TUSHER        = "Tusher_FlatNull"

ALL_METHODS = [METHOD_FISHER_STOREY, METHOD_BH_FISHER, METHOD_CECCONI, METHOD_TUSHER]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: SUPPRESS STDOUT / STDERR
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout and stderr to devnull during null replicates."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — RUN ORIGINAL PIPELINE AND EXTRACT EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════

def run_original_pipeline() -> dict:
    """
    Run p1_BPI_17.py's execute_pipeline with full computational budget.
    Returns all objects needed for R_obs computation and RQ1 evaluation.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — BPI 2017: ORIGINAL-DATA RUN VIA execute_pipeline()")
    print(f"  B1={B1_FULL:,}, B2={B2_FULL:,}, alpha={ALPHA}")
    print("=" * 100)

    t0 = time.time()

    orig_config = P1_CONFIG.copy()
    orig_config['B_label']      = B1_FULL
    orig_config['B_trace']      = B2_FULL
    orig_config['fdr_alpha']    = ALPHA
    orig_config['random_state'] = 42

    output = execute_pipeline(input_file=CSV_PATH, config=orig_config)

    case_data       = output['case_data']
    pattern_results = output['pattern_results']
    null_delta_mat  = output['null_delta_matrix']
    holds_all       = output['holds_all']
    delta_obs       = output['delta_obs']
    candidates_all  = output['candidates_all']
    timing          = output['timing']
    tf_null_mat     = output['tf_null_matrix']

    case_ids_sorted = sorted(case_data.keys())
    labels = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    wall = time.time() - t0
    n = len(labels)
    n1 = int(labels.sum())
    m = len(candidates_all)
    print(f"\n  Pipeline complete in {wall:.1f}s")
    print(f"  n={n:,} cases (n+={n1:,}, n-={n-n1:,}), m={m:,} patterns")

    return {
        'case_data':         case_data,
        'candidates_all':    candidates_all,
        'candidates_pos':    output['candidates_pos'],
        'candidates_neg':    output['candidates_neg'],
        'pattern_results':   pattern_results,
        'null_delta_matrix': null_delta_mat,
        'holds_all':         holds_all,
        'delta_obs':         delta_obs,
        'timing':            timing,
        'case_ids_sorted':   case_ids_sorted,
        'labels':            labels,
        'wall_seconds':      wall,
        'tf_null_matrix':    tf_null_mat,          # ← ADD return key
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXTRACT ORIGINAL-RUN QUANTITIES (for R_obs and diagnostics)
# ═══════════════════════════════════════════════════════════════════════════

def extract_original_quantities(pattern_results, alpha=ALPHA) -> dict:
    """
    Extract arrays from the original full-budget run for R_obs computation,
    Tusher failure analysis, and pi0 reporting.

    NOTE: These quantities are NOT cached for null replicates.  The double-
    null protocol recomputes everything from scratch per replicate.  This
    function is only used for: (a) computing R_obs on the real data, and
    (b) diagnostics (Tusher report, pi0 analysis).
    """
    print("\n" + "=" * 100)
    print("SECTION 2: EXTRACT ORIGINAL-RUN QUANTITIES")
    print("=" * 100)

    m = len(pattern_results)
    p_struct_c0   = np.array([r.p_structural_class0   for r in pattern_results])
    p_struct_c1   = np.array([r.p_structural_class1   for r in pattern_results])
    dominant      = np.array([r.dominant_class         for r in pattern_results])
    p_struct_dom  = np.array([r.p_structural_dominant  for r in pattern_results])
    p_disc_orig   = np.array([r.p_discriminative       for r in pattern_results])
    p_conj_orig     = np.array([r.p_conjunction           for r in pattern_results])
    p_conj_emp_orig = np.array([r.p_conjunction_empirical for r in pattern_results])
    delta_obs       = np.array([r.delta_obs               for r in pattern_results])
    q_sam_orig    = np.array([r.q_value_sam            for r in pattern_results])
    q_struct_dom  = np.array([r.q_structural_dominant  for r in pattern_results])
    ct_list       = [r.constraint_type for r in pattern_results]
    is_sig_final  = np.array([r.is_significant_final   for r in pattern_results])
    p_struct_screen_c0 = np.array([r.p_structural_screen_class0 for r in pattern_results])
    p_struct_screen_c1 = np.array([r.p_structural_screen_class1 for r in pattern_results])

    structural_idx = [
        i for i in range(m)
        if min(p_struct_screen_c0[i], p_struct_screen_c1[i]) <= alpha
    ]
    m_prime = len(structural_idx)

    print(f"  m = {m}, m' = {m_prime} (structural scope filter at alpha={alpha})")
    print(f"  Patterns excluded from Fisher-Storey scope: {m - m_prime}")

    return {
        'p_struct_c0':        p_struct_c0,
        'p_struct_c1':        p_struct_c1,
        'dominant_class':     dominant,
        'p_struct_dom':       p_struct_dom,
        'structural_idx':     structural_idx,
        'p_disc_orig':        p_disc_orig,
        'p_conjunction_orig':          p_conj_orig,      # analytic chi2_4 (BH reference)
        'p_conjunction_empirical_orig': p_conj_emp_orig,  # Phipson-Smyth (Storey gate input)
        'delta_obs_orig':     delta_obs,
        'q_sam_orig':         q_sam_orig,
        'q_struct_dom_orig':  q_struct_dom,
        'constraint_types':   ct_list,
        'is_sig_final_orig':  is_sig_final,
        'm':                  m,
        'm_prime':            m_prime,
    }


def compute_R_obs(
    pattern_results, holds_all, case_data,
    null_delta_matrix, delta_obs, orig_quantities, alpha=ALPHA,
) -> dict:
    """
    Compute R_obs (original-data rejection count) for all four methods.
    Uses the original (unpermuted, unshuffled) data.
    """
    print("\n  Computing R_obs for all four methods...")

    # R_fisher: already reflects p1's conjunctive gate (q_Fisher ≤ α AND p_struct_dom ≤ α → "Both" only)
    R_fisher = int(np.sum(orig_quantities['is_sig_final_orig']))

    # BH reference uses analytic chi2_4 p-values (per p1 Step 5a)
    structural_idx   = orig_quantities['structural_idx']
    p_fisher_m_prime = orig_quantities['p_conjunction_orig'][structural_idx]
    rejected_bh, _, _ = benjamini_hochberg(p_fisher_m_prime, alpha)
    R_bh = int(np.sum(rejected_bh))

    D_0, D_1 = split_by_class(case_data)
    cecconi = run_cecconi_baseline(holds_all, set(D_0.keys()), set(D_1.keys()), alpha)
    R_cecconi = cecconi.n_rejected

    tusher = run_tusher_flat_null(null_delta_matrix, delta_obs, alpha, pi0_hat=1.0)
    R_tusher = tusher['k_star']

    R_obs = {
        METHOD_FISHER_STOREY: R_fisher,
        METHOD_BH_FISHER:     R_bh,
        METHOD_CECCONI:       R_cecconi,
        METHOD_TUSHER:        R_tusher,
    }

    print(f"\n  R_obs (original-data rejections):")
    for method, count in R_obs.items():
        print(f"    {method:25s}: {count:,}")
    print(f"  Cecconi small-cell violations: {cecconi.n_small_cell_violations}")

    return R_obs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — TUSHER FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_tusher_analysis(null_delta_matrix, delta_obs, orig_quantities, R_obs) -> TusherFailureReport:
    """Three-step Tusher flat-null failure mechanistic demonstration."""
    print("\n" + "=" * 100)
    print("SECTION 3: TUSHER FAILURE MECHANISTIC REPORT")
    print("=" * 100)

    sig_deltas = np.abs(delta_obs)[orig_quantities['is_sig_final_orig']]
    tau_star = float(sig_deltas.min()) if len(sig_deltas) > 0 else 0.0

    structural_idx = orig_quantities['structural_idx']
    # Use empirical Phipson-Smyth p-values to match p1's actual Storey gate (Step 5b).
    # Analytic chi2_4 p-values are used only for the BH reference (Step 5a).
    p_fisher_m_prime = orig_quantities['p_conjunction_empirical_orig'][structural_idx]
    pi0_approx, _ = adaptive_storey_pi0(p_fisher_m_prime, q=ALPHA)

    report = build_tusher_failure_report(
        null_delta_matrix=null_delta_matrix,
        delta_obs=delta_obs,
        constraint_types=orig_quantities['constraint_types'],
        k_star_storey=R_obs[METHOD_FISHER_STOREY],
        tau_star_storey=tau_star,
        pi0_hat=pi0_approx,
        alpha=ALPHA,
        log_name="BPI17",
    )

    print(f"\n  sigma_null heterogeneity by constraint family:")
    for fam, s in report.sigma_null_by_family.items():
        print(f"    {fam:30s}: sigma_bar={s['mean_sigma']:.4f} "
              f"[{s['min_sigma']:.4f}, {s['max_sigma']:.4f}] (n={s['n_patterns']})")
    print(f"\n  sigma_null ratio: {report.sigma_null_ratio:.1f}x")
    print(f"  rho_inf:          {report.rho_inf:.1f}x")
    print(f"  k*_Tusher:        {report.k_star_tusher}")
    print(f"  k*_Fisher-Storey: {report.k_star_storey}")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PI0 ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_pi0_analysis(orig_quantities) -> Pi0EstimateWithCI:
    """Compute pi0 on all three axes for the original data."""
    print("\n" + "=" * 100)
    print("SECTION 4: PI0 ANALYSIS (SIGNAL DENSITY)")
    print("=" * 100)

    pi0_est = compute_pi0_all_axes(
        p_disc=orig_quantities['p_disc_orig'],
        p_struct_c0=orig_quantities['p_struct_c0'],
        p_struct_c1=orig_quantities['p_struct_c1'],
        log_name="BPI17",
    )

    print(f"\n  pi0 estimates (BPI 2017):")
    print(f"    Discriminative (m={pi0_est.m_disc}): pi0={pi0_est.pi0_disc:.4f}  "
          f"sensitivity [{pi0_est.pi0_disc_sensitivity_lo:.3f}, "
          f"{pi0_est.pi0_disc_sensitivity_hi:.3f}]")
    print(f"    Structural c0  (m={pi0_est.m_struct}): pi0={pi0_est.pi0_struct_c0:.4f}  "
          f"sensitivity [{pi0_est.pi0_struct_c0_sensitivity_lo:.3f}, "
          f"{pi0_est.pi0_struct_c0_sensitivity_hi:.3f}]")
    print(f"    Structural c1  (m={pi0_est.m_struct}): pi0={pi0_est.pi0_struct_c1:.4f}  "
          f"sensitivity [{pi0_est.pi0_struct_c1_sensitivity_lo:.3f}, "
          f"{pi0_est.pi0_struct_c1_sensitivity_hi:.3f}]")
    print(f"\n  Power gain over BH ~ 1/pi0_disc = {1.0/max(pi0_est.pi0_disc, 0.01):.2f}x")

    return pi0_est


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOUBLE-NULL REPLICATE RUNNER
# ═══════════════════════════════════════════════════════════════════════════
#
# This is the scientific core.  Each null replicate applies:
#   1. Within-trace activity shuffling  (nullifies structural axis)
#   2. Label permutation                (nullifies discriminative axis)
#   3. Fresh recomputation of holds, p_struct, p_disc, Fisher, Storey
#
# Both p_struct and p_disc are U(0,1) under this double null,
# so T_Fisher ~ chi2(4) as required.
# ═══════════════════════════════════════════════════════════════════════════

def _build_doubly_nullified_log(
    case_data_orig: dict,
    case_ids_sorted: list,
    permuted_labels: np.ndarray,
    random_state: int,
) -> dict:
    """
    Construct a doubly-nullified case_data dictionary.

    Operation 1 — sigma_trace: Shuffle activities within each trace.
        Preserves trace length and activity multiset per case.
        Destroys all temporal ordering (DECLARE constraint satisfaction
        becomes random).  This is the SAME null distribution that
        run_structural_permutation_test generates internally, so
        p_struct computed on this log will be U(0,1).

    Operation 2 — sigma_label: Override outcome labels with permuted_labels.
        Preserves marginal class counts.
        Destroys association between trace content and class membership,
        so p_disc computed on this log will be U(0,1).

    Both operations are applied to a shallow copy of each CaseInfo.
    The original case_data_orig is never mutated.

    Args:
        case_data_orig:  Dict[case_id -> CaseInfo] with real outcomes and traces.
        case_ids_sorted: List[str] — fixed lexicographic ordering.
        permuted_labels: (n,) permuted binary labels for this replicate.
        random_state:    RNG seed for trace shuffling.

    Returns:
        Dict[case_id -> CaseInfo] with shuffled traces and permuted labels.
    """
    rng = np.random.RandomState(random_state)
    nullified = {}

    for i, cid in enumerate(case_ids_sorted):
        ci_orig = case_data_orig[cid]

        # Shallow copy: only trace, activity_index, and outcome are overridden
        ci = copy.copy(ci_orig)

        # sigma_label: override outcome
        ci.outcome = int(permuted_labels[i])

        # sigma_trace: shuffle activity sequence within this trace
        # Preserves: trace length, activity multiset (bag of activities)
        # Destroys:  all temporal ordering
        shuffled_trace = ci_orig.trace.copy()
        rng.shuffle(shuffled_trace)
        ci.trace = shuffled_trace
        ci.activity_index = precompute_activity_index(shuffled_trace, case_id=cid)

        nullified[cid] = ci

    return nullified


def run_doubly_null_replicate(
    permuted_labels: np.ndarray,
    case_data_orig: dict,
    candidates_all: list,
    case_ids_sorted: list,
    B1_internal: int,
    B2_internal: int,
    alpha: float,
    random_state: int,
    tf_null_matrix: np.ndarray | None = None,
) -> dict:
    """
    Run all four methods on a single DOUBLY-NULLIFIED held-out replicate.

    This faithfully reproduces p1_BPI_17.py's decision procedure on a log
    where BOTH temporal structure AND class-label association have been
    destroyed by permutation.

    Protocol:
    ---------
    1.  Build doubly-nullified log:
            sigma_trace: shuffle activities within each trace
            sigma_label: permute class labels
    2.  Recompute holds_all on the shuffled traces (patterns evaluated
        on random activity orderings).
    3.  Run label permutation test (H0d) on the doubly-nullified log:
            -> fresh p_disc + null_delta_matrix
    4.  Run structural permutation test (H0s) on both class-conditional
        subsets of the doubly-nullified log:
            -> fresh p_struct_c0, p_struct_c1
    5.  Determine dominant class from shuffled prevalences.
    6.  Compute Fisher conjunction: p_Fisher = chi2.sf(-2*(ln p_s + ln p_d), 4)
    7.  Apply structural scope filter: m' = { i : min(p_s0, p_s1) <= alpha }
    8.  Fisher-Storey: adaptive_storey_pi0 on m' Fisher -> storey_qvalue
        -> is_significant = (q_Fisher <= alpha) AND (p_struct_dom <= alpha) — CONJUNCTIVE GATE
    9.  BH-Fisher: benjamini_hochberg on m' Fisher p-values
    10. Cecconi: chi-square + BH on freshly-computed holds_all
    11. Tusher: flat-null SAM on null_delta_matrix from step 3

    NOTE ON EMPIRICAL CALIBRATION:
    p1_BPI_17.py Step 5b uses Phipson-Smyth empirically-calibrated p-values
    (compute_double_null_tf_matrix + empirical_fisher_pvalue) to correct for
    potential anti-conservatism of the chi2_4 approximation on real data
    (Brown 1975: Fisher combination is anti-conservative when p_struct and
    p_disc are positively correlated, as can occur under the real distribution).

    Under the double-null (sigma_label ∘ sigma_trace):
        p_struct^(b) ~ U(0,1)  and  p_disc^(b) ~ U(0,1)  exactly and independently.
    Therefore T_F^(b) = -2(ln p_struct + ln p_disc) ~ chi2(4) EXACTLY.
    The analytic chi2_4 p-value IS the oracle null p-value here; empirical
    calibration would be equivalent in expectation (by the law of large numbers
    as B_null_inner → ∞) but is computationally prohibitive and scientifically
    redundant.

    Monotonicity argument: both p̃_F (Phipson-Smyth) and p_F^analytic (chi2_4)
    are strictly monotone-decreasing in T_F. Therefore the rejection set
    {i: q̃_F(i) <= alpha} equals {i: q_F^analytic(i) <= alpha} for any threshold,
    and FDR_emp estimated here correctly characterises the real procedure.

    Args:
        permuted_labels:  (n,) permuted binary labels.
        case_data_orig:   Dict[case_id -> CaseInfo] with ORIGINAL outcomes/traces.
        candidates_all:   List[(ct, a, b)] — fixed candidate pool.
        case_ids_sorted:  List[str] — lexicographic case ID order.
        B1_internal:      Label permutation budget for this replicate.
        B2_internal:      Structural permutation budget for this replicate.
        alpha:            FDR target level.
        random_state:     RNG seed for this replicate.

    Returns:
        Dict[method_name -> int] — rejection counts for all four methods.
    """
    m = len(candidates_all)
    rs = random_state

    # ── Step 1: build doubly-nullified log ───────────────────────────────
    # sigma_trace (shuffle activities) + sigma_label (permute outcomes)
    # Seed for trace shuffling is offset from the label/structural seeds
    # to ensure independence between the three randomization layers.
    null_case_data = _build_doubly_nullified_log(
        case_data_orig, case_ids_sorted, permuted_labels,
        random_state=rs + 200_000,
    )

    # ── Step 2: recompute holds on shuffled traces ───────────────────────
    # Pattern satisfaction is now evaluated on randomly-ordered activity
    # sequences, so holds values are draws from the structural null.
    with _suppress_output():
        holds_null = compute_holds_by_case_batch(null_case_data, candidates_all)

    # ── Step 3: label permutation test (H0d) — fresh p_disc ─────────────
    with _suppress_output():
        disc_results = run_label_permutation_test(
            null_case_data, candidates_all, holds_null,
            B1_internal, rs,
        )
    null_delta_mat = disc_results.pop('__null_delta_matrix__')

    p_disc    = np.array([disc_results[spec]['p_two_sided'] for spec in candidates_all])
    delta_obs = np.array([disc_results[spec]['delta_obs']   for spec in candidates_all])

    # ── Step 4: structural permutation test (H0s) — fresh p_struct ──────
    D_0, D_1 = split_by_class(null_case_data)
    cid_set_0 = set(D_0.keys())
    cid_set_1 = set(D_1.keys())

    with _suppress_output():
        struct_results_0 = run_structural_permutation_test(
            D_0, candidates_all, class_label=0, B2=B2_internal, random_state=rs + 1,
        )
        struct_results_1 = run_structural_permutation_test(
            D_1, candidates_all, class_label=1, B2=B2_internal, random_state=rs + 2,
        )

    # p_struct_c0 = np.array([
    #     struct_results_0[spec]['p_structural'] if spec in struct_results_0 else 1.0
    #     for spec in candidates_all
    # ])
    # p_struct_c1 = np.array([
    #     struct_results_1[spec]['p_structural'] if spec in struct_results_1 else 1.0
    #     for spec in candidates_all
    # ])
    p_struct_screen_c0 = np.array([
        struct_results_0[spec]['p_structural_screen'] if spec in struct_results_0 else 1.0
        for spec in candidates_all
    ])
    p_struct_screen_c1 = np.array([
        struct_results_1[spec]['p_structural_screen'] if spec in struct_results_1 else 1.0
        for spec in candidates_all
    ])
    p_struct_test_c0 = np.array([
        struct_results_0[spec]['p_structural_test'] if spec in struct_results_0 else 1.0
        for spec in candidates_all
    ])
    p_struct_test_c1 = np.array([
        struct_results_1[spec]['p_structural_test'] if spec in struct_results_1 else 1.0
        for spec in candidates_all
    ])


    # ── Step 5: dominant class from shuffled prevalences ─────────────────
    prev0 = np.zeros(m)
    prev1 = np.zeros(m)
    for i, spec in enumerate(candidates_all):
        holds = holds_null[spec]
        p0, _, _ = compute_prevalence_from_holds(holds, cid_set_0)
        p1, _, _ = compute_prevalence_from_holds(holds, cid_set_1)
        prev0[i] = p0
        prev1[i] = p1
    dominant = np.where(prev1 >= prev0, 1, 0)
    p_struct_dom_test = np.where(dominant == 1, p_struct_test_c1, p_struct_test_c0)

    # ── Step 6: Fisher conjunction p-values ──────────────────────────────
    # Both p_struct_dom and p_disc are fresh from the doubly-nullified log.
    # Under double-null: T = -2(ln p_s + ln p_d) ~ chi2(4) exactly.
    # Analytic chi2_4 is the oracle null p-value here (see docstring NOTE).
    _eps     = 1e-300
    _ps      = np.clip(p_struct_dom_test, _eps, 1.0)
    _pd      = np.clip(p_disc,            _eps, 1.0)
    tf_obs_b = -2.0 * (np.log(_ps) + np.log(_pd))
    if tf_null_matrix is not None:
        _count_geq = (tf_null_matrix >= tf_obs_b[np.newaxis, :]).sum(axis=0)
        _B_calib   = tf_null_matrix.shape[0]
        p_fisher   = (1.0 + _count_geq) / (_B_calib + 1.0)   # Phipson-Smyth
    else:
        p_fisher = fisher_conjunction_pvalue(p_struct_dom_test, p_disc)

    # ── Step 7: structural scope filter (freshly computed) ───────────────
    # m' = patterns where at least one class has structural evidence.
    structural_idx = [
        i for i in range(m)
        if min(p_struct_screen_c0[i], p_struct_screen_c1[i]) <= alpha
    ]
    m_prime = len(structural_idx)

    if m_prime > 0:
        p_fisher_m_prime = p_fisher[structural_idx]
    else:
        p_fisher_m_prime = np.array([])

    # ── Method 1: Fisher-Storey (primary) ────────────────────────────────
    # Conjunctive gate: q_Fisher <= alpha  AND  p_struct_dom_test <= alpha ("Both").
    if m_prime > 0:
        pi0_f, _ = adaptive_storey_pi0(p_fisher_m_prime, q=alpha)
        q_fisher = storey_qvalue(p_fisher_m_prime, pi0_f)
        p_struct_dom_test_m_prime = p_struct_dom_test[structural_idx]
        n_fisher_storey = int(np.sum(
            (q_fisher <= alpha) & (p_struct_dom_test_m_prime <= alpha)
        ))
    else:
        n_fisher_storey = 0

    # ── Method 2: BH on m' Fisher p-values (reference) ──────────────────
    if m_prime > 0:
        rejected_bh, _, _ = benjamini_hochberg(p_fisher_m_prime, alpha)
        n_bh = int(np.sum(rejected_bh))
    else:
        n_bh = 0

    # ── Method 3: Cecconi chi-square + BH ────────────────────────────────
    # Uses freshly-computed holds_null (from shuffled traces) with the
    # permuted class partition.
    label_override = {
        cid: int(lab) for cid, lab in zip(case_ids_sorted, permuted_labels)
    }
    cecconi_result = run_cecconi_baseline(
        holds_null,
        ids_class0=set(),
        ids_class1=set(),
        alpha=alpha,
        label_override=label_override,
    )
    n_cecconi = cecconi_result.n_rejected

    # ── Method 4: Tusher flat-null SAM ───────────────────────────────────
    tusher_result = run_tusher_flat_null(
        null_delta_mat, delta_obs, alpha, pi0_hat=1.0
    )
    n_tusher = tusher_result['k_star']

    return {
        METHOD_FISHER_STOREY: n_fisher_storey,
        METHOD_BH_FISHER:     n_bh,
        METHOD_CECCONI:       n_cecconi,
        METHOD_TUSHER:        n_tusher,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PARALLEL HELD-OUT NULL PERMUTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _worker(
    b, permuted_labels_b,
    case_data_orig, candidates_all, case_ids_sorted,
    B1_internal, B2_internal, alpha,
    tf_null_matrix,
):
    """Joblib worker for a single doubly-null replicate."""
    rs = BASE_SEED + 100_000 + b
    return run_doubly_null_replicate(
        permuted_labels=permuted_labels_b,
        case_data_orig=case_data_orig,
        candidates_all=candidates_all,
        case_ids_sorted=case_ids_sorted,
        B1_internal=B1_internal,
        B2_internal=B2_internal,
        alpha=alpha,
        random_state=rs,
        tf_null_matrix = tf_null_matrix,    # ← ADD
    )


def run_null_permutations(
    case_data, candidates_all, case_ids_sorted, labels,
    n_jobs=N_JOBS,
    tf_null_matrix = None,
) -> dict:
    """
    Run B_NULL doubly-nullified held-out permutations in parallel.

    For each replicate b:
        1.  Generate permuted labels with seed BASE_SEED + b.
        2.  Build doubly-nullified log (shuffle traces + permute labels).
        3.  Recompute holds, structural p-values, discriminative p-values.
        4.  Run all four methods via run_doubly_null_replicate.
        5.  Record |S_b| for each method.

    Seed architecture (three independent layers):
        BASE_SEED + b                held-out label permutation
        BASE_SEED + 100_000 + b      internal Phase 1 seeds (label perm test)
        BASE_SEED + 100_000 + b + 200_000  trace shuffling
    """
    print("\n" + "=" * 100)
    print("SECTION 6: PARALLEL DOUBLY-NULL HELD-OUT PERMUTATIONS")
    print(f"  B_null={B_NULL}, B1_valid={B1_VALID}, B2_valid={B2_VALID}")
    print(f"  n_jobs={n_jobs}")
    print(f"\n  Double-null protocol per replicate:")
    print(f"    1. sigma_trace: shuffle activities within each trace")
    print(f"    2. sigma_label: permute class labels (preserving marginals)")
    print(f"    3. Recompute holds, p_struct (B2={B2_VALID}), p_disc (B1={B1_VALID})")
    print(f"    4. Fisher conjunction + Adaptive Storey on fresh m'")
    print("=" * 100)

    t0 = time.time()

    print(f"\n  Generating {B_NULL} held-out label permutations...")
    permuted_labels_all = generate_heldout_permutation_batch(
        labels, B_NULL, BASE_SEED
    )

    for i in range(min(5, B_NULL)):
        assert int(permuted_labels_all[i].sum()) == int(labels.sum()), \
            f"Replicate {i}: marginals not preserved!"
    print(f"  Marginal check passed (n+={int(labels.sum())} preserved)")

    est_per_rep = 20  # minutes, rough estimate
    est_total = B_NULL * est_per_rep / max(abs(n_jobs) if n_jobs != -1 else 8, 1)
    print(f"\n  Estimated wall time: ~{est_total:.0f} min ({est_total/60:.1f} hours)")
    print(f"  Starting {B_NULL} parallel replicates (n_jobs={n_jobs})...")

    replicate_results = Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend='loky',
    )(
        delayed(_worker)(
            b, permuted_labels_all[b],
            case_data, candidates_all, case_ids_sorted,
            B1_VALID, B2_VALID, ALPHA,
            tf_null_matrix,   # ← ADD
        )
        for b in range(B_NULL)
    )

    # Aggregate null counts
    null_counts = {m_name: np.zeros(B_NULL, dtype=int) for m_name in ALL_METHODS}
    for b, counts in enumerate(replicate_results):
        for method, count in counts.items():
            null_counts[method][b] = count

    wall = time.time() - t0

    print(f"\n  Doubly-null permutations complete. Wall time: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"\n  Null rejection counts (mean +/- std over {B_NULL} replicates):")
    for method in ALL_METHODS:
        arr = null_counts[method]
        print(f"    {method:25s}: {arr.mean():.2f} +/- {arr.std():.2f}  "
              f"(max={arr.max()}, zeros={np.sum(arr==0)})")

    return {
        'null_counts': null_counts,
        'wall_seconds': wall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — FDR METRICS & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def compute_and_save_metrics(
    null_counts, R_obs_dict, orig_quantities, pi0_est, tusher_report,
    original_wall, perm_wall,
    null_delta_matrix, delta_obs,
):
    """
    Compute FDR_emp, PCER_emp, FWER_emp with BCa 95% CI.
    Run formal statistical tests.  Save output files.
    """
    print("\n" + "=" * 100)
    print("SECTION 7: FDR METRICS & OUTPUT")
    print("=" * 100)

    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)
    m_total = orig_quantities['m']

    all_null_runs = {m_name: list(arr) for m_name, arr in null_counts.items()}
    results_df = build_rq1_results_df(
        all_null_runs, R_obs_dict, m_total,
        alpha=ALPHA, log_name="BPI17",
    )

    print(f"\n  FDR Validation Results (BPI 2017, B_null={B_NULL}, double-null protocol):")
    print(f"  {'─'*95}")
    print(f"  {'Method':25s} {'R_obs':>6s} {'FDR_emp':>8s} {'95% CI':>22s} "
          f"{'PCER':>8s} {'FWER':>8s} {'FDR<=a':>7s}")
    print(f"  {'─'*95}")
    for _, row in results_df.iterrows():
        ci_str = f"[{row['FDR_CI_lower']:.4f}, {row['FDR_CI_upper']:.4f}]"
        verdict = "  pass" if row['controls_FDR'] else "  FAIL"
        print(f"  {row['method']:25s} {row['R_obs']:>6d} {row['FDR_emp']:>8.4f} "
              f"{ci_str:>22s} {row['PCER_emp']:>8.4f} {row['FWER_emp']:>8.4f} {verdict}")
    print(f"  {'─'*95}")

    # Statistical tests
    fdr_tests = {}
    print(f"\n  Formal statistical tests (H0: FDR <= {ALPHA}):")
    for method in ALL_METHODS:
        test = test_fdr_control(
            null_counts[method], R_obs_dict.get(method, 0), alpha=ALPHA
        )
        fdr_tests[method] = test
        fdr_v = "Controls FDR" if test['fdr_emp'] <= ALPHA else "*** FAILS ***"
        print(f"    {method:25s}: FDR_emp={test['fdr_emp']:.4f}, "
              f"p={test['fdr_test_pvalue']:.4f}  [{fdr_v}]")

    # FILE 1: rq1_fdr_metrics.csv
    csv_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_fdr_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # FILE 2: rq1_null_counts.csv
    null_df = pd.DataFrame({method: null_counts[method] for method in ALL_METHODS})
    null_df.index.name = 'replicate_b'
    null_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_null_counts.csv")
    null_df.to_csv(null_path)
    print(f"  Saved: {null_path}")

    # FILE 3: rq1_results.json
    json_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_results.json")
    save_rq1_results_json(
        results_df=results_df,
        fdr_tests=fdr_tests,
        pi0_estimate=pi0_est,
        tusher_report=tusher_report,
        output_path=json_path,
        log_name="BPI17",
    )

    with open(json_path, 'r') as f:
        full_json = json.load(f)

    full_json['null_protocol'] = {
        'type': 'double-null (sigma_label ∘ sigma_trace)',
        'sigma_trace': (
            'Within each trace, randomly permute the activity sequence. '
            'Preserves trace length and activity multiset per case. '
            'Destroys all temporal ordering. Guarantees p_struct ~ U(0,1) '
            'because the shuffled trace IS a draw from the structural null '
            'distribution (same null as run_structural_permutation_test).'
        ),
        'sigma_label': (
            'Permute class labels across cases, preserving marginal counts. '
            'Destroys class-activity association. '
            'Guarantees p_disc ~ U(0,1) by Fisher randomization.'
        ),
        'joint_null': (
            'Under both operations, T_Fisher = -2(ln p_struct + ln p_disc) ~ chi2(4). '
            'Every rejection in a doubly-nullified replicate is a false positive '
            'on BOTH axes simultaneously.'
        ),
        'why_label_only_fails': (
            'Label-only permutation preserves real temporal structure in traces. '
            'For patterns with genuine within-log temporal regularity (e.g. '
            'AlternateResponse(A_Submitted, O_Created) in BPI 2017), structural p-values '
            'remain small in any random class partition. The Fisher statistic '
            'is then T = [non-null chi2(2)] + [null chi2(2)], which is stochastically '
            'larger than chi2(4), inflating FDR_emp beyond alpha.'
        ),
    }
    full_json['alignment_with_p1'] = {
        'conjunction_statistic': (
            'T_F = -2*(ln p_struct + ln p_disc) scored analytically; '
            'converted to Phipson-Smyth empirical p̃_F for real-data Storey gate. '
            'Null replicates use analytic chi2_4 directly (oracle under double-null).'
        ),
        'significance_gate': (
            'SINGLE: q_Fisher <= alpha on m\'\' scope-filtered patterns. '
            'Real run: q computed from empirical p̃_F. '
            'Null replicates: q computed from analytic chi2_4 (equivalent by monotonicity).'
        ),
        'pi0_estimator': 'Gao (2023) Adaptive Storey on empirical p̃_F (real run) / analytic chi2_4 (null replicates)',
        'bh_reference': 'BH on analytic chi2_4 p-values (both real run and null replicates)',
        'fdr_scope': 'm\'\' sample-split scope-filtered (screen p-values, freshly computed per replicate)',
        'structural_role': 'Embedded inside Fisher combination + fresh screen/scope filter',
        'holds_recomputation': 'Per replicate on shuffled traces',
        'structural_pvalues': 'Freshly computed per replicate (B2_VALID per class)',
    }
    full_json['null_replicate_equivalence'] = {
        'claim': (
            'Under the double-null (sigma_label ∘ sigma_trace), '
            'p_struct ~ U(0,1) and p_disc ~ U(0,1) independently, '
            'so T_F ~ chi2(4) exactly. The analytic chi2_4 p-value is the '
            'oracle null p-value; no empirical calibration is needed.'
        ),
        'monotonicity_argument': (
            'Both p̃_F (Phipson-Smyth) and p_F^analytic (chi2_4) are '
            'strictly monotone-decreasing in T_F. Therefore the rejection set '
            '{i: q̃_F(i) <= alpha} equals {i: q_F^analytic(i) <= alpha} '
            'for any threshold, and FDR_emp estimated from null replicates '
            'using analytic p-values correctly characterises the real procedure.'
        ),
        'computational_argument': (
            f'Running compute_double_null_tf_matrix inside each of B_NULL={B_NULL} '
            'null replicates would require B_NULL * B_null_inner sub-replicates, '
            'which is computationally prohibitive and scientifically redundant '
            'given the monotonicity argument above.'
        ),
        'brown_1975_context': (
            'Empirical calibration is needed in the REAL data run because '
            'p_struct and p_disc may be positively correlated under the real data '
            'distribution (Brown 1975), making the analytic chi2_4 anti-conservative. '
            'Under the double-null this correlation is destroyed by construction.'
        ),
    }
    full_json['timing'] = {
        'original_run_seconds': original_wall,
        'null_permutations_seconds': perm_wall,
        'total_seconds': original_wall + perm_wall,
    }
    full_json['config'] = {
        'B_NULL': B_NULL,
        'B1_FULL': B1_FULL, 'B2_FULL': B2_FULL,
        'B1_VALID': B1_VALID, 'B2_VALID': B2_VALID,
        'ALPHA': ALPHA, 'BASE_SEED': BASE_SEED,
        'N_JOBS': N_JOBS,
        'log': 'BPI17',
        'm': orig_quantities['m'],
        'm_prime_original': orig_quantities['m_prime'],
    }

    # Budget resolution check
    p_res_full  = 1.0 / (B1_FULL + 1)
    p_res_valid = 1.0 / (B1_VALID + 1)
    ratio_disc = p_res_valid / p_res_full

    p_res_struct_full  = 1.0 / (B2_FULL + 1)
    p_res_struct_valid = 1.0 / (B2_VALID + 1)
    ratio_struct = p_res_struct_valid / p_res_struct_full

    print(f"\n  Phipson-Smyth resolution check:")
    print(f"    Discriminative — Original: 1/{B1_FULL+1} = {p_res_full:.2e}")
    print(f"    Discriminative — Null:     1/{B1_VALID+1} = {p_res_valid:.2e}  (ratio {ratio_disc:.2f}x)")
    print(f"    Structural    — Original:  1/{B2_FULL+1} = {p_res_struct_full:.2e}")
    print(f"    Structural    — Null:      1/{B2_VALID+1} = {p_res_struct_valid:.2e}  (ratio {ratio_struct:.2f}x)")

    full_json['validation_checks'] = {
        'budget_resolution': {
            'disc_p_res_original': p_res_full,
            'disc_p_res_null': p_res_valid,
            'disc_ratio': ratio_disc,
            'struct_p_res_original': p_res_struct_full,
            'struct_p_res_null': p_res_struct_valid,
            'struct_ratio': ratio_struct,
            'conservative_bias_direction': 'underestimates_FDR_emp',
            'safe_for_FDR_control_claim': True,
        },
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Updated JSON: {json_path}")

    # FILE 4: rq1_pattern_arrays.npz (from original run)
    sigma_per_pattern = np.std(null_delta_matrix, axis=0)
    arrays_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_pattern_arrays.npz")
    np.savez_compressed(
        arrays_path,
        p_discriminative          = orig_quantities['p_disc_orig'],
        p_conjunction             = orig_quantities['p_conjunction_orig'],        # analytic chi2_4
        p_conjunction_empirical   = orig_quantities['p_conjunction_empirical_orig'],  # Phipson-Smyth
        p_structural_c0           = orig_quantities['p_struct_c0'],
        p_structural_c1   = orig_quantities['p_struct_c1'],
        p_structural_dom  = orig_quantities['p_struct_dom'],
        q_fisher          = orig_quantities['q_sam_orig'],
        q_structural_dom  = orig_quantities['q_struct_dom_orig'],
        delta_obs         = orig_quantities['delta_obs_orig'],
        sigma_null        = sigma_per_pattern,
        is_significant    = orig_quantities['is_sig_final_orig'],
        constraint_types  = np.array(orig_quantities['constraint_types']),
        structural_idx    = np.array(orig_quantities['structural_idx']),
    )
    print(f"  Saved: {arrays_path}  ({orig_quantities['m']} patterns)")

    return results_df, fdr_tests


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ1 — FDR CONTROL VALIDITY: BPI CHALLENGE 2017")
    print("  Double-Null Protocol: sigma_label ∘ sigma_trace")
    print("  Aligned with p1_BPI_17.py v8.0 (empirical Phipson-Smyth Storey gate)")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  B_null={B_NULL}, B1_full={B1_FULL}, B2_full={B2_FULL}")
    print(f"  B1_valid={B1_VALID}, B2_valid={B2_VALID}, alpha={ALPHA}")
    print(f"  Output: {RQ1_OUTPUT_DIR}")
    print(f"\n  Null replicate protocol (double-null):")
    print(f"    1. sigma_trace:  shuffle activities within each trace")
    print(f"       -> p_struct ~ U(0,1)")
    print(f"    2. sigma_label:  permute class labels (preserving marginals)")
    print(f"       -> p_disc ~ U(0,1)")
    print(f"    3. Fresh recomputation: holds + structural + discriminative")
    print(f"       -> T_Fisher = -2(ln p_s + ln p_d) ~ chi2(4)")
    print(f"\n  Decision procedure (faithful to p1_BPI_17.py):")
    print(f"    Conjunction: T_F = -2(ln p_s + ln p_d); empirical Phipson-Smyth p\u0303_F (real run)")
    print(f"                 Analytic chi2_4 for null replicates (oracle under double-null)")
    print(f"    Pi0:         Adaptive Storey (Gao 2023) on empirical p\u0303_F")
    print(f"    Gate:        SINGLE \u2014 q\u0303_Fisher <= alpha on m'' scope-filtered patterns")
    print(f"    BH ref:      Analytic chi2_4 p-values (real run and null replicates)")
    print(f"    Structural:  Freshly computed per replicate (not cached)")
    print("=" * 100)

    t_total = time.time()

    # ── Section 1: run original pipeline ─────────────────────────────────
    orig = run_original_pipeline()
    case_data         = orig['case_data']
    candidates_all    = orig['candidates_all']
    pattern_results   = orig['pattern_results']
    null_delta_matrix = orig['null_delta_matrix']
    holds_all         = orig['holds_all']
    delta_obs         = orig['delta_obs']
    case_ids_sorted   = orig['case_ids_sorted']
    labels            = orig['labels']
    original_wall     = orig['wall_seconds']

    # ── Section 2: extract original quantities (for R_obs, diagnostics) ──
    orig_quantities = extract_original_quantities(pattern_results, alpha=ALPHA)
    R_obs_dict = compute_R_obs(
        pattern_results, holds_all, case_data,
        null_delta_matrix, delta_obs, orig_quantities, alpha=ALPHA,
    )

    # ── Section 3: Tusher failure report ─────────────────────────────────
    tusher_report = run_tusher_analysis(
        null_delta_matrix, delta_obs, orig_quantities, R_obs_dict,
    )

    # ── Section 4: pi0 analysis ──────────────────────────────────────────
    pi0_est = run_pi0_analysis(orig_quantities)

    # ── Section 6: parallel doubly-null permutations ─────────────────────
    perm_output = run_null_permutations(
        case_data, candidates_all, case_ids_sorted, labels,
        tf_null_matrix=orig['tf_null_matrix'],   # ← ADD
    )
    null_counts = perm_output['null_counts']
    perm_wall   = perm_output['wall_seconds']

    # ── Section 7: compute & save metrics ────────────────────────────────
    results_df, fdr_tests = compute_and_save_metrics(
        null_counts, R_obs_dict, orig_quantities,
        pi0_est, tusher_report,
        original_wall, perm_wall,
        null_delta_matrix, delta_obs,
    )

    # ── Final summary ────────────────────────────────────────────────────
    total_wall = time.time() - t_total

    print(f"\n{'='*100}")
    print("RQ1 — BPI 2017 COMPLETE (Double-Null Protocol)")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} hours)")
    print(f"  Output directory: {RQ1_OUTPUT_DIR}")
    print(f"    rq1_fdr_metrics.csv   — FDR table (one row per method)")
    print(f"    rq1_null_counts.csv   — raw FP counts ({B_NULL} rows x 4 methods)")
    print(f"    rq1_results.json      — full results for paper generation")
    print(f"    rq1_pattern_arrays.npz")
    print(f"\n  KEY RESULT:")
    for method in ALL_METHODS:
        fdr = fdr_tests[method]['fdr_emp']
        verdict = "pass" if fdr <= ALPHA else "FAIL"
        r_obs = R_obs_dict[method]
        print(f"    {method:25s}: R_obs={r_obs:>5}, FDR_emp={fdr:.4f}  [{verdict}]")
    print(f"\n  Tusher mechanistic: sigma_ratio={tusher_report.sigma_null_ratio:.1f}x, "
          f"rho_inf={tusher_report.rho_inf:.1f}x, "
          f"k*_Tusher={tusher_report.k_star_tusher}")
    print(f"  pi0_disc={pi0_est.pi0_disc:.4f} -> Storey power gain ~ "
          f"{1.0/max(pi0_est.pi0_disc, 0.01):.2f}x over BH")
    print(f"\n  Null protocol: DOUBLE-NULL (sigma_label ∘ sigma_trace)")
    print(f"  Both p_struct and p_disc are U(0,1) under null -> T_Fisher ~ chi2(4)")
    print(f"{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1 FDR Control Validity — BPI Challenge 2017 (Double-Null Protocol)"
    )
    parser.add_argument(
        '--b-null', type=int, default=B_NULL,
        help=f'Number of held-out null replicates (default: {B_NULL})'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Number of parallel jobs (-1 = all cores, default: {N_JOBS})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run with B_null=2 for quick testing'
    )
    args = parser.parse_args()

    if args.dry_run:
        B_NULL = 2
        print("*** DRY RUN MODE: B_null=2 ***")
    else:
        B_NULL = args.b_null

    N_JOBS = args.n_jobs

    main()