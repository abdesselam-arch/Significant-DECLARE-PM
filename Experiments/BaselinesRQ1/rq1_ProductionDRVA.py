#!/usr/bin/env python3
"""
rq1_ProductionDRVA.py  —  RQ1 FDR Control Validity: DRVA Baseline — Production
================================================================================
Doubly-Null Empirical FDR Evaluation of DRVA (Cecconi et al. BPM Forum 2021)

METHOD
------
DRVA performs a permutation test on ΔConfidence (shuffleLog = label
permutation on pre-cached trace evaluations) per candidate rule r ∈ M_all.
No FDR correction is applied — each rule is evaluated at the per-rule raw
α threshold.

WHAT DRVA'S NULL NULLIFIES
---------------------------
shuffleLog permutes pre-cached TRACE-LEVEL evaluations between groups.
Because evaluations come from the original (unshuffled) traces, temporal
structure within each trace is preserved across all π iterations.

    DRVA nullifies:         H₀ᵈ (discriminative axis — class–trace association)
    DRVA does NOT nullify:  H₀ˢ (structural axis — within-trace ordering)

Patterns with genuine temporal structure therefore inflate the null
distribution of |ΔConf|, driving empirical FDR above α under the
doubly-null protocol. This is the main mechanistic finding demonstrated.

DOUBLY-NULL PROTOCOL  (Pellegrina & Vandin 2018, adapted)
----------------------------------------------------------
Each held-out replicate b applies two independent operations:

    Null_b = σ_label ∘ σ_trace

    1. σ_trace:  Randomly permute activity sequence within each trace.
                 Preserves trace length and activity multiset per case.
                 Destroys all temporal ordering.
                 → p_struct^(b) ~ U(0,1) by Fisher randomisation.

    2. σ_label:  Permute class labels across cases (marginals preserved).
                 Destroys any class–trace association.
                 → p_disc^(b) ~ U(0,1) by Fisher randomisation.

    Every rejection on L^(b) is a false positive by construction.

EMPIRICAL FDR ESTIMATOR
------------------------
    FDR_emp = E[V_b] / max(R_obs, 1)        Pellegrina & Vandin (2018)

MECHANISTIC PREDICTION
-----------------------
FDR_emp >> α. No multiple-testing correction: under the null DRVA rejects
≈ α × m patterns per replicate. If R_obs is not proportionally larger,
FDR_emp = E[V_b] / R_obs >> α.

OUTPUT FILES
------------
    rq1_drva_fdr_metrics.csv     One row (FDR_emp, CI, pass/fail).
    rq1_drva_null_counts.csv     B_null rows of V_b counts.
    rq1_drva_results.json        Full results for paper generation.

Version : 1.0  (DRVA-only; doubly-null; α=0.05)
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

# ── Data loading and trace utilities (shared DECLARE semantics) ───────────
from P1_SDSM.p1_Production_hou import (
    load_and_preprocess_data,
    generate_candidate_patterns,
    compute_holds_by_case_batch,
    precompute_activity_index,
    CaseInfo,
    INPUT_FILE        as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
)

# ── DRVA baseline ─────────────────────────────────────────────────────────
from BaselinesRQ1.DRVA_Production import (
    run_drva,
    run_drva_on_doubly_null_log,
    DRVA_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH        = P1_INPUT_FILE
PHASE0_JSON     = P1_SPEC_FILE
RQ1_OUTPUT_DIR  = "RQ1_Production_DRVA"

# ── FDR target level ───────────────────────────────────────────────────────
ALPHA_DRVA = 0.05    # per-rule significance level (no FDR correction)

# ── Permutation budgets ────────────────────────────────────────────────────
PI_DRVA_FULL  = 2_000   # DRVA permutation iterations on real data
PI_DRVA_VALID = 200     # reduced budget per held-out null replicate (speed)

# ── Held-out null replicates ───────────────────────────────────────────────
B_NULL = 200

# ── Seed architecture ─────────────────────────────────────────────────────
#   Held-out σ_label:   BASE_SEED + b
#   σ_trace shuffle:    BASE_SEED + b + 200_000
#   DRVA internal:      BASE_SEED + b + 50_000
BASE_SEED = 20260521

assert B_NULL < 100_000, (
    f"B_NULL={B_NULL} ≥ 100,000 would cause seed-layer overlap."
)

# ── Parallelism ────────────────────────────────────────────────────────────
N_JOBS = -1


# ═══════════════════════════════════════════════════════════════════════════
# INLINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _generate_heldout_permutation_batch(
    labels: np.ndarray,
    B: int,
    base_seed: int,
) -> np.ndarray:
    """
    Generate B held-out label permutations preserving marginal class counts.
    Replicate b uses seed (base_seed + b).
    Returns (B, n) int8 array of permuted label vectors.
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
    """BCa bootstrap confidence interval."""
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

    FDR_emp  = E[V_b] / max(R_obs, 1)
    PCER_emp = E[V_b] / m_total
    FWER_emp = Pr[V_b > 0]
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


@contextlib.contextmanager
def _suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> dict:
    """
    Load the Production event log and build the shared M_all candidate pool.
    Returns case_data, candidates_all, labels, case_ids_sorted.
    """
    print("\n" + "=" * 100)
    print("SECTION 1 — DATA LOADING")
    print("=" * 100)

    case_data = load_and_preprocess_data(CSV_PATH)

    candidates_pos, candidates_neg = generate_candidate_patterns(case_data)
    pos_set        = set(candidates_pos)
    candidates_all = list(candidates_pos) + [p for p in candidates_neg if p not in pos_set]

    case_ids_sorted = sorted(case_data.keys())
    labels          = np.array([case_data[cid].outcome for cid in case_ids_sorted])
    n = len(labels)
    n1 = int(labels.sum())

    print(f"\n  n={n:,}  (n1={n1:,}, n0={n-n1:,})  M_all={len(candidates_all):,}")

    return {
        'case_data':       case_data,
        'candidates_all':  candidates_all,
        'case_ids_sorted': case_ids_sorted,
        'labels':          labels,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DRVA ORIGINAL-DATA RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_drva_original(
    case_data: dict,
    candidates_all: list,
) -> dict:
    """
    Run DRVA on the real Production data with the shared M_all candidate pool.

    Hierarchical simplification is DISABLED (hierarchical_pruning=False)
    and pre-processing thresholds are set to zero (mmin=0, mdiff_min=0)
    so that M_tested = M_all exactly.
    """
    print("\n" + "=" * 100)
    print("SECTION 2 — DRVA ORIGINAL-DATA RUN")
    print(f"  π={PI_DRVA_FULL:,}  α_DRVA={ALPHA_DRVA}"
          f"  hierarchical_pruning=False  mmin=0  mdiff_min=0")
    print("=" * 100)

    t0  = time.time()
    cfg = DRVA_CONFIG.copy()
    cfg['pi']                   = PI_DRVA_FULL
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
    drva_out['wall_seconds'] = wall

    print(f"\n  DRVA complete: {wall:.1f}s")
    print(f"  M_all={drva_out['m_all']:,}  M_tested={drva_out['m_tested']:,}  "
          f"R_obs(Cecconi p≤{ALPHA_DRVA})={drva_out['n_rejected_cecconi']:,}")

    return drva_out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DOUBLY-NULL LOG BUILDER
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
    σ_trace: randomly permute activities within each trace.
             Preserves trace length and activity multiset per case.
             Destroys all temporal ordering.

    Shallow copies of CaseInfo are created; case_data_orig is never mutated.
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
# SECTION 4 — PARALLEL DOUBLY-NULL HELD-OUT PERMUTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _null_worker(
    b,
    permuted_labels_b,
    case_data_orig,
    candidates_all,
    case_ids_sorted,
    pi_drva_valid,
    alpha_drva,
):
    """Joblib top-level worker for one doubly-null replicate (loky-safe)."""
    rs             = BASE_SEED + b
    null_case_data = _build_doubly_nullified_log(
        case_data_orig, case_ids_sorted, permuted_labels_b,
        random_state=rs + 200_000,
    )

    with _suppress_output():
        holds_null = compute_holds_by_case_batch(null_case_data, candidates_all)

    drva_cfg_null = DRVA_CONFIG.copy()
    drva_cfg_null['pi']                   = pi_drva_valid
    drva_cfg_null['alpha']                = alpha_drva
    drva_cfg_null['hierarchical_pruning'] = False
    drva_cfg_null['mmin']                 = 0.0
    drva_cfg_null['mdiff_min']            = 0.0

    with _suppress_output():
        n_drva = run_drva_on_doubly_null_log(
            null_case_data = null_case_data,
            candidates_all = candidates_all,
            alpha          = alpha_drva,
            replicate_seed = rs + 50_000,
            config         = drva_cfg_null,
            holds_all      = holds_null,
        )

    return n_drva


def run_null_permutations(
    case_data: dict,
    candidates_all: list,
    case_ids_sorted: list,
    labels: np.ndarray,
    pi_drva_valid: int = PI_DRVA_VALID,
    n_jobs: int = N_JOBS,
) -> dict:
    """
    Run B_NULL doubly-nullified held-out replicates in parallel (DRVA only).

    Seed architecture (non-overlapping layers, safe for B_NULL < 100,000):
        BASE_SEED + b              held-out σ_label permutation
        BASE_SEED + b + 200_000    σ_trace activity shuffling
        BASE_SEED + b + 50_000     DRVA internal shuffleLog
    """
    print("\n" + "=" * 100)
    print("SECTION 4 — PARALLEL DOUBLY-NULL HELD-OUT PERMUTATIONS  (DRVA)")
    print(f"  B_null={B_NULL}  PI_DRVA_valid={pi_drva_valid}  n_jobs={n_jobs}")
    print(f"\n  Per-replicate protocol:")
    print(f"    1. σ_trace: shuffle activities within each trace → p_struct ~ U(0,1)")
    print(f"    2. σ_label: permute class labels (marginals preserved) → p_disc ~ U(0,1)")
    print(f"\n  DRVA: holds fast path + shuffleLog (π={pi_drva_valid})")
    print(f"        per-rule p_Cecconi ≤ {ALPHA_DRVA}  (no FDR correction)")
    print("=" * 100)

    t0 = time.time()

    print(f"\n  Generating {B_NULL} held-out label permutations (seed={BASE_SEED})...")
    permuted_labels_all = _generate_heldout_permutation_batch(labels, B_NULL, BASE_SEED)
    for i in range(min(5, B_NULL)):
        assert int(permuted_labels_all[i].sum()) == int(labels.sum()), \
            f"Replicate {i}: class marginals not preserved"
    print(f"  Marginal check passed  (n+={int(labels.sum()):,} preserved)")

    print(f"\n  Launching {B_NULL} parallel workers (n_jobs={n_jobs})...\n")

    null_counts_list = Parallel(
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
            pi_drva_valid,
            ALPHA_DRVA,
        )
        for b in range(B_NULL)
    )

    null_counts = np.array(null_counts_list, dtype=int)
    wall        = time.time() - t0

    print(f"\n  All {B_NULL} replicates complete  |  wall={wall:.1f}s ({wall/60:.1f} min)")
    print(f"\n  DRVA null rejection counts V_b  (mean ± std over {B_NULL} replicates):")
    print(f"    mean={null_counts.mean():.2f}  std={null_counts.std():.2f}"
          f"  max={null_counts.max()}  zeros={int(np.sum(null_counts == 0))}")

    return {
        'null_counts':  null_counts,
        'wall_seconds': wall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FDR METRICS AND OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def compute_and_save_metrics(
    null_counts: np.ndarray,
    R_obs: int,
    m_total: int,
    original_wall: float,
    perm_wall: float,
) -> dict:
    """
    Compute FDR_emp, PCER_emp, FWER_emp with BCa 95% CI and save output files.
    """
    print("\n" + "=" * 100)
    print("SECTION 5 — FDR METRICS AND OUTPUT  (DRVA)")
    print("=" * 100)

    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    metrics = _compute_fdr_metrics(null_counts, R_obs, m_total, ALPHA_DRVA)

    ci_str  = f"[{metrics['FDR_CI_lower']:.4f}, {metrics['FDR_CI_upper']:.4f}]"
    verdict = "PASS" if metrics['controls_FDR'] else "FAIL"

    print(f"\n  FDR Results (Production DRVA, B_null={B_NULL}, double-null protocol):")
    print(f"  {'─'*80}")
    print(f"  {'Method':20s} {'α':>5s} {'R_obs':>6s} {'E[V_b]':>8s} "
          f"{'FDR_emp':>8s} {'95% CI':>22s} {'FWER':>7s} {'Pass?':>6s}")
    print(f"  {'─'*80}")
    print(f"  {'DRVA':20s} {ALPHA_DRVA:>5.3f} {R_obs:>6d} "
          f"{metrics['E_V_b']:>8.2f} {metrics['FDR_emp']:>8.4f} "
          f"{ci_str:>22s} {metrics['FWER_emp']:>7.4f}  {verdict}")
    print(f"  {'─'*80}")

    # ── FILE 1: rq1_drva_fdr_metrics.csv ─────────────────────────────────
    df       = pd.DataFrame([{'method': 'DRVA', **metrics}])
    csv_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_drva_fdr_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── FILE 2: rq1_drva_null_counts.csv ─────────────────────────────────
    null_df            = pd.DataFrame({'V_b': null_counts})
    null_df.index.name = 'replicate_b'
    null_path          = os.path.join(RQ1_OUTPUT_DIR, "rq1_drva_null_counts.csv")
    null_df.to_csv(null_path)
    print(f"  Saved: {null_path}")

    # ── FILE 3: rq1_drva_results.json ────────────────────────────────────
    full_json = {
        'rq1_version': '1.0',
        'method':      'DRVA',
        'log_name':    'Production',
        'timestamp':   datetime.now().isoformat(),

        'experiment_design': {
            'method':        'DRVA — Cecconi et al. BPM Forum 2021',
            'shared_pool':   'M_all from Phase 0 DECLARE spec',
            'null_protocol': (
                'Double-null: σ_label (permute class labels) ∘ σ_trace '
                '(shuffle activities within each trace). '
                'Under the double-null every rejection is a false positive.'
            ),
            'DRVA_gate': f'Per-rule p_Cecconi ≤ {ALPHA_DRVA} (no FDR correction)',
            'DRVA_structural_note': (
                'shuffleLog nullifies H₀ᵈ only. H₀ˢ (within-trace ordering) '
                'is never nullified → FDR_emp > α expected under doubly-null.'
            ),
        },

        'config': {
            'B_NULL':         B_NULL,
            'PI_DRVA_FULL':   PI_DRVA_FULL,
            'PI_DRVA_VALID':  PI_DRVA_VALID,
            'ALPHA_DRVA':     ALPHA_DRVA,
            'BASE_SEED':      BASE_SEED,
            'N_JOBS':         N_JOBS,
            'm_total':        m_total,
        },

        'R_obs':    R_obs,
        'fdr_metrics': metrics,

        'null_replicate_summary': {
            'mean_V_b':   float(null_counts.mean()),
            'std_V_b':    float(null_counts.std()),
            'max_V_b':    int(null_counts.max()),
            'n_zero_V_b': int(np.sum(null_counts == 0)),
        },

        'timing': {
            'original_drva_seconds':     original_wall,
            'null_permutations_seconds': perm_wall,
            'total_seconds':             original_wall + perm_wall,
        },

        'validation_checks': {
            'marginals_preserved': True,
            'seed_layers': {
                'held_out_label': 'BASE_SEED + b',
                'trace_shuffle':  'BASE_SEED + b + 200_000',
                'DRVA_internal':  'BASE_SEED + b + 50_000',
                'no_overlap':     f'B_NULL={B_NULL} < 100_000',
            },
        },
    }

    json_path = os.path.join(RQ1_OUTPUT_DIR, "rq1_drva_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("RQ1 — FDR CONTROL VALIDITY: DRVA BASELINE — PRODUCTION")
    print("Method: DRVA (Cecconi et al. BPM Forum 2021)")
    print("Double-null protocol: σ_label ∘ σ_trace")
    print("=" * 100)
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  B_null={B_NULL}  PI_DRVA_full={PI_DRVA_FULL}  PI_DRVA_valid={PI_DRVA_VALID}")
    print(f"  α_DRVA={ALPHA_DRVA}  (per-rule, no FDR correction)")
    print(f"  Output: {RQ1_OUTPUT_DIR}/")
    print(f"\n  Mechanistic prediction:")
    print(f"    DRVA: FDR_emp >> {ALPHA_DRVA}  (no FDR correction; H₀ˢ not nullified)")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    # ── Section 1: load data and shared M_all ─────────────────────────────
    data            = load_data()
    case_data       = data['case_data']
    candidates_all  = data['candidates_all']
    case_ids_sorted = data['case_ids_sorted']
    labels          = data['labels']

    # ── Section 2: DRVA full-budget run ───────────────────────────────────
    drva_orig     = run_drva_original(case_data, candidates_all)
    R_obs         = int(drva_orig['n_rejected_cecconi'])
    m_total       = int(drva_orig['m_all'])
    original_wall = drva_orig['wall_seconds']

    print(f"\n  R_obs (DRVA, p_Cecconi ≤ {ALPHA_DRVA}): {R_obs:,}")

    # ── Section 4: parallel doubly-null permutations ──────────────────────
    perm_out    = run_null_permutations(
        case_data       = case_data,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        labels          = labels,
        pi_drva_valid   = PI_DRVA_VALID,
        n_jobs          = N_JOBS,
    )
    null_counts = perm_out['null_counts']
    perm_wall   = perm_out['wall_seconds']

    # ── Section 5: FDR metrics and output ────────────────────────────────
    metrics = compute_and_save_metrics(
        null_counts   = null_counts,
        R_obs         = R_obs,
        m_total       = m_total,
        original_wall = original_wall,
        perm_wall     = perm_wall,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    total_wall = time.time() - t_total
    verdict    = "PASS" if metrics['controls_FDR'] else "FAIL"

    print(f"\n{'='*100}")
    print("RQ1 — DRVA PRODUCTION COMPLETE  (double-null protocol)")
    print(f"{'='*100}")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  Output: {RQ1_OUTPUT_DIR}/")
    print(f"\n  KEY RESULT:")
    print(f"  {'Method':20s} {'α':>5s} {'R_obs':>6s} {'E[V_b]':>8s} "
          f"{'FDR_emp':>8s} {'Pass?':>6s}")
    print(f"  {'─'*55}")
    print(f"  {'DRVA':20s} {ALPHA_DRVA:>5.3f} {R_obs:>6d} "
          f"{metrics['E_V_b']:>8.2f} {metrics['FDR_emp']:>8.4f} {verdict:>6s}")
    print(f"{'='*100}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1 FDR Validity — DRVA Baseline, Production (double-null protocol)"
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
        PI_DRVA_VALID = 20
        PI_DRVA_FULL  = 20
        print("*** DRY-RUN MODE: B_null=2, PI_DRVA=20 ***")
    else:
        B_NULL = args.b_null

    assert B_NULL < 100_000, f"B_NULL={B_NULL} ≥ 100,000 would cause seed-layer overlap."

    N_JOBS     = args.n_jobs
    ALPHA_DRVA = args.alpha_drva

    main()