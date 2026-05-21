"""
drva_RTFMP_parallel.py  —  DRVA Baseline: Road Traffic Fine Management Process Log
===================================================================================
Declarative Rules Variant Analysis (DRVA)
Cecconi, Augusto & Di Ciccio (BPM Forum 2021)

PURPOSE
-------
Faithful Python re-implementation of the DRVA permutation test
(Algorithm 1, Cecconi et al. 2021) on the RTFMP event log, using the
SAME fixed candidate pool M_all as Phase 1 (P1) and DeclareMiner.

DESIGN PRINCIPLE: SHARED CANDIDATE POOL
-----------------------------------------
The RQ1 experiment compares three testing procedures on the same universe of
hypotheses M_all, so that differences in empirical FDR and rejection counts
reflect only the testing procedure and not differences in candidate scope.

Allowing DRVA's hierarchical simplification would reduce its tested set to
M_DRVA ⊂ M_all, confounding the comparison:

    FDR_emp differences = (different test) + (different candidate scope)

To isolate the test-procedure factor, hierarchical simplification is
DISABLED by default (hierarchical_pruning = False).

The two generic pre-processing steps that do NOT change the candidate universe
are retained as optional filters:

    a.  min-difference:    remove r if |Conf_A − Conf_B| < mdiff_min.
        Default mdiff_min = 0.0  → no-op, M_tested = M_all.
    b.  min-interestingness: remove r if Conf_A < mmin AND Conf_B < mmin.
        Default mmin = 0.0  → no-op, M_tested = M_all.

With both defaults the tested pool equals M_all exactly.

DRVA PERMUTATION TEST  (Algorithm 1, Cecconi et al. 2021)
----------------------------------------------------------
For each rule r ∈ M_tested:
    1.  Ediff(r) = |Conf(r, L_A) − Conf(r, L_B)|.
    2.  Encode traces: m(r, t) ∈ {1.0, 0.0, NaN} (satisfied/violated/vacuous).
    3.  C(r) = 1  (Algorithm 1, line 12 — Phipson-Smyth numerator).
    4.  For i = 1 … π:
            shuffleLog: randomly reassign ALL n encoded trace vectors between
            L_A and L_B, preserving |L_A| and |L_B|.
            If |Conf(r, L_A^i) − Conf(r, L_B^i)| ≥ Ediff(r):  C(r) += 1.
    5.  p_Cecconi(r) = C(r) / π          (Algorithm 1, line 20; min = 1/π)
        p_Phipson(r)  = C(r) / (π + 1)   (Phipson & Smyth 2010; super-uniform)
    6.  Reject r iff p_Cecconi(r) ≤ α    (no FDR correction).

WHAT DRVA'S NULL DOES AND DOES NOT NULLIFY
-------------------------------------------
shuffleLog permutes pre-cached TRACE-LEVEL evaluations between groups.
Because evaluations come from the ORIGINAL (unshuffled) traces, the temporal
structure inside every trace is preserved across all π iterations.

Result: DRVA's null nullifies only H₀ᵈ (discriminative axis).
        H₀ˢ (structural axis / within-trace ordering) is never nullified.
Patterns with genuine temporal structure therefore inflate the null
distribution of |ΔConf|, driving empirical FDR above α under the doubly-null
protocol.  This is the main mechanistic finding demonstrated by RQ1.

THREE-METHOD HIERARCHY (RQ1)
-----------------------------
    Method        | Discriminative signal        | Structural | Multiple-testing
    --------------|------------------------------|------------|------------------
    DeclareMiner  | Δsupp ≥ τ  (hard threshold)  | None       | None
    DRVA          | Perm. p-val on ΔConf ≤ α     | None       | None (raw α)
    P1 (Ours)     | Hou T on p_disc               | p_struct   | Adaptive Storey

All three operate on the same M_all.  Each row adds exactly one layer of
statistical rigour over the previous.

DEFAULT PARAMETERS
------------------
    π                    = 1000    (§3.3 paper default)
    α                    = 0.01    (§3.5 paper default)
    mmin                 = 0.0     (no-op → M_tested = M_all)
    mdiff_min            = 0.0     (no-op → M_tested = M_all)
    hierarchical_pruning = False   (DISABLED for RQ1 shared-pool design)

OUTPUT FILES
------------
    drva_results.json               All tested rules with p-values.
    drva_significant_patterns.json  Rejected rules only.
    drva_report.txt                 Ranked human-readable output.

Version : 2.1  (hierarchical pruning disabled; shared M_all design; replicate-specific seeds in null loop)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References:
    Cecconi, Augusto & Di Ciccio (2021). Detection of Statistically Significant
        Differences Between Process Variants Through Declarative Rules.
        BPM Forum 2021, LNBIP 427, pp. 73–91.
    Phipson & Smyth (2010). Permutation p-values should never be zero.
        Stat. Appl. Genet. Mol. Biol. 9(1):Art. 39.
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
from tqdm import tqdm

# ─── PATH SETUP ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# All DECLARE evaluation primitives imported from Phase 1.
# Guarantees identical constraint semantics across P1, DRVA, DeclareMiner.
from P1_SDSM.p1_RTFMP_hou import (
    load_and_preprocess_data,
    generate_candidate_patterns,
    split_by_class,
    evaluate_pattern_fast,
    CaseInfo,
    INPUT_FILE        as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
    ALL_CONSTRAINT_TYPES,
)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

CSV_PATH          = P1_INPUT_FILE
DECLARE_SPEC_FILE = P1_SPEC_FILE
OUTPUT_DIR        = "DRVA_RTFMP"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRVA_CONFIG = {
    'pi':                   1000,   # permutation iterations (π)
    'alpha':                0.01,   # per-rule significance level α
    'mmin':                 0.0,    # minimum interestingness  (0 = no-op)
    'mdiff_min':            0.0,    # minimum difference       (0 = no-op)
    'hierarchical_pruning': False,  # DISABLED — preserves M_all for RQ1
    'random_state':         42,
}


# ─── TRACE-LEVEL ENCODING ────────────────────────────────────────────────────

def encode_log(
    case_data: Dict[str, CaseInfo],
    candidates: List[Tuple[str, str, Optional[str]]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Cache m(r, t) for every (rule, trace) pair.

    m(r, t) ∈ {1.0 (satisfied), 0.0 (violated), NaN (vacuous)}.
    Result shape: (n_traces, m_rules), dtype float32.
    Row order: sorted case IDs (deterministic).

    The key property enabling DRVA's shuffleLog: m(r, t) depends only on
    the trace t, not on which group t belongs to.  Reassigning traces to
    groups never requires re-evaluating DECLARE semantics — only the group
    mean changes.

    Returns:
        enc:            (n, m) float32 encoding matrix.
        case_id_order:  List[str] of case IDs in row order.
    """
    case_id_order = sorted(case_data.keys())
    n = len(case_id_order)
    m = len(candidates)

    enc = np.full((n, m), np.nan, dtype=np.float32)

    for t_idx, cid in enumerate(tqdm(case_id_order, desc="Encoding traces")):
        case = case_data[cid]
        for r_idx, (ct, a, b) in enumerate(candidates):
            result = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
            if result is not None:
                enc[t_idx, r_idx] = float(result)

    n_vacuous = int(np.isnan(enc).sum())
    print(f"   Encoding: {n} traces × {m} rules  |  "
          f"vacuous: {n_vacuous:,} ({n_vacuous / (n * m) * 100:.1f}%)")
    return enc, case_id_order


def _log_confidence(enc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Log-level Confidence for the trace subset selected by `mask`.

    Conf(r, L) = nanmean of enc[mask, r].
    Returns 0.0 for any rule where every trace in the subset is vacuous.

    Args:
        enc:   (n, m) float32.
        mask:  (n,) bool.

    Returns:
        (m,) float64.
    """
    subset = enc[mask].astype(np.float64)
    with np.errstate(all='ignore'):
        conf = np.nanmean(subset, axis=0)
    return np.where(np.isnan(conf), 0.0, conf)


# ─── GENERIC PRE-PROCESSING ──────────────────────────────────────────────────

def apply_generic_pruning(
    candidates: List[Tuple],
    conf_A: np.ndarray,
    conf_B: np.ndarray,
    ediff: np.ndarray,
    mdiff_min: float,
    mmin: float,
) -> Tuple[List[Tuple], np.ndarray]:
    """
    Apply the two scope-preserving DRVA pre-processing filters.

    These filters do not change the candidate universe (M_all remains the
    FDR denominator); they only remove rules carrying zero information.
    With default values (both 0.0) neither filter removes anything and
    M_tested = M_all exactly.

    Criterion 1 — min-difference (§3.2 Cecconi et al.):
        Remove r if |Conf_A − Conf_B| < mdiff_min.

    Criterion 2 — min-interestingness (§3.2 Cecconi et al.):
        Remove r if Conf_A < mmin AND Conf_B < mmin.

    Hierarchical simplification is NEVER applied here.

    Returns:
        candidates_kept: List[(ct, a, b)].
        keep_idx:        (k,) int array of surviving indices into `candidates`.
    """
    m = len(candidates)
    keep = np.ones(m, dtype=bool)

    if mdiff_min > 0.0:
        removed = int((ediff < mdiff_min).sum())
        keep &= (ediff >= mdiff_min)
        print(f"   min-difference  (mdiff_min={mdiff_min}): "
              f"removed {removed}, {int(keep.sum())} remain")
    else:
        print(f"   min-difference:      DISABLED (mdiff_min=0 → no-op)")

    if mmin > 0.0:
        uninteresting = (conf_A < mmin) & (conf_B < mmin)
        removed = int(uninteresting.sum())
        keep &= ~uninteresting
        print(f"   min-interestingness (mmin={mmin}): "
              f"removed {removed}, {int(keep.sum())} remain")
    else:
        print(f"   min-interestingness: DISABLED (mmin=0 → no-op)")

    print(f"   hierarchical simplification: DISABLED (shared M_all design)")

    keep_idx = np.where(keep)[0]
    return [candidates[i] for i in keep_idx], keep_idx


# ─── DRVA PERMUTATION TEST ───────────────────────────────────────────────────

def run_drva_permutation_test(
    enc: np.ndarray,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    col_idx: np.ndarray,
    ediff_obs: np.ndarray,
    pi: int,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """
    DRVA permutation test (Algorithm 1, lines 12–21, Cecconi et al. 2021).

    For each iteration i = 1 … π:
        shuffleLog: ALL n trace vectors are randomly assigned to L_A (n_A
        draws without replacement) and L_B (remaining).  Group sizes are
        preserved exactly.
        Log-level Confidence is recomputed from the shuffled assignment.
        C(r) is incremented whenever the permuted |ΔConf| ≥ observed Ediff(r).

    The inner loop is fully vectorised over all m_tested rules — no Python
    loop over rules inside the permutation loop.

    Args:
        enc:         (n, m_full) float32 — full encoding matrix.
        mask_A:      (n,) bool — rows belonging to L_A (class 1).
        mask_B:      (n,) bool — rows belonging to L_B (class 0).
        col_idx:     (m_tested,) int — column indices into enc for tested rules.
        ediff_obs:   (m_tested,) float — observed |Conf_A − Conf_B|.
        pi:          int — permutation iterations.
        random_state: int — RNG seed.

    Returns:
        dict:
            'p_cecconi':  (m_tested,) float — C(r)/π.
            'p_phipson':  (m_tested,) float — C(r)/(π+1).
            'counts':     (m_tested,) int.
            'null_diffs': (π, m_tested) float32 — all permuted |ΔConf| values.
    """
    n_total  = enc.shape[0]
    n_A      = int(mask_A.sum())
    m_tested = len(col_idx)

    # Extract and prepare the tested-rule submatrix once.
    enc_sub  = enc[:, col_idx].astype(np.float64)    # (n, m_tested)
    valid    = ~np.isnan(enc_sub)                     # (n, m_tested) bool
    enc_fill = np.where(valid, enc_sub, 0.0)          # NaN → 0 for masked sums
    # Pre-multiply once to avoid recomputation per iteration.
    valid_enc = enc_fill * valid                      # (n, m_tested)

    # Counters initialised at 1 per Algorithm 1 line 12 (Phipson-Smyth numerator).
    counts     = np.ones(m_tested, dtype=np.int64)
    null_diffs = np.zeros((pi, m_tested), dtype=np.float32)

    all_idx = np.arange(n_total)
    rng     = np.random.RandomState(random_state)

    print(f"\n   Running {pi:,} permutation iterations  "
          f"(n_A={n_A}, n_B={n_total - n_A}, m_tested={m_tested:,})...")

    for i in tqdm(range(pi), desc="DRVA permutation"):
        # shuffleLog: randomly draw n_A row indices for L_A (no replacement).
        perm_A_idx       = rng.choice(all_idx, size=n_A, replace=False)
        perm_A_mask      = np.zeros(n_total, dtype=bool)
        perm_A_mask[perm_A_idx] = True
        perm_B_mask      = ~perm_A_mask

        # Vectorised Confidence computation over all m_tested rules at once.
        cnt_A = valid[perm_A_mask].sum(axis=0).astype(np.float64)   # (m_tested,)
        cnt_B = valid[perm_B_mask].sum(axis=0).astype(np.float64)

        conf_A_i = np.where(
            cnt_A > 0,
            valid_enc[perm_A_mask].sum(axis=0) / np.maximum(cnt_A, 1.0),
            0.0,
        )
        conf_B_i = np.where(
            cnt_B > 0,
            valid_enc[perm_B_mask].sum(axis=0) / np.maximum(cnt_B, 1.0),
            0.0,
        )

        diff_i           = np.abs(conf_A_i - conf_B_i)
        null_diffs[i]    = diff_i.astype(np.float32)
        counts          += (diff_i >= ediff_obs).astype(np.int64)

    p_cecconi = counts.astype(np.float64) / pi          # Algorithm 1, line 20
    p_phipson = counts.astype(np.float64) / (pi + 1)    # Phipson & Smyth 2010

    return {
        'p_cecconi':  p_cecconi,
        'p_phipson':  p_phipson,
        'counts':     counts,
        'null_diffs': null_diffs,
    }


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_drva(
    config: Optional[dict] = None,
    case_data: Optional[Dict[str, CaseInfo]] = None,
    candidates_all: Optional[List[Tuple]] = None,
) -> dict:
    """
    Execute the full DRVA pipeline.

    Args:
        config:         Override DRVA_CONFIG parameters.
        case_data:      Pre-loaded case data (skip reload when called from RQ1).
        candidates_all: Pre-built M_all (shared with P1 and DeclareMiner).

    Returns:
        dict — all results and intermediate quantities for RQ1 integration.
    """
    cfg       = {**DRVA_CONFIG, **(config or {})}
    pi        = int(cfg['pi'])
    alpha     = float(cfg['alpha'])
    mmin      = float(cfg['mmin'])
    mdiff_min = float(cfg['mdiff_min'])
    rs        = int(cfg['random_state'])

    timing   = {}
    t0_total = time.time()

    # ── Step 0: Data loading ──────────────────────────────────────────────
    if case_data is None:
        print("\n" + "=" * 100)
        print("DRVA — STEP 0: DATA LOADING")
        print("=" * 100)
        case_data = load_and_preprocess_data(CSV_PATH)

    if candidates_all is None:
        print("\n" + "=" * 100)
        print("DRVA — STEP 1: CANDIDATE GENERATION FROM PHASE 0 SPEC")
        print("=" * 100)
        candidates_pos, candidates_neg = generate_candidate_patterns(case_data)
        pos_set = set(candidates_pos)
        candidates_all = list(candidates_pos) + [
            p for p in candidates_neg if p not in pos_set
        ]

    m_all = len(candidates_all)
    print(f"\n   M_all (fixed pool): {m_all:,} rules")

    # ── Step 1: Variant partition ─────────────────────────────────────────
    D_0, D_1 = split_by_class(case_data)
    n_A = len(D_1)   # L_A = Deviant (class 1)
    n_B = len(D_0)   # L_B = Normal  (class 0)
    n   = n_A + n_B
    print(f"   L_A (Deviant, class 1): {n_A:,}")
    print(f"   L_B (No Credit Collection, class 0): {n_B:,}")

    # ── Step 2: Trace-level encoding ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("DRVA — STEP 2: TRACE-LEVEL ENCODING")
    print("=" * 100)
    t_enc = time.time()
    enc, case_id_order = encode_log(case_data, candidates_all)
    timing['encoding'] = time.time() - t_enc

    mask_A = np.array(
        [case_data[cid].outcome == 1 for cid in case_id_order], dtype=bool
    )
    mask_B = ~mask_A
    assert int(mask_A.sum()) == n_A and int(mask_B.sum()) == n_B

    # ── Step 3: Observed Confidence and reference differences ─────────────
    print("\n" + "=" * 100)
    print("DRVA — STEP 3: LOG-LEVEL CONFIDENCE & EDIFF")
    print("=" * 100)
    conf_A_all = _log_confidence(enc, mask_A)
    conf_B_all = _log_confidence(enc, mask_B)
    ediff_all  = np.abs(conf_A_all - conf_B_all)

    print(f"   Ediff over M_all ({m_all:,} rules):")
    print(f"     mean={ediff_all.mean():.4f}  "
          f"median={np.median(ediff_all):.4f}  "
          f"max={ediff_all.max():.4f}")
    for t in [0.01, 0.05, 0.10, 0.20]:
        print(f"     Ediff >= {t:.2f}: {(ediff_all >= t).sum():,}")

    # ── Step 4: Generic pre-processing ───────────────────────────────────
    print("\n" + "=" * 100)
    print("DRVA — STEP 4: PRE-PROCESSING  (scope-preserving filters only)")
    print("=" * 100)
    t_prune = time.time()
    candidates_tested, keep_idx = apply_generic_pruning(
        candidates_all, conf_A_all, conf_B_all, ediff_all, mdiff_min, mmin
    )
    conf_A_tested = conf_A_all[keep_idx]
    conf_B_tested = conf_B_all[keep_idx]
    ediff_tested  = ediff_all[keep_idx]
    m_tested      = len(candidates_tested)
    timing['pruning'] = time.time() - t_prune

    print(f"\n   M_tested: {m_tested:,}  ({m_all - m_tested:,} removed by filters)")
    if m_tested < m_all:
        ct_counts = Counter(c[0] for c in candidates_tested)
        for ct in ALL_CONSTRAINT_TYPES:
            if ct in ct_counts:
                print(f"     {ct:<30s}: {ct_counts[ct]:,}")

    # ── Step 5: Permutation test ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("DRVA — STEP 5: PERMUTATION TEST  (Algorithm 1, Cecconi et al. 2021)")
    print(f"   π={pi:,}  α={alpha}")
    print(f"   Null: shuffleLog on trace-level encodings (DISCRIMINATIVE axis only).")
    print(f"   Structural axis NOT nullified → FDR > α expected under doubly-null.")
    print("=" * 100)
    t_perm = time.time()
    perm_out = run_drva_permutation_test(
        enc          = enc,
        mask_A       = mask_A,
        mask_B       = mask_B,
        col_idx      = keep_idx,
        ediff_obs    = ediff_tested,
        pi           = pi,
        random_state = rs,
    )
    timing['permutation_test'] = time.time() - t_perm

    p_cecconi = perm_out['p_cecconi']
    p_phipson = perm_out['p_phipson']
    counts    = perm_out['counts']

    # ── Step 6: Significance decision ─────────────────────────────────────
    rejected_cecconi = p_cecconi <= alpha
    rejected_phipson = p_phipson <= alpha
    n_rej_cecconi    = int(rejected_cecconi.sum())
    n_rej_phipson    = int(rejected_phipson.sum())
    direction        = np.where(conf_A_tested >= conf_B_tested, "Positive", "Negative")

    print(f"\n   Rejected (Cecconi p ≤ α={alpha}): {n_rej_cecconi:,}")
    print(f"   Rejected (Phipson p ≤ α={alpha}): {n_rej_phipson:,}")

    timing['total'] = time.time() - t0_total

    # ── Step 7: Assemble per-rule records ──────────────────────────────────
    results = []
    for r_idx, (ct, a, b) in enumerate(candidates_tested):
        pid = f"{ct}_{a}" + (f"_{b}" if b else "")
        results.append({
            'pattern_id':             pid,
            'constraint_type':        ct,
            'activity_a':             a,
            'activity_b':             b,
            'direction':              direction[r_idx],
            'conf_A':                 float(conf_A_tested[r_idx]),
            'conf_B':                 float(conf_B_tested[r_idx]),
            'ediff':                  float(ediff_tested[r_idx]),
            'p_cecconi':              float(p_cecconi[r_idx]),
            'p_phipson':              float(p_phipson[r_idx]),
            'count_C':                int(counts[r_idx]),
            'is_significant_cecconi': bool(rejected_cecconi[r_idx]),
            'is_significant_phipson': bool(rejected_phipson[r_idx]),
        })

    # Sort by Ediff descending (§3.4 DRVA default ranking).
    results.sort(key=lambda x: (-x['ediff'], -max(x['conf_A'], x['conf_B'])))

    print(f"\n   Timing:")
    for k, v in timing.items():
        print(f"     {k:25s}: {v:.1f}s")
    print(f"\n   Summary:")
    print(f"     M_all:              {m_all:,}")
    print(f"     M_tested:           {m_tested:,}")
    print(f"     Rejected (Cecconi): {n_rej_cecconi:,}")
    print(f"     Rejected (Phipson): {n_rej_phipson:,}")

    return {
        'results':             results,
        'rejected_cecconi':    rejected_cecconi,
        'rejected_phipson':    rejected_phipson,
        'n_rejected_cecconi':  n_rej_cecconi,
        'n_rejected_phipson':  n_rej_phipson,
        'candidates_all':      candidates_all,
        'm_all':               m_all,
        'candidates_tested':   candidates_tested,
        'keep_idx':            keep_idx,
        'm_tested':            m_tested,
        'conf_A_all':          conf_A_all,
        'conf_B_all':          conf_B_all,
        'ediff_all':           ediff_all,
        'conf_A_tested':       conf_A_tested,
        'conf_B_tested':       conf_B_tested,
        'ediff_tested':        ediff_tested,
        'p_cecconi':           p_cecconi,
        'p_phipson':           p_phipson,
        'counts':              counts,
        'null_diffs':          perm_out['null_diffs'],
        'enc':                 enc,
        'mask_A':              mask_A,
        'mask_B':              mask_B,
        'config':              cfg,
        'timing':              timing,
        'case_data':           case_data,
    }


# ─── OUTPUT GENERATION ────────────────────────────────────────────────────────

def save_outputs(drva_out: dict) -> None:
    """Save full JSON, significant-only JSON, and text report."""

    cfg           = drva_out['config']
    results       = drva_out['results']
    m_all         = drva_out['m_all']
    m_tested      = drva_out['m_tested']
    n_rej_cecconi = drva_out['n_rejected_cecconi']
    n_rej_phipson = drva_out['n_rejected_phipson']
    timing        = drva_out['timing']
    case_data     = drva_out['case_data']

    sig   = [r for r in results if r['is_significant_cecconi']]
    n_pos = sum(1 for r in sig if r['direction'] == 'Positive')
    n_neg = sum(1 for r in sig if r['direction'] == 'Negative')

    full_json = {
        'framework': 'DRVA — Declarative Rules Variant Analysis',
        'reference': (
            'Cecconi, Augusto & Di Ciccio (2021). '
            'BPM Forum 2021, LNBIP 427, pp. 73–91.'
        ),
        'version':   '2.1',
        'timestamp': datetime.now().isoformat(),
        'scientific_description': {
            'null_hypothesis': (
                'No difference in Confidence between Deviant (Sent for Credit Collection) '
                'and Normal (No Credit Collection) variants for rule r.'
            ),
            'permutation_scheme': (
                'shuffleLog: encoded trace vectors randomly reassigned between '
                'L_A and L_B preserving group sizes. '
                'Equivalent to label permutation on pre-cached evaluations. '
                'Structural axis (within-trace ordering) NOT nullified.'
            ),
            'p_value': (
                'p_Cecconi(r) = C(r)/π  (Algorithm 1 line 20; min = 1/π). '
                'p_Phipson(r) = C(r)/(π+1)  (super-uniform, Phipson & Smyth 2010).'
            ),
            'fdr_note': (
                'No FDR correction. Raw per-rule α threshold. '
                'Structural axis not nullified → empirical FDR > α under doubly-null.'
            ),
            'shared_pool_note': (
                'Hierarchical simplification DISABLED. '
                'M_tested = M_all at default mmin=0, mdiff_min=0. '
                'FDR comparison not confounded by candidate scope differences.'
            ),
        },
        'config': cfg,
        'dataset': {
            'n_total':   len(case_data),
            'n_deviant': sum(1 for c in case_data.values() if c.outcome == 1),
            'n_normal':  sum(1 for c in case_data.values() if c.outcome == 0),
        },
        'summary': {
            'm_all':                m_all,
            'm_tested':             m_tested,
            'n_rejected_cecconi':   n_rej_cecconi,
            'n_rejected_phipson':   n_rej_phipson,
            'n_rejected_positive':  n_pos,
            'n_rejected_negative':  n_neg,
        },
        'timing':           timing,
        'all_tested_rules': results,
        'significant_rules': sig,
    }

    path_full = os.path.join(OUTPUT_DIR, 'drva_results.json')
    with open(path_full, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON (full):        {path_full}")

    path_sig = os.path.join(OUTPUT_DIR, 'drva_significant_patterns.json')
    sig_json = {k: v for k, v in full_json.items() if k != 'all_tested_rules'}
    with open(path_sig, 'w', encoding='utf-8') as f:
        json.dump(sig_json, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON (significant): {path_sig}")

    lines = []
    lines.append("=" * 120)
    lines.append("DRVA — DECLARATIVE RULES VARIANT ANALYSIS  (Cecconi et al. 2021)")
    lines.append("RTFMP: Deviant (Sent for Credit Collection) vs. Normal (No Credit Collection)")
    lines.append("Shared M_all design: hierarchical pruning DISABLED")
    lines.append("=" * 120)
    lines.append(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"π={cfg['pi']:,}  α={cfg['alpha']}  "
        f"mmin={cfg['mmin']}  mdiff_min={cfg['mdiff_min']}  "
        f"hierarchical_pruning=False"
    )
    lines.append("")
    lines.append(f"M_all:              {m_all:,}")
    lines.append(f"M_tested:           {m_tested:,}  (= M_all at default params)")
    lines.append(f"Rejected (Cecconi): {n_rej_cecconi:,}")
    lines.append(f"Rejected (Phipson): {n_rej_phipson:,}")
    lines.append(f"  Positive (Deviant-dominant): {n_pos:,}")
    lines.append(f"  Negative (Normal-dominant):  {n_neg:,}")
    lines.append("")
    lines.append("=" * 120)
    lines.append("TOP SIGNIFICANT RULES — ranked by Ediff descending")
    lines.append("=" * 120)
    lines.append("")

    for rank, r in enumerate(sig[:50], 1):
        lines.append(f"Rank {rank:3d} | {r['pattern_id']}")
        lines.append(f"         Type:   {r['constraint_type']}")
        if r['activity_b']:
            lines.append(f"         Acts:   {r['activity_a']} → {r['activity_b']}")
        else:
            lines.append(f"         Act:    {r['activity_a']}")
        lines.append(f"         Dir:    {r['direction']}")
        lines.append(
            f"         Conf_A={r['conf_A']:.4f}  Conf_B={r['conf_B']:.4f}  "
            f"Ediff={r['ediff']:.4f}  "
            f"p_Cecconi={r['p_cecconi']:.4e}  "
            f"p_Phipson={r['p_phipson']:.4e}  "
            f"C(r)={r['count_C']:,}"
        )
        lines.append("")

    lines.append("=" * 120)
    lines.append("TIMING")
    lines.append("=" * 120)
    for k, v in timing.items():
        lines.append(f"  {k:25s}: {v:.1f}s")

    path_rpt = os.path.join(OUTPUT_DIR, 'drva_report.txt')
    with open(path_rpt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ Text report:        {path_rpt}")


# ─── RQ1 INTEGRATION ─────────────────────────────────────────────────────────

def run_drva_on_doubly_null_log(
    null_case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    alpha: float,
    replicate_seed: int,
    config: Optional[dict] = None,
    holds_all: Optional[Dict] = None,
) -> int:
    """
    Apply DRVA to a pre-built doubly-nullified log; return the rejection count.

    Called from the RQ1 null-replicate loop.  `null_case_data` has already
    had sigma_trace ∘ sigma_label applied externally.  Every rejection here
    is a false positive by construction.

    Generic pruning is re-applied on each null log because Ediff changes
    under the null.  With default thresholds (both 0.0) no rules are removed
    and the tested pool is always the full M_all — keeping the FDR denominator
    consistent across all replicates.

    `replicate_seed` must be distinct for every null replicate (e.g. BASE_SEED + b).
    This ensures the DRVA internal permutation sequences are independent across
    replicates, giving unbiased BCa confidence intervals on the FDR estimate.

    Args:
        null_case_data:  Doubly-nullified case_data.
        candidates_all:  Fixed M_all (same as real-data run).
        alpha:           DRVA per-rule significance level.
        replicate_seed:  RNG seed unique to this replicate (e.g. BASE_SEED + b).
        config:          Optional config overrides.
        holds_all:       Precomputed holds dict from the RQ1 loop (fast path).
                         Keys are specs; values are {case_id: bool/int} dicts.
                         When provided, encode_log is skipped entirely.

    Returns:
        n_rejected: int.
    """
    cfg       = {**DRVA_CONFIG, 'alpha': alpha, **(config or {})}
    pi        = int(cfg['pi'])
    mmin      = float(cfg['mmin'])
    mdiff_min = float(cfg['mdiff_min'])
    rs        = replicate_seed

    if holds_all is not None:
        case_id_order = sorted(null_case_data.keys())
        n = len(case_id_order)
        m = len(candidates_all)
        cid_to_idx = {cid: i for i, cid in enumerate(case_id_order)}
        enc_null = np.full((n, m), np.nan, dtype=np.float32)
        for r_idx, spec in enumerate(candidates_all):
            for cid, val in holds_all.get(spec, {}).items():
                t_idx = cid_to_idx.get(cid)
                if t_idx is not None:
                    enc_null[t_idx, r_idx] = float(val)
    else:
        enc_null, case_id_order = encode_log(null_case_data, candidates_all)

    mask_A = np.array(
        [null_case_data[cid].outcome == 1 for cid in case_id_order], dtype=bool
    )
    mask_B = ~mask_A

    conf_A = _log_confidence(enc_null, mask_A)
    conf_B = _log_confidence(enc_null, mask_B)
    ediff  = np.abs(conf_A - conf_B)

    candidates_tested, keep_idx = apply_generic_pruning(
        candidates_all, conf_A, conf_B, ediff, mdiff_min, mmin
    )
    if len(candidates_tested) == 0:
        return 0

    perm_out = run_drva_permutation_test(
        enc          = enc_null,
        mask_A       = mask_A,
        mask_B       = mask_B,
        col_idx      = keep_idx,
        ediff_obs    = ediff[keep_idx],
        pi           = pi,
        random_state = rs,
    )
    return int((perm_out['p_cecconi'] <= alpha).sum())


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DRVA Baseline — RTFMP  (shared M_all design, v2.1)"
    )
    parser.add_argument('--pi',        type=int,   default=DRVA_CONFIG['pi'])
    parser.add_argument('--alpha',     type=float, default=DRVA_CONFIG['alpha'])
    parser.add_argument('--mmin',      type=float, default=DRVA_CONFIG['mmin'])
    parser.add_argument('--mdiff-min', type=float, default=DRVA_CONFIG['mdiff_min'])
    parser.add_argument('--dry-run',   action='store_true',
                        help="Smoke test: run with π=10")
    args = parser.parse_args()

    config = {
        'pi':                   10 if args.dry_run else args.pi,
        'alpha':                args.alpha,
        'mmin':                 args.mmin,
        'mdiff_min':            args.mdiff_min,
        'hierarchical_pruning': False,
        'random_state':         42,
    }

    print("\n" + "=" * 100)
    print("DRVA — DECLARATIVE RULES VARIANT ANALYSIS  (shared M_all design, v2.1)")
    print("Cecconi, Augusto & Di Ciccio (BPM Forum 2021)")
    print("RTFMP: Deviant (Sent for Credit Collection) vs. Normal (No Credit Collection)")
    print("=" * 100)
    print(f"  π={config['pi']:,}  α={config['alpha']}  "
          f"mmin={config['mmin']}  mdiff_min={config['mdiff_min']}")
    print(f"  hierarchical_pruning=False  (M_all preserved for RQ1)")
    if args.dry_run:
        print("  *** DRY RUN: π=10 ***")
    print("=" * 100)

    drva_out = run_drva(config=config)
    save_outputs(drva_out)

    print(f"\n{'=' * 100}")
    print("DRVA COMPLETE")
    print(f"{'=' * 100}")
    print(f"  M_all:              {drva_out['m_all']:,}")
    print(f"  M_tested:           {drva_out['m_tested']:,}")
    print(f"  Rejected (Cecconi): {drva_out['n_rejected_cecconi']:,}")
    print(f"  Rejected (Phipson): {drva_out['n_rejected_phipson']:,}")
    print(f"  Total time:         {drva_out['timing']['total']:.1f}s")
    print(f"  Output:             {OUTPUT_DIR}/")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()