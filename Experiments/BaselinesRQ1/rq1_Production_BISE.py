#!/usr/bin/env python3
"""
rq1_Production_BISE.py  —  RQ1 FDR Control Validity: Production  [DvM BISE 2025]
==================================================================================
Doubly-Null Empirical FDR Estimation for the DvM BISE 2025 Baseline

METHOD
------
DvM BISE 2025 — Fisher Score + Coverage Selection + Rule Extraction
    Generalized Fisher score (Gu, Li, Han 2011) ranks Declare features by
    discriminative power between deviant and non-deviant classes.
    Greedy coverage selection adds features (in descending Fisher-score order)
    until each trace is "covered" coverage_threshold times.
    RipperK (k grid) and Decision Tree (max_depth grid) are then trained on
    the selected feature submatrix to extract rule sets.

FOUR OUTPUT SETS (single pipeline run per replicate)
-----------------------------------------------------
    sel_features   Fisher+coverage selection (feature names)
    ripper_rules   Rules extracted by best-k RipperK on sel_features submatrix
    dt_rules       Rules extracted by best-depth DT on sel_features submatrix
    ripper+dt      ripper_rules u dt_rules (unique rule strings)

All four sets are computed from ONE DvM evaluation per null replicate — no
re-running for each configuration.

EXPERIMENTAL DESIGN PRINCIPLE
-------------------------------
DvM BISE 2025 operates on a FIXED candidate pool M_all = {declare features
discovered from the original log by Apriori + template instantiation}.
For null replicates, the SAME M_all is encoded on the null log (not
re-discovered from the null log). This is the only valid design: V_b must
count false positives from the same candidate space as R_obs.

DOUBLY-NULL PROTOCOL  (Pellegrina & Vandin 2018, adapted)
----------------------------------------------------------
Each held-out replicate b applies TWO independent operations:

    Null_b = sigma_label composed with sigma_trace

    1. sigma_trace:  Randomly permute activity sequence within each trace.
    2. sigma_label:  Permute class labels across cases (marginals preserved).

    Every selection on L^(b) is a false positive by construction.

EMPIRICAL FDR ESTIMATOR  (Pellegrina & Vandin 2018)
----------------------------------------------------
    FDR_emp(S) = E[V_b^S] / max(R_obs^S, 1)

OUTPUT FILES
------------
    sel_features/
        rq1_dvm_bise2025_fdr_metrics.csv
        rq1_dvm_bise2025_null_counts.csv
        rq1_dvm_bise2025_diagnostics.csv
        rq1_dvm_bise2025_results.json
    rules/
        rq1_dvm_bise2025_fdr_metrics_ripper.csv
        rq1_dvm_bise2025_fdr_metrics_dt.csv
        rq1_dvm_bise2025_fdr_metrics_union.csv
        rq1_dvm_bise2025_null_counts_ripper.csv
        rq1_dvm_bise2025_null_counts_dt.csv
        rq1_dvm_bise2025_null_counts_union.csv
        rq1_dvm_bise2025_results_rules.json

Version : 2.0  (adds ripper_rules / dt_rules / union output sets)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Pellegrina & Vandin (2018/2020). KDD 2018 / TKDD 2020.
Di Francescomarino et al. (2025). BISE 67(6):877-894.
Gu, Li & Han (2011). Generalized Fisher Score for Feature Selection. UAI.
"""

import sys
import os
import copy
import io
import contextlib
import time
import json
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score as _sklearn_f1_score

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# DvM BISE 2025 Production module — all pipeline primitives resolved here
from DvM_Production import (
    DVM_CONFIG            as _DVM_CONFIG_BASE,
    CaseInfo,
    CSV_PATH,
    load_and_preprocess_data,
    build_log_structures,
    build_encoding_matrix,
    compute_generalized_fisher_scores,
    select_features_coverage,
    split_by_class,
    discover_declare_features,
    COVERAGE_THRESHOLD    as DVM_COVERAGE_THRESHOLD,
    SUPPORT_THRESHOLD     as DVM_SUPPORT_THRESHOLD,
    RANDOM_STATE          as DVM_RANDOM_STATE,
    extract_ripper_rules_text,
    extract_dt_rules_text,
    RIPPER_K_GRID,
    DT_DEPTH_GRID,
)

try:
    import wittgenstein as _lw_ripper
    _RIPPER_AVAILABLE = True
except ImportError:
    _lw_ripper = None
    _RIPPER_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

RQ1_OUTPUT_DIR = "RQ1_Production_DvM_BISE2025"

B_NULL    = 200
ALPHA     = 0.05
BASE_SEED = 20260521
N_JOBS    = 4

METHOD_DM   = "BISE2025_DvM"
ALL_METHODS = [METHOD_DM]

DVM_CONFIG = dict(_DVM_CONFIG_BASE)

_RIPPER_PRUNE_SIZE  = 0.33
_RULES_RANDOM_STATE = 42

assert B_NULL < 100_000, (
    f"B_NULL={B_NULL} >= 100,000 would cause seed-layer overlap."
)


# =============================================================================
# INLINE HELPERS
# =============================================================================

def _generate_heldout_permutation_batch(labels, B, base_seed):
    n   = len(labels)
    out = np.empty((B, n), dtype=np.int8)
    for b in range(B):
        rng    = np.random.RandomState(base_seed + b)
        out[b] = rng.permutation(labels).astype(np.int8)
    return out


def _bootstrap_bca_ci(data, stat_fn, B_boot=1000, ci=0.95, seed=42):
    rng   = np.random.RandomState(seed)
    n     = len(data)
    theta = stat_fn(data)
    boot  = np.array([stat_fn(rng.choice(data, n, replace=True)) for _ in range(B_boot)])
    z0    = stats.norm.ppf(float(np.mean(boot < theta)) + 1e-10)
    total = np.sum(data)
    jack  = (total - data) / (n - 1)
    jm    = jack.mean()
    num   = float(np.sum((jm - jack) ** 3))
    den   = float(6.0 * (np.sum((jm - jack) ** 2) ** 1.5))
    a     = num / den if abs(den) > 1e-15 else 0.0
    alpha_tail = (1.0 - ci) / 2.0
    def _adj(z_):
        return stats.norm.cdf(z0 + (z0 + z_) / (1.0 - a * (z0 + z_)))
    lo = float(np.clip(_adj(stats.norm.ppf(alpha_tail)),       0.001, 0.999))
    hi = float(np.clip(_adj(stats.norm.ppf(1.0 - alpha_tail)), 0.001, 0.999))
    return float(np.percentile(boot, lo * 100)), float(np.percentile(boot, hi * 100))


def _compute_fdr_metrics(null_counts, R_obs, m_total, alpha_nominal):
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
        'R_obs': R_obs, 'E_V_b': ev, 'FDR_emp': fdr_emp,
        'PCER_emp': pcer_emp, 'FWER_emp': fwer_emp,
        'FDR_CI_lower': ci_lo, 'FDR_CI_upper': ci_hi,
        'B_null': len(null_counts), 'm_total': m_total,
        'alpha': alpha_nominal, 'controls_FDR': bool(fdr_emp <= alpha_nominal),
    }


@contextlib.contextmanager
def _suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def _fit_rules_on_array(X_sel, y, sel_names):
    """
    Fit RipperK and DecisionTree on X_sel (already restricted to sel_features columns).
    Returns (ripper_rules: List[str], dt_rules: List[str]).
    """
    if len(sel_names) == 0 or len(np.unique(y)) < 2:
        return [], []

    ripper_rules = []
    if _RIPPER_AVAILABLE and _lw_ripper is not None:
        try:
            df = pd.DataFrame(X_sel, columns=sel_names)
            df["label"] = y
            best_f1, best_clf = -1.0, None
            for k in RIPPER_K_GRID:
                try:
                    clf = _lw_ripper.RIPPER(
                        k=k, prune_size=_RIPPER_PRUNE_SIZE,
                        random_state=_RULES_RANDOM_STATE,
                    )
                    clf.fit(df, class_feat="label", pos_class=1)
                    y_pred = np.array(clf.predict(df.drop(columns=["label"])), dtype=int)
                    f1 = float(_sklearn_f1_score(y, y_pred, zero_division=0))
                    if f1 > best_f1:
                        best_f1, best_clf = f1, clf
                except Exception:
                    continue
            if best_clf is not None:
                ripper_rules = extract_ripper_rules_text(best_clf, sel_names)
        except Exception:
            pass

    dt_rules = []
    try:
        best_f1, best_clf = -1.0, None
        for depth in DT_DEPTH_GRID:
            try:
                clf = DecisionTreeClassifier(max_depth=depth, random_state=_RULES_RANDOM_STATE)
                clf.fit(X_sel, y)
                y_pred = clf.predict(X_sel)
                f1 = float(_sklearn_f1_score(y, y_pred, zero_division=0))
                if f1 > best_f1:
                    best_f1, best_clf = f1, clf
            except Exception:
                continue
        if best_clf is not None:
            dt_rules = extract_dt_rules_text(best_clf, sel_names)
    except Exception:
        pass

    return ripper_rules, dt_rules


# =============================================================================
# DECLARE-ONLY STRIPPED PIPELINE  (skips sequential mining, data features, DeclD)
# =============================================================================

def extract_rule_constraint_names_ripper(rules: list) -> frozenset:
    """Distinct Declare feature names appearing in any RIPPER rule condition."""
    names = set()
    for rule in rules:
        for cond in rule.split(" AND "):
            feat = cond.split("=")[0].strip()
            if feat:
                names.add(feat)
    return frozenset(names)


def extract_rule_constraint_names_dt(rules: list) -> frozenset:
    """Distinct Declare feature names appearing in any DT rule path condition."""
    names = set()
    for rule in rules:
        for cond in rule.split(" AND "):
            feat = cond.split(" <= ")[0].split(" > ")[0].strip()
            if feat and feat != "⊤":
                names.add(feat)
    return frozenset(names)


def run_dvm_declare_only(case_data: dict, config: dict = None) -> dict:
    """
    Stripped DvM pipeline for the 'declare' encoding only.

    Skips three wasteful stages present in run_dvm():
        Step 1b  Sequential pattern mining   (O(n x L^2), not used by declare)
        Step 0c  Data feature extraction     (data_matrix unused by declare)
        Step 1d  Data-aware Declare (DeclD)  (only for decld/hybrid encodings)

    Returns dict with keys:
        declare_features, feat_names, sel_features, X_all, fisher_scores,
        DS_ripper, DS_dt, ripper_rules, dt_rules,
        case_data, all_ids, y, positional_log, sequence_log
    """
    cfg      = {**DVM_CONFIG, **(config or {})}
    cov_thr  = int(cfg.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))
    supp     = float(cfg.get("support_threshold", DVM_SUPPORT_THRESHOLD))
    fish_eps = float(cfg.get("fisher_eps", 1e-10))

    D_0, D_1   = split_by_class(case_data)
    ids_class0 = list(D_0.keys())
    ids_class1 = list(D_1.keys())
    all_ids    = ids_class0 + ids_class1
    y_all      = np.array([0] * len(ids_class0) + [1] * len(ids_class1), dtype=np.int32)
    print(f"   L-={len(ids_class0):,}  L+={len(ids_class1):,}  total={len(all_ids):,}", flush=True)

    print("   Step 0b: building log structures...", flush=True)
    positional_log, sequence_log, _ = build_log_structures(case_data)

    print("   Step 1a: discovering Declare features (Apriori + templates)...", flush=True)
    declare_features = discover_declare_features(
        case_ids_pos=ids_class1, case_ids_neg=ids_class0,
        positional_log=positional_log, support_threshold=supp,
    )

    _empty = {
        'declare_features': [], 'feat_names': [], 'sel_features': [],
        'X_all': np.zeros((len(all_ids), 0), dtype=np.float32),
        'fisher_scores': np.zeros(0, dtype=np.float64),
        'DS_ripper': frozenset(), 'DS_dt': frozenset(),
        'ripper_rules': [], 'dt_rules': [],
        'case_data': case_data, 'all_ids': all_ids, 'y': y_all,
        'positional_log': positional_log, 'sequence_log': sequence_log,
    }
    if not declare_features:
        print("   WARNING: no Declare features discovered — returning empty.", flush=True)
        return _empty

    print(f"   M_all = {len(declare_features):,} declare features discovered.", flush=True)
    print("   Step 1c: encoding declare matrix...", flush=True)
    X_all, feat_names = build_encoding_matrix(
        all_ids, positional_log, sequence_log,
        {}, [], {},
        declare_features, [], [],
        "declare",
    )
    if X_all.shape[1] == 0:
        print("   WARNING: encoding matrix is empty.", flush=True)
        _empty['declare_features'] = declare_features
        return _empty

    print(f"   X_all shape: {X_all.shape}  |  Step 2: Fisher scores + coverage selection...",
          flush=True)
    fisher_scores = compute_generalized_fisher_scores(X_all, y_all, eps=fish_eps)
    sel           = select_features_coverage(X_all, fisher_scores, cov_thr)
    sel_names     = [feat_names[i] for i in sel]
    X_sel         = X_all[:, sel] if len(sel) > 0 else np.zeros((len(all_ids), 0))
    print(f"   sel_features: {len(sel_names):,}  |  Step 3: RipperK + DT rule extraction...",
          flush=True)

    ripper_rules, dt_rules = _fit_rules_on_array(X_sel, y_all, sel_names)

    DS_ripper = extract_rule_constraint_names_ripper(ripper_rules)
    DS_dt     = extract_rule_constraint_names_dt(dt_rules)
    print(f"   ripper: {len(ripper_rules)} rules -> DS={len(DS_ripper)}  |  "
          f"dt: {len(dt_rules)} rules -> DS={len(DS_dt)}", flush=True)

    return {
        'declare_features': declare_features,
        'feat_names':       list(feat_names),
        'sel_features':     sel_names,
        'X_all':            X_all,
        'fisher_scores':    fisher_scores,
        'DS_ripper':        DS_ripper,
        'DS_dt':            DS_dt,
        'ripper_rules':     ripper_rules,
        'dt_rules':         dt_rules,
        'case_data':        case_data,
        'all_ids':          all_ids,
        'y':                y_all,
        'positional_log':   positional_log,
        'sequence_log':     sequence_log,
    }


# =============================================================================
# SECTION 1 — DvM BISE 2025 ORIGINAL-DATA RUN
# =============================================================================

def run_dm_original() -> dict:
    """
    Run the stripped DvM declare-only pipeline on the real Production data.

    Skips sequential mining, data feature extraction, and DeclD discovery.
    Computes sel_features via Fisher+coverage, then trains RIPPER and DT on
    the selected feature submatrix. R_obs for rule sets is counted as the
    number of distinct Declare constraint names (DS) appearing in rule conditions.
    All four output sets computed in a single pipeline execution.
    """
    cov_t  = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))
    supp_t = float(DVM_CONFIG.get("support_threshold", DVM_SUPPORT_THRESHOLD))

    print("\n" + "=" * 100)
    print("SECTION 1 — DvM BISE 2025 ORIGINAL-DATA RUN  (declare-only stripped pipeline)")
    print("  Skipping: sequential mining (Step 1b), data features (Step 0c), DeclD (Step 1d)")
    print("  Fisher score + coverage selection + RipperK + Decision Tree")
    print(f"  coverage_threshold = {cov_t}  |  support_threshold = {supp_t}")
    print("  Four output sets: sel_features, ripper_rules (DS), dt_rules (DS), ripper+dt (DS)")
    print("=" * 100)

    t0 = time.time()
    print("   Step 0: loading data from CSV...", flush=True)
    case_data = load_and_preprocess_data(CSV_PATH)
    print(f"   Data loaded: {len(case_data):,} cases.  Running declare-only pipeline...",
          flush=True)
    dvm_out = run_dvm_declare_only(case_data, DVM_CONFIG.copy())
    wall    = time.time() - t0

    declare_features  = dvm_out["declare_features"]
    m_total           = len(declare_features)
    sel_names         = dvm_out["sel_features"]
    R_obs_sel         = len(sel_names)
    ripper_rules_orig = dvm_out["ripper_rules"]
    dt_rules_orig     = dvm_out["dt_rules"]
    union_rules_orig  = list(set(ripper_rules_orig) | set(dt_rules_orig))
    DS_ripper_orig    = dvm_out["DS_ripper"]
    DS_dt_orig        = dvm_out["DS_dt"]
    DS_union_orig     = DS_ripper_orig | DS_dt_orig
    fisher_scores     = dvm_out["fisher_scores"]
    feat_names        = dvm_out["feat_names"]

    R_obs_all = {
        'sel':    R_obs_sel,
        'ripper': len(DS_ripper_orig),
        'dt':     len(DS_dt_orig),
        'union':  len(DS_union_orig),
    }

    print(f"\n  Declare-only pipeline complete: {wall:.1f}s")
    print(f"  M_all (declare features discovered) = {m_total:,}")
    print(f"  R_obs (sel_features)                = {R_obs_all['sel']:,}")
    print(f"  R_obs (ripper_rules / DS)           = {R_obs_all['ripper']:,}  "
          f"({len(ripper_rules_orig)} rules -> {len(DS_ripper_orig)} constraint names)")
    print(f"  R_obs (dt_rules / DS)               = {R_obs_all['dt']:,}  "
          f"({len(dt_rules_orig)} rules -> {len(DS_dt_orig)} constraint names)")
    print(f"  R_obs (ripper u dt / DS)            = {R_obs_all['union']:,}")

    dvm_out["R_obs"]              = R_obs_sel
    dvm_out["R_obs_all"]          = R_obs_all
    dvm_out["ripper_rules_orig"]  = ripper_rules_orig
    dvm_out["dt_rules_orig"]      = dt_rules_orig
    dvm_out["union_rules_orig"]   = union_rules_orig
    dvm_out["DS_ripper_orig"]     = DS_ripper_orig
    dvm_out["DS_dt_orig"]         = DS_dt_orig
    dvm_out["DS_union_orig"]      = DS_union_orig
    dvm_out["fisher_scores"]      = fisher_scores
    dvm_out["feat_names"]         = feat_names
    dvm_out["m_total"]            = m_total
    dvm_out["wall_seconds"]       = wall
    dvm_out["config"]             = DVM_CONFIG.copy()
    return dvm_out


# =============================================================================
# SECTION 2 — COLLECT R_obs
# =============================================================================

def collect_R_obs(dvm_out: dict) -> dict:
    R_obs_all = dvm_out["R_obs_all"]
    cov_t     = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))

    print("\n" + "=" * 100)
    print("SECTION 2 — R_obs (REAL-DATA SELECTIONS, DvM BISE 2025)")
    print("=" * 100)
    print(f"\n  {METHOD_DM} — four output sets:")
    print(f"    sel_features  : {R_obs_all['sel']:,}  "
          f"[Fisher+coverage, coverage_threshold={cov_t}]")
    print(f"    ripper_rules  : {R_obs_all['ripper']:,}  [best-k RipperK on sel submatrix]")
    print(f"    dt_rules      : {R_obs_all['dt']:,}  [best-depth DT on sel submatrix]")
    print(f"    ripper u dt   : {R_obs_all['union']:,}  [union of both rule sets]")

    return R_obs_all


# =============================================================================
# SECTION 3 — DIAGNOSTICS
# =============================================================================

def run_diagnostics_dvm(dvm_out: dict) -> dict:
    print("\n" + "=" * 100)
    print("SECTION 3 — DvM BISE 2025 DIAGNOSTICS")
    print("=" * 100)

    fisher_scores = dvm_out["fisher_scores"]
    m             = int(dvm_out["m_total"])
    R_obs_all     = dvm_out["R_obs_all"]
    cov_t         = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))

    finite_f    = fisher_scores[np.isfinite(fisher_scores)]
    n_nonfinite = int((~np.isfinite(fisher_scores)).sum())

    print(f"\n  M_all = {m:,}")
    print(f"  R_obs (sel_features)  = {R_obs_all['sel']:,}  (coverage_threshold={cov_t})")
    print(f"  R_obs (ripper_rules)  = {R_obs_all['ripper']:,}")
    print(f"  R_obs (dt_rules)      = {R_obs_all['dt']:,}")
    print(f"  R_obs (ripper u dt)   = {R_obs_all['union']:,}")

    if len(finite_f) > 0:
        print(f"\n  Fisher distribution:  mean={finite_f.mean():.4f}  "
              f"median={np.median(finite_f):.4f}  "
              f"max={finite_f.max():.4f}  "
              f"discriminative={int((finite_f > 0).sum()):,}")

    return {
        'm_total':            m,
        'R_obs_sel':          R_obs_all['sel'],
        'R_obs_ripper':       R_obs_all['ripper'],
        'R_obs_dt':           R_obs_all['dt'],
        'R_obs_union':        R_obs_all['union'],
        'coverage_threshold': cov_t,
        'fisher_mean':        float(finite_f.mean())     if len(finite_f) > 0 else 0.0,
        'fisher_median':      float(np.median(finite_f)) if len(finite_f) > 0 else 0.0,
        'fisher_min':         float(finite_f.min())      if len(finite_f) > 0 else 0.0,
        'fisher_max':         float(finite_f.max())      if len(finite_f) > 0 else 0.0,
        'n_nonzero_fisher':   int((finite_f > 0).sum())  if len(finite_f) > 0 else 0,
        'n_zero_fisher':      int((finite_f == 0).sum()) if len(finite_f) > 0 else 0,
        'n_nonfinite':        n_nonfinite,
    }


# =============================================================================
# SECTION 4 — DOUBLY-NULL LOG BUILDER
# =============================================================================

def _rebuild_activity_index(trace):
    index = {}
    for pos, act in enumerate(trace):
        if act not in index:
            index[act] = []
        index[act].append(pos)
    return index


def _build_doubly_nullified_log(case_data_orig, case_ids_sorted, permuted_labels, random_state):
    rng       = np.random.RandomState(random_state)
    nullified = {}
    for i, cid in enumerate(case_ids_sorted):
        ci_orig = case_data_orig[cid]
        ci      = copy.copy(ci_orig)
        ci.outcome = int(permuted_labels[i])
        shuffled_trace    = list(ci_orig.trace)
        rng.shuffle(shuffled_trace)
        ci.trace          = shuffled_trace
        ci.activity_index = _rebuild_activity_index(shuffled_trace)
        nullified[cid] = ci
    return nullified


# =============================================================================
# SECTION 5 — DOUBLY-NULL REPLICATE RUNNER
# =============================================================================

def _evaluate_null_dvm_all_sets(null_case_data, declare_features, coverage_threshold, fisher_eps):
    """
    Apply Fisher+coverage + rule extraction to a null log using FIXED feature pool.
    Returns {'sel': int, 'ripper': int, 'dt': int, 'union': int} in a single pass.
    """
    positional_log, sequence_log, _ = build_log_structures(null_case_data)
    D_0, D_1 = split_by_class(null_case_data)
    ids0, ids1 = list(D_0.keys()), list(D_1.keys())
    n0, n1 = len(ids0), len(ids1)
    if n0 == 0 or n1 == 0:
        return {'sel': 0, 'ripper': 0, 'dt': 0, 'union': 0}

    all_ids = ids0 + ids1
    y_null  = np.array([0] * n0 + [1] * n1, dtype=np.int32)

    X, fn = build_encoding_matrix(
        all_ids, positional_log, sequence_log,
        {}, [], {},
        declare_features, [], [],
        "declare",
    )
    if X.shape[1] == 0:
        return {'sel': 0, 'ripper': 0, 'dt': 0, 'union': 0}

    fisher  = compute_generalized_fisher_scores(X, y_null, eps=fisher_eps)
    sel     = select_features_coverage(X, fisher, coverage_threshold)
    V_b_sel = len(sel)

    if V_b_sel > 0:
        sel_names_null = [fn[i] for i in sel]
        X_sel_null     = X[:, sel]
        ripper_null, dt_null = _fit_rules_on_array(X_sel_null, y_null, sel_names_null)
        DS_r       = extract_rule_constraint_names_ripper(ripper_null)
        DS_d       = extract_rule_constraint_names_dt(dt_null)
        V_b_ripper = len(DS_r)
        V_b_dt     = len(DS_d)
        V_b_union  = len(DS_r | DS_d)
    else:
        V_b_ripper = V_b_dt = V_b_union = 0

    return {'sel': V_b_sel, 'ripper': V_b_ripper, 'dt': V_b_dt, 'union': V_b_union}


def run_doubly_null_replicate_dvm(permuted_labels, case_data_orig, declare_features,
                                   case_ids_sorted, random_state):
    null_case_data = _build_doubly_nullified_log(
        case_data_orig, case_ids_sorted, permuted_labels,
        random_state=random_state + 200_000,
    )
    with _suppress_output():
        counts = _evaluate_null_dvm_all_sets(
            null_case_data     = null_case_data,
            declare_features   = declare_features,
            coverage_threshold = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD)),
            fisher_eps         = float(DVM_CONFIG.get("fisher_eps", 1e-10)),
        )
    return counts


# =============================================================================
# SECTION 6 — PARALLEL HELD-OUT NULL PERMUTATIONS
# =============================================================================

def _null_worker_dvm(b, permuted_labels_b, case_data_orig, declare_features, case_ids_sorted):
    rs = BASE_SEED + 100_000 + b
    return run_doubly_null_replicate_dvm(
        permuted_labels  = permuted_labels_b,
        case_data_orig   = case_data_orig,
        declare_features = declare_features,
        case_ids_sorted  = case_ids_sorted,
        random_state     = rs,
    )


def run_null_permutations(case_data, declare_features, case_ids_sorted, labels, n_jobs=N_JOBS):
    """
    Run B_NULL doubly-nullified replicates in parallel.
    Each replicate returns V_b for all four sets from a single evaluation.
    """
    print("\n" + "=" * 100)
    print("SECTION 6 — PARALLEL DOUBLY-NULL HELD-OUT PERMUTATIONS (DvM BISE 2025)")
    print(f"  B_null={B_NULL}, n_jobs={n_jobs}")
    print(f"  Fixed pool: {len(declare_features):,} declare features from original log")
    print(f"  Four sets per replicate: sel, ripper, dt, ripper+dt (single evaluation)")
    print("=" * 100)

    t0 = time.time()

    print(f"\n  Generating {B_NULL} held-out label permutations...")
    permuted_labels_all = _generate_heldout_permutation_batch(labels, B_NULL, BASE_SEED)
    for i in range(min(5, B_NULL)):
        assert int(permuted_labels_all[i].sum()) == int(labels.sum())
    print(f"  Marginal check passed  (n+={int(labels.sum()):,})")

    print(f"\n  Launching {B_NULL} workers (n_jobs={n_jobs})...\n")
    replicate_results = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(_null_worker_dvm)(b, permuted_labels_all[b], case_data, declare_features,
                                   case_ids_sorted)
        for b in range(B_NULL)
    )

    null_counts = {k: np.zeros(B_NULL, dtype=int) for k in ('sel', 'ripper', 'dt', 'union')}
    for b, res in enumerate(replicate_results):
        for k in ('sel', 'ripper', 'dt', 'union'):
            null_counts[k][b] = res[k]

    wall = time.time() - t0
    print(f"\n  All {B_NULL} replicates complete  |  wall={wall:.1f}s ({wall/60:.1f} min)")
    for k in ('sel', 'ripper', 'dt', 'union'):
        arr = null_counts[k]
        print(f"    {k:10s}:  mean={arr.mean():.2f}  std={arr.std():.2f}  "
              f"max={arr.max()}  zeros={int(np.sum(arr == 0))}")

    return {'null_counts': null_counts, 'wall_seconds': wall}


# =============================================================================
# SECTION 7 — FDR METRICS AND OUTPUT
# =============================================================================

def compute_and_save_metrics(null_counts, R_obs, dvm_out, diagnostics,
                              original_wall, perm_wall):
    """
    Compute FDR_emp / PCER_emp / FWER_emp for all four output sets.
    Saves to sel_features/ and rules/ subfolders under RQ1_OUTPUT_DIR.
    """
    print("\n" + "=" * 100)
    print("SECTION 7 — FDR METRICS AND OUTPUT")
    print("=" * 100)

    sel_dir   = os.path.join(RQ1_OUTPUT_DIR, "sel_features")
    rules_dir = os.path.join(RQ1_OUTPUT_DIR, "rules")
    os.makedirs(sel_dir,   exist_ok=True)
    os.makedirs(rules_dir, exist_ok=True)

    m_total = diagnostics['m_total']
    cov_t   = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))

    SET_LABELS = {
        'sel':    'sel_features',
        'ripper': 'ripper_rules',
        'dt':     'dt_rules',
        'union':  'ripper+dt_rules',
    }

    all_metrics = {}
    fdr_tests   = {}

    print(f"\n  FDR Results (Production, B_null={B_NULL}, doubly-null protocol):")
    print(f"  {'─'*88}")
    print(f"  {'Set':18s} {'alpha':>5s} {'R_obs':>6s} {'E[V_b]':>8s} "
          f"{'FDR_emp':>8s} {'95% CI':>22s} {'FWER':>7s} {'Pass?':>6s}")
    print(f"  {'─'*88}")

    for k, label in SET_LABELS.items():
        m = _compute_fdr_metrics(null_counts[k], R_obs[k], m_total, ALPHA)
        all_metrics[k] = m
        fdr_tests[k]   = {
            'fdr_emp': m['FDR_emp'], 'controls_FDR': m['controls_FDR'],
            'FDR_CI_lower': m['FDR_CI_lower'], 'FDR_CI_upper': m['FDR_CI_upper'],
        }
        ci_str  = f"[{m['FDR_CI_lower']:.4f}, {m['FDR_CI_upper']:.4f}]"
        verdict = "PASS" if m['controls_FDR'] else "FAIL"
        print(f"  {label:18s} {ALPHA:>5.3f} {m['R_obs']:>6d} "
              f"{m['E_V_b']:>8.2f} {m['FDR_emp']:>8.4f} "
              f"{ci_str:>22s} {m['FWER_emp']:>7.4f} {verdict:>6s}")
    print(f"  {'─'*88}")

    # sel_features subfolder
    m_sel = all_metrics['sel']
    sel_df = pd.DataFrame([{'method': METHOD_DM, 'output_set': 'sel_features', **m_sel}])
    path = os.path.join(sel_dir, "rq1_dvm_bise2025_fdr_metrics.csv")
    sel_df.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    null_df            = pd.DataFrame({'V_b': null_counts['sel']})
    null_df.index.name = 'replicate_b'
    path = os.path.join(sel_dir, "rq1_dvm_bise2025_null_counts.csv")
    null_df.to_csv(path)
    print(f"  Saved: {path}")

    declare_features = dvm_out["declare_features"]
    fisher_scores    = dvm_out["fisher_scores"]
    sel_set          = set(dvm_out["sel_features"])
    diag_rows = []
    for j, (tmpl, acts) in enumerate(declare_features):
        fname = f"{tmpl}_({'_'.join(acts)})"
        fs    = float(fisher_scores[j]) if j < len(fisher_scores) else 0.0
        diag_rows.append({
            'pattern_id': fname, 'constraint_type': tmpl,
            'activity_a': acts[0] if acts else None,
            'activity_b': acts[1] if len(acts) > 1 else None,
            'fisher_score': fs, 'is_selected': fname in sel_set,
        })
    diag_df = pd.DataFrame(diag_rows)
    path    = os.path.join(sel_dir, "rq1_dvm_bise2025_diagnostics.csv")
    diag_df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(diag_df):,} constraints)")

    sel_json = {
        'rq1_version': '2.0', 'method': METHOD_DM,
        'output_set': 'sel_features', 'log_name': 'Production',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'B_NULL': B_NULL, 'ALPHA': ALPHA, 'BASE_SEED': BASE_SEED,
            'N_JOBS': N_JOBS, 'coverage_threshold': cov_t,
            'support_threshold': float(DVM_CONFIG.get('support_threshold', 0.0)),
            'fisher_eps': float(DVM_CONFIG.get('fisher_eps', 1e-10)),
            'R_obs_sel': R_obs['sel'], 'm_total': int(m_total),
        },
        'R_obs': R_obs['sel'], 'metrics': m_sel,
        'null_replicate_summary': {
            'mean_V_b': float(null_counts['sel'].mean()),
            'std_V_b':  float(null_counts['sel'].std()),
            'max_V_b':  int(null_counts['sel'].max()),
            'n_zero_V_b': int(np.sum(null_counts['sel'] == 0)),
        },
        'diagnostics': diagnostics,
        'timing': {
            'original_DvM_seconds': original_wall,
            'null_permutations_seconds': perm_wall,
            'total_seconds': original_wall + perm_wall,
        },
    }
    path = os.path.join(sel_dir, "rq1_dvm_bise2025_results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sel_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path}")

    # rules subfolder
    RULE_SETS = {
        'ripper': 'ripper_rules',
        'dt':     'dt_rules',
        'union':  'ripper+dt_rules',
    }
    for k, set_label in RULE_SETS.items():
        m_k = all_metrics[k]
        row_df = pd.DataFrame([{'method': METHOD_DM, 'output_set': set_label, **m_k}])
        path   = os.path.join(rules_dir, f"rq1_dvm_bise2025_fdr_metrics_{k}.csv")
        row_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

        nc_df            = pd.DataFrame({'V_b': null_counts[k]})
        nc_df.index.name = 'replicate_b'
        path = os.path.join(rules_dir, f"rq1_dvm_bise2025_null_counts_{k}.csv")
        nc_df.to_csv(path)
        print(f"  Saved: {path}")

    rules_json = {
        'rq1_version': '2.0', 'method': METHOD_DM,
        'log_name': 'Production', 'timestamp': datetime.now().isoformat(),
        'output_sets': ['ripper_rules', 'dt_rules', 'ripper+dt_rules'],
        'config': {
            'B_NULL': B_NULL, 'ALPHA': ALPHA, 'BASE_SEED': BASE_SEED, 'N_JOBS': N_JOBS,
            'coverage_threshold': cov_t,
            'RIPPER_K_GRID': list(RIPPER_K_GRID), 'DT_DEPTH_GRID': list(DT_DEPTH_GRID),
        },
        'R_obs_all': R_obs,
        'metrics': {k: all_metrics[k] for k in ('ripper', 'dt', 'union')},
        'rule_counts_orig': {
            'ripper': len(dvm_out.get('ripper_rules_orig', [])),
            'dt':     len(dvm_out.get('dt_rules_orig', [])),
            'union':  len(dvm_out.get('union_rules_orig', [])),
        },
        'null_replicate_summary': {
            k: {
                'mean_V_b': float(null_counts[k].mean()), 'std_V_b': float(null_counts[k].std()),
                'max_V_b':  int(null_counts[k].max()),    'n_zero_V_b': int(np.sum(null_counts[k] == 0)),
            }
            for k in ('ripper', 'dt', 'union')
        },
        'timing': {
            'original_DvM_seconds': original_wall,
            'null_permutations_seconds': perm_wall,
            'total_seconds': original_wall + perm_wall,
        },
    }
    path = os.path.join(rules_dir, "rq1_dvm_bise2025_results_rules.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rules_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path}")

    return all_metrics, fdr_tests


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 100)
    print("RQ1 — FDR CONTROL VALIDITY: PRODUCTION  [DvM BISE 2025]")
    print("Fisher Score + Coverage Selection + Rule Extraction Baseline")
    print("Double-null protocol: sigma_label composed with sigma_trace")
    print("Four output sets: sel_features, ripper_rules, dt_rules, ripper+dt")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  B_null={B_NULL}")
    print(f"  coverage_threshold = {DVM_CONFIG.get('coverage_threshold', DVM_COVERAGE_THRESHOLD)}")
    print(f"  Output: {RQ1_OUTPUT_DIR}/sel_features/  and  {RQ1_OUTPUT_DIR}/rules/")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ1_OUTPUT_DIR, exist_ok=True)

    dvm_orig = run_dm_original()

    case_data        = dvm_orig["case_data"]
    declare_features = dvm_orig["declare_features"]
    case_ids_sorted  = sorted(case_data.keys())
    labels           = np.array([case_data[cid].outcome for cid in case_ids_sorted])
    original_wall    = dvm_orig["wall_seconds"]

    R_obs = collect_R_obs(dvm_orig)

    diagnostics = run_diagnostics_dvm(dvm_orig)

    perm_out = run_null_permutations(
        case_data=case_data, declare_features=declare_features,
        case_ids_sorted=case_ids_sorted, labels=labels, n_jobs=N_JOBS,
    )
    null_counts = perm_out['null_counts']
    perm_wall   = perm_out['wall_seconds']

    all_metrics, fdr_tests = compute_and_save_metrics(
        null_counts=null_counts, R_obs=R_obs, dvm_out=dvm_orig,
        diagnostics=diagnostics, original_wall=original_wall, perm_wall=perm_wall,
    )

    total_wall = time.time() - t_total
    SET_LABELS = {'sel': 'sel_features', 'ripper': 'ripper_rules', 'dt': 'dt_rules', 'union': 'ripper+dt'}

    print(f"\n{'='*100}")
    print("RQ1 — PRODUCTION DvM BISE 2025 COMPLETE  (double-null protocol)")
    print(f"  Total wall time: {total_wall:.1f}s  |  Output: {RQ1_OUTPUT_DIR}/")
    print(f"\n  {'Output set':18s} {'R_obs':>6s} {'E[V_b]':>8s} {'FDR_emp':>8s} {'Pass?':>6s}")
    print(f"  {'─'*50}")
    for k, label in SET_LABELS.items():
        m = all_metrics[k]
        verdict = "PASS" if m['FDR_emp'] <= ALPHA else "FAIL"
        print(f"  {label:18s} {R_obs[k]:>6d} {m['E_V_b']:>8.2f} {m['FDR_emp']:>8.4f} {verdict:>6s}")
    print(f"{'='*100}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1 FDR Validity — Production (DvM BISE 2025, double-null protocol)"
    )
    parser.add_argument('--b-null', type=int, default=B_NULL)
    parser.add_argument('--n-jobs', type=int, default=N_JOBS)
    parser.add_argument('--alpha',  type=float, default=ALPHA)
    parser.add_argument('--dry-run', action='store_true', help='B_null=2 smoke test')
    args = parser.parse_args()

    if args.dry_run:
        B_NULL = 2
        print("*** DRY-RUN MODE: B_null=2 ***")
    else:
        B_NULL = args.b_null

    assert B_NULL < 100_000
    N_JOBS = args.n_jobs
    ALPHA  = args.alpha

    main()
