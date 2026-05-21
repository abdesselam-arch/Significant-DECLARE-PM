#!/usr/bin/env python3
"""
rq2_Sepsis_BISE.py  —  RQ2 Specification Quality Degradation: Sepsis  [DvM BISE 2025]
======================================================================================
Block B.3: Joint Label x Structural Noise Perturbation (2-D Cartesian grid)
DvM BISE 2025 — Fisher Score + Coverage Selection + Rule Extraction Baseline

RESEARCH QUESTION
-----------------
RQ2 (DvM BISE 2025): As signal is progressively corrupted along BOTH noise axes
simultaneously, how does DvM BISE 2025 degrade across all four output sets:
  sel_features, ripper_rules, dt_rules, ripper+dt_rules
in terms of FDR_ref, Precision, Recall, F1, and Jaccard_rq2?

FOUR OUTPUT SETS (single pipeline run per cell)
-----------------------------------------------
    sel_features   Fisher+coverage selection on (X_struct, y_perturbed)
    ripper_rules   RipperK rules trained on sel submatrix
    dt_rules       DT rules trained on sel submatrix
    ripper+dt      ripper_rules u dt_rules (unique rule strings)

All four sets are computed from ONE DvM evaluation per grid cell or null
replicate — no re-running the pipeline for each set.

JOINT PERTURBATION OPERATOR
-----------------------------
B.3 — Joint noise N_label(eps) composed with N_struct(rho):
    eps in {0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50}  (7 levels)
    rho in {0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00}  (7 levels)
    -> 49 cells

X REUSE STRATEGY
-----------------
X depends only on trace structure (not labels).
    rho = 0.00 : reuse X_orig (no structural perturbation).
    rho > 0.00 : precompute X once per rho row, shared across all 7 eps cells.
Fisher scores recomputed per cell (7 X computations, 49 Fisher passes).

REFERENCE SETS
--------------
  S_orig_full_{set}  — output set on original (unperturbed) data (ground truth).
  S_orig_rq2_{set}   — output set at (eps=0, rho=0) (single baseline).

PRIMARY METRICS PER CELL (for each of the four sets)
------------------------------------------------------
    FDR_ref       FP / R_obs
    Precision     TP / R_obs
    Recall        TP / R_full
    F1            harmonic mean(Precision, Recall)
    Jaccard_rq2   |S_pert inter S_orig_rq2| / |S_pert union S_orig_rq2|

OUTPUT FILES
------------
    sel_features/
        rq2_dvm_bise2025_joint_metrics_sel.csv
        rq2_dvm_bise2025_joint_null_counts_sel.json
        rq2_dvm_bise2025_joint_results_sel.json
        rq2_dvm_bise2025_joint_<metric>_sel.csv   (heatmaps)
    rules/
        rq2_dvm_bise2025_joint_metrics_ripper.csv
        rq2_dvm_bise2025_joint_metrics_dt.csv
        rq2_dvm_bise2025_joint_metrics_union.csv
        rq2_dvm_bise2025_joint_null_counts_rules.json
        rq2_dvm_bise2025_joint_results_rules.json
        rq2_dvm_bise2025_joint_<metric>_{ripper/dt/union}.csv   (heatmaps)

Version : 2.0  (adds ripper_rules / dt_rules / union output sets)
Author  : Ahmed Nour Abdesselam
Date    : May 2026

References
----------
Di Francescomarino, Donadello, Ghidini, Maggi, Puura (2025). BISE 67(6):877-894.
Pellegrina & Vandin (2018/2020). KDD 2018 / TKDD 2020.
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# DvM BISE 2025 Sepsis module
from DvM_Sepsis import (
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

RQ2_OUTPUT_DIR = "RQ2_Sepsis_DvM_BISE2025"

JOINT_LABEL_LEVELS  = [0.00, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
JOINT_STRUCT_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

JOINT_GRID = [
    (eps, rho)
    for eps in JOINT_LABEL_LEVELS
    for rho in JOINT_STRUCT_LEVELS
]

ANCHOR_CELLS = frozenset({
    (0.00, 0.00),
    (JOINT_LABEL_LEVELS[-1], 0.00),
    (0.00, JOINT_STRUCT_LEVELS[-1]),
    (JOINT_LABEL_LEVELS[-1], JOINT_STRUCT_LEVELS[-1]),
})

ALPHA         = 0.05
B_NULL_ANCHOR = 200
BASE_SEED     = 20260601
N_JOBS        = -1
INNER_N_JOBS  = 1

METHOD_DM   = "BISE2025_DvM"
ALL_METHODS = [METHOD_DM]

DVM_CONFIG = dict(_DVM_CONFIG_BASE)

_RIPPER_PRUNE_SIZE  = 0.33
_RULES_RANDOM_STATE = 42


# =============================================================================
# HELPERS
# =============================================================================

@contextlib.contextmanager
def _suppress():
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def _rebuild_activity_index(trace):
    index = {}
    for pos, act in enumerate(trace):
        if act not in index:
            index[act] = []
        index[act].append(pos)
    return index


def _fit_rules_on_array(X_sel, y, sel_names):
    """
    Fit RipperK and DT on X_sel (already restricted to sel_features columns).
    Returns (ripper_rules: List[str], dt_rules: List[str]).
    Single evaluation — used in both original-data and null-replicate paths.
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
# PERTURBATION OPERATORS
# =============================================================================

def apply_label_noise(case_data, epsilon, seed):
    if epsilon == 0.0:
        return case_data
    rng = np.random.RandomState(seed)
    result = {}
    for cid, case in case_data.items():
        if rng.random() < epsilon:
            ci         = copy.copy(case)
            ci.outcome = 1 - case.outcome
            result[cid] = ci
        else:
            result[cid] = case
    return result


def apply_structural_noise(case_data, rho, seed):
    if rho == 0.0:
        return case_data
    rng = np.random.RandomState(seed)
    result = {}
    for cid, case in case_data.items():
        trace = list(case.trace)
        n = len(trace)
        if n > 1:
            n_to_swap = int(round(rho * (n - 1)))
            if n_to_swap > 0:
                pair_idxs = rng.choice(n - 1, size=n_to_swap, replace=False)
                for i in sorted(pair_idxs):
                    trace[i], trace[i + 1] = trace[i + 1], trace[i]
        ci                = copy.copy(case)
        ci.trace          = trace
        ci.activity_index = _rebuild_activity_index(trace)
        result[cid]       = ci
    return result


def apply_joint_perturbation(struct_log, eps, eps_seed):
    return apply_label_noise(struct_log, eps, eps_seed)


# =============================================================================
# DOUBLY-NULL LOG BUILDER
# =============================================================================

def _build_doubly_null_log(case_data_perturbed, case_ids_sorted, permuted_labels, trace_seed):
    rng  = np.random.RandomState(trace_seed)
    null = {}
    for i, cid in enumerate(case_ids_sorted):
        ci_orig       = case_data_perturbed[cid]
        ci            = copy.copy(ci_orig)
        ci.outcome    = int(permuted_labels[i])
        shuffled      = list(ci_orig.trace)
        rng.shuffle(shuffled)
        ci.trace          = shuffled
        ci.activity_index = _rebuild_activity_index(shuffled)
        null[cid]         = ci
    return null


# =============================================================================
# DvM ALL-SETS EVALUATION ON PRECOMPUTED X
# =============================================================================

def _dvm_all_sets_on_X(
    y: np.ndarray,
    X_struct: np.ndarray,
    feat_names: list,
    coverage_threshold: int,
    fisher_eps: float = 1e-10,
) -> dict:
    """
    Apply Fisher+coverage + rule extraction to a precomputed feature matrix.

    Computes all four output sets from a SINGLE pass over (X_struct, y):
        S_sel    — frozenset of selected feature names
        S_ripper — frozenset of RipperK rule strings
        S_dt     — frozenset of DT rule strings
        S_union  — S_ripper | S_dt

    X_struct is precomputed per rho row and shared across all eps cells.
    For rules, classifiers are trained on X_struct[:, sel] (the sel submatrix).
    """
    fisher = compute_generalized_fisher_scores(X_struct, y, eps=fisher_eps)
    sel    = select_features_coverage(X_struct, fisher, coverage_threshold)
    S_sel  = frozenset(feat_names[i] for i in sel)

    if len(sel) > 0:
        sel_names = [feat_names[i] for i in sel]
        X_sel     = X_struct[:, sel]
        ripper_rules, dt_rules = _fit_rules_on_array(X_sel, y, sel_names)
        S_ripper = extract_rule_constraint_names_ripper(ripper_rules)
        S_dt     = extract_rule_constraint_names_dt(dt_rules)
        S_union  = S_ripper | S_dt
    else:
        S_ripper = frozenset()
        S_dt     = frozenset()
        S_union  = frozenset()

    return {
        'S_sel': S_sel, 'S_ripper': S_ripper, 'S_dt': S_dt, 'S_union': S_union,
        'R_sel': len(S_sel), 'R_ripper': len(S_ripper),
        'R_dt': len(S_dt),   'R_union': len(S_union),
        'fisher': fisher,
    }


# =============================================================================
# REFERENCE-ANCHORED METRICS
# =============================================================================

def compute_reference_metrics(S_pert, S_orig_full, S_orig_rq2) -> dict:
    """
    Reference-anchored performance metrics for one output set.
    S_pert, S_orig_full, S_orig_rq2 are frozensets of strings (features or rules).
    """
    R_obs  = len(S_pert)
    R_full = len(S_orig_full)

    TP = len(S_pert & S_orig_full)
    FP = len(S_pert - S_orig_full)
    FN = len(S_orig_full - S_pert)

    if R_obs == 0:
        precision = float('nan')
        fdr_ref   = float('nan')
        estimable = False
    else:
        precision = TP / R_obs
        fdr_ref   = FP / R_obs
        estimable = True

    recall = TP / R_full if R_full > 0 else float('nan')

    if (not estimable) or (precision != precision) or (recall != recall):
        f1 = float('nan')
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    union_rq2   = len(S_pert | S_orig_rq2)
    inter_rq2   = len(S_pert & S_orig_rq2)
    jaccard_rq2 = inter_rq2 / union_rq2 if union_rq2 > 0 else 1.0

    union_full   = len(S_pert | S_orig_full)
    jaccard_full = TP / union_full if union_full > 0 else 1.0

    fp_over_rfull = FP / R_full if R_full > 0 else float('nan')
    fn_over_rfull = FN / R_full if R_full > 0 else float('nan')

    return {
        'FDR_ref': fdr_ref, 'Precision': precision, 'Recall': recall,
        'F1': f1, 'TP': TP, 'FP': FP, 'FN': FN,
        'R_obs': R_obs, 'R_full': R_full,
        'Jaccard_rq2': jaccard_rq2, 'Jaccard_full': jaccard_full,
        'Gained_rq2': len(S_pert - S_orig_rq2), 'Lost_rq2': len(S_orig_rq2 - S_pert),
        'estimable': estimable, 'reliable': estimable and (R_obs >= 10),
        'FP_over_Rfull': fp_over_rfull, 'FN_over_Rfull': fn_over_rfull,
    }


# =============================================================================
# FDR HELPERS
# =============================================================================

def _bca_ci(data, B=800, seed=42):
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


def _null_fdr_skipped(reason):
    return {
        'FDR_emp': float('nan'), 'FDR_CI_lower': float('nan'),
        'FDR_CI_upper': float('nan'), 'E_V_b': float('nan'),
        'FWER_emp': float('nan'), 'controls_FDR': None,
        'estimable': False, 'skipped_reason': reason,
    }


def _compute_fdr_from_null(null_counts, R_obs, alpha_nominal):
    ev = float(np.mean(null_counts))
    if R_obs == 0:
        return {
            'FDR_emp': float('nan'), 'FDR_CI_lower': float('nan'),
            'FDR_CI_upper': float('nan'), 'E_V_b': ev,
            'FWER_emp': float(np.mean(null_counts > 0)),
            'controls_FDR': None, 'estimable': False,
            'skipped_reason': 'R_obs=0: FDR_emp undefined',
        }
    arr = null_counts.astype(float) / R_obs
    fdr = float(np.mean(arr))
    try:
        lo, hi = _bca_ci(arr)
    except Exception:
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        'FDR_emp': fdr, 'FDR_CI_lower': lo, 'FDR_CI_upper': hi, 'E_V_b': ev,
        'FWER_emp': float(np.mean(null_counts > 0)), 'controls_FDR': bool(fdr <= alpha_nominal),
        'estimable': True, 'skipped_reason': None,
    }


# =============================================================================
# NULL REPLICATE RUNNER FOR ANCHOR CELLS
# =============================================================================

def _one_anchor_replicate_dvm(
    b, case_data_perturbed, declare_features, case_ids_sorted,
    perm_labels_b, coverage_threshold, base_seed,
) -> dict:
    """
    One doubly-null anchor replicate — returns V_b for all four output sets.

    Builds sigma_label composed sigma_trace on L_{eps,rho}, rebuilds X on null log
    (sigma_trace changes Declare satisfaction values -> X_struct invalid),
    then runs Fisher+coverage + rule extraction in a single pass.

    Returns {'sel': int, 'ripper': int, 'dt': int, 'union': int}.
    """
    rs_trace = base_seed + 100_000 + b

    null_cd = _build_doubly_null_log(
        case_data_perturbed, case_ids_sorted, perm_labels_b, trace_seed=rs_trace,
    )

    D0_b, D1_b = split_by_class(null_cd)
    if len(D0_b) < 5 or len(D1_b) < 5:
        return {'sel': 0, 'ripper': 0, 'dt': 0, 'union': 0}

    null_pos_log, null_seq_log, _ = build_log_structures(null_cd)
    with _suppress():
        X_null, fn = build_encoding_matrix(
            case_ids_sorted, null_pos_log, null_seq_log,
            {}, [], {},
            declare_features, [], [],
            "declare",
        )

    if X_null.shape[1] == 0:
        return {'sel': 0, 'ripper': 0, 'dt': 0, 'union': 0}

    y_null   = perm_labels_b.astype(np.int32)
    fish_eps = float(DVM_CONFIG.get("fisher_eps", 1e-10))

    with _suppress():
        dvm_res = _dvm_all_sets_on_X(
            y=y_null, X_struct=X_null, feat_names=list(fn),
            coverage_threshold=coverage_threshold, fisher_eps=fish_eps,
        )

    return {
        'sel':    dvm_res['R_sel'],
        'ripper': dvm_res['R_ripper'],
        'dt':     dvm_res['R_dt'],
        'union':  dvm_res['R_union'],
    }


def run_null_replicates_for_level(
    case_data_perturbed, declare_features, case_ids_sorted,
    labels_perturbed, coverage_threshold, B_null, base_seed, n_jobs=1,
) -> dict:
    """
    Run B_null doubly-null replicates on a PERTURBED log (anchor cells only).
    Returns {'sel': arr, 'ripper': arr, 'dt': arr, 'union': arr}.
    """
    rng_outer       = np.random.RandomState(base_seed)
    perm_labels_all = np.stack([
        rng_outer.permutation(labels_perturbed).astype(np.int8)
        for _ in range(B_null)
    ], axis=0)

    reps = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_anchor_replicate_dvm)(
            b, case_data_perturbed, declare_features, case_ids_sorted,
            perm_labels_all[b], coverage_threshold, base_seed,
        )
        for b in range(B_null)
    )

    counts = {k: np.zeros(B_null, dtype=int) for k in ('sel', 'ripper', 'dt', 'union')}
    for b, r in enumerate(reps):
        for k in ('sel', 'ripper', 'dt', 'union'):
            counts[k][b] = r[k]
    return counts


# =============================================================================
# JOINT DRAW RUNNER
# =============================================================================

def _run_dvm_on_joint_draw(eps, eps_seed, struct_log, X_struct, feat_names,
                            case_ids_sorted, coverage_threshold):
    """Apply N_label(eps) to struct_log and run DvM all-sets evaluation."""
    case_data_pert = apply_joint_perturbation(struct_log, eps, eps_seed)

    y_perturbed = np.array(
        [case_data_pert[cid].outcome for cid in case_ids_sorted], dtype=np.int32
    )

    dvm_res = _dvm_all_sets_on_X(
        y=y_perturbed, X_struct=X_struct, feat_names=feat_names,
        coverage_threshold=coverage_threshold,
        fisher_eps=float(DVM_CONFIG.get("fisher_eps", 1e-10)),
    )

    return {
        'case_data_pert': case_data_pert,
        'y_perturbed':    y_perturbed,
        'S_pert_sel':     dvm_res['S_sel'],
        'S_pert_ripper':  dvm_res['S_ripper'],
        'S_pert_dt':      dvm_res['S_dt'],
        'S_pert_union':   dvm_res['S_union'],
        'R_obs_sel':      dvm_res['R_sel'],
        'R_obs_ripper':   dvm_res['R_ripper'],
        'R_obs_dt':       dvm_res['R_dt'],
        'R_obs_union':    dvm_res['R_union'],
        'eps_seed_used':  eps_seed,
    }


# =============================================================================
# PER-CELL ANALYSIS
# =============================================================================

def analyze_joint_cell(
    eps, rho, eps_idx, rho_idx,
    struct_log, X_struct, feat_names, declare_features, case_ids_sorted,
    S_orig_full: dict,   # {'sel': frozenset, 'ripper': frozenset, 'dt': frozenset, 'union': frozenset}
    S_orig_rq2: dict,    # same structure, or None at (0,0) anchor
    coverage_threshold,
    is_anchor_bootstrap=False,
) -> dict:
    """
    Full analysis for one joint noise cell (eps, rho), all four output sets.

    S_orig_full and S_orig_rq2 are dicts keyed by {'sel','ripper','dt','union'}.
    At (0,0) anchor (is_anchor_bootstrap=True), S_orig_rq2 is set to S_pert itself.
    """
    print(f"  [eps={eps:.2f}  rho={rho:.2f}]  Starting DvM BISE 2025 all-sets analysis...",
          flush=True)
    t0 = time.time()

    base_seed_cell = BASE_SEED + eps_idx * 1000 + rho_idx
    MAX_RETRIES    = 5
    RETRY_PRIME    = 999983

    draw      = None
    n_retries = 0

    for retry in range(MAX_RETRIES + 1):
        eps_seed_try = base_seed_cell + 1 + retry * RETRY_PRIME
        draw = _run_dvm_on_joint_draw(
            eps=eps, eps_seed=eps_seed_try, struct_log=struct_log,
            X_struct=X_struct, feat_names=feat_names,
            case_ids_sorted=case_ids_sorted, coverage_threshold=coverage_threshold,
        )
        if draw['R_obs_sel'] > 0:
            if retry > 0:
                print(f"  [eps={eps:.2f}  rho={rho:.2f}]  R_obs>0 after {retry} retries.",
                      flush=True)
            n_retries = retry
            break
        print(f"  [eps={eps:.2f}  rho={rho:.2f}]  R_obs_sel=0 on draw {retry} — ", end="",
              flush=True)
        if retry < MAX_RETRIES:
            print("retrying...", flush=True)
        else:
            print(f"MAX_RETRIES={MAX_RETRIES} exhausted.", flush=True)
            n_retries = retry

    case_data_pert = draw['case_data_pert']
    y_perturbed    = draw['y_perturbed']
    eps_seed_used  = draw['eps_seed_used']

    SET_KEYS = ('sel', 'ripper', 'dt', 'union')
    DRAW_S_KEYS = {
        'sel': 'S_pert_sel', 'ripper': 'S_pert_ripper',
        'dt':  'S_pert_dt',  'union':  'S_pert_union',
    }
    DRAW_R_KEYS = {
        'sel': 'R_obs_sel',    'ripper': 'R_obs_ripper',
        'dt':  'R_obs_dt',     'union':  'R_obs_union',
    }

    S_pert_all = {k: draw[DRAW_S_KEYS[k]] for k in SET_KEYS}
    R_obs_all  = {k: draw[DRAW_R_KEYS[k]] for k in SET_KEYS}

    # At (0,0) anchor, own output is the rq2 baseline
    if is_anchor_bootstrap or S_orig_rq2 is None:
        _S_rq2_all = S_pert_all
    else:
        _S_rq2_all = S_orig_rq2

    ref_metrics_all = {
        k: compute_reference_metrics(S_pert_all[k], S_orig_full[k], _S_rq2_all[k])
        for k in SET_KEYS
    }

    is_anchor   = is_anchor_bootstrap or (eps, rho) in ANCHOR_CELLS
    B_null_this = B_NULL_ANCHOR if (is_anchor and R_obs_all['sel'] > 0) else 0

    if B_null_this == 0:
        fdr_null_all = {k: _null_fdr_skipped('non-anchor cell') for k in SET_KEYS}
        null_counts_all = {k: np.zeros(0, dtype=int) for k in SET_KEYS}
    else:
        print(
            f"  [eps={eps:.2f}  rho={rho:.2f}]  "
            f"Running {B_null_this} null replicates (all four sets per rep)...",
            flush=True,
        )
        counts_dict = run_null_replicates_for_level(
            case_data_perturbed = case_data_pert,
            declare_features    = declare_features,
            case_ids_sorted     = case_ids_sorted,
            labels_perturbed    = y_perturbed,
            coverage_threshold  = coverage_threshold,
            B_null              = B_null_this,
            base_seed           = base_seed_cell + 50_000,
            n_jobs              = INNER_N_JOBS,
        )
        null_counts_all = counts_dict
        fdr_null_all = {
            k: _compute_fdr_from_null(counts_dict[k], R_obs_all[k], ALPHA)
            for k in SET_KEYS
        }

    wall = time.time() - t0
    _fdr_sel_str = (
        'n/a' if B_null_this == 0 else f"{fdr_null_all['sel']['FDR_emp']:.4f}"
    )
    print(
        f"  [eps={eps:.2f}  rho={rho:.2f}]  Done in {wall:.1f}s  "
        f"R_obs_sel={R_obs_all['sel']}  "
        f"FDR_ref_sel={ref_metrics_all['sel']['FDR_ref']:.4f}  "
        f"FDR_emp_sel={_fdr_sel_str}",
        flush=True,
    )

    result = {
        'eps': eps, 'rho': rho, 'eps_idx': eps_idx, 'rho_idx': rho_idx,
        'R_obs_all':      R_obs_all,
        'ref_metrics_all': ref_metrics_all,
        'fdr_null_all':    fdr_null_all,
        'null_counts_all': {k: null_counts_all[k].tolist() for k in SET_KEYS},
        'B_null':          B_null_this,
        'wall_seconds':    wall,
        'is_anchor':       is_anchor,
        'n_retries':       n_retries,
        'eps_seed_used':   eps_seed_used,
    }

    if is_anchor_bootstrap:
        result['S_pert_all_raw'] = S_pert_all

    return result


# =============================================================================
# SECTION 1 — ORIGINAL DATA RUN
# =============================================================================

def run_original_data_dvm() -> dict:
    """
    Run DvM BISE 2025 on the original (unperturbed) Sepsis log.

    Computes all four output sets (sel_features, ripper_rules, dt_rules, union)
    via a single pipeline execution. Returns orig_data dict with:
        S_orig_full — dict of frozensets for each output set (ground truth)
        X_orig, feat_names, case_data, declare_features, case_ids_sorted, labels_orig
    """
    cov_t    = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))
    supp_t   = float(DVM_CONFIG.get("support_threshold", DVM_SUPPORT_THRESHOLD))
    fish_eps = float(DVM_CONFIG.get("fisher_eps", 1e-10))

    print("\n" + "=" * 100)
    print("SECTION 1 — ORIGINAL DATA RUN  (declare-only stripped pipeline)")
    print("  Skipping: sequential mining (Step 1b), data features (Step 0c), DeclD (Step 1d)")
    print(f"  coverage_threshold = {cov_t}  |  support_threshold = {supp_t}")
    print(f"  Four output sets: sel_features, ripper_rules (DS), dt_rules (DS), ripper+dt (DS)")
    print(f"  X_orig computed here; reused as X_cache[rho=0.0] in grid.")
    print("=" * 100)

    t0 = time.time()
    print("   Step 0: loading data from CSV...", flush=True)
    case_data = load_and_preprocess_data(CSV_PATH)
    print(f"   Data loaded: {len(case_data):,} cases.  Running declare-only pipeline...",
          flush=True)
    dvm_out = run_dvm_declare_only(case_data, DVM_CONFIG.copy())
    wall    = time.time() - t0

    declare_features = dvm_out["declare_features"]
    case_ids_sorted  = sorted(case_data.keys())
    labels_orig      = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    positional_log = dvm_out["positional_log"]
    sequence_log   = dvm_out["sequence_log"]

    # Build X with case_ids_sorted ordering so rows align with labels_orig in grid
    X_orig, feat_names = build_encoding_matrix(
        case_ids_sorted, positional_log, sequence_log,
        {}, [], {},
        declare_features, [], [],
        "declare",
    )

    dvm_res_orig = _dvm_all_sets_on_X(
        y=labels_orig, X_struct=X_orig, feat_names=list(feat_names),
        coverage_threshold=cov_t, fisher_eps=fish_eps,
    )

    S_orig_full = {
        'sel':    dvm_res_orig['S_sel'],
        'ripper': dvm_res_orig['S_ripper'],
        'dt':     dvm_res_orig['S_dt'],
        'union':  dvm_res_orig['S_union'],
    }

    m_total = len(declare_features)
    print(f"\n  Declare-only pipeline complete: {wall:.1f}s")
    print(f"  M_all = {m_total:,}")
    print(f"  |S_orig_full| sel_features  = {len(S_orig_full['sel']):,}  (feature names)")
    print(f"  |S_orig_full| ripper_rules  = {len(S_orig_full['ripper']):,}  (DS: constraint names)")
    print(f"  |S_orig_full| dt_rules      = {len(S_orig_full['dt']):,}  (DS: constraint names)")
    print(f"  |S_orig_full| ripper+dt     = {len(S_orig_full['union']):,}  (DS union)")
    print(f"  X_orig shape: {X_orig.shape}")

    return {
        'case_data':        case_data,
        'declare_features': declare_features,
        'feat_names':       list(feat_names),
        'case_ids_sorted':  case_ids_sorted,
        'labels_orig':      labels_orig,
        'S_orig_full':      S_orig_full,
        'X_orig':           X_orig,
        'positional_log':   positional_log,
        'sequence_log':     sequence_log,
        'dvm_out':          dvm_out,
        'wall_seconds':     wall,
    }


# =============================================================================
# SECTION 2 — RQ2 JOINT PERTURBATION GRID
# =============================================================================

def _joint_cell_worker(
    eps, rho, eps_idx, rho_idx,
    struct_log, X_struct, feat_names, declare_features, case_ids_sorted,
    S_orig_full, S_orig_rq2, coverage_threshold,
):
    return analyze_joint_cell(
        eps=eps, rho=rho, eps_idx=eps_idx, rho_idx=rho_idx,
        struct_log=struct_log, X_struct=X_struct, feat_names=feat_names,
        declare_features=declare_features, case_ids_sorted=case_ids_sorted,
        S_orig_full=S_orig_full, S_orig_rq2=S_orig_rq2,
        coverage_threshold=coverage_threshold, is_anchor_bootstrap=False,
    )


def run_joint_grid(
    case_data_orig, declare_features, feat_names, case_ids_sorted,
    S_orig_full, X_orig, n_jobs=N_JOBS,
) -> tuple:
    """
    Run the full 7x7 joint noise grid in parallel.

    Step 0: Precompute one structural log + X matrix per rho row (7 total).
    Step 1: Run (0,0) anchor sequentially -> establish S_orig_rq2 (all four sets).
    Step 2: Run remaining 48 cells in parallel.

    Returns (all_results, S_orig_rq2)
    where S_orig_rq2 is a dict {'sel': frozenset, 'ripper': frozenset, ...}.
    """
    cov_t = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))

    print("\n" + "=" * 100)
    print("SECTION 2 — JOINT PERTURBATION GRID  (B.3: Label x Structural, DvM BISE 2025)")
    print(f"  eps levels: {JOINT_LABEL_LEVELS}")
    print(f"  rho levels: {JOINT_STRUCT_LEVELS}")
    print(f"  Grid size: {len(JOINT_GRID)} cells  |  Anchor cells: {sorted(ANCHOR_CELLS)}")
    print(f"  B_null (anchor)={B_NULL_ANCHOR}  (four sets per null rep, single evaluation)")
    print(f"  X_struct shared per rho row  |  coverage_threshold={cov_t}")
    print("=" * 100)

    t0 = time.time()

    # Step 0: Precompute structural logs + X matrices per rho row
    print("\n  [Step 0] Precomputing structural logs and X matrices per rho row...",
          flush=True)
    struct_logs = {}
    X_cache     = {}
    for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS):
        if rho == 0.0:
            struct_logs[rho] = case_data_orig
            X_cache[rho]     = X_orig
            print(f"    rho={rho:.2f}  -> reusing X_orig", flush=True)
        else:
            rho_struct_seed = BASE_SEED + rho_idx * 13
            t_rho = time.time()
            sl    = apply_structural_noise(case_data_orig, rho, rho_struct_seed)
            struct_logs[rho] = sl

            pos_log_rho, seq_log_rho, _ = build_log_structures(sl)
            with _suppress():
                X_struct, _ = build_encoding_matrix(
                    case_ids_sorted, pos_log_rho, seq_log_rho,
                    {}, [], {},
                    declare_features, [], [],
                    "declare",
                )
            X_cache[rho] = X_struct
            print(f"    rho={rho:.2f}  -> X computed in {time.time()-t_rho:.1f}s  "
                  f"(shape: {X_struct.shape})", flush=True)

    # Step 1: Run (0,0) anchor to establish S_orig_rq2 (all four sets)
    print("\n  [Step 1] Running (eps=0, rho=0) anchor -> S_orig_rq2...", flush=True)

    anchor_result = analyze_joint_cell(
        eps=0.0, rho=0.0, eps_idx=0, rho_idx=0,
        struct_log=struct_logs[0.0], X_struct=X_cache[0.0], feat_names=feat_names,
        declare_features=declare_features, case_ids_sorted=case_ids_sorted,
        S_orig_full=S_orig_full, S_orig_rq2=None,
        coverage_threshold=cov_t, is_anchor_bootstrap=True,
    )

    S_orig_rq2 = {k: frozenset(anchor_result['S_pert_all_raw'][k])
                  for k in ('sel', 'ripper', 'dt', 'union')}

    print(f"\n  S_orig_rq2 established:")
    for k in ('sel', 'ripper', 'dt', 'union'):
        print(f"    {k:10s}: {len(S_orig_rq2[k]):,} items  "
              f"(S_orig_full: {len(S_orig_full[k]):,})")

    # Step 2: Run remaining 48 cells in parallel
    remaining_jobs = [
        (eps, rho, eps_idx, rho_idx)
        for eps_idx, eps in enumerate(JOINT_LABEL_LEVELS)
        for rho_idx, rho in enumerate(JOINT_STRUCT_LEVELS)
        if not (eps == 0.0 and rho == 0.0)
    ]

    print(f"\n  [Step 2] Running {len(remaining_jobs)} remaining cells "
          f"(n_jobs={n_jobs})...", flush=True)

    results_flat = Parallel(n_jobs=n_jobs, verbose=5, backend='loky')(
        delayed(_joint_cell_worker)(
            eps, rho, eps_idx, rho_idx,
            struct_logs[rho], X_cache[rho], feat_names,
            declare_features, case_ids_sorted,
            S_orig_full, S_orig_rq2, cov_t,
        )
        for eps, rho, eps_idx, rho_idx in remaining_jobs
    )

    wall = time.time() - t0
    print(f"\n  Grid complete. Total wall time: {wall:.1f}s ({wall/3600:.2f} h)")

    all_results = [anchor_result] + list(results_flat)
    all_results.sort(key=lambda x: (x['eps'], x['rho']))

    return all_results, S_orig_rq2


# =============================================================================
# SECTION 3 — OUTPUT GENERATION
# =============================================================================

def build_metrics_table_for_set(all_results, set_key) -> pd.DataFrame:
    """Build a long-format metrics table for one output set across all 49 cells."""
    rows = []
    for res in all_results:
        rm  = res['ref_metrics_all'][set_key]
        fdr = res['fdr_null_all'][set_key]
        rows.append({
            'eps':            res['eps'],
            'rho':            res['rho'],
            'method':         METHOD_DM,
            'output_set':     set_key,
            'R_obs':          res['R_obs_all'][set_key],
            'FDR_ref':        rm['FDR_ref'],
            'Precision':      rm['Precision'],
            'Recall':         rm['Recall'],
            'F1':             rm['F1'],
            'TP':             rm['TP'],
            'FP':             rm['FP'],
            'FN':             rm['FN'],
            'R_full':         rm['R_full'],
            'Jaccard_rq2':    rm['Jaccard_rq2'],
            'Jaccard_full':   rm['Jaccard_full'],
            'Gained_rq2':     rm['Gained_rq2'],
            'Lost_rq2':       rm['Lost_rq2'],
            'reliable':       rm['reliable'],
            'FP_over_Rfull':  rm['FP_over_Rfull'],
            'FN_over_Rfull':  rm['FN_over_Rfull'],
            'FDR_emp':        fdr['FDR_emp'],
            'FDR_CI_lower':   fdr.get('FDR_CI_lower',  float('nan')),
            'FDR_CI_upper':   fdr.get('FDR_CI_upper',  float('nan')),
            'FWER_emp':       fdr.get('FWER_emp',      float('nan')),
            'E_V_b':          fdr.get('E_V_b',         float('nan')),
            'controls_FDR':   fdr.get('controls_FDR',  None),
            'estimable':      fdr.get('estimable',     False),
            'skipped_reason': fdr.get('skipped_reason', None),
            'B_null':         res['B_null'],
            'is_anchor':      res.get('is_anchor', False),
            'n_retries':      res.get('n_retries', 0),
            'eps_seed_used':  res.get('eps_seed_used', None),
        })
    return pd.DataFrame(rows)


def save_joint_outputs(all_results, orig_data, S_orig_rq2, total_wall) -> None:
    """
    Save all DvM BISE 2025 joint-design RQ2 output files.

    Structure:
        sel_features/  — metrics + heatmaps + null_counts + results JSON for sel_features
        rules/         — same for ripper_rules, dt_rules, ripper+dt_rules
    """
    sel_dir   = os.path.join(RQ2_OUTPUT_DIR, "sel_features")
    rules_dir = os.path.join(RQ2_OUTPUT_DIR, "rules")
    os.makedirs(sel_dir,   exist_ok=True)
    os.makedirs(rules_dir, exist_ok=True)

    S_orig_full = orig_data['S_orig_full']
    cov_t       = int(DVM_CONFIG.get("coverage_threshold", DVM_COVERAGE_THRESHOLD))
    m_total     = len(orig_data['declare_features'])

    PIVOT_METRICS = ['FDR_ref', 'Recall', 'F1', 'Jaccard_rq2',
                     'FP_over_Rfull', 'FN_over_Rfull', 'Jaccard_full', 'R_obs']

    SET_INFO = {
        'sel':    ('sel_features', sel_dir),
        'ripper': ('ripper_rules', rules_dir),
        'dt':     ('dt_rules',     rules_dir),
        'union':  ('ripper+dt',    rules_dir),
    }

    for set_key, (_, out_dir) in SET_INFO.items():
        df = build_metrics_table_for_set(all_results, set_key)

        # Long-format CSV
        suf   = set_key
        lf_path = os.path.join(out_dir, f'rq2_dvm_bise2025_joint_metrics_{suf}.csv')
        df.to_csv(lf_path, index=False)
        print(f"  Saved: {lf_path}")

        # Heatmap pivots
        for metric in PIVOT_METRICS:
            if metric not in df.columns:
                continue
            pivot = df.pivot(index='eps', columns='rho', values=metric)
            pivot.index.name = 'eps \\ rho'
            path = os.path.join(
                out_dir,
                f'rq2_dvm_bise2025_joint_{metric.lower()}_{suf}.csv',
            )
            pivot.to_csv(path)

        # Null counts JSON
        nc_json = {}
        for res in all_results:
            key = f"eps_{res['eps']:.4f}_rho_{res['rho']:.4f}"
            nc_json[key] = res['null_counts_all'][set_key]
        nc_path = os.path.join(out_dir, f'rq2_dvm_bise2025_joint_null_counts_{suf}.json')
        with open(nc_path, 'w', encoding='utf-8') as f:
            json.dump(nc_json, f, indent=2)
        print(f"  Saved: {nc_path}")

    # Consolidated sel_features JSON
    sel_json = {
        'rq2_version': '2.0', 'method': METHOD_DM,
        'output_set': 'sel_features', 'log_name': 'Sepsis',
        'timestamp': datetime.now().isoformat(),
        'experiment_design': {
            'block': 'B.3 Joint Label x Structural Noise',
            'joint_label_levels': JOINT_LABEL_LEVELS,
            'joint_struct_levels': JOINT_STRUCT_LEVELS,
            'n_cells': len(JOINT_GRID),
            'anchor_cells': [list(c) for c in sorted(ANCHOR_CELLS)],
            'X_reuse': '7 X computations per rho row',
            'null_reps_note': 'All four output sets from single null log evaluation',
        },
        'config': {
            'ALPHA': ALPHA, 'B_NULL_ANCHOR': B_NULL_ANCHOR,
            'BASE_SEED': BASE_SEED, 'coverage_threshold': cov_t,
            'support_threshold': float(DVM_CONFIG.get('support_threshold', 0.0)),
            'R_orig_sel': int(len(S_orig_full['sel'])), 'm_total': int(m_total),
        },
        'reference_sets': {
            'S_orig_full_sel': int(len(S_orig_full['sel'])),
            'S_orig_rq2_sel':  int(len(S_orig_rq2['sel'])),
        },
        'timing': {'total_seconds': total_wall},
        'joint_results': [
            {
                'eps': r['eps'], 'rho': r['rho'],
                'R_obs': r['R_obs_all']['sel'],
                'ref_metrics': r['ref_metrics_all']['sel'],
                'fdr_null': r['fdr_null_all']['sel'],
                'B_null': r['B_null'], 'is_anchor': r.get('is_anchor', False),
            }
            for r in all_results
        ],
    }
    path = os.path.join(sel_dir, 'rq2_dvm_bise2025_joint_results_sel.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sel_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path}")

    # Consolidated rules JSON
    rules_json = {
        'rq2_version': '2.0', 'method': METHOD_DM,
        'log_name': 'Sepsis', 'timestamp': datetime.now().isoformat(),
        'output_sets': ['ripper_rules', 'dt_rules', 'ripper+dt_rules'],
        'config': {
            'ALPHA': ALPHA, 'B_NULL_ANCHOR': B_NULL_ANCHOR,
            'BASE_SEED': BASE_SEED, 'coverage_threshold': cov_t,
            'RIPPER_K_GRID': list(RIPPER_K_GRID), 'DT_DEPTH_GRID': list(DT_DEPTH_GRID),
        },
        'reference_sets': {
            k: {
                'S_orig_full': int(len(S_orig_full[k])),
                'S_orig_rq2':  int(len(S_orig_rq2[k])),
            }
            for k in ('ripper', 'dt', 'union')
        },
        'timing': {'total_seconds': total_wall},
        'joint_results': [
            {
                'eps': r['eps'], 'rho': r['rho'],
                'R_obs_all': {k: r['R_obs_all'][k] for k in ('ripper', 'dt', 'union')},
                'ref_metrics': {k: r['ref_metrics_all'][k] for k in ('ripper', 'dt', 'union')},
                'fdr_null': {k: r['fdr_null_all'][k] for k in ('ripper', 'dt', 'union')},
                'B_null': r['B_null'], 'is_anchor': r.get('is_anchor', False),
            }
            for r in all_results
        ],
    }
    path = os.path.join(rules_dir, 'rq2_dvm_bise2025_joint_results_rules.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rules_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 4 — SUMMARY PRINTING
# =============================================================================

def _fmt(v, fmt='.4f') -> str:
    return f"{v:{fmt}}" if v == v else '  NaN'


def _print_joint_summary(all_results) -> None:
    """Print FDR_ref heatmap for sel_features and compact table."""
    print(f"\n  FDR_ref heatmap (sel_features) — {METHOD_DM}")
    print(f"  {'eps \\ rho':>8s}", end='')
    for rho in JOINT_STRUCT_LEVELS:
        print(f"  {rho:>6.2f}", end='')
    print()
    print(f"  {'─'*62}")
    for eps in JOINT_LABEL_LEVELS:
        print(f"  {eps:>8.2f}", end='')
        for rho in JOINT_STRUCT_LEVELS:
            res = next(
                (r for r in all_results if r['eps'] == eps and r['rho'] == rho), None
            )
            if res is None:
                print(f"  {'?':>6s}", end='')
            else:
                v = res['ref_metrics_all']['sel']['FDR_ref']
                print(f"  {_fmt(v, '.4f'):>6s}", end='')
        print()
    print(f"  {'─'*62}")

    print(f"\n  {'─'*120}")
    print(f"  {'eps':>5s}  {'rho':>5s}  "
          f"{'R_sel':>6s}  {'FDR_ref_sel':>11s}  "
          f"{'R_ripper':>8s}  {'FDR_ref_r':>9s}  "
          f"{'R_dt':>5s}  {'FDR_ref_dt':>10s}  "
          f"{'anchor':>6s}")
    print(f"  {'─'*120}")
    for res in sorted(all_results, key=lambda x: (x['eps'], x['rho'])):
        rm_sel    = res['ref_metrics_all']['sel']
        rm_ripper = res['ref_metrics_all']['ripper']
        rm_dt     = res['ref_metrics_all']['dt']
        print(
            f"  {res['eps']:>5.2f}  {res['rho']:>5.2f}  "
            f"{res['R_obs_all']['sel']:>6d}  {_fmt(rm_sel['FDR_ref']):>11s}  "
            f"{res['R_obs_all']['ripper']:>8d}  {_fmt(rm_ripper['FDR_ref']):>9s}  "
            f"{res['R_obs_all']['dt']:>5d}  {_fmt(rm_dt['FDR_ref']):>10s}  "
            f"{'YES' if res.get('is_anchor') else '-':>6s}"
        )
    print(f"  {'─'*120}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 100)
    print("RQ2 — SPECIFICATION QUALITY DEGRADATION: SEPSIS  [DvM BISE 2025]")
    print("Block B.3: Joint Label x Structural Noise  (7x7 = 49 cells)")
    print("Four output sets: sel_features, ripper_rules, dt_rules, ripper+dt")
    print("Single pipeline run per cell — no re-evaluation per set")
    print("=" * 100)
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  eps grid:  {JOINT_LABEL_LEVELS}")
    print(f"  rho grid:  {JOINT_STRUCT_LEVELS}")
    print(f"  Grid:    {len(JOINT_GRID)} cells  |  Anchors: {sorted(ANCHOR_CELLS)}")
    print(f"  coverage_threshold = {DVM_CONFIG.get('coverage_threshold', DVM_COVERAGE_THRESHOLD)}")
    print(f"  B_null (anchor) = {B_NULL_ANCHOR}")
    print(f"  Output: {RQ2_OUTPUT_DIR}/sel_features/  and  {RQ2_OUTPUT_DIR}/rules/")
    print("=" * 100)

    t_total = time.time()
    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    orig_data = run_original_data_dvm()

    all_results, S_orig_rq2 = run_joint_grid(
        case_data_orig   = orig_data['case_data'],
        declare_features = orig_data['declare_features'],
        feat_names       = orig_data['feat_names'],
        case_ids_sorted  = orig_data['case_ids_sorted'],
        S_orig_full      = orig_data['S_orig_full'],
        X_orig           = orig_data['X_orig'],
        n_jobs           = N_JOBS,
    )

    total_wall = time.time() - t_total

    print("\n" + "=" * 100)
    print("SECTION 3 — SAVING OUTPUTS")
    print("=" * 100)
    save_joint_outputs(all_results, orig_data, S_orig_rq2, total_wall)

    print(f"\n{'='*100}")
    print("RQ2 — SEPSIS DvM BISE 2025 COMPLETE  (Joint Noise, four output sets)")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/3600:.2f} h)")
    print(f"  Output: {RQ2_OUTPUT_DIR}/sel_features/  and  {RQ2_OUTPUT_DIR}/rules/")

    _print_joint_summary(all_results)

    print(f"\n{'='*100}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 Degradation — Sepsis (DvM BISE 2025, four output sets)  "
            "B.3 Joint Label x Structural Noise, 7x7 grid"
        )
    )
    parser.add_argument('--n-jobs', type=int, default=N_JOBS)
    parser.add_argument('--alpha',  type=float, default=ALPHA)
    parser.add_argument('--b-null-anchor', type=int, default=B_NULL_ANCHOR)
    parser.add_argument('--dry-run', action='store_true',
                        help='3x3 subgrid, B_null_anchor=2')
    args = parser.parse_args()

    if args.dry_run:
        JOINT_LABEL_LEVELS  = [0.00, 0.10, 0.50]
        JOINT_STRUCT_LEVELS = [0.00, 0.10, 1.00]
        JOINT_GRID[:] = [
            (eps, rho)
            for eps in JOINT_LABEL_LEVELS
            for rho in JOINT_STRUCT_LEVELS
        ]
        ANCHOR_CELLS = frozenset({
            (0.00, 0.00), (0.50, 0.00), (0.00, 1.00), (0.50, 1.00),
        })
        B_NULL_ANCHOR = 2
        print("*** DRY-RUN MODE: 3x3 grid, B_null_anchor=2 ***")
    else:
        B_NULL_ANCHOR = args.b_null_anchor

    N_JOBS = args.n_jobs
    ALPHA  = args.alpha

    main()
