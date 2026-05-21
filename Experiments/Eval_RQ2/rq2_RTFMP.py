"""
RQ2 — DO PATTERNS CARRY DISCRIMINATIVE SIGNAL?
===============================================
Full Evaluation Pipeline: Steps 1–6  (RTFMP)

STEP 1: Retrieve holds_all from Phase 1 output — reuse if in-memory or
        cached to disk; recompute only as last resort.
        CRITICAL: outcome signal ('Send for Credit Collection') is stripped
        from traces before any pattern evaluation, matching Phase 1 preprocessing.
STEP 2: Build binary feature matrices X ∈ {0,1}^{n×k} for 9 pattern sets.
STEP 3: 5-times repeated stratified 5-fold nested CV (25 outer folds).
         Inner loop: 5-fold grid search over LR-L1 and RF.
STEP 4: Wilcoxon signed-rank test on 25 paired AUROC differences,
         Holm-Bonferroni correction across 8 competitors.
STEP 5: Random-k baseline (30 random samples of k patterns, same CV).
STEP 6: Direction-aware post-hoc analysis:
         6a. Direction-stratified ablation (Ours_Positive, Ours_Negative)
         6b. Learned-direction consistency (LR β-sign vs Phase 1 direction)
         6c. Direction-weighted RF feature importance (MDI by direction)

PATTERN SETS (from Phase 1 v9.0 single-gate Hou-Storey architecture):
    PRIMARY SET:
    Ours             — is_significant_final == True
                       = "Both" ∪ "Discriminative only" (Hou-Storey q_Hou ≤ α)
                       Hou (2005) weighted T_Hou = -2[w_s ln p_s + w_d ln p_d],
                       w_d = B_label / (B_label + B2_test) = 1500/2500 = 0.60,
                       w_s = B2_test / (B_label + B2_test) = 1000/2500 = 0.40.
                       Gao (2023) adaptive Storey π̂₀, doubly-null calibrated.

    P1 TAXONOMY SUB-CATEGORIES (ablation competitors):
    Ours_Both        — significance_category == "Both"
                       (q_Hou ≤ α AND p_struct_dom ≤ α nominal)
    Ours_Disc_Only   — significance_category == "Discriminative only"
                       (q_Hou ≤ α AND p_struct_dom > α)
    Ours_Positive    — is_significant_final AND direction == "Positive"
    Ours_Negative    — is_significant_final AND direction == "Negative"

    EXTERNAL BASELINES:
    Structural       — significance_category == "Structural only"
    BH               — is_significant_bh == True (BH on analytic Hou p-value)
    Union            — is_significant_discriminative OR is_significant_structural
    All              — holds_all.keys()

STATISTICAL TESTING:
    Per log, for each of 8 competitors vs "Ours":
        Wilcoxon signed-rank test on 25 paired AUROC differences
        Holm-Bonferroni correction across 8 tests
        Rank-biserial r as effect size

RANDOM-k BASELINE:
    For r = 1..30: sample k patterns uniformly from holds_all.keys()
    Run same 5×5×5 CV; report mean ± std of 30×25 AUROC scores.

DIRECTION-AWARE ANALYSIS (Step 6):
    6a. Ablation: Ours vs Ours_Positive vs Ours_Negative — tests whether
        both directional sub-populations contribute complementary signal.
    6b. Consistency: For LR-L1 outer folds, verify sign(β_j) aligns with
        Phase 1 direction label for each feature j.
    6c. Importance: For RF outer folds, compute MDI per feature, stratified
        by Positive/Negative direction.

Version: 4.0-RTFMP-P1-ALIGNED
Author: Ahmed Nour Abdesselam
Institution: Free University of Bozen-Bolzano
Date: March 2026

References:
-----------
- Cawley & Talbot (2010). On Over-fitting in Model Selection. JMLR 11:2079-2107.
- Varma & Simon (2006). Bias in error estimation. BMC Bioinformatics 7:91.
- Kohavi (1995). A study of cross-validation. IJCAI 14:1137-1143.
- Demšar (2006). Statistical Comparisons of Classifiers. JMLR 7:1-30.
- Chicco & Jurman (2020). MCC advantages. BMC Genomics 21:6.
- Fawcett (2006). An introduction to ROC analysis. Pattern Recognit. Lett.
- Di Ciccio & Montali (2022). DECLARE constraint semantics.
- Di Francescomarino et al. (2022). Deviance mining with DECLARE encoding.
- Teinemaa et al. (2019). Outcome-oriented predictive process monitoring. TKDE.
- Tibshirani (1996). LASSO. JRSS-B 58(1):267-288.
- Breiman (2001). Random Forests. Machine Learning 45(1):5-32.
- Storey (2002). A direct approach to FDR. JRSS-B 64(3):479-498.
- Hou (2005). A simple approximation for the distribution of the weighted combination of non-independent or independent chi-squares. Stat. Prob. Lett. 73:179-187.
- Gao (2023). Adaptive Storey π̂₀ estimator. arXiv:2310.06357.
- Berger (1982). Multiparameter Hypothesis Testing. Technometrics. [IUT — superseded in P1 v9.0 by Hou 2005 combination]
- de Leoni & Mannhardt (2015). Road Traffic Fine Management Process Event Log. 4TU.ResearchData.
"""

import os
import sys
import json
import time
import pickle
import warnings
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import seaborn as sns

from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ============================================================================
# PUBLICATION-QUALITY PLOT SETTINGS
# ============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'ours':            '#332288',
    'ours_pos':        '#0072B2',
    'ours_neg':        '#D55E00',
    'discriminative':  '#117733',
    'structural':      '#882255',
    'bh':              '#44AA99',
    'union':           '#88CCEE',
    'all':             '#999999',
    'random':          '#DDCC77',
    'class0':          '#D55E00',
    'class1':          '#0072B2',
    'threshold':       '#CC0000',
    'accent':          '#009E73',
}

SET_COLORS = {
    'Ours':            COLORS['ours'],
    'Ours_Both':       COLORS['discriminative'],   # "Both" sub-category — forest green
    'Ours_Disc_Only':  COLORS['accent'],           # "Discriminative only" sub-category
    'Ours_Positive':   COLORS['ours_pos'],
    'Ours_Negative':   COLORS['ours_neg'],
    'Structural':      COLORS['structural'],
    'BH':              COLORS['bh'],
    'Union':           COLORS['union'],
    'All':             COLORS['all'],
    'Random-k':        COLORS['random'],
}

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_SEED = 42  # Master seed — all random states derived from this

# ── Log configuration ────────────────────────────────────────────────────────
# outcome_mode='activity_present': class 1 iff outcome_signal ∈ trace activities.
# outcome_strip_from_trace: if True, strip outcome_signal from trace BEFORE
#   pattern evaluation. CRITICAL for RTFMP: Phase 1 strips 'Send for Credit Collection'
#   from traces to prevent trivially dominant patterns; RQ2 must match.
LOG_CONFIGS = {
    'RTFMP': {
        'csv': 'Road_Traffic_Fine_Management_Process.csv',
        'declare_spec': 'phase0_RTFMP.json',
        'phase1_dir': 'RTFMP_Results',
        'phase1_json': 'RTFMP_Results/three_hypothesis_houfdr_results.json',
        'outcome_signal': 'Send for Credit Collection',
        'outcome_mode': 'activity_present',
        'outcome_strip_from_trace': True,   # CRITICAL: match Phase 1 preprocessing
        'case_col': 'case:concept:name',
        'act_col': 'concept:name',
        'ts_col': 'time:timestamp',
    },
}

# RQ2 output directory
RQ2_OUTPUT_DIR = 'RQ2_RTFMP_Parallel'

# LOG_CONFIGS = {
#     'RTFMP': {
#         'csv': '../Phase 1 - KM Catalog Construction/Experiments data/CSV/Road_Traffic_Fine_Management_Process.csv',
#         'declare_spec': '../Phase 1 - KM Catalog Construction/Experiments data/Experiments/Results/DECspec_RTFMP/phase0_RTFMP.json',
#         'phase1_dir': '../Phase 1 - KM Catalog Construction/Experiments data/Experiments/Results/RTFMP_Results',
#         'phase1_json': '../Phase 1 - KM Catalog Construction/Experiments data/Experiments/Results/RTFMP_Results/three_hypothesis_houfdr_results.json',
#         'outcome_signal': 'Send for Credit Collection',
#         'outcome_mode': 'activity_present',
#         'outcome_strip_from_trace': True,   # CRITICAL: match Phase 1 preprocessing
#         'case_col': 'case:concept:name',
#         'act_col': 'concept:name',
#         'ts_col': 'time:timestamp',
#     },
# }

# # RQ2 output directory
# RQ2_OUTPUT_DIR = '../Experiments data/Experiments/Results/RQ2_RTFMP'

# ── CV configuration (fixed across all logs and pattern sets) ─────────────
N_OUTER_SPLITS  = 5
N_OUTER_REPEATS = 5   # → 25 outer test folds per pattern set per log
N_INNER_SPLITS  = 5

# ── Hyperparameter grids ─────────────────────────────────────────────────
LR_C_GRID      = [0.01, 0.1, 1, 10, 100]
RF_DEPTH_GRID  = [3, 5, 10, None]
RF_N_ESTIMATORS = 500

# ── Random-k baseline ────────────────────────────────────────────────────
N_RANDOM_SAMPLES = 30   # Number of random pattern samples
N_JOBS = -1             # Outer parallelism for Step 5 (-1 = all cores, set via --n-jobs)

# ── Feature encoding ─────────────────────────────────────────────────────
MIN_FEATURES = 3        # k < 3 → skip

# ── Ternary sensitivity analysis ─────────────────────────────────────────
RUN_TERNARY_SENSITIVITY = True

# ── Required Phase 1 JSON keys (for schema validation) ──────────────────
REQUIRED_PHASE1_KEYS = {
    'constraint_type',
    'activity_a',
    'direction',
    'storey_fdr',   # sub-object
    'bh_fdr',       # sub-object
}
REQUIRED_STOREY_KEYS = {
    'significance_category',
    'is_significant_structural',
    'is_significant_discriminative',
    'is_significant_final',
}
REQUIRED_BH_KEYS = {'is_significant'}


# ============================================================================
# DECLARE CONSTRAINT CHECKERS (self-contained, identical to Phase 1)
# ============================================================================

UNARY_CONSTRAINTS = ['Init', 'End']
BINARY_POSITIVE_CONSTRAINTS = [
    'Response', 'AlternateResponse', 'ChainResponse',
    'Succession', 'AlternateSuccession', 'ChainSuccession',
]
BINARY_NEGATIVE_CONSTRAINTS = ['NotResponse', 'NotChainSuccession']
ALL_CONSTRAINT_TYPES = (
    UNARY_CONSTRAINTS + BINARY_POSITIVE_CONSTRAINTS + BINARY_NEGATIVE_CONSTRAINTS
)
# NOTE: Precedence family and NotChainResponse/NotSuccession are supported in
# evaluate_pattern_fast dispatch (identical to p1_RTFMP_hou.py) but are NOT
# listed in ALL_CONSTRAINT_TYPES — p1_RTFMP_hou.py generates candidates from
# the 10 types above only. The dispatch handles them if they appear in Phase 0 spec.


def precompute_activity_index(trace, case_id=None):
    idx = {}
    for i, act in enumerate(trace):
        if act not in idx:
            idx[act] = []
        idx[act].append(i)
    return idx


def check_init_fast(activity_index, trace, activity, **kw):
    if len(trace) == 0: return None
    return 1 if trace[0] == activity else 0

def check_end_fast(activity_index, trace, activity, **kw):
    if len(trace) == 0: return None
    return 1 if trace[-1] == activity else 0

def check_Response_trace(idx, trace, x, y, **kw):
    if x not in idx: return None
    yp = set(idx.get(y, []))
    for xp in idx[x]:
        if not any(j > xp for j in yp): return 0
    return 1

def check_AlternateResponse_trace(idx, trace, x, y, **kw):
    if x not in idx: return None
    xpos = sorted(idx[x])
    yp = set(idx.get(y, []))
    for i, xp in enumerate(xpos):
        nx = xpos[i + 1] if i + 1 < len(xpos) else None
        if nx is None: continue
        if not any(xp < j < nx for j in yp): return 0
    return 1

def check_ChainResponse_trace(idx, trace, x, y, **kw):
    if x not in idx: return None
    for xp in idx[x]:
        if not (xp + 1 < len(trace) and trace[xp + 1] == y): return 0
    return 1

def check_Succession_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    yp = set(idx.get(y, []))
    for xp in idx[x]:
        if not any(j > xp for j in yp): return 0
    xp_set = set(idx.get(x, []))
    for yp_val in idx[y]:
        if not any(j < yp_val for j in xp_set): return 0
    return 1

def check_AlternateSuccession_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    xpos = sorted(idx[x])
    yp = set(idx.get(y, []))
    for i, xp in enumerate(xpos):
        nx = xpos[i + 1] if i + 1 < len(xpos) else None
        if nx is None: continue
        if not any(xp < j < nx for j in yp): return 0
    ypos = sorted(idx[y])
    xpos_set = sorted(idx.get(x, []))
    for i, yp_val in enumerate(ypos):
        if i == 0: continue
        lower = ypos[i - 1] + 1
        if not any(lower <= j < yp_val for j in xpos_set): return 0
    return 1

def check_ChainSuccession_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    for xp in idx[x]:
        if not (xp + 1 < len(trace) and trace[xp + 1] == y): return 0
    for yp_val in idx[y]:
        if not (yp_val > 0 and trace[yp_val - 1] == x): return 0
    return 1

def check_NotResponse_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    yp = set(idx[y])
    for xp in idx[x]:
        if any(j > xp for j in yp): return 0
    return 1

def check_NotChainSuccession_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    for xp in idx[x]:
        if xp + 1 < len(trace) and trace[xp + 1] == y: return 0
    for yp_val in idx[y]:
        if yp_val > 0 and trace[yp_val - 1] == x: return 0
    return 1

# ── Precedence-family and Not-Precedence checkers (identical to p1_RTFMP_hou.py)
#    These are in the evaluate_pattern_fast dispatch but not in ALL_CONSTRAINT_TYPES.
#    Defined here for full dispatch coverage across all DECLARE log types. ──

def check_Precedence_trace(idx, trace, x, y, **kw):
    if y not in idx: return None
    xp = set(idx.get(x, []))
    for yp in idx[y]:
        if not any(j < yp for j in xp): return 0
    return 1

def check_AlternatePrecedence_trace(idx, trace, x, y, **kw):
    if y not in idx: return None
    xpos = sorted(idx.get(x, []))
    ypos = sorted(idx[y])
    for i, yp in enumerate(ypos):
        if i == 0: continue
        lower = ypos[i - 1] + 1
        if not any(lower <= j < yp for j in xpos): return 0
    return 1

def check_ChainPrecedence_trace(idx, trace, x, y, **kw):
    if y not in idx: return None
    for yp in idx[y]:
        if not (yp > 0 and trace[yp - 1] == x): return 0
    return 1

def check_NotChainResponse_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    for xp in idx[x]:
        if xp + 1 < len(trace) and trace[xp + 1] == y: return 0
    return 1

def check_NotSuccession_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    yp = set(idx.get(y, []))
    for xp in idx[x]:
        if any(j > xp for j in yp): return 0
    xp_set = set(idx.get(x, []))
    for yp_val in idx[y]:
        if any(j < yp_val for j in xp_set): return 0
    return 1

def check_NotPrecedence_trace(idx, trace, x, y, **kw):
    if y not in idx: return None
    xp = set(idx.get(x, []))
    for yp in idx[y]:
        if any(j < yp for j in xp): return 0
    return 1

def check_NotChainPrecedence_trace(idx, trace, x, y, **kw):
    if x not in idx or y not in idx: return None
    for yp in idx[y]:
        if yp > 0 and trace[yp - 1] == x: return 0
    return 1


def evaluate_pattern_fast(constraint_type, activity_a, activity_b, trace, activity_index, **kw):
    dispatch = {
        'Init': lambda: check_init_fast(activity_index, trace, activity_a),
        'End': lambda: check_end_fast(activity_index, trace, activity_a),
        'Response': lambda: check_Response_trace(activity_index, trace, activity_a, activity_b),
        'AlternateResponse': lambda: check_AlternateResponse_trace(activity_index, trace, activity_a, activity_b),
        'ChainResponse': lambda: check_ChainResponse_trace(activity_index, trace, activity_a, activity_b),
        'Precedence': lambda: check_Precedence_trace(activity_index, trace, activity_a, activity_b),
        'AlternatePrecedence': lambda: check_AlternatePrecedence_trace(activity_index, trace, activity_a, activity_b),
        'ChainPrecedence': lambda: check_ChainPrecedence_trace(activity_index, trace, activity_a, activity_b),
        'Succession': lambda: check_Succession_trace(activity_index, trace, activity_a, activity_b),
        'AlternateSuccession': lambda: check_AlternateSuccession_trace(activity_index, trace, activity_a, activity_b),
        'ChainSuccession': lambda: check_ChainSuccession_trace(activity_index, trace, activity_a, activity_b),
        'NotResponse': lambda: check_NotResponse_trace(activity_index, trace, activity_a, activity_b),
        'NotChainResponse': lambda: check_NotChainResponse_trace(activity_index, trace, activity_a, activity_b),
        'NotSuccession': lambda: check_NotSuccession_trace(activity_index, trace, activity_a, activity_b),
        'NotChainSuccession': lambda: check_NotChainSuccession_trace(activity_index, trace, activity_a, activity_b),
    }
    fn = dispatch.get(constraint_type)
    if fn is None: return None
    if activity_b is None and constraint_type not in ('Init', 'End'): return None
    try:
        return fn()
    except Exception:
        return None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CaseInfo:
    """
    Minimal case structure for RQ2.

    IMPORTANT — `trace` excludes the outcome signal activity. Retaining it
    would produce trivially dominant DECLARE constraints (e.g. Init/End/Response
    involving 'Send for Credit Collection' at 100% in Class 1 and 0% in Class 0)
    that encode the label rather than genuine process behaviour. This matches P1's
    load_and_preprocess_data() which strips the outcome signal before any
    pattern evaluation (see outcome_strip_from_trace in LOG_CONFIGS).
    """
    case_id: str
    outcome: int
    trace: List[str]
    activity_index: Dict[str, List[int]] = field(default_factory=dict)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_event_log(
    csv_path: str,
    log_config: dict,
    log_name: str,
) -> Dict[str, CaseInfo]:
    """
    Load event log and extract case information.

    CRITICAL FOR RTFMP: When outcome_strip_from_trace is True, the outcome
    signal activity is stripped from traces AFTER labelling but BEFORE pattern
    evaluation. This matches Phase 1 (p1_RTFMP_hou.py) preprocessing exactly:

        Phase 1:  trace = [a for a in all_activities if a != TARGET_ACTIVITY]
        RQ2:      trace = [a for a in raw_trace if a != outcome_signal]

    Without this, Tier 3 recomputation would produce holds_all values
    inconsistent with Phase 1 (patterns evaluated on traces containing
    'Send for Credit Collection' would have different satisfaction values).
    """
    print(f"\n   Loading event log: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop rows with no case assignment (matches Phase 1)
    case_col = log_config['case_col']
    act_col  = log_config['act_col']
    ts_col   = log_config['ts_col']
    df = df.dropna(subset=[case_col])

    outcome_signal = log_config['outcome_signal']
    outcome_mode   = log_config['outcome_mode']
    strip_outcome  = log_config.get('outcome_strip_from_trace', False)

    assert case_col in df.columns, f"Column '{case_col}' not found. Columns: {list(df.columns)}"
    assert act_col in df.columns,  f"Column '{act_col}' not found. Columns: {list(df.columns)}"
    assert ts_col in df.columns,   f"Column '{ts_col}' not found. Columns: {list(df.columns)}"

    print(f"   Columns: case={case_col}, act={act_col}, ts={ts_col}")
    print(f"   Events: {len(df):,}, Cases: {df[case_col].nunique():,}")
    print(f"   Outcome signal: '{outcome_signal}' (mode: {outcome_mode})")
    print(f"   Strip outcome from trace: {strip_outcome}")

    case_data = {}
    n_stripped_events = 0

    for case_id, group in df.groupby(case_col):
        case_events = group.sort_values(ts_col)
        raw_trace = case_events[act_col].tolist()

        if len(raw_trace) == 0:
            continue

        # Determine outcome BEFORE stripping
        if outcome_mode == 'activity_present':
            activities_in_case = set(raw_trace)
            outcome = 1 if outcome_signal in activities_in_case else 0
        else:
            raise ValueError(f"Unknown outcome_mode: '{outcome_mode}'.")

        # Strip outcome signal from trace (CRITICAL for RTFMP Phase 1 consistency)
        if strip_outcome:
            trace = [a for a in raw_trace if a != outcome_signal]
            n_stripped_events += len(raw_trace) - len(trace)
        else:
            trace = raw_trace

        if len(trace) == 0:
            continue

        activity_index = precompute_activity_index(trace, case_id=str(case_id))
        case_data[str(case_id)] = CaseInfo(
            case_id=str(case_id),
            outcome=outcome,
            trace=trace,
            activity_index=activity_index,
        )

    n1 = sum(1 for c in case_data.values() if c.outcome == 1)
    n0 = len(case_data) - n1

    assert n1 > 0, (
        f"FATAL: All {len(case_data)} cases have outcome=0. "
        f"Check LOG_CONFIGS['{log_name}']['outcome_signal']."
    )
    assert n0 > 0, (
        f"FATAL: All {len(case_data)} cases have outcome=1. "
        f"Check LOG_CONFIGS['{log_name}']['outcome_signal']."
    )

    print(f"   Processed: {len(case_data):,} cases")
    if strip_outcome:
        print(f"   Stripped {n_stripped_events:,} '{outcome_signal}' events from traces")
    print(f"   Class 1 (Sent for Credit Collection / Deviant): {n1:,} ({n1/len(case_data)*100:.1f}%)")
    print(f"   Class 0 (No Credit Collection / Normal):        {n0:,} ({n0/len(case_data)*100:.1f}%)")

    return case_data


def load_phase1_results(json_path: str) -> dict:
    """
    Load Phase 1 JSON and construct two working pattern pools.

    P1 v9.0 JSON sub-lists and their formats:
        significant_patterns       — nested pddict (storey_fdr / bh_fdr sub-objects)
                                     contains "Both" ∪ "Discriminative only" (is_significant_final)
        structural_only_patterns   — flat compact dict (fields at top level)
        discriminative_only_patterns — flat compact dict
        all_patterns               — flat compact dict; includes "Neither"; is_significant_bh
                                     at top level (not nested under bh_fdr)

    NOTE: significant_patterns already contains "Discriminative only" (sigresults =
    [r for r in patternresults if r.is_significant_final]). discriminative_only_patterns
    stores the same records again. all_nested is deduplicated by spec (ct, a, b) to
    prevent duplicate columns in Ours / Ours_Disc_Only feature matrices.

    Internal pools:
        _working_patterns      — deduplicated non-Neither patterns (Ours, Structural sets)
        _working_patterns_full — all_patterns if available (BH, Union sets — includes "Neither")
    """
    print(f"   Loading Phase 1 results: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ── Load all sub-lists from Phase 1 JSON ──────────────────────────
    sig_pats  = data.get('significant_patterns', [])       # "Both"
    str_pats  = data.get('structural_only_patterns', [])   # "Structural only"
    dis_pats  = data.get('discriminative_only_patterns', [])# "Discriminative only"
    all_pats  = data.get('all_patterns', [])               # ALL tested (includes "Neither")

    n_sig  = len(sig_pats)
    n_str  = len(str_pats)
    n_dis  = len(dis_pats)
    n_full = len(all_pats)

    print(f"   Phase 1 patterns loaded from sub-lists:")
    print(f"     Both (significant_patterns):        {n_sig}  [nested pddict format]")
    print(f"     Structural only:                    {n_str}  [flat compact format]")
    print(f"     Discriminative only:                {n_dis}  [flat compact format]")

    # Deduplicate by (constraint_type, activity_a, activity_b).
    # significant_patterns = "Both" ∪ "Discriminative only"; discriminative_only_patterns
    # stores the same "Discriminative only" records again → dedup prevents duplicate columns.
    seen_specs = set()
    all_nested = []
    for p in sig_pats + str_pats + dis_pats:
        spec = (p['constraint_type'], p['activity_a'], p.get('activity_b'))
        if spec not in seen_specs:
            seen_specs.add(spec)
            all_nested.append(p)
    n_raw = n_sig + n_str + n_dis
    n_removed = n_raw - len(all_nested)
    n_all = len(all_nested)
    if n_removed > 0:
        print(f"     Deduplicated: removed {n_removed} duplicate specs "
              f"(Disc_only appears in both sig_pats and dis_pats)")
    print(f"     Total non-Neither (deduplicated):   {n_all}")
    if all_pats:
        print(f"     all_patterns (full list):           {n_full}  ← used for BH/Union sets")
    else:
        print(f"     all_patterns:                       NOT PRESENT in JSON")

    # ── BH/Union coverage: use all_patterns if available ──────────────
    # "Neither" patterns (q_Hou > α, p_struct_dom > α) are excluded from sub-lists.
    # A "Neither" pattern could still be BH-significant on the analytic Hou p-value.
    # Loading all_patterns ensures BH and Union sets are not undercounted.
    metadata = data.get('metadata', data.get('summary', {}))
    n_tested = metadata.get('total_patterns_tested', None)
    bh_ref   = metadata.get('bh_rejections_reference', None)
    if n_tested is not None:
        n_neither = n_tested - n_all
        if n_neither > 0 and not all_pats:
            print(f"   ⚠️  WARNING: {n_neither} 'Neither' patterns are absent "
                  f"(not stored in any sub-list and all_patterns key not found). "
                  f"BH and Union sets will be built from {n_all}/{n_tested} tested "
                  f"patterns — counts may be lower than Phase 1 reference ({bh_ref}).")
        elif n_neither > 0 and all_pats:
            print(f"   ✓ BH/Union coverage: using all_patterns ({n_full}) "
                  f"which includes {n_neither} 'Neither' patterns.")
        else:
            print(f"   ✓ Coverage: sub-lists cover all {n_tested} tested patterns.")

    if not all_nested:
        print(f"   ⚠️  All sub-lists are empty in Phase 1 JSON.")
        data['_working_patterns']      = []
        data['_working_patterns_full'] = all_pats if all_pats else []
        return data

    # ── Schema validation ─────────────────────────────────────────────
    # Nested pddict: validate via first sig_pats entry (always pddict)
    first = sig_pats[0] if sig_pats else all_nested[0]
    actual_top = set(first.keys())
    missing_top = REQUIRED_PHASE1_KEYS - actual_top
    assert not missing_top, (
        f"Phase 1 JSON schema mismatch — missing top-level keys: {missing_top}. "
        f"Actual keys: {sorted(actual_top)}."
    )
    storey_sample = first.get('storey_fdr', {})
    missing_storey = REQUIRED_STOREY_KEYS - set(storey_sample.keys())
    assert not missing_storey, (
        f"storey_fdr sub-object missing keys: {missing_storey}."
    )
    bh_sample = first.get('bh_fdr', {})
    missing_bh = REQUIRED_BH_KEYS - set(bh_sample.keys())
    assert not missing_bh, (
        f"bh_fdr sub-object missing keys: {missing_bh}."
    )
    print(f"   ✓ Nested pddict schema validated on significant_patterns[0].")

    # Flat compact: validate first str_pats entry (structural_only uses compact format)
    if str_pats:
        first_compact = str_pats[0]
        assert 'significance_category' in first_compact, (
            f"compact_pattern_dict missing 'significance_category' at top level. "
            f"Keys: {sorted(first_compact.keys())}"
        )
        print(f"   ✓ Flat compact schema validated on structural_only_patterns[0].")

    # ── BH count cross-check (best-effort) ────────────────────────────
    # all_nested uses nested pddict for sig_pats entries — bh nested under bh_fdr.
    # all_pats uses flat format — is_significant_bh at top level (no bh_fdr sub-object).
    bh_count_nonnether = sum(
        1 for p in all_nested
        if p.get('bh_fdr', {}).get('is_significant', False)
    )
    bh_count_full = sum(
        1 for p in all_pats
        if p.get('is_significant_bh', False)   # flat format: top-level key
    ) if all_pats else bh_count_nonnether
    if bh_ref is not None:
        if bh_count_full < bh_ref:
            print(f"   ⚠️  BH count (all_patterns): {bh_count_full} "
                  f"(Phase 1 reference: {bh_ref}). "
                  f"Deficit of {bh_ref - bh_count_full} — possible schema mismatch.")
        elif bh_count_nonnether < bh_ref:
            print(f"   ✓ BH count from all_patterns: {bh_count_full} == metadata.bh_rejections_reference"
                  f"  (non-Neither sub-lists alone: {bh_count_nonnether})")
        else:
            print(f"   ✓ BH count: {bh_count_full} == metadata.bh_rejections_reference")

    # ── Direction distributions ───────────────────────────────────────
    dir_all  = Counter(p.get('direction', 'Unknown') for p in all_nested)
    dir_both = Counter(p.get('direction', 'Unknown') for p in sig_pats)
    print(f"   Direction distribution (all sub-lists, n={n_all}): {dict(dir_all)}")
    print(f"   Direction distribution (Both only,    n={n_sig}):  {dict(dir_both)}")

    # ── Inject working pools under consistent internal keys ───────────
    # _working_patterns:      deduplicated non-Neither (Ours, Structural sets)
    # _working_patterns_full: all_patterns (flat, deduplicated by P1) if available,
    #                         else fall back to all_nested (BH/Union may be undercounted)
    data['_working_patterns']      = all_nested
    data['_working_patterns_full'] = all_pats if all_pats else all_nested
    return data


# ============================================================================
# STEP 1 — RETRIEVE / RECONSTRUCT HOLDS_ALL
# ============================================================================

def compute_holds_all(
    case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
) -> Dict[Tuple, Dict[str, int]]:
    """
    Compute holds_all from scratch — O(n × m × L_avg).
    Used only when no cached or in-memory holds_all is available.
    """
    print(f"\n   Computing holds_all for {len(candidates_all):,} patterns "
          f"on {len(case_data):,} cases (from scratch)...")
    holds_all = {}
    case_list = list(case_data.values())

    for ct, a, b in tqdm(candidates_all, desc="   holds_all"):
        holds = {}
        for case in case_list:
            result = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
            if result is not None:
                holds[case.case_id] = result
        holds_all[(ct, a, b)] = holds

    return holds_all


def _load_candidates_from_phase0(spec_path: str) -> List[Tuple]:
    """Load Phase 0 DECLARE spec and return the union candidate list."""
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    allowed = set(ALL_CONSTRAINT_TYPES)

    def _extract(clist):
        return [
            (c['constraint_type'], c['param_a'], c.get('param_b'))
            for c in clist if c['constraint_type'] in allowed
        ]

    lpos = _extract(spec.get('Lpos', {}).get('constraints', []))
    lneg = _extract(spec.get('Lneg', {}).get('constraints', []))
    pos_set = set(lpos)
    union = list(lpos) + [p for p in lneg if p not in pos_set]
    return union


def retrieve_phase1_artifacts(
    log_config: dict,
    log_name: str,
    holds_all_precomputed: Optional[Dict] = None,
) -> Tuple[Dict, Dict, List[str], np.ndarray, List[Tuple], dict]:
    """
    STEP 1: Extract all Phase 1 artifacts required for RQ2.

    Three-tier holds_all resolution:
        1. In-memory reuse
        2. Disk cache (holds_all_cache.pkl)
        3. Recompute (traces already stripped if outcome_strip_from_trace)
    """
    print(f"\n{'='*80}")
    print(f"STEP 1: RETRIEVE PHASE 1 ARTIFACTS — {log_name}")
    print(f"{'='*80}")

    case_data = load_event_log(log_config['csv'], log_config, log_name)
    phase1_results = load_phase1_results(log_config['phase1_json'])

    case_ids_ordered = sorted(case_data.keys())
    y = np.array([case_data[cid].outcome for cid in case_ids_ordered], dtype=np.int8)
    n1, n0 = int(y.sum()), int((1 - y).sum())

    print(f"\n   [Phase 1 → RQ2] n={len(case_ids_ordered)}, "
          f"class-1={n1} ({n1/len(y)*100:.1f}%), class-0={n0} ({n0/len(y)*100:.1f}%)")

    cache_path = os.path.join(log_config['phase1_dir'], 'holds_all_cache.pkl')

    if holds_all_precomputed is not None:
        holds_all = holds_all_precomputed
        print(f"   [Step 1] holds_all REUSED from Phase 1 memory — "
              f"zero recomputation. |holds_all| = {len(holds_all):,}")
    elif os.path.exists(cache_path):
        print(f"   [Step 1] Loading cached holds_all from {cache_path}...")
        with open(cache_path, 'rb') as f:
            holds_all = pickle.load(f)
        print(f"   [Step 1] holds_all loaded from disk cache. "
              f"|holds_all| = {len(holds_all):,}")
    else:
        print(f"   ⚠️  [Step 1] No in-memory or cached holds_all available.")
        print(f"   ⚠️  Recomputing from scratch — traces already stripped "
              f"(outcome_strip_from_trace={log_config.get('outcome_strip_from_trace', False)}).")
        candidates_phase0 = _load_candidates_from_phase0(log_config['declare_spec'])
        print(f"   Phase 0 candidates: {len(candidates_phase0):,}")
        holds_all = compute_holds_all(case_data, candidates_phase0)
        print(f"   Saving holds_all cache to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(holds_all, f, protocol=4)
        print(f"   ✓ Cache saved ({os.path.getsize(cache_path) / 1e6:.1f} MB)")

    candidates_all = list(holds_all.keys())
    print(f"   candidates_all: {len(candidates_all):,} (from holds_all.keys())")

    return holds_all, case_data, case_ids_ordered, y, candidates_all, phase1_results


# ============================================================================
# STEP 2 — BUILD BINARY FEATURE MATRICES FOR ALL 8 PATTERN SETS
# ============================================================================

def _classify_pattern_from_json(pat: dict) -> dict:
    """
    Extract classification fields from a P1 v9.0 JSON entry.

    P1 v9.0 writes TWO structurally different formats depending on which JSON
    sub-list a pattern appears in:

    Nested pddict (significant_patterns only):
        storey_fdr: { significance_category, is_significant_structural,
                      is_significant_discriminative, is_significant_final }
        bh_fdr:     { is_significant }  ← 'is_significant', not 'is_significant_bh'

    Flat compact dict (structural_only_patterns, discriminative_only_patterns,
                       all_patterns):
        significance_category, is_significant_structural, is_significant_discriminative,
        is_significant_final, is_significant_bh  ← all at TOP LEVEL, no sub-objects.

    is_significant_structural: RAW NOMINAL label (p_struct_dom ≤ α) — NOT a Storey
        q-value gate; stored for taxonomy only (P1 Step 5c).
    is_significant_final = is_significant_discriminative = (q_Hou ≤ α) — SOLE gate.
    """
    if 'storey_fdr' in pat:
        # Nested pddict format — significant_patterns only
        storey = pat['storey_fdr']
        is_bh  = pat.get('bh_fdr', {}).get('is_significant', False)
    else:
        # Flat compact format — all other sub-lists and all_patterns
        storey = pat
        is_bh  = pat.get('is_significant_bh', False)

    # is_significant_final may be absent in compact_pattern_dict for older P1 output;
    # fall back to is_significant_discriminative (P1 invariant: final ≡ discriminative).
    is_final = storey.get(
        'is_significant_final',
        storey.get('is_significant_discriminative', False),
    )

    return {
        'spec': (
            pat['constraint_type'],
            pat['activity_a'],
            pat.get('activity_b'),
        ),
        # RAW NOMINAL (p_struct_dom ≤ α) — NOT a Storey q-value gate
        'significance_category':         storey.get('significance_category', 'Neither'),
        'is_significant_structural':     storey.get('is_significant_structural', False),
        # Hou-Storey primary gate (q_Hou ≤ α, Step 5b)
        'is_significant_discriminative': storey.get('is_significant_discriminative', False),
        # Identical to is_significant_discriminative — SOLE significance gate in P1 v9.0
        'is_significant_final':          is_final,
        # BH on analytic Satterthwaite p_Hou — reference comparison only
        'is_significant_bh':             is_bh,
        'direction':                     pat.get('direction', 'Unknown'),
        'dominant_class':                pat.get('dominant_class', None),
    }


# ============================================================================
# PATTERN SET DEFINITIONS — Aligned with P1 v9.0 single-gate architecture
# ============================================================================
#
# P1 SINGLE GATE: is_significant_final = is_significant_discriminative = (q_Hou ≤ α)
#   is_significant_final = True  iff  significance_category in {"Both", "Discriminative only"}
#
# P1 TAXONOMY (descriptive, not a gate):
#   Both               = q_Hou ≤ α  AND  p_struct_dom ≤ α (raw nominal)
#   Discriminative only= q_Hou ≤ α  AND  p_struct_dom > α
#   Structural only    = q_Hou > α  AND  p_struct_dom ≤ α (raw nominal)
#   Neither            = both criteria fail
#
# WEIGHT ALIGNMENT (P1 CONFIG v9.0):
#   B_label = 1500, B2_test = B_trace // 2 = 1000
#   w_d = 1500 / 2500 = 0.60  (discriminative weight, Hou 2005 precision-proportional)
#   w_s = 1000 / 2500 = 0.40  (structural weight, Hou 2005 precision-proportional)
#
# DIRECTION DEFINITION (P1, PatternTestResult.direction):
#   "Positive" = class 1 (Sent for Credit Collection / deviant) has higher prevalence (prev1 ≥ prev0)
#   "Negative" = class 0 (No Credit Collection / normal) has higher prevalence (prev0 > prev1)
# ============================================================================

PATTERN_SET_DEFINITIONS = {
    # PRIMARY: Full P1 output — is_significant_final = True
    # = "Both" ∪ "Discriminative only"  (Hou-Storey gate q_Hou ≤ α, Step 5b)
    'Ours':            lambda p: p['is_significant_final'],

    # P1 taxonomy sub-categories of Ours (ablation competitors)
    'Ours_Both':       lambda p: p['significance_category'] == 'Both',
    'Ours_Disc_Only':  lambda p: p['significance_category'] == 'Discriminative only',

    # Direction-stratified sub-populations (within Ours)
    'Ours_Positive':   lambda p: p['is_significant_final'] and p['direction'] == 'Positive',
    'Ours_Negative':   lambda p: p['is_significant_final'] and p['direction'] == 'Negative',

    # External baselines
    'Structural':      lambda p: p['significance_category'] == 'Structural only',
    'BH':              lambda p: p['is_significant_bh'],
    'Union':           lambda p: p['is_significant_discriminative'] or p['is_significant_structural'],
    'All':             None,  # uses holds_all.keys() directly
}

SET_NAMES_ORDERED = [
    'Ours',
    'Ours_Both', 'Ours_Disc_Only',
    'Ours_Positive', 'Ours_Negative',
    'Structural', 'BH', 'Union', 'All',
]

# The "primary" sets for Wilcoxon comparison (Ours vs each of these)
COMPETITOR_SETS = [
    'Ours_Both', 'Ours_Disc_Only',
    'Ours_Positive', 'Ours_Negative',
    'Structural', 'BH', 'Union', 'All',
]


def build_feature_matrix(
    pattern_specs: List[Tuple],
    holds_all: Dict,
    case_ids_ordered: List[str],
    vacuous_fill: int = 0,
) -> np.ndarray:
    """
    Build X ∈ {0,1}^{n × k} for a given ordered list of pattern specs.

    Encoding (Di Francescomarino et al. 2022):
        X[i, j] = holds_all[pat_j].get(case_i, vacuous_fill)

    NOTE on ternary sensitivity (vacuous_fill=2): creates ordinal encoding
    {0=violated, 1=satisfied, 2=vacuous}. Numerically 2 > 1, implying
    "vacuous > satisfied" — no semantic grounding. This is a sensitivity
    analysis only; binary CWA (vacuous_fill=0) is the canonical encoding.
    """
    n = len(case_ids_ordered)
    k = len(pattern_specs)
    X = np.full((n, k), vacuous_fill, dtype=np.int8)
    cid_to_row = {cid: i for i, cid in enumerate(case_ids_ordered)}

    for j, pspec in enumerate(pattern_specs):
        holds = holds_all.get(pspec, {})
        for cid, val in holds.items():
            row = cid_to_row.get(cid)
            if row is not None:
                X[row, j] = int(val)

    return X


def build_all_feature_matrices(
    phase1_results: dict,
    holds_all: Dict,
    case_ids_ordered: List[str],
    candidates_all: List[Tuple],
) -> Tuple[Dict[str, dict], List[dict]]:
    """
    STEP 2: Construct feature matrices for all 9 pattern sets.

    BH/Union sets draw from _working_patterns_full (all_patterns, includes "Neither")
    to avoid undercounting BH-significant "Neither" patterns.
    All other sets draw from _working_patterns (non-Neither only — correct by construction).

    Returns:
        feature_matrices: Dict[set_name → {X, k, patterns, sparsity, direction_labels}]
        all_pats_classified: List[dict] — non-Neither classified patterns for Step 6
    """
    print(f"\n{'='*80}")
    print("STEP 2: BUILD FEATURE MATRICES — 9 PATTERN SETS (direction-aware)")
    print(f"{'='*80}")

    # Non-Neither patterns: used for Ours/Ours_Both/Ours_Disc_Only/Structural sets
    all_pats = [_classify_pattern_from_json(p) for p in phase1_results.get('_working_patterns', [])]
    # Full pattern list (includes "Neither"): used for BH/Union sets to avoid undercounting
    all_pats_full = [_classify_pattern_from_json(p) for p in phase1_results.get('_working_patterns_full', [])]
    print(f"   Loaded {len(all_pats):,} non-Neither classifications from Phase 1 JSON")
    print(f"   Loaded {len(all_pats_full):,} total classifications (for BH/Union sets)")

    n_ours = sum(1 for p in all_pats if p['is_significant_final'])
    n_both = sum(1 for p in all_pats if p['significance_category'] == 'Both')
    categories_found = Counter(p['significance_category'] for p in all_pats)
    print(f"   Phase 1 verdict distribution: {dict(categories_found)}")
    print(f"   Ours (is_significant_final): {n_ours}  = Both({n_both}) + Disc_Only({n_ours - n_both})")

    if n_ours == 0 and len(all_pats) > 0:
        print(f"   ⚠️  WARNING: Zero 'Ours' patterns (is_significant_final=True) found.")

    # Build spec → direction lookup for direction-aware analysis (Step 6)
    spec_to_direction = {}
    for p in all_pats_full:
        spec_to_direction[p['spec']] = p['direction']

    json_specs = {p['spec'] for p in all_pats_full}
    holds_specs = set(holds_all.keys())
    json_only = json_specs - holds_specs
    if json_only:
        print(f"   ⚠️  {len(json_only)} JSON pattern specs not found in holds_all "
              f"(will be excluded from feature matrices).")

    print(f"\n   {'Set':>15s} {'k':>6s} {'Shape':>14s} {'Sparsity':>10s} {'Pos/Neg':>10s}")
    print(f"   {'─'*65}")

    # Sets that need the full pattern pool (including "Neither") for correct BH/Union counts
    FULL_POOL_SETS = {'BH', 'Union'}

    feature_matrices = {}

    for set_name in SET_NAMES_ORDERED:
        criterion = PATTERN_SET_DEFINITIONS[set_name]
        pat_pool = all_pats_full if set_name in FULL_POOL_SETS else all_pats

        if criterion is None:
            selected = list(candidates_all)
        else:
            selected = [
                p['spec'] for p in pat_pool
                if criterion(p) and p['spec'] in holds_specs
            ]

        k = len(selected)
        X = build_feature_matrix(selected, holds_all, case_ids_ordered)
        sparsity = 1.0 - X.mean() if k > 0 else float('nan')

        # Direction labels for each column (for Step 6 analysis)
        direction_labels = [spec_to_direction.get(s, 'Unknown') for s in selected]
        n_pos = sum(1 for d in direction_labels if d == 'Positive')
        n_neg = sum(1 for d in direction_labels if d == 'Negative')

        print(f"   {set_name:>15s} {k:>6d} {str(X.shape):>14s} {sparsity:>9.3f} "
              f"{n_pos:>4d}/{n_neg:<4d}")

        feature_matrices[set_name] = {
            'X': X,
            'k': k,
            'patterns': selected,
            'sparsity': sparsity,
            'direction_labels': direction_labels,   # DIRECTION-AWARE
        }

    return feature_matrices, all_pats


# ============================================================================
# STEP 3 — NESTED 5×5×5 CROSS-VALIDATION (direction-aware model capture)
# ============================================================================

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """Compute four evaluation metrics on a single held-out fold."""
    return {
        'auroc':  float(roc_auc_score(y_true, y_score)),
        'mcc':    float(matthews_corrcoef(y_true, y_pred)),
        'balacc': float(balanced_accuracy_score(y_true, y_pred)),
        'kappa':  float(cohen_kappa_score(y_true, y_pred)),
    }


def run_nested_cv_for_set(
    X: np.ndarray,
    y: np.ndarray,
    set_name: str,
    base_seed: int = BASE_SEED,
    capture_models: bool = False,
    sklearn_n_jobs: int = -1,  # n_jobs for GridSearchCV and RF; set to 1 when called from within a joblib.Parallel worker
) -> dict:
    """
    5-times repeated stratified 5-fold nested CV for one pattern set.

    When capture_models=True, stores the best model object per outer fold
    for post-hoc direction analysis (Step 6). This is only enabled for
    the 'Ours' set to avoid unnecessary memory usage.
    """
    n, k = X.shape

    if k < MIN_FEATURES:
        print(f"   [{set_name:>15s}] SKIPPED — k={k} < {MIN_FEATURES}")
        return {
            'skipped': True, 'k': k,
            'auroc_scores': np.array([]),
            'mcc_scores': np.array([]),
            'balacc_scores': np.array([]),
            'kappa_scores': np.array([]),
            'best_estimators': [],
            'captured_models': [],
            'mean_auroc': float('nan'),
            'std_auroc': float('nan'),
        }

    print(f"   [{set_name:>15s}] k={k}, n={n} — "
          f"running {N_OUTER_REPEATS}×{N_OUTER_SPLITS}-fold nested CV"
          f"{' (capturing models for Step 6)' if capture_models else ''}...")

    n_total_folds = N_OUTER_REPEATS * N_OUTER_SPLITS
    auroc_scores  = np.empty(n_total_folds)
    mcc_scores    = np.empty(n_total_folds)
    balacc_scores = np.empty(n_total_folds)
    kappa_scores  = np.empty(n_total_folds)
    best_ests     = []
    captured_models = []  # DIRECTION-AWARE: store model objects for Step 6

    fold_counter = 0
    for repeat in range(N_OUTER_REPEATS):
        outer_seed = base_seed + repeat * 1000
        outer_cv = StratifiedKFold(
            n_splits=N_OUTER_SPLITS, shuffle=True, random_state=outer_seed
        )

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            inner_seed = outer_seed + fold_idx * 100
            inner_cv = StratifiedKFold(
                n_splits=N_INNER_SPLITS, shuffle=True, random_state=inner_seed
            )

            # ── Inner loop: LR-L1 ────────────────────────────────────────
            lr_gs = GridSearchCV(
                LogisticRegression(
                    penalty='l1', solver='saga', max_iter=5000,
                    class_weight='balanced', random_state=inner_seed,
                ),
                param_grid={'C': LR_C_GRID},
                cv=inner_cv, scoring='roc_auc',
                n_jobs=sklearn_n_jobs, refit=True,
            )
            lr_gs.fit(X_tr, y_tr)

            # ── Inner loop: RF ────────────────────────────────────────────
            rf_gs = GridSearchCV(
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    class_weight='balanced',
                    random_state=inner_seed, n_jobs=sklearn_n_jobs,
                ),
                param_grid={'max_depth': RF_DEPTH_GRID},
                cv=inner_cv, scoring='roc_auc',
                n_jobs=sklearn_n_jobs, refit=True,
            )
            rf_gs.fit(X_tr, y_tr)

            # ── Family selection ──────────────────────────────────────────
            if lr_gs.best_score_ >= rf_gs.best_score_:
                best_model = lr_gs.best_estimator_
                best_info = {
                    'type': 'LR-L1',
                    'C': lr_gs.best_params_['C'],
                    'inner_auroc': float(lr_gs.best_score_),
                }
            else:
                best_model = rf_gs.best_estimator_
                best_info = {
                    'type': 'RF',
                    'max_depth': rf_gs.best_params_['max_depth'],
                    'inner_auroc': float(rf_gs.best_score_),
                }

            # ── Outer evaluation ──────────────────────────────────────────
            y_score = best_model.predict_proba(X_te)[:, 1]
            y_pred  = best_model.predict(X_te)
            m = compute_fold_metrics(y_te, y_pred, y_score)

            auroc_scores[fold_counter]  = m['auroc']
            mcc_scores[fold_counter]    = m['mcc']
            balacc_scores[fold_counter] = m['balacc']
            kappa_scores[fold_counter]  = m['kappa']
            best_ests.append(best_info)

            # DIRECTION-AWARE: capture trained model for post-hoc analysis
            if capture_models:
                captured_models.append({
                    'model': best_model,
                    'type': best_info['type'],
                    'fold': fold_counter,
                })

            fold_counter += 1

    mu, sigma = auroc_scores.mean(), auroc_scores.std()
    print(f"   [{set_name:>15s}] AUROC = {mu:.4f} ± {sigma:.4f} "
          f"[min={auroc_scores.min():.4f}, max={auroc_scores.max():.4f}]")

    return {
        'skipped': False,
        'k': k,
        'auroc_scores': auroc_scores,
        'mcc_scores': mcc_scores,
        'balacc_scores': balacc_scores,
        'kappa_scores': kappa_scores,
        'best_estimators': best_ests,
        'captured_models': captured_models,
        'mean_auroc': float(mu),
        'std_auroc': float(sigma),
    }


def run_step3_all_sets(
    feature_matrices: Dict[str, dict],
    y: np.ndarray,
    base_seed: int = BASE_SEED,
) -> Dict[str, dict]:
    """STEP 3: Run nested CV for all 9 pattern sets."""
    print(f"\n{'='*80}")
    print("STEP 3: NESTED 5×5×5 CROSS-VALIDATION — 25 OUTER FOLDS PER SET")
    print(f"{'='*80}")
    print(f"   Models: LR-L1 (C ∈ {LR_C_GRID}) + RF (n_est={RF_N_ESTIMATORS}, "
          f"depth ∈ {RF_DEPTH_GRID})")

    cv_results = {}
    for set_name in SET_NAMES_ORDERED:
        fm = feature_matrices[set_name]
        # Capture models only for 'Ours' — needed for Step 6 direction analysis
        capture = (set_name == 'Ours')
        cv_results[set_name] = run_nested_cv_for_set(
            fm['X'], y, set_name=set_name, base_seed=base_seed,
            capture_models=capture,
        )

    return cv_results


# ============================================================================
# STEP 3-BIS — TERNARY SENSITIVITY ANALYSIS (§A.3)
# ============================================================================

def run_ternary_sensitivity(
    feature_matrices: Dict[str, dict],
    holds_all: Dict,
    case_ids_ordered: List[str],
    y: np.ndarray,
    base_seed: int = BASE_SEED,
) -> Dict[str, dict]:
    """Ternary encoding sensitivity analysis: vacuous → 2 instead of 0."""
    print(f"\n{'='*80}")
    print("STEP 3-BIS: TERNARY SENSITIVITY ANALYSIS (§A.3)")
    print(f"{'='*80}")

    ternary_results = {}
    for set_name in ['Ours', 'All']:
        fm = feature_matrices.get(set_name)
        if fm is None or fm['k'] < MIN_FEATURES:
            ternary_results[set_name] = {'skipped': True, 'k': fm['k'] if fm else 0}
            continue

        X_ternary = build_feature_matrix(
            fm['patterns'], holds_all, case_ids_ordered, vacuous_fill=2
        )
        ternary_results[set_name] = run_nested_cv_for_set(
            X_ternary, y, set_name=f"{set_name}_ternary", base_seed=base_seed
        )

    for set_name in ['Ours', 'All']:
        tr = ternary_results.get(set_name, {})
        if not tr.get('skipped', True):
            print(f"\n   [{set_name}] Ternary AUROC = "
                  f"{tr['mean_auroc']:.4f} ± {tr['std_auroc']:.4f}")

    return ternary_results


# ============================================================================
# STEP 4 — WILCOXON SIGNED-RANK TEST + HOLM-BONFERRONI CORRECTION
# ============================================================================

def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation r as effect size for Wilcoxon test."""
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    abs_diff = np.abs(diff)
    ranks = sp_stats.rankdata(abs_diff)
    T_plus = np.sum(ranks[diff > 0])
    T_minus = np.sum(ranks[diff < 0])
    return float((T_plus - T_minus) / (n * (n + 1) / 2))


def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05):
    """Holm-Bonferroni step-down procedure for FWER control."""
    K = len(p_values)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    adjusted = np.zeros(K)
    running_max = 0.0
    for i in range(K):
        adj_val = (K - i) * sorted_p[i]
        running_max = max(running_max, adj_val)
        adjusted[i] = min(running_max, 1.0)

    adjusted_original = np.ones(K)
    adjusted_original[sort_idx] = adjusted
    rejected = adjusted_original <= alpha
    return adjusted_original, rejected


def run_step4_statistical_tests(
    cv_results: Dict[str, dict],
    alpha: float = 0.05,
) -> dict:
    """
    STEP 4: Wilcoxon signed-rank test on 25 paired AUROC differences.
    Now includes Ours_Positive and Ours_Negative as competitors.
    """
    print(f"\n{'='*80}")
    print("STEP 4: WILCOXON SIGNED-RANK TEST + HOLM-BONFERRONI")
    print(f"{'='*80}")

    ours_result = cv_results.get('Ours')
    if ours_result is None or ours_result.get('skipped', True):
        print("   ⚠️  'Ours' pattern set was skipped — cannot run Step 4.")
        return {'skipped': True, 'reason': 'Ours was skipped (k < MIN_FEATURES)'}

    ours_auroc = ours_result['auroc_scores']

    raw_p_values = []
    test_details = []

    print(f"\n   {'Competitor':>15s} {'k':>5s} {'AUROC':>12s} {'Δ_mean':>9s} "
          f"{'W-stat':>8s} {'p_raw':>10s} {'r_rb':>7s}")
    print(f"   {'─'*75}")

    for comp_name in COMPETITOR_SETS:
        comp_result = cv_results.get(comp_name)
        if comp_result is None or comp_result.get('skipped', True):
            test_details.append({
                'competitor': comp_name,
                'skipped': True,
                'k': comp_result['k'] if comp_result else 0,
                'p_raw': 1.0, 'statistic': float('nan'),
                'r_rb': 0.0, 'delta_mean': float('nan'),
                'ours_mean': float(ours_auroc.mean()),
                'comp_mean': float('nan'),
            })
            raw_p_values.append(1.0)
            print(f"   {comp_name:>15s} {'---':>5s} {'SKIPPED':>12s}")
            continue

        comp_auroc = comp_result['auroc_scores']
        assert len(ours_auroc) == len(comp_auroc) == N_OUTER_REPEATS * N_OUTER_SPLITS

        diff = ours_auroc - comp_auroc
        delta_mean = float(diff.mean())

        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
        else:
            stat, p_val = sp_stats.wilcoxon(ours_auroc, comp_auroc, alternative='two-sided')

        r_rb = rank_biserial_r(ours_auroc, comp_auroc)

        test_details.append({
            'competitor': comp_name, 'skipped': False,
            'k': comp_result['k'], 'p_raw': float(p_val),
            'statistic': float(stat), 'r_rb': float(r_rb),
            'delta_mean': delta_mean,
            'ours_mean': float(ours_auroc.mean()),
            'comp_mean': float(comp_auroc.mean()),
            'ours_std': float(ours_auroc.std()),
            'comp_std': float(comp_auroc.std()),
        })
        raw_p_values.append(float(p_val))

        print(f"   {comp_name:>15s} {comp_result['k']:>5d} "
              f"{comp_auroc.mean():.4f}±{comp_auroc.std():.4f} "
              f"{delta_mean:>+8.4f} {stat:>8.1f} {p_val:>10.4e} {r_rb:>+6.3f}")

    # Holm-Bonferroni correction
    raw_p_arr = np.array(raw_p_values)
    adjusted_p, rejected = holm_bonferroni(raw_p_arr, alpha=alpha)

    print(f"\n   Holm-Bonferroni corrected p-values (α={alpha}):")
    print(f"   {'Competitor':>15s} {'p_raw':>10s} {'p_adj':>10s} {'Reject H₀':>10s}")
    print(f"   {'─'*50}")
    for i, comp_name in enumerate(COMPETITOR_SETS):
        td = test_details[i]
        td['p_adjusted'] = float(adjusted_p[i])
        td['rejected'] = bool(rejected[i])
        print(f"   {comp_name:>15s} {raw_p_values[i]:>10.4e} "
              f"{adjusted_p[i]:>10.4e} {'YES' if rejected[i] else 'no':>10s}")

    return {
        'skipped': False, 'alpha': alpha,
        'ours_k': ours_result['k'],
        'ours_mean_auroc': float(ours_auroc.mean()),
        'ours_std_auroc': float(ours_auroc.std()),
        'tests': test_details,
    }


# ============================================================================
# STEP 5 — RANDOM-k BASELINE
# ============================================================================

def run_step5_random_k_baseline(
    candidates_all: List[Tuple],
    holds_all: Dict,
    case_ids_ordered: List[str],
    y: np.ndarray,
    k_target: int,
    n_random_samples: int = N_RANDOM_SAMPLES,
    base_seed: int = BASE_SEED,
    n_jobs: int = 1,   # >1 → parallelize the 30-sample loop via joblib.Parallel
) -> dict:
    """STEP 5: Random-k baseline."""
    print(f"\n{'='*80}")
    print(f"STEP 5: RANDOM-k BASELINE (k={k_target}, R={n_random_samples}, n_jobs={n_jobs})")
    print(f"{'='*80}")

    if k_target < MIN_FEATURES:
        print(f"   ⚠️  k_target={k_target} < {MIN_FEATURES} — skipping.")
        return {'skipped': True, 'k': k_target}

    if k_target > len(candidates_all):
        print(f"   ⚠️  k_target={k_target} > |candidates_all|={len(candidates_all)} — skipping.")
        return {'skipped': True, 'k': k_target}

    def _worker_random(r: int) -> dict:
        rng = np.random.RandomState(base_seed + r)
        sample_idx = rng.choice(len(candidates_all), size=k_target, replace=False)
        sampled_patterns = [candidates_all[i] for i in sample_idx]
        X_rand = build_feature_matrix(sampled_patterns, holds_all, case_ids_ordered)
        # sklearn_n_jobs=1: outer joblib.Parallel is the active parallelism layer
        return run_nested_cv_for_set(
            X_rand, y, set_name=f"Rand-{r:02d}", base_seed=base_seed,
            sklearn_n_jobs=1 if n_jobs != 1 else -1,
        )

    if n_jobs != 1:
        raw_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_worker_random)(r) for r in range(1, n_random_samples + 1)
        )
    else:
        raw_results = [_worker_random(r) for r in range(1, n_random_samples + 1)]

    all_auroc_scores = []
    per_sample_means = []
    for result in raw_results:
        if not result['skipped']:
            all_auroc_scores.append(result['auroc_scores'])
            per_sample_means.append(result['mean_auroc'])

    if len(all_auroc_scores) == 0:
        return {'skipped': True, 'k': k_target}

    all_auroc = np.concatenate(all_auroc_scores)
    per_sample_means = np.array(per_sample_means)

    print(f"\n   Random-k Baseline Summary (k={k_target}, R={len(per_sample_means)}):")
    print(f"   Grand mean AUROC:  {all_auroc.mean():.4f} ± {all_auroc.std():.4f}")
    print(f"   Per-sample means:  {per_sample_means.mean():.4f} ± {per_sample_means.std():.4f}")

    return {
        'skipped': False, 'k': k_target,
        'n_random_samples': len(per_sample_means),
        'all_auroc_flat': all_auroc,
        'per_sample_means': per_sample_means,
        'grand_mean': float(all_auroc.mean()),
        'grand_std': float(all_auroc.std()),
        'per_sample_mean_mean': float(per_sample_means.mean()),
        'per_sample_mean_std': float(per_sample_means.std()),
    }


# ============================================================================
# STEP 6 — DIRECTION-AWARE POST-HOC ANALYSIS
# ============================================================================

def run_step6_direction_analysis(
    cv_results: Dict[str, dict],
    feature_matrices: Dict[str, dict],
) -> dict:
    """
    STEP 6: Direction-aware post-hoc analysis.

    6a. Direction-stratified ablation summary:
        Compare Ours vs Ours_Positive vs Ours_Negative AUROC.
        Scientific question: Do both directional sub-populations contribute
        complementary discriminative information?

    6b. Learned-direction consistency:
        For each LR-L1 model captured in Step 3 (Ours set), verify that
        sign(β_j) aligns with Phase 1 direction label for feature j.
            direction_consistency(j) = 1 if:
                (direction_j == "Positive" AND β_j > 0) OR
                (direction_j == "Negative" AND β_j < 0)
        High consistency (>85%) validates Phase 1 statistical direction.

    6c. Direction-weighted RF feature importance:
        For each RF model, compute mean decrease in impurity (MDI) per feature,
        then aggregate by direction (Positive vs Negative).
    """
    print(f"\n{'='*80}")
    print("STEP 6: DIRECTION-AWARE POST-HOC ANALYSIS")
    print(f"{'='*80}")

    direction_analysis = {
        'ablation': {},
        'consistency': {},
        'importance': {},
    }

    ours_res = cv_results.get('Ours', {})
    if ours_res.get('skipped', True):
        print("   ⚠️  'Ours' was skipped — cannot run direction analysis.")
        return direction_analysis

    ours_fm = feature_matrices.get('Ours', {})
    direction_labels = ours_fm.get('direction_labels', [])
    k = ours_fm.get('k', 0)

    # ── 6a: Direction-stratified ablation summary ────────────────────────
    print(f"\n   ── 6a: Direction-Stratified Ablation ──")

    for sn in ['Ours', 'Ours_Positive', 'Ours_Negative']:
        res = cv_results.get(sn, {})
        if res.get('skipped', True):
            print(f"   {sn:>15s}: SKIPPED (k={res.get('k', 0)})")
            direction_analysis['ablation'][sn] = {'skipped': True, 'k': res.get('k', 0)}
        else:
            print(f"   {sn:>15s}: k={res['k']:>4d}, AUROC = {res['mean_auroc']:.4f} ± {res['std_auroc']:.4f}")
            direction_analysis['ablation'][sn] = {
                'skipped': False, 'k': res['k'],
                'mean_auroc': res['mean_auroc'],
                'std_auroc': res['std_auroc'],
            }

    # Step 6a Wilcoxon tests are EXPLORATORY (post-hoc, uncorrected for FWER).
    # Step 4 (Holm-Bonferroni across 8 competitors) is the primary confirmatory family.
    # These 2 comparisons are descriptive: they quantify directional sub-population
    # contribution and do not constitute a separate null-hypothesis test family.
    ours_pos_res = cv_results.get('Ours_Positive', {})
    ours_neg_res = cv_results.get('Ours_Negative', {})

    for sub_name, sub_res in [('Ours_Positive', ours_pos_res), ('Ours_Negative', ours_neg_res)]:
        if sub_res.get('skipped', True):
            continue
        diff = ours_res['auroc_scores'] - sub_res['auroc_scores']
        if not np.all(diff == 0):
            stat, pval = sp_stats.wilcoxon(ours_res['auroc_scores'], sub_res['auroc_scores'],
                                           alternative='two-sided')
        else:
            stat, pval = 0.0, 1.0
        r_rb = rank_biserial_r(ours_res['auroc_scores'], sub_res['auroc_scores'])
        print(f"   Ours vs {sub_name}: Δ={diff.mean():+.4f}, p={pval:.4e}, r_rb={r_rb:+.3f}")
        direction_analysis['ablation'][f'ours_vs_{sub_name}'] = {
            'delta_mean': float(diff.mean()),
            'p_wilcoxon': float(pval),
            'r_rb': float(r_rb),
        }

    # Complementarity test: if Ours > max(Ours_Positive, Ours_Negative), both directions help
    if not ours_pos_res.get('skipped', True) and not ours_neg_res.get('skipped', True):
        best_sub = max(ours_pos_res['mean_auroc'], ours_neg_res['mean_auroc'])
        complementarity_delta = ours_res['mean_auroc'] - best_sub
        print(f"\n   Complementarity Δ = Ours - max(Ours_Pos, Ours_Neg) = {complementarity_delta:+.4f}")
        if complementarity_delta > 0:
            print(f"   ✓ Positive complementarity: combining both directions yields better AUROC")
        else:
            print(f"   ⚠️  No complementarity: best sub-set matches or exceeds combined")
        direction_analysis['ablation']['complementarity_delta'] = float(complementarity_delta)

    # ── 6b: Learned-direction consistency ────────────────────────────────
    print(f"\n   ── 6b: Learned-Direction Consistency (LR β-sign) ──")

    captured = ours_res.get('captured_models', [])
    if not captured or k == 0:
        print("   ⚠️  No captured models available for consistency analysis.")
    else:
        all_consistencies = []
        n_lr_folds = 0
        n_rf_folds = 0
        per_feature_consistency = np.zeros(k)
        per_feature_count = np.zeros(k)

        for cm in captured:
            model = cm['model']
            model_type = cm['type']

            if model_type == 'LR-L1' and hasattr(model, 'coef_'):
                n_lr_folds += 1
                coefs = model.coef_[0]  # shape (k,)
                assert len(coefs) == k, f"Coefficient length mismatch: {len(coefs)} vs k={k}"

                fold_consistent = 0
                fold_total = 0
                for j in range(k):
                    if direction_labels[j] == 'Unknown':
                        continue
                    fold_total += 1
                    # Positive direction: pattern is more prevalent in class 1
                    # → LR should learn β > 0 (X=1 increases P(Y=1))
                    # Negative direction: pattern is more prevalent in class 0
                    # → LR should learn β < 0 (X=1 decreases P(Y=1))
                    if coefs[j] == 0.0:
                        # L1 zeroed this feature — skip (no directional evidence)
                        fold_total -= 1
                        continue

                    if direction_labels[j] == 'Positive' and coefs[j] > 0:
                        fold_consistent += 1
                        per_feature_consistency[j] += 1
                    elif direction_labels[j] == 'Negative' and coefs[j] < 0:
                        fold_consistent += 1
                        per_feature_consistency[j] += 1

                    per_feature_count[j] += 1

                if fold_total > 0:
                    all_consistencies.append(fold_consistent / fold_total)
            else:
                n_rf_folds += 1

        if all_consistencies:
            mean_consistency = float(np.mean(all_consistencies))
            std_consistency = float(np.std(all_consistencies))
            print(f"   LR-L1 folds analysed: {n_lr_folds}")
            print(f"   RF folds (skipped for β analysis): {n_rf_folds}")
            print(f"   Mean direction consistency: {mean_consistency:.3f} ± {std_consistency:.3f}")
            print(f"   (1.0 = perfect alignment between Phase 1 direction and learned β sign)")

            # Thresholds are informal benchmarks (no formal citation):
            #   ≥0.85 = HIGH, ≥0.70 = MODERATE, <0.70 = LOW.
            if mean_consistency >= 0.85:
                print(f"   ✓ HIGH consistency (≥0.85) — Phase 1 direction labels validated by LR")
            elif mean_consistency >= 0.70:
                print(f"   ~ MODERATE consistency (≥0.70) — most directions confirmed, some noise")
            else:
                print(f"   ⚠️  LOW consistency (<0.70) — Phase 1 direction labels partially contradicted")

            # Per-feature consistency (across LR folds)
            per_feature_rate = np.where(
                per_feature_count > 0,
                per_feature_consistency / per_feature_count,
                np.nan
            )
            n_always_consistent = int(np.nansum(per_feature_rate == 1.0))
            n_never_consistent = int(np.nansum(per_feature_rate == 0.0))
            print(f"   Per-feature: {n_always_consistent}/{k} always consistent, "
                  f"{n_never_consistent}/{k} never consistent")

            direction_analysis['consistency'] = {
                'n_lr_folds': n_lr_folds,
                'n_rf_folds': n_rf_folds,
                'mean_consistency': mean_consistency,
                'std_consistency': std_consistency,
                'per_fold_consistency': [float(c) for c in all_consistencies],
                'per_feature_consistency_rate': [
                    float(r) if not np.isnan(r) else None
                    for r in per_feature_rate
                ],
                'n_always_consistent': n_always_consistent,
                'n_never_consistent': n_never_consistent,
            }
        else:
            print("   ⚠️  No LR-L1 folds captured — all outer folds selected RF.")
            direction_analysis['consistency'] = {'n_lr_folds': 0, 'note': 'All folds selected RF'}

    # ── 6c: Direction-weighted RF feature importance ─────────────────────
    print(f"\n   ── 6c: Direction-Weighted RF Feature Importance (MDI) ──")

    rf_models = [cm for cm in captured if cm['type'] == 'RF' and hasattr(cm['model'], 'feature_importances_')]

    if not rf_models or k == 0:
        print("   ⚠️  No RF models captured for importance analysis.")
    else:
        # Aggregate MDI across all RF folds
        all_importances = np.zeros((len(rf_models), k))
        for i, cm in enumerate(rf_models):
            all_importances[i] = cm['model'].feature_importances_

        mean_importance = all_importances.mean(axis=0)  # (k,)

        # Stratify by direction
        pos_mask = np.array([d == 'Positive' for d in direction_labels])
        neg_mask = np.array([d == 'Negative' for d in direction_labels])

        imp_pos = mean_importance[pos_mask]
        imp_neg = mean_importance[neg_mask]

        total_imp_pos = float(imp_pos.sum()) if len(imp_pos) > 0 else 0.0
        total_imp_neg = float(imp_neg.sum()) if len(imp_neg) > 0 else 0.0
        total_imp = float(mean_importance.sum())

        print(f"   RF folds analysed: {len(rf_models)}")
        print(f"   Positive features: n={pos_mask.sum()}, total MDI share = "
              f"{total_imp_pos/total_imp*100:.1f}%")
        print(f"   Negative features: n={neg_mask.sum()}, total MDI share = "
              f"{total_imp_neg/total_imp*100:.1f}%")
        if len(imp_pos) > 0:
            print(f"   Positive MDI: mean={imp_pos.mean():.4f}, max={imp_pos.max():.4f}")
        if len(imp_neg) > 0:
            print(f"   Negative MDI: mean={imp_neg.mean():.4f}, max={imp_neg.max():.4f}")

        # Top-5 features by importance
        top_idx = np.argsort(mean_importance)[::-1][:min(10, k)]
        patterns = ours_fm.get('patterns', [])
        print(f"\n   Top-10 features by mean MDI:")
        print(f"   {'Rank':>4s} {'Dir':>8s} {'MDI':>8s} {'Pattern'}")
        print(f"   {'─'*80}")
        for rank, idx in enumerate(top_idx, 1):
            pat = patterns[idx] if idx < len(patterns) else ('?', '?', '?')
            dlabel = direction_labels[idx] if idx < len(direction_labels) else '?'
            pat_str = f"{pat[0]}({pat[1]}" + (f", {pat[2]})" if pat[2] else ")")
            print(f"   {rank:>4d} {dlabel:>8s} {mean_importance[idx]:>8.4f} {pat_str}")

        direction_analysis['importance'] = {
            'n_rf_folds': len(rf_models),
            'n_positive_features': int(pos_mask.sum()),
            'n_negative_features': int(neg_mask.sum()),
            'total_mdi_positive': total_imp_pos,
            'total_mdi_negative': total_imp_neg,
            'mdi_share_positive': float(total_imp_pos / total_imp) if total_imp > 0 else 0.0,
            'mdi_share_negative': float(total_imp_neg / total_imp) if total_imp > 0 else 0.0,
            'mean_mdi_positive': float(imp_pos.mean()) if len(imp_pos) > 0 else 0.0,
            'mean_mdi_negative': float(imp_neg.mean()) if len(imp_neg) > 0 else 0.0,
            'top_features': [
                {
                    'rank': rank + 1,
                    'pattern': patterns[idx] if idx < len(patterns) else None,
                    'direction': direction_labels[idx] if idx < len(direction_labels) else 'Unknown',
                    'mean_mdi': float(mean_importance[idx]),
                }
                for rank, idx in enumerate(top_idx)
            ],
        }

    return direction_analysis


# ============================================================================
# OUTPUT GENERATION — TABLES, PLOTS, JSON
# ============================================================================

def save_plot_pdf(fig, filepath, dpi=300):
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"      ✓ Saved: {os.path.basename(filepath)}")


def generate_rq2_outputs(
    log_name: str,
    cv_results: Dict[str, dict],
    feature_matrices: Dict[str, dict],
    stat_tests: dict,
    random_baseline: dict,
    ternary_results: Optional[Dict[str, dict]],
    direction_analysis: dict,
    y: np.ndarray,
    output_dir: str,
    timing: dict,
):
    """Generate all RQ2 outputs: JSON, text report, and plots."""

    log_dir = os.path.join(output_dir, log_name)
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    n1 = int(y.sum())
    n0 = len(y) - n1

    # ════════════════════════════════════════════════════════════════════════
    # JSON OUTPUT
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n   Generating JSON output...")

    json_out = {
        'framework': 'RQ2 — Discriminative Signal Evaluation (Direction-Aware)',
        'version': '4.0-RTFMP-P1-ALIGNED',
        'log': log_name,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'base_seed': BASE_SEED,
            'n_outer_splits': N_OUTER_SPLITS,
            'n_outer_repeats': N_OUTER_REPEATS,
            'n_inner_splits': N_INNER_SPLITS,
            'lr_c_grid': LR_C_GRID,
            'rf_depth_grid': [str(d) for d in RF_DEPTH_GRID],
            'rf_n_estimators': RF_N_ESTIMATORS,
            'min_features': MIN_FEATURES,
            'n_random_samples': N_RANDOM_SAMPLES,
            'run_ternary_sensitivity': RUN_TERNARY_SENSITIVITY,
        },
        'dataset': {
            'n_total': len(y), 'n_class1': n1, 'n_class0': n0,
            'class_ratio': float(n1 / len(y)),
        },
        'pattern_set_sizes': {
            name: {
                'k': fm['k'],
                'n_positive': sum(1 for d in fm['direction_labels'] if d == 'Positive'),
                'n_negative': sum(1 for d in fm['direction_labels'] if d == 'Negative'),
            }
            for name, fm in feature_matrices.items()
        },
        'cv_results': {},
        'statistical_tests': stat_tests,
        'random_baseline': {
            k: v for k, v in random_baseline.items()
            if k not in ('all_auroc_flat', 'per_sample_means')
        },
        'direction_analysis': direction_analysis,
        'timing': timing,
    }

    # CV results
    for name, res in cv_results.items():
        entry = {'k': res['k'], 'skipped': res.get('skipped', False)}
        if not res.get('skipped', False):
            entry.update({
                'mean_auroc': res['mean_auroc'],
                'std_auroc': res['std_auroc'],
                'mean_mcc': float(res['mcc_scores'].mean()),
                'std_mcc': float(res['mcc_scores'].std()),
                'mean_balacc': float(res['balacc_scores'].mean()),
                'std_balacc': float(res['balacc_scores'].std()),
                'mean_kappa': float(res['kappa_scores'].mean()),
                'std_kappa': float(res['kappa_scores'].std()),
                'auroc_scores': res['auroc_scores'].tolist(),
                'mcc_scores': res['mcc_scores'].tolist(),
                'balacc_scores': res['balacc_scores'].tolist(),
                'kappa_scores': res['kappa_scores'].tolist(),
                'best_estimators': res['best_estimators'],
            })
        json_out['cv_results'][name] = entry

    if not random_baseline.get('skipped', True):
        json_out['random_baseline']['per_sample_means'] = random_baseline['per_sample_means'].tolist()

    if ternary_results is not None:
        json_out['ternary_sensitivity'] = {}
        for name, tr in ternary_results.items():
            if not tr.get('skipped', True):
                json_out['ternary_sensitivity'][name] = {
                    'mean_auroc': tr['mean_auroc'], 'std_auroc': tr['std_auroc'],
                    'auroc_scores': tr['auroc_scores'].tolist(),
                }
            else:
                json_out['ternary_sensitivity'][name] = {'skipped': True}

    json_path = os.path.join(log_dir, 'rq2_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✓ JSON: {json_path}")

    # ════════════════════════════════════════════════════════════════════════
    # TEXT REPORT
    # ════════════════════════════════════════════════════════════════════════
    print(f"   Generating text report...")
    rpt = []
    rpt.append("=" * 100)
    rpt.append(f"RQ2 — DO PATTERNS CARRY DISCRIMINATIVE SIGNAL?  [{log_name}]")
    rpt.append(f"Version: 4.0-RTFMP-P1-ALIGNED")
    rpt.append("=" * 100)
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append("")

    rpt.append("─" * 100)
    rpt.append("DATASET")
    rpt.append("─" * 100)
    rpt.append(f"  n = {len(y):,}  (Class 1 Sent for Credit Collection: {n1:,}, Class 0 No Credit Collection: {n0:,})")
    rpt.append(f"  Class ratio: {n1/len(y)*100:.1f}% / {n0/len(y)*100:.1f}%")
    rpt.append(f"  Outcome signal stripped from traces: True (matching Phase 1)")
    rpt.append("")

    rpt.append("─" * 100)
    rpt.append("EXPERIMENTAL SETUP (reproducibility)")
    rpt.append("─" * 100)
    rpt.append(f"  Nested CV: {N_OUTER_REPEATS}×{N_OUTER_SPLITS} outer × {N_INNER_SPLITS} inner = 25 folds")
    rpt.append(f"  LR-L1: penalty='l1', solver='saga', max_iter=5000, class_weight='balanced'")
    rpt.append(f"         C ∈ {LR_C_GRID}")
    rpt.append(f"  RF:    n_estimators={RF_N_ESTIMATORS}, class_weight='balanced'")
    rpt.append(f"         max_depth ∈ {RF_DEPTH_GRID}")
    rpt.append(f"  Scoring: 'roc_auc' (inner loop)")
    rpt.append(f"  Feature gate: k < {MIN_FEATURES} → skip")
    rpt.append(f"  Random-k: R={N_RANDOM_SAMPLES} samples, k = |Ours|")
    rpt.append(f"  Base seed: {BASE_SEED}")
    rpt.append("")

    rpt.append("─" * 100)
    rpt.append("PATTERN SET SIZES (direction-aware)")
    rpt.append("─" * 100)
    for name in SET_NAMES_ORDERED:
        fm = feature_matrices[name]
        n_pos = sum(1 for d in fm['direction_labels'] if d == 'Positive')
        n_neg = sum(1 for d in fm['direction_labels'] if d == 'Negative')
        rpt.append(f"  {name:>15s}: k = {fm['k']:>6,}  sparsity = {fm['sparsity']:.3f}  "
                    f"(Pos: {n_pos}, Neg: {n_neg})")
    rpt.append("")

    rpt.append("─" * 100)
    rpt.append("STEP 3 RESULTS: NESTED CV")
    rpt.append("─" * 100)
    rpt.append(f"  {'Set':>15s} {'k':>6s} {'AUROC':>14s} {'MCC':>14s} "
               f"{'BalAcc':>14s} {'Cohen κ':>14s}")
    rpt.append(f"  {'─'*85}")
    for name in SET_NAMES_ORDERED:
        res = cv_results[name]
        if res.get('skipped', False):
            rpt.append(f"  {name:>15s} {res['k']:>6d} {'SKIPPED':>14s}")
        else:
            rpt.append(
                f"  {name:>15s} {res['k']:>6d} "
                f"{res['mean_auroc']:.4f}±{res['std_auroc']:.4f} "
                f"{res['mcc_scores'].mean():.4f}±{res['mcc_scores'].std():.4f} "
                f"{res['balacc_scores'].mean():.4f}±{res['balacc_scores'].std():.4f} "
                f"{res['kappa_scores'].mean():.4f}±{res['kappa_scores'].std():.4f}"
            )
    rpt.append("")

    # Model selection stability
    rpt.append("─" * 100)
    rpt.append("MODEL SELECTION STABILITY")
    rpt.append("─" * 100)
    for name in SET_NAMES_ORDERED:
        res = cv_results[name]
        if res.get('skipped', False) or not res.get('best_estimators'):
            continue
        ests = res['best_estimators']
        total = len(ests)
        n_lr = sum(1 for e in ests if e['type'] == 'LR-L1')
        rpt.append(f"  {name:>15s}: LR-L1 = {n_lr}/{total} ({n_lr/total*100:.0f}%)  "
                    f"RF = {total - n_lr}/{total} ({(total - n_lr)/total*100:.0f}%)")
    rpt.append("")

    # Step 4
    if not stat_tests.get('skipped', True):
        rpt.append("─" * 100)
        rpt.append("STEP 4: WILCOXON SIGNED-RANK TESTS (Ours vs competitors)")
        rpt.append("─" * 100)
        rpt.append(f"  Ours AUROC: {stat_tests['ours_mean_auroc']:.4f} ± {stat_tests['ours_std_auroc']:.4f}")
        rpt.append(f"  α = {stat_tests['alpha']}, Holm-Bonferroni corrected")
        rpt.append("")
        rpt.append(f"  {'Competitor':>15s} {'Δ_mean':>9s} {'p_raw':>10s} {'p_adj':>10s} "
                    f"{'r_rb':>7s} {'Reject':>8s}")
        rpt.append(f"  {'─'*65}")
        for td in stat_tests['tests']:
            if td.get('skipped', False):
                rpt.append(f"  {td['competitor']:>15s} {'SKIPPED':>9s}")
            else:
                rpt.append(
                    f"  {td['competitor']:>15s} {td['delta_mean']:>+8.4f} "
                    f"{td['p_raw']:>10.4e} {td['p_adjusted']:>10.4e} "
                    f"{td['r_rb']:>+6.3f} {'YES' if td['rejected'] else 'no':>8s}"
                )
        rpt.append("")

    # Step 5
    if not random_baseline.get('skipped', True):
        rpt.append("─" * 100)
        rpt.append(f"STEP 5: RANDOM-k BASELINE (k={random_baseline['k']}, R={random_baseline['n_random_samples']})")
        rpt.append("─" * 100)
        rpt.append(f"  Grand mean AUROC: {random_baseline['grand_mean']:.4f} ± {random_baseline['grand_std']:.4f}")
        ours_res = cv_results.get('Ours')
        if ours_res and not ours_res.get('skipped', False):
            delta = ours_res['mean_auroc'] - random_baseline['grand_mean']
            rpt.append(f"  Ours − Random-k: {delta:+.4f}")
        rpt.append("")

    # Ternary sensitivity
    if ternary_results is not None:
        rpt.append("─" * 100)
        rpt.append("§A.3: TERNARY SENSITIVITY ANALYSIS (vacuous → 2)")
        rpt.append("─" * 100)
        for name in ['Ours', 'All']:
            tr = ternary_results.get(name, {})
            binary_res = cv_results.get(name, {})
            if tr.get('skipped', True) or binary_res.get('skipped', True):
                continue
            delta = tr['mean_auroc'] - binary_res['mean_auroc']
            rpt.append(f"  {name}: binary AUROC = {binary_res['mean_auroc']:.4f}, "
                        f"ternary AUROC = {tr['mean_auroc']:.4f}, Δ = {delta:+.4f}")
        rpt.append("")

    # Step 6: Direction analysis
    rpt.append("─" * 100)
    rpt.append("STEP 6: DIRECTION-AWARE POST-HOC ANALYSIS")
    rpt.append("─" * 100)
    rpt.append("")
    rpt.append("  6a. Direction-Stratified Ablation:")
    for sn in ['Ours', 'Ours_Positive', 'Ours_Negative']:
        abl = direction_analysis.get('ablation', {}).get(sn, {})
        if abl.get('skipped', True):
            rpt.append(f"    {sn:>15s}: SKIPPED (k={abl.get('k', 0)})")
        else:
            rpt.append(f"    {sn:>15s}: k={abl['k']}, AUROC = {abl['mean_auroc']:.4f} ± {abl['std_auroc']:.4f}")
    comp_delta = direction_analysis.get('ablation', {}).get('complementarity_delta')
    if comp_delta is not None:
        rpt.append(f"    Complementarity Δ = {comp_delta:+.4f}")
    rpt.append("")

    rpt.append("  6b. Learned-Direction Consistency (LR β-sign):")
    cons = direction_analysis.get('consistency', {})
    if cons.get('n_lr_folds', 0) > 0:
        rpt.append(f"    LR-L1 folds: {cons['n_lr_folds']}")
        rpt.append(f"    Mean consistency: {cons['mean_consistency']:.3f} ± {cons['std_consistency']:.3f}")
        rpt.append(f"    Always consistent: {cons.get('n_always_consistent', 0)}/{feature_matrices['Ours']['k']}")
    else:
        rpt.append(f"    No LR-L1 folds available.")
    rpt.append("")

    rpt.append("  6c. Direction-Weighted RF Feature Importance:")
    imp = direction_analysis.get('importance', {})
    if imp.get('n_rf_folds', 0) > 0:
        rpt.append(f"    RF folds: {imp['n_rf_folds']}")
        rpt.append(f"    Positive features: n={imp['n_positive_features']}, "
                    f"MDI share = {imp['mdi_share_positive']*100:.1f}%")
        rpt.append(f"    Negative features: n={imp['n_negative_features']}, "
                    f"MDI share = {imp['mdi_share_negative']*100:.1f}%")
    else:
        rpt.append(f"    No RF folds available.")
    rpt.append("")

    # Timing
    rpt.append("─" * 100)
    rpt.append("TIMING")
    rpt.append("─" * 100)
    for k_t, v_t in timing.items():
        rpt.append(f"  {k_t:>30s}: {v_t:.1f}s")
    rpt.append("")
    rpt.append("=" * 100)

    report_path = os.path.join(log_dir, 'rq2_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rpt))
    print(f"   ✓ Report: {report_path}")

    # ════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ════════════════════════════════════════════════════════════════════════
    generate_rq2_plots(
        log_name, cv_results, feature_matrices, stat_tests,
        random_baseline, ternary_results, direction_analysis, plots_dir
    )


def generate_rq2_plots(
    log_name: str,
    cv_results: Dict[str, dict],
    feature_matrices: Dict[str, dict],
    stat_tests: dict,
    random_baseline: dict,
    ternary_results: Optional[Dict[str, dict]],
    direction_analysis: dict,
    plots_dir: str,
):
    """Generate publication-quality RQ2 visualizations including direction-aware plots."""

    print(f"\n   Generating plots...")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 1: AUROC Boxplot Comparison (all 8 sets + Random-k)
    # ═══════════════════════════════════════════════════════════════════════
    print("      [1/8] AUROC Boxplot Comparison...")

    plot_data, plot_labels, plot_colors = [], [], []
    for name in SET_NAMES_ORDERED:
        res = cv_results[name]
        if not res.get('skipped', False):
            plot_data.append(res['auroc_scores'])
            plot_labels.append(f"{name}\n(k={res['k']})")
            plot_colors.append(SET_COLORS.get(name, '#999999'))

    if not random_baseline.get('skipped', True):
        plot_data.append(random_baseline['all_auroc_flat'])
        plot_labels.append(f"Random-k\n(k={random_baseline['k']})")
        plot_colors.append(SET_COLORS['Random-k'])

    if plot_data:
        fig, ax = plt.subplots(figsize=(14, 6))
        bp = ax.boxplot(
            plot_data, patch_artist=True, widths=0.6,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(markersize=4, alpha=0.5),
        )
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        ax.set_xticklabels(plot_labels, fontsize=8)
        ax.set_ylabel('AUROC', fontweight='bold')
        ax.set_title(f'RQ2: AUROC Comparison Across Pattern Sets — {log_name}', fontweight='bold')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance (0.5)')
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
        for sp in ax.spines.values(): sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, os.path.join(plots_dir, '01_auroc_boxplot.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 2: Multi-Metric Bar Chart
    # ═══════════════════════════════════════════════════════════════════════
    print("      [2/8] Multi-Metric Bar Chart...")

    metrics = ['auroc', 'mcc', 'balacc', 'kappa']
    metric_labels = ['AUROC', 'MCC', 'Balanced Acc.', "Cohen's κ"]
    active_sets = [n for n in SET_NAMES_ORDERED if not cv_results[n].get('skipped', False)]

    if active_sets:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
        x = np.arange(len(active_sets))
        w = 0.6

        for ax, metric, metric_label in zip(axes, metrics, metric_labels):
            means, stds, colors = [], [], []
            for name in active_sets:
                scores = cv_results[name][f'{metric}_scores']
                means.append(float(scores.mean()))
                stds.append(float(scores.std()))
                colors.append(SET_COLORS.get(name, '#999999'))

            ax.bar(x, means, w, yerr=stds, capsize=4,
                   color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n}\n(k={cv_results[n]['k']})" for n in active_sets],
                               fontsize=7, rotation=45, ha='right')
            ax.set_title(metric_label, fontweight='bold')
            for sp in ax.spines.values(): sp.set_visible(True)

        plt.suptitle(f'RQ2: Multi-Metric Comparison — {log_name}', fontweight='bold', y=1.02)
        plt.tight_layout()
        save_plot_pdf(fig, os.path.join(plots_dir, '02_multimetric_bars.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 3: Effect Size Plot
    # ═══════════════════════════════════════════════════════════════════════
    print("      [3/8] Effect Size Plot...")

    if not stat_tests.get('skipped', True):
        tests = [t for t in stat_tests['tests'] if not t.get('skipped', False)]
        if tests:
            fig, ax = plt.subplots(figsize=(10, 6))
            comp_names = [t['competitor'] for t in tests]
            r_rbs = [t['r_rb'] for t in tests]
            bar_colors = [SET_COLORS.get(n, '#999999') for n in comp_names]

            y_pos = np.arange(len(comp_names))
            bars = ax.barh(y_pos, r_rbs, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.8)

            for i, (bar, r_val) in enumerate(zip(bars, r_rbs)):
                sig = '*' if tests[i]['rejected'] else ''
                label = f"{r_val:+.3f}{sig}"
                x_pos = r_val + 0.01 if r_val >= 0 else r_val - 0.01
                ha = 'left' if r_val >= 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        label, va='center', ha=ha, fontsize=10, fontweight='bold')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(comp_names, fontsize=10)
            ax.set_xlabel('Rank-biserial $r$ (Ours − Competitor)', fontweight='bold')
            ax.set_title(f'RQ2: Effect Sizes [{log_name}]\n'
                         f'(* = significant after Holm-Bonferroni, α=0.05)', fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8)
            for sp in ax.spines.values(): sp.set_visible(True)
            plt.tight_layout()
            save_plot_pdf(fig, os.path.join(plots_dir, '03_effect_sizes.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 4: Random-k Baseline Distribution
    # ═══════════════════════════════════════════════════════════════════════
    print("      [4/8] Random-k Baseline Distribution...")

    ours_res = cv_results.get('Ours')
    if (not random_baseline.get('skipped', True) and
            ours_res and not ours_res.get('skipped', False)):

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(random_baseline['per_sample_means'], bins=15,
                color=COLORS['random'], alpha=0.6, edgecolor='black', linewidth=1.2,
                label=f"Random-k (R={random_baseline['n_random_samples']})")

        ours_mean = ours_res['mean_auroc']
        ax.axvline(ours_mean, color=COLORS['ours'], linewidth=3,
                   label=f'Ours mean = {ours_mean:.4f}')
        rand_mean = random_baseline['per_sample_mean_mean']
        ax.axvline(rand_mean, color=COLORS['random'], linewidth=2, linestyle='--',
                   label=f'Random mean = {rand_mean:.4f}')

        ax.set_xlabel('Mean AUROC (per random sample)', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'RQ2: Ours vs Random-k Baseline — {log_name}', fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        for sp in ax.spines.values(): sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, os.path.join(plots_dir, '04_random_k_baseline.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 5: Model Selection Stability
    # ═══════════════════════════════════════════════════════════════════════
    print("      [5/8] Model Selection Stability...")

    active_with_ests = [
        n for n in SET_NAMES_ORDERED
        if not cv_results[n].get('skipped', False) and cv_results[n].get('best_estimators')
    ]
    if active_with_ests:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(active_with_ests))
        w = 0.35
        lr_fracs, rf_fracs = [], []
        for name in active_with_ests:
            ests = cv_results[name]['best_estimators']
            total = len(ests)
            n_lr = sum(1 for e in ests if e['type'] == 'LR-L1')
            lr_fracs.append(n_lr / total)
            rf_fracs.append(1.0 - n_lr / total)

        ax.bar(x - w/2, lr_fracs, w, color='#4477AA', edgecolor='black', linewidth=1.2, label='LR-L1')
        ax.bar(x + w/2, rf_fracs, w, color='#EE6677', edgecolor='black', linewidth=1.2, label='RF')
        ax.set_xticks(x)
        ax.set_xticklabels(active_with_ests, fontsize=9, rotation=30, ha='right')
        ax.set_ylabel('Fraction of Outer Folds', fontweight='bold')
        ax.set_title(f'RQ2: Model Selection Stability — {log_name}', fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.set_ylim(0, 1.05)
        for sp in ax.spines.values(): sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, os.path.join(plots_dir, '05_model_selection_stability.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 6: Ternary Sensitivity
    # ═══════════════════════════════════════════════════════════════════════
    print("      [6/8] Ternary Sensitivity Plot...")

    if ternary_results is not None:
        pairs = []
        for name in ['Ours', 'All']:
            tr = ternary_results.get(name, {})
            br = cv_results.get(name, {})
            if not tr.get('skipped', True) and not br.get('skipped', True):
                pairs.append((name, br, tr))

        if pairs:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(pairs))
            w = 0.35
            binary_means  = [p[1]['mean_auroc'] for p in pairs]
            ternary_means = [p[2]['mean_auroc'] for p in pairs]
            binary_stds   = [p[1]['std_auroc'] for p in pairs]
            ternary_stds  = [p[2]['std_auroc'] for p in pairs]

            ax.bar(x - w/2, binary_means, w, yerr=binary_stds, capsize=5,
                   color=COLORS['ours'], alpha=0.8, edgecolor='black', linewidth=1.2,
                   label='Binary (CWA)')
            ax.bar(x + w/2, ternary_means, w, yerr=ternary_stds, capsize=5,
                   color=COLORS['accent'], alpha=0.8, edgecolor='black', linewidth=1.2,
                   label='Ternary (vac → 2)')

            ax.set_xticks(x)
            ax.set_xticklabels([p[0] for p in pairs], fontsize=11)
            ax.set_ylabel('AUROC', fontweight='bold')
            ax.set_title(f'§A.3: Binary vs Ternary Encoding — {log_name}', fontweight='bold')
            ax.legend(frameon=True, fancybox=False, edgecolor='black')
            for sp in ax.spines.values(): sp.set_visible(True)
            plt.tight_layout()
            save_plot_pdf(fig, os.path.join(plots_dir, '06_ternary_sensitivity.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 7: Direction-Stratified Ablation (Step 6a)
    # ═══════════════════════════════════════════════════════════════════════
    print("      [7/8] Direction-Stratified Ablation...")

    dir_sets = ['Ours', 'Ours_Positive', 'Ours_Negative']
    dir_active = [s for s in dir_sets if not cv_results.get(s, {}).get('skipped', True)]

    if len(dir_active) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        dir_data = [cv_results[s]['auroc_scores'] for s in dir_active]
        dir_labels = [f"{s}\n(k={cv_results[s]['k']})" for s in dir_active]
        dir_colors = [SET_COLORS.get(s, '#999999') for s in dir_active]

        bp = ax.boxplot(dir_data, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        for patch, color in zip(bp['boxes'], dir_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        ax.set_xticklabels(dir_labels, fontsize=10)
        ax.set_ylabel('AUROC', fontweight='bold')
        ax.set_title(f'Step 6a: Direction-Stratified Ablation — {log_name}\n'
                     f'Do both Positive and Negative patterns contribute?', fontweight='bold')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        for sp in ax.spines.values(): sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, os.path.join(plots_dir, '07_direction_ablation.pdf'))

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 8: Direction Consistency + Importance (Step 6b/c combined)
    # ═══════════════════════════════════════════════════════════════════════
    print("      [8/8] Direction Consistency & Importance...")

    cons = direction_analysis.get('consistency', {})
    imp = direction_analysis.get('importance', {})

    has_cons = cons.get('n_lr_folds', 0) > 0
    has_imp = imp.get('n_rf_folds', 0) > 0

    if has_cons or has_imp:
        n_panels = int(has_cons) + int(has_imp)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]
        ax_idx = 0

        if has_cons:
            ax = axes[ax_idx]; ax_idx += 1
            fold_cons = cons.get('per_fold_consistency', [])
            ax.hist(fold_cons, bins=10, color=COLORS['ours'], alpha=0.7,
                    edgecolor='black', linewidth=1.2)
            ax.axvline(cons['mean_consistency'], color=COLORS['threshold'], linewidth=2,
                       linestyle='--', label=f"Mean = {cons['mean_consistency']:.3f}")
            ax.axvline(0.85, color='gray', linewidth=1.5, linestyle=':',
                       label='High threshold (0.85)')
            ax.set_xlabel('Direction Consistency (per fold)', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Step 6b: LR β-Sign vs Phase 1 Direction', fontweight='bold')
            ax.legend(frameon=True, fancybox=False, edgecolor='black')
            ax.set_xlim(0, 1.05)
            for sp in ax.spines.values(): sp.set_visible(True)

        if has_imp:
            ax = axes[ax_idx]; ax_idx += 1
            labels = ['Positive\n(class 1 dom.)', 'Negative\n(class 0 dom.)']
            shares = [imp['mdi_share_positive'] * 100, imp['mdi_share_negative'] * 100]
            bar_colors = [COLORS['class1'], COLORS['class0']]

            bars = ax.bar(labels, shares, color=bar_colors, edgecolor='black', linewidth=1.5)
            for bar, v in zip(bars, shares):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                        f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
            ax.set_ylabel('Share of Total MDI (%)', fontweight='bold')
            ax.set_title(f'Step 6c: RF Feature Importance by Direction\n'
                         f'(n_pos={imp["n_positive_features"]}, n_neg={imp["n_negative_features"]})',
                         fontweight='bold')
            ax.set_ylim(0, max(shares) * 1.3 if shares else 100)
            for sp in ax.spines.values(): sp.set_visible(True)

        plt.tight_layout()
        save_plot_pdf(fig, os.path.join(plots_dir, '08_direction_consistency_importance.pdf'))

    print(f"   ✓ Generated visualizations in {plots_dir}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_rq2_single_log(
    log_name: str,
    log_config: dict,
    output_dir: str,
    base_seed: int = BASE_SEED,
    holds_all_precomputed: Optional[Dict] = None,
    n_jobs: int = 1,   # forwarded to run_step5_random_k_baseline
) -> dict:
    """Execute Steps 1–6 for a single event log and generate all outputs."""
    timing = {}
    t0 = time.time()

    # ── Step 1 ──────────────────────────────────────────────────────────
    t_step = time.time()
    holds_all, case_data, case_ids_ordered, y, candidates_all, phase1_results = \
        retrieve_phase1_artifacts(log_config, log_name,
                                 holds_all_precomputed=holds_all_precomputed)
    timing['step1_retrieve'] = time.time() - t_step

    # ── Step 2 ──────────────────────────────────────────────────────────
    t_step = time.time()
    feature_matrices, all_pats_classified = build_all_feature_matrices(
        phase1_results, holds_all, case_ids_ordered, candidates_all
    )
    timing['step2_feature_matrices'] = time.time() - t_step

    # ── Step 3 ──────────────────────────────────────────────────────────
    t_step = time.time()
    cv_results = run_step3_all_sets(feature_matrices, y, base_seed=base_seed)
    timing['step3_nested_cv'] = time.time() - t_step

    # ── Step 3-bis: Ternary sensitivity ─────────────────────────────────
    ternary_results = None
    if RUN_TERNARY_SENSITIVITY:
        t_step = time.time()
        ternary_results = run_ternary_sensitivity(
            feature_matrices, holds_all, case_ids_ordered, y, base_seed=base_seed
        )
        timing['step3bis_ternary'] = time.time() - t_step

    # ── Step 4 ──────────────────────────────────────────────────────────
    t_step = time.time()
    stat_tests = run_step4_statistical_tests(cv_results, alpha=0.05)
    timing['step4_wilcoxon'] = time.time() - t_step

    # ── Step 5 ──────────────────────────────────────────────────────────
    ours_k = feature_matrices['Ours']['k']
    t_step = time.time()
    random_baseline = run_step5_random_k_baseline(
        candidates_all, holds_all, case_ids_ordered, y,
        k_target=ours_k, n_random_samples=N_RANDOM_SAMPLES, base_seed=base_seed,
        n_jobs=n_jobs,
    )
    timing['step5_random_k'] = time.time() - t_step

    # ── Step 6: Direction-aware analysis ────────────────────────────────
    t_step = time.time()
    direction_analysis = run_step6_direction_analysis(cv_results, feature_matrices)
    timing['step6_direction'] = time.time() - t_step

    timing['total'] = time.time() - t0

    # ── Output generation ───────────────────────────────────────────────
    generate_rq2_outputs(
        log_name, cv_results, feature_matrices, stat_tests,
        random_baseline, ternary_results, direction_analysis,
        y, output_dir, timing,
    )

    return {
        'log': log_name,
        'cv_results': cv_results,
        'feature_matrices': feature_matrices,
        'stat_tests': stat_tests,
        'random_baseline': random_baseline,
        'ternary_results': ternary_results,
        'direction_analysis': direction_analysis,
        'y': y,
        'case_ids_ordered': case_ids_ordered,
        'candidates_all': candidates_all,
        'timing': timing,
    }


# ============================================================================
# FILE-PATH INTEGRITY CHECK
# ============================================================================

def validate_log_config(log_name: str, log_config: dict) -> List[str]:
    """Validate that all required files exist for a log configuration."""
    missing = []
    for key in ['csv', 'declare_spec', 'phase1_json']:
        path = log_config.get(key, '')
        if not os.path.exists(path):
            missing.append(f"  {key}: {path}")
    phase1_dir = log_config.get('phase1_dir', '')
    if not os.path.isdir(phase1_dir):
        missing.append(f"  phase1_dir: {phase1_dir}")
    return missing


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("RQ2 — DO PATTERNS CARRY DISCRIMINATIVE SIGNAL?")
    print("Version 4.0 — SEPSIS — P1-ALIGNED (is_significant_final, Hou-Storey single gate)")
    print("=" * 100)
    print(f"\n🎯 EXPERIMENTAL DESIGN:")
    print(f"   9 pattern sets: Ours (is_significant_final), Ours_Both, Ours_Disc_Only, "
          f"Ours_Positive, Ours_Negative, Structural, BH, Union, All")
    print(f"   Nested CV: {N_OUTER_REPEATS}×{N_OUTER_SPLITS} outer × {N_INNER_SPLITS} inner")
    print(f"   Models: LR-L1 (C ∈ {LR_C_GRID}) + RF (n_est={RF_N_ESTIMATORS}, "
          f"depth ∈ {RF_DEPTH_GRID})")
    print(f"   Statistical test: Wilcoxon signed-rank + Holm-Bonferroni ({len(COMPETITOR_SETS)} comparisons)")
    print(f"   Random baseline: {N_RANDOM_SAMPLES} samples × same CV")
    print(f"   Ternary sensitivity: {RUN_TERNARY_SENSITIVITY}")
    print(f"   Direction analysis: Step 6 (ablation + β-sign consistency + RF MDI)")
    print(f"   Outcome stripping: 'Send for Credit Collection' removed from traces (Phase 1 consistency)")
    print(f"   Base seed: {BASE_SEED}")
    print("=" * 100)

    os.makedirs(RQ2_OUTPUT_DIR, exist_ok=True)

    for log_name, log_config in LOG_CONFIGS.items():
        missing = validate_log_config(log_name, log_config)
        if missing:
            print(f"\n⚠️  SKIPPING {log_name} — missing files:")
            for m in missing:
                print(f"   {m}")
            continue

        print(f"\n{'═'*100}")
        print(f"PROCESSING LOG: {log_name}")
        print(f"{'═'*100}")

        total_start = time.time()

        rq2_result = run_rq2_single_log(
            log_name, log_config, RQ2_OUTPUT_DIR, base_seed=BASE_SEED,
            n_jobs=N_JOBS,
        )

        total_time = time.time() - total_start

        # ── Final summary ────────────────────────────────────────────────
        print(f"\n{'='*100}")
        print(f"✅ RQ2 COMPLETE — {log_name}")
        print(f"{'='*100}")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   Output dir: {os.path.join(RQ2_OUTPUT_DIR, log_name)}")

        ours = rq2_result['cv_results'].get('Ours', {})
        if not ours.get('skipped', False):
            print(f"\n   📊 KEY RESULTS:")
            print(f"   Ours (k={ours['k']}): AUROC = {ours['mean_auroc']:.4f} ± {ours['std_auroc']:.4f}")

            # Direction sub-sets
            for sub in ['Ours_Positive', 'Ours_Negative']:
                sub_res = rq2_result['cv_results'].get(sub, {})
                if not sub_res.get('skipped', False):
                    print(f"   {sub} (k={sub_res['k']}): AUROC = "
                          f"{sub_res['mean_auroc']:.4f} ± {sub_res['std_auroc']:.4f}")

            rb = rq2_result.get('random_baseline', {})
            if not rb.get('skipped', True):
                delta = ours['mean_auroc'] - rb['grand_mean']
                print(f"   Random-k (k={rb['k']}): AUROC = {rb['grand_mean']:.4f} ± "
                      f"{rb['grand_std']:.4f}  (Δ = {delta:+.4f})")

            st = rq2_result.get('stat_tests', {})
            if not st.get('skipped', True):
                n_sig = sum(1 for t in st['tests'] if t.get('rejected', False))
                print(f"   Wilcoxon: {n_sig}/{len(COMPETITOR_SETS)} competitors significantly "
                      f"different (Holm-Bonferroni α=0.05)")

            # Direction consistency
            da = rq2_result.get('direction_analysis', {})
            cons = da.get('consistency', {})
            if cons.get('n_lr_folds', 0) > 0:
                print(f"   Direction consistency: {cons['mean_consistency']:.3f} ± "
                      f"{cons['std_consistency']:.3f} (across {cons['n_lr_folds']} LR folds)")

            # Complementarity
            comp_d = da.get('ablation', {}).get('complementarity_delta')
            if comp_d is not None:
                print(f"   Complementarity Δ: {comp_d:+.4f} "
                      f"({'✓ both directions help' if comp_d > 0 else '⚠️ no complementarity'})")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RQ2 — Discriminative Signal Evaluation")
    parser.add_argument(
        '--n-jobs', type=int, default=N_JOBS,
        help=f'Number of parallel workers for Step 5 random-k baseline (-1 = all cores, default: {N_JOBS})'
    )
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main()