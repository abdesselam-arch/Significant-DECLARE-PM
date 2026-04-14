"""
RQ3 DIRECTION 2 — SENSITIVITY ANALYSIS UNDER CONTROLLED LOG PERTURBATIONS
===========================================================================
Sommers et al. (2025) Model-Transformation Perturbation Framework
Applied to the Dual-Axis Storey FDR Discriminative Specification Mining Pipeline

SCIENTIFIC FRAMING:
-------------------
The central claim of Direction 2 is epistemic and distinct from RQ1/RQ2:
NOT whether the framework controls FDR at nominal α under ideal conditions (RQ1),
but whether the dual-axis statistical machinery DEGRADES GRACEFULLY when the
event log deviates from ideality.

This is the robustness dimension of the Process Mining Manifesto
(van der Aalst et al., 2012, criterion 6), formalized by Sommers et al. (2025)
as the ability of a technique "to cope with variations in the data and noise".

PERTURBATION TAXONOMY (Sommers et al. 2025):
---------------------------------------------
P1 — Activity label noise (RIa_in): Swap two activity positions per trace
     with probability ε.  Targets H₀ˢ (structural axis).
     Preserves: activity multiset, class labels.
     Destroys: temporal ordering (partially pre-applies the structural null).

P2 — Class label noise (RIc_in): Flip outcome label for fraction ε of cases.
     Targets H₀ᵈ (discriminative axis).
     Preserves: all trace structures.
     Destroys: class-outcome association.

P3 — Stochastic case-level trace truncation (RIe_mi, v2.0):
         Randomly select fraction ε of cases and truncate each to
         first ⌈(1−ε)·L⌉ events. random_state governs case selection.
         Fix v2.0: previous deterministic version used random_state=ignored;
         corrected to ensure replicates differ across seeds.
     Targets both axes via vacuity inflation.
     Simulates incomplete recording at end-of-process.

INTENSITY LEVELS:
    ε ∈ {0.02, 0.05, 0.10, 0.20, 0.30}
    R = 10 independent replicates per (perturbation_type, ε) pair
    → 3 × 5 × 10 = 150 perturbed logs + 1 clean baseline

METRICS:
    M1 — Jaccard specification stability: |S_clean ∩ S_pert| / |S_clean ∪ S_pert|
    M2 — Empirical FDR under perturbation: |S_pert \ S_clean| / max(|S_pert|, 1)
    M3 — Vacuity rate: fraction of surviving patterns that are vacuously satisfied
    M4 — Axis-specific breakdown: four-category verdict migration tracking

INFLECTION POINT:
    Smallest ε where 95% bootstrap CI of M1 drops below 0.80.

DECLARE FAMILY STRATIFICATION:
    Unary (Init, End), Binary Positive (Response, ChainResponse, ...),
    Binary Negative (NotResponse, NotChainSuccession).

Version: 1.0
Author: Ahmed Nour Abdesselam
Institution: Free University of Bozen-Bolzano
Date: March 2026

References:
-----------
- Sommers et al. (2025). A Taxonomy of Behavioural Deviations in Process Mining.
  Springer. DOI: 10.1007/s44311-025-00006-8
- van der Aalst et al. (2012). Process Mining Manifesto. BPM 2011 Workshops, LNBIP 99.
- Meinshausen & Bühlmann (2010). Stability Selection. JRSS-B 72(4):417-473.
- Storey (2002). A direct approach to false discovery rates. JRSS-B 64(3):479-498.
- Phipson & Smyth (2010). Permutation p-values should never be zero.
- Berger (1982). Multiparameter Hypothesis Testing. Technometrics.
"""

import sys
import os
import json
import time
import copy
import hashlib
import warnings
import math
import argparse
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np             
import pandas as pd
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- File paths (adjust to your directory structure) ---
# INPUT_CSV = "../Phase 1 - KM Catalog Construction/Experiments data/CSV/Sepsis_EL.csv"
# DECLARE_SPEC_FILE = "../Phase 1 - KM Catalog Construction/Experiments data/Experiments/Results/DECspec_Sepsis/phase0_declare_specification_CC.json"
# CLEAN_RESULTS_FILE = "../Phase 1 - KM Catalog Construction/Experiments data/Experiments/Results/Sepsis_ThreeHyp_SAM2/three_hypothesis_samfdr_results.json"
# OUTPUT_DIR = "../Phase 1 - KM Catalog Construction/Experiments data/Experiments/Results/Sepsis_RQ3_Direction"
INPUT_CSV = "Production.csv"
DECLARE_SPEC_FILE = "phase0_Production.json"
CLEAN_RESULTS_FILE = "Production_ThreeHyp_SAM_Parallel/three_hypothesis_samfdr_results.json"
OUTPUT_DIR = "Production_RQ3_Direction"

PLOTS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPLICATES_DIR = os.path.join(OUTPUT_DIR, "replicates")
os.makedirs(REPLICATES_DIR, exist_ok=True)

# --- Import Phase 1 pipeline ---
# Add parent directory to path so we can import the Phase 1 module
# PHASE1_DIR = ""
# sys.path.insert(0, PHASE1_DIR)

# Import from Phase 1 script (Production version)
try:
    from Experiments.P1_SDSM.p1_Production_parallel import (
        CaseInfo, PatternTestResult,
        precompute_activity_index,
        evaluate_pattern_fast,
        compute_holds_by_case_batch,
        compute_prevalence_from_holds,
        split_by_class,
        run_label_permutation_test,
        run_structural_permutation_test,
        fisher_conjunction_pvalue,
        benjamini_hochberg,
        adaptive_storey_pi0,
        storey_qvalue,
        compute_double_null_tf_matrix,
        empirical_fisher_pvalue,
        extract_subgroups_from_case_data,
        determine_applicable_subgroups_with_cases,
        compute_discrimination_for_pattern,
        UNARY_CONSTRAINTS,
        BINARY_POSITIVE_CONSTRAINTS,
        BINARY_NEGATIVE_CONSTRAINTS,
        ALL_CONSTRAINT_TYPES,
        COLORS,
    )
    PHASE1_IMPORTED = True
    print("✓ Phase 1 module imported successfully")
except ImportError as e:
    print(f"⚠️  Phase 1 import failed ({e}). Using inline definitions.")
    PHASE1_IMPORTED = False

# ============================================================================
# INLINE PHASE 1 DEFINITIONS (fallback if import fails)
# ============================================================================
# If the Phase 1 module cannot be imported, we define the essential structures
# and functions inline. The full constraint checker code is replicated from
# the Phase 1 script to ensure scientific reproducibility.

if not PHASE1_IMPORTED:
    from dataclasses import dataclass, field

    UNARY_CONSTRAINTS = ['Init', 'End']
    BINARY_POSITIVE_CONSTRAINTS = [
        'Response', 'AlternateResponse', 'ChainResponse',
        'Succession', 'AlternateSuccession', 'ChainSuccession',
    ]
    BINARY_NEGATIVE_CONSTRAINTS = ['NotResponse', 'NotChainSuccession']
    ALL_CONSTRAINT_TYPES = (
        UNARY_CONSTRAINTS + BINARY_POSITIVE_CONSTRAINTS + BINARY_NEGATIVE_CONSTRAINTS
    )

    COLORS = {
        'class0': '#D55E00', 'class1': '#0072B2', 'both': '#CC79A7',
        'neither': '#999999', 'null': '#E5E5E5', 'accent1': '#009E73',
        'accent2': '#F0E442', 'accent3': '#56B4E9', 'mc': '#E69F00',
        'threshold': '#CC0000', 'structural': '#882255',
        'discriminative': '#117733', 'conjunction': '#332288',
    }

    @dataclass
    class CaseInfo:
        """Information about a single Production manufacturing case."""
        case_id: str
        outcome: int              # 1 = Deviant (Re-submission Required), 0 = Regular (Normal)
        trace: List[str]          # Activity sequence (no stripping — label is a column attribute)
        part_desc: str            # Product type (Part_Desc_ column)
        report_type: str          # Report type: B / D / S (Report_Type column)
        start_timestamp: datetime
        activity_index: Dict[str, List[int]] = field(default_factory=dict)

    @dataclass
    class PatternTestResult:
        pattern_id: str
        constraint_type: str
        activity_a: str
        activity_b: Optional[str]
        prevalence_class0: float
        prevalence_class1: float
        n_applicable_class0: int
        n_applicable_class1: int
        n_satisfied_class0: int
        n_satisfied_class1: int
        delta_obs: float
        p_structural_class0: float
        p_structural_class1: float
        p_structural_dominant: float
        null_mean_class0: float
        null_mean_class1: float
        null_std_class0: float
        null_std_class1: float
        p_discriminative: float
        p_discriminative_onesided: float
        null_delta_mean: float
        null_delta_std: float
        p_conjunction: float
        is_significant_bh: bool
        bh_rank: Optional[int]
        bh_threshold: Optional[float]
        dominant_class: int
        direction: str
        q_value_sam: float = 1.0
        is_significant_sam: bool = False
        is_significant_discriminative: bool = False
        fdp_estimate: float = 1.0
        tau_star_sam: float = float('inf')
        q_structural_class0: float = 1.0
        q_structural_class1: float = 1.0
        q_structural_dominant: float = 1.0
        is_significant_structural: bool = False
        is_significant_final: bool = False
        significance_category: str = "Neither"
        p_structural_screen_class0: float = 1.0   # D4: SCREEN half for scope filter (D3)
        p_structural_screen_class1: float = 1.0   # D4: SCREEN half for scope filter (D3)
        p_conjunction_empirical: float = 1.0      # D4: empirical calibration path
        applicable_subgroups: List[str] = field(default_factory=list)
        subgroup_to_cases: Dict[str, List[str]] = field(default_factory=dict)

    def precompute_activity_index(trace, case_id=None):
        index = {}
        for i, act in enumerate(trace):
            if act not in index:
                index[act] = []
            index[act].append(i)
        return index

    # ---------- DECLARE constraint checkers (copied from Phase 1) ----------

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
        xpos = sorted(idx[x]); yp = set(idx.get(y, []))
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
        if x in idx:
            yp = set(idx.get(y, []))
            for xp in idx[x]:
                if not any(j > xp for j in yp): return 0
        if y in idx:
            xp = set(idx.get(x, []))
            for yp in idx[y]:
                if not any(j < yp for j in xp): return 0
        return 1

    def check_AlternateSuccession_trace(idx, trace, x, y, **kw):
        if x not in idx or y not in idx: return None
        xpos = sorted(idx[x]); yp = set(idx.get(y, []))
        for i, xp in enumerate(xpos):
            nx = xpos[i + 1] if i + 1 < len(xpos) else None
            if nx is None: continue
            if not any(xp < j < nx for j in yp): return 0
        ypos = sorted(idx[y]); xpos_set = sorted(idx.get(x, []))
        for i, yp_val in enumerate(ypos):
            if i == 0: continue
            lower = ypos[i - 1] + 1
            if not any(lower <= j < yp_val for j in xpos_set): return 0
        return 1

    def check_ChainSuccession_trace(idx, trace, x, y, **kw):
        if x not in idx or y not in idx: return None
        for xp in idx[x]:
            if not (xp + 1 < len(trace) and trace[xp + 1] == y): return 0
        for yp in idx[y]:
            if not (yp > 0 and trace[yp - 1] == x): return 0
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
            'Succession': lambda: check_Succession_trace(activity_index, trace, activity_a, activity_b),
            'AlternateSuccession': lambda: check_AlternateSuccession_trace(activity_index, trace, activity_a, activity_b),
            'ChainSuccession': lambda: check_ChainSuccession_trace(activity_index, trace, activity_a, activity_b),
            'NotResponse': lambda: check_NotResponse_trace(activity_index, trace, activity_a, activity_b),
            'NotChainSuccession': lambda: check_NotChainSuccession_trace(activity_index, trace, activity_a, activity_b),
        }
        fn = dispatch.get(constraint_type)
        if fn is None: return None
        if activity_b is None and constraint_type not in ('Init', 'End'): return None
        try: return fn()
        except Exception: return None

    def compute_holds_by_case_batch(cases, candidates):
        results = {}
        case_list = list(cases.values())
        _in_worker = os.environ.get('RQ3_IN_WORKER') == '1'
        for ct, a, b in tqdm(candidates, desc="Holds-by-case", disable=_in_worker):
            holds = {}
            for case in case_list:
                result = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
                if result is not None:
                    holds[case.case_id] = result
            results[(ct, a, b)] = holds
        return results

    def compute_prevalence_from_holds(holds, case_ids):
        n_sat = 0; n_app = 0
        for cid in case_ids:
            if cid in holds:
                n_app += 1
                if holds[cid] == 1: n_sat += 1
        prev = n_sat / n_app if n_app > 0 else 0.0
        return prev, n_sat, n_app

    def split_by_class(case_data):
        D_0 = {cid: c for cid, c in case_data.items() if c.outcome == 0}
        D_1 = {cid: c for cid, c in case_data.items() if c.outcome == 1}
        return D_0, D_1

    def adaptive_storey_pi0(pvals, q=0.05, lambda_max=0.80, delta=None,
                            use_robust=True, **kwargs):
        """
        Gao (2023) Adaptive Storey null proportion estimator (D5 fix).
        Matches p1_Production_parallel.py — replaces Bootstrap-MSE storey_pi0_bootstrap.
        Returns (pi0_final, lambda_star) — same shape as storey_pi0_bootstrap.
        """
        pvals = np.asarray(pvals, dtype=np.float64)
        m = len(pvals)
        if m == 0:
            return 1.0, q
        if np.all(pvals <= q):
            return float(np.minimum(np.mean(pvals > q) / max(1.0 - q, 1e-9), 1.0)), q
        if delta is None:
            delta = max((lambda_max - q) / 50.0, 1.0 / m)
        lambdas = np.arange(q + delta, lambda_max + 1e-9, delta)
        if len(lambdas) < 2:
            lam_fb = min(0.5, lambda_max)
            pi0_fb = float(np.minimum(np.mean(pvals > lam_fb) / (1.0 - lam_fb), 1.0))
            return pi0_fb, lam_fb
        pi0_grid = np.array(
            [np.sum(pvals > lam) / (m * (1.0 - lam)) for lam in lambdas],
            dtype=np.float64,
        )
        pi0_grid = np.minimum(pi0_grid, 1.0)
        if use_robust:
            V_grid = (pi0_grid * (1.0 - (1.0 - lambdas) * pi0_grid)
                      / (m * (1.0 - lambdas)))
            V_grid = np.maximum(V_grid, 0.0)
            psi_grid = pi0_grid + V_grid
        else:
            psi_grid = pi0_grid.copy()
        # τ* = inf{λ_{j+1} : ψ(λ_{j+1}) ≥ ψ(λ_j)}, truncated at lambda_max
        stop_idx = len(lambdas) - 1
        for j in range(len(lambdas) - 1):
            if psi_grid[j + 1] >= psi_grid[j]:
                stop_idx = j
                break
        return float(np.minimum(pi0_grid[stop_idx], 1.0)), float(lambdas[stop_idx])

    def storey_qvalue(p_vals, pi0_hat):
        m = len(p_vals)
        if m == 0: return np.array([])
        sort_idx = np.argsort(p_vals)
        sorted_p = p_vals[sort_idx]
        ranks = np.arange(1, m + 1, dtype=np.float64)
        fdp_hat = pi0_hat * m * sorted_p / ranks
        q_by_rank = np.minimum.accumulate(fdp_hat[::-1])[::-1]
        q_by_rank = np.minimum(q_by_rank, 1.0)
        q_values = np.ones(m)
        q_values[sort_idx] = q_by_rank
        return q_values

    def benjamini_hochberg(p_values, alpha, method='BH'):
        m = len(p_values)
        if m == 0: return np.array([], dtype=bool), np.array([]), 0
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        c_m = np.sum(1.0 / np.arange(1, m + 1)) if method == 'BY' else 1.0
        ranks = np.arange(1, m + 1)
        bh_critical = ranks * alpha / (m * c_m)
        k_star = 0
        for k in range(m, 0, -1):
            if sorted_p[k - 1] <= bh_critical[k - 1]:
                k_star = k; break
        rejected = np.zeros(m, dtype=bool)
        if k_star > 0: rejected[sorted_idx[:k_star]] = True
        bh_thresholds = np.zeros(m)
        for i, orig_idx in enumerate(sorted_idx):
            bh_thresholds[orig_idx] = bh_critical[i]
        return rejected, bh_thresholds, k_star

    def iut_conjunction_pvalue(p_struct_dom, p_disc):
        return np.maximum(np.asarray(p_struct_dom, dtype=np.float64),
                          np.asarray(p_disc, dtype=np.float64))

    def extract_subgroups_from_case_data(case_data):
        """Production subgroups: PartType × ReportType (matches p1_Production_parallel.py)."""
        case_to_subgroups = {}; subgroup_to_cases = {}
        for cid, case in case_data.items():
            sgs = []
            if case.part_desc not in ('UNKNOWN', '', 'nan', 'None'):
                sg = f"Part_{case.part_desc.replace(' ', '_')}"
                sgs.append(sg); subgroup_to_cases.setdefault(sg, set()).add(cid)
            if case.report_type not in ('UNKNOWN', '', 'nan', 'None'):
                sg = f"ReportType_{case.report_type}"
                sgs.append(sg); subgroup_to_cases.setdefault(sg, set()).add(cid)
            if sgs: case_to_subgroups[cid] = sgs
        return case_to_subgroups, subgroup_to_cases

    def determine_applicable_subgroups_with_cases(holds_by_case, case_to_subgroups):
        sg_map = {}
        for cid in holds_by_case:
            if cid in case_to_subgroups:
                for sg in case_to_subgroups[cid]:
                    sg_map.setdefault(sg, []).append(cid)
        return {sg: sorted(cids) for sg, cids in sg_map.items()}


# ============================================================================
# PUBLICATION-QUALITY PLOTTING CONFIGURATION
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

# Extended color palette for RQ3
RQ3_COLORS = {
    **COLORS,
    'P1': '#E69F00',     # Orange — activity label noise
    'P2': '#56B4E9',     # Sky blue — class label noise
    'P3': '#009E73',     # Bluish green — trace truncation
    'jaccard': '#332288',   # Indigo — Jaccard metric
    'fdr_emp': '#CC0000',   # Red — empirical FDR
    'vacuity': '#AA4499',   # Purple — vacuity rate
    'inflection': '#000000', # Black — inflection markers
}


# ============================================================================
# RQ3 EXPERIMENT CONFIGURATION
# ============================================================================

RQ3_CONFIG = {
    # Perturbation intensity levels (Sommers et al. 2025)
    'epsilon_levels': [0.02, 0.05, 0.10, 0.20, 0.30],

    # Number of independent replicates per (perturbation_type, epsilon)
    'n_replicates': 10,

    # Phase 1 pipeline parameters (must match clean run)
    'B_label': 2000,      # B₁: label permutation resamples
    'B_trace': 1000,      # B₂: trace permutation resamples
    'fdr_alpha': 0.05,    # Target FDR level α
    'fdr_method': 'BH',   # BH reference method
    # D9: Empirical calibration budget — matches P1 so S_pert and S_clean are defined
    # by the same decision rule (empirical Phipson-Smyth gate, not analytic χ²₄).
    # Resolution: 1/(B_null+1) ≈ 0.005 << α=0.05.
    'B_null':  200,   # double-null replicates for T_F empirical calibration
    'B1_null': 75,    # label perm budget per null replicate (reduced vs B_label)
    'B2_null': 75,    # trace perm budget per null replicate per class (reduced vs B_trace)
    'n_jobs':  -1,    # joblib parallelism for calibration (-1 = all cores)

    # Inflection threshold (Meinshausen & Bühlmann 2010 analogue)
    'jaccard_threshold': 0.80,

    # Bootstrap CIs
    'bootstrap_n': 1000,
    'ci_level': 0.95,

    # Random seed base
    'random_state': 42,

    # Constraint types (must match Phase 1)
    'constraint_types': [
        'Init', 'End',
        'Response', 'AlternateResponse', 'ChainResponse',
        'Succession', 'AlternateSuccession', 'ChainSuccession',
        'NotResponse', 'NotChainSuccession',
    ],
}


# ============================================================================
# BUDGET TIERS — ε-LEVEL-SPECIFIC PERMUTATION BUDGETS
# ============================================================================
#
# Scientific justification:
#   M1-M4 metrics are set-membership decisions (pattern ∈ S_pert or not).
#   At high ε (≥0.20), degradation is large and unambiguous; lower permutation
#   budgets are sufficient for reliable verdict assignment.
#   At low ε (≤0.05), subtle signal requires full precision.
#
#   Minimum resolution: 1/(B_null+1) must be << α=0.05.
#   At B_null=75: resolution = 1/76 ≈ 0.013 << 0.05 ✓
#   At B_null=200: resolution = 1/201 ≈ 0.005 << 0.05 ✓
#
# Paper disclosure (one sentence):
#   "Permutation budgets were tiered by ε level: full budgets (B₁=2000,
#    B₂=1000, B_null=200) at ε ≤ 0.05; reduced budgets (B₁=1000, B₂=500,
#    B_null=75) at ε=0.30 where degradation is unambiguous; intermediate
#    budgets at ε=0.10 and ε=0.20."

EPS_BUDGET_TIERS: Dict[float, Dict[str, int]] = {
    0.02: {'B_label': 2000, 'B_trace': 1000, 'B_null': 200, 'B1_null': 75, 'B2_null': 75},
    0.05: {'B_label': 2000, 'B_trace': 1000, 'B_null': 200, 'B1_null': 75, 'B2_null': 75},
    0.10: {'B_label': 2000, 'B_trace': 1000, 'B_null': 150, 'B1_null': 75, 'B2_null': 75},
    0.20: {'B_label': 1500, 'B_trace':  750, 'B_null': 100, 'B1_null': 50, 'B2_null': 50},
    0.30: {'B_label': 1000, 'B_trace':  500, 'B_null':  75, 'B1_null': 40, 'B2_null': 40},
}


# ============================================================================
# DECLARE FAMILY CLASSIFICATION
# ============================================================================

def classify_constraint_family(constraint_type: str) -> str:
    """
    Classify a constraint type into its DECLARE family for stratified analysis.

    Families:
        'Unary':           Init, End
        'Binary Positive': Response, AlternateResponse, ChainResponse,
                           Succession, AlternateSuccession, ChainSuccession
        'Binary Negative': NotResponse, NotChainSuccession
    """
    if constraint_type in UNARY_CONSTRAINTS:
        return 'Unary'
    elif constraint_type in BINARY_POSITIVE_CONSTRAINTS:
        return 'Binary Positive'
    elif constraint_type in BINARY_NEGATIVE_CONSTRAINTS:
        return 'Binary Negative'
    else:
        return 'Unknown'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_event_log(filepath: str) -> Dict[str, CaseInfo]:
    """
    Load Production manufacturing event log and extract case information.

    Column mapping:
        Case ID            → case identifier
        Complete Timestamp → event timestamp
        Activity           → activity name
        label              → pre-encoded outcome label ('deviant' / 'regular')
        Part_Desc_         → product type (subgroup dimension 1)
        Report_Type        → report type B/D/S (subgroup dimension 2)

    Outcome determination:
        Class 1 (Deviant — Re-submission Required) — label == 'deviant'
        Class 0 (Regular — No Re-submission)       — label == 'regular'

    No outcome-signal stripping required: the label is a case-level column
    attribute, not an activity in the trace. Retaining all activities is correct.
    """
    print(f"\n{'='*100}")
    print("📊 LOADING PRODUCTION EVENT LOG")
    print(f"{'='*100}")

    df = pd.read_csv(filepath, sep=';', low_memory=False)
    df = df.dropna(subset=['Case ID'])
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'], errors='coerce')
    print(f"   ✓ Loaded {len(df):,} events from {df['Case ID'].nunique():,} cases")

    case_data = {}

    for case_id, group in df.groupby('Case ID'):
        case_events = group.sort_values('Complete Timestamp')
        trace = case_events['Activity'].dropna().tolist()
        if len(trace) == 0:
            continue

        # Outcome: read from pre-encoded label column (no stripping needed)
        label_series = case_events['label'].dropna()
        raw_label = str(label_series.iloc[0]).strip() if len(label_series) > 0 else 'regular'
        outcome = 1 if raw_label == 'deviant' else 0

        activity_index = precompute_activity_index(trace, case_id=str(case_id))

        # Subgroup dimensions
        part_desc_raw = case_events['Part_Desc_'].iloc[0] if 'Part_Desc_' in case_events.columns else float('nan')
        part_desc = str(part_desc_raw).strip() if not pd.isna(part_desc_raw) else 'UNKNOWN'

        if 'Report_Type' in case_events.columns:
            rt_mode = case_events['Report_Type'].dropna().mode()
            report_type = str(rt_mode.iloc[0]).strip() if len(rt_mode) > 0 else 'UNKNOWN'
        else:
            report_type = 'UNKNOWN'

        ts_raw = case_events['Complete Timestamp'].iloc[0]
        start_ts = ts_raw if not pd.isna(ts_raw) else pd.Timestamp('1970-01-01')

        case_data[str(case_id)] = CaseInfo(
            case_id=str(case_id),
            outcome=outcome,
            trace=trace,
            part_desc=part_desc,
            report_type=report_type,
            start_timestamp=start_ts,
            activity_index=activity_index,
        )

    n1 = sum(1 for c in case_data.values() if c.outcome == 1)
    n0 = len(case_data) - n1
    print(f"   ✓ Processed {len(case_data):,} cases")
    print(f"   Class 1 (Deviant — Re-submission Required): {n1:,} ({n1/len(case_data)*100:.1f}%)")
    print(f"   Class 0 (Regular — No Re-submission):       {n0:,} ({n0/len(case_data)*100:.1f}%)")
    ir = max(n0, n1) / max(min(n0, n1), 1)
    print(f"   Imbalance ratio (maj/min):                   {ir:.3f}")

    return case_data


def load_candidates_from_spec(spec_file: str) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """Load candidate patterns from Phase 0 CC specification."""
    with open(spec_file, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    allowed = set(ALL_CONSTRAINT_TYPES)

    def _filter(clist):
        out = []
        for c in clist:
            if c['constraint_type'] in allowed:
                out.append((c['constraint_type'], c['param_a'], c.get('param_b')))
        return out

    lpos = _filter(spec.get('Lpos', {}).get('constraints', []))
    lneg = _filter(spec.get('Lneg', {}).get('constraints', []))

    pos_set = set(lpos)
    union = list(lpos) + [p for p in lneg if p not in pos_set]

    print(f"   ✓ L+ candidates: {len(lpos):,}, L− candidates: {len(lneg):,},"
          f"Union: {len(union):,}")
    return lpos, lneg, union

def load_clean_specification(results_file: str) -> Tuple[Set[str], Dict[str, str]]:
    """
    Load the significant specification S_clean from the clean Phase 1 run.

    Returns:
        S_clean              — set of pattern_id strings for all 'Both' or
                               'Discriminative only' patterns (is_significant_final=True).
        all_pattern_categories — dict mapping every pattern_id → significance_category.

    Schema-tolerant (D6 fix):
        Older Phase 1 JSON versions (pre-v8.0) may store significance_category:
          (a) nested under a 'storeyfdr' sub-dict, or
          (b) absent entirely, requiring derivation from boolean flags
              is_significant_structural + is_significant_discriminative.
        The assert fires only when the category genuinely cannot be resolved to
        a valid v8.0 value, i.e. the JSON is irrecoverably malformed.
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        clean_results = json.load(f)

    def _resolve_category(p: dict) -> str:
        """
        Robust significance_category extraction with three-level fallback:
          1. Top-level field (v8.0+ schema).
          2. Nested under 'storeyfdr' sub-dict (v7.x schema).
          3. Derived from is_significant_structural / is_significant_discriminative
             boolean flags (v6.x schema).
        """
        # Level 1: v8.0+ top-level field
        cat = p.get('significance_category', '')
        if cat:
            return cat

        # Level 2: legacy 'storey_fdr' nesting
        cat = p.get('storey_fdr', {}).get('significance_category', '')
        if cat:
            return cat

        # Level 3: derive from boolean flags
        is_struct = bool(p.get('is_significant_structural', False))
        is_disc   = bool(
            p.get('is_significant_discriminative',
                  p.get('is_significant_sam', False))
        )
        if is_struct and is_disc:
            return 'Both'
        if is_struct and not is_disc:
            return 'Structural only'
        if not is_struct and is_disc:
            return 'Discriminative only'
        return 'Neither'

    S_clean: Set[str] = set()
    for p in clean_results.get('significant_patterns', []):
        cat = _resolve_category(p)
        assert cat in ('Both', 'Discriminative only'), (
            f"S_clean contains pattern '{p['pattern_id']}' with category '{cat}' "
            f"— rerun P1 with single-gate (v8.0+). D6 guard."
        )
        S_clean.add(p['pattern_id'])

    all_pattern_categories: Dict[str, str] = {}
    for p in clean_results.get('all_patterns', []):
        all_pattern_categories[p['pattern_id']] = _resolve_category(p)

    print(f"   ✓ Clean specification |S_clean| = {len(S_clean):,}")
    print(f"   ✓ Total patterns in clean run:   {len(all_pattern_categories):,}")
    return S_clean, all_pattern_categories


# ============================================================================
# PERTURBATION GENERATORS (Sommers et al. 2025)
# ============================================================================

def apply_P1_activity_label_noise(
    case_data: Dict[str, CaseInfo],
    epsilon: float,
    random_state: int,
) -> Dict[str, CaseInfo]:
    """
    P1 — Activity label noise (operationalizes RIa_in).

    For each trace, independently with probability ε, swap two randomly
    selected distinct activity positions in the trace.

    Preserves: activity multiset per trace, class labels.
    Destroys: temporal ordering (partially pre-applies the structural null).
    Targets: H₀ˢ (structural axis).

    Scientific mechanism: the trace-activity permutation null exactly shuffles
    the same multiset — the perturbed log partially pre-applies the null
    permutation, compressing the signal-to-noise ratio for the structural test.

    Production note: the Production log has longer traces than clinical logs.
    Even at low ε, position swaps in multi-step quality workflows disrupt
    ChainSuccession and ChainResponse constraints that encode strict step ordering.
    The structural axis degrades proportionally; the discriminative axis is unaffected.
    """
    rng = np.random.RandomState(random_state)
    perturbed = {}
    n_swapped = 0

    for cid, case in case_data.items():
        trace = case.trace.copy()

        if len(trace) >= 2 and rng.random() < epsilon:
            # Swap two randomly selected distinct positions
            i, j = rng.choice(len(trace), size=2, replace=False)
            trace[i], trace[j] = trace[j], trace[i]
            n_swapped += 1

        activity_index = precompute_activity_index(trace, case_id=cid)
        perturbed[cid] = CaseInfo(
            case_id=cid, outcome=case.outcome, trace=trace,
            part_desc=case.part_desc, report_type=case.report_type,
            start_timestamp=case.start_timestamp,
            activity_index=activity_index,
        )

    return perturbed


def apply_P2_class_label_noise(
    case_data: Dict[str, CaseInfo],
    epsilon: float,
    random_state: int,
) -> Dict[str, CaseInfo]:
    """
    P2 — Class label noise (operationalizes RIc_in).

    Select a random fraction ε of cases and flip their outcome label (0↔1).

    Preserves: all trace structures.
    Destroys: class-outcome association.
    Targets: H₀ᵈ (discriminative axis).

    Scientific mechanism: the label permutation null redistributes labels
    across cases — the perturbed log injects synthetic null-like cases into
    the observed data. Δ_obs collapses toward zero as ε → 0.5.

    Production note: the Deviant class (Re-submission Required) is the minority.
    Label flipping at ε=0.10 dilutes Deviant signal disproportionately, collapsing
    Δ_obs faster than in balanced logs. The structural axis is fully unaffected
    since all trace structures are preserved.
    """
    rng = np.random.RandomState(random_state)
    case_ids = sorted(case_data.keys())
    n_to_flip = max(1, int(round(len(case_ids) * epsilon)))
    flip_ids = set(rng.choice(case_ids, size=n_to_flip, replace=False))

    perturbed = {}
    for cid, case in case_data.items():
        new_outcome = 1 - case.outcome if cid in flip_ids else case.outcome
        perturbed[cid] = CaseInfo(
            case_id=cid, outcome=new_outcome, trace=case.trace.copy(),
            part_desc=case.part_desc, report_type=case.report_type,
            start_timestamp=case.start_timestamp,
            activity_index=case.activity_index.copy(),
        )

    return perturbed


def apply_P3_trace_truncation(
    case_data: Dict[str, CaseInfo],
    epsilon: float,
    random_state: int,
) -> Dict[str, CaseInfo]:
    """
    P3 — Stochastic Trace Truncation (operationalizes RIe_mi).

    FIX v2.0 (scientific correction): random_state is now USED.
    Previous version was fully deterministic — all 10 replicates at the
    same ε produced identical perturbed logs, making replication scientifically void.

    NEW MECHANISM:
    Randomly select a fraction ε of cases (without replacement) whose traces
    are truncated to their first ⌈(1 − ε) · L⌉ events. The remaining (1−ε)
    fraction of cases are left intact. This operationalizes RIe_mi as a
    *random missing-data process at the case level* rather than a deterministic
    global shortening, which is both more realistic (incomplete ICU documentation
    affects specific patients) and scientifically correct (replicates now differ).

    Preserves: class labels, activity multisets of non-truncated traces.
    Destroys: tail events of selected cases — vacuity inflation for Response,
              Succession, ChainSuccession families.
    Targets: Both axes.

    Production note: truncation simulates incomplete step-completion recording
    in the manufacturing workflow (e.g., missing final quality sign-off events).
    The Succession family is most affected; Init/End constraints are immune.
    """
    rng = np.random.RandomState(random_state)
    perturbed = {}

    case_ids = sorted(case_data.keys())
    n_to_truncate = max(1, int(round(len(case_ids) * epsilon)))

    # Randomly select which cases are truncated — THIS is the stochasticity
    truncate_set = set(rng.choice(case_ids, size=n_to_truncate, replace=False))

    for cid, case in case_data.items():
        if cid in truncate_set:
            L = len(case.trace)
            retain_length = max(1, math.ceil((1.0 - epsilon) * L))
            new_trace = case.trace[:retain_length]
        else:
            new_trace = case.trace.copy()  # untruncated

        activity_index = precompute_activity_index(new_trace, case_id=cid)
        perturbed[cid] = CaseInfo(
            case_id=cid, outcome=case.outcome, trace=new_trace,
            part_desc=case.part_desc, report_type=case.report_type,
            start_timestamp=case.start_timestamp,
            activity_index=activity_index,
        )

    return perturbed


PERTURBATION_GENERATORS = {
    'P1': apply_P1_activity_label_noise,
    'P2': apply_P2_class_label_noise,
    'P3': apply_P3_trace_truncation,
}


# ============================================================================
# PHASE 1 PIPELINE EXECUTION ON PERTURBED LOG
# ============================================================================

def run_phase1_on_perturbed_log(
    perturbed_case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple[str, str, Optional[str]]],
    config: Dict,
    precomputed_holds: Optional[Dict] = None,  # NEW: skip holds computation for P2
) -> List[PatternTestResult]:
    """
    Execute the full Phase 1 three-hypothesis pipeline on a perturbed log.

    This is the core computation: for each perturbed log, we rerun the entire
    pipeline identically to the clean run, including:
      - Holds-by-case computation on perturbed traces
      - Label permutation test (H₀ᵈ) with B₁ resamples
      - Trace-activity permutation test (H₀ˢ) with B₂ resamples per class
      - IUT conjunction p-values
      - Dual-axis Storey FDR control (Steps 5b + 5c)

    CRITICAL: The B₁ null permutations computed on the perturbed log are exactly
    what is needed — since the Phipson-Smyth exact p-value formula is conditioned
    on the observed log, rerunning the pipeline on L_pert automatically generates
    the correct null distribution for that perturbed log.

    Args:
        precomputed_holds: If provided (P2 only), skip compute_holds_by_case_batch.
                           Holds are trace-only; since P2 does not alter traces,
                           the clean holds dict is mathematically identical to
                           what would be computed on the P2-perturbed log.
                           Passing precomputed_holds=None (default) preserves
                           the original behaviour for P1 and P3.

    Returns:
        List[PatternTestResult] — one per candidate pattern
    """
    # D7: Guard — fallback block does not define run_structural_permutation_test,
    # run_label_permutation_test, or fisher_conjunction_pvalue; crash here rather
    # than with a cryptic NameError deep inside the pipeline.
    if not PHASE1_IMPORTED:
        raise RuntimeError(
            "run_phase1_on_perturbed_log requires the Phase 1 module (p1_Production_parallel). "
            "The import failed — check the PHASE1_DIR path and fix the import error "
            "before running RQ3. (D7)"
        )

    B1 = config['B_label']
    B2 = config['B_trace']
    alpha = config['fdr_alpha']
    rs = config['random_state']
    B_null           = config.get('B_null',          200)
    B1_null          = config.get('B1_null',          75)
    B2_null          = config.get('B2_null',          75)
    n_jobs           = config.get('n_jobs',           -1)
    n_workers_struct = config.get('n_workers_struct',  1)  # NEW: for structural permutation

    # Sorted case list + label vector needed by compute_double_null_tf_matrix
    case_ids_sorted = sorted(perturbed_case_data.keys())
    labels = np.array([perturbed_case_data[cid].outcome for cid in case_ids_sorted])

    D_0, D_1 = split_by_class(perturbed_case_data)
    m_total = len(candidates_all)

    # Step 2: Holds-by-case on perturbed log
    # P2 optimisation: P2 only modifies outcome labels, never traces.
    # holds_by_case is a pure function of (trace, candidate) — mathematically
    # identical between clean and P2-perturbed logs for all ε.
    # Passing precomputed_holds bypasses this O(m·n) step for P2 workers.
    if precomputed_holds is not None:
        holds_all = precomputed_holds
    else:
        holds_all = compute_holds_by_case_batch(perturbed_case_data, candidates_all)

    # Step 3: Label permutation (H₀ᵈ)
    disc_results = run_label_permutation_test(
        perturbed_case_data, candidates_all, holds_all, B1, rs
    )
    null_delta_matrix = disc_results.pop('__null_delta_matrix__')

    # Step 4: Trace-activity permutation (H₀ˢ) — both classes on full union
    #
    # v2.0: D_0 and D_1 are disjoint sets with independent RNGs.
    # When running in the outer serial mode (n_jobs_outer=1), they CAN be
    # executed concurrently. When inside a loky worker (n_jobs_outer>1),
    # we use sequential execution + n_workers_struct-chunked B₂ parallelism.
    #
    # GUARD: do NOT use Parallel(n_jobs=2) here if we are already inside
    # a loky worker — triple nesting would occur. Detect via env flag.
    _in_outer_worker = os.environ.get('RQ3_IN_WORKER') == '1'

    if (not _in_outer_worker) and n_workers_struct <= 1:
        # Serial outer + serial struct: run D0/D1 concurrently (2 threads safe)
        struct_results_0, struct_results_1 = Parallel(n_jobs=2, backend='loky')([
            delayed(run_structural_permutation_test)(
                D_0, candidates_all, class_label=0, B2=B2,
                random_state=rs, n_workers=1,
            ),
            delayed(run_structural_permutation_test)(
                D_1, candidates_all, class_label=1, B2=B2,
                random_state=rs + 1, n_workers=1,
            ),
        ])
    elif _in_outer_worker:
        # Inside loky worker: use n_workers_struct for B₂ chunking, sequential D0/D1
        struct_results_0 = run_structural_permutation_test(
            D_0, candidates_all, class_label=0, B2=B2,
            random_state=rs, n_workers=n_workers_struct,
        )
        struct_results_1 = run_structural_permutation_test(
            D_1, candidates_all, class_label=1, B2=B2,
            random_state=rs + 1, n_workers=n_workers_struct,
        )
    else:
        # Serial outer + chunked struct: both sequential
        struct_results_0 = run_structural_permutation_test(
            D_0, candidates_all, class_label=0, B2=B2,
            random_state=rs, n_workers=n_workers_struct,
        )
        struct_results_1 = run_structural_permutation_test(
            D_1, candidates_all, class_label=1, B2=B2,
            random_state=rs + 1, n_workers=n_workers_struct,
        )

    # Subgroup extraction
    case_to_sg, sg_to_cases = extract_subgroups_from_case_data(perturbed_case_data)

    # Step 5a: Assemble per-pattern results
    cid_set_0 = set(D_0.keys())
    cid_set_1 = set(D_1.keys())
    pattern_results = []
    tf_obs_all = np.zeros(m_total)   # raw T_F scores; needed for empirical calibration (D9)

    for p_idx, pspec in enumerate(candidates_all):
        ct, a, b = pspec
        pid = f"{ct}_{a}" + (f"_{b}" if b else "")
        holds = holds_all[pspec]

        prev0, nsat0, napp0 = compute_prevalence_from_holds(holds, cid_set_0)
        prev1, nsat1, napp1 = compute_prevalence_from_holds(holds, cid_set_1)
        delta = prev1 - prev0
        dominant = 1 if prev1 >= prev0 else 0
        direction = "Positive" if dominant == 1 else "Negative"

        d = disc_results[pspec]
        p_disc_two = d['p_two_sided']
        p_disc_one = d['p_one_sided']

        s0 = struct_results_0.get(pspec, {})
        p_s0 = s0.get('p_structural_test', 1.0)    # TEST half enters Fisher (D1 fix)
        p_s0_screen = s0.get('p_structural_screen', 1.0)  # SCREEN half for scope filter (D3 fix)
        null_mean_0 = s0.get('null_mean', prev0)
        null_std_0 = s0.get('null_std', 0.0)

        s1 = struct_results_1.get(pspec, {})
        p_s1 = s1.get('p_structural_test', 1.0)    # TEST half enters Fisher (D1 fix)
        p_s1_screen = s1.get('p_structural_screen', 1.0)  # SCREEN half for scope filter (D3 fix)
        null_mean_1 = s1.get('null_mean', prev1)
        null_std_1 = s1.get('null_std', 0.0)

        p_struct_dom = p_s1 if dominant == 1 else p_s0

        # Raw T_F score — stored for empirical calibration (D9); mirrors P1 exactly
        _eps = 1e-300
        tf_obs_i = -2.0 * (np.log(max(float(p_struct_dom), _eps))
                           + np.log(max(float(p_disc_two),  _eps)))
        tf_obs_all[p_idx] = tf_obs_i

        p_conj = float(fisher_conjunction_pvalue(
            np.array([p_struct_dom]), np.array([p_disc_two])
        )[0])

        sg_cases = determine_applicable_subgroups_with_cases(holds, case_to_sg)

        pattern_results.append(PatternTestResult(
            pattern_id=pid, constraint_type=ct,
            activity_a=a, activity_b=b,
            prevalence_class0=prev0, prevalence_class1=prev1,
            n_applicable_class0=napp0, n_applicable_class1=napp1,
            n_satisfied_class0=nsat0, n_satisfied_class1=nsat1,
            delta_obs=delta,
            p_structural_class0=p_s0, p_structural_class1=p_s1,
            p_structural_screen_class0=p_s0_screen,  # SCREEN half (scope filter only)
            p_structural_screen_class1=p_s1_screen,  # SCREEN half (scope filter only)
            p_structural_dominant=p_struct_dom,
            null_mean_class0=null_mean_0, null_mean_class1=null_mean_1,
            null_std_class0=null_std_0, null_std_class1=null_std_1,
            p_discriminative=p_disc_two,
            p_discriminative_onesided=p_disc_one,
            null_delta_mean=d['null_delta_mean'],
            null_delta_std=d['null_delta_std'],
            p_conjunction=p_conj,
            is_significant_bh=False, bh_rank=None, bh_threshold=None,
            dominant_class=dominant, direction=direction,
            applicable_subgroups=sorted(sg_cases.keys()),
            subgroup_to_cases=sg_cases,
        ))

    # Step 5b_pre: Empirical calibration of T_F (D9 fix — mirrors P1 exactly)
    # Replaces analytic χ²₄ p-values with Phipson-Smyth empirical p-values so that
    # S_pert and S_clean are defined by the same decision rule.
    tf_null_matrix = compute_double_null_tf_matrix(
        case_data       = perturbed_case_data,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        labels          = labels,
        B_null          = B_null,
        B1_null         = B1_null,
        B2_null         = B2_null,
        alpha           = alpha,
        random_state    = rs + 300_000,   # offset from all production seeds (mirrors P1)
        n_jobs          = n_jobs,
    )
    p_empirical_all = empirical_fisher_pvalue(tf_obs_all, tf_null_matrix)
    # p̃_F(i) = (1 + #{b: T_F^(b)(i) ≥ T_F_obs(i)}) / (B_null + 1)
    for pidx, r in enumerate(pattern_results):
        r.p_conjunction_empirical = float(p_empirical_all[pidx])

    # Step 5b: Storey Q-Value FDR on empirical Fisher p-values (discriminative axis)
    p_conj_values = np.array([r.p_conjunction for r in pattern_results])  # analytic; BH reference only
    structural_idx = [
        i for i, r in enumerate(pattern_results)
        if min(r.p_structural_screen_class0, r.p_structural_screen_class1) <= alpha  # SCREEN half (D3 fix — avoids selection bias)
    ]
    m_prime = len(structural_idx)

    if m_prime > 0:
        p_conj_filtered = p_empirical_all[structural_idx]  # empirical gate (D9 fix — was analytic p_conjunction)
        pi0_f, _ = adaptive_storey_pi0(p_conj_filtered, q=alpha)  # Gao (2023) AS — D5 fix
        q_values_f = storey_qvalue(p_conj_filtered, pi0_f)

        for sam_i, orig_i in enumerate(structural_idx):
            r = pattern_results[orig_i]
            r.q_value_sam = float(q_values_f[sam_i])
            r.is_significant_sam = bool(q_values_f[sam_i] <= alpha)
            r.is_significant_discriminative = r.is_significant_sam

    # Step 5c: Storey FDR on structural p-values (per class)
    p_struct_c0 = np.array([r.p_structural_class0 for r in pattern_results])
    p_struct_c1 = np.array([r.p_structural_class1 for r in pattern_results])

    if len(p_struct_c0) > 0:
        pi0_s0, _ = adaptive_storey_pi0(p_struct_c0, q=alpha)  # Gao (2023) AS — D5 fix
        pi0_s1, _ = adaptive_storey_pi0(p_struct_c1, q=alpha)  # Gao (2023) AS — D5 fix
        q_sc0 = storey_qvalue(p_struct_c0, pi0_s0)
        q_sc1 = storey_qvalue(p_struct_c1, pi0_s1)

        for i, r in enumerate(pattern_results):
            r.q_structural_class0 = float(q_sc0[i])
            r.q_structural_class1 = float(q_sc1[i])
            q_dom = float(q_sc1[i]) if r.dominant_class == 1 else float(q_sc0[i])
            r.q_structural_dominant = q_dom
            r.is_significant_structural = r.p_structural_dominant <= alpha  # raw nominal label — taxonomy only (mirrors Phase 1)

            if r.is_significant_structural and r.is_significant_discriminative:
                r.significance_category = "Both"
            elif r.is_significant_structural and not r.is_significant_discriminative:
                r.significance_category = "Structural only"
            elif not r.is_significant_structural and r.is_significant_discriminative:
                r.significance_category = "Discriminative only"
            else:
                r.significance_category = "Neither"
            r.is_significant_final = r.is_significant_discriminative  # Fisher sole gate (mirrors P1 — D2 fix)

    # BH reference
    rejected, bh_thresh, _ = benjamini_hochberg(p_conj_values, alpha, method='BH')
    for i, r in enumerate(pattern_results):
        r.is_significant_bh = bool(rejected[i])
        r.bh_threshold = float(bh_thresh[i])

    return pattern_results


# ============================================================================
# METRIC COMPUTATION (M1–M4)
# ============================================================================

@dataclass
class ReplicateMetrics:
    """Metrics for a single (perturbation_type, epsilon, replicate) run."""
    perturbation_type: str
    epsilon: float
    replicate: int
    random_seed: int

    # M1 — Jaccard specification stability
    jaccard: float

    # M2 — Empirical FDR under perturbation
    empirical_fdr: float

    # M3 — Vacuity rate
    vacuity_rate: float

    # M4 — Four-category verdict counts
    n_both: int
    n_structural_only: int
    n_discriminative_only: int
    n_neither: int

    # Additional diagnostics
    n_S_pert: int          # |S_pert|
    n_intersection: int    # |S_clean ∩ S_pert|
    n_union: int           # |S_clean ∪ S_pert|
    n_false_discoveries: int  # |S_pert \ S_clean|

    # Per-family breakdown (Dict[family → {n_both, n_struct, n_disc, n_neither}])
    family_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Per-family Jaccard
    family_jaccard: Dict[str, float] = field(default_factory=dict)


def compute_replicate_metrics(
    perturbed_results: List[PatternTestResult],
    S_clean: Set[str],
    perturbation_type: str,
    epsilon: float,
    replicate: int,
    random_seed: int,
    n_total: int,
) -> ReplicateMetrics:
    """
    Compute M1–M4 metrics for a single replicate run.

    M1 — Jaccard specification stability:
        J(S_clean, S_pert) = |S_clean ∩ S_pert| / |S_clean ∪ S_pert|

    M2 — Empirical FDR under perturbation:
        FDR_pert = |S_pert \ S_clean| / max(|S_pert|, 1)
        S_clean serves as oracle — under α control, all surviving clean
        constraints are genuine by construction from RQ1.

    M3 — Vacuity rate:
        Fraction of significant patterns that are vacuously satisfied
        (n_applicable = 0 in at least one class).

    M4 — Four-category verdict breakdown:
        Track Both / Structural-only / Discriminative-only / Neither counts.
    """
    # Build perturbed specification set
    S_pert = set()
    for r in perturbed_results:
        if r.is_significant_final:
            S_pert.add(r.pattern_id)

    # M1: Jaccard
    intersection = S_clean & S_pert
    union = S_clean | S_pert
    jaccard = len(intersection) / len(union) if len(union) > 0 else 1.0

    # M2: Empirical FDR
    false_discoveries = S_pert - S_clean
    empirical_fdr = len(false_discoveries) / max(len(S_pert), 1)

    # M3: Vacuity rate (among significant patterns)
    vacuity_thresh = 0.05 * n_total
    n_vacuous = 0
    n_sig = 0
    for r in perturbed_results:
        if r.is_significant_final:
            n_sig += 1
            if r.n_applicable_class0 < vacuity_thresh or r.n_applicable_class1 < vacuity_thresh:
                n_vacuous += 1
    vacuity_rate = n_vacuous / max(n_sig, 1)

    # M4: Four-category counts
    n_both = sum(1 for r in perturbed_results if r.significance_category == "Both")
    n_struct = sum(1 for r in perturbed_results if r.significance_category == "Structural only")
    n_disc = sum(1 for r in perturbed_results if r.significance_category == "Discriminative only")
    n_neither = sum(1 for r in perturbed_results if r.significance_category == "Neither")

    # Per-family breakdown
    family_breakdown = {}
    family_sig_clean = defaultdict(set)   # family → set of clean significant PIDs
    family_sig_pert = defaultdict(set)    # family → set of pert significant PIDs

    for r in perturbed_results:
        fam = classify_constraint_family(r.constraint_type)
        if fam not in family_breakdown:
            family_breakdown[fam] = {'Both': 0, 'Structural only': 0,
                                     'Discriminative only': 0, 'Neither': 0}
        family_breakdown[fam][r.significance_category] += 1

        if r.is_significant_final:
            family_sig_pert[fam].add(r.pattern_id)

    # Clean family sets (from S_clean + constraint type info)
    for r in perturbed_results:
        fam = classify_constraint_family(r.constraint_type)
        if r.pattern_id in S_clean:
            family_sig_clean[fam].add(r.pattern_id)

    # Per-family Jaccard
    family_jaccard = {}
    for fam in set(list(family_sig_clean.keys()) + list(family_sig_pert.keys())):
        sc = family_sig_clean.get(fam, set())
        sp = family_sig_pert.get(fam, set())
        union_f = sc | sp
        family_jaccard[fam] = len(sc & sp) / len(union_f) if len(union_f) > 0 else 1.0

    return ReplicateMetrics(
        perturbation_type=perturbation_type,
        epsilon=epsilon,
        replicate=replicate,
        random_seed=random_seed,
        jaccard=jaccard,
        empirical_fdr=empirical_fdr,
        vacuity_rate=vacuity_rate,
        n_both=n_both,
        n_structural_only=n_struct,
        n_discriminative_only=n_disc,
        n_neither=n_neither,
        n_S_pert=len(S_pert),
        n_intersection=len(intersection),
        n_union=len(union),
        n_false_discoveries=len(false_discoveries),
        family_breakdown=family_breakdown,
        family_jaccard=family_jaccard,
    )


# ============================================================================
# CHECKPOINT HELPERS (fault-tolerant resume)
# ============================================================================

def _metrics_to_dict(m: ReplicateMetrics) -> Dict:
    """Serialize ReplicateMetrics to a JSON-safe dict."""
    return asdict(m)


def _dict_to_metrics(d: Dict) -> ReplicateMetrics:
    """
    Deserialize ReplicateMetrics from a JSON dict.

    Schema-tolerant: extra keys from an older checkpoint version are silently
    dropped; missing keys that have dataclass defaults are handled by Python.
    This prevents TypeError on checkpoint files written by a prior version of
    ReplicateMetrics with a different field set.
    """
    valid_fields = {f.name for f in fields(ReplicateMetrics)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return ReplicateMetrics(**filtered)


def _checkpoint_path(ptype: str, eps: float, rep: int) -> str:
    """
    Canonical path for a single replicate's checkpoint JSON.
    Key: ptype_epsXXXX_repYY.json  (deterministic, collision-free).
    """
    eps_str = f"{eps:.4f}".replace('.', 'p')
    fname = f"{ptype}_eps{eps_str}_rep{rep:02d}.json"
    return os.path.join(REPLICATES_DIR, fname)


def _rq3_worker_with_checkpoint(
    ptype: str,
    eps: float,
    rep: int,
    case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    S_clean: Set[str],
    pipeline_config: Dict,
    base_seed: int,
    clean_holds: Optional[Dict] = None,
) -> ReplicateMetrics:
    """
    Fault-tolerant wrapper around _rq3_worker.

    - Checks for a cached result at REPLICATES_DIR/<key>.json before running.
    - If cache hit: deserializes and returns immediately (idempotent).
    - If cache miss: runs the full worker, saves result, then returns.

    This makes the entire experiment resumable: re-running the script after
    a HPC job failure will skip all already-completed replicates.
    Seeds are deterministic — a resumed replicate produces the identical result.

    CRITICAL INVARIANT: the checkpoint is written AFTER the full worker
    completes successfully. A partial write (crash mid-worker) leaves no
    checkpoint file, forcing a clean re-run of that task.
    """
    ckpt = _checkpoint_path(ptype, eps, rep)

    # ── Cache hit ────────────────────────────────────────────────────────────
    if os.path.exists(ckpt):
        try:
            with open(ckpt, 'r', encoding='utf-8') as f:
                d = json.load(f)
            metrics = _dict_to_metrics(d)
            # Verify key fields match (guard against stale/corrupt cache)
            assert metrics.perturbation_type == ptype
            assert abs(metrics.epsilon - eps) < 1e-9
            assert metrics.replicate == rep
            return metrics
        except Exception as e:
            # Corrupt checkpoint — delete and recompute
            warnings.warn(f"Corrupt checkpoint {ckpt}: {e}. Recomputing.", RuntimeWarning)
            os.remove(ckpt)

    # ── Cache miss: run the full computation ─────────────────────────────────
    metrics = _rq3_worker(
        ptype, eps, rep,
        case_data, candidates_all, S_clean,
        pipeline_config, base_seed,
        clean_holds=clean_holds,
    )

    # ── Atomic write: write to .tmp first, then rename ───────────────────────
    # This prevents partial writes from creating a corrupt checkpoint.
    tmp_path = ckpt + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(_metrics_to_dict(metrics), f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, ckpt)   # atomic on POSIX (SLURM NFS safe)
    except Exception as e:
        warnings.warn(f"Failed to write checkpoint {ckpt}: {e}", RuntimeWarning)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return metrics


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Returns: (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha_half = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, 100 * alpha_half))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha_half)))
    return float(np.mean(values)), ci_lower, ci_upper


# ============================================================================
# INFLECTION POINT ESTIMATION
# ============================================================================

def estimate_inflection_point(
    epsilon_levels: List[float],
    metric_values_by_epsilon: Dict[float, np.ndarray],
    threshold: float = 0.80,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Optional[float]:
    """
    Estimate the inflection point: smallest ε where the 95% bootstrap CI
    of the Jaccard metric drops below the threshold (0.80).

    Analogous to the 80%-overlap criterion in stability analysis of
    statistical learning procedures (Meinshausen & Bühlmann 2010).
    """
    for eps in sorted(epsilon_levels):
        vals = metric_values_by_epsilon.get(eps)
        if vals is None or len(vals) == 0:
            continue
        _, ci_lower, _ = bootstrap_ci(vals, n_bootstrap, ci_level, random_state)
        if ci_lower < threshold:
            return eps
    return None  # No inflection detected within tested range


# ============================================================================
# TWO-PHASE ADAPTIVE DESIGN HELPERS
# ============================================================================

def identify_augmentation_targets(
    pilot_metrics: List[ReplicateMetrics],
    config: Dict,
    additional_reps: int = 7,
) -> List[Tuple[str, float, List[int]]]:
    """
    After Phase A (R=3 pilot), determine which (ptype, ε) cells need
    more replicates and which replicate indices to assign.

    Decision rule per (ptype, ε):
      DEFINITE ABOVE  — CI lower > threshold + 0.10:  skip (stable, no inflection risk)
      DEFINITE BELOW  — CI upper < threshold - 0.10:  skip (already degraded past threshold)
      BORDERLINE      — otherwise:                    add `additional_reps` replicates

    Returns:
        List of (ptype, eps, [rep_indices]) for Phase B augmentation.
        rep_indices start at n_pilot_reps to avoid seed collision with Phase A.
    """
    n_boot = config.get('bootstrap_n', 1000)
    ci_level = config.get('ci_level', 0.95)
    rs = config.get('random_state', 42)
    threshold = config.get('jaccard_threshold', 0.80)
    epsilon_levels = config['epsilon_levels']
    ptypes = ['P1', 'P2', 'P3']

    # Group pilot metrics by (ptype, eps); n_pilot = len(metrics) is used directly.
    grouped: Dict[Tuple, List] = defaultdict(list)
    for m in pilot_metrics:
        grouped[(m.perturbation_type, m.epsilon)].append(m)

    augment_tasks = []

    for ptype in ptypes:
        for eps in epsilon_levels:
            metrics = grouped.get((ptype, eps), [])
            if not metrics:
                # No pilot data — run full additional_reps starting at 0
                augment_tasks.append((ptype, eps, list(range(0, additional_reps))))
                continue

            j_vals = np.array([m.jaccard for m in metrics])
            _, ci_lo, ci_hi = bootstrap_ci(j_vals, n_boot, ci_level, rs)
            n_pilot = len(metrics)

            definite_above = ci_lo > threshold + 0.10
            definite_below = ci_hi < threshold - 0.10

            if definite_above or definite_below:
                # CI is unambiguous — no extra replicates needed
                print(f"   [SKIP] {ptype} ε={eps:.2f}: CI=[{ci_lo:.3f},{ci_hi:.3f}] "
                      f"({'above' if definite_above else 'below'} threshold ± 0.10)")
            else:
                # Borderline — needs augmentation
                new_rep_indices = list(range(n_pilot, n_pilot + additional_reps))
                augment_tasks.append((ptype, eps, new_rep_indices))
                print(f"   [AUGMENT] {ptype} ε={eps:.2f}: CI=[{ci_lo:.3f},{ci_hi:.3f}] "
                      f"→ adding reps {new_rep_indices}")

    return augment_tasks


# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def _rq3_worker(
    ptype: str,
    eps: float,
    rep: int,
    case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    S_clean: Set[str],
    pipeline_config: Dict,
    base_seed: int,
    clean_holds: Optional[Dict] = None,  # NEW: pre-computed holds for P2
) -> ReplicateMetrics:
    """Joblib worker for a single (ptype, ε, replicate) task."""
    # Fix 1 (critical): collision-free seed — PTYPE_BASES >> max(ε×10⁴ + rep×7919)
    # P1 block [100_042, 174_313], P2 [200_042, 274_313], P3 [300_042, 374_313] — no overlap.
    PTYPE_BASES = {'P1': 100_000, 'P2': 200_000, 'P3': 300_000}
    seed = base_seed + PTYPE_BASES[ptype] + int(round(eps * 10_000)) + rep * 7_919

    # Fix 2 (moderate): propagate per-task seed into all permutation stages so each
    # replicate draws an independent null distribution (not CRN shared across 150 workers).
    # Apply ε-level budget tier — reduces compute for high-ε cells where
    # degradation is unambiguous and precision on p-values is not required
    tier = EPS_BUDGET_TIERS.get(eps, {})
    local_config = {
        **pipeline_config,
        'random_state': seed,
        # Override budgets with tier values if defined
        **{k: tier[k] for k in ('B_label', 'B_trace', 'B_null', 'B1_null', 'B2_null')
           if k in tier},
    }

    # Fix 3 (v2.0): silence tqdm inside loky workers to avoid garbled SLURM logs.
    # Use try/finally to restore the previous env-var value so that serial-mode
    # execution (n_jobs_outer=1, dry-run, adaptive Phase A) does not poison the
    # main-process environment for all subsequent replicates.
    _prev_worker_flag = os.environ.get('RQ3_IN_WORKER', '')
    os.environ['RQ3_IN_WORKER'] = '1'
    try:
        generator = PERTURBATION_GENERATORS[ptype]
        perturbed_data = generator(case_data, eps, seed)

        # P2 invariant guard: traces must be unchanged for clean_holds reuse to be valid.
        # Use sorted keys (deterministic) and check min(50, n) cases — sufficient to
        # catch any systematic trace mutation introduced by a future P2 change.
        if ptype == 'P2' and clean_holds is not None:
            _check_ids = sorted(case_data.keys())[:min(50, len(case_data))]
            assert all(
                perturbed_data[cid].trace == case_data[cid].trace
                for cid in _check_ids
            ), "P2 must not modify traces — clean_holds reuse is invalid."

        # P2: traces unchanged — reuse clean holds to skip O(m·n) recomputation
        holds_to_use = clean_holds if (ptype == 'P2' and clean_holds is not None) else None

        pert_results = run_phase1_on_perturbed_log(
            perturbed_data, candidates_all, local_config,
            precomputed_holds=holds_to_use,
        )
        return compute_replicate_metrics(
            pert_results, S_clean, ptype, eps, rep, seed, len(perturbed_data)
        )
    finally:
        # Restore: prevents main-process env poisoning in serial mode
        if _prev_worker_flag:
            os.environ['RQ3_IN_WORKER'] = _prev_worker_flag
        else:
            os.environ.pop('RQ3_IN_WORKER', None)


def run_rq3_experiment(
    case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    S_clean: Set[str],
    config: Dict = None,
    n_jobs_outer: int = 1,
) -> List[ReplicateMetrics]:
    """
    Run the complete RQ3 Direction 2 sensitivity analysis.

    For each (perturbation_type, ε, replicate):
      1. Generate perturbed log
      2. Execute full Phase 1 pipeline on perturbed log
      3. Compute M1–M4 metrics against S_clean
      4. Store results

    Total runs: 3 perturbation types × 5 ε levels × R replicates = 150.
    """
    if config is None:
        config = RQ3_CONFIG

    # Fix 4 (v2.0): Replace blunt n_jobs=1 guard with per-tier budget arithmetic.
    # Previous version forced n_jobs=1 inside ALL workers whenever n_jobs_outer>1,
    # serialising the B_null=200 double-null calibration completely.
    # New version splits available cores proportionally between:
    #   - n_workers_struct: structural permutation (B₂ chunking)
    #   - n_jobs_calib:     double-null calibration (compute_double_null_tf_matrix)
    # INVARIANT: inside _run_one_replicate (inside compute_double_null_tf_matrix),
    #            run_structural_permutation_test is always called with n_workers=1
    #            to avoid triple nesting. This is enforced by P1 code (unchanged).
    import joblib as _jl
    C = _jl.cpu_count() or 1
    if n_jobs_outer > 1:
        # Multiple outer workers: divide cores evenly
        cores_per_task = max(1, C // n_jobs_outer)
    else:
        # Single outer worker: use all cores inside
        cores_per_task = C

    n_workers_struct = max(1, cores_per_task // 2)
    n_jobs_calib     = max(1, cores_per_task - n_workers_struct)

    print(f"   Core budget: C={C}, n_jobs_outer={n_jobs_outer}, "
          f"cores_per_task={cores_per_task}, "
          f"n_workers_struct={n_workers_struct}, n_jobs_calib={n_jobs_calib}")

    config = {
        **config,
        'n_jobs':           n_jobs_calib,     # → compute_double_null_tf_matrix
        'n_workers_struct': n_workers_struct,  # → run_structural_permutation_test
    }

    epsilon_levels = config['epsilon_levels']
    n_replicates = config['n_replicates']
    base_seed = config['random_state']

    perturbation_types = ['P1', 'P2', 'P3']
    total_runs = len(perturbation_types) * len(epsilon_levels) * n_replicates

    print(f"\n{'='*100}")
    print("🔬 RQ3 DIRECTION 2 — SENSITIVITY ANALYSIS UNDER CONTROLLED LOG PERTURBATIONS")
    print(f"{'='*100}")
    print(f"   Perturbation types: {perturbation_types}")
    print(f"   Intensity levels ε: {epsilon_levels}")
    print(f"   Replicates R:       {n_replicates}")
    print(f"   Total pipeline runs: {total_runs}")
    print(f"   B₁ (label perm):   {config['B_label']:,}")
    print(f"   B₂ (trace perm):   {config['B_trace']:,}")
    print(f"   FDR α:             {config['fdr_alpha']}")
    print(f"   |S_clean|:         {len(S_clean):,}")
    print(f"   Candidates:        {len(candidates_all):,}")
    print(f"{'='*100}")

    t_experiment_start = time.time()

    pipeline_config = {
        'B_label':           config['B_label'],
        'B_trace':           config['B_trace'],
        'fdr_alpha':         config['fdr_alpha'],
        'fdr_method':        config.get('fdr_method', 'BH'),
        'random_state':      base_seed,
        'B_null':            config.get('B_null',  200),
        'B1_null':           config.get('B1_null', 75),
        'B2_null':           config.get('B2_null', 75),
        'n_jobs':            config.get('n_jobs',  n_jobs_calib),
        'n_workers_struct':  config.get('n_workers_struct', n_workers_struct),
    }

    # Pre-compute holds on the clean log ONCE.
    # Reused by all P2 workers (P2 does not modify traces).
    # P1 and P3 workers receive clean_holds=None and compute their own.
    # If called from run_rq3_adaptive, holds may already be pre-computed
    # (stored as config['_clean_holds']) — reuse them to avoid O(m·n) repetition.
    _injected_holds = config.get('_clean_holds', None)
    if _injected_holds is not None:
        clean_holds = _injected_holds
        print("   ✓ Clean holds reused from caller (adaptive phase, no recomputation)")
    else:
        print("   Pre-computing clean holds (shared by all P2 workers)...")
        t_holds = time.time()
        clean_holds = compute_holds_by_case_batch(case_data, candidates_all)
        n_p2_workers = len([t for t in perturbation_types if t == 'P2']) * len(epsilon_levels) * n_replicates
        print(f"   ✓ Clean holds computed in {time.time()-t_holds:.1f}s "
              f"— reused by {n_p2_workers} P2 workers")

    tasks = [
        (ptype, eps, rep)
        for ptype in perturbation_types
        for eps in epsilon_levels
        for rep in range(n_replicates)
    ]

    all_metrics: List[ReplicateMetrics] = Parallel(
        n_jobs=n_jobs_outer, verbose=10, backend='loky',
    )(
        delayed(_rq3_worker_with_checkpoint)(
            ptype, eps, rep,
            case_data, candidates_all, S_clean,
            pipeline_config, base_seed,
            clean_holds=(clean_holds if ptype == 'P2' else None),  # P2 gets reuse
        )
        for ptype, eps, rep in tasks
    )

    total_time = time.time() - t_experiment_start
    print(f"\n{'='*100}")
    print(f"✅ RQ3 EXPERIMENT COMPLETE — {total_runs} runs in {total_time:.0f}s "
          f"({total_time/60:.1f} min)")
    print(f"{'='*100}")

    return all_metrics


def run_rq3_adaptive(
    case_data: Dict[str, CaseInfo],
    candidates_all: List[Tuple],
    S_clean: Set[str],
    config: Dict = None,
    n_jobs_outer: int = 1,
    n_pilot_reps: int = 3,
    n_augment_reps: int = 7,
    use_adaptive: bool = True,
) -> List[ReplicateMetrics]:
    """
    Two-Phase Adaptive Sensitivity Experiment.

    Phase A — Pilot sweep (n_pilot_reps replicates per cell):
        Run all 15 (ptype, ε) cells with R=n_pilot_reps.
        Total: 3 × 5 × n_pilot_reps = 45 runs (default).
        Compute bootstrap CIs → identify ambiguous cells near ε*.

    Phase B — Targeted augmentation (n_augment_reps per borderline cell):
        Only cells where CI straddles the 0.80 threshold get extra replicates.
        This focuses compute where the inflection point matters most.

    Scientific validity:
        This is sequential adaptive experimental design (see Wald 1947, SPRT;
        and adaptive stopping rules in sensitivity analysis — Saltelli et al. 2008).
        The two-phase design is fully disclosed in the methods section.
        The combined Phase A + Phase B results are pooled identically to a
        fixed-design experiment: each replicate is independent, seeds are
        collision-free, the decision rule (CI straddling threshold) does not
        condition on the test statistic.

    Paper disclosure:
        "We employed a two-phase adaptive design: a pilot sweep of R=3 replicates
         per (perturbation type, ε) cell, followed by targeted augmentation of R=7
         additional replicates for cells whose 95% bootstrap CI of J(S_clean, S_pert)
         straddled the 0.80 threshold. The final analysis pools all replicates."
    """
    if config is None:
        config = RQ3_CONFIG

    if not use_adaptive:
        # Fall through to full fixed design
        return run_rq3_experiment(case_data, candidates_all, S_clean,
                                  config, n_jobs_outer)

    print(f"\n{'='*100}")
    print("🔬 RQ3 ADAPTIVE TWO-PHASE DESIGN")
    print(f"{'='*100}")
    print(f"   Phase A: R={n_pilot_reps} pilot replicates × 15 cells = "
          f"{3 * len(config['epsilon_levels']) * n_pilot_reps} runs")
    print(f"   Phase B: Up to R={n_augment_reps} augmentation × borderline cells")

    # Pre-compute clean holds ONCE here — used by Phase A's P2 workers
    # (threaded through pilot_config) AND by Phase B directly.
    # This avoids the O(m·n) recomputation that previously occurred twice:
    # once inside run_rq3_experiment for Phase A, and once again for Phase B.
    print("   Pre-computing clean holds for adaptive experiment (shared across both phases)...")
    t_holds = time.time()
    clean_holds_adaptive = compute_holds_by_case_batch(case_data, candidates_all)
    print(f"   ✓ Clean holds ready in {time.time()-t_holds:.1f}s")

    # ── Phase A: Pilot ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE A — PILOT SWEEP")
    print(f"{'─'*60}")

    pilot_config = {**config, 'n_replicates': n_pilot_reps,
                    '_clean_holds': clean_holds_adaptive}
    pilot_metrics = run_rq3_experiment(
        case_data, candidates_all, S_clean,
        pilot_config, n_jobs_outer,
    )

    # ── Identify borderline cells ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE A → PHASE B: Identifying augmentation targets")
    print(f"{'─'*60}")
    augment_tasks = identify_augmentation_targets(
        pilot_metrics, config, additional_reps=n_augment_reps
    )

    if not augment_tasks:
        print("   ✓ All cells unambiguous — Phase B not needed.")
        return pilot_metrics

    # ── Phase B: Targeted augmentation ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE B — AUGMENTATION ({len(augment_tasks)} cells)")
    print(f"{'─'*60}")

    # Reuse the holds already computed above — no recomputation needed here.
    clean_holds = clean_holds_adaptive

    base_seed = config['random_state']

    # Build augment task list with exact rep indices
    augment_job_list = [
        (ptype, eps, rep)
        for ptype, eps, rep_indices in augment_tasks
        for rep in rep_indices
    ]

    # Compute inner core budget (mirrors run_rq3_experiment logic)
    import joblib as _jl
    C = _jl.cpu_count() or 1
    if n_jobs_outer > 1:
        cores_per_task = max(1, C // n_jobs_outer)
    else:
        cores_per_task = C
    n_workers_struct = max(1, cores_per_task // 2)
    n_jobs_calib = max(1, cores_per_task - n_workers_struct)

    phase_b_config = {
        'B_label':           config['B_label'],
        'B_trace':           config['B_trace'],
        'fdr_alpha':         config['fdr_alpha'],
        'fdr_method':        config.get('fdr_method', 'BH'),
        'random_state':      base_seed,
        'B_null':            config.get('B_null', 200),
        'B1_null':           config.get('B1_null', 75),
        'B2_null':           config.get('B2_null', 75),
        'n_jobs':            n_jobs_calib,
        'n_workers_struct':  n_workers_struct,
    }

    augment_metrics: List[ReplicateMetrics] = Parallel(
        n_jobs=n_jobs_outer, verbose=10, backend='loky',
    )(
        delayed(_rq3_worker_with_checkpoint)(
            ptype, eps, rep,
            case_data, candidates_all, S_clean,
            phase_b_config, base_seed,
            clean_holds=(clean_holds if ptype == 'P2' else None),
        )
        for ptype, eps, rep in augment_job_list
    )

    all_metrics = pilot_metrics + augment_metrics
    print(f"\n   ✓ Adaptive experiment complete: "
          f"{len(pilot_metrics)} pilot + {len(augment_metrics)} augmentation "
          f"= {len(all_metrics)} total replicates")
    return all_metrics


# ============================================================================
# RESULTS AGGREGATION AND ANALYSIS
# ============================================================================

@dataclass
class AggregatedResults:
    """Aggregated RQ3 results across all replicates."""
    # Per (ptype, epsilon): mean, CI for each metric
    jaccard_stats: Dict[str, Dict[float, Tuple[float, float, float]]]
    fdr_stats: Dict[str, Dict[float, Tuple[float, float, float]]]
    vacuity_stats: Dict[str, Dict[float, Tuple[float, float, float]]]

    # Four-category means
    category_means: Dict[str, Dict[float, Dict[str, float]]]

    # Family-level Jaccard
    family_jaccard_stats: Dict[str, Dict[float, Dict[str, Tuple[float, float, float]]]]

    # Inflection points
    inflection_points: Dict[str, Optional[float]]

    # Raw metrics for detailed analysis
    raw_metrics: List[ReplicateMetrics]


def aggregate_results(
    all_metrics: List[ReplicateMetrics],
    config: Dict,
) -> AggregatedResults:
    """
    Aggregate replicate metrics into summary statistics with bootstrap CIs.
    """
    print(f"\n{'='*100}")
    print("📊 AGGREGATING RESULTS")
    print(f"{'='*100}")

    n_boot = config.get('bootstrap_n', 1000)
    ci_level = config.get('ci_level', 0.95)
    rs = config.get('random_state', 42)
    threshold = config.get('jaccard_threshold', 0.80)
    epsilon_levels = config['epsilon_levels']

    perturbation_types = sorted(set(m.perturbation_type for m in all_metrics))

    # Group metrics by (ptype, epsilon)
    grouped = defaultdict(list)
    for m in all_metrics:
        grouped[(m.perturbation_type, m.epsilon)].append(m)

    jaccard_stats = {}
    fdr_stats = {}
    vacuity_stats = {}
    category_means = {}
    family_jaccard_stats = {}
    inflection_points = {}

    for ptype in perturbation_types:
        jaccard_stats[ptype] = {}
        fdr_stats[ptype] = {}
        vacuity_stats[ptype] = {}
        category_means[ptype] = {}
        family_jaccard_stats[ptype] = {}

        # For inflection point estimation
        jaccard_by_eps = {}

        for eps in epsilon_levels:
            metrics_list = grouped.get((ptype, eps), [])
            if not metrics_list:
                continue

            # M1: Jaccard
            j_vals = np.array([m.jaccard for m in metrics_list])
            jaccard_stats[ptype][eps] = bootstrap_ci(j_vals, n_boot, ci_level, rs)
            jaccard_by_eps[eps] = j_vals

            # M2: Empirical FDR
            f_vals = np.array([m.empirical_fdr for m in metrics_list])
            fdr_stats[ptype][eps] = bootstrap_ci(f_vals, n_boot, ci_level, rs)

            # M3: Vacuity
            v_vals = np.array([m.vacuity_rate for m in metrics_list])
            vacuity_stats[ptype][eps] = bootstrap_ci(v_vals, n_boot, ci_level, rs)

            # M4: Category means
            category_means[ptype][eps] = {
                'Both': np.mean([m.n_both for m in metrics_list]),
                'Structural only': np.mean([m.n_structural_only for m in metrics_list]),
                'Discriminative only': np.mean([m.n_discriminative_only for m in metrics_list]),
                'Neither': np.mean([m.n_neither for m in metrics_list]),
            }

            # Family Jaccard
            families = set()
            for m in metrics_list:
                families.update(m.family_jaccard.keys())

            family_jaccard_stats[ptype][eps] = {}
            for fam in families:
                fam_vals = np.array([
                    m.family_jaccard.get(fam, 0.0) for m in metrics_list
                ])
                family_jaccard_stats[ptype][eps][fam] = bootstrap_ci(
                    fam_vals, n_boot, ci_level, rs
                )

        # Inflection point
        inflection_points[ptype] = estimate_inflection_point(
            epsilon_levels, jaccard_by_eps, threshold, n_boot, ci_level, rs
        )

    # Print summary
    for ptype in perturbation_types:
        print(f"\n   {ptype} — Perturbation Summary:")
        print(f"   {'ε':>6s}  {'Jaccard':>10s}  {'95% CI':>20s}  "
              f"{'Emp FDR':>10s}  {'Vacuity':>10s}")
        print(f"   {'─'*65}")
        for eps in epsilon_levels:
            if eps in jaccard_stats[ptype]:
                j_m, j_lo, j_hi = jaccard_stats[ptype][eps]
                f_m, _, _ = fdr_stats[ptype][eps]
                v_m, _, _ = vacuity_stats[ptype][eps]
                print(f"   {eps:>6.2f}  {j_m:>10.4f}  [{j_lo:.4f}, {j_hi:.4f}]  "
                      f"{f_m:>10.4f}  {v_m:>10.4f}")

        inf_pt = inflection_points[ptype]
        if inf_pt is not None:
            print(f"   → Inflection point (J < {threshold}): ε* = {inf_pt:.2f}")
        else:
            print(f"   → Inflection point (J < {threshold}): not reached within ε ≤ 0.30")

    return AggregatedResults(
        jaccard_stats=jaccard_stats,
        fdr_stats=fdr_stats,
        vacuity_stats=vacuity_stats,
        category_means=category_means,
        family_jaccard_stats=family_jaccard_stats,
        inflection_points=inflection_points,
        raw_metrics=all_metrics,
    )


# ============================================================================
# VISUALIZATION — PUBLICATION QUALITY
# ============================================================================

def save_plot_pdf(fig, filename, dpi=300):
    pdf_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"      ✓ Saved: {filename}")


def generate_rq3_visualizations(
    agg: AggregatedResults,
    config: Dict,
):
    """Generate comprehensive publication-quality visualizations for RQ3."""

    print(f"\n{'='*100}")
    print("📊 GENERATING RQ3 VISUALIZATIONS")
    print(f"{'='*100}")

    epsilon_levels = config['epsilon_levels']
    ptypes = ['P1', 'P2', 'P3']
    ptype_labels = {
        'P1': 'Activity label noise (P1)',
        'P2': 'Class label noise (P2)',
        'P3': 'Trace truncation (P3)',
    }

    # ========================================================================
    # PLOT 1: Jaccard Stability Curves — All Perturbation Types
    # ========================================================================
    print("   [1/8] Jaccard Stability Curves...")

    fig, ax = plt.subplots(figsize=(10, 7))

    for ptype in ptypes:
        means = []; lowers = []; uppers = []
        for eps in epsilon_levels:
            if eps in agg.jaccard_stats[ptype]:
                m, lo, hi = agg.jaccard_stats[ptype][eps]
                means.append(m); lowers.append(lo); uppers.append(hi)
            else:
                means.append(np.nan); lowers.append(np.nan); uppers.append(np.nan)

        ax.plot(epsilon_levels, means, 'o-', color=RQ3_COLORS[ptype],
                linewidth=2.5, markersize=8, label=ptype_labels[ptype])
        ax.fill_between(epsilon_levels, lowers, uppers,
                        alpha=0.15, color=RQ3_COLORS[ptype])

    # Inflection threshold
    ax.axhline(config['jaccard_threshold'], color=RQ3_COLORS['inflection'],
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Stability threshold (J = {config["jaccard_threshold"]})')

    # Mark inflection points
    for ptype in ptypes:
        inf_pt = agg.inflection_points.get(ptype)
        if inf_pt is not None:
            ax.axvline(inf_pt, color=RQ3_COLORS[ptype], linestyle=':',
                       linewidth=1.0, alpha=0.5)
            ax.annotate(f'ε*={inf_pt:.2f}',
                        xy=(inf_pt, config['jaccard_threshold']),
                        xytext=(inf_pt + 0.02, config['jaccard_threshold'] - 0.08),
                        fontsize=9, color=RQ3_COLORS[ptype],
                        arrowprops=dict(arrowstyle='->', color=RQ3_COLORS[ptype]))

    ax.set_xlabel('Perturbation intensity ε', fontweight='bold')
    ax.set_ylabel('Jaccard specification stability $J(S_{clean}, S_{pert})$',
                  fontweight='bold')
    ax.set_title('M1 — Specification Stability Under Controlled Perturbations',
                 fontweight='bold')
    ax.set_xlim(-0.01, max(epsilon_levels) + 0.01)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    for sp in ax.spines.values(): sp.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_01_jaccard_stability_curves.pdf')

    # ========================================================================
    # PLOT 2: Empirical FDR Under Perturbation
    # ========================================================================
    print("   [2/8] Empirical FDR Curves...")

    fig, ax = plt.subplots(figsize=(10, 7))

    for ptype in ptypes:
        means = []; lowers = []; uppers = []
        for eps in epsilon_levels:
            if eps in agg.fdr_stats[ptype]:
                m, lo, hi = agg.fdr_stats[ptype][eps]
                means.append(m); lowers.append(lo); uppers.append(hi)
            else:
                means.append(np.nan); lowers.append(np.nan); uppers.append(np.nan)

        ax.plot(epsilon_levels, means, 's-', color=RQ3_COLORS[ptype],
                linewidth=2.5, markersize=8, label=ptype_labels[ptype])
        ax.fill_between(epsilon_levels, lowers, uppers,
                        alpha=0.15, color=RQ3_COLORS[ptype])

    ax.axhline(config['fdr_alpha'], color=RQ3_COLORS['threshold'],
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Nominal FDR α = {config["fdr_alpha"]}')

    ax.set_xlabel('Perturbation intensity ε', fontweight='bold')
    ax.set_ylabel('Empirical FDR = $|S_{pert} \\setminus S_{clean}| / |S_{pert}|$',
                  fontweight='bold')
    ax.set_title('M2 — Empirical False Discovery Rate Under Perturbation',
                 fontweight='bold')
    ax.set_xlim(-0.01, max(epsilon_levels) + 0.01)
    ax.set_ylim(-0.02, max(0.5, max(m for pt in ptypes for m, _, _ in
                                      [agg.fdr_stats[pt].get(e, (0,0,0))
                                       for e in epsilon_levels]) + 0.05))
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    for sp in ax.spines.values(): sp.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_02_empirical_fdr_curves.pdf')

    # ========================================================================
    # PLOT 3: Vacuity Rate (primarily for P3)
    # ========================================================================
    print("   [3/8] Vacuity Rate Curves...")

    fig, ax = plt.subplots(figsize=(10, 7))

    for ptype in ptypes:
        means = []; lowers = []; uppers = []
        for eps in epsilon_levels:
            if eps in agg.vacuity_stats[ptype]:
                m, lo, hi = agg.vacuity_stats[ptype][eps]
                means.append(m); lowers.append(lo); uppers.append(hi)
            else:
                means.append(np.nan); lowers.append(np.nan); uppers.append(np.nan)

        ax.plot(epsilon_levels, means, '^-', color=RQ3_COLORS[ptype],
                linewidth=2.5, markersize=8, label=ptype_labels[ptype])
        ax.fill_between(epsilon_levels, lowers, uppers,
                        alpha=0.15, color=RQ3_COLORS[ptype])

    ax.set_xlabel('Perturbation intensity ε', fontweight='bold')
    ax.set_ylabel('Vacuity rate (fraction of significant patterns)', fontweight='bold')
    ax.set_title('M3 — Vacuity Rate Under Controlled Perturbations', fontweight='bold')
    ax.set_xlim(-0.01, max(epsilon_levels) + 0.01)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    for sp in ax.spines.values(): sp.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_03_vacuity_rate_curves.pdf')

    # ========================================================================
    # PLOT 4: Four-Category Verdict Migration (one subplot per perturbation)
    # ========================================================================
    print("   [4/8] Four-Category Verdict Migration...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    categories = ['Both', 'Structural only', 'Discriminative only', 'Neither']
    cat_colors = [RQ3_COLORS['conjunction'], RQ3_COLORS['structural'],
                  RQ3_COLORS['discriminative'], RQ3_COLORS['neither']]

    for ax_idx, ptype in enumerate(ptypes):
        ax = axes[ax_idx]
        x = np.arange(len(epsilon_levels))
        width = 0.18

        for c_idx, cat in enumerate(categories):
            vals = [agg.category_means[ptype].get(eps, {}).get(cat, 0)
                    for eps in epsilon_levels]
            ax.bar(x + c_idx * width, vals, width,
                   label=cat if ax_idx == 0 else None,
                   color=cat_colors[c_idx], edgecolor='black', linewidth=0.8)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([f'{e:.2f}' for e in epsilon_levels])
        ax.set_xlabel('ε', fontweight='bold')
        ax.set_title(ptype_labels[ptype], fontweight='bold', fontsize=11)
        for sp in ax.spines.values(): sp.set_visible(True)

    axes[0].set_ylabel('Mean pattern count', fontweight='bold')
    axes[0].legend(frameon=True, fancybox=False, edgecolor='black',
                   fontsize=9, ncol=2)

    plt.suptitle('M4 — Four-Category Verdict Distribution Under Perturbation',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_04_verdict_migration.pdf')

    # ========================================================================
    # PLOT 5: DECLARE Family Stratification — Jaccard per Family
    # ========================================================================
    print("   [5/8] DECLARE Family Stratification...")

    families = ['Unary', 'Binary Positive', 'Binary Negative']
    fam_colors = ['#44AA99', '#882255', '#DDCC77']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, ptype in enumerate(ptypes):
        ax = axes[ax_idx]
        for f_idx, fam in enumerate(families):
            means = []; lowers = []; uppers = []
            for eps in epsilon_levels:
                fam_stats = agg.family_jaccard_stats.get(ptype, {}).get(eps, {})
                if fam in fam_stats:
                    m, lo, hi = fam_stats[fam]
                    means.append(m); lowers.append(lo); uppers.append(hi)
                else:
                    means.append(np.nan); lowers.append(np.nan); uppers.append(np.nan)

            ax.plot(epsilon_levels, means, 'o-', color=fam_colors[f_idx],
                    linewidth=2, markersize=6,
                    label=fam if ax_idx == 0 else None)
            ax.fill_between(epsilon_levels, lowers, uppers,
                            alpha=0.1, color=fam_colors[f_idx])

        ax.axhline(config['jaccard_threshold'], color='black',
                   linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('ε', fontweight='bold')
        ax.set_title(ptype_labels[ptype], fontweight='bold', fontsize=11)
        ax.set_xlim(-0.01, max(epsilon_levels) + 0.01)
        ax.set_ylim(-0.05, 1.05)
        for sp in ax.spines.values(): sp.set_visible(True)

    axes[0].set_ylabel('Jaccard stability (per family)', fontweight='bold')
    axes[0].legend(frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    plt.suptitle('DECLARE Family Stratification — Differential Degradation',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_05_family_stratification.pdf')

    # ========================================================================
    # PLOT 6: Axis Independence — P1 vs P2 Comparative Panel
    # ========================================================================
    print("   [6/8] Axis Independence Panel...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0,0): P1 → structural "Both" shrinks to "Neither"
    # Panel (0,1): P1 → discriminative axis unaffected
    # Panel (1,0): P2 → structural axis unaffected
    # Panel (1,1): P2 → discriminative "Both" shrinks to "Structural only"

    for row, ptype in enumerate(['P1', 'P2']):
        for col, cat_pair in enumerate([
            ('Both', 'Structural only'),
            ('Discriminative only', 'Neither')
        ]):
            ax = axes[row][col]
            for cat, ls in zip(cat_pair, ['-', '--']):
                vals = [agg.category_means[ptype].get(eps, {}).get(cat, 0)
                        for eps in epsilon_levels]
                color = {
                    'Both': RQ3_COLORS['conjunction'],
                    'Structural only': RQ3_COLORS['structural'],
                    'Discriminative only': RQ3_COLORS['discriminative'],
                    'Neither': RQ3_COLORS['neither'],
                }[cat]
                ax.plot(epsilon_levels, vals, 'o' + ls, color=color,
                        linewidth=2, markersize=6, label=cat)

            ax.set_xlabel('ε', fontweight='bold')
            ax.set_ylabel('Mean count', fontweight='bold')
            ax.set_title(f'{ptype_labels[ptype][:15]}...', fontweight='bold', fontsize=10)
            ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
            for sp in ax.spines.values(): sp.set_visible(True)

    plt.suptitle('Axis Independence: Category Migration Under P1 vs P2',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_06_axis_independence.pdf')

    # ========================================================================
    # PLOT 7: Heatmap — All Metrics × All Perturbations
    # ========================================================================
    print("   [7/8] Summary Heatmap...")

    metric_names = ['Jaccard (M1)', 'Emp FDR (M2)', 'Vacuity (M3)']
    n_metrics = len(metric_names)
    n_ptypes = len(ptypes)
    n_eps = len(epsilon_levels)

    heatmap_data = np.zeros((n_metrics * n_ptypes, n_eps))
    row_labels = []

    for m_idx, (metric_key, stats_dict) in enumerate([
        ('Jaccard', agg.jaccard_stats),
        ('Emp FDR', agg.fdr_stats),
        ('Vacuity', agg.vacuity_stats),
    ]):
        for p_idx, ptype in enumerate(ptypes):
            row = m_idx * n_ptypes + p_idx
            row_labels.append(f'{metric_names[m_idx]} — {ptype}')
            for e_idx, eps in enumerate(epsilon_levels):
                if eps in stats_dict[ptype]:
                    heatmap_data[row, e_idx] = stats_dict[ptype][eps][0]  # mean

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_eps))
    ax.set_xticklabels([f'ε={e:.2f}' for e in epsilon_levels])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(n_eps):
            val = heatmap_data[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    # Add separator lines between metric groups
    for m_idx in range(1, n_metrics):
        ax.axhline(m_idx * n_ptypes - 0.5, color='black', linewidth=2)

    plt.colorbar(im, ax=ax, label='Metric value', shrink=0.8)
    ax.set_title('RQ3 Sensitivity Analysis — Summary Heatmap', fontweight='bold')
    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_07_summary_heatmap.pdf')

    # ========================================================================
    # PLOT 8: Summary Dashboard
    # ========================================================================
    print("   [8/8] Summary Dashboard...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Jaccard curves (compact)
    ax1 = fig.add_subplot(gs[0, 0])
    for ptype in ptypes:
        means = [agg.jaccard_stats[ptype].get(e, (np.nan,))[0] for e in epsilon_levels]
        ax1.plot(epsilon_levels, means, 'o-', color=RQ3_COLORS[ptype],
                 linewidth=2, markersize=5, label=ptype)
    ax1.axhline(config['jaccard_threshold'], color='black', linestyle='--',
                linewidth=1, alpha=0.5)
    ax1.set_ylabel('Jaccard', fontweight='bold')
    ax1.set_xlabel('ε', fontweight='bold')
    ax1.set_title('M1: Stability', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=8, frameon=True)
    for sp in ax1.spines.values(): sp.set_visible(True)

    # Panel 2: FDR curves (compact)
    ax2 = fig.add_subplot(gs[0, 1])
    for ptype in ptypes:
        means = [agg.fdr_stats[ptype].get(e, (np.nan,))[0] for e in epsilon_levels]
        ax2.plot(epsilon_levels, means, 's-', color=RQ3_COLORS[ptype],
                 linewidth=2, markersize=5, label=ptype)
    ax2.axhline(config['fdr_alpha'], color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel('Emp FDR', fontweight='bold')
    ax2.set_xlabel('ε', fontweight='bold')
    ax2.set_title('M2: Empirical FDR', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=8, frameon=True)
    for sp in ax2.spines.values(): sp.set_visible(True)

    # Panel 3: Vacuity curves (compact)
    ax3 = fig.add_subplot(gs[0, 2])
    for ptype in ptypes:
        means = [agg.vacuity_stats[ptype].get(e, (np.nan,))[0] for e in epsilon_levels]
        ax3.plot(epsilon_levels, means, '^-', color=RQ3_COLORS[ptype],
                 linewidth=2, markersize=5, label=ptype)
    ax3.set_ylabel('Vacuity rate', fontweight='bold')
    ax3.set_xlabel('ε', fontweight='bold')
    ax3.set_title('M3: Vacuity', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=8, frameon=True)
    for sp in ax3.spines.values(): sp.set_visible(True)

    # Panels 4-6: Category migration per perturbation type
    categories = ['Both', 'Structural only', 'Discriminative only', 'Neither']
    cat_colors_list = [RQ3_COLORS['conjunction'], RQ3_COLORS['structural'],
                       RQ3_COLORS['discriminative'], RQ3_COLORS['neither']]

    for p_idx, ptype in enumerate(ptypes):
        ax = fig.add_subplot(gs[1, p_idx])
        for c_idx, cat in enumerate(categories):
            vals = [agg.category_means[ptype].get(e, {}).get(cat, 0)
                    for e in epsilon_levels]
            ax.plot(epsilon_levels, vals, 'o-', color=cat_colors_list[c_idx],
                    linewidth=1.5, markersize=4,
                    label=cat if p_idx == 0 else None)
        ax.set_xlabel('ε', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f'M4: {ptype}', fontweight='bold', fontsize=11)
        if p_idx == 0:
            ax.legend(fontsize=7, frameon=True, ncol=2)
        for sp in ax.spines.values(): sp.set_visible(True)

    # Panel 7: Text summary
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    inf_text = ""
    for ptype in ptypes:
        inf_pt = agg.inflection_points.get(ptype)
        inf_text += f"  {ptype}: ε* = {inf_pt:.2f}\n" if inf_pt else f"  {ptype}: not reached\n"

    summary = f"""
    RQ3 DIRECTION 2 — SENSITIVITY ANALYSIS SUMMARY  [Production Log]
    ════════════════════════════════════════════════════════════════════════
    Framework: Sommers et al. (2025) Model-Transformation Perturbation
    Pipeline:  Dual-Axis Storey FDR (v8.0-DUAL-AXIS-STOREY-FDR)
    Log:       Production Manufacturing Log — Deviant (Re-submission) vs. Regular
    Outcome:   Deviant (Re-submission Required) vs. Regular (No Re-submission)
    ε levels:  {epsilon_levels}
    Replicates: {config['n_replicates']} per (perturbation, ε)
    B₁={config['B_label']:,}, B₂={config['B_trace']:,}, α={config['fdr_alpha']}

    Inflection Points (J < {config['jaccard_threshold']}):
{inf_text}
    Expected Degradation Patterns (Production-specific):
      P1 (activity noise):  Structural axis degrades; longer manufacturing traces
                             dampen per-swap impact vs. short clinical logs
      P2 (class noise):     Deviant minority diluted; Δ_obs collapses rapidly;
                             inflection driven by minority-class size
      P3 (truncation):      Both axes degrade; Succession family most affected;
                             simulates incomplete step-completion recording
    """

    ax_text.text(0.05, 0.5, summary, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='black', linewidth=1.5))

    plt.tight_layout()
    save_plot_pdf(fig, 'RQ3_08_summary_dashboard.pdf')

    print(f"\n   ✓ Generated 8 RQ3 visualizations in {PLOTS_DIR}")


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_rq3_outputs(
    agg: AggregatedResults,
    config: Dict,
):
    """Save all RQ3 results to JSON and text report."""

    print(f"\n{'='*100}")
    print("📊 SAVING RQ3 OUTPUTS")
    print(f"{'='*100}")

    # ── JSON output ──────────────────────────────────────────────────────
    json_out = {
        'framework': 'RQ3 Direction 2 — Sensitivity Analysis Under Controlled Log Perturbations',
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'scientific_basis': 'Sommers et al. (2025) model-transformation perturbation framework',
        'configuration': {
            'epsilon_levels': config['epsilon_levels'],
            'n_replicates': config['n_replicates'],
            'B_label': config['B_label'],
            'B_trace': config['B_trace'],
            'fdr_alpha': config['fdr_alpha'],
            'jaccard_threshold': config['jaccard_threshold'],
            'bootstrap_n': config['bootstrap_n'],
        },
        'inflection_points': {
            pt: agg.inflection_points.get(pt) for pt in ['P1', 'P2', 'P3']
        },
        'metrics_summary': {},
        'family_stratification': {},
        'replicate_details': [],
    }

    for ptype in ['P1', 'P2', 'P3']:
        json_out['metrics_summary'][ptype] = {}
        for eps in config['epsilon_levels']:
            j = agg.jaccard_stats[ptype].get(eps, (None, None, None))
            f = agg.fdr_stats[ptype].get(eps, (None, None, None))
            v = agg.vacuity_stats[ptype].get(eps, (None, None, None))
            cats = agg.category_means[ptype].get(eps, {})

            json_out['metrics_summary'][ptype][str(eps)] = {
                'jaccard': {'mean': j[0], 'ci_lower': j[1], 'ci_upper': j[2]},
                'empirical_fdr': {'mean': f[0], 'ci_lower': f[1], 'ci_upper': f[2]},
                'vacuity_rate': {'mean': v[0], 'ci_lower': v[1], 'ci_upper': v[2]},
                'category_means': cats,
            }

        # Family stratification
        json_out['family_stratification'][ptype] = {}
        for eps in config['epsilon_levels']:
            fam_data = agg.family_jaccard_stats.get(ptype, {}).get(eps, {})
            json_out['family_stratification'][ptype][str(eps)] = {
                fam: {'mean': s[0], 'ci_lower': s[1], 'ci_upper': s[2]}
                for fam, s in fam_data.items()
            }

    # Replicate details (compact)
    for m in agg.raw_metrics:
        json_out['replicate_details'].append({
            'perturbation_type': m.perturbation_type,
            'epsilon': m.epsilon,
            'replicate': m.replicate,
            'jaccard': m.jaccard,
            'empirical_fdr': m.empirical_fdr,
            'vacuity_rate': m.vacuity_rate,
            'n_both': m.n_both,
            'n_structural_only': m.n_structural_only,
            'n_discriminative_only': m.n_discriminative_only,
            'n_neither': m.n_neither,
            'n_S_pert': m.n_S_pert,
            'n_intersection': m.n_intersection,
            'n_false_discoveries': m.n_false_discoveries,
            'family_jaccard': m.family_jaccard,
        })

    json_path = os.path.join(OUTPUT_DIR, 'rq3_direction2_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✓ JSON: {json_path}")

    # ── Text report ──────────────────────────────────────────────────────
    rpt = []
    rpt.append("=" * 120)
    rpt.append("RQ3 DIRECTION 2 — SENSITIVITY ANALYSIS UNDER CONTROLLED LOG PERTURBATIONS")
    rpt.append("Sommers et al. (2025) Model-Transformation Perturbation Framework")
    rpt.append("=" * 120)
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Version: 1.0")
    rpt.append("")
    rpt.append(f"Configuration:")
    rpt.append(f"  ε levels:      {config['epsilon_levels']}")
    rpt.append(f"  Replicates:    {config['n_replicates']}")
    rpt.append(f"  B₁ (label):    {config['B_label']:,}")
    rpt.append(f"  B₂ (trace):    {config['B_trace']:,}")
    rpt.append(f"  FDR α:         {config['fdr_alpha']}")
    rpt.append(f"  J threshold:   {config['jaccard_threshold']}")
    rpt.append("")

    for ptype in ['P1', 'P2', 'P3']:
        ptype_desc = {
            'P1': 'Activity label noise (RIa_in) → targets H₀ˢ',
            'P2': 'Class label noise (RIc_in) → targets H₀ᵈ',
            'P3': 'Trace truncation (RIe_mi) → targets both axes',
        }[ptype]
        rpt.append("=" * 120)
        rpt.append(f"{ptype} — {ptype_desc}")
        rpt.append("=" * 120)
        rpt.append("")
        rpt.append(f"  {'ε':>6s}  {'Jaccard':>10s}  {'95% CI':>22s}  "
                    f"{'Emp FDR':>10s}  {'95% CI':>22s}  {'Vacuity':>10s}")
        rpt.append(f"  {'─'*90}")

        for eps in config['epsilon_levels']:
            j = agg.jaccard_stats[ptype].get(eps, (0, 0, 0))
            f = agg.fdr_stats[ptype].get(eps, (0, 0, 0))
            v = agg.vacuity_stats[ptype].get(eps, (0, 0, 0))
            rpt.append(f"  {eps:>6.2f}  {j[0]:>10.4f}  [{j[1]:.4f}, {j[2]:.4f}]  "
                        f"{f[0]:>10.4f}  [{f[1]:.4f}, {f[2]:.4f}]  {v[0]:>10.4f}")

        inf_pt = agg.inflection_points.get(ptype)
        rpt.append("")
        if inf_pt:
            rpt.append(f"  Inflection point: ε* = {inf_pt:.2f}")
        else:
            rpt.append(f"  Inflection point: not reached (J ≥ {config['jaccard_threshold']} for all ε)")
        rpt.append("")

        # Category means
        rpt.append(f"  Four-Category Verdict Means:")
        rpt.append(f"  {'ε':>6s}  {'Both':>8s}  {'Struct':>8s}  {'Disc':>8s}  {'Neither':>8s}")
        rpt.append(f"  {'─'*45}")
        for eps in config['epsilon_levels']:
            cats = agg.category_means[ptype].get(eps, {})
            rpt.append(f"  {eps:>6.2f}  {cats.get('Both', 0):>8.1f}  "
                        f"{cats.get('Structural only', 0):>8.1f}  "
                        f"{cats.get('Discriminative only', 0):>8.1f}  "
                        f"{cats.get('Neither', 0):>8.1f}")
        rpt.append("")

        # Family stratification
        rpt.append(f"  DECLARE Family Stratification (Jaccard means):")
        rpt.append(f"  {'ε':>6s}  {'Unary':>10s}  {'Bin Pos':>10s}  {'Bin Neg':>10s}")
        rpt.append(f"  {'─'*40}")
        for eps in config['epsilon_levels']:
            fam_data = agg.family_jaccard_stats.get(ptype, {}).get(eps, {})
            u = fam_data.get('Unary', (0, 0, 0))[0]
            bp = fam_data.get('Binary Positive', (0, 0, 0))[0]
            bn = fam_data.get('Binary Negative', (0, 0, 0))[0]
            rpt.append(f"  {eps:>6.2f}  {u:>10.4f}  {bp:>10.4f}  {bn:>10.4f}")
        rpt.append("")

    # Scientific interpretation
    rpt.append("=" * 120)
    rpt.append("SCIENTIFIC INTERPRETATION — PRODUCTION MANUFACTURING LOG")
    rpt.append("=" * 120)
    rpt.append("")
    rpt.append("Dataset: Production Manufacturing Event Log")
    rpt.append("Outcome: Deviant — Re-submission Required (Class 1) vs. Regular — No Re-submission (Class 0)")
    rpt.append("Label encoding: pre-encoded 'label' column attribute; no signal stripping required")
    rpt.append("Subgroups: Part_Desc_ (product type) × Report_Type (B/D/S)")
    rpt.append("")
    rpt.append("Expected differential degradation patterns (Production-specific):")
    rpt.append("")
    rpt.append("P1 (Activity label noise — targets H₀ˢ):")
    rpt.append("  - Structural axis degrades monotonically in ε")
    rpt.append("  - Longer manufacturing traces (vs. clinical logs) dampen per-swap impact:")
    rpt.append("    a single position swap in a long workflow sequence has lower relative")
    rpt.append("    disruption than in short traces; inflection point may be at higher ε")
    rpt.append("  - Discriminative axis unaffected (class labels unchanged)")
    rpt.append("  - Expected verdict migration: 'Both' → 'Discriminative-only',")
    rpt.append("    'Structural-only' → 'Neither'")
    rpt.append("")
    rpt.append("P2 (Class label noise — targets H₀ᵈ):")
    rpt.append("  - Discriminative axis degrades; Deviant minority class diluted by label flips")
    rpt.append("  - Δ_obs collapses as flipped cases inject null-like signal into L+")
    rpt.append("  - Inflection driven by minority-class proportion; smaller Deviant fraction")
    rpt.append("    → earlier inflection point than in balanced logs")
    rpt.append("  - Structural axis unaffected (trace structures fully preserved)")
    rpt.append("  - Expected: 'Both' → 'Structural-only', phase transition at low ε")
    rpt.append("")
    rpt.append("P3 (Trace truncation — targets both axes via vacuity):")
    rpt.append("  - Simulates incomplete step-completion recording in manufacturing")
    rpt.append("    (e.g., missing final quality sign-off or inspection events)")
    rpt.append("  - Vacuity rate rises monotonically; Succession family most affected")
    rpt.append("  - Init/End constraints immune (unary, position-based)")
    rpt.append("  - ChainSuccession/AlternateSuccession disproportionately eliminated")
    rpt.append("    as tail events encoding process completion steps are removed")
    rpt.append("  - q-values inflate across both axes as applicability counts drop")
    rpt.append("")
    rpt.append("=" * 120)
    rpt.append("END OF REPORT")
    rpt.append("=" * 120)

    report_path = os.path.join(OUTPUT_DIR, 'rq3_direction2_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rpt))
    print(f"   ✓ Text report: {report_path}")

    # ── CSV summary for easy plotting ────────────────────────────────────
    rows = []
    for m in agg.raw_metrics:
        rows.append({
            'perturbation_type': m.perturbation_type,
            'epsilon': m.epsilon,
            'replicate': m.replicate,
            'jaccard': m.jaccard,
            'empirical_fdr': m.empirical_fdr,
            'vacuity_rate': m.vacuity_rate,
            'n_both': m.n_both,
            'n_structural_only': m.n_structural_only,
            'n_discriminative_only': m.n_discriminative_only,
            'n_neither': m.n_neither,
            'n_S_pert': m.n_S_pert,
            'n_intersection': m.n_intersection,
            'n_false_discoveries': m.n_false_discoveries,
        })
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'rq3_replicate_metrics.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"   ✓ CSV: {csv_path}")

    print(f"\n✅ All RQ3 outputs saved to: {OUTPUT_DIR}")


# ============================================================================
# MAIN
# ============================================================================

def main(n_jobs_outer: int = 1, args=None):
    print("\n" + "=" * 100)
    print("RQ3 DIRECTION 2 — SENSITIVITY ANALYSIS UNDER CONTROLLED LOG PERTURBATIONS")
    print("Sommers et al. (2025) Model-Transformation Perturbation Framework")
    print("Applied to Dual-Axis Storey FDR Discriminative Specification Mining Pipeline")
    print("LOG: Production Manufacturing Log — Deviant (Re-submission Required) vs. Regular")
    print("=" * 100)

    print("\n🎯 SCIENTIFIC CLAIM:")
    print("   Not whether the framework controls FDR at nominal α (RQ1),")
    print("   but whether it degrades gracefully when the log deviates from ideality.")
    print("")
    print("   Perturbation types:")
    print("     P1 — Activity label noise (RIa_in) → H₀ˢ structural axis")
    print("     P2 — Class label noise (RIc_in) → H₀ᵈ discriminative axis")
    print("     P3 — Trace truncation (RIe_mi) → both axes via vacuity")
    print("")
    print(f"   ε ∈ {RQ3_CONFIG['epsilon_levels']}")
    print(f"   R = {RQ3_CONFIG['n_replicates']} replicates per (type, ε)")
    print(f"   Total pipeline runs: "
          f"{3 * len(RQ3_CONFIG['epsilon_levels']) * RQ3_CONFIG['n_replicates']}")
    print("=" * 100)

    t_start = time.time()

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n📊 STEP 1: LOADING DATA AND CLEAN SPECIFICATION")
    case_data = load_event_log(INPUT_CSV)
    candidates_pos, candidates_neg, candidates_all = load_candidates_from_spec(DECLARE_SPEC_FILE)
    S_clean, clean_categories = load_clean_specification(CLEAN_RESULTS_FILE)

    # ── Step 2: Run experiment ───────────────────────────────────────────
    print("\n📊 STEP 2: RUNNING PERTURBATION EXPERIMENT")
    _n_pilot    = getattr(args, 'n_pilot_reps',   3)
    _n_augment  = getattr(args, 'n_augment_reps',  7)
    _no_adaptive = getattr(args, 'no_adaptive',   False)
    all_metrics = run_rq3_adaptive(
        case_data, candidates_all, S_clean, RQ3_CONFIG,
        n_jobs_outer=n_jobs_outer,
        n_pilot_reps=_n_pilot,
        n_augment_reps=_n_augment,
        use_adaptive=not _no_adaptive,
    )

    # ── Step 3: Aggregate results ────────────────────────────────────────
    print("\n📊 STEP 3: AGGREGATING RESULTS")
    agg = aggregate_results(all_metrics, RQ3_CONFIG)

    # ── Step 4: Generate visualizations ──────────────────────────────────
    print("\n📊 STEP 4: GENERATING VISUALIZATIONS")
    generate_rq3_visualizations(agg, RQ3_CONFIG)

    # ── Step 5: Save outputs ─────────────────────────────────────────────
    print("\n📊 STEP 5: SAVING OUTPUTS")
    save_rq3_outputs(agg, RQ3_CONFIG)

    total_time = time.time() - t_start

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("✅ RQ3 DIRECTION 2 — COMPLETE")
    print(f"{'='*100}")
    print(f"\n   Total execution time: {total_time:.0f}s ({total_time/60:.1f} min, "
          f"{total_time/3600:.2f} hr)")
    print(f"\n   Inflection points (J < {RQ3_CONFIG['jaccard_threshold']}):")
    for ptype in ['P1', 'P2', 'P3']:
        inf_pt = agg.inflection_points.get(ptype)
        if inf_pt:
            print(f"     {ptype}: ε* = {inf_pt:.2f}")
        else:
            print(f"     {ptype}: not reached within ε ≤ 0.30")

    print(f"\n📁 Outputs: {OUTPUT_DIR}")
    print(f"   • rq3_direction2_results.json    (full results)")
    print(f"   • rq3_direction2_report.txt      (text report)")
    print(f"   • rq3_replicate_metrics.csv      (replicate-level CSV)")
    print(f"   • visualizations/")
    print(f"     – RQ3_01_jaccard_stability_curves.pdf")
    print(f"     – RQ3_02_empirical_fdr_curves.pdf")
    print(f"     – RQ3_03_vacuity_rate_curves.pdf")
    print(f"     – RQ3_04_verdict_migration.pdf")
    print(f"     – RQ3_05_family_stratification.pdf")
    print(f"     – RQ3_06_axis_independence.pdf")
    print(f"     – RQ3_07_summary_heatmap.pdf")
    print(f"     – RQ3_08_summary_dashboard.pdf")
    print(f"\n{'='*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ3 Direction 2 — Sensitivity Analysis (Parallel)"
    )
    parser.add_argument('--n-jobs', type=int, default=-1,
        help='Outer parallelism over 150 (ptype, ε, rep) tasks (-1 = all cores)')
    parser.add_argument('--n-jobs-inner', type=int, default=None,
        help='Inner calibration parallelism (default: auto-computed from --n-jobs '
             'and total cores). Set explicitly to override auto-budget.')
    parser.add_argument('--n-replicates', type=int,
        default=RQ3_CONFIG['n_replicates'],
        help=f'Replicates per (ptype, ε) (default: {RQ3_CONFIG["n_replicates"]})')
    parser.add_argument('--n-pilot-reps', type=int, default=3,
        help='Phase A pilot replicates per (ptype, ε) cell (default: 3)')
    parser.add_argument('--n-augment-reps', type=int, default=7,
        help='Phase B augmentation replicates per borderline cell (default: 7)')
    parser.add_argument('--no-adaptive', action='store_true',
        help='Disable two-phase design; run full fixed R=n_replicates design')
    parser.add_argument('--dry-run', action='store_true',
        help='Run 1 replicate at ε=0.02 only for quick smoke-test')
    args = parser.parse_args()

    if args.dry_run:
        RQ3_CONFIG['n_replicates'] = 1
        RQ3_CONFIG['epsilon_levels'] = [0.02]
        print("*** DRY RUN MODE: 1 replicate, ε=0.02 only ***")
    else:
        RQ3_CONFIG['n_replicates'] = args.n_replicates

    if args.n_jobs_inner is not None:
        # Manual override — skip auto-budget arithmetic in run_rq3_experiment
        RQ3_CONFIG['n_jobs'] = args.n_jobs_inner
        RQ3_CONFIG['n_workers_struct'] = 1
    # else: auto-budget computed inside run_rq3_experiment (Category 6)

    main(n_jobs_outer=args.n_jobs, args=args)