"""
THREE-HYPOTHESIS DISCRIMINATIVE SPECIFICATION MINING (PHASE 1)
===============================================================
Storey CONJUNCTION TEST FOR DECLARE PATTERNS

SCIENTIFIC FRAMEWORK:
---------------------
This implementation tests three distinct null hypotheses per pattern,
following the formalization of Ojala & Garriga (2010) for classifier
significance testing, adapted to discriminative DECLARE specification mining.

NULL HYPOTHESIS 1 — H₀ˢ (Structural Null):
    "The observed prevalence of pattern p in class y is consistent with
     purely random temporal ordering within class y."
    Permutation: Activities within each trace are shuffled.
    Preserves:   Trace length, activity multiset, class labels.
    Destroys:    Sequential/temporal structure.
    Purpose:     Confirms the pattern captures genuine temporal regularity,
                 not merely a marginal-frequency artifact.

NULL HYPOTHESIS 2 — H₀ᵈ (Discriminative Null):
    "There is no difference in pattern prevalence between classes."
    Permutation: Class labels are randomly reassigned across cases.
    Preserves:   Full trace temporal structure, marginal class counts.
    Destroys:    Association between traces and class membership.
    Purpose:     Directly tests whether the pattern discriminates between
                 Not-Granted vs. Granted — the primary scientific claim.

NULL HYPOTHESIS 3 — H₀ᶜ (Conjunction):
    "Either H₀ˢ or H₀ᵈ (or both) hold."
    Test:        Fisher (1932) combination under joint null H₀ᶜ = H₀ˢ ∩ H₀ᵈ.
    Statistic:   T_F(p) = −2(ln p_struct_dom(p) + ln p_disc(p)) ~ χ²_4 under H₀ᶜ.
    p-value:     p_Fisher(p) = P(χ²_4 ≥ T_F(p))  [uniformly more powerful than IUT]
    Purpose:     Pattern is either structurally non-random or class-specific.
    Note:        p_struct and p_disc arise from independent permutation schemes;
                 under dependence the χ²_4 approximation is conservative (safe).

STATISTICAL CONTROL:
--------------------
- Phipson & Smyth (2010) exact permutation p-values:
      p = (1 + #{T_b ≥ T_obs}) / (B + 1)
  This formula is exact and yields stochastically uniform p-values under H₀.

- Storey (2002) Q-Value FDR applied to per-pattern Phipson-Smyth p-values:
      q(p) = min_{k'≥k} [π̂₀ · m' · p_(k') / k']
  where p_p^PS = (1 + #{b : |Δ_b,p| ≥ |Δ_obs,p|}) / (B₁ + 1) is the
  per-pattern exact p-value from column p of null_delta_matrix.
- Gao (2023) Adaptive Storey (AS) π̂₀ estimator applied to all three axes.
  Replaces bootstrap MSE-minimisation (Storey-Taylor-Siegmund 2004).
  Algorithm 1 + robust stopping time τ* (Eq. 7):
      τ* = inf{ λ_{j+1} ≤ 0.80 : ψ(λ_{j+1}) ≥ ψ(λ_j) }
      ψ(λ) = π̂₀(λ) + V(λ),  V = Binomial plug-in variance of π̂₀(λ).
  FDR validity: Proposition 1 (Gao 2023) — super-martingale + optional
  stopping theorem; valid for super-uniform (Phipson-Smyth) p-values.
  This replaces the Tusher flat-null approach which fails at m' < 500:
  pooling all B₁×m' null values cross-contaminates high-variance patterns
  (Init/End: σ_null ≈ 0.3) with low-variance ones (NotChainSuccession:
  σ_null ≈ 0.006), inflating Ê[V(τ)] ~80× and collapsing k* to 0.
  The per-pattern approach has zero cross-contamination and is exact at
  any m' ≥ 1 with no extra permutation cost.
  Final significance: q_value_sam ≤ α  (Fisher-Storey is the primary gate).
  Structural q-values on m' scope are used for taxonomy only (not gating).

  BH (Benjamini-Hochberg 1995) is retained as a secondary reference for
  comparison; it is not the primary significance criterion.

STATISTICAL DESIGN — SINGLE-GATE ARCHITECTURE:
------------------------------------------------
Fisher-Storey (Step 5b) is the sole significance gate:
    is_significant_final = is_significant_discriminative = (q_Fisher ≤ α)

Structural evidence is already encoded inside p_Fisher via the Fisher combination
statistic T_F = −2(ln p_struct_dom + ln p_disc).  Using structural p-values as a
second gate would penalise patterns twice for the same evidence, and using them as
both selector (structural_idx) and test statistic introduces selection bias.

Step 5c computes Storey q-values on m' structural p-values for JSON transparency,
but they NEVER gate is_significant_final.  is_significant_structural is a raw
nominal label (p_structural_dominant ≤ α) used solely for taxonomy.

FOUR-CATEGORY TAXONOMY (descriptive, within Fisher-significant set):
---------------------------------------------------------------------
    Both               q_Fisher ≤ α  AND  p_struct_dom ≤ α  (nominal)
                       Fisher-significant with corroborating nominal structural signal.
    Discriminative only q_Fisher ≤ α  AND  p_struct_dom > α
                       Fisher-significant; structural signal is weak (Fisher still valid
                       because T_F penalises large p_struct automatically).
    Structural only    q_Fisher > α  AND  p_struct_dom ≤ α  (nominal)
                       Real temporal regularity; not class-discriminative.
    Neither            Both criteria fail.  Not retained.

is_significant_final = True iff "Both" or "Discriminative only".

COMPUTATIONAL ARCHITECTURE:
---------------------------
Step 1: Candidate generation from Phase 0 CC specification.
Step 2: Pre-compute holds_by_case on observed log (once, reused everywhere).
Step 3: Label permutation (B₁ resamples) — O(B₁ · m · n), very cheap.
        Stores null_delta_matrix (B₁ × m, float32) for SAM FDR estimation.
Step 4: Trace-activity permutation (B₂ resamples) — O(B₂ · m · n · L), expensive.
        Run on full union for both classes; every pattern has a real p_structural
        for both D₀ and D₁ (no sentinel 1.0 values).
Step 5a: Fisher conjunction p_Fisher = chi2.sf(−2(ln p_struct_dom + ln p_disc), 4)
         + BH-FDR on m' structurally-touched patterns (reference).
Step 5b: Storey (2002) Q-Value FDR on p_Fisher (discriminative axis).
         q_disc = storey_qvalue(p_Fisher, π̂₀) over m' patterns.
         is_significant_discriminative = q_disc ≤ α.
Step 5c: Storey (2002) Q-Value FDR on structural p-values — for transparency only.
         Two corrections on m' (structural_idx): q_sc0, q_sc1 stored in JSON.
         is_significant_structural = p_structural_dominant ≤ α  (raw nominal,
           avoids selection bias from using the same p-values as selector+test).
         is_significant_final = is_significant_discriminative  (Fisher sole gate).
         Four-category taxonomy assigned here (descriptive, not a second gate).

KEY ADVANTAGES OVER BH-FDR:
----------------------------
1. π̂₀ correction: Storey (2002) estimates the null fraction π̂₀ from
   per-pattern Phipson-Smyth p-values. When π̂₀ < 1 (signals present),
   FDR̂ is deflated by factor π̂₀, yielding strictly more rejections than BH.
2. Per-pattern comparison: p_p^PS uses only column p of null_delta_matrix —
   zero cross-contamination from other patterns' null ranges. The flat-null
   pooling (Tusher 2001) fails at m' < 500 due to heterogeneous σ_null.
3. Exact at any m': no large-m approximation; valid FDR control even at m'=1.
4. Sparsity-aware: π̂₀ down-weights the FDR estimate when signals are dense.
5. No distributional assumptions: purely non-parametric.
6. Computational efficiency: null_delta_matrix is filled in zero extra passes
   — all B₁ permutation Δ-vectors are already computed in step 3.

Version: 8.0-DUAL-AXIS-STOREY-FDR
Author: Ahmed Nour Abdesselam
Institution: Free University of Bozen-Bolzano
Date: March 2026

References:
-----------
- Storey (2002): A direct approach to false discovery rates. JRSS-B 64(3):479-498.
- Storey & Tibshirani (2003): Statistical significance for genomewide studies.
  PNAS 100(16):9440-9445.
- Tusher, Tibshirani & Chu (2001): Significance analysis of microarrays applied
  to the ionizing radiation response. PNAS 98(9):5116-5121.
- Ojala & Garriga (2010): Permutation Tests for Studying Classifier Performance. JMLR.
- Phipson & Smyth (2010): Permutation p-values should never be zero.
  Stat. Appl. Genet. Mol. Biol. 9(1):Article 39.
- Benjamini & Hochberg (1995): Controlling the False Discovery Rate. JRSS-B.
- Benjamini & Yekutieli (2001): The Control of the FDR Under Dependency. Ann. Stat.
- Berger (1982): Multiparameter Hypothesis Testing and Acceptance Sampling. Technometrics.
- Di Ciccio & Montali (2022): DECLARE constraint semantics.
- Pellegrina et al. (2021): Statistical Significance in Pattern Mining.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
import warnings
import os
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# ============================================================================
# SCIENTIFIC PLOTTING CONFIGURATION — PUBLICATION QUALITY
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
    'class0': '#D55E00',     # Vermillion — Class 0 (Granted / Normal)
    'class1': '#0072B2',     # Blue — Class 1 (Not-Granted / Deviant)
    'both': '#CC79A7',       # Reddish purple — both
    'neither': '#999999',    # Gray — neither
    'null': '#E5E5E5',       # Light gray — null
    'accent1': '#009E73',    # Bluish green
    'accent2': '#F0E442',    # Yellow
    'accent3': '#56B4E9',    # Sky blue
    'mc': '#E69F00',         # Orange — MC components
    'threshold': '#CC0000',  # Deep red — thresholds
    'structural': '#882255', # Wine — structural p-values
    'discriminative': '#117733',  # Forest green — discriminative p-values
    'conjunction': '#332288',     # Indigo — conjunction p-values
}

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "../Experiments data/CSV/BPI_Challenge_2015_1.csv"
OUTPUT_DIR = "../Experiments data/Experiments/Results/BPI15_ThreeHyp_SAM2"
DECLARE_SPEC_FILE = "../Experiments data/Experiments/Results/DECspec_BPI15/phase0_declare_specification_CC.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Three-Hypothesis Storey Configuration
CONFIG = {
    # Label permutation (discriminative test H₀ᵈ) — cheap
    # Also fills null_delta_matrix (B₁ × m) for SAM FDR estimation at no extra cost
    'B_label': 4000,       # B₁: label permutation resamples

    # Trace-activity permutation (structural test H₀ˢ) — expensive
    'B_trace': 2000,        # B₂: trace permutation resamples

    # ── EMPIRICAL CALIBRATION (NEW) ───────────────────────────────────────
    # B_null double-null replicates are run inside execute_three_hypothesis_protocol
    # to calibrate T_F empirically per pattern, replacing the analytic chi2_4 lookup.
    # Each replicate applies sigma_trace o sigma_label (both axes nullified),
    # recomputes holds, runs label perm (B1_null) and structural perm (B2_null),
    # then records T_F^(b)(i) for every pattern i.
    # Phipson-Smyth: p_tilde_F(i) = (1 + #{b: T_F^(b)(i) >= T_F_obs(i)}) / (B_null + 1)
    # Resolution: 1/(B_null+1) ≈ 0.0099 << alpha=0.05. Adequate for q-value step.
    # Wall-time estimate: B_null × (B1_null + 2×B2_null) / n_cores ≈ 20 min on 12 cores.
    'B_null':  200,     # number of double-null replicates for empirical calibration
    'B1_null': 100,     # label perm budget per null replicate (reduced vs B_label)
    'B2_null':  100,     # trace perm budget per null replicate per class (reduced vs B_trace)
    # ─────────────────────────────────────────────────────────────────────

    # FDR control — SAM is the primary method; BH is retained for comparison
    'fdr_alpha': 0.05,      # Target FDR level α (used by both SAM and BH)
    'fdr_method': 'BH',     # BH reference method: 'BH' or 'BY'

    # Pattern mining parameters
    'min_activity_frequency': 1,
    'max_patterns': None,

    # Random seed
    'random_state': 42,

    # Constraint types
    'constraint_types': [
        'Init', 'End',
        'Response', 'AlternateResponse', 'ChainResponse',
        'Succession', 'AlternateSuccession', 'ChainSuccession',
        'NotResponse',
        'NotChainSuccession',
    ],
}

# ============================================================================
# DECLARE CONSTRAINT TYPE SETS
# ============================================================================

UNARY_CONSTRAINTS = ['Init', 'End']

BINARY_POSITIVE_CONSTRAINTS = [
    'Response', 'AlternateResponse', 'ChainResponse',
    'Succession', 'AlternateSuccession', 'ChainSuccession',
]

BINARY_NEGATIVE_CONSTRAINTS = [
    'NotResponse',
    'NotChainSuccession',
]

ALL_CONSTRAINT_TYPES = (
    UNARY_CONSTRAINTS + BINARY_POSITIVE_CONSTRAINTS + BINARY_NEGATIVE_CONSTRAINTS
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CaseInfo:
    """Information about a single BPI Challenge 2015 (municipality permit) case."""
    case_id: str
    outcome: int                   # 1 = Not-Granted (Deviant), 0 = Granted (Normal)
    trace: List[str]               # Full activity sequence (complete events only); no signal stripping
    last_phase: str                # case:last_phase — final phase reached (case-level)
    term_name: str                 # case:termName  — building/project term category (case-level)
    start_timestamp: datetime
    activity_index: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class PatternTestResult:
    """
    Complete three-hypothesis test result for a single DECLARE pattern.

    Contains p-values and statistics from all three tests plus independently
    FDR-controlled verdicts for both the structural and discriminative axes.

    Significance taxonomy (four categories):
        Both               — q_structural_dominant ≤ α  AND  q_value_sam ≤ α
                             Genuine temporal regularity AND class-discriminative.
        Structural only    — q_structural_dominant ≤ α  AND  q_value_sam > α
                             Real temporal pattern; class-agnostic.
        Discriminative only— q_structural_dominant > α  AND  q_value_sam ≤ α
                             Class-specific; may be a frequency artifact.
        Neither            — Both q-values > α.

    FDR control:
        Discriminative: Storey (2002) q-values on m' Fisher conjunction
                        p-values (Step 5b).  Sole significance gate.
        Structural:     Storey (2002) q-values on m' (structural_idx) p-values
                        (Step 5c) stored for transparency only — not a gate.
        is_significant_final   = is_significant_discriminative  (q_Fisher ≤ α)
        is_significant_structural = p_structural_dominant ≤ α  (raw nominal label)
    """
    pattern_id: str
    constraint_type: str
    activity_a: str
    activity_b: Optional[str]

    # Observed statistics
    prevalence_class0: float          # P̂₀(p) on observed D₀
    prevalence_class1: float          # P̂₁(p) on observed D₁
    n_applicable_class0: int          # cases where pattern was evaluated in D₀
    n_applicable_class1: int          # cases where pattern was evaluated in D₁
    n_satisfied_class0: int
    n_satisfied_class1: int
    delta_obs: float                  # P̂₁(p) − P̂₀(p), observed discriminative statistic

    # H₀ˢ — Structural null (trace-activity permutation)
    p_structural_class0: float        # TEST half p-value for structural test in class 0 (enters Fisher)
    p_structural_class1: float        # TEST half p-value for structural test in class 1 (enters Fisher)
    p_structural_dominant: float      # TEST half p-value for structural test in dominant class
    null_mean_class0: float           # E_H₀ˢ[P̂₀(p)]
    null_mean_class1: float           # E_H₀ˢ[P̂₁(p)]
    null_std_class0: float
    null_std_class1: float

    # H₀ᵈ — Discriminative null (label permutation)
    p_discriminative: float           # two-sided p-value for |Δ| test
    p_discriminative_onesided: float  # one-sided p-value (class 1 > class 0)
    null_delta_mean: float            # E_H₀ᵈ[Δ]
    null_delta_std: float             # Std_H₀ᵈ[Δ]

    # # H₀ᶜ — Fisher conjunction
    # p_conjunction: float              # chi2.sf(−2(ln p_struct_dom + ln p_disc), df=4)
    # H₀ᶜ — IUT conjunction (was: Fisher conjunction)
    p_conjunction: float   # analytic chi2_4 Fisher p-value (KEPT for BH reference, backward compat)
                        # Formerly: max(p_struct_dom, p_disc) — IUT, Berger (1982)
                        # chi2.sf(−2(ln p_s + ln p_d), df=4)

    # BH-FDR result
    is_significant_bh: bool           # Rejected by BH at FDR level α
    bh_rank: Optional[int]            # Rank in BH ordering (None if not significant)
    bh_threshold: Optional[float]     # BH critical value k·α/m for this pattern

    # Direction and dominance
    dominant_class: int               # Class with higher prevalence (0 or 1)
    direction: str                    # "Positive" (class 1 dominant) or "Negative" (class 0)

    # H₀ˢ — Structural null: SCREEN p-values (sample-split, for scope filter only)
    # Independent of test p-values by construction (disjoint permutation indices).
    p_structural_screen_class0: float = 1.0
    p_structural_screen_class1: float = 1.0

    # NEW — Phipson-Smyth empirical calibration of T_F under the double-null.
    # p_tilde_F(i) = (1 + #{b: T_F^(b)(i) >= T_F_obs(i)}) / (B_null + 1)
    # Replaces p_conjunction as the input to the Storey q-value gate (Step 5b).
    # p_conjunction (analytic) is still used for the BH reference (Step 5a).
    p_conjunction_empirical: float = 1.0

    # Storey FDR — discriminative axis (Step 5b, filled after Fisher-Storey call)
    q_value_sam:               float = 1.0          # Storey q-value on Fisher conjunction p-values
    is_significant_sam:        bool  = False        # q_value_sam ≤ α  (used internally)
    is_significant_discriminative: bool = False     # q_value_sam ≤ α  (public alias, set in Step 5c)
    fdp_estimate:              float = 1.0          # FDP̂ at τ* for this pattern's rank
    tau_star_sam:              float = float('inf') # p-value threshold τ* from Fisher-Storey

    # Storey FDR — structural axis (Step 5c, transparency only — not a gate)
    q_structural_class0:       float = 1.0   # Storey q-value for H₀ˢ in class 0 (stored for JSON)
    q_structural_class1:       float = 1.0   # Storey q-value for H₀ˢ in class 1 (stored for JSON)
    q_structural_dominant:     float = 1.0   # Storey q-value, dominant class (stored for JSON)
    is_significant_structural: bool  = False  # p_structural_dominant ≤ α (raw nominal — taxonomy label only)

    # Final verdict (Step 5c)
    is_significant_final:      bool  = False  # is_significant_discriminative (q_Fisher ≤ α — sole gate)
    significance_category:     str   = "Neither"  # "Both" | "Structural only" | "Discriminative only" | "Neither"

    # Subgroup applicability
    applicable_subgroups: List[str] = field(default_factory=list)
    subgroup_to_cases: Dict[str, List[str]] = field(default_factory=dict)


# ============================================================================
# OPTIMIZED ACTIVITY INDEXING
# ============================================================================

def precompute_activity_index(
    trace: List[str], case_id: Optional[str] = None
) -> Dict[str, List[int]]:
    """
    Pre-index activity positions for O(1) lookup.

    Returns:
        dict: {activity: [pos0, pos1, ...]} where positions are within this trace.
    """
    if trace is None:
        raise ValueError(f"Cannot compute activity index: trace is None (case_id={case_id})")
    index: Dict[str, List[int]] = {}
    for i, act in enumerate(trace):
        if act not in index:
            index[act] = []
        index[act].append(i)
    return index


# ============================================================================
# DECLARE CONSTRAINT CHECKERS
# ============================================================================
# All check_* functions operate on a SINGLE case's trace and its
# corresponding activity_index.  Returns: 1 (satisfied), 0 (violated),
# None (not applicable / vacuous).
# ============================================================================

def check_init_fast(activity_index, trace, activity, **kw) -> Optional[int]:
    if len(trace) == 0:
        return None
    return 1 if trace[0] == activity else 0

def check_end_fast(activity_index, trace, activity, **kw) -> Optional[int]:
    if len(trace) == 0:
        return None
    return 1 if trace[-1] == activity else 0

def check_Response_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx:
        return None
    yp = set(idx.get(y, []))
    for xp in idx[x]:
        if not any(j > xp for j in yp):
            return 0
    return 1

def check_AlternateResponse_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx:
        return None
    xpos = sorted(idx[x])
    yp = set(idx.get(y, []))
    for i, xp in enumerate(xpos):
        nx = xpos[i + 1] if i + 1 < len(xpos) else None
        if nx is None:
            continue
        if not any(xp < j < nx for j in yp):
            return 0
    return 1

def check_ChainResponse_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx:
        return None
    for xp in idx[x]:
        if not (xp + 1 < len(trace) and trace[xp + 1] == y):
            return 0
    return 1

def check_Precedence_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if y not in idx:
        return None
    xp = set(idx.get(x, []))
    for yp in idx[y]:
        if not any(j < yp for j in xp):
            return 0
    return 1

def check_AlternatePrecedence_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if y not in idx:
        return None
    xpos = sorted(idx.get(x, []))
    ypos = sorted(idx[y])
    for i, yp in enumerate(ypos):
        if i == 0:
            continue
        lower = ypos[i - 1] + 1
        if not any(lower <= j < yp for j in xpos):
            return 0
    return 1

def check_ChainPrecedence_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if y not in idx:
        return None
    for yp in idx[y]:
        if not (yp > 0 and trace[yp - 1] == x):
            return 0
    return 1

def check_NotResponse_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    yp = set(idx[y])
    for xp in idx[x]:
        if any(j > xp for j in yp):
            return 0
    return 1

def check_NotChainResponse_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    for xp in idx[x]:
        if xp + 1 < len(trace) and trace[xp + 1] == y:
            return 0
    return 1

def check_NotPrecedence_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if y not in idx:
        return None
    xp = set(idx.get(x, []))
    for yp in idx[y]:
        if any(j < yp for j in xp):
            return 0
    return 1

def check_NotChainPrecedence_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    for yp in idx[y]:
        if yp > 0 and trace[yp - 1] == x:
            return 0
    return 1

# --- Succession constraints (dual activation: x ∨ y) ---

def check_Succession_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    if x in idx:
        yp = set(idx.get(y, []))
        for xp in idx[x]:
            if not any(j > xp for j in yp):
                return 0
    if y in idx:
        xp = set(idx.get(x, []))
        for yp in idx[y]:
            if not any(j < yp for j in xp):
                return 0
    return 1

def check_AlternateSuccession_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    if x in idx:
        xpos = sorted(idx[x])
        yp = set(idx.get(y, []))
        for i, xp in enumerate(xpos):
            nx = xpos[i + 1] if i + 1 < len(xpos) else None
            if nx is None:
                continue
            if not any(xp < j < nx for j in yp):
                return 0
    if y in idx:
        ypos = sorted(idx[y])
        xpos_set = sorted(idx.get(x, []))
        for i, yp in enumerate(ypos):
            if i == 0:
                continue
            lower = ypos[i - 1] + 1
            if not any(lower <= j < yp for j in xpos_set):
                return 0
    return 1

def check_ChainSuccession_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    if x in idx:
        for xp in idx[x]:
            if not (xp + 1 < len(trace) and trace[xp + 1] == y):
                return 0
    if y in idx:
        for yp in idx[y]:
            if not (yp > 0 and trace[yp - 1] == x):
                return 0
    return 1

def check_NotSuccession_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    if x in idx:
        yp = set(idx.get(y, []))
        for xp in idx[x]:
            if any(j > xp for j in yp):
                return 0
    if y in idx:
        xp = set(idx.get(x, []))
        for yp in idx[y]:
            if any(j < yp for j in xp):
                return 0
    return 1

def check_NotChainSuccession_trace(idx, trace, x, y, **kw) -> Optional[int]:
    if x not in idx or y not in idx:
        return None
    if x in idx:
        for xp in idx[x]:
            if xp + 1 < len(trace) and trace[xp + 1] == y:
                return 0
    if y in idx:
        for yp in idx[y]:
            if yp > 0 and trace[yp - 1] == x:
                return 0
    return 1

def evaluate_pattern_fast(
    constraint_type: str,
    activity_a: str,
    activity_b: Optional[str],
    trace: List[str],
    activity_index: Dict[str, List[int]],
    **kw,
) -> Optional[int]:
    """Dispatch to specific constraint checker. Returns 1/0/None."""
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
    if fn is None:
        return None
    if activity_b is None and constraint_type not in ('Init', 'End'):
        return None
    try:
        return fn()
    except Exception:
        return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_preprocess_data(filepath: str) -> Dict[str, CaseInfo]:
    """Load BPI Challenge 2015 event log and extract case information.

    Column mapping:
        case:concept:name    → case identifier (integer cast to str)
        time:timestamp       → event timestamp
        concept:name         → activity name (fine-grained municipality codes)
        lifecycle:transition → event lifecycle state; only 'complete' events are retained
        case:caseStatus      → outcome attribute: 'G' = Granted, all others = Not-Granted
        case:last_phase      → final process phase the case reached (case-level)
        case:termName        → building/project term category (case-level)

    Outcome determination (binary):
        Class 1 (Not-Granted / Deviant) — case:caseStatus ≠ 'G'.
            Canonical label: Teinemaa et al. (TKDE 2019) — BPI 2015 binary outcome.
            Covers outcomes O (objection) and T (other terminations), both representing
            failure to obtain a building permit under the submitted application.
        Class 0 (Granted / Normal)      — case:caseStatus == 'G'.
        Skipped                         — missing case ID or empty trace after filtering.

    No outcome-signal stripping: the outcome is encoded in the case attribute
    case:caseStatus, not as an activity in the trace.  All activities in the
    trace are genuine process steps and must be retained.

    Lifecycle filter: the BPI 2015 log records both 'start' and 'complete' lifecycle
    events.  Only 'complete' rows are retained before building traces; retaining
    starts would duplicate every activity, inflating trace lengths and distorting
    all pattern prevalence estimates.

    Case-level attributes extracted (first non-null value per case):
        case:last_phase  → last_phase  (e.g., '01_HOOFD', '02_BEZWAAR', etc.)
        case:termName    → term_name   (building project category)
    """
    print("\n" + "=" * 100)
    print("THREE-HYPOTHESIS DISCRIMINATIVE SPECIFICATION MINING — PHASE 1")
    print("Storey CONJUNCTION TEST (H₀ˢ ∧ H₀ᵈ) — BPI CHALLENGE 2015 DATASET")
    print("=" * 100)
    print("\n📊 STEP 0: DATA LOADING & PREPROCESSING")
    print("=" * 100)

    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna(subset=['case:concept:name'])
    print(f"\n✓ Loaded {len(df):,} raw events from {df['case:concept:name'].nunique():,} cases")

    # ── Lifecycle filter: keep only 'complete' events ────────────────────────
    # The BPI 2015 log contains both start and complete rows.  Keeping starts would
    # duplicate each activity in every trace, inflating pattern satisfaction counts.
    if 'lifecycle:transition' in df.columns:
        before = len(df)
        df = df[df['lifecycle:transition'].str.lower() == 'complete'].copy()
        print(f"   Lifecycle filter ('complete' only): {before:,} → {len(df):,} events "
              f"({before - len(df):,} non-complete rows dropped)")
    else:
        print("   ⚠️  'lifecycle:transition' column not found — no lifecycle filter applied")

    act_counts = df['concept:name'].value_counts()
    print(f"\n   Activity vocabulary ({len(act_counts)} activities):")
    for act, cnt in act_counts.items():
        print(f"     {act:<55s}: {cnt:,}")

    case_data = {}
    skipped = {'empty_trace': 0}

    for case_id, group in df.groupby('case:concept:name'):
        case_events = group.sort_values('time:timestamp')
        trace = case_events['concept:name'].tolist()

        if len(trace) == 0:
            skipped['empty_trace'] += 1
            continue

        # Outcome: 1 if permit was NOT granted (caseStatus ≠ 'G'), 0 if granted
        # case:caseStatus is constant within a case; read from the first row
        status_raw = case_events['case:caseStatus'].iloc[0]
        outcome = 0 if str(status_raw).strip().upper() == 'G' else 1

        activity_index = precompute_activity_index(trace, case_id=str(case_id))

        # case:last_phase — final phase reached, constant within case
        lp_raw = case_events['case:last_phase'].iloc[0]
        last_phase = str(lp_raw).strip() if not pd.isna(lp_raw) else 'UNKNOWN'

        # case:termName — building term category; first non-null within case
        tn_series = case_events['case:termName'].dropna()
        term_name = str(tn_series.iloc[0]).strip() if len(tn_series) > 0 else 'UNKNOWN'

        case_data[str(case_id)] = CaseInfo(
            case_id=str(case_id),
            outcome=outcome,
            trace=trace,
            last_phase=last_phase,
            term_name=term_name,
            start_timestamp=pd.to_datetime(case_events['time:timestamp'].iloc[0]),
            activity_index=activity_index,
        )

    print(f"\n✓ Processed {len(case_data):,} cases")
    for reason, count in skipped.items():
        if count:
            print(f"   ⚠️  Skipped ({reason}): {count:,}")

    n1 = sum(1 for c in case_data.values() if c.outcome == 1)
    n0 = len(case_data) - n1
    print(f"\n📊 Outcome Distribution:")
    print(f"   Class 1 (Not-Granted / Deviant): {n1:,} ({n1/len(case_data)*100:.1f}%)")
    print(f"   Class 0 (Granted / Normal):      {n0:,} ({n0/len(case_data)*100:.1f}%)")
    ir = max(n0, n1) / min(n0, n1) if min(n0, n1) > 0 else float('inf')
    print(f"   Imbalance ratio (maj/min): {ir:.3f}")

    return case_data


# ============================================================================
# CLASS-CONDITIONAL LOG SPLITTING
# ============================================================================

def split_by_class(
    case_data: Dict[str, CaseInfo]
) -> Tuple[Dict[str, CaseInfo], Dict[str, CaseInfo]]:
    D_0 = {cid: c for cid, c in case_data.items() if c.outcome == 0}
    D_1 = {cid: c for cid, c in case_data.items() if c.outcome == 1}
    return D_0, D_1


# ============================================================================
# STEP 2 — PRE-COMPUTE HOLDS-BY-CASE ON OBSERVED LOG
# ============================================================================
# This is the single most important data structure.  It is computed ONCE on
# the observed log and reused by both permutation strategies.
# holds_matrix[pattern_spec] = {case_id: 0 or 1}   (None-result excluded)
# ============================================================================

def compute_holds_by_case_batch(
    cases: Dict[str, CaseInfo],
    candidates: List[Tuple[str, str, Optional[str]]],
) -> Dict[Tuple, Dict[str, int]]:
    """
    For every pattern, record which cases it holds on (non-vacuous only).

    Returns:
        Dict[pattern_spec → Dict[case_id → int]]
    """
    results: Dict[Tuple, Dict[str, int]] = {}
    case_list = list(cases.values())

    print(f"   Computing holds-by-case for {len(candidates):,} patterns "
          f"on {len(case_list):,} cases...")

    for ct, a, b in tqdm(candidates, desc="Holds-by-case"):
        holds: Dict[str, int] = {}
        for case in case_list:
            result = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
            if result is not None:
                holds[case.case_id] = result
        results[(ct, a, b)] = holds

    return results


def compute_prevalence_from_holds(
    holds: Dict[str, int],
    case_ids: Set[str],
) -> Tuple[float, int, int]:
    """
    Compute prevalence of a pattern within a subset of cases.

    Args:
        holds:    Dict[case_id → 0/1] from holds_by_case (non-vacuous only)
        case_ids: Set of case_ids forming the subset (e.g., all class-0 cases)

    Returns:
        (prevalence, n_satisfied, n_applicable)
        where n_applicable = cases in the intersection of holds and case_ids.
    """
    n_sat = 0
    n_app = 0
    for cid in case_ids:
        if cid in holds:
            n_app += 1
            if holds[cid] == 1:
                n_sat += 1
    prev = n_sat / n_app if n_app > 0 else 0.0
    return prev, n_sat, n_app


# ============================================================================
# SUBGROUP EXTRACTION
# ============================================================================

def extract_subgroups_from_case_data(
    case_data: Dict[str, CaseInfo],
) -> Tuple[Dict[str, List[str]], Dict[str, set]]:
    case_to_subgroups: Dict[str, List[str]] = {}
    subgroup_to_cases: Dict[str, set] = {}
    for cid, case in case_data.items():
        # BPI 2015 subgroups: LastPhase × TermName
        # Dimension 1: case:last_phase — final phase reached by the permit application.
        #              This partitions cases along the procedural depth axis: cases that
        #              reached later phases (e.g., objection/appeal) have structurally
        #              different activity sequences from first-phase closures.
        # Dimension 2: case:termName — building project term/category.
        #              Different project types (e.g., regular vs. extended procedure) follow
        #              distinct regulatory workflows, making term name a meaningful
        #              process partition for DECLARE pattern stratification.
        _sentinel = ('UNKNOWN', '', 'nan', 'None', 'none', 'NaN')
        sgs = []
        if case.last_phase not in _sentinel:
            sg_lp = f"LastPhase_{case.last_phase.replace(' ', '_').replace('/', '_')}"
            sgs.append(sg_lp)
            subgroup_to_cases.setdefault(sg_lp, set()).add(cid)
        if case.term_name not in _sentinel:
            sg_tn = f"TermName_{case.term_name.replace(' ', '_').replace('/', '_')}"
            sgs.append(sg_tn)
            subgroup_to_cases.setdefault(sg_tn, set()).add(cid)
        if sgs:
            case_to_subgroups[cid] = sgs
    return case_to_subgroups, subgroup_to_cases


def determine_applicable_subgroups_with_cases(
    holds_by_case: Dict[str, int],
    case_to_subgroups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    sg_map: Dict[str, List[str]] = {}
    for cid in holds_by_case:
        if cid in case_to_subgroups:
            for sg in case_to_subgroups[cid]:
                sg_map.setdefault(sg, []).append(cid)
    return {sg: sorted(cids) for sg, cids in sg_map.items()}


# ============================================================================
# STEP 3 — LABEL PERMUTATION TEST (H₀ᵈ — Discriminative Null)
# ============================================================================
# Complexity per resample: O(m · n)  [no trace permutation needed]
# Total:                   O(B₁ · m · n)
#
# For each resample b:
#   1. Shuffle the label vector: O(n)
#   2. For each pattern p: recompute Δ_b = P̂₁^(b) − P̂₀^(b)
#      using the pre-computed holds_by_case: O(n) per pattern
#   3. Accumulate |Δ_b| ≥ |Δ_obs| counts for Phipson-Smyth p-values
#   4. Store delta_b into null_delta_matrix[b, :] for SAM FDR estimation
#      (one float32 row per resample — no extra permutation passes needed)
# ============================================================================

def run_label_permutation_test(
    case_data: Dict[str, CaseInfo],
    candidates: List[Tuple[str, str, Optional[str]]],
    holds_all: Dict[Tuple, Dict[str, int]],
    B1: int,
    random_state: int,
) -> Dict[Tuple, Dict]:
    """
    Label permutation test for H₀ᵈ (discriminative null).

    For each pattern, tests whether the observed prevalence difference
    Δ_obs = P̂₁(p) − P̂₀(p) is significantly different from zero.

    Uses Phipson & Smyth (2010) exact permutation p-value:
        p = (1 + #{|Δ_b| ≥ |Δ_obs|}) / (B₁ + 1)
    This is exact and yields stochastically uniform p-values under H₀ᵈ.

    Additionally builds null_delta_matrix (B₁ × m, float32), which stores all
    permuted Δ_b vectors. This matrix is passed to sam_permutation_fdr() in
    step 5b to compute the SAM empirical FDR estimate without any additional
    permutation passes.

    Memory: null_delta_matrix at float32 is B₁ × m × 4 bytes
            (e.g. 10,000 × 20,000 × 4 ≈ 800 MB for the full pattern space).

    Returns:
        Dict with one entry per pattern_spec:
            'p_two_sided':      float  — Phipson-Smyth two-sided p-value
            'p_one_sided':      float  — one-sided p-value (class 1 > class 0)
            'delta_obs':        float  — observed Δ = P̂₁(p) − P̂₀(p)
            'null_delta_mean':  float  — E_H₀ᵈ[Δ] over B₁ resamples
            'null_delta_std':   float  — Std_H₀ᵈ[Δ] over B₁ resamples
        Plus a special key:
            '__null_delta_matrix__': np.ndarray, shape (B₁, m), float32
    """
    print(f"\n{'='*100}")
    print("📊 STEP 3: LABEL PERMUTATION TEST (H₀ᵈ — Discriminative Null)")
    print(f"{'='*100}")
    print(f"   B₁ = {B1:,} label resamples")
    print(f"   m  = {len(candidates):,} candidate patterns")
    print(f"   n  = {len(case_data):,} total cases")
    print(f"   Complexity: O(B₁ · m · n) = O({B1} · {len(candidates):,} · {len(case_data):,})")

    # Build ordered arrays for fast label shuffling
    case_ids_ordered = sorted(case_data.keys())
    labels_original = np.array([case_data[cid].outcome for cid in case_ids_ordered])
    n_total = len(case_ids_ordered)
    n1 = int(labels_original.sum())
    n0 = n_total - n1

    print(f"   n₀ = {n0}, n₁ = {n1}")
    print(f"\n   Permutation preserves: full trace temporal structure, marginal class counts")
    print(f"   Permutation destroys:  association between traces and class labels")

    # Map case_id → position in ordered array
    cid_to_idx = {cid: i for i, cid in enumerate(case_ids_ordered)}

    # Pre-build holds vectors per pattern: shape (m, n_total), NaN for not-applicable
    print(f"\n   Building holds matrix...")
    m = len(candidates)
    holds_matrix = np.full((m, n_total), np.nan)
    for p_idx, pspec in enumerate(candidates):
        holds = holds_all[pspec]
        for cid, val in holds.items():
            if cid in cid_to_idx:
                holds_matrix[p_idx, cid_to_idx[cid]] = float(val)

    # Compute observed Δ for each pattern
    delta_obs = np.zeros(m)
    for p_idx in range(m):
        h = holds_matrix[p_idx]
        mask1 = (labels_original == 1) & ~np.isnan(h)
        mask0 = (labels_original == 0) & ~np.isnan(h)
        prev1 = np.nanmean(h[mask1]) if mask1.any() else 0.0
        prev0 = np.nanmean(h[mask0]) if mask0.any() else 0.0
        delta_obs[p_idx] = prev1 - prev0

    # Run label permutations
    rng = np.random.RandomState(random_state)
    count_extreme_two = np.zeros(m, dtype=int)  # #{|Δ_b| ≥ |Δ_obs|}
    count_extreme_one = np.zeros(m, dtype=int)  # #{Δ_b ≥ Δ_obs} (one-sided)
    null_delta_sum = np.zeros(m)
    null_delta_sq_sum = np.zeros(m)

    abs_delta_obs = np.abs(delta_obs)

    # Precompute validity mask and zero-filled matrix once (constant across perms)
    valid = ~np.isnan(holds_matrix)              # (m, n_total)
    holds_filled = np.where(valid, holds_matrix, 0.0)  # NaN → 0 for masked sums

    # SAM: pre-allocate full null delta matrix (B1 × m) for empirical FDR estimation
    null_delta_matrix = np.zeros((B1, m), dtype=np.float32)

    print(f"\n   Running {B1:,} label permutations...")
    for b_idx in tqdm(range(B1), desc="Label permutation"):
        shuffled_labels = rng.permutation(labels_original)

        # Vectorize over all m patterns simultaneously — eliminates B1×m Python loop
        mask1 = (shuffled_labels == 1)[None, :] & valid   # (m, n_total)
        mask0 = (shuffled_labels == 0)[None, :] & valid   # (m, n_total)
        cnt1 = mask1.sum(axis=1)                           # (m,)
        cnt0 = mask0.sum(axis=1)                           # (m,)
        prev1_b = np.where(cnt1 > 0, (holds_filled * mask1).sum(axis=1) / np.maximum(cnt1, 1), 0.0)
        prev0_b = np.where(cnt0 > 0, (holds_filled * mask0).sum(axis=1) / np.maximum(cnt0, 1), 0.0)
        delta_b = prev1_b - prev0_b                        # (m,)

        null_delta_matrix[b_idx] = delta_b.astype(np.float32)  # SAM: store each resample
        null_delta_sum     += delta_b
        null_delta_sq_sum  += delta_b ** 2
        count_extreme_two  += (np.abs(delta_b) >= abs_delta_obs).astype(int)
        count_extreme_one  += (delta_b >= delta_obs).astype(int)

    # Phipson-Smyth exact p-values
    results_disc: Dict[Tuple, Dict] = {}
    for p_idx, pspec in enumerate(candidates):
        p_two = (1 + count_extreme_two[p_idx]) / (B1 + 1)
        p_one = (1 + count_extreme_one[p_idx]) / (B1 + 1)
        null_mean = null_delta_sum[p_idx] / B1
        null_var = null_delta_sq_sum[p_idx] / B1 - null_mean ** 2
        null_std = np.sqrt(max(null_var, 0.0))

        results_disc[pspec] = {
            'p_two_sided': float(p_two),
            'p_one_sided': float(p_one),
            'delta_obs': float(delta_obs[p_idx]),
            'null_delta_mean': float(null_mean),
            'null_delta_std': float(null_std),
        }

    # SAM: attach full null delta matrix for empirical FDR estimation
    results_disc['__null_delta_matrix__'] = null_delta_matrix  # shape (B1, m)

    # Summary
    sig_005 = sum(1 for k, v in results_disc.items() if k != '__null_delta_matrix__' and v['p_two_sided'] <= 0.05)
    sig_001 = sum(1 for k, v in results_disc.items() if k != '__null_delta_matrix__' and v['p_two_sided'] <= 0.01)
    print(f"\n   ✓ Label permutation complete")
    print(f"   Patterns with p_disc ≤ 0.05: {sig_005:,}")
    print(f"   Patterns with p_disc ≤ 0.01: {sig_001:,}")

    return results_disc


# ============================================================================
# STEP 4 — TRACE-ACTIVITY PERMUTATION TEST (H₀ˢ — Structural Null)
# ============================================================================
# For each class y, permute activities within each trace, compute permuted
# prevalences, and derive per-pattern exact p-values.
# Complexity: O(B₂ · m · n_y · L_avg) per class.
# ============================================================================

@dataclass
class PermutedLog:
    """A single permuted version of a class-conditional log."""
    cases: List[Tuple[str, List[str], Dict[str, List[int]]]]


def generate_permuted_log(
    cases: Dict[str, CaseInfo], resample_idx: int, random_state: int
) -> PermutedLog:
    """Generate a complete permuted log (all traces shuffled once)."""
    permuted = []
    for cid, case in cases.items():
        # seed = random_state + resample_idx * 100000 + hash(cid) % 100000
        import hashlib

        det_hash = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 100000
        seed = random_state + resample_idx * 100000 + det_hash
        rng = np.random.RandomState(seed)
        perm_trace = case.trace.copy()
        rng.shuffle(perm_trace)
        perm_idx = precompute_activity_index(perm_trace, case_id=cid)
        permuted.append((cid, perm_trace, perm_idx))
    return PermutedLog(cases=permuted)


def run_structural_permutation_test(
    D_y: Dict[str, CaseInfo],
    candidates_y: List[Tuple[str, str, Optional[str]]],
    class_label: int,
    B2: int,
    random_state: int,
) -> Dict[Tuple, Dict]:
    """
    Trace-activity permutation test for H₀ˢ (structural null) within one class.

    For each pattern p, tests whether the observed prevalence P̂_y(p) deviates
    significantly from what would be expected under random temporal ordering.

    Phipson-Smyth exact p-value (one-sided, no centering):
        p = (1 + #{P̂_y^(b)(p) ≥ P̂_y(p)}) / (B₂ + 1)
    Under H₀ˢ, (P̂_y_obs, P̂_y^(1), ..., P̂_y^(B₂)) are exchangeable by the rank
    argument, so this is exact. Centering around mean(P̂_y^(b)) — which excludes
    P̂_y_obs — breaks exchangeability and introduces O(1/B₂) bias (Phipson & Smyth).

    Returns:
        Dict[pattern_spec → {
            'p_structural': float,
            'prevalence_obs': float,
            'null_mean': float,
            'null_std': float,
            'n_applicable': int,
            'n_satisfied': int,
        }]
    """
    class_name = "Not-Granted" if class_label == 1 else "Granted"
    n_y = len(D_y)
    m_y = len(candidates_y)

    print(f"\n{'─'*100}")
    print(f"H₀ˢ TEST — Class {class_label} ({class_name})")
    print(f"   n_y = {n_y}, m_y = {m_y:,}, B₂ = {B2}")
    print(f"{'─'*100}")

    # Compute observed prevalences
    case_ids_y = set(D_y.keys())
    case_list = list(D_y.values())

    obs_prev = np.zeros(m_y)
    obs_nsat = np.zeros(m_y, dtype=int)
    obs_napp = np.zeros(m_y, dtype=int)

    print(f"   Computing observed prevalences...")
    for p_idx, (ct, a, b) in enumerate(tqdm(candidates_y, desc="Obs prevalence")):
        nsat = 0
        napp = 0
        for case in case_list:
            r = evaluate_pattern_fast(ct, a, b, case.trace, case.activity_index)
            if r is not None:
                napp += 1
                if r == 1:
                    nsat += 1
        obs_prev[p_idx] = nsat / napp if napp > 0 else 0.0
        obs_nsat[p_idx] = nsat
        obs_napp[p_idx] = napp

    # Run trace permutations
    perm_prev_matrix = np.zeros((B2, m_y))  # B₂ × m_y

    print(f"   Running {B2} trace-activity permutations...")
    for b_idx in tqdm(range(B2), desc=f"Trace perm (class {class_label})"):
        perm_log = generate_permuted_log(D_y, b_idx, random_state)

        for p_idx, (ct, a, b) in enumerate(candidates_y):
            nsat = 0
            napp = 0
            for cid, perm_trace, perm_idx in perm_log.cases:
                r = evaluate_pattern_fast(ct, a, b, perm_trace, perm_idx)
                if r is not None:
                    napp += 1
                    if r == 1:
                        nsat += 1
            perm_prev_matrix[b_idx, p_idx] = nsat / napp if napp > 0 else 0.0

    # Null statistics (for diagnostics only — not used in the test statistic)
    null_means = perm_prev_matrix.mean(axis=0)  # (m_y,)
    null_stds = perm_prev_matrix.std(axis=0)

    # ── SAMPLE SPLITTING (Fithian & Lei 2022; Cox 1975) ──────────────────────
    # Split B2 permutations into two independent halves using disjoint indices:
    #   Screen half (even indices): for scope filtering only
    #   Test half (odd indices):    for Fisher combination only
    # Independence: disjoint permutation draws → independent p-values under H₀ˢ.
    B2_screen = B2 // 2          # even-indexed permutations
    B2_test   = B2 - B2_screen   # odd-indexed permutations

    perm_screen = perm_prev_matrix[0::2, :]   # (B2_screen, m_y)
    perm_test   = perm_prev_matrix[1::2, :]   # (B2_test, m_y)

    print(f"   Sample splitting: B2_screen={B2_screen}, B2_test={B2_test}")
    print(f"   Screen resolution: 1/{B2_screen+1} = {1/(B2_screen+1):.4e}")
    print(f"   Test resolution:   1/{B2_test+1} = {1/(B2_test+1):.4e}")

    # Phipson-Smyth p-values on each half independently
    count_upper_screen = (perm_screen >= obs_prev[None, :]).sum(axis=0)
    p_upper_screen = (1 + count_upper_screen) / (B2_screen + 1)

    count_upper_test = (perm_test >= obs_prev[None, :]).sum(axis=0)
    p_upper_test = (1 + count_upper_test) / (B2_test + 1)

    # Build mask for negative constraint types
    is_negative = np.array(
        [ct in BINARY_NEGATIVE_CONSTRAINTS for (ct, a, b) in candidates_y],
        dtype=bool
    )   # (m_y,)

    if is_negative.any():
        # Screen half
        count_lower_screen = (perm_screen <= obs_prev[None, :]).sum(axis=0)
        p_lower_screen = (1 + count_lower_screen) / (B2_screen + 1)
        p_twosided_screen = np.minimum(2.0 * np.minimum(p_upper_screen, p_lower_screen), 1.0)
        p_screen = np.where(is_negative, p_twosided_screen, p_upper_screen)

        # Test half
        count_lower_test = (perm_test <= obs_prev[None, :]).sum(axis=0)
        p_lower_test = (1 + count_lower_test) / (B2_test + 1)
        p_twosided_test = np.minimum(2.0 * np.minimum(p_upper_test, p_lower_test), 1.0)
        p_test = np.where(is_negative, p_twosided_test, p_upper_test)

        n_neg_two = int(is_negative.sum())
        print(f"   Test direction: upper-tail for {m_y - n_neg_two} positive patterns, "
            f"two-sided for {n_neg_two} negative (NotResponse/NotChainSuccession)")
    else:
        p_screen = p_upper_screen
        p_test   = p_upper_test
        print(f"   Test direction: one-sided upper-tail for all {m_y} patterns")

    # Pack results — return BOTH screen and test p-values
    results_struct: Dict[Tuple, Dict] = {}
    for p_idx, pspec in enumerate(candidates_y):
        results_struct[pspec] = {
            'p_structural_screen': float(p_screen[p_idx]),   # for scope filter only
            'p_structural_test':   float(p_test[p_idx]),     # for Fisher combination only
            'p_structural':        float(p_test[p_idx]),     # backward compat (= test)
            'prevalence_obs': float(obs_prev[p_idx]),
            'null_mean': float(null_means[p_idx]),
            'null_std': float(null_stds[p_idx]),
            'n_applicable': int(obs_napp[p_idx]),
            'n_satisfied': int(obs_nsat[p_idx]),
            'centered_deviation': float(obs_prev[p_idx] - null_means[p_idx]),
        }

    sig_screen = sum(1 for v in results_struct.values() if v['p_structural_screen'] <= 0.05)
    sig_test   = sum(1 for v in results_struct.values() if v['p_structural_test']   <= 0.05)
    print(f"\n   ✓ Structural test complete for class {class_label}")
    print(f"   Screen p ≤ 0.05: {sig_screen:,}  |  Test p ≤ 0.05: {sig_test:,}")

    return results_struct


# ============================================================================
# FISHER CONJUNCTION P-VALUE (Fisher 1932; Littell & Folks 1973)
# ============================================================================

def fisher_conjunction_pvalue(
    p_struct_dom: np.ndarray,
    p_disc: np.ndarray,
    eps: float = 1e-300,
) -> np.ndarray:
    """
    Fisher (1932) combination for H₀ᶜ = H₀ˢ ∩ H₀ᵈ.

    T_F(p) = −2 (ln p_struct_dom(p) + ln p_disc(p))  ~  χ²_4  under H₀ᶜ.

    Uniformly more powerful than the IUT max() [Littell & Folks 1973]:
    a very strong signal on one component can compensate for a moderate signal
    on the other, as long as the joint evidence is sufficient.

    Args:
        p_struct_dom: (m,) dominant-class structural p-values (Phipson-Smyth).
        p_disc:       (m,) two-sided discriminative p-values (Phipson-Smyth).
        eps:          floor applied before log to avoid log(0).

    Returns:
        (m,) Fisher combination p-values ~ U(0,1) under H₀ᶜ.

    References:
        Fisher (1932). Statistical Methods for Research Workers, §21.1.
        Littell & Folks (1973). Asymptotic optimality of Fisher's method.
            J. Am. Stat. Assoc. 68(341):193-194.
    """
    ps = np.clip(np.asarray(p_struct_dom, dtype=np.float64), eps, 1.0)
    pd = np.clip(np.asarray(p_disc,       dtype=np.float64), eps, 1.0)
    T_F = -2.0 * (np.log(ps) + np.log(pd))
    return stats.chi2.sf(T_F, df=4)

# ============================================================================
# IUT CONJUNCTION P-VALUE (Berger 1982, Technometrics)
# ============================================================================

def iut_conjunction_pvalue(
    p_struct_dom: np.ndarray,
    p_disc: np.ndarray,
) -> np.ndarray:
    """
    Berger (1982) Intersection-Union Test (IUT) p-value for H₀ᶜ = H₀ˢ ∪ H₀ᵈ.

    p_conj^IUT(p) = max(p_struct_dom(p), p_disc(p))

    Formal guarantee (Berger 1982, Theorem 1):
        sup_{θ ∈ H₀ˢ ∪ H₀ᵈ} P(p_conj ≤ α) ≤ α

    Proof: if H₀ˢ holds,
        P(max(p_s, p_d) ≤ α | H₀ˢ) ≤ P(p_s ≤ α | H₀ˢ) ≤ α.
    Symmetrically for H₀ᵈ. The sup over the union is ≤ α unconditionally.
    No independence between p_struct_dom and p_disc is required.

    Note: Fisher's method tests the JOINT null H₀ˢ ∩ H₀ᵈ (both hold simultaneously)
    and can reject H₀ˢ ∩ H₀ᵈ purely because p_disc is small — contaminating
    the 'Both' category with 'Discriminative only' patterns under H₀ˢ.
    IUT enforces that BOTH component tests must independently reject.

    Args:
        p_struct_dom: (m,) dominant-class structural Phipson-Smyth p-values.
        p_disc:       (m,) two-sided discriminative Phipson-Smyth p-values.

    Returns:
        (m,) IUT conjunction p-values, stochastically ≤ U(0,1) under H₀ᶜ.

    References:
        Berger (1982). Multiparameter Hypothesis Testing and Acceptance Sampling.
            Technometrics 24(4):295-300.
        Casella & Berger (2002). Statistical Inference, Theorem 8.3.23.
    """
    ps = np.asarray(p_struct_dom, dtype=np.float64)
    pd = np.asarray(p_disc,       dtype=np.float64)
    return np.maximum(ps, pd)          # elementwise max — no clipping needed


# ============================================================================
# STEP 5a — BH-FDR PROCEDURE ON FISHER CONJUNCTION P-VALUES (reference)
# ============================================================================
# Retained as a secondary significance criterion for paper comparison.
# The primary criterion is Storey Q-Value FDR on p_Fisher in step 5b.
# ============================================================================

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float,
    method: str = 'BH',
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Benjamini-Hochberg (1995) or Benjamini-Yekutieli (2001) step-up procedure.

    NOTE: This function is retained as a secondary/reference significance
    procedure. The primary FDR control is sam_permutation_fdr(). BH is run
    on the conjunction p-values p_conj = max(p_struct, p_disc) and its results
    are stored in PatternTestResult.is_significant_bh / bh_rank / bh_threshold
    for comparison against the SAM results in the paper.

    Args:
        p_values: Array of m p-values (here: conjunction p-values p_conj).
        alpha:    Target FDR level.
        method:   'BH' for independence/positive dependence (default),
                  'BY' for arbitrary dependence (conservative).

    Returns:
        (rejected, bh_thresholds, k_star)
        rejected:       Boolean array, True if pattern is rejected by BH.
        bh_thresholds:  Critical values k·α/m (or k·α/(m·C_m) for BY),
                        mapped back to original pattern order.
        k_star:         Largest k such that p_(k) ≤ k·α/m.

    References:
        Benjamini & Hochberg (1995). JRSS-B 57(1):289-300.
        Benjamini & Yekutieli (2001). Ann. Stat. 29(4):1165-1188.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool), np.array([]), 0

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Correction factor
    if method == 'BY':
        # Harmonic number C_m = Σ_{k=1}^{m} 1/k
        c_m = np.sum(1.0 / np.arange(1, m + 1))
    else:
        c_m = 1.0

    # BH critical values
    ranks = np.arange(1, m + 1)
    bh_critical = ranks * alpha / (m * c_m)

    # Find largest k such that p_(k) ≤ bh_critical[k]
    k_star = 0
    for k in range(m, 0, -1):
        if sorted_p[k - 1] <= bh_critical[k - 1]:
            k_star = k
            break

    # Reject all patterns with rank ≤ k_star
    rejected = np.zeros(m, dtype=bool)
    if k_star > 0:
        rejected_sorted_positions = sorted_idx[:k_star]
        rejected[rejected_sorted_positions] = True

    # Map BH thresholds back to original order
    bh_thresholds_sorted = bh_critical.copy()
    bh_thresholds_original = np.zeros(m)
    for i, orig_idx in enumerate(sorted_idx):
        bh_thresholds_original[orig_idx] = bh_thresholds_sorted[i]

    return rejected, bh_thresholds_original, k_star


# ============================================================================
# BOOTSTRAP π̂₀ ESTIMATOR (Storey, Taylor & Siegmund 2002)
# ============================================================================

# def storey_pi0_bootstrap(
#     p_vals: np.ndarray,
#     lambdas: Optional[np.ndarray] = None,
#     B: int = 100,
#     random_state: int = 42,
# ) -> Tuple[float, float]:
#     """
#     Storey-Taylor-Siegmund (2002) bootstrap π̂₀ estimator.

#     Evaluates π̂₀(λ) = mean(p > λ)/(1−λ) across a λ grid and selects the λ*
#     that minimises bootstrap MSE = Var_B[π̂₀(λ)] + (bias proxy)². This corrects
#     the upward bias of any single fixed λ when m' is small (≈100–200).

#     Does NOT change α — only reduces the overestimation of the null fraction.

#     Args:
#         p_vals:       (m,) array of per-pattern Phipson-Smyth p-values.
#         lambdas:      λ grid; defaults to np.arange(0.05, 0.90, 0.05).
#         B:            number of bootstrap resamples (100 is sufficient).
#         random_state: RNG seed for reproducibility.

#     Returns:
#         (pi0_final, lambda_star)

#     References:
#         Storey, Taylor & Siegmund (2002). Strong control, conservative point
#             estimation and simultaneous conservative consistency of false
#             discovery rates: a unified approach. JRSS-B 66(1):187-205.
#     """
#     if lambdas is None:
#         lambdas = np.arange(0.05, 0.90, 0.05)
#     m = len(p_vals)
#     rng = np.random.RandomState(random_state)

#     # π̂₀(λ) on the observed p-values for each λ
#     pi0_grid = np.array([np.mean(p_vals > lam) / (1.0 - lam) for lam in lambdas])
#     pi0_grid = np.minimum(pi0_grid, 1.0)  # conservative cap

#     # Bootstrap MSE for each λ
#     mse = np.zeros(len(lambdas))
#     for i, lam in enumerate(lambdas):
#         boot_ests = np.zeros(B)
#         for b in range(B):
#             p_boot = rng.choice(p_vals, size=m, replace=True)
#             boot_ests[b] = min(np.mean(p_boot > lam) / (1.0 - lam), 1.0)
#         # MSE = variance + squared bias (using pi0_grid[i] as the reference)
#         mse[i] = np.var(boot_ests) + (np.mean(boot_ests) - pi0_grid[-1]) ** 2

#     best_idx = int(np.argmin(mse))
#     lambda_star = float(lambdas[best_idx])
#     pi0_final = float(pi0_grid[best_idx])
#     return pi0_final, lambda_star

def storey_pi0_bootstrap(
    p_vals: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    B: int = 100,           # retained for API compatibility only, unused in closed form
    random_state: int = 42, # retained for API compatibility only, unused
) -> Tuple[float, float]:
    """
    Storey-Taylor-Siegmund (2004) bootstrap π̂₀ estimator — closed-form variant.

    Matches the canonical qvalue R package (StoreyLab/qvalue, pi0est.R):
        minpi0 = quantile(pi0_grid, 0.10)
        MSE[i] = (W[i] / (m² (1−λ[i])²)) × (1 − W[i]/m) + (π̂₀(λ[i]) − minpi0)²
        λ* = argmin MSE;  π̂₀* = π̂₀(λ*)

    FIX vs prior version:
        WRONG:   bias term used pi0_grid[-1]  (estimate at λ_max = 0.85)
        CORRECT: bias term uses quantile(pi0_grid, 0.10) — robust lower-envelope proxy
                 for true π₀, per StoreyLab/qvalue source (rdrr.io/github/StoreyLab/qvalue/src/R/pi0est.R)

    The analytical variance term (W/m²(1-λ)² × (1-W/m)) replaces the B=100
    inner bootstrap loop — exact, deterministic, and O(|λ_grid|) instead of O(B·|λ_grid|).

    References:
        Storey, Taylor & Siegmund (2004). Strong control, conservative point
            estimation, and simultaneous conservative consistency of false
            discovery rates: a unified approach. JRSS-B 66(1):187-205.
        StoreyLab/qvalue (2024). pi0est.R, lines 80-84 (closed-form bootstrap).
    """
    if lambdas is None:
        lambdas = np.arange(0.05, 0.90, 0.05)
    lambdas = np.sort(lambdas)
    m = len(p_vals)

    # π̂₀(λ) grid on observed p-values
    pi0_grid = np.array([np.mean(p_vals > lam) / (1.0 - lam) for lam in lambdas])
    pi0_grid = np.minimum(pi0_grid, 1.0)   # conservative cap

    # W[i] = #{p > λ[i]}
    W = np.array([np.sum(p_vals > lam) for lam in lambdas], dtype=np.float64)

    # FIX: bias reference = 10th percentile of pi0_grid (matches qvalue R package)
    # WRONG was: pi0_grid[-1]
    min_pi0 = float(np.quantile(pi0_grid, 0.10))

    # Closed-form MSE = analytical variance + squared bias
    variance_term = (W / (m**2 * (1.0 - lambdas)**2)) * (1.0 - W / m)
    bias_sq_term  = (pi0_grid - min_pi0)**2
    mse = variance_term + bias_sq_term

    best_idx     = int(np.argmin(mse))
    lambda_star  = float(lambdas[best_idx])
    pi0_final    = float(np.minimum(pi0_grid[best_idx], 1.0))
    return pi0_final, lambda_star

def adaptive_storey_pi0(
    pvals: np.ndarray,
    q: float = 0.05,
    lambda_max: float = 0.80,
    delta: Optional[float] = None,
    use_robust: bool = True,
    **kwargs,
) -> Tuple[float, float]:
    """
    Gao (2023) Adaptive Storey null proportion estimator.
    Replaces storey_pi0_bootstrap in Steps 5b and 5c.

    Implements Algorithm 1 + robust stopping time τ* (Eq. 7) from:
        Gao, Z. (2023). Adaptive Storey's null proportion estimator.
        arXiv:2310.06357v1 [stat.ME].

    The estimator walks a fine λ-grid starting from q upward, computes
    π̂₀(λ) at each step, and stops at the first λ where the loss
        ψ(λ) = π̂₀(λ) + V(λ)   [robust variant, Eq. 7]
    stops decreasing.  V(λ) is the Binomial plug-in variance of π̂₀(λ):
        V(λ) = π̂₀(λ)·(1 − (1−λ)·π̂₀(λ)) / (m·(1−λ))
    truncated at λ_max = 0.80.

    FDR validity: Proposition 1 (Gao 2023) — the adaptive BH combined with
    the AS estimator controls FDR at level q whenever null p-values satisfy
    conditional stochastic dominance over U[0,1] on (q, 1], which holds
    for Phipson-Smyth permutation p-values (super-uniform by construction).

    Args:
        pvals      : (m,) array of p-values (Fisher conjunction OR structural).
        q          : FDR level; lower hard bound for the λ-grid. Pass fdralpha.
        lambda_max : Upper truncation (0.80 per paper §3 simulation).
        delta      : Grid step size. None → δ = (lambda_max − q) / 50.
        use_robust : If True (default), use ψ = π̂₀ + V  (Eq. 7).
        **kwargs   : Absorbs legacy B, random_state arguments silently.

    Returns:
        (pi0_final, lambda_star) — identical shape to storey_pi0_bootstrap.
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
        V_grid = (
            pi0_grid * (1.0 - (1.0 - lambdas) * pi0_grid)
            / (m * (1.0 - lambdas))
        )
        V_grid = np.maximum(V_grid, 0.0)
        psi_grid = pi0_grid + V_grid
    else:
        psi_grid = pi0_grid.copy()

    # τ* = inf{ λ_{j+1} : ψ(λ_{j+1}) ≥ ψ(λ_j) }  truncated at lambda_max
    stop_idx = len(lambdas) - 1
    for j in range(len(lambdas) - 1):
        if psi_grid[j + 1] >= psi_grid[j]:
            stop_idx = j
            break

    lambda_star = float(lambdas[stop_idx])
    pi0_final   = float(np.minimum(pi0_grid[stop_idx], 1.0))
    return pi0_final, lambda_star

def storey_qvalue(p_vals: np.ndarray, pi0_hat: float) -> np.ndarray:
    """
    Compute Storey (2002) q-values from pre-computed p-values and a given π̂₀.

    q_(k) = min_{k'≥k} [ π̂₀ · m · p_(k') / k' ]

    Monotonicity is enforced via right-to-left running minimum. Q-values are
    returned in the ORIGINAL order of p_vals (unsorted).

    Args:
        p_vals:   (m,) array of p-values (e.g. Fisher conjunction p-values).
        pi0_hat:  estimated null fraction, typically from storey_pi0_bootstrap().

    Returns:
        (m,) array of q-values in the same order as p_vals.
    """
    m = len(p_vals)
    if m == 0:
        return np.array([])
    sort_idx = np.argsort(p_vals)
    sorted_p = p_vals[sort_idx]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    fdp_hat = pi0_hat * m * sorted_p / ranks
    q_by_rank = np.minimum.accumulate(fdp_hat[::-1])[::-1]
    q_by_rank = np.minimum(q_by_rank, 1.0)
    q_values = np.ones(m)
    q_values[sort_idx] = q_by_rank
    return q_values


# ============================================================================
# Storey ESTIMATOR (Tusher et al. 2001; Storey & Tibshirani 2003)
# ============================================================================

def sam_permutation_fdr(
    delta_obs: np.ndarray,
    null_delta_matrix: np.ndarray,
    alpha: float = 0.05,
    pi0_correction: bool = True,
    pi0_bootstrap: bool = True,
    pi0_lambda: float = 0.5,
    pi0_bootstrap_B: int = 100,
) -> dict:
    """
    Per-Pattern Storey (2002) Q-Value FDR Estimator.

    Replaces the Tusher et al. (2001) flat-null SAM procedure, which fails
    at small m' (m' ≈ 100–200) because it pools all B1×m' null d-values into
    one flat distribution. At small m', high-variance constraint types
    (Init/End: σ_null ≈ 0.3) contaminate the expected false-count estimate
    for low-variance types (NotChainSuccession: σ_null ≈ 0.006), inflating
    Ê[V(τ)] by a factor of ~80× at the best pattern's threshold and forcing
    FDP̂ > α for all k (k* = 0).

    This replacement computes per-pattern Phipson-Smyth exact p-values
    (already available from null_delta_matrix at zero extra cost), applies
    Storey's (2002) π̂₀-corrected BH procedure, and enforces monotone
    q-values via right-to-left running minimum.

    Scientific properties:
    ─────────────────────
    1. Per-pattern comparison: p_p^PS uses only column p of abs_null —
       zero cross-contamination from other patterns' null ranges.
    2. π̂₀ < 1: strictly more powerful than BH (Storey & Tibshirani 2003),
       by factor 1/π̂₀ ≈ 1.20 at π̂₀ = 0.833.
    3. Valid FDR control: p_p^PS is stochastically uniform under H₀ᵈ
       (Phipson & Smyth 2010), and π̂₀ is conservative under positive
       dependence (Storey 2002, Theorem 1), maintaining FDR ≤ α.
    4. Exact at any m' ≥ 1: no large-m assumption required.
    5. No extra permutations: all B1 label permutations are already stored
       in null_delta_matrix from Step 3.

    Args:
        delta_obs:          (m,) observed prevalence differences Δ_obs(p)
        null_delta_matrix:  (B1, m) permuted Δ_b(p) from label permutation
        alpha:              target FDR level
        pi0_correction:     if True, apply Storey π̂₀ correction (recommended)
        pi0_bootstrap:      if True (default), use bootstrap MSE-minimisation
                            (Storey-Taylor-Siegmund 2002) to estimate π̂₀ —
                            recommended for small m' (< 500). If False, fall
                            back to the fixed-λ estimator.
        pi0_lambda:         fixed-λ fallback value (used only if
                            pi0_bootstrap=False). Default 0.5.
        pi0_bootstrap_B:    number of bootstrap resamples for π̂₀ estimation.
                            100 is sufficient; increase to 200 for publication.

    Returns:
        dict with keys:
            significant     : (m,) bool — patterns with q_p ≤ α
            k_star          : int   — number of significant patterns
            tau_star        : float — p-value threshold (on p-value scale)
            fdp_at_tau_star : float — q-value at k* (= FDP̂ at rejection boundary)
            pi0_hat         : float — estimated null fraction
            fdp_hat         : (m,) — FDP̂_k sorted by p-value ascending
            q_by_rank       : (m,) — monotone q-values in sorted order
            q_values        : (m,) — q-values in ORIGINAL pattern order
            E_V             : (m,) — per-pattern π̂₀·m·p_(k)/k (diagnostic)
            sort_idx        : (m,) — argsort(p_disc_ps) ascending
            sorted_d        : (m,) — d_obs in sort_idx order (diagnostic)
            sigma_null      : (m,) — per-pattern null std (diagnostic)
            s0              : float — SAM fudge factor (diagnostic only)
            d_obs           : (m,) — standardized statistics (diagnostic)
            p_disc_ps       : (m,) — per-pattern Phipson-Smyth p-values

    References:
        Storey (2002). A direct approach to false discovery rates. JRSS-B.
        Storey & Tibshirani (2003). Statistical significance for genomewide
            studies. PNAS 100(16):9440-9445.
        Phipson & Smyth (2010). Permutation p-values should never be zero.
            Stat. Appl. Genet. Mol. Biol. 9(1):Article 39.
        Benjamini & Hochberg (1995). Controlling the FDR. JRSS-B.
    """
    m  = len(delta_obs)
    B1 = null_delta_matrix.shape[0]
    abs_obs  = np.abs(delta_obs).astype(np.float64)
    abs_null = np.abs(null_delta_matrix).astype(np.float64)

    # ── Diagnostic: standardization (kept for output compatibility) ───────
    # σ_null and d_obs are retained in output for paper diagnostics only.
    # They are NOT used in the FDR computation below.
    sigma_null = abs_null.std(axis=0)                        # (m,)
    s0 = float(np.percentile(sigma_null, 50))                # median, not 5th pct
    print(f"   σ_null: [{sigma_null.min():.4f}, {sigma_null.max():.4f}], "
          f"median={np.median(sigma_null):.4f}, s₀(median)={s0:.4f}")
    denom = sigma_null + s0
    d_obs = abs_obs / denom                                  # (m,) — diagnostic only

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Per-pattern Phipson-Smyth exact p-values
    # p_p = (1 + #{b : |Δ_b,p| ≥ |Δ_obs,p|}) / (B1 + 1)
    # Comparison is per-column → zero cross-contamination between patterns.
    # ═══════════════════════════════════════════════════════════════════════
    count_ext = (abs_null >= abs_obs[None, :]).sum(axis=0)   # (m,) column-wise
    p_disc_ps = (1 + count_ext) / (B1 + 1)                  # exact, uniform under H₀ᵈ

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Storey π̂₀ estimation
    # Primary: bootstrap MSE-minimisation (Storey-Taylor-Siegmund 2002),
    #          which corrects the upward bias of fixed λ at small m'.
    # Fallback: fixed λ estimator π̂₀ = #{p > λ} / (m·(1−λ)).
    # Both are conservative under PRDS → valid FDR control downstream.
    # ═══════════════════════════════════════════════════════════════════════
    if pi0_correction:
        if pi0_bootstrap:
            pi0_hat, lambda_star = storey_pi0_bootstrap(
                p_disc_ps,
                B=pi0_bootstrap_B,
                random_state=CONFIG['random_state'],
            )
            print(f"   π̂₀ = {pi0_hat:.4f}  (bootstrap, λ*={lambda_star:.2f}, "
                  f"B={pi0_bootstrap_B}, m={m})")
        else:
            n_above_lambda = float(np.sum(p_disc_ps > pi0_lambda))
            pi0_hat = min(n_above_lambda / (m * (1.0 - pi0_lambda)), 1.0)
            print(f"   π̂₀ = {pi0_hat:.4f}  (fixed λ={pi0_lambda}, "
                  f"n_above={int(n_above_lambda)})")
    else:
        pi0_hat = 1.0                                        # conservative fallback
        print(f"   π̂₀ = 1.0000  (correction disabled)")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — Sort p-values ascending, assign BH-style ranks
    # FDP̂_k = π̂₀ · m · p_(k) / k
    # At rank k: R(τ_k) = k (by definition of sorted order).
    # E[V(τ_k)] ≈ π̂₀ · m · p_(k) under uniform-null assumption —
    # exact when p-values are i.i.d. uniform under H₀, conservative under PRDS.
    # ═══════════════════════════════════════════════════════════════════════
    sort_idx  = np.argsort(p_disc_ps)                        # ascending p-values
    sorted_p  = p_disc_ps[sort_idx]                          # p_(1) ≤ ... ≤ p_(m)
    sorted_d  = d_obs[sort_idx]                              # for diagnostic output
    ranks     = np.arange(1, m + 1, dtype=np.float64)       # 1, 2, ..., m

    E_V       = pi0_hat * m * sorted_p                       # π̂₀ · m · p_(k) per rank
    fdp_hat   = E_V / ranks                                  # FDP̂_k = π̂₀·m·p_(k) / k

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Enforce monotonicity: right-to-left running minimum
    # q_(k) = min_{k'≥k} FDP̂_{k'}
    # Without this, a low-ranked pattern could have q < a high-ranked one,
    # which violates the step-up property. The [::-1] trick computes the
    # right-to-left minimum in one vectorized pass.
    # ═══════════════════════════════════════════════════════════════════════
    q_by_rank = np.minimum.accumulate(fdp_hat[::-1])[::-1]  # right-to-left min
    q_by_rank = np.minimum(q_by_rank, 1.0)                  # q-value ≤ 1 always

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5 — Significance decision: q_p ≤ α
    # k* = #{p : q_(k) ≤ α} by the step-up property
    # τ* is the largest p-value threshold where q ≤ α still holds.
    # ═══════════════════════════════════════════════════════════════════════
    kstar     = int(np.sum(q_by_rank <= alpha))
    tau_star  = float(sorted_p[kstar - 1]) if kstar > 0 else 0.0
    fdp_atstar = float(q_by_rank[kstar - 1]) if kstar > 0 else 1.0

    # Reject in original order: pattern p is significant iff p_disc_ps[p] ≤ τ*
    significant = (p_disc_ps <= tau_star) if kstar > 0 else np.zeros(m, dtype=bool)

    # Map q-values back to original (unsorted) pattern order
    q_values  = np.ones(m)
    q_values[sort_idx] = q_by_rank                          # invert the sort

    # ── Diagnostics ───────────────────────────────────────────────────────
    best_p_ps = float(sorted_p[0])
    best_E_V  = float(E_V[0])
    print(f"   Best p_disc_ps = {best_p_ps:.6e}  (pattern rank 1)")
    print(f"   Ê[V(best, per-pattern)] = {best_E_V:.6e}  "
          f"= {pi0_hat:.4f} × {m} × {best_p_ps:.4e}")
    print(f"   k* = {kstar}, FDP̂ = {fdp_atstar:.4f}, τ*_p = {tau_star:.6e}")

    return dict(
        significant=significant,
        k_star=kstar,
        tau_star=tau_star,
        fdp_at_tau_star=fdp_atstar,
        pi0_hat=pi0_hat,
        fdp_hat=fdp_hat,
        q_by_rank=q_by_rank,
        q_values=q_values,
        E_V=E_V,
        sort_idx=sort_idx,
        sorted_d=sorted_d,
        sigma_null=sigma_null,
        s0=s0,
        d_obs=d_obs,
        p_disc_ps=p_disc_ps,
    )


# ============================================================================
# CANDIDATE GENERATION FROM PHASE 0 CC SPECIFICATION
# ============================================================================

def generate_candidate_patterns(
    _case_data: Dict[str, CaseInfo],
) -> Tuple[List[Tuple[str, str, Optional[str]]], List[Tuple[str, str, Optional[str]]]]:
    """
    Generate class-specific candidate pools from Phase 0 CC specification.

    Returns:
        candidates_pos: patterns from Lpos (tested on D₁)
        candidates_neg: patterns from Lneg (tested on D₀)
    """
    print(f"\n{'='*100}")
    print("📊 STEP 1: CANDIDATE GENERATION FROM PHASE 0 CC SPEC")
    print(f"{'='*100}")

    with open(DECLARE_SPEC_FILE, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    lpos = spec.get('Lpos', {}).get('constraints', [])
    lneg = spec.get('Lneg', {}).get('constraints', [])

    print(f"   Raw L+ constraints: {len(lpos):,}")
    print(f"   Raw L− constraints: {len(lneg):,}")

    allowed = set(ALL_CONSTRAINT_TYPES)

    def _filter(clist, label):
        out = []
        for c in clist:
            if c['constraint_type'] in allowed:
                out.append((c['constraint_type'], c['param_a'], c.get('param_b')))
        tc = Counter(x[0] for x in out)
        print(f"\n   {label}: {len(out):,} candidates after type filter")
        for ct in ALL_CONSTRAINT_TYPES:
            if ct in tc:
                print(f"     • {ct:<30s}: {tc[ct]:,}")
        return out

    candidates_pos = _filter(lpos, "L+ (Not-Granted)")
    candidates_neg = _filter(lneg, "L− (Granted)")

    # Build union
    pos_set = set(candidates_pos)
    neg_set = set(candidates_neg)
    union = list(candidates_pos) + [p for p in candidates_neg if p not in pos_set]

    print(f"\n📊 Final pools:")
    print(f"   L+ candidates: {len(candidates_pos):,}")
    print(f"   L− candidates: {len(candidates_neg):,}")
    print(f"   Union:          {len(union):,}")
    print(f"   Shared (L+∩L−): {len(pos_set & neg_set):,}")

    return candidates_pos, candidates_neg


# ============================================================================
# EMPIRICAL CALIBRATION — DOUBLE-NULL T_F MATRIX
# ============================================================================

def compute_double_null_tf_matrix(
    case_data:       Dict[str, CaseInfo],
    candidates_all:  List[Tuple[str, str, Optional[str]]],
    case_ids_sorted: List[str],
    labels:          np.ndarray,          # (n,) original binary labels
    B_null:          int,
    B1_null:         int,
    B2_null:         int,
    alpha:           float,
    random_state:    int,
    n_jobs:          int = -1,
) -> np.ndarray:
    """
    Run B_null doubly-nullified replicates and record per-pattern T_F^(b) values.

    Protocol per replicate b
    ─────────────────────────
    1. sigma_label  — permute class labels (marginals preserved).
    2. sigma_trace  — shuffle activities within every trace.
       Together these guarantee p_s^(b) ~ U(0,1) and p_d^(b) ~ U(0,1).
    3. Recompute holds on the shuffled traces.
    4. Run label permutation test  (B1_null resamples) → p_d^(b)  per pattern.
    5. Run structural permutation  (B2_null resamples, both classes) → p_s^(b).
    6. T_F^(b)(i) = -2(ln p_s_dom^(b)(i) + ln p_d^(b)(i))   [chi2_4 under true null]

    Returns
    ───────
    tf_null_matrix : ndarray, shape (B_null, m)
        tf_null_matrix[b, i] = T_F^(b)(i) for pattern i in replicate b.
        Patterns are in the same order as candidates_all.
    """
    import copy
    from joblib import Parallel, delayed

    m = len(candidates_all)
    BASE_SEED = random_state + 500_000   # offset from production seeds

    # ── Generate B_null permuted label vectors (preserving marginals) ──
    rng_label = np.random.RandomState(BASE_SEED)
    permuted_labels_all = np.stack(
        [rng_label.permutation(labels) for _ in range(B_null)], axis=0
    )  # (B_null, n)

    def _run_one_replicate(b: int) -> np.ndarray:
        """Returns T_F^(b) of shape (m,)."""
        rs = BASE_SEED + 100_000 * b

        # ── sigma_trace: shuffle activities within each trace ──────────
        rng_trace = np.random.RandomState(rs + 200_000)
        null_case_data = {}
        for i, cid in enumerate(case_ids_sorted):
            ci_orig = case_data[cid]
            ci = copy.copy(ci_orig)                      # shallow copy
            ci.outcome = int(permuted_labels_all[b, i])  # sigma_label
            shuffled_trace = ci_orig.trace.copy()
            rng_trace.shuffle(shuffled_trace)            # sigma_trace
            ci.trace = shuffled_trace
            ci.activity_index = precompute_activity_index(shuffled_trace, case_id=cid)
            null_case_data[cid] = ci

        # ── Recompute holds on shuffled traces ─────────────────────────
        holds_null = compute_holds_by_case_batch(null_case_data, candidates_all)

        # ── H_0^d: label permutation test (B1_null resamples) ──────────
        disc_results = run_label_permutation_test(
            null_case_data, candidates_all, holds_null, B1_null, rs
        )
        disc_results.pop('__null_delta_matrix__', None)
        p_disc = np.array(
            [disc_results[spec]['p_two_sided'] for spec in candidates_all]
        )

        # ── H_0^s: structural permutation test (B2_null per class) ─────
        D0_null, D1_null = split_by_class(null_case_data)
        struct0 = run_structural_permutation_test(
            D0_null, candidates_all, class_label=0, B2=B2_null, random_state=rs + 1
        )
        struct1 = run_structural_permutation_test(
            D1_null, candidates_all, class_label=1, B2=B2_null, random_state=rs + 2
        )

        # ── Dominant class from shuffled prevalences ────────────────────
        cid_set0 = set(D0_null.keys())
        cid_set1 = set(D1_null.keys())
        p_s_test_c0 = np.array([
            struct0[spec]['p_structural_test'] if spec in struct0 else 1.0
            for spec in candidates_all
        ])
        p_s_test_c1 = np.array([
            struct1[spec]['p_structural_test'] if spec in struct1 else 1.0
            for spec in candidates_all
        ])

        prev0 = np.zeros(m)
        prev1 = np.zeros(m)
        for i, spec in enumerate(candidates_all):
            holds = holds_null[spec]
            p0, _, _ = compute_prevalence_from_holds(holds, cid_set0)
            p1, _, _ = compute_prevalence_from_holds(holds, cid_set1)
            prev0[i] = p0
            prev1[i] = p1
        dominant = np.where(prev1 >= prev0, 1, 0)
        p_s_dom = np.where(dominant == 1, p_s_test_c1, p_s_test_c0)

        # ── T_F^(b) ─────────────────────────────────────────────────────
        eps = 1e-300
        ps = np.clip(p_s_dom, eps, 1.0)
        pd = np.clip(p_disc,  eps, 1.0)
        tf_b = -2.0 * (np.log(ps) + np.log(pd))   # shape (m,)
        return tf_b

    # ── Run B_null replicates in parallel ──────────────────────────────
    print(f"\n  Running {B_null} double-null replicates for empirical T_F calibration "
          f"(B1_null={B1_null}, B2_null={B2_null}, n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, verbose=5, backend='loky')(
        delayed(_run_one_replicate)(b) for b in range(B_null)
    )
    tf_null_matrix = np.stack(results, axis=0)  # (B_null, m)
    return tf_null_matrix


def empirical_fisher_pvalue(
    tf_obs:        np.ndarray,   # (m,)  observed T_F values (from chi2_4 score)
    tf_null_matrix: np.ndarray,  # (B_null, m) null T_F values per replicate
) -> np.ndarray:
    """
    Phipson-Smyth (2010) exact permutation p-value for the Fisher combined statistic.

        p̃_F(i) = (1 + #{b : T_F^(b)(i) ≥ T_F_obs(i)}) / (B_null + 1)

    This is stochastically super-uniform under the joint null (sigma_trace ∘ sigma_label)
    by the exchangeability argument: under the double-null, T_F_obs(i) is exchangeable
    with T_F^(1)(i), ..., T_F^(B)(i), so its rank among them is uniform.

    Power preservation: the ranking of patterns by T_F_obs is IDENTICAL to the ranking
    by p̃_F (both are monotone-decreasing in T_F_obs). The Storey q-value procedure
    applied to p̃_F uses the same ordered sequence — only calibration shifts.

    Args
    ────
    tf_obs         : (m,)         observed Fisher statistics, one per pattern.
    tf_null_matrix : (B_null, m)  null Fisher statistics under double-null.

    Returns
    ───────
    p_tilde : (m,) empirically calibrated p-values, in (0, 1].
    """
    B_null, m = tf_null_matrix.shape
    # Count how many null values >= observed value (upper-tail, consistent with chi2_4)
    count_geq = (tf_null_matrix >= tf_obs[np.newaxis, :]).sum(axis=0)  # (m,)
    p_tilde = (1.0 + count_geq) / (B_null + 1.0)
    return p_tilde  # in [1/(B+1), 1.0]  — never exactly 0 (Phipson-Smyth)


# ============================================================================
# MAIN EXECUTION — THREE-HYPOTHESIS PROTOCOL
# ============================================================================

def execute_three_hypothesis_protocol(
    case_data:       Dict[str, CaseInfo],
    candidates_pos:  List[Tuple[str, str, Optional[str]]],
    candidates_neg:  List[Tuple[str, str, Optional[str]]],
    case_ids_sorted: Optional[List[str]] = None,   # NEW — needed for sigma_trace
    labels:          Optional[np.ndarray] = None,   # NEW — needed for sigma_label
) -> Tuple[List[PatternTestResult], Dict, Dict]:
    """
    Execute the complete three-hypothesis Storey protocol.

    Architecture:
    1. Build union candidate pool (L+ ∪ L−).
    2. Pre-compute holds-by-case on observed log (once, reused everywhere).
    3. Label permutation test H₀ᵈ on the union pool.
       Produces per-pattern Phipson-Smyth p-values AND null_delta_matrix
       (B₁ × m, float32) for use in step 5b.
    4. Trace-activity permutation test H₀ˢ:
       - Structural test on D₀ for all union patterns
       - Structural test on D₁ for all union patterns
    5a. Fisher conjunction p-values + BH-FDR (reference comparison):
        - T_F(p) = −2(ln p_struct_dominant(p) + ln p_disc(p)) ~ χ²_4   [Fisher 1932]
        - p_Fisher(p) = chi2.sf(T_F, df=4)
        - BH step-up on {p_Fisher(p)} at level α
    5b. Storey (2002) Q-Value FDR on p_Fisher (primary significance criterion):
        - storey_qvalue(p_Fisher) with bootstrap π̂₀ over structurally-touched m'
        - q(p) = min_{k'≥k} [π̂₀ · m' · p_Fisher_(k') / k'] ≤ α
        - Final verdict: is_significant_final = q_Fisher(p) ≤ α
          (structural evidence already embedded in p_Fisher via Fisher combination)
    """
    # If not provided, derive them (backward compatibility)
    if case_ids_sorted is None:
        case_ids_sorted = sorted(case_data.keys())
    if labels is None:
        labels = np.array([case_data[cid].outcome for cid in case_ids_sorted])

    timing = {}
    t0 = time.time()

    B1 = CONFIG['B_label']
    B2 = CONFIG['B_trace']
    alpha = CONFIG['fdr_alpha']
    fdr_method = CONFIG['fdr_method']
    rs = CONFIG['random_state']

    # ── Build union ──────────────────────────────────────────────────────
    pos_set = set(candidates_pos)
    neg_set = set(candidates_neg)
    candidates_all = list(candidates_pos)
    for p in candidates_neg:
        if p not in pos_set:
            candidates_all.append(p)
    m_total = len(candidates_all)

    print(f"\n{'='*100}")
    print("📊 THREE-HYPOTHESIS Storey PROTOCOL")
    print(f"{'='*100}")
    print(f"   m_total   = {m_total:,} union patterns")
    print(f"   B₁ (label)= {B1:,}")
    print(f"   B₂ (trace)= {B2:,}")
    print(f"   FDR α     = {alpha}")
    print(f"   BH ref.   = {fdr_method}  (for comparison)")

    # ── Split by class ───────────────────────────────────────────────────
    D_0, D_1 = split_by_class(case_data)
    print(f"   n₀ = {len(D_0):,}, n₁ = {len(D_1):,}")

    # ── Step 2: Pre-compute holds-by-case ────────────────────────────────
    print(f"\n{'='*100}")
    print("📊 STEP 2: PRE-COMPUTE HOLDS-BY-CASE ON OBSERVED LOG")
    print(f"{'='*100}")

    t_holds = time.time()
    holds_all = compute_holds_by_case_batch(case_data, candidates_all)
    timing['holds_computation'] = time.time() - t_holds

    # ── Step 3: Label permutation (H₀ᵈ) ─────────────────────────────────
    t_label = time.time()
    disc_results = run_label_permutation_test(
        case_data, candidates_all, holds_all, B1, rs
    )
    null_delta_matrix = disc_results.pop('__null_delta_matrix__')
    timing['label_permutation'] = time.time() - t_label

    # ── Step 4: Trace-activity permutation (H₀ˢ) ────────────────────────
    print(f"\n{'='*100}")
    print("📊 STEP 4: TRACE-ACTIVITY PERMUTATION (H₀ˢ — Structural Null)")
    print(f"{'='*100}")

    # Structural test on full union for both classes.
    # Running on candidates_all (not candidates_neg/pos) ensures every union pattern
    # has a structural p-value for both classes, so no pattern is forced to
    # p_conj = 1.0 or excluded from SAM simply because its dominant direction
    # switched relative to Phase 0.
    t_struct0 = time.time()
    struct_results_0 = run_structural_permutation_test(
        D_0, candidates_all, class_label=0, B2=B2, random_state=rs
    )
    timing['structural_class0'] = time.time() - t_struct0

    t_struct1 = time.time()
    struct_results_1 = run_structural_permutation_test(
        D_1, candidates_all, class_label=1, B2=B2, random_state=rs + 1
    )
    timing['structural_class1'] = time.time() - t_struct1

    # ── Subgroup extraction ──────────────────────────────────────────────
    case_to_sg, sg_to_cases = extract_subgroups_from_case_data(case_data)

    # ── Step 5a: Fisher conjunction p-values + BH-FDR (reference) ───────
    print(f"\n{'='*100}")
    print("📊 STEP 5a: FISHER CONJUNCTION P-VALUES + BH-FDR (reference)")
    print(f"{'='*100}")
    print(f"   p_Fisher(p) = chi2.sf(−2(ln p_struct_dom + ln p_disc), df=4)  [Fisher 1932]")
    print(f"   BH-{fdr_method} at α = {alpha}  (retained for comparison only)")

    cid_set_0 = set(D_0.keys())
    cid_set_1 = set(D_1.keys())

    # Assemble per-pattern results
    pattern_results: List[PatternTestResult] = []
    p_conj_values = np.ones(m_total)
    tf_obs_all    = np.zeros(m_total)   # NEW — stores T_F score before chi2 conversion

    for p_idx, pspec in enumerate(tqdm(candidates_all, desc="Assembling results")):
        ct, a, b = pspec
        pid = f"{ct}_{a}" + (f"_{b}" if b else "")

        holds = holds_all[pspec]

        # Observed prevalences
        prev0, nsat0, napp0 = compute_prevalence_from_holds(holds, cid_set_0)
        prev1, nsat1, napp1 = compute_prevalence_from_holds(holds, cid_set_1)
        delta = prev1 - prev0

        # Dominant class
        dominant = 1 if prev1 >= prev0 else 0
        direction = "Positive" if dominant == 1 else "Negative"

        # ── H₀ᵈ result ──────────────────────────────────────────────────
        d = disc_results[pspec]
        p_disc_two = d['p_two_sided']
        p_disc_one = d['p_one_sided']

        # ── H₀ˢ results (sample-split) ──────────────────────────────────
        # TEST p-values enter Fisher combination.
        # SCREEN p-values are stored separately for scope filter only.
        if pspec in struct_results_0:
            s0 = struct_results_0[pspec]
            p_s0_test   = s0['p_structural_test']
            p_s0_screen = s0['p_structural_screen']
            null_mean_0 = s0['null_mean']
            null_std_0  = s0['null_std']
        else:
            p_s0_test = 1.0; p_s0_screen = 1.0
            null_mean_0 = prev0; null_std_0 = 0.0

        if pspec in struct_results_1:
            s1 = struct_results_1[pspec]
            p_s1_test   = s1['p_structural_test']
            p_s1_screen = s1['p_structural_screen']
            null_mean_1 = s1['null_mean']
            null_std_1  = s1['null_std']
        else:
            p_s1_test = 1.0; p_s1_screen = 1.0
            null_mean_1 = prev1; null_std_1 = 0.0

        # TEST p-value of the dominant class enters Fisher combination
        p_struct_dom_test = p_s1_test if dominant == 1 else p_s0_test

        # ── H₀ᶜ — Fisher conjunction (TEST p-values only) ───────────
        # p_struct_dom_test is independent of p_disc (disjoint permutation schemes):
        #   structural test shuffles trace order; discriminative test shuffles labels.
        # Additionally, p_struct_dom_test uses only the test half of structural
        # permutations, which is disjoint from the screen half used in the scope filter.
        eps = 1e-300
        ps_clip = max(float(p_struct_dom_test), eps)
        pd_clip = max(float(p_disc_two),        eps)
        tf_obs_i = -2.0 * (np.log(ps_clip) + np.log(pd_clip))   # raw score

        # Analytic p-value — KEEP for BH reference (Step 5a) and backward-compat JSON
        from scipy.stats import chi2 as _chi2
        pconj_analytic = float(_chi2.sf(tf_obs_i, df=4))

        p_conj_values[p_idx] = pconj_analytic   # analytic, used for BH in Step 5a
        tf_obs_all[p_idx]    = tf_obs_i          # raw score, used for empirical in Step 5b

        # ── Subgroup applicability ───────────────────────────────────────
        sg_cases = determine_applicable_subgroups_with_cases(holds, case_to_sg)

        pattern_results.append(PatternTestResult(
            pattern_id=pid,
            constraint_type=ct,
            activity_a=a,
            activity_b=b,

            prevalence_class0=prev0,
            prevalence_class1=prev1,
            n_applicable_class0=napp0,
            n_applicable_class1=napp1,
            n_satisfied_class0=nsat0,
            n_satisfied_class1=nsat1,
            delta_obs=delta,

            p_structural_class0=p_s0_test,        # TEST half (enters Fisher)
            p_structural_class1=p_s1_test,        # TEST half (enters Fisher)
            p_structural_dominant=p_struct_dom_test,  # TEST half (enters Fisher)
            null_mean_class0=null_mean_0,
            null_mean_class1=null_mean_1,
            null_std_class0=null_std_0,
            null_std_class1=null_std_1,

            p_structural_screen_class0=p_s0_screen,   # SCREEN half (scope filter only)
            p_structural_screen_class1=p_s1_screen,   # SCREEN half (scope filter only)

            p_discriminative=p_disc_two,
            p_discriminative_onesided=p_disc_one,
            null_delta_mean=d['null_delta_mean'],
            null_delta_std=d['null_delta_std'],

            p_conjunction=pconj_analytic,

            is_significant_bh=False,  # will be set below
            bh_rank=None,
            bh_threshold=None,

            dominant_class=dominant,
            direction=direction,

            applicable_subgroups=sorted(sg_cases.keys()),
            subgroup_to_cases=sg_cases,
        ))

    # ── BH-FDR ───────────────────────────────────────────────────────────
    rejected, bh_thresholds, _ = benjamini_hochberg(
        p_conj_values, alpha, method=fdr_method
    )

    # Assign BH results (overwritten for structural subset by Step 5b m'-BH)
    sorted_idx = np.argsort(p_conj_values)
    for rank_pos, orig_idx in enumerate(sorted_idx):
        pattern_results[orig_idx].bh_threshold = float(bh_thresholds[orig_idx])
        if rejected[orig_idx]:
            pattern_results[orig_idx].is_significant_bh = True
            pattern_results[orig_idx].bh_rank = rank_pos + 1

    # # ── STEP 5b: Storey Q-Value FDR on Fisher conjunction p-values ───────────
    # # Fisher already combines structural and discriminative evidence, so no
    # # separate structural pre-filter is needed before FDR.  The scope filter
    # # retains only patterns that have been structurally tested in at least one
    # # class (min(p_struct_c0, p_struct_c1) < 1.0 — i.e., not a sentinel 1.0),
    # # which is every pattern in candidates_all since the structural test is run
    # # on the full union.  The filter at α keeps the pool from ballooning with
    # # patterns that have zero structural evidence (T_F collapses to χ²_2 when
    # # p_struct_dom → 1).
    # print(f"\n{'='*100}")
    # print("📊 STEP 5b: STOREY Q-VALUE FDR ON FISHER CONJUNCTION P-VALUES (PRIMARY)")
    # print(f"{'='*100}")
    # print(f"   p_Fisher(p) already encodes both H₀ˢ and H₀ᵈ evidence.")
    # print(f"   q(p) = min_{{k'≥k}} [π̂₀·m'·p_Fisher_(k')/k']  [Storey 2002, bootstrap π̂₀]")

    # structural_idx = [
    #     pidx for pidx, r in enumerate(pattern_results)
    #     if min(r.p_structural_class0, r.p_structural_class1) <= alpha
    # ]
    # m_prime = len(structural_idx)
    # print(f"   Structural scope filter: m={m_total} → m'={m_prime} "
    #       f"({m_total - m_prime} excluded by min(p_struct_c0, p_struct_c1) > {alpha})")

    # # Fisher p-values for structurally-touched patterns (already stored in p_conjunction)
    # p_fisher_filtered = np.array(
    #     [pattern_results[i].p_conjunction for i in structural_idx]
    # )

    # # Adaptive Storey π̂₀ (Gao 2023) on Fisher p-values, then Storey q-values
    # t_sam = time.time()
    # pi0_fisher, lambda_star_fisher = adaptive_storey_pi0(
    #     p_fisher_filtered, q=alpha,
    # )
    # q_values_fisher = storey_qvalue(p_fisher_filtered, pi0_fisher)
    # timing['sam_fdr'] = time.time() - t_sam

    # # τ* = largest Fisher p-value threshold where q ≤ α
    # sort_idx_f = np.argsort(p_fisher_filtered)
    # q_sorted_f = q_values_fisher[sort_idx_f]
    # p_sorted_f = p_fisher_filtered[sort_idx_f]
    # kstar_fisher  = int(np.sum(q_values_fisher <= alpha))
    # tau_star_f    = float(p_sorted_f[kstar_fisher - 1]) if kstar_fisher > 0 else 0.0
    # fdp_at_tau_f  = float(q_sorted_f[kstar_fisher - 1]) if kstar_fisher > 0 else 1.0

    # print(f"\n   Fisher-Storey results (on m'={m_prime} structurally-touched patterns):")
    # print(f"     π̂₀  = {pi0_fisher:.4f}  (AS Gao 2023, λ*={lambda_star_fisher:.2f})")
    # print(f"     k*  = {kstar_fisher:,}  (Fisher-Storey significant patterns)")
    # print(f"     τ*_p = {tau_star_f:.6e}  (threshold on Fisher p-value scale)")
    # print(f"     FDP̂ = {fdp_at_tau_f:.4f}  (at τ*)")

    # # Map back to original indices (discriminative axis only — final verdict set in Step 5c)
    # for sam_i, orig_i in enumerate(structural_idx):
    #     r = pattern_results[orig_i]
    #     r.q_value_sam                  = float(q_values_fisher[sam_i])
    #     r.is_significant_sam           = bool(q_values_fisher[sam_i] <= alpha)
    #     r.is_significant_discriminative = r.is_significant_sam
    #     r.fdp_estimate                 = fdp_at_tau_f
    #     r.tau_star_sam                 = tau_star_f
    #     # is_significant_final and significance_category are set in Step 5c

    # # BH reference — on m' Fisher p-values (fair comparison against primary)
    # rejected_bh_f, bh_thresh_f, kstar_bh_f = benjamini_hochberg(
    #     p_fisher_filtered, alpha, method=fdr_method
    # )
    # for bh_i, orig_i in enumerate(structural_idx):
    #     if rejected_bh_f[bh_i]:
    #         pattern_results[orig_i].is_significant_bh = True
    #         pattern_results[orig_i].bh_rank = int(np.sum(p_fisher_filtered <=
    #                                                p_fisher_filtered[bh_i]))
    #         pattern_results[orig_i].bh_threshold = float(bh_thresh_f[bh_i])

    # n_bh_f = int(np.sum(rejected_bh_f))
    # print(f"     BH on m':      {n_bh_f:,}  (reference, m'={m_prime})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5b_pre: EMPIRICAL CALIBRATION OF T_F (NEW)
    # ══════════════════════════════════════════════════════════════════════
    # We now replace p_conj_values (analytic chi2_4) with Phipson-Smyth empirical
    # p-values computed from B_null double-null replicates.
    # The scope filter still uses screen p-values (unchanged — no selection bias).
    # Only the Storey gate input changes: analytic → empirical.
    # ──────────────────────────────────────────────────────────────────────
    B_null  = CONFIG.get('B_null',  100)
    B1_null = CONFIG.get('B1_null', 200)
    B2_null = CONFIG.get('B2_null',  50)

    t_calib = time.time()
    tf_null_matrix = compute_double_null_tf_matrix(
        case_data       = case_data,
        candidates_all  = candidates_all,
        case_ids_sorted = case_ids_sorted,
        labels          = labels,
        B_null          = B_null,
        B1_null         = B1_null,
        B2_null         = B2_null,
        alpha           = alpha,
        random_state    = rs + 300_000,       # offset from all production seeds
        n_jobs          = CONFIG.get('n_jobs', -1),
    )
    p_empirical_all = empirical_fisher_pvalue(tf_obs_all, tf_null_matrix)
    # p_empirical_all[i] = (1 + #{b: T_F^(b)(i) >= T_F_obs(i)}) / (B_null + 1)
    # Shape (m_total,). Values in [1/(B_null+1), 1.0].
    timing['double_null_calibration'] = time.time() - t_calib

    print(f"  Empirical calibration complete ({B_null} replicates, {timing['double_null_calibration']:.1f}s)")
    print(f"  p̃_F resolution: 1/{B_null+1} = {1.0/(B_null+1):.4e}  (vs alpha={alpha})")
    print(f"  Patterns with p̃_F ≤ alpha: {(p_empirical_all <= alpha).sum()}  "
          f"(analytic: {(p_conj_values <= alpha).sum()})")

    # Back-fill empirical p-value into each PatternTestResult for JSON output
    for pidx, r in enumerate(pattern_results):
        r.p_conjunction_empirical = float(p_empirical_all[pidx])

    # ── STEP 5b: Sample-split scope filter + Storey Q-Value FDR ─────────────
    # Scope filter: SCREEN p-values (independent of TEST p-values in Fisher).
    # Under H₀ˢ, p_struct_screen is super-uniform on B2_screen permutations.
    # Filtering on p_struct_screen ≤ α and testing with p_struct_test (via Fisher)
    # is valid because the two halves used disjoint permutation draws → independence.
    # Formal guarantee: conditional on passing the screen, p_struct_test remains
    # super-uniform (Fithian & Lei 2022) → Fisher p-values super-uniform on m'' →
    # Storey q-values control FDR ≤ α (Storey 2002, Theorem 1).
    print(f"\n{'='*100}")
    print("📊 STEP 5b: SAMPLE-SPLIT SCOPE FILTER + STOREY Q-VALUE FDR (PRIMARY)")
    print(f"{'='*100}")
    print(f"   Scope filter uses SCREEN p-values (independent of TEST p-values in Fisher).")
    print(f"   Fisher uses EMPIRICAL p-values — Phipson-Smyth calibrated under double-null.")
    print(f"   q(p) = min_{{k'≥k}} [π̂₀·m''·p̃_F_(k')/k']  [Storey 2002]")
    print(f"   Calibration: empirical Phipson-Smyth  "
          f"(B_null={B_null}, resolution=1/{B_null+1}={1/(B_null+1):.4e})")
    print(f"   Analytic χ²₄ p-values retained in p_conjunction field (BH reference only)")

    # Scope filter: patterns where min(p_screen_c0, p_screen_c1) ≤ α
    structural_idx = [
        pidx for pidx, r in enumerate(pattern_results)
        if min(r.p_structural_screen_class0, r.p_structural_screen_class1) <= alpha
    ]
    m_prime = len(structural_idx)
    print(f"   Sample-split scope filter: m={m_total} → m''={m_prime} "
          f"({m_total - m_prime} excluded by min(p_screen_c0, p_screen_c1) > {alpha})")
    print(f"   Screen p-values are INDEPENDENT of test p-values in Fisher → no selection bias")

    # PRIMARY CHANGE: use empirical p-values instead of analytic for Storey gate
    p_fisher_filtered = p_empirical_all[structural_idx]  # CHANGED (was p_conj_values[...])

    t_sam = time.time()
    if m_prime > 0:
        pi0_fisher, lambda_star_fisher = adaptive_storey_pi0(
            p_fisher_filtered, q=alpha,
        )
        q_values_fisher = storey_qvalue(p_fisher_filtered, pi0_fisher)
    else:
        pi0_fisher, lambda_star_fisher = 1.0, alpha
        q_values_fisher = np.ones(0)
    timing['sam_fdr'] = time.time() - t_sam

    if m_prime > 0:
        sort_idx_f = np.argsort(p_fisher_filtered)
        q_sorted_f = q_values_fisher[sort_idx_f]
        p_sorted_f = p_fisher_filtered[sort_idx_f]
        kstar_fisher  = int(np.sum(q_values_fisher <= alpha))
        tau_star_f    = float(p_sorted_f[kstar_fisher - 1]) if kstar_fisher > 0 else 0.0
        fdp_at_tau_f  = float(q_sorted_f[kstar_fisher - 1]) if kstar_fisher > 0 else 1.0
    else:
        kstar_fisher = 0; tau_star_f = 0.0; fdp_at_tau_f = 1.0

    print(f"\n   Fisher-Storey results (on m''={m_prime} scope-filtered patterns):")
    print(f"     π̂₀  = {pi0_fisher:.4f}  (AS Gao 2023, λ*={lambda_star_fisher:.2f})")
    print(f"     k*  = {kstar_fisher:,}  (Fisher-Storey significant patterns)")
    print(f"     τ*_p = {tau_star_f:.6e}  (threshold on Fisher p-value scale)")
    print(f"     FDP̂ = {fdp_at_tau_f:.4f}  (at τ*)")

    # Map back to original indices via structural_idx
    for sam_i, orig_i in enumerate(structural_idx):
        r = pattern_results[orig_i]
        r.q_value_sam                   = float(q_values_fisher[sam_i])
        r.is_significant_sam            = bool(q_values_fisher[sam_i] <= alpha)
        r.is_significant_discriminative = r.is_significant_sam
        r.fdp_estimate                  = fdp_at_tau_f
        r.tau_star_sam                  = tau_star_f

    # BH reference — on m'' Fisher p-values (fair comparison against primary)
    if m_prime > 0:
        rejected_bh_f, bh_thresh_f, _ = benjamini_hochberg(
            p_fisher_filtered, alpha, method=fdr_method
        )
        for bh_i, orig_i in enumerate(structural_idx):
            if rejected_bh_f[bh_i]:
                pattern_results[orig_i].is_significant_bh = True
                pattern_results[orig_i].bh_rank = int(np.sum(p_fisher_filtered <=
                                                       p_fisher_filtered[bh_i]))
                pattern_results[orig_i].bh_threshold = float(bh_thresh_f[bh_i])
        n_bh_f = int(np.sum(rejected_bh_f))
    else:
        n_bh_f = 0

    print(f"     BH on m'':     {n_bh_f:,}  (reference, m''={m_prime})")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5c  —  STRUCTURAL FDR ON m' (structurally-touched) PATTERNS ONLY
    #             Symmetric scope with Step 5b.  Corrects the fatal m-asymmetry.
    # ════════════════════════════════════════════════════════════════════════
    # print(f"\n{'='*100}")
    # print("📊 STEP 5c: STOREY Q-VALUE FDR ON STRUCTURAL P-VALUES (per class, on m' scope)")
    # print(f"{'='*100}")
    # print(f"   Structural FDR restricted to m'={m_prime} structurally-touched patterns")
    # print(f"   (same structural_idx as Step 5b — symmetric comparison family)")

    # t_sc = time.time()

    # # ── Extract structural p-values FOR m' ONLY (not all m) ─────────────────
    # p_struct_c0 = np.array([pattern_results[i].p_structural_class0 for i in structural_idx])
    # p_struct_c1 = np.array([pattern_results[i].p_structural_class1 for i in structural_idx])

    # # ── AS π̂₀ on m' structural p-values ────────────────────────────────────
    # pi0_s0, lam_s0 = adaptive_storey_pi0(p_struct_c0, q=alpha)
    # pi0_s1, lam_s1 = adaptive_storey_pi0(p_struct_c1, q=alpha)

    # # ── Storey q-values on m' ────────────────────────────────────────────────
    # q_sc0 = storey_qvalue(p_struct_c0, pi0_s0)   # shape (m',)
    # q_sc1 = storey_qvalue(p_struct_c1, pi0_s1)   # shape (m',)

    # timing['structural_storey'] = time.time() - t_sc

    # k_sc0 = int(np.sum(q_sc0 <= alpha))
    # k_sc1 = int(np.sum(q_sc1 <= alpha))
    # print(f"   Class 0: π̂₀={pi0_s0:.4f} (λ*={lam_s0:.2f})  →  k*_struct_c0 = {k_sc0}")
    # print(f"   Class 1: π̂₀={pi0_s1:.4f} (λ*={lam_s1:.2f})  →  k*_struct_c1 = {k_sc1}")

    # # ── Step 1: set all m patterns to default ───────────────────────────────
    # for r in pattern_results:
    #     r.q_structural_class0    = 1.0
    #     r.q_structural_class1    = 1.0
    #     r.q_structural_dominant  = 1.0
    #     # is_significant_structural = raw nominal label (not an FDR gate)
    #     r.is_significant_structural = (r.p_structural_dominant <= alpha)

    # # ── Step 2: store Storey q-values on m' for transparency (JSON only) ────
    # # q_structural_dominant is recorded for inspection but NEVER gates
    # # is_significant_final.  is_significant_structural uses the raw nominal
    # # p_structural_dominant so that structural p-values are NOT used as both
    # # selector (structural_idx) and test statistic (avoids selection bias).
    # for sam_i, orig_i in enumerate(structural_idx):
    #     r = pattern_results[orig_i]
    #     r.q_structural_class0   = float(q_sc0[sam_i])
    #     r.q_structural_class1   = float(q_sc1[sam_i])
    #     q_dom = float(q_sc1[sam_i]) if r.dominant_class == 1 else float(q_sc0[sam_i])
    #     r.q_structural_dominant = q_dom
    #     # is_significant_structural already set in Step 1 (raw nominal) — do not overwrite

    # # ── Step 3: final verdict — Fisher-Storey is the single gate ────────────
    # # is_significant_final = is_significant_discriminative (q_Fisher ≤ α only).
    # # The four-category taxonomy is purely descriptive within the significant set;
    # # is_significant_structural (nominal) characterises structural evidence only.
    # cat_counts = {"Both": 0, "Structural only": 0, "Discriminative only": 0, "Neither": 0}

    # for r in pattern_results:
    #     r.is_significant_final = r.is_significant_discriminative  # single axis

    #     if r.is_significant_discriminative and r.is_significant_structural:
    #         r.significance_category = "Both"           # Fisher passed + nominal structural

    #     elif r.is_significant_discriminative and not r.is_significant_structural:
    #         r.significance_category = "Discriminative only"  # Fisher passed, structural weak

    #     elif r.is_significant_structural and not r.is_significant_discriminative:
    #         r.significance_category = "Structural only"      # nominal structural, Fisher failed

    #     else:
    #         r.significance_category = "Neither"

    #     cat_counts[r.significance_category] += 1

    # # k_final = k* (all Fisher-Storey rejections); Both/Disc-only is a descriptive split
    # n_sam_final = cat_counts["Both"] + cat_counts["Discriminative only"]
    # print(f"\n   Four-Category Taxonomy (α = {alpha}):")
    # print(f"     Both (Fisher ∧ nominal-struct):  {cat_counts['Both']:,}")
    # print(f"     Structural only (nominal only):  {cat_counts['Structural only']:,}")
    # print(f"     Discriminative only (Fisher):    {cat_counts['Discriminative only']:,}")
    # print(f"     Neither:                         {cat_counts['Neither']:,}")
    # print(f"     Fisher-Storey significant (k*):  {n_sam_final:,}  (= Both + Discriminative only)")
    # print(f"     BH on m' (reference):            {n_bh_f:,}")
    
    # ════════════════════════════════════════════════════════════════════════
    # STEP 5c  —  STRUCTURAL STOREY Q-VALUES (all m, transparency only)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("📊 STEP 5c: STOREY Q-VALUE FDR ON STRUCTURAL P-VALUES (per class, all m)")
    print(f"{'='*100}")
    print(f"   Structural FDR on all m={m_total} patterns (transparency only — not a gate)")

    t_sc = time.time()

    p_struct_c0_all = np.array([r.p_structural_class0 for r in pattern_results])
    p_struct_c1_all = np.array([r.p_structural_class1 for r in pattern_results])

    pi0_s0, lam_s0 = adaptive_storey_pi0(p_struct_c0_all, q=alpha)
    pi0_s1, lam_s1 = adaptive_storey_pi0(p_struct_c1_all, q=alpha)

    q_sc0 = storey_qvalue(p_struct_c0_all, pi0_s0)
    q_sc1 = storey_qvalue(p_struct_c1_all, pi0_s1)

    timing['structural_storey'] = time.time() - t_sc

    k_sc0 = int(np.sum(q_sc0 <= alpha))
    k_sc1 = int(np.sum(q_sc1 <= alpha))
    print(f"   Class 0: π̂₀={pi0_s0:.4f} (λ*={lam_s0:.2f})  →  k*_struct_c0 = {k_sc0}")
    print(f"   Class 1: π̂₀={pi0_s1:.4f} (λ*={lam_s1:.2f})  →  k*_struct_c1 = {k_sc1}")

    # Assign structural q-values and nominal label to ALL patterns
    for p_idx, r in enumerate(pattern_results):
        r.q_structural_class0   = float(q_sc0[p_idx])
        r.q_structural_class1   = float(q_sc1[p_idx])
        q_dom = float(q_sc1[p_idx]) if r.dominant_class == 1 else float(q_sc0[p_idx])
        r.q_structural_dominant = q_dom
        r.is_significant_structural = (r.p_structural_dominant <= alpha)

    # Final verdict — conjunctive gate: q_Fisher ≤ α  AND  p_struct_dom ≤ α  ("Both")
    cat_counts = {"Both": 0, "Structural only": 0, "Discriminative only": 0, "Neither": 0}

    for r in pattern_results:
        r.is_significant_final = r.is_significant_discriminative and r.is_significant_structural

        if r.is_significant_discriminative and r.is_significant_structural:
            r.significance_category = "Both"
        elif r.is_significant_discriminative and not r.is_significant_structural:
            r.significance_category = "Discriminative only"
        elif r.is_significant_structural and not r.is_significant_discriminative:
            r.significance_category = "Structural only"
        else:
            r.significance_category = "Neither"

        cat_counts[r.significance_category] += 1

    n_sam_final = cat_counts["Both"]
    print(f"\n   Four-Category Taxonomy (α = {alpha}):")
    print(f"     Both (Fisher ∧ nominal-struct):  {cat_counts['Both']:,}")
    print(f"     Structural only (nominal only):  {cat_counts['Structural only']:,}")
    print(f"     Discriminative only (Fisher):    {cat_counts['Discriminative only']:,}")
    print(f"     Neither:                         {cat_counts['Neither']:,}")
    print(f"     Fisher-Storey significant (k*):  {n_sam_final:,}  (= Both only)")
    print(f"     BH on m'' (reference):           {n_bh_f:,}")

    timing['total'] = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("📊 THREE-HYPOTHESIS SAM FDR RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"\n   Total patterns tested:    {m_total:,}")
    print(f"   Sample-split scope (m''): {m_prime:,} ({m_prime/m_total*100:.1f}%) — screen p ≤ {alpha}")
    print(f"   BH on m'' (reference):    {n_bh_f:,}")
    print(f"\n   ┌────────────────────────────────────────┬──────────────┐")
    print(f"   │ Category                               │   Count      │")
    print(f"   ├────────────────────────────────────────┼──────────────┤")
    print(f"   │ Both (q_Fisher ≤ α ∧ p_struct_nom ≤ α) │ {cat_counts['Both']:>6,}       │")
    print(f"   │ Structural only (p_struct_nom ≤ α)     │ {cat_counts['Structural only']:>6,}       │")
    print(f"   │ Discriminative only (q_Fisher ≤ α)     │ {cat_counts['Discriminative only']:>6,}       │")
    print(f"   │ Neither                                │ {cat_counts['Neither']:>6,}       │")
    print(f"   └────────────────────────────────────────┴──────────────┘")

    # Direction breakdown for 'Both' patterns
    n_pos = sum(1 for r in pattern_results if r.is_significant_final and r.direction == "Positive")
    n_neg = sum(1 for r in pattern_results if r.is_significant_final and r.direction == "Negative")
    print(f"\n   Significant patterns (Fisher-primary) by direction:")
    print(f"     Positive (Not-Granted-dominant): {n_pos:,}")
    print(f"     Negative (Granted-dominant):     {n_neg:,}")

    # Q-value summary
    all_q_disc   = [r.q_value_sam            for r in pattern_results]
    all_q_struct = [r.q_structural_dominant   for r in pattern_results]
    all_p_conj   = [r.p_conjunction           for r in pattern_results]
    print(f"\n   Q-value / p-value distributions (median [min, max]):")
    print(f"     q_disc:  {np.median(all_q_disc):.4f} [{min(all_q_disc):.4e}, {max(all_q_disc):.4f}]")
    print(f"     q_struct:{np.median(all_q_struct):.4f} [{min(all_q_struct):.4e}, {max(all_q_struct):.4f}]")
    print(f"     p_conj:  {np.median(all_p_conj):.4f} [{min(all_p_conj):.4e}, {max(all_p_conj):.4f}]")

    # Timing
    print(f"\n   ⏱️  Timing:")
    for k, v in timing.items():
        print(f"     {k:30s}: {v:.1f}s")

    delta_obs = np.array([r.delta_obs for r in pattern_results])
    return pattern_results, timing, {
        'null_delta_matrix': null_delta_matrix,
        'tf_null_matrix':    tf_null_matrix, # ← (B_null × m, float64)
        'holds_all':         holds_all,
        'delta_obs':         delta_obs,
        'candidates_all':    candidates_all,
    }


# ============================================================================
# DISCRIMINATION METRICS (per-pattern binary classifier evaluation)
# ============================================================================

@dataclass
class PatternDiscriminationMetrics:
    pattern_id: str
    target_class: int
    constraint_type: str
    activity_a: str
    activity_b: Optional[str]
    tp: int; fp: int; fn: int; tn: int; n_total: int
    precision: float; recall: float; f1_score: float
    accuracy: float; balanced_accuracy: float
    specificity: float; mcc: float; auroc: float
    p_conjunction: float; delta_obs: float


def compute_discrimination_for_pattern(
    r: PatternTestResult,
) -> PatternDiscriminationMetrics:
    """
    Evaluate pattern as a binary classifier predicting its dominant class.
    """
    target = r.dominant_class
    if target == 1:
        tp, fn = r.n_satisfied_class1, r.n_applicable_class1 - r.n_satisfied_class1
        fp, tn = r.n_satisfied_class0, r.n_applicable_class0 - r.n_satisfied_class0
    else:
        tp, fn = r.n_satisfied_class0, r.n_applicable_class0 - r.n_satisfied_class0
        fp, tn = r.n_satisfied_class1, r.n_applicable_class1 - r.n_satisfied_class1

    n_total = tp + fp + fn + tn
    eps = 1e-10
    prec = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec + eps) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (n_total + eps) if n_total > 0 else 0.0
    spec = tn / (tn + fp + eps) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2
    mcc_d = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / mcc_d if mcc_d > 0 else 0.0
    auroc = (rec + spec) / 2

    return PatternDiscriminationMetrics(
        pattern_id=r.pattern_id, target_class=target,
        constraint_type=r.constraint_type,
        activity_a=r.activity_a, activity_b=r.activity_b,
        tp=tp, fp=fp, fn=fn, tn=tn, n_total=n_total,
        precision=prec, recall=rec, f1_score=f1,
        accuracy=acc, balanced_accuracy=bal_acc,
        specificity=spec, mcc=mcc, auroc=auroc,
        p_conjunction=r.p_conjunction, delta_obs=r.delta_obs,
    )


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_plot_pdf(fig, filename, dpi=300):
    pdf_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"      ✓ Saved: {filename}")


def generate_outputs(
    pattern_results: List[PatternTestResult],
    case_data: Dict[str, CaseInfo],
    timing: Dict,
):
    """Generate JSON, text report, and visualizations."""

    print(f"\n{'='*100}")
    print("📊 GENERATING OUTPUTS")
    print(f"{'='*100}")

    D_0, D_1 = split_by_class(case_data)
    sig_results       = [r for r in pattern_results if r.is_significant_final]
    struct_only       = [r for r in pattern_results if r.significance_category == "Structural only"]
    disc_only         = [r for r in pattern_results if r.significance_category == "Discriminative only"]

    # ── JSON Output ──────────────────────────────────────────────────────
    print("\n🔄 Generating JSON output...")

    json_out = {
        'framework': 'Three-Hypothesis SAM-Permutation FDR Conjunction Test',
        'version': '8.0-DUAL-AXIS-STOREY-FDR',
        'timestamp': datetime.now().isoformat(),
        'scientific_description': {
            'H0_structural': 'Trace-activity permutation within each class. Tests non-random temporal ordering.',
            'H0_discriminative': 'Label permutation across cases. Tests class-conditional prevalence difference.',
            'H0_conjunction': 'IUT: max(p_struct, p_disc). Pattern is both structured AND discriminative.',
            'p_value_method': 'Phipson & Smyth (2010) exact permutation p-values.',
            'fdr_control': (
                f'Storey (2002) Q-Value FDR (per-pattern Phipson-Smyth p-values) at α = {CONFIG["fdr_alpha"]}. '
                f'BH-{CONFIG["fdr_method"]} retained for comparison.'
            ),
            'sam_note': (
                'q(p) = min_{{k\'>=k}} [pi0_hat * m\' * p_(k\') / k\']. '
                'pi0_hat (Storey 2002, lambda=0.5) is conservative under PRDS. '
                'Replaces Tusher (2001) flat-null which fails at m\' < 500 due to heterogeneous sigma_null.'
            ),
        },
        'references': [
            'Tusher, Tibshirani & Chu (2001). Significance analysis of microarrays. PNAS 98(9):5116-5121.',
            'Storey & Tibshirani (2003). Statistical significance for genomewide studies. PNAS 100(16):9440-9445.',
            'Ojala & Garriga (2010). Permutation Tests for Studying Classifier Performance. JMLR.',
            'Phipson & Smyth (2010). Permutation p-values should never be zero.',
            'Benjamini & Hochberg (1995). Controlling the FDR. JRSS-B.',
            'Berger (1982). Multiparameter Hypothesis Testing. Technometrics.',
            'Di Ciccio & Montali (2022). DECLARE constraint semantics.',
        ],
        'configuration': CONFIG,
        'dataset_statistics': {
            'total_cases': len(case_data),
            'class_0_cases': len(D_0),
            'class_1_cases': len(D_1),
        },
        'summary': {
            'total_patterns_tested': len(pattern_results),
            'four_category_counts': {
                'both': len(sig_results),
                'structural_only': len(struct_only),
                'discriminative_only': len(disc_only),
                'neither': sum(1 for r in pattern_results if r.significance_category == "Neither"),
            },
            'both_positive': sum(1 for r in sig_results if r.direction == "Positive"),
            'both_negative': sum(1 for r in sig_results if r.direction == "Negative"),
            'bh_rejections_reference': sum(1 for r in pattern_results if r.is_significant_bh),
        },
        'timing': timing,
        'significant_patterns': [],      # category: Both
        'structural_only_patterns': [],  # category: Structural only
        'discriminative_only_patterns': [], # category: Discriminative only
        'all_patterns': [],
    }

    # Sort significant patterns by conjunction p-value
    for r in sorted(sig_results, key=lambda x: x.p_conjunction):
        pd_dict = {
            'pattern_id': r.pattern_id,
            'constraint_type': r.constraint_type,
            'activity_a': r.activity_a,
            'activity_b': r.activity_b,
            'direction': r.direction,
            'dominant_class': r.dominant_class,
            'prevalence_class0': r.prevalence_class0,
            'prevalence_class1': r.prevalence_class1,
            'delta_obs': r.delta_obs,
            'n_applicable_class0': r.n_applicable_class0,
            'n_applicable_class1': r.n_applicable_class1,
            'n_satisfied_class0': r.n_satisfied_class0,
            'n_satisfied_class1': r.n_satisfied_class1,
            'p_values': {
                'p_structural_class0': r.p_structural_class0,
                'p_structural_class1': r.p_structural_class1,
                'p_structural_dominant': r.p_structural_dominant,
                'p_discriminative_two_sided': r.p_discriminative,
                'p_discriminative_one_sided': r.p_discriminative_onesided,
                'p_conjunction': r.p_conjunction,                          # analytic chi2_4
                'p_conjunction_empirical': r.p_conjunction_empirical,      # Phipson-Smyth
            },
            'null_statistics': {
                'null_mean_class0': r.null_mean_class0,
                'null_mean_class1': r.null_mean_class1,
                'null_std_class0': r.null_std_class0,
                'null_std_class1': r.null_std_class1,
                'null_delta_mean': r.null_delta_mean,
                'null_delta_std': r.null_delta_std,
            },
            'storey_fdr': {
                'significance_category': r.significance_category,
                'is_significant_structural': r.is_significant_structural,
                'is_significant_discriminative': r.is_significant_discriminative,
                'is_significant_final': r.is_significant_final,
                'q_structural_class0': r.q_structural_class0,
                'q_structural_class1': r.q_structural_class1,
                'q_structural_dominant': r.q_structural_dominant,
                'q_value_disc': r.q_value_sam,
                'fdp_estimate': r.fdp_estimate,
                'tau_star_disc': r.tau_star_sam,
            },
            'bh_fdr': {
                'is_significant': r.is_significant_bh,
                'bh_rank': r.bh_rank,
                'bh_threshold': r.bh_threshold,
            },
            'applicable_subgroups': {
                'subgroups': r.applicable_subgroups,
                'n_subgroups': len(r.applicable_subgroups),
                'subgroup_to_cases': r.subgroup_to_cases,
            },
        }
        json_out['significant_patterns'].append(pd_dict)

    # ── Structural-only patterns (compact) ───────────────────────────────
    def _compact_pattern_dict(r: PatternTestResult) -> dict:
        return {
            'pattern_id': r.pattern_id,
            'constraint_type': r.constraint_type,
            'activity_a': r.activity_a,
            'activity_b': r.activity_b,
            'direction': r.direction,
            'significance_category': r.significance_category,
            'prevalence_class0': r.prevalence_class0,
            'prevalence_class1': r.prevalence_class1,
            'delta_obs': r.delta_obs,
            'q_structural_dominant': r.q_structural_dominant,
            'q_value_disc': r.q_value_sam,
            'p_conjunction': r.p_conjunction,                      # analytic chi2_4
            'p_conjunction_empirical': r.p_conjunction_empirical,  # Phipson-Smyth
            'is_significant_structural': r.is_significant_structural,
            'is_significant_discriminative': r.is_significant_discriminative,
        }

    for r in sorted(struct_only, key=lambda x: x.q_structural_dominant):
        json_out['structural_only_patterns'].append(_compact_pattern_dict(r))

    for r in sorted(disc_only, key=lambda x: x.q_value_sam):
        json_out['discriminative_only_patterns'].append(_compact_pattern_dict(r))

    # All patterns (compact), sorted by q_structural_dominant then q_value_disc
    for r in sorted(pattern_results,
                    key=lambda x: (x.q_structural_dominant, x.q_value_sam)):
        json_out['all_patterns'].append({
            'pattern_id': r.pattern_id,
            'constraint_type': r.constraint_type,
            'activity_a': r.activity_a,
            'activity_b': r.activity_b,
            'direction': r.direction,
            'significance_category': r.significance_category,
            'prevalence_class0': r.prevalence_class0,
            'prevalence_class1': r.prevalence_class1,
            'delta_obs': r.delta_obs,
            'p_structural_dominant': r.p_structural_dominant,
            'p_discriminative': r.p_discriminative,
            'p_conjunction': r.p_conjunction,                      # analytic chi2_4
            'p_conjunction_empirical': r.p_conjunction_empirical,  # Phipson-Smyth
            'q_structural_class0': r.q_structural_class0,
            'q_structural_class1': r.q_structural_class1,
            'q_structural_dominant': r.q_structural_dominant,
            'q_value_disc': r.q_value_sam,
            'is_significant_structural': r.is_significant_structural,
            'is_significant_discriminative': r.is_significant_discriminative,
            'is_significant_final': r.is_significant_final,
            'is_significant_bh': r.is_significant_bh,
            'bh_rank': r.bh_rank,
            'null_mean_class0': r.null_mean_class0,
            'null_mean_class1': r.null_mean_class1,
            'null_std_class0': r.null_std_class0,
            'null_std_class1': r.null_std_class1,
        })

    json_path = os.path.join(OUTPUT_DIR, 'three_hypothesis_samfdr_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON: {json_path}")

    # Significant-only JSON (all non-Neither categories + metadata, without all_patterns)
    sig_json = {k: v for k, v in json_out.items() if k != 'all_patterns'}
    sig_path = os.path.join(OUTPUT_DIR, 'significant_patterns_only.json')
    with open(sig_path, 'w', encoding='utf-8') as f:
        json.dump(sig_json, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON (significant only): {sig_path}")

    # ── Text Report ──────────────────────────────────────────────────────
    print("\n🔄 Generating text report...")
    generate_text_report(pattern_results, case_data, timing)

    # ── Visualizations ───────────────────────────────────────────────────
    print("\n🔄 Generating visualizations...")
    generate_visualizations(pattern_results, case_data, timing)

    # ── Discrimination Metrics ───────────────────────────────────────────
    print("\n🔄 Computing discrimination metrics...")
    disc_metrics = []
    for r in sig_results:
        dm = compute_discrimination_for_pattern(r)
        disc_metrics.append(dm)

    if disc_metrics:
        print(f"   {len(disc_metrics):,} significant patterns evaluated as classifiers")
        f1s = [d.f1_score for d in disc_metrics]
        mccs = [d.mcc for d in disc_metrics]
        print(f"   F1-score: mean={np.mean(f1s):.4f}, median={np.median(f1s):.4f}")
        print(f"   MCC:      mean={np.mean(mccs):.4f}, median={np.median(mccs):.4f}")

        # Save discrimination JSON
        disc_json = {
            'framework_version': '8.0-DUAL-AXIS-STOREY-FDR',
            'n_significant_patterns': len(disc_metrics),
            'patterns': sorted(
                [
                    {
                        'pattern_id': d.pattern_id,
                        'target_class': d.target_class,
                        'tp': d.tp, 'fp': d.fp, 'fn': d.fn, 'tn': d.tn,
                        'precision': d.precision, 'recall': d.recall,
                        'f1_score': d.f1_score, 'mcc': d.mcc,
                        'balanced_accuracy': d.balanced_accuracy,
                        'auroc': d.auroc,
                        'p_conjunction': d.p_conjunction,
                        'delta_obs': d.delta_obs,
                    }
                    for d in disc_metrics
                ],
                key=lambda x: x['f1_score'],
                reverse=True,
            )
        }
        disc_path = os.path.join(OUTPUT_DIR, 'discrimination_metrics.json')
        with open(disc_path, 'w', encoding='utf-8') as f:
            json.dump(disc_json, f, indent=2, ensure_ascii=False)
        print(f"✓ Discrimination metrics: {disc_path}")

    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


def generate_text_report(
    results: List[PatternTestResult],
    case_data: Dict[str, CaseInfo],
    timing: Dict,
):
    """Generate detailed text report."""
    D_0, D_1 = split_by_class(case_data)
    sig = [r for r in results if r.is_significant_final]
    rpt = []

    rpt.append("=" * 120)
    rpt.append("THREE-HYPOTHESIS DISCRIMINATIVE SPECIFICATION MINING — PHASE 1")
    rpt.append("Storey CONJUNCTION TEST")
    rpt.append("=" * 120)
    rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append(f"Version: 8.0-DUAL-AXIS-STOREY-FDR")
    rpt.append("")

    rpt.append("=" * 120)
    rpt.append("THEORETICAL FRAMEWORK")
    rpt.append("=" * 120)
    rpt.append("")
    rpt.append("Null Hypothesis 1 — H₀ˢ (Structural):")
    rpt.append("  Trace-activity permutation within each class.")
    rpt.append("  Tests: pattern captures genuine temporal regularity.")
    rpt.append("")
    rpt.append("Null Hypothesis 2 — H₀ᵈ (Discriminative):")
    rpt.append("  Label permutation across cases.")
    rpt.append("  Tests: pattern prevalence differs between classes.")
    rpt.append("")
    rpt.append("Null Hypothesis 3 — H₀ᶜ (Conjunction):")
    rpt.append("  p_conj = max(p_struct_dominant, p_disc)  [IUT, Berger 1982]")
    rpt.append("  Pattern is either structurally non-random OR class-discriminative.")
    rpt.append("")
    rpt.append("P-value: Phipson & Smyth (2010) exact formula:")
    rpt.append("  p = (1 + #{T_b ≥ T_obs}) / (B + 1)")
    rpt.append("")
    rpt.append(f"FDR Control: Storey (Tusher et al. 2001) at α = {CONFIG['fdr_alpha']}")
    rpt.append(f"             BH-{CONFIG['fdr_method']} retained for comparison.")
    rpt.append("")

    rpt.append("=" * 120)
    rpt.append("CONFIGURATION")
    rpt.append("=" * 120)
    for k, v in CONFIG.items():
        rpt.append(f"  {k}: {v}")
    rpt.append("")

    rpt.append("=" * 120)
    rpt.append("DATASET")
    rpt.append("=" * 120)
    rpt.append(f"  Total cases:   {len(case_data):,}")
    rpt.append(f"  Class 0 (NA):  {len(D_0):,} ({len(D_0)/len(case_data)*100:.1f}%)")
    rpt.append(f"  Class 1 (A):   {len(D_1):,} ({len(D_1)/len(case_data)*100:.1f}%)")
    rpt.append("")

    rpt.append("=" * 120)
    rpt.append("RESULTS SUMMARY — FOUR-CATEGORY VERDICT (α = {:.2f})".format(CONFIG['fdr_alpha']))
    rpt.append("=" * 120)
    n_bh       = sum(1 for r in results if r.is_significant_bh)
    n_both     = len(sig)
    n_str_only = sum(1 for r in results if r.significance_category == "Structural only")
    n_dis_only = sum(1 for r in results if r.significance_category == "Discriminative only")
    n_neither  = sum(1 for r in results if r.significance_category == "Neither")
    m          = len(results)

    rpt.append(f"  Total patterns tested:                  {m:,}")
    rpt.append(f"")
    rpt.append(f"  {'Category':<40s} {'Count':>6}  {'%':>6}")
    rpt.append(f"  {'─'*54}")
    rpt.append(f"  {'Both  (q_struct ≤ α ∧ q_disc ≤ α)':<40s} {n_both:>6,}  {n_both/m*100:>5.1f}%")
    rpt.append(f"  {'Structural only  (q_struct ≤ α)':<40s} {n_str_only:>6,}  {n_str_only/m*100:>5.1f}%")
    rpt.append(f"  {'Discriminative only  (q_disc ≤ α)':<40s} {n_dis_only:>6,}  {n_dis_only/m*100:>5.1f}%")
    rpt.append(f"  {'Neither':<40s} {n_neither:>6,}  {n_neither/m*100:>5.1f}%")
    rpt.append(f"  {'─'*54}")
    rpt.append(f"  {'BH on m\' (reference)':<40s} {n_bh:>6,}  {n_bh/m*100:>5.1f}%")
    rpt.append(f"")

    n_pos = sum(1 for r in sig if r.direction == "Positive")
    n_neg = sum(1 for r in sig if r.direction == "Negative")
    rpt.append(f"  'Both' by direction:")
    rpt.append(f"    Positive (Not-Granted-dominant): {n_pos:,}")
    rpt.append(f"    Negative (Granted-dominant):     {n_neg:,}")
    rpt.append("")

    def _fmt_pattern_block(r: PatternTestResult, rank: int, category_label: str) -> List[str]:
        lines = []
        lines.append(f"{'─'*120}")
        lines.append(f"Rank {rank}: {r.pattern_id}  [{category_label}]")
        lines.append(f"{'─'*120}")
        lines.append(f"  Constraint: {r.constraint_type}")
        if r.activity_b:
            lines.append(f"  Activities: {r.activity_a[:50]} → {r.activity_b[:50]}")
        else:
            lines.append(f"  Activity:   {r.activity_a[:60]}")
        lines.append(f"  Direction:  {r.direction} (dominant class = {r.dominant_class})")
        lines.append("")
        lines.append(f"  Prevalence:  Class 0 = {r.prevalence_class0:.4f} ({r.n_satisfied_class0}/{r.n_applicable_class0})")
        lines.append(f"               Class 1 = {r.prevalence_class1:.4f} ({r.n_satisfied_class1}/{r.n_applicable_class1})")
        lines.append(f"  Δ_obs:       {r.delta_obs:+.4f}")
        lines.append("")
        lines.append(f"  P-VALUES (raw, Phipson-Smyth):")
        lines.append(f"    p_struct(class 0) = {r.p_structural_class0:.4e}")
        lines.append(f"    p_struct(class 1) = {r.p_structural_class1:.4e}")
        lines.append(f"    p_struct(dominant)= {r.p_structural_dominant:.4e}")
        lines.append(f"    p_disc(two-sided) = {r.p_discriminative:.4e}")
        lines.append(f"    p_disc(one-sided) = {r.p_discriminative_onesided:.4e}")
        lines.append(f"    p_conjunction     = {r.p_conjunction:.4e}")
        lines.append(f"  STOREY Q-VALUES (FDR-controlled):")
        lines.append(f"    q_structural_c0  = {r.q_structural_class0:.4e}")
        lines.append(f"    q_structural_c1  = {r.q_structural_class1:.4e}")
        lines.append(f"    q_structural_dom = {r.q_structural_dominant:.4e}  [is_significant_structural = {r.is_significant_structural}]")
        lines.append(f"    q_disc (Fisher)  = {r.q_value_sam:.4e}  [τ* = {r.tau_star_sam:.6f}, FDP̂ = {r.fdp_estimate:.4f}]")
        lines.append(f"    BH rank = {r.bh_rank}, BH threshold = {r.bh_threshold:.4e}")
        lines.append("")
        lines.append(f"  NULL STATISTICS:")
        lines.append(f"    Structural: null_mean₀={r.null_mean_class0:.4f}, null_mean₁={r.null_mean_class1:.4f}")
        lines.append(f"    Discriminative: null_Δ_mean={r.null_delta_mean:.4f}, null_Δ_std={r.null_delta_std:.4f}")
        lines.append("")
        return lines

    rpt.append("=" * 120)
    rpt.append("TOP 30 — CATEGORY: BOTH  (q_struct ≤ α ∧ q_disc ≤ α)  [sorted by q_disc]")
    rpt.append("=" * 120)
    rpt.append("")
    for i, r in enumerate(sorted(sig, key=lambda x: x.q_value_sam)[:30], 1):
        rpt.extend(_fmt_pattern_block(r, i, "Both"))

    struct_only_results = [r for r in results if r.significance_category == "Structural only"]
    rpt.append("=" * 120)
    rpt.append("TOP 30 — CATEGORY: STRUCTURAL ONLY  (q_struct ≤ α, q_disc > α)  [sorted by q_struct_dom]")
    rpt.append("Scientific meaning: genuine temporal regularity; class-agnostic behaviour.")
    rpt.append("=" * 120)
    rpt.append("")
    for i, r in enumerate(sorted(struct_only_results, key=lambda x: x.q_structural_dominant)[:30], 1):
        rpt.extend(_fmt_pattern_block(r, i, "Structural only"))

    disc_only_results = [r for r in results if r.significance_category == "Discriminative only"]
    rpt.append("=" * 120)
    rpt.append("TOP 30 — CATEGORY: DISCRIMINATIVE ONLY  (q_disc ≤ α, q_struct > α)  [sorted by q_disc]")
    rpt.append("Scientific meaning: class-specific but may be a frequency artifact (no temporal structure).")
    rpt.append("=" * 120)
    rpt.append("")
    for i, r in enumerate(sorted(disc_only_results, key=lambda x: x.q_value_sam)[:30], 1):
        rpt.extend(_fmt_pattern_block(r, i, "Discriminative only"))

    rpt.append("=" * 120)
    rpt.append("TIMING")
    rpt.append("=" * 120)
    for k, v in timing.items():
        rpt.append(f"  {k:30s}: {v:.1f}s")
    rpt.append("")
    rpt.append("=" * 120)
    rpt.append("END OF REPORT")
    rpt.append("=" * 120)

    path = os.path.join(OUTPUT_DIR, 'three_hypothesis_sam_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rpt))
    print(f"✓ Text report: {path}")


# ============================================================================
# VISUALIZATIONS — PUBLICATION QUALITY
# ============================================================================

def generate_visualizations(
    results: List[PatternTestResult],
    case_data: Dict[str, CaseInfo],
    timing: Dict,
):
    """Generate comprehensive visualizations for the three-hypothesis framework."""

    sig = [r for r in results if r.is_significant_final]
    D_0, D_1 = split_by_class(case_data)

    # ========================================================================
    # PLOT 1: BH-FDR Step-Up Plot (reference comparison)
    # ========================================================================
    print("   [1/10] BH-FDR Step-Up Plot (reference)...")

    p_conj_sorted = np.sort([r.p_conjunction for r in results])
    m = len(p_conj_sorted)
    ranks = np.arange(1, m + 1)
    alpha = CONFIG['fdr_alpha']
    c_m = np.sum(1.0 / ranks) if CONFIG['fdr_method'] == 'BY' else 1.0
    bh_line = ranks * alpha / (m * c_m)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(ranks, p_conj_sorted, s=8, alpha=0.5, c=COLORS['conjunction'],
               edgecolors='none', label='$p_\\mathrm{conj}$ (sorted)')
    ax.plot(ranks, bh_line, '--', color=COLORS['threshold'], linewidth=2,
            label=f'BH threshold $k\\alpha/m$ ($\\alpha$={alpha})')

    k_star = sum(1 for r in results if r.is_significant_bh)
    if k_star > 0:
        ax.axvline(k_star, color=COLORS['accent1'], linestyle=':', linewidth=1.5,
                   label=f'$k^*$ = {k_star}')
        ax.fill_between(ranks[:k_star], 0, p_conj_sorted[:k_star],
                        alpha=0.15, color=COLORS['accent1'])

    ax.set_xlabel('Rank $k$', fontweight='bold')
    ax.set_ylabel('$p_\\mathrm{conj}^{(k)}$', fontweight='bold')
    ax.set_title(f'BH-{CONFIG["fdr_method"]} Step-Up Procedure on Conjunction P-values (reference)',
                 fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.set_xlim(0, m + 1)
    ax.set_ylim(-0.02, 1.02)
    for s in ax.spines.values():
        s.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, '01_bh_fdr_stepup.pdf')

    # ========================================================================
    # PLOT 2: P-value Comparison Scatter (p_disc vs p_struct)
    # ========================================================================
    print("   [2/10] P-value Scatter: p_disc vs p_struct...")

    fig, ax = plt.subplots(figsize=(9, 9))
    p_d = [r.p_discriminative for r in results]
    p_s = [r.p_structural_dominant for r in results]
    is_sig = [r.is_significant_bh for r in results]

    colors_scatter = [COLORS['conjunction'] if s else COLORS['neither'] for s in is_sig]
    sizes = [50 if s else 15 for s in is_sig]

    ax.scatter(p_s, p_d, c=colors_scatter, s=sizes, alpha=0.6, edgecolors='black',
               linewidths=0.3)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(alpha, color=COLORS['discriminative'], linestyle=':', linewidth=1,
               alpha=0.5, label=f'$\\alpha$ = {alpha}')
    ax.axvline(alpha, color=COLORS['structural'], linestyle=':', linewidth=1,
               alpha=0.5)

    ax.set_xlabel('$p_\\mathrm{struct}$ (dominant class)', fontweight='bold')
    ax.set_ylabel('$p_\\mathrm{disc}$ (label permutation)', fontweight='bold')
    ax.set_title('Structural vs. Discriminative P-values', fontweight='bold')
    legend_elems = [
        Patch(facecolor=COLORS['conjunction'], edgecolor='black',
              label=f'BH-significant ({sum(is_sig)})'),
        Patch(facecolor=COLORS['neither'], edgecolor='black',
              label=f'Non-significant ({sum(not s for s in is_sig)})'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', frameon=True,
              fancybox=False, edgecolor='black')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    for s in ax.spines.values():
        s.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, '02_pvalue_scatter_disc_vs_struct.pdf')

    # ========================================================================
    # PLOT 3: Volcano Plot — Δ_obs vs −log₁₀(p_conj)
    # ========================================================================
    print("   [3/10] Volcano Plot...")

    fig, ax = plt.subplots(figsize=(10, 7))
    deltas = np.array([r.delta_obs for r in results])
    neg_log_p = -np.log10(np.clip([r.p_conjunction for r in results], 1e-300, 1.0))
    is_sig_arr = np.array([r.is_significant_bh for r in results])

    ax.scatter(deltas[~is_sig_arr], neg_log_p[~is_sig_arr],
               c=COLORS['neither'], s=20, alpha=0.4, edgecolors='none',
               label='Non-significant')
    # Color significant by direction
    for r_idx, r in enumerate(results):
        if r.is_significant_bh:
            c = COLORS['class1'] if r.direction == "Positive" else COLORS['class0']
            ax.scatter(deltas[r_idx], neg_log_p[r_idx], c=c, s=50,
                       alpha=0.7, edgecolors='black', linewidths=0.5)

    ax.axhline(-np.log10(alpha), color='gray', linestyle=':', linewidth=1.5,
               alpha=0.5, label=f'$\\alpha$ = {alpha}')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('$\\Delta_\\mathrm{obs} = \\hat{P}_1 - \\hat{P}_0$', fontweight='bold')
    ax.set_ylabel('$-\\log_{10}(p_\\mathrm{conj})$', fontweight='bold')
    ax.set_title('Volcano Plot: Discriminative Effect vs. Conjunction Significance',
                 fontweight='bold')
    legend_elems = [
        Patch(facecolor=COLORS['class1'], edgecolor='black', label='Significant Positive'),
        Patch(facecolor=COLORS['class0'], edgecolor='black', label='Significant Negative'),
        Patch(facecolor=COLORS['neither'], edgecolor='black', label='Non-significant'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True,
              fancybox=False, edgecolor='black')
    for s in ax.spines.values():
        s.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, '03_volcano_plot.pdf')

    # ========================================================================
    # PLOT 4: P-value Histograms (3 panels)
    # ========================================================================
    print("   [4/10] P-value Histograms...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, pvals, title, color in [
        (axes[0], [r.p_structural_dominant for r in results],
         '$p_\\mathrm{struct}$ (dominant)', COLORS['structural']),
        (axes[1], [r.p_discriminative for r in results],
         '$p_\\mathrm{disc}$ (label perm)', COLORS['discriminative']),
        (axes[2], [r.p_conjunction for r in results],
         '$p_\\mathrm{conj}$ (IUT)', COLORS['conjunction']),
    ]:
        ax.hist(pvals, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.8)
        ax.axvline(alpha, color=COLORS['threshold'], linestyle='--', linewidth=2)
        ax.set_xlabel('P-value', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        for sp in ax.spines.values():
            sp.set_visible(True)

    plt.tight_layout()
    save_plot_pdf(fig, '04_pvalue_histograms.pdf')

    # ========================================================================
    # PLOT 5: Top Significant Patterns — Horizontal Bar
    # ========================================================================
    print("   [5/10] Top Significant Patterns Bar Chart...")

    top_n = min(25, len(sig))
    if top_n > 0:
        top_sig = sorted(sig, key=lambda x: x.p_conjunction)[:top_n]

        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
        labels = []
        neg_log_ps = []
        bar_colors = []

        for r in top_sig:
            lab = f"{r.constraint_type[:12]}"
            if r.activity_b:
                lab += f"\n{r.activity_a[:20]}→{r.activity_b[:18]}"
            else:
                lab += f"\n{r.activity_a[:30]}"
            labels.append(lab)
            neg_log_ps.append(-np.log10(max(r.p_conjunction, 1e-300)))
            bar_colors.append(COLORS['class1'] if r.direction == "Positive" else COLORS['class0'])

        y = np.arange(len(labels))
        bars = ax.barh(y, neg_log_ps, color=bar_colors, edgecolor='black', linewidth=0.8)
        for bar, nlp in zip(bars, neg_log_ps):
            ax.text(nlp + max(neg_log_ps) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{nlp:.2f}', va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('$-\\log_{10}(p_\\mathrm{conj})$', fontweight='bold')
        ax.invert_yaxis()
        legend_elems = [
            Patch(facecolor=COLORS['class1'], edgecolor='black', label='Positive (class 1)'),
            Patch(facecolor=COLORS['class0'], edgecolor='black', label='Negative (class 0)'),
        ]
        ax.legend(handles=legend_elems, loc='lower right', frameon=True,
                  fancybox=False, edgecolor='black')
        for sp in ax.spines.values():
            sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, '05_top_patterns_barh.pdf')
    else:
        print("      ⚠️  No significant patterns to plot")

    # ========================================================================
    # PLOT 6: Cross-Class Prevalence Scatter
    # ========================================================================
    print("   [6/10] Cross-Class Prevalence Scatter...")

    fig, ax = plt.subplots(figsize=(9, 9))
    prev0 = [r.prevalence_class0 for r in results]
    prev1 = [r.prevalence_class1 for r in results]
    is_sig_list = [r.is_significant_bh for r in results]
    dirs = [r.direction for r in results]

    for i, r in enumerate(results):
        if r.is_significant_bh:
            c = COLORS['class1'] if r.direction == "Positive" else COLORS['class0']
            ax.scatter(prev0[i], prev1[i], c=c, s=50, alpha=0.7,
                       edgecolors='black', linewidths=0.5)
        else:
            ax.scatter(prev0[i], prev1[i], c=COLORS['neither'], s=10, alpha=0.3,
                       edgecolors='none')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='$\\hat{P}_0 = \\hat{P}_1$')
    ax.set_xlabel('$\\hat{P}_0$ (Granted)', fontweight='bold')
    ax.set_ylabel('$\\hat{P}_1$ (Not-Granted)', fontweight='bold')
    ax.set_title('Cross-Class Prevalence', fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    for sp in ax.spines.values():
        sp.set_visible(True)
    plt.tight_layout()
    save_plot_pdf(fig, '06_cross_class_prevalence.pdf')

    # ========================================================================
    # PLOT 7: Constraint Type Breakdown (Significant)
    # ========================================================================
    print("   [7/10] Constraint Type Breakdown...")

    ct_counts = Counter()
    ct_pos = Counter()
    ct_neg = Counter()
    for r in sig:
        ct_counts[r.constraint_type] += 1
        if r.direction == "Positive":
            ct_pos[r.constraint_type] += 1
        else:
            ct_neg[r.constraint_type] += 1

    if ct_counts:
        cts = sorted(ct_counts.keys())
        x = np.arange(len(cts))
        w = 0.7

        fig, ax = plt.subplots(figsize=(12, 6))
        pos_vals = [ct_pos.get(ct, 0) for ct in cts]
        neg_vals = [ct_neg.get(ct, 0) for ct in cts]
        ax.bar(x, pos_vals, w, label='Positive (class 1)',
               color=COLORS['class1'], edgecolor='black', linewidth=1.2)
        ax.bar(x, neg_vals, w, bottom=pos_vals, label='Negative (class 0)',
               color=COLORS['class0'], edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(cts, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Significant Patterns', fontweight='bold')
        ax.set_title('Significant Patterns by Constraint Type and Direction', fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        for sp in ax.spines.values():
            sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, '07_constraint_type_breakdown.pdf')

    # ========================================================================
    # PLOT 8: Four-Category Verdict Distribution
    # ========================================================================
    print("   [8/10] Four-Category Verdict Distribution...")

    cat_labels = ['Both', 'Structural\nonly', 'Discriminative\nonly', 'Neither']
    cat_vals = [
        sum(1 for r in results if r.significance_category == "Both"),
        sum(1 for r in results if r.significance_category == "Structural only"),
        sum(1 for r in results if r.significance_category == "Discriminative only"),
        sum(1 for r in results if r.significance_category == "Neither"),
    ]
    cat_colors = [COLORS['conjunction'], COLORS['structural'], COLORS['discriminative'], COLORS['neither']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute counts
    bars = axes[0].bar(cat_labels, cat_vals, color=cat_colors, edgecolor='black', linewidth=1.5)
    for bar, v in zip(bars, cat_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v,
                     f'{v}\n({v/len(results)*100:.1f}%)',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Number of Patterns', fontweight='bold')
    axes[0].set_title('Four-Category Significance Verdict', fontweight='bold')
    for sp in axes[0].spines.values():
        sp.set_visible(True)

    # Right: q-value scatter — structural vs discriminative
    q_s = np.array([r.q_structural_dominant for r in results])
    q_d = np.array([r.q_value_sam for r in results])
    alpha_val = CONFIG['fdr_alpha']
    scatter_colors = [
        COLORS['conjunction'] if r.significance_category == "Both" else
        COLORS['structural'] if r.significance_category == "Structural only" else
        COLORS['discriminative'] if r.significance_category == "Discriminative only" else
        COLORS['neither']
        for r in results
    ]
    axes[1].scatter(q_d, q_s, c=scatter_colors, s=15, alpha=0.5, edgecolors='none')
    axes[1].axvline(alpha_val, color=COLORS['threshold'], linestyle='--', linewidth=1.5, label=f'α={alpha_val}')
    axes[1].axhline(alpha_val, color=COLORS['threshold'], linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('$q_\\mathrm{disc}$ (Fisher-Storey)', fontweight='bold')
    axes[1].set_ylabel('$q_\\mathrm{struct}$ (Storey per class)', fontweight='bold')
    axes[1].set_title('Q-value Landscape: Structural vs Discriminative', fontweight='bold')
    axes[1].set_xlim(-0.02, 1.05)
    axes[1].set_ylim(-0.02, 1.05)
    legend_patches = [
        Patch(facecolor=COLORS['conjunction'],    label='Both'),
        Patch(facecolor=COLORS['structural'],      label='Structural only'),
        Patch(facecolor=COLORS['discriminative'],  label='Discriminative only'),
        Patch(facecolor=COLORS['neither'],         label='Neither'),
    ]
    axes[1].legend(handles=legend_patches, loc='upper right', frameon=True,
                   fancybox=False, edgecolor='black', fontsize=9)
    for sp in axes[1].spines.values():
        sp.set_visible(True)

    plt.tight_layout()
    save_plot_pdf(fig, '08_four_category_verdict.pdf')

    # ========================================================================
    # PLOT 9: Execution Time Breakdown
    # ========================================================================
    print("   [9/10] Execution Time Breakdown...")

    if timing:
        fig, ax = plt.subplots(figsize=(10, 6))
        comps = [(k, v) for k, v in timing.items() if k != 'total']
        comps.sort(key=lambda x: x[1], reverse=True)
        labels_t = [c[0].replace('_', '\n') for c in comps]
        vals_t = [c[1] for c in comps]
        colors_t = plt.cm.Set2(np.linspace(0, 1, len(comps)))

        y = np.arange(len(labels_t))
        bars = ax.barh(y, vals_t, color=colors_t, edgecolor='black', linewidth=1.2)
        for bar, v in zip(bars, vals_t):
            pct = v / timing.get('total', sum(vals_t)) * 100
            ax.text(v + max(vals_t) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}s ({pct:.0f}%)', va='center', fontsize=10, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels_t, fontsize=10)
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_title(f'Execution Time (total: {timing.get("total", 0):.1f}s)', fontweight='bold')
        for sp in ax.spines.values():
            sp.set_visible(True)
        plt.tight_layout()
        save_plot_pdf(fig, '09_execution_time.pdf')

    # ========================================================================
    # PLOT 10: Summary Dashboard
    # ========================================================================
    print("   [10/10] Summary Dashboard...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Four-category counts
    ax1 = fig.add_subplot(gs[0, 0])
    n_pos = sum(1 for r in sig if r.direction == "Positive")
    n_neg = sum(1 for r in sig if r.direction == "Negative")
    n_both     = len(sig)
    n_str_only = sum(1 for r in results if r.significance_category == "Structural only")
    n_dis_only = sum(1 for r in results if r.significance_category == "Discriminative only")
    cats  = ['Both', 'Struct\nonly', 'Disc\nonly', 'Neither']
    vals  = [n_both, n_str_only, n_dis_only, len(results) - n_both - n_str_only - n_dis_only]
    cols  = [COLORS['conjunction'], COLORS['structural'], COLORS['discriminative'], COLORS['neither']]
    bars = ax1.bar(cats, vals, color=cols, edgecolor='black', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                 ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Four-Category Verdict', fontweight='bold', fontsize=11)
    for sp in ax1.spines.values():
        sp.set_visible(True)

    # Panel 2: Class sizes
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(['Class 0\n(Granted)', 'Class 1\n(Not-Granted)'],
                   [len(D_0), len(D_1)],
                   color=[COLORS['class0'], COLORS['class1']],
                   edgecolor='black', linewidth=1.5)
    for bar, v in zip(bars, [len(D_0), len(D_1)]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                 ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Cases', fontweight='bold')
    ax2.set_title('Class Sizes', fontweight='bold', fontsize=11)
    for sp in ax2.spines.values():
        sp.set_visible(True)

    # Panel 3: Resample counts
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(['$B_1$ (Label)', '$B_2$ (Trace)'],
                   [CONFIG['B_label'], CONFIG['B_trace']],
                   color=[COLORS['discriminative'], COLORS['structural']],
                   edgecolor='black', linewidth=1.5)
    for bar, v in zip(bars, [CONFIG['B_label'], CONFIG['B_trace']]):
        ax3.text(bar.get_x() + bar.get_width() / 2, v, f'{v:,}',
                 ha='center', va='bottom', fontweight='bold')
    ax3.set_ylabel('Resamples', fontweight='bold')
    ax3.set_title('Permutation Budget', fontweight='bold', fontsize=11)
    for sp in ax3.spines.values():
        sp.set_visible(True)

    # Panel 4: Text summary
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    n_bh_ref = sum(1 for r in results if r.is_significant_bh)
    summary_text = f"""
    THREE-HYPOTHESIS DISCRIMINATIVE SPECIFICATION MINING — SEPSIS SUMMARY
    ══════════════════════════════════════════════════════════════════════════════

    Framework: Dual-axis Storey FDR (structural + discriminative, Step 5b/5c)
    P-values:  Phipson & Smyth (2010) exact permutation p-values
    FDR:       Storey (2002) q-values at α = {CONFIG['fdr_alpha']}
               BH-{CONFIG['fdr_method']} retained for comparison

    Null Hypotheses:
      H₀ˢ: No temporal structure (trace-activity permutation, B₂ = {CONFIG['B_trace']:,})
      H₀ᵈ: No class difference   (label permutation, B₁ = {CONFIG['B_label']:,})

    FDR Control:
      Structural:    Storey q-values on m={len(results):,} structural p-values (per class)
      Discriminative: Storey q-values on m\'={sum(1 for r in results if r.q_value_sam < 1.0):,} Fisher conjunction p-values

    Dataset: {len(case_data):,} cases (Class 0 No-Return-ER: {len(D_0):,}, Class 1 Return-ER: {len(D_1):,})
    Patterns Tested:             {len(results):,}
    Both  (struct ∧ disc):       {n_both:,} ({n_both/len(results)*100:.1f}%)  — Positive: {n_pos}, Negative: {n_neg}
    Structural only:             {n_str_only:,} ({n_str_only/len(results)*100:.1f}%)
    Discriminative only:         {n_dis_only:,} ({n_dis_only/len(results)*100:.1f}%)
    BH-FDR (reference):          {n_bh_ref:,} ({n_bh_ref/len(results)*100:.1f}%)

    Total Execution Time: {timing.get('total', 0):.1f}s
    """

    ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

    plt.tight_layout()
    save_plot_pdf(fig, '10_summary_dashboard.pdf')

    print(f"\n   ✓ Generated 10 visualizations in {PLOTS_DIR}")


# ============================================================================
# IMPORTABLE PIPELINE FUNCTION
# ============================================================================

def execute_pipeline(input_file=None, config=None):
    """
    Single entry point for Phase 1 three-hypothesis execution.

    Args:
        input_file: Path to CSV event log (default: global INPUT_FILE)
        config:     Override CONFIG dict (default: global CONFIG)

    Returns:
        Dictionary with all results.
    """
    global CONFIG

    if input_file is None:
        input_file = INPUT_FILE

    original_config = CONFIG.copy()
    if config is not None:
        CONFIG.update(config)

    try:
        case_data = load_and_preprocess_data(input_file)
        candidates_pos, candidates_neg = generate_candidate_patterns(case_data)
        case_ids_sorted = sorted(case_data.keys())
        labels = np.array([case_data[cid].outcome for cid in case_ids_sorted])
        pattern_results, timing, internals = execute_three_hypothesis_protocol(
            case_data, candidates_pos, candidates_neg,
            case_ids_sorted=case_ids_sorted,
            labels=labels,
        )
        return {
            'pattern_results':   pattern_results,
            'timing':            timing,
            'case_data':         case_data,
            'candidates_pos':    candidates_pos,
            'candidates_neg':    candidates_neg,
            'null_delta_matrix': internals['null_delta_matrix'],
            'holds_all':         internals['holds_all'],
            'delta_obs':         internals['delta_obs'],
            'candidates_all':    internals['candidates_all'],
            'tf_null_matrix': internals['tf_null_matrix'],
        }
    finally:
        CONFIG.clear()
        CONFIG.update(original_config)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("THREE-HYPOTHESIS DISCRIMINATIVE SPECIFICATION MINING — PHASE 1")
    print("Storey CONJUNCTION TEST")
    print("=" * 100)
    print("\n🎯 SCIENTIFIC FRAMEWORK:")
    print("   H₀ˢ: Structural null — trace-activity permutation within each class")
    print("   H₀ᵈ: Discriminative null — label permutation across cases")
    print("   H₀ᶜ: IUT at q-value level (Berger 1982) — q_value_sam ≤ α ∧ q_structural_dominant ≤ α")
    print("   P-values: Phipson & Smyth (2010) exact formula")
    print(f"   FDR: Storey (2002) Q-Value at α = {CONFIG['fdr_alpha']}  [per-pattern, replaces Tusher flat-null]")
    print(f"        BH-{CONFIG['fdr_method']} retained as reference comparison")
    print("=" * 100)

    start = time.time()

    output = execute_pipeline()
    pattern_results = output['pattern_results']
    timing = output['timing']
    case_data = output['case_data']

    generate_outputs(pattern_results, case_data, timing)

    total = time.time() - start

    sig_final = [r for r in pattern_results if r.is_significant_final]
    sig_bh    = [r for r in pattern_results if r.is_significant_bh]
    n_pos = sum(1 for r in sig_final if r.direction == "Positive")
    n_neg = sum(1 for r in sig_final if r.direction == "Negative")

    print(f"\n{'='*100}")
    print("✅ THREE-HYPOTHESIS PHASE 1 COMPLETE")
    print(f"{'='*100}")
    print(f"\n⏱️  Total: {total:.1f}s ({total/60:.1f} min)")
    print(f"\n📁 Outputs: {OUTPUT_DIR}")
    print(f"   • three_hypothesis_samfdr_results.json")
    print(f"   • significant_patterns_only.json")
    print(f"   • discrimination_metrics.json")
    print(f"   • three_hypothesis_sam_report.txt")
    print(f"   • visualizations/")
    print(f"\n📊 KEY RESULTS (Storey, primary):")
    print(f"   Patterns tested:              {len(pattern_results):,}")
    print(f"   SAM ∧ structural (final):     {len(sig_final):,} ({len(sig_final)/len(pattern_results)*100:.1f}%)")
    print(f"   Positive (class 1 dominant):  {n_pos:,}")
    print(f"   Negative (class 0 dominant):  {n_neg:,}")
    print(f"   BH-FDR rejections (ref.):     {len(sig_bh):,} ({len(sig_bh)/len(pattern_results)*100:.1f}%)")
    print(f"\n   B₁ (label perm):  {CONFIG['B_label']:,}")
    print(f"   B₂ (trace perm):  {CONFIG['B_trace']:,}")
    print(f"   FDR α:            {CONFIG['fdr_alpha']}")
    print(f"   BH ref. method:   {CONFIG['fdr_method']}")

    # Key scientific advantage: conjunction test suppresses non-discriminative Not*** patterns
    n_not = sum(1 for r in pattern_results
                if r.constraint_type.startswith('Not') and r.is_significant_final)
    n_not_total = sum(1 for r in pattern_results if r.constraint_type.startswith('Not'))
    print(f"\n📊 Not*** SUPPRESSION CHECK:")
    print(f"   Not*** patterns tested:     {n_not_total:,}")
    print(f"   Not*** patterns significant:{n_not:,} "
          f"({n_not/n_not_total*100:.1f}% of Not*** patterns)" if n_not_total > 0 else "")
    print(f"   → Conjunction test naturally suppresses non-discriminative Not*** patterns")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()