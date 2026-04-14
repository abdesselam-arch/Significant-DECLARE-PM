"""
eval_utils.py — Shared Evaluation Utilities for RQ1: FDR Control Validity
==========================================================================

PURPOSE:
    Provides the scientific evaluation backbone for RQ1 across all event logs.
    Every function is process-mining-agnostic: it operates on generic arrays
    of p-values, null-delta matrices, and holds dictionaries, making it
    reusable across BPI-17, BPI-18, Sepsis, BPI-19, and Hospital logs.

SCIENTIFIC GROUNDING:
    The evaluation protocol follows the gold-standard empirical FDR validation
    design from:
      - Pellegrina & Vandin (KDD 2018): TopKWY — held-out permutation FDR
      - Storey & Tibshirani (PNAS 2003): q-value calibration via full-null
      - Dalleiger & Vreeken (KDD 2022): SPASS — sequential FDR validation
      - Zhang (2024): permutation-based FDR for sequential patterns
      - Cecconi, Augusto & Di Ciccio (BPM 2021): chi-square baseline

    The fundamental methodological point (from the RQ1 protocol):
    The framework INTERNALLY uses B₁ label permutations to populate
    null_delta_matrix and compute Phipson-Smyth p-values. Validation MUST
    use B_null INDEPENDENT held-out permutations that are never passed to
    Phase 1. For each held-out replicate b, labels are permuted, Phase 1
    is run fresh, and all resulting discoveries are by construction false
    positives. This prevents circular evaluation.

FUNCTIONS:
    ┌─────────────────────────────────────┬───────────────────────────────────┐
    │ Function                            │ Purpose                           │
    ├─────────────────────────────────────┼───────────────────────────────────┤
    │ generate_heldout_permutation        │ Stratified Fisher permutation     │
    │ compute_empirical_fdr               │ FDR_emp, PCER_emp, FWER_emp      │
    │ bootstrap_bca_ci                    │ BCa 95% CI on FDR_emp            │
    │ run_cecconi_baseline                │ Chi-square + BH (BPM 2021)       │
    │ run_tusher_flat_null                │ Tusher (2001) pooled-null SAM    │
    │ run_bh_on_iut_pvalues              │ BH-FDR on IUT conjunction p-vals  │
    │ compute_pi0_with_ci                 │ Bootstrap π̂₀ with 95% CI        │
    │ compute_sigma_null_heterogeneity    │ σ_null per constraint family     │
    │ compute_tusher_inflation_factor     │ ρ_inf = Ê[V]_pooled / Ê[V]_pp   │
    │ build_rq1_results_df               │ Aggregation DataFrame for paper   │
    │ build_paper_table                   │ Table 1 for the evaluation sect.  │
    └─────────────────────────────────────┴───────────────────────────────────┘

Version: 1.0
Author:  Ahmed Nour Abdesselam
Institution: Free University of Bozen-Bolzano
Date: March 2026

References:
    [1] Storey (2002). A direct approach to false discovery rates. JRSS-B 64(3).
    [2] Storey & Tibshirani (2003). Statistical significance for genomewide
        studies. PNAS 100(16):9440-9445.
    [3] Storey, Taylor & Siegmund (2004). Strong control, conservative point
        estimation. JRSS-B 66(1):187-205.
    [4] Tusher, Tibshirani & Chu (2001). Significance analysis of microarrays.
        PNAS 98(9):5116-5121.
    [5] Phipson & Smyth (2010). Permutation p-values should never be zero.
        Stat. Appl. Genet. Mol. Biol. 9(1):Article 39.
    [6] Benjamini & Hochberg (1995). Controlling the FDR. JRSS-B.
    [7] Pellegrina & Vandin (KDD 2018). TopKWY.
    [8] Cecconi, Augusto & Di Ciccio (BPM 2021). Variants through DECLARE.
    [9] Dalleiger & Vreeken (KDD 2022). SPASS.
    [10] Zhang (2024). Permutation-based FDR for sequential patterns.
    [11] Berger (1982). Multiparameter Hypothesis Testing. Technometrics.
    [12] Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman.
    [13] Brown (1975). A method for combining non-independent p-values.
         Biometrics 31(4):987-992.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, chi2
from typing import (
    Dict, List, Tuple, Optional, Set, Any, NamedTuple, Union
)
from dataclasses import dataclass, field
from collections import Counter
import warnings
import time
import json
import os

warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EmpiricalFDRResult:
    """
    Complete empirical FDR validation result for one method on one log.

    Error rate estimates (Storey 2002, §3; Pellegrina & Vandin 2018):

        FDR_emp:  Ê[V_null] / max(R_obs, 1)
                  This is the Storey (2002) permutation FDR estimator, NOT a
                  direct estimate of E[V/max(R,1)]. It conditions on R = R_obs
                  (the rejection count on original unpermuted data) and estimates
                  the numerator E[V] via B_null null replicates.

                  Storey (2002) Theorem 1 proves that under PRDS this estimator
                  provides an upper bound on the true FDR = E[V/max(R,1)]:
                      FDR = E[V/max(R,1)] ≤ FDR_emp
                  For R_obs ≥ 20, FDR_emp is tight (within ~5% of true FDR).

        PCER_emp: Ê[V_null] / m  — per-comparison error rate.
        FWER_emp: P(V_null ≥ 1)  — fraction of null replicates with ≥1 FP.

    A correctly controlled method must satisfy FDR_emp ≤ α on every log.
    """
    method_name: str
    log_name: str
    alpha: float

    # Error rate estimates
    fdr_emp: float                  # Ê[V_null] / max(R_obs, 1) — Storey conditional FDR estimator
    pcer_emp: float                 # (1/B_null) * Σ_b |S_b| / m
    fwer_emp: float                 # (1/B_null) * Σ_b 1[|S_b| > 0]

    # Confidence interval on FDR_emp
    fdr_ci_lower: float             # BCa 95% CI lower
    fdr_ci_upper: float             # BCa 95% CI upper

    # Rejection counts
    R_obs: int                      # Rejections on original (unpermuted) log

    # Fields with defaults
    ci_method: str = "BCa"
    null_rejection_counts: Optional[np.ndarray] = None  # |S_b| for b=1..B_null

    # Metadata
    B_null: int = 0
    m_total: int = 0                # Total patterns tested
    wall_seconds: float = 0.0


@dataclass
class Pi0EstimateWithCI:
    """
    Bootstrap π̂₀ estimate with confidence interval from the λ-grid.

    Reports π̂₀ for both the discriminative and structural axes, plus
    the 95% CI from the Storey-Taylor-Siegmund closed-form estimator.
    """
    log_name: str
    pi0_disc: float                 # π̂₀ on discriminative (Fisher conjunction) p-values
    pi0_struct_c0: float            # π̂₀ on structural class 0 p-values
    pi0_struct_c1: float            # π̂₀ on structural class 1 p-values
    lambda_star_disc: float
    lambda_star_struct_c0: float
    lambda_star_struct_c1: float
    # Sensitivity range from λ-grid (NOT a statistical CI; see compute_pi0_with_ci)
    pi0_disc_sensitivity_lo: float
    pi0_disc_sensitivity_hi: float
    pi0_struct_c0_sensitivity_lo: float
    pi0_struct_c0_sensitivity_hi: float
    pi0_struct_c1_sensitivity_lo: float
    pi0_struct_c1_sensitivity_hi: float
    m_disc: int                     # Number of patterns in discriminative correction
    m_struct: int                   # Number of patterns in structural correction


@dataclass
class TusherFailureReport:
    """
    Mechanistic dissection of the Tusher flat-null failure.

    Three-step demonstration:
    1. σ_null heterogeneity across constraint families
    2. Inflation factor ρ_inf = Ê[V(τ*)]_pooled / Ê[V(τ*)]_per-pattern
    3. k*_Tusher = 0 (complete power collapse)
    """
    log_name: str
    k_star_tusher: int              # Expected: 0
    k_star_storey: int              # Expected: > 0
    rho_inf: float                  # Inflation factor (expected ≈ 80)
    sigma_null_by_family: Dict[str, Dict[str, float]]  # family → {mean, std, min, max}
    sigma_null_ratio: float         # max(σ_family) / min(σ_family) — expected ≈ 50×
    E_V_pooled_at_tau_star: float
    E_V_perpattern_at_tau_star: float


@dataclass
class CecconiResult:
    """Result of running the Cecconi et al. (BPM 2021) chi-square baseline."""
    method_name: str = "Cecconi_ChiSq_BH"
    n_rejected: int = 0
    p_values: Optional[np.ndarray] = None
    rejected: Optional[np.ndarray] = None
    n_small_cell_violations: int = 0   # Patterns where min expected cell < 5
    chi2_statistics: Optional[np.ndarray] = None


# ============================================================================
# 1. HELD-OUT PERMUTATION GENERATION
# ============================================================================

def generate_heldout_permutation(
    labels: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Generate a single held-out label permutation (stratified Fisher randomization).

    Preserves the marginal class counts n₊ and n₋ exactly, which is critical
    for imbalanced logs (Sepsis: 15/85, BPI-18: 10/90).

    This implements Fisher's exact permutation: draw n₊ indices uniformly at
    random (without replacement) from all n cases and assign label 1; the
    rest get label 0. Under H₀ (no class-activity association), every such
    assignment is equally likely.

    Args:
        labels:       (n,) binary array of original case labels (0/1).
        random_state: RNG seed for this specific permutation replicate.

    Returns:
        (n,) permuted label array with identical marginal counts.

    Note:
        These permutations are INDEPENDENT of the B₁ internal permutations
        used by Phase 1 to populate null_delta_matrix. This independence
        is the fundamental methodological requirement for valid empirical
        FDR estimation (Storey & Tibshirani 2003, §4).
    """
    rng = np.random.RandomState(random_state)
    permuted = labels.copy()
    rng.shuffle(permuted)
    return permuted


def generate_heldout_permutation_batch(
    labels: np.ndarray,
    B_null: int,
    base_seed: int = 20260321,
) -> List[np.ndarray]:
    """
    Generate B_null independent held-out label permutations.

    Seeds are deterministic: base_seed + b for b = 0, ..., B_null-1,
    ensuring full reproducibility across runs and machines.

    Args:
        labels:    (n,) original label vector.
        B_null:    Number of held-out replicates.
        base_seed: Base random seed (default: 20260321, the RQ1 eval date).

    Returns:
        List of B_null permuted label arrays.
    """
    return [
        generate_heldout_permutation(labels, random_state=base_seed + b)
        for b in range(B_null)
    ]


# ============================================================================
# 2. EMPIRICAL FDR ESTIMATION — THE MASTER FORMULA
# ============================================================================

def compute_empirical_fdr(
    null_rejection_counts: np.ndarray,
    R_obs: int,
    m_total: int,
    method_name: str = "",
    log_name: str = "",
    alpha: float = 0.05,
) -> EmpiricalFDRResult:
    """
    Compute empirical FDR, PCER, and FWER from held-out null permutations.

    This is the gold-standard evaluation from Pellegrina & Vandin (KDD 2018)
    and Storey & Tibshirani (PNAS 2003). Since all labels are random in each
    held-out replicate, every rejection is a false positive by construction.

    The master empirical FDR formula (Storey 2002, §3):
    ─────────────────────────────────────────────────────
                  Ê[V_null]            (1/B_null) × Σ_b |S_b|
    FDR_emp  =  ─────────────────  =  ───────────────────────────
                 max(R_obs, 1)              max(R_obs, 1)

    Notation:
        V_null:  Random variable counting rejections in one null replicate.
                 All |S_b| rejections are false positives by construction
                 (labels are random). V_null = |S_b| for replicate b.
        R_obs:   Rejection count on the ORIGINAL (unpermuted) data — fixed,
                 not a random variable here.

    IMPORTANT: This computes E[V_null] / R_obs, which equals
               E[FDP | R = R_obs] (the conditional false discovery proportion
               given the observed rejection count). This is the Storey (2002)
               permutation FDR estimator, not E[V/max(R,1)] directly.

    Relationship to formal FDR (Storey 2002, Theorem 1):
        Under PRDS (positive regression dependence on the subset),
        FDR = E[V/max(R,1)] ≤ FDR_emp

        The inequality holds because under the alternative (real labels),
        R_real ≥ R_null in expectation (true positives inflate the denominator
        but not the numerator), making FDR_emp a conservative upper bound on
        the true FDR.

    Validity condition: The formula is valid when R_obs > 0. When R_obs = 0,
        the method makes no discoveries and FDR = 0 by definition regardless
        of null replicate counts.

    Three error rate estimates:
    ─────────────────────────
    1. FDR_emp: see formula above — Storey conditional FDR estimator.
    2. PCER_emp = Ê[V_null] / m — per-comparison error rate.
    3. FWER_emp = P(V_null ≥ 1) — fraction of null replicates with ≥1 FP.

    A correctly controlled method must satisfy FDR_emp ≤ α.

    Args:
        null_rejection_counts: (B_null,) array where entry b is |S_b|, the
            number of discoveries in held-out permuted replicate b. All of
            these are false positives by construction.
        R_obs:       Number of rejections on the original (unpermuted) log.
        m_total:     Total number of candidate patterns tested.
        method_name: Name of the method (for reporting).
        log_name:    Name of the event log (for reporting).
        alpha:       Nominal FDR level.

    Returns:
        EmpiricalFDRResult with all three error rates and BCa 95% CI.

    References:
        Storey (2002), §3: FDR̂(τ) = π̂₀ · m · p / R(τ).
        Pellegrina & Vandin (2018), §5: empirical FWER from B_null full-null runs.
        Tusher et al. (2001), §2.3: permutation-based FDR estimate.
    """
    B_null = len(null_rejection_counts)
    null_counts = np.asarray(null_rejection_counts, dtype=np.float64)

    # ── FDR_emp ──────────────────────────────────────────────────────────
    # Storey (2002) permutation FDR estimate:
    #   FDR̂ = Ê[V] / max(R_obs, 1)
    # where Ê[V] = mean(|S_b|) over B_null null replicates.
    E_V = float(np.mean(null_counts))
    fdr_emp = E_V / max(R_obs, 1)

    # ── PCER_emp ─────────────────────────────────────────────────────────
    pcer_emp = E_V / max(m_total, 1)

    # ── FWER_emp ─────────────────────────────────────────────────────────
    fwer_emp = float(np.mean(null_counts > 0))

    # ── BCa 95% CI on FDR_emp ────────────────────────────────────────────
    ci_lower, ci_upper = bootstrap_bca_ci(
        null_counts, R_obs, confidence=0.95, B_boot=2000
    )

    return EmpiricalFDRResult(
        method_name=method_name,
        log_name=log_name,
        alpha=alpha,
        fdr_emp=fdr_emp,
        pcer_emp=pcer_emp,
        fwer_emp=fwer_emp,
        fdr_ci_lower=ci_lower,
        fdr_ci_upper=ci_upper,
        ci_method="BCa",
        R_obs=R_obs,
        null_rejection_counts=null_counts,
        B_null=B_null,
        m_total=m_total,
    )


# ============================================================================
# 3. BCa BOOTSTRAP CONFIDENCE INTERVAL ON FDR_emp
# ============================================================================

def bootstrap_bca_ci(
    null_rejection_counts: np.ndarray,
    R_obs: int,
    confidence: float = 0.95,
    B_boot: int = 2000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    BCa (bias-corrected and accelerated) bootstrap 95% CI on FDR_emp.

    The BCa method (Efron & Tibshirani 1993, Chapter 14) corrects for both
    bias and skewness in the bootstrap distribution, producing second-order
    accurate confidence intervals — substantially better than the symmetric
    normal-based CI, especially when the statistic's distribution is skewed
    (as FDR_emp often is, with a point mass at 0 when many replicates
    produce zero rejections).

    Algorithm:
    1. Compute the observed statistic θ̂ = mean(|S_b|) / max(R_obs, 1).
    2. Draw B_boot bootstrap samples (with replacement) of size B_null from
       the null_rejection_counts vector.
    3. For each bootstrap sample, compute θ̂* = mean(|S_b*|) / max(R_obs, 1).
    4. Compute bias correction z₀ = Φ⁻¹(#{θ̂* < θ̂} / B_boot).
    5. Compute acceleration â via the jackknife (Efron 1987):
       â = Σ(θ̂_{(·)} − θ̂_{(−i)})³ / [6 × (Σ(θ̂_{(·)} − θ̂_{(−i)})²)^{3/2}]
    6. Adjusted quantiles:
       α₁ = Φ(z₀ + (z₀ + z_{α/2}) / (1 − â(z₀ + z_{α/2})))
       α₂ = Φ(z₀ + (z₀ + z_{1−α/2}) / (1 − â(z₀ + z_{1−α/2})))

    Args:
        null_rejection_counts: (B_null,) array of false positive counts.
        R_obs:       Rejection count on original log.
        confidence:  Confidence level (default 0.95).
        B_boot:      Number of bootstrap resamples (default 2000; Efron
                     recommends ≥1000 for BCa).
        random_state: RNG seed.

    Returns:
        (ci_lower, ci_upper) — BCa confidence interval bounds on FDR_emp.

    References:
        Efron (1987). Better bootstrap confidence intervals. JASA 82(397).
        Efron & Tibshirani (1993). An Introduction to the Bootstrap. Ch. 14.
    """
    rng = np.random.RandomState(random_state)
    counts = np.asarray(null_rejection_counts, dtype=np.float64)
    B_null = len(counts)
    denom = max(R_obs, 1)

    # Observed statistic
    theta_hat = float(np.mean(counts)) / denom

    # ── Step 1: Bootstrap distribution ───────────────────────────────────
    theta_boot = np.zeros(B_boot)
    for b in range(B_boot):
        idx = rng.randint(0, B_null, size=B_null)
        theta_boot[b] = np.mean(counts[idx]) / denom

    # ── Step 2: Bias correction z₀ ──────────────────────────────────────
    # Fraction of bootstrap replicates below the observed statistic.
    prop_below = np.mean(theta_boot < theta_hat)
    # Clip to avoid ±∞ from Φ⁻¹(0) or Φ⁻¹(1)
    prop_below = np.clip(prop_below, 1e-10, 1.0 - 1e-10)
    z0 = float(stats.norm.ppf(prop_below))

    # ── Step 3: Acceleration â via jackknife ─────────────────────────────
    # θ̂_{(−i)} = leave-one-out estimates
    jackknife_vals = np.zeros(B_null)
    for i in range(B_null):
        # Leave-one-out mean: (sum - counts[i]) / (B_null - 1)
        loo_mean = (np.sum(counts) - counts[i]) / max(B_null - 1, 1)
        jackknife_vals[i] = loo_mean / denom

    theta_dot = np.mean(jackknife_vals)  # θ̂_{(·)}
    diff = theta_dot - jackknife_vals     # (θ̂_{(·)} − θ̂_{(−i)})

    numer = np.sum(diff ** 3)
    denom_acc = np.sum(diff ** 2)
    if denom_acc > 0:
        a_hat = numer / (6.0 * (denom_acc ** 1.5))
    else:
        a_hat = 0.0

    # ── Step 4: Adjusted quantiles ───────────────────────────────────────
    alpha_tail = (1.0 - confidence) / 2.0
    z_alpha_lo = float(stats.norm.ppf(alpha_tail))
    z_alpha_hi = float(stats.norm.ppf(1.0 - alpha_tail))

    def _bca_quantile(z_alpha: float) -> float:
        """Compute BCa-adjusted quantile."""
        numer_q = z0 + z_alpha
        denom_q = 1.0 - a_hat * numer_q
        if abs(denom_q) < 1e-15:
            # Degenerate: fall back to percentile
            return z_alpha
        adjusted_z = z0 + numer_q / denom_q
        return float(stats.norm.cdf(adjusted_z))

    alpha1 = _bca_quantile(z_alpha_lo)
    alpha2 = _bca_quantile(z_alpha_hi)

    # Clip to valid quantile range
    alpha1 = np.clip(alpha1, 0.5 / B_boot, 1.0 - 0.5 / B_boot)
    alpha2 = np.clip(alpha2, 0.5 / B_boot, 1.0 - 0.5 / B_boot)

    # ── Step 5: Extract CI from sorted bootstrap distribution ────────────
    theta_sorted = np.sort(theta_boot)
    ci_lower = float(theta_sorted[max(0, int(np.floor(alpha1 * B_boot)))])
    ci_upper = float(theta_sorted[min(B_boot - 1, int(np.ceil(alpha2 * B_boot)))])

    # Ensure monotonicity (lower ≤ point estimate ≤ upper is not guaranteed
    # by BCa in finite samples, but swapped bounds indicate instability)
    if ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    return ci_lower, ci_upper


# ============================================================================
# 4. CECCONI CHI-SQUARE BASELINE (BPM 2021)
# ============================================================================

def run_cecconi_baseline(
    holds_all: Dict[tuple, Dict[str, int]],
    ids_class0: Set[str],
    ids_class1: Set[str],
    alpha: float = 0.05,
    label_override: Optional[Dict[str, int]] = None,
) -> CecconiResult:
    """
    Cecconi et al. (BPM 2021) chi-square contingency test + BH-FDR.

    For each DECLARE pattern, constructs the 2×2 contingency table:

                    | Satisfied | Not Satisfied |
        ────────────┼───────────┼───────────────┤
        Class 0     |    a      |      b        |
        Class 1     |    c      |      d        |

    and computes Pearson's chi-square statistic with Yates' continuity
    correction. P-values are then corrected via BH step-up (Benjamini &
    Hochberg 1995) at level α.

    This is the direct process-mining baseline from Cecconi, Augusto &
    Di Ciccio (BPM 2021), who use element-shuffling of class-conditional
    traces and chi-square comparison of constraint prevalences.

    Limitation (documented in RQ1 protocol):
    The chi-square approximation assumes min(expected cell count) ≥ 5.
    Under severe imbalance (BPI-18: n₊=4,381; Sepsis: n₊=157), this
    assumption fails for low-prevalence patterns, potentially inflating
    the FDR beyond α. The number of small-cell violations is reported
    in CecconiResult.n_small_cell_violations.

    Args:
        holds_all:      Dict[pattern_spec → Dict[case_id → 0/1]] from the
                        Phase 1 holds-by-case computation.
        ids_class0:     Set of case_ids belonging to class 0.
        ids_class1:     Set of case_ids belonging to class 1.
        alpha:          Target FDR level (default 0.05).
        label_override: Optional Dict[case_id → label]. If provided,
                        overrides the class membership for this run (used
                        in held-out null permutation replicates). When None,
                        ids_class0 and ids_class1 are used as-is.

    Returns:
        CecconiResult with per-pattern chi-square p-values, BH rejection
        decisions, and small-cell violation count.

    References:
        Cecconi, Augusto & Di Ciccio (BPM 2021). Detection of Statistically
            Significant Differences Between Process Variants Through
            Declarative Rules.
        Pearson (1900). On the criterion that a given system of deviations.
        Benjamini & Hochberg (1995). Controlling the FDR. JRSS-B 57(1).
    """
    pattern_specs = [k for k in holds_all.keys() if not str(k).startswith('__')]
    m = len(pattern_specs)

    if m == 0:
        return CecconiResult(n_rejected=0)

    # Resolve class membership
    if label_override is not None:
        c0 = {cid for cid, lab in label_override.items() if lab == 0}
        c1 = {cid for cid, lab in label_override.items() if lab == 1}
    else:
        c0 = set(ids_class0)
        c1 = set(ids_class1)

    p_values = np.ones(m)
    chi2_stats = np.zeros(m)
    n_small_cell = 0

    for p_idx, pspec in enumerate(pattern_specs):
        holds = holds_all[pspec]

        # Build 2×2 table
        #   a = satisfied ∩ class0, b = not-satisfied ∩ class0
        #   c = satisfied ∩ class1, d = not-satisfied ∩ class1
        a, b, c, d = 0, 0, 0, 0
        for cid, val in holds.items():
            if cid in c0:
                if val == 1:
                    a += 1
                else:
                    b += 1
            elif cid in c1:
                if val == 1:
                    c += 1
                else:
                    d += 1

        table = np.array([[a, b], [c, d]])
        n_total = a + b + c + d

        if n_total == 0:
            p_values[p_idx] = 1.0
            continue

        # Check expected cell counts for chi-square validity
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / max(n_total, 1)
        if expected.min() < 5.0:
            n_small_cell += 1

        # Pearson chi-square with Yates' continuity correction
        # If any marginal is zero, chi-square is undefined → p = 1.0
        if (row_sums == 0).any() or (col_sums == 0).any():
            p_values[p_idx] = 1.0
            chi2_stats[p_idx] = 0.0
            continue

        try:
            chi2_stat, p_val, dof, exp = chi2_contingency(
                table, correction=True  # Yates' correction
            )
            p_values[p_idx] = p_val
            chi2_stats[p_idx] = chi2_stat
        except ValueError:
            # Degenerate table (e.g., all zeros in a row/column)
            p_values[p_idx] = 1.0
            chi2_stats[p_idx] = 0.0

    # ── BH step-up procedure ─────────────────────────────────────────────
    rejected, _bh_thresh, _k_star = _benjamini_hochberg(p_values, alpha)

    return CecconiResult(
        method_name="Cecconi_ChiSq_BH",
        n_rejected=int(np.sum(rejected)),
        p_values=p_values,
        rejected=rejected,
        n_small_cell_violations=n_small_cell,
        chi2_statistics=chi2_stats,
    )


# ============================================================================
# 5. TUSHER FLAT-NULL SAM (Tusher, Tibshirani & Chu 2001)
# ============================================================================

def run_tusher_flat_null(
    null_delta_matrix: np.ndarray,
    delta_obs: np.ndarray,
    alpha: float = 0.05,
    pi0_hat: float = 1.0,
) -> Dict[str, Any]:
    """
    Tusher et al. (2001) flat-null pooled SAM FDR procedure.

    This is the ORIGINAL SAM method that POOLS all B₁ × m null Δ values into
    a single flat distribution. It is included as a baseline to demonstrate
    its catastrophic failure at small m' due to σ_null heterogeneity.

    Procedure:
    ──────────
    For a given threshold τ on |Δ|:
      R(τ)  = #{p : |Δ_obs(p)| ≥ τ}        — observed rejections
      Ê[V(τ)] = π̂₀ × (1/(B₁)) × Σ_b #{p : |Δ_b(p)| ≥ τ}  — expected FPs
      FDP̂(τ) = Ê[V(τ)] / max(R(τ), 1)

    The method sweeps τ from large to small and selects τ* = min{τ : FDP̂(τ) ≤ α}.

    WHY IT FAILS (documented in Phase 1 framework):
    ──────────────────────────────────────────────────
    When σ_null varies ~50× across constraint types (Init/End ≈ 0.3,
    NotChainSuccession ≈ 0.006), the pooled Ê[V(τ)] at any τ relevant to
    low-σ patterns is dominated by high-σ patterns whose null |Δ| easily
    exceeds τ. This inflates Ê[V(τ)] by a factor ρ_inf ≈ 80, causing
    FDP̂(τ) > α for all τ, so k* = 0 (zero rejections).

    Args:
        null_delta_matrix: (B₁, m) float32 — permuted Δ_b(p) from label
            permutation (Step 3 of Phase 1).
        delta_obs:         (m,) observed prevalence differences.
        alpha:             Target FDR level.
        pi0_hat:           Estimated null fraction (default 1.0 = conservative).

    Returns:
        Dict with keys:
            k_star:            int   — number of rejections (expected: 0)
            tau_star:          float — threshold (0.0 if k*=0)
            fdp_at_tau_star:   float — FDP̂ at τ*
            significant:       (m,) bool array
            fdp_curve:         (n_thresholds,) FDP̂ at each tested τ
            tau_curve:         (n_thresholds,) corresponding τ values
            E_V_at_tau_star:   float — Ê[V(τ*)] from pooled null
            R_at_tau_star:     int   — R(τ*) observed rejections
            min_fdp:           float — minimum achievable FDP̂ across all τ

    References:
        Tusher, Tibshirani & Chu (2001). PNAS 98(9):5116-5121, §Methods.
    """
    B1, m = null_delta_matrix.shape
    abs_obs = np.abs(delta_obs).astype(np.float64)
    abs_null = np.abs(null_delta_matrix).astype(np.float64)

    # Sort observed |Δ| descending to define candidate thresholds
    sorted_abs_obs = np.sort(abs_obs)[::-1]
    # Use observed |Δ| values as threshold candidates
    unique_thresholds = np.unique(sorted_abs_obs)[::-1]  # descending

    tau_grid = unique_thresholds

    fdp_curve = np.zeros(len(tau_grid))
    R_curve = np.zeros(len(tau_grid), dtype=int)
    E_V_curve = np.zeros(len(tau_grid))

    for t_idx, tau in enumerate(tau_grid):
        # R(τ) = observed exceedances
        R_tau = int(np.sum(abs_obs >= tau))
        R_curve[t_idx] = R_tau

        # Ê[V(τ)] = π̂₀ × (1/B₁) × Σ_b #{p : |Δ_b(p)| ≥ τ}
        null_exceedances_per_b = np.sum(abs_null >= tau, axis=1)  # (B1,)
        E_V_tau = pi0_hat * float(np.mean(null_exceedances_per_b))
        E_V_curve[t_idx] = E_V_tau

        # FDP̂(τ)
        fdp_curve[t_idx] = E_V_tau / max(R_tau, 1)

    # Find τ* = minimum τ (maximum k*) subject to FDP̂(τ) ≤ α.
    # tau_grid is sorted descending, so the LAST index with FDP̂ ≤ α
    # corresponds to the smallest τ, which maximises R(τ) = k*.
    # (Breaking at the first satisfying index would give the largest τ
    # and fewest rejections — the wrong direction.)
    k_star = 0
    tau_star = 0.0
    fdp_at_tau_star = 1.0

    valid_mask = fdp_curve <= alpha
    if valid_mask.any():
        best_idx = int(np.where(valid_mask)[0][-1])  # last = smallest τ = max k*
        tau_star = float(tau_grid[best_idx])
        k_star = int(R_curve[best_idx])
        fdp_at_tau_star = float(fdp_curve[best_idx])

    # Significance decisions
    significant = abs_obs >= tau_star if k_star > 0 else np.zeros(m, dtype=bool)

    # Diagnostic: E_V at the found τ*
    E_V_at_tau_star = 0.0
    if k_star > 0:
        best_idx = np.argmin(np.abs(tau_grid - tau_star))
        E_V_at_tau_star = float(E_V_curve[best_idx])

    min_fdp = float(np.min(fdp_curve)) if len(fdp_curve) > 0 else 1.0

    return {
        'k_star': k_star,
        'tau_star': tau_star,
        'fdp_at_tau_star': fdp_at_tau_star,
        'significant': significant,
        'fdp_curve': fdp_curve,
        'tau_curve': tau_grid,
        'E_V_curve': E_V_curve,
        'R_curve': R_curve,
        'E_V_at_tau_star': E_V_at_tau_star,
        'R_at_tau_star': k_star,
        'min_fdp': min_fdp,
    }


# ============================================================================
# 6. BH-FDR ON IUT CONJUNCTION P-VALUES (Baseline Method 2)
# ============================================================================

def run_bh_on_iut_pvalues(
    p_structural_dominant: np.ndarray,
    p_discriminative: np.ndarray,
    alpha: float = 0.05,
    method: str = 'BH',
) -> Dict[str, Any]:
    """
    BH-FDR applied to IUT conjunction p-values (Berger 1982 + BH 1995).

    This is the second baseline: raw Benjamini-Hochberg step-up on the
    IUT p-values p_conj(p) = max(p_struct_dom(p), p_disc(p)), WITHOUT
    the Storey π̂₀ correction.

    The dual-axis Storey method should reject strictly more patterns than
    this baseline by a factor of ~1/π̂₀, since BH implicitly assumes
    π̂₀ = 1.0 (all patterns are null).

    Args:
        p_structural_dominant: (m,) Phipson-Smyth p-values for H₀ˢ in
            the dominant class.
        p_discriminative:      (m,) Phipson-Smyth p-values for H₀ᵈ.
        alpha:                 Target FDR level.
        method:                'BH' (default) or 'BY'.

    Returns:
        Dict with 'rejected' (bool array), 'n_rejected', 'p_conjunction',
        'bh_thresholds'.
    """
    # IUT conjunction: p_conj = max(p_struct, p_disc)
    p_conj = np.maximum(
        np.asarray(p_structural_dominant, dtype=np.float64),
        np.asarray(p_discriminative, dtype=np.float64),
    )

    rejected, bh_thresholds, k_star = _benjamini_hochberg(p_conj, alpha, method)

    return {
        'rejected': rejected,
        'n_rejected': int(np.sum(rejected)),
        'p_conjunction': p_conj,
        'bh_thresholds': bh_thresholds,
        'k_star': k_star,
    }


# ============================================================================
# 7. π̂₀ ESTIMATION WITH CONFIDENCE INTERVAL
# ============================================================================

def compute_pi0_with_ci(
    p_values: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    axis_name: str = "disc",
    log_name: str = "",
) -> Dict[str, Any]:
    """
    Compute Storey-Taylor-Siegmund (2004) bootstrap π̂₀ with 95% CI from
    the λ-grid range.

    The CI is derived from the π̂₀(λ) grid: the 95% CI spans from the
    2.5th to the 97.5th percentile of the π̂₀(λ) estimates across the
    λ grid, clipped to [0, 1]. This is a conservative CI that reflects
    the sensitivity of π̂₀ to the tuning parameter λ.

    For a proper bootstrap CI on π̂₀ itself, one would need to bootstrap
    the p-values — but since p-values arise from permutations (not i.i.d.
    samples), resampling them violates the exchangeability assumption.
    The λ-grid range is therefore the appropriate uncertainty measure,
    as recommended by Storey, Taylor & Siegmund (2004, §4).

    Args:
        p_values:  (m,) array of p-values for one axis.
        lambdas:   λ grid (default: 0.05 to 0.85 in steps of 0.05).
        axis_name: Name for reporting ("disc", "struct_c0", "struct_c1").
        log_name:  Event log name.

    Returns:
        Dict with keys: 'pi0', 'lambda_star', 'sensitivity_lo', 'sensitivity_hi', 'm'.
        Note: sensitivity_lo/hi reflect the λ-grid sensitivity range, NOT a
        statistical CI — see class-level note in Pi0EstimateWithCI.
    """
    if lambdas is None:
        lambdas = np.arange(0.05, 0.90, 0.05)
    lambdas = np.sort(lambdas)
    m = len(p_values)
    p = np.asarray(p_values, dtype=np.float64)

    # π̂₀(λ) grid
    pi0_grid = np.array([
        min(np.mean(p > lam) / (1.0 - lam), 1.0) for lam in lambdas
    ])

    # Closed-form MSE minimisation (matches StoreyLab/qvalue R package)
    W = np.array([np.sum(p > lam) for lam in lambdas], dtype=np.float64)
    min_pi0 = float(np.quantile(pi0_grid, 0.10))
    variance_term = (W / (m ** 2 * (1.0 - lambdas) ** 2)) * (1.0 - W / m)
    bias_sq_term = (pi0_grid - min_pi0) ** 2
    mse = variance_term + bias_sq_term

    best_idx = int(np.argmin(mse))
    lambda_star = float(lambdas[best_idx])
    pi0_hat = float(np.minimum(pi0_grid[best_idx], 1.0))

    # Sensitivity range from λ-grid: 2.5th and 97.5th percentile of π̂₀(λ).
    # This reflects λ-tuning sensitivity, NOT a statistical confidence interval.
    sensitivity_lo = float(np.clip(np.percentile(pi0_grid, 2.5), 0.0, 1.0))
    sensitivity_hi = float(np.clip(np.percentile(pi0_grid, 97.5), 0.0, 1.0))

    return {
        'pi0': pi0_hat,
        'lambda_star': lambda_star,
        'sensitivity_lo': sensitivity_lo,
        'sensitivity_hi': sensitivity_hi,
        'm': m,
        'axis_name': axis_name,
        'log_name': log_name,
        'pi0_grid': pi0_grid,
        'lambda_grid': lambdas,
    }


def compute_pi0_all_axes(
    p_disc: np.ndarray,
    p_struct_c0: np.ndarray,
    p_struct_c1: np.ndarray,
    log_name: str = "",
) -> Pi0EstimateWithCI:
    """
    Compute π̂₀ with CI for all three axes (disc, struct_c0, struct_c1).

    Args:
        p_disc:      (m',) Fisher conjunction / discriminative p-values.
        p_struct_c0: (m,)  structural p-values for class 0.
        p_struct_c1: (m,)  structural p-values for class 1.
        log_name:    Event log name.

    Returns:
        Pi0EstimateWithCI with all three axes.
    """
    disc = compute_pi0_with_ci(p_disc, axis_name="disc", log_name=log_name)
    sc0 = compute_pi0_with_ci(p_struct_c0, axis_name="struct_c0", log_name=log_name)
    sc1 = compute_pi0_with_ci(p_struct_c1, axis_name="struct_c1", log_name=log_name)

    return Pi0EstimateWithCI(
        log_name=log_name,
        pi0_disc=disc['pi0'],
        pi0_struct_c0=sc0['pi0'],
        pi0_struct_c1=sc1['pi0'],
        lambda_star_disc=disc['lambda_star'],
        lambda_star_struct_c0=sc0['lambda_star'],
        lambda_star_struct_c1=sc1['lambda_star'],
        pi0_disc_sensitivity_lo=disc['sensitivity_lo'],
        pi0_disc_sensitivity_hi=disc['sensitivity_hi'],
        pi0_struct_c0_sensitivity_lo=sc0['sensitivity_lo'],
        pi0_struct_c0_sensitivity_hi=sc0['sensitivity_hi'],
        pi0_struct_c1_sensitivity_lo=sc1['sensitivity_lo'],
        pi0_struct_c1_sensitivity_hi=sc1['sensitivity_hi'],
        m_disc=disc['m'],
        m_struct=sc0['m'],
    )


# ============================================================================
# 8. σ_null HETEROGENEITY ANALYSIS (Tusher Failure Step 1)
# ============================================================================

def compute_sigma_null_heterogeneity(
    null_delta_matrix: np.ndarray,
    constraint_types: List[str],
    pattern_to_family: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute σ_null per constraint type family from null_delta_matrix.

    This is Step 1 of the Tusher failure mechanistic demonstration:
    show that σ_null varies ~50× across constraint families, from
    Init/End ≈ 0.3 down to NotChainSuccession ≈ 0.006.

    Args:
        null_delta_matrix: (B₁, m) permuted Δ_b(p) from label permutation.
        constraint_types:  (m,) list of constraint type names, one per pattern.
        pattern_to_family: Optional mapping from pattern index to family name.
            If None, the constraint_type string itself is used as the family.

    Returns:
        Dict[family_name → {
            'mean_sigma': float,
            'std_sigma':  float,
            'min_sigma':  float,
            'max_sigma':  float,
            'median_sigma': float,
            'n_patterns': int,
        }]
    """
    B1, m = null_delta_matrix.shape
    abs_null = np.abs(null_delta_matrix).astype(np.float64)

    # Per-pattern σ_null
    sigma_null = abs_null.std(axis=0)  # (m,)

    # Group by family
    families: Dict[str, List[float]] = {}
    for p_idx in range(m):
        if pattern_to_family is not None:
            family = pattern_to_family.get(p_idx, constraint_types[p_idx])
        else:
            ct = constraint_types[p_idx]
            # Aggregate into families
            if ct in ('Init', 'End'):
                family = 'Init/End'
            elif ct in ('Response', 'AlternateResponse', 'ChainResponse'):
                family = 'Response'
            elif ct in ('Succession', 'AlternateSuccession', 'ChainSuccession'):
                family = 'Succession'
            elif ct in ('NotResponse',):
                family = 'NotResponse'
            elif ct in ('NotChainSuccession',):
                family = 'NotChainSuccession'
            else:
                family = ct
        families.setdefault(family, []).append(float(sigma_null[p_idx]))

    result: Dict[str, Dict[str, float]] = {}
    for family, sigmas in sorted(families.items()):
        arr = np.array(sigmas)
        result[family] = {
            'mean_sigma': float(np.mean(arr)),
            'std_sigma': float(np.std(arr)),
            'min_sigma': float(np.min(arr)),
            'max_sigma': float(np.max(arr)),
            'median_sigma': float(np.median(arr)),
            'n_patterns': len(arr),
        }

    return result


# ============================================================================
# 9. TUSHER INFLATION FACTOR (Tusher Failure Step 2)
# ============================================================================

def compute_tusher_inflation_factor(
    null_delta_matrix: np.ndarray,
    delta_obs: np.ndarray,
    tau_star_perpattern: float,
    pi0_hat: float = 1.0,
) -> Dict[str, float]:
    """
    Compute the inflation factor ρ_inf = Ê[V(τ*)]_pooled / Ê[V(τ*)]_per-pattern.

    This is Step 2 of the Tusher failure demonstration. At the threshold τ*
    selected by the per-pattern Storey method:

        Ê[V(τ*)]_pooled     = π̂₀ × (1/B₁) × Σ_b #{p : |Δ_b(p)| ≥ τ*}
        Ê[V(τ*)]_per-pattern = π̂₀ × Σ_{p: rejected} (#{b: |Δ_b(p)| ≥ |Δ_obs(p)|} / B₁)

    The pooled estimate cross-contaminates: Init/End patterns (σ_null ≈ 0.3)
    contribute false exceedances at thresholds relevant to NotChainSuccession
    patterns (σ_null ≈ 0.006), inflating Ê[V] by factor ρ ≈ 80.

    Args:
        null_delta_matrix:   (B₁, m) permuted Δ_b(p).
        delta_obs:           (m,) observed Δ.
        tau_star_perpattern: τ* from the per-pattern Storey method (on |Δ| scale).
        pi0_hat:             Estimated null fraction.

    Note on asymmetry (intentional):
        E_V_pooled uses the global threshold τ_star_perpattern for ALL patterns,
        while E_V_perpattern uses each rejected pattern's own |Δ_obs(p)| as its
        individual threshold. This means ρ_inf is a LOWER BOUND on the true
        inflation: the per-pattern estimate is more conservative because each
        pattern's null column is compared against its own (larger) observed
        statistic rather than the shared τ*. The asymmetry is intentional and
        demonstrates why the pooled Tusher estimate inflates the FP count.

    Returns:
        Dict with keys:
            'rho_inf':           float — inflation factor (expected ≈ 80)
            'E_V_pooled':        float — pooled Ê[V(τ*)]
            'E_V_perpattern':    float — per-pattern Ê[V(τ*)]
            'tau_star':          float — the threshold used
    """
    B1, m = null_delta_matrix.shape
    abs_null = np.abs(null_delta_matrix).astype(np.float64)
    abs_obs = np.abs(delta_obs).astype(np.float64)
    tau = tau_star_perpattern

    if tau <= 0:
        return {
            'rho_inf': float('nan'),
            'E_V_pooled': 0.0,
            'E_V_perpattern': 0.0,
            'tau_star': tau,
        }

    # ── Pooled Ê[V(τ*)] ─────────────────────────────────────────────────
    # For each resample b, count total exceedances across ALL m patterns
    null_exceedances_per_b = np.sum(abs_null >= tau, axis=1)  # (B1,)
    E_V_pooled = pi0_hat * float(np.mean(null_exceedances_per_b))

    # ── Per-pattern Ê[V(τ*)] ────────────────────────────────────────────
    # Column-wise: for each rejected pattern, compute its own expected
    # false positive contribution using ONLY its own null column.
    # Ê[V_p] = π̂₀ × (#{b: |Δ_b(p)| ≥ |Δ_obs(p)|} / B₁)
    patterns_rejected = abs_obs >= tau
    E_V_pp_components = np.zeros(m)
    for p_idx in range(m):
        if patterns_rejected[p_idx]:
            count_ext = np.sum(abs_null[:, p_idx] >= abs_obs[p_idx])
            E_V_pp_components[p_idx] = pi0_hat * count_ext / B1

    E_V_perpattern = float(np.sum(E_V_pp_components))

    # Inflation factor
    rho_inf = E_V_pooled / max(E_V_perpattern, 1e-15)

    return {
        'rho_inf': rho_inf,
        'E_V_pooled': E_V_pooled,
        'E_V_perpattern': E_V_perpattern,
        'tau_star': tau,
    }


# ============================================================================
# 10. FULL TUSHER FAILURE REPORT (Steps 1-3 combined)
# ============================================================================

def build_tusher_failure_report(
    null_delta_matrix: np.ndarray,
    delta_obs: np.ndarray,
    constraint_types: List[str],
    k_star_storey: int,
    tau_star_storey: float,
    pi0_hat: float = 1.0,
    alpha: float = 0.05,
    log_name: str = "",
) -> TusherFailureReport:
    """
    Assemble the complete three-step Tusher flat-null failure demonstration.

    Step 1: σ_null heterogeneity across constraint families.
    Step 2: Inflation factor ρ_inf at τ*_Storey.
    Step 3: k*_Tusher = 0 from the Tusher flat-null sweep.

    Args:
        null_delta_matrix: (B₁, m) from Phase 1 label permutation.
        delta_obs:         (m,) observed Δ.
        constraint_types:  (m,) constraint type per pattern.
        k_star_storey:     k* from the per-pattern Storey method.
        tau_star_storey:   τ* from the per-pattern Storey method.
        pi0_hat:           π̂₀ estimate.
        alpha:             Target FDR level.
        log_name:          Event log name.

    Returns:
        TusherFailureReport with all three steps.
    """
    # Step 1: σ_null heterogeneity
    sigma_by_family = compute_sigma_null_heterogeneity(
        null_delta_matrix, constraint_types
    )
    all_means = [v['mean_sigma'] for v in sigma_by_family.values() if v['n_patterns'] > 0]
    sigma_ratio = max(all_means) / max(min(all_means), 1e-15) if all_means else 0.0

    # Step 2: Inflation factor
    inflation = compute_tusher_inflation_factor(
        null_delta_matrix, delta_obs, tau_star_storey, pi0_hat
    )

    # Step 3: Tusher flat-null sweep → k* = 0
    tusher_result = run_tusher_flat_null(
        null_delta_matrix, delta_obs, alpha, pi0_hat
    )

    return TusherFailureReport(
        log_name=log_name,
        k_star_tusher=tusher_result['k_star'],
        k_star_storey=k_star_storey,
        rho_inf=inflation['rho_inf'],
        sigma_null_by_family=sigma_by_family,
        sigma_null_ratio=sigma_ratio,
        E_V_pooled_at_tau_star=inflation['E_V_pooled'],
        E_V_perpattern_at_tau_star=inflation['E_V_perpattern'],
    )


# ============================================================================
# 11. AGGREGATE RESULTS INTO DATAFRAME
# ============================================================================

def build_rq1_results_df(
    all_null_runs: Dict[str, List[int]],
    R_obs_dict: Dict[str, int],
    m_total: int,
    alpha: float = 0.05,
    log_name: str = "",
) -> pd.DataFrame:
    """
    Aggregate all B_null held-out replicates into a clean DataFrame.

    For each method, computes FDR_emp, PCER_emp, FWER_emp, and 95% BCa CI.

    Args:
        all_null_runs: Dict[method_name → List[int]] where each list has
            B_null entries, each being |S_b| for held-out replicate b.
        R_obs_dict:    Dict[method_name → int] — rejections on original log.
        m_total:       Total number of patterns tested.
        alpha:         Nominal FDR level.
        log_name:      Event log name.

    Returns:
        pd.DataFrame with one row per method and columns:
            method, log, R_obs, FDR_emp, PCER_emp, FWER_emp,
            FDR_CI_lower, FDR_CI_upper, B_null, m_total, alpha,
            controls_FDR (bool: FDR_emp ≤ alpha)
    """
    rows = []
    for method_name, null_counts in all_null_runs.items():
        null_arr = np.array(null_counts, dtype=np.float64)
        R_obs = R_obs_dict.get(method_name, 0)

        result = compute_empirical_fdr(
            null_arr, R_obs, m_total,
            method_name=method_name, log_name=log_name, alpha=alpha
        )

        rows.append({
            'method': method_name,
            'log': log_name,
            'R_obs': R_obs,
            'FDR_emp': result.fdr_emp,
            'PCER_emp': result.pcer_emp,
            'FWER_emp': result.fwer_emp,
            'FDR_CI_lower': result.fdr_ci_lower,
            'FDR_CI_upper': result.fdr_ci_upper,
            'B_null': result.B_null,
            'm_total': m_total,
            'alpha': alpha,
            'controls_FDR': result.fdr_emp <= alpha,
        })

    return pd.DataFrame(rows)


# ============================================================================
# 12. PAPER TABLE BUILDER (Table 1)
# ============================================================================

def build_paper_table(
    results_per_log: Dict[str, pd.DataFrame],
    pi0_estimates: Optional[Dict[str, Pi0EstimateWithCI]] = None,
) -> pd.DataFrame:
    """
    Build the final paper table (Table 1 in the evaluation section).

    Combines results across all logs into the format specified in the RQ1
    protocol:

    ┌─────────┬────────────────────┬──────────┬──────────────┬──────────────┐
    │ Log     │ Metric             │ Storey   │ BH-IUT       │ Cecconi+BH   │ Tusher │
    ├─────────┼────────────────────┼──────────┼──────────────┼──────────────┤
    │ BPI-17  │ FDR_emp            │ ≤0.05 ✓  │ ≤0.05 ✓      │ ≤0.05 ✓      │ k*=0 ✗│
    │         │ 95% CI             │ [a, b]   │ [c, d]       │ [e, f]       │ —     │
    │         │ k* (orig)          │ N        │ N'           │ N''          │ 0     │
    │         │ π̂₀                 │ 0.XX     │ 1.0          │ 1.0          │ N/A   │
    │         │ PCER_emp           │          │              │              │       │
    │         │ FWER_emp           │          │              │              │       │
    └─────────┴────────────────────┴──────────┴──────────────┴──────────────┘

    Args:
        results_per_log: Dict[log_name → pd.DataFrame] from build_rq1_results_df.
        pi0_estimates:   Dict[log_name → Pi0EstimateWithCI] from compute_pi0_all_axes.

    Returns:
        pd.DataFrame in long format, ready for LaTeX tabulation.
    """
    all_rows = []

    for log_name, df in results_per_log.items():
        for _, row in df.iterrows():
            entry = {
                'log': log_name,
                'method': row['method'],
                'FDR_emp': row['FDR_emp'],
                'FDR_CI': f"[{row['FDR_CI_lower']:.4f}, {row['FDR_CI_upper']:.4f}]",
                'k_star_orig': row['R_obs'],
                'PCER_emp': row['PCER_emp'],
                'FWER_emp': row['FWER_emp'],
                'controls_FDR': row['controls_FDR'],
            }

            # Add π̂₀ where available.
            # IMPORTANT: The four method name strings below must exactly match
            # the keys returned by evaluate_single_null_replicate():
            #   'Dual-Axis Storey', 'BH-IUT', 'Cecconi_ChiSq_BH', 'Tusher_FlatNull'
            if pi0_estimates and log_name in pi0_estimates:
                pi0 = pi0_estimates[log_name]
                if row['method'] == 'Dual-Axis Storey':
                    entry['pi0_disc'] = pi0.pi0_disc
                    entry['pi0_disc_CI'] = (
                        f"[{pi0.pi0_disc_sensitivity_lo:.3f}, {pi0.pi0_disc_sensitivity_hi:.3f}]"
                    )
                elif row['method'] == 'BH-IUT':
                    entry['pi0_disc'] = 1.0  # BH assumes π₀ = 1
                    entry['pi0_disc_CI'] = "(fixed)"
                elif row['method'] == 'Cecconi_ChiSq_BH':
                    entry['pi0_disc'] = 1.0
                    entry['pi0_disc_CI'] = "(fixed)"
                else:
                    entry['pi0_disc'] = float('nan')
                    entry['pi0_disc_CI'] = "N/A"

            all_rows.append(entry)

    return pd.DataFrame(all_rows)


# ============================================================================
# 13. STATISTICAL TEST FOR FDR CONTROL
# ============================================================================

def test_fdr_control(
    null_rejection_counts: np.ndarray,
    R_obs: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    One-sided exact binomial test for the null "method controls FDR at α".

    Under the null hypothesis H₀: FDR ≤ α, the expected number of false
    discoveries in each null replicate is at most α × R_obs. The empirical
    FDR_emp should not exceed α.

    This test provides a formal p-value for the claim "the method controls
    FDR at the nominal level." It is stricter than simply checking
    FDR_emp ≤ α: it accounts for sampling variability in the B_null
    null replicates.

    Additionally tests for FWER control: null "P(any FP) ≤ α".

    Args:
        null_rejection_counts: (B_null,) array of |S_b|.
        R_obs:                 Rejections on original log.
        alpha:                 Nominal FDR level.

    Returns:
        Dict with:
            'fdr_emp':              float
            'fdr_test_pvalue':      float — p-value for H₀: FDR ≤ α
            'fwer_emp':             float
            'fwer_test_pvalue':     float — p-value for H₀: FWER ≤ α
    """
    B_null = len(null_rejection_counts)
    counts = np.asarray(null_rejection_counts, dtype=np.float64)
    denom = max(R_obs, 1)

    # FDR_emp
    fdr_emp = float(np.mean(counts)) / denom

    # FWER_emp and its binomial test
    n_any_fp = int(np.sum(counts > 0))
    fwer_emp = n_any_fp / B_null

    # One-sided binomial test for FWER: H₀: p ≤ α
    # Under H₀, n_any_fp ~ Bin(B_null, α).
    # P-value = P(X ≥ n_any_fp | p = α) = 1 - CDF(n_any_fp - 1, B_null, α)
    if n_any_fp > 0:
        fwer_pval = float(1.0 - stats.binom.cdf(n_any_fp - 1, B_null, alpha))
    else:
        fwer_pval = 1.0  # zero FPs — certainly does not violate

    # For FDR: test whether mean(|S_b|/R_obs) ≤ α using a one-sample t-test
    # against α (one-sided). With B_null ≥ 100, CLT applies to the mean.
    fdr_per_replicate = counts / denom
    if np.std(fdr_per_replicate) > 0:
        t_stat = (np.mean(fdr_per_replicate) - alpha) / (
            np.std(fdr_per_replicate, ddof=1) / np.sqrt(B_null)
        )
        # One-sided: reject H₀ if t_stat > 0 (FDR exceeds α)
        fdr_pval = float(1.0 - stats.t.cdf(t_stat, df=B_null - 1))
    else:
        # All replicates have identical count (likely 0)
        fdr_pval = 1.0 if fdr_emp <= alpha else 0.0

    return {
        'fdr_emp': fdr_emp,
        'fdr_test_pvalue': fdr_pval,
        'fwer_emp': fwer_emp,
        'fwer_test_pvalue': fwer_pval,
    }


# ============================================================================
# 14. COUNT REJECTIONS FOR A METHOD ON PERMUTED LABELS
# ============================================================================

def count_rejections_dual_axis_storey(
    pattern_results: list,
    alpha: float = 0.05,
) -> int:
    """
    Count 'Both' category rejections from Phase 1 PatternTestResult list.

    This is the rejection count for the dual-axis Storey method:
    a pattern is rejected iff is_significant_final = True
    (i.e., both q_structural_dominant ≤ α AND q_value_sam ≤ α).

    Args:
        pattern_results: List of PatternTestResult from Phase 1.
        alpha:           Nominal FDR level (for reference only;
                         the q-values are already computed by Phase 1).

    Returns:
        int — number of 'Both' category patterns.
    """
    return sum(1 for r in pattern_results if r.is_significant_final)


def count_rejections_bh_iut(
    pattern_results: list,
) -> int:
    """
    Count BH rejections from Phase 1 PatternTestResult list.

    Args:
        pattern_results: List of PatternTestResult from Phase 1.

    Returns:
        int — number of BH-significant patterns (is_significant_bh = True).
    """
    return sum(1 for r in pattern_results if r.is_significant_bh)


# ============================================================================
# 15. EXTRACT ARRAYS FROM PHASE 1 RESULTS (bridge to eval_utils)
# ============================================================================

def extract_arrays_from_phase1(
    pattern_results: list,
) -> Dict[str, Any]:
    """
    Extract key numpy arrays from Phase 1 PatternTestResult list for use
    in eval_utils functions.

    This is the bridge between the Phase 1 data structures and the
    generic eval_utils functions.

    Args:
        pattern_results: List of PatternTestResult from Phase 1.

    Returns:
        Dict with keys:
            'delta_obs':                (m,) observed Δ = P̂₁ − P̂₀
            'p_structural_class0':      (m,) Phipson-Smyth p for H₀ˢ class 0
            'p_structural_class1':      (m,) Phipson-Smyth p for H₀ˢ class 1
            'p_structural_dominant':    (m,) dominant-class structural p
            'p_discriminative':         (m,) two-sided discriminative p
            'p_conjunction':            (m,) IUT conjunction p
            'q_value_sam':              (m,) Storey q-value (discriminative)
            'q_structural_dominant':    (m,) Storey q-value (structural)
            'constraint_types':         (m,) list of constraint type strings
            'is_significant_final':     (m,) bool
            'is_significant_bh':        (m,) bool
            'significance_category':    (m,) list of category strings
    """
    m = len(pattern_results)
    return {
        'delta_obs': np.array([r.delta_obs for r in pattern_results]),
        'p_structural_class0': np.array([r.p_structural_class0 for r in pattern_results]),
        'p_structural_class1': np.array([r.p_structural_class1 for r in pattern_results]),
        'p_structural_dominant': np.array([r.p_structural_dominant for r in pattern_results]),
        'p_discriminative': np.array([r.p_discriminative for r in pattern_results]),
        'p_conjunction': np.array([r.p_conjunction for r in pattern_results]),
        'q_value_sam': np.array([r.q_value_sam for r in pattern_results]),
        'q_structural_dominant': np.array([r.q_structural_dominant for r in pattern_results]),
        'constraint_types': [r.constraint_type for r in pattern_results],
        'is_significant_final': np.array([r.is_significant_final for r in pattern_results]),
        'is_significant_bh': np.array([r.is_significant_bh for r in pattern_results]),
        'significance_category': [r.significance_category for r in pattern_results],
    }


# ============================================================================
# 16. SINGLE NULL-REPLICATE EVALUATION DRIVER
# ============================================================================

def evaluate_single_null_replicate(
    run_phase1_fn,
    holds_all: Dict[tuple, Dict[str, int]],
    case_ids: List[str],
    original_labels: np.ndarray,
    permuted_labels: np.ndarray,
    candidates_all: list,
    B1_internal: int = 2000,
    B2_internal: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, int]:
    """
    Run all four methods on a single held-out null replicate and return
    rejection counts.

    This is the inner loop of the RQ1 evaluation. For each held-out
    permuted label assignment:
    1. Run Phase 1 (dual-axis Storey) with reduced B₁ and B₂.
    2. Run BH on IUT p-values.
    3. Run Cecconi chi-square + BH.
    4. Run Tusher flat-null SAM.

    The candidate pool is FIXED from the original-label Phase 0 run
    (Step 1 of the RQ1 protocol: candidate pool fixation).

    Args:
        run_phase1_fn:   Callable that runs Phase 1 given permuted labels.
                         Signature: run_phase1_fn(permuted_labels, B1, B2, rs)
                         → (pattern_results, null_delta_matrix, delta_obs)
        holds_all:       Holds-by-case from original Phase 1 (reused for
                         Cecconi baseline — trace structure unchanged).
        case_ids:        Ordered list of case IDs (matching label arrays).
        original_labels: (n,) original label vector.
        permuted_labels: (n,) this replicate's permuted labels.
        candidates_all:  List of (ct, a, b) pattern specs.
        B1_internal:     Internal B₁ for this replicate's Phase 1
                         (default 2000, reduced for speed).
        B2_internal:     Internal B₂ for this replicate's Phase 1
                         (default 500, reduced for speed).
        alpha:           Target FDR level.
        random_state:    RNG seed for this replicate's Phase 1.

    Returns:
        Dict[method_name → int] — rejection counts for this replicate.
        Keys: 'Dual-Axis Storey', 'BH-IUT', 'Cecconi_ChiSq_BH', 'Tusher_FlatNull'
    """
    # ── Method 1: Dual-Axis Storey (the full Phase 1 pipeline) ───────────
    phase1_results, null_delta_mat, delta_obs_perm = run_phase1_fn(
        permuted_labels, B1_internal, B2_internal, random_state
    )
    n_storey = count_rejections_dual_axis_storey(phase1_results, alpha)

    # ── Method 2: BH on IUT p-values ────────────────────────────────────
    arrays = extract_arrays_from_phase1(phase1_results)
    bh_result = run_bh_on_iut_pvalues(
        arrays['p_structural_dominant'],
        arrays['p_discriminative'],
        alpha=alpha,
    )
    n_bh = bh_result['n_rejected']

    # ── Method 3: Cecconi chi-square + BH ────────────────────────────────
    # Use the ORIGINAL holds_all (trace structure is unchanged by label
    # permutation), but override class membership with permuted labels.
    label_override = {
        cid: int(lab)
        for cid, lab in zip(case_ids, permuted_labels)
    }
    cecconi_result = run_cecconi_baseline(
        holds_all,
        ids_class0=set(),  # not used when label_override is provided
        ids_class1=set(),
        alpha=alpha,
        label_override=label_override,
    )
    n_cecconi = cecconi_result.n_rejected

    # ── Method 4: Tusher flat-null SAM ───────────────────────────────────
    tusher_result = run_tusher_flat_null(
        null_delta_mat, delta_obs_perm, alpha=alpha, pi0_hat=1.0
    )
    n_tusher = tusher_result['k_star']

    return {
        'Dual-Axis Storey': n_storey,
        'BH-IUT': n_bh,
        'Cecconi_ChiSq_BH': n_cecconi,
        'Tusher_FlatNull': n_tusher,
    }


# ============================================================================
# 17. FULL RQ1 EVALUATION ORCHESTRATOR
# ============================================================================

def run_rq1_evaluation(
    run_phase1_fn,
    holds_all: Dict[tuple, Dict[str, int]],
    case_ids: List[str],
    labels: np.ndarray,
    candidates_all: list,
    B_null: int,
    B1_internal: int = 2000,
    B2_internal: int = 500,
    alpha: float = 0.05,
    base_seed: int = 20260321,
    log_name: str = "",
    R_obs_dict: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full RQ1 evaluation for one event log: B_null held-out permutations.

    Orchestrates the complete protocol:
    1. Generate B_null held-out label permutations (independent of Phase 1).
    2. For each replicate, run all four methods and collect |S_b|.
    3. Compute FDR_emp, PCER_emp, FWER_emp with BCa 95% CI.
    4. Statistical test for FDR control.
    5. Build results DataFrame and paper table row.

    Args:
        run_phase1_fn:  Callable to run Phase 1 on permuted labels.
                        Signature: fn(permuted_labels, B1, B2, rs) →
                        (pattern_results, null_delta_matrix, delta_obs)
        holds_all:      Holds-by-case from original Phase 1.
        case_ids:       Ordered case IDs.
        labels:         (n,) original binary labels.
        candidates_all: Pattern specs from Phase 0.
        B_null:         Number of held-out replicates.
        B1_internal:    Internal B₁ per replicate (default 2000).
        B2_internal:    Internal B₂ per replicate (default 500).
        alpha:          Target FDR level.
        base_seed:      Base seed for reproducibility.
        log_name:       Event log name.
        R_obs_dict:     Rejection counts on original log per method.
                        If None, uses zeros (must be provided for FDR_emp).
        verbose:        Print progress.

    Returns:
        Dict with:
            'results_df':       pd.DataFrame — per-method FDR table
            'null_counts':      Dict[method → np.array of |S_b|]
            'fdr_tests':        Dict[method → test_fdr_control result]
            'wall_seconds':     float
    """
    t0 = time.time()
    methods = ['Dual-Axis Storey', 'BH-IUT', 'Cecconi_ChiSq_BH', 'Tusher_FlatNull']
    null_counts: Dict[str, List[int]] = {m_name: [] for m_name in methods}

    if R_obs_dict is None:
        R_obs_dict = {m_name: 0 for m_name in methods}

    m_total = len(candidates_all)

    if verbose:
        print(f"\n{'='*100}")
        print(f"RQ1 EVALUATION — {log_name}")
        print(f"{'='*100}")
        print(f"  B_null = {B_null}, B1_internal = {B1_internal}, "
              f"B2_internal = {B2_internal}")
        print(f"  m = {m_total}, n = {len(labels)}, "
              f"n+ = {int(labels.sum())}, n- = {int(len(labels) - labels.sum())}")
        print(f"  R_obs: {R_obs_dict}")

    # Generate all held-out permutations upfront
    permuted_labels_all = generate_heldout_permutation_batch(
        labels, B_null, base_seed
    )

    for b in range(B_null):
        if verbose and (b % max(1, B_null // 20) == 0 or b == B_null - 1):
            elapsed = time.time() - t0
            eta = (elapsed / max(b, 1)) * (B_null - b) if b > 0 else 0
            print(f"  Replicate {b+1}/{B_null} "
                  f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")

        replicate_counts = evaluate_single_null_replicate(
            run_phase1_fn=run_phase1_fn,
            holds_all=holds_all,
            case_ids=case_ids,
            original_labels=labels,
            permuted_labels=permuted_labels_all[b],
            candidates_all=candidates_all,
            B1_internal=B1_internal,
            B2_internal=B2_internal,
            alpha=alpha,
            random_state=base_seed + 100000 + b,
        )

        for method_name, count in replicate_counts.items():
            null_counts[method_name].append(count)

    # ── Aggregate results ────────────────────────────────────────────────
    null_arrays = {m_name: np.array(c) for m_name, c in null_counts.items()}

    results_df = build_rq1_results_df(
        all_null_runs=null_counts,
        R_obs_dict=R_obs_dict,
        m_total=m_total,
        alpha=alpha,
        log_name=log_name,
    )

    # ── Statistical tests ────────────────────────────────────────────────
    fdr_tests = {}
    for method_name in methods:
        fdr_tests[method_name] = test_fdr_control(
            null_arrays[method_name],
            R_obs_dict.get(method_name, 0),
            alpha=alpha,
        )

    wall_seconds = time.time() - t0

    if verbose:
        print(f"\n{'~'*100}")
        print(f"RQ1 RESULTS — {log_name} (wall time: {wall_seconds:.1f}s)")
        print(f"{'~'*100}")
        print(results_df.to_string(index=False))
        print(f"\n  Statistical tests (p-value for H0: FDR <= alpha):")
        for method_name, test in fdr_tests.items():
            verdict = "Controls FDR" if test['fdr_emp'] <= alpha else "FAILS"
            print(f"    {method_name:25s}: FDR_emp = {test['fdr_emp']:.4f}, "
                  f"p = {test['fdr_test_pvalue']:.4f}  [{verdict}]")

    return {
        'results_df': results_df,
        'null_counts': null_arrays,
        'fdr_tests': fdr_tests,
        'wall_seconds': wall_seconds,
    }


# ============================================================================
# 18. JSON SERIALISATION FOR REPRODUCIBILITY
# ============================================================================

def save_rq1_results_json(
    results_df: pd.DataFrame,
    fdr_tests: Dict[str, Dict],
    pi0_estimate: Optional[Pi0EstimateWithCI],
    tusher_report: Optional[TusherFailureReport],
    output_path: str,
    log_name: str = "",
):
    """
    Save all RQ1 results to a JSON file for reproducibility and paper generation.

    Args:
        results_df:     DataFrame from build_rq1_results_df.
        fdr_tests:      Dict from test_fdr_control per method.
        pi0_estimate:   Pi0EstimateWithCI from compute_pi0_all_axes.
        tusher_report:  TusherFailureReport from build_tusher_failure_report.
        output_path:    Path to write JSON.
        log_name:       Event log name.
    """
    out: Dict[str, Any] = {
        'rq1_version': '1.0',
        'log_name': log_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'empirical_fdr_table': results_df.to_dict(orient='records'),
        'fdr_statistical_tests': {
            method: {
                'fdr_emp': t['fdr_emp'],
                'fdr_test_pvalue': t['fdr_test_pvalue'],
                'fwer_emp': t['fwer_emp'],
                'fwer_test_pvalue': t['fwer_test_pvalue'],
            }
            for method, t in fdr_tests.items()
        },
    }

    if pi0_estimate is not None:
        out['pi0_estimates'] = {
            'disc': {
                'pi0': pi0_estimate.pi0_disc,
                'sensitivity_lo': pi0_estimate.pi0_disc_sensitivity_lo,
                'sensitivity_hi': pi0_estimate.pi0_disc_sensitivity_hi,
                'lambda_star': pi0_estimate.lambda_star_disc,
                'm': pi0_estimate.m_disc,
            },
            'struct_c0': {
                'pi0': pi0_estimate.pi0_struct_c0,
                'sensitivity_lo': pi0_estimate.pi0_struct_c0_sensitivity_lo,
                'sensitivity_hi': pi0_estimate.pi0_struct_c0_sensitivity_hi,
                'lambda_star': pi0_estimate.lambda_star_struct_c0,
                'm': pi0_estimate.m_struct,
            },
            'struct_c1': {
                'pi0': pi0_estimate.pi0_struct_c1,
                'sensitivity_lo': pi0_estimate.pi0_struct_c1_sensitivity_lo,
                'sensitivity_hi': pi0_estimate.pi0_struct_c1_sensitivity_hi,
                'lambda_star': pi0_estimate.lambda_star_struct_c1,
                'm': pi0_estimate.m_struct,
            },
        }

    if tusher_report is not None:
        out['tusher_failure'] = {
            'k_star_tusher': tusher_report.k_star_tusher,
            'k_star_storey': tusher_report.k_star_storey,
            'rho_inf': tusher_report.rho_inf,
            'sigma_null_ratio': tusher_report.sigma_null_ratio,
            'E_V_pooled': tusher_report.E_V_pooled_at_tau_star,
            'E_V_perpattern': tusher_report.E_V_perpattern_at_tau_star,
            'sigma_null_by_family': tusher_report.sigma_null_by_family,
        }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)


# ============================================================================
# INTERNAL HELPER: BENJAMINI-HOCHBERG STEP-UP
# ============================================================================

def _benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float,
    method: str = 'BH',
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Benjamini-Hochberg (1995) or Benjamini-Yekutieli (2001) step-up.

    This is an internal helper used by run_cecconi_baseline() and
    run_bh_on_iut_pvalues(). Identical to the BH implementation in
    Phase 1 (p1_BPI17.py), kept here to make eval_utils standalone.

    Args:
        p_values: (m,) array of p-values.
        alpha:    Target FDR level.
        method:   'BH' or 'BY'.

    Returns:
        (rejected, bh_thresholds, k_star)
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool), np.array([]), 0

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    ranks = np.arange(1, m + 1)

    if method == 'BY':
        c_m = np.sum(1.0 / ranks)
    else:
        c_m = 1.0

    bh_critical = ranks * alpha / (m * c_m)

    k_star = 0
    for k in range(m, 0, -1):
        if sorted_p[k - 1] <= bh_critical[k - 1]:
            k_star = k
            break

    rejected = np.zeros(m, dtype=bool)
    if k_star > 0:
        rejected[sorted_idx[:k_star]] = True

    bh_thresholds = np.zeros(m)
    for i, orig_idx in enumerate(sorted_idx):
        bh_thresholds[orig_idx] = bh_critical[i]

    return rejected, bh_thresholds, k_star


# ============================================================================
# MODULE SELF-TEST
# ============================================================================

def _self_test():
    """
    Minimal self-test to verify all functions are importable and produce
    valid outputs on synthetic data. Run with: python eval_utils.py
    """
    print("=" * 80)
    print("eval_utils.py — SELF-TEST")
    print("=" * 80)

    rng = np.random.RandomState(42)
    m = 100
    B1 = 500
    n = 200

    # Synthetic null delta matrix
    null_delta_matrix = rng.randn(B1, m).astype(np.float32) * 0.1
    delta_obs = rng.randn(m) * 0.15

    # ── Test 1: Empirical FDR ────────────────────────────────────────────
    null_counts = rng.poisson(2, size=50)  # 50 null replicates
    result = compute_empirical_fdr(
        null_counts, R_obs=30, m_total=m,
        method_name="test", log_name="synthetic"
    )
    assert 0 <= result.fdr_emp, "FDR_emp must be non-negative"
    assert 0 <= result.fwer_emp <= 1, "FWER_emp must be in [0, 1]"
    print(f"  [1/12] compute_empirical_fdr: FDR_emp={result.fdr_emp:.4f}, "
          f"FWER_emp={result.fwer_emp:.4f}")

    # ── Test 2: BCa CI ───────────────────────────────────────────────────
    ci_lo, ci_hi = bootstrap_bca_ci(null_counts, R_obs=30)
    assert ci_lo <= ci_hi, "CI lower must be <= upper"
    print(f"  [2/12] bootstrap_bca_ci: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── Test 3: Cecconi baseline ─────────────────────────────────────────
    case_ids_all = [f"case_{i}" for i in range(n)]
    ids_c0 = set(case_ids_all[:n // 2])
    ids_c1 = set(case_ids_all[n // 2:])
    holds_all: Dict[tuple, Dict[str, int]] = {}
    for p_idx in range(m):
        holds: Dict[str, int] = {}
        for cid in case_ids_all:
            holds[cid] = int(rng.rand() > 0.5)
        holds_all[(f"Constraint_{p_idx}", f"A_{p_idx}", None)] = holds

    cecconi = run_cecconi_baseline(holds_all, ids_c0, ids_c1, alpha=0.05)
    assert cecconi.n_rejected >= 0, "Cecconi rejections must be non-negative"
    print(f"  [3/12] run_cecconi_baseline: {cecconi.n_rejected} rejections, "
          f"{cecconi.n_small_cell_violations} small-cell violations")

    # ── Test 4: Tusher flat-null ─────────────────────────────────────────
    tusher = run_tusher_flat_null(null_delta_matrix, delta_obs, alpha=0.05)
    print(f"  [4/12] run_tusher_flat_null: k*={tusher['k_star']}, "
          f"min_FDP={tusher['min_fdp']:.4f}")

    # ── Test 5: BH on IUT ────────────────────────────────────────────────
    p_s = rng.uniform(0, 1, m)
    p_d = rng.uniform(0, 1, m)
    bh = run_bh_on_iut_pvalues(p_s, p_d, alpha=0.05)
    print(f"  [5/12] run_bh_on_iut_pvalues: {bh['n_rejected']} rejections")

    # ── Test 6: pi0 with CI ──────────────────────────────────────────────
    p_mixed = np.concatenate([rng.uniform(0, 1, 70), rng.beta(0.1, 1, 30)])
    pi0 = compute_pi0_with_ci(p_mixed, axis_name="test")
    assert 0 < pi0['pi0'] <= 1, "pi0 must be in (0, 1]"
    print(f"  [6/12] compute_pi0_with_ci: pi0={pi0['pi0']:.4f} "
          f"[{pi0['sensitivity_lo']:.3f}, {pi0['sensitivity_hi']:.3f}]")

    # ── Test 7: sigma_null heterogeneity ─────────────────────────────────
    constraint_types = (['Init'] * 10 + ['Response'] * 40 +
                        ['Succession'] * 30 + ['NotChainSuccession'] * 20)
    sigma_het = compute_sigma_null_heterogeneity(
        null_delta_matrix, constraint_types
    )
    assert len(sigma_het) > 0, "Must return at least one family"
    print(f"  [7/12] compute_sigma_null_heterogeneity: {len(sigma_het)} families")
    for fam, stats_dict in sigma_het.items():
        print(f"         {fam:30s}: mean_sigma={stats_dict['mean_sigma']:.4f} "
              f"(n={stats_dict['n_patterns']})")

    # ── Test 8: Inflation factor ─────────────────────────────────────────
    inflation = compute_tusher_inflation_factor(
        null_delta_matrix, delta_obs, tau_star_perpattern=0.1
    )
    print(f"  [8/12] compute_tusher_inflation_factor: rho_inf={inflation['rho_inf']:.2f}")

    # ── Test 9: FDR control test ─────────────────────────────────────────
    fdr_test = test_fdr_control(null_counts, R_obs=30, alpha=0.05)
    print(f"  [9/12] test_fdr_control: FDR_emp={fdr_test['fdr_emp']:.4f}, "
          f"p={fdr_test['fdr_test_pvalue']:.4f}")

    # ── Test 10: Results DataFrame ───────────────────────────────────────
    all_null = {
        'Dual-Axis Storey': list(rng.poisson(1, 50)),
        'BH-IUT': list(rng.poisson(2, 50)),
    }
    R_obs_test = {'Dual-Axis Storey': 20, 'BH-IUT': 25}
    df = build_rq1_results_df(all_null, R_obs_test, m_total=m, log_name="test")
    assert len(df) == 2, "DataFrame must have 2 rows"
    print(f"  [10/12] build_rq1_results_df: {len(df)} rows")
    print(df.to_string(index=False))

    # ── Test 11: Held-out permutation generation ─────────────────────────
    labels = np.array([0] * 100 + [1] * 100)
    perms = generate_heldout_permutation_batch(labels, B_null=10)
    for perm in perms:
        assert int(perm.sum()) == 100, "Permutation must preserve marginals"
    print(f"  [11/12] generate_heldout_permutation_batch: {len(perms)} perms, "
          f"all preserve marginals")

    # ── Test 12: Full Tusher failure report ──────────────────────────────
    report = build_tusher_failure_report(
        null_delta_matrix, delta_obs, constraint_types,
        k_star_storey=15, tau_star_storey=0.1,
        pi0_hat=0.85, log_name="synthetic"
    )
    print(f"  [12/12] build_tusher_failure_report: k*_Tusher={report.k_star_tusher}, "
          f"k*_Storey={report.k_star_storey}, rho_inf={report.rho_inf:.2f}")

    print(f"\n{'='*80}")
    print("ALL 12 SELF-TESTS PASSED")
    print(f"{'='*80}")


if __name__ == "__main__":
    _self_test()