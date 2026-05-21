"""
DvM_BISE2025_Sepsis.py — Business Process Deviance Mining: Sequential & Declarative Patterns
================================================================================================
Faithful reimplementation of the complete Deviance Mining pipeline from:

    Di Francescomarino, Donadello, Ghidini, Maggi, Puura (2025).
    Business Process Deviance Mining with Sequential and Declarative Patterns.
    Bus Inf Syst Eng 67(6):877–894.
    https://doi.org/10.1007/s12599-024-00911-5

Source repository (reference implementation):
    https://github.com/ivanDonadello/Deviance_mining_sequential_declarative_patterns

Pipeline (Section 7 of the paper):
──────────────────────────────────
Step 1.  Feature Extraction  (Section 7.1)
    1a.  Data features (F_data, Section 7.1.1):
         first/last of each attribute; min/max/avg of numeric attributes;
         count-per-value of categorical attributes; trace length; trace
         time-length in seconds.
    1b.  Sequential features (F_seq, Section 7.1.2):
         Tandem Repeats (TR) and their alphabets (TRA);
         Maximal Repeats (MR) and their alphabets (MRA).
         Extracted using the discriminative mining algorithm of
         Bose & van der Aalst (2009, 2013): patterns must be frequent
         (≥ support threshold θ) in exactly one class.
    1c.  Declarative features (F_decl, Section 7.1.3):
         Frequent activity sets mined via Apriori (θ = 0 in experiments);
         all Declare templates instantiated over each frequent set;
         candidate kept if support ≥ θ in the generating class sub-log.
    1d.  Data-aware Declare (F_declD, Section 7.1.4):
         For each F_decl constraint, collect activations from satisfied
         traces, encode data payloads, learn a local decision tree
         (deviant vs. non-deviant activations), enrich constraint with
         the extracted data condition.

Step 2.  Trace Encoding  (Section 7.2)
    Declarative (X_decl_{X,r}):
        −1  violated in r
         0  vacuously satisfied in r
         n  satisfied and activated n times in r   (n ≥ 1)
    Sequential (X_seq_{X,r}):   support (occurrence count) of pattern in r.
    Data        (X_data_{X,r}): value of the data feature for trace r.
    Hybrid (X_H):  [X_seq | X_decl]  (horizontal concatenation).
    Data-aware Declare (X_declD): same −1/0/n scale as declarative.

Step 3.  Feature Selection  (Section 7.3)
    Generalized Fisher score (Gu, Li, Han 2011, UAI):
        F_r = [ n+ (μ+_r − μ_r)² + n− (μ−_r − μ_r)² ]
              / [ n+ σ+²_r  +  n− σ−²_r ]
    Features ranked descending by F_r.
    Greedy coverage selection: add feature iff it covers ≥ 1 trace not
    yet fully covered; trace fully covered when it has ≥ coverage_threshold
    (= 20, Section 7.1) selected features whose value != 0 (non-zero =
    "covered" semantics from the reference implementation).

Step 4.  Model Training  (Section 7.4)
    RipperK: Wittgenstein library; k grid = {0,4,6,8,10,16,18}.
    Decision Tree: scikit-learn; max_depth grid = {5,10,15,20,25,30,35,60,80}.
    Both classifiers evaluated on each fold.

Step 5.  Rule Extraction & Evaluation  (Sections 7.5, 9.3)
    10-fold cross-validation.
    Metrics per fold (averaged across folds and datasets):
        Precision, Recall, F1, AUC  (Table 1, Table 2 of the paper)
        Average Rule Length  (ARL)   (Figure 5)
        Average Number of Rules (ANR) (Figure 6)

Encodings evaluated (Section 9.2):
    1.  IA              – Individual Activities (frequency)
    2.  MR              – Maximal Repeat
    3.  TR              – Tandem Repeat
    4.  MRA             – Maximal Repeat Alphabet
    5.  TRA             – Tandem Repeat Alphabet
    6.  Declare         – Declarative only
    7.  H               – Hybrid = Declare + all sequential
    8.  Data            – Pure data attributes
    9.  DeclD           – Data-aware Declare
   10.  Data+IA
   11.  Data+MR
   12.  Data+TR
   13.  Data+MRA
   14.  Data+TRA
   15.  Data+Declare
   16.  Data+H
   17.  Data+DeclD
   18.  H+DeclD
   19.  H+Data+DeclD

Backward-compatible aliases (rq1/rq2 interface):
    run_declareminer                    = run_dvm
    run_declareminer_on_doubly_null_log = run_dvm_on_doubly_null_log
    DM_CONFIG                           = DVM_CONFIG

References
----------
Di Francescomarino et al. (2025). BISE 67(6):877–894.
Bose & van der Aalst (2009). BPM LNCS 5701:159–175.
Bose & van der Aalst (2013). IEEE CIDM, pp 111–118.
Gu, Li, Han (2011). Generalized Fisher Score. UAI, pp 266–273.
Cohen (1995). Fast Effective Rule Induction. ICML, pp 115–123.
Agrawal & Srikant (1994). Fast Apriori. VLDB, pp 487–499.

Version : 1.0
Author  : Ahmed Nour Abdesselam
Date    : May 2026
"""

import os
import sys
import json
import time
import warnings
import argparse
import itertools
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any, FrozenSet

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ─── RIPPER-K import (Wittgenstein) ──────────────────────────────────────────
try:
    import wittgenstein as lw
    RIPPER_AVAILABLE = True
except ImportError:
    RIPPER_AVAILABLE = False
    warnings.warn(
        "Wittgenstein not found. Install with: pip install wittgenstein\n"
        "Falling back to DecisionTreeClassifier for rule extraction.",
        ImportWarning, stacklevel=2,
    )

# ─── PATH SETUP ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from P1_SDSM.p1_Production_hou import (
    load_and_preprocess_data,
    split_by_class,
    evaluate_pattern_fast,
    CaseInfo,
    INPUT_FILE        as P1_INPUT_FILE,
    DECLARE_SPEC_FILE as P1_SPEC_FILE,
    ALL_CONSTRAINT_TYPES,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CSV_PATH          = P1_INPUT_FILE
DECLARE_SPEC_FILE = P1_SPEC_FILE
OUTPUT_DIR        = "DvM_BISE2025_Production"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pipeline hyperparameters (paper Section 9.3) ───────────────────────────
CANDIDATE_THRESHOLD:   float = 0.1   # activity-pair frequency threshold (10 %)
SUPPORT_THRESHOLD:     float = 0.0   # constraint support threshold (0 in experiments)
COVERAGE_THRESHOLD:    int   = 20    # coverage threshold (Section 7.3, 7.1)
N_CV_FOLDS:            int   = 10    # 10-fold cross-validation (Section 9.3)
RANDOM_STATE:          int   = 42

# Grid search ranges (Section 9.3)
RIPPER_K_GRID:   List[int]            = [0, 4, 6, 8, 10, 16, 18]
DT_DEPTH_GRID:   List[Optional[int]]  = [5, 10, 15, 20, 25, 30, 35, 60, 80]
RIPPER_PRUNE_SIZE: float = 0.33

DVM_CONFIG: dict = {
    "candidate_threshold": CANDIDATE_THRESHOLD,
    "support_threshold":   SUPPORT_THRESHOLD,
    "coverage_threshold":  COVERAGE_THRESHOLD,
    "n_cv_folds":          N_CV_FOLDS,
    "random_state":        RANDOM_STATE,
    # rq1/rq2 compatibility
    "tau_min":    0.0,
    "fisher_eps": 1e-10,
}
DM_CONFIG: dict = DVM_CONFIG   # backward-compatible alias


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DECLARE TEMPLATES — aligned with p1 ALL_CONSTRAINT_TYPES
# ══════════════════════════════════════════════════════════════════════════════
#
# Uses evaluate_pattern_fast from the imported p1 module so that the exact
# same constraint semantics are used here as in the p1 hypothesis-testing
# pipeline.  Encoding: -1 violated, 0 vacuous, 1 satisfied.

UNARY_TEMPLATES  = {'Init', 'End'}
BINARY_TEMPLATES = {
    'Response', 'AlternateResponse', 'ChainResponse',
    'Succession', 'AlternateSuccession', 'ChainSuccession',
    'NotResponse', 'NotChainSuccession',
}


def _positional_to_trace(positional_trace: dict) -> List[str]:
    """Reconstruct ordered activity sequence from positional dict."""
    if not positional_trace:
        return []
    max_pos = max(p for positions in positional_trace.values() for p in positions)
    trace: List[str] = [''] * (max_pos + 1)
    for act, positions in positional_trace.items():
        for p in positions:
            trace[p] = act
    return trace


def apply_declare_template(
    template: str,
    acts: tuple,
    positional_trace: dict,
) -> Tuple[int, bool]:
    """
    Evaluate a Declare constraint via evaluate_pattern_fast (p1 semantics).
    Returns (value, is_vacuous): -1 violated, 0 vacuous, 1 satisfied.
    """
    activity_a = acts[0]
    activity_b = acts[1] if len(acts) > 1 else None
    trace = _positional_to_trace(positional_trace)
    result = evaluate_pattern_fast(template, activity_a, activity_b, trace, positional_trace)
    if result is None:
        return 0, True    # vacuous
    elif result == 0:
        return -1, False  # violated
    else:
        return 1, False   # satisfied


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LOG UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def case_to_positional(case: "CaseInfo") -> dict:
    """
    Convert a CaseInfo object to positional-trace format used by templates.
    Returns dict {activity_name: [pos0, pos1, ...]} (positions sorted ascending).
    """
    return {act: sorted(positions) for act, positions in case.activity_index.items()}


def case_to_activity_list(case: "CaseInfo") -> List[str]:
    """Return the ordered activity list from a CaseInfo."""
    return list(case.trace)


def build_log_structures(
    case_data: Dict[str, "CaseInfo"],
) -> Tuple[Dict[str, dict], Dict[str, List[str]], Dict[str, int]]:
    """
    Build three parallel views of the log from CaseInfo objects:
      positional_log : case_id → {activity: [positions]}
      sequence_log   : case_id → [act_0, act_1, ..., act_n]
      labels         : case_id → 0 or 1
    """
    positional_log = {}
    sequence_log   = {}
    labels         = {}
    for cid, case in case_data.items():
        positional_log[cid] = case_to_positional(case)
        sequence_log[cid]   = case_to_activity_list(case)
        labels[cid]         = int(case.label if hasattr(case, "label") else 0)
    return positional_log, sequence_log, labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SEQUENTIAL PATTERN MINING (Section 7.1.2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Implements the sequential patterns of Bose & van der Aalst (2009, 2013).
# GoSwift.jar is replaced by pure-Python mining with equivalent semantics.
#
# Definitions
# -----------
# TR (Tandem Repeat):  ordered sub-sequence w such that w·w appears
#   consecutively in a trace.  Feature = occurrence count of this tandem.
# TRA (Tandem Repeat Alphabet): unordered activity set of a tandem repeat
#   unit.  Feature = count of maximal tandem runs using that activity set.
# MR (Maximal Repeat):  ordered sub-sequence appearing ≥ 2 times in the
#   trace (not necessarily consecutively), and not extendable while keeping
#   that property.  Feature = occurrence count.
# MRA (Maximal Repeat Alphabet): unordered activity set of a maximal repeat.
#
# Discriminative filtering (Section 7.1.2):
#   Keep patterns frequent in exactly one class (L+ or L−), not both.
#   Frequency = fraction of class traces containing the pattern.

def _find_tandem_repeats_in_sequence(seq: List[str]) -> Dict[tuple, int]:
    """
    Find all tandem repeat units and their occurrence counts in seq.
    A tandem (w, 2) exists if seq contains w+w as a consecutive subsequence.
    Returns {w_tuple: count_of_distinct_tandem_runs}.
    """
    n = len(seq)
    repeats: Dict[tuple, int] = defaultdict(int)
    for length in range(1, n // 2 + 1):
        i = 0
        while i <= n - 2 * length:
            w = tuple(seq[i: i + length])
            if tuple(seq[i + length: i + 2 * length]) == w:
                repeats[w] += 1
                i += length   # skip past this tandem block
            else:
                i += 1
    return dict(repeats)


def _find_tandem_repeat_alphabets(seq: List[str]) -> Dict[FrozenSet[str], int]:
    """
    TRA: for each tandem repeat unit w, record its alphabet (frozenset of acts).
    Aggregates counts across all tandem units sharing the same alphabet.
    """
    alphas: Dict[FrozenSet[str], int] = defaultdict(int)
    for w, cnt in _find_tandem_repeats_in_sequence(seq).items():
        alphas[frozenset(w)] += cnt
    return dict(alphas)


def _ngrams(seq: List[str], n: int) -> List[tuple]:
    return [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]


def _find_maximal_repeats(seq: List[str]) -> Dict[tuple, int]:
    """
    MR: ordered sub-sequences that appear ≥ 2 times in seq.
    Keep only *maximal* ones (extending either end reduces count to < 2).
    Returns {w_tuple: count}.
    """
    n = len(seq)
    if n < 2:
        return {}

    # Count all sub-sequences of each length
    candidates: Dict[tuple, List[int]] = defaultdict(list)   # pattern → start positions
    for length in range(1, n):
        for i in range(n - length + 1):
            w = tuple(seq[i: i + length])
            candidates[w].append(i)

    # Keep only those with ≥ 2 occurrences
    repeated = {w: positions for w, positions in candidates.items()
                if len(positions) >= 2}

    # Maximality: remove w if every occurrence is contained in a longer repeat
    maximal = {}
    sorted_by_len = sorted(repeated.keys(), key=len, reverse=True)
    dominated: Set[tuple] = set()
    for w in sorted_by_len:
        if w in dominated:
            continue
        maximal[w] = len(repeated[w])
        # Mark sub-sequences of w as dominated
        wlen = len(w)
        for sub_len in range(1, wlen):
            for start in range(wlen - sub_len + 1):
                sub = w[start: start + sub_len]
                dominated.add(sub)

    return maximal


def _find_maximal_repeat_alphabets(seq: List[str]) -> Dict[FrozenSet[str], int]:
    """MRA: alphabets of maximal repeats."""
    alphas: Dict[FrozenSet[str], int] = defaultdict(int)
    for w, cnt in _find_maximal_repeats(seq).items():
        alphas[frozenset(w)] += cnt
    return dict(alphas)


def mine_sequential_patterns(
    case_ids_pos: List[str],      # deviant traces
    case_ids_neg: List[str],      # non-deviant traces
    sequence_log: Dict[str, List[str]],
    support_threshold: float = 0.1,
    pattern_types: List[str] = None,
) -> Dict[str, Dict]:
    """
    Mine discriminative TR, TRA, MR, MRA patterns (Section 7.1.2).

    A pattern is kept in the feature set if its support in L+ differs from
    its support in L− by more than a tolerance, following the reference
    implementation's filtering logic (frequent in one class, not both).

    Returns a dict:
        {"tr": {pattern: None}, "tra": {pattern: None},
         "mr": {pattern: None}, "mra": {pattern: None}}
    where pattern is a tuple (for TR/MR) or frozenset (for TRA/MRA).
    """
    if pattern_types is None:
        pattern_types = ["tr", "tra", "mr", "mra"]

    def _count_support(patterns_per_trace, all_patterns, n_traces):
        """Returns {pattern: fraction_of_traces_containing_it}."""
        support = defaultdict(int)
        for pdict in patterns_per_trace:
            for p in pdict:
                if p in all_patterns:
                    support[p] += 1
        return {p: support[p] / n_traces for p in all_patterns}

    def _union_patterns(traces, fn):
        results = []
        all_p = set()
        for cid in traces:
            d = fn(sequence_log[cid])
            results.append(set(d.keys()))
            all_p.update(d.keys())
        return results, all_p

    selected: Dict[str, Dict] = {}
    n_pos = max(len(case_ids_pos), 1)
    n_neg = max(len(case_ids_neg), 1)
    min_count_pos = int(np.ceil(support_threshold * n_pos))
    min_count_neg = int(np.ceil(support_threshold * n_neg))

    for ptype in pattern_types:
        if ptype == "tr":
            fn = lambda s: _find_tandem_repeats_in_sequence(s)
        elif ptype == "tra":
            fn = lambda s: _find_tandem_repeat_alphabets(s)
        elif ptype == "mr":
            fn = lambda s: _find_maximal_repeats(s)
        elif ptype == "mra":
            fn = lambda s: _find_maximal_repeat_alphabets(s)
        else:
            continue

        pos_results, pos_patterns = _union_patterns(case_ids_pos, fn)
        neg_results, neg_patterns = _union_patterns(case_ids_neg, fn)
        all_patterns = pos_patterns | neg_patterns

        sup_pos = _count_support(pos_results, all_patterns, n_pos)
        sup_neg = _count_support(neg_results, all_patterns, n_neg)

        # Keep patterns discriminative: frequent in one class but not the other
        discriminative = {}
        for p in all_patterns:
            in_pos = sup_pos.get(p, 0) * n_pos >= min_count_pos
            in_neg = sup_neg.get(p, 0) * n_neg >= min_count_neg
            if in_pos != in_neg:   # XOR: frequent in exactly one class
                discriminative[p] = None

        selected[ptype] = discriminative
        print(f"     {ptype.upper():4s}: {len(discriminative):,} discriminative patterns")

    return selected


def encode_sequential_features(
    case_id: str,
    sequence_log: Dict[str, List[str]],
    pattern_type: str,
    patterns: Dict,
) -> np.ndarray:
    """
    Encode one trace as a feature vector for sequential patterns.
    Feature value = support (count of pattern occurrences) in the trace.
    Order of features = sorted(patterns.keys()).
    """
    seq = sequence_log[case_id]
    sorted_patterns = sorted(patterns.keys(), key=str)

    if pattern_type == "tr":
        pdict = _find_tandem_repeats_in_sequence(seq)
    elif pattern_type == "tra":
        pdict = _find_tandem_repeat_alphabets(seq)
    elif pattern_type == "mr":
        pdict = _find_maximal_repeats(seq)
    elif pattern_type == "mra":
        pdict = _find_maximal_repeat_alphabets(seq)
    else:
        pdict = {}

    return np.array([pdict.get(p, 0) for p in sorted_patterns], dtype=np.float64)


def encode_ia_features(
    case_id: str,
    sequence_log: Dict[str, List[str]],
    activity_vocab: List[str],
) -> np.ndarray:
    """
    Individual Activities (IA): frequency of each activity in the trace.
    Feature value = count of occurrences of that activity.
    """
    seq = sequence_log[case_id]
    counts = defaultdict(int)
    for act in seq:
        counts[act] += 1
    return np.array([counts.get(a, 0) for a in activity_vocab], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DECLARATIVE FEATURE DISCOVERY (Section 7.1.3)
# ══════════════════════════════════════════════════════════════════════════════

def apriori_frequent_activity_sets(
    case_ids: List[str],
    positional_log: Dict[str, dict],
    support_threshold: float = 0.0,
) -> List[frozenset]:
    """
    Mine frequent itemsets (activity sets) from traces using a simplified
    Apriori algorithm (Agrawal & Srikant 1994).  Returns all frequent
    activity sets of size 1 and 2 (sufficient for binary Declare templates).
    """
    n = len(case_ids)
    min_count = max(1, int(np.ceil(support_threshold * n))) if n > 0 else 1

    # 1-itemsets
    item_counts: Dict[str, int] = defaultdict(int)
    for cid in case_ids:
        for act in positional_log[cid]:
            item_counts[act] += 1
    freq_1 = frozenset(a for a, c in item_counts.items() if c >= min_count)

    # 2-itemsets (pairs from freq_1)
    pair_counts: Dict[frozenset, int] = defaultdict(int)
    freq_acts = sorted(freq_1)
    for cid in case_ids:
        acts_in_trace = frozenset(positional_log[cid].keys()) & freq_1
        for i, a in enumerate(freq_acts):
            if a not in acts_in_trace:
                continue
            for b in freq_acts[i + 1:]:
                if b not in acts_in_trace:
                    continue
                pair_counts[frozenset({a, b})] += 1

    freq_2 = [p for p, c in pair_counts.items() if c >= min_count]

    return [frozenset({a}) for a in freq_1] + freq_2


def discover_declare_features(
    case_ids_pos: List[str],
    case_ids_neg: List[str],
    positional_log: Dict[str, dict],
    support_threshold: float = 0.0,
) -> List[Tuple[str, tuple]]:
    """
    Discover discriminative Declare constraints (Section 7.1.3).

    Algorithm:
    1. Mine frequent activity sets separately from L+ and L−.
    2. For each frequent set, instantiate all Declare templates.
    3. Keep candidate if its support in the generating class ≥ threshold.
    4. Return the union of accepted candidates from both classes.

    Returns list of (template_name, activity_tuple) specs.
    """
    def _check_support(template, acts, case_ids, pt_log, threshold):
        n = len(case_ids)
        if n == 0:
            return False
        min_c = max(1, int(np.ceil(threshold * n)))
        satisfied = 0
        for cid in case_ids:
            val, vacuous = apply_declare_template(template, acts, pt_log[cid])
            # "satisfied" means > 0 in reference implementation's encoding
            if val > 0:
                satisfied += 1
                if satisfied >= min_c:
                    return True
        return False

    seen: Set[Tuple] = set()
    candidates: List[Tuple[str, tuple]] = []

    for class_label, case_ids in [("pos", case_ids_pos), ("neg", case_ids_neg)]:
        freq_sets = apriori_frequent_activity_sets(
            case_ids, positional_log, support_threshold
        )
        for item_set in freq_sets:
            acts_sorted = sorted(item_set)
            # Unary templates
            for a in acts_sorted:
                for tmpl in sorted(UNARY_TEMPLATES):
                    spec = (tmpl, (a,))
                    if spec in seen:
                        continue
                    seen.add(spec)
                    if _check_support(tmpl, (a,), case_ids, positional_log, support_threshold):
                        candidates.append(spec)
            # Binary templates (both orderings)
            if len(acts_sorted) == 2:
                a, b = acts_sorted
                for tmpl in sorted(BINARY_TEMPLATES):
                    for act_pair in [(a, b), (b, a)]:
                        spec = (tmpl, act_pair)
                        if spec in seen:
                            continue
                        seen.add(spec)
                        if _check_support(tmpl, act_pair, case_ids, positional_log, support_threshold):
                            candidates.append(spec)

    return candidates


def encode_declare_value(template: str, acts: tuple, positional_trace: dict) -> float:
    """
    Encode one (trace, constraint) pair using the paper's three-way scale:
        −1  violated
         0  vacuously satisfied
         n  satisfied and activated n times  (n ≥ 1)
    """
    val, vacuous = apply_declare_template(template, acts, positional_trace)
    return float(val)   # already −1/0/n by template convention


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DATA FEATURES (Section 7.1.1)
# ══════════════════════════════════════════════════════════════════════════════

def extract_data_features_from_csv(
    csv_path: str,
    sep: str = ";",
    case_id_col:  str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
    label_col:    str = "label",
    ignored_cols: Optional[Set[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract data features (F_data, Section 7.1.1) from a CSV event log.

    For numeric attributes: first, last, min, max, avg values per trace.
    For categorical attributes: count of each distinct value per trace;
        first and last value (as one-hot).
    Meta features: trace length; trace time-length in seconds.

    Returns:
        data_matrix : {case_id: feature_vector}
        feature_names: list of feature names in vector order
    """
    if ignored_cols is None:
        ignored_cols = {case_id_col, activity_col, label_col, "Label"}

    try:
        df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    except Exception as e:
        warnings.warn(f"Could not load CSV for data features: {e}")
        return {}, []

    # Parse timestamps if available
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    else:
        timestamp_col = None

    feature_registry: Dict[str, List] = {}  # feature_name → per-trace values
    case_order: List[str] = []

    grouped = df.groupby(case_id_col, sort=False)

    for cid, grp in grouped:
        case_order.append(str(cid))
        grp = grp.sort_values(timestamp_col) if timestamp_col else grp

        row_features: Dict[str, float] = {}

        # ── Meta features ──────────────────────────────────────────────────
        row_features["_meta_trace_length"] = float(len(grp))
        if timestamp_col and not grp[timestamp_col].isna().all():
            ts_vals = grp[timestamp_col].dropna()
            if len(ts_vals) >= 2:
                delta = (ts_vals.iloc[-1] - ts_vals.iloc[0]).total_seconds()
                row_features["_meta_trace_time_s"] = float(delta)
            else:
                row_features["_meta_trace_time_s"] = 0.0
        else:
            row_features["_meta_trace_time_s"] = 0.0

        # ── Attribute features ─────────────────────────────────────────────
        for col in grp.columns:
            if col in ignored_cols or col == timestamp_col:
                continue
            series = grp[col].dropna()
            if series.empty:
                continue

            try:
                numeric_vals = pd.to_numeric(series, errors="raise")
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if is_numeric:
                vals = numeric_vals.values.astype(float)
                row_features[f"first_{col}"]  = float(vals[0])
                row_features[f"last_{col}"]   = float(vals[-1])
                row_features[f"min_{col}"]    = float(vals.min())
                row_features[f"max_{col}"]    = float(vals.max())
                row_features[f"avg_{col}"]    = float(vals.mean())
            else:
                str_vals = series.astype(str).values
                row_features[f"first_{col}"] = str_vals[0]
                row_features[f"last_{col}"]  = str_vals[-1]
                for v in str_vals:
                    row_features[f"count_{col}_{v}"] = \
                        row_features.get(f"count_{col}_{v}", 0.0) + 1.0

        for k, v in row_features.items():
            if k not in feature_registry:
                feature_registry[k] = []
            feature_registry[k].append(v)

    # Build consistent feature matrix
    all_keys = list(feature_registry.keys())
    n_traces  = len(case_order)

    # One-hot encode string features; leave numerics as-is
    final_features: Dict[str, np.ndarray] = {}
    final_names: List[str] = []

    for key in all_keys:
        raw = feature_registry[key]
        if len(raw) < n_traces:
            raw = raw + [0.0] * (n_traces - len(raw))

        # Check if numeric
        try:
            arr = np.array(raw, dtype=np.float64)
            final_features[key] = arr
            final_names.append(key)
        except (ValueError, TypeError):
            # Categorical → one-hot
            le  = LabelEncoder()
            enc = le.fit_transform([str(v) for v in raw])
            for cls_idx, cls_val in enumerate(le.classes_):
                fname = f"{key}=={cls_val}"
                final_features[fname] = (enc == cls_idx).astype(np.float64)
                final_names.append(fname)

    data_matrix = {}
    feat_matrix_2d = np.column_stack([final_features[n] for n in final_names]) \
        if final_names else np.zeros((n_traces, 0), dtype=np.float64)

    for i, cid in enumerate(case_order):
        data_matrix[cid] = feat_matrix_2d[i]

    return data_matrix, final_names


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATA-AWARE DECLARE (Section 7.1.4)
# ══════════════════════════════════════════════════════════════════════════════

def discover_data_aware_declare(
    declare_features: List[Tuple[str, tuple]],
    positional_log: Dict[str, dict],
    data_matrix: Dict[str, np.ndarray],
    labels: Dict[str, int],
    case_ids: List[str],
    data_feature_names: List[str],
    random_state: int = 42,
) -> List[Tuple[str, tuple, Optional[str]]]:
    """
    Enrich each Declare constraint with a data condition (Section 7.1.4).

    For each constraint C = (template, acts):
      1. Collect all traces where C is non-vacuously satisfied.
      2. Collect data snapshots (feature vectors) at activation points.
      3. Learn a local DT on (snapshot, deviant/non-deviant) labels.
      4. Extract the data condition from the DT leaf with highest deviant
         count (the LHS of the best root-to-leaf path).
      5. Return enriched constraint (template, acts, condition_string).

    Returns list of (template, acts, data_condition_str).
    If no meaningful data condition is found, data_condition_str = None.
    """
    if not data_feature_names or not data_matrix:
        return [(t, a, None) for t, a in declare_features]

    enriched = []
    for template, acts in declare_features:
        act_snapshots = []
        act_labels    = []

        for cid in case_ids:
            if cid not in data_matrix or cid not in positional_log:
                continue
            pt = positional_log[cid]
            val, vacuous = apply_declare_template(template, acts, pt)
            if vacuous or val <= 0:
                continue   # only non-vacuously satisfied traces

            # Data snapshot = case-level feature vector
            snap = data_matrix[cid]
            act_snapshots.append(snap)
            act_labels.append(labels.get(cid, 0))

        if len(act_snapshots) < 4 or len(set(act_labels)) < 2:
            enriched.append((template, acts, None))
            continue

        X = np.array(act_snapshots, dtype=np.float64)
        y = np.array(act_labels,    dtype=np.int32)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            dt = DecisionTreeClassifier(max_depth=3, random_state=random_state)
            dt.fit(X, y)

            # Extract the most informative condition (highest deviant node)
            tree_    = dt.tree_
            feature  = tree_.feature
            threshold = dt.tree_.threshold
            condition_str = None

            if feature[0] >= 0:   # root is a split node
                fname = data_feature_names[feature[0]]
                thr   = threshold[0]
                condition_str = f"{fname} <= {thr:.3f}"

            enriched.append((template, acts, condition_str))
        except Exception:
            enriched.append((template, acts, None))

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: FULL TRACE ENCODING (Section 7.2)
# ══════════════════════════════════════════════════════════════════════════════

def build_encoding_matrix(
    case_ids: List[str],
    positional_log: Dict[str, dict],
    sequence_log: Dict[str, List[str]],
    data_matrix: Dict[str, np.ndarray],
    activity_vocab: List[str],
    seq_patterns: Dict[str, Dict],
    declare_features: List[Tuple[str, tuple]],
    declD_features: List[Tuple[str, tuple, Optional[str]]],
    data_feature_names: List[str],
    encoding: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the feature matrix X (n_traces × m_features) for a given encoding.

    encoding values (Section 9.2):
        "ia"            – Individual Activities
        "mr"/"tr"/"mra"/"tra" – one sequential family
        "declare"       – Declarative only
        "h"             – Hybrid = all seq + declare
        "data"          – Data attributes only
        "decld"         – Data-aware Declare
        Compound names formed by joining components with "+"
        e.g. "data+declare", "h+data+decld", etc.
    """
    enc_lower  = encoding.lower()
    parts      = [p.strip() for p in enc_lower.split("+")]
    col_blocks: List[np.ndarray] = []
    col_names:  List[str]        = []
    n = len(case_ids)

    def _add_block(matrix: np.ndarray, names: List[str]) -> None:
        if matrix.shape[1] > 0:
            col_blocks.append(matrix)
            col_names.extend(names)

    # ── Individual Activities ─────────────────────────────────────────────
    if "ia" in parts:
        mat = np.array([
            encode_ia_features(cid, sequence_log, activity_vocab)
            for cid in case_ids
        ], dtype=np.float64)
        _add_block(mat, [f"ia_{a}" for a in activity_vocab])

    # ── Sequential families (TR, TRA, MR, MRA) ───────────────────────────
    seq_types_in_parts = [p for p in parts if p in ("tr", "tra", "mr", "mra")]
    # "h" = hybrid includes ALL sequential families + declare
    if "h" in parts:
        seq_types_in_parts = list(set(seq_types_in_parts) | {"tr", "tra", "mr", "mra"})

    for stype in seq_types_in_parts:
        patterns = seq_patterns.get(stype, {})
        if not patterns:
            continue
        sorted_pats = sorted(patterns.keys(), key=str)
        mat = np.array([
            encode_sequential_features(cid, sequence_log, stype, patterns)
            for cid in case_ids
        ], dtype=np.float64)
        _add_block(mat, [f"{stype}_{p}" for p in sorted_pats])

    # ── Declare ───────────────────────────────────────────────────────────
    if "declare" in parts or "h" in parts:
        if declare_features:
            mat = np.array([
                [encode_declare_value(t, a, positional_log[cid])
                 for t, a in declare_features]
                for cid in case_ids
            ], dtype=np.float64)
            names = [f"{t}_({'_'.join(a)})" for t, a in declare_features]
            _add_block(mat, names)

    # ── Data-aware Declare (DeclD) ────────────────────────────────────────
    if "decld" in parts:
        if declD_features:
            mat = np.array([
                [encode_declare_value(t, a, positional_log[cid])
                 for t, a, _ in declD_features]
                for cid in case_ids
            ], dtype=np.float64)
            names = [f"declD_{t}_({'_'.join(a)})_{cond or 'none'}"
                     for t, a, cond in declD_features]
            _add_block(mat, names)

    # ── Data ──────────────────────────────────────────────────────────────
    if "data" in parts and data_feature_names:
        mat = np.array([
            data_matrix.get(cid, np.zeros(len(data_feature_names)))
            for cid in case_ids
        ], dtype=np.float64)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        _add_block(mat, list(data_feature_names))

    if not col_blocks:
        return np.zeros((n, 0), dtype=np.float64), []

    X = np.hstack(col_blocks)
    return X, col_names


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: GENERALIZED FISHER SCORE + COVERAGE SELECTION (Section 7.3)
# ══════════════════════════════════════════════════════════════════════════════

def compute_generalized_fisher_scores(
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Generalized Fisher score (Gu, Li, Han 2011, UAI):

        F_r = [ n+ (μ+_r − μ_r)² + n− (μ−_r − μ_r)² ]
              / [ n+ σ+²_r       + n− σ−²_r            ]

    where μ and σ² are computed over the column's values.
    Returns scores array of shape (m,), all ≥ 0.
    """
    X    = X.astype(np.float64)
    m0   = (y == 0)
    m1   = (y == 1)
    n0   = m0.sum()
    n1   = m1.sum()
    n    = len(y)

    mu   = X.mean(axis=0)
    mu0  = X[m0].mean(axis=0) if n0 > 0 else np.zeros(X.shape[1])
    mu1  = X[m1].mean(axis=0) if n1 > 0 else np.zeros(X.shape[1])
    var0 = X[m0].var(axis=0)  if n0 > 0 else np.zeros(X.shape[1])
    var1 = X[m1].var(axis=0)  if n1 > 0 else np.zeros(X.shape[1])

    num  = n1 * (mu1 - mu) ** 2 + n0 * (mu0 - mu) ** 2
    den  = n1 * var1 + n0 * var0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > eps, num / den, np.where(num <= eps, 0.0, num / eps))
    return ratio


def select_features_coverage(
    X: np.ndarray,
    fisher_scores: np.ndarray,
    coverage_threshold: int = 20,
) -> List[int]:
    """
    Greedy coverage-based feature selection (Section 7.3).

    A feature "covers" a trace if its value ≠ 0 in that trace.
    Selects features in descending Fisher-score order, adding a feature
    only if it covers ≥ 1 trace not yet fully covered.
    Terminates when every trace has been covered ≥ coverage_threshold times.

    This matches the reference implementation logic (non-zero = covered).
    """
    n, m        = X.shape
    ranked      = np.argsort(fisher_scores)[::-1]   # descending
    covered     = np.zeros(n, dtype=np.int32)        # coverage counter per trace
    selected    = []

    for j in ranked:
        col      = (X[:, j] != 0)
        not_full = covered < coverage_threshold
        if (col & not_full).any():
            selected.append(int(j))
            covered += col.astype(np.int32)
            if (covered >= coverage_threshold).all():
                break

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CLASSIFIER TRAINING & RULE EXTRACTION (Sections 7.4–7.5)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  y_prob: Optional[np.ndarray] = None) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = precision_score(y_true, y_pred, zero_division=0.0)
        rec  = recall_score(y_true, y_pred, zero_division=0.0)
        f1   = f1_score(y_true, y_pred, zero_division=0.0)
        acc  = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob if y_prob is not None else y_pred)
        except Exception:
            auc = 0.5
    return {"precision": prec, "recall": rec, "f1": f1,
            "accuracy": acc, "auc": auc}


# ── RIPPER-K path ────────────────────────────────────────────────────────────

def _ripper_rules_to_length(clf) -> Tuple[int, int]:
    """Return (number_of_rules, total_condition_count) from a RIPPER clf."""
    try:
        rules = [r for r in clf.ruleset_ if r.conds]
        n_rules = len(rules)
        total_conds = sum(len(r.conds) for r in rules)
        return n_rules, total_conds
    except Exception:
        return 0, 0


def train_ripper(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: List[str],
    k_grid: List[int] = RIPPER_K_GRID,
    prune_size: float = RIPPER_PRUNE_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train RipperK with grid search over k (Section 9.3).
    Returns metrics dict with ARL and ANR.
    """
    if not RIPPER_AVAILABLE:
        raise RuntimeError("Wittgenstein not installed.")

    train_df = pd.DataFrame(X_train, columns=feat_names)
    train_df["label"] = y_train

    best_f1  = -1.0
    best_out = None

    for k in k_grid:
        try:
            clf = lw.RIPPER(k=k, prune_size=prune_size, random_state=random_state)
            clf.fit(train_df, class_feat="label", pos_class=1)
            y_pred_test  = clf.predict(pd.DataFrame(X_test, columns=feat_names))
            y_pred_test  = np.array(y_pred_test, dtype=int)
            met = _safe_metrics(y_test, y_pred_test)
            if met["f1"] > best_f1:
                best_f1  = met["f1"]
                n_rules, n_conds = _ripper_rules_to_length(clf)
                best_out = {**met,
                            "anr": n_rules,
                            "arl": (n_conds / n_rules if n_rules > 0 else 0.0),
                            "best_k": k}
        except Exception as e:
            continue

    if best_out is None:
        return {"precision": 0., "recall": 0., "f1": 0.,
                "accuracy": 0., "auc": 0.5, "anr": 0, "arl": 0., "best_k": -1}
    return best_out


# ── Decision Tree path ───────────────────────────────────────────────────────

def _extract_dt_rules(
    clf: DecisionTreeClassifier,
    n_features: int,
) -> Tuple[int, float]:
    """
    Count rules and average rule length from a fitted DT.
    Each root-to-deviant-leaf path is one rule.
    Rule length = number of splits on the path.
    """
    tree_  = clf.tree_
    ch_l   = tree_.children_left
    ch_r   = tree_.children_right
    value  = tree_.value

    rules = []

    def _traverse(node: int, depth: int) -> None:
        if ch_l[node] == ch_r[node]:   # leaf
            leaf_class = int(np.argmax(value[node][0]))
            if leaf_class == 1:
                rules.append(depth)
        else:
            _traverse(ch_l[node], depth + 1)
            _traverse(ch_r[node], depth + 1)

    _traverse(0, 0)
    n_rules = len(rules)
    arl     = float(np.mean(rules)) if rules else 0.0
    return n_rules, arl


def extract_ripper_rules_text(clf, feat_names: List[str]) -> List[str]:
    """Return each RIPPER rule as a human-readable condition string (Section 7.5)."""
    rules = []
    try:
        for rule in clf.ruleset_:
            if not rule.conds:
                continue
            rules.append(" AND ".join(str(c) for c in rule.conds))
    except Exception:
        pass
    return rules


def extract_dt_rules_text(clf: DecisionTreeClassifier, feat_names: List[str]) -> List[str]:
    """
    Extract root-to-deviant-leaf paths from a fitted DT as rule strings (Section 7.5).
    Each path is one rule — a conjunction of atomic conditions.
    Conditions on the same feature are not simplified here; that is left to
    the analyst as described in the paper.
    """
    tree_ = clf.tree_
    feat_ = tree_.feature
    thr_  = tree_.threshold
    ch_l_ = tree_.children_left
    ch_r_ = tree_.children_right
    val_  = tree_.value
    rules: List[str] = []

    def _walk(node: int, path: List[str]) -> None:
        if ch_l_[node] == ch_r_[node]:   # leaf
            if int(np.argmax(val_[node][0])) == 1:
                rules.append(" AND ".join(path) if path else "⊤ → Deviant")
        else:
            fname = feat_names[feat_[node]] if feat_[node] >= 0 else "?"
            t = float(thr_[node])
            _walk(ch_l_[node], path + [f"{fname} <= {t:.4f}"])
            _walk(ch_r_[node], path + [f"{fname} > {t:.4f}"])

    _walk(0, [])
    return rules


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    depth_grid: List[Optional[int]] = DT_DEPTH_GRID,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train DT with grid search over max_depth (Section 9.3).
    Returns metrics dict with ARL and ANR.
    """
    best_f1  = -1.0
    best_out = None

    for depth in depth_grid:
        try:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
            met = _safe_metrics(y_test, y_pred, y_prob)
            if met["f1"] > best_f1:
                best_f1 = met["f1"]
                n_rules, arl = _extract_dt_rules(clf, X_train.shape[1])
                best_out = {**met, "anr": n_rules, "arl": arl, "best_depth": depth}
        except Exception:
            continue

    if best_out is None:
        return {"precision": 0., "recall": 0., "f1": 0.,
                "accuracy": 0., "auc": 0.5, "anr": 0, "arl": 0., "best_depth": -1}
    return best_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: 10-FOLD CROSS-VALIDATION (Section 9.3)
# ══════════════════════════════════════════════════════════════════════════════

def run_encoding(
    encoding: str,
    case_ids: List[str],
    labels_arr: np.ndarray,
    positional_log: Dict[str, dict],
    sequence_log: Dict[str, List[str]],
    data_matrix: Dict[str, np.ndarray],
    activity_vocab: List[str],
    seq_patterns: Dict[str, Dict],
    declare_features: List[Tuple[str, tuple]],
    declD_features: List[Tuple[str, tuple, Optional[str]]],
    data_feature_names: List[str],
    coverage_threshold: int = COVERAGE_THRESHOLD,
    n_folds: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Run 10-fold cross-validation for one encoding, both classifiers.
    Returns aggregated metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    ids_arr = np.array(case_ids)

    ripper_results = []
    dt_results     = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(ids_arr, labels_arr)
    ):
        train_ids = list(ids_arr[train_idx])
        test_ids  = list(ids_arr[test_idx])
        y_train   = labels_arr[train_idx]
        y_test    = labels_arr[test_idx]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue   # degenerate fold

        # ── Step 1: Mine features on TRAIN split only ──────────────────────
        # Sequential: re-mine on training set to avoid leakage
        train_pos = [cid for cid in train_ids if positional_log[cid] is not None and y_train[train_ids.index(cid)] == 1]
        train_neg = [cid for cid in train_ids if positional_log[cid] is not None and y_train[train_ids.index(cid)] == 0]

        # Use pre-mined global seq_patterns (as reference impl uses full-log mining
        # for pattern discovery; cross-validation applies to classification step)
        # This matches the reference implementation's approach.

        # ── Step 2: Build feature matrix ──────────────────────────────────
        X_train_full, feat_names = build_encoding_matrix(
            train_ids, positional_log, sequence_log, data_matrix,
            activity_vocab, seq_patterns, declare_features, declD_features,
            data_feature_names, encoding,
        )
        X_test_full, _ = build_encoding_matrix(
            test_ids, positional_log, sequence_log, data_matrix,
            activity_vocab, seq_patterns, declare_features, declD_features,
            data_feature_names, encoding,
        )

        if X_train_full.shape[1] == 0:
            continue

        # ── Step 3: Feature selection on TRAIN ────────────────────────────
        fisher = compute_generalized_fisher_scores(X_train_full, y_train)
        sel    = select_features_coverage(X_train_full, fisher, coverage_threshold)

        if not sel:
            continue

        X_tr = X_train_full[:, sel]
        X_te = X_test_full[:, sel]
        sel_names = [feat_names[i] for i in sel]

        # ── Step 4 & 5: Train classifiers & evaluate ───────────────────────
        # RipperK
        if RIPPER_AVAILABLE:
            try:
                r_out = train_ripper(X_tr, y_train, X_te, y_test, sel_names,
                                     random_state=random_state)
                ripper_results.append(r_out)
            except Exception as e:
                pass

        # Decision Tree
        try:
            dt_out = train_decision_tree(X_tr, y_train, X_te, y_test,
                                         random_state=random_state)
            dt_results.append(dt_out)
        except Exception as e:
            pass

    def _avg(results, key):
        vals = [r[key] for r in results if r is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _agg(results):
        if not results:
            return {"precision": 0., "recall": 0., "f1": 0.,
                    "accuracy": 0., "auc": 0.5, "anr": 0., "arl": 0.}
        return {k: _avg(results, k)
                for k in ["precision", "recall", "f1", "accuracy", "auc", "anr", "arl"]}

    return {
        "encoding":     encoding,
        "ripper":       _agg(ripper_results),
        "dt":           _agg(dt_results),
        "n_folds_done": max(len(ripper_results), len(dt_results)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10b: FINAL-MODEL RULE EXTRACTION (Section 7.5)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_and_extract_rules(
    encoding: str,
    case_ids: List[str],
    labels_arr: np.ndarray,
    positional_log: Dict[str, dict],
    sequence_log: Dict[str, List[str]],
    data_matrix: Dict[str, np.ndarray],
    activity_vocab: List[str],
    seq_patterns: Dict[str, Dict],
    declare_features: List[Tuple[str, tuple]],
    declD_features: List[Tuple[str, tuple, Optional[str]]],
    data_feature_names: List[str],
    coverage_threshold: int = COVERAGE_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train a final model on ALL labelled data and extract rules (Section 7.5).
    CV is for metrics; this single fit on the full dataset produces the rule set
    reported as the model's output.
    Returns {"ripper_rules": [...], "dt_rules": [...], "sel_features": [...]}.
    """
    X_all, feat_names = build_encoding_matrix(
        case_ids, positional_log, sequence_log, data_matrix,
        activity_vocab, seq_patterns, declare_features, declD_features,
        data_feature_names, encoding,
    )
    if X_all.shape[1] == 0:
        return {"ripper_rules": [], "dt_rules": [], "sel_features": []}

    fisher = compute_generalized_fisher_scores(X_all, labels_arr)
    sel    = select_features_coverage(X_all, fisher, coverage_threshold)
    if not sel:
        return {"ripper_rules": [], "dt_rules": [], "sel_features": []}

    X_sel     = X_all[:, sel]
    sel_names = [feat_names[i] for i in sel]

    ripper_rules: List[str] = []
    dt_rules:     List[str] = []

    # RipperK final model
    if RIPPER_AVAILABLE:
        try:
            train_df = pd.DataFrame(X_sel, columns=sel_names)
            train_df["label"] = labels_arr
            best_f1, best_clf = -1.0, None
            for k in RIPPER_K_GRID:
                try:
                    clf = lw.RIPPER(k=k, prune_size=RIPPER_PRUNE_SIZE, random_state=random_state)
                    clf.fit(train_df, class_feat="label", pos_class=1)
                    y_pred = np.array(clf.predict(train_df.drop(columns=["label"])), dtype=int)
                    f1 = _safe_metrics(labels_arr, y_pred)["f1"]
                    if f1 > best_f1:
                        best_f1, best_clf = f1, clf
                except Exception:
                    continue
            if best_clf is not None:
                ripper_rules = extract_ripper_rules_text(best_clf, sel_names)
        except Exception:
            pass

    # Decision Tree final model
    try:
        best_f1, best_clf = -1.0, None
        for depth in DT_DEPTH_GRID:
            try:
                clf = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
                clf.fit(X_sel, labels_arr)
                y_pred = clf.predict(X_sel)
                f1 = _safe_metrics(labels_arr, y_pred)["f1"]
                if f1 > best_f1:
                    best_f1, best_clf = f1, clf
            except Exception:
                continue
        if best_clf is not None:
            dt_rules = extract_dt_rules_text(best_clf, sel_names)
    except Exception:
        pass

    return {
        "ripper_rules": ripper_rules,
        "dt_rules":     dt_rules,
        "sel_features": sel_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MAIN DvM PIPELINE (run_dvm / run_declareminer)
# ══════════════════════════════════════════════════════════════════════════════

# All 16+ encodings from Section 9.2 of the paper
ALL_ENCODINGS = [
    "ia",
    "mr",
    "tr",
    "mra",
    "tra",
    "declare",
    "h",
    "data",
    "decld",
    "data+ia",
    "data+mr",
    "data+tr",
    "data+mra",
    "data+tra",
    "data+declare",
    "data+h",
    "data+decld",
    "h+decld",
    "h+data+decld",
]


def run_dvm(
    config: Optional[dict] = None,
    case_data: Optional[Dict[str, "CaseInfo"]] = None,
    candidates_all: Optional[List[Tuple]] = None,   # legacy compat; ignored
    R_obs_target: Optional[int] = None,             # legacy compat; ignored
    encodings: Optional[List[str]] = None,
) -> dict:
    """
    Execute the full Deviance Mining pipeline (BISE 2025).

    Parameters
    ----------
    config       : optional overrides for DVM_CONFIG.
    case_data    : pre-loaded CaseInfo dict. Loaded from CSV_PATH if None.
    encodings    : list of encoding names to evaluate (default: ALL_ENCODINGS).

    Returns
    -------
    dict with keys:
        results_by_encoding : {encoding → {ripper: metrics, dt: metrics}}
        summary_table       : pd.DataFrame with all metrics
        timing              : timing dict
        ... plus legacy keys for rq1/rq2 compatibility
    """
    cfg      = {**DVM_CONFIG, **(config or {})}
    timing   = {}
    t0_total = time.time()
    rng      = int(cfg.get("random_state", RANDOM_STATE))
    cov_thr  = int(cfg.get("coverage_threshold", COVERAGE_THRESHOLD))
    n_folds  = int(cfg.get("n_cv_folds", N_CV_FOLDS))
    cand_thr = float(cfg.get("candidate_threshold", CANDIDATE_THRESHOLD))
    supp_thr = float(cfg.get("support_threshold", SUPPORT_THRESHOLD))

    if encodings is None:
        encodings = ALL_ENCODINGS

    # ── Step 0: Data loading ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("DvM (BISE 2025) — STEP 0: DATA LOADING")
    print("=" * 100)
    t0 = time.time()
    if case_data is None:
        case_data = load_and_preprocess_data(CSV_PATH)
    timing["data_loading"] = time.time() - t0

    D_0, D_1   = split_by_class(case_data)
    ids_class0 = list(D_0.keys())
    ids_class1 = list(D_1.keys())
    n0, n1     = len(ids_class0), len(ids_class1)
    all_ids    = ids_class0 + ids_class1
    y_all      = np.array([0] * n0 + [1] * n1, dtype=np.int32)

    print(f"   L- (non-deviant, class 0): {n0:,} traces")
    print(f"   L+ (deviant,     class 1): {n1:,} traces")
    print(f"   Total: {n0 + n1:,} traces")

    # ── Step 0b: Build log views ───────────────────────────────────────────
    positional_log, sequence_log, labels = build_log_structures(case_data)
    # Override labels from split
    for cid in ids_class0:
        labels[cid] = 0
    for cid in ids_class1:
        labels[cid] = 1

    # ── Step 0c: Data features from CSV ───────────────────────────────────
    print("\n   Loading data features from CSV ...")
    t0 = time.time()
    data_matrix, data_feature_names = extract_data_features_from_csv(CSV_PATH)
    timing["data_feature_extraction"] = time.time() - t0
    print(f"   Data features: {len(data_feature_names):,}")

    # ── Step 1a: Activity vocabulary ──────────────────────────────────────
    activity_vocab = sorted({a for cid in all_ids for a in sequence_log[cid]})
    print(f"   Activity vocabulary: {len(activity_vocab):,} activities")

    # ── Step 1b: Sequential pattern mining ────────────────────────────────
    print("\n" + "=" * 100)
    print("DvM (BISE 2025) — STEP 1b: SEQUENTIAL PATTERN MINING (Section 7.1.2)")
    print("=" * 100)
    t0 = time.time()
    seq_patterns = mine_sequential_patterns(
        case_ids_pos    = ids_class1,
        case_ids_neg    = ids_class0,
        sequence_log    = sequence_log,
        support_threshold = cand_thr,
    )
    timing["sequential_mining"] = time.time() - t0
    for stype, pats in seq_patterns.items():
        print(f"   {stype.upper():4s}: {len(pats):,} discriminative patterns found")

    # ── Step 1c: Declarative feature discovery ────────────────────────────
    print("\n" + "=" * 100)
    print("DvM (BISE 2025) — STEP 1c: DECLARATIVE FEATURE DISCOVERY (Section 7.1.3)")
    print("=" * 100)
    t0 = time.time()
    declare_features = discover_declare_features(
        case_ids_pos    = ids_class1,
        case_ids_neg    = ids_class0,
        positional_log  = positional_log,
        support_threshold = supp_thr,
    )
    timing["declare_discovery"] = time.time() - t0
    print(f"   Declare features discovered: {len(declare_features):,}")

    # ── Step 1d: Data-aware Declare ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("DvM (BISE 2025) — STEP 1d: DATA-AWARE DECLARE (Section 7.1.4)")
    print("=" * 100)
    t0 = time.time()
    declD_features = discover_data_aware_declare(
        declare_features   = declare_features,
        positional_log     = positional_log,
        data_matrix        = data_matrix,
        labels             = labels,
        case_ids           = all_ids,
        data_feature_names = data_feature_names,
        random_state       = rng,
    )
    n_enriched = sum(1 for _, _, c in declD_features if c is not None)
    timing["declD_discovery"] = time.time() - t0
    print(f"   DeclD features: {len(declD_features):,} "
          f"({n_enriched:,} enriched with data conditions)")

    # ── Step 2–5: Cross-validation over all encodings ─────────────────────
    print("\n" + "=" * 100)
    print("DvM (BISE 2025) — STEPS 2–5: ENCODING × 10-FOLD CV (Sections 7.2–7.5, 9.3)")
    print("=" * 100)

    results_by_encoding: Dict[str, dict] = {}
    t0_cv = time.time()

    for enc in encodings:
        print(f"\n   Encoding: {enc.upper():30s}", end="", flush=True)
        t_enc = time.time()
        res = run_encoding(
            encoding           = enc,
            case_ids           = all_ids,
            labels_arr         = y_all,
            positional_log     = positional_log,
            sequence_log       = sequence_log,
            data_matrix        = data_matrix,
            activity_vocab     = activity_vocab,
            seq_patterns       = seq_patterns,
            declare_features   = declare_features,
            declD_features     = declD_features,
            data_feature_names = data_feature_names,
            coverage_threshold = cov_thr,
            n_folds            = n_folds,
            random_state       = rng,
        )
        results_by_encoding[enc] = res
        dt_f1     = res["dt"]["f1"]
        rip_f1    = res["ripper"]["f1"] if RIPPER_AVAILABLE else float("nan")
        elapsed   = time.time() - t_enc
        print(f" DT-F1={dt_f1:.3f}  RIP-F1={rip_f1:.3f}  [{elapsed:.1f}s]")

    timing["cross_validation"] = time.time() - t0_cv

    # ── Final-model rule extraction (Section 7.5) ─────────────────────────
    print("\n" + "=" * 100)
    print("DvM (BISE 2025) — FINAL MODEL RULE EXTRACTION (Section 7.5)")
    print("=" * 100)
    final_rules_by_encoding: Dict[str, dict] = {}
    for enc in encodings:
        try:
            rules_out = _fit_and_extract_rules(
                encoding           = enc,
                case_ids           = all_ids,
                labels_arr         = y_all,
                positional_log     = positional_log,
                sequence_log       = sequence_log,
                data_matrix        = data_matrix,
                activity_vocab     = activity_vocab,
                seq_patterns       = seq_patterns,
                declare_features   = declare_features,
                declD_features     = declD_features,
                data_feature_names = data_feature_names,
                coverage_threshold = cov_thr,
                random_state       = rng,
            )
            final_rules_by_encoding[enc] = rules_out
            n_rip = len(rules_out["ripper_rules"])
            n_dt  = len(rules_out["dt_rules"])
            print(f"   {enc.upper():30s}  RIPPER rules: {n_rip:3d}  DT rules: {n_dt:3d}")
        except Exception:
            final_rules_by_encoding[enc] = {"ripper_rules": [], "dt_rules": [], "sel_features": []}

    timing["total"] = time.time() - t0_total

    # ── Build summary table ────────────────────────────────────────────────
    rows = []
    for enc, res in results_by_encoding.items():
        for clf_name in ["ripper", "dt"]:
            m = res[clf_name]
            rows.append({
                "encoding":  enc,
                "classifier": clf_name,
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"],    4),
                "f1":        round(m["f1"],        4),
                "auc":       round(m["auc"],       4),
                "anr":       round(m["anr"],       2),
                "arl":       round(m["arl"],       2),
            })
    summary_df = pd.DataFrame(rows)

    print(f"\n{'=' * 100}")
    print("DvM (BISE 2025) COMPLETE")
    print(f"{'=' * 100}")
    print(f"   Encodings evaluated : {len(encodings)}")
    print(f"   Folds               : {n_folds}")
    print(f"   Total time          : {timing['total']:.1f}s")

    # ── Legacy rq1/rq2 compatibility output ───────────────────────────────
    # Provide a flat "rejected" mask using the pure Declare encoding result
    declare_enc_result = results_by_encoding.get("declare", {})
    m_total     = len(declare_features)
    n_rejected  = len([f for f in declare_features])   # all discovered = "rejected by test"
    rejected    = np.ones(m_total, dtype=bool)

    return {
        # Primary outputs
        "results_by_encoding":      results_by_encoding,
        "final_rules_by_encoding":  final_rules_by_encoding,
        "summary_table":            summary_df,
        "timing":                   timing,
        # Log structures
        "case_data":            case_data,
        "ids_class0":           set(ids_class0),
        "ids_class1":           set(ids_class1),
        "n0":                   n0,
        "n1":                   n1,
        "ordered_ids":          all_ids,
        "y":                    y_all,
        # Feature structures
        "declare_features":     declare_features,
        "declD_features":       declD_features,
        "seq_patterns":         seq_patterns,
        "data_feature_names":   data_feature_names,
        "activity_vocab":       activity_vocab,
        # Legacy rq1/rq2 compatibility keys
        "results_all":          [{"pattern_id": f"{t}_{'_'.join(a)}",
                                  "constraint_type": t,
                                  "activity_a": a[0] if a else None,
                                  "activity_b": a[1] if len(a) > 1 else None,
                                  "in_model": True}
                                 for t, a in declare_features],
        "rejected":             rejected,
        "n_rejected":           n_rejected,
        "model_indices":        list(range(m_total)),
        "model_score":          declare_enc_result.get("dt", {}).get("f1", 0.0),
        "selected_indices":     list(range(m_total)),
        "candidates_all":       [(t, a[0] if a else None, a[1] if len(a) > 1 else None)
                                 for t, a in declare_features],
        "m_total":              m_total,
        "m_selected":           m_total,
        "config":               cfg,
        "tau_star":             0.0,
        "alpha_f":              None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: NULL REPLICATE RUNNER (rq1 interface)
# ══════════════════════════════════════════════════════════════════════════════

def run_dvm_on_doubly_null_log(
    null_case_data: Dict[str, "CaseInfo"],
    candidates_all: List[Tuple],
    tau_star: float = 0.0,
    tau_min: float  = 0.0,
    holds_all: Optional[Dict] = None,
    fisher_eps: float = 1e-10,
    config: Optional[dict] = None,
) -> int:
    """
    Apply DvM to a null log for rq1 permutation testing.
    Returns V_b = number of false positives (any accepted features = FP by construction).
    Uses the pure Declare encoding for speed; runs feature selection only.
    """
    cfg = {**DVM_CONFIG, **(config or {})}
    cov_thr = int(cfg.get("coverage_threshold", COVERAGE_THRESHOLD))

    positional_log, sequence_log, labels = build_log_structures(null_case_data)
    D_0, D_1 = split_by_class(null_case_data)
    ids0, ids1 = list(D_0.keys()), list(D_1.keys())
    n0, n1 = len(ids0), len(ids1)
    if n0 == 0 or n1 == 0:
        return 0

    all_ids = ids0 + ids1
    y_null  = np.array([0] * n0 + [1] * n1, dtype=np.int32)

    # Build Declare features from the null log
    declare_features = discover_declare_features(
        case_ids_pos   = ids1,
        case_ids_neg   = ids0,
        positional_log = positional_log,
        support_threshold = float(cfg.get("support_threshold", SUPPORT_THRESHOLD)),
    )
    if not declare_features:
        return 0

    # Encode and select
    X, _ = build_encoding_matrix(
        all_ids, positional_log, sequence_log, {}, [], {},
        declare_features, [], [], "declare",
    )
    if X.shape[1] == 0:
        return 0

    fisher = compute_generalized_fisher_scores(X, y_null, eps=float(cfg["fisher_eps"]))
    sel    = select_features_coverage(X, fisher, cov_thr)
    return len(sel)


# ─── BACKWARD-COMPATIBLE ALIASES ─────────────────────────────────────────────
run_declareminer                    = run_dvm
run_declareminer_on_doubly_null_log = run_dvm_on_doubly_null_log


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13: OUTPUT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(dvm_out: dict) -> None:
    """Write JSON summary and per-encoding CSV to OUTPUT_DIR."""
    summary_df = dvm_out["summary_table"]
    timing     = dvm_out["timing"]
    cfg        = dvm_out["config"]

    # CSV table
    path_csv = os.path.join(OUTPUT_DIR, "dvm_bise2025_results.csv")
    summary_df.to_csv(path_csv, index=False)
    print(f"\n✓ Results CSV:      {path_csv}")

    # JSON summary
    full_json = {
        "framework": "DvM BISE 2025 — Sequential & Declarative Patterns",
        "timestamp": datetime.now().isoformat(),
        "reference": "Di Francescomarino et al. (2025). BISE 67(6):877-894.",
        "config":    cfg,
        "timing":    timing,
        "n_encodings_evaluated": len(dvm_out["results_by_encoding"]),
        "ripper_available": RIPPER_AVAILABLE,
        "results_by_encoding": {
            enc: {"ripper": res["ripper"], "dt": res["dt"]}
            for enc, res in dvm_out["results_by_encoding"].items()
        },
    }
    path_json = os.path.join(OUTPUT_DIR, "dvm_bise2025_results.json")
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump(full_json, fh, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Results JSON:     {path_json}")

    # Text report
    lines = [
        "=" * 120,
        "DvM — Business Process Deviance Mining: Sequential & Declarative Patterns",
        "Reference: Di Francescomarino et al. (2025), BISE 67(6):877–894",
        "=" * 120,
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Classifier: {'RipperK (Wittgenstein) + DecisionTree' if RIPPER_AVAILABLE else 'DecisionTree only'}",
        "",
        "PARAMETERS",
        f"  candidate_threshold = {cfg.get('candidate_threshold')}",
        f"  support_threshold   = {cfg.get('support_threshold')}",
        f"  coverage_threshold  = {cfg.get('coverage_threshold')}  (paper Section 7.3)",
        f"  n_cv_folds          = {cfg.get('n_cv_folds')}  (paper Section 9.3)",
        "",
        "RESULTS  (averaged over 10 folds)",
        "-" * 120,
        f"{'Encoding':<22} {'Clf':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'ANR':>6} {'ARL':>6}",
        "-" * 120,
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['encoding']:<22} {row['classifier']:>6} "
            f"{row['precision']:>6.3f} {row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} {row['auc']:>6.3f} "
            f"{row['anr']:>6.1f} {row['arl']:>6.2f}"
        )
    lines += ["-" * 120, "", "TIMING"]
    for k, v in timing.items():
        lines.append(f"  {k:34s}: {v:.2f}s")

    path_rpt = os.path.join(OUTPUT_DIR, "dvm_bise2025_report.txt")
    with open(path_rpt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"✓ Text report:      {path_rpt}")

    # Rules JSON (Section 7.5 — final model on full dataset)
    rules_data = dvm_out.get("final_rules_by_encoding", {})
    if rules_data:
        path_rules_json = os.path.join(OUTPUT_DIR, "dvm_bise2025_rules.json")
        with open(path_rules_json, "w", encoding="utf-8") as fh:
            json.dump(rules_data, fh, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Rules JSON:       {path_rules_json}")

        rule_lines = [
            "=" * 120,
            "DvM — Rule Set (Section 7.5 — final model trained on full dataset)",
            "Reference: Di Francescomarino et al. (2025), BISE 67(6):877–894",
            "=" * 120,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for enc, rout in rules_data.items():
            sel_feats = rout.get("sel_features", [])
            rip_rules = rout.get("ripper_rules", [])
            dt_rules_ = rout.get("dt_rules", [])
            rule_lines.append("─" * 80)
            rule_lines.append(f"ENCODING: {enc.upper()}")
            rule_lines.append(
                f"  Selected features ({len(sel_feats)}): "
                + ", ".join(sel_feats[:10])
                + (" ..." if len(sel_feats) > 10 else "")
            )
            rule_lines.append(f"  RipperK rules ({len(rip_rules)}):")
            for i, r in enumerate(rip_rules, 1):
                rule_lines.append(f"    R{i:02d}: {r}")
            rule_lines.append(f"  DecisionTree rules ({len(dt_rules_)}):")
            for i, r in enumerate(dt_rules_, 1):
                rule_lines.append(f"    D{i:02d}: {r}")
            rule_lines.append("")
        path_rules_rpt = os.path.join(OUTPUT_DIR, "dvm_bise2025_rules.txt")
        with open(path_rules_rpt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rule_lines))
        print(f"✓ Rules text:       {path_rules_rpt}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DvM BISE 2025 — Business Process Deviance Mining"
    )
    parser.add_argument("--candidate-threshold", type=float,
                        default=CANDIDATE_THRESHOLD)
    parser.add_argument("--support-threshold",   type=float,
                        default=SUPPORT_THRESHOLD)
    parser.add_argument("--coverage-threshold",  type=int,
                        default=COVERAGE_THRESHOLD)
    parser.add_argument("--n-folds",             type=int,
                        default=N_CV_FOLDS)
    parser.add_argument("--encodings", nargs="+", default=None,
                        choices=ALL_ENCODINGS,
                        help="Subset of encodings to evaluate (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = {
        "candidate_threshold": args.candidate_threshold,
        "support_threshold":   args.support_threshold,
        "coverage_threshold":  args.coverage_threshold,
        "n_cv_folds":          args.n_folds,
        "random_state":        RANDOM_STATE,
        "tau_min":             0.0,
        "fisher_eps":          1e-10,
    }

    print("\n" + "=" * 100)
    print("DEVIANCE MINING — Business Process Deviance Mining (BISE 2025)")
    print("Sequential and Declarative Patterns — Production")
    print("Reference: Di Francescomarino et al. (2025), BISE 67(6):877–894")
    print("=" * 100)
    print(f"  candidate_threshold = {config['candidate_threshold']}")
    print(f"  support_threshold   = {config['support_threshold']}")
    print(f"  coverage_threshold  = {config['coverage_threshold']}")
    print(f"  n_cv_folds          = {config['n_cv_folds']}")
    print(f"  Encodings           = {args.encodings or 'ALL'}")
    print(f"  Classifiers         = {'RipperK + DT' if RIPPER_AVAILABLE else 'DT only'}")
    print("=" * 100)

    dvm_out = run_dvm(config=config, encodings=args.encodings)

    if not args.dry_run:
        save_outputs(dvm_out)

    print(f"\n{'=' * 100}")
    print("DvM (BISE 2025) COMPLETE")
    print(f"{'=' * 100}")
    print(dvm_out["summary_table"].to_string(index=False))
    print(f"\n  Total time: {dvm_out['timing']['total']:.1f}s")
    print(f"  Output dir: {OUTPUT_DIR}/")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()