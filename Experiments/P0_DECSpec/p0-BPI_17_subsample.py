"""
PHASE 0: CLASS-CONDITIONAL DECLARE SPECIFICATION DISCOVERY
===========================================================
Exhaustive pattern enumeration with MINERful-style activation/target measures,
subsumption hierarchy pruning, and negation pair pruning — applied independently
to two class-conditional sublogs derived from the outcome label:

    L+  =  { σ ∈ L  |  outcome(σ) = Not-Accepted (Deviant) }
    L−  =  { σ ∈ L  |  outcome(σ) = Accepted (Normal) }

Each sublog yields its own DECLARE specification over its class-specific activity
alphabet (activities absent from a class are excluded to avoid vacuous constraints).
The full discovery pipeline (candidate generation → MINERful measure computation →
threshold filtering → subsumption pruning → language-equivalence pruning →
negation pruning → transitivity pruning → FSA entailment pruning) is executed
independently for L+ and L−, producing:
    • phase0_declare_specification_Lpos.decl   — Not-Accepted (Deviant) behaviour
    • phase0_declare_specification_Lneg.decl   — Accepted (Normal) behaviour
    • phase0_declare_specification_CC.json     — enriched output with both
      specifications, class-level measure tables, and set-algebra summary
      (shared constraints, L+-exclusive, L−-exclusive).

Constraint semantics follow Di Ciccio & Montali (2022), Table 2.
Measure definitions follow Iacometta & Di Ciccio (2025), Table 2.

Constraint repertoire:
  Unary:   Init, End
  Binary+: RespondedExistence, Response, AlternateResponse, ChainResponse,
           Precedence, AlternatePrecedence, ChainPrecedence,
           Succession, AlternateSuccession, ChainSuccession
  Binary-: NotRespondedExistence, NotResponse, NotChainResponse,
           NotPrecedence, NotChainPrecedence,
           NotSuccession, NotChainSuccession

Author: Ahmed Nour Abdesselam
Institution: Free University of Bozen-Bolzano
Date: February 2026
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, FrozenSet, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
import itertools
import os
import time
from datetime import datetime
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "../Experiments data/CSV/BPI_Challenge_17_subsample.csv"
OUTPUT_DIR = "../Experiments data/Experiments/Results/DECspec_BPI17_subsample"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    # MINERful filtering thresholds (Definition 14 & 15 from Di Ciccio & Montali)
    'min_event_confidence': 0.0,
    'min_event_support': 0.0,
    'min_trace_confidence': 0.0,
    'min_trace_support': 0.0,
    # At least one measure pair must exceed these to be retained
    'min_event_confidence_hard': 0.0,
    'min_trace_confidence_hard': 0.0,
    # Activity frequency filter
    'min_activity_frequency': 1,
    # Random seed
    'random_state': 42,
    # Tier 3: full FSA-based language-inclusion entailment check (Di Ciccio & Montali 2022, §4.2)
    # Computationally complete but exponential in |K|; disabled if constraint count exceeds limit.
    'enable_fsa_entailment': False,
    'fsa_entailment_max_constraints': 500,     # per-NEIGHBOURHOOD limit (local pair scope)
}

# =============================================================================
# CONSTRAINT TYPES
# =============================================================================

UNARY_CONSTRAINTS = ['Init', 'End']

BINARY_POSITIVE_CONSTRAINTS = [
    # 'RespondedExistence',
    'Response', 'AlternateResponse', 'ChainResponse',
    # 'Precedence', 'AlternatePrecedence', 'ChainPrecedence',
    'Succession', 'AlternateSuccession', 'ChainSuccession',
]

BINARY_NEGATIVE_CONSTRAINTS = [
    # 'NotRespondedExistence',
    'NotResponse',
    # 'NotChainResponse',
    # 'NotPrecedence',
    # 'NotChainPrecedence',
    # 'NotSuccession',
    'NotChainSuccession',
]

ALL_CONSTRAINT_TYPES = UNARY_CONSTRAINTS + BINARY_POSITIVE_CONSTRAINTS + BINARY_NEGATIVE_CONSTRAINTS

# =============================================================================
# SUBSUMPTION HIERARCHY (Figure 8, Di Ciccio & Montali 2022)
# =============================================================================
# If A subsumes B (A ⊑ B in the paper's notation: A is MORE restrictive),
# then every trace satisfying A also satisfies B.
# Stored as: subsuming -> subsumed (i.e., A -> B means A subsumes B)
# Equivalently: if A has measures >= B, then B is redundant (keep A).

SUBSUMPTION_CHAINS = {
    # Forward-looking chain: ChainResponse ⊑ AlternateResponse ⊑ Response ⊑ RespondedExistence
    'ChainResponse': ['AlternateResponse', 'Response'],  # 'RespondedExistence' excluded
    'AlternateResponse': ['Response'],  # 'RespondedExistence' excluded
    # 'Response': ['RespondedExistence'],  # excluded

    # Backward-looking chain: ChainPrecedence ⊑ AlternatePrecedence ⊑ Precedence ⊑ RespondedExistence(y,x)
    'ChainPrecedence': ['AlternatePrecedence', 'Precedence'],
    'AlternatePrecedence': ['Precedence'],

    # Negative forward: NotRespondedExistence ⊑ NotResponse ⊑ NotChainResponse
    # 'NotRespondedExistence': ['NotResponse', 'NotChainResponse'],  # excluded
    'NotResponse': ['NotChainResponse'],

    # Negative backward: NotRespondedExistence ⊑ NotPrecedence ⊑ NotChainPrecedence
    # Note: NotRespondedExistence already listed above for NotResponse chain
    # We add the precedence chain here
    # 'NotPrecedence': ['NotChainPrecedence'],  # NotPrecedence excluded

    # ── Succession family (Figure 8, Di Ciccio & Montali 2022) ───────────────
    # ChainSuccession ⊑ AlternateSuccession ⊑ Succession
    # Each also ⊑ its respective Response and Precedence components
    'ChainSuccession': [
        'AlternateSuccession',           # ChainSucc ⊑ AltSucc (tighter alternation)
        'Succession',                    # ChainSucc ⊑ Succ (transitive)
        'ChainResponse',                 # ChainSucc ⊑ ChainResp (forward component)
        'ChainPrecedence',               # ChainSucc ⊑ ChainPrec (backward component)
        'AlternateResponse',             # transitive via ChainResp ⊑ AltResp
        'AlternatePrecedence',           # transitive via ChainPrec ⊑ AltPrec
        'Response',                      # transitive
        'Precedence',                    # transitive
    ],
    'AlternateSuccession': [
        'Succession',                    # AltSucc ⊑ Succ
        'AlternateResponse',             # AltSucc ⊑ AltResp (forward component)
        'AlternatePrecedence',           # AltSucc ⊑ AltPrec (backward component)
        'Response',                      # transitive
        'Precedence',                    # transitive
    ],
    'Succession': [
        'Response',                      # Succ ⊑ Resp (forward component)
        'Precedence',                    # Succ ⊑ Prec (backward component)
    ],

    # ── Negative Succession family ────────────────────────────────────────────
    # NotSuccession ⊑ NotResponse (x-branch identical to NotResponse)
    # NotSuccession ⊑ NotPrecedence (y-branch identical to NotPrecedence)
    # NotSuccession ⊑ NotChainSuccession (forbidding all occurrence is stronger than forbidding adjacency)
    # NotChainSuccession ⊑ NotChainResponse (x-branch)
    # NotChainSuccession ⊑ NotChainPrecedence (y-branch)
    'NotSuccession': [
        'NotChainSuccession',            # NotSucc ⊑ NotChainSucc (no y ever → not immediately)
        'NotResponse',                   # NotSucc ⊑ NotResp (x-branch is identical)
        'NotChainResponse',              # transitive via NotResp ⊑ NotChainResp
    ],
    'NotChainSuccession': [
        'NotChainResponse',              # NotChainSucc ⊑ NotChainResp (x-branch)
        'NotChainPrecedence',            # NotChainSucc ⊑ NotChainPrec (y-branch)
    ],
}
# NotRespondedExistence also subsumes NotPrecedence — excluded since NotRespondedExistence is disabled
# SUBSUMPTION_CHAINS.setdefault('NotRespondedExistence', [])
# if 'NotPrecedence' not in SUBSUMPTION_CHAINS['NotRespondedExistence']:
#     SUBSUMPTION_CHAINS['NotRespondedExistence'].append('NotPrecedence')
# if 'NotChainPrecedence' not in SUBSUMPTION_CHAINS['NotRespondedExistence']:
#     SUBSUMPTION_CHAINS['NotRespondedExistence'].append('NotChainPrecedence')

# =============================================================================
# PAIRWISE CROSS-FAMILY ENTAILMENT CATALOGUE  (Tier 2.5)
# Offline FSA language-inclusion: L(A_T1 ⊗ A_T2) ⊆ L(A_T3)
# Verified over symbolic alphabet {A=param_a, B=param_b, O=other}
# Di Ciccio & Montali (2022), §4.2, Definitions 6–7
# =============================================================================
PAIRWISE_ENTAILMENTS_SAME: Dict[FrozenSet[str], List[str]] = {
    # ── Category A: non-trivial cross-family ──────────────────────────────
    frozenset({'AlternateResponse',   'ChainPrecedence'  }): ['ChainResponse'],
    frozenset({'AlternatePrecedence', 'ChainResponse'    }): ['ChainPrecedence'],
    # ── Category B: vacuous-collapse (rarely fires after negation pruning) ─
    frozenset({'Response',            'NotResponse'      }): ['AlternateResponse', 'ChainResponse'],
    frozenset({'Precedence',          'NotResponse'      }): ['AlternatePrecedence', 'ChainPrecedence'],
    frozenset({'AlternateResponse',   'NotResponse'      }): ['ChainResponse'],
    frozenset({'AlternatePrecedence', 'NotResponse'      }): ['ChainPrecedence'],
    frozenset({'ChainResponse',       'NotChainResponse' }): ['NotResponse'],

    # ── Succession assembly: T1 ∧ T2 ⊨ T3 (T3 redundant given T1,T2) ────────
    # L(Response) ∩ L(Precedence) = L(Succession) → Succession redundant
    frozenset({'Response',            'Precedence'           }): ['Succession'],
    frozenset({'AlternateResponse',   'AlternatePrecedence'  }): ['AlternateSuccession'],
    frozenset({'ChainResponse',       'ChainPrecedence'      }): ['ChainSuccession'],

    # ── Negative Succession assembly ─────────────────────────────────────────
    # L(NotResponse) ∩ L(NotPrecedence) = L(NotSuccession) → NotSuccession redundant
    frozenset({'NotResponse',         'NotPrecedence'        }): ['NotSuccession'],
    frozenset({'NotChainResponse',    'NotChainPrecedence'   }): ['NotChainSuccession'],

    # ── Vacuous-collapse (Category B): Succession ∧ Not* = contradiction ─────
    # Succession(x,y) ∧ NotSuccession(x,y): x and y cannot co-occur in any valid
    # trace → all stronger constraints vacuously satisfied
    frozenset({'Succession',          'NotSuccession'        }): ['AlternateSuccession', 'ChainSuccession'],
    frozenset({'Succession',          'NotResponse'          }): ['AlternateSuccession', 'ChainSuccession'],
    frozenset({'AlternateSuccession', 'NotResponse'          }): ['ChainSuccession'],

    # ChainSuccession(x,y) ∧ NotChainSuccession(x,y):
    #   x-direction: x→⊙y ∧ x→¬⊙y = x never occurs
    #   y-direction: y→⊖x ∧ y→¬⊖x = y never occurs
    #   → neither x nor y appears → NotSuccession vacuously satisfied
    frozenset({'ChainSuccession',     'NotChainSuccession'   }): ['NotSuccession'],
}
MUTUAL_COLLAPSE_FAMILIES: Dict[str, List[str]] = {
    'ResponseFamily':   ['Response', 'AlternateResponse', 'ChainResponse'],
    'PrecedenceFamily': ['Precedence', 'AlternatePrecedence', 'ChainPrecedence'],
}

# =============================================================================
# CROSS-PARAMETER SUBSUMPTIONS (Figure 8, Di Ciccio & Montali 2022)
# =============================================================================
# These are subsumptions where the subsuming constraint on (a, b) logically
# implies the subsumed constraint on (b, a) — parameter roles are SWAPPED.
#
# Formal basis (Table 2, Di Ciccio & Montali 2022):
#   Precedence(a,b):          □(b → ◆a)         — every b has an a somewhere before it
#   RespondedExistence(b,a):  ◇b → ◇a           — if b occurs, a occurs somewhere
#
#   Since □(b → ◆a) implies ◇b → ◇a, we have:
#       Precedence(a,b)          ⊑  RespondedExistence(b,a)   [params swapped]
#       AlternatePrecedence(a,b) ⊑  RespondedExistence(b,a)   [transitive via Precedence]
#       ChainPrecedence(a,b)     ⊑  RespondedExistence(b,a)   [transitive via Precedence]
#
# Stored as: (subsuming_type_on_ab, subsumed_type_on_ba)
# Ordered from most to least restrictive to handle cascades correctly.
# =============================================================================

CROSS_PARAM_SUBSUMPTIONS: List[Tuple[str, str]] = [
    # ('ChainPrecedence',     'RespondedExistence'),  # excluded
    # ('AlternatePrecedence', 'RespondedExistence'),  # excluded
    # ('Precedence',          'RespondedExistence'),  # excluded
]

# =============================================================================
# NEGATION PAIRS (Section 4.2, Di Ciccio & Montali 2022)
# =============================================================================
# Pairs of constraints that are negated versions of each other.
# They share the same activation but have incompatible targets.
# Both should not be in the specification simultaneously.

NEGATION_PAIRS = [
    # ('RespondedExistence', 'NotRespondedExistence'),  # excluded
    # --- Direct negation pairs (same family) ---
    ('Response',           'NotResponse'),
    ('Precedence',         'NotPrecedence'),
    ('ChainResponse',      'NotChainResponse'),
    ('ChainPrecedence',    'NotChainPrecedence'),
    # --- Subsumption-derived pairs (same-family, existing) ---
    ('AlternateResponse',  'NotResponse'),      # AltResp ⊑ Resp; (¬a U b) ∧ □¬b = ⊥
    ('ChainResponse',      'NotResponse'),      # ChainResp ⊑ Resp; ⊙b ∧ □¬b = ⊥
    ('AlternatePrecedence','NotPrecedence'),    # AltPrec ⊑ Prec; (¬b S a) ∧ □⁻¬a = ⊥
    ('ChainPrecedence',    'NotPrecedence'),    # ChainPrec ⊑ Prec; ⊖a ∧ □⁻¬a = ⊥
    # --- Cross-family pairs derived from FSA language equivalences E1/E2 ---
    # Via E1: NotResponse(a,b) ≡ NotPrecedence(a,b) (identical FSAs)
    ('Response',           'NotPrecedence'),    # ≡ (Response, NotResponse) via E1
    ('AlternateResponse',  'NotPrecedence'),    # AltResp ⊑ Resp; NotPrec ≡ NotResp
    ('ChainResponse',      'NotPrecedence'),    # ChainResp ⊑ Resp; NotPrec ≡ NotResp
    ('Precedence',         'NotResponse'),      # ≡ (Prec, NotPrec) via E1
    ('AlternatePrecedence','NotResponse'),      # AltPrec ⊑ Prec; NotResp ≡ NotPrec
    ('ChainPrecedence',    'NotResponse'),      # ChainPrec ⊑ Prec; NotResp ≡ NotPrec
    # Via E2: NotChainResponse(a,b) ≡ NotChainPrecedence(a,b) (identical FSAs)
    ('ChainResponse',      'NotChainPrecedence'),  # ≡ (ChainResp, NotChainResp) via E2
    ('ChainPrecedence',    'NotChainResponse'),    # ≡ (ChainPrec, NotChainPrec) via E2

    # ── Direct Succession negation pairs ─────────────────────────────────────
    ('Succession',          'NotSuccession'),        # direct: Succ ∧ ¬Succ = ⊥
    ('ChainSuccession',     'NotChainSuccession'),   # direct: ChainSucc ∧ ¬ChainSucc = ⊥

    # ── Succession ⊑ Response/Precedence → inherits their negation pairs ─────
    ('Succession',          'NotResponse'),          # Succ ⊑ Resp; (Resp,NotResp) is a pair
    ('Succession',          'NotPrecedence'),        # Succ ⊑ Prec; (Prec,NotPrec) via E1
    ('AlternateSuccession', 'NotResponse'),          # AltSucc ⊑ AltResp ⊑ Resp
    ('AlternateSuccession', 'NotPrecedence'),        # AltSucc ⊑ AltPrec ⊑ Prec; via E1
    ('AlternateSuccession', 'NotSuccession'),        # AltSucc ⊑ Succ; (Succ,NotSucc) is a pair

    # ── ChainSuccession ⊑ ChainResponse/ChainPrecedence ──────────────────────
    ('ChainSuccession',     'NotChainResponse'),     # ChainSucc ⊑ ChainResp; (ChainResp,NotChainResp)
    ('ChainSuccession',     'NotChainPrecedence'),   # ChainSucc ⊑ ChainPrec; (ChainPrec,NotChainPrec) via E2
    ('ChainSuccession',     'NotResponse'),          # transitive: ChainSucc ⊑ Resp
    ('ChainSuccession',     'NotPrecedence'),        # transitive: ChainSucc ⊑ Prec; via E1
    ('ChainSuccession',     'NotSuccession'),        # ChainSucc ⊑ Succ; (Succ,NotSucc) is a pair
]

# =============================================================================
# LANGUAGE EQUIVALENCES (Tier 1 — FSA analysis, Di Ciccio & Montali 2022)
# =============================================================================
# Pairs of constraint templates whose local FSAs are isomorphic — they accept
# exactly the same language over any alphabet.  One is always unconditionally
# redundant when both appear in the same specification on the same (a, b).
#
# E1: NotResponse(a,b) ≡ NotPrecedence(a,b)
#     Both encode: "a never precedes b in any trace"
#     FSA violation condition: a seen, then b seen (identical transition function)
#
# E2: NotChainResponse(a,b) ≡ NotChainPrecedence(a,b)
#     Both encode: "a is never immediately followed by b in any trace"
#     FSA violation condition: b seen immediately after a (identical)

LANGUAGE_EQUIVALENCES: List[Tuple[str, str]] = [
    ('NotResponse',      'NotPrecedence'),       # E1 — NotResponse ≡ NotPrecedence (NotPrecedence disabled but equivalence defined)
    ('NotChainResponse', 'NotChainPrecedence'),  # E2 — NotChainResponse ≡ NotChainPrecedence

    # E3: NotSuccession(x,y) ≡ NotResponse(x,y)
    #   Proof: x-branch of NotSuccession IS NotResponse; y-branch IS NotPrecedence ≡ NotResponse (E1).
    #   The dual activation does not change the accepted language — only measure coverage differs.
    ('NotSuccession',    'NotResponse'),         # E3a
    ('NotSuccession',    'NotPrecedence'),       # E3b (via E1)

    # E4: NotChainSuccession(x,y) ≡ NotChainResponse(x,y)
    #   Same argument: x-branch IS NotChainResponse; y-branch IS NotChainPrecedence ≡ NotChainResponse (E2).
    ('NotChainSuccession', 'NotChainResponse'),  # E4a
    ('NotChainSuccession', 'NotChainPrecedence'),# E4b (via E2)
]

# =============================================================================
# TRANSITIVITY FAMILIES (Tier 2 — LTLf transitivity entailments)
# =============================================================================
# Response transitivity:
#   T1(a,b) ∧ T2(b,c)  ⊨  Response(a,c)
#   where T1, T2 ∈ RESPONSE_TRANSITIVITY_FAMILY
#   Proof: □(a→◇b) ∧ □(b→◇c)  ⊨  □(a→◇c)
#
# Precedence transitivity:
#   T1(a,b) ∧ T2(b,c)  ⊨  Precedence(a,c)
#   where T1, T2 ∈ PRECEDENCE_TRANSITIVITY_FAMILY
#   Proof: □(b→◆a) ∧ □(c→◆b)  ⊨  □(c→◆a)
#
# NOTE: AlternateResponse/ChainResponse transitivity does NOT lift to
# AlternateResponse(a,c) — counterexample: ⟨a,b,a,c,b,c⟩ satisfies
# AlternateResponse(a,b) ∧ AlternateResponse(b,c) but NOT AlternateResponse(a,c).
# The entailment reaches only Response(a,c).

RESPONSE_TRANSITIVITY_FAMILY: List[str] = [
    'Response', 'AlternateResponse', 'ChainResponse',
    'Succession', 'AlternateSuccession', 'ChainSuccession',  # Succ ⊑ Resp: □(a→◇b) ∧ □(b→◇c) ⊨ □(a→◇c) ✓
]

PRECEDENCE_TRANSITIVITY_FAMILY: List[str] = [
    'Precedence', 'AlternatePrecedence', 'ChainPrecedence',
    'Succession', 'AlternateSuccession', 'ChainSuccession',  # Succ ⊑ Prec: □(b→◆a) ∧ □(c→◆b) ⊨ □(c→◆a) ✓
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CaseTrace:
    """A single case with its PC activity trace."""
    case_id: str
    outcome: int          # 1=Not-Accepted (Deviant), 0=Accepted (Normal)
    pc_activities: List[str]
    activity_positions: Dict[str, List[int]]  # activity -> list of positions


@dataclass
class ConstraintMeasures:
    """MINERful interestingness measures for a constraint (Table 2, Iacometta & Di Ciccio 2025)."""
    # Event-based (Definition 15, Di Ciccio & Montali 2022)
    event_confidence: float
    event_coverage: float
    event_support: float
    # Trace-based (Definition 14, Di Ciccio & Montali 2022)
    trace_confidence: float
    trace_coverage: float
    trace_support: float
    # Raw counts for debugging
    n_events_activated_and_satisfied: int
    n_events_activated: int
    n_events_total: int
    n_traces_activated_and_satisfied: int
    n_traces_activated: int
    n_traces_total: int


@dataclass
class DiscoveredConstraint:
    """A discovered DECLARE constraint with its measures."""
    constraint_type: str
    param_a: str
    param_b: Optional[str]  # None for unary
    measures: ConstraintMeasures
    # For output
    pattern_id: str


# =============================================================================
# DATA LOADING
# =============================================================================

def load_event_log(
    filepath: str,
) -> Tuple[List[CaseTrace], List[CaseTrace], List[CaseTrace], Dict]:
    """Load BPI Challenge 2017 event log, extract full activity traces, and
    partition into class sublogs following Teinemaa et al. (TKDE 2019).

    BPI 2017 column mapping
    -----------------------
        case:concept:name    → case identifier
        concept:name         → activity name (A_ / O_ / W_ namespaces retained)
        time:timestamp       → event timestamp
        lifecycle:transition → event lifecycle state (filter to 'complete' only)

    Activity namespaces:
        A_  — Application lifecycle events (case-level decisions)
        O_  — Offer lifecycle events       (offer-level decisions)
        W_  — Work-item events             (resource/task execution)
    All three namespaces are retained in the trace; cross-namespace DECLARE
    constraints (e.g. AlternateResponse[A_Submitted, O_Created]) capture
    genuine process ordering that would be lost by namespace filtering.

    Lifecycle filter: the BPI 2017 log records both 'start' and 'complete'
    events per activity.  Only 'complete' rows are retained before building
    traces; retaining starts would duplicate every activity, inflating trace
    lengths and distorting all DECLARE pattern prevalence estimates.

    Outcome labelling (Teinemaa et al. TKDE 2019 — bpic2017_accepted sub-log)
    --------------------------------------------------------------------------
    The label is derived from the LAST O_ (offer) event in the (complete) trace:
        terminal O_ == 'O_Accepted'                → outcome = 0 (Normal)   → L−
        terminal O_ == 'O_Refused' or 'O_Cancelled'→ outcome = 1 (Deviant)  → L+
    Cases with no O_ events after lifecycle filtering are skipped (cannot label).

    Outcome-signal removal: all three terminal offer outcome activities
    {'O_Accepted', 'O_Refused', 'O_Cancelled'} are stripped from the trace
    BEFORE building the DECLARE alphabet.  These are exclusively terminal states;
    retaining them would produce trivially class-deterministic constraints such as
    End[O_Accepted] holding in 100 % of L− and 0 % of L+.

    Returns
    -------
    cases     : full log L  (all valid labelled cases)
    cases_pos : L+  =  { σ ∈ L  |  outcome(σ) = Not-Accepted (Deviant) }
    cases_neg : L−  =  { σ ∈ L  |  outcome(σ) = Accepted (Normal) }
    stats     : logging/audit dictionary
    """
    # Terminal offer outcome activities — used for labelling, then stripped.
    TERMINAL_OFFER_OUTCOMES: Set = {'O_Accepted', 'O_Refused', 'O_Cancelled'}

    print("=" * 100)
    print("PHASE 0 (CLASS-CONDITIONAL): DECLARE SPECIFICATION DISCOVERY")
    print("MINERful-style Activation/Target Measures — L+ and L− independently")
    print("=" * 100)
    print(f"\nLoading event log from: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna(subset=['case:concept:name'])
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True, errors='coerce')
    print(f"  Total raw events: {len(df):,}")
    print(f"  Total cases:      {df['case:concept:name'].nunique():,}")

    # Lifecycle filter: keep only 'complete' events
    if 'lifecycle:transition' in df.columns:
        before = len(df)
        df = df[df['lifecycle:transition'].str.lower() == 'complete'].copy()
        print(f"  Lifecycle filter ('complete' only): {before:,} → {len(df):,} events "
              f"({before - len(df):,} non-complete rows dropped)")
    else:
        print("  ⚠  'lifecycle:transition' column not found — no lifecycle filter applied")

    cases: List[CaseTrace] = []
    stats = {
        'total_cases': 0,
        'skipped_empty': 0,
        'skipped_no_offer': 0,
        'class_0': 0,   # L−  (Accepted / Normal)
        'class_1': 0,   # L+  (Not-Accepted / Deviant)
    }

    for case_id, group in df.groupby('case:concept:name'):
        # All complete events sorted by timestamp
        case_events = group.sort_values('time:timestamp')
        all_activities = case_events['concept:name'].tolist()

        if len(all_activities) == 0:
            stats['skipped_empty'] += 1
            continue

        # Labelling: terminal O_ event determines outcome
        offer_activities = [a for a in all_activities if a.startswith('O_')]
        if not offer_activities:
            stats['skipped_no_offer'] += 1
            continue
        terminal_offer = offer_activities[-1]
        outcome = 0 if terminal_offer == 'O_Accepted' else 1

        # Strip all three terminal offer outcome activities from the trace.
        # They are exclusively terminal states that directly encode the label.
        activities = [a for a in all_activities if a not in TERMINAL_OFFER_OUTCOMES]

        if len(activities) == 0:
            stats['skipped_empty'] += 1
            continue

        # Build position index
        activity_positions: Dict[str, List[int]] = {}
        for i, act in enumerate(activities):
            activity_positions.setdefault(act, []).append(i)

        cases.append(CaseTrace(
            case_id=str(case_id),
            outcome=outcome,
            pc_activities=activities,
            activity_positions=activity_positions,
        ))

        if outcome == 1:
            stats['class_1'] += 1
        else:
            stats['class_0'] += 1

    stats['total_cases'] = len(cases)

    # --- Class-conditional partition ---
    cases_pos = [c for c in cases if c.outcome == 1]   # L+  (Not-Accepted / Deviant)
    cases_neg = [c for c in cases if c.outcome == 0]   # L−  (Accepted / Normal)

    print(f"\n  Processed cases: {stats['total_cases']:,}")
    print(f"  L+ | Class 1 (Not-Accepted / Deviant): {stats['class_1']:,} traces")
    print(f"  L− | Class 0 (Accepted / Normal):      {stats['class_0']:,} traces")
    print(f"  Skipped (no O_ events):    {stats['skipped_no_offer']:,}")
    print(f"  Skipped (empty):           {stats['skipped_empty']:,}")
    print(f"\n  Sublog partitioning:")
    print(f"    L  = {len(cases):,} traces")
    print(f"    L+ = {len(cases_pos):,} traces  (outcome = Not-Accepted / Deviant)")
    print(f"    L− = {len(cases_neg):,} traces  (outcome = Accepted / Normal)")

    return cases, cases_pos, cases_neg, stats


def get_frequent_activities(
    cases: List[CaseTrace],
    min_freq: int,
    class_label: str = "",
) -> List[str]:
    """Get activities meeting minimum frequency threshold within a (sub-)log.

    Computing the alphabet per class-conditional sublog is scientifically
    necessary: an activity that never appears in L+ generates only vacuously-
    satisfied constraints for L+ and therefore carries no discriminative
    information.  Using a global alphabet would inflate the candidate space
    and introduce spurious pruning artefacts.
    """
    tag = f" [{class_label}]" if class_label else ""
    counts: Counter = Counter()
    for case in cases:
        for act in case.pc_activities:
            counts[act] += 1

    frequent = sorted([a for a, c in counts.items() if c >= min_freq])
    print(f"\n  Activity alphabet{tag}:")
    print(f"    Unique PC activities: {len(counts)}")
    print(f"    Frequent (>={min_freq} occurrences): {len(frequent)}")
    return frequent


# =============================================================================
# ACTIVATION/TARGET CONSTRAINT CHECKERS
# =============================================================================
# Each function returns, for a given trace and event position:
#   (activated: bool, satisfied: bool)
#
# For TRACE-BASED evaluation:
#   activated = does ANY position activate?
#   satisfied = are ALL activations satisfied?
#
# Semantics strictly follow Di Ciccio & Montali (2022) Table 2.
# =============================================================================

# ---------------------------------------------------------------------------
# UNARY CONSTRAINTS
# ---------------------------------------------------------------------------

def check_Init_event(trace: List[str], pos: int, x: str) -> Tuple[bool, bool]:
    """
    Init(x): □(start → x)
    Activation: start (position 0 fires exactly once)
    Target: x is the first activity

    Event-based: only position 0 is an activation event.
    """
    if pos == 0:
        return (True, trace[0] == x)
    return (False, False)


def check_End_event(trace: List[str], pos: int, x: str) -> Tuple[bool, bool]:
    """
    End(x): □(end → x)
    Activation: end (last position fires exactly once)
    Target: x is the last activity

    Event-based: only the last position is an activation event.
    """
    if pos == len(trace) - 1:
        return (True, trace[-1] == x)
    return (False, False)


# ---------------------------------------------------------------------------
# POSITIVE RELATION CONSTRAINTS (Forward-looking, activated by x)
# ---------------------------------------------------------------------------

def check_RespondedExistence_event(trace: List[str], pos: int,
                                   x: str, y: str,
                                   y_positions: Set[int]) -> Tuple[bool, bool]:
    """
    RespondedExistence(x, y): □(x → ◇y ∨ ◇⁻y)
    Activation: each occurrence of x
    Target: y occurs somewhere in the trace (before or after x)

    Event-based: position i activates if trace[i] == x.
    Satisfied if y exists anywhere in the trace.
    """
    if trace[pos] != x:
        return (False, False)
    # Activated: x occurs at pos
    # Target: y exists anywhere in trace
    satisfied = len(y_positions) > 0
    return (True, satisfied)


def check_Response_event(trace: List[str], pos: int,
                         x: str, y: str,
                         y_positions: Set[int]) -> Tuple[bool, bool]:
    """
    Response(x, y): □(x → ◇y)
    Activation: each occurrence of x
    Target: y eventually occurs at some later position after this x

    Event-based: position i activates if trace[i] == x.
    Satisfied if ∃ j > i such that trace[j] == y.
    """
    if trace[pos] != x:
        return (False, False)
    # Check if any y position is after pos
    satisfied = any(j > pos for j in y_positions)
    return (True, satisfied)


def check_AlternateResponse_event(trace: List[str], pos: int,
                                   x: str, y: str,
                                   x_positions: List[int],
                                   y_positions: Set[int]) -> Tuple[bool, bool]:
    """
    AlternateResponse(x, y): □(x → (¬x U y)) / ⊙(¬x U y)
    Activation: each occurrence of x
    Target: y occurs after this x, and no other x occurs before that y

    Event-based: position i activates if trace[i] == x.
    Satisfied if y occurs before the next x (or end of trace if no next x).
    """
    if trace[pos] != x:
        return (False, False)

    # Find next x after pos — required for activation (pair-based semantics)
    next_x = None
    for xp in x_positions:
        if xp > pos:
            next_x = xp
            break

    # Last x with no subsequent x: not activated (vacuously not triggered)
    if next_x is None:
        return (False, False)

    # y must occur strictly between this x and the next x
    satisfied = any(pos < j < next_x for j in y_positions)
    return (True, satisfied)


def check_ChainResponse_event(trace: List[str], pos: int,
                               x: str, y: str) -> Tuple[bool, bool]:
    """
    ChainResponse(x, y): □(x → ⊙y)
    Activation: each occurrence of x
    Target: y is the immediately next activity

    Event-based: position i activates if trace[i] == x.
    Satisfied if i+1 < len(trace) and trace[i+1] == y.
    """
    if trace[pos] != x:
        return (False, False)
    satisfied = (pos + 1 < len(trace)) and (trace[pos + 1] == y)
    return (True, satisfied)


# ---------------------------------------------------------------------------
# POSITIVE RELATION CONSTRAINTS (Backward-looking, activated by y)
# ---------------------------------------------------------------------------

def check_Precedence_event(trace: List[str], pos: int,
                           x: str, y: str,
                           x_positions: Set[int]) -> Tuple[bool, bool]:
    """
    Precedence(x, y): □(y → ◇⁻x)
    Activation: each occurrence of y
    Target: x has occurred at some earlier position before this y

    Event-based: position i activates if trace[i] == y.
    Satisfied if ∃ j < i such that trace[j] == x.
    """
    if trace[pos] != y:
        return (False, False)
    satisfied = any(j < pos for j in x_positions)
    return (True, satisfied)


def check_AlternatePrecedence_event(trace: List[str], pos: int,
                                     x: str, y: str,
                                     x_positions: List[int],
                                     y_positions: List[int]) -> Tuple[bool, bool]:
    """
    AlternatePrecedence(x, y): □(y → ⊖(¬y S x))
    Activation: each occurrence of y
    Target: x occurred before this y with no intervening y between that x and this y

    Event-based: position i activates if trace[i] == y.
    Satisfied if there is an x between the previous y (exclusive) and this y (exclusive).
    """
    if trace[pos] != y:
        return (False, False)

    # Find previous y before pos — required for activation (pair-based semantics)
    prev_y = None
    for yp in reversed(y_positions):
        if yp < pos:
            prev_y = yp
            break

    # First y with no preceding y: not activated (vacuously not triggered)
    if prev_y is None:
        return (False, False)

    # x must occur strictly between the previous y and this y
    satisfied = any(prev_y < j < pos for j in x_positions)
    return (True, satisfied)


def check_ChainPrecedence_event(trace: List[str], pos: int,
                                 x: str, y: str) -> Tuple[bool, bool]:
    """
    ChainPrecedence(x, y): □(y → ⊖x)
    Activation: each occurrence of y
    Target: x is the immediately preceding activity

    Event-based: position i activates if trace[i] == y.
    Satisfied if i > 0 and trace[i-1] == x.
    """
    if trace[pos] != y:
        return (False, False)
    satisfied = (pos > 0) and (trace[pos - 1] == x)
    return (True, satisfied)


# ---------------------------------------------------------------------------
# NEGATIVE RELATION CONSTRAINTS
# ---------------------------------------------------------------------------

def check_NotRespondedExistence_event(trace: List[str], pos: int,
                                       x: str, y: str,
                                       y_positions: Set[int]) -> Tuple[bool, bool]:
    """
    NotRespondedExistence(x, y): (x → (□¬y ∧ □⁻¬y))
    Activation: each occurrence of x
    Target: y does NOT occur anywhere in the trace

    Event-based: position i activates if trace[i] == x.
    Satisfied if y is completely absent from the trace.
    """
    if trace[pos] != x:
        return (False, False)
    satisfied = len(y_positions) == 0
    return (True, satisfied)


def check_NotResponse_event(trace: List[str], pos: int,
                             x: str, y: str,
                             y_positions: Set[int]) -> Tuple[bool, bool]:
    """
    NotResponse(x, y): (x → □¬y)
    Activation: each occurrence of x, but only when y also appears in the trace.
    If y is absent from the trace, the forbidden pattern can never occur, so
    no activation (mirrors NotChainSuccession guard).
    Target: y does NOT occur at any later position after this x.

    Event-based: position i activates if trace[i] == x AND y ∈ trace.
    Satisfied if no y occurs at position j > i.
    """
    if trace[pos] != x:
        return (False, False)
    if y not in trace:
        return (False, False)
    satisfied = not any(j > pos for j in y_positions)
    return (True, satisfied)


def check_NotChainResponse_event(trace: List[str], pos: int,
                                  x: str, y: str) -> Tuple[bool, bool]:
    """
    NotChainResponse(x, y): (x → ¬⊙y)
    Activation: each occurrence of x
    Target: y is NOT the immediately next activity

    Event-based: position i activates if trace[i] == x.
    Satisfied if i+1 >= len(trace) or trace[i+1] != y.
    """
    if trace[pos] != x:
        return (False, False)
    if pos + 1 >= len(trace):
        satisfied = True  # No next event, so y cannot immediately follow
    else:
        satisfied = trace[pos + 1] != y
    return (True, satisfied)


def check_NotPrecedence_event(trace: List[str], pos: int,
                               x: str, y: str,
                               x_positions: Set[int]) -> Tuple[bool, bool]:
    """
    NotPrecedence(x, y): (y → □⁻¬x)
    Activation: each occurrence of y
    Target: x has NOT occurred at any earlier position before this y

    Event-based: position i activates if trace[i] == y.
    Satisfied if no x occurs at position j < i.
    """
    if trace[pos] != y:
        return (False, False)
    satisfied = not any(j < pos for j in x_positions)
    return (True, satisfied)


def check_NotChainPrecedence_event(trace: List[str], pos: int,
                                    x: str, y: str) -> Tuple[bool, bool]:
    """
    NotChainPrecedence(y, x): (y → ¬⊖x)
    Activation: each occurrence of y
    Target: x is NOT the immediately preceding activity

    NOTE: Di Ciccio & Montali Table 2 writes NotChainPrecedence(y, x)
    with reversed parameter order. The activation is y, checking that x
    did NOT immediately precede. We follow the convention that the user
    specifies NotChainPrecedence(x, y) meaning "x does not immediately
    precede y", where y is the activator.

    Event-based: position i activates if trace[i] == y.
    Satisfied if i == 0 or trace[i-1] != x.
    """
    if trace[pos] != y:
        return (False, False)
    if pos == 0:
        satisfied = True  # No predecessor, so x cannot immediately precede
    else:
        satisfied = trace[pos - 1] != x
    return (True, satisfied)


# ---------------------------------------------------------------------------
# SUCCESSION CONSTRAINTS (dual activation: x ∨ y)
# ---------------------------------------------------------------------------

def check_Succession_event(
    trace: List[str], pos: int,
    x: str, y: str,
    x_positions: List[int], y_positions: List[int]
) -> Tuple[bool, bool]:
    """
    Succession(x, y):  □(x → ◇y)  ∧  □(y → ◆x)
    = Response(x, y) ∧ Precedence(x, y)

    Activation: x ∨ y  (dual — each conjunct activates independently)
      • x at pos i  →  ∃ j > i : trace[j] = y        [Response branch]
      • y at pos i  →  ∃ j < i : trace[j] = x        [Precedence branch]

    Di Ciccio & Montali (2022), Table 2.
    """
    if trace[pos] == x:
        satisfied = any(j > pos for j in y_positions)
        return (True, satisfied)
    elif trace[pos] == y:
        satisfied = any(j < pos for j in x_positions)
        return (True, satisfied)
    return (False, False)


def check_AlternateSuccession_event(
    trace: List[str], pos: int,
    x: str, y: str,
    x_positions: List[int], y_positions: List[int]
) -> Tuple[bool, bool]:
    """
    AlternateSuccession(x, y):  □(x → ⊙(¬x U y))  ∧  □(y → ⊖(¬y S x))
    = AlternateResponse(x, y) ∧ AlternatePrecedence(x, y)

    Activation uses a pair-based (weak-next) semantics:
      • x at pos i activates only if a next x exists at pos j (i < j);
        satisfaction: y appears strictly in (i, j)            [AltResponse branch]
      • y at pos i activates only if a previous y exists at pos k (k < i);
        satisfaction: x appears strictly in (k, i)           [AltPrecedence branch]

    The last x (no subsequent x) and the first y (no preceding y) are
    vacuously not activated, avoiding spurious violations at trace boundaries.

    Di Ciccio & Montali (2022), Table 2.
    """
    if trace[pos] == x:
        # AlternateResponse: activated only when a next x exists (consecutive x-x pair)
        next_x = next((xp for xp in x_positions if xp > pos), None)
        if next_x is None:
            return (False, False)
        satisfied = any(pos < j < next_x for j in y_positions)
        return (True, satisfied)
    elif trace[pos] == y:
        # AlternatePrecedence: activated only when a previous y exists (consecutive y-y pair)
        prev_y = next((yp for yp in reversed(y_positions) if yp < pos), None)
        if prev_y is None:
            return (False, False)
        satisfied = any(prev_y < j < pos for j in x_positions)
        return (True, satisfied)
    return (False, False)


def check_ChainSuccession_event(
    trace: List[str], pos: int,
    x: str, y: str
) -> Tuple[bool, bool]:
    """
    ChainSuccession(x, y):  □(x → ⊙y)  ∧  □(y → ⊖x)
    = ChainResponse(x, y) ∧ ChainPrecedence(x, y)

    Activation: x ∨ y  (dual)
      • x at pos i  →  trace[i+1] = y                [ChainResponse branch]
      • y at pos i  →  trace[i-1] = x                [ChainPrecedence branch]

    Di Ciccio & Montali (2022), Table 2.
    """
    if trace[pos] == x:
        satisfied = (pos + 1 < len(trace)) and (trace[pos + 1] == y)
        return (True, satisfied)
    elif trace[pos] == y:
        satisfied = (pos > 0) and (trace[pos - 1] == x)
        return (True, satisfied)
    return (False, False)


def check_NotSuccession_event(
    trace: List[str], pos: int,
    x: str, y: str,
    x_positions: List[int], y_positions: List[int]
) -> Tuple[bool, bool]:
    """
    NotSuccession(x, y):  □(x → □¬y)  ∧  □(y → □⁻¬x)
    = NotResponse(x, y) ∧ NotPrecedence(x, y)

    Activation: x ∨ y  (dual — symmetric with Succession)
      • x at pos i  →  ¬∃ j > i : trace[j] = y      [NotResponse branch]
      • y at pos i  →  ¬∃ j < i : trace[j] = x      [NotPrecedence branch]

    Note: NotResponse ≡ NotPrecedence by language (E1), but dual activation
    gives separate event-level statistics for x-coverage and y-coverage,
    providing richer MINERful measures than single-activation NotResponse.

    Di Ciccio & Montali (2022), Table 2; E1 equivalence §4.2.
    """
    if trace[pos] == x:
        satisfied = not any(j > pos for j in y_positions)
        return (True, satisfied)
    elif trace[pos] == y:
        satisfied = not any(j < pos for j in x_positions)
        return (True, satisfied)
    return (False, False)


def check_NotChainSuccession_event(
    trace: List[str], pos: int,
    x: str, y: str
) -> Tuple[bool, bool]:
    """
    NotChainSuccession(x, y):  □(x → ¬⊙y)  ∧  □(y → ¬⊖x)
    = NotChainResponse(x, y) ∧ NotChainPrecedence(x, y)

    Activation: x ∨ y  (dual — symmetric with ChainSuccession), but only
    when BOTH x and y appear in the trace. If the other activity is absent,
    the immediate-succession pattern can never occur, so no activation.
      • x at pos i (y in trace) →  trace[i+1] ≠ y  [NotChainResponse branch]
      • y at pos i (x in trace) →  trace[i-1] ≠ x  [NotChainPrecedence branch]

    Di Ciccio & Montali (2022), Table 2; E2 equivalence §4.2.
    """
    if trace[pos] == x:
        if y not in trace:
            return (False, False)
        satisfied = not ((pos + 1 < len(trace)) and (trace[pos + 1] == y))
        return (True, satisfied)
    elif trace[pos] == y:
        if x not in trace:
            return (False, False)
        satisfied = not ((pos > 0) and (trace[pos - 1] == x))
        return (True, satisfied)
    return (False, False)


# =============================================================================
# UNIFIED EVALUATION ENGINE
# =============================================================================

def evaluate_constraint_on_trace(
    constraint_type: str,
    param_a: str,
    param_b: Optional[str],
    trace: List[str],
    activity_positions: Dict[str, List[int]]
) -> Tuple[int, int, int, bool, bool]:
    """
    Evaluate a constraint on a single trace, returning both event-based
    and trace-based counts.

    Returns:
        (n_activated_events, n_satisfied_events, n_total_events,
         trace_activated, trace_satisfied)

    Where:
      - n_activated_events: count of events where activation fires
      - n_satisfied_events: count of events where activation fires AND target satisfied
      - n_total_events: total events in trace (len(trace))
      - trace_activated: True if activation fires in at least one event
      - trace_satisfied: True if constraint is non-vacuously satisfied
        (activated at least once, and ALL activations are satisfied)
    """
    n = len(trace)
    if n == 0:
        return (0, 0, 0, False, False)

    # Pre-compute position sets for efficiency
    a_positions_list = activity_positions.get(param_a, [])  # sorted by construction
    a_positions_set = set(a_positions_list)

    if param_b is not None:
        b_positions_list = activity_positions.get(param_b, [])
        b_positions_set = set(b_positions_list)
    else:
        b_positions_list = []
        b_positions_set = set()

    n_activated = 0
    n_satisfied = 0

    for pos in range(n):
        activated = False
        satisfied = False

        # --- UNARY ---
        if constraint_type == 'Init':
            activated, satisfied = check_Init_event(trace, pos, param_a)

        elif constraint_type == 'End':
            activated, satisfied = check_End_event(trace, pos, param_a)

        # --- BINARY POSITIVE (forward: activated by x=param_a) ---
        # elif constraint_type == 'RespondedExistence':
        #     activated, satisfied = check_RespondedExistence_event(
        #         trace, pos, param_a, param_b, b_positions_set)

        elif constraint_type == 'Response':
            activated, satisfied = check_Response_event(
                trace, pos, param_a, param_b, b_positions_set)

        elif constraint_type == 'AlternateResponse':
            activated, satisfied = check_AlternateResponse_event(
                trace, pos, param_a, param_b, a_positions_list, b_positions_set)

        elif constraint_type == 'ChainResponse':
            activated, satisfied = check_ChainResponse_event(
                trace, pos, param_a, param_b)

        # --- BINARY POSITIVE (backward: activated by y=param_b) ---
        elif constraint_type == 'Precedence':
            activated, satisfied = check_Precedence_event(
                trace, pos, param_a, param_b, a_positions_set)

        elif constraint_type == 'AlternatePrecedence':
            activated, satisfied = check_AlternatePrecedence_event(
                trace, pos, param_a, param_b, a_positions_list, b_positions_list)

        elif constraint_type == 'ChainPrecedence':
            activated, satisfied = check_ChainPrecedence_event(
                trace, pos, param_a, param_b)

        # --- BINARY NEGATIVE (forward: activated by x=param_a) ---
        # elif constraint_type == 'NotRespondedExistence':
        #     activated, satisfied = check_NotRespondedExistence_event(
        #         trace, pos, param_a, param_b, b_positions_set)

        elif constraint_type == 'NotResponse':
            activated, satisfied = check_NotResponse_event(
                trace, pos, param_a, param_b, b_positions_set)

        elif constraint_type == 'NotChainResponse':
            activated, satisfied = check_NotChainResponse_event(
                trace, pos, param_a, param_b)

        # --- BINARY NEGATIVE (backward: activated by y=param_b) ---
        elif constraint_type == 'NotPrecedence':
            activated, satisfied = check_NotPrecedence_event(
                trace, pos, param_a, param_b, a_positions_set)

        elif constraint_type == 'NotChainPrecedence':
            activated, satisfied = check_NotChainPrecedence_event(
                trace, pos, param_a, param_b)

        # --- SUCCESSION CONSTRAINTS (dual activation: x ∨ y) ---
        elif constraint_type == 'Succession':
            activated, satisfied = check_Succession_event(
                trace, pos, param_a, param_b, a_positions_list, b_positions_list)

        elif constraint_type == 'AlternateSuccession':
            activated, satisfied = check_AlternateSuccession_event(
                trace, pos, param_a, param_b, a_positions_list, b_positions_list)

        elif constraint_type == 'ChainSuccession':
            activated, satisfied = check_ChainSuccession_event(
                trace, pos, param_a, param_b)

        elif constraint_type == 'NotSuccession':
            activated, satisfied = check_NotSuccession_event(
                trace, pos, param_a, param_b, a_positions_list, b_positions_list)

        elif constraint_type == 'NotChainSuccession':
            activated, satisfied = check_NotChainSuccession_event(
                trace, pos, param_a, param_b)

        if activated:
            n_activated += 1
            if satisfied:
                n_satisfied += 1

    trace_activated = n_activated > 0
    trace_satisfied = (n_activated > 0) and (n_satisfied == n_activated)

    return (n_activated, n_satisfied, n, trace_activated, trace_satisfied)


def compute_measures(
    constraint_type: str,
    param_a: str,
    param_b: Optional[str],
    cases: List[CaseTrace]
) -> ConstraintMeasures:
    """
    Compute MINERful event-based and trace-based measures for a constraint
    across all cases.

    Definitions (Di Ciccio & Montali 2022, Def 14 & 15; Iacometta & Di Ciccio 2025, Table 2):

    Event-based:
      confidence = #e(L, if(κ) ∧ then(κ)) / max{1, #e(L, if(κ))}
      coverage   = #e(L, if(κ))            / max{1, #e(L, ⊤)}
      support    = #e(L, if(κ) ∧ then(κ)) / max{1, #e(L, ⊤)}

    Trace-based:
      confidence = #t(L, if(κ) ∧ then(κ)) / max{1, #t(L, if(κ))}
      coverage   = #t(L, if(κ))            / max{1, #t(L, ⊤)}
      support    = #t(L, if(κ) ∧ then(κ)) / max{1, #t(L, ⊤)}

    IMPORTANT: For trace-based, "if(κ) ∧ then(κ)" means the trace is
    NON-VACUOUSLY satisfied: activation fires AND all activations are satisfied.
    "if(κ)" means the activation fires at least once in the trace.
    """
    total_events = 0
    total_activated_events = 0
    total_satisfied_events = 0

    total_traces = len(cases)
    total_activated_traces = 0
    total_satisfied_traces = 0

    for case in cases:
        n_act, n_sat, n_events, t_act, t_sat = evaluate_constraint_on_trace(
            constraint_type, param_a, param_b,
            case.pc_activities, case.activity_positions
        )
        total_events += n_events
        total_activated_events += n_act
        total_satisfied_events += n_sat

        if t_act:
            total_activated_traces += 1
        if t_sat:
            total_satisfied_traces += 1

    # Event-based measures
    event_conf = total_satisfied_events / max(1, total_activated_events)
    event_cov = total_activated_events / max(1, total_events)
    event_supp = total_satisfied_events / max(1, total_events)

    # Trace-based measures
    trace_conf = total_satisfied_traces / max(1, total_activated_traces)
    trace_cov = total_activated_traces / max(1, total_traces)
    trace_supp = total_satisfied_traces / max(1, total_traces)

    return ConstraintMeasures(
        event_confidence=event_conf,
        event_coverage=event_cov,
        event_support=event_supp,
        trace_confidence=trace_conf,
        trace_coverage=trace_cov,
        trace_support=trace_supp,
        n_events_activated_and_satisfied=total_satisfied_events,
        n_events_activated=total_activated_events,
        n_events_total=total_events,
        n_traces_activated_and_satisfied=total_satisfied_traces,
        n_traces_activated=total_activated_traces,
        n_traces_total=total_traces,
    )


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

_OPPOSITE_DIRECTIONS: Dict[str, str] = {
    'INCREASE': 'DECREASE', 'DECREASE': 'INCREASE',
    'ENABLE':   'DISABLE',  'DISABLE':  'ENABLE',
}

def _same_activity_opposite_direction(a: str, b: str) -> bool:
    """
    Return True if a and b name the same underlying activity but with
    opposite direction tokens (INCREASE/DECREASE or ENABLE/DISABLE).

    Comparison is token-level: exactly one token must differ, and that
    differing token must be an opposite-direction pair.
    Example: "PC INCREASE expansion" vs "PC DECREASE expansion" → True.
    """
    tokens_a = a.split()
    tokens_b = b.split()
    if len(tokens_a) != len(tokens_b):
        return False
    diffs = [(ta, tb) for ta, tb in zip(tokens_a, tokens_b) if ta != tb]
    if len(diffs) != 1:
        return False
    ta, tb = diffs[0]
    return _OPPOSITE_DIRECTIONS.get(ta.upper()) == tb.upper()


def generate_candidates(
    frequent_activities: List[str]
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Generate exhaustive candidate set: all (constraint_type, param_a, param_b).

    For unary constraints: all activities.
    For binary constraints: all ordered pairs (a != b), excluding pairs where
    the two activities are the same underlying concept with opposite directions
    (INCREASE/DECREASE or ENABLE/DISABLE) — e.g. "PC INCREASE expansion" with
    "PC DECREASE expansion" would be meaningless as a relational constraint.
    """
    candidates = []

    # Unary
    for act in frequent_activities:
        for ct in UNARY_CONSTRAINTS:
            candidates.append((ct, act, None))

    # Binary: all ordered pairs, skipping opposite-direction same-activity pairs
    for a, b in itertools.permutations(frequent_activities, 2):
        if _same_activity_opposite_direction(a, b):
            continue
        for ct in BINARY_POSITIVE_CONSTRAINTS + BINARY_NEGATIVE_CONSTRAINTS:
            candidates.append((ct, a, b))

    return candidates


# =============================================================================
# THRESHOLD FILTERING (Algorithm 1, lines 4-6)
# =============================================================================

def passes_threshold(m: ConstraintMeasures) -> bool:
    """
    Check if measures pass minimum thresholds.
    Following Algorithm 1 from Di Ciccio & Montali (2022):
    Remove if any measure falls below its threshold.

    We use the 'hard' thresholds for confidence as the main filter.
    """
    if m.event_confidence < CONFIG['min_event_confidence_hard']:
        return False
    if m.trace_confidence < CONFIG['min_trace_confidence_hard']:
        return False
    if m.event_support < CONFIG['min_event_support']:
        return False
    if m.trace_support < CONFIG['min_trace_support']:
        return False
    return True


# =============================================================================
# SUBSUMPTION PRUNING (Algorithm 1, lines 7-14)
# =============================================================================

def measures_leq(m1: ConstraintMeasures, m2: ConstraintMeasures) -> bool:
    """Check if all measures of m1 <= all measures of m2."""
    return (
        m1.event_confidence <= m2.event_confidence and
        m1.event_support <= m2.event_support and
        m1.trace_confidence <= m2.trace_confidence and
        m1.trace_support <= m2.trace_support
    )


def apply_subsumption_pruning(
    constraints: List[DiscoveredConstraint]
) -> List[DiscoveredConstraint]:
    """
    Apply subsumption-based pruning (Algorithm 1, lines 7-14).

    For each pair (k, k') where k subsumes k' (same parameters):
    - If measures(k) >= measures(k'): remove k' (subsumed, keep more restrictive k)
    - Else: remove k (subsuming constraint has worse measures, less informative)

    This implements Figure 8's subsumption hierarchy.
    """
    print("\n  Applying subsumption pruning...")

    # Index constraints by (param_a, param_b) for efficient lookup
    by_params = defaultdict(dict)
    for c in constraints:
        key = (c.param_a, c.param_b)
        by_params[key][c.constraint_type] = c

    to_remove = set()
    n_removed = 0

    for _, type_map in by_params.items():
        for subsuming_type, subsumed_types in SUBSUMPTION_CHAINS.items():
            if subsuming_type not in type_map:
                continue
            k_sub = type_map[subsuming_type]  # the more restrictive constraint

            for subsumed_type in subsumed_types:
                if subsumed_type not in type_map:
                    continue
                k_sup = type_map[subsumed_type]  # the less restrictive constraint

                if k_sub.pattern_id in to_remove or k_sup.pattern_id in to_remove:
                    continue

                # k_sub subsumes k_sup (k_sub is more restrictive)
                if measures_leq(k_sup.measures, k_sub.measures):
                    # measures(k_sub) >= measures(k_sup): remove k_sup (redundant)
                    to_remove.add(k_sup.pattern_id)
                    n_removed += 1
                else:
                    # k_sub has worse measures than k_sup: remove k_sub
                    to_remove.add(k_sub.pattern_id)
                    n_removed += 1

    result = [c for c in constraints if c.pattern_id not in to_remove]
    print(f"    Removed {n_removed} constraints via subsumption")
    print(f"    Remaining: {len(result)}")
    return result

def apply_cross_param_subsumption_pruning(
    constraints: List[DiscoveredConstraint]
) -> List[DiscoveredConstraint]:
    """
    Handle cross-parameter subsumptions (Figure 8, Di Ciccio & Montali 2022).

    The same-parameter logic in apply_subsumption_pruning indexes by (param_a, param_b)
    and therefore CANNOT detect pairs like:
        Precedence(a, b) [more restrictive] ⊑ RespondedExistence(b, a) [less restrictive]
    where the subsuming and subsumed constraints carry swapped activity roles.

    Pruning rule (mirrors Algorithm 1, Di Ciccio & Montali 2022):
      Let k  = subsuming_type on (a, b)  [more restrictive]
      Let k' = subsumed_type  on (b, a)  [less restrictive]

      Case 1: allm(k) >= allm(k')  → k' is logically AND empirically redundant → remove k'
      Case 2: allm(k) <  allm(k')  → k is over-claiming relative to the data   → remove k
              (e.g. Precedence conf=0.80, RespondedExistence(b,a) conf=0.97:
               the log supports co-occurrence but not strict ordering → drop Precedence)

    NOTE: Because activation of Precedence(a,b) [= each b occurrence] and
    RespondedExistence(b,a) [= ◇b, whole-trace] are structurally aligned on the
    same activity b, the trace-level confidence values ARE directly comparable.
    """
    print("\n  Applying cross-parameter subsumption pruning...")
    print("  Handles: Precedence/AlternatePrecedence/ChainPrecedence(a,b) ⊑ RespondedExistence(b,a)")

    # Build a fast lookup: (constraint_type, param_a, param_b) -> DiscoveredConstraint
    index: Dict[Tuple[str, str, Optional[str]], DiscoveredConstraint] = {
        (c.constraint_type, c.param_a, c.param_b): c
        for c in constraints
    }

    to_remove: Set[str] = set()
    log_entries = []

    for subsuming_type, subsumed_type in CROSS_PARAM_SUBSUMPTIONS:
        for c in constraints:
            if c.constraint_type != subsuming_type:
                continue
            if c.pattern_id in to_remove:
                continue
            if c.param_b is None:
                continue  # safety: only binary constraints have cross-param

            # k  = subsuming_type(a, b) — more restrictive
            # k' = subsumed_type(b, a)  — less restrictive (swapped params)
            swapped_key = (subsumed_type, c.param_b, c.param_a)
            if swapped_key not in index:
                continue

            k_prime = index[swapped_key]  # the less restrictive, swapped-param constraint
            if k_prime.pattern_id in to_remove:
                continue

            # Apply Algorithm 1 measure comparison
            if measures_leq(k_prime.measures, c.measures):
                # Case 1: subsuming (k) has >= measures → k' is redundant → remove k'
                to_remove.add(k_prime.pattern_id)
                log_entries.append(
                    f"    REMOVE {k_prime.pattern_id} "
                    f"(subsumed by {c.pattern_id}, measures: "
                    f"tconf {c.measures.trace_confidence:.3f} >= {k_prime.measures.trace_confidence:.3f})"
                )
            else:
                # Case 2: k' has strictly better measures → k over-claims → remove k
                to_remove.add(c.pattern_id)
                log_entries.append(
                    f"    REMOVE {c.pattern_id} "
                    f"(over-claims vs {k_prime.pattern_id}, measures: "
                    f"tconf {c.measures.trace_confidence:.3f} < {k_prime.measures.trace_confidence:.3f})"
                )

    for entry in log_entries:
        print(entry)

    result = [c for c in constraints if c.pattern_id not in to_remove]
    print(f"    Removed {len(to_remove)} constraints via cross-parameter subsumption")
    print(f"    Remaining: {len(result)}")
    return result

# =============================================================================
# NEGATION PRUNING (Algorithm 1, lines 15-17)
# =============================================================================

def apply_negation_pruning(
    constraints: List[DiscoveredConstraint]
) -> List[DiscoveredConstraint]:
    """
    Apply negation-pair pruning (Algorithm 1, lines 15-17).

    For each negation pair (k, ¬k) on the same parameters:
    - Keep whichever has strictly better measures
    - If equal, keep both (user decides)
    """
    print("\n  Applying negation pruning...")

    by_params = defaultdict(dict)
    for c in constraints:
        key = (c.param_a, c.param_b)
        by_params[key][c.constraint_type] = c

    to_remove = set()
    n_removed = 0

    for _, type_map in by_params.items():
        for pos_type, neg_type in NEGATION_PAIRS:
            if pos_type not in type_map or neg_type not in type_map:
                continue
            k_pos = type_map[pos_type]
            k_neg = type_map[neg_type]

            if k_pos.pattern_id in to_remove or k_neg.pattern_id in to_remove:
                continue

            # Compare: keep whichever has strictly better measures
            if measures_leq(k_neg.measures, k_pos.measures) and not measures_leq(k_pos.measures, k_neg.measures):
                # pos strictly better: remove neg
                to_remove.add(k_neg.pattern_id)
                n_removed += 1
            elif measures_leq(k_pos.measures, k_neg.measures) and not measures_leq(k_neg.measures, k_pos.measures):
                # neg strictly better: remove pos
                to_remove.add(k_pos.pattern_id)
                n_removed += 1
            # If neither strictly dominates, keep both

    result = [c for c in constraints if c.pattern_id not in to_remove]
    print(f"    Removed {n_removed} constraints via negation pruning")
    print(f"    Remaining: {len(result)}")
    return result


# =============================================================================
# FINITE STATE AUTOMATON (FSA) IMPLEMENTATION
# =============================================================================
# Implements the local automata from Definition 4 and the product construction
# from Definition 7 of Di Ciccio & Montali (2022).
# Used by Tier 1 (language equivalence), Tier 2 (transitivity), and the full
# Tier 3 language-inclusion entailment check (Section 4.2 / Example 14).
# =============================================================================

class FSA:
    """Deterministic, complete FSA over a finite alphabet."""

    def __init__(
        self,
        states: Set,
        alphabet: Set[str],
        transitions: Dict,   # (state, symbol) -> state
        initial: object,
        accepting: Set,
    ):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial = initial
        self.accepting = accepting

    # ------------------------------------------------------------------
    def complement(self) -> 'FSA':
        """Return the complement FSA (flip accepting / non-accepting).

        The FSAs produced by _make_fsa are already complete (total transition
        function), so no implicit dead-state completion is needed.  We keep
        the logic for safety in case an incomplete FSA is ever passed in.
        """
        _DEAD = '__DEAD__'
        new_trans = dict(self.transitions)
        needs_dead = False
        for state in self.states:
            for sym in self.alphabet:
                if (state, sym) not in new_trans:
                    new_trans[(state, sym)] = _DEAD
                    needs_dead = True
        if needs_dead:
            for sym in self.alphabet:
                new_trans[(_DEAD, sym)] = _DEAD
            new_states = self.states | {_DEAD}
        else:
            new_states = self.states

        # Flip: non-accepting in original become accepting in complement.
        # The dead trap state (_DEAD) remains non-accepting in the complement
        # (it represents traces that are already in violation — not "new" models).
        new_accepting = (new_states - self.accepting) - {_DEAD}

        return FSA(
            states=new_states,
            alphabet=self.alphabet,
            transitions=new_trans,
            initial=self.initial,
            accepting=new_accepting,
        )

    # ------------------------------------------------------------------
    def product(self, other: 'FSA') -> 'FSA':
        """Synchronous product (Definition 7, Di Ciccio & Montali 2022).

        Accepts iff BOTH component FSAs simultaneously accept.
        Product states are (s1, s2) tuples; only reachable states are built.
        Missing transitions are mapped to an absorbing dead-pair state.
        """
        from collections import deque
        _DEAD = '__DEAD__'

        def step(fsa: 'FSA', state, sym):
            nxt = fsa.transitions.get((state, sym))
            return _DEAD if nxt is None else nxt

        initial_pair = (self.initial, other.initial)
        new_states: Set = set()
        new_trans: Dict = {}
        new_accepting: Set = set()

        queue: deque = deque([initial_pair])
        visited: Set = {initial_pair}

        while queue:
            (s1, s2) = queue.popleft()
            new_states.add((s1, s2))
            if s1 in self.accepting and s2 in other.accepting:
                new_accepting.add((s1, s2))

            for sym in self.alphabet:
                t1 = step(self, s1, sym)
                t2 = step(other, s2, sym)
                pair = (t1, t2)
                new_trans[((s1, s2), sym)] = pair
                if pair not in visited:
                    visited.add(pair)
                    queue.append(pair)

        # Add dead-pair self-loops if needed
        if (_DEAD, _DEAD) in new_states or any(
            _DEAD in p for p in new_states if isinstance(p, tuple)
        ):
            for sym in self.alphabet:
                new_trans[((_DEAD, _DEAD), sym)] = (_DEAD, _DEAD)
            new_states.add((_DEAD, _DEAD))

        return FSA(
            states=new_states,
            alphabet=self.alphabet,
            transitions=new_trans,
            initial=initial_pair,
            accepting=new_accepting,
        )

    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Return True iff the accepted language is empty (no accepting state reachable)."""
        from collections import deque
        if self.initial in self.accepting:
            return False
        visited: Set = {self.initial}
        queue: deque = deque([self.initial])
        while queue:
            state = queue.popleft()
            for sym in self.alphabet:
                nxt = self.transitions.get((state, sym))
                if nxt is None:
                    continue
                if nxt in self.accepting:
                    return False
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return True


def _make_fsa(
    constraint_type: str,
    a: str,
    b: Optional[str],
    sigma: Set[str],
) -> FSA:
    """Build the deterministic FSA for a DECLARE constraint template.

    The FSA is defined over the full alphabet sigma.  For each symbol, the
    transition function distinguishes three equivalence classes:
        'a'     — matches param_a (the activating activity for forward templates)
        'b'     — matches param_b (the target activity)
        'other' — any symbol in sigma that is neither a nor b

    State naming follows the FSA catalogue in the analysis:
      Response/Precedence family: s0 (ACC), s1 (REJ), s_dead (trap)
      Negative family: s_free (ACC), s_xseen (ACC), s_dead (trap)
      etc.

    FSA semantics follow Di Ciccio & Montali (2022), Table 2 and Definition 4.
    """

    def sc(sym: str) -> str:
        if sym == a:
            return 'a'
        if b is not None and sym == b:
            return 'b'
        return 'other'

    trans: Dict = {}

    # ------------------------------------------------------------------
    if constraint_type == 'Init':
        # □(start → a): first event must be a.
        # s0 (ACC, initial) — s_ok (ACC) — s_dead (trap)
        states = {'s0', 's_ok', 's_dead'}
        accepting = {'s0', 's_ok'}
        initial = 's0'
        for sym in sigma:
            c = sc(sym)
            trans[('s0', sym)]     = 's_ok'   if c == 'a' else 's_dead'
            trans[('s_ok', sym)]   = 's_ok'
            trans[('s_dead', sym)] = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type == 'End':
        # □(end → a): last event must be a.
        # s_nota (REJ, initial) — s_a (ACC)
        states = {'s_nota', 's_a'}
        accepting = {'s_a'}
        initial = 's_nota'
        for sym in sigma:
            c = sc(sym)
            trans[('s_nota', sym)] = 's_a'    if c == 'a' else 's_nota'
            trans[('s_a',    sym)] = 's_a'    if c == 'a' else 's_nota'

    # ------------------------------------------------------------------
    elif constraint_type == 'Response':
        # □(a → ◇b)
        # s0 (ACC) — s1 (REJ: pending a)
        states = {'s0', 's1'}
        accepting = {'s0'}
        initial = 's0'
        for sym in sigma:
            c = sc(sym)
            trans[('s0', sym)] = 's1' if c == 'a' else 's0'
            trans[('s1', sym)] = 's0' if c == 'b' else 's1'

    # ------------------------------------------------------------------
    elif constraint_type == 'AlternateResponse':
        # □(a → (¬a U b))
        # s0 (ACC) — s1 (REJ: a seen, must reach b before next a) — s_dead (trap)
        states = {'s0', 's1', 's_dead'}
        accepting = {'s0'}
        initial = 's0'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s0', sym)]     = 's1'
                trans[('s1', sym)]     = 's_dead'   # second a before b
                trans[('s_dead', sym)] = 's_dead'
            elif c == 'b':
                trans[('s0', sym)]     = 's0'
                trans[('s1', sym)]     = 's0'        # b satisfies
                trans[('s_dead', sym)] = 's_dead'
            else:
                trans[('s0', sym)]     = 's0'
                trans[('s1', sym)]     = 's1'
                trans[('s_dead', sym)] = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type == 'ChainResponse':
        # □(a → ⊙b)
        # s0 (ACC) — s1 (REJ: a seen, immediate next must be b) — s_dead (trap)
        states = {'s0', 's1', 's_dead'}
        accepting = {'s0'}
        initial = 's0'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s0', sym)]     = 's1'
                trans[('s1', sym)]     = 's_dead'   # second a without b
                trans[('s_dead', sym)] = 's_dead'
            elif c == 'b':
                trans[('s0', sym)]     = 's0'
                trans[('s1', sym)]     = 's0'        # b satisfies
                trans[('s_dead', sym)] = 's_dead'
            else:
                trans[('s0', sym)]     = 's0'
                trans[('s1', sym)]     = 's_dead'   # non-b after a
                trans[('s_dead', sym)] = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type == 'Precedence':
        # □(b → ◆a)
        # s_blocked (ACC, initial) — s_open (ACC) — s_dead (trap)
        states = {'s_blocked', 's_open', 's_dead'}
        accepting = {'s_blocked', 's_open'}
        initial = 's_blocked'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s_blocked', sym)] = 's_open'
                trans[('s_open',    sym)] = 's_open'
                trans[('s_dead',    sym)] = 's_dead'
            elif c == 'b':
                trans[('s_blocked', sym)] = 's_dead'   # b before any a
                trans[('s_open',    sym)] = 's_open'
                trans[('s_dead',    sym)] = 's_dead'
            else:
                trans[('s_blocked', sym)] = 's_blocked'
                trans[('s_open',    sym)] = 's_open'
                trans[('s_dead',    sym)] = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type == 'AlternatePrecedence':
        # □(b → (¬b S a))
        # s_hasA (ACC, initial) — s_hasB (ACC) — s_dead (trap)
        states = {'s_hasA', 's_hasB', 's_dead'}
        accepting = {'s_hasA', 's_hasB'}
        initial = 's_hasB'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s_hasA', sym)] = 's_hasA'
                trans[('s_hasB', sym)] = 's_hasA'   # a resets
                trans[('s_dead', sym)] = 's_dead'
            elif c == 'b':
                trans[('s_hasA', sym)] = 's_hasB'
                trans[('s_hasB', sym)] = 's_dead'   # b before any a since last b
                trans[('s_dead', sym)] = 's_dead'
            else:
                trans[('s_hasA', sym)] = 's_hasA'
                trans[('s_hasB', sym)] = 's_hasB'
                trans[('s_dead', sym)] = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type == 'ChainPrecedence':
        # □(b → ⊖a)
        # s_notA (ACC, initial) — s_A (ACC) — s_dead (trap)
        states = {'s_notA', 's_A', 's_dead'}
        accepting = {'s_notA', 's_A'}
        initial = 's_notA'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s_notA', sym)] = 's_A'
                trans[('s_A',    sym)] = 's_A'
                trans[('s_dead', sym)] = 's_dead'
            elif c == 'b':
                trans[('s_notA', sym)] = 's_dead'   # b without immediate prior a
                trans[('s_A',    sym)] = 's_notA'   # b after a: satisfied; reset
                trans[('s_dead', sym)] = 's_dead'
            else:
                trans[('s_notA', sym)] = 's_notA'
                trans[('s_A',    sym)] = 's_notA'   # non-b after a: a no longer "immediate"
                trans[('s_dead', sym)] = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type in ('NotResponse', 'NotPrecedence'):
        # E1: both have identical FSAs — same accepted language.
        # NotResponse(a,b):  □(a → □¬b)    — violation: a seen, then b
        # NotPrecedence(a,b): □(b → □⁻¬a)  — violation: a before b
        # Both reduce to: "a is never followed by b anywhere in the trace"
        # s_free (ACC, initial) — s_xseen (ACC: a seen, b now forbidden) — s_dead (trap)
        states = {'s_free', 's_xseen', 's_dead'}
        accepting = {'s_free', 's_xseen'}
        initial = 's_free'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s_free',   sym)] = 's_xseen'
                trans[('s_xseen', sym)]  = 's_xseen'
                trans[('s_dead',  sym)]  = 's_dead'
            elif c == 'b':
                trans[('s_free',   sym)] = 's_free'
                trans[('s_xseen', sym)]  = 's_dead'   # b after a: violation
                trans[('s_dead',  sym)]  = 's_dead'
            else:
                trans[('s_free',   sym)] = 's_free'
                trans[('s_xseen', sym)]  = 's_xseen'
                trans[('s_dead',  sym)]  = 's_dead'

    # ------------------------------------------------------------------
    elif constraint_type in ('NotChainResponse', 'NotChainPrecedence'):
        # E2: both have identical FSAs — same accepted language.
        # NotChainResponse(a,b):   □(a → ¬⊙b)   — violation: b immediately after a
        # NotChainPrecedence(a,b): □(b → ¬⊖a)   — violation: a immediately before b
        # Both reduce to: "a is never immediately followed by b"
        # s_safe (ACC, initial) — s_a (ACC: a seen, check next) — s_dead (trap)
        states = {'s_safe', 's_a', 's_dead'}
        accepting = {'s_safe', 's_a'}
        initial = 's_safe'
        for sym in sigma:
            c = sc(sym)
            if c == 'a':
                trans[('s_safe', sym)] = 's_a'
                trans[('s_a',    sym)] = 's_a'
                trans[('s_dead', sym)] = 's_dead'
            elif c == 'b':
                trans[('s_safe', sym)] = 's_safe'
                trans[('s_a',    sym)] = 's_dead'   # b immediately after a: violation
                trans[('s_dead', sym)] = 's_dead'
            else:
                trans[('s_safe', sym)] = 's_safe'
                trans[('s_a',    sym)] = 's_safe'   # non-b after a: reset
                trans[('s_dead', sym)] = 's_dead'

    else:
        raise ValueError(f"_make_fsa: unsupported constraint type '{constraint_type}'")

    return FSA(
        states=states,
        alphabet=sigma,
        transitions=trans,
        initial=initial,
        accepting=accepting,
    )


# =============================================================================
# MAIN DISCOVERY PIPELINE
# =============================================================================

# =============================================================================
# PHASE 4c — LANGUAGE EQUIVALENCE PRUNING (Tier 1)
# =============================================================================

def apply_language_equivalence_pruning(
    constraints: List[DiscoveredConstraint],
) -> List[DiscoveredConstraint]:
    """Remove one member of each language-equivalent template pair (E1, E2).

    E1: NotResponse(a,b) ≡ NotPrecedence(a,b)   — identical FSAs
    E2: NotChainResponse(a,b) ≡ NotChainPrecedence(a,b) — identical FSAs

    Both constraints in a pair accept the same language, so one is always
    unconditionally redundant.  We keep the one with strictly higher measures;
    ties are broken by keeping the first element of the pair (arbitrary but
    deterministic).
    """
    print("\n  Applying language-equivalence pruning (Tier 1: E1, E2)...")
    print("  E1: NotResponse ≡ NotPrecedence        (identical FSA / accepted language)")
    print("  E2: NotChainResponse ≡ NotChainPrecedence  (identical FSA / accepted language)")

    by_params: Dict = defaultdict(dict)
    for c in constraints:
        by_params[(c.param_a, c.param_b)][c.constraint_type] = c

    to_remove: Set[str] = set()
    log_entries = []

    for _, type_map in by_params.items():
        for t1, t2 in LANGUAGE_EQUIVALENCES:
            if t1 not in type_map or t2 not in type_map:
                continue
            k1 = type_map[t1]
            k2 = type_map[t2]
            if k1.pattern_id in to_remove or k2.pattern_id in to_remove:
                continue

            # Keep the one with strictly higher measures; ties → keep k1 (first in pair)
            if measures_leq(k2.measures, k1.measures):
                # k1 >= k2 on all dimensions: k2 is dominated → remove k2
                to_remove.add(k2.pattern_id)
                log_entries.append(
                    f"    REMOVE {k2.pattern_id}  ≡  {k1.pattern_id}  "
                    f"(tconf {k1.measures.trace_confidence:.3f} >= {k2.measures.trace_confidence:.3f})"
                )
            else:
                # k2 strictly better: remove k1
                to_remove.add(k1.pattern_id)
                log_entries.append(
                    f"    REMOVE {k1.pattern_id}  ≡  {k2.pattern_id}  "
                    f"(tconf {k2.measures.trace_confidence:.3f} > {k1.measures.trace_confidence:.3f})"
                )

    for entry in log_entries:
        print(entry)

    result = [c for c in constraints if c.pattern_id not in to_remove]
    print(f"    Removed {len(to_remove)} constraints via language equivalence")
    print(f"    Remaining: {len(result)}")
    return result


# =============================================================================
# PHASE 6a — TRANSITIVITY ENTAILMENT PRUNING (Tier 2)
# =============================================================================

def apply_transitivity_pruning(
    constraints: List[DiscoveredConstraint],
) -> List[DiscoveredConstraint]:
    """Remove constraints entailed by transitivity of Response / Precedence families.

    Response transitivity (LTLf):
        T1(a,b) ∧ T2(b,c)  ⊨  Response(a,c)
        T1, T2 ∈ {Response, AlternateResponse, ChainResponse}
        Proof: □(a→◇b) ∧ □(b→◇c)  ⊨  □(a→◇c)

    Precedence transitivity (LTLf):
        T1(a,b) ∧ T2(b,c)  ⊨  Precedence(a,c)
        T1, T2 ∈ {Precedence, AlternatePrecedence, ChainPrecedence}
        Proof: □(b→◆a) ∧ □(c→◆b)  ⊨  □(c→◆a)

    NOTE: This phase runs AFTER negation pruning so that the "support chain"
    constraints (T1, T2) have already survived all earlier pruning steps.
    This prevents over-pruning in cases where a support constraint itself gets
    removed later.

    NOTE: AlternateResponse/ChainResponse transitivity does NOT lift to
    AlternateResponse(a,c) — see counterexample ⟨a,b,a,c,b,c⟩.
    The entailment only lifts to Response(a,c) and Precedence(a,c).
    """
    print("\n  Applying transitivity-based entailment pruning (Tier 2)...")

    # Build lookup: (constraint_type, param_a, param_b) -> constraint
    index: Dict[Tuple[str, str, Optional[str]], DiscoveredConstraint] = {
        (c.constraint_type, c.param_a, c.param_b): c
        for c in constraints
    }

    to_remove: Set[str] = set()
    log_entries = []

    def _check_chain(family: List[str], entailed_type: str) -> None:
        for c1 in constraints:
            if c1.constraint_type not in family:
                continue
            if c1.param_b is None:
                continue
            a, b = c1.param_a, c1.param_b

            for c2 in constraints:
                if c2.constraint_type not in family:
                    continue
                if c2.param_b is None:
                    continue
                if c2.param_a != b:
                    continue
                c_val = c2.param_b
                if c_val is None or c_val == a:
                    continue

                # c1(a,b) ∧ c2(b,c) ⊨ entailed_type(a,c)
                key = (entailed_type, a, c_val)
                if key not in index:
                    continue
                entailed = index[key]
                if entailed.pattern_id in to_remove:
                    continue
                # Do not remove if the entailed constraint itself is a support member
                # (prevents removing a constraint that is also part of another chain)
                to_remove.add(entailed.pattern_id)
                log_entries.append(
                    f"    REMOVE {entailed.pattern_id}  "
                    f"(entailed: {c1.pattern_id} ∧ {c2.pattern_id}  ⊨  {entailed_type}({a},{c_val}))"
                )

    _check_chain(RESPONSE_TRANSITIVITY_FAMILY,   'Response')
    _check_chain(PRECEDENCE_TRANSITIVITY_FAMILY, 'Precedence')

    for entry in log_entries:
        print(entry)

    result = [c for c in constraints if c.pattern_id not in to_remove]
    print(f"    Removed {len(to_remove)} constraints via transitivity")
    print(f"    Remaining: {len(result)}")
    return result


# =============================================================================
# PHASE 6b — FULL FSA ENTAILMENT PRUNING (Tier 3)
# =============================================================================

# def apply_automaton_entailment_pruning(
#     constraints: List[DiscoveredConstraint],
#     alphabet: Set[str],
# ) -> List[DiscoveredConstraint]:
#     """Full language-inclusion entailment check via automata product (Tier 3).

#     For each constraint k ∈ K (Definition 13, Di Ciccio & Montali 2022):
#         1. Build A_k     = local FSA for k
#         2. Build Ā_k     = complement of A_k
#         3. Build A_{K\\{k}} = synchronous product of all other FSAs
#         4. Compute intersection: A_{K\\{k}} ⊗ Ā_k
#         5. If L(A_{K\\{k}} ⊗ Ā_k) = ∅ → every model-trace of K\\{k} satisfies k
#            → k is redundant → remove k

#     This check is complete (subsumes Tiers 1 and 2) but has worst-case complexity
#     exponential in |K| due to the product construction.  It is skipped when the
#     remaining constraint count exceeds CONFIG['fsa_entailment_max_constraints'].

#     Handles Example 14 (Di Ciccio & Montali 2022) and multi-constraint entailments
#     that no pairwise rule can detect.
#     """
#     print("\n  Applying FSA entailment pruning (Tier 3: full language-inclusion)...")

#     if not CONFIG['enable_fsa_entailment']:
#         print("    Skipped (enable_fsa_entailment = False)")
#         return constraints

#     if len(constraints) > CONFIG['fsa_entailment_max_constraints']:
#         print(
#             f"    Skipped: {len(constraints)} constraints > "
#             f"limit {CONFIG['fsa_entailment_max_constraints']} "
#             f"(set fsa_entailment_max_constraints higher to enable)"
#         )
#         return constraints

#     print(f"    Building FSAs for {len(constraints)} constraints over |Σ| = {len(alphabet)}…")

#     # Build local FSA for each constraint; skip on unknown template
#     fsa_map: Dict[str, FSA] = {}
#     for c in constraints:
#         try:
#             fsa_map[c.pattern_id] = _make_fsa(c.constraint_type, c.param_a, c.param_b, alphabet)
#         except ValueError as e:
#             print(f"    Warning: {e} — skipping {c.pattern_id}")

#     to_remove: Set[str] = set()
#     log_entries = []

#     surviving_ids = [c.pattern_id for c in constraints if c.pattern_id in fsa_map]

#     for k in constraints:
#         if k.pattern_id not in fsa_map:
#             continue
#         if k.pattern_id in to_remove:
#             continue

#         # Product of all OTHER surviving constraints' FSAs
#         others = [
#             pid for pid in surviving_ids
#             if pid != k.pattern_id and pid not in to_remove
#         ]
#         if not others:
#             continue

#         a_others = fsa_map[others[0]]
#         for pid in others[1:]:
#             a_others = a_others.product(fsa_map[pid])

#         # Complement of k's FSA
#         a_k_bar = fsa_map[k.pattern_id].complement()

#         # Intersection
#         intersection = a_others.product(a_k_bar)

#         if intersection.is_empty():
#             to_remove.add(k.pattern_id)
#             log_entries.append(
#                 f"    REMOVE {k.pattern_id}  "
#                 f"(entailed by K\\{{k}}: L(A_{{K\\{{k}}}} ⊗ Ā_k) = ∅)"
#             )

#     for entry in log_entries:
#         print(entry)

#     result = [c for c in constraints if c.pattern_id not in to_remove]
#     print(f"    Removed {len(to_remove)} constraints via FSA entailment")
#     print(f"    Remaining: {len(result)}")
#     return result

def apply_pairwise_entailment_pruning(
    constraints: List[DiscoveredConstraint],
) -> List[DiscoveredConstraint]:
    """Tier 2.5: Static pairwise cross-family entailment pruning.

    O(|K|) runtime — zero automata construction. Each activity pair requires
    at most 7 dict lookups against the offline-verified catalogue.
    Scientific basis: Di Ciccio & Montali (2022), §4.2, language inclusion.
    """
    print("\n  Applying pairwise cross-family entailment pruning (Tier 2.5)...")
    by_pair: Dict[Tuple, Dict[str, DiscoveredConstraint]] = defaultdict(dict)
    for c in constraints:
        by_pair[(c.param_a, c.param_b)][c.constraint_type] = c

    to_remove: Set[str] = set()
    log_entries: List[str] = []
    n_warnings = 0

    for (pa, pb), type_map in by_pair.items():
        if pb is None:
            continue
        present = frozenset(type_map.keys())
        # Same-pair entailment
        for premise_pair, entailed_types in PAIRWISE_ENTAILMENTS_SAME.items():
            if not premise_pair.issubset(present):
                continue
            t1, t2 = tuple(premise_pair)
            for t3 in entailed_types:
                if t3 not in type_map or type_map[t3].pattern_id in to_remove:
                    continue
                to_remove.add(type_map[t3].pattern_id)
                log_entries.append(
                    f"    REMOVE {type_map[t3].pattern_id}\n"
                    f"      (entailed: {t1}({pa},{pb}) ∧ {t2}({pa},{pb}) ⊨ {t3}({pa},{pb}))"
                )
        # Cross-pair mutual collapse warning
        reverse_map = by_pair.get((pb, pa), {})
        if reverse_map:
            for fname, ftypes in MUTUAL_COLLAPSE_FAMILIES.items():
                ab_hit = [t for t in ftypes if t in type_map]
                ba_hit = [t for t in ftypes if t in reverse_map]
                if ab_hit and ba_hit:
                    n_warnings += 1
                    log_entries.append(
                        f"    WARNING mutual-{fname}: {ab_hit[0]}({pa},{pb}) ∧ "
                        f"{ba_hit[0]}({pb},{pa}) → L={{ε}} (logical inconsistency)"
                    )
                    break

    for entry in log_entries:
        print(entry)
    if n_warnings:
        print(f"    ⚠  {n_warnings} activity pairs with mutually collapsing constraints")
    result = [c for c in constraints if c.pattern_id not in to_remove]
    print(f"    Removed {len(to_remove)} constraints via pairwise entailment")
    print(f"    Remaining: {len(result)}")
    return result


def apply_automaton_entailment_pruning_local(
    constraints: List[DiscoveredConstraint],
    alphabet: Set[str],
) -> List[DiscoveredConstraint]:
    """Local-neighbourhood FSA entailment pruning (Tier 3 — scalable variant).

    For each constraint k(a, b), restricts the 'others' product to constraints
    that share at least one activity with k (param_a or param_b).

    Scientific basis: A constraint on a disjoint activity set is insensitive to
    the positions of a and b in any trace — its FSA maps both a and b to 'other'
    internally, so it cannot distinguish between k being satisfied vs. violated.
    Therefore the complement intersection L(A_{K\{k}} ⊗ Ā_k) can never be empty
    by virtue of those disjoint constraints alone.  Only the local neighbourhood
    (constraints mentioning a or b) can generate the entailment.

    Soundness: identical to the cluster proof — applied at pair granularity.
    Reference: Di Ciccio & Montali (2022), §4.2, Example 14.
    """
    print("\n  Applying local-neighbourhood FSA entailment pruning (Tier 3: local)...")

    if not CONFIG['enable_fsa_entailment']:
        print("    Skipped (enable_fsa_entailment = False)")
        return constraints

    # ------------------------------------------------------------------
    # Step 1 — Build activity → pattern_id index
    # ------------------------------------------------------------------
    act_to_pids: Dict[str, Set[str]] = defaultdict(set)
    for c in constraints:
        act_to_pids[c.param_a].add(c.pattern_id)
        if c.param_b is not None:
            act_to_pids[c.param_b].add(c.pattern_id)

    # ------------------------------------------------------------------
    # Step 2 — Build FSA map once (reused for all k)
    # ------------------------------------------------------------------
    fsa_map: Dict[str, FSA] = {}
    for c in constraints:
        try:
            fsa_map[c.pattern_id] = _make_fsa(
                c.constraint_type, c.param_a, c.param_b, alphabet
            )
        except ValueError as e:
            print(f"    Warning: {e} — skipping {c.pattern_id}")

    # ------------------------------------------------------------------
    # Step 3 — Per-constraint local entailment check
    # ------------------------------------------------------------------
    global_to_remove: Set[str] = set()
    n_skipped = 0
    log_entries = []

    for k in constraints:
        if k.pattern_id not in fsa_map:
            continue
        if k.pattern_id in global_to_remove:
            continue

        # Local neighbourhood: constraints sharing ≥1 activity with k
        neighbour_pids: Set[str] = act_to_pids[k.param_a].copy()
        if k.param_b is not None:
            neighbour_pids |= act_to_pids[k.param_b]
        neighbour_pids.discard(k.pattern_id)

        others = [
            pid for pid in neighbour_pids
            if pid in fsa_map and pid not in global_to_remove
        ]

        if not others:
            continue

        # Per-neighbourhood guard
        if len(others) > CONFIG['fsa_entailment_max_constraints']:
            n_skipped += 1
            continue

        # Build A_{K\{k}} over the LOCAL neighbourhood only
        a_others = fsa_map[others[0]]
        for pid in others[1:]:
            a_others = a_others.product(fsa_map[pid])

        # Complement of k's FSA
        a_k_bar = fsa_map[k.pattern_id].complement()

        # Intersection: empty iff k is entailed by its neighbourhood
        intersection = a_others.product(a_k_bar)

        if intersection.is_empty():
            global_to_remove.add(k.pattern_id)
            log_entries.append(
                f"    REMOVE {k.pattern_id}  "
                f"(entailed by local neighbourhood: L(A_{{K\\{{k}}}} ⊗ Ā_k) = ∅)"
            )

    for entry in log_entries:
        print(entry)

    if n_skipped:
        print(f"    Skipped {n_skipped} constraints (neighbourhood exceeded per-pair limit)")

    result = [c for c in constraints if c.pattern_id not in global_to_remove]
    print(f"    Removed {len(global_to_remove)} constraints via local FSA entailment")
    print(f"    Remaining: {len(result)}")
    return result


def discover_declare_specification(
    cases: List[CaseTrace],
    frequent_activities: List[str],
    class_label: str = "",
) -> List[DiscoveredConstraint]:
    """
    Main discovery pipeline following Algorithm 1 (Di Ciccio & Montali 2022)
    extended with FSA-based entailment pruning (Section 4.2).

    Designed to run on any sublog (full L, L+, or L−).  The class_label
    parameter is used exclusively for console output clarity.

    Phase 1:  Generate candidate constraints (all template-activity combinations)
    Phase 2:  Compute interestingness measures for each candidate
    Phase 3:  Filter by thresholds
    Phase 4a: Same-parameter subsumption pruning
    Phase 4b: Cross-parameter subsumption pruning
    Phase 4c: Language-equivalence pruning (Tier 1 / E1, E2)
    Phase 5:  Negation pruning (extended with cross-family pairs)
    Phase 6a: Transitivity entailment pruning (Tier 2)
    Phase 6b: Full FSA entailment pruning (Tier 3 — local neighbourhood scope)
    """
    tag = f" [{class_label}]" if class_label else ""

    # Phase 1: Generate candidates
    print("\n" + "=" * 100)
    print(f"STEP 1: CANDIDATE GENERATION{tag}")
    print("=" * 100)
    candidates = generate_candidates(frequent_activities)
    print(f"  Total candidates: {len(candidates):,}")

    n_unary = sum(1 for c in candidates if c[2] is None)
    n_binary = len(candidates) - n_unary
    print(f"  Unary candidates: {n_unary:,}")
    print(f"  Binary candidates: {n_binary:,}")

    # Phase 2: Compute measures
    print("\n" + "=" * 100)
    print(f"STEP 2: MEASURE COMPUTATION{tag}")
    print("=" * 100)
    print(f"  Computing measures for {len(candidates):,} candidates on {len(cases):,} traces...")

    discovered: List[DiscoveredConstraint] = []
    for ct, pa, pb in tqdm(candidates, desc=f"  Computing measures{tag}"):
        m = compute_measures(ct, pa, pb, cases)
        pid = f"{ct}({pa}" + (f", {pb})" if pb else ")")
        discovered.append(DiscoveredConstraint(
            constraint_type=ct,
            param_a=pa,
            param_b=pb,
            measures=m,
            pattern_id=pid,
        ))

    print(f"  Computed measures for {len(discovered):,} constraints")

    # Phase 3: Threshold filtering
    print("\n" + "=" * 100)
    print(f"STEP 3: THRESHOLD FILTERING{tag}")
    print("=" * 100)
    print(f"  Thresholds:")
    print(f"    event_confidence >= {CONFIG['min_event_confidence_hard']}")
    print(f"    trace_confidence >= {CONFIG['min_trace_confidence_hard']}")

    before = len(discovered)
    discovered = [c for c in discovered if passes_threshold(c.measures)]
    print(f"  Before: {before:,} -> After: {len(discovered):,}")
    print(f"  Removed: {before - len(discovered):,}")

    # Phase 4a: Same-parameter subsumption pruning
    print("\n" + "=" * 100)
    print(f"STEP 4a: SAME-PARAMETER SUBSUMPTION PRUNING{tag}")
    print("=" * 100)
    discovered = apply_subsumption_pruning(discovered)

    # Phase 4b: Cross-parameter subsumption pruning
    print("\n" + "=" * 100)
    print(f"STEP 4b: CROSS-PARAMETER SUBSUMPTION PRUNING{tag}")
    print("=" * 100)
    print("  Scientific basis: Precedence(a,b) ⊑ RespondedExistence(b,a)")
    print("  (Di Ciccio & Montali 2022, Figure 8, Table 2)")
    discovered = apply_cross_param_subsumption_pruning(discovered)

    # Phase 4c: Language-equivalence pruning (Tier 1 / E1, E2)
    print("\n" + "=" * 100)
    print(f"STEP 4c: LANGUAGE-EQUIVALENCE PRUNING (Tier 1){tag}")
    print("=" * 100)
    print("  Scientific basis: FSA isomorphism — Di Ciccio & Montali (2022), Table 2")
    discovered = apply_language_equivalence_pruning(discovered)

    # Phase 5: Negation pruning (extended with cross-family pairs via E1/E2)
    print("\n" + "=" * 100)
    print(f"STEP 5: NEGATION PRUNING (extended with cross-family pairs){tag}")
    print("=" * 100)
    discovered = apply_negation_pruning(discovered)

    # Phase 6a: Transitivity entailment pruning (Tier 2)
    # Runs AFTER negation pruning so the support chain is stable.
    print("\n" + "=" * 100)
    print(f"STEP 6a: TRANSITIVITY ENTAILMENT PRUNING (Tier 2){tag}")
    print("=" * 100)
    print("  Scientific basis: LTLf transitivity — Di Ciccio & Montali (2022), Table 2")
    discovered = apply_transitivity_pruning(discovered)

    # Phase 6.5: Pairwise cross-family entailment pruning (Tier 2.5)
    print("\n" + "=" * 100)
    print(f"STEP 6.5: PAIRWISE CROSS-FAMILY ENTAILMENT (Tier 2.5){tag}")
    print("=" * 100)
    print("  Scientific basis: Offline FSA catalogue — Di Ciccio & Montali (2022), §4.2")
    discovered = apply_pairwise_entailment_pruning(discovered)

    # Phase 6b: Full FSA entailment pruning (Tier 3 — local neighbourhood)
    print("\n" + "=" * 100)
    print(f"STEP 6b: FULL FSA ENTAILMENT PRUNING (Tier 3){tag}")
    print("=" * 100)
    print("  Scientific basis: Language inclusion — Di Ciccio & Montali (2022), §4.2, Example 14")
    alphabet_set: Set[str] = set(frequent_activities)
    discovered = apply_automaton_entailment_pruning_local(discovered, alphabet_set)

    return discovered


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def write_decl_file(
    discovered: List[DiscoveredConstraint],
    output_path: str,
    header_comment: str = "",
) -> None:
    """
    Write discovered constraints to a standard .decl file.

    Format follows the MINERful / Declare4Py / ProM convention:

        // <header_comment>   ← optional provenance comment
        activity <name>
        ...
        <blank line>
        <ConstraintType>[<param_a>]             # unary
        <ConstraintType>[<param_a>, <param_b>]  # binary
        ...

    Constraints are grouped by type in canonical order and sorted
    alphabetically within each group.

    Reference: Di Ciccio & Montali (2022); MINERful tool format.
    """
    activity_set: set = set()
    for c in discovered:
        activity_set.add(c.param_a)
        if c.param_b is not None:
            activity_set.add(c.param_b)

    sorted_activities = sorted(activity_set)
    lines: List[str] = []

    # Provenance header
    if header_comment:
        for hline in header_comment.strip().splitlines():
            lines.append(f"// {hline}")
        lines.append("")

    # Activity declarations
    for act in sorted_activities:
        lines.append(f"activity {act}")

    lines.append("")  # blank line separating declarations from constraints

    # Constraints grouped by type in canonical order
    for ctype in ALL_CONSTRAINT_TYPES:
        group = [c for c in discovered if c.constraint_type == ctype]
        if not group:
            continue
        group.sort(key=lambda c: (c.param_a, c.param_b or ""))
        for c in group:
            if c.param_b is None:
                lines.append(f"{c.constraint_type}[{c.param_a}]")
            else:
                lines.append(f"{c.constraint_type}[{c.param_a}, {c.param_b}]")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved: {output_path}")


def _constraint_entries(discovered: List[DiscoveredConstraint]) -> List[Dict]:
    """Serialise a list of DiscoveredConstraint objects to JSON-ready dicts."""
    entries = []
    for c in discovered:
        entries.append({
            'pattern_id': c.pattern_id,
            'constraint_type': c.constraint_type,
            'param_a': c.param_a,
            'param_b': c.param_b,
            'measures': {
                'event_based': {
                    'confidence': round(c.measures.event_confidence, 6),
                    'coverage':   round(c.measures.event_coverage,   6),
                    'support':    round(c.measures.event_support,    6),
                },
                'trace_based': {
                    'confidence': round(c.measures.trace_confidence, 6),
                    'coverage':   round(c.measures.trace_coverage,   6),
                    'support':    round(c.measures.trace_support,    6),
                },
            },
            'raw_counts': {
                'events_activated_and_satisfied': c.measures.n_events_activated_and_satisfied,
                'events_activated':               c.measures.n_events_activated,
                'events_total':                   c.measures.n_events_total,
                'traces_activated_and_satisfied': c.measures.n_traces_activated_and_satisfied,
                'traces_activated':               c.measures.n_traces_activated,
                'traces_total':                   c.measures.n_traces_total,
            },
        })
    return entries


def _sort_constraints(discovered: List[DiscoveredConstraint]) -> List[DiscoveredConstraint]:
    """Sort by trace confidence desc, event confidence desc, trace support desc."""
    return sorted(
        discovered,
        key=lambda c: (
            c.measures.trace_confidence,
            c.measures.event_confidence,
            c.measures.trace_support,
        ),
        reverse=True,
    )


def _print_top(discovered: List[DiscoveredConstraint], label: str, n: int = 20) -> None:
    """Pretty-print top-n constraints for one class."""
    print(f"\n  TOP {n} CONSTRAINTS — {label} (by trace confidence):")
    print(f"  {'Pattern':<65} {'T-Conf':>7} {'T-Cov':>7} {'T-Supp':>7} {'E-Conf':>7} {'E-Supp':>7}")
    print(f"  {'-'*65} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for c in discovered[:n]:
        m = c.measures
        print(
            f"  {c.pattern_id:<65} "
            f"{m.trace_confidence:7.4f} {m.trace_coverage:7.4f} "
            f"{m.trace_support:7.4f} {m.event_confidence:7.4f} {m.event_support:7.4f}"
        )


def generate_output(
    discovered_pos: List[DiscoveredConstraint],
    discovered_neg: List[DiscoveredConstraint],
    cases_pos: List[CaseTrace],
    cases_neg: List[CaseTrace],
    stats: Dict,
    freq_pos: List[str],
    freq_neg: List[str],
    timing: float,
) -> Dict:
    """Generate class-conditional JSON and two .decl files.

    Outputs
    -------
    phase0_declare_specification_Lpos.decl  — L+ (Not-Accepted / Deviant) specification
    phase0_declare_specification_Lneg.decl  — L− (Accepted / Normal) specification
    phase0_declare_specification_CC.json    — enriched output:
        • per-class constraint tables with full MINERful measures
        • set-algebra summary:
            shared       = pattern_ids present in BOTH specifications
            Lpos_only    = pattern_ids found only in L+
            Lneg_only    = pattern_ids found only in L−
          These sets characterise what distinguishes Not-Accepted from
          Accepted process behaviour.
    """
    print("\n" + "=" * 100)
    print("GENERATING CLASS-CONDITIONAL OUTPUT")
    print("=" * 100)

    # Sort each specification for deterministic, ranked output
    discovered_pos = _sort_constraints(discovered_pos)
    discovered_neg = _sort_constraints(discovered_neg)

    by_type_pos = Counter(c.constraint_type for c in discovered_pos)
    by_type_neg = Counter(c.constraint_type for c in discovered_neg)

    # --- Set-algebra summary -------------------------------------------
    ids_pos: Set[str] = {c.pattern_id for c in discovered_pos}
    ids_neg: Set[str] = {c.pattern_id for c in discovered_neg}
    shared_ids      = ids_pos & ids_neg
    lpos_only_ids   = ids_pos - ids_neg
    lneg_only_ids   = ids_neg - ids_pos

    lpos_only    = [c for c in discovered_pos if c.pattern_id in lpos_only_ids]
    lneg_only    = [c for c in discovered_neg if c.pattern_id in lneg_only_ids]

    print(f"\n  L+ (Not-Accepted / Deviant) specification: {len(discovered_pos):,} constraints")
    for ct in ALL_CONSTRAINT_TYPES:
        if ct in by_type_pos:
            print(f"    {ct}: {by_type_pos[ct]}")

    print(f"\n  L− (Accepted / Normal) specification: {len(discovered_neg):,} constraints")
    for ct in ALL_CONSTRAINT_TYPES:
        if ct in by_type_neg:
            print(f"    {ct}: {by_type_neg[ct]}")

    print(f"\n  Set-algebra summary:")
    print(f"    Shared constraints (L+ ∩ L−):    {len(shared_ids):,}")
    print(f"    L+-exclusive constraints:         {len(lpos_only_ids):,}")
    print(f"    L−-exclusive constraints:         {len(lneg_only_ids):,}")

    # --- Build JSON -------------------------------------------------------
    output = {
        'framework': 'Phase 0 — Class-Conditional DECLARE Specification Discovery',
        'version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'description': (
            'Exhaustive DECLARE pattern enumeration with MINERful-style '
            'activation/target measures, subsumption pruning, language-equivalence '
            'pruning (Tier 1: E1/E2 FSA isomorphisms), negation pruning (extended '
            'with cross-family pairs), transitivity entailment pruning (Tier 2), '
            'and full FSA language-inclusion entailment pruning (Tier 3), applied '
            'independently to L+ (Not-Accepted / Deviant) and L− (Accepted / Normal) sublogs. '
            'Constraint semantics follow Di Ciccio & Montali (2022) Table 2. '
            'Measure definitions follow Iacometta & Di Ciccio (2025) Table 2.'
        ),
        'references': [
            'Di Ciccio & Montali (2022): Declarative Process Specifications: '
            'Reasoning, Discovery, Monitoring',
            'Iacometta & Di Ciccio (2025): Declarative Process Mining with MINERful, Reloaded',
        ],
        'configuration': {
            'min_event_confidence': CONFIG['min_event_confidence_hard'],
            'min_trace_confidence': CONFIG['min_trace_confidence_hard'],
            'min_event_support':    CONFIG['min_event_support'],
            'min_trace_support':    CONFIG['min_trace_support'],
            'min_activity_frequency': CONFIG['min_activity_frequency'],
        },
        'dataset': {
            'total_cases':    stats['total_cases'],
            'class_Lpos_cases': stats['class_1'],   # Not-Accepted (Deviant)
            'class_Lneg_cases': stats['class_0'],   # Accepted (Normal)
            'class_Lpos_freq_activities': len(freq_pos),
            'class_Lneg_freq_activities': len(freq_neg),
            'freq_activities_Lpos': freq_pos,
            'freq_activities_Lneg': freq_neg,
        },
        'execution_time_seconds': round(timing, 2),
        'constraint_types_info': {
            'unary':           UNARY_CONSTRAINTS,
            'binary_positive': BINARY_POSITIVE_CONSTRAINTS,
            'binary_negative': BINARY_NEGATIVE_CONSTRAINTS,
        },
        'subsumption_hierarchy': {k: v for k, v in SUBSUMPTION_CHAINS.items()},
        'negation_pairs': [list(p) for p in NEGATION_PAIRS],
        'language_equivalences': [list(p) for p in LANGUAGE_EQUIVALENCES],
        'response_transitivity_family': RESPONSE_TRANSITIVITY_FAMILY,
        'precedence_transitivity_family': PRECEDENCE_TRANSITIVITY_FAMILY,
        'fsa_entailment_enabled': CONFIG['enable_fsa_entailment'],
        'fsa_entailment_max_constraints': CONFIG['fsa_entailment_max_constraints'],

        # --- L+ (Not-Accepted / Deviant) specification ---
        'Lpos': {
            'label': 'L+ (Not-Accepted / Deviant)',
            'n_traces': len(cases_pos),
            'n_constraints': len(discovered_pos),
            'by_type': dict(by_type_pos),
            'constraints': _constraint_entries(discovered_pos),
        },

        # --- L− (Accepted / Normal) specification ---
        'Lneg': {
            'label': 'L− (Accepted / Normal)',
            'n_traces': len(cases_neg),
            'n_constraints': len(discovered_neg),
            'by_type': dict(by_type_neg),
            'constraints': _constraint_entries(discovered_neg),
        },

        # --- Set-algebra summary ---
        'set_algebra': {
            'description': (
                'Class-discriminative analysis. '
                'shared: constraints in both L+ and L−. '
                'Lpos_only: constraints exclusively characterising Not-Accepted (Deviant) behaviour. '
                'Lneg_only: constraints exclusively characterising Accepted (Normal) behaviour.'
            ),
            'n_shared':    len(shared_ids),
            'n_Lpos_only': len(lpos_only_ids),
            'n_Lneg_only': len(lneg_only_ids),
            'shared':    sorted(shared_ids),
            'Lpos_only': [c.pattern_id for c in lpos_only],
            'Lneg_only': [c.pattern_id for c in lneg_only],
        },
    }

    # --- Save JSON --------------------------------------------------------
    json_path = os.path.join(OUTPUT_DIR, 'phase0_declare_specification_CC.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    # --- Save .decl files -------------------------------------------------
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    decl_pos_path = os.path.join(OUTPUT_DIR, 'phase0_declare_specification_Lpos.decl')
    write_decl_file(
        discovered_pos,
        decl_pos_path,
        header_comment=(
            f"Phase 0 Class-Conditional DECLARE Specification — L+ (Not-Accepted / Deviant)\n"
            f"Generated: {ts}\n"
            f"Traces: {len(cases_pos)}  |  "
            f"Constraints: {len(discovered_pos)}  |  "
            f"Activities: {len(freq_pos)}\n"
            f"Thresholds: event_conf>={CONFIG['min_event_confidence_hard']}, "
            f"trace_conf>={CONFIG['min_trace_confidence_hard']}"
        ),
    )

    decl_neg_path = os.path.join(OUTPUT_DIR, 'phase0_declare_specification_Lneg.decl')
    write_decl_file(
        discovered_neg,
        decl_neg_path,
        header_comment=(
            f"Phase 0 Class-Conditional DECLARE Specification — L− (Accepted / Normal)\n"
            f"Generated: {ts}\n"
            f"Traces: {len(cases_neg)}  |  "
            f"Constraints: {len(discovered_neg)}  |  "
            f"Activities: {len(freq_neg)}\n"
            f"Thresholds: event_conf>={CONFIG['min_event_confidence_hard']}, "
            f"trace_conf>={CONFIG['min_trace_confidence_hard']}"
        ),
    )

    # --- Console summaries ------------------------------------------------
    _print_top(discovered_pos, "L+ (Not-Accepted / Deviant)")
    _print_top(discovered_neg, "L− (Accepted / Normal)")

    if lpos_only:
        print(f"\n  L+-EXCLUSIVE constraints (Not-Accepted-only behaviour, {len(lpos_only)}):")
        for c in lpos_only[:15]:
            m = c.measures
            print(f"    {c.pattern_id:<65} tconf={m.trace_confidence:.4f}")

    if lneg_only:
        print(f"\n  L−-EXCLUSIVE constraints (Accepted-only behaviour, {len(lneg_only)}):")
        for c in lneg_only[:15]:
            m = c.measures
            print(f"    {c.pattern_id:<65} tconf={m.trace_confidence:.4f}")

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    start = time.time()

    # -----------------------------------------------------------------------
    # 1. Load and partition the event log into L+  and  L−
    # -----------------------------------------------------------------------
    _, cases_pos, cases_neg, stats = load_event_log(INPUT_FILE)

    # -----------------------------------------------------------------------
    # 2. Compute class-conditional activity alphabets
    #
    #    Using class-specific alphabets is scientifically necessary:
    #    an activity absent from L+ has zero occurrences there, making
    #    every constraint involving it either vacuously satisfied or trivially
    #    false — neither contributes discriminative information.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ACTIVITY ALPHABET COMPUTATION (class-conditional)")
    print("=" * 100)
    freq_pos = get_frequent_activities(
        cases_pos, CONFIG['min_activity_frequency'], class_label="L+ Not-Accepted"
    )
    freq_neg = get_frequent_activities(
        cases_neg, CONFIG['min_activity_frequency'], class_label="L− Accepted"
    )

    # -----------------------------------------------------------------------
    # 3. Run discovery pipeline independently on L+ and L−
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DISCOVERY PIPELINE — L+ (Not-Accepted / Deviant)")
    print("=" * 100)
    discovered_pos = discover_declare_specification(
        cases_pos, freq_pos, class_label="L+ Not-Accepted"
    )

    print("\n" + "=" * 100)
    print("DISCOVERY PIPELINE — L− (Accepted / Normal)")
    print("=" * 100)
    discovered_neg = discover_declare_specification(
        cases_neg, freq_neg, class_label="L− Accepted"
    )

    # -----------------------------------------------------------------------
    # 4. Generate class-conditional output
    # -----------------------------------------------------------------------
    timing = time.time() - start
    output = generate_output(
        discovered_pos, discovered_neg,
        cases_pos, cases_neg,
        stats,
        freq_pos, freq_neg,
        timing,
    )

    print(f"\n{'=' * 100}")
    print(f"PHASE 0 (CLASS-CONDITIONAL) COMPLETE")
    print(f"{'=' * 100}")
    print(f"  Total time: {timing:.1f}s")
    print(f"  L+ (Not-Accepted / Deviant) constraints: {len(discovered_pos)}")
    print(f"  L− (Accepted / Normal)      constraints: {len(discovered_neg)}")
    print(f"  Outputs written to: {OUTPUT_DIR}/")
    print(f"    phase0_declare_specification_Lpos.decl")
    print(f"    phase0_declare_specification_Lneg.decl")
    print(f"    phase0_declare_specification_CC.json")

    return output


if __name__ == "__main__":
    main()