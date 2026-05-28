# Significant DECLARE Process Deviance Mining
 
**Statistical Testing and False Discovery Rate Control for Process Deviance Mining**
 
Nour A. Abdesselam, Fabrizio Maria Maggi, Thomas Villgrattner, Angelika Peer, and Markus Zanker  
Free University of Bozen-Bolzano · GKN Powder Metallurgy 

---

## Overview
 
This repository contains the full implementation for the paper. The paper addresses a fundamental problem in process deviance mining: when many DECLARE constraints are tested simultaneously on the same event log, some will appear significant purely by chance—a *multiple-testing problem*. Without correction, the number of false discoveries grows with the size of the candidate space.
 
The framework controls the **False Discovery Rate (FDR)** across the full candidate space of DECLARE constraints. It does this by combining *dual-axis permutation testing* with *adaptive Storey–Gao FDR correction*. A constraint is only reported if it provides simultaneous evidence of:
 
- **R1 — Temporal regularity**: the observed fulfillment rate of the constraint exceeds what is expected under random reordering of activities within traces.
- **R2 — Discriminative signal**: the difference in fulfillment rate between the deviant and non-deviant class exceeds what is expected under random reassignment of class labels.

---

## Requirements
 
```
python >= 3.9
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
tqdm
joblib
```
 
Install dependencies:
 
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn tqdm joblib
```
 
---
 
## Reproducing the Results
 
All data paths in the scripts point to a local `Experiments data/` directory that must contain the event logs in CSV format. Update `INPUT_FILE` and `OUTPUT_DIR` at the top of each script to match your local setup before running.
 
**Step 1 — Run Phase 0 (candidate set construction):**
 
```bash
python Experiments/P0_DECSpec/p0-Production.py
python Experiments/P0_DECSpec/p0-Sepsis.py
```
 
This writes the class-conditional specifications to `Experiments/Results/DECspec_Production/` and `Experiments/Results/DECspec_Sepsis/`.
 
**Step 2 — Run Phase 1 (dual-axis testing and FDR control):**
 
```bash
python Experiments/P1_SDSM/p1_Production_hou.py
python Experiments/P1_SDSM/p1_Sepsis_hou.py
```
 
**Step 3 — Run the RQ1 evaluation (empirical FDR):**
 
```bash
python Experiments/Eval_RQ1/rq1_Production_parallel.py
python Experiments/Eval_RQ1/rq1_Sepsis_parallel.py
```
 
**Step 4 — Run the RQ2 evaluation (perturbation robustness):**
 
```bash
python Experiments/Eval_RQ2/rq2_Production_parallel_copy.py
python Experiments/Eval_RQ2/rq2_Sepsis_parallel_copy.py
```

---

## Acknowledgments
 
This work is supported by a scholarship funded by the European Union — Next Generation EU, Mission 4 Component 1 CUP I52B24000520005, and by GKN Sinter Metals.
Pre-computed results for all methods and both research questions are available in the corresponding result folders (`RQ1_*`, `RQ2_*`, `DRVA_*`, `DeclareMiner_*`, `DvM_BISE2025_*`).
 
---
