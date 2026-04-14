#!/bin/bash
#SBATCH --job-name=rq1_Sepsis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=32:00:00
#SBATCH --output=logs/rq1_Sepsis_%j.out
#SBATCH --error=logs/rq1_Sepsis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Aabdesselam@unibz.it

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load python/3.11        # adjust to your cluster's module name

WORKDIR="$HOME/process_mining/evaluation"
cd "$WORKDIR" || { echo "Cannot cd to $WORKDIR"; exit 1; }

source venv/bin/activate

# CRITICAL: Prevent NumPy/BLAS from spawning per-worker thread pools.
# Without these, 8 loky workers × 8 BLAS threads = 64 threads on 8 cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# ── Logging header ───────────────────────────────────────────────────────
echo "============================================================"
echo "  JOB: $SLURM_JOB_ID — rq1_BPI17"
echo "  Node: $SLURMD_NODENAME"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Mem:  $SLURM_MEM_PER_NODE MB"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

mkdir -p logs RQ1_BPI17

# ── Dry-run smoke test (30 s) — comment out for production ──────────────
# python rq1_BPI17.py --dry-run --n-jobs 2
# if [ $? -ne 0 ]; then
#     echo "DRY-RUN FAILED — aborting production run"; exit 1
# fi

# ── Production run ───────────────────────────────────────────────────────
python rq1_BPI17.py \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --b-null 200

EXIT_CODE=$?
echo "============================================================"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S') | Exit: $EXIT_CODE"
echo "============================================================"

# ── Post-run: generate figures if run succeeded ──────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Generating figures..."
    python rq1_figures.py
    echo "  ✓ Figures saved to RQ1_BPI17/figures/"
else
    echo "  ✗ rq1_BPI17.py failed — skipping figures"
fi

exit $EXIT_CODE