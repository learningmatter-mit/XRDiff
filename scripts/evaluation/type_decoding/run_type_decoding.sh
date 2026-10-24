#!/bin/bash
# Usage: sbatch scripts/evaluation/type_decoding/run_type_decoding.sh <run_dir> [label ...]
# label = eval_diff_<label>.pt, e.g. test, rruff_pv, rruff_pv_n_evals_10 (default: test)
#SBATCH --job-name=type_decode
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=0-12:00:00
#SBATCH --output=out/type_decode_%j.out
#SBATCH --error=out/type_decode_%j.err

source /etc/profile
source $HOME/.bashrc
export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$PWD}   # submit from the repo root
cd $PROJECT_ROOT

RUN=$1
shift
LABELS=${@:-test}
for L in $LABELS; do
    if [ ! -f "$RUN/type_logits_$L.pt" ]; then
        python scripts/evaluation/type_decoding/recover_type_logits.py --root_path "$RUN" --label $L
    fi
    python scripts/evaluation/type_decoding/decode_and_match.py --root_path "$RUN" --label $L --num_cpus 40
done
