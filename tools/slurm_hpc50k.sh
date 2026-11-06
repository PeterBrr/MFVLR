#!/bin/bash
#SBATCH --job-name=mfvlr50k
#SBATCH --output=mfvlr50k_%j.log
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:2g.35gb:1

source /datastore/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mfvlr
export PYTHONUNBUFFERED=1

cd /datastore/khanhnd/deepfake/MFVLR/
mkdir -p logs

# Pack train/val npz once (needs /datastore/$USER/deepfake/genface50k unzipped).
[ -f artifacts/genface50k/package_001/package_report.json ] || \
    python3 tools/pack_genface50k_npz.py --extracted /datastore/$USER/deepfake/genface50k \
        --output artifacts/genface50k/package_001 >> logs/pack.log 2>&1

B=configs/experiments/hpc50k_dicefw2.json
C=configs/experiments/hpc50k_dicefw2_textpatch_clip.json
R=configs/experiments/hpc50k_dicefw2_textpatch_random.json

python3 tools/run_experiment.py --configs $B --seeds 0 --output runs/hpc50k >> logs/baseline_seed0.log 2>&1
python3 tools/run_experiment.py --configs $C --seeds 0 --output runs/hpc50k >> logs/textpatch_clip_seed0.log 2>&1
python3 tools/run_experiment.py --configs $R --seeds 0 --output runs/hpc50k >> logs/textpatch_random_seed0.log 2>&1
