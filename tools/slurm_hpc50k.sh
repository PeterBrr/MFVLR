#!/bin/bash
#SBATCH --job-name=mfvlr50k
#SBATCH --output=mfvlr50k_%j.log
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1g.18gb:1

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
L=configs/experiments/hpc50k_dicefw2_region_language.json

python3 tools/run_experiment.py --configs $B --seeds 0 --output runs/hpc50k >> logs/baseline_seed0.log 2>&1
python3 tools/run_experiment.py --configs $L --seeds 0 --output runs/hpc50k >> logs/language_seed0.log 2>&1
python3 tools/run_experiment.py --configs $B --seeds 1 --output runs/hpc50k >> logs/baseline_seed1.log 2>&1
python3 tools/run_experiment.py --configs $L --seeds 1 --output runs/hpc50k >> logs/language_seed1.log 2>&1
python3 tools/run_experiment.py --configs $B --seeds 2 --output runs/hpc50k >> logs/baseline_seed2.log 2>&1
python3 tools/run_experiment.py --configs $L --seeds 2 --output runs/hpc50k >> logs/language_seed2.log 2>&1