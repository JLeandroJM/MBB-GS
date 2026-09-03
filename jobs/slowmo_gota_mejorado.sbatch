#!/bin/bash
#SBATCH --job-name=slowmo_gota_mej
#SBATCH --partition=data-science,gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jose.machaca@utec.edu.pe

set -e
cd $HOME/MBB-GS

module load cuda/12.8
module load gnu12/12.4.0
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate mbb-gs

echo "=== INFO GPU ==="; nvidia-smi
# 1080p + 100k gaussianas, solo 120 frames (sec 2-4). Si OOM: baja n_gaussianas.

echo "=== TRAIN gota_mejorado sec2-4 1080p $(date) ==="
python scripts/train.py --config configs/slowmo/slowmo_gota_mejorado_s2s4.json \
    --nombre-experimento slowmo_gota_mejorado_s2s4

echo "=== SLOW-MO x3 $(date) ==="
python scripts/render_slowmo.py \
    --checkpoint outputs/slowmo_gota_mejorado_s2s4/checkpoints/checkpoint_final.pt \
    --salida outputs/slowmo_gota_mejorado_s2s4/slowmo \
    --factor 3 --crear_video --fps 30

echo "=== LISTO $(date) ==="
