#!/bin/bash
#SBATCH --job-name=slowmo_gota_100f_x5
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
# Usa el clip gota_mejorado_s2s4_1080p YA subido, limitado a 100 frames.

echo "=== TRAIN 100 frames 1080p $(date) ==="
python scripts/train.py --config configs/slowmo/slowmo_gota_mej_100f.json \
    --nombre-experimento slowmo_gota_mej_100f

echo "=== SLOW-MO x5 $(date) ==="
python scripts/render_slowmo.py \
    --checkpoint outputs/slowmo_gota_mej_100f/checkpoints/checkpoint_final.pt \
    --salida outputs/slowmo_gota_mej_100f/slowmo_x5 \
    --factor 5 --crear_video --fps 30

echo "=== LISTO $(date) ==="
