#!/bin/bash
#SBATCH --job-name=slowmo_gota_motion
#SBATCH --partition=data-science,gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
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

# barrido de beta (smoothness asimetrica) sobre gota_mejorado_s2s4_1080p, 1080p.
# Cada uno: entrena (motion + smooth asimetrica) y renderiza slow-mo x5.
CONFIGS=(gota_motion_b1e6 gota_motion_b5e6 gota_motion_b2e5)

for E in "${CONFIGS[@]}"; do
  echo "=== TRAIN $E $(date) ==="
  python scripts/train.py --config configs/slowmo/${E}.json --nombre-experimento ${E}

  echo "=== SLOW-MO x5 $E $(date) ==="
  python scripts/render_slowmo.py \
    --checkpoint outputs/${E}/checkpoints/checkpoint_final.pt \
    --salida outputs/${E}/slowmo_x5 \
    --factor 5 --crear_video --fps 30
  echo "=== DONE $E $(date) ==="
done

echo "=== TODO LISTO $(date) ==="
