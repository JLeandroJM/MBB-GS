#!/bin/bash
#SBATCH --job-name=slowmo_final_gota
#SBATCH --partition=data-science,gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
# 1080p + 150k gaussianas. Si da OOM: baja n_gaussianas_inicial (p.ej. 100000)
# o cambia frames_en_gpu_uint8->false y frames_en_cpu->true en las configs.

# ---------- 1) PRODUCTO: entrena TODO -> slow-mo x3 ----------
echo "=== TRAIN FULL $(date) ==="
python scripts/train.py --config configs/slowmo/slowmo_final_gota_FULL.json \
    --nombre-experimento slowmo_final_gota_FULL

echo "=== SLOW-MO x3 $(date) ==="
python scripts/render_slowmo.py \
    --checkpoint outputs/slowmo_final_gota_FULL/checkpoints/checkpoint_final.pt \
    --salida outputs/slowmo_final_gota_FULL/slowmo \
    --factor 3 --crear_video --fps 30

# ---------- 2) MEDICION: termometro holdout-x2 ----------
echo "=== TRAIN TERM (termometro) $(date) ==="
python scripts/train.py --config configs/slowmo/slowmo_final_gota_TERM.json \
    --nombre-experimento slowmo_final_gota_TERM
python scripts/analizar_holdout.py --exp outputs/slowmo_final_gota_TERM || true

echo "=== TODO LISTO $(date) ==="
