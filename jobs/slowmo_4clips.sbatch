#!/bin/bash
#SBATCH --job-name=slowmo_4clips
#SBATCH --partition=gpu,data-science
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
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
# Si cae en una Tesla y el raster da "no kernel image", cancelar y reintentar
# (que toque A6000/A100/MIG en ag001/g00x).

# tag -> config base por clip
EXPS=(
  slowmo_gota_base
  slowmo_rampaA_base
  slowmo_rampaB_base
  slowmo_estr_base
)

for E in "${EXPS[@]}"; do
  echo "=== TRAIN $E $(date) ==="
  python scripts/train.py --config configs/slowmo/${E}.json --nombre-experimento ${E}

  echo "=== SLOW-MO x3 $E $(date) ==="
  python scripts/render_slowmo.py \
    --checkpoint outputs/${E}/checkpoints/checkpoint_final.pt \
    --salida outputs/${E}/slowmo \
    --factor 3 --crear_video --fps 30

  # analisis holdout (corre tambien aqui; CPU, barato)
  python scripts/analizar_holdout.py --exp outputs/${E} || true
  echo "=== DONE $E $(date) ==="
done

echo "=== TODO LISTO ==="
