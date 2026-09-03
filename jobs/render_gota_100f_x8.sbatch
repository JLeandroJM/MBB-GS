#!/bin/bash
#SBATCH --job-name=render_gota_100f_x8
#SBATCH --partition=data-science,gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
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

# Solo RENDER x8 del MISMO modelo (100 frames) que entrena el job de x5.
# No re-entrena: usa el checkpoint_final.pt que deja slowmo_gota_mej_100f.
CKPT=outputs/slowmo_gota_mej_100f/checkpoints/checkpoint_final.pt
if [ ! -f "$CKPT" ]; then
  echo "ERROR: no existe $CKPT (el job de x5 aun no termino de entrenar)"; exit 1
fi

echo "=== SLOW-MO x8 $(date) ==="
python scripts/render_slowmo.py \
    --checkpoint "$CKPT" \
    --salida outputs/slowmo_gota_mej_100f/slowmo_x8 \
    --factor 8 --crear_video --fps 30

echo "=== LISTO x8 $(date) ==="
