#!/bin/bash
#SBATCH --job-name=slowmo_gota_abl
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
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
# NOTA: si cae en una Tesla y el rasterizador da "no kernel image", recancelar
# y reintentar para que toque A6000/A100/MIG.

# Las 9 variantes de ablacion (termometro holdout-x2). FINAL se corre aparte.
CONFIGS=(
  slowmo_gota_A0_control
  slowmo_gota_A1_smoothasym_b1e7
  slowmo_gota_A2_smoothasym_b1e6
  slowmo_gota_A3_accel_1e7
  slowmo_gota_A4_accel_1e6
  slowmo_gota_A5_combo
  slowmo_gota_A6_gradomu20
  slowmo_gota_A7_gradomu40
  slowmo_gota_A8_g16k
)

for nombre in "${CONFIGS[@]}"; do
  echo "=== START $nombre $(date) ==="
  python scripts/train.py --config configs/slowmo/${nombre}.json --nombre-experimento ${nombre}
  python scripts/analizar_holdout.py --exp outputs/${nombre} || true
  echo "=== END $nombre $(date) ==="
done

echo "=== RANKING FINAL ==="
python scripts/ranking_slowmo.py --exps outputs/slowmo_gota_* \
    --baseline outputs/baseline_lineal/gota.csv || true
