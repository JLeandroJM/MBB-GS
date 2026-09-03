#!/bin/bash
#SBATCH --job-name=slowmo_gota_FAST
#SBATCH --partition=data-science,gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

# 1) bajar el clip 1080p (ya subido) a 480x270 si no existe
if [ ! -f data/clips/video_final_gota_480x270/frame_0000.png ]; then
  echo "=== downscale a 480x270 ==="
  python scripts/downscale_clip.py \
    --src data/clips/video_final_gota_1080p \
    --dst data/clips/video_final_gota_480x270 --H 270 --W 480
fi

# 2) entrenar rapido (checkpoints intermedios cada 100 epochs)
echo "=== TRAIN FAST $(date) ==="
python scripts/train.py --config configs/slowmo/slowmo_final_gota_FAST.json \
    --nombre-experimento slowmo_final_gota_FAST

# 3) slow-mo x3
echo "=== SLOW-MO x3 $(date) ==="
python scripts/render_slowmo.py \
    --checkpoint outputs/slowmo_final_gota_FAST/checkpoints/checkpoint_final.pt \
    --salida outputs/slowmo_final_gota_FAST/slowmo \
    --factor 3 --crear_video --fps 30

echo "=== LISTO $(date) ==="
