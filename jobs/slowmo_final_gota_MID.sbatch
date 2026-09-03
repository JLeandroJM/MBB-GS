#!/bin/bash
#SBATCH --job-name=slowmo_gota_MID
#SBATCH --partition=data-science,gpu
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
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

# 1) bajar el clip 1080p a 960x540 si no existe
if [ ! -f data/clips/video_final_gota_960x540/frame_0000.png ]; then
  echo "=== downscale a 960x540 ==="
  python scripts/downscale_clip.py \
    --src data/clips/video_final_gota_1080p \
    --dst data/clips/video_final_gota_960x540 --H 540 --W 960
fi

echo "=== TRAIN MID $(date) ==="
python scripts/train.py --config configs/slowmo/slowmo_final_gota_MID.json \
    --nombre-experimento slowmo_final_gota_MID

echo "=== SLOW-MO x3 $(date) ==="
python scripts/render_slowmo.py \
    --checkpoint outputs/slowmo_final_gota_MID/checkpoints/checkpoint_final.pt \
    --salida outputs/slowmo_final_gota_MID/slowmo \
    --factor 3 --crear_video --fps 30

echo "=== LISTO $(date) ==="
