#!/bin/bash
#SBATCH --job-name=dvd_mult2_motion
#SBATCH --partition=gpu
#SBATCH --exclude=ag001
#SBATCH --nodelist=g002
#SBATCH --gres=shard:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"

# === clip dvd_20s: 720x1164 (aspecto nativo 2940:1818), 30fps, 20s = 600 frames ===
if [ ! -f data/clips/dvd_20s/frame_0000.png ]; then
  echo "=== extrayendo clip dvd_20s ==="
  python scripts/extraer_clips_720p.py \
      --video data/videos/dvd.mp4 \
      --nombre_clip dvd_20s \
      --inicio_seg 0 --duracion_seg 20 \
      --fps 30 --H 720 --W 1164
else
  echo "=== clip dvd_20s ya existe ==="
fi

echo "=== START $(date) ==="
python scripts/train.py \
    --config configs/dvd_20s_holdout_mult2_motion.json \
    --nombre-experimento dvd_20s_holdout_mult2_motion
echo "=== END $(date) ==="
