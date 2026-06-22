#!/bin/bash
#SBATCH --job-name=dvd_full_fast
#SBATCH --partition=gpu
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

# clip dvd_20s: 720x1164, 30fps, 20s = 600 frames
if [ ! -f data/clips/dvd_20s/frame_0000.png ]; then
  python scripts/extraer_clips_720p.py \
      --video data/videos/dvd.mp4 --nombre_clip dvd_20s \
      --inicio_seg 0 --duracion_seg 20 --fps 30 --H 720 --W 1164
fi

echo "=== START $(date) ==="
python scripts/train.py --config configs/dvd_20s_full_l1dssim_fast.json \
    --nombre-experimento dvd_20s_full_l1dssim_fast
echo "=== END $(date) ==="
