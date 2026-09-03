#!/bin/bash
#SBATCH --job-name=mus_g80_gaps20
#SBATCH --partition=gpu
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

if [ ! -f data/clips/musical_20s/frame_0000.png ]; then
  python scripts/extraer_clips_720p.py \
      --video data/videos/musical.mp4 --nombre_clip musical_20s \
      --inicio_seg 0 --duracion_seg 20 --H 720 --W 1080
fi

echo "=== START $(date) ==="
python scripts/train.py --config configs/musical_20s_g80_gaps20_l1dssim.json --nombre-experimento musical_20s_g80_gaps20_l1dssim
echo "=== END $(date) ==="
