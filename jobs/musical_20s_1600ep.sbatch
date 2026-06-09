#!/bin/bash
#SBATCH --job-name=musical_20s_1600ep
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

echo "=== INFO GPU ==="
nvidia-smi
echo "=== INFO PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# === extraer el clip 720p (nativo 720x1080, sin deformar) si aun no existe ===
# 20 s a 24 fps = 480 frames. Idempotente: si ya hay PNGs no re-extrae (sin --forzar).
if [ ! -f data/clips/musical_20s/frame_0000.png ]; then
  echo "=== extrayendo clip musical_20s ==="
  python scripts/extraer_clips_720p.py \
      --video data/videos/musical.mp4 \
      --nombre_clip musical_20s \
      --inicio_seg 0 \
      --duracion_seg 20 \
      --H 720 --W 1080
else
  echo "=== clip musical_20s ya existe, no re-extraigo ==="
fi

echo "=== START $(date) ==="

python scripts/train.py \
    --config configs/musical_20s_1600ep.json \
    --nombre-experimento musical_20s_1600ep

echo "=== END $(date) ==="
