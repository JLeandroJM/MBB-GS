#!/bin/bash
#SBATCH --job-name=fuego_musical
#SBATCH --partition=data-science
#SBATCH --gres=shard:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

EXP=outputs/musical_20s_1600ep
CKPT=$EXP/checkpoints/checkpoint_final.pt
CKPT_FUEGO=$EXP/checkpoints/checkpoint_final_fuego.pt

# 1) Inyectar el fuego (CPU, rapido) -> genera checkpoint_final_fuego.pt en Khipu
echo "=== INYECTANDO FUEGO $(date) ==="
python scripts/inyectar_efecto_fuego.py --checkpoint "$CKPT"

# 2) Rasterizar el clip con la fogata compuesta (GPU) + MP4
#    Salida aislada en $EXP/fuego/ para no pisar el video original del clip.
echo "=== RASTERIZANDO $(date) ==="
python scripts/regenerar_clip_desde_checkpoint_streaming.py \
    --checkpoint "$CKPT_FUEGO" \
    --salida "$EXP/fuego/frames" \
    --device cuda \
    --crear_video --fps 24

echo "=== END $(date) ==="
