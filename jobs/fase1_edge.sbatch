#!/bin/bash
#SBATCH --job-name=fase1_edge
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jose.machaca@utec.edu.pe

set -e
cd $HOME/MBB-GS

# Modulos confirmados en Khipu (nombres reales del 'module avail').
module load cuda/12.8
module load gnu12/12.4.0
module load python3/3.11.11

# Activa el entorno conda creado en la guia de instalacion.
source $HOME/miniconda3/etc/profile.d/conda.sh || \
    source /opt/ohpc/pub/utils/miniconda/3.0/etc/profile.d/conda.sh
conda activate mbb-gs

echo "=== INFO GPU ==="
nvidia-smi
echo "=== INFO PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "=== START $(date) ==="

python scripts/train.py \
    --config configs/fase1_edge.json \
    --nombre-experimento fase1_edge

echo "=== END $(date) ==="
