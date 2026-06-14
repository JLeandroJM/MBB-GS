"""
Generador de configs + sbatch para los experimentos de holdout temporal.

Tras la FASE 0 (descartar loss), corre este script con --loss <ganador> para
generar de un golpe todos los configs y sbatch de:

  FASE 1 (los 3 del profe):
    - multiplos de 2   (supervisa pares)
    - multiplos de 3   (supervisa 1 de cada 3)
    - aleatorio 1 de cada bloque de 3

  CURVA DE DEGRADACION:
    - multiplos de 2, 3, 4, 5  (PSNR_holdout vs fraccion supervisada)

Parte de un config base (default: el de fase 0 con el loss elegido) y solo
cambia: nombre_experimento, holdout y el bloque de loss. Asi no duplicas a mano.

Uso:
    python scripts/generar_configs_holdout.py --loss l1dssim
    python scripts/generar_configs_holdout.py --loss motion --solo_curva
"""
import argparse
import copy
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = RAIZ / "configs"
JOBS = RAIZ / "jobs"

# Bloques de loss segun el ganador de fase 0.
LOSS_BLOQUES = {
    "motion": {"tipo_loss": "motion", "lambda_dssim": 0.25, "lambda_motion": 2.0},
    "l1dssim": {"tipo_loss": "baseline", "lambda_dssim": 0.25, "lambda_motion": 0.0},
}

# Experimentos: (sufijo, holdout_cfg)
FASE1 = [
    ("mult2", {"modo": "multiplos", "k": 2}),
    ("mult3", {"modo": "multiplos", "k": 3}),
    ("aleat_bloque3", {"modo": "aleatorio_por_bloque", "bloque": 3, "por_bloque": 1}),
]
CURVA = [
    ("curva_mult2", {"modo": "multiplos", "k": 2}),
    ("curva_mult3", {"modo": "multiplos", "k": 3}),
    ("curva_mult4", {"modo": "multiplos", "k": 4}),
    ("curva_mult5", {"modo": "multiplos", "k": 5}),
]

SBATCH_TPL = """#!/bin/bash
#SBATCH --job-name={job}
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

# clip dvd_20s: 720x1164, 30fps, 20s = 600 frames
if [ ! -f data/clips/{clip}/frame_0000.png ]; then
  python scripts/extraer_clips_720p.py \\
      --video data/videos/dvd.mp4 --nombre_clip {clip} \\
      --inicio_seg 0 --duracion_seg 20 --fps 30 --H 720 --W 1164
fi

echo "=== START $(date) ==="
python scripts/train.py --config configs/{nombre}.json --nombre-experimento {nombre}
echo "=== END $(date) ==="
"""


def cargar_base(ruta_base, loss):
    config = json.loads(Path(ruta_base).read_text(encoding="utf-8"))
    # aplicar bloque de loss
    bloque = LOSS_BLOQUES[loss]
    config.update(bloque)
    return config


def emitir(config_base, sufijo, holdout_cfg, loss):
    nombre = f"dvd_20s_holdout_{sufijo}_{loss}"
    config = copy.deepcopy(config_base)
    config["nombre_experimento"] = nombre
    config["holdout"] = holdout_cfg
    config["_comentario"] = f"Holdout {holdout_cfg} | loss {loss} | generado por generar_configs_holdout.py"

    ruta_cfg = CONFIGS / f"{nombre}.json"
    ruta_cfg.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    job = nombre.replace("dvd_20s_holdout_", "dvd_").replace(f"_{loss}", f"_{loss}")
    ruta_job = JOBS / f"{nombre}.sbatch"
    ruta_job.write_text(
        SBATCH_TPL.format(job=job[:30], clip=config["clip"], nombre=nombre),
        encoding="utf-8",
    )
    print(f"  {ruta_cfg.relative_to(RAIZ)}  +  {ruta_job.relative_to(RAIZ)}")
    return nombre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["motion", "l1dssim"], required=True,
                    help="loss ganador de fase 0")
    ap.add_argument("--base", default=None,
                    help="config base (default: configs/dvd_20s_holdout_mult2_<loss>.json)")
    ap.add_argument("--solo_fase1", action="store_true")
    ap.add_argument("--solo_curva", action="store_true")
    args = ap.parse_args()

    base = args.base or (CONFIGS / f"dvd_20s_holdout_mult2_{ 'motion' if args.loss=='motion' else 'l1dssim'}.json")
    config_base = cargar_base(base, args.loss)
    print(f"=== generar_configs_holdout (loss={args.loss}, base={Path(base).name}) ===")

    generados = []
    if not args.solo_curva:
        print("FASE 1:")
        for sufijo, h in FASE1:
            generados.append(emitir(config_base, sufijo, h, args.loss))
    if not args.solo_fase1:
        print("CURVA:")
        for sufijo, h in CURVA:
            generados.append(emitir(config_base, sufijo, h, args.loss))

    print(f"\nGenerados {len(generados)} experimentos. Lanza en Khipu con:")
    for n in generados:
        print(f"  sbatch jobs/{n}.sbatch")


if __name__ == "__main__":
    main()
