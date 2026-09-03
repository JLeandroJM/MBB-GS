# Correr la ablación de loss en el cluster Khipu (UTEC)

Guía pensada para correr cada uno de los 14 experimentos del estudio de
ablación de loss en el cluster Khipu, desde una Mac. No requiere cambios
al código del repo: el rasterizador CUDA se compila en el cluster la
primera vez.

Documentación oficial: https://docs.khipu.utec.edu.pe/

---

## 0. TL;DR

```bash
# Desde tu Mac (una vez)
ssh tu_usuario@khipu.utec.edu.pe
# en Khipu
git clone <tu-repo> MBB-GS && cd MBB-GS
module load cuda/12.8 python3/3.11.11 gnu12/12.4.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
# subir el video desde la Mac (en otra terminal)
# en Khipu, dentro de un job interactivo:
srun --gres=shard:1 -p debug-gpu --pty /bin/bash
python scripts/data/extraer_clips_720p.py
exit
# correr cada experimento como job batch
sbatch jobs/fase1_baseline.sbatch
```

---

## 1. Conexión SSH desde la Mac

### 1.1 Conexión básica

```bash
ssh tu_usuario@khipu.utec.edu.pe
```

Te va a pedir tu contraseña institucional (misma que la del SSO de UTEC).

**ALERTA**: Khipu **bloquea tu IP** tras 3 intentos fallidos consecutivos.
Si te pasa, escribe a https://servicedesk.utec.edu.pe para que te
desbloqueen. No es un autobloqueo temporal: requiere intervención humana.

### 1.2 Conexión con clave SSH (recomendado)

Para evitar tipear la pass cada vez:

```bash
# En la Mac
ssh-keygen -t ed25519 -C "tu_correo@utec.edu.pe"          # si no tienes ya
ssh-copy-id tu_usuario@khipu.utec.edu.pe
# probar
ssh tu_usuario@khipu.utec.edu.pe
```

### 1.3 Atajo de conexión

Añade a `~/.ssh/config` en la Mac:

```
Host khipu
  HostName khipu.utec.edu.pe
  User tu_usuario
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

Luego basta con: `ssh khipu`.

---

## 2. Entorno de software en Khipu

### 2.1 Módulos disponibles

Khipu usa el sistema **Lmod** para cargar software. Para ver qué hay:

```bash
module avail        # o:  ml avail
```

Versiones relevantes para este proyecto:

| Componente | Versión recomendada | Comando |
|---|---|---|
| CUDA       | 12.8                | `module load cuda/12.8` |
| GCC        | 12.4.0              | `module load gnu/12.4.0` |
| Python     | 3.11.11             | `module load python/3.11.11` |

CUDA 12.8 matchea con PyTorch wheel `cu128`. Si prefieres más
conservador, usa `cuda/11.8` con `cu118`.

### 2.2 Crear venv y dependencias

Khipu NO trae PyTorch como módulo. Lo instalas en un venv:

```bash
module load cuda/12.8 python3/3.11.11 gnu12/12.4.0
cd ~
python -m venv .venv-mbb-gs
source .venv-mbb-gs/bin/activate
pip install --upgrade pip
pip install -r ~/MBB-GS/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Verificación**:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# Esperado en login node: cuda.is_available() == False (es normal, no hay GPU en login).
# La verdad la ves dentro de un job con GPU.
```

### 2.3 Compilar el rasterizador CUDA del repo

La primera vez que `train.py` importa `raster_cuda` (en
`src/gs2d_video/render/cuda_tiled.py`), PyTorch compila la extensión.
Esto **requiere estar dentro de un job con GPU** (no en el login node)
y que las variables de entorno de CUDA estén bien.

```bash
# dentro de un srun interactivo con shard (ver seccion 4.1)
cd ~/MBB-GS
source ~/.venv-mbb-gs/bin/activate
module load cuda/12.8 gnu12/12.4.0
# probar:
python -c "import torch; from gs2d_video.render.cuda_tiled import raster_cuda; print(dir(raster_cuda))"
```

La primera vez tarda 1-2 min compilando. Las siguientes usa el binario
en cache (`~/.cache/torch_extensions/`).

---

## 3. Transferir el video y los configs desde la Mac

### 3.1 Subir el video original

Desde la Mac:

```bash
# subir video.mp4
scp -C /Users/jleandrojm/TESIS/MBB-GS/video/video.mp4 \
    khipu:~/MBB-GS/video/video.mp4
```

(Si usaste el atajo de `~/.ssh/config`, `khipu` funciona como host.)

### 3.2 Subir el repo

Opción A (recomendada, vía git): hace `git push` desde la Mac y
`git clone`/`git pull` en Khipu.

Opción B (rsync directo):

```bash
rsync -azP --exclude='.venv*' --exclude='__pycache__' --exclude='outputs' \
      --exclude='data' --exclude='video' --exclude='.git' \
      /Users/jleandrojm/TESIS/MBB-GS/ \
      khipu:~/MBB-GS/
```

### 3.3 Bajar outputs después de cada experimento

```bash
# trae solo metricas.json + CSV + checkpoint (lo ligero) del experimento
rsync -azP --include='metricas.json' --include='metricas_por_frame.csv' \
       --include='config_usada.json' --include='info_clip.json' \
       --include='checkpoints/***' --include='logs/***' \
       --exclude='frames_renderizados/*' \
       khipu:~/MBB-GS/outputs/fase1_baseline/ \
       ./outputs_khipu/fase1_baseline/
```

Si quieres también los PNGs renderizados (700 MB - 1 GB por experimento
a 720p), quita el `--exclude='frames_renderizados/*'`.

---

## 4. Slurm: cómo correr los experimentos

Khipu usa **Slurm** con **sharding de GPU** (las GPUs físicas se
comparten en fracciones llamadas *shards*). Tu petición de recursos
usa `--gres=shard:N` y NO `--gres=gpu:N` como en clusters estándar.

### 4.1 Sesión interactiva (debug y compilación inicial del kernel)

Para probar antes de meter un batch largo:

```bash
# pide 1 shard de GPU en la particion debug-gpu (max 30 min)
srun --gres=shard:1 -p debug-gpu --time=00:30:00 --pty /bin/bash

# dentro del job:
cd ~/MBB-GS
source ~/.venv-mbb-gs/bin/activate
module load cuda/12.8 gnu12/12.4.0
nvidia-smi                                    # ver la GPU que te toco
python -c "import torch; print(torch.cuda.get_device_name(0))"
# correr 5 epochs para chequear que entra en VRAM:
python scripts/train.py --config configs/video/exp3_loss/fase1_baseline.json \
       --nombre-experimento smoke_test
```

Si OOM, baja `sub_batch_frames` a 1 en el config y vuelve a probar.
Si sale "no CUDA device" es porque pediste solo `-p debug` sin
`-p debug-gpu` o sin `--gres=shard:N`.

### 4.2 Job batch (para correr 1 experimento)

Crea la carpeta `jobs/` en el repo y dentro un `.sbatch` por
experimento. Plantilla:

```bash
#!/bin/bash
#SBATCH --job-name=fase1_baseline
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tu_correo@utec.edu.pe

set -e
cd $HOME/MBB-GS

module load cuda/12.8 gnu12/12.4.0 python3/3.11.11
source $HOME/.venv-mbb-gs/bin/activate

nvidia-smi
echo "=== START $(date) ==="
python scripts/train.py --config configs/video/exp3_loss/fase1_baseline.json \
       --nombre-experimento fase1_baseline
echo "=== END $(date) ==="
```

Cosas a notar:
- `--gres=shard:1` te da una porción de GPU. Si necesitas más VRAM,
  `shard:2` o `shard:4` (cada shard suele ser ~5-10 GB de VRAM, pero
  esto **lo debes confirmar con `nvidia-smi --query-gpu=memory.total`
  dentro de un job interactivo en `g001`**).
- `--time=12:00:00` (12 horas) es una estimación. A 720p+100k+1200 epochs
  puede tomar 4-8 h por experimento. Mejor pedir holgura.
- `--mem=32G` es RAM, no VRAM. Para `frames_en_cpu=true` con 900 frames
  720p uint8 (~2.5 GB) más overhead, 32 GB sobra.
- `--cpus-per-task=8` para el dataloader y la transferencia CPU→GPU.
- Crea `logs_slurm/` antes (`mkdir logs_slurm`).

**Envío**:

```bash
sbatch jobs/fase1_baseline.sbatch
```

Te imprime `Submitted batch job 12345`.

### 4.3 Monitorear

```bash
squeue -u $USER              # tus jobs en cola/corriendo
squeue -p gpu                # todos los jobs de la particion gpu
sacct -X -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList
                             # historial de tus jobs
scancel 12345                # cancelar job por ID
tail -f logs_slurm/fase1_baseline_12345.out
                             # ver salida en tiempo real
```

### 4.4 Email al terminar

Las directivas `--mail-type=END,FAIL --mail-user=...` mandan correo
cuando el job termina o falla. Útil para no estar revisando.

---

## 5. Scripts batch para los 14 experimentos

En lugar de crear 14 archivos a mano, este script bash en el repo genera
todos los `.sbatch` desde una plantilla. Lo metes en `jobs/generar_sbatchs.sh`:

```bash
#!/bin/bash
# generar_sbatchs.sh -- crea jobs/<nombre>.sbatch para cada config en configs/
set -e
mkdir -p jobs logs_slurm

MAIL="tu_correo@utec.edu.pe"          # cambia esto

for CFG in configs/fase*.json; do
    NOMBRE=$(basename "$CFG" .json)
    cat > "jobs/${NOMBRE}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=${NOMBRE}
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -e
cd \$HOME/MBB-GS
module load cuda/12.8 gnu12/12.4.0 python3/3.11.11
source \$HOME/.venv-mbb-gs/bin/activate
nvidia-smi
echo "=== START \$(date) ==="
python scripts/train.py --config ${CFG} --nombre-experimento ${NOMBRE}
echo "=== END \$(date) ==="
EOF
done
echo "Creados $(ls jobs/*.sbatch | wc -l) sbatch en jobs/"
```

Uso:

```bash
chmod +x jobs/generar_sbatchs.sh
./jobs/generar_sbatchs.sh
# corre el primero
sbatch jobs/fase1_baseline.sbatch
```

### 5.1 Estrategia recomendada por fase

- **Fase 1** (6 experimentos): los lanzas todos juntos en cola con
  `for s in jobs/fase1_*.sbatch; do sbatch $s; done`. Khipu los corre
  en paralelo si hay shards libres, secuencial si no.
- **Fase 2** (5 experimentos): NO los lances antes de mirar Fase 1.
  Identificas el ganador y, si es distinto al asumido (`baseline`),
  editas el `tipo_loss` y lambdas en los 5 configs de `fase2_*.json`
  ANTES de lanzar.
- **Fase 3** (3 experimentos): igual, esperas a Fase 2.

---

## 6. Workflow completo de inicio a fin

```bash
# === en la Mac ===
ssh khipu                                            # login

# === en Khipu (login node, una vez) ===
git clone <tu-repo> ~/MBB-GS
cd ~/MBB-GS
module load cuda/12.8 gnu12/12.4.0 python3/3.11.11
python -m venv ~/.venv-mbb-gs
source ~/.venv-mbb-gs/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
mkdir -p video data/clips outputs logs_slurm jobs

# === en la Mac (otra terminal) ===
scp video/video.mp4 khipu:~/MBB-GS/video/video.mp4

# === en Khipu, sesion interactiva con GPU ===
srun --gres=shard:1 -p debug-gpu --time=00:30:00 --pty /bin/bash
cd ~/MBB-GS && source ~/.venv-mbb-gs/bin/activate
module load cuda/12.8 gnu12/12.4.0
nvidia-smi                                           # confirma GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/data/extraer_clips_720p.py                 # extrae PNGs a data/clips/test30s_clips/
# smoke test: corre 5 epochs editando temporalmente n_epochs
python -c "
import json
c = json.load(open('configs/video/exp3_loss/fase1_baseline.json'))
c['n_epochs'] = 5
json.dump(c, open('configs/_smoke.json','w'), indent=2)
"
python scripts/train.py --config configs/_smoke.json --nombre-experimento smoke
exit                                                  # sale del srun

# === en Khipu (login node) ===
./jobs/generar_sbatchs.sh
# lanzar Fase 1 entera
for s in jobs/fase1_*.sbatch; do sbatch $s; done
squeue -u $USER

# === despues de horas/dias ===
sacct -X -u $USER --format=JobID,JobName,State,Elapsed
# bajar metricas
# (en la Mac)
rsync -azP --exclude='frames_renderizados/*' \
      khipu:~/MBB-GS/outputs/ ./outputs_khipu/
```

---

## 7. Cosas que tienes que CONFIRMAR la primera vez en Khipu

Cuando entres a la sesión interactiva, corre esto y guarda los outputs
porque ajustan decisiones:

```bash
# modelo y VRAM de la GPU que te toco
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# que CUDA expone PyTorch
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability())"

# cuantos shards hay por nodo en la particion gpu
sinfo -p gpu -o "%n %G"
scontrol show node g001 | grep -E "Gres|CfgTRES|RealMemory"
```

Con esos números:

- Si la VRAM del shard que te dan es **< 8 GB**: subes a
  `--gres=shard:2` (o más). Con cada shard duplicado obtienes el doble
  de slice de GPU.
- Si el modelo es **A100/H100** (>= 40 GB total): un shard
  probablemente ya te da > 8 GB y `sub_batch_frames=2` o más es
  cómodo.
- Si el modelo es **V100/RTX**: chequea VRAM y ajusta sub_batch.

---

## 8. Apptainer (opcional, NO necesario para este flujo)

Si en algún punto quieres "encapsular" CUDA+PyTorch+repo en una imagen
reproducible:

```bash
# bajar imagen oficial PyTorch+CUDA de NVIDIA NGC
apptainer pull docker://nvcr.io/nvidia/pytorch:24.10-py3
# correr tu codigo con GPU
apptainer exec --nv pytorch_24.10-py3.sif python scripts/train.py --config ...
```

No lo necesitas para el estudio actual — el venv + modules es
suficiente y más simple.

---

## 9. Problemas comunes y soluciones

| Síntoma | Causa probable | Fix |
|---|---|---|
| `torch.cuda.is_available()=False` en job | No pediste `--gres=shard:N` o partición sin GPU | usa `-p gpu --gres=shard:1` |
| `nvcc: command not found` al compilar raster | No cargaste `module load cuda/12.8` | recarga el módulo |
| OOM en training a 720p+100k | shard muy chico | sube a `--gres=shard:2` o más, o baja `sub_batch_frames=1` |
| El kernel CUDA del repo no compila | mismatch nvcc ↔ PyTorch CUDA | matchea: PyTorch `cu128` ↔ `module load cuda/12.8` |
| `OSError: [Errno 122] Disk quota exceeded` | te llenaste el HOME | borra outputs viejos, usa `--exclude` al subir |
| IP bloqueada por SSH | >3 intentos fallidos | mesa de ayuda, no autobloqueo temporal |
| Job killed a los 30 min sin razón | partición `debug` o `debug-gpu` | usa `-p gpu` para jobs largos |

---

## 10. Archivos relevantes del repo

- [scripts/train.py](scripts/train.py) — entry point del entrenamiento.
- [scripts/data/extraer_clips_720p.py](scripts/data/extraer_clips_720p.py) — extrae PNGs del MP4.
- [configs/fase*.json](configs/) — los 14 configs del estudio.
- [cuda/raster_cuda/setup.py](cuda/raster_cuda/setup.py) — se compila
  automáticamente la primera vez vía PyTorch JIT.
- [requirements.txt](requirements.txt) — deps de Python (sin PyTorch:
  PyTorch va aparte para matchear CUDA).
