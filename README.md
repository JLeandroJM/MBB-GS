# MBB-GS: Moving Gaussians

Representación temporal de video mediante **2D Gaussian Splatting**, con parámetros que evolucionan en el tiempo usando **polinomios de Chebyshev**, rasterización diferenciable acelerada con CUDA, reducción del modelo mediante pruning y cuantización `uint16`, e integración experimental de audio mediante **átomos de Gabor**.

Repositorio:

```text
https://github.com/JLeandroJM/MBB-GS
```

---

## Tabla de contenidos

1. [Descripción general](#1-descripción-general)
2. [Idea principal](#2-idea-principal)
3. [Representación temporal del video](#3-representación-temporal-del-video)
4. [Rasterización 2D](#4-rasterización-2d)
5. [Arquitectura del repositorio](#5-arquitectura-del-repositorio)
6. [Requisitos](#6-requisitos)
7. [Instalación](#7-instalación)
8. [Compilación de las extensiones CUDA](#8-compilación-de-las-extensiones-cuda)
9. [Preparación de datos de video](#9-preparación-de-datos-de-video)
10. [Configuración de un experimento de video](#10-configuración-de-un-experimento-de-video)
11. [Entrenamiento de video](#11-entrenamiento-de-video)
12. [Función de pérdida](#12-función-de-pérdida)
13. [Reconstrucción desde un checkpoint](#13-reconstrucción-desde-un-checkpoint)
14. [Interpolación temporal y aumento de FPS](#14-interpolación-temporal-y-aumento-de-fps)
15. [Métricas de video](#15-métricas-de-video)
16. [Pruning adaptativo](#16-pruning-adaptativo)
17. [Cuantización UINT16](#17-cuantización-uint16)
18. [Representación de audio con átomos de Gabor](#18-representación-de-audio-con-átomos-de-gabor)
19. [Audio estéreo y dominio Mid-Side](#19-audio-estéreo-y-dominio-mid-side)
20. [Pipeline audiovisual completo](#20-pipeline-audiovisual-completo)
21. [Configuraciones incluidas](#21-configuraciones-incluidas)
22. [Scripts principales](#22-scripts-principales)
23. [Tests oficiales](#23-tests-oficiales)
24. [Resultados representativos](#24-resultados-representativos)
25. [Uso de memoria y entrenamiento a alta resolución](#25-uso-de-memoria-y-entrenamiento-a-alta-resolución)
26. [Checkpoints y formato interno](#26-checkpoints-y-formato-interno)
27. [Reproducibilidad](#27-reproducibilidad)
28. [Solución de problemas](#28-solución-de-problemas)
29. [Limitaciones actuales](#29-limitaciones-actuales)
30. [Relación con la tesis](#30-relación-con-la-tesis)

---

# 1. Descripción general

MBB-GS estudia una representación alternativa de video basada en una población fija de **gaussianas 2D temporales**.

En un video convencional se almacenan imágenes independientes:

```text
frame_0
frame_1
frame_2
...
frame_T
```

En MBB-GS no se aprende una representación independiente por frame. Se mantiene una misma población de gaussianas durante todo el clip y se aprende cómo cambian sus atributos con el tiempo:

```text
Gaussiana i
    posición     mu_i(t)
    opacidad     alpha_i(t)
    color        c_i(t)
    escala       s_i(t)
    rotación     theta_i(t)
    profundidad  d_i(t)
```

Cada atributo temporal se representa mediante coeficientes de una base polinómica.

La base principal del proyecto es **Chebyshev**.

Esto permite interpretar el video como una representación continua en el tiempo:

```text
tiempo t
   |
   v
evaluación de coeficientes temporales
   |
   v
atributos de todas las gaussianas en t
   |
   v
rasterizador 2D
   |
   v
frame reconstruido
```

El mismo modelo puede utilizarse posteriormente para:

- reconstruir los frames originales;
- evaluar instantes intermedios;
- aumentar la frecuencia temporal del video;
- analizar el movimiento de las gaussianas;
- eliminar gaussianas con pruning;
- cuantizar los coeficientes;
- generar una representación audiovisual al combinarlo con el modelo Gabor de audio.

El **video es el eje principal del framework**. La representación de audio con Gabor constituye una extensión complementaria.

---

# 2. Idea principal

Sea un video con `F` frames.

En lugar de aprender los atributos de una gaussiana de forma independiente para cada frame, MBB-GS aprende una función temporal para cada atributo.

Para una gaussiana `i` y un atributo `p`:

```math
p_i(t) = \sum_{k=0}^{K_p} a_{i,p,k} T_k(\tau(t))
```

donde:

- `a_{i,p,k}` es un coeficiente aprendido;
- `K_p` es el grado temporal asignado al atributo;
- `T_k` es el polinomio de Chebyshev de grado `k`;
- `tau(t)` transforma el tiempo al intervalo `[-1, 1]`.

Para los frames originales:

```math
\tau(j) = 2 \frac{j}{F-1} - 1
```

con:

```text
j = 0, 1, ..., F-1
```

La población de gaussianas permanece fija. Lo que cambia a lo largo del video son sus parámetros.

Una configuración típica puede asignar capacidades temporales diferentes:

```json
"grados": {
  "mu": 100,
  "opacity": 80,
  "color": 30,
  "scale": 12,
  "theta": 6,
  "depth": 4
}
```

La posición y la opacidad reciben más capacidad temporal porque suelen necesitar describir cambios más complejos. La escala, rotación y profundidad pueden utilizar grados menores.

---

# 3. Representación temporal del video

## 3.1 Polinomios de Chebyshev

La base de Chebyshev de primer tipo se construye mediante:

```math
T_0(x) = 1
```

```math
T_1(x) = x
```

```math
T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)
```

El módulo:

```text
src/gs2d_video/core/bases.py
```

construye la matriz temporal:

```text
B.shape = [n_frames, grado_max + 1]
```

donde:

```text
B[j, k] = T_k(tau_j)
```

Para mejorar la precisión numérica de la recurrencia, la matriz se construye primero en CPU con `float64` y posteriormente se convierte al `dtype` y dispositivo objetivo.

## 3.2 Base monomial

El repositorio conserva una implementación de base monomial:

```math
1, t, t^2, ..., t^K
```

Esta base existe principalmente para:

- reproducir la ablación Chebyshev vs. monomial;
- comparar estabilidad;
- conservar compatibilidad con checkpoints experimentales.

La ruta principal del framework utiliza Chebyshev.

El cargador central de checkpoints conserva la base temporal almacenada en la configuración. Si un checkpoint antiguo no especifica `base_temporal`, se interpreta como Chebyshev.

## 3.3 Parámetros de cada gaussiana

El modelo principal se encuentra en:

```text
src/gs2d_video/core/modelo.py
```

La clase central es:

```python
GaussianasPolinomial2D
```

Cada gaussiana posee seis grupos de atributos:

| Atributo | Dimensión conceptual | Uso |
|---|---:|---|
| `mu` | 2 | posición vertical y horizontal |
| `opacity` | 1 | opacidad |
| `color` | 3 | RGB |
| `scale` | 2 | escalas principales de la elipse |
| `theta` | 1 | rotación |
| `depth` | 1 | orden de profundidad |

Los coeficientes se almacenan separando:

```text
<atributo>_a0
<atributo>_high
```

Por ejemplo:

```text
mu_a0
mu_high

opacity_a0
opacity_high

color_a0
color_high

scale_a0
scale_high

theta_a0
theta_high

depth_a0
depth_high
```

`a0` contiene el término de orden cero y `high` contiene los coeficientes temporales de orden superior.

## 3.4 Activaciones

Los valores temporales evaluados no siempre son utilizados directamente.

El modelo aplica transformaciones según el significado físico del parámetro:

```text
mu       -> valor directo
theta    -> valor directo
depth    -> valor directo
opacity  -> sigmoid
color    -> sigmoid
scale    -> exp sobre un rango acotado
```

De esta manera:

```text
opacity in (0, 1)
color   in (0, 1)
scale   > 0
```

## 3.5 Evaluación de un frame

Para renderizar el frame `j`:

```text
1. seleccionar B[j]
2. evaluar los coeficientes de cada atributo
3. aplicar las activaciones
4. construir los parámetros 2D
5. rasterizar las gaussianas
```

Conceptualmente:

```math
P_j = A B_j
```

donde `A` contiene coeficientes aprendidos y `B_j` es la fila temporal correspondiente al frame.

---

# 4. Rasterización 2D

El renderer de producción se encuentra principalmente en:

```text
src/gs2d_video/render/renderer.py
src/gs2d_video/render/cuda_tiled.py
cuda/raster_cuda/
```

La extensión CUDA implementa el rasterizador diferenciable utilizado durante el entrenamiento y la reconstrucción.

## 4.1 Gaussiana 2D

Una gaussiana elíptica puede interpretarse de forma general como:

```math
G_i(x) =
\exp\left(
-\frac{1}{2}
(x-\mu_i)^T
\Sigma_i^{-1}
(x-\mu_i)
\right)
```

La matriz de covarianza está determinada por:

```text
scale_x
scale_y
theta
```

La opacidad efectiva de la gaussiana se combina con su valor espacial.

## 4.2 Composición

Las gaussianas se procesan siguiendo un orden de profundidad y se utiliza composición alfa.

Para cada píxel se acumula color mientras disminuye la transmitancia.

De forma conceptual:

```text
T = 1
color = 0

para cada gaussiana ordenada:
    alpha = opacity * gaussian_value
    color += T * alpha * gaussian_color
    T *= 1 - alpha
```

El rasterizador utiliza tiles para evitar evaluar cada gaussiana contra todos los píxeles de la imagen.

## 4.3 Flujo CUDA

El pipeline CUDA contiene operaciones para:

```text
construcción de la cónica
        |
        v
preprocess por tiles
        |
        v
ordenamiento / asociación gaussiana-tile
        |
        v
forward raster
        |
        v
loss o gradiente del render
        |
        v
backward raster
        |
        v
gradientes hacia parámetros
```

Entre las funciones internas de la extensión se encuentran operaciones equivalentes a:

```text
build_conic
preprocess_tiled
forward_tiled_train
forward_tiled_train_loss
backward_tiled_fast
grad_conic_to_scale_theta
```

## 4.4 Profundidad

`depth` interviene en el ordenamiento de las gaussianas.

El ordenamiento es una operación discreta. Por ello, el flujo actual no interpreta el cambio de orden como una operación continuamente diferenciable.

---

# 5. Arquitectura del repositorio

La estructura conceptual principal es:

```text
MBB-GS/
|
|-- configs/
|   |-- comparacion_bases/
|   |-- comparacion_capacidad/
|   |-- gabor/
|   |-- audio_only/
|   |-- rockyourbody_10s_1ep/
|   |-- thriller_10s_1ep/
|   |-- fase1_*.json
|   |-- fase2_*.json
|   |-- fase3_*.json
|   |-- motion_150k_*.json
|   `-- ganador_motion_200k_1200ep.json
|
|-- cuda/
|   |-- raster_cuda/
|   `-- gabor_audio_cuda/
|
|-- data/
|   `-- datos locales no versionados
|
|-- jobs/
|   `-- scripts SLURM usados en experimentos
|
|-- scripts/
|   |-- train.py
|   |-- _carga_checkpoint.py
|   |-- extraer_clips_720p.py
|   |-- regenerar_clip_desde_checkpoint_streaming.py
|   |-- regenerar_fps_interpolado.py
|   |-- analizar_interpolacion_fps.py
|   |-- comparar_frames_psnr.py
|   |-- lpips_post_hoc.py
|   |-- run_binary_pruning_adaptativo.py
|   |-- pack_checkpoint_uint16.py
|   |-- unpack_checkpoint_uint16.py
|   |-- pack_checkpoint_uint16_all.py
|   |-- unpack_checkpoint_uint16_all.py
|   |-- run_pipeline_video_audio.py
|   |-- GABOR_SCRIPTS/
|   `-- OLD_TESTS_PRUNNING/
|
|-- src/
|   |-- gs2d_video/
|   |   |-- core/
|   |   |-- io/
|   |   |-- metrics/
|   |   |-- render/
|   |   |-- training/
|   |   `-- viz/
|   |
|   `-- gs2d_gabor/
|       |-- core/
|       `-- render/
|
|-- tests/
|   |-- conftest.py
|   |-- test_bases_modelo.py
|   |-- test_checkpoint_loader.py
|   `-- test_gabor_gradientes.py
|
|-- .gitattributes
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

Los directorios `outputs/`, los datasets, checkpoints y binarios CUDA generados localmente están excluidos del control de versiones.

---

# 6. Requisitos

## 6.1 Software

El proyecto requiere:

```text
Python >= 3.10
PyTorch
CUDA
compilador C/C++ compatible con PyTorch
Ninja
NumPy
SciPy
OpenCV
Pillow
ImageIO
Matplotlib
pytest
```

Para las métricas adicionales:

```text
LPIPS
pytorch-msssim
```

Para el pipeline audiovisual:

```text
FFmpeg
FFprobe
```

Opcionalmente:

```text
7-Zip
```

se utiliza para comprimir sin pérdida los paquetes finales.

## 6.2 Hardware

Para el flujo principal de video se recomienda una GPU NVIDIA compatible con CUDA.

El framework fue desarrollado pensando en entrenamientos que pueden alcanzar:

```text
720p
decenas o cientos de miles de gaussianas
centenares de frames
grados temporales altos
```

Por tanto, el consumo de:

```text
VRAM
RAM
tiempo de GPU
```

depende fuertemente de la configuración.

Los tests unitarios principales pueden ejecutarse sin realizar un entrenamiento completo.

---

# 7. Instalación

## 7.1 Clonar el repositorio

```powershell
git clone https://github.com/JLeandroJM/MBB-GS.git
cd MBB-GS
```

## 7.2 Crear un entorno

Ejemplo con Conda:

```powershell
conda create -n mbb-gs python=3.10 -y
conda activate mbb-gs
```

Actualizar `pip`:

```powershell
python -m pip install --upgrade pip
```

## 7.3 Instalar dependencias

```powershell
pip install -r requirements.txt
```

Instalar el proyecto local en modo editable:

```powershell
pip install -e .
```

La instalación editable permite importar:

```python
import gs2d_video
import gs2d_gabor
```

directamente desde el código presente en `src/`.

## 7.4 Verificar PyTorch y CUDA

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('cuda disponible:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

La versión CUDA utilizada por PyTorch, el driver NVIDIA y el toolkit empleado para compilar las extensiones deben ser compatibles entre sí.

El archivo `requirements.txt` documenta el conjunto de dependencias del entorno del proyecto, pero una instalación CUDA puede requerir adaptar la distribución de PyTorch a la plataforma donde se ejecute.

---

# 8. Compilación de las extensiones CUDA

MBB-GS contiene dos extensiones CUDA principales:

```text
cuda/raster_cuda
cuda/gabor_audio_cuda
```

Los binarios compilados no se almacenan en Git. Deben generarse en cada entorno.

## 8.1 Rasterizador CUDA de video

Desde la raíz del repositorio, en PowerShell:

```powershell
Push-Location .\cuda\raster_cuda
python setup.py build_ext --inplace
Pop-Location
```

Equivalente:

```powershell
cd .\cuda\raster_cuda
python setup.py build_ext --inplace
cd ..\..
```

Después de compilar debe aparecer un binario similar a:

```text
raster_cuda.cp310-win_amd64.pyd
```

en Windows, o:

```text
raster_cuda*.so
```

en Linux.

Verificación:

```powershell
python -c "import torch, sys; sys.path.insert(0, r'cuda/raster_cuda'); import raster_cuda; print('raster_cuda OK')"
```

### `RASTER_BATCH_SIZE`

La compilación del rasterizador permite utilizar la variable:

```text
RASTER_BATCH_SIZE
```

Por ejemplo:

```powershell
$env:RASTER_BATCH_SIZE="256"
Push-Location .\cuda\raster_cuda
python setup.py build_ext --inplace
Pop-Location
```

Cambiar este valor requiere recompilar la extensión.

## 8.2 CUDA para audio Gabor

```powershell
Push-Location .\cuda\gabor_audio_cuda
python setup.py build_ext --inplace
Pop-Location
```

Verificación:

```powershell
python -c "import torch, sys; sys.path.insert(0, r'cuda/gabor_audio_cuda'); import gabor_audio_cuda; print('gabor_audio_cuda OK')"
```

El wrapper de Python también busca esta extensión mediante:

```text
RUTA_GABOR_AUDIO_CUDA
```

cuando se define explícitamente.

## 8.3 Linux

La lógica es la misma:

```bash
cd cuda/raster_cuda
python setup.py build_ext --inplace
cd ../..

cd cuda/gabor_audio_cuda
python setup.py build_ext --inplace
cd ../..
```

En servidores con distintas arquitecturas NVIDIA puede ser útil definir explícitamente las arquitecturas que se desean compilar. Esto debe adaptarse al hardware real del servidor.

Ejemplo:

```bash
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
```

No es obligatorio utilizar exactamente esa lista.

---

# 9. Preparación de datos de video

Los clips utilizados por `train.py` se esperan normalmente como:

```text
data/
`-- clips/
    `-- mi_clip/
        |-- frame_0000.png
        |-- frame_0001.png
        |-- frame_0002.png
        `-- ...
```

El script recomendado para extraer un clip es:

```text
scripts/extraer_clips_720p.py
```

Ejemplo:

```powershell
python .\scripts\extraer_clips_720p.py `
    --video "D:\Videos\entrada.mp4" `
    --nombre_clip "mi_clip" `
    --inicio_seg 0 `
    --duracion_seg 10 `
    --fps 30 `
    --H 720 `
    --W 1280 `
    --forzar
```

Esto genera:

```text
data/clips/mi_clip/frame_0000.png
data/clips/mi_clip/frame_0001.png
...
```

Parámetros principales:

| Parámetro | Significado |
|---|---|
| `--video` | archivo de video de entrada |
| `--nombre_clip` | nombre dentro de `data/clips/` |
| `--inicio_seg` | tiempo inicial |
| `--duracion_seg` | duración del fragmento |
| `--max_frames` | límite máximo de frames |
| `--fps` | FPS de salida |
| `--H` | altura |
| `--W` | ancho |
| `--forzar` | reemplazar extracción existente |

Si no se especifica `--fps`, el script puede utilizar la frecuencia del video original.

---

# 10. Configuración de un experimento de video

El entrenamiento se controla principalmente mediante archivos JSON.

Ejemplo conceptual:

```json
{
  "nombre_experimento": "mi_experimento",
  "clip": "mi_clip",
  "base_temporal": "chebyshev",
  "max_frames": 300,
  "device": "cuda",
  "seed": 42,

  "n_gaussianas_inicial": 100000,
  "inicializar_color_desde_frame0": true,
  "escala_inicial_px": 2.5,

  "grados": {
    "mu": 100,
    "opacity": 80,
    "color": 30,
    "scale": 12,
    "theta": 6,
    "depth": 4
  },

  "n_epochs": 800,
  "sub_batch_frames": 1,

  "tipo_loss": "motion",
  "lambda_dssim": 0.25,
  "lambda_motion": 2.0,

  "exponente_frame": 1.0,
  "usar_max_frame": false,

  "usar_cuda_tiled": true,
  "cuda_tile_size": 16,
  "cuda_k_sigma": 3.5,

  "frames_en_cpu": true,
  "evitar_render_completo_en_train": true,
  "usar_metricas_streaming": true
}
```

Este fragmento no reemplaza las configuraciones incluidas en `configs/`. Su objetivo es mostrar los grupos principales de parámetros.

## 10.1 `base_temporal`

Valores soportados por el flujo actual:

```text
chebyshev
monomial
```

Chebyshev es la opción principal.

Los alias históricos aceptados por el cargador se normalizan a una de estas bases.

## 10.2 Número de gaussianas

```json
"n_gaussianas_inicial": 100000
```

Más gaussianas permiten mayor capacidad espacial, pero incrementan:

```text
memoria
tamaño del checkpoint
costo de rasterización
tiempo de entrenamiento
```

## 10.3 Grados temporales

```json
"grados": {
  "mu": 100,
  "opacity": 80,
  "color": 30,
  "scale": 12,
  "theta": 6,
  "depth": 4
}
```

Cada atributo puede tener una capacidad temporal independiente.

## 10.4 Learning rates

El optimizador permite learning rates separados para:

```text
mu_a0
mu_high
opacity_a0
opacity_high
color_a0
color_high
scale_a0
scale_high
theta_a0
theta_high
depth_a0
depth_high
```

Esto permite controlar por separado:

```text
valor base
evolución temporal
```

de cada atributo.

## 10.5 Regularización temporal

La configuración puede incluir:

```json
"beta_smoothness": 1e-6,
"pesos_smoothness": {
  "mu": 0.0,
  "opacity": 0.0,
  "color": 0.0,
  "scale": 0.0,
  "theta": 1.0,
  "depth": 2.0
}
```

La regularización penaliza coeficientes temporales de orden alto mediante un peso dependiente del grado.

---

# 11. Entrenamiento de video

El punto de entrada principal es:

```text
scripts/train.py
```

Uso:

```powershell
python .\scripts\train.py --config .\configs\ganador_motion_200k_1200ep.json
```

También puede modificarse el nombre del experimento cuando el script/config utilizado lo permite.

## 11.1 Flujo general

```text
config JSON
   |
   v
carga de frames
   |
   v
matrices temporales
   |
   v
GaussianasPolinomial2D
   |
   v
optimizador
   |
   v
entrenamiento
   |
   v
checkpoint_final.pt
   |
   v
render / métricas / visualizaciones
```

## 11.2 Inicialización

El modelo puede inicializar el color a partir del primer frame:

```json
"inicializar_color_desde_frame0": true
```

Esto permite que las gaussianas comiencen con información relacionada con la imagen objetivo en lugar de partir únicamente de colores arbitrarios.

## 11.3 Entrenamiento con frames en CPU

Para clips grandes:

```json
"frames_en_cpu": true
```

mantiene los frames como `uint8` en RAM y transfiere a GPU únicamente los datos necesarios durante el entrenamiento.

Esto evita reservar en VRAM todo el video como `float32`.

## 11.4 Frames `uint8` en GPU

Existe también:

```json
"frames_en_gpu_uint8": true
```

para entornos con suficiente VRAM.

## 11.5 Rasterización completa vs. streaming

En configuraciones grandes se recomienda:

```json
"evitar_render_completo_en_train": true,
"usar_metricas_streaming": true
```

El modo streaming evita apilar todos los frames reconstruidos simultáneamente en GPU.

En su lugar:

```text
render frame
   |
   v
calcular métricas
   |
   v
guardar PNG
   |
   v
liberar temporales
   |
   v
siguiente frame
```

Esta ruta es especialmente importante en 720p con poblaciones grandes de gaussianas.

## 11.6 Salidas

Un experimento produce una estructura similar a:

```text
outputs/<nombre_experimento>/
|
|-- frames_renderizados/
|   |-- frame_0000.png
|   |-- frame_0001.png
|   `-- ...
|
|-- checkpoints/
|   |-- checkpoint_final.pt
|   `-- modelo_pruneado.pt
|
|-- logs/
|   |-- log_entrenamiento.csv
|   |-- trayectorias.png
|   |-- heatmap_opacity_temporal.png
|   |-- evolucion_parametros.png
|   `-- coeficientes_magnitudes.png
|
|-- config_usada.json
|-- info_clip.json
|-- metricas.json
|-- metricas_por_frame.csv
`-- metricas_compresion.json
```

La presencia exacta de algunos archivos depende de las opciones habilitadas.

---

# 12. Función de pérdida

El módulo principal es:

```text
src/gs2d_video/core/perdidas.py
```

El repositorio conserva varias variantes experimentales, entre ellas:

```text
baseline
l1_mse
motion
hard
edge
temporal
motion_temporal
combo
```

No todas representan la configuración final de la tesis; varias existen para reproducir ablaciones.

## 12.1 Pérdida `motion`

La variante seleccionada en los experimentos principales da más peso a regiones dinámicas.

A partir del ground truth:

```math
M_t =
\operatorname{normalize}
(
|I_t-I_{t-1}|
)
```

se construye un peso:

```math
W_t = 1 + \lambda_{motion} M_t
```

y el error absoluto ponderado puede escribirse conceptualmente como:

```math
L_{motion}
=
\frac{
\sum_p W_t(p)
|\hat I_t(p)-I_t(p)|
}{
\sum_p W_t(p)
}
```

La máscara de movimiento se calcula a partir del video objetivo, no a partir de la predicción.

La configuración utilizada en los experimentos finales combina este término con DSSIM.

Ejemplo:

```json
"tipo_loss": "motion",
"lambda_motion": 2.0,
"lambda_dssim": 0.25
```

## 12.2 Agregación entre frames

El framework permite modificar el énfasis relativo entre frames mediante:

```json
"exponente_frame": 1.0,
"usar_max_frame": false
```

Los experimentos mostraron que `q = 1`, equivalente al promedio convencional, fue superior a valores mayores.

La configuración recomendada para el camino principal es:

```text
exponente_frame = 1
```

## 12.3 Smoothness

Además del error de reconstrucción se puede incorporar una penalización sobre coeficientes temporales de alto orden.

La idea es evitar que los términos de gran grado crezcan sin control cuando no son necesarios.

---

# 13. Reconstrucción desde un checkpoint

El script principal para reconstrucción eficiente es:

```text
scripts/regenerar_clip_desde_checkpoint_streaming.py
```

Ejemplo:

```powershell
python .\scripts\regenerar_clip_desde_checkpoint_streaming.py `
    --checkpoint ".\outputs\mi_experimento\checkpoints\checkpoint_final.pt" `
    --salida ".\outputs\mi_experimento\recon_streaming" `
    --device cuda `
    --inicio 0 `
    --fin 300 `
    --fps 30 `
    --crear_video
```

Opciones importantes:

```text
--checkpoint
--salida
--clip
--device
--inicio
--fin
--crear_video
--fps
--comparaciones
--factor_diff
--limpiar_cache_cada
```

## 13.1 Cargador central

La reconstrucción utiliza:

```text
scripts/_carga_checkpoint.py
```

como lógica central para:

- leer `state_dict_coefs`;
- reconstruir `GaussianasPolinomial2D`;
- recuperar dimensiones;
- recuperar grados;
- recuperar número de gaussianas;
- recuperar número de frames;
- normalizar la base temporal;
- mantener compatibilidad con checkpoints antiguos.

Esto evita que cada script implemente su propia interpretación del checkpoint.

## 13.2 Rango de frames

En:

```text
--inicio 0 --fin 300
```

`fin` funciona como límite superior exclusivo.

Por tanto se renderizan:

```text
0 ... 299
```

## 13.3 Comparaciones visuales

Con:

```powershell
--comparaciones
```

el script puede producir composiciones:

```text
original | reconstrucción | diferencia amplificada
```

para inspección visual.

---

# 14. Interpolación temporal y aumento de FPS

Uno de los usos más importantes de la representación temporal es evaluar el modelo en tiempos que no pertenecían a la secuencia de entrenamiento.

El script es:

```text
scripts/regenerar_fps_interpolado.py
```

Ejemplo:

```powershell
python .\scripts\regenerar_fps_interpolado.py `
    --checkpoint ".\outputs\mi_experimento\checkpoints\checkpoint_final.pt" `
    --salida ".\outputs\mi_experimento\interpolado_120fps" `
    --fps_origen 30 `
    --fps_salida 120 `
    --device cuda `
    --forzar
```

El número de frames de salida se calcula preservando aproximadamente la duración:

```math
F_{out}
=
round
\left[
(F_{in}-1)
\frac{fps_{out}}{fps_{in}}
\right]
+1
```

## 14.1 Qué significa interpolar en MBB-GS

No se entrena una segunda red.

No se copian frames.

No se realiza únicamente una mezcla lineal de imágenes.

Se construyen nuevas posiciones temporales y se evalúan los mismos coeficientes aprendidos:

```text
checkpoint
    |
    v
coeficientes temporales aprendidos
    |
    +------ tiempo original
    |
    +------ tiempo intermedio
    |
    +------ tiempo intermedio
    |
    v
nuevos estados de las gaussianas
    |
    v
nuevos frames
```

El flujo actual respeta la base temporal almacenada en el checkpoint:

```text
Chebyshev -> interpolación con Chebyshev
monomial  -> interpolación con monomial
```

## 14.2 Convertir frames interpolados a video

```powershell
python .\scripts\frames_a_video.py `
    --frames ".\outputs\mi_experimento\interpolado_120fps" `
    --salida ".\outputs\mi_experimento\interpolado_120fps.mp4" `
    --fps 120
```

---

# 15. Métricas de video

El módulo:

```text
src/gs2d_video/metrics/calidad.py
```

incluye métricas de calidad utilizadas para evaluar reconstrucciones.

Entre las principales:

```text
PSNR
SSIM
LPIPS
PSNR temporal
```

## 15.1 PSNR

Mide fidelidad píxel a píxel a partir del MSE.

Mayor es mejor.

## 15.2 SSIM

Mide similitud estructural.

Mayor es mejor.

## 15.3 LPIPS

Compara similitud perceptual utilizando características profundas.

Menor es mejor.

## 15.4 PSNR temporal

Compara la evolución temporal de la reconstrucción y permite evaluar si el movimiento se reproduce de forma coherente.

## 15.5 Comparación entre carpetas de frames

Para comparar dos reconstrucciones:

```powershell
python .\scripts\comparar_frames_psnr.py `
    --a ".\ruta\frames_A" `
    --b ".\ruta\frames_B" `
    --out ".\resultado.csv"
```

Este script es utilizado también por el pipeline de pruning y cuantización.

## 15.6 LPIPS post-hoc

Para calcular LPIPS posteriormente sobre resultados ya generados:

```text
scripts/lpips_post_hoc.py
```

Esto es útil cuando se desea evitar el costo de LPIPS durante una ejecución larga y calcularlo después.

---

# 16. Pruning adaptativo

El pruning busca reducir el número de gaussianas después del entrenamiento.

El flujo principal está implementado en:

```text
scripts/run_binary_pruning_adaptativo.py
```

El nombre del archivo conserva la nomenclatura histórica. El procedimiento actual no debe interpretarse estrictamente como una búsqueda binaria clásica: evalúa porcentajes progresivos y conserva la mejor versión que cumple el criterio de calidad.

## 16.1 Flujo

```text
checkpoint completo
       |
       v
estadísticas por gaussiana
       |
       v
ranking adaptativo
       |
       v
probar porcentaje
       |
       v
render reducido
       |
       v
comparar contra baseline
       |
       +---- PSNR suficiente -> continuar / conservar
       |
       `---- PSNR insuficiente -> retroceder
```

## 16.2 Estadísticas

El análisis utiliza información generada por:

```text
scripts/viz_gaussian_stats.py
```

Entre las estadísticas:

```text
path_length_px
color_path
op_mean
op_max
op_std
active_frac
scale_std
```

## 16.3 Protección de gaussianas transitorias

El selector protege gaussianas que presentan:

```text
op_max >= 0.90
active_frac <= 0.10
```

Este criterio busca evitar eliminar primitivas que pueden ser importantes durante una fracción corta del video.

## 16.4 Candidatas

La implementación actual forma la población candidata utilizando:

```text
op_mean >= 0.05
```

y excluyendo primero las gaussianas protegidas por el criterio transitorio.

## 16.5 Score adaptativo

Las métricas se convierten a rangos percentiles y se combinan con:

```text
0.30 * path_length_px
0.25 * color_path
0.25 * op_std
0.20 * scale_std
```

Las gaussianas con los scores más bajos son las primeras candidatas a eliminación.

Esto no debe confundirse con eliminar únicamente por opacidad.

## 16.6 Búsqueda por porcentaje

Ejemplo:

```powershell
python .\scripts\run_binary_pruning_adaptativo.py `
    --exp ".\outputs\mi_experimento" `
    --checkpoint ".\outputs\mi_experimento\checkpoints\checkpoint_final.pt" `
    --baseline_frames ".\outputs\mi_experimento\frames_renderizados" `
    --fps 30 `
    --device cuda `
    --inicio_pct 10 `
    --paso_pct 5 `
    --min_pct 10 `
    --max_pct 50 `
    --psnr_min 65 `
    --crear_video_ganador
```

Con esos valores se pueden evaluar niveles como:

```text
10 %
15 %
20 %
25 %
...
```

La aceptación utiliza **PSNR global**, calculado a partir del MSE promedio entre frames.

## 16.7 Salidas

El directorio típico es:

```text
outputs/<experimento>/binary_pruning/
|
|-- ids/
|-- tests/
|-- metricas/
|-- checkpoints/
|-- resumen_busqueda_pruning.csv
|-- resumen_busqueda_pruning.txt
`-- ganador/
```

El checkpoint ganador queda separado del checkpoint original.

---

# 17. Cuantización UINT16

Después del pruning, el modelo puede reducirse todavía más mediante cuantización afín por tensor.

La idea general es:

```math
q =
round
\left(
\frac{x-x_{min}}{scale}
\right)
```

con:

```math
scale =
\frac{x_{max}-x_{min}}{65535}
```

y la reconstrucción:

```math
\hat x = x_{min} + scale \cdot q
```

donde:

```text
q in uint16
```

Esta estrategia utiliza los `65536` niveles enteros dentro del rango real de cada tensor.

## 17.1 UINT16 SAFE

Scripts:

```text
scripts/pack_checkpoint_uint16.py
scripts/unpack_checkpoint_uint16.py
```

La variante utilizada por el pipeline audiovisual cuantiza principalmente:

```text
mu_high
color_high
opacity_high
scale_high
```

mientras conserva otros valores en `float32`.

Ejemplo:

```powershell
python .\scripts\pack_checkpoint_uint16.py `
    --in_ckpt ".\modelo_pruneado.pt" `
    --out_pkg ".\modelo_uint16_safe.pkg.pt" `
    --other_float fp32 `
    --quant_tensors mu_high color_high opacity_high scale_high `
    --omit_zero_depth_high
```

Reconstrucción del checkpoint:

```powershell
python .\scripts\unpack_checkpoint_uint16.py `
    --in_pkg ".\modelo_uint16_safe.pkg.pt" `
    --out_ckpt ".\modelo_uint16_safe_render.pt" `
    --out_float fp32
```

## 17.2 UINT16 ALL

Scripts:

```text
scripts/pack_checkpoint_uint16_all.py
scripts/unpack_checkpoint_uint16_all.py
```

Ejemplo:

```powershell
python .\scripts\pack_checkpoint_uint16_all.py `
    --in_ckpt ".\modelo_pruneado.pt" `
    --out_pkg ".\modelo_uint16_all.pkg.pt" `
    --omit_zero_depth_high
```

Reconstrucción:

```powershell
python .\scripts\unpack_checkpoint_uint16_all.py `
    --in_pkg ".\modelo_uint16_all.pkg.pt" `
    --out_ckpt ".\modelo_uint16_all_render.pt"
```

## 17.3 Omisión de `depth_high`

Cuando:

```text
depth_high
```

es completamente cero dentro de una tolerancia pequeña, puede omitirse del paquete.

Durante el unpack se reconstruye como un tensor de ceros con la forma esperada.

No se elimina si contiene valores significativos.

## 17.4 Validación

La cuantización no se considera válida únicamente porque reduzca el tamaño.

El flujo correcto es:

```text
modelo FP32
   |
   v
cuantización
   |
   v
decuantización
   |
   v
render
   |
   v
comparación contra baseline
```

El pipeline calcula PSNR entre los frames cuantizados y la reconstrucción de referencia.

---

# 18. Representación de audio con átomos de Gabor

La extensión de audio utiliza una representación distinta a la de video.

La ruta principal es:

```text
src/gs2d_gabor/
scripts/GABOR_SCRIPTS/
cuda/gabor_audio_cuda/
```

El audio se representa directamente en el dominio temporal.

## 18.1 Átomo de Gabor

Cada átomo es:

```math
g_i(t)
=
A_i
\exp
\left[
-\frac{(t-\mu_i)^2}{2\sigma_i^2}
\right]
\cos
\left[
2\pi f_i(t-\mu_i)+\phi_i
\right]
```

La señal reconstruida es:

```math
\hat x(t) = \sum_i g_i(t)
```

Parámetros:

| Parámetro | Significado |
|---|---|
| `mu` | posición temporal |
| `sigma` | duración/ancho temporal |
| `amp` | amplitud |
| `fnorm` / frecuencia | frecuencia de oscilación |
| `phi` | fase |

A diferencia de un enfoque basado únicamente en magnitud espectral, la fase forma parte explícita del modelo.

## 18.2 CUDA Gabor

La extensión:

```text
cuda/gabor_audio_cuda/
```

implementa:

```text
forward
backward
```

para:

```text
mu
sigma
amp
fnorm
phi
```

El kernel utiliza soporte local:

```text
[mu - k_sigma*sigma, mu + k_sigma*sigma]
```

de forma que un átomo no necesita evaluarse contra todas las muestras cuando su envolvente ya es despreciable.

## 18.3 Fallback PyTorch

Existe una implementación diferenciable en PyTorch para:

```text
validación
CPU
señales pequeñas
comparaciones numéricas
```

Para audio largo y poblaciones grandes de átomos, CUDA es la ruta práctica.

---

# 19. Audio estéreo y dominio Mid-Side

El entrenamiento estéreo se encuentra en:

```text
scripts/GABOR_SCRIPTS/train_gabor_stereo.py
```

Hay dos dominios soportados.

## 19.1 L/R

```text
Left  -> modelo Gabor
Right -> modelo Gabor
```

Los canales se entrenan de forma independiente.

## 19.2 Mid-Side

Se transforma:

```math
M = \frac{L+R}{2}
```

```math
S = \frac{L-R}{2}
```

y se entrenan:

```text
Mid  -> modelo Gabor
Side -> modelo Gabor
```

La reconstrucción vuelve a L/R mediante:

```math
L = M + S
```

```math
R = M - S
```

Este esquema permite asignar cantidades de átomos distintas a Mid y Side.

Ejemplo de configuración de alta capacidad:

```json
{
  "n_atomos": 160000,
  "dominio": "MS",
  "n_atomos_mid": 96000,
  "n_atomos_side": 64000
}
```

## 19.3 Función de pérdida de audio

El módulo:

```text
src/gs2d_gabor/core/perdidas_gabor.py
```

permite combinar:

```text
waveform L1
MR-STFT
-SI-SDR
```

Conceptualmente:

```math
L =
\lambda_{wave}L_{wave}
+
\lambda_{mrstft}L_{mrstft}
+
\lambda_{sisdr}L_{sisdr}
```

Ejemplo:

```json
"lambda_wave": 1.0,
"lambda_mrstft": 0.6,
"lambda_sisdr": 0.2,
"mrstft_ffts": [512, 1024, 2048, 4096]
```

## 19.4 Inicialización por energía

Las configuraciones Gabor pueden usar:

```json
"init_modo": "energia"
```

con parámetros de STFT para distribuir inicialmente los átomos en regiones relevantes de la señal.

Ejemplo:

```json
"init_alpha": 0.7,
"init_n_fft": 2048,
"init_hop": 512
```

## 19.5 Entrenamiento mono

```powershell
python .\scripts\GABOR_SCRIPTS\train_gabor.py `
    --config ".\configs\gabor\gabor_rock_2s_smoke.json"
```

Salida típica:

```text
outputs/gabor/<experimento>/
|
|-- checkpoints/
|   `-- checkpoint_final.pt
|-- config_usada.json
|-- info_audio.json
|-- metricas.json
|-- log_entrenamiento.csv
|-- loss_curve.png
|-- recon.wav
`-- target.wav
```

## 19.6 Entrenamiento estéreo

```powershell
python .\scripts\GABOR_SCRIPTS\train_gabor_stereo.py `
    --config ".\configs\gabor\gabor_rock_31_40_stereo_MS_160k_6000ep.json"
```

Entre las métricas disponibles:

```text
SNR
SI-SDR
PSNR waveform
MSE waveform
LSD
Mel-L1
MR-STFT
SNR Mid
SNR Side
SI-SDR Mid
SI-SDR Side
```

---

# 20. Pipeline audiovisual completo

El punto de entrada integrado es:

```text
scripts/run_pipeline_video_audio.py
```

Este pipeline ejecuta el flujo de video y audio sobre el mismo segmento temporal de un MP4.

## 20.1 Flujo completo

```text
MP4 original
    |
    +-------------------------+
    |                         |
    v                         v
frames PNG                audio WAV estéreo
    |                         |
    v                         v
MBB-GS video              Gabor audio
    |                         |
    v                         |
checkpoint                   |
    |                         |
    v                         |
pruning adaptativo           |
    |                         |
    v                         |
UINT16 SAFE / ALL            |
    |                         |
    v                         v
video reconstruido       recon_stereo.wav
    |                         |
    +------------+------------+
                 |
                 v
            FFmpeg mux
                 |
                 v
          audiovisual final
```

## 20.2 Etapas

El pipeline:

1. inspecciona el MP4 con `ffprobe`;
2. extrae los frames del intervalo seleccionado;
3. extrae el audio estéreo del mismo intervalo;
4. genera configuraciones runtime;
5. entrena el modelo de video;
6. entrena el modelo de audio Gabor;
7. ejecuta pruning adaptativo;
8. genera `UINT16 SAFE`;
9. genera `UINT16 ALL`;
10. reconstruye ambos paquetes;
11. mide la desviación respecto al baseline;
12. opcionalmente genera archivos `.7z`;
13. selecciona el modo de video final;
14. combina video y audio;
15. guarda un resumen JSON y TXT.

## 20.3 Configuración maestra

Un ejemplo incluido en el repositorio sigue esta estructura:

```json
{
  "nombre_pipeline": "rockyourbody_10s_1ep",
  "mp4_original": "data/videos/rockyourbody.mp4",
  "inicio_segundos": 0.0,
  "duracion_segundos": 10.0,
  "fps": 30,
  "resolucion": [720, 1280],
  "device": "cuda",

  "nombre_clip": "rockyourbody_10s_1ep_clip",
  "config_video": "video.json",
  "config_audio": "audio.json",

  "forzar_extraccion": true,

  "pruning": {
    "inicio_pct": 10,
    "paso_pct": 5,
    "min_pct": 10,
    "max_pct": 50,
    "psnr_global_min": 65.0
  },

  "video_cuantizacion": {
    "generar_uint16_safe": true,
    "generar_uint16_all": true,
    "usar_para_video_final": "uint16_safe"
  },

  "video_final": {
    "bitrate_audio_kbps": 192
  }
}
```

## 20.4 Ejecutar

Ejemplo real de estructura existente:

```powershell
python .\scripts\run_pipeline_video_audio.py `
    --config ".\configs\thriller_10s_1ep\pipeline.json"
```

o:

```powershell
python .\scripts\run_pipeline_video_audio.py `
    --config ".\configs\rockyourbody_10s_1ep\pipeline.json"
```

## 20.5 Dependencias externas

El pipeline requiere que estén disponibles en `PATH`:

```text
ffmpeg
ffprobe
```

Para compresión `.7z`, `7z` es opcional.

Verificación:

```powershell
ffmpeg -version
ffprobe -version
7z
```

---

# 21. Configuraciones incluidas

El directorio:

```text
configs/
```

no contiene únicamente ejemplos arbitrarios. Conserva configuraciones utilizadas para distintas etapas experimentales.

## 21.1 Ablación de pérdida

```text
fase1_baseline.json
fase1_edge.json
fase1_motion.json
fase1_temporal.json
fase1_l1_mse.json
fase1_combo.json
...
```

## 21.2 Agregación entre frames

```text
fase2_qframe1.json
fase2_qframe2.json
fase2_qframe4.json
fase2_qframe8.json
fase2_maxframe.json
```

## 21.3 Agregación a nivel píxel

```text
fase3_qpixel1.json
fase3_qpixel2.json
fase3_qpixel4.json
```

## 21.4 Comparación de bases

```text
configs/comparacion_bases/
```

contiene configuraciones para reproducir:

```text
Chebyshev
vs.
monomial
```

## 21.5 Capacidad del modelo

```text
configs/comparacion_capacidad/
```

contiene experimentos que modifican:

```text
número de gaussianas
grados temporales
épocas
```

## 21.6 Escalamiento de la configuración `motion`

```text
motion_150k_0400ep.json
motion_150k_0800ep.json
motion_150k_1200ep.json
motion_150k_1600ep.json
ganador_motion_200k_1200ep.json
```

## 21.7 Gabor

```text
configs/gabor/
```

incluye:

```text
smoke tests
ablaciones de loss
ablaciones por número de átomos
mono
estéreo LR
estéreo Mid-Side
configuraciones de alta capacidad
```

## 21.8 Pipeline audiovisual

Ejemplos:

```text
configs/rockyourbody_10s_1ep/
configs/thriller_10s_1ep/
```

Cada carpeta contiene:

```text
pipeline.json
video.json
audio.json
```

---

# 22. Scripts principales

Esta sección distingue herramientas activas de material histórico.

## 22.1 Entrenamiento y carga

| Script | Función |
|---|---|
| `scripts/train.py` | entrenamiento principal de video |
| `scripts/_carga_checkpoint.py` | cargador central de checkpoints y base temporal |
| `scripts/run_tests_secuencial.py` | ejecución de configuraciones de ablación; no es la suite oficial de tests |

## 22.2 Preparación y video

| Script | Función |
|---|---|
| `extraer_clips_720p.py` | extrae clips a PNG |
| `extraer_frames_originales.py` | extracción auxiliar de frames originales |
| `extraer_video_gt.py` | genera video ground truth de un rango |
| `frames_a_video.py` | convierte frames PNG a MP4 |
| `extraer.py` | extractor histórico/auxiliar; para el flujo reproducible se prefiere `extraer_clips_720p.py` |

## 22.3 Reconstrucción e interpolación

| Script | Función |
|---|---|
| `regenerar_clip_desde_checkpoint_streaming.py` | reconstrucción frame a frame desde checkpoint |
| `regenerar_fps_interpolado.py` | evalúa la representación en una frecuencia temporal diferente |
| `analizar_interpolacion_fps.py` | análisis de resultados de interpolación |

## 22.4 Métricas

| Script | Función |
|---|---|
| `comparar_frames_psnr.py` | compara dos carpetas de frames |
| `lpips_post_hoc.py` | calcula LPIPS después de generar resultados |

## 22.5 Pruning y cuantización

| Script | Función |
|---|---|
| `viz_gaussian_stats.py` | calcula estadísticas temporales por gaussiana |
| `viz_render_subset.py` | renderiza subconjuntos de gaussianas |
| `viz_prune_checkpoint_by_stats.py` | crea checkpoint podado |
| `run_binary_pruning_adaptativo.py` | búsqueda adaptativa de porcentaje de pruning |
| `pack_checkpoint_uint16.py` | paquete UINT16 SAFE/selectivo |
| `unpack_checkpoint_uint16.py` | reconstruye paquete SAFE |
| `pack_checkpoint_uint16_all.py` | cuantiza todos los tensores float aplicables |
| `unpack_checkpoint_uint16_all.py` | reconstruye paquete ALL |

## 22.6 Visualizaciones

| Script | Función |
|---|---|
| `viz_atributos_gaussiana_tiempo.py` | evolución temporal de atributos |
| `viz_elipses_velocidades.py` | visualización espacial/dinámica |
| `viz_rank_gaussianas_rango.py` | ranking sobre un intervalo temporal |
| `viz_tira_evolucion.py` | evolución visual en secuencia |
| `viz_trayectorias_marcadores.py` | trayectorias de gaussianas |
| `viz_trayectorias_marcadores_rango.py` | trayectorias dentro de un rango temporal |

## 22.7 Audio Gabor

| Script | Función |
|---|---|
| `GABOR_SCRIPTS/train_gabor.py` | entrenamiento mono |
| `GABOR_SCRIPTS/train_gabor_stereo.py` | entrenamiento estéreo LR o MS |
| `GABOR_SCRIPTS/visualizar.py` | visualización de resultados Gabor |

## 22.8 Pipeline integrado

| Script | Función |
|---|---|
| `run_pipeline_video_audio.py` | flujo audiovisual completo |

## 22.9 Material histórico

```text
scripts/OLD_TESTS_PRUNNING/
cuda/raster_cuda/tests/
```

contienen pruebas, benchmarks y experimentos de etapas anteriores del desarrollo.

No constituyen la suite oficial actual de validación.

No es necesario ejecutarlos para comprobar el estado normal del repositorio.

---

# 23. Tests oficiales

La suite oficial se encuentra en:

```text
tests/
```

Actualmente contiene:

```text
tests/conftest.py
tests/test_bases_modelo.py
tests/test_checkpoint_loader.py
tests/test_gabor_gradientes.py
```

`pyproject.toml` configura `pytest` para buscar tests en ese directorio.

## 23.1 Ejecutar

```powershell
pytest -q
```

En la revisión de limpieza del repositorio:

```text
5 passed
```

## 23.2 Qué se valida

La suite cubre:

```text
construcción de base Chebyshev
construcción de base monomial
evaluación básica del modelo
metadata del modelo
carga de checkpoint Chebyshev
carga de checkpoint monomial
gradientes analíticos del modelo Gabor frente a autograd
```

## 23.3 Compilación sintáctica

También puede comprobarse todo el código Python:

```powershell
python -m compileall -q .\src .\scripts .\tests
```

## 23.4 Tests CUDA antiguos

El directorio `cuda/raster_cuda/tests/` no debe utilizarse como suite general de validación.

Ese directorio conserva tests y benchmarks históricos con imports correspondientes a versiones anteriores del rasterizador.

La suite actual que debe ejecutarse con:

```powershell
pytest -q
```

está limitada a:

```text
tests/
```

---

# 24. Resultados representativos

Los siguientes resultados corresponden a experimentos documentados en la tesis y sirven para contextualizar las decisiones implementadas.

No representan una garantía de calidad para cualquier video: la calidad depende del contenido, duración, resolución, capacidad del modelo y entrenamiento.

## 24.1 Chebyshev vs. monomial

Experimento con:

```text
8000 gaussianas
360 épocas
mismos grados
mismo clip
misma loss
```

| Base | PSNR promedio | PSNR mínimo | SSIM | PSNR temporal | Std. PSNR |
|---|---:|---:|---:|---:|---:|
| Monomial | 23.16 dB | 20.19 dB | 0.750 | 30.61 dB | 2.61 |
| Chebyshev | 29.95 dB | 27.70 dB | 0.926 | 34.04 dB | 1.09 |

Chebyshev obtuvo mayor calidad y mayor estabilidad temporal.

## 24.2 Ablación de pérdida

| Loss | PSNR prom. | PSNR mín. | PSNR p5 | SSIM | LPIPS | PSNR temp. |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 31.83 | 28.10 | 29.35 | 0.962 | 0.0594 | 35.17 |
| Edge | 31.88 | 28.05 | 29.30 | 0.963 | 0.0574 | 35.23 |
| Motion | 33.24 | 30.32 | 31.12 | 0.965 | 0.0483 | 36.88 |
| Temporal | 31.93 | 28.32 | 29.42 | 0.963 | 0.0588 | 35.32 |

La variante `motion` fue la seleccionada para los experimentos posteriores.

## 24.3 Agregación entre frames

| q | PSNR | PSNR mín. | SSIM | LPIPS | PSNR temp. |
|---:|---:|---:|---:|---:|---:|
| 1 | 33.24 | 30.32 | 0.965 | 0.048 | 36.90 |
| 2 | 29.77 | 27.43 | 0.942 | 0.099 | 33.55 |
| 4 | 21.19 | 16.84 | 0.859 | 0.336 | 30.33 |
| 8 | 14.70 | 13.37 | 0.517 | 0.821 | 30.03 |

El promedio convencional (`q = 1`) fue la opción más estable.

## 24.4 Interpolación temporal

En uno de los experimentos:

```text
90000 gaussianas
180 frames originales
720p
2200 épocas
Chebyshev
loss motion
```

se obtuvo:

| Métrica | Resultado |
|---|---:|
| PSNR promedio | 40.97 dB |
| PSNR mínimo | 36.83 dB |
| SSIM | 0.985 |
| LPIPS | 0.028 |
| PSNR temporal | 43.78 dB |

El mismo modelo pudo evaluarse en instantes intermedios para generar versiones de mayor FPS.

## 24.5 Reducción del modelo

Experimento de referencia:

```text
150000 gaussianas iniciales
750 frames
720p
1600 épocas
grados 100/80/30/12/6/4
loss motion
```

Reconstrucción completa:

| Métrica | Resultado |
|---|---:|
| PSNR promedio | 35.94 dB |
| PSNR mínimo | 27.40 dB |
| PSNR p5 | 30.71 dB |
| SSIM | 0.970 |
| LPIPS | 0.048 |
| PSNR temporal | 39.59 dB |

Una prueba de pruning adaptativo del `20 %` redujo:

```text
150000 -> 120000 gaussianas
```

y la posterior cuantización permitió obtener aproximadamente:

```text
236 MB -> 85 MB
```

con un PSNR cercano a:

```text
63 dB
```

al comparar la versión reducida/cuantizada contra la reconstrucción completa.

## 24.6 Audio Gabor

Ejemplo estéreo de alta capacidad:

```text
9 segundos
44100 Hz
160000 átomos
96000 Mid
64000 Side
6000 épocas
```

Resultados:

| Métrica | Resultado |
|---|---:|
| SNR estéreo | 27.78 dB |
| SI-SDR estéreo | 27.77 dB |
| PSNR estéreo | 44.24 dB |
| LSD promedio | 4.57 dB |
| Mel-L1 promedio | 0.1985 |
| MR-STFT promedio | 1.8532 |
| SNR Mid | 28.50 dB |
| SNR Side | 20.01 dB |

---

# 25. Uso de memoria y entrenamiento a alta resolución

Una secuencia de video completa en `float32` puede consumir gran cantidad de memoria.

Por ejemplo, un clip:

```text
T frames
720 x 1280
RGB
float32
```

requiere aproximadamente:

```math
T \cdot 720 \cdot 1280 \cdot 3 \cdot 4
```

bytes únicamente para almacenar los frames.

Por esta razón el framework incorpora rutas para:

```text
frames CPU uint8
transferencia frame a frame
render streaming
métricas streaming
liberación periódica de cache CUDA
```

Configuración recomendada para entrenamientos grandes:

```json
"frames_en_cpu": true,
"frames_en_gpu_uint8": false,
"evitar_render_completo_en_train": true,
"usar_metricas_streaming": true
```

Si aparece OOM, reducir uno o varios de:

```text
n_gaussianas_inicial
resolución
cantidad de frames
sub_batch_frames
grados temporales
```

También debe verificarse que el problema no sea RAM del sistema. Mantener frames en CPU reduce VRAM, pero aumenta el uso de RAM.

---

# 26. Checkpoints y formato interno

Un checkpoint de video contiene principalmente:

```python
{
    "state_dict_coefs": ...,
    "config": ...
}
```

## 26.1 Metadata

`state_dict_coefs` almacena además información necesaria para reconstruir el modelo:

```text
N
H
W
n_frames
grados
```

junto a los coeficientes.

## 26.2 Compatibilidad

El cargador central intenta obtener la información del checkpoint antes de depender del archivo de configuración externo.

Esto permite reconstruir modelos incluso cuando:

```text
el nombre de la carpeta cambió
el config original fue movido
el número de gaussianas cambió por pruning
```

## 26.3 Checkpoint completo y checkpoint podado

El checkpoint inmediatamente después del entrenamiento se guarda normalmente como:

```text
checkpoint_final.pt
```

Un checkpoint generado después de pruning puede tener un número de gaussianas menor.

Los scripts deben utilizar la metadata interna `N` del checkpoint y no asumir que sigue siendo igual a `n_gaussianas_inicial`.

---

# 27. Reproducibilidad

## 27.1 Semilla

Las configuraciones incluyen:

```json
"seed": 42
```

El código inicializa las semillas relevantes de PyTorch.

## 27.2 Config utilizada

Cada experimento almacena una copia de:

```text
config_usada.json
```

para registrar los parámetros de ejecución.

## 27.3 Metadata del clip

También se genera:

```text
info_clip.json
```

con información del video utilizado.

## 27.4 Control de versiones

Los binarios CUDA, checkpoints, datasets y outputs no deben sustituir el control de versiones del código.

Para reproducir un resultado de tesis se recomienda registrar:

```text
commit Git
config JSON
nombre del clip
cantidad de frames
resolución
versión de PyTorch
versión CUDA
GPU
```

## 27.5 Finales de línea

El repositorio incluye:

```text
.gitattributes
```

para mantener finales de línea consistentes entre Windows y Linux.

Los archivos de código y configuración se normalizan principalmente a LF, mientras que scripts específicos de Windows pueden utilizar CRLF.

---

# 28. Solución de problemas

## 28.1 `ModuleNotFoundError: No module named 'raster_cuda'`

La extensión no fue compilada, no está en el directorio esperado o fue compilada para otro entorno de Python.

Recompilar:

```powershell
Push-Location .\cuda\raster_cuda
python setup.py build_ext --inplace
Pop-Location
```

Después:

```powershell
python -c "import torch, sys; sys.path.insert(0, r'cuda/raster_cuda'); import raster_cuda; print('OK')"
```

En Windows es recomendable importar `torch` antes de importar directamente una extensión PyTorch/CUDA para que sus DLL estén cargadas.

## 28.2 El `.pyd` existe pero no importa

Comprobar:

```powershell
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
nvidia-smi
```

Un archivo:

```text
*.cp310-win_amd64.pyd
```

está ligado a una versión y ABI de Python concretas.

No debe copiarse un binario compilado en otro entorno esperando compatibilidad automática.

## 28.3 Error al compilar CUDA

Verificar:

```text
CUDA Toolkit
nvcc
compilador C++
PyTorch
arquitectura GPU
```

En Windows, PyTorch CUDA extensions requieren una toolchain C++ compatible.

En Linux/SLURM, verificar que los módulos cargados correspondan al entorno utilizado.

## 28.4 `torch.cuda.is_available() == False`

Ejecutar:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Si PyTorch no detecta CUDA, el problema debe resolverse antes de compilar o entrenar el rasterizador.

## 28.5 Out Of Memory

Para video:

```json
"frames_en_cpu": true,
"evitar_render_completo_en_train": true,
"usar_metricas_streaming": true
```

También puede reducirse:

```text
N
resolución
frames
grado
sub-batch
```

En un cluster, comprobar tanto memoria GPU como RAM asignada al job.

## 28.6 `ffmpeg` no encontrado

Verificar:

```powershell
ffmpeg -version
ffprobe -version
```

El pipeline audiovisual necesita ambos programas accesibles mediante `PATH`.

## 28.7 `7z` no encontrado

La cuantización y reconstrucción siguen siendo utilizables.

`7z` se utiliza únicamente como etapa adicional de compresión sin pérdida del paquete.

## 28.8 `pytest` intenta ejecutar tests CUDA antiguos

La validación oficial debe ser:

```powershell
pytest -q
```

y `pyproject.toml` debe contener la configuración que limita descubrimiento a:

```text
tests/
```

No utilizar como suite general:

```text
cuda/raster_cuda/tests/
```

## 28.9 Audio Gabor no encuentra CUDA

Compilar:

```powershell
Push-Location .\cuda\gabor_audio_cuda
python setup.py build_ext --inplace
Pop-Location
```

El propio wrapper permite indicar manualmente la ubicación mediante:

```text
RUTA_GABOR_AUDIO_CUDA
```

## 28.10 Diferencias pequeñas entre Gabor CUDA y PyTorch

El kernel CUDA utiliza una ventana local controlada por `k_sigma`.

El fallback PyTorch denso puede evaluar colas gaussianas más allá de esa ventana.

Por ello pueden aparecer diferencias numéricas pequeñas sin que esto implique necesariamente un error del kernel.

---

# 29. Limitaciones actuales

## 29.1 El entrenamiento depende de CUDA para el flujo principal de video

El modelo matemático puede inspeccionarse en CPU y la suite unitaria no requiere entrenar un video completo, pero el rasterizador de producción está diseñado para GPU NVIDIA.

## 29.2 Interpolación no equivale a un método VFI especializado

La interpolación es una consecuencia de la representación continua.

Esto permite generar frames intermedios sin una red adicional, pero no garantiza que todas las trayectorias entre frames observados sean visualmente perfectas.

En movimientos rápidos pueden aparecer:

```text
rastros
deformaciones
borrosidad
trayectorias intermedias imperfectas
```

## 29.3 El pruning depende del video

La redundancia aprendida no es idéntica para todos los clips.

Un porcentaje de pruning aceptable en un modelo puede degradar significativamente otro.

Por eso el pipeline valida cada porcentaje mediante reconstrucción y PSNR.

## 29.4 UINT16 es post-training

La cuantización implementada es una etapa posterior al entrenamiento.

El modelo no se entrena actualmente simulando cuantización durante el forward.

## 29.5 Gabor es una extensión complementaria

La parte de audio demuestra que el principio de primitivas explícitas optimizables puede trasladarse a otra modalidad, pero el desarrollo central del proyecto corresponde al video temporal con gaussianas 2D.

---

# 30. Relación con la tesis

El repositorio implementa los componentes computacionales utilizados en la investigación MBB-GS / Moving Gaussians:

```text
representación temporal con Chebyshev
rasterizador diferenciable CUDA
entrenamiento de video
ablaciones de capacidad y loss
reconstrucción
interpolación temporal
pruning
cuantización
audio Gabor
pipeline audiovisual
métricas y visualizaciones
```

El flujo reproducible general es:

```text
1. clonar repositorio
2. crear entorno Python
3. instalar dependencias
4. instalar el proyecto con pip install -e .
5. compilar raster_cuda
6. compilar gabor_audio_cuda si se utilizará audio
7. preparar frames o audio
8. seleccionar un config
9. entrenar
10. reconstruir
11. calcular métricas
12. opcionalmente aplicar pruning
13. opcionalmente cuantizar
14. opcionalmente interpolar
15. opcionalmente ejecutar el pipeline audiovisual
```

Para una comprobación rápida del estado del código:

```powershell
python -m compileall -q .\src .\scripts .\tests
pytest -q
```

La representación final de video puede resumirse como:

```text
una población fija de gaussianas 2D
+
atributos que evolucionan con Chebyshev
+
rasterización diferenciable
=
una representación explícita y continua del video en el tiempo
```

La extensión de audio sigue un principio relacionado:

```text
una población de átomos de Gabor
+
parámetros optimizables de tiempo, frecuencia, duración, amplitud y fase
=
una representación explícita de la waveform
```

El objetivo del repositorio no es reemplazar los formatos de video o audio convencionales como producto final, sino proporcionar una implementación reproducible para estudiar **representaciones multimedia explícitas, temporales, diferenciables y optimizables**.

---

## Comandos de referencia rápida

Instalación:

```powershell
conda create -n mbb-gs python=3.10 -y
conda activate mbb-gs
pip install -r requirements.txt
pip install -e .
```

Compilar video CUDA:

```powershell
Push-Location .\cuda\raster_cuda
python setup.py build_ext --inplace
Pop-Location
```

Compilar Gabor CUDA:

```powershell
Push-Location .\cuda\gabor_audio_cuda
python setup.py build_ext --inplace
Pop-Location
```

Extraer frames:

```powershell
python .\scripts\extraer_clips_720p.py `
    --video ".\data\videos\video.mp4" `
    --nombre_clip "mi_clip" `
    --inicio_seg 0 `
    --duracion_seg 10 `
    --fps 30 `
    --H 720 `
    --W 1280 `
    --forzar
```

Entrenar video:

```powershell
python .\scripts\train.py `
    --config ".\configs\ganador_motion_200k_1200ep.json"
```

Reconstruir:

```powershell
python .\scripts\regenerar_clip_desde_checkpoint_streaming.py `
    --checkpoint ".\outputs\mi_experimento\checkpoints\checkpoint_final.pt" `
    --salida ".\outputs\mi_experimento\recon" `
    --device cuda `
    --crear_video `
    --fps 30
```

Interpolar:

```powershell
python .\scripts\regenerar_fps_interpolado.py `
    --checkpoint ".\outputs\mi_experimento\checkpoints\checkpoint_final.pt" `
    --salida ".\outputs\mi_experimento\fps120" `
    --fps_origen 30 `
    --fps_salida 120 `
    --device cuda `
    --forzar
```

Pruning:

```powershell
python .\scripts\run_binary_pruning_adaptativo.py `
    --exp ".\outputs\mi_experimento" `
    --checkpoint ".\outputs\mi_experimento\checkpoints\checkpoint_final.pt" `
    --baseline_frames ".\outputs\mi_experimento\frames_renderizados" `
    --fps 30 `
    --device cuda `
    --inicio_pct 10 `
    --paso_pct 5 `
    --min_pct 10 `
    --max_pct 50 `
    --psnr_min 65 `
    --crear_video_ganador
```

Gabor estéreo:

```powershell
python .\scripts\GABOR_SCRIPTS\train_gabor_stereo.py `
    --config ".\configs\gabor\gabor_rock_31_40_stereo_MS_160k_6000ep.json"
```

Pipeline audiovisual:

```powershell
python .\scripts\run_pipeline_video_audio.py `
    --config ".\configs\thriller_10s_1ep\pipeline.json"
```

Tests:

```powershell
python -m compileall -q .\src .\scripts .\tests
pytest -q
```

---

**MBB-GS / Moving Gaussians**

Repositorio de investigación para representación temporal de video mediante 2D Gaussian Splatting y representación complementaria de audio mediante átomos de Gabor.
