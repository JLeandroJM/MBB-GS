# Prompt de inicializacion para el repositorio oficial

Pegar como primer mensaje en una sesion nueva de Claude Code abierta en
`MBB-GS-Moving-Gaussians`. El trabajo de ordenamiento ya esta hecho; esa sesion
es para refinar el README y la wiki.

---

Estoy trabajando en el repositorio oficial de mi tesis, el que vamos a enviar a
revision junto con un paper. El codigo ya esta ordenado y funcional: **no
quiero reorganizarlo**. Lo que quiero trabajar contigo es el README y la wiki.

## Que es el proyecto

MBB-GS (Moving Gaussians) representa senales multimedia con primitivas
explicitas en vez de arreglos de frames o muestras.

Un video es una poblacion fija de N gaussianas 2D. Cada atributo de cada
gaussiana (posicion `mu`, opacidad, color, escala, rotacion `theta` y
profundidad) es un polinomio de Chebyshev en el tiempo normalizado a [-1, 1]:

    p_i(t) = a_{i,0} + sum_{k=1..K_p} a_{i,k} * T_k(tau(t))

El modelo guarda coeficientes, no frames. Para renderizar el frame t se evaluan
los polinomios y las gaussianas resultantes pasan por un rasterizador CUDA
diferenciable. De ahi salen dos propiedades: la interpolacion temporal es
gratis (se evalua el modelo entre frames originales, sin entrenar otra red) y
el modelo se puede podar y cuantizar despues de entrenar.

Como extension complementaria, el audio se representa con atomos de Gabor
(gaussianas moduladas por una sinusoide) directamente sobre la waveform, sin
STFT y sin polinomios temporales.

Autores: Jose Leandro Machaca Soloaga y Mauro Ianfranco Bobadilla Castillo.
Asesor: Eric Biagioli. Universidad de Ingenieria y Tecnologia (UTEC), 2026.
Los experimentos corrieron en el cluster HPC Khipu de UTEC.

El paper cubre las tres secciones: video, audio y el framework integrado.

## Estructura del repositorio

    src/gs2d_video/    modelo de video: bases temporales, modelo de gaussianas,
                       perdidas, rasterizador CUDA, loop de entrenamiento,
                       metricas, entrada/salida
    src/gs2d_gabor/    modelo de audio: atomos de Gabor, perdidas, render CUDA
    cuda/              extensiones CUDA, se compilan en cada entorno
    scripts/           entry points, agrupados por rol
    configs/           un JSON por experimento, agrupados por experimento del paper
    jobs/              scripts de Slurm usados en Khipu
    tests/             suite de tests (8, corren en CPU, sin GPU)
    results/           registros livianos de 33 experimentos
    wiki/              fuente de la documentacion

`scripts/` tiene subcarpetas por rol: `data`, `reconstruction`, `compression`,
`metrics`, `visualization`, `pipeline`, `audio`, con `train.py` en la raiz como
unico entry point de entrenamiento. Cada subcarpeta tiene su README.

## Que ya se hizo (no rehacer)

- `scripts/` reorganizado por rol, con todas las rutas de subprocess, los
  `.sbatch` y la documentacion actualizadas.
- `configs/` agrupado por experimento. `configs/README.md` mapea cada carpeta
  con la tabla del paper que produce.
- El cargador de checkpoints vive en `src/gs2d_video/io/checkpoints.py` y es la
  unica ruta para interpretar un checkpoint.
- El proyecto se instala con `pip install -e .`; no hay parches de `sys.path`.
- `requirements.txt` lista dependencias directas y funciona en cualquier
  plataforma. `requirements-tesis-khipu.txt` congela el entorno exacto de
  Khipu, con ruedas `+cu126`.
- README reescrito, corto y en ingles. `wiki/` con 15 paginas en ingles.
- `LICENSE` (MIT) y `CITATION.cff`.
- Se corrigio un bug: el pruning post-entrenamiento evaluaba la opacidad
  siempre en base Chebyshev, ignorando la base con la que se entreno. Hay un
  test de regresion.

## Convenciones que hay que respetar

1. **README y wiki en ingles. Codigo, comentarios, claves de config y nombres
   de archivo en espanol.** Es deliberado: las carpetas orientan al lector, el
   codigo es interno. `wiki/Home.md` lo explica al lector.
2. **Sin emojis** en ningun documento.
3. **No renombrar archivos** de `scripts/`, ni las carpetas de salida que crea
   el entrenamiento (`frames_renderizados`, `checkpoints`, `logs`): romperia
   los comandos documentados y la lectura de los experimentos ya generados.
4. **No tocar los kernels CUDA ni la matematica** (perdidas, representacion
   temporal, activaciones, pruning, cuantizacion) salvo bug demostrable.
5. Chebyshev es el camino principal; monomial existe solo para la ablacion y
   compatibilidad con checkpoints antiguos.
6. `exponente_frame = 1` es el valor validado experimentalmente.
7. Antes de documentar un comando, **verificar los flags contra el `argparse`
   real del script**. Ya aparecieron dos comandos con flags inexistentes en la
   documentacion anterior.

## Convenciones de la wiki

Los archivos estan en `wiki/`. El titulo de cada pagina en GitHub debe ser
exactamente el nombre del archivo sin `.md`, con los guiones tal cual, porque
los enlaces entre paginas son de la forma `[Training](Training)`. `Home` es la
portada.

Paginas actuales: `Home`, `Model-and-Temporal-Representation`,
`CUDA-Rasterizer`, `Installation`, `Data-Preparation`, `Training`,
`Loss-Functions`, `Reconstruction-and-Interpolation`, `Metrics`,
`Pruning-and-Quantization`, `Gabor-Audio`, `Audiovisual-Pipeline`, `Results`,
`Reproducibility`, `Troubleshooting`.

Si agregas o renombras una pagina, verifica que todos los enlaces internos
sigan resolviendo.

## Resultados reportados (no inventar numeros)

Base temporal, 8000 gaussianas, 360 epocas, todo lo demas fijo:

| Base | PSNR | PSNR min | SSIM | PSNR temporal |
|---|---|---|---|---|
| Monomial | 23.16 | 20.19 | 0.750 | 30.61 |
| Chebyshev | 29.95 | 27.70 | 0.926 | 34.04 |

Ablacion de perdida, 600 frames, 720p, 100k gaussianas, 800 epocas:

| Perdida | PSNR | PSNR min | PSNR p5 | SSIM | LPIPS | PSNR temporal |
|---|---|---|---|---|---|---|
| Baseline | 31.83 | 28.10 | 29.35 | 0.962 | 0.0594 | 35.17 |
| Edge | 31.88 | 28.05 | 29.30 | 0.963 | 0.0574 | 35.23 |
| Temporal | 31.93 | 28.32 | 29.42 | 0.963 | 0.0588 | 35.32 |
| Motion | 33.24 | 30.32 | 31.12 | 0.965 | 0.0483 | 36.88 |

Agregacion entre frames (exponente q): q=1 da 33.24 dB, q=2 da 29.77,
q=4 da 21.19, q=8 da 14.70. La calidad cae de forma monotona.

Interpolacion temporal, 90k gaussianas, 180 frames, 720p, 2200 epocas:
PSNR 40.97 dB, SSIM 0.985, LPIPS 0.028, PSNR temporal 43.78 dB.

Reduccion del modelo, 150k gaussianas, 750 frames, 720p, 1600 epocas:
reconstruccion completa 35.94 dB. Poda adaptativa 20 % lleva a 120k gaussianas
y la cuantizacion uint16 de ~236 MB a ~85 MB, con ~63 dB contra la
reconstruccion completa.

Audio Gabor, 9 s estereo 44.1 kHz, 160k atomos (96k Mid, 64k Side), 6000
epocas: SNR estereo 27.78 dB, SI-SDR 27.77 dB, PSNR 44.24 dB, LSD 4.57 dB.

La receta final de video es perdida `motion` con `lambda_motion: 2.0`,
`lambda_dssim: 0.25` y `exponente_frame: 1`.

El informe completo de la tesis esta en el repositorio de trabajo, en
`MBB-GS/documentacion/`. Este repositorio no lo incluye.

## Marcadores pendientes

Quedan tres marcadores `TODO_` por completar. Si los ves y todavia no te di el
valor, preguntame antes de inventar una URL:

- `TODO_REPO_URL` en README.md, wiki/Home.md, wiki/Installation.md, CITATION.cff
- `TODO_DRIVE_URL` en README.md y wiki/Results.md (Drive publico con los videos
  y resultados pesados)
- `TODO_THESIS_URL` en CITATION.cff

## Lo que nunca debe entrar a este repositorio

Los videos y el audio son grabaciones comerciales y no se redistribuyen. Los
resultados pesados (frames, checkpoints, videos) van a un Drive publico
enlazado desde el README. Aqui solo van los registros livianos de cada
experimento, que ya estan en `results/`.

## Como validar

    pip install -e .
    pytest -q          # deben pasar 8

Los scripts de video fallan con `ModuleNotFoundError: No module named
'raster_cuda'` en una maquina sin CUDA compilada: es esperado, no es un bug.

## Lo que quiero de ti ahora

Empieza leyendo el README, `wiki/Home.md` y las paginas de la wiki para tener
el contexto. Despues dime que mejorarias antes de tocar nada: quiero que el
README y la wiki queden a la altura de un repositorio que acompana un paper en
revision. Trabaja de a poco y explicame cada cambio.
