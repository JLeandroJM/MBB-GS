# Como publicar la wiki

Las paginas viven en `wiki/` como archivos Markdown. Se versionan aqui para que
la documentacion cambie junto con el codigo; la wiki de GitHub es solo la copia
publicada.

## Requisito previo

En GitHub, en el repositorio: Settings -> Features -> marcar **Wikis**. Luego
crear una primera pagina cualquiera desde la web, porque hasta que exista una
pagina el repositorio de la wiki no se crea.

## Opcion A: copiar y pegar

En la web, "New Page" por cada archivo de `wiki/`. El **titulo de la pagina
debe ser exactamente el nombre del archivo sin `.md`**, con los guiones tal
cual:

| Archivo | Titulo de la pagina |
| --- | --- |
| `Home.md` | `Home` |
| `Model-and-Temporal-Representation.md` | `Model-and-Temporal-Representation` |
| `CUDA-Rasterizer.md` | `CUDA-Rasterizer` |
| `Installation.md` | `Installation` |
| `Data-Preparation.md` | `Data-Preparation` |
| `Training.md` | `Training` |
| `Loss-Functions.md` | `Loss-Functions` |
| `Reconstruction-and-Interpolation.md` | `Reconstruction-and-Interpolation` |
| `Metrics.md` | `Metrics` |
| `Pruning-and-Quantization.md` | `Pruning-and-Quantization` |
| `Gabor-Audio.md` | `Gabor-Audio` |
| `Audiovisual-Pipeline.md` | `Audiovisual-Pipeline` |
| `Results.md` | `Results` |
| `Reproducibility.md` | `Reproducibility` |
| `Troubleshooting.md` | `Troubleshooting` |

Los enlaces entre paginas son de la forma `[Training](Training)`, que es como
resuelve la wiki de GitHub. Si un titulo no coincide exactamente, los enlaces
que apuntan a esa pagina quedan rotos.

`Home` es la portada. No hace falta crear un `_Sidebar`: GitHub lista las
paginas automaticamente.

## Opcion B: un push (bastante mas rapido)

La wiki es un repositorio git aparte:

```bash
git clone https://github.com/<usuario>/<repo>.wiki.git
cp wiki/*.md <repo>.wiki/
cd <repo>.wiki
git add .
git commit -m "Publish wiki"
git push
```

Para actualizarla despues, repetir la copia y el push.

## Antes de publicar

Reemplazar los marcadores `TODO_REPO_URL`, `TODO_DRIVE_URL` y
`TODO_THESIS_URL` por las URLs reales. Aparecen en `README.md`,
`CITATION.cff`, `wiki/Home.md`, `wiki/Installation.md` y `wiki/Results.md`.

## Nota

Este archivo es documentacion interna del repositorio de trabajo. No es una
pagina de la wiki y no se copia al repositorio oficial.
