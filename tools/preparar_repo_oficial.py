"""
Arma la carpeta del repositorio oficial copiando unicamente lo aprobado.

Este repositorio es el de trabajo: contiene material con derechos de autor,
resultados pesados y notas personales. El repositorio que se envia a revision
debe contener solo lo que participa en producir un resultado del paper.

Uso:
    python tools/preparar_repo_oficial.py --destino ../MBB-GS-oficial
    python tools/preparar_repo_oficial.py --destino ../MBB-GS-oficial --forzar

El script solo copia. No borra ni modifica nada de este repositorio.

Que copia:
  - codigo: src/, cuda/, scripts/, tests/
  - configuraciones: configs/
  - jobs de Slurm: jobs/, con el correo de notificacion sustituido
  - documentacion: README.md, wiki/, LICENSE, CITATION.cff
  - empaquetado: pyproject.toml, requirements*.txt, .gitignore, .gitattributes
  - resultados livianos extraidos de outputs_khipu/ a results/

Que NO copia, y por que:
  - data/            material con derechos de autor; se regenera con scripts/data/
  - outputs/, outputs_khipu/, comparacion_slowmo/   demasiado pesado; va al Drive
  - README_KHIPU.md, run.txt, rutas_importantes.txt notas de trabajo internas
  - cuda/raster_cuda/tests/                         benchmarks historicos que no pasan
  - documentacion/                                  el informe se enlaza, no se versiona
  - tools/                                          esta herramienta
"""

import argparse
import re
import shutil
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]

# Carpetas y archivos que se copian tal cual.
INCLUIR = [
    "src",
    "cuda",
    "scripts",
    "tests",
    "configs",
    "jobs",
    "wiki",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "requirements-tesis-khipu.txt",
    ".gitignore",
    ".gitattributes",
]

# Se excluye dentro de lo copiado.
EXCLUIR_DIRS = {
    "__pycache__", ".git", ".venv", "build", ".pytest_cache",
    "tests_historicos",
}
EXCLUIR_RUTAS = {
    "cuda/raster_cuda/tests",
}
EXCLUIR_SUFIJOS = {".pyc", ".pyd", ".so", ".o", ".obj", ".lib", ".exp", ".pt", ".pth"}

# Archivos livianos por experimento que si van al repositorio.
RESULTADOS_LIVIANOS = (
    "metricas.json",
    "metricas_por_frame.csv",
    "config_usada.json",
    "info_clip.json",
)
RESULTADOS_LOGS = ("log_entrenamiento.csv",)

CORREO = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CORREO_PLACEHOLDER = "TU_CORREO@utec.edu.pe"


def _excluido(ruta_rel: Path) -> bool:
    partes = ruta_rel.parts
    if any(p in EXCLUIR_DIRS for p in partes):
        return True
    if ruta_rel.suffix in EXCLUIR_SUFIJOS:
        return True
    ruta_txt = ruta_rel.as_posix()
    return any(ruta_txt == e or ruta_txt.startswith(e + "/") for e in EXCLUIR_RUTAS)


def copiar_arbol(origen: Path, destino: Path, contador: dict) -> None:
    for src in sorted(origen.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(RAIZ)
        if _excluido(rel):
            contador["omitidos"] += 1
            continue
        dst = destino / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        contador["copiados"] += 1


def sanitizar_jobs(destino: Path) -> int:
    """Sustituye correos reales en los .sbatch por un marcador."""
    n = 0
    for p in sorted((destino / "jobs").glob("*.sbatch")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        nuevo = CORREO.sub(CORREO_PLACEHOLDER, txt)
        if nuevo != txt:
            p.write_text(nuevo, encoding="utf-8")
            n += 1
    return n


def copiar_resultados_livianos(destino: Path) -> tuple[int, int]:
    """Extrae solo metricas y configs de cada experimento a resultados/<exp>/."""
    origen = RAIZ / "outputs_khipu"
    if not origen.is_dir():
        return 0, 0

    n_archivos = 0
    experimentos = set()

    for exp in sorted(p for p in origen.iterdir() if p.is_dir()):
        for nombre in RESULTADOS_LIVIANOS:
            src = exp / nombre
            if src.is_file():
                dst = destino / "results" / exp.name / nombre
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n_archivos += 1
                experimentos.add(exp.name)

        for nombre in RESULTADOS_LOGS:
            src = exp / "logs" / nombre
            if src.is_file():
                dst = destino / "results" / exp.name / nombre
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n_archivos += 1
                experimentos.add(exp.name)

    return len(experimentos), n_archivos


def escribir_readme_resultados(destino: Path, n_exp: int) -> None:
    ruta = destino / "results" / "README.md"
    if not ruta.parent.is_dir():
        return
    ruta.write_text(
        "# Experiment records\n\n"
        f"Lightweight records for {n_exp} experiments: the aggregated and\n"
        "per-frame metrics, the exact configuration each run used, and the clip\n"
        "metadata. They are here so the numbers reported in the paper can be\n"
        "checked without retraining anything.\n\n"
        "| File | Contents |\n"
        "| --- | --- |\n"
        "| `metricas.json` | aggregates and per-frame values, split pre/post pruning |\n"
        "| `metricas_por_frame.csv` | one row per frame |\n"
        "| `config_usada.json` | the exact configuration used |\n"
        "| `info_clip.json` | clip metadata: frames, resolution, seed |\n"
        "| `log_entrenamiento.csv` | per-epoch losses and timings |\n\n"
        "Rendered frames, videos and checkpoints are published separately; see\n"
        "the link in the main README.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--destino", required=True, help="carpeta del repositorio oficial")
    ap.add_argument("--forzar", action="store_true", help="permite un destino no vacio")
    ap.add_argument("--sin-resultados", action="store_true",
                    help="no copia los resultados livianos de outputs_khipu")
    args = ap.parse_args()

    destino = Path(args.destino).resolve()
    if destino == RAIZ:
        raise SystemExit("El destino no puede ser este mismo repositorio.")
    if destino.exists() and any(destino.iterdir()) and not args.forzar:
        raise SystemExit(f"El destino no esta vacio: {destino}\nUsa --forzar para continuar.")
    destino.mkdir(parents=True, exist_ok=True)

    contador = {"copiados": 0, "omitidos": 0}

    print(f"origen  : {RAIZ}")
    print(f"destino : {destino}\n")

    for nombre in INCLUIR:
        src = RAIZ / nombre
        if not src.exists():
            print(f"  aviso: no existe, se omite -> {nombre}")
            continue
        if src.is_dir():
            copiar_arbol(src, destino, contador)
        else:
            shutil.copy2(src, destino / nombre)
            contador["copiados"] += 1
        print(f"  copiado: {nombre}")

    n_jobs = sanitizar_jobs(destino)
    print(f"\n  jobs sanitizados (correo -> {CORREO_PLACEHOLDER}): {n_jobs}")

    if not args.sin_resultados:
        n_exp, n_arch = copiar_resultados_livianos(destino)
        escribir_readme_resultados(destino, n_exp)
        print(f"  resultados livianos: {n_arch} archivos de {n_exp} experimentos")

    print(f"\n  archivos copiados: {contador['copiados']}")
    print(f"  archivos omitidos por filtro: {contador['omitidos']}")

    pendientes = []
    for p in destino.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".cff", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "TODO_" in txt:
                pendientes.append(p.relative_to(destino))

    if pendientes:
        print("\n  marcadores TODO_ por completar antes de publicar:")
        for p in sorted(pendientes):
            print(f"    {p}")

    print(
        "\nSiguiente paso: inicializar el repositorio con historia nueva, no clonar\n"
        "este, para no arrastrar el material con derechos de autor que vive en el\n"
        "historial de git:\n"
        f"    cd {destino}\n"
        "    git init && git add . && git commit -m \"Initial release\"\n"
    )


if __name__ == "__main__":
    main()
