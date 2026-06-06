
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAIZ = Path(__file__).resolve().parents[2]


def leer_csv_log(ruta):
    filas = []

    if not ruta.exists():
        return filas

    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                filas.append({
                    "epoch": int(float(row["epoch"])),
                    "loss": float(row["loss"]),
                    "l1": float(row["l1"]),
                    "mse": float(row["mse"]),
                    "psnr_logmag": float(row["psnr_logmag"]),
                    "fase": row.get("fase", "base"),
                })
            except Exception:
                continue

    return filas


def plot_metric(out_png, filas_l, filas_r, clave, titulo, ylabel):
    n = min(len(filas_l), len(filas_r))

    if n <= 0:
        return

    ep = [filas_l[i]["epoch"] for i in range(n)]
    y_l = [filas_l[i][clave] for i in range(n)]
    y_r = [filas_r[i][clave] for i in range(n)]
    y_avg = [(y_l[i] + y_r[i]) / 2.0 for i in range(n)]

    plt.figure(figsize=(12, 6))
    plt.plot(ep, y_l, label="LEFT")
    plt.plot(ep, y_r, label="RIGHT")
    plt.plot(ep, y_avg, label="PROMEDIO", linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def guardar_resumen_csv(out_csv, filas_l, filas_r):
    n = min(len(filas_l), len(filas_r))

    if n <= 0:
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow([
            "epoch",
            "loss_left",
            "loss_right",
            "loss_promedio",
            "psnr_left",
            "psnr_right",
            "psnr_promedio",
            "l1_left",
            "l1_right",
            "l1_promedio",
            "mse_left",
            "mse_right",
            "mse_promedio",
            "fase_left",
            "fase_right",
        ])

        for i in range(n):
            l = filas_l[i]
            r = filas_r[i]

            w.writerow([
                l["epoch"],
                l["loss"],
                r["loss"],
                (l["loss"] + r["loss"]) / 2.0,
                l["psnr_logmag"],
                r["psnr_logmag"],
                (l["psnr_logmag"] + r["psnr_logmag"]) / 2.0,
                l["l1"],
                r["l1"],
                (l["l1"] + r["l1"]) / 2.0,
                l["mse"],
                r["mse"],
                (l["mse"] + r["mse"]) / 2.0,
                l.get("fase", "base"),
                r.get("fase", "base"),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimento", required=True)
    args = parser.parse_args()

    out_exp = RAIZ / "outputs" / "audio" / args.experimento

    left_log = out_exp / "LEFT" / "log_entrenamiento.csv"
    right_log = out_exp / "RIGHT" / "log_entrenamiento.csv"

    filas_l = leer_csv_log(left_log)
    filas_r = leer_csv_log(right_log)

    if not filas_l or not filas_r:
        raise RuntimeError(
            f"No se pudieron leer logs LEFT/RIGHT: {left_log} | {right_log}"
        )

    out_dir = out_exp / "graficas"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        out_dir / "loss_left_right_promedio.png",
        filas_l,
        filas_r,
        "loss",
        "Loss por epoch - LEFT vs RIGHT vs promedio",
        "loss",
    )

    plot_metric(
        out_dir / "psnr_left_right_promedio.png",
        filas_l,
        filas_r,
        "psnr_logmag",
        "PSNR logmag por epoch - LEFT vs RIGHT vs promedio",
        "PSNR logmag",
    )

    plot_metric(
        out_dir / "l1_left_right_promedio.png",
        filas_l,
        filas_r,
        "l1",
        "L1 por epoch - LEFT vs RIGHT vs promedio",
        "L1",
    )

    plot_metric(
        out_dir / "mse_left_right_promedio.png",
        filas_l,
        filas_r,
        "mse",
        "MSE por epoch - LEFT vs RIGHT vs promedio",
        "MSE",
    )

    guardar_resumen_csv(out_dir / "resumen_curvas.csv", filas_l, filas_r)

    print(f"OK graficas stereo: {out_dir}")


if __name__ == "__main__":
    main()
