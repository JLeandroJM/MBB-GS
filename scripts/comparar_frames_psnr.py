import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def calcular_psnr(a, b):
    diff = a - b
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    if mse <= 0:
        return float("inf"), mse, mae

    psnr = float(-10.0 * np.log10(mse))
    return psnr, mse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Carpeta A: baseline")
    parser.add_argument("--b", required=True, help="Carpeta B: pruning/ablacion")
    parser.add_argument("--out", default=None, help="CSV de salida")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    a_dir = Path(args.a)
    b_dir = Path(args.b)

    a_paths = sorted(a_dir.glob("frame_*.png"))
    b_paths = sorted(b_dir.glob("frame_*.png"))

    if args.max_frames is not None:
        a_paths = a_paths[:args.max_frames]
        b_paths = b_paths[:args.max_frames]

    n = min(len(a_paths), len(b_paths))

    if n == 0:
        raise RuntimeError(
            f"No encontre frames para comparar.\n"
            f"A: {a_dir} -> {len(a_paths)} frames\n"
            f"B: {b_dir} -> {len(b_paths)} frames"
        )

    rows = []
    psnrs = []
    mses = []
    maes = []

    for i in range(n):
        img_a = Image.open(a_paths[i]).convert("RGB")
        img_b = Image.open(b_paths[i]).convert("RGB")

        arr_a = np.asarray(img_a, dtype=np.float32) / 255.0
        arr_b = np.asarray(img_b, dtype=np.float32) / 255.0

        if arr_a.shape != arr_b.shape:
            raise RuntimeError(
                f"Shape distinto en frame {i}: {arr_a.shape} vs {arr_b.shape}"
            )

        psnr, mse, mae = calcular_psnr(arr_a, arr_b)

        psnrs.append(psnr)
        mses.append(mse)
        maes.append(mae)

        rows.append([
            i,
            a_paths[i].name,
            b_paths[i].name,
            psnr,
            mse,
            mae,
        ])

    psnrs_np = np.asarray(psnrs, dtype=np.float64)
    mses_np = np.asarray(mses, dtype=np.float64)
    maes_np = np.asarray(maes, dtype=np.float64)

    print("=== Comparacion de frames ===")
    print(f"A: {a_dir}")
    print(f"B: {b_dir}")
    print(f"frames comparados: {n}")
    print("")
    print(f"PSNR promedio : {psnrs_np.mean():.4f}")
    print(f"PSNR min      : {psnrs_np.min():.4f}")
    print(f"PSNR p5       : {np.percentile(psnrs_np, 5):.4f}")
    print(f"PSNR max      : {psnrs_np.max():.4f}")
    print(f"PSNR std      : {psnrs_np.std():.4f}")
    print("")
    print(f"MSE promedio  : {mses_np.mean():.10f}")
    print(f"MAE promedio  : {maes_np.mean():.10f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "frame_a", "frame_b", "psnr", "mse", "mae"])
            for row in rows:
                writer.writerow([
                    row[0],
                    row[1],
                    row[2],
                    f"{row[3]:.6f}",
                    f"{row[4]:.10f}",
                    f"{row[5]:.10f}",
                ])

        print("")
        print(f"CSV guardado en: {out}")


if __name__ == "__main__":
    main()
