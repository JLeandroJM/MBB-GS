import argparse
from pathlib import Path
import torch


def mb(x):
    return x / 1024 / 1024


def analyze_tensor(name, q):
    q = q.detach().cpu()
    q64 = q.to(torch.int64)

    numel = q.numel()
    bytes_u16 = numel * 2
    bytes_u8_direct = numel * 1

    qmin = int(q64.min().item())
    qmax = int(q64.max().item())
    qrange = qmax - qmin

    print("")
    print("======================================================")
    print(f"TENSOR: {name}")
    print("======================================================")
    print(f"shape              : {tuple(q.shape)}")
    print(f"numel              : {numel:,}")
    print(f"u16 MB actual       : {mb(bytes_u16):.4f}")
    print(f"q min/max/range     : {qmin} / {qmax} / {qrange}")

    direct_ok = qmax <= 255 and qmin >= 0
    print(f"UINT8 directo exacto: {direct_ok}")

    if direct_ok:
        print(f"u8 directo MB       : {mb(bytes_u8_direct):.4f}")
        print(f"ahorro              : {100*(1 - bytes_u8_direct/bytes_u16):.2f}%")
    else:
        print("u8 directo NO es exacto porque hay valores > 255.")

    # Forma [N, D]
    if q.ndim == 1:
        q2 = q64.reshape(q.shape[0], 1)
    else:
        q2 = q64.reshape(q.shape[0], -1)

    N, D = q2.shape

    # ==================================================
    # Analisis por coeficiente: para cada columna D,
    # mirar rango entre todas las gaussianas.
    # Guardaria: base uint16 por columna + delta uint8 por valor.
    # ==================================================
    col_min = q2.min(dim=0).values
    col_max = q2.max(dim=0).values
    col_range = col_max - col_min
    col_fit = col_range <= 255
    n_col_fit = int(col_fit.sum().item())

    bytes_col_delta_all = D * 2 + numel * 1
    bytes_col_mixed = n_col_fit * 2 + n_col_fit * N * 1 + (D - n_col_fit) * N * 2

    print("")
    print("--- Packing por coeficiente ---")
    print(f"columnas D                  : {D}")
    print(f"coeficientes con rango<=255 : {n_col_fit}/{D} ({100*n_col_fit/D:.2f}%)")
    print(f"rango columna promedio      : {float(col_range.float().mean().item()):.2f}")
    print(f"rango columna p50           : {float(torch.quantile(col_range.float(), 0.50).item()):.2f}")
    print(f"rango columna p90           : {float(torch.quantile(col_range.float(), 0.90).item()):.2f}")
    print(f"rango columna p99           : {float(torch.quantile(col_range.float(), 0.99).item()):.2f}")
    print(f"rango columna max           : {int(col_range.max().item())}")

    if n_col_fit == D:
        print(f"packing columna exacto MB   : {mb(bytes_col_delta_all):.4f}")
        print(f"ahorro exacto               : {100*(1 - bytes_col_delta_all/bytes_u16):.2f}%")
    else:
        print(f"packing mixto estimado MB   : {mb(bytes_col_mixed):.4f}")
        print(f"ahorro mixto estimado       : {100*(1 - bytes_col_mixed/bytes_u16):.2f}%")

    # ==================================================
    # Analisis por gaussiana: cada fila N.
    # Guardaria: base uint16 por gaussiana + delta uint8 por valor.
    # ==================================================
    row_min = q2.min(dim=1).values
    row_max = q2.max(dim=1).values
    row_range = row_max - row_min
    row_fit = row_range <= 255
    n_row_fit = int(row_fit.sum().item())

    bytes_row_delta_all = N * 2 + numel * 1
    bytes_row_mixed = n_row_fit * 2 + n_row_fit * D * 1 + (N - n_row_fit) * D * 2

    print("")
    print("--- Packing por gaussiana ---")
    print(f"gaussianas N                : {N}")
    print(f"gaussianas con rango<=255   : {n_row_fit}/{N} ({100*n_row_fit/N:.2f}%)")
    print(f"rango gaussiana promedio    : {float(row_range.float().mean().item()):.2f}")
    print(f"rango gaussiana p50         : {float(torch.quantile(row_range.float(), 0.50).item()):.2f}")
    print(f"rango gaussiana p90         : {float(torch.quantile(row_range.float(), 0.90).item()):.2f}")
    print(f"rango gaussiana p99         : {float(torch.quantile(row_range.float(), 0.99).item()):.2f}")
    print(f"rango gaussiana max         : {int(row_range.max().item())}")

    if n_row_fit == N:
        print(f"packing gauss exacto MB     : {mb(bytes_row_delta_all):.4f}")
        print(f"ahorro exacto               : {100*(1 - bytes_row_delta_all/bytes_u16):.2f}%")
    else:
        print(f"packing mixto estimado MB   : {mb(bytes_row_mixed):.4f}")
        print(f"ahorro mixto estimado       : {100*(1 - bytes_row_mixed/bytes_u16):.2f}%")

    return {
        "name": name,
        "bytes_u16": bytes_u16,
        "direct_ok": direct_ok,
        "n_col_fit": n_col_fit,
        "D": D,
        "bytes_col_mixed": bytes_col_mixed,
        "n_row_fit": n_row_fit,
        "N": N,
        "bytes_row_mixed": bytes_row_mixed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    args = ap.parse_args()

    path = Path(args.pkg)
    pkg = torch.load(path, map_location="cpu")

    quantized = pkg.get("quantized", {})
    if not quantized:
        raise RuntimeError("No encontre 'quantized' en el package.")

    print("======================================================")
    print("ANALISIS UINT16 -> UINT8 LOSSLESS")
    print("======================================================")
    print(f"archivo: {path}")
    print(f"MB archivo: {path.stat().st_size / 1024 / 1024:.2f}")

    results = []

    for name, info in quantized.items():
        if not isinstance(info, dict) or "q" not in info:
            continue
        results.append(analyze_tensor(name, info["q"]))

    print("")
    print("======================================================")
    print("RESUMEN GLOBAL")
    print("======================================================")

    total_u16 = sum(r["bytes_u16"] for r in results)
    total_col_mixed = sum(r["bytes_col_mixed"] for r in results)
    total_row_mixed = sum(r["bytes_row_mixed"] for r in results)

    print(f"TOTAL q UINT16 actual MB        : {mb(total_u16):.4f}")
    print(f"TOTAL packing columna mixto MB  : {mb(total_col_mixed):.4f}")
    print(f"TOTAL packing gauss mixto MB    : {mb(total_row_mixed):.4f}")

    print("")
    print("Ahorro estimado columna mixto:")
    print(f"{100*(1 - total_col_mixed/total_u16):.2f}%")

    print("")
    print("Ahorro estimado gauss mixto:")
    print(f"{100*(1 - total_row_mixed/total_u16):.2f}%")

    print("")
    print("LECTURA:")
    print("- Si UINT8 directo exacto es True, ese tensor puede pasar a uint8 sin perdida.")
    print("- Si muchas columnas tienen rango <=255, se puede guardar base por coeficiente + delta uint8.")
    print("- Si muchas gaussianas tienen rango <=255, se puede guardar base por gaussiana + delta uint8.")
    print("- Si los rangos son grandes, pasar a uint8 solo seria con perdida.")


if __name__ == "__main__":
    main()
