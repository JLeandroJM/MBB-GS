import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def buscar_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict_coefs" in ckpt:
        return ckpt["state_dict_coefs"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("No pude encontrar state_dict en el checkpoint.")


def tensor_mb(numel, bytes_por_elem):
    return numel * bytes_por_elem / 1024 / 1024


def stats_array(x_np):
    x = x_np.reshape(-1).astype(np.float64)
    if x.size == 0:
        return {}

    percentiles = np.percentile(x, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])

    return {
        "min": percentiles[0],
        "p001": percentiles[1],
        "p01": percentiles[2],
        "p05": percentiles[3],
        "p25": percentiles[4],
        "p50": percentiles[5],
        "p75": percentiles[6],
        "p95": percentiles[7],
        "p99": percentiles[8],
        "p999": percentiles[9],
        "max": percentiles[10],
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "abs_max": float(np.max(np.abs(x))),
        "frac_abs_lt_1e_4": float(np.mean(np.abs(x) < 1e-4)),
        "frac_abs_lt_1e_3": float(np.mean(np.abs(x) < 1e-3)),
        "frac_abs_lt_1e_2": float(np.mean(np.abs(x) < 1e-2)),
    }


def error_roundtrip(x_np, y_np):
    x = x_np.reshape(-1).astype(np.float64)
    y = y_np.reshape(-1).astype(np.float64)
    e = y - x
    ae = np.abs(e)
    mse = float(np.mean(e ** 2))
    denom = float(np.mean(x ** 2))
    snr = float("inf") if mse <= 0 else 10.0 * np.log10((denom + 1e-30) / mse)

    return {
        "err_mae": float(np.mean(ae)),
        "err_p95": float(np.percentile(ae, 95)),
        "err_p99": float(np.percentile(ae, 99)),
        "err_max": float(np.max(ae)),
        "err_mse": mse,
        "tensor_snr_db": snr,
    }


def quant_uint_affine(x_np, bits):
    x = x_np.astype(np.float32)
    qmax = (2 ** bits) - 1

    mn = float(np.min(x))
    mx = float(np.max(x))

    if mx == mn:
        q = np.zeros_like(x, dtype=np.uint8 if bits == 8 else np.uint16)
        deq = np.full_like(x, mn, dtype=np.float32)
        return q, deq, mn, 0.0

    scale = (mx - mn) / qmax
    q = np.round((x - mn) / scale)
    q = np.clip(q, 0, qmax)

    if bits == 8:
        q = q.astype(np.uint8)
    else:
        q = q.astype(np.uint16)

    deq = q.astype(np.float32) * scale + mn
    return q, deq, mn, scale


def evaluar_cheb(T, high, a0):
    """
    Evalua coeficientes Chebyshev para tensores del checkpoint.

    Casos soportados:
      high [N, G]       con a0 [N, 1] o [N]
      high [N, C, G]    con a0 [N, C, 1] o [N, C]
      high [N, G, C]    con a0 [N, C, 1] o [N, C]
    Devuelve:
      [N, Teval] o [N, Teval, C]
    """

    # Normalizar a0: quitar ultimo eje si es coeficiente a0 con shape [..., 1]
    if a0.ndim >= 2 and a0.shape[-1] == 1:
        a0v = a0.squeeze(-1)
    else:
        a0v = a0

    G = T.shape[1]

    if high.ndim == 2:
        # high [N, G]
        base = high @ T.T  # [N, Teval]

        if a0v.ndim == 2 and a0v.shape[1] == 1:
            a0v = a0v.squeeze(1)

        return a0v[:, None] + base

    if high.ndim == 3:
        # Caso normal del modelo: [N, C, G]
        if high.shape[-1] == G:
            base = torch.einsum("ncg,tg->ntc", high, T)  # [N, Teval, C]

        # Por si algun checkpoint viniera como [N, G, C]
        elif high.shape[1] == G:
            base = torch.einsum("ngc,tg->ntc", high, T)  # [N, Teval, C]

        else:
            raise RuntimeError(
                f"No puedo inferir eje de grado. high={tuple(high.shape)}, T={tuple(T.shape)}"
            )

        if a0v.ndim == 1:
            a0v = a0v[:, None]

        return a0v[:, None, :] + base

    raise RuntimeError(f"Forma no soportada para high: {tuple(high.shape)}")


def construir_cheb(n_frames, grado, sample_every):
    ts = torch.linspace(-1.0, 1.0, n_frames, dtype=torch.float32)[::sample_every]
    if grado <= 0:
        return ts, torch.empty((ts.numel(), 0), dtype=torch.float32)

    T = torch.empty((ts.numel(), grado), dtype=torch.float32)
    T[:, 0] = 1.0
    if grado > 1:
        T[:, 1] = ts
        for k in range(2, grado):
            T[:, k] = 2.0 * ts * T[:, k - 1] - T[:, k - 2]

    return ts, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--export_color_sample", action="store_true")
    ap.add_argument("--sample_ids", type=int, default=10)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = buscar_state_dict(ckpt)

    rows = []

    for name, tensor in sd.items():
        if not torch.is_tensor(tensor):
            continue
        if not tensor.is_floating_point():
            continue

        x = tensor.detach().cpu().float().numpy()
        numel = int(tensor.numel())

        st = stats_array(x)

        # FP16 roundtrip
        x_fp16 = tensor.detach().cpu().half().float().numpy()
        e_fp16 = error_roundtrip(x, x_fp16)

        # UINT8 affine por tensor
        _, x_u8, u8_min, u8_scale = quant_uint_affine(x, 8)
        e_u8 = error_roundtrip(x, x_u8)

        # UINT16 affine por tensor
        _, x_u16, u16_min, u16_scale = quant_uint_affine(x, 16)
        e_u16 = error_roundtrip(x, x_u16)

        rows.append({
            "tensor": name,
            "shape": "x".join(str(v) for v in tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": numel,

            "MB_actual": tensor_mb(numel, tensor.element_size()),
            "MB_fp32": tensor_mb(numel, 4),
            "MB_fp16": tensor_mb(numel, 2),
            "MB_uint16_datos": tensor_mb(numel, 2),
            "MB_uint8_datos": tensor_mb(numel, 1),

            **st,

            "fp16_mae": e_fp16["err_mae"],
            "fp16_p99": e_fp16["err_p99"],
            "fp16_max": e_fp16["err_max"],
            "fp16_snr_db": e_fp16["tensor_snr_db"],

            "uint16_mae": e_u16["err_mae"],
            "uint16_p99": e_u16["err_p99"],
            "uint16_max": e_u16["err_max"],
            "uint16_snr_db": e_u16["tensor_snr_db"],
            "uint16_min": u16_min,
            "uint16_scale": u16_scale,

            "uint8_mae": e_u8["err_mae"],
            "uint8_p99": e_u8["err_p99"],
            "uint8_max": e_u8["err_max"],
            "uint8_snr_db": e_u8["tensor_snr_db"],
            "uint8_min": u8_min,
            "uint8_scale": u8_scale,
        })

    csv_path = out_dir / "tensor_storage_stats.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # resumen por prefijo: mu/color/opacity/scale/theta/depth
    grupos = {}
    for r in rows:
        pref = r["tensor"].split("_")[0]
        grupos.setdefault(pref, []).append(r)

    resumen = []
    for pref, rs in grupos.items():
        resumen.append({
            "param": pref,
            "MB_actual": sum(float(r["MB_actual"]) for r in rs),
            "MB_fp32": sum(float(r["MB_fp32"]) for r in rs),
            "MB_fp16": sum(float(r["MB_fp16"]) for r in rs),
            "MB_uint16_datos": sum(float(r["MB_uint16_datos"]) for r in rs),
            "MB_uint8_datos": sum(float(r["MB_uint8_datos"]) for r in rs),
            "num_tensores": len(rs),
            "numel": sum(int(r["numel"]) for r in rs),
            "fp16_snr_db_prom": float(np.mean([float(r["fp16_snr_db"]) for r in rs])),
            "uint16_snr_db_prom": float(np.mean([float(r["uint16_snr_db"]) for r in rs])),
            "uint8_snr_db_prom": float(np.mean([float(r["uint8_snr_db"]) for r in rs])),
        })

    resumen_path = out_dir / "param_storage_summary.csv"
    if resumen:
        with open(resumen_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
            w.writeheader()
            w.writerows(resumen)

    # Evaluación extra de color/opacidad reales a través del tiempo
    eval_rows = []

    grados = sd.get("grados", {})
    n_frames = int(sd.get("n_frames", ckpt.get("n_frames", 750) if isinstance(ckpt, dict) else 750))

    for param, final_fn in [
        ("color", "sigmoid"),
        ("opacity", "sigmoid"),
        ("scale", "exp"),
        ("depth", "identity"),
    ]:
        a0_key = f"{param}_a0"
        high_key = f"{param}_high"

        if a0_key not in sd or high_key not in sd:
            continue
        if not torch.is_tensor(sd[a0_key]) or not torch.is_tensor(sd[high_key]):
            continue

        a0 = sd[a0_key].detach().cpu().float()
        high = sd[high_key].detach().cpu().float()
        grado = int(grados.get(param, high.shape[-1]))

        _, T = construir_cheb(n_frames, grado, args.sample_every)
        vals = evaluar_cheb(T, high[..., :grado], a0)

        if final_fn == "sigmoid":
            vals_final = torch.sigmoid(vals)
        elif final_fn == "exp":
            vals_final = torch.exp(vals)
        else:
            vals_final = vals

        vals_np = vals_final.numpy()
        st = stats_array(vals_np)

        eval_rows.append({
            "param_final": param,
            "transformacion": final_fn,
            "shape_eval": "x".join(str(v) for v in vals_final.shape),
            "sample_every": args.sample_every,
            **st,
        })

        if param == "color" and args.export_color_sample:
            n = min(args.sample_ids, vals_final.shape[0])
            sample_path = out_dir / "color_final_sample_por_gaussiana.csv"

            with open(sample_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "canal", "min", "max", "mean", "std", "p05", "p50", "p95"])
                arr = vals_final[:n].numpy()

                for gid in range(n):
                    for c in range(arr.shape[-1]):
                        v = arr[gid, :, c]
                        w.writerow([
                            gid,
                            c,
                            float(np.min(v)),
                            float(np.max(v)),
                            float(np.mean(v)),
                            float(np.std(v)),
                            float(np.percentile(v, 5)),
                            float(np.percentile(v, 50)),
                            float(np.percentile(v, 95)),
                        ])

    eval_path = out_dir / "param_final_values_summary.csv"
    if eval_rows:
        with open(eval_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_rows)

    meta = {
        "checkpoint": str(ckpt_path),
        "out_dir": str(out_dir),
        "sample_every": args.sample_every,
        "archivos": [
            str(csv_path),
            str(resumen_path),
            str(eval_path),
        ],
    }

    with open(out_dir / "README_resultados.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=== inspeccion storage checkpoint ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"out_dir   : {out_dir}")
    print("")
    print(f"CSV tensores       : {csv_path}")
    print(f"CSV resumen params : {resumen_path}")
    print(f"CSV valores finales: {eval_path}")
    print("")
    print("Top por MB actual:")
    for r in sorted(rows, key=lambda z: float(z["MB_actual"]), reverse=True)[:12]:
        print(
            f"{r['tensor']:18s} shape={r['shape']:18s} "
            f"MB={float(r['MB_actual']):8.2f} "
            f"fp16_snr={float(r['fp16_snr_db']):8.2f} "
            f"u8_snr={float(r['uint8_snr_db']):8.2f}"
        )


if __name__ == "__main__":
    main()

