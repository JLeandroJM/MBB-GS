import argparse
from pathlib import Path
import torch


def mb(nbytes):
    return nbytes / 1024 / 1024


def tensor_nbytes(x):
    return x.numel() * x.element_size()


def short_list(x, max_items=8):
    x = x.detach().cpu().flatten()
    if x.numel() == 0:
        return []
    vals = x[:max_items].tolist()
    return [round(float(v), 6) for v in vals]


def print_dict_brief(d, name, indent=0):
    sp = " " * indent
    print(f"{sp}{name}: dict con {len(d)} claves")
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"{sp}  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype} MB={mb(tensor_nbytes(v)):.4f}")
        elif isinstance(v, dict):
            print(f"{sp}  {k}: dict con {len(v)} claves")
        else:
            s = str(v)
            if len(s) > 160:
                s = s[:160] + "..."
            print(f"{sp}  {k}: {type(v).__name__} = {s}")


def dequant_tensor(info):
    q = info["q"].detach().cpu()
    mn = float(info["min"])
    scale = float(info["scale"])
    return q.float() * scale + mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--ids", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--coef_preview", type=int, default=8)
    args = ap.parse_args()

    path = Path(args.pkg)
    obj = torch.load(path, map_location="cpu")

    print("======================================================")
    print("INSPECCION PACKAGE")
    print("======================================================")
    print(f"archivo: {path}")
    print(f"MB archivo: {mb(path.stat().st_size):.2f}")
    print(f"tipo raiz: {type(obj).__name__}")

    if not isinstance(obj, dict):
        print("El archivo no es dict. No puedo inspeccionar formato esperado.")
        return

    print("")
    print("=== claves top-level ===")
    for k in obj.keys():
        print("-", k)

    print("")
    print("=== objetos no tensor / metadata ===")
    for k, v in obj.items():
        if k in ["quantized", "raw_tensors"]:
            continue
        if isinstance(v, dict):
            print_dict_brief(v, k, indent=0)
        elif torch.is_tensor(v):
            print(f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype} MB={mb(tensor_nbytes(v)):.4f}")
        else:
            s = str(v)
            if len(s) > 300:
                s = s[:300] + "..."
            print(f"{k}: {type(v).__name__} = {s}")

    raw = obj.get("raw_tensors", {})
    quant = obj.get("quantized", {})

    print("")
    print("======================================================")
    print("TENSORES RAW / FP32")
    print("======================================================")
    total_raw = 0
    if isinstance(raw, dict):
        for k, v in raw.items():
            if torch.is_tensor(v):
                nb = tensor_nbytes(v)
                total_raw += nb
                print(f"{k:20s} shape={str(tuple(v.shape)):24s} dtype={str(v.dtype):12s} MB={mb(nb):8.4f}")
            else:
                print(f"{k:20s} {type(v).__name__}")
    print(f"TOTAL RAW MB: {mb(total_raw):.4f}")

    print("")
    print("======================================================")
    print("TENSORES CUANTIZADOS")
    print("======================================================")
    total_q = 0
    if isinstance(quant, dict):
        for k, info in quant.items():
            if isinstance(info, dict) and "q" in info:
                q = info["q"]
                nb = tensor_nbytes(q)
                total_q += nb
                print(
                    f"{k:20s} q_shape={str(tuple(q.shape)):24s} "
                    f"dtype={str(q.dtype):12s} MB={mb(nb):8.4f} "
                    f"min={float(info.get('min', 0.0)):.6g} "
                    f"scale={float(info.get('scale', 0.0)):.6g} "
                    f"bits={info.get('bits', '?')}"
                )
            else:
                print(f"{k:20s} formato no reconocido: {type(info).__name__}")
    print(f"TOTAL QUANT Q MB: {mb(total_q):.4f}")

    print("")
    print("======================================================")
    print("RESUMEN 2 GAUSSIANAS")
    print("======================================================")
    ids = args.ids

    print("")
    print("---- RAW tensors por gaussiana ----")
    for gid in ids:
        print(f"\nGAUSSIANA {gid}")
        for k, v in raw.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] > gid:
                arr = v[gid]
                print(
                    f"  {k:18s} shape={tuple(arr.shape)} "
                    f"dtype={arr.dtype} preview={short_list(arr, args.coef_preview)}"
                )

    print("")
    print("---- QUANT tensors por gaussiana, reconstruidos a float ----")
    for gid in ids:
        print(f"\nGAUSSIANA {gid}")
        for k, info in quant.items():
            if not (isinstance(info, dict) and "q" in info):
                continue
            q = info["q"]
            if q.ndim >= 1 and q.shape[0] > gid:
                x = dequant_tensor(info)[gid]
                q_gid = q[gid]
                print(
                    f"  {k:18s} float_shape={tuple(x.shape)} "
                    f"float_preview={short_list(x, args.coef_preview)}"
                )
                print(
                    f"  {'q_'+k:18s} q_shape={tuple(q_gid.shape)} "
                    f"q_preview={q_gid.detach().cpu().flatten()[:args.coef_preview].tolist()}"
                )

    print("")
    print("======================================================")
    print("LECTURA")
    print("======================================================")
    print("Si TOTAL RAW + TOTAL QUANT es casi igual al tamaño del archivo,")
    print("entonces casi todo el peso son parámetros reales de gaussianas.")
    print("La metadata/config/texto pesa poco.")


if __name__ == "__main__":
    main()
