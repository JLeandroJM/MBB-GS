import argparse
from pathlib import Path
import torch


DEFAULT_QUANT = {
    "mu_high": 16,
    "color_high": 8,
    "opacity_high": 8,
    "scale_high": 8,
}


def find_sd(ckpt):
    if isinstance(ckpt, dict) and "state_dict_coefs" in ckpt:
        return ckpt["state_dict_coefs"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("No encontre state_dict.")


def quant_affine(x, bits):
    x = x.detach().cpu().float().contiguous()
    mn = float(x.min().item())
    mx = float(x.max().item())
    qmax = (2 ** bits) - 1

    if mx == mn:
        dtype = torch.uint8 if bits == 8 else torch.uint16
        q = torch.zeros_like(x, dtype=dtype)
        scale = 0.0
    else:
        scale = (mx - mn) / qmax
        q = torch.round((x - mn) / scale).clamp(0, qmax)
        q = q.to(torch.uint8 if bits == 8 else torch.uint16)

    return q, mn, scale, list(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_pkg", required=True)
    ap.add_argument("--omit_zero_depth_high", action="store_true")
    ap.add_argument("--zero_eps", type=float, default=1e-12)
    args = ap.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_pkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")
    sd = find_sd(ckpt)

    quantized = {}
    omitted_zeros = {}

    for name, bits in DEFAULT_QUANT.items():
        if name not in sd:
            print(f"[WARN] no existe tensor: {name}")
            continue

        x = sd.pop(name)

        q, mn, scale, shape = quant_affine(x, bits)

        quantized[name] = {
            "q": q,
            "bits": bits,
            "min": mn,
            "scale": scale,
            "shape": shape,
            "original_dtype": str(x.dtype),
        }

        print(
            f"UINT{bits:<2d} {name:15s} "
            f"shape={shape} min={mn:.8g} max={float(x.max().item()):.8g} scale={scale:.8g}"
        )

    if args.omit_zero_depth_high and "depth_high" in sd and torch.is_tensor(sd["depth_high"]):
        x = sd["depth_high"].detach().cpu()
        max_abs = float(x.float().abs().max().item())

        if max_abs <= args.zero_eps:
            omitted_zeros["depth_high"] = {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            }
            sd.pop("depth_high")
            print(f"OMIT depth_high porque max_abs={max_abs:.3e}")
        else:
            print(f"[WARN] depth_high no es cero. max_abs={max_abs:.3e}. No se omite.")

    pkg = {
        "format": "gs2d_mixed_affine_package_v1",
        "source_checkpoint": str(in_path),
        "ckpt_skeleton": ckpt,
        "quantized": quantized,
        "omitted_zeros": omitted_zeros,
    }

    torch.save(pkg, out_path)

    print("")
    print("=== paquete MIXED creado ===")
    print(f"entrada: {in_path}")
    print(f"salida : {out_path}")
    print(f"MB entrada: {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB paquete: {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
