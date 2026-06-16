import argparse
from pathlib import Path
import torch


def find_sd(ckpt):
    if isinstance(ckpt, dict) and "state_dict_coefs" in ckpt:
        return ckpt["state_dict_coefs"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("No encontre state_dict.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pkg", required=True)
    ap.add_argument("--out_ckpt", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_pkg)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkg = torch.load(in_path, map_location="cpu")

    if pkg.get("format") != "gs2d_mixed_affine_package_v1":
        raise RuntimeError("Formato no reconocido.")

    ckpt = pkg["ckpt_skeleton"]
    sd = find_sd(ckpt)

    for name, info in pkg["quantized"].items():
        q = info["q"].cpu()
        mn = float(info["min"])
        scale = float(info["scale"])
        bits = int(info["bits"])

        x = q.float() * scale + mn
        x = x.float().contiguous()

        sd[name] = x
        print(f"DEQUANT UINT{bits:<2d} {name:15s} shape={tuple(x.shape)} dtype={x.dtype}")

    for name, info in pkg.get("omitted_zeros", {}).items():
        sd[name] = torch.zeros(tuple(info["shape"]), dtype=torch.float32)
        print(f"RESTORE ZERO {name:15s} shape={tuple(info['shape'])} dtype=torch.float32")

    torch.save(ckpt, out_path)

    print("")
    print("=== checkpoint dequantizado creado ===")
    print(f"entrada package: {in_path}")
    print(f"salida ckpt    : {out_path}")
    print(f"MB package     : {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB ckpt salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
