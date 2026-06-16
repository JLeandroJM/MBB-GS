import argparse
from pathlib import Path
import torch


PARAMS = ["mu", "opacity", "color", "scale", "theta", "depth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)

    ap.add_argument("--mu", type=int, default=None)
    ap.add_argument("--opacity", type=int, default=None)
    ap.add_argument("--color", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--theta", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)

    args = ap.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")

    if "state_dict_coefs" not in ckpt:
        raise RuntimeError("El checkpoint no tiene state_dict_coefs")

    sd = ckpt["state_dict_coefs"]

    grados_old = dict(sd.get("grados", ckpt.get("config", {}).get("grados", {})))
    grados_new = dict(grados_old)

    cambios = {
        "mu": args.mu,
        "opacity": args.opacity,
        "color": args.color,
        "scale": args.scale,
        "theta": args.theta,
        "depth": args.depth,
    }

    print("=== truncar grados ===")
    print(f"entrada: {in_path}")
    print(f"salida : {out_path}")
    print(f"grados old: {grados_old}")

    for p in PARAMS:
        if cambios[p] is None:
            continue

        old_g = int(grados_old[p])
        new_g = int(cambios[p])

        if new_g < 0:
            raise RuntimeError(f"{p}: grado nuevo no puede ser negativo: {new_g}")

        if new_g > old_g:
            raise RuntimeError(f"{p}: nuevo grado {new_g} > grado viejo {old_g}")

        key = f"{p}_high"

        if key not in sd:
            raise RuntimeError(f"No existe tensor {key}")

        old_shape = tuple(sd[key].shape)
        sd[key] = sd[key][..., :new_g].contiguous()
        new_shape = tuple(sd[key].shape)

        grados_new[p] = new_g

        print(f"{p}: {old_g} -> {new_g} | {key}: {old_shape} -> {new_shape}")

    sd["grados"] = grados_new

    if "config" in ckpt:
        ckpt["config"]["grados"] = grados_new

    torch.save(ckpt, out_path)

    print(f"grados new: {grados_new}")
    print(f"MB entrada: {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
