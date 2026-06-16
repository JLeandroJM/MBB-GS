import argparse
from pathlib import Path
import torch


PARAMS_DIMS = {
    "mu": 2,
    "opacity": 1,
    "color": 3,
    "scale": 2,
    "theta": 1,
    "depth": 1,
}


def construir_matriz_chebyshev(n_frames, grado_max, dtype=torch.float64):
    if n_frames < 2:
        t = torch.zeros(n_frames, dtype=dtype)
    else:
        idx = torch.arange(n_frames, dtype=dtype)
        t = 2.0 * (idx / (n_frames - 1)) - 1.0

    B = torch.empty(n_frames, grado_max + 1, dtype=dtype)
    B[:, 0] = 1.0
    if grado_max >= 1:
        B[:, 1] = t
    for k in range(2, grado_max + 1):
        B[:, k] = 2.0 * t * B[:, k - 1] - B[:, k - 2]

    return B


def normalizar_a0(a0, dim):
    if dim == 1:
        if a0.ndim == 1:
            return a0[:, None]
        if a0.ndim == 2:
            return a0
        if a0.ndim == 3:
            return a0.squeeze(-1)
    else:
        if a0.ndim == 3:
            return a0.squeeze(-1)
        if a0.ndim == 2:
            return a0
    raise RuntimeError(f"shape a0 no soportado: {tuple(a0.shape)}")


def restaurar_a0_shape(a0_new, old_a0, dim):
    if dim == 1:
        if old_a0.ndim == 1:
            return a0_new[:, 0].contiguous()
        if old_a0.ndim == 2:
            return a0_new.contiguous()
        if old_a0.ndim == 3:
            return a0_new[:, :, None].contiguous()
    else:
        if old_a0.ndim == 3:
            return a0_new[:, :, None].contiguous()
        if old_a0.ndim == 2:
            return a0_new.contiguous()
    raise RuntimeError(f"shape old_a0 no soportado: {tuple(old_a0.shape)}")


def refit_param(sd, nombre, new_g, chunk):
    dim = PARAMS_DIMS[nombre]
    key_a0 = f"{nombre}_a0"
    key_hi = f"{nombre}_high"

    a0_old = sd[key_a0].detach().cpu().float()
    hi_old = sd[key_hi].detach().cpu().float()

    old_g = hi_old.shape[-1]

    if new_g > old_g:
        raise RuntimeError(f"{nombre}: new_g {new_g} > old_g {old_g}")

    if new_g == old_g:
        print(f"{nombre}: mismo grado {old_g}, no se toca")
        return

    n_frames = int(sd["n_frames"])

    print(f"\n=== REFIT {nombre}: {old_g} -> {new_g} ===")
    print(f"a0 old: {tuple(a0_old.shape)}")
    print(f"hi old: {tuple(hi_old.shape)}")

    a0_mat = normalizar_a0(a0_old, dim)          # [N, dim]
    N = a0_mat.shape[0]

    if dim == 1:
        hi_mat = hi_old.reshape(N, 1, old_g)
    else:
        hi_mat = hi_old.reshape(N, dim, old_g)

    coefs_old = torch.cat([a0_mat[:, :, None], hi_mat], dim=-1)  # [N, dim, old_g+1]
    flat_old = coefs_old.reshape(N * dim, old_g + 1).double()

    B_old = construir_matriz_chebyshev(n_frames, old_g)          # [T, old_g+1]
    B_new = construir_matriz_chebyshev(n_frames, new_g)          # [T, new_g+1]

    # beta = Y @ pinv(B_new).T
    pinv_new_T = torch.linalg.pinv(B_new).T                      # [T, new_g+1]
    B_old_T = B_old.T                                            # [old_g+1, T]

    flat_new_parts = []

    for ini in range(0, flat_old.shape[0], chunk):
        fin = min(ini + chunk, flat_old.shape[0])

        c = flat_old[ini:fin]                                    # [M, old_g+1]
        y = c @ B_old_T                                          # [M, T]
        beta = y @ pinv_new_T                                    # [M, new_g+1]

        flat_new_parts.append(beta.float().cpu())

        if ini == 0 or fin == flat_old.shape[0]:
            print(f"chunk {ini}:{fin} / {flat_old.shape[0]}")

    flat_new = torch.cat(flat_new_parts, dim=0)
    coefs_new = flat_new.reshape(N, dim, new_g + 1)

    a0_new = coefs_new[:, :, 0]
    hi_new = coefs_new[:, :, 1:].contiguous()

    sd[key_a0] = restaurar_a0_shape(a0_new, a0_old, dim)

    if dim == 1:
        sd[key_hi] = hi_new[:, 0, :].contiguous()
    else:
        sd[key_hi] = hi_new.contiguous()

    sd["grados"][nombre] = new_g

    print(f"a0 new: {tuple(sd[key_a0].shape)}")
    print(f"hi new: {tuple(sd[key_hi].shape)}")


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

    ap.add_argument("--chunk", type=int, default=8192)

    args = ap.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")

    if "state_dict_coefs" not in ckpt:
        raise RuntimeError("El checkpoint no tiene state_dict_coefs")

    sd = ckpt["state_dict_coefs"]

    if "grados" not in sd:
        if "config" in ckpt and "grados" in ckpt["config"]:
            sd["grados"] = dict(ckpt["config"]["grados"])
        else:
            raise RuntimeError("No encuentro grados en state_dict_coefs ni config")

    sd["grados"] = dict(sd["grados"])

    cambios = {
        "mu": args.mu,
        "opacity": args.opacity,
        "color": args.color,
        "scale": args.scale,
        "theta": args.theta,
        "depth": args.depth,
    }

    print("=== refit grados checkpoint ===")
    print(f"entrada: {in_path}")
    print(f"salida : {out_path}")
    print(f"grados old: {sd['grados']}")

    for nombre, new_g in cambios.items():
        if new_g is not None:
            refit_param(sd, nombre, int(new_g), args.chunk)

    if "config" in ckpt:
        ckpt["config"]["grados"] = dict(sd["grados"])

    torch.save(ckpt, out_path)

    print("")
    print("=== listo ===")
    print(f"grados new: {sd['grados']}")
    print(f"MB entrada: {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
