import argparse
from pathlib import Path

import torch


def convertir_fp16(obj):
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.half()
        return obj

    if isinstance(obj, dict):
        return {k: convertir_fp16(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convertir_fp16(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(convertir_fp16(v) for v in obj)

    return obj


def contar_tensores(obj):
    total = 0
    floats = 0
    fp16 = 0
    fp32 = 0

    if torch.is_tensor(obj):
        total += 1
        if obj.is_floating_point():
            floats += 1
            if obj.dtype == torch.float16:
                fp16 += 1
            if obj.dtype == torch.float32:
                fp32 += 1
        return total, floats, fp16, fp32

    if isinstance(obj, dict):
        for v in obj.values():
            a, b, c, d = contar_tensores(v)
            total += a
            floats += b
            fp16 += c
            fp32 += d

    elif isinstance(obj, (list, tuple)):
        for v in obj:
            a, b, c, d = contar_tensores(v)
            total += a
            floats += b
            fp16 += c
            fp32 += d

    return total, floats, fp16, fp32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ckpt", required=True)
    parser.add_argument("--out_ckpt", required=True)
    args = parser.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")

    total0, floats0, fp160, fp320 = contar_tensores(ckpt)
    ckpt_fp16 = convertir_fp16(ckpt)
    total1, floats1, fp161, fp321 = contar_tensores(ckpt_fp16)

    torch.save(ckpt_fp16, out_path)

    print("=== conversion checkpoint FP16 ===")
    print(f"entrada : {in_path}")
    print(f"salida  : {out_path}")
    print("")
    print("Antes:")
    print(f"  tensores total : {total0}")
    print(f"  tensores float : {floats0}")
    print(f"  float16        : {fp160}")
    print(f"  float32        : {fp320}")
    print("")
    print("Despues:")
    print(f"  tensores total : {total1}")
    print(f"  tensores float : {floats1}")
    print(f"  float16        : {fp161}")
    print(f"  float32        : {fp321}")
    print("")
    print(f"MB entrada: {in_path.stat().st_size / 1024 / 1024:.2f}")
    print(f"MB salida : {out_path.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
