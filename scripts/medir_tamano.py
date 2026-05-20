"""
Calcula el tamano en bytes de un checkpoint (.pt) y el ratio de compresion
vs el video crudo (num_frames * H * W * 3 bytes en uint8).

Uso:
    python tools/medir_tamano.py checkpoint.pt --num_frames 90 --res 256
"""
import argparse
import os
import torch



# bytes por dtype de torch
BYTES_POR_DTYPE = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
}



def medir_tamano_tensor(t):
    """Cuenta exacta: numel * bytes_por_dtype."""
    bytes_por_elem = BYTES_POR_DTYPE.get(t.dtype, 4)
    return t.numel() * bytes_por_elem



def medir_tamano_checkpoint(ruta_pt):
    """Carga el .pt y suma los bytes de todos los tensores que encuentra."""
    obj = torch.load(ruta_pt, map_location='cpu', weights_only=False)

    total = 0
    detalles = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                b = medir_tamano_tensor(v)
                total += b
                detalles[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": b}
    elif torch.is_tensor(obj):
        total = medir_tamano_tensor(obj)
        detalles["<tensor>"] = {"shape": list(obj.shape), "dtype": str(obj.dtype), "bytes": total}

    return total, detalles



def tamano_raw_video(num_frames, H, W, canales=3):
    return num_frames * H * W * canales



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="ruta al .pt")
    parser.add_argument("--num_frames", type=int, default=90)
    parser.add_argument("--res", type=int, default=256)
    args = parser.parse_args()

    total, detalles = medir_tamano_checkpoint(args.checkpoint)
    raw = tamano_raw_video(args.num_frames, args.res, args.res)
    ratio = raw / total if total > 0 else float('inf')

    print(f"\ncheckpoint: {args.checkpoint}")
    for k, d in detalles.items():
        print(f"  {k}: shape={d['shape']} dtype={d['dtype']} bytes={d['bytes']}")
    print(f"\ntotal modelo: {total} bytes ({total/1024:.2f} KiB)")
    print(f"raw video:    {raw} bytes ({raw/1024:.2f} KiB)")
    print(f"ratio compresion: {ratio:.1f}x")



if __name__ == "__main__":
    main()
