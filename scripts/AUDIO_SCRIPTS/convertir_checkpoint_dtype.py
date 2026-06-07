# scripts/AUDIO_SCRIPTS/convertir_checkpoint_dtype.py

import argparse
from pathlib import Path
import torch

def convertir_obj(obj, dtype):
    if torch.is_tensor(obj) and torch.is_floating_point(obj):
        return obj.to(dtype)
    if isinstance(obj, dict):
        return {k: convertir_obj(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convertir_obj(v, dtype) for v in obj]
    return obj

parser = argparse.ArgumentParser()
parser.add_argument("--in-ckpt", required=True)
parser.add_argument("--out-ckpt", required=True)
parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
args = parser.parse_args()

dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

ckpt = torch.load(args.in_ckpt, map_location="cpu")
ckpt_q = convertir_obj(ckpt, dtype)

Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt_q, args.out_ckpt)

print("OK")
print("entrada:", args.in_ckpt)
print("salida :", args.out_ckpt)
print("dtype  :", args.dtype)

'''
python scripts\AUDIO_SCRIPTS\convertir_checkpoint_dtype.py `
  --in-ckpt outputs\audio\_batches\ft2_sigma4_winner\rock_61s_30s_stereo_ft2_N10000_g200_40_360_sigma4\RIGHT\checkpoint_final.pt `
  --out-ckpt outputs\audio\_batches\ft2_sigma4_winner\rock_61s_30s_stereo_ft2_N10000_g200_40_360_sigma4\RIGHT\checkpoint_final_fp16.pt `
  --dtype fp16


python scripts\AUDIO_SCRIPTS\recon_audio_desde_checkpoint.py `
  --ckpt outputs\audio\_batches\ft2_sigma4_winner\rock_61s_30s_stereo_ft2_N10000_g200_40_360_sigma4\LEFT\checkpoint_final_fp16.pt `
  --config outputs\audio\_batches\ft2_sigma4_winner\rock_61s_30s_stereo_ft2_N10000_g200_40_360_sigma4\LEFT\config_usada.json `
  --out outputs\audio\_batches\ft2_sigma4_winner\rock_61s_30s_stereo_ft2_N10000_g200_40_360_sigma4\LEFT\recon_fase_original_fp16.wav
  


'''