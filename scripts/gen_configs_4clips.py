"""
Genera UNA config base (control) por cada uno de los 4 clips nuevos, con:
  - termometro holdout-x2 (mide interpolacion: train vs retenidos)
  - guardar_frames_rasterizados=true  -> para armar reconstruido.mp4
Escribe en configs/slowmo/slowmo_<tag>_base.json

Esto es la FOTO BASE por clip (sin palancas). La ablacion de palancas
(smoothness asimetrica + aceleracion) vive en gen_configs_slowmo.py sobre gota.

Uso:  python scripts/gen_configs_4clips.py
"""
import copy
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from gen_configs_slowmo import base_config       # reusa la misma base

DEST = RAIZ / "configs" / "slowmo"

# (clip, max_frames, tag)
CLIPS = [
    ("gota_s6s16_480x270", 300, "gota"),
    ("rampa_alta_s4s9_480x270", 150, "rampaA"),
    ("rampa_baja_s9s15_480x270", 180, "rampaB"),
    ("estructura_s6s14_480x270", 240, "estr"),
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for clip, mf, tag in CLIPS:
        cfg = copy.deepcopy(base_config(clip, mf))
        cfg["nombre_experimento"] = f"slowmo_{tag}_base"
        cfg["_comentario"] = f"foto base (control) clip {clip} | termometro holdout-x2"
        cfg["guardar_frames_rasterizados"] = True       # <- para reconstruido.mp4
        ruta = DEST / f"slowmo_{tag}_base.json"
        ruta.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {ruta.name}  (clip={clip}, max_frames={mf})")
    print(f"=== 4 configs base en {DEST} ===")


if __name__ == "__main__":
    main()
