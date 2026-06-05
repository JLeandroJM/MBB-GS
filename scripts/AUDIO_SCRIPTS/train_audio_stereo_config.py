import json
import subprocess
import sys
from pathlib import Path
from scipy.io import wavfile
import numpy as np


RAIZ = Path(__file__).resolve().parents[2]


def guardar_wav_stereo(ruta, sr, left, right):
    n = min(len(left), len(right))
    y = np.stack([left[:n], right[:n]], axis=1).astype(np.float32)
    y = np.nan_to_num(y)
    m = np.max(np.abs(y))
    if m > 1.0:
        y = y / m
    wavfile.write(str(ruta), sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    ruta_cfg = Path(args.config)
    if not ruta_cfg.is_absolute():
        ruta_cfg = RAIZ / ruta_cfg

    with open(ruta_cfg, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    nombre_base = cfg["nombre_experimento"]

    cfg_L = dict(cfg)
    cfg_R = dict(cfg)

    cfg_L["nombre_experimento"] = nombre_base + "_L"
    cfg_R["nombre_experimento"] = nombre_base + "_R"

    cfg_L["canal"] = "left"
    cfg_R["canal"] = "right"

    tmp_dir = RAIZ / "configs" / "audio" / "_tmp_stereo"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ruta_L = tmp_dir / (nombre_base + "_L.json")
    ruta_R = tmp_dir / (nombre_base + "_R.json")

    with open(ruta_L, "w", encoding="utf-8") as f:
        json.dump(cfg_L, f, indent=2)

    with open(ruta_R, "w", encoding="utf-8") as f:
        json.dump(cfg_R, f, indent=2)

    print("=== ENTRENANDO CANAL L ===")
    subprocess.check_call([
        sys.executable,
        str(RAIZ / "scripts" / "AUDIO_SCRIPTS" / "train_audio_config.py"),
        "--config",
        str(ruta_L),
    ], cwd=str(RAIZ))

    print("=== ENTRENANDO CANAL R ===")
    subprocess.check_call([
        sys.executable,
        str(RAIZ / "scripts" / "AUDIO_SCRIPTS" / "train_audio_config.py"),
        "--config",
        str(ruta_R),
    ], cwd=str(RAIZ))

    out_L = RAIZ / "outputs" / "audio" / cfg_L["nombre_experimento"]
    out_R = RAIZ / "outputs" / "audio" / cfg_R["nombre_experimento"]
    out_stereo = RAIZ / "outputs" / "audio" / nombre_base
    out_stereo.mkdir(parents=True, exist_ok=True)

    sr_L, recon_L = wavfile.read(str(out_L / "recon_fase_original.wav"))
    sr_R, recon_R = wavfile.read(str(out_R / "recon_fase_original.wav"))

    if sr_L != sr_R:
        raise RuntimeError("Los sample rates L/R no coinciden")

    recon_L = recon_L.astype(np.float32) / 32768.0
    recon_R = recon_R.astype(np.float32) / 32768.0

    guardar_wav_stereo(out_stereo / "recon_stereo_fase_original.wav", sr_L, recon_L, recon_R)

    sr_L, gl_L = wavfile.read(str(out_L / "recon_griffinlim.wav"))
    sr_R, gl_R = wavfile.read(str(out_R / "recon_griffinlim.wav"))

    gl_L = gl_L.astype(np.float32) / 32768.0
    gl_R = gl_R.astype(np.float32) / 32768.0

    guardar_wav_stereo(out_stereo / "recon_stereo_griffinlim.wav", sr_L, gl_L, gl_R)

    print("listo.")
    print(f"salida stereo: {out_stereo}")
    print("escucha: recon_stereo_fase_original.wav y recon_stereo_griffinlim.wav")


if __name__ == "__main__":
    main()
