import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy import signal


RAIZ = Path(__file__).resolve().parents[2]


def leer_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def guardar_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def audio_to_float32(data):
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128.0) / 128.0
    else:
        y = data.astype(np.float32)

    y = np.nan_to_num(y)
    return np.clip(y, -1.0, 1.0)


def cargar_audio_stereo(path, sr_objetivo=None, max_segundos=None):
    sr, data = wavfile.read(str(path))
    y = audio_to_float32(data)

    if y.ndim == 1:
        y = np.stack([y, y], axis=1)

    if y.shape[1] > 2:
        y = y[:, :2]

    if sr_objetivo is not None and int(sr_objetivo) != int(sr):
        gcd = np.gcd(int(sr), int(sr_objetivo))
        up = int(sr_objetivo) // gcd
        down = int(sr) // gcd

        left = signal.resample_poly(y[:, 0], up, down).astype(np.float32)
        right = signal.resample_poly(y[:, 1], up, down).astype(np.float32)

        n = min(len(left), len(right))
        y = np.stack([left[:n], right[:n]], axis=1)
        sr = int(sr_objetivo)

    if max_segundos is not None:
        n = int(float(max_segundos) * int(sr))
        y = y[:n]

    return int(sr), y


def leer_wav_float(path):
    sr, data = wavfile.read(str(path))
    y = audio_to_float32(data)
    return sr, y


def guardar_wav_stereo(path, sr, left, right):
    n = min(len(left), len(right))
    y = np.stack([left[:n], right[:n]], axis=1).astype(np.float32)
    y = np.nan_to_num(y)

    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs > 1.0:
        y = y / max_abs

    wavfile.write(
        str(path),
        int(sr),
        (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16),
    )


def metricas_wave(original, recon):
    n = min(len(original), len(recon))
    original = original[:n].astype(np.float32)
    recon = recon[:n].astype(np.float32)

    err = original - recon
    mse = float(np.mean(err * err))
    power = float(np.mean(original * original))
    snr = 10.0 * np.log10(power / max(mse, 1e-12))
    psnr = 10.0 * np.log10((2.0 ** 2) / max(mse, 1e-12))

    return mse, float(snr), float(psnr)


def leer_metricas_canal(path):
    if not path.exists():
        return {}
    return leer_json(path)


def escribir_metricas_txt(path, metricas):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metricas.items():
            if isinstance(v, float):
                f.write(f"{k}={v:.8f}\n")
            else:
                f.write(f"{k}={v}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    ruta_cfg = Path(args.config)
    if not ruta_cfg.is_absolute():
        ruta_cfg = (RAIZ / ruta_cfg).resolve()

    cfg = leer_json(ruta_cfg)

    if str(cfg.get("modo_audio", "stereo")).lower().strip() != "stereo":
        print("AVISO: modo_audio no es stereo, pero se ejecutara como stereo.")

    nombre_base = cfg.get("nombre_experimento") or ruta_cfg.stem
    out_stereo = RAIZ / "outputs" / "gabor" / nombre_base

    out_stereo.mkdir(parents=True, exist_ok=True)
    (out_stereo / "_tmp_configs").mkdir(parents=True, exist_ok=True)

    guardar_json(out_stereo / "config_stereo_original.json", cfg)

    print("============================================================")
    print(" TRAIN GABOR STEREO")
    print("============================================================")
    print(f"config base : {ruta_cfg}")
    print(f"salida      : {out_stereo}")
    print(f"LEFT        : {out_stereo / 'LEFT'}")
    print(f"RIGHT       : {out_stereo / 'RIGHT'}")
    print("============================================================")

    base_cfg = dict(cfg)
    base_cfg.pop("modo_audio", None)

    left_cfg = dict(base_cfg)
    left_cfg["canal"] = "left"
    left_cfg["nombre_experimento"] = f"{nombre_base}/LEFT"
    left_cfg["sobreescribir_salida"] = True

    right_cfg = dict(base_cfg)
    right_cfg["canal"] = "right"
    right_cfg["nombre_experimento"] = f"{nombre_base}/RIGHT"
    right_cfg["sobreescribir_salida"] = True

    left_cfg_path = out_stereo / "_tmp_configs" / "config_LEFT.json"
    right_cfg_path = out_stereo / "_tmp_configs" / "config_RIGHT.json"

    guardar_json(left_cfg_path, left_cfg)
    guardar_json(right_cfg_path, right_cfg)

    train_script = RAIZ / "scripts" / "GABOR_SCRIPTS" / "train_gabor.py"

    print("")
    print("=== ENTRENANDO CANAL LEFT ===")
    subprocess.check_call([
        sys.executable,
        str(train_script),
        "--config",
        str(left_cfg_path),
    ], cwd=str(RAIZ))

    print("")
    print("=== ENTRENANDO CANAL RIGHT ===")
    subprocess.check_call([
        sys.executable,
        str(train_script),
        "--config",
        str(right_cfg_path),
    ], cwd=str(RAIZ))

    print("")
    print("=== UNIENDO STEREO ===")

    left_dir = out_stereo / "LEFT"
    right_dir = out_stereo / "RIGHT"

    sr_l, recon_l = leer_wav_float(left_dir / "recon.wav")
    sr_r, recon_r = leer_wav_float(right_dir / "recon.wav")

    if sr_l != sr_r:
        raise RuntimeError(f"Sample rates distintos: LEFT={sr_l}, RIGHT={sr_r}")

    sr = sr_l
    n = min(len(recon_l), len(recon_r))
    recon_l = recon_l[:n]
    recon_r = recon_r[:n]

    guardar_wav_stereo(out_stereo / "recon_stereo.wav", sr, recon_l, recon_r)

    audio_path = Path(cfg["audio"])
    if not audio_path.is_absolute():
        audio_path = RAIZ / audio_path

    sr_orig, original = cargar_audio_stereo(
        audio_path,
        sr_objetivo=cfg.get("sr"),
        max_segundos=cfg.get("max_segundos"),
    )

    if sr_orig != sr:
        raise RuntimeError(f"Sample rate original distinto: original={sr_orig}, recon={sr}")

    original = original[:n]
    guardar_wav_stereo(out_stereo / "target_stereo.wav", sr, original[:, 0], original[:, 1])

    mse_l, snr_l, psnr_l = metricas_wave(original[:, 0], recon_l)
    mse_r, snr_r, psnr_r = metricas_wave(original[:, 1], recon_r)

    recon_st = np.stack([recon_l, recon_r], axis=1)
    mse_st = float(np.mean((original - recon_st) ** 2))
    power_st = float(np.mean(original ** 2))
    snr_st = 10.0 * np.log10(power_st / max(mse_st, 1e-12))
    psnr_st = 10.0 * np.log10((2.0 ** 2) / max(mse_st, 1e-12))

    met_l = leer_metricas_canal(left_dir / "metricas.json")
    met_r = leer_metricas_canal(right_dir / "metricas.json")

    bytes_modelo_total = int(met_l.get("bytes_modelo", 0)) + int(met_r.get("bytes_modelo", 0))
    bytes_wav_stereo = int(n * 2 * 2)  # samples * canales * int16
    ratio_stereo = bytes_wav_stereo / max(1, bytes_modelo_total)

    metricas = {
        "experimento": nombre_base,
        "sr": sr,
        "samples": int(n),
        "duracion_s": float(n / sr),

        "snr_left_db": snr_l,
        "snr_right_db": snr_r,
        "snr_promedio_lr_db": (snr_l + snr_r) / 2.0,
        "snr_stereo_db": float(snr_st),

        "psnr_left_db": psnr_l,
        "psnr_right_db": psnr_r,
        "psnr_promedio_lr_db": (psnr_l + psnr_r) / 2.0,
        "psnr_stereo_db": float(psnr_st),

        "mse_left": mse_l,
        "mse_right": mse_r,
        "mse_promedio_lr": (mse_l + mse_r) / 2.0,
        "mse_wave_stereo": mse_st,

        "n_atomos_por_canal": int(cfg.get("n_atomos", -1)),
        "n_atomos_total": int(cfg.get("n_atomos", 0)) * 2,

        "bytes_modelo_total": bytes_modelo_total,
        "bytes_wav_int16_stereo": bytes_wav_stereo,
        "ratio_compresion_vs_wav_stereo": ratio_stereo,

        "snr_left_train_json": met_l.get("snr_db", ""),
        "snr_right_train_json": met_r.get("snr_db", ""),
        "loss_left_final": met_l.get("loss_final", ""),
        "loss_right_final": met_r.get("loss_final", ""),
    }

    guardar_json(out_stereo / "metricas_stereo.json", metricas)
    escribir_metricas_txt(out_stereo / "metricas_stereo.txt", metricas)

    print("")
    print("============================================================")
    print(" LISTO GABOR STEREO")
    print("============================================================")
    print(f"salida stereo: {out_stereo}")
    print(f"recon        : {out_stereo / 'recon_stereo.wav'}")
    print(f"metricas     : {out_stereo / 'metricas_stereo.txt'}")
    print(f"SNR_ST       : {snr_st:.4f} dB")
    print(f"PSNR_ST      : {psnr_st:.4f} dB")
    print(f"MSE_ST       : {mse_st:.8f}")
    print(f"ratio stereo : {ratio_stereo:.4f}x")


if __name__ == "__main__":
    main()
