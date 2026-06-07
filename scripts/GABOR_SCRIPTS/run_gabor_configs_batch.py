import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[2]


def leer_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def limpiar_nombre(nombre):
    return str(nombre).replace("\\", "/").strip("/").replace("/", "__")


def exp_completo(exp_dir):
    stereo_ok = (
        (exp_dir / "metricas_stereo.json").exists()
        and (exp_dir / "recon_stereo.wav").exists()
        and (exp_dir / "LEFT" / "checkpoints" / "checkpoint_final.pt").exists()
        and (exp_dir / "RIGHT" / "checkpoints" / "checkpoint_final.pt").exists()
    )

    mono_ok = (
        (exp_dir / "metricas.json").exists()
        and (exp_dir / "recon.wav").exists()
        and (exp_dir / "checkpoints" / "checkpoint_final.pt").exists()
    )

    return stereo_ok or mono_ok


def leer_metricas(exp_dir):
    stereo_path = exp_dir / "metricas_stereo.json"

    if stereo_path.exists():
        m = leer_json(stereo_path)
        cfg_path = exp_dir / "config_stereo_original.json"
        cfg = leer_json(cfg_path) if cfg_path.exists() else {}

        row = {
            "experimento": exp_dir.name,
            "modo": "stereo",
            "snr_db": m.get("snr_stereo_db", ""),
            "psnr_db": m.get("psnr_stereo_db", ""),
            "mse_wave": m.get("mse_wave_stereo", ""),
            "si_sdr_db": m.get("si_sdr_db", ""),
            "lsd_db_promedio": m.get("lsd_db_promedio", ""),
            "mel_l1_log": m.get("mel_l1_log", ""),
            "mrstft_final": m.get("mrstft_final", ""),
            "spectral_overshoot_final": m.get("spectral_overshoot_final", ""),
            "snr_left_db": m.get("snr_left_db", ""),
            "snr_right_db": m.get("snr_right_db", ""),
            "snr_promedio_lr_db": m.get("snr_promedio_lr_db", ""),
            "psnr_left_db": m.get("psnr_left_db", ""),
            "psnr_right_db": m.get("psnr_right_db", ""),
            "psnr_promedio_lr_db": m.get("psnr_promedio_lr_db", ""),
            "loss_final": "",
            "ratio_compresion_vs_wav": m.get("ratio_compresion_vs_wav_stereo", ""),
            "bytes_modelo": m.get("bytes_modelo_total", ""),
            "bytes_wav_int16": m.get("bytes_wav_int16_stereo", ""),
            "n_atomos": m.get("n_atomos_total", ""),
            "n_atomos_por_canal": m.get("n_atomos_por_canal", cfg.get("n_atomos", "")),
            "samples": m.get("samples", ""),
            "sr": m.get("sr", ""),
            "duracion_s": m.get("duracion_s", ""),
            "canal": "stereo",
            "epochs": cfg.get("epochs", ""),
            "lambda_wave": cfg.get("lambda_wave", ""),
            "lambda_mrstft": cfg.get("lambda_mrstft", ""),
            "sigma_inicial_samples": cfg.get("sigma_inicial_samples", ""),
            "k_sigma": cfg.get("k_sigma", ""),
            "mrstft_ffts": json.dumps(cfg.get("mrstft_ffts", [])),
            "ruta": str(exp_dir),
        }

        lrs = cfg.get("lrs", {})
        for k in ["mu_t", "log_sigma", "amp", "freq_raw", "phi"]:
            row[f"lr_{k}"] = lrs.get(k, "")

        return row

    metricas_path = exp_dir / "metricas.json"
    config_path = exp_dir / "config_usada.json"
    info_path = exp_dir / "info_audio.json"

    metricas = leer_json(metricas_path) if metricas_path.exists() else {}
    config = leer_json(config_path) if config_path.exists() else {}
    info = leer_json(info_path) if info_path.exists() else {}

    row = {
        "experimento": exp_dir.name,
        "modo": "mono",
        "snr_db": metricas.get("snr_db", ""),
        "psnr_db": metricas.get("psnr_db", ""),
        "mse_wave": metricas.get("mse_wave", ""),
        "si_sdr_db": metricas.get("si_sdr_db", ""),
        "lsd_db_promedio": metricas.get("lsd_db_promedio", ""),
        "mel_l1_log": metricas.get("mel_l1_log", ""),
        "mrstft_final": metricas.get("mrstft_final", ""),
        "spectral_overshoot_final": metricas.get("spectral_overshoot_final", ""),
        "snr_left_db": "",
        "snr_right_db": "",
        "snr_promedio_lr_db": "",
        "psnr_left_db": "",
        "psnr_right_db": "",
        "psnr_promedio_lr_db": "",
        "loss_final": metricas.get("loss_final", ""),
        "ratio_compresion_vs_wav": metricas.get("ratio_compresion_vs_wav", ""),
        "bytes_modelo": metricas.get("bytes_modelo", ""),
        "bytes_wav_int16": metricas.get("bytes_wav_int16", ""),
        "n_atomos": metricas.get("n_atomos", config.get("n_atomos", "")),
        "n_atomos_por_canal": metricas.get("n_atomos", config.get("n_atomos", "")),
        "samples": metricas.get("samples", info.get("samples", "")),
        "sr": metricas.get("sr", info.get("sr", "")),
        "duracion_s": metricas.get("duracion_s", info.get("duracion_s", "")),
        "canal": config.get("canal", info.get("canal", "")),
        "epochs": config.get("epochs", ""),
        "lambda_wave": config.get("lambda_wave", ""),
        "lambda_mrstft": config.get("lambda_mrstft", ""),
        "sigma_inicial_samples": config.get("sigma_inicial_samples", ""),
        "k_sigma": config.get("k_sigma", ""),
        "mrstft_ffts": json.dumps(config.get("mrstft_ffts", [])),
        "ruta": str(exp_dir),
    }

    lrs = config.get("lrs", {})
    for k in ["mu_t", "log_sigma", "amp", "freq_raw", "phi"]:
        row[f"lr_{k}"] = lrs.get(k, "")

    return row


def generar_resumen(batch_root):
    filas = []

    for exp_dir in sorted(batch_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith("_"):
            continue

        if (exp_dir / "metricas_stereo.json").exists() or (exp_dir / "metricas.json").exists():
            filas.append(leer_metricas(exp_dir))

    def score(row):
        try:
            return float(row.get("snr_db", -999))
        except Exception:
            return -999

    filas.sort(key=score, reverse=True)

    resumen_dir = batch_root / "_resumen"
    resumen_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resumen_dir / "resumen_gabor_batch.csv"
    txt_path = resumen_dir / "resumen_gabor_batch.txt"

    campos = [
        "experimento",
        "modo",
        "snr_db",
        "psnr_db",
        "mse_wave",
        "si_sdr_db",
        "lsd_db_promedio",
        "mel_l1_log",
        "mrstft_final",
        "spectral_overshoot_final",
        "snr_left_db",
        "snr_right_db",
        "snr_promedio_lr_db",
        "psnr_left_db",
        "psnr_right_db",
        "psnr_promedio_lr_db",
        "loss_final",
        "ratio_compresion_vs_wav",
        "bytes_modelo",
        "bytes_wav_int16",
        "n_atomos",
        "n_atomos_por_canal",
        "samples",
        "sr",
        "duracion_s",
        "canal",
        "epochs",
        "lambda_wave",
        "lambda_mrstft",
        "sigma_inicial_samples",
        "k_sigma",
        "mrstft_ffts",
        "lr_mu_t",
        "lr_log_sigma",
        "lr_amp",
        "lr_freq_raw",
        "lr_phi",
        "ruta",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    def fmt_float(x, default=-1.0):
        try:
            return float(x)
        except Exception:
            return default

    def fmt_int(x, default=-1):
        try:
            return int(x)
        except Exception:
            return default

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RANKING GABOR BATCH - ordenado por snr_db\n")
        f.write("=" * 145 + "\n")
        f.write(
            f"{'#':>2} | {'experimento':54s} | {'modo':>6} | {'SNR':>8} | {'SI-SDR':>8} | {'LSD':>8} | {'MelL1':>8} | "
            f"{'MRSTFT':>8} | {'Oversht':>8} | {'PSNR':>8} | {'MSE':>10} | {'ratio':>8} | {'Ntot':>8}\n"
        )
        f.write("-" * 145 + "\n")

        for i, r in enumerate(filas, start=1):
            f.write(
                f"{i:>2} | "
                f"{r['experimento'][:54]:54s} | "
                f"{str(r['modo'])[:6]:>6s} | "
                f"{fmt_float(r['snr_db']):8.4f} | "
                f"{fmt_float(r.get('si_sdr_db', '')):8.4f} | "
                f"{fmt_float(r.get('lsd_db_promedio', '')):8.4f} | "
                f"{fmt_float(r.get('mel_l1_log', '')):8.4f} | "
                f"{fmt_float(r.get('mrstft_final', '')):8.4f} | "
                f"{fmt_float(r.get('spectral_overshoot_final', '')):8.4f} | "
                f"{fmt_float(r['psnr_db']):8.4f} | "
                f"{fmt_float(r['mse_wave']):10.6f} | "
                f"{fmt_float(r['ratio_compresion_vs_wav']):8.3f} | "
                f"{fmt_int(r['n_atomos']):8d}\n"
            )

    return csv_path, txt_path, filas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--continuar-si-falla", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    batch_name = args.batch_name
    if not batch_name:
        batch_name = "gabor_batch_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_name = limpiar_nombre(batch_name)

    batch_root = RAIZ / "outputs" / "gabor" / "_batches" / batch_name
    configs_dir = batch_root / "_configs_usadas"
    logs_dir = batch_root / "_logs"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    configs = [Path(x) for x in args.configs]
    configs = [p if p.is_absolute() else RAIZ / p for p in configs]

    print("============================================================")
    print(" BATCH GABOR CONFIGS")
    print("============================================================")
    print(f"batch : {batch_name}")
    print(f"root  : {batch_root}")
    print(f"configs: {len(configs)}")
    print(f"skip completed: {args.skip_completed}")
    print("============================================================")

    estados = []

    for idx, cfg_path in enumerate(configs, start=1):
        cfg = leer_json(cfg_path)

        nombre_original = cfg.get("nombre_experimento") or cfg_path.stem
        nombre_limpio = limpiar_nombre(nombre_original)

        cfg_batch = dict(cfg)
        cfg_batch["nombre_experimento"] = f"_batches/{batch_name}/{nombre_limpio}"
        cfg_batch["sobreescribir_salida"] = bool(cfg.get("sobreescribir_salida", True))

        cfg_batch_path = configs_dir / cfg_path.name

        with open(cfg_batch_path, "w", encoding="utf-8") as f:
            json.dump(cfg_batch, f, indent=2)

        shutil.copy2(cfg_path, configs_dir / (cfg_path.stem + ".original.json"))

        exp_dir = batch_root / nombre_limpio

        if args.skip_completed and exp_completo(exp_dir):
            print("------------------------------------------------------------")
            print(f"[{idx}/{len(configs)}] SKIP completado: {nombre_limpio}")
            estados.append((nombre_limpio, "SKIP_COMPLETED", 0.0))
            continue

        modo_audio = str(cfg.get("modo_audio", "mono")).lower().strip()

        if modo_audio == "stereo":
            script_train = RAIZ / "scripts" / "GABOR_SCRIPTS" / "train_gabor_stereo_config.py"
        else:
            script_train = RAIZ / "scripts" / "GABOR_SCRIPTS" / "train_gabor.py"

        log_path = logs_dir / f"{idx:02d}_{nombre_limpio}.log"

        cmd = [
            sys.executable,
            str(script_train),
            "--config",
            str(cfg_batch_path),
        ]

        print("------------------------------------------------------------")
        print(f"[{idx}/{len(configs)}] {cfg_path.name}")
        print(f"modo: {modo_audio}")
        print(f"experimento original: {nombre_original}")
        print(f"experimento batch   : {cfg_batch['nombre_experimento']}")
        print(f"log: {log_path}")
        print("------------------------------------------------------------")

        t0 = time.time()

        with open(log_path, "w", encoding="utf-8", errors="replace") as flog:
            flog.write(f"CONFIG_ORIGINAL: {cfg_path}\n")
            flog.write(f"CONFIG_BATCH: {cfg_batch_path}\n")
            flog.write(f"EXPERIMENTO_BATCH: {cfg_batch['nombre_experimento']}\n")
            flog.write(f"COMANDO: {' '.join(cmd)}\n\n")
            flog.flush()

            p = subprocess.Popen(
                cmd,
                cwd=str(RAIZ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for line in p.stdout:
                print(line, end="")
                flog.write(line)
                flog.flush()

            code = p.wait()

        dur = time.time() - t0
        estado = "OK" if code == 0 else f"FALLO_{code}"
        estados.append((nombre_limpio, estado, dur))

        generar_resumen(batch_root)

        print(f"[{estado}] terminado en {dur / 60.0:.2f} min")

        if code != 0 and not args.continuar_si_falla:
            print("Se detuvo porque una config fallo.")
            break

    estado_path = batch_root / "_estado_batch.txt"
    with open(estado_path, "w", encoding="utf-8") as f:
        for nombre, estado, dur in estados:
            f.write(f"{estado:16s} | {dur / 60.0:8.2f} min | {nombre}\n")

    csv_path, txt_path, filas = generar_resumen(batch_root)

    print("============================================================")
    print("RESUMEN FINAL")
    print("============================================================")

    for nombre, estado, dur in estados:
        print(f"{estado:16s} | {dur / 60.0:8.2f} min | {nombre}")

    print("")
    print(f"batch root: {batch_root}")
    print(f"estado    : {estado_path}")
    print(f"resumen   : {txt_path}")
    print(f"csv       : {csv_path}")


if __name__ == "__main__":
    main()
