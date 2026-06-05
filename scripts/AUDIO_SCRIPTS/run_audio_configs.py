import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


RAIZ = Path(__file__).resolve().parents[2]




def resolver_carpeta_salida(nombre):
    """
    Busca primero en outputs/audio/<experimento>.
    Si no existe, usa outputs/<experimento> para compatibilidad con corridas antiguas.
    """
    salida_audio = RAIZ / "outputs" / "audio" / nombre
    salida_root = RAIZ / "outputs" / nombre

    if (salida_audio / "metricas_audio.txt").exists():
        return salida_audio

    if (salida_root / "metricas_audio.txt").exists():
        return salida_root

    return salida_audio


def leer_json(ruta):
    with open(ruta, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def leer_metricas_txt(ruta):
    datos = {}

    if not ruta.exists():
        return datos

    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or "=" not in linea:
                continue

            k, v = linea.split("=", 1)
            k = k.strip()
            v = v.strip()

            try:
                datos[k] = float(v)
            except ValueError:
                datos[k] = v

    return datos


def experimento_completo(carpeta_salida):
    return (
        (carpeta_salida / "metricas_audio.txt").exists()
        and (carpeta_salida / "recon_fase_original.wav").exists()
    )


def recolectar_resumen(configs, carpeta_logs):
    filas = []

    for cfg_path in configs:
        try:
            cfg = leer_json(cfg_path)
        except Exception:
            continue

        nombre = cfg.get("nombre_experimento") or cfg.get("nombre") or cfg_path.stem
        salida = resolver_carpeta_salida(nombre)

        metricas = leer_metricas_txt(salida / "metricas_audio.txt")

        fila = {
            "config": cfg_path.name,
            "experimento": nombre,
            "tipo_loss": cfg.get("tipo_loss", ""),
            "n_gaussianas": cfg.get("n_gaussianas", ""),
            "epochs": cfg.get("epochs", ""),
            "batch_temporal": cfg.get("batch_temporal", ""),
            "grado_mu_f": cfg.get("grado_mu_f", ""),
            "grado_sigma_f": cfg.get("grado_sigma_f", ""),
            "grado_amp": cfg.get("grado_amp", ""),
            "usar_render_cuda_audio": cfg.get("usar_render_cuda_audio", ""),
            "usar_loss_cuda_audio": cfg.get("usar_loss_cuda_audio", ""),
            "lambda_mse": cfg.get("lambda_mse", ""),
            "lambda_motion": cfg.get("lambda_motion", ""),
            "lambda_hard": cfg.get("lambda_hard", ""),
            "lambda_temporal": cfg.get("lambda_temporal", ""),
            "lambda_freq_edge": cfg.get("lambda_freq_edge", ""),
            "lambda_dssim": cfg.get("lambda_dssim", ""),
            "loss_final": metricas.get("loss_final", ""),
            "mse_wave": metricas.get("mse_wave", ""),
            "snr_db": metricas.get("snr_db", ""),
            "psnr_logmag_promedio": metricas.get("psnr_logmag_promedio", ""),
            "psnr_logmag_min": metricas.get("psnr_logmag_min", ""),
            "psnr_logmag_max": metricas.get("psnr_logmag_max", ""),
            "psnr_logmag_p5": metricas.get("psnr_logmag_p5", ""),
            "psnr_logmag_std": metricas.get("psnr_logmag_std", ""),
            "salida": str(salida),
        }

        filas.append(fila)

    if not filas:
        return None

    campos = list(filas[0].keys())

    ruta_csv = carpeta_logs / "resumen_audio_configs.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    ruta_txt = carpeta_logs / "resumen_audio_configs.txt"
    ordenadas = sorted(
        filas,
        key=lambda x: float(x["psnr_logmag_promedio"]) if x["psnr_logmag_promedio"] != "" else -999,
        reverse=True,
    )

    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("RESUMEN AUDIO CONFIGS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Ordenado por psnr_logmag_promedio desc\n\n")

        for r in ordenadas:
            f.write(
                f"{r['experimento']}\n"
                f"  loss={r['tipo_loss']}  N={r['n_gaussianas']}  "
                f"render_cuda={r['usar_render_cuda_audio']}  fused_cuda={r['usar_loss_cuda_audio']}\n"
                f"  psnr_prom={r['psnr_logmag_promedio']}  "
                f"min={r['psnr_logmag_min']}  p5={r['psnr_logmag_p5']}  "
                f"snr={r['snr_db']}  loss_final={r['loss_final']}\n"
                f"  salida={r['salida']}\n\n"
            )

    return ruta_csv, ruta_txt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="lista de configs. Si no se pasa, corre configs/audio/*.json",
    )
    parser.add_argument("--continuar-si-falla", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    if args.configs:
        configs = [Path(x) for x in args.configs]
    else:
        configs = sorted((RAIZ / "configs" / "audio").glob("*.json"))

    configs = [p if p.is_absolute() else RAIZ / p for p in configs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_logs = RAIZ / "outputs" / "_batch_audio_logs" / f"batch_audio_{timestamp}"
    carpeta_logs.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(" BATCH AUDIO CONFIGS")
    print("============================================================")
    print(f"configs: {len(configs)}")
    print(f"logs   : {carpeta_logs}")
    print(f"skip completed: {args.skip_completed}")
    print("============================================================")

    resumen_estado = []

    for idx, cfg_path in enumerate(configs, start=1):
        cfg = leer_json(cfg_path)
        nombre = cfg.get("nombre_experimento") or cfg.get("nombre") or cfg_path.stem
        salida = RAIZ / "outputs"/ "audio" / nombre

        if args.skip_completed and experimento_completo(salida):
            print("------------------------------------------------------------")
            print(f"[{idx}/{len(configs)}] SKIP completado: {cfg_path.name}")
            print(f"salida: {salida}")
            resumen_estado.append((cfg_path.name, "SKIP_COMPLETED", 0.0))
            continue

        ruta_log = carpeta_logs / f"{idx:02d}_{cfg_path.stem}.log"

        print("------------------------------------------------------------")
        print(f"[{idx}/{len(configs)}] {cfg_path.name}")
        print(f"experimento: {nombre}")
        print(f"log: {ruta_log}")
        print("------------------------------------------------------------")

        cmd = [
            sys.executable,
            str(RAIZ / "scripts" / "train_audio_config.py"),
            "--config",
            str(cfg_path),
        ]

        inicio = time.time()

        with open(ruta_log, "w", encoding="utf-8", errors="replace") as f:
            f.write(f"CONFIG: {cfg_path}\n")
            f.write(f"EXPERIMENTO: {nombre}\n")
            f.write(f"COMANDO: {' '.join(cmd)}\n\n")
            f.flush()

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

            for linea in p.stdout:
                print(linea, end="")
                f.write(linea)
                f.flush()

            codigo = p.wait()

        dur = time.time() - inicio
        estado = "OK" if codigo == 0 else f"FALLO_{codigo}"
        resumen_estado.append((cfg_path.name, estado, dur))

        print(f"[{estado}] terminado en {dur / 60.0:.2f} min")

        if codigo != 0 and not args.continuar_si_falla:
            print("Se detuvo porque una config fallo. Usa --continuar-si-falla para seguir.")
            break

    ruta_estado = carpeta_logs / "estado_batch_audio.txt"
    with open(ruta_estado, "w", encoding="utf-8") as f:
        for cfg_name, estado, dur in resumen_estado:
            f.write(f"{estado:16s} | {dur / 60.0:8.2f} min | {cfg_name}\n")

    rutas_resumen = recolectar_resumen(configs, carpeta_logs)

    print("============================================================")
    print("RESUMEN DE EJECUCION")
    print("============================================================")
    for cfg_name, estado, dur in resumen_estado:
        print(f"{estado:16s} | {dur / 60.0:8.2f} min | {cfg_name}")

    print(f"\nEstado guardado en: {ruta_estado}")

    if rutas_resumen is not None:
        ruta_csv, ruta_txt = rutas_resumen
        print(f"Resumen CSV: {ruta_csv}")
        print(f"Resumen TXT: {ruta_txt}")


if __name__ == "__main__":
    main()
