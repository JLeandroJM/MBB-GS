"""
Genera las configs de la ablacion de SLOW-MO (interpolacion sub-frame) a partir
de una base comun. Escribe JSONs en configs/slowmo/.

Cada variante de ablacion incluye el TERMOMETRO holdout-x2
({modo: multiplos, k: 2, anclar_bordes: 4}): entrena con frames pares, deja los
impares como test de interpolacion. analizar_holdout.py separa PSNR/SSIM
train vs retenidos. NO es el experimento de reconstruccion descartado; es solo
el medidor para elegir receta.

La variante "FINAL" (sin holdout, entrena con TODOS los frames) se genera aparte
para el render x3 una vez elegida la mejor palanca.

Uso:
    python scripts/gen_configs_slowmo.py
    python scripts/gen_configs_slowmo.py --clip estructura_s6s14_480x270 --tag estr
"""
import argparse
import copy
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DEST = RAIZ / "configs" / "slowmo"


def base_config(clip, max_frames):
    return {
        "_comentario": "ablacion slow-mo (interpolacion sub-frame) | termometro holdout-x2",
        "nombre_experimento": None,                  # lo pone cada variante
        "clip": clip,
        "video_mp4": None,
        "max_frames": max_frames,
        "device": "cuda",
        "seed": 42,

        "n_gaussianas_inicial": 8000,
        "inicializar_color_desde_frame0": True,
        "escala_inicial_px": 2.5,

        "grados": {"mu": 30, "opacity": 13, "color": 10, "scale": 4, "theta": 4, "depth": 3},

        "lrs": {
            "mu_a0": 0.001, "mu_high": 0.0075,
            "opacity_a0": 0.05, "opacity_high": 0.04,
            "color_a0": 0.01, "color_high": 0.008,
            "scale_a0": 0.005, "scale_high": 0.004,
            "theta_a0": 0.005, "theta_high": 0.0025,
            "depth_a0": 0.001, "depth_high": 0.0001,
        },

        # TERMOMETRO: entrena pares, mide impares. Se quita en la config FINAL.
        "holdout": {"modo": "multiplos", "k": 2, "anclar_bordes": 4},

        "n_epochs": 600,
        "checkpoint_cada_n_epochs": 300,
        "sub_batch_frames": 2,

        "tipo_loss": "baseline",
        "lambda_dssim": 0.25,
        "lambda_mse": 0.0, "lambda_motion": 0.0, "lambda_hard": 0.0,
        "lambda_edge": 0.0, "lambda_temporal": 0.0,
        "exponente_pixel": 1.0, "exponente_frame": 1.0,
        "usar_max_pixel": False, "usar_max_frame": False, "usar_pnorm_root": False,

        # palancas (off por defecto; cada variante las prende)
        "beta_smoothness": 0.0,
        "pesos_smoothness": None,
        "lambda_accel": 0.0,
        "accel_muestras": 600,

        # rasterizador
        "usar_cuda_tiled": True, "cuda_tile_size": 16, "cuda_k_sigma": 3.0,

        # scheduler
        "usar_scheduler_lento": True, "scheduler_lento_window": 200,
        "scheduler_lento_min_delta": 0.0001, "scheduler_lento_factor": 0.5,
        "scheduler_lento_min_lr": 1e-7, "scheduler_lento_cooldown": 150,

        # rendimiento / salida
        "frames_en_gpu_uint8": True,
        "evitar_render_completo_en_train": False,
        "ejecutar_pruning_post": False,
        "calcular_metricas": True, "usar_ssim": True, "usar_lpips": False,
        "calcular_compresion": False,
        "guardar_gif": False, "guardar_visualizaciones": False,
        "guardar_frames_rasterizados": False, "guardar_checkpoints_intermedios": False,
        "guardar_verificacion_visual": False,
        "sobreescribir_salida": True,
    }


# (sufijo, overrides) -- solo cambian estos campos respecto a la base.
VARIANTES = [
    # 0) control: sin ninguna palanca
    ("A0_control", {}),

    # 1) smoothness ASIMETRICA: castiga opacity/color (mata cross-fade), suelta mu
    ("A1_smoothasym_b1e7", {
        "beta_smoothness": 1e-7,
        "pesos_smoothness": {"mu": 0.0, "opacity": 1.0, "color": 1.0,
                              "scale": 0.0, "theta": 0.0, "depth": 0.0},
    }),
    ("A2_smoothasym_b1e6", {
        "beta_smoothness": 1e-6,
        "pesos_smoothness": {"mu": 0.0, "opacity": 1.0, "color": 1.0,
                              "scale": 0.0, "theta": 0.0, "depth": 0.0},
    }),

    # 2) aceleracion de mu (trayectoria suave). Dos magnitudes.
    ("A3_accel_1e7", {"lambda_accel": 1e-7}),
    ("A4_accel_1e6", {"lambda_accel": 1e-6}),

    # 3) combo: smoothness asimetrica + aceleracion
    ("A5_combo", {
        "beta_smoothness": 1e-7,
        "pesos_smoothness": {"mu": 0.0, "opacity": 1.0, "color": 1.0,
                              "scale": 0.0, "theta": 0.0, "depth": 0.0},
        "lambda_accel": 1e-7,
    }),

    # 4) barrido de grado de mu
    ("A6_gradomu20", {"grados": {"mu": 20, "opacity": 13, "color": 10, "scale": 4, "theta": 4, "depth": 3}}),
    ("A7_gradomu40", {"grados": {"mu": 40, "opacity": 13, "color": 10, "scale": 4, "theta": 4, "depth": 3}}),

    # 5) capacidad: mas gaussianas (nitidez por frame)
    ("A8_g16k", {"n_gaussianas_inicial": 16000}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="gota_s6s16_480x270")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--tag", default="gota", help="prefijo corto para los nombres")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    base = base_config(args.clip, args.max_frames)

    escritos = []
    for sufijo, overrides in VARIANTES:
        cfg = copy.deepcopy(base)
        cfg.update(overrides)
        nombre = f"slowmo_{args.tag}_{sufijo}"
        cfg["nombre_experimento"] = nombre
        ruta = DEST / f"{nombre}.json"
        ruta.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        escritos.append(ruta.name)

    # config FINAL: sin holdout (entrena TODOS los frames), para el render x3.
    # Arranca = control; se editara a mano con la palanca ganadora.
    final = copy.deepcopy(base)
    final.pop("holdout", None)
    final["nombre_experimento"] = f"slowmo_{args.tag}_FINAL_alltrain"
    final["_comentario"] = "FINAL slow-mo: entrena TODOS los frames (sin holdout) -> render x3"
    (DEST / f"slowmo_{args.tag}_FINAL_alltrain.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    escritos.append(f"slowmo_{args.tag}_FINAL_alltrain.json")

    print(f"=== {len(escritos)} configs en {DEST} ===")
    for n in escritos:
        print(f"  {n}")


if __name__ == "__main__":
    main()
