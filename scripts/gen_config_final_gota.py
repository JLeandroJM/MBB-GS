"""
Genera las 2 configs del EXPERIMENTO DEFINITIVO de slow-mo sobre
video_final_gota_1080p (1920x1080, 60fps, 255 frames, shutter 1/1800):

  slowmo_final_gota_FULL.json  -> entrena TODOS los frames (sin holdout).
                                  Es el PRODUCTO: de aqui sale el slow-mo x3.
  slowmo_final_gota_TERM.json  -> termometro holdout-x2 (entrena pares, mide
                                  impares con GT). Da el NUMERO de interpolacion.

Dimensionado a full-res 1080p: mas gaussianas, degradado VRAM-safe
(frames_en_gpu_uint8 + metricas streaming).

Uso:  python scripts/gen_config_final_gota.py
"""
import copy
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from gen_configs_slowmo import base_config

DEST = RAIZ / "configs" / "slowmo"
CLIP = "video_final_gota_1080p"


def config_definitivo():
    cfg = copy.deepcopy(base_config(CLIP, 255))

    # --- resolucion / capacidad (1080p) ---
    cfg["n_gaussianas_inicial"] = 150000
    cfg["escala_inicial_px"] = 4.0
    cfg["grados"] = {"mu": 50, "opacity": 25, "color": 15, "scale": 8, "theta": 6, "depth": 4}

    # --- entrenamiento ---
    cfg["n_epochs"] = 600
    cfg["checkpoint_cada_n_epochs"] = 300
    cfg["sub_batch_frames"] = 2
    cfg["tipo_loss"] = "baseline"
    cfg["lambda_dssim"] = 0.25

    # --- palancas (defaults por teoria; el ablation de gota las refina) ---
    cfg["beta_smoothness"] = 1e-7
    cfg["pesos_smoothness"] = {"mu": 0.0, "opacity": 1.0, "color": 1.0,
                                "scale": 0.0, "theta": 0.0, "depth": 0.0}
    cfg["lambda_accel"] = 1e-7
    cfg["accel_muestras"] = 512

    # --- VRAM-safe para 1080p ---
    cfg["frames_en_gpu_uint8"] = True
    cfg["evitar_render_completo_en_train"] = True     # metricas streaming, no apila
    cfg["guardar_frames_rasterizados"] = True         # para reconstruido.mp4
    cfg["calcular_metricas"] = True
    cfg["usar_ssim"] = True
    cfg["usar_lpips"] = False                          # activar al final si quieres
    cfg["ejecutar_pruning_post"] = False
    cfg["sobreescribir_salida"] = True
    return cfg


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    # FULL: producto (todos los frames, sin holdout) -> render x3
    full = config_definitivo()
    full.pop("holdout", None)
    full["nombre_experimento"] = "slowmo_final_gota_FULL"
    full["_comentario"] = "DEFINITIVO slow-mo | video_final_gota 1080p | entrena TODO -> render x3"
    (DEST / "slowmo_final_gota_FULL.json").write_text(
        json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")

    # TERM: termometro holdout-x2 (mide interpolacion con GT real)
    term = config_definitivo()
    term["holdout"] = {"modo": "multiplos", "k": 2, "anclar_bordes": 4}
    term["nombre_experimento"] = "slowmo_final_gota_TERM"
    term["_comentario"] = "DEFINITIVO termometro | holdout-x2 | mide PSNR interpolado"
    (DEST / "slowmo_final_gota_TERM.json").write_text(
        json.dumps(term, indent=2, ensure_ascii=False), encoding="utf-8")

    # FAST: version chica para entrenar RAPIDO (deadline). 480x270, menos
    # gaussianas, 400 epochs, checkpoints intermedios cada 100 (para agarrar antes).
    fast = config_definitivo()
    fast.pop("holdout", None)
    fast["clip"] = "video_final_gota_480x270"
    fast["n_gaussianas_inicial"] = 12000
    fast["grados"] = {"mu": 40, "opacity": 20, "color": 12, "scale": 6, "theta": 5, "depth": 3}
    fast["escala_inicial_px"] = 2.5
    fast["n_epochs"] = 400
    fast["checkpoint_cada_n_epochs"] = 100
    fast["guardar_checkpoints_intermedios"] = True     # <- checkpoint cada 100 epochs
    fast["accel_muestras"] = 512
    fast["evitar_render_completo_en_train"] = False     # a 480x270 no hace falta streaming
    fast["nombre_experimento"] = "slowmo_final_gota_FAST"
    fast["_comentario"] = "RAPIDO (deadline) | 480x270 12k gauss 400ep | checkpoints cada 100"
    (DEST / "slowmo_final_gota_FAST.json").write_text(
        json.dumps(fast, indent=2, ensure_ascii=False), encoding="utf-8")

    # MID: intermedio 960x540 (mitad de 1080p). Mas nitido que FAST, ~2-3h.
    mid = config_definitivo()
    mid.pop("holdout", None)
    mid["clip"] = "video_final_gota_960x540"
    mid["n_gaussianas_inicial"] = 30000
    mid["grados"] = {"mu": 50, "opacity": 25, "color": 15, "scale": 8, "theta": 6, "depth": 4}
    mid["escala_inicial_px"] = 3.0
    mid["n_epochs"] = 300
    mid["checkpoint_cada_n_epochs"] = 75
    mid["guardar_checkpoints_intermedios"] = True
    mid["evitar_render_completo_en_train"] = True       # streaming (960x540)
    # --- FIX gota: motion loss up-pesa el objeto que se mueve; palancas OFF
    # (suprimian el transitorio rapido). Ver por que en el chat. ---
    mid["tipo_loss"] = "motion"
    mid["lambda_motion"] = 1.0
    mid["motion_clip"] = 10.0
    mid["beta_smoothness"] = 0.0
    mid["pesos_smoothness"] = None
    mid["lambda_accel"] = 0.0
    mid["accel_muestras"] = 512
    mid["nombre_experimento"] = "slowmo_final_gota_MID"
    mid["_comentario"] = "INTERMEDIO | 960x540 30k 300ep | loss=motion palancas OFF (para que la gota aparezca)"
    (DEST / "slowmo_final_gota_MID.json").write_text(
        json.dumps(mid, indent=2, ensure_ascii=False), encoding="utf-8")

    for n in ("slowmo_final_gota_FULL.json", "slowmo_final_gota_TERM.json",
              "slowmo_final_gota_FAST.json", "slowmo_final_gota_MID.json"):
        c = json.loads((DEST / n).read_text())
        print(f"  {n}: N={c['n_gaussianas_inicial']} grados_mu={c['grados']['mu']} "
              f"epochs={c['n_epochs']} holdout={'holdout' in c}")
    print(f"=== 2 configs en {DEST} ===")


if __name__ == "__main__":
    main()
