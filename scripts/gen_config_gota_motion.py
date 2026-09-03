"""
Configs del experimento "MOTION-FORZADO": intentar que la gota INTERPOLE
MOVIMIENTO (una gaussiana que se mueve) en vez de cross-fade de opacidad.

Idea (ver chat): motion loss hace que la gota importe; la smoothness ASIMETRICA
prohibe el parpadeo RAPIDO de opacidad (coefs de orden alto) pero permite la
aparicion LENTA genuina -> la unica forma de ajustar los frames pasa a ser MOVER
la gaussiana. Menos gaussianas = mas presion a reusar/mover.

Barrido de beta_smoothness (la variable critica: muy bajo -> cross-fade;
muy alto -> la gota no aparece). 3 valores.

Clip: gota_mejorado_s2s4_1080p (REAL), resolucion original 1080p.

Uso:  python scripts/gen_config_gota_motion.py
"""
import copy
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from gen_configs_slowmo import base_config

DEST = RAIZ / "configs" / "slowmo"

BETAS = [("b1e6", 1e-6), ("b5e6", 5e-6), ("b2e5", 2e-5)]


def config_motion(beta):
    cfg = copy.deepcopy(base_config("gota_mejorado_s2s4_1080p", 120))
    cfg.pop("holdout", None)

    cfg["n_gaussianas_inicial"] = 40000          # menos gaussianas -> presion a mover
    cfg["escala_inicial_px"] = 4.0
    cfg["grados"] = {"mu": 40, "opacity": 20, "color": 15, "scale": 8, "theta": 6, "depth": 3}

    cfg["n_epochs"] = 300
    cfg["checkpoint_cada_n_epochs"] = 75
    cfg["guardar_checkpoints_intermedios"] = True
    cfg["sub_batch_frames"] = 2

    # motion: la gota importa
    cfg["tipo_loss"] = "motion"
    cfg["lambda_motion"] = 1.0
    cfg["motion_clip"] = 10.0
    cfg["lambda_dssim"] = 0.25

    # smoothness ASIMETRICA: castiga opacidad/color (mata el flicker), mu LIBRE
    cfg["beta_smoothness"] = beta
    cfg["pesos_smoothness"] = {"mu": 0.0, "opacity": 1.0, "color": 0.3,
                                "scale": 0.0, "theta": 0.0, "depth": 0.0}
    # aceleracion suave sobre mu (trayectoria limpia)
    cfg["lambda_accel"] = 1e-7
    cfg["accel_muestras"] = 512

    # VRAM-safe 1080p
    cfg["frames_en_gpu_uint8"] = True
    cfg["evitar_render_completo_en_train"] = True
    cfg["guardar_frames_rasterizados"] = True
    cfg["guardar_visualizaciones"] = False        # robustez: viz aparte en la Mac
    cfg["calcular_metricas"] = True
    cfg["usar_ssim"] = True
    cfg["usar_lpips"] = False
    cfg["ejecutar_pruning_post"] = False
    cfg["sobreescribir_salida"] = True
    return cfg


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for tag, beta in BETAS:
        cfg = config_motion(beta)
        nombre = f"gota_motion_{tag}"
        cfg["nombre_experimento"] = nombre
        cfg["_comentario"] = f"MOTION-FORZADO 1080p | motion + smooth asimetrica beta={beta} | 40k gauss"
        (DEST / f"{nombre}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {nombre}.json  beta_smoothness={beta}  N=40000  loss=motion")
    print(f"=== {len(BETAS)} configs en {DEST} ===")


if __name__ == "__main__":
    main()
