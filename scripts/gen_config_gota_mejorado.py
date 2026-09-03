"""
Config del experimento GOTA_MEJORADO: video 1920x1080 60fps bien expuesto, gota
grande y nitida. Usamos SOLO el recorte sec 2-4 (120 frames, donde caen mas gotas)
a RESOLUCION ORIGINAL 1080p.

Incorpora el drop-fix aprendido: tipo_loss=motion (up-pesa el objeto que se mueve)
y palancas OFF (smoothness/accel suprimian el transitorio rapido). VRAM-safe.

Uso:  python scripts/gen_config_gota_mejorado.py
"""
import copy
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from gen_configs_slowmo import base_config

DEST = RAIZ / "configs" / "slowmo"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_config("gota_mejorado_s2s4_1080p", 120))
    cfg.pop("holdout", None)                       # producto: entrena TODO -> render x3

    # resolucion original 1080p, 120 frames (2s)
    cfg["n_gaussianas_inicial"] = 100000
    cfg["escala_inicial_px"] = 4.0
    cfg["grados"] = {"mu": 40, "opacity": 20, "color": 15, "scale": 8, "theta": 6, "depth": 4}

    cfg["n_epochs"] = 300
    cfg["checkpoint_cada_n_epochs"] = 75
    cfg["guardar_checkpoints_intermedios"] = True   # fallback si vas justo de tiempo
    cfg["sub_batch_frames"] = 2

    # drop-fix: motion loss + palancas OFF (para que la gota aparezca)
    cfg["tipo_loss"] = "motion"
    cfg["lambda_motion"] = 1.0
    cfg["motion_clip"] = 10.0
    cfg["lambda_dssim"] = 0.25
    cfg["beta_smoothness"] = 0.0
    cfg["pesos_smoothness"] = None
    cfg["lambda_accel"] = 0.0

    # VRAM-safe 1080p
    cfg["frames_en_gpu_uint8"] = True
    cfg["evitar_render_completo_en_train"] = True
    cfg["guardar_frames_rasterizados"] = True
    cfg["calcular_metricas"] = True
    cfg["usar_ssim"] = True
    cfg["usar_lpips"] = False
    cfg["ejecutar_pruning_post"] = False
    cfg["sobreescribir_salida"] = True

    cfg["nombre_experimento"] = "slowmo_gota_mejorado_s2s4"
    cfg["_comentario"] = "GOTA_MEJORADO sec2-4 1080p | 100k | loss=motion palancas OFF | render x3"

    ruta = DEST / "slowmo_gota_mejorado_s2s4.json"
    ruta.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {ruta.name}: clip={cfg['clip']} N={cfg['n_gaussianas_inicial']} "
          f"res=1080p frames=120 loss={cfg['tipo_loss']} epochs={cfg['n_epochs']}")

    # variante 100 frames (para el slow-mo x5). Reusa el MISMO clip ya subido,
    # solo limita a las primeras 100 frames. El factor x5 va en render_slowmo.
    cfg100 = copy.deepcopy(cfg)
    cfg100["max_frames"] = 100
    cfg100["nombre_experimento"] = "slowmo_gota_mej_100f"
    cfg100["_comentario"] = "GOTA_MEJORADO 100 frames 1080p | loss=motion palancas OFF | render x5"
    ruta100 = DEST / "slowmo_gota_mej_100f.json"
    ruta100.write_text(json.dumps(cfg100, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {ruta100.name}: clip={cfg100['clip']} max_frames=100 -> render x5 (n_out=496)")


if __name__ == "__main__":
    main()
