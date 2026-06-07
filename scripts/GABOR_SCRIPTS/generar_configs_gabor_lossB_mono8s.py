import json
from pathlib import Path


out_dir = Path("configs/gabor_sweeps/lossB_mono8s_ft")
out_dir.mkdir(parents=True, exist_ok=True)

base = {
    "audio": "data/audio/RockThatBody_61s_8s_mono_44k.wav",
    "canal": "mono",
    "max_segundos": 8.0,
    "sr": 44100,
    "device": "cuda",
    "seed": 42,

    "n_atomos": 16000,
    "k_sigma": 4.0,
    "f_min_hz": 40.0,
    "f_max_hz": None,
    "sigma_inicial_samples": None,

    "epochs": 1500,
    "finetune_epochs": 300,
    "finetune_lr_scale": 0.1,
    "finetune_reset_scheduler": True,

    "log_cada": 50,

    "tipo_wave_loss": "l1",
    "lambda_wave": 1.0,
    "lambda_mse_wave": 0.0,
    "lambda_mrstft": 0.3,
    "mrstft_ffts": [512, 1024, 2048],

    "lrs": {
        "mu_t": 0.3,
        "log_sigma": 0.005,
        "amp": 0.01,
        "freq_raw": 0.01,
        "phi": 0.1
    },

    "usar_scheduler": True,
    "scheduler_factor": 0.5,
    "scheduler_paciencia": 250,
    "scheduler_min_lr": 1e-5,

    "sobreescribir_salida": True,
    "forzar_pytorch": False
}

tests = [
    {
        "tag": "base_l1_mr03",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [512, 1024, 2048],
    },
    {
        "tag": "l1_mse025_mr03",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.25,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [512, 1024, 2048],
    },
    {
        "tag": "charb_mr03",
        "tipo_wave_loss": "charbonnier",
        "charbonnier_eps": 1e-3,
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [512, 1024, 2048],
    },
    {
        "tag": "l1_mr01",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.1,
        "mrstft_ffts": [512, 1024, 2048],
    },
    {
        "tag": "l1_mr05",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.5,
        "mrstft_ffts": [512, 1024, 2048],
    },
    {
        "tag": "l1_mr03_fft256_512_1024_2048",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [256, 512, 1024, 2048],
    },
    {
        "tag": "l1_mr03_fft512_1024_2048_4096",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [512, 1024, 2048, 4096],
    },
    {
        "tag": "stage_mr05_to_wave2_mr01",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.0,
        "lambda_mrstft": 0.5,
        "mrstft_ffts": [512, 1024, 2048],
        "finetune_loss_overrides": {
            "lambda_wave": 2.0,
            "lambda_mrstft": 0.1,
            "lambda_mse_wave": 0.0
        }
    },
    {
        "tag": "stage_l1mse_to_wave2_mse05_mr01",
        "tipo_wave_loss": "l1",
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.25,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [512, 1024, 2048],
        "finetune_loss_overrides": {
            "lambda_wave": 2.0,
            "lambda_mse_wave": 0.5,
            "lambda_mrstft": 0.1
        }
    },
    {
        "tag": "charb_mse025_mr03",
        "tipo_wave_loss": "charbonnier",
        "charbonnier_eps": 1e-3,
        "lambda_wave": 1.0,
        "lambda_mse_wave": 0.25,
        "lambda_mrstft": 0.3,
        "mrstft_ffts": [512, 1024, 2048],
    },
]

for t in tests:
    cfg = dict(base)
    cfg.update(t)

    cfg["nombre_experimento"] = "gabor_rock_8s_lossB_" + t["tag"]

    path = out_dir / (cfg["nombre_experimento"] + ".json")
    cfg.pop("tag")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(path)

print("")
print(f"OK: configs generadas: {len(tests)}")
