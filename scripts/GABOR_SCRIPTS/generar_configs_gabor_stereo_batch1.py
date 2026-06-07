import json
from pathlib import Path


out_dir = Path("configs/gabor_sweeps/stereo_batch1")
out_dir.mkdir(parents=True, exist_ok=True)

base = {
    "modo_audio": "stereo",
    "audio": "data/audio/RockThatBody_61s_30s_stereo_44k.wav",

    "max_segundos": 30.0,
    "sr": 44100,
    "device": "cuda",
    "seed": 42,

    "k_sigma": 4.0,
    "f_min_hz": 40.0,
    "f_max_hz": None,

    "epochs": 3000,
    "log_cada": 100,

    "lambda_wave": 1.0,
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
    "scheduler_paciencia": 300,
    "scheduler_min_lr": 1e-5,

    "sobreescribir_salida": True,
    "forzar_pytorch": False
}

tests = [
    {
        "tag": "N16k_sigmaAuto_wave1_mr03",
        "n_atomos": 16000,
        "sigma_inicial_samples": None,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.3,
    },
    {
        "tag": "N24k_sigmaAuto_wave1_mr03",
        "n_atomos": 24000,
        "sigma_inicial_samples": None,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.3,
    },
    {
        "tag": "N32k_sigmaAuto_wave1_mr03",
        "n_atomos": 32000,
        "sigma_inicial_samples": None,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.3,
    },
    {
        "tag": "N16k_sigma16_wave1_mr03",
        "n_atomos": 16000,
        "sigma_inicial_samples": 16,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.3,
    },
    {
        "tag": "N24k_sigma16_wave1_mr03",
        "n_atomos": 24000,
        "sigma_inicial_samples": 16,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.3,
    },
    {
        "tag": "N24k_sigma32_wave1_mr03",
        "n_atomos": 24000,
        "sigma_inicial_samples": 32,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.3,
    },
    {
        "tag": "N24k_sigmaAuto_wave1_mr01",
        "n_atomos": 24000,
        "sigma_inicial_samples": None,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.1,
    },
    {
        "tag": "N24k_sigmaAuto_wave1_mr05",
        "n_atomos": 24000,
        "sigma_inicial_samples": None,
        "lambda_wave": 1.0,
        "lambda_mrstft": 0.5,
    },
]

for t in tests:
    cfg = dict(base)
    cfg.update(t)

    cfg["nombre_experimento"] = "gabor_rock_30s_stereo_batch1_" + t["tag"]

    path = out_dir / (cfg["nombre_experimento"] + ".json")
    cfg.pop("tag")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(path)

print("")
print(f"OK: configs generadas: {len(tests)}")
