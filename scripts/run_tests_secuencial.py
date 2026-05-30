from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


# ============================================================
# CONFIGS A EJECUTAR
# ============================================================

#CONFIGS = [
#    "configs/config_loss_01_baseline.json",
#    "configs/config_loss_02_motion.json",
#    "configs/config_loss_03_motion_temporal.json",
#    "configs/config_loss_04_l1_mse.json",
#    "configs/config_loss_05_hard.json",
#    "configs/config_loss_06_edge.json",
#    "configs/config_loss_07_temporal_loss.json",
#]

'''
CONFIGS = [
    "configs/config_loss_01_baseline.json",
    "configs/config_loss_02_motion.json",
    "configs/config_loss_03_motion_temporal.json",
    "configs/config_loss_05_hard.json",
    "configs/config_loss_08_hard_05.json",
    "configs/config_loss_09_combo_hard_motion.json",
]'''

CONFIGS = [
    #"configs/config_loss_05_hard2.json",
    "configs/config_loss_09_combo_hard_motion2.json",
]

# Si True: si un test falla, sigue con el siguiente.
# Si False: si un test falla, se detiene todo.
CONTINUAR_SI_FALLA = True


def main():
    raiz = Path(__file__).resolve().parents[1]
    script_train = raiz / "scripts" / "train.py"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_logs = raiz / "outputs" / "batch_tests" / f"batch_{timestamp}"
    carpeta_logs.mkdir(parents=True, exist_ok=True)

    resumen = []

    print("============================================================")
    print(" EJECUCION SECUENCIAL DE TESTS")
    print("============================================================")
    print(f"Raiz proyecto : {raiz}")
    print(f"Python usado  : {sys.executable}")
    print(f"Logs en       : {carpeta_logs}")
    print("")

    inicio_total = time.time()

    for idx, config_rel in enumerate(CONFIGS, start=1):
        config_path = raiz / config_rel

        if not config_path.exists():
            msg = f"[ERROR] No existe config: {config_path}"
            print(msg)
            resumen.append((config_rel, "NO_EXISTE", 0.0))
            if not CONTINUAR_SI_FALLA:
                break
            continue

        nombre_log = f"{idx:02d}_{config_path.stem}.log"
        ruta_log = carpeta_logs / nombre_log

        print("------------------------------------------------------------")
        print(f"[{idx}/{len(CONFIGS)}] Ejecutando: {config_rel}")
        print(f"Log: {ruta_log}")
        print("------------------------------------------------------------")

        comando = [
            sys.executable,
            str(script_train),
            "--config",
            str(config_path),
        ]

        inicio = time.time()

        with open(ruta_log, "w", encoding="utf-8", errors="replace") as f:
            f.write("============================================================\n")
            f.write(f"CONFIG: {config_rel}\n")
            f.write(f"INICIO: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"COMANDO: {' '.join(comando)}\n")
            f.write("============================================================\n\n")
            f.flush()

            proceso = subprocess.Popen(
                comando,
                cwd=str(raiz),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            for linea in proceso.stdout:
                print(linea, end="")
                f.write(linea)
                f.flush()

            codigo = proceso.wait()

        duracion = time.time() - inicio
        estado = "OK" if codigo == 0 else f"FALLO_{codigo}"
        resumen.append((config_rel, estado, duracion))

        print("")
        print(f"[{estado}] {config_rel} terminado en {duracion / 60:.1f} min")
        time.sleep(10)
        print("")

        if codigo != 0 and not CONTINUAR_SI_FALLA:
            print("Se detiene la ejecucion porque CONTINUAR_SI_FALLA=False")
            break

    duracion_total = time.time() - inicio_total

    ruta_resumen = carpeta_logs / "resumen_batch.txt"
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        f.write("RESUMEN DE TESTS\n")
        f.write("================\n")
        f.write(f"Inicio batch: {timestamp}\n")
        f.write(f"Duracion total min: {duracion_total / 60:.2f}\n\n")

        for config_rel, estado, duracion in resumen:
            f.write(f"{estado:12s} | {duracion / 60:8.2f} min | {config_rel}\n")

    print("================================================------------")
    print("RESUMEN")
    print("================================================------------")
    for config_rel, estado, duracion in resumen:
        print(f"{estado:12s} | {duracion / 60:8.2f} min | {config_rel}")

    print("")
    print(f"Resumen guardado en: {ruta_resumen}")


if __name__ == "__main__":
    main()