"""
Cruza los `metricas.json` de todos los experimentos y produce tablas + graficos
comparativos.

Outputs:
  - tabla_resumen.csv  : una fila por (experimento, clip) con metricas agregadas
  - rd_global.png      : curva Rate-Distortion comparando exp05 (tu metodo) vs
                         exp07 (H.264/H.265), por clip
  - bars_exp_vs_psnr.png: barras agrupadas comparando PSNR de cada exp por clip

Uso:
    python tools/comparar_experimentos.py
"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTOS_ORDEN = [
    "exp00_baseline_estatico",
    "exp01_solo_posicion_temporal",
    "exp02_posicion_opacidad_temporal",
    "exp03_full_temporal",
    "exp04_grado_polinomio",
    "exp05_numero_gaussianas",
    "exp06_cuantizacion",
    "exp07_codecs_clasicos",
]



def leer_metricas_simples(ruta_json):
    """Lee metricas.json y devuelve dict con campos comunes (PSNR, SSIM, bytes, ratio)."""
    if not os.path.isfile(ruta_json):
        return None
    with open(ruta_json) as f:
        d = json.load(f)
    return d



def recolectar(resultados_dir):
    """
    Recorre resultados/<exp>/<clip>/metricas.json y arma una lista de filas
    para la tabla resumen.
    """
    filas = []
    for exp in EXPERIMENTOS_ORDEN:
        carpeta_exp = os.path.join(resultados_dir, exp)
        if not os.path.isdir(carpeta_exp):
            continue
        for clip in sorted(os.listdir(carpeta_exp)):
            ruta_json = os.path.join(carpeta_exp, clip, "metricas.json")
            d = leer_metricas_simples(ruta_json)
            if d is None:
                continue
            # extraemos los campos disponibles, lo que no este lo dejamos vacio
            filas.append({
                "exp": exp,
                "clip": clip,
                "psnr": d.get("psnr_promedio"),
                "ssim": d.get("ssim_promedio"),
                "lpips": d.get("lpips_promedio"),
                "bytes": d.get("bytes_modelo"),
                "ratio": d.get("ratio_compresion"),
                "tiempo_seg": d.get("tiempo_total_seg") or d.get("tiempo_entrenamiento_seg"),
            })
    return filas



def escribir_tabla_csv(filas, ruta):
    cabecera = ["exp", "clip", "PSNR", "SSIM", "LPIPS", "bytes", "ratio", "tiempo_seg"]
    with open(ruta, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        for d in filas:
            w.writerow([
                d["exp"], d["clip"],
                f"{d['psnr']:.3f}" if d["psnr"] is not None else "",
                f"{d['ssim']:.4f}" if d["ssim"] is not None else "",
                f"{d['lpips']:.4f}" if d["lpips"] is not None else "",
                d["bytes"] if d["bytes"] is not None else "",
                f"{d['ratio']:.2f}" if d["ratio"] is not None else "",
                f"{d['tiempo_seg']:.1f}" if d["tiempo_seg"] is not None else "",
            ])



def graficar_rd_global_por_clip(resultados_dir, salida_dir):
    """
    Para cada clip, junta los puntos (bytes, PSNR) de exp05 y exp07 en un grafico.
    Estos son los dos experimentos que producen curvas R-D directamente.
    """
    exp05_dir = os.path.join(resultados_dir, "exp05_numero_gaussianas")
    exp07_dir = os.path.join(resultados_dir, "exp07_codecs_clasicos")

    if not os.path.isdir(exp05_dir) and not os.path.isdir(exp07_dir):
        print("no hay resultados de exp05 ni exp07 — salto el grafico R-D global")
        return

    # buscamos clips comunes
    clips = set()
    if os.path.isdir(exp05_dir):
        clips |= set(os.listdir(exp05_dir))
    if os.path.isdir(exp07_dir):
        clips |= set(os.listdir(exp07_dir))

    for clip in sorted(clips):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        algo_dibujado = False

        # exp05: leer metricas.json del clip (tiene psnr_promedio_por_N y bytes_modelo_por_N)
        ruta_e05 = os.path.join(exp05_dir, clip, "metricas.json")
        if os.path.isfile(ruta_e05):
            d = leer_metricas_simples(ruta_e05)
            if d and "bytes_modelo_por_N" in d:
                bytes_arr = d["bytes_modelo_por_N"]
                psnr_arr = d["psnr_promedio_por_N"]
                ax.plot(bytes_arr, psnr_arr, 'o-', label="2DGS Temporal (exp05)", color='tab:red', markersize=8)
                for x, y, n in zip(bytes_arr, psnr_arr, d.get("Ns", range(len(bytes_arr)))):
                    ax.annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
                algo_dibujado = True

        # exp07: leer metricas.json (tiene tabla por codec/crf)
        ruta_e07 = os.path.join(exp07_dir, clip, "metricas.json")
        if os.path.isfile(ruta_e07):
            d = leer_metricas_simples(ruta_e07)
            if d and "tabla" in d:
                tabla = d["tabla"]
                for codec in sorted(set(r["codec"] for r in tabla)):
                    sub = [r for r in tabla if r["codec"] == codec]
                    bytes_arr = [r["bytes"] for r in sub]
                    psnr_arr  = [r["PSNR"]  for r in sub]
                    ax.plot(bytes_arr, psnr_arr, 's-', label=codec, alpha=0.85)
                algo_dibujado = True

        if not algo_dibujado:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        ax.set_xlabel("bytes")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"Rate-Distortion comparativo — {clip}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        ruta_png = os.path.join(salida_dir, f"rd_global_{clip}.png")
        fig.savefig(ruta_png)
        plt.close(fig)
        print(f"  generado: {ruta_png}")



def graficar_barras_psnr(filas, ruta):
    """Barras agrupadas: PSNR de cada experimento por clip."""
    if not filas:
        return
    # filtrar a experimentos que reportan un solo PSNR agregado (exp00..exp03)
    exps_simples = ["exp00_baseline_estatico", "exp01_solo_posicion_temporal",
                     "exp02_posicion_opacidad_temporal", "exp03_full_temporal"]
    filas_f = [f for f in filas if f["exp"] in exps_simples and f["psnr"] is not None]
    if not filas_f:
        return

    clips = sorted(set(f["clip"] for f in filas_f))
    exps = [e for e in exps_simples if any(f["exp"] == e for f in filas_f)]

    width = 0.8 / max(len(exps), 1)
    x = np.arange(len(clips))

    fig, ax = plt.subplots(figsize=(max(8, len(clips) * 1.5), 5), dpi=100)
    for i, exp in enumerate(exps):
        valores = []
        for clip in clips:
            v = next((f["psnr"] for f in filas_f if f["exp"] == exp and f["clip"] == clip), None)
            valores.append(v if v is not None else 0)
        ax.bar(x + (i - len(exps)/2 + 0.5) * width, valores, width=width, label=exp)

    ax.set_xticks(x)
    ax.set_xticklabels(clips, rotation=20, ha='right')
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Comparacion de PSNR por clip y experimento")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, ".."))
    resultados_dir = os.path.join(raiz, "resultados")
    salida_dir = os.path.join(raiz, "resultados", "_resumen_global")
    os.makedirs(salida_dir, exist_ok=True)

    filas = recolectar(resultados_dir)
    print(f"recolectadas {len(filas)} filas de metricas")

    escribir_tabla_csv(filas, os.path.join(salida_dir, "tabla_resumen.csv"))
    print(f"escrita tabla: {os.path.join(salida_dir, 'tabla_resumen.csv')}")

    graficar_rd_global_por_clip(resultados_dir, salida_dir)
    graficar_barras_psnr(filas, os.path.join(salida_dir, "bars_exp_vs_psnr.png"))
    print(f"\nlisto. resumen en: {salida_dir}")



if __name__ == "__main__":
    main()
