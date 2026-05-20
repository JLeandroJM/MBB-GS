"""
Compara las corridas de chebyshev vs monomial sobre el MISMO clip.

Lee:
    resultados/exp09/exp09_chebyshev/<clip>/metricas_calidad.json
    resultados/exp09/exp09_chebyshev/<clip>/metricas_compresion.json
    resultados/exp09/exp09_chebyshev/<clip>/log_entrenamiento.csv
    (y los equivalentes para exp09_monomial)

Produce:
    resultados/exp09/comparacion_<clip>/
        tabla_comparativa.csv
        bars_calidad.png
        curvas_loss_superpuestas.png
        rd_tamano_vs_ssim.png
        comparacion.md
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt



def cargar_si_existe(ruta):
    if not os.path.isfile(ruta):
        return None
    with open(ruta) as f:
        return json.load(f)


def cargar_csv(ruta):
    if not os.path.isfile(ruta):
        return None
    filas = []
    with open(ruta) as f:
        r = csv.reader(f)
        cabecera = next(r)
        for fila in r:
            filas.append(fila)
    return cabecera, filas



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="drifting_circles")
    args = parser.parse_args()

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))
    resultados_exp09 = os.path.join(raiz, "resultados", "exp09")

    clip = args.clip
    rutas = {
        'chebyshev': os.path.join(resultados_exp09, "exp09_chebyshev", clip),
        'monomial':  os.path.join(resultados_exp09, "exp09_monomial",  clip),
    }

    datos = {}
    for base, ruta in rutas.items():
        datos[base] = {
            'calidad':  cargar_si_existe(os.path.join(ruta, "metricas_calidad.json")),
            'comp':     cargar_si_existe(os.path.join(ruta, "metricas_compresion.json")),
            'log':      cargar_csv(os.path.join(ruta, "log_entrenamiento.csv")),
        }
        if datos[base]['calidad'] is None:
            print(f"[warn] no encuentro metricas_calidad.json para {base}: {ruta}")

    if not any(datos[b]['calidad'] is not None for b in rutas):
        print("nada que comparar -- corre primero ambas corridas.", file=sys.stderr)
        sys.exit(1)

    salida = os.path.join(resultados_exp09, f"comparacion_{clip}")
    os.makedirs(salida, exist_ok=True)

    # ===== tabla comparativa CSV =========================================
    encabezado = ["metrica", "chebyshev", "monomial"]
    filas = []

    def agregar(metrica, getter):
        vals = []
        for base in ['chebyshev', 'monomial']:
            try:
                vals.append(getter(datos[base]))
            except Exception:
                vals.append("")
        filas.append([metrica] + vals)

    agregar("PSNR pre-pruning", lambda d: f"{d['calidad']['pre_pruning']['psnr_promedio']:.3f}")
    agregar("SSIM pre-pruning", lambda d: f"{d['calidad']['pre_pruning']['ssim_promedio']:.4f}")
    agregar("PSNR post-pruning", lambda d: f"{d['calidad']['post_pruning']['psnr_promedio']:.3f}")
    agregar("SSIM post-pruning", lambda d: f"{d['calidad']['post_pruning']['ssim_promedio']:.4f}")
    agregar("LPIPS post-pruning",
             lambda d: f"{d['calidad']['post_pruning']['lpips_promedio']:.4f}"
                       if d['calidad']['post_pruning']['lpips_promedio'] is not None else "n/a")
    agregar("N pre", lambda d: d['calidad']['pre_pruning']['N'])
    agregar("N post", lambda d: d['calidad']['post_pruning']['N'])
    agregar("bytes_modelo",     lambda d: d['comp']['tamano_modelo_bytes'])
    agregar("bytes_video_orig", lambda d: d['comp']['tamano_video_original_bytes'])
    agregar("ratio vs video original", lambda d: f"{d['comp']['ratio_compresion_vs_original']:.2f}x")
    agregar("AVIF q80 originales (total)",
             lambda d: d['comp']['avif_originales_por_calidad']['80']['total_bytes']
                       if isinstance(d['comp']['avif_originales_por_calidad'], dict) and '80' in d['comp']['avif_originales_por_calidad']
                       else d['comp']['avif_originales_por_calidad'][80]['total_bytes'])
    agregar("AVIF q80 rasterizados (total)",
             lambda d: d['comp']['avif_rasterizados_por_calidad']['80']['total_bytes']
                       if isinstance(d['comp']['avif_rasterizados_por_calidad'], dict) and '80' in d['comp']['avif_rasterizados_por_calidad']
                       else d['comp']['avif_rasterizados_por_calidad'][80]['total_bytes'])

    with open(os.path.join(salida, "tabla_comparativa.csv"), "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(encabezado)
        for r in filas:
            w.writerow(r)

    # ===== barras de calidad =============================================
    bases = [b for b in ['chebyshev', 'monomial'] if datos[b]['calidad'] is not None]
    psnrs = [datos[b]['calidad']['post_pruning']['psnr_promedio'] for b in bases]
    ssims = [datos[b]['calidad']['post_pruning']['ssim_promedio'] for b in bases]
    lpipss = [datos[b]['calidad']['post_pruning']['lpips_promedio'] or 0.0 for b in bases]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=100)
    for ax, vals, titulo in zip(axes, [psnrs, ssims, lpipss],
                                  ["PSNR (dB)", "SSIM", "LPIPS"]):
        ax.bar(bases, vals, color=['tab:blue', 'tab:orange'])
        ax.set_title(titulo)
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f"Comparativa de calidad post-pruning — {clip}")
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "bars_calidad.png"))
    plt.close(fig)

    # ===== curvas de loss superpuestas ===================================
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    for base, color in [('chebyshev', 'tab:blue'), ('monomial', 'tab:orange')]:
        cs = datos[base]['log']
        if cs is None:
            continue
        cabec, fil = cs
        idx_loss = cabec.index("loss_render")
        losses = [float(r[idx_loss]) for r in fil]
        ax.plot(losses, label=base, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss_render")
    ax.set_yscale("log")
    ax.set_title(f"Loss render durante training — {clip}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "curvas_loss_superpuestas.png"))
    plt.close(fig)

    # ===== Rate-Distortion ==============================================
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    for base, color in [('chebyshev', 'tab:blue'), ('monomial', 'tab:orange')]:
        if datos[base]['calidad'] is None or datos[base]['comp'] is None:
            continue
        b = datos[base]['comp']['tamano_modelo_bytes']
        s = datos[base]['calidad']['post_pruning']['ssim_promedio']
        ax.scatter([b], [s], s=120, color=color, label=base, zorder=3)
        ax.annotate(base, (b, s), textcoords="offset points", xytext=(8, 8), fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("bytes modelo")
    ax.set_ylabel("SSIM (post-pruning)")
    ax.set_title(f"Rate-Distortion: tamano vs SSIM — {clip}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "rd_tamano_vs_ssim.png"))
    plt.close(fig)

    # ===== reporte md ====================================================
    md = [f"# Comparacion chebyshev vs monomial — clip `{clip}`\n"]
    md.append("## Calidad post-pruning\n")
    md.append("| metrica | chebyshev | monomial |")
    md.append("|---|---|---|")
    for fila in filas:
        md.append("| " + " | ".join(str(x) for x in fila) + " |")
    md.append("\n## Interpretacion\n")
    if datos['chebyshev']['calidad'] and datos['monomial']['calidad']:
        psnr_c = datos['chebyshev']['calidad']['post_pruning']['psnr_promedio']
        psnr_m = datos['monomial']['calidad']['post_pruning']['psnr_promedio']
        ssim_c = datos['chebyshev']['calidad']['post_pruning']['ssim_promedio']
        ssim_m = datos['monomial']['calidad']['post_pruning']['ssim_promedio']
        if psnr_c > psnr_m + 0.3:
            md.append(f"- **Chebyshev gana** en PSNR ({psnr_c:.2f} vs {psnr_m:.2f}). "
                       "Consistente con la teoria: base ortogonal -> mejor condicionamiento "
                       "para grados altos.")
        elif psnr_m > psnr_c + 0.3:
            md.append(f"- **Monomial gana** en PSNR ({psnr_m:.2f} vs {psnr_c:.2f}). "
                       "Resultado inesperado; revisar inicializacion y/o estabilidad numerica.")
        else:
            md.append(f"- Empate en PSNR (cheb {psnr_c:.2f} vs mono {psnr_m:.2f}). "
                       "Para este clip las dos bases tienen capacidad similar.")
        md.append(f"- SSIM: chebyshev {ssim_c:.4f} | monomial {ssim_m:.4f}.")
    md.append("")

    with open(os.path.join(salida, "comparacion.md"), "w") as f:
        f.write("\n".join(md))

    print(f"listo. comparacion en: {salida}")



if __name__ == "__main__":
    main()
