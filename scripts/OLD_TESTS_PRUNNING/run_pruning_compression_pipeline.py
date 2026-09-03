import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import torch


RULES = {
    "loose": {
        "op_mean_min": 0.05,
        "path_max": 10.0,
        "color_max": 0.10,
        "opstd_max": 0.03,
        "scale_max": 0.10,
    },
    "hard_60": {
        "op_mean_min": 0.05,
        "path_max": 60.0,
        "color_max": 0.60,
        "opstd_max": 0.125,
        "scale_max": 0.60,
    },
}


def run(cmd, capture=False, log_path=None):
    print("")
    print(">", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        [str(x) for x in cmd],
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )

    if capture:
        text = result.stdout or ""
        print(text)

        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(text, encoding="utf-8")

        return text

    return ""


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_info(path):
    ckpt = load_checkpoint(path)
    sd = ckpt["state_dict_coefs"]
    config = ckpt.get("config", {})

    n_frames = sd.get("n_frames", config.get("max_frames"))
    if n_frames is None:
        raise RuntimeError("No se pudo determinar n_frames.")

    N = sd.get("N")
    if N is None:
        for value in sd.values():
            if torch.is_tensor(value) and value.ndim >= 1:
                N = value.shape[0]
                break

    if N is None:
        raise RuntimeError("No se pudo determinar N.")

    return int(N), int(n_frames)


def file_mb(path):
    path = Path(path)
    if not path.exists():
        return None
    return path.stat().st_size / 1024 / 1024


def frames_complete(frames_dir, expected):
    frames_dir = Path(frames_dir)
    return frames_dir.exists() and len(list(frames_dir.glob("frame_*.png"))) >= expected


def create_rule_csv(stats_csv, out_csv, rule):
    stats_csv = Path(stats_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with stats_csv.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "id" not in fieldnames:
        raise RuntimeError(f"{stats_csv} no tiene columna id.")

    def value(row, key):
        return float(row[key])

    selected = [
        row for row in rows
        if value(row, "op_mean") >= rule["op_mean_min"]
        and value(row, "path_length_px") <= rule["path_max"]
        and value(row, "color_path") <= rule["color_max"]
        and value(row, "op_std") <= rule["opstd_max"]
        and value(row, "scale_std") <= rule["scale_max"]
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    return len(rows), len(selected)


def parse_metrics(text):
    patterns = {
        "psnr_mean": r"PSNR promedio\s*:\s*([0-9.+\-eEinfINF]+)",
        "psnr_min": r"PSNR min\s*:\s*([0-9.+\-eEinfINF]+)",
        "psnr_p5": r"PSNR p5\s*:\s*([0-9.+\-eEinfINF]+)",
        "psnr_max": r"PSNR max\s*:\s*([0-9.+\-eEinfINF]+)",
        "psnr_std": r"PSNR std\s*:\s*([0-9.+\-eEinfINFnanNAN]+)",
        "mse_mean": r"MSE promedio\s*:\s*([0-9.+\-eE]+)",
        "mae_mean": r"MAE promedio\s*:\s*([0-9.+\-eE]+)",
    }

    result = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        result[key] = match.group(1) if match else ""

    return result


def render_checkpoint(
    python_exe,
    script,
    checkpoint,
    output_dir,
    device,
    n_frames,
    fps,
    force,
):
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    video_path = output_dir / "video_reconstruido.mp4"

    if not force and frames_complete(frames_dir, n_frames):
        print(f"[SKIP] Ya existen {n_frames} frames en {frames_dir}")

        if not video_path.exists():
            frames_script = script.parent / "frames_a_video.py"
            run([
                python_exe,
                frames_script,
                "--frames", frames_dir,
                "--salida", video_path,
                "--fps", str(int(round(fps))),
            ])
        return frames_dir, video_path

    run([
        python_exe,
        script,
        "--checkpoint", checkpoint,
        "--salida", frames_dir,
        "--device", device,
        "--inicio", "0",
        "--fin", str(n_frames),
        "--crear_video",
        "--fps", str(int(round(fps))),
    ])

    return frames_dir, video_path


def compare_frames(
    python_exe,
    script,
    frames_a,
    frames_b,
    out_csv,
    log_path,
):
    text = run([
        python_exe,
        script,
        "--a", frames_a,
        "--b", frames_b,
        "--out", out_csv,
    ], capture=True, log_path=log_path)

    return parse_metrics(text)


def add_metrics(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rules", nargs="+", default=["loose", "hard_60"])
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument(
        "--seven_zip",
        default=r"C:\Program Files\7-Zip\7z.exe",
    )
    parser.add_argument(
        "--gt_frames",
        default=None,
        help="Carpeta opcional con frames originales para medir contra GT.",
    )
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable)

    exp = Path(args.exp)
    if not exp.is_absolute():
        exp = root / exp

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else exp / "checkpoints" / "checkpoint_final.pt"
    )

    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint

    scripts = root / "scripts"
    pipeline_dir = exp / "pipeline_pruning"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = exp / "viz_stats" / "gaussian_stats.csv"

    N_original, n_frames = checkpoint_info(checkpoint)

    print("======================================================")
    print("PIPELINE PRUNING + UINT16 ALL")
    print("======================================================")
    print(f"Experimento: {exp}")
    print(f"Checkpoint : {checkpoint}")
    print(f"N original : {N_original}")
    print(f"Frames     : {n_frames}")
    print(f"FPS        : {args.fps}")

    # --------------------------------------------------
    # 1. Estadísticas por gaussiana
    # --------------------------------------------------
    if args.force or not stats_csv.exists():
        run([
            python_exe,
            scripts / "viz_gaussian_stats.py",
            "--checkpoint", checkpoint,
            "--chunk", "2048",
            "--sample_every", str(args.sample_every),
        ])

    if not stats_csv.exists():
        raise RuntimeError(
            f"No se generó el archivo esperado: {stats_csv}"
        )

    # --------------------------------------------------
    # 2. Baseline completo
    # --------------------------------------------------
    baseline_dir = pipeline_dir / "baseline"
    baseline_frames, baseline_video = render_checkpoint(
        python_exe,
        scripts / "regenerar_clip_desde_checkpoint_streaming.py",
        checkpoint,
        baseline_dir,
        args.device,
        n_frames,
        args.fps,
        args.force,
    )

    gt_frames = Path(args.gt_frames) if args.gt_frames else None
    if gt_frames and not gt_frames.is_absolute():
        gt_frames = root / gt_frames

    rows = []

    # --------------------------------------------------
    # 3. Loose y hard_60
    # --------------------------------------------------
    for rule_name in args.rules:
        if rule_name not in RULES:
            raise RuntimeError(
                f"Regla desconocida: {rule_name}. "
                f"Disponibles: {list(RULES)}"
            )

        rule = RULES[rule_name]

        print("")
        print("======================================================")
        print(f"REGLA: {rule_name}")
        print("======================================================")

        ids_csv = pipeline_dir / "ids" / f"{rule_name}.csv"
        n_total, n_removed = create_rule_csv(
            stats_csv,
            ids_csv,
            rule,
        )

        n_remaining = n_total - n_removed
        pct_removed = 100.0 * n_removed / n_total

        print(f"Eliminadas : {n_removed}")
        print(f"Restantes  : {n_remaining}")
        print(f"Porcentaje : {pct_removed:.4f}%")

        # ----------------------------------------------
        # 3.1 Crear checkpoint pruneado FP32
        # ----------------------------------------------
        pruned_ckpt = (
            pipeline_dir
            / "checkpoints"
            / f"checkpoint_{rule_name}_fp32.pt"
        )
        pruned_ckpt.parent.mkdir(parents=True, exist_ok=True)

        if args.force or not pruned_ckpt.exists():
            run([
                python_exe,
                scripts / "viz_prune_checkpoint_by_stats.py",
                "--checkpoint", checkpoint,
                "--ids_csv", ids_csv,
                "--modo", "remove_ids",
                "--salida", pruned_ckpt,
            ])

        # ----------------------------------------------
        # 3.2 Render pruneado FP32
        # ----------------------------------------------
        pruned_render_dir = (
            pipeline_dir / "renders" / f"{rule_name}_fp32"
        )

        pruned_frames, pruned_video = render_checkpoint(
            python_exe,
            scripts / "regenerar_clip_desde_checkpoint_streaming.py",
            pruned_ckpt,
            pruned_render_dir,
            args.device,
            n_frames,
            args.fps,
            args.force,
        )

        # ----------------------------------------------
        # 3.3 Métrica baseline vs pruneado
        # ----------------------------------------------
        metrics_dir = pipeline_dir / "metricas"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        m_base_pruned = compare_frames(
            python_exe,
            scripts / "comparar_frames_psnr.py",
            baseline_frames,
            pruned_frames,
            metrics_dir / f"baseline_vs_{rule_name}_fp32.csv",
            metrics_dir / f"baseline_vs_{rule_name}_fp32.txt",
        )

        # ----------------------------------------------
        # 3.4 UINT16 ALL
        # ----------------------------------------------
        u16_pkg = (
            pipeline_dir
            / "checkpoints"
            / f"checkpoint_{rule_name}_u16all.pkg.pt"
        )

        if args.force or not u16_pkg.exists():
            run([
                python_exe,
                scripts / "pack_checkpoint_uint16_all.py",
                "--in_ckpt", pruned_ckpt,
                "--out_pkg", u16_pkg,
                "--omit_zero_depth_high",
            ])

        u16_render_ckpt = (
            pipeline_dir
            / "checkpoints"
            / f"checkpoint_{rule_name}_u16all_render.pt"
        )

        if args.force or not u16_render_ckpt.exists():
            run([
                python_exe,
                scripts / "unpack_checkpoint_uint16_all.py",
                "--in_pkg", u16_pkg,
                "--out_ckpt", u16_render_ckpt,
            ])

        # ----------------------------------------------
        # 3.5 Render UINT16 ALL
        # ----------------------------------------------
        u16_render_dir = (
            pipeline_dir / "renders" / f"{rule_name}_u16all"
        )

        u16_frames, u16_video = render_checkpoint(
            python_exe,
            scripts / "regenerar_clip_desde_checkpoint_streaming.py",
            u16_render_ckpt,
            u16_render_dir,
            args.device,
            n_frames,
            args.fps,
            args.force,
        )

        # ----------------------------------------------
        # 3.6 Métricas UINT16 ALL
        # ----------------------------------------------
        m_pruned_u16 = compare_frames(
            python_exe,
            scripts / "comparar_frames_psnr.py",
            pruned_frames,
            u16_frames,
            metrics_dir / f"{rule_name}_fp32_vs_u16all.csv",
            metrics_dir / f"{rule_name}_fp32_vs_u16all.txt",
        )

        m_base_u16 = compare_frames(
            python_exe,
            scripts / "comparar_frames_psnr.py",
            baseline_frames,
            u16_frames,
            metrics_dir / f"baseline_vs_{rule_name}_u16all.csv",
            metrics_dir / f"baseline_vs_{rule_name}_u16all.txt",
        )

        # ----------------------------------------------
        # 3.7 Métricas contra GT opcional
        # ----------------------------------------------
        m_gt_pruned = {}
        m_gt_u16 = {}

        if gt_frames:
            m_gt_pruned = compare_frames(
                python_exe,
                scripts / "comparar_frames_psnr.py",
                gt_frames,
                pruned_frames,
                metrics_dir / f"gt_vs_{rule_name}_fp32.csv",
                metrics_dir / f"gt_vs_{rule_name}_fp32.txt",
            )

            m_gt_u16 = compare_frames(
                python_exe,
                scripts / "comparar_frames_psnr.py",
                gt_frames,
                u16_frames,
                metrics_dir / f"gt_vs_{rule_name}_u16all.csv",
                metrics_dir / f"gt_vs_{rule_name}_u16all.txt",
            )

        # ----------------------------------------------
        # 3.8 Compresión 7z lossless
        # ----------------------------------------------
        seven_zip_path = Path(args.seven_zip)
        u16_7z = u16_pkg.with_suffix(".7z")

        if seven_zip_path.exists():
            if args.force and u16_7z.exists():
                u16_7z.unlink()

            if not u16_7z.exists():
                run([
                    seven_zip_path,
                    "a",
                    "-t7z",
                    u16_7z,
                    u16_pkg,
                    "-mx=9",
                    "-y",
                ])
        else:
            print(f"[WARN] No se encontró 7-Zip: {seven_zip_path}")

        # ----------------------------------------------
        # 3.9 Fila consolidada
        # ----------------------------------------------
        row = {
            "regla": rule_name,
            "op_mean_min": rule["op_mean_min"],
            "path_max": rule["path_max"],
            "color_max": rule["color_max"],
            "opstd_max": rule["opstd_max"],
            "scale_max": rule["scale_max"],
            "n_original": n_total,
            "n_eliminadas": n_removed,
            "n_restantes": n_remaining,
            "pct_eliminado": f"{pct_removed:.4f}",
            "checkpoint_original_mb": file_mb(checkpoint),
            "checkpoint_pruned_fp32_mb": file_mb(pruned_ckpt),
            "u16all_pkg_mb": file_mb(u16_pkg),
            "u16all_7z_mb": file_mb(u16_7z),
            "video_pruned_fp32_mb": file_mb(pruned_video),
            "video_u16all_mb": file_mb(u16_video),
        }

        add_metrics(row, "baseline_vs_pruned", m_base_pruned)
        add_metrics(row, "pruned_vs_u16all", m_pruned_u16)
        add_metrics(row, "baseline_vs_u16all", m_base_u16)

        if m_gt_pruned:
            add_metrics(row, "gt_vs_pruned", m_gt_pruned)

        if m_gt_u16:
            add_metrics(row, "gt_vs_u16all", m_gt_u16)

        rows.append(row)

    # --------------------------------------------------
    # 4. CSV consolidado
    # --------------------------------------------------
    summary_csv = pipeline_dir / "resumen_pipeline_pruning.csv"

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("======================================================")
    print("PIPELINE FINALIZADO")
    print("======================================================")
    print(f"Resumen: {summary_csv}")
    print(f"Resultados: {pipeline_dir}")


if __name__ == "__main__":
    main()


