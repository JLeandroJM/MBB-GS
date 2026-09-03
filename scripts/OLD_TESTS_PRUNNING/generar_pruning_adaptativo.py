import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = (
    "path_length_px",
    "color_path",
    "op_std",
    "scale_std",
)


def read_float(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def percentile_ranks(values):
    """
    Convierte los valores a rangos relativos [0, 1].
    El valor más pequeño recibe aproximadamente 0.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / (n - 1)

    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--porcentajes",
        nargs="+",
        type=float,
        default=[15, 20, 25, 30],
    )
    parser.add_argument("--op_mean_min", type=float, default=0.05)

    # Protege gaussianas breves con picos altos de opacidad.
    parser.add_argument("--transient_op_max", type=float, default=0.90)
    parser.add_argument("--transient_active_frac", type=float, default=0.10)

    # Pesos del score.
    parser.add_argument("--w_path", type=float, default=0.30)
    parser.add_argument("--w_color", type=float, default=0.25)
    parser.add_argument("--w_opstd", type=float, default=0.25)
    parser.add_argument("--w_scale", type=float, default=0.20)

    args = parser.parse_args()

    stats_path = Path(args.stats_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with stats_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "id" not in fieldnames:
        raise RuntimeError("El CSV debe contener la columna 'id'.")

    for metric in METRICS:
        if metric not in fieldnames:
            raise RuntimeError(f"Falta la columna requerida: {metric}")

    n_total = len(rows)

    candidate_indices = []
    protected_indices = []

    for i, row in enumerate(rows):
        op_mean = read_float(row, "op_mean")
        op_max = read_float(row, "op_max")
        active_frac = read_float(row, "active_frac", 1.0)

        is_transient = (
            op_max >= args.transient_op_max
            and active_frac <= args.transient_active_frac
        )

        if is_transient:
            protected_indices.append(i)
            continue

        if op_mean >= args.op_mean_min:
            candidate_indices.append(i)

    if not candidate_indices:
        raise RuntimeError("No se encontraron gaussianas candidatas.")

    metric_values = {}

    for metric in METRICS:
        values = [
            read_float(rows[i], metric)
            for i in candidate_indices
        ]
        metric_values[metric] = percentile_ranks(values)

    scores = (
        args.w_path * metric_values["path_length_px"]
        + args.w_color * metric_values["color_path"]
        + args.w_opstd * metric_values["op_std"]
        + args.w_scale * metric_values["scale_std"]
    )

    sorted_local = np.argsort(scores, kind="mergesort")

    summary_rows = []

    print("======================================================")
    print("PRUNING ADAPTATIVO")
    print("======================================================")
    print(f"Stats             : {stats_path}")
    print(f"N total           : {n_total}")
    print(f"Candidatas        : {len(candidate_indices)}")
    print(f"Transitorias protegidas: {len(protected_indices)}")
    print("")
    print("Score:")
    print(
        f"{args.w_path:.2f}*path + "
        f"{args.w_color:.2f}*color + "
        f"{args.w_opstd:.2f}*op_std + "
        f"{args.w_scale:.2f}*scale"
    )
    print("")

    out_fields = fieldnames + [
        "adaptive_score",
        "adaptive_rank",
    ]

    for pct in args.porcentajes:
        target = int(round(n_total * pct / 100.0))

        if target > len(candidate_indices):
            raise RuntimeError(
                f"No hay suficientes candidatas para eliminar {pct}%: "
                f"objetivo={target}, candidatas={len(candidate_indices)}"
            )

        selected_local = sorted_local[:target]
        selected_global = [
            candidate_indices[int(local_idx)]
            for local_idx in selected_local
        ]

        selected_rows = []

        for rank, local_idx in enumerate(selected_local):
            global_idx = candidate_indices[int(local_idx)]
            output_row = dict(rows[global_idx])
            output_row["adaptive_score"] = (
                f"{float(scores[int(local_idx)]):.12g}"
            )
            output_row["adaptive_rank"] = str(rank)
            selected_rows.append(output_row)

        pct_name = (
            str(int(pct))
            if float(pct).is_integer()
            else str(pct).replace(".", "_")
        )

        out_csv = out_dir / f"adaptativo_{pct_name}.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(selected_rows)

        remaining = n_total - target
        cutoff_score = float(scores[int(selected_local[-1])])

        summary_rows.append({
            "regla": f"adaptativo_{pct_name}",
            "porcentaje_objetivo": pct,
            "n_original": n_total,
            "n_eliminadas": target,
            "n_restantes": remaining,
            "cutoff_score": cutoff_score,
            "candidatas": len(candidate_indices),
            "transitorias_protegidas": len(protected_indices),
            "csv": str(out_csv),
        })

        print(
            f"adaptativo_{pct_name:>2s}: "
            f"elimina={target:6d} "
            f"restantes={remaining:6d} "
            f"({pct:5.2f}%) "
            f"cutoff={cutoff_score:.6f}"
        )

    summary_path = out_dir / "resumen_adaptativo.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
