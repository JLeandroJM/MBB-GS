import argparse
import csv
from pathlib import Path


BASE = {
    "op_mean_min": 0.05,
    "path_max": 60.0,
    "color_max": 0.60,
    "opstd_max": 0.125,
    "scale_max": 0.60,
}


RULES = {
    # base
    "base_hard60": BASE,


    # solo cambia op_std
    
    "only_opstd_020": {
        **BASE,
        "opstd_max": 5,
    },

    # solo cambia scale
  
    "only_scale_090": {
        **BASE,
        "scale_max": 100,
    },

    # solo cambia op_mean
    # bajar op_mean_min permite eliminar gaussianas con menor opacidad media

}


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def has_id_column(fieldnames):
    for c in ["id", "idx", "gaussian_id", "gauss_id", "indice"]:
        if c in fieldnames:
            return c
    return None


def keep(row, rule):
    return (
        f(row, "op_mean") >= rule["op_mean_min"] and
        f(row, "path_length_px") <= rule["path_max"] and
        f(row, "color_path") <= rule["color_max"] and
        f(row, "op_std") <= rule["opstd_max"] and
        f(row, "scale_std") <= rule["scale_max"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    stats_path = Path(args.stats_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    id_col = has_id_column(fieldnames)

    if id_col is None:
        fieldnames_out = ["id"] + fieldnames
        for i, row in enumerate(rows):
            row["id"] = str(i)
    else:
        fieldnames_out = fieldnames

    print("=== Variantes one-at-a-time desde hard_60 ===")
    print(f"stats: {stats_path}")
    print(f"N total: {len(rows)}")
    print(f"id_col: {id_col if id_col else 'NO HABIA, se creo id por indice'}")
    print("")

    resumen_rows = []

    for name, rule in RULES.items():
        selected = [r for r in rows if keep(r, rule)]
        out_csv = out_dir / f"{name}.csv"

        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_out)
            writer.writeheader()
            writer.writerows(selected)

        n = len(selected)
        pct = 100.0 * n / max(1, len(rows))

        resumen = {
            "regla": name,
            "n_eliminadas": str(n),
            "pct_eliminadas": f"{pct:.4f}",
            "op_mean_min": str(rule["op_mean_min"]),
            "path_max": str(rule["path_max"]),
            "color_max": str(rule["color_max"]),
            "opstd_max": str(rule["opstd_max"]),
            "scale_max": str(rule["scale_max"]),
            "csv": str(out_csv),
        }
        resumen_rows.append(resumen)

        print(
            f"{name:18s} -> {n:7d} ({pct:6.2f}%) | "
            f"op_mean>={rule['op_mean_min']} "
            f"path<={rule['path_max']} "
            f"color<={rule['color_max']} "
            f"opstd<={rule['opstd_max']} "
            f"scale<={rule['scale_max']}"
        )

    resumen_path = out_dir / "resumen_oat_hard60.csv"
    fields = [
        "regla",
        "n_eliminadas",
        "pct_eliminadas",
        "op_mean_min",
        "path_max",
        "color_max",
        "opstd_max",
        "scale_max",
        "csv",
    ]

    with open(resumen_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(resumen_rows)

    print("")
    print(f"Resumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()
