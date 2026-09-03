import argparse
import csv
from pathlib import Path


RULES = {
    "hard_60_base": {
        "op_mean_min": 0.05,
        "path_max": 60.0,
        "color_max": 0.60,
        "opstd_max": 0.125,
        "scale_max": 0.60,
    },
    "hard_60_op015": {
        "op_mean_min": 0.05,
        "path_max": 60.0,
        "color_max": 0.60,
        "opstd_max": 0.150,
        "scale_max": 0.60,
    },
    "hard_60_op020": {
        "op_mean_min": 0.05,
        "path_max": 60.0,
        "color_max": 0.60,
        "opstd_max": 0.200,
        "scale_max": 0.60,
    },
    "hard_80": {
        "op_mean_min": 0.05,
        "path_max": 80.0,
        "color_max": 0.80,
        "opstd_max": 0.150,
        "scale_max": 0.80,
    },
    "hard_100": {
        "op_mean_min": 0.05,
        "path_max": 100.0,
        "color_max": 1.00,
        "opstd_max": 0.180,
        "scale_max": 1.00,
    },
    "hard_120": {
        "op_mean_min": 0.05,
        "path_max": 120.0,
        "color_max": 1.20,
        "opstd_max": 0.220,
        "scale_max": 1.20,
    },
    "low_opacity_010_040": {
        "op_mean_max": 0.10,
        "op_max_max": 0.40,
    },
    "low_opacity_015_050": {
        "op_mean_max": 0.15,
        "op_max_max": 0.50,
    },
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
    if "op_mean_min" in rule and f(row, "op_mean") < rule["op_mean_min"]:
        return False
    if "op_mean_max" in rule and f(row, "op_mean") > rule["op_mean_max"]:
        return False
    if "op_max_max" in rule and f(row, "op_max") > rule["op_max_max"]:
        return False
    if "path_max" in rule and f(row, "path_length_px") > rule["path_max"]:
        return False
    if "color_max" in rule and f(row, "color_path") > rule["color_max"]:
        return False
    if "opstd_max" in rule and f(row, "op_std") > rule["opstd_max"]:
        return False
    if "scale_max" in rule and f(row, "scale_std") > rule["scale_max"]:
        return False
    return True


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

    print("=== Generando variantes hard ===")
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

        resumen_row = {
            "regla": name,
            "n_eliminadas": str(n),
            "pct_eliminadas": f"{pct:.4f}",
            "csv": str(out_csv),
        }
        for k, v in rule.items():
            resumen_row[k] = str(v)

        resumen_rows.append(resumen_row)

        print(f"{name:22s} -> {n:7d} gaussianas ({pct:6.2f}%) | {out_csv}")

    resumen_fields = []
    for r in resumen_rows:
        for k in r.keys():
            if k not in resumen_fields:
                resumen_fields.append(k)

    resumen_path = out_dir / "resumen_variantes_hard.csv"
    with open(resumen_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=resumen_fields)
        writer.writeheader()
        writer.writerows(resumen_rows)

    print("")
    print(f"Resumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()
