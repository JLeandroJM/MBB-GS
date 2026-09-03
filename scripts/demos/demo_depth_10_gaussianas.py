from pathlib import Path
import math
import cv2
import numpy as np


# ============================================================
# Demo depth con 10 gaussianas
# Genera 2 versiones:
#   A) rank_panel   -> mas visual / presentacion
#   B) depth_rows   -> mas analitica / depth mas legible
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]
OUT_DIR = RAIZ / "outputs" / "demo_depth_multi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUTA_A = OUT_DIR / "depth_demo_10_gaussianas_final_1080p.mp4"
RUTA_B = OUT_DIR / "depth_demo_v2_depth_rows_1080p.mp4"

H = 1080
W = 1080
FPS = 30
DURACION_SEG = 20
N_FRAMES = FPS * DURACION_SEG
N_GAUSS = 10

SCENE_CENTER = np.array([470.0, 455.0], dtype=np.float32)  # y, x


# ============================================================
# Utilidades
# ============================================================

def cheb3(t):
    T0 = 1.0
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    return np.array([T0, T1, T2, T3], dtype=np.float32)


def eval_cheb(a0, high, t):
    """
    p(t) = a0 + a1*T1 + a2*T2 + a3*T3
    """
    B = cheb3(t)[1:]  # T1, T2, T3
    a0 = np.asarray(a0, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    return a0 + high @ B


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def rgb01_to_bgr255(c):
    c = np.asarray(c, dtype=np.float32)
    return tuple(int(round(v * 255.0)) for v in c[::-1])


def draw_text(img, text, org, scale=0.75, thickness=2, color=(242, 242, 242)):
    # sombra / contorno
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (18, 18, 18), thickness + 3, cv2.LINE_AA
    )
    # texto principal
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA
    )


def build_background(kind="A"):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = xx / (W - 1)
    yn = yy / (H - 1)

    bg = np.zeros((H, W, 3), dtype=np.float32)

    # base oscura azulada
    bg[..., 0] = 0.035 + 0.015 * (1.0 - yn)   # R
    bg[..., 1] = 0.050 + 0.020 * (1.0 - yn)   # G
    bg[..., 2] = 0.085 + 0.040 * (1.0 - yn)   # B

    # glow central
    cx = 0.40 if kind == "A" else 0.46
    cy = 0.42
    r2 = ((xn - cx) / 0.34) ** 2 + ((yn - cy) / 0.30) ** 2
    glow = np.exp(-1.8 * r2)
    bg += glow[..., None] * np.array([0.030, 0.040, 0.060], dtype=np.float32)

    # vignette
    dist = np.sqrt((xn - 0.5) ** 2 + (yn - 0.5) ** 2)
    vig = np.clip(1.05 - 0.90 * dist, 0.65, 1.0)
    bg *= vig[..., None]

    return clip01(bg)


BG_A = build_background("A")
BG_B = build_background("B")


# ============================================================
# Banco de gaussianas
# ============================================================

def build_bank(seed=22):
    rng = np.random.default_rng(seed)

    # paleta fija para mantener identidad visual
    palette = np.array([
        [0.95, 0.38, 0.32],  # coral
        [0.22, 0.78, 0.95],  # cyan
        [0.34, 0.95, 0.58],  # green
        [0.98, 0.80, 0.28],  # yellow
        [0.80, 0.48, 0.98],  # purple
        [0.98, 0.52, 0.72],  # pink
        [0.55, 0.72, 1.00],  # blue
        [1.00, 0.62, 0.28],  # orange
        [0.62, 1.00, 0.46],  # lime
        [0.74, 0.74, 0.82],  # silver
    ], dtype=np.float32)

    bank = []
    base_depths = np.linspace(-0.22, 0.22, N_GAUSS)

    for i in range(N_GAUSS):
        ang = 2.0 * math.pi * i / N_GAUSS
        rad = 80.0 + 28.0 * math.sin(1.6 * i)

        mu_a0 = SCENE_CENTER + np.array([
            rad * math.sin(ang),
            rad * math.cos(ang)
        ], dtype=np.float32) + rng.normal(0.0, 12.0, size=2).astype(np.float32)

        mu_high = rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32) * np.array(
            [[26.0, 15.0, 8.0],
             [28.0, 16.0, 10.0]],
            dtype=np.float32
        )

        opacity_a0 = np.float32(0.43 + 0.08 * math.sin(0.9 * i))
        opacity_high = rng.uniform(-1.0, 1.0, size=3).astype(np.float32) * np.array(
            [0.10, 0.06, 0.035], dtype=np.float32
        )

        scale_a0 = np.array([
            72.0 + rng.uniform(-12.0, 12.0),
            92.0 + rng.uniform(-15.0, 15.0),
        ], dtype=np.float32)
        scale_high = rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32) * np.array(
            [[10.0, 6.0, 4.0],
             [11.0, 7.0, 4.0]],
            dtype=np.float32
        )

        theta_a0 = np.float32(rng.uniform(-1.2, 1.2))
        theta_high = rng.uniform(-1.0, 1.0, size=3).astype(np.float32) * np.array(
            [0.35, 0.20, 0.12], dtype=np.float32
        )

        # depth con cruces suaves
        depth_a0 = np.float32(base_depths[i])
        depth_high = rng.uniform(-1.0, 1.0, size=3).astype(np.float32) * np.array(
            [0.55, 0.24, 0.10], dtype=np.float32
        )

        bank.append({
            "id": i,
            "name": f"G{i+1}",
            "color": palette[i],
            "mu_a0": mu_a0,
            "mu_high": mu_high,
            "opacity_a0": opacity_a0,
            "opacity_high": opacity_high,
            "scale_a0": scale_a0,
            "scale_high": scale_high,
            "theta_a0": theta_a0,
            "theta_high": theta_high,
            "depth_a0": depth_a0,
            "depth_high": depth_high,
        })

    return bank


BANK = build_bank()


def state_at(gdef, tau):
    mu = eval_cheb(gdef["mu_a0"], gdef["mu_high"], tau)
    mu[0] = np.clip(mu[0], 150.0, 760.0)
    mu[1] = np.clip(mu[1], 150.0, 760.0)

    opacity = float(eval_cheb(gdef["opacity_a0"], gdef["opacity_high"], tau))
    opacity = float(np.clip(opacity, 0.25, 0.68))

    scale = eval_cheb(gdef["scale_a0"], gdef["scale_high"], tau)
    scale = np.clip(scale, [48.0, 56.0], [100.0, 126.0])

    theta = float(eval_cheb(gdef["theta_a0"], gdef["theta_high"], tau))

    depth = float(eval_cheb(gdef["depth_a0"], gdef["depth_high"], tau))
    depth = float(np.clip(depth, -1.0, 1.0))

    return {
        "id": gdef["id"],
        "name": gdef["name"],
        "color": gdef["color"],
        "mu": mu.astype(np.float32),
        "opacity": opacity,
        "scale": scale.astype(np.float32),
        "theta": theta,
        "depth": depth,
    }


# ============================================================
# Render gaussiano
# ============================================================

def render_gaussian_patch(img, g, k_sigma=3.2):
    cy, cx = float(g["mu"][0]), float(g["mu"][1])
    sy, sx = float(g["scale"][0]), float(g["scale"][1])
    theta = float(g["theta"])
    opacity = float(g["opacity"])
    color = np.asarray(g["color"], dtype=np.float32)

    c = math.cos(theta)
    s = math.sin(theta)

    extent_y = k_sigma * math.sqrt((c * sy) ** 2 + (s * sx) ** 2)
    extent_x = k_sigma * math.sqrt((s * sy) ** 2 + (c * sx) ** 2)

    y0 = max(0, int(math.floor(cy - extent_y)))
    y1 = min(H - 1, int(math.ceil(cy + extent_y)))
    x0 = max(0, int(math.floor(cx - extent_x)))
    x1 = min(W - 1, int(math.ceil(cx + extent_x)))

    if y1 <= y0 or x1 <= x0:
        return

    yy, xx = np.mgrid[y0:y1+1, x0:x1+1].astype(np.float32)
    dy = yy - cy
    dx = xx - cx

    y_rot = c * dy + s * dx
    x_rot = -s * dy + c * dx

    gauss = np.exp(-0.5 * ((y_rot / sy) ** 2 + (x_rot / sx) ** 2)).astype(np.float32)
    alpha = np.clip(opacity * gauss, 0.0, 1.0)[..., None]

    patch = img[y0:y1+1, x0:x1+1, :]
    img[y0:y1+1, x0:x1+1, :] = patch * (1.0 - alpha) + color[None, None, :] * alpha


def ellipse_points(g, radius=2.0, n=180):
    cy, cx = float(g["mu"][0]), float(g["mu"][1])
    sy, sx = float(g["scale"][0]), float(g["scale"][1])
    theta = float(g["theta"])

    phis = np.linspace(0.0, 2.0 * math.pi, n, endpoint=True)
    y_local = radius * sy * np.sin(phis)
    x_local = radius * sx * np.cos(phis)

    c = math.cos(theta)
    s = math.sin(theta)

    dy = c * y_local - s * x_local
    dx = s * y_local + c * x_local

    pts = np.stack([cx + dx, cy + dy], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts.round().astype(np.int32).reshape(-1, 1, 2)


def draw_soft_contour(img_u8, g):
    """
    Contorno sutil cuya intensidad depende de opacity.
    No muy visible, solo para ayudar a seguir la superposicion.
    """
    pts = ellipse_points(g, radius=2.0, n=160)
    base = np.array(rgb01_to_bgr255(g["color"]), dtype=np.float32)

    # suavemente mas brillante si opacity es mayor
    boost = 55.0 + 85.0 * float(g["opacity"])
    line = np.clip(base * 0.42 + boost, 0.0, 255.0).astype(np.uint8)
    thickness = 1 if g["opacity"] < 0.45 else 2

    cv2.polylines(img_u8, [pts], True, tuple(int(x) for x in line), thickness, cv2.LINE_AA)

    cx = int(round(float(g["mu"][1])))
    cy = int(round(float(g["mu"][0])))
    cv2.circle(img_u8, (cx, cy), 3, tuple(int(x) for x in line), -1, cv2.LINE_AA)


# ============================================================
# Paneles visuales
# ============================================================

def draw_top_left_title(img, title, subtitle):
    overlay = img.copy()
    cv2.rectangle(overlay, (20, 20), (690, 120), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.42, img, 0.58, 0)

    draw_text(img, title, (40, 60), scale=1.05, thickness=3)
    draw_text(img, subtitle, (40, 100), scale=0.58, thickness=2, color=(220, 220, 220))


def draw_panel_A(img, states, tau):
    # panel derecho
    x0, y0, x1, y1 = 760, 26, 1054, 1054
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.47, img, 0.53, 0)

    draw_text(img, "Orden de profundidad", (782, 62), scale=0.82, thickness=2)
    draw_text(img, "Mayor depth = se dibuja mas adelante", (782, 96), scale=0.50, thickness=1, color=(220,220,220))
    draw_text(img, f"tiempo normalizado = {tau:+.2f}", (782, 126), scale=0.48, thickness=1, color=(220,220,220))

    front_sorted = sorted(states, key=lambda z: z["depth"], reverse=True)
    top_names = " > ".join(g["name"] for g in front_sorted[:4])
    draw_text(img, f"Orden actual: {top_names}", (782, 158), scale=0.50, thickness=1, color=(230,230,230))

    base_y = 200
    row_h = 80

    for rank, g in enumerate(front_sorted):
        y = base_y + rank * row_h

        # resaltar el primero
        if rank == 0:
            ov = img.copy()
            cv2.rectangle(ov, (775, y - 22), (1040, y + 34), (40, 40, 40), -1)
            img[:] = cv2.addWeighted(ov, 0.35, img, 0.65, 0)

        color_box = rgb01_to_bgr255(g["color"])
        cv2.rectangle(img, (784, y - 10), (812, y + 18), color_box, -1)
        cv2.rectangle(img, (784, y - 10), (812, y + 18), (230, 230, 230), 1)

        draw_text(img, f"#{rank+1}  {g['name']}", (824, y + 5), scale=0.60, thickness=2)
        draw_text(
            img,
            f"depth={g['depth']:+.2f}   alpha={g['opacity']:.2f}",
            (824, y + 30),
            scale=0.47,
            thickness=1,
            color=(225, 225, 225)
        )

    draw_text(
        img,
        "Contorno suave segun opacidad",
        (782, 1018),
        scale=0.50,
        thickness=1,
        color=(220, 220, 220)
    )


def draw_panel_B(img, states, tau):
    # titulo superior
    overlay = img.copy()
    cv2.rectangle(overlay, (20, 20), (1060, 122), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)

    draw_text(img, "Version B  |  Fila por gaussiana", (38, 62), scale=1.00, thickness=3)
    draw_text(
        img,
        "Cada fila muestra el depth actual de una gaussiana, sin superponer textos.",
        (38, 102),
        scale=0.58,
        thickness=2,
        color=(225, 225, 225)
    )

    # panel inferior
    py0, py1 = 772, 1055
    overlay = img.copy()
    cv2.rectangle(overlay, (20, py0), (1060, py1), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.50, img, 0.50, 0)

    front_sorted = sorted(states, key=lambda z: z["depth"], reverse=True)
    draw_text(img, f"tau = {tau:+.2f}", (36, 806), scale=0.58, thickness=1, color=(230,230,230))
    draw_text(
        img,
        "Frente actual: " + " > ".join(g["name"] for g in front_sorted[:5]),
        (180, 806),
        scale=0.52,
        thickness=1,
        color=(230,230,230)
    )

    # eje depth
    x_axis0 = 315
    x_axis1 = 1000
    cv2.line(img, (x_axis0, 835), (x_axis1, 835), (200, 200, 200), 1, cv2.LINE_AA)
    draw_text(img, "fondo  -1", (x_axis0 - 34, 827), scale=0.42, thickness=1, color=(220,220,220))
    draw_text(img, "frente  +1", (x_axis1 - 82, 827), scale=0.42, thickness=1, color=(220,220,220))

    rows = sorted(states, key=lambda z: z["id"])
    y_start = 865
    row_gap = 18

    for i, g in enumerate(rows):
        y = y_start + i * row_gap

        cv2.line(img, (x_axis0, y), (x_axis1, y), (80, 80, 80), 1, cv2.LINE_AA)

        color_box = rgb01_to_bgr255(g["color"])
        cv2.rectangle(img, (38, y - 10), (60, y + 8), color_box, -1)
        cv2.rectangle(img, (38, y - 10), (60, y + 8), (230, 230, 230), 1)

        draw_text(
            img,
            f"{g['name']}   d={g['depth']:+.2f}   a={g['opacity']:.2f}",
            (72, y + 4),
            scale=0.43,
            thickness=1,
            color=(235,235,235)
        )

        # dot en el eje
        x = int(round(x_axis0 + ((g["depth"] + 1.0) / 2.0) * (x_axis1 - x_axis0)))
        radius = 6 if g["opacity"] < 0.45 else 7
        cv2.circle(img, (x, y), radius, color_box, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius + 1, (235, 235, 235), 1, cv2.LINE_AA)

    draw_text(
        img,
        "Contornos sutiles en escena: intensidad guiada por opacity",
        (36, 1030),
        scale=0.50,
        thickness=1,
        color=(225, 225, 225)
    )


# ============================================================
# Generacion
# ============================================================

def render_scene(states, bg):
    img = bg.copy()

    # back-to-front
    back_to_front = sorted(states, key=lambda z: z["depth"])
    for g in back_to_front:
        render_gaussian_patch(img, g)

    img_u8 = (clip01(img) * 255.0).astype(np.uint8)

    # contornos sutiles
    for g in back_to_front:
        draw_soft_contour(img_u8, g)

    return img_u8


def generar_video(version, ruta_salida):
    if version == "A":
        bg = BG_A
    else:
        bg = BG_B

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(ruta_salida), fourcc, FPS, (W, H))

    if not writer.isOpened():
        raise RuntimeError(f"No se pudo abrir VideoWriter para {ruta_salida}")

    print(f"Generando {version}: {ruta_salida}")

    for j in range(N_FRAMES):
        tau = -1.0 + 2.0 * (j / (N_FRAMES - 1))
        states = [state_at(gd, tau) for gd in BANK]

        frame = render_scene(states, bg)

        if version == "A":
            draw_top_left_title(
                frame,
                "Gaussianas 2D temporales",
                "Superposicion, alpha blending y cambios suaves de profundidad."
            )
            draw_panel_A(frame, states, tau)
        else:
            draw_panel_B(frame, states, tau)

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (j + 1) % 60 == 0:
            print(f"  {version}: frame {j+1}/{N_FRAMES}")

    writer.release()
    print(f"Listo: {ruta_salida}")


def main():
    generar_video("A", RUTA_A)

    print("")
    print("Video final generado:")
    print(f"  - {RUTA_A}")


if __name__ == "__main__":
    main()
