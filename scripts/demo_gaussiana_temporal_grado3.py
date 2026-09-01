from pathlib import Path
import math
import cv2
import numpy as np


# ============================================================
# Demo visual: una gaussiana 2D temporal con Chebyshev grado 3
# ============================================================
# Genera un MP4 de maximo 20 segundos mostrando:
# 1) gaussiana base
# 2) cambio de mu(t)
# 3) cambio de opacity(t)
# 4) cambio de color(t)
# 5) cambio de scale(t)
# 6) cambio de theta(t)
# 7) cambio de depth(t) con una gaussiana auxiliar
# 8) todos los parametros cambiando a la vez
#
# Nota:
# - Para que depth se vea, se usa una gaussiana auxiliar fija.
#   Con una sola gaussiana, depth no produce cambio visual porque no hay
#   otra primitiva con la cual ordenar la composicion.
# ============================================================


# -----------------------------
# Configuracion general
# -----------------------------
RAIZ = Path(__file__).resolve().parents[1]
OUT_DIR = RAIZ / "outputs" / "demo_gaussiana_temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUTA_VIDEO = OUT_DIR / "demo_gaussiana_grado3_40s_1080p.mp4"

H, W = 1080, 1080
FPS = 30
TOTAL_MAX_SEG = 40
TOTAL_MAX_FRAMES = FPS * TOTAL_MAX_SEG

# Secciones: total = 600 frames = 20 segundos exactos a 30 fps
SECCIONES = [
    ("base",       "Gaussiana base",                            90),   # 3.0 s
    ("mu",         "Cambio de posicion: mu(t)",                 150),  # 5.0 s
    ("opacity",    "Cambio de opacidad: alpha(t)",              180),  # 6.0 s
    ("color",      "Cambio de color: RGB(t)",                   120),  # 4.0 s
    ("scale",      "Cambio de escala: scale_y(t), scale_x(t)",  150),  # 5.0 s
    ("theta",      "Cambio de rotacion: theta(t)",              150),  # 5.0 s
    ("depth",      "Cambio de profundidad: depth(t)",           180),  # 6.0 s
    ("all",        "Todos los parametros cambian",              180),  # 6.0 s
]

assert sum(n for _, _, n in SECCIONES) <= TOTAL_MAX_FRAMES

# Grillas para evaluar la gaussiana
GRID_Y, GRID_X = np.mgrid[0:H, 0:W].astype(np.float32)


# -----------------------------
# Chebyshev grado 3
# -----------------------------
def cheb3(t):
    """
    t en [-1, 1]
    Retorna [T0, T1, T2, T3]
    """
    T0 = 1.0
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    return np.array([T0, T1, T2, T3], dtype=np.float32)


def evaluar_cheb(a0, high, t):
    """
    p(t) = a0 + a1*T1(t) + a2*T2(t) + a3*T3(t)

    a0   : escalar o vector
    high : coeficientes [a1, a2, a3]
           si p es vectorial, shape = (dim, 3)
    """
    B = cheb3(t)[1:]  # T1, T2, T3
    a0 = np.asarray(a0, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    if high.ndim == 1:
        return a0 + float(np.dot(high, B))

    return a0 + high @ B


# -----------------------------
# Render de gaussianas
# -----------------------------
def render_gaussiana(img, mu, scale, theta, opacity, color):
    """
    Renderiza una gaussiana eliptica sobre img usando alpha blending.

    mu      : [y, x]
    scale   : [scale_y, scale_x]
    theta   : rotacion en radianes
    opacity : alpha maximo [0,1]
    color   : RGB en [0,1]
    """
    cy, cx = float(mu[0]), float(mu[1])
    sy, sx = float(scale[0]), float(scale[1])

    sy = max(sy, 1.0)
    sx = max(sx, 1.0)

    dy = GRID_Y - cy
    dx = GRID_X - cx

    c = math.cos(theta)
    s = math.sin(theta)

    # Coordenadas rotadas
    y_rot = c * dy + s * dx
    x_rot = -s * dy + c * dx

    g = np.exp(-0.5 * ((y_rot / sy) ** 2 + (x_rot / sx) ** 2)).astype(np.float32)

    alpha = np.clip(opacity * g, 0.0, 1.0)[..., None]
    color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)

    img[:] = img * (1.0 - alpha) + color * alpha
    return img


def render_escena(gaussianas):
    """
    gaussianas: lista de dicts con:
      mu, scale, theta, opacity, color, depth

    Para la demo:
    - menor depth se dibuja primero
    - mayor depth queda encima
    """
    fondo = np.zeros((H, W, 3), dtype=np.float32)
    fondo[:] = np.array([0.025, 0.025, 0.035], dtype=np.float32)

    gaussianas_ordenadas = sorted(gaussianas, key=lambda g: g["depth"])

    img = fondo
    for g in gaussianas_ordenadas:
        img = render_gaussiana(
            img=img,
            mu=g["mu"],
            scale=g["scale"],
            theta=g["theta"],
            opacity=g["opacity"],
            color=g["color"],
        )

    return np.clip(img, 0.0, 1.0)


# -----------------------------
# Dibujo auxiliar
# -----------------------------
def draw_text(img_u8, text, org, scale=0.9, thickness=2):
    # contorno oscuro para mejorar visibilidad en pantalla grande
    cv2.putText(
        img_u8,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (15, 15, 15),
        thickness + 3,
        cv2.LINE_AA,
    )
    # texto principal claro
    cv2.putText(
        img_u8,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (240, 240, 240),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(img_u8, titulo, formula, info, nota=None):
    overlay = img_u8.copy()

    # Panel superior mas alto para 1080x1080
    cv2.rectangle(overlay, (0, 0), (W, 255), (0, 0, 0), -1)
    img_u8[:] = cv2.addWeighted(overlay, 0.55, img_u8, 0.45, 0)

    # Titulo grande
    draw_text(img_u8, titulo, (34, 54), scale=1.18, thickness=3)

    # Formula
    draw_text(img_u8, formula, (34, 98), scale=0.75, thickness=2)

    # Explicacion de tau
    draw_text(
        img_u8,
        "tau = tiempo normalizado del segmento: inicio=-1 | mitad=0 | final=1",
        (34, 138),
        scale=0.72,
        thickness=2,
    )

    partes = str(info).split(" || ")

    # Linea 1: mu, alpha, scale, theta, depth
    draw_text(img_u8, partes[0], (34, 185), scale=0.73, thickness=2)

    # Linea 2: RGB
    if len(partes) > 1:
        draw_text(img_u8, partes[1], (34, 226), scale=0.73, thickness=2)

    # Nota inferior
    if nota:
        draw_text(img_u8, nota, (34, H - 36), scale=0.78, thickness=2)


def draw_elipse_y_centro(img_u8, g, color=(245, 245, 245)):
    """
    Dibuja el contorno 2-sigma usando la MISMA transformacion del render.
    Esto evita el desfase visual que pasaba con cv2.ellipse al interpretar
    theta/ejes en otra convencion de coordenadas.
    """
    cy, cx = float(g["mu"][0]), float(g["mu"][1])
    sy, sx = float(g["scale"][0]), float(g["scale"][1])
    theta = float(g["theta"])

    phis = np.linspace(0.0, 2.0 * math.pi, 240, endpoint=True)

    # Contorno en coordenadas locales de la gaussiana.
    # Usamos 2 sigma para que el ovalo recubra claramente la zona visible.
    y_local = 2.0 * sy * np.sin(phis)
    x_local = 2.0 * sx * np.cos(phis)

    c = math.cos(theta)
    ss = math.sin(theta)

    # Inversa de:
    # y_rot =  c*dy + s*dx
    # x_rot = -s*dy + c*dx
    dy = c * y_local - ss * x_local
    dx = ss * y_local + c * x_local

    pts = np.stack([cx + dx, cy + dy], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    pts = pts.round().astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        img_u8,
        [pts],
        isClosed=True,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    center = (int(round(cx)), int(round(cy)))
    cv2.circle(img_u8, center, 5, color, -1, lineType=cv2.LINE_AA)


def draw_trayectoria(img_u8, puntos):
    if len(puntos) < 2:
        return

    pts = np.array([(int(x), int(y)) for y, x in puntos], dtype=np.int32)
    cv2.polylines(img_u8, [pts], False, (220, 220, 220), 2, cv2.LINE_AA)

    for p in pts[-8:]:
        cv2.circle(img_u8, tuple(p), 3, (220, 220, 220), -1, cv2.LINE_AA)


# -----------------------------
# Estados de la gaussiana
# -----------------------------
BASE = {
    "mu": np.array([H * 0.52, W * 0.50], dtype=np.float32),
    "opacity": 0.85,
    "color": np.array([0.15, 0.55, 1.00], dtype=np.float32),
    "scale": np.array([90.0, 115.0], dtype=np.float32),
    "theta": math.radians(0.0),
    "depth": 0.0,
}


# Coeficientes aleatorios controlados para la ultima parte
rng = np.random.default_rng(123)

ALL_COEFS = {
    "mu_a0": np.array([H * 0.52, W * 0.50], dtype=np.float32),
    "mu_high": rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
               * np.array([110.0, 80.0, 55.0], dtype=np.float32),

    "opacity_a0": 0.60,
    "opacity_high": rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                    * np.array([0.28, 0.18, 0.10], dtype=np.float32),

    "color_a0": np.array([0.55, 0.55, 0.55], dtype=np.float32),
    "color_high": rng.uniform(-1.0, 1.0, size=(3, 3)).astype(np.float32)
                  * np.array([0.35, 0.25, 0.18], dtype=np.float32),

    "scale_a0": np.array([85.0, 95.0], dtype=np.float32),
    "scale_high": rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
                  * np.array([35.0, 25.0, 18.0], dtype=np.float32),

    "theta_a0": 0.0,
    "theta_high": rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                  * np.array([1.5, 0.9, 0.5], dtype=np.float32),

    "depth_a0": 0.0,
    "depth_high": rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                  * np.array([0.85, 0.35, 0.20], dtype=np.float32),
}


def estado_base():
    return dict(BASE)


def estado_para_seccion(nombre, t):
    """
    t local de la seccion en [-1, 1]
    """
    g = estado_base()
    nota = None
    usar_trayectoria = False

    if nombre == "base":
        pass

    elif nombre == "mu":
        # Movimiento curvo usando Chebyshev grado 3
        g["mu"] = evaluar_cheb(
            a0=np.array([H * 0.52, W * 0.50], dtype=np.float32),
            high=np.array([
                [80.0, -65.0, 55.0],    # y(t)
                [145.0, 40.0, -75.0],   # x(t)
            ], dtype=np.float32),
            t=t,
        )
        g["mu"] = np.clip(g["mu"], [120, 120], [H - 120, W - 120])
        usar_trayectoria = True

    elif nombre == "opacity":
        alpha = evaluar_cheb(
            a0=0.58,
            high=np.array([0.34, -0.23, 0.12], dtype=np.float32),
            t=t,
        )
        g["opacity"] = float(np.clip(alpha, 0.10, 0.95))

    elif nombre == "color":
        color = evaluar_cheb(
            a0=np.array([0.55, 0.55, 0.55], dtype=np.float32),
            high=np.array([
                [0.42, -0.20, 0.10],
                [-0.15, 0.38, -0.25],
                [-0.38, -0.20, 0.32],
            ], dtype=np.float32),
            t=t,
        )
        g["color"] = np.clip(color, 0.05, 1.0)

    elif nombre == "scale":
        scale = evaluar_cheb(
            a0=np.array([82.0, 95.0], dtype=np.float32),
            high=np.array([
                [34.0, -25.0, 14.0],   # scale_y(t)
                [-28.0, 32.0, 20.0],   # scale_x(t)
            ], dtype=np.float32),
            t=t,
        )
        g["scale"] = np.clip(scale, [35.0, 35.0], [140.0, 150.0])

    elif nombre == "theta":
        g["scale"] = np.array([65.0, 145.0], dtype=np.float32)
        theta = evaluar_cheb(
            a0=0.0,
            high=np.array([1.30, 0.55, 0.95], dtype=np.float32),
            t=t,
        )
        g["theta"] = float(theta)

    elif nombre == "depth":
        # Para ver depth se necesita otra gaussiana.
        g["mu"] = np.array([H * 0.53, W * 0.47], dtype=np.float32)
        g["scale"] = np.array([95.0, 125.0], dtype=np.float32)
        g["color"] = np.array([1.0, 0.35, 0.15], dtype=np.float32)
        g["opacity"] = 0.78

        depth = evaluar_cheb(
            a0=0.0,
            high=np.array([1.10, 0.00, 0.00], dtype=np.float32),
            t=t,
        )
        g["depth"] = float(np.clip(depth, -1.0, 1.0))

        referencia = {
            "mu": np.array([H * 0.50, W * 0.53], dtype=np.float32),
            "scale": np.array([105.0, 120.0], dtype=np.float32),
            "theta": math.radians(-20),
            "opacity": 0.70,
            "color": np.array([0.25, 0.65, 1.00], dtype=np.float32),
            "depth": 0.0,
        }

        nota = "Nota: depth solo se aprecia comparando contra una segunda gaussiana auxiliar."
        return [referencia, g], nota, usar_trayectoria

    elif nombre == "all":
        g["mu"] = evaluar_cheb(ALL_COEFS["mu_a0"], ALL_COEFS["mu_high"], t)
        g["mu"] = np.clip(g["mu"], [120, 120], [H - 120, W - 120])

        g["opacity"] = float(np.clip(
            evaluar_cheb(ALL_COEFS["opacity_a0"], ALL_COEFS["opacity_high"], t),
            0.20,
            0.95,
        ))

        g["color"] = np.clip(
            evaluar_cheb(ALL_COEFS["color_a0"], ALL_COEFS["color_high"], t),
            0.08,
            1.0,
        )

        g["scale"] = np.clip(
            evaluar_cheb(ALL_COEFS["scale_a0"], ALL_COEFS["scale_high"], t),
            [35.0, 35.0],
            [145.0, 155.0],
        )

        g["theta"] = float(evaluar_cheb(
            ALL_COEFS["theta_a0"],
            ALL_COEFS["theta_high"],
            t,
        ))

        g["depth"] = float(np.clip(
            evaluar_cheb(ALL_COEFS["depth_a0"], ALL_COEFS["depth_high"], t),
            -1.0,
            1.0,
        ))

        referencia = {
            "mu": np.array([H * 0.50, W * 0.50], dtype=np.float32),
            "scale": np.array([115.0, 115.0], dtype=np.float32),
            "theta": 0.0,
            "opacity": 0.28,
            "color": np.array([0.85, 0.85, 0.85], dtype=np.float32),
            "depth": 0.0,
        }

        nota = "Parametros aleatorios controlados: mu, alpha, RGB, scale, theta y depth."
        usar_trayectoria = True
        return [referencia, g], nota, usar_trayectoria

    return [g], nota, usar_trayectoria


def info_gaussiana(g):
    y, x = g["mu"]
    sy, sx = g["scale"]
    theta_deg = math.degrees(g["theta"])

    r, gr, b = g["color"]
    r255 = int(round(float(r) * 255))
    g255 = int(round(float(gr) * 255))
    b255 = int(round(float(b) * 255))

    linea1 = (
        f"mu=({y:6.1f},{x:6.1f})  "
        f"alpha={g['opacity']:.2f}  "
        f"scale=({sy:5.1f},{sx:5.1f})  "
        f"theta={theta_deg:6.1f} deg  "
        f"depth={g['depth']:+.2f}"
    )

    linea2 = (
        f"RGB=[{r:.2f}, {gr:.2f}, {b:.2f}] en [0,1]   "
        f"RGB255=({r255}, {g255}, {b255})"
    )

    return linea1 + " || " + linea2


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(RUTA_VIDEO), fourcc, FPS, (W, H))

    if not writer.isOpened():
        raise RuntimeError(
            "No se pudo abrir el VideoWriter. Prueba cambiando el codec mp4v o revisa OpenCV."
        )

    formula = "Chebyshev grado 3: p(tau)=a0+a1*T1+a2*T2+a3*T3"

    total = sum(n for _, _, n in SECCIONES)
    frame_global = 0

    print(f"Generando video: {RUTA_VIDEO}")
    print(f"Duracion: {total / FPS:.2f} s | FPS: {FPS} | Frames: {total}")

    for nombre, titulo, n_frames in SECCIONES:
        print(f"  - {titulo} ({n_frames / FPS:.1f} s)")

        trayectoria = []

        for k in range(n_frames):
            if n_frames <= 1:
                t = 0.0
            else:
                t = -1.0 + 2.0 * (k / (n_frames - 1))

            gaussianas, nota, usar_trayectoria = estado_para_seccion(nombre, t)

            # La protagonista es la ultima gaussiana de la lista.
            protagonista = gaussianas[-1]

            img = render_escena(gaussianas)
            img_u8 = (img * 255).astype(np.uint8)

            if usar_trayectoria:
                trayectoria.append(protagonista["mu"].copy())
                draw_trayectoria(img_u8, trayectoria)

            # Dibujar contorno de la protagonista
            draw_elipse_y_centro(img_u8, protagonista)

            # Si hay referencia, dibujar su contorno mas tenue
            if len(gaussianas) > 1:
                draw_elipse_y_centro(img_u8, gaussianas[0], color=(150, 150, 150))

            draw_panel(
                img_u8,
                titulo=titulo,
                formula=formula,
                info=info_gaussiana(protagonista),
                nota=nota,
            )

            # OpenCV escribe BGR
            frame_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

            frame_global += 1

    writer.release()

    print("")
    print("Listo.")
    print(f"Video guardado en: {RUTA_VIDEO}")


if __name__ == "__main__":
    main()
