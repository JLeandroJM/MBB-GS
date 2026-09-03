"""
Holdout temporal: decide que frames se SUPERVISAN durante el entrenamiento.

Idea del experimento (propuesto por el profesor): como cada parametro de cada
gaussiana es un polinomio en tiempo continuo, el modelo se puede evaluar en
cualquier frame. Si entrenamos supervisando solo un SUBCONJUNTO de frames y el
modelo reconstruye bien los que NO vio (holdout), demostramos que la
representacion polinomica INTERPOLA el movimiento (no memoriza frames).

IMPORTANTE: el eje de tiempo siempre es el total (n_frames). Aqui solo elegimos
QUE frames entran al loss; los demas conservan su slot temporal y se reconstruyen
por evaluacion del polinomio.

Modos (config["holdout"] = dict con "modo" + parametros):

  {"modo": "multiplos", "k": 2}
      Supervisa los frames cuyo indice es multiplo de k.
      k=2 -> supervisa pares (0,2,4,...), holdout = impares.
      k=3 -> supervisa 1 de cada 3, holdout = 2/3 de los frames.

  {"modo": "aleatorio_por_bloque", "bloque": 3, "por_bloque": 1}
      Divide en bloques consecutivos de 'bloque' frames y de cada bloque toma
      'por_bloque' al azar (semilla fija). Muestreo irregular, mas realista.

  {"modo": "gap", "inicio": 200, "longitud": 5}
      Supervisa TODO menos un bloque contiguo [inicio, inicio+longitud).
      Caso "pelota / dron": reconstruir un hueco de frames perdidos.

  {"modo": "gaps_aleatorios", "n": 5}
      Quita n frames sueltos al azar (simula frames perdidos dispersos).
"""
import torch


def generar_indices_supervisados(n_frames, holdout_cfg, seed=42):
    """
    Devuelve (supervisados, holdout): dos listas ordenadas de indices de frame,
    disjuntas, cuya union es range(n_frames).

    supervisados: entran al loss.
    holdout     : NO se supervisan; se reconstruyen por el polinomio.
    """
    n = int(n_frames)
    todos = set(range(n))
    modo = str(holdout_cfg.get("modo", "")).lower().strip()

    if modo == "multiplos":
        k = int(holdout_cfg["k"])
        if k < 2:
            raise ValueError("holdout multiplos: k debe ser >= 2")
        sup = set(j for j in range(n) if j % k == 0)

    elif modo == "aleatorio_por_bloque":
        bloque = int(holdout_cfg.get("bloque", 3))
        por_bloque = int(holdout_cfg.get("por_bloque", 1))
        if bloque < 1 or por_bloque < 1 or por_bloque > bloque:
            raise ValueError("holdout aleatorio_por_bloque: 1 <= por_bloque <= bloque")
        g = torch.Generator().manual_seed(int(seed))
        sup = set()
        for inicio in range(0, n, bloque):
            fin = min(inicio + bloque, n)
            idx_bloque = list(range(inicio, fin))
            tomar = min(por_bloque, len(idx_bloque))
            perm = torch.randperm(len(idx_bloque), generator=g).tolist()
            sup.update(idx_bloque[p] for p in perm[:tomar])

    elif modo == "gap":
        # Un gap: "inicio"+"longitud".  Varios gaps: "huecos": [[inicio,long],...]
        huecos_cfg = holdout_cfg.get("huecos")
        hueco = set()
        if huecos_cfg:
            for ini, lon in huecos_cfg:
                hueco |= set(range(int(ini), min(int(ini) + int(lon), n)))
        else:
            inicio = int(holdout_cfg["inicio"])
            longitud = int(holdout_cfg["longitud"])
            hueco = set(range(inicio, min(inicio + longitud, n)))
        sup = todos - hueco

    elif modo == "gaps_aleatorios":
        cuantos = int(holdout_cfg["n"])
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=g).tolist()
        hueco = set(perm[:cuantos])
        sup = todos - hueco

    else:
        raise ValueError(f"holdout modo desconocido: {modo!r}")

    # Anclar bordes: forzar que los primeros K y ultimos K frames SIEMPRE se
    # supervisen (nunca sean holdout). Fija los extremos del intervalo temporal
    # t in [-1, 1], donde el polinomio de grado alto oscila (efecto Runge) si no
    # tiene datos. Sin esto, los frames holdout de los extremos se reconstruyen
    # pesimo (p.ej. frame 1 y frame N-1).
    anclar = int(holdout_cfg.get("anclar_bordes", 0))
    if anclar > 0:
        bordes = set(range(0, min(anclar, n))) | set(range(max(0, n - anclar), n))
        sup = sup | bordes

    supervisados = sorted(sup)
    holdout = sorted(todos - sup)

    if not supervisados:
        raise ValueError("holdout dejo 0 frames supervisados; revisa la config")

    return supervisados, holdout


def describir_holdout(n_frames, holdout_cfg, seed=42):
    """Texto resumen para logs."""
    sup, hold = generar_indices_supervisados(n_frames, holdout_cfg, seed)
    return (
        f"holdout modo={holdout_cfg.get('modo')} | "
        f"supervisados={len(sup)}/{n_frames} | holdout={len(hold)}"
    )
