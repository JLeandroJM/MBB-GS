"""
Bases polinomicas para los coeficientes temporales.

Soportamos dos bases, ambas indexadas en el frame j de un clip de N_frames:

Base monomial:
--------------
    t_norm(j) = j / (N_frames - 1)         en [0, 1]
    fila B[j] = [t_norm^0, t_norm^1, ..., t_norm^q]

Es la base "natural": p(t) = sum a_k t^k. Es facil de implementar pero
tiene un problema numerico clasico: las columnas son MUY similares entre si
(t^10 y t^11 cuando t ~ 0.5 son casi iguales), lo que produce una matriz de
diseno con condicion alta. Para grados >~ 15 el ajuste por minimos cuadrados
empieza a oscilar (fenomeno de Runge en los bordes).

Base Chebyshev (primer tipo):
-----------------------------
    t_cheb(j) = 2 * j / (N_frames - 1) - 1     en [-1, 1]
    recurrencia:
        T_0(t) = 1
        T_1(t) = t
        T_k(t) = 2 t T_{k-1}(t) - T_{k-2}(t)
    fila B[j] = [T_0(t_cheb), T_1(t_cheb), ..., T_q(t_cheb)]

Propiedades clave:
  - |T_k(t)| <= 1 en [-1, 1]: base acotada.
  - Es una base ORTOGONAL con respecto al peso 1/sqrt(1-t^2). En la practica
    los coefs son aproximadamente independientes y el problema de optimizacion
    esta mucho mejor condicionado para grados altos.
  - Error de interpolacion distribuido uniformemente -- no explota en los bordes.

Ambas funciones devuelven la matriz B de shape (N_frames, q+1). El modelo
hace simplemente:
    params(t_j) = coefs @ B[j, :q+1]
"""
import torch



# DECISION: construimos primero en CPU con float64 para precision de la
# recurrencia (sobre todo en Chebyshev con grado >= 30 hay cancelacion
# importante si se usa float32 nativo), y despues movemos al device en el
# dtype objetivo. MPS no soporta float64 nativo. CUDA si pero el costo de
# crear esta matriz una sola vez es despreciable, asi que no varia.
def _arange_normalizado_01(n_frames):
    """j / (N-1) en [0, 1], como float64."""
    if n_frames < 2:
        return torch.zeros(n_frames, dtype=torch.float64)
    idx = torch.arange(n_frames, dtype=torch.float64)
    return idx / (n_frames - 1)



def construir_matriz_monomial(n_frames, grado_max, device='cpu', dtype=torch.float32):
    """
    Matriz B (n_frames, grado_max+1) en base monomial.
    B[j, k] = (j / (N - 1)) ** k.
    """
    t = _arange_normalizado_01(n_frames)                            # [0, 1]

    B = torch.empty(n_frames, grado_max + 1, dtype=torch.float64)
    B[:, 0] = 1.0
    if grado_max >= 1:
        B[:, 1] = t
    for k in range(2, grado_max + 1):
        # forma directa: t**k. Lo hacemos con la recurrencia multiplicativa
        # para minimizar acumulacion de errores cuando grado es alto.
        B[:, k] = B[:, k - 1] * t

    return B.to(dtype=dtype, device=device)



def construir_matriz_chebyshev(n_frames, grado_max, device='cpu', dtype=torch.float32):
    """
    Matriz B (n_frames, grado_max+1) en base de Chebyshev de primer tipo.
    B[j, k] = T_k(t_cheb_j), con t_cheb_j = 2 * j / (N - 1) - 1.
    """
    t01 = _arange_normalizado_01(n_frames)
    t = 2.0 * t01 - 1.0                                              # [-1, 1]

    B = torch.empty(n_frames, grado_max + 1, dtype=torch.float64)
    B[:, 0] = 1.0
    if grado_max >= 1:
        B[:, 1] = t
    for k in range(2, grado_max + 1):
        B[:, k] = 2.0 * t * B[:, k - 1] - B[:, k - 2]

    return B.to(dtype=dtype, device=device)



# DECISION: dispatcher por nombre. Asi el modelo y el trainer reciben el
# parametro "base" del config y no necesitan if/else internos.
def construir_matriz(base, n_frames, grado_max, device='cpu', dtype=torch.float32):
    if base == 'chebyshev':
        return construir_matriz_chebyshev(n_frames, grado_max, device, dtype)
    if base == 'monomial':
        return construir_matriz_monomial(n_frames, grado_max, device, dtype)
    raise ValueError(f"base desconocida: {base!r}  (esperaba 'chebyshev' o 'monomial')")



# ===========================================================================
# tests
# ===========================================================================

def _tests():
    """Verifica identidades basicas de cada base."""
    # --- Chebyshev ---
    B = construir_matriz_chebyshev(5, 4)
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert torch.allclose(B[:, 0], torch.ones(5))                       # T_0 = 1
    assert torch.allclose(B[:, 1], t)                                    # T_1 = t
    assert torch.allclose(B[:, 2], 2 * t ** 2 - 1, atol=1e-6)            # T_2
    assert torch.allclose(B[:, 3], 4 * t ** 3 - 3 * t, atol=1e-6)        # T_3
    assert torch.allclose(B[:, 4], 8 * t ** 4 - 8 * t ** 2 + 1, atol=1e-6)
    assert (B.abs() <= 1.0 + 1e-6).all(), "Chebyshev fuera de [-1, 1]"

    # --- monomial ---
    B = construir_matriz_monomial(5, 4)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.allclose(B[:, 0], torch.ones(5))
    assert torch.allclose(B[:, 1], t)
    assert torch.allclose(B[:, 2], t ** 2, atol=1e-6)
    assert torch.allclose(B[:, 3], t ** 3, atol=1e-6)
    assert torch.allclose(B[:, 4], t ** 4, atol=1e-6)

    # --- dispatcher ---
    B1 = construir_matriz('chebyshev', 10, 5)
    B2 = construir_matriz_chebyshev(10, 5)
    assert torch.allclose(B1, B2)

    # --- estabilidad numerica: con grado 40 los valores de Cheb deben quedarse <= 1
    Bbig = construir_matriz_chebyshev(90, 40)
    assert (Bbig.abs() <= 1.0 + 1e-5).all(), "Cheb grado 40: overflow inesperado"

    # ... mientras que la monomial llega a t=1 -> 1, a t cerca de 0 -> 0
    Bmono = construir_matriz_monomial(90, 40)
    # las potencias altas con t<1 se acercan a 0 -- esperado
    assert Bmono[0, 40].item() == 0.0     # t=0
    assert Bmono[-1, 40].item() == 1.0    # t=1

    print("[bases] tests OK")


if __name__ == "__main__":
    _tests()
