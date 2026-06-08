"""
Cuantizacion post-entrenamiento de los coeficientes Gabor, para medir el
trade-off compresion (bytes) vs calidad (SNR) de forma honesta.

DETALLE TEORICO IMPORTANTE sobre mu_t:
    float16 tiene ~11 bits de mantisa -> solo representa enteros EXACTOS hasta
    2048; mas alla aparecen huecos. Para 30 s a 44.1 kHz, mu_t llega a ~1.3e6
    samples, donde el paso de fp16 es de ~128 samples (¡varios ms de error de
    posicion!). Por eso mu_t NO se debe guardar en fp16. La posicion temporal es
    intrinsecamente un entero de sample: se redondea y se guarda en int32 (4 B).
    Los otros 4 parametros (log_sigma, amp, freq_raw, phi) tienen rango pequeno
    y sí toleran fp16 (2 B) sin degradacion audible.

Esquemas:
    fp32     : todo float32            -> 20 B/atomo (baseline sin perdida)
    fp16mix  : mu_t int32 + resto fp16 -> 12 B/atomo (recomendado)
    fp16full : todo fp16 (mu incluido) -> 10 B/atomo (muestra el dano en mu_t)
"""
import torch

_PARAMS = ["mu_t", "log_sigma", "amp", "freq_raw", "phi"]

# bytes por parametro segun tipo de almacenamiento
_BYTES = {"fp32": 4, "fp16": 2, "int": 4}

ESQUEMAS = {
    "fp32":    {p: "fp32" for p in _PARAMS},
    "fp16mix": {"mu_t": "int", "log_sigma": "fp16", "amp": "fp16",
                "freq_raw": "fp16", "phi": "fp16"},
    "fp16full": {p: "fp16" for p in _PARAMS},
}


def bytes_por_atomo(esquema):
    return sum(_BYTES[esquema[p]] for p in _PARAMS)


@torch.no_grad()
def render_cuantizado(modelo, esquema):
    """
    Aplica la cuantizacion del `esquema` a los parametros RAW del modelo,
    renderiza la waveform con esos valores degradados y RESTAURA los originales.
    Devuelve x_hat [T] (detached). No modifica el modelo de forma permanente.
    """
    originales = {p: getattr(modelo, p).detach().clone() for p in _PARAMS}
    try:
        for p in _PARAMS:
            tensor = getattr(modelo, p)
            modo = esquema[p]
            if modo == "fp16":
                tensor.data = tensor.data.half().float()
            elif modo == "int":
                tensor.data = tensor.data.round()
            # "fp32": sin cambios
        x_hat = modelo.render().detach()
    finally:
        for p in _PARAMS:
            getattr(modelo, p).data = originales[p]
    return x_hat
