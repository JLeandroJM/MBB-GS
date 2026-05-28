import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _get_raster_batch_size():
    value = os.environ.get("RASTER_BATCH_SIZE", "256")

    try:
        value_int = int(value)
    except ValueError:
        raise ValueError("RASTER_BATCH_SIZE debe ser un entero. Ejemplo: 128, 256, 512")

    if value_int <= 0:
        raise ValueError("RASTER_BATCH_SIZE debe ser mayor que 0")

    return value_int


raster_batch_size = _get_raster_batch_size()

# Build flags focused on release speed.
# --use_fast_math can slightly change floating point results.
cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    f"-DRASTER_BATCH_SIZE={raster_batch_size}",
]

print(f"[setup] RASTER_BATCH_SIZE={raster_batch_size}")

setup(
    name="raster_cuda",
    ext_modules=[
        CUDAExtension(
            name="raster_cuda",
            sources=[
                "raster_cuda.cpp",
                "raster_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)