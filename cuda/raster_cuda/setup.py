import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Build flags focused on release speed.
# --use_fast_math can slightly change floating point results.
cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
]

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
