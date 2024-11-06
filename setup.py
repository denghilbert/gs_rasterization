from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import multiprocessing

# Determine the number of available CPU cores
num_cores = multiprocessing.cpu_count()

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/utils.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(parallel=num_cores)
    }
)

