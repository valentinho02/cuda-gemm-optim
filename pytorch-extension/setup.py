from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_gemm_ext',
    ext_modules=[
        CUDAExtension(
            name='my_gemm_ext',
            sources=[
                'gemm_binding.cpp',
                'gemm_kernel.cu'
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)