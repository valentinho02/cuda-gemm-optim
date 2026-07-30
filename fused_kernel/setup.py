from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_gelu_ext',
    ext_modules=[
        CUDAExtension(
            name='my_gelu_ext',
            sources=[
                'csrc/ln_gelu_binding.cpp',
                'csrc/ln_gelu_kernel.cu'
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)