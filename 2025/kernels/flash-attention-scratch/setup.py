import os
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))


def get_nvcc_flags():
    args = [
        "-O3",
        "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xptxas=-warn-spills",
        "-Xptxas=-warn-lmem-usage",
        "--resource-usage",
        "--keep",
        "-gencode=arch=compute_89,code=sm_89",
        "-g"
    ]

    return args


ext_modules = [
    CUDAExtension(
        name="my_flash_attn_2_cuda",
        sources=[
            "src/flash_api.cpp",
            "src/flash.cu"
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "nvcc": get_nvcc_flags()
        },
        include_dirs=[
            Path(this_dir)],
    )
]


setup(
    name="my_flash_attn",
    packages=find_packages(
        exclude=(
            "build", "src", "my_flash_attn.egg-info", "dist", "tests"
        )
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
