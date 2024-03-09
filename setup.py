from setuptools import Extension, setup
from torch.utils import cpp_extension

src_files = [
    "earth2grid/third_party/healpy_bare/healpix_bare_wrapper.cpp",
]

setup(
    name='earth2grid',
    ext_modules=[
        cpp_extension.CppExtension(
            'earth2grid._healpix_bare', src_files, extra_compile_args=["-Wno-error=c++11-narrowing"]
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
