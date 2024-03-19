import subprocess
from typing import List

from setuptools import setup
from torch.utils import cpp_extension


def get_compiler():
    try:
        # Try to get the compiler path from the CC environment variable
        # If not set, it will default to gcc (which could be symlinked to clang or g++)
        compiler = subprocess.check_output(["gcc", "--version"], universal_newlines=True)

        if "clang" in compiler:
            return "clang"
        elif "g++" in compiler or "gcc" in compiler:
            return "gnu"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error detecting compiler: {e}")
        return "unknown"


compiler_type = get_compiler()
extra_compile_args: List[str] = []

if compiler_type == "clang":
    print("Detected Clang compiler.")
    # Additional settings or flags specific to Clang can be added here
    extra_compile_args += ["-Wno-error=c++11-narrowing", "-Wno-c++11-narrowing"]
elif compiler_type == "gnu":
    print("Detected GNU compiler.")
    # Additional settings or flags specific to G++ can be added here
else:
    print("Could not detect compiler or unknown compiler detected.")


src_files = [
    "earth2grid/third_party/healpy_bare/healpix_bare_wrapper.cpp",
]

setup(
    name='earth2grid',
    ext_modules=[
        cpp_extension.CppExtension(
            'earth2grid._healpix_bare',
            src_files,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
