"""Print the toolchain paths CMake needs to build cusadi2.jax._ffi, all taken from
pip packages in the interpreter running the build (no system CUDA):

    NVCC=...            nvidia-cuda-nvcc
    CUDA_INCLUDE=...    nvidia-cuda-runtime / cccl / crt headers
    CUDART=...          libcudart.so.<major> (the wheels ship no unversioned symlink)
    XLA_INCLUDE=...     jaxlib's XLA FFI headers
    NANOBIND_DIR=...    nanobind's CMake config

Exit status 1 with a reason on stderr when something is missing, so CMake can
skip the extension instead of failing the whole install.
"""
import sys
from importlib.util import find_spec
from pathlib import Path


def base(spec_name):
    spec = find_spec(spec_name)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"{spec_name} not installed")
    return Path(list(spec.submodule_search_locations)[0])


def main():
    cu = base("nvidia.cu13")
    nvcc = cu / "bin" / "nvcc"
    if not nvcc.exists():
        raise ImportError("nvidia-cuda-nvcc (CUDA 13) is not installed")
    cudart = sorted((cu / "lib").glob("libcudart.so*"), key=lambda p: len(p.name))
    if not cudart:
        raise ImportError("nvidia-cuda-runtime (CUDA 13) is not installed")
    from jax import ffi as jffi
    import nanobind
    print(f"NVCC={nvcc}")
    print(f"CUDA_INCLUDE={cu / 'include'}")
    print(f"CUDART={cudart[-1].resolve()}")
    print(f"XLA_INCLUDE={jffi.include_dir()}")
    print(f"NANOBIND_DIR={nanobind.cmake_dir()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"{type(e).__name__}: {e}\n")
        sys.exit(1)
