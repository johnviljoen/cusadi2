"""Python bridge to the CUDA QP solver (mpc_qp_lib.cu): pass device arrays from any framework that exposes
`__cuda_array_interface__` (JAX, Warp, CuPy, PyTorch) straight to the solver, no host round trip.

    qp = QPSolver(build_dir, B)                     # loads <build_dir>/libmpc_qp_double.so
    dx = qp.solve(blk, q, z_lb, z_ub)               # (B, N, NS) numpy; arrays must be complete (e.g. block_until_ready)
"""
import ctypes, os, numpy as np

def device_pointer(a, shape, itemsize):
    """Device address of a contiguous GPU array, after checking shape and element size."""
    cai = a.__cuda_array_interface__
    if tuple(cai["shape"]) != tuple(shape): raise ValueError(f"expected shape {tuple(shape)}, got {tuple(cai['shape'])}")
    if cai["typestr"] != f"<f{itemsize}": raise ValueError(f"expected <f{itemsize}, got {cai['typestr']}")
    c_strides = tuple(itemsize * int(np.prod(shape[i + 1:])) for i in range(len(shape)))
    if cai.get("strides") not in (None, c_strides): raise ValueError("array is not C-contiguous")
    return ctypes.c_void_p(cai["data"][0])

class QPSolver:
    def __init__(self, build_dir, B, dtype="double"):
        self.lib = lib = ctypes.CDLL(os.path.join(build_dir, f"libmpc_qp_{dtype}.so"))
        lib.mpcqp_create.restype = ctypes.c_void_p; lib.mpcqp_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.mpcqp_solve.argtypes = [ctypes.c_void_p] * 5; lib.mpcqp_dims.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.mpcqp_get_dx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; lib.mpcqp_destroy.argtypes = [ctypes.c_void_p]
        lib.mpcqp_dx.restype = ctypes.c_void_p; lib.mpcqp_dx.argtypes = [ctypes.c_void_p]
        self.h = lib.mpcqp_create(os.fsencode(build_dir), B)
        if not self.h: raise RuntimeError("mpcqp_create failed")
        dims = (ctypes.c_int * 6)(); lib.mpcqp_dims(self.h, dims)
        self.N, self.NS, self.NG, self.NBLK, self.B, self.itemsize = dims
        self.np_dtype = np.float64 if self.itemsize == 8 else np.float32

    def launch(self, blk, q, z_lb, z_ub):
        """Solve; returns once the step is on the device (device address: self.dx_pointer)."""
        s = (self.B, self.N); ptr = lambda a, w: device_pointer(a, s + (w,), self.itemsize)
        if self.lib.mpcqp_solve(self.h, ptr(blk, self.NBLK), ptr(q, self.NS), ptr(z_lb, self.NG), ptr(z_ub, self.NG)):
            raise RuntimeError("mpcqp_solve failed")

    @property
    def dx_pointer(self): return self.lib.mpcqp_dx(self.h)

    def dx(self):
        out = np.empty((self.B, self.N, self.NS), self.np_dtype)
        if self.lib.mpcqp_get_dx(self.h, out.ctypes.data_as(ctypes.c_void_p)): raise RuntimeError("mpcqp_get_dx failed")
        return out

    def solve(self, blk, q, z_lb, z_ub):
        self.launch(blk, q, z_lb, z_ub); return self.dx()

    def close(self):
        if self.h: self.lib.mpcqp_destroy(self.h); self.h = None
    __del__ = close

class DeviceArray:
    """A device buffer owned by the library, presented through `__cuda_array_interface__`."""
    def __init__(self, ptr, shape, itemsize):
        self.__cuda_array_interface__ = dict(shape=tuple(shape), typestr=f"<f{itemsize}", data=(ptr, False), version=2)

class Cusadi2Assembler:
    """The cusadi2 plan `mpc_assemble` behind the same kind of call as the MJX assembly:
    assemble(x_pad, p_stage, p_shared) -> (blk, q, z_lb, z_ub) on the device, complete on return."""
    def __init__(self, build_dir, B, qp):
        self.lib = lib = qp.lib; self.B, self.N, self.NS, self.itemsize = B, qp.N, qp.NS, qp.itemsize
        lib.mpcasm_create.restype = ctypes.c_void_p; lib.mpcasm_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.mpcasm_evaluate.argtypes = [ctypes.c_void_p] * 4; lib.mpcasm_destroy.argtypes = [ctypes.c_void_p]
        lib.mpcasm_out.restype = ctypes.c_void_p; lib.mpcasm_out.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.h = lib.mpcasm_create(os.fsencode(build_dir), B)
        if not self.h: raise RuntimeError("mpcasm_create failed")
        self.out = tuple(DeviceArray(lib.mpcasm_out(self.h, i), (B, qp.N, w), qp.itemsize) for i, w in enumerate((qp.NBLK, qp.NS, qp.NG, qp.NG)))

    def __call__(self, x_pad, p_stage, p_shared):
        ptr = lambda a: ctypes.c_void_p(a.__cuda_array_interface__["data"][0])
        if tuple(x_pad.__cuda_array_interface__["shape"]) != (self.B, self.N + 2, self.NS): raise ValueError("x_pad shape")
        if self.lib.mpcasm_evaluate(self.h, ptr(x_pad), ptr(p_stage), ptr(p_shared)): raise RuntimeError("mpcasm_evaluate failed")
        return self.out

    def close(self):
        if self.h: self.lib.mpcasm_destroy(self.h); self.h = None
    __del__ = close

if __name__ == "__main__":
    # Self-test: the benchmark QP assembled on the CPU by the CasADi stage function, uploaded as Warp arrays,
    # solved through the bridge and compared with the reference step.   python mpc_bridge.py <build_dir> [B]
    import sys, casadi as ca, warp as wp
    build = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    qp = QPSolver(build, B); N, NS = qp.N, qp.NS
    rd = lambda n: np.fromfile(os.path.join(build, "inputs", n + ".f64"))
    x, dt, ps, pg = rd("x_pad").reshape(N + 2, NS), rd("dt_pad"), rd("p_stage").reshape(N, -1), rd("p_shared")
    f = ca.Function.load(os.path.join(build, "fns", "mpc_assemble.casadi"))
    outs = [f(x[k], x[k + 1], x[k + 2], dt[k], dt[k + 1], float(k == 0), float(k < N - 1), ps[k], pg) for k in range(N)]
    host = [np.stack([np.asarray(o[i]).reshape(-1) for o in outs]) for i in range(4)]
    dev = [wp.array(np.broadcast_to(a, (B,) + a.shape).copy(), dtype=wp.float64, device="cuda") for a in host]
    err = np.abs(qp.solve(*dev) - rd("dx_ref").reshape(N, NS)).max()
    print(f"bridge self-test (Warp arrays, B = {B}): max |dx - dx_ref| = {err:.3e}", "OK" if err < 1e-8 else "FAIL")
    sys.exit(0 if err < 1e-8 else 1)
