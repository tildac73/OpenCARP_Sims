import sys, numpy as np
import scipy.signal as S, scipy.sparse as SP
sys.path.insert(0, "shim"); 
import importlib.util
def load(name):
    spec = importlib.util.spec_from_file_location(name, f"shim/scipy/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sig, sp = load("signal"), load("sparse")
rng = np.random.default_rng(0)
ref = S.butter(2, [30, 450], btype="bandpass", fs=5000, output="sos")
assert np.allclose(ref, sig.butter(2, [30, 450], btype="bandpass", fs=5000, output="sos"), rtol=0, atol=1e-15)
x = rng.normal(size=(7, 16, 3001)).astype(np.float32); x[..., 1000] += 50
a, b = S.sosfilt(ref, x, axis=-1), sig.sosfilt(ref, x, axis=-1)
print("sosfilt max abs diff", np.abs(a - b).max(), "dtype", a.dtype, b.dtype)
r = rng.integers(0, 50, 20000); c = rng.integers(0, 3000, 20000); d = rng.normal(size=20000)
A = SP.coo_matrix((d, (r, c)), shape=(50, 3000)).tocsr(); B = sp.coo_matrix((d, (r, c)), shape=(50, 3000)).tocsr()
V = rng.normal(size=(3000, 40))
print("sparse matmul max abs diff", np.abs(A @ V - B @ V).max())
