import torch
from pinneapple_solvers.fft import FFTSolver

def test_fft_solver_smoke():
    solver = FFTSolver()
    x = torch.randn(4, 128)
    out = solver(x)
    assert hasattr(out, "result")
    assert out.result.shape[0] == 4
