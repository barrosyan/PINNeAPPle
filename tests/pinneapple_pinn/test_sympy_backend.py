import torch

from pinneapple_pinn.factory.pinn_factory import PINNFactory, PINNProblemSpec


def test_sympy_backend_basic_residual():
    # PDE: u_t + u = 0
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(u(t,x), t) + u(t,x)"],
        conditions=[],
        independent_vars=["t", "x"],
        dependent_vars=["u"],
        inverse_params=[],
        verbose=False,
    )

    factory = PINNFactory(spec)
    loss_fn = factory.generate_loss_function()

    # Modelo que depende de t (pra autograd funcionar), mas ainda retorna 0
    class M(torch.nn.Module):
        def forward(self, t, x):
            # depende de t, mas é identicamente 0
            return t * 0.0

    model = M()
    t = torch.zeros((8, 1), requires_grad=True)
    x = torch.zeros((8, 1), requires_grad=True)

    loss, comps = loss_fn(model, {"collocation": (t, x)})

    assert torch.isfinite(loss).all()
    assert float(loss.item()) >= 0.0
    assert isinstance(comps, dict)
    assert "total" in comps or "pde" in comps
