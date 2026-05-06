import torch

from pinneaple_pinn.factory.pinn_factory import PINNFactory, PINNProblemSpec

spec = PINNProblemSpec(
    pde_residuals=["Derivative(u(t,x), t) + u(t,x)"],
    conditions=[],
    independent_vars=["t", "x"],
    dependent_vars=["u"],
    inverse_params=[],
    verbose=True,
)

factory = PINNFactory(spec)
loss_fn = factory.generate_loss_function()

class Tiny(torch.nn.Module):
    def forward(self, t, x):
        return torch.zeros((t.shape[0], 1), device=t.device, dtype=t.dtype)

model = Tiny()
t = torch.randn(32, 1, requires_grad=True)
x = torch.randn(32, 1, requires_grad=True)

loss, comps = loss_fn(model, {"collocation": (t, x)})

print("loss:", float(loss.item()))
print("components:", comps)
