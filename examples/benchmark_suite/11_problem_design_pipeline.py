"""
Exemplo 11 — Pipeline Completo: Problem Design → Resolution

Demonstra como o fluxo de trabalho do pesquisador (definição do problema)
se conecta diretamente com o pipeline de resolução do PINNeAPPle.

Fluxo:
  1. Pesquisador define o problema com ProblemBuilder (→ ProblemSpec)
  2. ProblemSpec alimenta o Arena pipeline sem passar por YAML
  3. Comparação de múltiplos modelos no mesmo problema
  4. Registro do problema como preset reutilizável

Por que isso importa:
  Antes, o pesquisador precisava escrever YAML configs OU criar presets
  manualmente no registry. Agora pode definir o problema em Python puro
  e rodar imediatamente — o ProblemSpec é o contrato compartilhado entre
  o design do problema e o pipeline de resolução.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── imports principais ─────────────────────────────────────────────────────────
from pinneaple_environment import ProblemBuilder, ProblemSpec, PDETermSpec
from pinneaple_environment import get_preset, list_presets
from pinneaple_arena.api import Arena, ArenaResult

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 1: Definição de problema com ProblemBuilder
# ══════════════════════════════════════════════════════════════════════════════

def demo_problem_builder():
    """
    ProblemBuilder: API fluente para definir PDEs sem YAML nem presets.

    O ProblemSpec resultante é idêntico ao produzido pelos presets internos —
    é a mesma estrutura que o Arena, o solver e o YAML pipeline consomem.
    """
    print("\n" + "═" * 60)
    print("  PARTE 1 — ProblemBuilder → ProblemSpec")
    print("═" * 60)

    # Problema 1: Equação do calor 1D
    #   u_t = α · u_xx     em x ∈ [0,1], t ∈ [0,1]
    #   u(x,0) = sin(πx)
    #   u(0,t) = u(1,t) = 0
    #   Solução analítica: u(x,t) = exp(-α π² t) · sin(πx)
    alpha = 0.05

    heat_spec: ProblemSpec = (
        ProblemBuilder("heat_1d_custom")
        .domain(x=(0.0, 1.0), t=(0.0, 1.0))
        .fields("u")
        .pde("heat_1d", alpha=alpha)
        .ic(
            field="u",
            fn=lambda X: np.sin(np.pi * X[:, 0:1]),      # u(x,0) = sin(πx)
            weight=10.0,
        )
        .bc("dirichlet", field="u", value=0.0, on="x_boundary", weight=10.0)
        .sample(interior=3000, boundary=600, ic=600)
        .solver(name="fdm", method="heat_1d", nx=256, nt=256, alpha=alpha)
        .reference("1D heat equation — analytical: u = exp(-α π² t) sin(πx)")
        .build()
    )

    print(f"\n  Problema criado: {heat_spec.name}")
    print(f"  PDE:    {heat_spec.pde.kind}  params={heat_spec.pde.params}")
    print(f"  Campos: {heat_spec.fields}")
    print(f"  Coords: {heat_spec.coords}  dim={heat_spec.dim}")
    print(f"  Domínio: {heat_spec.domain_bounds}")
    print(f"  Condições: {len(heat_spec.conditions)}")
    for c in heat_spec.conditions:
        print(f"    - [{c.kind}] {c.name}  weight={c.weight}")
    print(f"  Sample defaults: {heat_spec.sample_defaults}")

    # Problema 2: Oscilador de Duffing (não-linear)
    #   ẍ + δẋ + αx + βx³ = γcos(ωt)
    #   → como sistema de 1a ordem: [x, ẋ]
    duffing_spec: ProblemSpec = (
        ProblemBuilder("duffing_oscillator")
        .domain(t=(0.0, 10.0))
        .fields("x", "v")                                    # posição e velocidade
        .pde("system_ode", delta=0.3, alpha=-1.0, beta=1.0, gamma=0.5, omega=1.2)
        .pde_meta(system_type="duffing", order=1)
        .ic(field="x", value=0.1, name="x0", weight=20.0)   # x(0) = 0.1
        .ic(field="v", value=0.0, name="v0", weight=20.0)   # ẋ(0) = 0
        .sample(interior=5000, ic=10)
        .reference("Duffing oscillator — chaotic regime")
        .build()
    )

    print(f"\n  Problema criado: {duffing_spec.name}")
    print(f"  Campos: {duffing_spec.fields}  (t-domain, ODE)")
    print(f"  Params: {duffing_spec.pde.params}")

    return heat_spec, duffing_spec


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2: ProblemSpec → Arena (pipeline de resolução)
# ══════════════════════════════════════════════════════════════════════════════

def demo_arena_from_spec(heat_spec: ProblemSpec) -> ArenaResult:
    """
    Arena.from_spec(spec): o ProblemSpec vai diretamente para o pipeline.

    Sem YAML. Sem presets. Sem configuração extra.
    O ProblemSpec é o contrato: define o problema, o Arena resolve.
    """
    print("\n" + "═" * 60)
    print("  PARTE 2 — Arena.from_spec(spec) → Resolução direta")
    print("═" * 60)

    arena = Arena.from_spec(heat_spec)
    print(f"\n  {arena}")

    result = arena.run(
        model="VanillaPINN",
        epochs=2000,
        lr=1e-3,
        device="auto",
        seed=42,
        verbose=True,
    )

    print(f"\n  {result.summary()}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3: Comparação de múltiplos modelos no mesmo problema
# ══════════════════════════════════════════════════════════════════════════════

def demo_compare_models(heat_spec: ProblemSpec):
    """
    Arena.compare(): treina vários modelos no mesmo ProblemSpec e faz ranking.
    """
    print("\n" + "═" * 60)
    print("  PARTE 3 — Arena.compare() — múltiplos modelos")
    print("═" * 60)

    arena = Arena.from_spec(heat_spec)

    compare_result = arena.compare(
        models=["VanillaPINN", "InversePINN"],
        epochs=1000,     # reduzido para demo rápido
        lr=1e-3,
        device="auto",
        seed=42,
        verbose=False,   # silencioso para demo
    )

    print(f"\n  Leaderboard:")
    print(compare_result.leaderboard())
    return compare_result


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 4: Registro como preset reutilizável
# ══════════════════════════════════════════════════════════════════════════════

def demo_register_preset():
    """
    Depois de projetar um problema com ProblemBuilder, registrá-lo como
    preset permite que outros usem o mesmo problema via YAML ou get_preset().
    """
    print("\n" + "═" * 60)
    print("  PARTE 4 — Registrar problema como preset")
    print("═" * 60)

    # Define e registra
    (
        ProblemBuilder("kdv_1d")
        .domain(x=(-10.0, 10.0), t=(0.0, 2.0))
        .fields("u")
        .pde("kdv", c1=6.0, c2=1.0)            # KdV: u_t + c1·u·u_x + c2·u_xxx = 0
        .ic(
            field="u",
            fn=lambda X: (1 / np.cosh(X[:, 0:1])) ** 2,   # 1-soliton IC
            weight=20.0,
        )
        .bc("dirichlet", field="u", value=0.0, on="x_boundary", weight=5.0)
        .sample(interior=8000, boundary=500, ic=1000)
        .reference("Korteweg–de Vries equation — 1-soliton solution")
        .register("kdv_1d")       # ← registra no preset registry global
    )

    # Agora qualquer um pode usar:
    registered = list_presets()
    if "kdv_1d" in registered:
        print(f"\n  ✓ 'kdv_1d' registrado no registry")

        spec = get_preset("kdv_1d")
        print(f"  Acessível via get_preset('kdv_1d'): {spec.name}")
        print(f"  Presets disponíveis inclui KdV: {'kdv_1d' in registered}")

        # E também via YAML:
        yaml_snippet = """
  # config.yaml — agora funciona sem nenhum código Python extra:
  problem:
    id: kdv_1d
    params: {}
  models:
    - id: pinn_kdv
      model:
        hidden: [64, 64, 64]
        activation: tanh
      train:
        epochs: 10000
        lr: 0.001
"""
        print(f"\n  Equivalente YAML:{yaml_snippet}")
    else:
        print(f"\n  Presets disponíveis: {registered[:10]}...")


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 5: Preset existente → Arena (integração reversa)
# ══════════════════════════════════════════════════════════════════════════════

def demo_preset_to_arena():
    """
    Arena.from_preset(): pega um preset existente e roda com a nova API.
    Mesma pipeline, mesmo contrato (ProblemSpec).
    """
    print("\n" + "═" * 60)
    print("  PARTE 5 — Arena.from_preset() — presets existentes")
    print("═" * 60)

    # Lista presets disponíveis
    presets = list_presets()
    print(f"\n  Presets registrados ({len(presets)} total):")
    for p in sorted(presets)[:12]:
        print(f"    - {p}")
    if len(presets) > 12:
        print(f"    ... +{len(presets)-12} mais")

    # Usa o preset de Burgers via nova API
    arena = Arena.from_preset("burgers_1d", nu=0.01)
    print(f"\n  Rodando: {arena}")

    result = arena.run(
        model="VanillaPINN",
        epochs=500,       # rápido para demo
        lr=1e-3,
        verbose=False,
    )
    print(f"  Loss final: {result.history[-1]['loss']:.4e}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 6: Visualização e comparação
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(heat_result: ArenaResult, output_path: Path):
    """Plota curva de convergência e campo predito."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # ── Plot 1: Curva de convergência ─────────────────────────────────────────
    ax = axes[0]
    losses = [h["loss"] for h in heat_result.history]
    ax.semilogy(losses, color="#58a6ff", linewidth=1.5)
    ax.set_title("Convergência da Loss", color="white", fontsize=11)
    ax.set_xlabel("Epoch", color="#8b949e")
    ax.set_ylabel("Loss", color="#8b949e")
    ax.grid(True, alpha=0.2, color="#30363d")

    # ── Plot 2: Campo predito u(x,t) ──────────────────────────────────────────
    ax = axes[1]
    model = heat_result.model
    if model is not None:
        model.eval()
        nx, nt = 50, 50
        x_vals = np.linspace(0, 1, nx, dtype=np.float32)
        t_vals = np.linspace(0, 1, nt, dtype=np.float32)
        XX, TT = np.meshgrid(x_vals, t_vals)
        pts = np.column_stack([XX.ravel(), TT.ravel()])
        xt = torch.from_numpy(pts)
        with torch.no_grad():
            u_pred = model(xt)
            if hasattr(u_pred, "y"):
                u_pred = u_pred.y
            if isinstance(u_pred, torch.Tensor):
                u_pred = u_pred.numpy()
        U = u_pred.reshape(nt, nx)
        im = ax.pcolormesh(x_vals, t_vals, U, cmap="RdBu_r", shading="auto")
        plt.colorbar(im, ax=ax, label="u(x,t)")
        ax.set_title("u(x,t) predito — VanillaPINN", color="white", fontsize=11)
        ax.set_xlabel("x", color="#8b949e")
        ax.set_ylabel("t", color="#8b949e")

    # ── Plot 3: Diagrama do pipeline ───────────────────────────────────────────
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Pipeline Integrado", color="white", fontsize=11)

    pipeline = [
        (9.0, "[1] ProblemBuilder",   "#3fb950", "define PDE, domínio, BCs, ICs"),
        (7.2, "[2] ProblemSpec",      "#58a6ff", "contrato frozen entre design e solução"),
        (5.4, "[3] Arena.from_spec()",  "#d29922", "carrega spec, monta batch"),
        (3.6, "[4] generate_dataset()", "#79c0ff", "coloca pontos no domínio"),
        (1.8, "[5] train_loop()",       "#3fb950", "treina com physics loss compiler"),
        (0.0, "[6] ArenaResult",        "#58a6ff", "metrics, model, history, artifacts"),
    ]
    for y, label, color, detail in pipeline:
        ax.text(0.3, y + 0.5, label, color=color, fontsize=9, fontweight="bold")
        ax.text(0.3, y - 0.1, detail, color="#8b949e", fontsize=7)
        if y > 0.5:
            ax.annotate("", xy=(1.5, y - 0.5), xytext=(1.5, y - 1.1),
                        arrowprops=dict(arrowstyle="->", color="#30363d"))

    fig.suptitle(
        "Exemplo 11 — Problem Design → Arena Pipeline",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    plt.savefig(str(output_path), dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Figura salva: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Exemplo 11 — Problem Design + Arena Pipeline")
    print("=" * 60)

    # Parte 1: Define problemas
    heat_spec, duffing_spec = demo_problem_builder()

    # Parte 2: Resolve com Arena diretamente do spec
    heat_result = demo_arena_from_spec(heat_spec)

    # Parte 3: Compara modelos
    # demo_compare_models(heat_spec)  # descomente para rodar (mais lento)

    # Parte 4: Registra como preset
    demo_register_preset()

    # Parte 5: Usa preset via nova API
    demo_preset_to_arena()

    # Parte 6: Visualização
    plot_results(heat_result, OUTPUT_DIR / "problem_design_pipeline.png")

    # ── Resumo ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMO — Contratos da integração")
    print("=" * 60)
    print("""
  PROBLEM DESIGN  (pinneaple_environment)
  ────────────────────────────────────────
  ProblemBuilder  → fluent API para pesquisadores
       ↓ .build()
  ProblemSpec     → contrato frozen (dim, coords, fields, pde,
                    conditions, domain_bounds, sample_defaults)
       ↓
  RESOLUTION PIPELINE  (pinneaple_arena)
  ────────────────────────────────────────
  Arena.from_spec(spec)   ← entrada principal
  Arena.from_preset(id)   ← presets registrados
  Arena.from_yaml(path)   ← YAML workflow existente
       ↓ .run(model, epochs, ...)
  ArenaResult             → model, history, metrics, artifacts
       ↓ .compare(models)
  ArenaCompareResult      → leaderboard multi-modelo

  REUTILIZAÇÃO
  ────────────────────────────────────────
  builder.register("meu_preset")  → disponível em YAML e get_preset()
  spec salvo como JSON / pickle   → reproduzibilidade total
""")


if __name__ == "__main__":
    main()
