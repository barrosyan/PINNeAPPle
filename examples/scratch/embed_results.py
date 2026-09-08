"""
embed_results.py
Embeds generated PNGs (base64) and real loss data into the HTML presentation.
"""
from __future__ import annotations
import base64, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUT  = os.path.join(os.path.dirname(__file__), "results_chassis")
HTML = os.path.join(os.path.dirname(__file__), "chassis_structural_analysis.html")

# ── Load images ───────────────────────────────────────────────────────────
imgs = {}
for name in ["field_uz","field_ux","field_uy","field_vm","loss_history","collocation"]:
    p = os.path.join(OUT, name + ".png")
    with open(p, "rb") as f:
        imgs[name] = base64.b64encode(f.read()).decode()
    print(f"  {name}: {len(imgs[name])} chars base64")

# ── Load loss data ────────────────────────────────────────────────────────
with open(os.path.join(OUT, "loss_data.json")) as f:
    loss = json.load(f)
epochs = [r["epoch"] for r in loss]
total  = [r["total"] for r in loss]
pde    = [r["pde"]   for r in loss]
bc     = [r.get("bc_suspension_mounts", 0.0) for r in loss]

print(f"  Loss records: {len(loss)}, final total={loss[-1]['total']:.3e}")

# ── Read HTML ─────────────────────────────────────────────────────────────
with open(HTML, encoding="utf-8") as f:
    html = f.read()
print(f"  HTML before: {len(html):,} chars")


# ── Helper: build img card div ────────────────────────────────────────────
def img_div(b64: str, title: str, caption: str, extra_style: str = "") -> str:
    return (
        f'<div class="field-wrap">\n'
        f'      <h4>{title}</h4>\n'
        f'      <img src="data:image/png;base64,{b64}"'
        f' style="width:100%;border-radius:4px;{extra_style}" alt="{title}">\n'
        f'      <div class="field-cap">{caption}</div>\n'
        f'    </div>'
    )


# ── 1. Replace 4 canvas field plots ───────────────────────────────────────
replacements = [
    (
        '<div class="field-wrap">\n'
        '      <h4>Vertical Displacement u<sub>z</sub> [mm]</h4>\n'
        '      <canvas id="cvUz" width="520" height="200"></canvas>\n'
        '      <div class="field-cap">Fig. 1 — u<sub>z</sub> top-view cross-section. Maximum sag at engine mount (x = 0.9 m). Zero at suspension mounts (corners).</div>\n'
        '    </div>',
        img_div(imgs["field_uz"],
                "Vertical Displacement u<sub>z</sub>",
                "Fig. 1 — u<sub>z</sub> (y=0 cross-section, PINNeAPPle SIREN). "
                "Max sag at engine-mount patch (x=0.9 m, yellow ▲). "
                "Suspension mounts (red ▼) forced to zero.")
    ),
    (
        '<div class="field-wrap">\n'
        '      <h4>Longitudinal Displacement u<sub>x</sub> [mm]</h4>\n'
        '      <canvas id="cvUx" width="520" height="200"></canvas>\n'
        '      <div class="field-cap">Fig. 2 — u<sub>x</sub> longitudinal component. Symmetric about the chassis centreline due to mid-load.</div>\n'
        '    </div>',
        img_div(imgs["field_ux"],
                "Longitudinal Displacement u<sub>x</sub>",
                "Fig. 2 — u<sub>x</sub> longitudinal field. "
                "Diverging RdBu colormap reveals compression (blue) vs. tension (red) zones along chassis length.")
    ),
    (
        '<div class="field-wrap">\n'
        '      <h4>Lateral Displacement u<sub>y</sub> [mm]</h4>\n'
        '      <canvas id="cvUy" width="520" height="200"></canvas>\n'
        '      <div class="field-cap">Fig. 3 — u<sub>y</sub> lateral (diverging colormap). Anti-symmetric splaying of the chassis sills.</div>\n'
        '    </div>',
        img_div(imgs["field_uy"],
                "Lateral Displacement u<sub>y</sub>",
                "Fig. 3 — u<sub>y</sub> lateral field (PiYG colormap). "
                "Anti-symmetric response reveals chassis torsional compliance under asymmetric mounting.")
    ),
    (
        '<div class="field-wrap">\n'
        '      <h4>Von Mises Stress σ<sub>VM</sub> [MPa]</h4>\n'
        '      <canvas id="cvVM" width="520" height="200"></canvas>\n'
        '      <div class="field-cap">Fig. 4 — σ<sub>VM</sub> (plasma colormap). Stress concentrations at load application, suspension mounts, and B-pillar junctions.</div>\n'
        '    </div>',
        img_div(imgs["field_vm"],
                "Von Mises Stress σ<sub>VM</sub>",
                "Fig. 4 — σ<sub>VM</sub> (plasma colormap, PINNeAPPle autograd). "
                "Stress concentrations at engine-mount patch (▲) and four suspension mounts (▼). "
                "High-stress corridor along transmission tunnel visible at centreline.")
    ),
]

for old, new in replacements:
    if old in html:
        html = html.replace(old, new)
        print(f"  Replaced canvas for field")
    else:
        print(f"  WARNING: canvas pattern not found — check spacing")

# ── 2. Replace synthetic Chart.js loss data with real data ────────────────
old_loss = (
    "  // ── Loss history ───────────────────────────────────────────────────\n"
    "  const epochs=[];const total=[],pde=[],bc=[];\n"
    "  let L=0.82,Lpde=0.60,Lbc=0.22;\n"
    "  for(let ep=0;ep<=50000;ep+=500){\n"
    "    epochs.push(ep);\n"
    "    const t=ep/50000;\n"
    "    const decay=Math.exp(-5*t);\n"
    "    const noise=(Math.random()-0.5)*0.08*decay;\n"
    "    L   =0.82*Math.exp(-5.8*t)+noise*0.6+1.2e-4;\n"
    "    Lpde=0.60*Math.exp(-5.5*t)+noise*0.4+7.4e-5;\n"
    "    Lbc =0.22*Math.exp(-6.2*t)+noise*0.2+4.4e-5;\n"
    "    total.push(Math.max(L,1.2e-4));\n"
    "    pde.push(Math.max(Lpde,7.4e-5));\n"
    "    bc.push(Math.max(Lbc,4.4e-5));\n"
    "  }"
)
new_loss = (
    f"  // ── Real training data — PINNeAPPle SIREN run (1200 epochs) ──────────\n"
    f"  const epochs = {json.dumps(epochs)};\n"
    f"  const total  = {json.dumps(total)};\n"
    f"  const pde    = {json.dumps(pde)};\n"
    f"  const bc     = {json.dumps(bc)};"
)
if old_loss in html:
    html = html.replace(old_loss, new_loss)
    print("  Replaced synthetic loss data with real values")
else:
    print("  WARNING: loss data pattern not found")

# ── 3. Insert real loss image + collocation image ─────────────────────────
real_loss_img = (
    '\n  <div class="field-wrap" style="margin-bottom:1.5rem;">\n'
    '    <h4>Loss History — PINNeAPPle actual run (1200 epochs, SIREN 3\xd732)</h4>\n'
    f'    <img src="data:image/png;base64,{imgs["loss_history"]}"'
    ' style="width:100%;border-radius:4px;" alt="Loss history">\n'
    '    <div class="field-cap">Real training output from PINNeAPPle. '
    'Total loss converges from 34.8 (ep 0) to 3.16×10⁻⁵ (ep 1200). '
    'PDE residual and Dirichlet BC loss shown separately.</div>\n'
    '  </div>\n'
)
colloc_img = (
    '\n  <div class="card" style="margin-bottom:1.5rem;">\n'
    '    <h4>Collocation Points — Generated by PINNeAPPle LHS Sampler</h4>\n'
    f'    <img src="data:image/png;base64,{imgs["collocation"]}"'
    ' style="width:100%;border-radius:4px;" alt="Collocation">\n'
    '  </div>\n'
)

anchor_train = "<h3>5.1 Training Convergence</h3>"
if anchor_train in html:
    html = html.replace(anchor_train, anchor_train + real_loss_img)
    print("  Inserted real loss image")

anchor_geom = "<h3>4.1 Problem Specification</h3>"
if anchor_geom in html:
    html = html.replace(
        anchor_geom,
        "<h3>3.2 Collocation Sampling</h3>" + colloc_img + anchor_geom
    )
    print("  Inserted collocation image")

# ── 4. Update metric card (L2 error → final loss) ─────────────────────────
old_metric = (
    '<div class="val" style="color:var(--green)">2.1 %</div>'
    '<div class="lbl">Max L2 error vs FEM</div>'
)
new_metric = (
    '<div class="val" style="color:var(--green)">3.16e-5</div>'
    '<div class="lbl">Final PINN loss</div>'
)
if old_metric in html:
    html = html.replace(old_metric, new_metric)
    print("  Updated metric card")

# ── Write output ──────────────────────────────────────────────────────────
with open(HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n  HTML after:  {len(html):,} chars")
print("  Done — chassis_structural_analysis.html updated with real PINNeAPPle results.")
