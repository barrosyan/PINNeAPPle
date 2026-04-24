import Plot from "react-plotly.js";
import type { TrainingEntry } from "@/types";

interface LossChartProps {
  history:   TrainingEntry[];
  height?:   number;
  logScale?: boolean;
  showComponents?: boolean;
}

const LAYOUT_BASE = {
  template:     "plotly_dark" as const,
  paper_bgcolor: "transparent",
  plot_bgcolor:  "transparent",
  font:  { color: "#8892a4", size: 11 },
  xaxis: { title: "Epoch", gridcolor: "#2a3a5c", zerolinecolor: "#2a3a5c" },
  margin: { l: 50, r: 20, t: 20, b: 40 },
  legend: { bgcolor: "transparent", bordercolor: "#2a3a5c" },
};

export default function LossChart({
  history,
  height = 280,
  logScale = true,
  showComponents = false,
}: LossChartProps) {
  if (!history || history.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-surface2 rounded-xl border border-border"
        style={{ height }}
      >
        <span className="text-muted text-sm">No training data yet</span>
      </div>
    );
  }

  const epochs = history.map((h) => h.epoch);
  const losses = history.map((h) => h.loss);

  const traces: Plotly.Data[] = [
    {
      x: epochs,
      y: losses,
      mode:  "lines",
      name:  "Total loss",
      line:  { color: "#FF6B35", width: 2 },
    },
  ];

  if (showComponents) {
    const pde = history.map((h) => h.pde).filter(Boolean) as number[];
    const bc  = history.map((h) => h.bc).filter(Boolean) as number[];
    if (pde.length) {
      traces.push({
        x: epochs.slice(-pde.length),
        y: pde,
        mode: "lines",
        name: "PDE residual",
        line: { color: "#4ECDC4", width: 1.5, dash: "dash" },
      });
    }
    if (bc.length) {
      traces.push({
        x: epochs.slice(-bc.length),
        y: bc,
        mode: "lines",
        name: "BC loss",
        line: { color: "#FFE66D", width: 1.5, dash: "dot" },
      });
    }
  }

  const layout: Partial<Plotly.Layout> = {
    ...LAYOUT_BASE,
    height,
    yaxis: {
      title:        "Loss",
      type:         logScale ? "log" : "linear",
      gridcolor:    "#2a3a5c",
      zerolinecolor:"#2a3a5c",
    },
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
