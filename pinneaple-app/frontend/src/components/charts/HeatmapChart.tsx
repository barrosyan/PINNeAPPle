import Plot from "react-plotly.js";

interface HeatmapChartProps {
  z:          number[][];
  x?:         number[];
  y?:         number[];
  title?:     string;
  colorscale?: string;
  reversescale?: boolean;
  zmin?:      number;
  zmax?:      number;
  height?:    number;
  xlabel?:    string;
  ylabel?:    string;
  maskWith?:  boolean[][] | null;
}

export default function HeatmapChart({
  z,
  x,
  y,
  title,
  colorscale = "Viridis",
  reversescale = false,
  zmin,
  zmax,
  height = 320,
  xlabel = "x",
  ylabel = "y",
  maskWith,
}: HeatmapChartProps) {
  if (!z || z.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-surface2 rounded-xl border border-border"
        style={{ height }}
      >
        <span className="text-muted text-sm">No data</span>
      </div>
    );
  }

  // Apply obstacle mask
  let displayZ = z;
  if (maskWith) {
    displayZ = z.map((row, i) =>
      row.map((val, j) => (maskWith[i]?.[j] ? null : val) as number)
    );
  }

  // Transpose for Plotly (Plotly heatmap expects z[y][x])
  const zT = displayZ[0].map((_, ci) => displayZ.map((row) => row[ci]));

  const trace: Plotly.Data = {
    type:         "heatmap",
    z:            zT,
    x:            x,
    y:            y,
    colorscale,
    reversescale,
    zmin,
    zmax,
    showscale:    true,
    colorbar: {
      thickness: 12,
      tickfont: { color: "#8892a4", size: 10 },
    },
  };

  const layout: Partial<Plotly.Layout> = {
    title:        title ? { text: title, font: { color: "#E8E8E8", size: 13 } } : undefined,
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:   { color: "#8892a4", size: 11 },
    height,
    xaxis: { title: xlabel, gridcolor: "#2a3a5c", scaleanchor: "y" },
    yaxis: { title: ylabel, gridcolor: "#2a3a5c" },
    margin: { l: 50, r: 60, t: title ? 40 : 10, b: 40 },
  };

  return (
    <Plot
      data={[trace]}
      layout={layout}
      config={{ displayModeBar: true, responsive: true }}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
