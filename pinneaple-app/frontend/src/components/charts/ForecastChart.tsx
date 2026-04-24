import Plot from "react-plotly.js";

interface ForecastChartProps {
  history:   number[];
  forecast:  number[];
  inputLen?: number;
  title?:    string;
  height?:   number;
}

export default function ForecastChart({
  history,
  forecast,
  inputLen,
  title = "Timeseries Forecast",
  height = 300,
}: ForecastChartProps) {
  const nHist  = history.length;
  const tHist  = Array.from({ length: nHist }, (_, i) => i);
  const tFore  = Array.from({ length: forecast.length }, (_, i) => nHist + i);

  const traces: Plotly.Data[] = [
    {
      x:    tHist,
      y:    history,
      mode: "lines",
      name: "History",
      line: { color: "#4ECDC4", width: 1.5 },
    },
    {
      x:    tFore,
      y:    forecast,
      mode: "lines+markers",
      name: "Forecast",
      line: { color: "#FF6B35", width: 2, dash: "dash" },
      marker: { size: 4, color: "#FF6B35" },
    },
  ];

  if (inputLen) {
    traces.push({
      x:    [nHist - inputLen, nHist - inputLen],
      y:    [Math.min(...history), Math.max(...history)],
      mode: "lines",
      name: "Context start",
      line: { color: "#FFE66D", width: 1, dash: "dot" },
      showlegend: true,
    });
  }

  const layout: Partial<Plotly.Layout> = {
    title:         title ? { text: title, font: { color: "#E8E8E8", size: 13 } } : undefined,
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:   { color: "#8892a4", size: 11 },
    height,
    xaxis:  { title: "t", gridcolor: "#2a3a5c", zerolinecolor: "#2a3a5c" },
    yaxis:  { title: "value", gridcolor: "#2a3a5c", zerolinecolor: "#2a3a5c" },
    legend: { bgcolor: "transparent", bordercolor: "#2a3a5c" },
    margin: { l: 50, r: 20, t: title ? 40 : 10, b: 40 },
    shapes: inputLen
      ? [{
          type:   "rect",
          x0:     nHist - inputLen,
          x1:     nHist,
          y0:     0,
          y1:     1,
          xref:   "x",
          yref:   "paper",
          fillcolor: "#FF6B35",
          opacity:   0.07,
          line: { width: 0 },
        }]
      : [],
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: true, responsive: true }}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
