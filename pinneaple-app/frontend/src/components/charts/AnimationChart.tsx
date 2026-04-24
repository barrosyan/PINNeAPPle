import { useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";

interface AnimationChartProps {
  frames:     number[][][];          // [frame][nx][ny]
  framesUy?:  number[][][];
  obstacle?:  boolean[][] | null;
  field?:     "velocity" | "ux" | "uy";
  title?:     string;
  height?:    number;
}

export default function AnimationChart({
  frames,
  framesUy,
  obstacle,
  field = "velocity",
  title = "LBM Trajectory",
  height = 340,
}: AnimationChartProps) {
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing,  setPlaying]  = useState(false);

  const maxFrame = frames.length - 1;

  const advance = useCallback(() => {
    setFrameIdx((i) => (i >= maxFrame ? 0 : i + 1));
  }, [maxFrame]);

  // Simple requestAnimationFrame loop
  const togglePlay = () => {
    setPlaying((p) => {
      if (!p) {
        const loop = () => {
          setFrameIdx((i) => {
            const next = i >= maxFrame ? 0 : i + 1;
            if (next === 0) setPlaying(false);
            return next;
          });
        };
        const id = setInterval(loop, 150);
        // Store interval id on window for cleanup
        (window as unknown as Record<string, unknown>)._pinnAnimId = id;
      } else {
        clearInterval((window as unknown as Record<string, unknown>)._pinnAnimId as number);
      }
      return !p;
    });
  };

  if (!frames || frames.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-surface2 rounded-xl border border-border"
        style={{ height }}
      >
        <span className="text-muted text-sm">No frames to display</span>
      </div>
    );
  }

  const raw = frames[frameIdx];
  let z = raw;
  if (field === "velocity" && framesUy) {
    const uy = framesUy[frameIdx];
    z = raw.map((row, i) => row.map((ux, j) => Math.sqrt(ux ** 2 + (uy[i]?.[j] ?? 0) ** 2)));
  }

  // Apply mask
  let displayZ = z;
  if (obstacle) {
    displayZ = z.map((row, i) =>
      row.map((val, j) => (obstacle[i]?.[j] ? null : val) as number)
    );
  }

  const zT = displayZ[0].map((_, ci) => displayZ.map((row) => row[ci]));

  const trace: Plotly.Data = {
    type:      "heatmap",
    z:         zT,
    colorscale: "Viridis",
    showscale:  true,
    colorbar: { thickness: 12, tickfont: { color: "#8892a4", size: 10 } },
  };

  const layout: Partial<Plotly.Layout> = {
    title:         title
      ? { text: `${title} — frame ${frameIdx + 1}/${frames.length}`,
          font: { color: "#E8E8E8", size: 12 } }
      : undefined,
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:   { color: "#8892a4", size: 11 },
    height,
    xaxis:  { scaleanchor: "y", gridcolor: "#2a3a5c" },
    yaxis:  { gridcolor: "#2a3a5c" },
    margin: { l: 40, r: 60, t: title ? 40 : 10, b: 30 },
  };

  return (
    <div>
      <Plot
        data={[trace]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height }}
        useResizeHandler
      />
      {/* Controls */}
      <div className="flex items-center gap-3 mt-2 px-2">
        <button className="btn-ghost p-1" onClick={() => setFrameIdx(0)}>
          <SkipBack size={14} />
        </button>
        <button className="btn-ghost p-1" onClick={togglePlay}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button className="btn-ghost p-1" onClick={() => setFrameIdx(maxFrame)}>
          <SkipForward size={14} />
        </button>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={frameIdx}
          onChange={(e) => setFrameIdx(Number(e.target.value))}
          className="flex-1 h-1 accent-primary"
        />
        <span className="text-xs text-muted font-mono w-16 text-right">
          {frameIdx + 1}/{frames.length}
        </span>
      </div>
    </div>
  );
}
