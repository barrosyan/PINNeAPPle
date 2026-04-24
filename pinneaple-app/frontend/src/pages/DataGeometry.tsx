import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useActiveProject } from "@/hooks/useProject";
import { uploadFile, getFiles, getFileData, deleteFile } from "@/api/files";
import { useAppStore } from "@/store";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";
import { TabsUnderline } from "@/components/ui/Tabs";
import { Spinner } from "@/components/ui/Spinner";
import Plot from "react-plotly.js";
import type { UploadedFile } from "@/types";
import { Upload, Trash2, FileText, Box, Database, RefreshCw } from "lucide-react";
import toast from "react-hot-toast";
import { useQueryClient } from "@tanstack/react-query";

export default function DataGeometry() {
  const { data: project }       = useActiveProject();
  const projectId               = project?.id;
  const { setTsData, tsData, tsCol } = useAppStore();
  const [tab, setTab]           = useState("colloc");
  const [nInterior, setNInterior] = useState(2000);
  const [nBoundary, setNBoundary] = useState(400);
  const [collocPts, setCollocPts] = useState<{ interior: number[][]; boundary: number[][] } | null>(null);
  const [selectedCsvCol, setSelectedCsvCol] = useState("");
  const [csvData, setCsvData]   = useState<{ columns: string[]; data: Record<string, number[]>; rows: number } | null>(null);
  const qc = useQueryClient();

  const { data: files, isLoading: filesLoading } = useQuery({
    queryKey: ["files", projectId],
    queryFn:  () => getFiles(projectId),
    enabled:  !!projectId,
  });

  const uploadMut = useMutation({
    mutationFn: (file: File) => uploadFile(file, projectId),
    onSuccess:  () => { qc.invalidateQueries({ queryKey: ["files", projectId] }); toast.success("File uploaded"); },
    onError:    (e: Error) => toast.error(e.message),
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteFile(id),
    onSuccess:  () => { qc.invalidateQueries({ queryKey: ["files", projectId] }); toast.success("File deleted"); },
  });

  const onDrop = useCallback((accepted: File[]) => {
    accepted.forEach((f) => uploadMut.mutate(f));
  }, [uploadMut]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "model/stl": [".stl"], "text/csv": [".csv"], "application/octet-stream": [".npy"] },
    multiple: true,
  });

  const generateColloc = () => {
    if (!project) { toast.error("No active project"); return; }
    const domain = project.problem_spec?.domain ?? { x: [0, 1], y: [0, 1] };
    const keys   = Object.keys(domain);
    const lo     = keys.map((k) => domain[k][0]);
    const hi     = keys.map((k) => domain[k][1]);

    const interior: number[][] = Array.from({ length: nInterior }, () =>
      lo.map((l, i) => l + Math.random() * (hi[i] - l))
    );
    const boundary: number[][] = [];
    for (let di = 0; di < keys.length; di++) {
      for (const val of [lo[di], hi[di]]) {
        for (let k = 0; k < Math.floor(nBoundary / (2 * keys.length)); k++) {
          const pt = lo.map((l, i) => l + Math.random() * (hi[i] - l));
          pt[di] = val;
          boundary.push(pt);
        }
      }
    }
    setCollocPts({ interior, boundary });
    toast.success(`Generated ${interior.length} interior + ${boundary.length} boundary points`);
  };

  const loadCsvData = async (file: UploadedFile) => {
    try {
      const d = await getFileData(file.id);
      setCsvData(d);
      if (d.columns.length > 0) setSelectedCsvCol(d.columns[0]);
    } catch (e: unknown) {
      toast.error((e as Error).message);
    }
  };

  const setTsFromCsv = () => {
    if (!csvData || !selectedCsvCol) return;
    const col = csvData.data[selectedCsvCol];
    if (!col) return;
    const nums = col.filter((v) => typeof v === "number" && !isNaN(v));
    setTsData(nums, selectedCsvCol);
    toast.success(`Set timeseries: ${nums.length} samples from "${selectedCsvCol}"`);
  };

  const TABS = [
    { id: "colloc", label: "Collocation Points" },
    { id: "files",  label: "Files & Geometry"   },
    { id: "ts",     label: "Timeseries Data"     },
  ];

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-text">Data & Geometry</h1>
        <p className="text-muted text-sm mt-1">
          Generate collocation points, upload STL geometry, and load CSV data.
        </p>
      </div>

      <TabsUnderline tabs={TABS} active={tab} onChange={setTab} />

      {/* ── Collocation Points ─────────────────────────────────────── */}
      {tab === "colloc" && (
        <div className="space-y-4">
          <Card title="Generate Collocation Points" subtitle="Random sampling of the problem domain for PINN training">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
              <Slider
                label="Interior points"
                value={nInterior}
                onChange={setNInterior}
                min={100}
                max={10000}
                step={100}
              />
              <Slider
                label="Boundary points"
                value={nBoundary}
                onChange={setNBoundary}
                min={50}
                max={2000}
                step={50}
              />
            </div>
            {project && (
              <div className="mb-4 p-3 bg-surface2 rounded-lg border border-border">
                <div className="text-xs text-muted mb-1">Domain</div>
                <div className="flex flex-wrap gap-3 font-mono text-xs">
                  {Object.entries(project.problem_spec?.domain ?? {}).map(([k, v]) => (
                    <span key={k} className="text-secondary">
                      {k} ∈ [{v[0]}, {v[1]}]
                    </span>
                  ))}
                </div>
              </div>
            )}
            <button className="btn-primary" onClick={generateColloc}>
              <RefreshCw size={14} /> Generate
            </button>
          </Card>

          {collocPts && (
            <Card
              title="Collocation Points"
              subtitle={`${collocPts.interior.length} interior + ${collocPts.boundary.length} boundary`}
            >
              <Plot
                data={[
                  {
                    x: collocPts.interior.map((p) => p[0]),
                    y: collocPts.interior.map((p) => p[1]),
                    mode: "markers",
                    marker: { size: 2, color: "#4ECDC4", opacity: 0.5 },
                    name: "Interior",
                    type: "scatter",
                  },
                  {
                    x: collocPts.boundary.map((p) => p[0]),
                    y: collocPts.boundary.map((p) => p[1]),
                    mode: "markers",
                    marker: { size: 4, color: "#FF6B35" },
                    name: "Boundary",
                    type: "scatter",
                  },
                ]}
                layout={{
                  paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                  font: { color: "#8892a4" }, height: 360,
                  xaxis: { gridcolor: "#2a3a5c" }, yaxis: { gridcolor: "#2a3a5c" },
                  legend: { bgcolor: "transparent" },
                  margin: { l: 40, r: 20, t: 10, b: 40 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
                useResizeHandler
              />
              <div className="mt-3 flex gap-2">
                <button
                  className="btn-secondary text-xs"
                  onClick={() => {
                    const csv = ["x,y", ...collocPts.interior.map((p) => p.join(","))].join("\n");
                    const a = document.createElement("a");
                    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
                    a.download = "interior_points.csv"; a.click();
                  }}
                >
                  Download interior CSV
                </button>
                <button
                  className="btn-secondary text-xs"
                  onClick={() => {
                    const csv = ["x,y", ...collocPts.boundary.map((p) => p.join(","))].join("\n");
                    const a = document.createElement("a");
                    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
                    a.download = "boundary_points.csv"; a.click();
                  }}
                >
                  Download boundary CSV
                </button>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Files & Geometry ────────────────────────────────────────── */}
      {tab === "files" && (
        <div className="space-y-4">
          <Card title="Upload Files" subtitle="STL geometry, CSV data, or NumPy arrays">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
                          ${isDragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}
            >
              <input {...getInputProps()} />
              <Upload size={36} className="mx-auto text-muted mb-3" />
              <p className="text-muted text-sm">
                {isDragActive ? "Drop files here…" : "Drag & drop .stl, .csv, .npy files, or click to browse"}
              </p>
              {uploadMut.isPending && (
                <div className="mt-2 flex items-center justify-center gap-2">
                  <Spinner size="sm" />
                  <span className="text-muted text-xs">Uploading…</span>
                </div>
              )}
            </div>
          </Card>

          {filesLoading ? (
            <div className="flex justify-center py-8"><Spinner /></div>
          ) : files && files.length > 0 ? (
            <Card title="Uploaded Files">
              <div className="space-y-2">
                {files.map((f) => (
                  <div
                    key={f.id}
                    className="flex items-center gap-3 p-3 bg-surface2 rounded-lg border border-border"
                  >
                    <FileIcon type={f.file_type} />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-text truncate">{f.name}</div>
                      <div className="text-xs text-muted">
                        {f.file_type.toUpperCase()} ·{" "}
                        {f.file_type === "stl" && `${f.meta.n_triangles ?? "?"} triangles`}
                        {f.file_type === "csv" && `${f.meta.rows ?? "?"} rows × ${Array.isArray(f.meta.columns) ? f.meta.columns.length : "?"} cols`}
                        {f.file_type === "npy" && `shape: ${JSON.stringify(f.meta.shape ?? [])}`}
                      </div>
                    </div>
                    {f.file_type === "csv" && (
                      <button
                        className="btn-secondary text-xs py-1"
                        onClick={() => loadCsvData(f)}
                      >
                        Load
                      </button>
                    )}
                    <button
                      className="btn-ghost text-error hover:text-error p-1"
                      onClick={() => deleteMut.mutate(f.id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </Card>
          ) : (
            <div className="text-center py-8 text-muted text-sm">No files uploaded yet</div>
          )}
        </div>
      )}

      {/* ── Timeseries Data ─────────────────────────────────────────── */}
      {tab === "ts" && (
        <div className="space-y-4">
          {tsData ? (
            <Card
              title={`Timeseries: ${tsCol}`}
              subtitle={`${tsData.length} samples`}
              action={
                <button className="btn-ghost text-xs text-error" onClick={() => useAppStore.getState().clearTsData()}>
                  Clear
                </button>
              }
            >
              <Plot
                data={[{
                  y: tsData, mode: "lines",
                  line: { color: "#4ECDC4", width: 1.5 },
                  type: "scatter",
                }]}
                layout={{
                  paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                  font: { color: "#8892a4" }, height: 250,
                  xaxis: { title: "t", gridcolor: "#2a3a5c" },
                  yaxis: { title: tsCol ?? "value", gridcolor: "#2a3a5c" },
                  margin: { l: 50, r: 20, t: 10, b: 40 },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
                useResizeHandler
              />
            </Card>
          ) : (
            <Card title="Load Timeseries from CSV">
              <p className="text-muted text-sm mb-4">
                Upload a CSV file in the <strong>Files</strong> tab, then click "Load" to preview and select a column.
              </p>
            </Card>
          )}

          {csvData && (
            <Card title="CSV Preview" subtitle={`${csvData.rows} rows · ${csvData.columns.length} columns`}>
              <div className="flex items-end gap-3 mb-4">
                <Select
                  label="Select column"
                  value={selectedCsvCol}
                  onChange={(e) => setSelectedCsvCol(e.target.value)}
                  options={csvData.columns.map((c) => ({ value: c, label: c }))}
                  className="max-w-xs"
                />
                <button className="btn-primary" onClick={setTsFromCsv}>
                  <Database size={14} /> Use as timeseries
                </button>
              </div>
              {selectedCsvCol && csvData.data[selectedCsvCol] && (
                <Plot
                  data={[{
                    y: csvData.data[selectedCsvCol] as number[],
                    mode: "lines", type: "scatter",
                    line: { color: "#FF6B35", width: 1.5 },
                  }]}
                  layout={{
                    paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                    font: { color: "#8892a4" }, height: 200,
                    xaxis: { gridcolor: "#2a3a5c" },
                    yaxis: { title: selectedCsvCol, gridcolor: "#2a3a5c" },
                    margin: { l: 50, r: 20, t: 10, b: 40 },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%" }}
                  useResizeHandler
                />
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

function FileIcon({ type }: { type: string }) {
  const icons: Record<string, JSX.Element> = {
    stl: <Box size={18} className="text-primary" />,
    csv: <FileText size={18} className="text-secondary" />,
    npy: <Database size={18} className="text-accent" />,
  };
  return icons[type] ?? <FileText size={18} className="text-muted" />;
}
