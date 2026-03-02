import React, { useEffect, useMemo, useState } from 'react'

type ModelSpec = {
  name: string
  family: string
  description: string
  tags: string[]
  input_kind: string
  supports_physics_loss: boolean
  expects: string[]
  predicts: string[]
}

async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...opts, headers: { 'Content-Type': 'application/json', ...(opts?.headers || {}) } })
  if (!res.ok) throw new Error(await res.text())
  return await res.json()
}

function MetalHeader() {
  return (
    <div style={{padding:'18px 22px', borderBottom:'1px solid rgba(255,255,255,.08)'}}>
      <div style={{display:'flex', alignItems:'baseline', gap:10}}>
        <div style={{fontSize:18, letterSpacing:.6, fontWeight:800}}>PINNEAPLE</div>
        <div style={{opacity:.75}}>Industrial Physics OS · Celery-powered workloads</div>
      </div>
    </div>
  )
}

function ModelPicker({ models, selected, onChange, multiple=false }:{
  models: ModelSpec[], selected: string[]|string, onChange:(v:any)=>void, multiple?: boolean
}) {
  const [q, setQ] = useState('')
  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase()
    if (!qq) return models
    return models.filter(m => (m.name + ' ' + m.family + ' ' + (m.tags||[]).join(' ')).toLowerCase().includes(qq))
  }, [models, q])

  return (
    <div>
      <input placeholder="Search models (name/family/tags)" value={q} onChange={e=>setQ(e.target.value)} />
      <div style={{height:10}} />
      <select multiple={multiple} size={multiple ? 10 : 1} value={selected as any} onChange={e=>{
        if (!multiple) onChange(e.target.value)
        else onChange(Array.from(e.target.selectedOptions).map(o=>o.value))
      }}>
        {filtered.map(m => (
          <option key={m.name} value={m.name}>
            {m.name} · {m.family}{m.supports_physics_loss ? ' · physics' : ''} · {m.input_kind}
          </option>
        ))}
      </select>
      <div style={{opacity:.7, marginTop:8, fontSize:12}}>
        Showing {filtered.length} of {models.length}
      </div>
    </div>
  )
}

function useJobPolling(jobId: string) {
  const [status, setStatus] = useState<string>('')
  const [logs, setLogs] = useState<string>('')

  useEffect(() => {
    if (!jobId) return
    let stop = false
    const t = setInterval(async () => {
      if (stop) return
      try {
        const j = await api<any>(`/api/jobs/${jobId}`)
        setStatus(j.status)
        const lg = await api<any>(`/api/jobs/${jobId}/logs?tail=200`)
        setLogs(lg.events.map((e:any)=>`[${e.ts}] ${e.level}: ${e.message}`).join('\n'))
      } catch {}
    }, 1500)
    return () => { stop = true; clearInterval(t) }
  }, [jobId])

  return { status, logs }
}

function OperatorTemplates(name: string) {
  const templates: Record<string, any> = {
    deeponet: { branch_dim: 128, trunk_dim: 2, out_dim: 1, hidden: 128, modes: 64 },
    multiscale_deeponet: { branch_dim: 128, trunk_dim: 2, out_dim: 1, hidden: 128, scales: [32,64,128] },
    fno: { in_channels: 1, out_channels: 1, width: 64, modes: 16, layers: 4, use_grid: true },
    gno: { in_dim: 1, out_dim: 1, basis_dim: 64, basis_hidden: 128 },
    uno: { dim: 2, in_channels: 1, out_channels: 1, mesh_mode: "grid", width_grid: 64, levels_grid: 3, depth_grid: 2 },
    pino: {
      operator: { name: "uno", kwargs: { dim: 2, in_channels: 1, out_channels: 1, mesh_mode: "grid" } },
      lambda_pde: 1.0, lambda_bc: 1.0, lambda_ic: 1.0, lambda_physics: 1.0
    },
  }
  return templates[name] || {}
}

function QueueRow({queue, setQueue, useCache, setUseCache, onRun}:{queue:any,setQueue:any,useCache:any,setUseCache:any,onRun:()=>void}) {
  return (
    <div style={{display:'flex', gap:10, alignItems:'center'}}>
      <select value={queue} onChange={e=>setQueue(e.target.value as any)} style={{width:140}}>
        <option value="cpu">CPU queue</option>
        <option value="gpu">GPU queue</option>
      </select>
      <label style={{display:'flex', gap:8, alignItems:'center', opacity:.85}}>
        <input type="checkbox" checked={useCache} onChange={e=>setUseCache(e.target.checked)} style={{width:16}} />
        Cache
      </label>
      <button className="btn" onClick={onRun}>Run</button>
    </div>
  )
}

function VerticalA({ operators }:{ operators: ModelSpec[] }) {
  const [queue, setQueue] = useState<'cpu'|'gpu'>('cpu')
  const [useCache, setUseCache] = useState(true)
  const [model, setModel] = useState<string>(operators[0]?.name || 'deeponet')
  const [jobId, setJobId] = useState<string>('')

  const [config, setConfig] = useState<string>(() => JSON.stringify({
    model: operators[0]?.name || "deeponet",
    model_kwargs: OperatorTemplates(operators[0]?.name || "deeponet"),
    dataset: { batch_size: 8, H: 96, W: 96, L: 256, branch_dim: 128, n_coords: 1024, n_points: 2048 },
    geometry_params: { cx: 0.5, cy: 0.5, r: 0.30, ex: 0.55, ey: 0.45, a: 0.22, b: 0.12, theta: 0.35, k: 40.0 },
    solver_cfg: { name: "fdm", equation: "heat2d" },
    epochs: 250,
    lr: 0.001,
    device: "cpu",
    config_name: ""
  }, null, 2))

  const { status, logs } = useJobPolling(jobId)

  // --- Geometry Lab (STEP → mesh → point cloud)
  const [stepFile, setStepFile] = useState<File|null>(null)
  const [geomId, setGeomId] = useState<string>('')
  const [meshId, setMeshId] = useState<string>('')
  const [pcId, setPcId] = useState<string>('')
  const [meshSize, setMeshSize] = useState<number>(0.02)
  const [pcN, setPcN] = useState<number>(8192)
  const [geomInfo, setGeomInfo] = useState<string>('')

  useEffect(() => {
    try {
      const obj = JSON.parse(config)
      obj.model = model
      if (!obj.model_kwargs || Object.keys(obj.model_kwargs).length === 0) {
        obj.model_kwargs = OperatorTemplates(model)
      }
      setConfig(JSON.stringify(obj, null, 2))
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model])

  async function start() {
    const cfg = JSON.parse(config)
    cfg.model = model
    const out = await api<{job_id:string, cached:boolean}>('/api/jobs/start', {
      method:'POST',
      body: JSON.stringify({ vertical: "vertical_a", config: cfg, queue, use_cache: useCache })
    })
    setJobId(out.job_id)
  }

  async function uploadStep() {
    if (!stepFile) return
    const fd = new FormData()
    fd.append('file', stepFile)
    const res = await fetch('/api/geom/step/upload', { method:'POST', body: fd })
    if (!res.ok) throw new Error(await res.text())
    const j = await res.json()
    setGeomId(j.geom_id)
    setMeshId('')
    setPcId('')
    setGeomInfo(`Uploaded: ${j.filename} (geom_id=${j.geom_id})`)
  }

  async function buildMesh() {
    if (!geomId) return
    const j = await api<any>('/api/geom/step/mesh', { method:'POST', body: JSON.stringify({ geom_id: geomId, mesh_size: meshSize, dim: 3 }) })
    setMeshId(j.mesh_id)
    setPcId('')
    setGeomInfo(`Mesh: points=${j.n_points}, cells=${j.n_cells}, types=${(j.cell_types||[]).join(', ')}`)
  }

  async function buildPointCloud() {
    if (!meshId) return
    const j = await api<any>('/api/geom/step/pointcloud', { method:'POST', body: JSON.stringify({ mesh_id: meshId, n: pcN, seed: 0 }) })
    setPcId(j.pc_id)
    setGeomInfo(`PointCloud: n=${j.n} | bounds=[${j.bounds_min.map((x:number)=>x.toFixed(3)).join(', ')}] → [${j.bounds_max.map((x:number)=>x.toFixed(3)).join(', ')}]`)
  }

  return (
    <div className="panel" style={{display:'grid', gap:14}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12}}>
        <div style={{fontWeight:800}}>Vertical A — Surrogate Engineering Platform (SDF geometry + solver hook)</div>
        <QueueRow queue={queue} setQueue={setQueue} useCache={useCache} setUseCache={setUseCache} onRun={start} />
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:14}}>
        <div>
          <div style={{opacity:.85, marginBottom:8}}>Neural Operator model</div>
          <ModelPicker models={operators} selected={model} onChange={setModel} multiple={false} />
          <div style={{opacity:.7, fontSize:12, marginTop:10}}>
            Pipeline: SDF domain → marching squares boundary → build u_t → solver step → train operator u_t→u_{'{'}t+1{'}'}.
          </div>
        </div>
        <div>
          <div style={{opacity:.85, marginBottom:8}}>Config (JSON)</div>
          <textarea rows={14} value={config} onChange={e=>setConfig(e.target.value)} />
        </div>
      </div>

      <div style={{borderTop:'1px solid rgba(255,255,255,.08)', paddingTop:12}}>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12}}>
          <div style={{fontWeight:800}}>Geometry Lab (STEP → mesh → point cloud)</div>
          <div style={{opacity:.75, fontSize:12}}>Use for multi-geometry sweeps + operator training datasets</div>
        </div>

        <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:14, marginTop:10}}>
          <div style={{display:'grid', gap:10}}>
            <input type="file" accept=".step,.stp" onChange={e=>setStepFile(e.target.files?.[0] || null)} />
            <div style={{display:'flex', gap:10, alignItems:'center'}}>
              <button className="btn" onClick={uploadStep} disabled={!stepFile}>Upload STEP</button>
              <div style={{opacity:.8, fontSize:12}}>geom_id: {geomId || '-'}</div>
            </div>

            <div style={{display:'flex', gap:10, alignItems:'center'}}>
              <label style={{opacity:.85}}>mesh_size</label>
              <input type="number" step="0.005" value={meshSize} onChange={e=>setMeshSize(parseFloat(e.target.value||'0.02'))} style={{width:110}} />
              <button className="btn" onClick={buildMesh} disabled={!geomId}>Build mesh</button>
              {meshId && <a href={`/api/geom/step/mesh/${meshId}.msh`} target="_blank" rel="noreferrer">download .msh</a>}
            </div>

            <div style={{display:'flex', gap:10, alignItems:'center'}}>
              <label style={{opacity:.85}}>n_points</label>
              <input type="number" step="1024" value={pcN} onChange={e=>setPcN(parseInt(e.target.value||'8192'))} style={{width:110}} />
              <button className="btn" onClick={buildPointCloud} disabled={!meshId}>Sample point cloud</button>
              {pcId && <a href={`/api/geom/step/pointcloud/${pcId}.npz`} target="_blank" rel="noreferrer">download .npz</a>}
            </div>
          </div>

          <div>
            <div style={{opacity:.85, marginBottom:8}}>Status</div>
            <textarea rows={7} value={geomInfo} readOnly />
            <div style={{opacity:.7, fontSize:12, marginTop:8}}>
              Tip: once you have point clouds, you can train (i) a VAE for latent geometry sweeps, or (ii) a GNN encoder to condition a neural operator.
            </div>
          </div>
        </div>
      </div>

      <div style={{display:'flex', gap:12, alignItems:'center', opacity:.9}}>
        <div>Job:</div>
        <div style={{fontFamily:'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas', fontSize:12}}>{jobId || '-'}</div>
        <div style={{marginLeft:18}}>Status:</div>
        <div>{status || '-'}</div>
        {jobId && (
          <a href={`/api/jobs/${jobId}/artifacts.zip`} target="_blank" rel="noreferrer" style={{marginLeft:'auto'}}>
            Download artifacts.zip
          </a>
        )}
      </div>

      <div>
        <div style={{opacity:.85, marginBottom:8}}>Logs</div>
        <textarea rows={10} value={logs} readOnly />
      </div>
    </div>
  )
}

function VerticalB() {
  const [queue, setQueue] = useState<'cpu'|'gpu'>('cpu')
  const [useCache, setUseCache] = useState(true)
  const [jobId, setJobId] = useState<string>('')

  const [config, setConfig] = useState<string>(() => JSON.stringify({
    T: 2000,
    dt: 0.01,
    note: "MVP: replace with pinneaple_timeseries + pinneaple_pdb continuous recalibration."
  }, null, 2))

  const { status, logs } = useJobPolling(jobId)

  async function start() {
    const cfg = JSON.parse(config)
    const out = await api<{job_id:string, cached:boolean}>('/api/jobs/start', {
      method:'POST',
      body: JSON.stringify({ vertical: "vertical_b", config: cfg, queue, use_cache: useCache })
    })
    setJobId(out.job_id)
  }

  return (
    <div className="panel" style={{display:'grid', gap:14}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12}}>
        <div style={{fontWeight:800}}>Vertical B — Digital Twin Builder (streaming MVP)</div>
        <QueueRow queue={queue} setQueue={setQueue} useCache={useCache} setUseCache={setUseCache} onRun={start} />
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:14}}>
        <div style={{opacity:.85}}>
          This MVP generates a synthetic sensor stream and produces a baseline forecast + artifact plot.
          In production, connect your streaming ingestion to <b>pinneaple_timeseries</b> and persist in <b>pinneaple_pdb</b>.
        </div>
        <div>
          <div style={{opacity:.85, marginBottom:8}}>Config (JSON)</div>
          <textarea rows={12} value={config} onChange={e=>setConfig(e.target.value)} />
        </div>
      </div>

      <div style={{display:'flex', gap:12, alignItems:'center', opacity:.9}}>
        <div>Job:</div>
        <div style={{fontFamily:'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas', fontSize:12}}>{jobId || '-'}</div>
        <div style={{marginLeft:18}}>Status:</div>
        <div>{status || '-'}</div>
        {jobId && (
          <a href={`/api/jobs/${jobId}/artifacts.zip`} target="_blank" rel="noreferrer" style={{marginLeft:'auto'}}>
            Download artifacts.zip
          </a>
        )}
      </div>

      <div>
        <div style={{opacity:.85, marginBottom:8}}>Logs</div>
        <textarea rows={10} value={logs} readOnly />
      </div>
    </div>
  )
}

function VerticalC({ models }:{ models: ModelSpec[] }) {
  const [queue, setQueue] = useState<'cpu'|'gpu'>('cpu')
  const [useCache, setUseCache] = useState(true)
  const [selected, setSelected] = useState<string[]>(models.slice(0, 5).map(m=>m.name))
  const [jobId, setJobId] = useState<string>('')

  const [config, setConfig] = useState<string>(() => JSON.stringify({
    config_name: "",
    models: selected,
    parallelism: "process",
    gpus: "auto",
    ddp_per_model: false,
    problem_spec: {},
    geometry: {},
    solver_cfg: { name: "fdm", equation: "heat2d" }
  }, null, 2))

  const { status, logs } = useJobPolling(jobId)

  async function start() {
    const cfg = JSON.parse(config)
    cfg.models = selected
    const out = await api<{job_id:string, cached:boolean}>('/api/jobs/start', {
      method:'POST',
      body: JSON.stringify({ vertical: "vertical_c", config: cfg, queue, use_cache: useCache })
    })
    setJobId(out.job_id)
  }

  return (
    <div className="panel" style={{display:'grid', gap:14}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12}}>
        <div style={{fontWeight:800}}>Vertical C — Scientific Benchmark Arena (All Models)</div>
        <QueueRow queue={queue} setQueue={setQueue} useCache={useCache} setUseCache={setUseCache} onRun={start} />
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:14}}>
        <div>
          <div style={{opacity:.85, marginBottom:8}}>Benchmark models</div>
          <ModelPicker models={models} selected={selected} onChange={setSelected} multiple={true} />
        </div>
        <div>
          <div style={{opacity:.85, marginBottom:8}}>Config (JSON)</div>
          <textarea rows={14} value={config} onChange={e=>setConfig(e.target.value)} />
        </div>
      </div>

      <div style={{display:'flex', gap:12, alignItems:'center', opacity:.9}}>
        <div>Job:</div>
        <div style={{fontFamily:'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas', fontSize:12}}>{jobId || '-'}</div>
        <div style={{marginLeft:18}}>Status:</div>
        <div>{status || '-'}</div>
        {jobId && (
          <a href={`/api/jobs/${jobId}/artifacts.zip`} target="_blank" rel="noreferrer" style={{marginLeft:'auto'}}>
            Download artifacts.zip
          </a>
        )}
      </div>

      <div>
        <div style={{opacity:.85, marginBottom:8}}>Logs</div>
        <textarea rows={10} value={logs} readOnly />
      </div>
    </div>
  )
}

function VerticalD() {
  const [queue, setQueue] = useState<'cpu'|'gpu'>('cpu')
  const [useCache, setUseCache] = useState(true)
  const [jobId, setJobId] = useState<string>('')

  const [config, setConfig] = useState<string>(() => JSON.stringify({
    T: 2000,
    dt: 0.02,
    ramp_limit: 0.08,
    lambda_physics: 1.0,
    note: "MVP: replace with pinneaple_timeseries + pinneaple_pinn physics loss (PDE/constraints)."
  }, null, 2))

  const { status, logs } = useJobPolling(jobId)

  async function start() {
    const cfg = JSON.parse(config)
    const out = await api<{job_id:string, cached:boolean}>('/api/jobs/start', {
      method:'POST',
      body: JSON.stringify({ vertical: "vertical_d", config: cfg, queue, use_cache: useCache })
    })
    setJobId(out.job_id)
  }

  return (
    <div className="panel" style={{display:'grid', gap:14}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12}}>
        <div style={{fontWeight:800}}>Vertical D — Physics + Time Series Fusion (MVP)</div>
        <QueueRow queue={queue} setQueue={setQueue} useCache={useCache} setUseCache={setUseCache} onRun={start} />
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:14}}>
        <div style={{opacity:.85}}>
          This MVP adds a physics-like penalty (ramp-rate constraint) on the forecast.
          In production, plug a TS model and build physics losses using your <b>pinneaple_pinn</b> loss builder.
        </div>
        <div>
          <div style={{opacity:.85, marginBottom:8}}>Config (JSON)</div>
          <textarea rows={12} value={config} onChange={e=>setConfig(e.target.value)} />
        </div>
      </div>

      <div style={{display:'flex', gap:12, alignItems:'center', opacity:.9}}>
        <div>Job:</div>
        <div style={{fontFamily:'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas', fontSize:12}}>{jobId || '-'}</div>
        <div style={{marginLeft:18}}>Status:</div>
        <div>{status || '-'}</div>
        {jobId && (
          <a href={`/api/jobs/${jobId}/artifacts.zip`} target="_blank" rel="noreferrer" style={{marginLeft:'auto'}}>
            Download artifacts.zip
          </a>
        )}
      </div>

      <div>
        <div style={{opacity:.85, marginBottom:8}}>Logs</div>
        <textarea rows={10} value={logs} readOnly />
      </div>
    </div>
  )
}

function JobsPanel() {
  const [jobs, setJobs] = useState<any[]>([])
  useEffect(() => {
    let stop = false
    const t = setInterval(async () => {
      if (stop) return
      try {
        const out = await api<any>(`/api/jobs/recent?limit=50`)
        setJobs(out)
      } catch {}
    }, 2000)
    return () => { stop = true; clearInterval(t) }
  }, [])
  return (
    <div className="panel" style={{display:'grid', gap:10}}>
      <div style={{fontWeight:800}}>Recent Jobs</div>
      <div style={{display:'grid', gap:6}}>
        {jobs.map(j => (
          <div key={j.job_id} style={{display:'grid', gridTemplateColumns:'1fr 120px 90px 140px', gap:10, alignItems:'center', padding:'8px 10px', border:'1px solid rgba(255,255,255,.08)', borderRadius:12}}>
            <div style={{fontFamily:'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas', fontSize:12, opacity:.95}}>{j.job_id}</div>
            <div style={{opacity:.9}}>{j.vertical}</div>
            <div style={{opacity:.9}}>{j.status}</div>
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:10}}>
              <a href={`/api/jobs/${j.job_id}/artifacts.zip`} target="_blank" rel="noreferrer">zip</a>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [operators, setOperators] = useState<ModelSpec[]>([])
  const [tab, setTab] = useState<'vertical_a'|'vertical_b'|'vertical_c'|'vertical_d'|'jobs'>('vertical_a')

  useEffect(() => {
    api<{models: ModelSpec[]}>('/api/models').then(out => setModels(out.models || [])).catch(()=>setModels([]))
    api<{models: ModelSpec[]}>('/api/models?family=neural_operators').then(out => setOperators(out.models || [])).catch(()=>setOperators([]))
  }, [])

  return (
    <div>
      <MetalHeader />
      <div style={{maxWidth:1200, margin:'0 auto', padding:22, display:'grid', gap:16}}>
        <div style={{display:'flex', gap:10, flexWrap:'wrap'}}>
          <button className={tab==='vertical_a' ? 'btn' : 'btn2'} onClick={()=>setTab('vertical_a')}>Vertical A</button>
          <button className={tab==='vertical_b' ? 'btn' : 'btn2'} onClick={()=>setTab('vertical_b')}>Vertical B</button>
          <button className={tab==='vertical_c' ? 'btn' : 'btn2'} onClick={()=>setTab('vertical_c')}>Vertical C</button>
          <button className={tab==='vertical_d' ? 'btn' : 'btn2'} onClick={()=>setTab('vertical_d')}>Vertical D</button>
          <button className={tab==='jobs' ? 'btn' : 'btn2'} onClick={()=>setTab('jobs')}>Jobs</button>
        </div>

        {tab === 'vertical_a' ? (
          <VerticalA operators={operators} />
        ) : tab === 'vertical_b' ? (
          <VerticalB />
        ) : tab === 'vertical_c' ? (
          <VerticalC models={models} />
        ) : tab === 'vertical_d' ? (
          <VerticalD />
        ) : (
          <JobsPanel />
        )}
      </div>
    </div>
  )
}
