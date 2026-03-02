import React from 'react'
import JsonEditor from './JsonEditor'

async function postJSON(path: string, body: any) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config: body })
  })
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `HTTP ${res.status}`)
  }
  return res.json()
}

export default function RunPanel({
  title,
  subtitle,
  endpoint,
  defaultConfig
}: {
  title: string
  subtitle: string
  endpoint: string
  defaultConfig: any
}) {
  const [cfg, setCfg] = React.useState(defaultConfig)
  const [loading, setLoading] = React.useState(false)
  const [err, setErr] = React.useState<string | null>(null)
  const [result, setResult] = React.useState<any>(null)

  const run = async () => {
    setLoading(true)
    setErr(null)
    try {
      const r = await postJSON(endpoint, cfg)
      setResult(r)
    } catch (e: any) {
      setErr(e?.message ?? 'Run failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="metal-card shadow-metal rounded-2xl p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-lg font-semibold">{title}</div>
            <div className="text-sm text-slate-300/80 mt-1">{subtitle}</div>
          </div>
          <button
            onClick={run}
            disabled={loading}
            className="metal-btn rounded-xl px-4 py-2 text-sm font-semibold hover:bg-slate-200/10 disabled:opacity-60"
          >
            {loading ? 'Running…' : 'Run'}
          </button>
        </div>
        {err && <div className="mt-3 text-sm text-rose-300">{err}</div>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <JsonEditor value={cfg} onChange={setCfg} />

        <div className="metal-card rounded-2xl p-3 shadow-metal">
          <div className="text-sm font-semibold mb-2">Output</div>
          {!result && <div className="text-sm text-slate-300/80">Run the module to see results.</div>}
          {result && (
            <div className="space-y-3">
              {result.metrics && (
                <div className="bg-black/20 border border-slate-200/10 rounded-xl p-3">
                  <div className="text-xs uppercase tracking-widest text-slate-300/70">Metrics</div>
                  <pre className="mt-2 text-xs text-slate-100 overflow-auto">{JSON.stringify(result.metrics, null, 2)}</pre>
                </div>
              )}

              {result.ranking && (
                <div className="bg-black/20 border border-slate-200/10 rounded-xl p-3">
                  <div className="text-xs uppercase tracking-widest text-slate-300/70">Ranking</div>
                  <pre className="mt-2 text-xs text-slate-100 overflow-auto">{JSON.stringify(result.ranking, null, 2)}</pre>
                </div>
              )}

              {result.plot_base64_png && (
                <div className="bg-black/20 border border-slate-200/10 rounded-xl p-3">
                  <div className="text-xs uppercase tracking-widest text-slate-300/70">Visualization</div>
                  <img
                    className="mt-2 rounded-xl border border-slate-200/10"
                    src={`data:image/png;base64,${result.plot_base64_png}`}
                    alt="plot"
                  />
                </div>
              )}

              {result.plots && (
                <div className="space-y-3">
                  {Object.entries(result.plots).map(([k, v]: any) => (
                    <div key={k} className="bg-black/20 border border-slate-200/10 rounded-xl p-3">
                      <div className="text-xs uppercase tracking-widest text-slate-300/70">{k}</div>
                      <img
                        className="mt-2 rounded-xl border border-slate-200/10"
                        src={`data:image/png;base64,${v}`}
                        alt={k}
                      />
                    </div>
                  ))}
                </div>
              )}

              <details className="bg-black/20 border border-slate-200/10 rounded-xl p-3">
                <summary className="text-sm cursor-pointer text-slate-200">Raw JSON</summary>
                <pre className="mt-2 text-xs text-slate-100 overflow-auto">{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
