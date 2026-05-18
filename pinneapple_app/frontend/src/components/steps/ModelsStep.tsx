import React, { useEffect, useState } from 'react'
import clsx from 'clsx'
import toast from 'react-hot-toast'
import { api, type ModelItem } from '../../api'
import { useStore } from '../../store'

const FAMILY_COLORS: Record<string, string> = {
  pinns:            'badge-blue',
  group_b:          'badge-green',
  neural_operators: 'badge-amber',
  graphnn:          'badge-red',
  transformers:     'badge-blue',
  recurrent:        'badge-green',
  physics_aware:    'badge-amber',
  continuous:       'badge-blue',
  benchmarks:       'badge-green',
}

export function ModelsStep() {
  const { selectedProblem, selectedModels, setSelectedModels,
          selectedMetrics, setSelectedMetrics, trainConfig, setTrainConfig, setStep } = useStore()

  const [allModels, setAllModels]       = useState<ModelItem[]>([])
  const [allMetrics, setAllMetrics]     = useState<{ key: string; label: string }[]>([])
  const [filterFamily, setFilterFamily] = useState('all')
  const [search, setSearch]             = useState('')
  const [loading, setLoading]           = useState(true)

  useEffect(() => {
    Promise.all([api.models.list(), api.models.metrics()])
      .then(([mRes, meRes]) => {
        setAllModels(mRes.data)
        setAllMetrics(meRes.data.available)
      })
      .catch(() => toast.error('Failed to load models'))
      .finally(() => setLoading(false))

    // Auto-recommend if problem is selected
    if (selectedProblem && selectedModels.length === 0) {
      api.models.recommend(selectedProblem.family, 5).then((r) => {
        const names = r.data.recommendations
        setAllModels((prev) => {
          const recommended = prev.filter((m) => names.includes(m.name))
          if (recommended.length > 0) setSelectedModels(recommended)
          return prev
        })
      })
    }
  }, [])

  const families = ['all', ...Array.from(new Set(allModels.map((m) => m.family))).sort()]

  const filtered = allModels.filter((m) => {
    const matchSearch = m.name.includes(search) || m.description.toLowerCase().includes(search.toLowerCase())
    const matchFamily = filterFamily === 'all' || m.family === filterFamily
    return matchSearch && matchFamily
  })

  function toggleModel(m: ModelItem) {
    const already = selectedModels.some((s) => s.name === m.name)
    if (already) setSelectedModels(selectedModels.filter((s) => s.name !== m.name))
    else setSelectedModels([...selectedModels, m])
  }

  function toggleMetric(key: string) {
    if (selectedMetrics.includes(key)) setSelectedMetrics(selectedMetrics.filter((k) => k !== key))
    else setSelectedMetrics([...selectedMetrics, key])
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Select Models & Metrics</h1>
        <p className="text-gray-400 text-sm mt-1">
          Choose models to benchmark. {selectedProblem && (
            <span className="text-brand-400">Recommended for <strong>{selectedProblem.family}</strong>: {selectedProblem.recommended_models.join(', ')}</span>
          )}
        </p>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <input className="input max-w-xs" placeholder="Search models…" value={search}
          onChange={(e) => setSearch(e.target.value)} />
        <select className="input max-w-[180px]" value={filterFamily} onChange={(e) => setFilterFamily(e.target.value)}>
          {families.map((f) => <option key={f} value={f}>{f === 'all' ? 'All families' : f}</option>)}
        </select>
        {selectedModels.length > 0 && (
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span className="badge badge-blue">{selectedModels.length} selected</span>
            <button className="text-xs underline text-gray-500 hover:text-gray-300" onClick={() => setSelectedModels([])}>clear</button>
          </div>
        )}
      </div>

      {/* Model grid */}
      {loading ? (
        <div className="text-gray-500 text-sm">Loading models…</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {filtered.map((m) => {
            const selected = selectedModels.some((s) => s.name === m.name)
            return (
              <button
                key={m.name}
                onClick={() => toggleModel(m)}
                className={clsx(
                  'text-left p-3 rounded-xl border transition-all',
                  selected
                    ? 'border-brand-500 bg-brand-900/30 ring-1 ring-brand-500'
                    : 'border-gray-800 bg-gray-900 hover:border-gray-600',
                )}
              >
                <div className="flex items-start justify-between gap-1 mb-1">
                  <span className="text-xs font-semibold text-gray-100 font-mono break-all">{m.name}</span>
                  {selected && <span className="text-brand-400 text-sm shrink-0">✓</span>}
                </div>
                <span className={clsx('badge text-[10px]', FAMILY_COLORS[m.family] ?? 'badge-blue')}>{m.family}</span>
                {m.description && (
                  <p className="text-[11px] text-gray-500 mt-1.5 line-clamp-2">{m.description}</p>
                )}
              </button>
            )
          })}
        </div>
      )}

      {/* Metrics */}
      <div className="card">
        <div className="text-sm font-semibold text-gray-200 mb-3">Evaluation Metrics</div>
        <div className="flex flex-wrap gap-2">
          {allMetrics.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => toggleMetric(key)}
              className={clsx(
                'px-3 py-1.5 rounded-lg border text-xs font-medium transition-all',
                selectedMetrics.includes(key)
                  ? 'border-brand-500 bg-brand-900/30 text-brand-300'
                  : 'border-gray-700 text-gray-400 hover:border-gray-500',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Training config */}
      <div className="card">
        <div className="text-sm font-semibold text-gray-200 mb-4">Training Configuration</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <label className="label">Epochs</label>
            <input type="number" className="input" value={trainConfig.epochs}
              onChange={(e) => setTrainConfig({ epochs: parseInt(e.target.value) || 100 })} />
          </div>
          <div>
            <label className="label">Learning Rate</label>
            <input type="number" step="0.0001" className="input" value={trainConfig.lr}
              onChange={(e) => setTrainConfig({ lr: parseFloat(e.target.value) || 1e-3 })} />
          </div>
          <div>
            <label className="label">Device</label>
            <select className="input" value={trainConfig.device} onChange={(e) => setTrainConfig({ device: e.target.value })}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA (GPU)</option>
              <option value="mps">MPS (Apple)</option>
            </select>
          </div>
          <div>
            <label className="label">Seed</label>
            <input type="number" className="input" value={trainConfig.seed}
              onChange={(e) => setTrainConfig({ seed: parseInt(e.target.value) || 42 })} />
          </div>
        </div>
      </div>

      <div className="flex justify-between">
        <button className="btn-ghost" onClick={() => setStep('geometry')}>← Back</button>
        <button
          className="btn-primary"
          disabled={selectedModels.length === 0}
          onClick={() => setStep('run')}
        >
          Run Benchmark →
        </button>
      </div>
    </div>
  )
}
