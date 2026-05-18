import React, { useEffect, useState } from 'react'
import clsx from 'clsx'
import toast from 'react-hot-toast'
import { api, type ProblemItem } from '../../api'
import { useStore } from '../../store'

const FAMILY_COLORS: Record<string, string> = {
  fluid:      'badge-blue',
  thermal:    'badge-amber',
  structural: 'badge-green',
  wave:       'badge-red',
  diffusion:  'badge-blue',
  finance:    'badge-green',
  biological: 'badge-amber',
  generic:    'badge-blue',
}

export function ProblemStep() {
  const { selectedProblem, setSelectedProblem, setStep } = useStore()
  const [problems, setProblems] = useState<ProblemItem[]>([])
  const [search, setSearch] = useState('')
  const [filterFamily, setFilterFamily] = useState<string>('all')
  const [showCustom, setShowCustom] = useState(false)
  const [loading, setLoading] = useState(true)

  // Custom form state
  const [customName, setCustomName]   = useState('')
  const [customEqs,  setCustomEqs]    = useState('')
  const [customBCs,  setCustomBCs]    = useState('')
  const [customDim,  setCustomDim]    = useState('2')

  useEffect(() => {
    api.problems.list()
      .then((r) => setProblems(r.data))
      .catch(() => toast.error('Failed to load problems'))
      .finally(() => setLoading(false))
  }, [])

  const families = ['all', ...Array.from(new Set(problems.map((p) => p.family))).sort()]

  const filtered = problems.filter((p) => {
    const matchSearch = p.name.includes(search) || p.description.toLowerCase().includes(search.toLowerCase())
    const matchFamily = filterFamily === 'all' || p.family === filterFamily
    return matchSearch && matchFamily
  })

  async function handleCustomSubmit() {
    const eqs = customEqs.split('\n').map((e) => e.trim()).filter(Boolean)
    const bcs  = customBCs.split('\n').map((b) => b.trim()).filter(Boolean)
    const dim  = parseInt(customDim, 10)
    const bounds: Record<string, number[]> = {}
    const axes = ['x', 'y', 'z', 't'].slice(0, dim)
    axes.forEach((a) => (bounds[a] = [0, 1]))

    const body = {
      name: customName || 'Custom Problem',
      equations: eqs,
      boundary_conditions: bcs.map((b) => ({ kind: 'dirichlet', location: b, value: 0 })),
      domain_bounds: bounds,
      dim,
    }

    try {
      await api.problems.validate(body)
      const { setCustomProblem } = useStore.getState()
      setCustomProblem(body)
      toast.success('Custom problem configured')
      setStep('geometry')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail?.join('\n') ?? 'Validation failed')
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Select a Problem</h1>
        <p className="text-gray-400 text-sm mt-1">
          Choose a pre-defined physics problem or define your own equations.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        <button className={clsx('btn', !showCustom ? 'btn-primary' : 'btn-ghost')} onClick={() => setShowCustom(false)}>
          Preset Problems
        </button>
        <button className={clsx('btn', showCustom ? 'btn-primary' : 'btn-ghost')} onClick={() => setShowCustom(true)}>
          Custom Problem
        </button>
      </div>

      {!showCustom ? (
        <>
          {/* Filters */}
          <div className="flex gap-3 flex-wrap">
            <input
              className="input max-w-xs"
              placeholder="Search problems…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select className="input max-w-[160px]" value={filterFamily} onChange={(e) => setFilterFamily(e.target.value)}>
              {families.map((f) => <option key={f} value={f}>{f === 'all' ? 'All families' : f}</option>)}
            </select>
          </div>

          {/* Grid */}
          {loading ? (
            <div className="text-gray-500 text-sm">Loading problems…</div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {filtered.map((p) => (
                <button
                  key={p.name}
                  onClick={() => setSelectedProblem(p)}
                  className={clsx(
                    'text-left p-4 rounded-xl border transition-all',
                    selectedProblem?.name === p.name
                      ? 'border-brand-500 bg-brand-900/30 ring-1 ring-brand-500'
                      : 'border-gray-800 bg-gray-900 hover:border-gray-600',
                  )}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="text-sm font-medium text-gray-100 break-all">{p.name}</span>
                    <span className={clsx('badge shrink-0', FAMILY_COLORS[p.family] ?? 'badge-blue')}>{p.family}</span>
                  </div>
                  <p className="text-xs text-gray-400 line-clamp-2">{p.description}</p>
                  {p.time_dependent && <span className="mt-2 inline-block badge badge-amber text-[10px]">time-dependent</span>}
                </button>
              ))}
              {filtered.length === 0 && (
                <div className="col-span-3 text-gray-500 text-sm py-8 text-center">No problems match your filters.</div>
              )}
            </div>
          )}

          {selectedProblem && (
            <div className="card flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-gray-100">Selected: <span className="text-brand-400">{selectedProblem.name}</span></div>
                <div className="text-xs text-gray-400 mt-0.5">Recommended models: {selectedProblem.recommended_models.join(', ')}</div>
              </div>
              <button className="btn-primary" onClick={() => setStep('geometry')}>Continue →</button>
            </div>
          )}
        </>
      ) : (
        /* Custom problem form */
        <div className="card space-y-4 max-w-2xl">
          <h2 className="text-base font-semibold text-gray-200">Define Custom Problem</h2>

          <div>
            <label className="label">Problem Name</label>
            <input className="input" placeholder="e.g. My Diffusion Problem" value={customName} onChange={(e) => setCustomName(e.target.value)} />
          </div>

          <div>
            <label className="label">Spatial Dimension</label>
            <select className="input max-w-[120px]" value={customDim} onChange={(e) => setCustomDim(e.target.value)}>
              {[1,2,3].map((d) => <option key={d} value={d}>{d}D</option>)}
            </select>
          </div>

          <div>
            <label className="label">PDE Equations (one per line, e.g. <code className="font-mono text-brand-400">u_xx + u_yy = 0</code>)</label>
            <textarea
              className="input h-28 resize-none font-mono text-xs"
              placeholder="u_xx + u_yy + f = 0&#10;v_t + u*v_x = nu*v_xx"
              value={customEqs}
              onChange={(e) => setCustomEqs(e.target.value)}
            />
          </div>

          <div>
            <label className="label">Boundary Conditions (one per line, location description)</label>
            <textarea
              className="input h-20 resize-none text-xs"
              placeholder="x=0 (left wall)&#10;x=1 (right wall)&#10;y=0 (bottom)"
              value={customBCs}
              onChange={(e) => setCustomBCs(e.target.value)}
            />
          </div>

          <button className="btn-primary w-full" onClick={handleCustomSubmit} disabled={!customEqs.trim()}>
            Configure Custom Problem →
          </button>
        </div>
      )}
    </div>
  )
}
