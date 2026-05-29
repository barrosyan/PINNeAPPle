import React from 'react'
import clsx from 'clsx'
import { useStore } from '../../store'

const STRATEGIES = [
  { value: 'lhs',      label: 'Latin Hypercube (LHS)',     desc: 'Good default — space-filling, low discrepancy.' },
  { value: 'sobol',    label: 'Sobol quasi-random',         desc: 'Excellent uniformity, ideal for higher dimensions.' },
  { value: 'halton',   label: 'Halton sequence',            desc: 'Deterministic low-discrepancy sequence.' },
  { value: 'uniform',  label: 'Uniform random',             desc: 'Simple Monte Carlo sampling.' },
  { value: 'grid',     label: 'Structured grid',            desc: 'Regular grid — best for 1-D and 2-D problems.' },
  { value: 'adaptive', label: 'Residual-adaptive',          desc: 'Refines points in high-residual regions (slower).' },
  { value: 'meshfree', label: 'Meshfree (RBF)',             desc: 'Point clouds with RBF interpolation support.' },
]

export function GeometryStep() {
  const { selectedProblem, isCustomProblem, geometry, setGeometry, dataConfig, setDataConfig, setStep } = useStore()
  const problem = selectedProblem

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Geometry & Collocation</h1>
        <p className="text-gray-400 text-sm mt-1">
          Configure geometry (optional) and how collocation points are sampled.
        </p>
      </div>

      {/* Geometry toggle */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-gray-200">Use Geometry</div>
            <div className="text-xs text-gray-400 mt-0.5">Enable to apply a geometric SDF constraint for collocation sampling.</div>
          </div>
          <button
            onClick={() => setGeometry({ useGeometry: !geometry.useGeometry })}
            className={clsx(
              'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
              geometry.useGeometry ? 'bg-brand-600' : 'bg-gray-700',
            )}
          >
            <span className={clsx('inline-block h-4 w-4 rounded-full bg-white shadow transition-transform',
              geometry.useGeometry ? 'translate-x-6' : 'translate-x-1')} />
          </button>
        </div>

        {geometry.useGeometry && (
          <div>
            <label className="label">Domain Name (from pinneaple_geom)</label>
            <input
              className="input"
              placeholder="e.g. channel_2d, lid_driven_cavity_2d"
              value={geometry.domainName}
              onChange={(e) => setGeometry({ domainName: e.target.value })}
            />
            <p className="text-xs text-gray-500 mt-1">Leave blank for default box domain.</p>
          </div>
        )}
      </div>

      {/* Collocation strategy */}
      <div className="card space-y-4">
        <div className="text-sm font-semibold text-gray-200">Collocation Strategy</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {STRATEGIES.map((s) => (
            <button
              key={s.value}
              onClick={() => setGeometry({ collocationStrategy: s.value })}
              className={clsx(
                'text-left p-3 rounded-lg border transition-all',
                geometry.collocationStrategy === s.value
                  ? 'border-brand-500 bg-brand-900/30'
                  : 'border-gray-800 bg-gray-800/50 hover:border-gray-600',
              )}
            >
              <div className="text-sm font-medium text-gray-100">{s.label}</div>
              <div className="text-xs text-gray-400 mt-0.5">{s.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Point counts */}
      <div className="card">
        <div className="text-sm font-semibold text-gray-200 mb-4">Point Counts</div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {[
            { key: 'nInterior', label: 'Interior points' },
            { key: 'nBoundary', label: 'Boundary points' },
            { key: 'nInitial',  label: 'Initial condition points', disabled: !problem?.time_dependent },
          ].map(({ key, label, disabled }) => (
            <div key={key} className={clsx(disabled && 'opacity-40')}>
              <label className="label">{label}</label>
              <input
                type="number"
                className="input"
                disabled={!!disabled}
                value={(geometry as any)[key]}
                onChange={(e) => setGeometry({ [key]: parseInt(e.target.value) || 0 })}
              />
            </div>
          ))}
          <div>
            <label className="label">Random seed</label>
            <input type="number" className="input" value={geometry.seed}
              onChange={(e) => setGeometry({ seed: parseInt(e.target.value) || 0 })} />
          </div>
        </div>
      </div>

      {/* Data / solver config */}
      <div className="card space-y-4">
        <div className="text-sm font-semibold text-gray-200">Data Generation</div>
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            id="use-solver"
            className="h-4 w-4 rounded accent-brand-500"
            checked={dataConfig.useSolver}
            onChange={(e) => setDataConfig({ useSolver: e.target.checked })}
          />
          <label htmlFor="use-solver" className="text-sm text-gray-300">
            Generate reference data with solver (FDM/FEM)
          </label>
        </div>

        {dataConfig.useSolver && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <label className="label">Solver override</label>
              <input className="input" placeholder="auto" value={dataConfig.solverKey}
                onChange={(e) => setDataConfig({ solverKey: e.target.value })} />
            </div>
            <div>
              <label className="label">Snapshots</label>
              <input type="number" className="input" value={dataConfig.nSnapshots}
                onChange={(e) => setDataConfig({ nSnapshots: parseInt(e.target.value) || 1 })} />
            </div>
            <div>
              <label className="label">Grid resolution</label>
              <input type="number" className="input" value={dataConfig.gridResolution}
                onChange={(e) => setDataConfig({ gridResolution: parseInt(e.target.value) || 16 })} />
            </div>
            <div>
              <label className="label">t_end</label>
              <input type="number" step="0.1" className="input" value={dataConfig.tEnd}
                onChange={(e) => setDataConfig({ tEnd: parseFloat(e.target.value) || 1.0 })} />
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button className="btn-ghost" onClick={() => setStep('problem')}>← Back</button>
        <button className="btn-primary" onClick={() => setStep('models')}>Continue →</button>
      </div>
    </div>
  )
}
