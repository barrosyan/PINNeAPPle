import React, { useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../store'

type Tab = 'leaderboard' | 'charts' | 'summary'

export function ResultsStep() {
  const { benchmarkPayload, setStep } = useStore()
  const [tab, setTab] = useState<Tab>('leaderboard')

  if (!benchmarkPayload) {
    return (
      <div className="max-w-4xl mx-auto space-y-4">
        <h1 className="text-2xl font-bold text-gray-100">Results</h1>
        <p className="text-gray-400">No results available yet. Run an experiment first.</p>
        <button className="btn-ghost" onClick={() => setStep('run')}>← Go to Run</button>
      </div>
    )
  }

  const { leaderboard, charts, summary, errors, problem_name, experiment_id, completed_at } = benchmarkPayload

  const CHART_LABELS: Record<string, string> = {
    loss_curves:        'Training Curves',
    metric_comparison:  'Model Comparison',
    time_vs_accuracy:   'Accuracy vs. Cost',
    parameter_count:    'Model Size',
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Benchmark Results</h1>
          <p className="text-gray-400 text-sm mt-1">
            Problem: <span className="text-brand-400">{problem_name}</span> · ID: <span className="font-mono text-xs text-gray-500">{experiment_id}</span>
          </p>
        </div>
        <button className="btn-ghost text-sm" onClick={() => setStep('run')}>← Rerun</button>
      </div>

      {/* Winner banner */}
      {leaderboard.length > 0 && (
        <div className="card border-green-700/50 bg-green-900/10 flex items-center gap-4">
          <div className="text-3xl">🏆</div>
          <div>
            <div className="text-sm font-semibold text-green-300">Best Model</div>
            <div className="text-lg font-bold text-gray-100 font-mono">{leaderboard[0].model}</div>
            <div className="text-xs text-gray-400">
              L2 error: {typeof leaderboard[0].l2_relative === 'number'
                ? leaderboard[0].l2_relative.toExponential(3)
                : 'N/A'}
              {' '} · params: {leaderboard[0].n_params?.toLocaleString()}
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-900 rounded-lg p-1 max-w-sm">
        {(['leaderboard', 'charts', 'summary'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={clsx('flex-1 py-1.5 rounded-md text-sm font-medium transition-all capitalize',
              tab === t ? 'bg-gray-700 text-gray-100' : 'text-gray-500 hover:text-gray-300')}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Leaderboard */}
      {tab === 'leaderboard' && (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-2 px-3 text-gray-400 font-medium">#</th>
                <th className="text-left py-2 px-3 text-gray-400 font-medium">Model</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">L2 Error</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">MSE</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">PDE Residual</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Train Time</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Params</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((row, i) => (
                <tr key={row.model} className={clsx('border-b border-gray-800/50', i === 0 && 'bg-green-900/10')}>
                  <td className="py-2 px-3 text-gray-500">{i + 1}</td>
                  <td className="py-2 px-3 font-mono text-gray-100">{row.model}</td>
                  <td className="py-2 px-3 text-right text-gray-300">
                    {typeof row.l2_relative === 'number' ? row.l2_relative.toExponential(3) : '—'}
                  </td>
                  <td className="py-2 px-3 text-right text-gray-300">
                    {typeof row.mse === 'number' ? row.mse.toExponential(3) : '—'}
                  </td>
                  <td className="py-2 px-3 text-right text-gray-300">
                    {typeof row.pde_residual === 'number' ? row.pde_residual.toExponential(3) : '—'}
                  </td>
                  <td className="py-2 px-3 text-right text-gray-300">
                    {typeof row.train_time_s === 'number' ? `${row.train_time_s.toFixed(1)}s` : '—'}
                  </td>
                  <td className="py-2 px-3 text-right text-gray-300">
                    {typeof row.n_params === 'number' ? row.n_params.toLocaleString() : '—'}
                  </td>
                </tr>
              ))}
              {leaderboard.length === 0 && (
                <tr><td colSpan={7} className="py-8 text-center text-gray-500">No results yet.</td></tr>
              )}
            </tbody>
          </table>

          {Object.keys(errors).length > 0 && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg">
              <div className="text-sm font-medium text-red-400 mb-2">Failed models:</div>
              {Object.entries(errors).map(([name, err]) => (
                <div key={name} className="text-xs text-red-300 font-mono">{name}: {err}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      {tab === 'charts' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.entries(charts).map(([key, b64]) => (
            <div key={key} className="card">
              <div className="text-sm font-medium text-gray-300 mb-3">{CHART_LABELS[key] ?? key}</div>
              <img
                src={`data:image/png;base64,${b64}`}
                alt={key}
                className="w-full rounded-lg"
              />
            </div>
          ))}
          {Object.keys(charts).length === 0 && (
            <div className="col-span-2 text-gray-500 text-sm py-8 text-center">No charts generated.</div>
          )}
        </div>
      )}

      {/* Summary */}
      {tab === 'summary' && (
        <div className="card">
          <pre className="text-xs font-mono text-gray-300 whitespace-pre-wrap leading-relaxed">{summary}</pre>
          <div className="mt-4 text-xs text-gray-600">Completed at: {completed_at}</div>
        </div>
      )}

      {/* Export */}
      <div className="flex justify-end">
        <button
          className="btn-ghost text-xs"
          onClick={() => {
            const blob = new Blob([JSON.stringify(benchmarkPayload, null, 2)], { type: 'application/json' })
            const url  = URL.createObjectURL(blob)
            const a    = document.createElement('a')
            a.href = url; a.download = `pinneaple_${experiment_id}.json`
            a.click(); URL.revokeObjectURL(url)
          }}
        >
          ↓ Export JSON
        </button>
      </div>
    </div>
  )
}
