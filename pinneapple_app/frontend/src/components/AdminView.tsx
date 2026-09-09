import React, { useState } from 'react'
import axios from 'axios'
import { useStore } from '../store'

interface ModelSummaryItem {
  problem_id: string
  latest_model_version: string | null
  stage: string | null
  dataset_scenarios: string[]
}

interface AdminSummary {
  registry_root: string
  n_presets: number
  n_architectures: number
  models: ModelSummaryItem[]
}

// A minimal internal curation/status view -- lists every problem_id the
// local artifact registry (pinneapple_registry.ArtifactRegistry) knows
// about, its latest model version + stage, and its dataset scenarios.
// Gated server-side by an X-Admin-Token header (see backend/core/admin_auth.py)
// -- there is no public/anonymous admin access here, by design.
export function AdminView() {
  const setShowAdmin = useStore((s) => s.setShowAdmin)
  const [token, setToken] = useState('')
  const [summary, setSummary] = useState<AdminSummary | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function loadSummary() {
    setLoading(true)
    setError('')
    try {
      const res = await axios.get<AdminSummary>('/api/admin/summary', {
        headers: { 'X-Admin-Token': token },
      })
      setSummary(res.data)
    } catch (e) {
      const msg = axios.isAxiosError(e) ? e.response?.data?.detail ?? e.message : String(e)
      setError(msg)
      setSummary(null)
    } finally {
      setLoading(false)
    }
  }

  async function promote(problemId: string, version: string, stage: string) {
    try {
      await axios.post(
        '/api/admin/models/promote',
        { problem_id: problemId, version, stage },
        { headers: { 'X-Admin-Token': token } },
      )
      loadSummary()
    } catch (e) {
      const msg = axios.isAxiosError(e) ? e.response?.data?.detail ?? e.message : String(e)
      setError(msg)
    }
  }

  return (
    <div className="max-w-4xl mx-auto text-gray-100">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">Admin — Registry Curation</h1>
        <button
          className="text-sm text-gray-400 hover:text-gray-100"
          onClick={() => setShowAdmin(false)}
        >
          ← Back to lab
        </button>
      </div>

      <div className="flex gap-2 mb-4">
        <input
          type="password"
          placeholder="X-Admin-Token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm"
        />
        <button
          onClick={loadSummary}
          disabled={loading || !token}
          className="bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-lg"
        >
          {loading ? 'Loading…' : 'Load summary'}
        </button>
      </div>

      {error && <div className="text-red-400 text-sm mb-4">{error}</div>}

      {summary && (
        <>
          <div className="grid grid-cols-3 gap-3 mb-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-2xl font-semibold">{summary.n_presets}</div>
              <div className="text-xs text-gray-500 mt-1">Registered presets</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-2xl font-semibold">{summary.n_architectures}</div>
              <div className="text-xs text-gray-500 mt-1">Registered architectures</div>
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="text-2xl font-semibold">{summary.models.length}</div>
              <div className="text-xs text-gray-500 mt-1">Problems with a stored model</div>
            </div>
          </div>

          <table className="w-full text-sm">
            <thead className="text-gray-500 text-left">
              <tr>
                <th className="pb-2">Problem ID</th>
                <th className="pb-2">Latest version</th>
                <th className="pb-2">Stage</th>
                <th className="pb-2">Dataset scenarios</th>
                <th className="pb-2"></th>
              </tr>
            </thead>
            <tbody>
              {summary.models.map((m) => (
                <tr key={m.problem_id} className="border-t border-gray-800">
                  <td className="py-2">{m.problem_id}</td>
                  <td className="py-2 text-gray-400">{m.latest_model_version ?? '—'}</td>
                  <td className="py-2 text-gray-400">{m.stage ?? '—'}</td>
                  <td className="py-2 text-gray-400">{m.dataset_scenarios.length}</td>
                  <td className="py-2">
                    {m.latest_model_version && m.stage !== 'production' && (
                      <button
                        className="text-xs text-brand-400 hover:text-brand-300"
                        onClick={() => promote(m.problem_id, m.latest_model_version as string, 'production')}
                      >
                        Promote to production
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
