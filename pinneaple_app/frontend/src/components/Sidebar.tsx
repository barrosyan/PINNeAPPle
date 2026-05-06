import React from 'react'
import clsx from 'clsx'
import { useStore, type Step } from '../store'

const STEPS: { id: Step; label: string; icon: string }[] = [
  { id: 'problem',  label: '1  Problem',     icon: '⚛' },
  { id: 'geometry', label: '2  Geometry',    icon: '⬡' },
  { id: 'models',   label: '3  Models',      icon: '🧠' },
  { id: 'run',      label: '4  Run',         icon: '▶' },
  { id: 'results',  label: '5  Results',     icon: '📊' },
]

export function Sidebar() {
  const { step, setStep, selectedProblem, isCustomProblem, selectedModels, experimentId } = useStore()

  function canNavigate(id: Step): boolean {
    const order = STEPS.map((s) => s.id)
    const current = order.indexOf(step)
    const target  = order.indexOf(id)
    if (target <= current) return true
    if (id === 'geometry' && (selectedProblem || isCustomProblem)) return true
    if (id === 'models'   && (selectedProblem || isCustomProblem)) return true
    if (id === 'run'      && selectedModels.length > 0)            return true
    if (id === 'results'  && experimentId !== null)                return true
    return false
  }

  return (
    <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col py-6 px-3">
      {/* Logo */}
      <div className="px-2 mb-8">
        <div className="text-brand-500 font-bold text-lg tracking-tight">🍍 PINNeAPPle</div>
        <div className="text-gray-500 text-xs mt-0.5">Physics AI Lab</div>
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1">
        {STEPS.map(({ id, label, icon }) => {
          const active  = step === id
          const enabled = canNavigate(id)
          return (
            <button
              key={id}
              onClick={() => enabled && setStep(id)}
              disabled={!enabled}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all',
                active  && 'bg-brand-600 text-white',
                !active && enabled  && 'text-gray-400 hover:bg-gray-800 hover:text-gray-100',
                !active && !enabled && 'text-gray-700 cursor-not-allowed',
              )}
            >
              <span className="text-base">{icon}</span>
              <span>{label}</span>
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-2 pt-4 border-t border-gray-800 text-xs text-gray-600">
        <div>PINNeAPPle v1.0</div>
        <div className="mt-0.5">Physics-Informed AI</div>
      </div>
    </aside>
  )
}
