import React from 'react'
import { useStore } from './store'
import { ProblemStep }    from './components/steps/ProblemStep'
import { GeometryStep }   from './components/steps/GeometryStep'
import { ModelsStep }     from './components/steps/ModelsStep'
import { RunStep }        from './components/steps/RunStep'
import { ResultsStep }    from './components/steps/ResultsStep'
import { Sidebar }        from './components/Sidebar'

const STEPS = ['problem', 'geometry', 'models', 'run', 'results'] as const

export default function App() {
  const step = useStore((s) => s.step)

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        {step === 'problem'  && <ProblemStep />}
        {step === 'geometry' && <GeometryStep />}
        {step === 'models'   && <ModelsStep />}
        {step === 'run'      && <RunStep />}
        {step === 'results'  && <ResultsStep />}
      </main>
    </div>
  )
}
