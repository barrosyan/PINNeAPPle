import React, { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import toast from 'react-hot-toast'
import { api, connectTrainingWs, type ExperimentRequest } from '../../api'
import { useStore } from '../../store'
import { LossChart } from '../charts/LossChart'

export function RunStep() {
  const {
    selectedProblem, isCustomProblem, customProblemDef,
    geometry, dataConfig, selectedModels, selectedMetrics, trainConfig,
    experimentId, experimentStatus, experimentProgress, trainingEvents,
    setExperimentId, setExperimentStatus, setExperimentProgress,
    pushTrainingEvent, resetRun, setBenchmarkPayload, setStep,
  } = useStore()

  const [launching, setLaunching] = useState(false)
  const wsCleanup = useRef<(() => void) | null>(null)

  useEffect(() => () => { wsCleanup.current?.() }, [])

  async function launch() {
    setLaunching(true)
    resetRun()

    const problemName = isCustomProblem ? '__custom__' : (selectedProblem?.name ?? '')

    const body: ExperimentRequest = {
      problem_name: problemName,
      custom_problem: isCustomProblem ? (customProblemDef as any) : undefined,
      collocation: {
        strategy:    geometry.collocationStrategy,
        n_interior:  geometry.nInterior,
        n_boundary:  geometry.nBoundary,
        n_initial:   geometry.nInitial,
        seed:        geometry.seed,
        use_geometry: geometry.useGeometry,
        domain_name: geometry.domainName || undefined,
      },
      data: {
        solver_key:      dataConfig.solverKey || undefined,
        n_snapshots:     dataConfig.nSnapshots,
        grid_resolution: dataConfig.gridResolution,
        t_end:           dataConfig.tEnd,
        use_solver:      dataConfig.useSolver,
      },
      models:  selectedModels.map((m) => ({ name: m.name, extra_kwargs: {} })),
      metrics: selectedMetrics,
      epochs:  trainConfig.epochs,
      lr:      trainConfig.lr,
      device:  trainConfig.device,
      seed:    trainConfig.seed,
    }

    try {
      const res = await api.experiments.launch(body)
      const { experiment_id } = res.data
      setExperimentId(experiment_id)
      setExperimentStatus('running')
      toast.success(`Experiment ${experiment_id} started`)

      // Connect WebSocket for live progress
      wsCleanup.current = connectTrainingWs(
        experiment_id,
        (ev) => {
          if (ev.type === 'training') {
            pushTrainingEvent({
              model:           String(ev.model),
              epoch:           Number(ev.epoch),
              totalEpochs:     Number(ev.total_epochs),
              loss:            Number(ev.loss),
              physLoss:        Number(ev.phys_loss),
              bcLoss:          Number(ev.bc_loss),
              overallProgress: Number(ev.overall_progress),
            })
            setExperimentProgress(Number(ev.overall_progress))
          }
          if (ev.type === 'progress') {
            setExperimentProgress(Number(ev.progress))
          }
        },
        async () => {
          const status = await api.experiments.status(experiment_id)
          setExperimentStatus(status.data.status)
          setExperimentProgress(status.data.progress)

          if (status.data.status === 'done') {
            const results = await api.experiments.results(experiment_id)
            setBenchmarkPayload(results.data)
            toast.success('Experiment complete! View results →')
          } else if (status.data.status === 'failed') {
            toast.error(`Experiment failed: ${status.data.message}`)
          }
        },
      )
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Launch failed')
    } finally {
      setLaunching(false)
    }
  }

  const isDone   = experimentStatus === 'done'
  const isFailed = experimentStatus === 'failed'
  const isRunning = experimentStatus === 'running'

  // Group events by model for the chart
  const modelEvents: Record<string, { epoch: number; loss: number; physLoss: number }[]> = {}
  trainingEvents.forEach((ev) => {
    if (!modelEvents[ev.model]) modelEvents[ev.model] = []
    modelEvents[ev.model].push({ epoch: ev.epoch, loss: ev.loss, physLoss: ev.physLoss })
  })

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Run Experiment</h1>
        <p className="text-gray-400 text-sm mt-1">
          Train {selectedModels.length} model{selectedModels.length > 1 ? 's' : ''} on{' '}
          <span className="text-brand-400">{isCustomProblem ? 'Custom Problem' : selectedProblem?.name}</span> for {trainConfig.epochs} epochs.
        </p>
      </div>

      {/* Config summary */}
      <div className="card grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
        <div><div className="label">Models</div><div className="text-gray-200">{selectedModels.map(m => m.name).join(', ')}</div></div>
        <div><div className="label">Collocation</div><div className="text-gray-200">{geometry.collocationStrategy} · {geometry.nInterior.toLocaleString()} pts</div></div>
        <div><div className="label">Solver data</div><div className="text-gray-200">{dataConfig.useSolver ? `${dataConfig.nSnapshots} snapshots` : 'Collocation only'}</div></div>
        <div><div className="label">Device</div><div className="text-gray-200">{trainConfig.device}</div></div>
      </div>

      {/* Progress bar */}
      {(isRunning || isDone || isFailed) && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-300">
              {isDone ? '✓ Complete' : isFailed ? '✗ Failed' : `Training… ${experimentProgress.toFixed(0)}%`}
            </span>
            {experimentId && <span className="font-mono text-xs text-gray-500">ID: {experimentId}</span>}
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div
              className={clsx('h-2 rounded-full transition-all', isDone ? 'bg-green-500' : isFailed ? 'bg-red-500' : 'bg-brand-500')}
              style={{ width: `${Math.min(experimentProgress, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Live loss charts */}
      {Object.keys(modelEvents).length > 0 && (
        <div className="card">
          <div className="text-sm font-semibold text-gray-200 mb-4">Live Training Loss</div>
          <LossChart modelEvents={modelEvents} />
        </div>
      )}

      {/* Action buttons */}
      <div className="flex justify-between items-center">
        <button className="btn-ghost" onClick={() => setStep('models')}>← Back</button>
        <div className="flex gap-3">
          {!isRunning && !isDone && (
            <button className="btn-primary px-6" onClick={launch} disabled={launching || selectedModels.length === 0}>
              {launching ? 'Launching…' : '▶  Start Experiment'}
            </button>
          )}
          {isDone && (
            <button className="btn-primary px-6" onClick={() => setStep('results')}>
              View Results →
            </button>
          )}
          {(isDone || isFailed) && (
            <button className="btn-ghost" onClick={() => { resetRun(); setBenchmarkPayload(null) }}>
              Reset
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
