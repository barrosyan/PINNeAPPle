import React from 'react'
import RunPanel from '../components/RunPanel'

export default function Surrogate() {
  return (
    <RunPanel
      title="Vertical A — Surrogate Engineering Platform"
      subtitle="Direct replacement of CFD/FEA with fast surrogates (demo: Heat2D solver → train surrogate → instant inference)."
      endpoint="/api/vertical-a/run"
      defaultConfig={{
        grid: 64,
        steps: 240,
        snapshots: 8,
        alpha: 0.02,
        dt: 0.001,
        train_steps: 700,
        batch_size: 4096,
        width: 128,
        depth: 4,
        lr: 0.002,
        prefer_cuda: true,
        seed: 0
      }}
    />
  )
}
