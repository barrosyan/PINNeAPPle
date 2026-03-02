import React from 'react'
import RunPanel from '../components/RunPanel'

export default function Fusion() {
  return (
    <RunPanel
      title="Vertical D — Physics + Time Series Fusion"
      subtitle="Critical forecasting under physical constraints (demo: damped oscillator; compare supervised vs physics-fused)."
      endpoint="/api/vertical-d/run"
      defaultConfig={{
        T: 1200,
        dt: 0.01,
        zeta: 0.05,
        w0: 6.283185307179586,
        noise_std: 0.02,
        lookback: 32,
        hidden: 64,
        train_steps: 1200,
        batch_size: 128,
        lr: 0.002,
        w_phys: 0.5,
        prefer_cuda: true,
        seed: 0
      }}
    />
  )
}
