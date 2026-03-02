import React from 'react'
import RunPanel from '../components/RunPanel'

export default function DigitalTwin() {
  return (
    <RunPanel
      title="Vertical B — Digital Twin Builder"
      subtitle="Streaming sensors → continuous recalibration (demo: battery-like voltage model with online parameter update)."
      endpoint="/api/vertical-b/run"
      defaultConfig={{
        T: 600,
        dt: 1.0,
        noise_std: 0.01,
        R_true: 0.08,
        R_init: 0.2,
        lr: 0.05,
        seed: 0
      }}
    />
  )
}
