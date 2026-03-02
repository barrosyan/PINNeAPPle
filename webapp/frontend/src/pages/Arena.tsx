import React from 'react'
import RunPanel from '../components/RunPanel'

export default function Arena() {
  return (
    <RunPanel
      title="Vertical C — Scientific Benchmark Arena"
      subtitle="Standardized benchmark + model ranking (demo: 5 models on Heat2D; PINN models add PDE/BC/IC losses)."
      endpoint="/api/vertical-c/run"
      defaultConfig={{
        alpha: 0.02,
        n_data: 20000,
        n_col: 8000,
        n_bc: 4000,
        n_ic: 4000,
        train_steps: 1200,
        batch_size: 2048,
        lr: 0.002,
        w_data: 1.0,
        w_pde: 1.0,
        w_bc: 5.0,
        w_ic: 5.0,
        grid: 64,
        t_test: 0.6,
        prefer_cuda: true,
        seed: 0
      }}
    />
  )
}
