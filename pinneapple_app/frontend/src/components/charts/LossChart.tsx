import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const COLORS = ['#38bdf8','#4ade80','#fb923c','#a78bfa','#f472b6','#facc15','#67e8f9','#86efac']

interface Props {
  modelEvents: Record<string, { epoch: number; loss: number; physLoss: number }[]>
}

export function LossChart({ modelEvents }: Props) {
  const models = Object.keys(modelEvents)
  if (models.length === 0) return null

  // Merge into unified epoch-indexed array for the chart
  const maxEpoch = Math.max(...models.flatMap((m) => modelEvents[m].map((e) => e.epoch)))
  const step = Math.max(1, Math.floor(maxEpoch / 80))

  const merged: Record<number, Record<string, number>> = {}
  models.forEach((m, i) => {
    modelEvents[m].forEach(({ epoch, loss }) => {
      const bucket = Math.floor(epoch / step) * step
      if (!merged[bucket]) merged[bucket] = { epoch: bucket }
      merged[bucket][m] = loss
    })
  })

  const data = Object.values(merged).sort((a, b) => (a.epoch as number) - (b.epoch as number))

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 10 }} />
        <YAxis
          scale="log"
          domain={['auto', 'auto']}
          tick={{ fill: '#64748b', fontSize: 10 }}
          tickFormatter={(v: number) => v.toExponential(0)}
        />
        <Tooltip
          contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
          labelStyle={{ color: '#94a3b8', fontSize: 11 }}
          itemStyle={{ fontSize: 11 }}
          formatter={(v: number) => v.toExponential(3)}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
        {models.map((m, i) => (
          <Line
            key={m}
            type="monotone"
            dataKey={m}
            stroke={COLORS[i % COLORS.length]}
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
