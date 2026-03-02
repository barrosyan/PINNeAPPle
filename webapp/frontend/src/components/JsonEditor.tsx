import React from 'react'

export default function JsonEditor({ value, onChange }: { value: any; onChange: (v: any) => void }) {
  const [text, setText] = React.useState(JSON.stringify(value, null, 2))
  const [err, setErr] = React.useState<string | null>(null)

  React.useEffect(() => {
    setText(JSON.stringify(value, null, 2))
  }, [value])

  const apply = () => {
    try {
      const parsed = JSON.parse(text)
      setErr(null)
      onChange(parsed)
    } catch (e: any) {
      setErr(e?.message ?? 'Invalid JSON')
    }
  }

  return (
    <div className="metal-card rounded-2xl p-3 shadow-metal">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-semibold">Config</div>
        <button className="metal-btn rounded-xl px-3 py-1.5 text-xs hover:bg-slate-200/10" onClick={apply}>
          Apply
        </button>
      </div>
      <textarea
        className="w-full h-56 bg-black/20 border border-slate-200/10 rounded-xl p-3 font-mono text-xs text-slate-100 outline-none focus:border-slate-200/25"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      {err && <div className="mt-2 text-xs text-rose-300">{err}</div>}
    </div>
  )
}
