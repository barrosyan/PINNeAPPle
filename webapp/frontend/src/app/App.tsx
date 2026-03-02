import React from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import Surrogate from '../pages/Surrogate'
import DigitalTwin from '../pages/DigitalTwin'
import Arena from '../pages/Arena'
import Fusion from '../pages/Fusion'

const NavItem = ({ to, label, sub }: { to: string; label: string; sub: string }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      `block rounded-xl px-3 py-2 transition ${isActive ? 'bg-slate-200/10 border border-slate-200/20' : 'hover:bg-slate-200/5'}
      `
    }
  >
    <div className="text-sm font-semibold text-slate-100">{label}</div>
    <div className="text-xs text-slate-300/80 mt-0.5">{sub}</div>
  </NavLink>
)

export default function App() {
  return (
    <div className="min-h-screen bg-metal-gradient text-slate-100">
      <div className="max-w-[1400px] mx-auto px-4 py-6">
        <header className="flex items-center justify-between">
          <div>
            <div className="text-xl font-semibold tracking-tight">Pinneaple</div>
            <div className="text-sm text-slate-300/80">Industrial Physics OS — interactive modules</div>
          </div>
          <div className="text-xs text-slate-300/80 metal-card rounded-xl px-3 py-2 shadow-metal">
            Modern • Minimal • Metallic
          </div>
        </header>

        <div className="mt-6 grid grid-cols-12 gap-4">
          <aside className="col-span-12 md:col-span-3">
            <div className="metal-card shadow-metal rounded-2xl p-3">
              <div className="text-xs uppercase tracking-widest text-slate-300/70 px-1 pb-2">Business Verticals</div>
              <div className="space-y-2">
                <NavItem to="/" label="A — Surrogate Engineering" sub="CFD/FEA → neural surrogate → instant inference" />
                <NavItem to="/digital-twin" label="B — Digital Twin Builder" sub="Streaming sensors → recalibration → predictive maintenance" />
                <NavItem to="/arena" label="C — Benchmark Arena" sub="Standardized datasets → ranking → reproducible runs" />
                <NavItem to="/fusion" label="D — Physics + Time Series" sub="Forecasting with physical constraints" />
              </div>
              <div className="mt-3 text-xs text-slate-300/70 px-1">
                Tip: run a module, inspect metrics, compare models.
              </div>
            </div>
          </aside>

          <main className="col-span-12 md:col-span-9">
            <Routes>
              <Route path="/" element={<Surrogate />} />
              <Route path="/digital-twin" element={<DigitalTwin />} />
              <Route path="/arena" element={<Arena />} />
              <Route path="/fusion" element={<Fusion />} />
            </Routes>
          </main>
        </div>
      </div>
    </div>
  )
}
