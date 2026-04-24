import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useAppStore } from "@/store";
import Layout         from "@/components/layout/Layout";
import PrivateRoute   from "@/components/auth/PrivateRoute";
import Landing        from "@/pages/Landing";
import Login          from "@/pages/Login";
import Register       from "@/pages/Register";
import Dashboard      from "@/pages/Dashboard";
import ProblemSetup   from "@/pages/ProblemSetup";
import DataGeometry   from "@/pages/DataGeometry";
import Models         from "@/pages/Models";
import Training       from "@/pages/Training";
import Inference      from "@/pages/Inference";
import Visualization  from "@/pages/Visualization";
import Benchmarks     from "@/pages/Benchmarks";
import Profile        from "@/pages/Profile";

/** Redirect logged-in users away from /login and /register */
function PublicOnlyRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAppStore((s) => !!s.accessToken);
  return isAuthenticated ? <Navigate to="/dashboard" replace /> : <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* ── Public ────────────────────────────────────────────────────── */}
        <Route path="/" element={<Landing />} />

        <Route
          path="/login"
          element={
            <PublicOnlyRoute>
              <Login />
            </PublicOnlyRoute>
          }
        />
        <Route
          path="/register"
          element={
            <PublicOnlyRoute>
              <Register />
            </PublicOnlyRoute>
          }
        />

        {/* ── Private (requires auth) ────────────────────────────────────── */}
        <Route element={<PrivateRoute />}>
          <Route element={<Layout />}>
            <Route path="dashboard"     element={<Dashboard />} />
            <Route path="problem"       element={<ProblemSetup />} />
            <Route path="data"          element={<DataGeometry />} />
            <Route path="models"        element={<Models />} />
            <Route path="training"      element={<Training />} />
            <Route path="inference"     element={<Inference />} />
            <Route path="visualization" element={<Visualization />} />
            <Route path="benchmarks"    element={<Benchmarks />} />
            <Route path="profile"       element={<Profile />} />
          </Route>
        </Route>

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
