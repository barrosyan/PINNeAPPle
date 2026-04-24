import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAppStore } from "@/store";

export default function PrivateRoute() {
  const isAuthenticated = useAppStore((s) => !!s.accessToken);
  const location        = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
}
