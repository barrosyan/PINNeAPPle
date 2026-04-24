import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import TopBar  from "./TopBar";
import { useAppStore } from "@/store";

export default function Layout() {
  const collapsed = useAppStore((s) => s.sidebarCollapsed);

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <Sidebar />
      <div
        className="flex flex-col flex-1 overflow-hidden transition-all duration-300"
        style={{ marginLeft: collapsed ? 64 : 240 }}
      >
        <TopBar />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
