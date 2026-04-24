import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#16213E",
            color:      "#E8E8E8",
            border:     "1px solid #2a3a5c",
            borderRadius: "8px",
          },
          success: { iconTheme: { primary: "#2ED573", secondary: "#16213E" } },
          error:   { iconTheme: { primary: "#FF4757", secondary: "#16213E" } },
        }}
      />
    </QueryClientProvider>
  </React.StrictMode>
);
