/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:        "#0D0D1A",
        surface:   "#16213E",
        surface2:  "#1a2444",
        surface3:  "#1e2d58",
        border:    "#2a3a5c",
        primary:   "#FF6B35",
        "primary-dark": "#e8521a",
        secondary: "#4ECDC4",
        accent:    "#FFE66D",
        text:      "#E8E8E8",
        muted:     "#8892a4",
        error:     "#FF4757",
        success:   "#2ED573",
        warning:   "#FFA502",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      boxShadow: {
        card:   "0 2px 12px rgba(0,0,0,0.4)",
        glow:   "0 0 20px rgba(255,107,53,0.3)",
        "glow-teal": "0 0 20px rgba(78,205,196,0.3)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "spin-slow":  "spin 3s linear infinite",
        "fade-in":    "fadeIn 0.3s ease-in-out",
        "slide-in":   "slideIn 0.3s ease-out",
      },
      keyframes: {
        fadeIn:  { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideIn: { "0%": { transform: "translateX(-10px)", opacity: "0" },
                   "100%": { transform: "translateX(0)", opacity: "1" } },
      },
    },
  },
  plugins: [],
};
