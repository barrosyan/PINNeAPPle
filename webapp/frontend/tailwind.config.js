/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}'
  ],
  theme: {
    extend: {
      boxShadow: {
        'metal': '0 20px 60px rgba(0,0,0,0.45)',
      },
      backgroundImage: {
        'metal-gradient': 'radial-gradient(1200px 600px at 10% 0%, rgba(148,163,184,0.25), transparent 60%), radial-gradient(900px 400px at 90% 10%, rgba(203,213,225,0.18), transparent 55%), linear-gradient(180deg, rgba(15,23,42,1), rgba(2,6,23,1))',
      }
    },
  },
  plugins: [],
}
