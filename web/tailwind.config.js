/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'royal-purple': '#4b0082',
        'deep-black': '#050505',
        'neon-purple': '#b026ff',
      }
    },
  },
  plugins: [],
}
