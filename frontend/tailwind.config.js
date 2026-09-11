/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0f1419",
          900: "#1a2332",
          700: "#334155",
          500: "#64748b",
        },
        paper: {
          50: "#f7f6f3",
          100: "#efece6",
        },
        accent: {
          700: "#0f5c66",
          600: "#147a86",
        },
        caution: {
          50: "#fbf6ea",
          800: "#6b4e12",
        },
      },
      fontFamily: {
        sans: ["Source Sans 3", "Segoe UI", "system-ui", "sans-serif"],
        display: ["Source Serif 4", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
