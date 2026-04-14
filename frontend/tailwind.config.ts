import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        pitch: { DEFAULT: "#0f3d2e", light: "#1a5c40" },
        ball: "#f4d03f",
      },
    },
  },
  plugins: [],
};

export default config;
