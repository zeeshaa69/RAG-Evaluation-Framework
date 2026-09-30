import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// base "./" lets the built site be served from any path or opened as a static export.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  test: { environment: "node" },
});
