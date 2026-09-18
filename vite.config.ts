import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The production site is served from the root of www.airest.health.
  base: "/",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});
