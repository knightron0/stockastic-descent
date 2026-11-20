import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// base "./" = relative asset paths, so the same build works on Vercel and on a GitHub Pages sub-path.
export default defineConfig({ plugins: [svelte()], base: "./", build: { chunkSizeWarningLimit: 900 } });
