import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Flask backend runs on 127.0.0.1:5000. Proxying keeps the browser on a
// single origin during development, so no CORS preflight is involved.
const BACKEND = process.env.VITE_BACKEND_URL || 'http://127.0.0.1:5000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/predict': { target: BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
