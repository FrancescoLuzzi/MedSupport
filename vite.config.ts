import 'dotenv/config';
import { defineConfig } from 'vite';
import { readFileSync } from 'fs';

export default defineConfig({
  root: 'frontend',
  base: './',
  optimizeDeps: {
    include: [
      '@huggingface/transformers',
      '@tursodatabase/database-wasm'
    ]
  },
  build: {
    outDir: '../dist/frontend',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            return 'vendor';
          }
        }
      }
    }
  },
  server: {
    port: Number(process.env.PORT) || 4433,
    https: {
      key: readFileSync(process.env.KEY || 'key.pem'),
      cert: readFileSync(process.env.CERT || 'cert.pem')
    },
    host: true
  }
});
