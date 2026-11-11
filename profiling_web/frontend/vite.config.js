import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000', '/perfetto': 'http://127.0.0.1:8000' },
    fs: { allow: [fileURLToPath(new URL('.', import.meta.url)), fileURLToPath(new URL('../../web_shared', import.meta.url))] },
  },
  build: { rollupOptions: { output: { manualChunks: { charts: ['echarts/core', 'echarts/charts', 'echarts/components', 'echarts/renderers'] } } } },
})
