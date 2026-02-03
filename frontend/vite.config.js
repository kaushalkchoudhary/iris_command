import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  appType: 'spa',
  server: {
    proxy: {
      // Proxy HLS streams from MediaMTX
      '/hls': {
        target: 'http://127.0.0.1:8888',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/hls/, ''),
      },
      // Proxy WebRTC from MediaMTX
      '/webrtc': {
        target: 'http://127.0.0.1:8889',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/webrtc/, ''),
        ws: true,
      },
      // Proxy backend control API
      '/api': {
        target: 'http://127.0.0.1:9010',
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9010',
        changeOrigin: true,
      },
    },
  },
})
