import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// VITE_BASE: the path the site is served under (GitHub Pages: "/<repo>/").
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react()],
})
