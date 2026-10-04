# web/ — MTG Price Range frontend

React + Vite + TypeScript. Reads the static data in `public/data/` (written nightly
by `scripts/app/score_all.py`) for existing printings, and calls the custom-card
API (`api/app.py`) for custom cards. Design: [../docs/APP_DESIGN.md](../docs/APP_DESIGN.md).

```bash
npm install
npm run dev                                   # http://localhost:5173, API at http://localhost:8000
VITE_API_URL=https://… npm run build          # production build -> dist/
```

Build-time env: `VITE_API_URL` (custom-card API), `VITE_BASE` (path the site is served
under; the GitHub Pages workflow sets `/<repo>/`).
