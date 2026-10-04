# web/ — MTG Price Range frontend

React + Vite + TypeScript, fully static. Existing printings come from `public/data/`
(written nightly by `scripts/app/score_all.py`); custom cards are priced in the browser
by Models A and B (`src/model/`, exported by `scripts/app/export_web_models.py`). Design: [../docs/APP_DESIGN.md](../docs/APP_DESIGN.md).

```bash
npm install
npm test        # parity: the TypeScript port must match Python (needs the nightly export)
npm run dev     # http://localhost:5173
npm run build   # -> dist/
```

Build-time env: `VITE_BASE` (path the site is served under; the GitHub Pages workflow
sets `/<repo>/`).
