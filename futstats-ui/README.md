# FutStats UI

Dark-theme, FotMob-inspired frontend for the FutStats analytics pipeline. Pure HTML/CSS/JS — no build step, no dependencies to install.

## Run it

Just open `index.html` in a browser. That's it — the page ships with sample data baked in (`js/data.js`) so it renders fully out of the box.

If you'd rather serve it (recommended once you wire in real assets, since `file://` blocks some image/video loading in some browsers):

```
npx serve .
# or
python3 -m http.server 8000
```

## Structure

```
index.html            All markup, all tabs (single page, JS toggles visibility)
css/styles.css         Design tokens + all component styles
js/data.js              Sample data object (window.FUTSTATS_DATA) — used as a fallback
js/app.js                All rendering logic, tab routing, table sort/filter
data/results.sample.json Reference schema for your backend's real output
assets/                  Put generated heatmaps / pass networks / video here
```

## Wiring in real pipeline output

1. Have your pipeline write a `results.json` (same shape as `data/results.sample.json`) to `data/results.json`.
2. `app.js` already tries `fetch('data/results.json')` first and only falls back to the sample data in `js/data.js` if that fetch fails — so dropping the file in is enough, no code changes needed.
3. Point the `assets` paths in your `results.json` at wherever the generated heatmap/pass-network images and annotated video actually live (relative to `index.html`, or a full URL).
4. Missing or broken image paths degrade gracefully to a small "not generated yet" placeholder frame — the page never shows a broken image icon.

## What's intentionally NOT here

Per the product spec, no UI slot exists for real-world distance/speed (km/h) or shot/goal detection — both are documented as not implemented, not hidden. If you build those later, add them as their own cards rather than repurposing an existing one.

## Notes on the player identity caveat

The small caption above the player table ("Player identity is tracked automatically and may occasionally split one player into multiple entries...") is a permanent, low-key UI element — it's meant to stay visible, not be dismissed or hidden once the tracker improves. Update the copy if the underlying behavior changes, don't remove it silently.
