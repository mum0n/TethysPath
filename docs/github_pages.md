# Interactive Dashboard: Local Viewing and GitHub Pages Deployment

```powershell
git switch gh-pages
Copy-Item ..\lakehouse\TethysPath\movement_dashboard.html .\index.html
Copy-Item ..\lakehouse\TethysPath\movement_dashboard.data.zz .\movement_dashboard.data.zz
git add index.html movement_dashboard.data.zz
git commit --amend --no-edit
git push --force-with-lease origin gh-pages
git switch main
```

`export_dashboards` writes two paired files to the configured output directory (for example, `../lakehouse/TethysPath/`):

- `movement_dashboard.html`: Canvas-backed (`preferCanvas: true`) Leaflet.js dashboard shell with left-docked legends and controls, tab bar (`Smoothed Trajectories` default first tab, `Paths & HSI`, `Corridors`, `Directed Flux & Bottlenecks`, `Corridor Canalization`, `1-Yr Effective Dispersal`, `Compass Bearing Roses (100 km)`, `Circuit Flow`, `Agent Space-Use`, `Tessellation & Depth`, plus report tabs), and click-on-map selection defaulting to `Nearest Marks / Releases`.
- `movement_dashboard.data.zz`: RFC 1950 zlib-compressed JSON sidecar containing the shoreline-clipped hexagonal marine cells, `5`-knot centripetal Catmull-Rom spline trajectories across all mark-recapture events, direction arrowheads, `100 km` SVG bearing roses, and diagnostic field layers.

Zarr checkpoints (`movement_checkpoint.zarr`, `movement_results_checkpoint.zarr`) are used only by the Julia pipeline and are not required by the browser.

## Opening Locally

- **Local static HTTP server (automatic loading)**:
  - Start a static HTTP server in the directory containing both files:
    ```powershell
    Set-Location ..\lakehouse\TethysPath
    python -m http.server 8000
    ```
  - Open `http://localhost:8000/movement_dashboard.html`. The page fetches and decompresses `./movement_dashboard.data.zz` automatically via `DecompressionStream("deflate")`, matching GitHub Pages behavior.
- **Direct open (`file://`)**:
  - Open `movement_dashboard.html` directly in a browser.
  - Because browsers block `fetch()` requests over `file://`, the loader overlay displays a **Select `movement_dashboard.data.zz`** button.
  - Select `movement_dashboard.data.zz` from the same directory to decompress and render the dashboard in memory.

## Publishing with an Orphan `gh-pages` Branch

An orphan `gh-pages` branch keeps binary `.data.zz` sidecars out of the `main` branch history.

- **Create the orphan branch and clear the tracked index** (from the `TethysPath` repository root with a clean working tree on `main`):
  ```powershell
  git switch --orphan gh-pages
  git rm -rf .
  ```
  - `.git` and any files that were untracked or gitignored on `main` (such as `Manifest.toml` or local `inputs/`) remain on disk and should be left untouched so they persist when switching back to `main`.
- **Copy the dashboard files and add `.nojekyll`**:
  ```powershell
  Copy-Item ..\lakehouse\TethysPath\movement_dashboard.html .\index.html
  Copy-Item ..\lakehouse\TethysPath\movement_dashboard.data.zz .\movement_dashboard.data.zz
  New-Item -ItemType File -Name .nojekyll
  ```
- **Stage explicitly, commit, push, and return to `main`**:
  ```powershell
  git add index.html movement_dashboard.data.zz .nojekyll
  git commit -m "Publish snow crab movement connectivity dashboard"
  git push -u origin gh-pages
  git switch main
  ```
  - Avoid `git add .` on the orphan branch, because `.gitignore` is absent there and untracked files would be staged.
- **Enable GitHub Pages**:
  - In the repository settings on GitHub, open **Settings -> Pages -> Build and deployment**, set **Source** to **Deploy from a branch**, select **`gh-pages`** and **`/ (root)`**, and save.

## Updating Published Results

- Switch to `gh-pages`, copy the regenerated `.html` and `.data.zz` files, and amend the snapshot to avoid repository growth:
  ```powershell
  git switch gh-pages
  Copy-Item ..\lakehouse\TethysPath\movement_dashboard.html .\index.html
  Copy-Item ..\lakehouse\TethysPath\movement_dashboard.data.zz .\movement_dashboard.data.zz
  git add index.html movement_dashboard.data.zz
  git commit --amend --no-edit
  git push --force-with-lease origin gh-pages
  git switch main
  ```

## Deployment Constraints

- Keep `index.html` (or `movement_dashboard.html`) and `movement_dashboard.data.zz` in the same directory and retain the `movement_dashboard.data.zz` filename (`index.html` fetches `./movement_dashboard.data.zz`).
- Commit `movement_dashboard.data.zz` with standard git rather than Git LFS, because GitHub Pages serves raw LFS pointer files rather than the binary payload.
