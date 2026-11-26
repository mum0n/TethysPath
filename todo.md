# TethysPath — Remaining Work Items

Tracking items for the `TethysPath` movement-ecology package.

## Completed Architecture & Diagnostic Milestones
- [x] Consolidate inputs under `inputs/` (including `cfagulf.dat` covering CFA 19 and Southern Gulf of St. Lawrence) and store outputs/checkpoints as Zarr archives (`movement_checkpoint.zarr`, `movement_results_checkpoint.zarr`) in `lakehouse/TethysPath`.
- [x] Upgrade shoreline and inland-lake masking to `1.25`-arcminute `"h"` global raster (`land_mask_from_global_mask`), sever overland/isthmus adjacency edges (`sever_raster_land_edges!`), mask enclosed inland pockets (`mask_enclosed_inland_pockets!`), and radially clip coastal polygon vertices to ocean shorelines (`clip_polygons_to_ocean!`).
- [x] Replace deterministic `A*` routing, endpoint tortuosity/efficiency ratios, 1-day uniform regional matrices, and cell-wise Brier scores with conditioned Markov bridges (`:bridge`), 1-year ($\mathbf{P}^{365}$) HSI-weighted regional connectivity, posterior predictive KL divergence, spatial corridor canalization $K(u)$, $100 \times 100\text{ km}$ map-anchored compass bearing roses, bathymetric/thermal/HSI gradient taxis projections, and stratified `OTN` acoustic vs. `BIO` commercial trap inference.
- [x] Export consolidated single-file interactive Leaflet dashboard (`movement_dashboard.html`), including Canvas-rendered locally smoothed predicted trajectories across every mark-recapture event colour-coded by liberty interval with `TethysDrift`-style click-map path highlighting.

## Numerics & Modeling Follow-Ups
- [ ] Joint survey density likelihood vs. telemetry fit: evaluate sensitivity of Negative Binomial dispersion $r$ to survey zero-inflation when paired trawl abundance data (`surveydata_file`) are supplied.
- [ ] Higher-efficiency MCMC proposals (e.g., adaptive covariance Haario-Bardenet RW-MH or slice/NUTS with cached sparse sensitivity) to raise effective sample size (`ESS`) on multi-thousand-observation telemetry likelihoods.
- [ ] Audit `synthesize_adr_results` implicit Euler resolvent normalization (`M_prop = I - dt*(v*A - D*L)`) against direct sparse matrix exponential uniformization for large step sizes.

## Maintainability & Documentation
- [ ] Partition `src/movement.jl` (~8.4k lines) into focused sub-files (`kernels.jl`, `bridges.jl`, `diagnostics.jl`, `synthetic.jl`) under the unified `TethysPath` module namespace.
- [ ] Optional `Documenter.jl` build configuration (`docs/make.jl`) to render API docstrings alongside `docs/movement_analysis.md` and `docs/TethysPathNotes.md`.
