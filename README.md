# TethysPath.jl

Discrete-space animal movement inference, conditioned Markov-bridge trajectory reconstruction, and spatial connectivity analysis in Julia.

## Overview

`TethysPath.jl` estimates discrete-time movement transition kernels over marine spatial lattices from mark-recapture and acoustic telemetry encounter records, reconstructs trajectories conditioned on observed endpoints, and evaluates regional demographic connectivity and topological transport bottlenecks. Empirical observables (release and recapture coordinates, time at liberty, biological traits, and data-source modality) are separated from model-dependent quantities inferred under a fitted row-stochastic Markov operator $\mathbf{P} \in [0, 1]^{S \times S}$.

- **Telemetry ingestion and marine domain discretization**: Projects encounter histories onto hexagonal lattices or Centroidal Voronoi Tessellations via `Tethys.jl`, enforces high-resolution (`1.25`-arcminute) global ocean/lake raster masking, severs overland and causeway adjacency edges, masks enclosed inland water pockets, and clips coastal polygon boundaries to the ocean shoreline.
- **Habitat suitability coupling and regional domains**: Pairs survey-derived Habitat Suitability Index (HSI) fields with homogeneous baseline suitability (`0.50`) across unsurveyed management zones (such as CFA 19 and the Southern Gulf of St. Lawrence) so movement in unsurveyed waters reduces to two-dimensional marine diffusion rather than collapsing onto shortest-path channels.
- **Bayesian transition kernel inference**: Infers advection weight $\alpha \in [0, 1]$, residence probability $\rho \in [0, 1)$, and habitat-gradient sensitivity $\gamma \in \mathbb{R}$ via `Turing.jl`, both pooled across all records and stratified between fishery-independent acoustic receiver arrays (`datasource == "otn"`) and commercial pot-trap recaptures (`datasource == "bio"`, incorporating an empirical trap-effort offset).
- **Conditioned trajectory reconstruction and marine B-spline smoothing**: Samples conditioned forward-backward Markov bridges and Viterbi maximum-probability paths across marine edges, and constructs locally smoothed quadratic B-spline trajectories (guarded against shoreline excursions) for every mark-recapture event stratified by time at liberty.
- **Directional, environmental, and corridor diagnostics**: Evaluates spatially discretized $100 \times 100\text{ km}$ compass bearing roses and Rayleigh uniformity tests, projects displacement vectors onto local bathymetric ($\nabla z$), thermal ($\nabla T$), and habitat ($\nabla\text{HSI}$) gradients, and quantifies normalized outgoing step entropy $H(u) \in [0, 1]$ alongside corridor canalization $K(u) \in [0, 1]$.
- **Circuit-theoretic and regional connectivity**: Models the marine graph as a resistive network ($\mathbf{L}_C \mathbf{V} = \mathbf{I}_{\text{ext}}$) to locate low-degree current pinch-points, computes 1-year ($\mathbf{P}^{365}$) HSI-weighted regional transition probabilities across management areas, and evaluates exact mean first-passage times and multistate Cormack-Jolly-Seber survival/detection separation.
- **Lakehouse persistence and dashboard export**: Writes Zarr checkpoints (`movement_checkpoint.zarr`, `movement_results_checkpoint.zarr`), diagnostic CSV tables, `TethysBio` movement parameter summaries (`movement_stats.toml`), and a self-contained multi-tab Leaflet HTML dashboard (`movement_dashboard.html`) to `lakehouse/TethysPath`.

Mathematical formulations, phase specifications, and configuration parameters are documented in [docs/movement_analysis.md](docs/movement_analysis.md).

## Pipeline Stages

Execution in `run_movement_analysis` proceeds sequentially through six stages:

1. **Data ingestion and marine topology (`load_movement_data`)**: Loads encounter records, transfers signed bathymetric elevation onto the spatial lattice, applies `1.25`-arcminute land and lake masking, severs overland adjacency edges, clips coastal cell vertices to ocean pixels, and assigns regional polygon memberships.
2. **Bayesian kernel estimation (`fit_movement_models`)**: Samples pooled and data-source-stratified (`otn` vs. `bio`) discrete-time Markov transition models using prior-scaled random-walk Metropolis-Hastings proposals and $k$-step forward probability caches.
3. **Transition operator extraction (`extract_transition_kernels`)**: Assembles posterior-mean and per-draw transition matrices $\mathbf{P}$, writes `movement_checkpoint.zarr`, and simulates forward individual-based agent trajectories when `:agent` is active.
4. **Trajectory, corridor, and bottleneck reconstruction (`reconstruct_paths_and_diagnostics`)**: Samples conditioned Markov-bridge trajectories, accumulates interior bridge transit densities, and evaluates structural bottleneck scores $B(u) = C_{\text{domain}}(u) / \max(1, \deg_{\text{marine}}(u))$ and net directed edge fluxes.
5. **Circuit, connectivity, and validation analyses (`compute_advanced_diagnostics`, `execute_validation_analyses`)**: Solves multi-pair Kirchhoff current equations, computes spatial corridor canalization $K(u)$, evaluates 1-year ($\mathbf{P}^{365}$) HSI-weighted regional stock connectivity and credible intervals, and performs posterior predictive Kullback-Leibler divergence checks.
6. **Dashboard and lakehouse export (`export_dashboards`)**: Reconstructs locally smoothed marine trajectories across all mark-recapture events, computes $100 \times 100\text{ km}$ bearing roses and gradient taxis projections, writes CSV summaries, `movement_dashboard.html`, `movement_stats.toml`, and `movement_results_checkpoint.zarr`.

## Quick Start

```julia
using TethysPath

# Load TOML configuration and execute the end-to-end pipeline
config = load_config(config_path = "configs/snowcrab.toml")
results = run_movement_analysis(config)

# Or execute individual stages explicitly
loaded = load_movement_data(config)
fitted = fit_movement_models(loaded, config)
kernels = extract_transition_kernels(loaded, fitted, config)
path_res = reconstruct_paths_and_diagnostics(loaded, kernels, config)
```

## Command-Line Interface

The CLI entry point `TethysPathRun.jl` accepts configuration files and flag overrides:

```powershell
# Full snow crab analysis from Phase 1
julia --project=. TethysPathRun.jl --config=configs/snowcrab.toml

# Resume from saved MCMC checkpoint (skip Phases 1-3)
julia --project=. TethysPathRun.jl --config=configs/snowcrab.toml --resume=true

# Regenerate HTML dashboard and CSV summaries from full results checkpoint
julia --project=. TethysPathRun.jl --config=configs/snowcrab.toml --figures-only=true
```

## Results Inspection and Notes

Instructions for loading Zarr checkpoints, inspecting posterior transition kernels, and querying exported tables are provided in [docs/TethysPathNotes.md](docs/TethysPathNotes.md). Local HTTP viewing and GitHub Pages deployment of `movement_dashboard.html` and `movement_dashboard.data.zz` are documented in [docs/github_pages.md](docs/github_pages.md). The biological and mathematical specification for Northwest Atlantic snow crab (*Chionoecetes opilio*) is given in [docs/snowcrab_movement_connectivity_research_document.md](docs/snowcrab_movement_connectivity_research_document.md).


## GitHub serving:

```
git push -u origin gh-pages

git switch gh-pages
Copy-Item ..\lakehouse\TethysPath\movement_dashboard.html .\index.html
Copy-Item ..\lakehouse\TethysPath\movement_dashboard.data.zz .\movement_dashboard.data.zz
git add index.html movement_dashboard.data.zz
git commit --amend --no-edit
git push --force-with-lease origin gh-pages
git switch main
```
