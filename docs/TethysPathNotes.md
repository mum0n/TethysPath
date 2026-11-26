# TethysPath Results Viewer and Diagnostic Notes

This document describes the persistent Zarr checkpoints, CSV tables, TOML parameter exports, and interactive HTML views produced by `TethysPath`, along with Julia snippets for inspecting stored runs without repeating MCMC inference.

## Checkpoint Loading and Structure

Pipeline execution (`run_movement_analysis`) writes two Zarr archives via `Tethys.write_data` into the configured output directory (default `../lakehouse/TethysPath`):

- `movement_checkpoint.zarr`: written at the end of Phase 3; stores `loaded`, `fitted`, and `kernels`. Used by `--resume=true` (`resume_from_checkpoint = true`).
- `movement_results_checkpoint.zarr`: written at the end of Phase 6; stores `loaded`, `kernels`, `path_res`, `diagnostics`, `validation`, `agent_trajectories`, and `agent_space_use`. Used by `--figures-only=true`.

Sparse adjacency matrices and polygon coordinate vectors are stored in Zarr-compatible arrays and restored via `TethysPath._zarr_checkpoint_restore_loaded`:

```julia
using TethysPath, Tethys, DataFrames, Statistics, SparseArrays

store_dir = Tethys.lakehouse_dir("TethysPath")
results_ckpt = joinpath(store_dir, "movement_results_checkpoint.zarr")
interm_ckpt  = joinpath(store_dir, "movement_checkpoint.zarr")

if ispath(results_ckpt)
    rc = Tethys.read_data(results_ckpt)
    loaded = TethysPath._zarr_checkpoint_restore_loaded(rc["loaded"])
    kernels = rc["kernels"]
    path_res = rc["path_res"]
    diagnostics = rc["diagnostics"]
    validation = rc["validation"]
    agent_trajectories = get(rc, "agent_trajectories", nothing)
    agent_space_use = get(rc, "agent_space_use", nothing)
elseif ispath(interm_ckpt)
    ic = Tethys.read_data(interm_ckpt)
    loaded = TethysPath._zarr_checkpoint_restore_loaded(ic["loaded"])
    fitted = TethysPath._zarr_checkpoint_restore_fitted(ic["fitted"])
    kernels = ic["kernels"]
end
```

To regenerate the interactive dashboard and summary exports from an existing results checkpoint:

```julia
cfg = load_config(
    config_path = "configs/snowcrab.toml",
    overrides = Dict(:figures_only => true)
)
run_movement_analysis(cfg)
```

## Spatial Domain and Marine Lattice Summary

The restored `loaded` NamedTuple contains the active marine mesh (`mesh`), mark-recapture encounter table (`obs_df`), marine-masked adjacency matrix (`W`), climatological and monthly habitat suitability arrays (`hsi_vec`, `monthly_hsi`), bathymetric depths (`resharded_depths`), bottom temperatures (`bottom_temp`), land/lake mask (`land_mask`), and management-region assignments (`region_map`, `region_labels`).

```julia
n_units  = loaded.mesh.n_units
n_obs    = nrow(loaded.obs_df)
n_marine = isnothing(loaded.land_mask) ? n_units : count(!, loaded.land_mask)
println("Active units: $n_units ($n_marine marine), encounter rows: $n_obs")
```

## Fitted Transition Kernels and Data-Source Stratification

The `kernels` structure holds posterior estimates for the advection weight $\alpha \in [0, 1]$, residence probability $\rho \in [0, 1)$, and habitat-gradient sensitivity $\gamma \in \mathbb{R}$, alongside the $S \times S$ row-stochastic transition operator `P_kernel` ($\sum_v P_{uv} = 1$). When `obs_df.datasource` contains both acoustic receiver detections (`"otn"`) and commercial pot-trap recaptures (`"bio"`), `kernels.datasource_summary` (exported to `acoustic_vs_trap_inference.csv`) contrasts parameter posteriors and effective dispersal scales across observation modalities.

Normalized outgoing step entropy on non-self marine transitions is defined at each marine unit $u$ with marine degree $d_u = \deg_{\text{marine}}(u) \ge 2$ as:

$$H(u) = -\frac{1}{\log d_u} \sum_{v \ne u} \tilde{P}_{uv} \log \tilde{P}_{uv} \in [0, 1], \qquad \tilde{P}_{uv} = \frac{P_{uv}}{\sum_{w \ne u} P_{uw}},$$

where $H(u) = 1$ indicates locally isotropic marine dispersal (attained across homogeneous-HSI regions where $\Delta\text{HSI} = 0$) and $H(u) \to 0$ indicates deterministic concentration onto a single outgoing edge.

```julia
P = sparse(Float64.(kernels.P_kernel))
step_H = TethysPath._unit_step_entropy(P, loaded.land_mask)
println("Median outgoing step entropy H(u): ", median(step_H[.!loaded.land_mask]))
```

## Trajectories, Corridors, and Structural Bottlenecks

The `path_res` structure stores conditioned Markov-bridge or Viterbi trajectories (`paths`) for spatially stratified tags, interior bridge transit densities (`corridors`), and domain-wide bottleneck metrics (`domain_bottlenecks`), including nodal transit intensity $C_{\text{domain}}(u)$, net skew-symmetric edge flux $F_{\text{net}}(u, v) = \max(0, F(u, v) - F(v, u))$, and structural bottleneck scores $B(u) = C_{\text{domain}}(u) / \max(1, \deg_{\text{marine}}(u))$.

To reconstruct locally smoothed predicted trajectories across every mark-recapture event in `loaded.obs_df`:

```julia
all_events = reconstruct_all_event_trajectories(
    loaded, kernels;
    step_years = 1.0 / 365.25,
    seed = 42,
    resolution = "h",
    grid_minutes = 1.25
)
```

Each returned record contains `event_id`, `tagid`, `datasource`, `liberty_days`, `liberty_bin`, `liberty_label` (`< 1 month`, `1 mo - 1 yr`, `1 - 2 yr`, `2 - 3 yr`, `3 - 4 yr`, `>= 4 yr`), `displacement_km`, `path_length_km`, and marine-guarded quadratic B-spline coordinates `coords`.

## Corridor Canalization, Regional Connectivity, and Validation

When `:validation` is enabled in `diagnostics`, `validation` provides:

- `corridor_canalization` (`corridor_canalization_summary.csv`): per-unit step entropy $H(u)$, interior bridge transit probability, and canalization index $K(u) = \sqrt{\tilde{C}(u)} \,(1 - 0.5\,H(u)) \in [0, 1]$, replacing endpoint-indexed path credible intervals.
- `connectivity_matrix` and `connectivity_uncertainty` (`stock_connectivity_summary.csv`, `stock_connectivity_credible_intervals.csv`): 1-year ($\mathbf{P}^{365}$) HSI-weighted regional transition probabilities across configured management polygons (`["N-ENS", "S-ENS", "4X", "CFA19 / S.Gulf"]`, plus `"Unassigned"` if active units fall outside polygon boundaries), together with posterior 95% credible intervals.
- `posterior_predictive` (`posterior_predictive_summary.txt`): Kullback-Leibler divergence ($\mathrm{KL}(p_{\text{obs}} \parallel p_{\text{pred}})$ in nats) across posterior parameter draws.

Optional diagnostics `:connectivity` and `:recapture` additionally populate `validation.connectivity_assessment` (exact regional mean first-passage times, arrival CDFs, modularity communities, and minimum cuts) and `validation.recapture_separation` (multistate Cormack-Jolly-Seber survival $\phi$, detection $p_{\text{det}}$, and movement $\psi$).

## Consolidated Interactive HTML Dashboard

When `render_html = true`, `export_dashboards` writes `movement_dashboard.html` paired with the RFC 1950 zlib-compressed payload sidecar `movement_dashboard.data.zz` (loaded automatically over HTTP/HTTPS on GitHub Pages via `DecompressionStream("deflate")`, with a file-picker fallback over `file://`; see [docs/github_pages.md](github_pages.md)):

- **Smoothed Trajectories** (default first tab): Canvas-rendered loop-free `5`-knot centripetal Catmull-Rom cubic spline trajectories for every mark-recapture event, colour-coded by time between mark and recapture (`< 1 month` through `>= 4 yr`) with Mark/Release points, Recapture stars, inline movement-direction arrowheads, adjustable base opacity, and click-on-map isolation of a user-configurable number $n$ of paths.
- **Paths & HSI**: stratified Markov-bridge trajectories over habitat suitability polygons.
- **Corridors**: cumulative interior Markov-bridge transit density and top-8% core corridor nodes.
- **Directed Flux & Bottlenecks**: structural bottleneck index $B(u)$ overlaid with net directed migration flux vectors.
- **Corridor Canalization** and **Step Entropy**: spatial maps of $K(u)$ conduits and outgoing marine step entropy $H(u)$.
- **Compass Bearing Roses (100 km)**: SVG polar bearing roses and Rayleigh test vectors anchored directly at marine sector coordinates `(lat, lon)`.
- **Circuit Flow**, **Agent Space-Use**, and **Tessellation & Depth**: Kirchhoff current density and robust pinch-points, forward agent visit probabilities, and bathymetric depth across the marine hexagonal mesh.
- **Path Analysis & Diagnostics** and **Posterior Predictive (PPC)**: tabular summaries of pooled and `OTN` vs. `BIO` kernel posteriors, bathymetric/thermal/HSI gradient taxis cosines, multi-horizon MSD scaling ($\text{MSD} \propto \Delta t^\beta$) and null-model $\Delta\text{LPPD}$, ontogenetic depth shifts $\Delta z$, biological-state stratification, 1-year regional transition matrices, and posterior predictive KL divergence traces.
