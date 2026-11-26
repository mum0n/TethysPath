# Movement Analysis: Pipeline Guide and Technical Reference

## 1. Overview and Architecture

`TethysPath` estimates discrete-space animal movement operators from mark-recapture and acoustic telemetry records, reconstructs conditioned marine trajectories, and quantifies spatial corridors, circuit-theoretic pinch-points, and regional management connectivity. Empirical observables (release and recapture locations, elapsed liberty intervals, biological traits, and gear/receiver modality) are kept distinct from model-derived constructs (transition kernels, conditioned Markov-bridge paths, step entropy, canalization indices, and multi-step regional exchange probabilities).

Execution in `run_movement_analysis` proceeds sequentially through six phases:

```text
[Phase 1] Data Ingestion & Marine Mesh Topology (load_movement_data)
   |      - Mark-recapture / acoustic telemetry loading & trait enrichment
   |      - 1.25-arcmin global ocean/lake raster masking & signed elevation transfer
   |      - Overland edge severing, enclosed-pocket masking & shoreline polygon clipping
   |      - Survey HSI assignment + homogeneous regional HSI (0.50) in unsurveyed zones
   v
[Phase 2] Bayesian Kernel Inference (fit_movement_models)
   |      - Pooled discrete-time Markov likelihood via prior-scaled RW-MH
   |      - Stratified Acoustic (OTN) vs. Commercial Trap (BIO + effort offset) models
   v
[Phase 3] Transition Kernel Extraction & Forward Agents (extract_transition_kernels)
   |      - Posterior mean & draw-level row-stochastic operators P
   |      - Intermediate Zarr checkpoint: movement_checkpoint.zarr
   |      - Optional forward individual-based agent projections (:agent)
   v
[Phase 4] Conditioned Trajectories, Corridors & Bottlenecks (reconstruct_paths_and_diagnostics)
   |      - Conditioned forward-backward Markov bridges (:bridge) or Viterbi (:viterbi)
   |      - Interior Markov-bridge transit corridors
   |      - Domain transit intensity C(u), net directed flux F_net(u, v) & bottlenecks B(u)
   v
[Phase 5] Circuit, Connectivity & Validation Diagnostics
   |      - Multi-pair Kirchhoff current density & stochastic pinch-points (:circuit)
   |      - Outgoing step entropy H(u) & spatial corridor canalization K(u) (:validation)
   |      - 1-year (P^365) HSI-weighted regional transition matrix & CIs (:validation)
   |      - Posterior predictive KL divergence (:validation)
   |      - Optional first-passage connectivity (:connectivity) & CJS separation (:recapture)
   v
[Phase 6] Synthesis Exports & Consolidated Dashboard (export_dashboards)
          - Locally smoothed marine B-spline trajectories for all mark-recapture events
          - 100 x 100 km marine compass bearing roses & gradient taxis projections
          - Self-contained Leaflet dashboard: movement_dashboard.html
          - TethysBio movement summary: movement_stats.toml
          - Full results Zarr checkpoint: movement_results_checkpoint.zarr
```

---

## 2. Pipeline Execution Phases

### Phase 1: Data Ingestion, Marine Masking, and Domain Construction

Function: `load_movement_data(params)`

- **Encounter loading**: Reads empirical mark-recapture and acoustic telemetry tables (`tagging_file`, `hsi_file`, `sppoly_file`) or generates synthetic multi-segment histories when input paths are unset. Each encounter row records `release` and `recapture` unit indices, elapsed discrete steps `k` ($\Delta t / \delta t$, with $\delta t = 1/365.25\text{ yr}$ for daily resolution), decimal release year `rel_time`, biological traits (`cw`, `cc`, `sex`, `mat`), and observation modality (`datasource`, e.g. `"otn"` vs. `"bio"`).
- **Strict marine classification and shoreline geometry**: Signed bathymetric elevation (retaining positive terrestrial elevations prior to area-weighted resharding so coastal land elevations do not average with adjacent water into artificial shallow channels) is combined with the GMT/basemap `1.25`-arcminute (`~1.6-2.3 km`) high-resolution raster (`land_mask_resolution = "h"`, `land_mask_grid_minutes = 1.25`), treating both terrestrial land (`1`) and inland lakes (`>= 2`, such as Bras d'Or Lake) as non-marine (`true`). Four geometric invariants enforce strictly marine topology:
  - `land_mask_from_global_mask`: masks units whose centroids or interior half-radius vertices fall on non-ocean raster cells.
  - `sever_raster_land_edges!` and `sever_land_crossing_edges!`: sample 49 interior points along each adjacency edge in $\mathbf{W}$ and sever any edge crossing terrestrial isthmuses, causeways, islands, or headlands.
  - `mask_enclosed_inland_pockets!`: masks small enclosed lagoon or fjord-lake components (`< 2%` of the primary connected marine graph).
  - `clip_polygons_to_ocean!`: retracts coastal polygon vertices radially toward their marine centroid wherever a vertex ray enters non-ocean pixels, preventing rendered cells from protruding inland.
- **Homogeneous suitability in unsurveyed management zones**: Within surveyed footprints (`sppoly_file`), empirical HSI time series (`hsi_file`) parameterize local habitat gradients. Across in-depth marine waters inside configured regional polygons (`region_polygon_files`, such as `inputs/cfagulf.dat` covering CFA 19 and the Southern Gulf of St. Lawrence) where survey HSI estimates are absent, units receive a homogeneous suitability baseline (`0.50`). Because $\Delta\text{HSI} = 0$ across homogeneous-HSI neighbours, the advective kernel reduces locally to an unbiased marine random walk, permitting two-dimensional diffusive dispersal rather than forcing trajectories through narrow artificial channels.
- **Depth-window encoding (`depth_barrier_mode`)**:
  - `:hsi_only` (default): preserves full marine graph connectivity in $\mathbf{W}$ while clamping HSI on out-of-depth units (`depth < min_d` or `depth > max_d`) to `hsi_ood_floor` (`0.01`), discouraging traversal via $\exp(\gamma(H_v - H_u))$ without fragmenting continuous shelf basins.
  - `:hard`: removes out-of-depth units structurally from $\mathbf{W}$.
  - `:bridge`: reconnects in-depth marine components exclusively through navigable marine units (`!land_mask && !out_of_depth`), dropping tiny isolated pockets (`< 25` units) without bridging across land.

---

### Phase 2: Bayesian Transition Kernel Inference

Function: `fit_movement_models(loaded, params)`

Movement parameters are estimated in `Turing.jl` via `movement_sampler`, a random-walk Metropolis-Hastings sampler scaled to prior standard deviations (`mh_proposal_scale * prior_sd`). Forward $k$-step transition vectors $[(\mathbf{P}^\top)^k \mathbf{e}_{u_{\text{rel}}}]_{u_{\text{rec}}}$ are evaluated once per unique release unit and step horizon via `kstep_transition_cache`.

- **Prior and parameter mapping**: The model samples `mu_velocity` $v \sim \mathcal{N}_{[0, 0.95]}(0.3, 0.2^2)$, `mu_diffusion` $D \sim \mathcal{N}_{[0, \infty)}(0.1, 0.2^2)$, and `mu_gamma` $\gamma \sim \mathcal{N}(1.0, 1.0^2)$, mapped deterministically via `movement_alpha_rho(v, D)` to:

$$\alpha = \frac{v}{v + D + \epsilon} \in [0, 1], \qquad \rho = \frac{1}{1 + v + D + \epsilon} \in (0, 1],$$

where $\epsilon = 10^{-12}$ prevents division by zero when $v = D = 0$.
- **Stratified Acoustic (`OTN`) vs. Commercial Trap (`BIO`) inference**: When `obs_df.datasource` contains at least 15 fishery-independent acoustic detections (`"otn"`) and 15 commercial trap recaptures (`"bio"`), `fit_datasource_stratified_kernels!` fits independent posteriors to each modality. Because commercial recaptures require pot-haul effort at the destination cell, the `"bio"` observation likelihood weights destination probabilities by a spatially smoothed empirical trap-intensity vector $\mathbf{w}_{\text{eff}} \in [0.05, 1]^S$:

$$\mathbb{P}_{\text{bio}}(Y = u_{\text{rec}} \mid X_0 = u_{\text{rel}}, k) = \frac{w_{\text{eff}}(u_{\text{rec}})\, [\mathbf{P}^k]_{u_{\text{rel}}, u_{\text{rec}}}}{\sum_{u=1}^S w_{\text{eff}}(u)\, [\mathbf{P}^k]_{u_{\text{rel}}, u}}.$$

---

### Phase 3: Transition Kernel Extraction and Forward Agent Projection

Function: `extract_transition_kernels(loaded, fitted, params)`

For active marine units $u, v \in \{1, \dots, S\}$ with adjacency $W_{uv} > 0$, the one-step row-stochastic transition operator $\mathbf{P}$ is a convex mixture of local residence, undirected topological diffusion $\mathbf{T}_{\text{diff}}$, and habitat-gradient taxis $\mathbf{A}(H)$:

$$\mathbf{P} = (1 - \rho)\bigl[(1 - \alpha)\mathbf{T}_{\text{diff}} + \alpha \mathbf{A}(H)\bigr] + \rho \mathbf{I},$$

where $T_{\text{diff}, uv} = W_{uv} / \sum_w W_{uw}$ and $A_{uv}(H) \propto W_{uv} \exp(\gamma (H_v - H_u))$. Land units ($u \in \mathcal{L}$) have zero incoming marine weight and $P_{uu} = 1$.

At the conclusion of Phase 3, `movement_checkpoint.zarr` is written to the output store. When `:agent` is included in `model_modes`, `forward_project_agents` simulates individual trajectories forward from empirical release locations under static or 12-month climatological seasonal kernels (`dynamic_kernels = true`).

---

### Phase 4: Conditioned Trajectory Reconstruction and Bottlenecks

Function: `reconstruct_paths_and_diagnostics(loaded, kernels, params)`

Deterministic shortest-path routing (`A*`) is excluded because minimizing $-\log P_{uv}$ collapses multi-step diffusive transport onto single-file minimum-hop channels across homogeneous habitats. Instead, trajectories between observed endpoints $(u_{\text{rel}}, u_{\text{rec}})$ over $K$ steps are reconstructed by:

- **Conditioned forward-backward Markov bridges (`:bridge`, `predict_bridge_path`)**: Computes L1-normalized backward reachability vectors $\mathbf{b}_\tau = \mathbf{P}\mathbf{b}_{\tau+1}$ from $\mathbf{b}_{K+1} = \mathbf{e}_{u_{\text{rec}}}$ and samples forward transitions at step $\tau \in \{1, \dots, K\}$ according to:

$$\mathbb{P}(X_\tau = v \mid X_{\tau-1} = u, X_K = u_{\text{rec}}) = \frac{P_{uv}\, b_{\tau+1}(v)}{\sum_{w} P_{uw}\, b_{\tau+1}(w)}.$$

- **Maximum-a-posteriori Viterbi trellis (`:viterbi`, `predict_path`)**: Exact dynamic programming maximization of $\sum_{\tau=1}^K \log P_{u_{\tau-1}, u_\tau}$ subject to $u_0 = u_{\text{rel}}$ and $u_K = u_{\text{rec}}$.
- **Interior transit corridors and structural bottlenecks**: Averaging interior bridge marginals $\tau \in \{1, \dots, K-1\}$ excludes delta-function spikes at pinned release/recapture pots. Across unique non-stationary origin-destination pairs weighted by $w_{\text{pair}} = 1 + \log n_{\text{pair}}$ and propagated across posterior parameter draws, nodal transit intensity $C_{\text{domain}}(u)$ and net directed edge flux $F_{\text{net}}(u, v) = \max(0, F_{\text{domain}}(u, v) - F_{\text{domain}}(v, u))$ define the structural bottleneck index:

$$B(u) = \frac{C_{\text{domain}}(u)}{\max(1, \deg_{\text{marine}}(u))},$$

isolating marine units in the upper decile of $B(u)$ where elevated population flux passes through restricted topological degree.

---

### Phase 5: Circuit Theory, Canalization, and Connectivity Validation

Functions: `compute_advanced_diagnostics`, `execute_validation_analyses`

- **Circuit-theoretic current density (`:circuit`)**: Solves $\mathbf{L}_C \mathbf{V} = \mathbf{I}_{\text{ext}}$ with edge conductances $C_{uv} = W_{uv}\sqrt{H_u H_v}$ (evaluated on the 12-month annual mean HSI when `dynamic_kernels = true`) to quantify nodal current density $J(u)$ and stochastic pinch-points persistent across interannual HSI realizations ($P \ge 0.80$).
- **Outgoing step entropy and corridor canalization (`:validation`, `compute_corridor_canalization`)**: Combines normalized non-self marine step entropy $H(u) \in [0, 1]$ with interior Markov-bridge passage density $\tilde{C}(u) = C(u) / P_{99}(C)$ into the spatial canalization index:

$$K(u) = \sqrt{\min(1, \tilde{C}(u))}\,\bigl(1 - 0.5\,H(u)\bigr) \in [0, 1],$$

distinguishing narrow habitat-guided conduits (high $\tilde{C}$, low $H$) from broad diffusive dispersal zones (high $H$). Exported to `corridor_canalization_summary.csv`.
- **1-year HSI-weighted regional transition probabilities (`:validation`, `compute_stock_connectivity_matrix`)**: Coarse-grains the daily kernel promoted to an annual horizon ($\mathbf{P}^{365}$, or $H_{\text{yr}} = \text{round}(1 / \delta t_{\text{yr}})$ steps) using stationary suitability weights $w_u \propto H_u$ within each management region $R_a$:

$$M_{ab}^{(1\text{ yr})} = \frac{\sum_{u \in R_a} H_u \sum_{v \in R_b} [\mathbf{P}^{365}]_{uv}}{\sum_{u \in R_a} H_u}, \qquad \sum_{b} M_{ab}^{(1\text{ yr})} = 1.$$

Posterior means, standard deviations, and 95% credible intervals across MCMC draws are exported to `stock_connectivity_summary.csv` and `stock_connectivity_credible_intervals.csv`.
- **Posterior predictive KL calibration (`:validation`, `posterior_predictive_check`)**: Evaluates $\mathrm{KL}(p_{\text{obs}} \parallel p_{\text{pred}}^{(s)})$ across posterior draws $s$, avoiding sparse-lattice zero-inflation artifacts inherent to global cell-wise Brier scores.
- **First-passage connectivity (`:connectivity`) and CJS recapture separation (`:recapture`)**: Independently gated diagnostics computing exact regional mean first-passage times, arrival CDFs, modularity communities, minimum cuts, and Arnason-Schwarz survival/detection/movement separation.

---

### Phase 6: Spatial Rose/Taxis Diagnostics, All-Event Smoothing, and Dashboard Export

Function: `export_dashboards`

- **Smoothed all-event predicted trajectories (`reconstruct_all_event_trajectories`, `smooth_marine_polyline`)**: Reconstructs a conditioned Markov-bridge path for every row in `obs_df`, applies Lawler loop erasure and pre-spline moving-average relaxation, and fits a loop-free `5`-knot centripetal Catmull-Rom cubic spline where every control chord and curve span is verified against the `1.25`-arcminute ocean/lake raster (subdividing around peninsulas and damping coastal overshoot so zero segments cross land). Events are colour-coded across six intervals of time between mark and recapture (`< 1 month`, `1 mo - 1 yr`, `1 - 2 yr`, `2 - 3 yr`, `3 - 4 yr`, and `>= 4 yr`) and rendered on the default first tab with Mark/Release points, Recapture stars, directional arrowheads, and left-docked controls.
- **Spatially discretized $100 \times 100\text{ km}$ compass bearing roses (`compute_spatial_bearing_roses`)**: Partitions non-stationary displacements by release sector on a planar $100 \times 100\text{ km}$ grid (`min_events = 8`), computing circular mean bearing $\bar{\theta}$, resultant vector length $\bar{R} \in [0, 1]$, von Mises concentration $\hat{\kappa}$, and exact small-sample Rayleigh $Z = n\bar{R}^2$ uniformity $p$-values. Sector anchors are snapped to the nearest occupied marine unit centroid and rendered as interactive SVG polar roses directly at their `(lat, lon)` coordinates on the Leaflet map (`spatial_bearing_roses_100km.csv`).
- **Bathymetric, thermal, and habitat taxis projections (`analyze_gradient_taxis_projections`)**: Estimates local planar gradients $(\nabla z, \nabla T, \nabla\text{HSI})$ by weighted least squares over adjacent marine centroids (with minimum-norm directional derivative fallback on collinear coastal chains) and projects unit displacement vectors $\hat{\mathbf{d}}_i$ onto each gradient to test downslope ($\cos\psi_z$), along-isobath ($|\cos\psi_z| < \sin 25^\circ$), warmward ($\cos\psi_T$), and up-HSI ($\cos\psi_h$) taxis (`gradient_taxis_summary.csv`).
- **Multi-horizon dispersal scaling and null-model LPPD (`evaluate_dispersal_scaling_and_null`)**: Estimates the empirical mean-squared displacement scaling exponent $\beta$ in $\log\text{MSD} = \alpha + \beta\log\Delta t$ across liberty horizons (`<= 30 d` through `> 2 yr`) and computes the log pointwise predictive density gain $\Delta\text{LPPD}$ of the fitted habitat-advection kernel over an parameter-free isotropic marine random walk ($\alpha = 0$, $\rho = \hat{\rho}$).
- **TethysBio parameter coupling (`write_tethysbio_movement_stats`)**: Writes `movement_stats.toml` containing calibrated `max_move_m_per_year`, `movement_softmax_temperature` ($1 / \hat{\gamma}$), advection/residence means, and effective diffusivity $D_{\text{eff}}$ ($\text{km}^2\,\text{yr}^{-1}$) for downstream consumption by `TethysBio`.

---

## 3. Configuration Parameters

All settings reside in `TethysPathConfig` (`src/config.jl`), overridden in order by TOML config files (`configs/*.toml`), CLI flags, and programmatic `overrides`:

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_modes` | `Vector{Symbol}` | `[:telemetry, :telemetry_and_survey, :agent]` | Inference and simulation modes (`MODEL_MODE_CHOICES`) |
| `tagging_file` | `Union{Nothing, String}` | `nothing` | Mark-recapture / telemetry JLD2 or CSV input (`nothing` = synthetic) |
| `hsi_file` | `Union{Nothing, String}` | `nothing` | Habitat suitability Zarr or JLD2 input |
| `sppoly_file` | `Union{Nothing, String}` | `nothing` | Survey spatial polygon definitions |
| `surveydata_file` | `Union{Nothing, String}` | `nothing` | Optional paired survey abundance data |
| `land_source` | `Symbol` | `:landmask` | `:landmask`, `:depth`, `:polygons`, or `:none` |
| `land_mask_resolution` | `String` | `"l"` | GMT/basemap coastline tier (`"c"`, `"l"`, `"i"`, `"h"`, `"f"`; `"h"` in `snowcrab.toml`) |
| `land_mask_grid_minutes` | `Float64` | `5.0` | Raster spacing in arc-minutes (`1.25`, `2.5`, `5.0`, `10.0`; `1.25` in `snowcrab.toml`) |
| `land_polygon_files` | `Vector{String}` | `String[]` | Supplementary shoreline barrier polygons (e.g. `inputs/sable_island.geojson`) |
| `reshard_hex` | `Bool` | `false` | Reshard input polygons onto a regular hexagonal lattice |
| `hex_radius_km` | `Float64` | `10.0` | Hexagonal circumradius in km (`5.0` in `snowcrab.toml`) |
| `depth_range` | `Union{Nothing, Vector{Float64}}` | `nothing` | Benthic depth bounds `[min_m, max_m]` (`[25.0, 350.0]` in `snowcrab.toml`) |
| `depth_barrier_mode` | `Symbol` | `:hsi_only` | `:hsi_only`, `:hard`, or `:bridge` |
| `hsi_ood_floor` | `Float64` | `0.01` | Suitability floor applied to out-of-depth marine units under `:hsi_only` |
| `max_paths` | `Int` | `1000` | Spatially stratified tags reconstructed in `Paths & HSI` (`300` in `snowcrab.toml`) |
| `path_methods` | `Vector{Symbol}` | `[:bridge]` | `:bridge` and/or `:viterbi` (`PATH_METHOD_CHOICES`) |
| `dynamic_kernels` | `Bool` | `false` | Apply monthly/seasonal HSI series in trajectory and agent routing |
| `n_samples` | `Int` | `200` | Posterior MCMC draw count (`500` in `snowcrab.toml`) |
| `n_warmup` | `Int` | `100` | MCMC warmup iterations |
| `mh_proposal_scale` | `Float64` | `0.05` | Prior-scaled random-walk MH step multiplier |
| `diagnostics` | `Vector{Symbol}` | `[:circuit, :bottlenecks, :validation, :bayesian_ensemble]` | Active post-fit diagnostics (`DIAGNOSTIC_CHOICES`, plus `:connectivity`, `:recapture`) |
| `fpt_step_years` | `Float64` | `1/12` | Duration of one transition kernel step in years (`1/365.25` for daily snow crab) |
| `region_labels` | `Vector{String}` | `String[]` | Management region names (e.g. `["N-ENS", "S-ENS", "4X", "CFA19 / S.Gulf"]`) |
| `region_polygon_files` | `Vector{String}` | `String[]` | Management polygon boundary files (`*.dat`, `.geojson`, `.shp`) |
| `render_html` | `Bool` | `true` | Export consolidated interactive Leaflet dashboard (`movement_dashboard.html`) |
| `output_dir` | `String` | `"../lakehouse/TethysPath"` | Output directory for Zarr checkpoints, CSVs, TOML, and HTML |
| `resume_from_checkpoint` | `Bool` | `false` | Load `movement_checkpoint.zarr` and skip Phases 1-3 |
| `figures_only` | `Bool` | `false` | Load `movement_results_checkpoint.zarr` and regenerate Phase 6 outputs only |
| `tessellation_only` | `Bool` | `false` | Build Phase 1 marine mesh, render tessellation preview, and exit |

---

## 4. Output Store Artifacts (`lakehouse/TethysPath`)

```text
lakehouse/TethysPath/
|-- movement_checkpoint.zarr                  # Phase 1-3 checkpoint (loaded, fitted, kernels)
|-- movement_results_checkpoint.zarr          # Full Phase 1-6 checkpoint for --figures-only
|-- movement_dashboard.html                   # Leaflet dashboard shell (GitHub Pages ready)
|-- movement_dashboard.data.zz                # Zlib-compressed JSON spatial & trajectory sidecar
|-- movement_stats.toml                       # Calibrated movement parameters for TethysBio
|-- movement_path_metrics.csv                 # Stratified Markov-bridge path lengths & speeds
|-- movement_agent_space_use.csv              # Forward-projected agent visitation probabilities
|-- acoustic_vs_trap_inference.csv            # Stratified OTN acoustic vs. BIO trap posteriors
|-- spatial_bearing_roses_100km.csv           # 100 x 100 km circular statistics & Rayleigh tests
|-- gradient_taxis_summary.csv                # Bathymetric, thermal & HSI gradient projections
|-- corridor_canalization_summary.csv         # Per-unit step entropy H(u) & canalization K(u)
|-- stock_connectivity_summary.csv            # 1-year (P^365) HSI-weighted regional transitions
|-- stock_connectivity_credible_intervals.csv # Posterior 95% CIs on 1-year regional transitions
`-- posterior_predictive_summary.txt          # Posterior predictive KL divergence summary
```

When `:connectivity` or `:recapture` are added to `diagnostics`, `connectivity_*.csv` and `recapture_*.csv` tables are written alongside the artifacts above.

---

## 5. References

- Choi, J. S., Zisserson, B. M., & Cameron, B. J. (2022). Snow crab (*Chionoecetes opilio*) assessment and ecological dynamics on the Scotian Shelf. *DFO Canadian Science Advisory Secretariat Research Document*.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
- Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *Annals of Mathematical Statistics*, 22(1), 79-86.
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley.
- McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B. (2008). Using circuit theory to model connectivity in ecology, evolution, and conservation. *Ecology*, 89(10), 2712-2724.
- Turchin, P. (1998). *Quantitative Analysis of Movement: Measuring and Modeling Population Redistribution in Animals and Plants*. Sinauer Associates.
