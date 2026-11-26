---
title: >
  Inferring Snow Crab (*Chionoecetes opilio*) Movement, Dispersal, and
  Spatiotemporal Connectivity in Atlantic Canada: A Mechanistic Bayesian
  Advection-Diffusion and Probabilistic Network Framework
subtitle: >
  Integrating Acoustic Telemetry, Commercial Mark-Recapture, Conditioned Markov
  Bridges, and Circuit Theory within TethysPath
author: "TethysPath Group"
date: "2026-11-26"
status: >
  Technical specification synchronized with the active `TethysPath` implementation
  (`configs/snowcrab.toml`, `src/movement.jl`, `src/spatial_sources.jl`, and
  `src/pipeline.jl`). See `docs/movement_analysis.md` and `docs/TethysPathNotes.md`
  for CLI usage and Zarr checkpoint inspection.
format:
  html:
    toc: true
    toc-depth: 4
    number-sections: true
    code-fold: show
    code-summary: "Show Julia Code"
    theme: cosmo
    highlight-style: github
  pdf:
    toc: true
    number-sections: true
    latex-engine: lualatex
bibliography: ../references.bib
csl: ../chicago-author-date.csl
---

# Abstract

Spatial redistribution, directional taxis, and cross-boundary demographic connectivity govern how benthic marine populations respond to exploitation and ocean warming. Snow crab (*Chionoecetes opilio*) supports major crustacean fisheries across the Scotian Shelf and Southern Gulf of St. Lawrence (NAFO Divisions 4VWX and 4T), yet harvest management remains partitioned into static Crab Fishing Areas (N-ENS, S-ENS, 4X, and CFA 19 / Area 12) whose boundaries do not coincide with bathymetric conduits.

This document specifies the discrete-space Bayesian movement and connectivity framework implemented in `TethysPath` for Northwest Atlantic snow crab, separating empirical encounter observables from model-based transition and trajectory representations:

- **Shoreline-Constrained Marine Hexagonal Topology**: Discretization across ~3,487 active 5 km marine hexagonal units spanning the Scotian Shelf, Cabot Strait, CFA 19, and Southern Gulf of St. Lawrence (`25-350 m` benthic habitat band), with `1.25`-arcminute global ocean/lake raster masking, overland causeway/isthmus edge severing, enclosed inland-lake exclusion (e.g., Bras d'Or Lake), and radial shoreline clipping of coastal polygon vertices.
- **Survey Suitability and Unsurveyed Diffusive Domains**: Coupling of empirical Scotian Shelf Habitat Suitability Index (HSI) fields with a homogeneous baseline (`0.50`) across unsurveyed CFA 19 and Southern Gulf waters (`inputs/cfagulf.dat`), ensuring that movement across unsurveyed shelves reduces locally to two-dimensional marine diffusion rather than collapsing onto artificial narrow channels.
- **Modality-Stratified Bayesian Transition Kernels**: Estimation of discrete convex mixture operators $\mathbf{P} = (1 - \rho)[(1 - \alpha)\mathbf{T}_{\text{diff}} + \alpha\mathbf{A}(H)] + \rho\mathbf{I}$ in `Turing.jl`, both pooled and stratified between fishery-independent Ocean Tracking Network acoustic arrays (`datasource == "otn"`) and commercial pot-trap recaptures (`datasource == "bio"`, corrected for spatial trap-haul effort).
- **Conditioned Markov-Bridge Routing and Marine B-Spline Smoothing**: Exact forward-backward Markov bridge trajectory sampling conditioned on observed endpoints $(u_{\text{rel}}, u_{\text{rec}}, k)$, paired with raster-guarded quadratic B-spline smoothing across every mark-recapture event stratified by time at liberty (`< 1 month` to `>= 4 yr`).
- **Spatial Directional Roses, Environmental Taxis, and Corridor Canalization**: Evaluation of $100 \times 100\text{ km}$ compass bearing roses and Rayleigh tests anchored at marine coordinates, displacement vector projections onto local bathymetric ($\nabla z$), thermal ($\nabla T$), and habitat ($\nabla\text{HSI}$) gradients, outgoing step entropy $H(u) \in [0, 1]$, and spatial corridor canalization $K(u) \in [0, 1]$.
- **Circuit Pinch-Points and Annual Regional Exchange**: Kirchhoff current-density analysis ($\mathbf{L}_C\mathbf{V} = \mathbf{I}_{\text{ext}}$) and 1-year ($\mathbf{P}^{365}$) HSI-weighted transition probability matrices across N-ENS, S-ENS, 4X, and CFA 19 / Southern Gulf.

---

# 1. Ecological Context and Spatial Frame of Reference

## 1.1. Thermal Stenothermy and Bathymetric Partitioning

Snow crab (*Chionoecetes opilio*, Brachyura: Oregoniidae) is a cold-water stenotherm whose benthic distribution and metabolic scope are bounded by bottom temperature and seabed geomorphology [@Foyle_et_al_1989; @Kuhn_Choi_2011; @Choi_et_al_2022]:

- **Core Thermal Envelope**: Peak post-settlement densities occur between $-1.0^\circ\text{C}$ and $3.0^\circ\text{C}$ within the Cold Intermediate Layer (CIL) supplied by winter convection and Labrador Current / Gulf outflow [@Choi_2011; @Choi_et_al_2022]. Bottom temperatures above $6.0\text{--}7.0^\circ\text{C}$ elevate metabolic maintenance costs and restrict prolonged occupancy [@Foyle_et_al_1989].
- **Bathymetric Structure (`25-350 m`)**: Across the Scotian Shelf, shallow offshore banks (`< 60 m`; Sable Island, Western, and Banquereau Banks) separate cold interior basins (`150-300 m`; Emerald, Misaine, and Roseway Basins) linked by cross-shelf troughs, submarine gullies, and the Laurentian Channel margin [@Zwanenburg_et_al_2002; @Choi_et_al_2005a].
- **Ontogeny and Terminal Molt**: Determinate growth culminates in a terminal molt to morphometric maturity [@Comeau_Conan_1992; @Sainte-Marie_1993], followed by progressive carapace aging from clean hard shell (CC 2) through intermediate/mature hard shell (CC 3-4) to senescent shell (CC 5). Large hard-shelled males undertake seasonal and ontogenetic redistributions across basin slopes, whereas multiparous females exhibit higher spatial site fidelity [@Sainte-Marie_Hazel_1992; @Tremblay_et_al_2007; @Boudreau_et_al_2011].

## 1.2. Observational Modalities vs. Model-Based Representations

Empirical encounter data comprise two distinct observation processes whose sampling biases must not be conflated with underlying biological movement:

- **Acoustic Receiver Detections (`OTN`)**: Fixed Ocean Tracking Network hydrophone lines and gates detect acoustically tagged crabs independently of commercial fishing seasons, recording high-frequency residency and directional crossings near receiver arrays.
- **Commercial Pot-Trap Recaptures (`BIO`)**: T-bar anchor tags returned by the commercial fishery span multi-year liberties across broad shelf extents, but recapture locations are conditioned on the spatiotemporal distribution of pot-haul effort.

Endpoints $(u_{\text{rel}}, u_{\text{rec}})$ and elapsed durations $\Delta t$ are direct empirical observables. Intermediate trajectories, nodal transit densities, step entropies, and annual regional transition matrices are model-based expectations conditioned on the discrete marine graph and fitted Markov transition kernel. Accordingly, endpoint-derived straight-line ratios (such as unconditioned chord-to-path "tortuosity" or behavioural bout labels inferred without continuous intermediate fixes) are omitted in favour of conditioned Markov-bridge ensembles and explicit gradient projections.

---

# 2. Marine Spatial Discretization and Boundary Invariants

## 2.1. Shoreline-Clipped 5 km Hexagonal Lattice

The spatial domain spans NAFO Divisions 4VWX and southern 4T on a planar 5 km hexagonal tessellation (~3,487 active marine units within the `25-350 m` habitat band). Preventing artificial overland routing across narrow Maritime barriers (Canso Causeway, Chignecto Isthmus, Prince Edward Island, Magdalen Islands, Sable Island, and Bras d'Or Lake in central Cape Breton) requires four coupled geometric constraints:

- **Signed Elevation Transfer**: Bathymetric elevation $z_{\text{elev}} \in \mathbb{R}$ (positive on land, negative in water) is area-transferred prior to zero-clamping so coastal units overlapping steep shorelines are not averaged with zero-clamped land into synthetic `30-60 m` channels.
- **High-Resolution Ocean/Lake Raster Masking (`land_mask_from_global_mask`)**: Evaluates the `1.25`-arcminute (`~1.6-2.3 km`) GMT/basemap raster (`resolution = "h"`), classifying all non-zero codes (`1` = terrestrial land, `>= 2` = inland lakes/ponds) as non-ocean (`true`) and screening interior half-radius polygon vertices.
- **Overland Edge Severing and Pocket Exclusion (`sever_raster_land_edges!`, `mask_enclosed_inland_pockets!`)**: Tests 49 interior points along each adjacency chord $\overline{\mathbf{c}_u \mathbf{c}_v}$ against the `1.25`-arcminute raster and supplementary GeoJSON polygons (`inputs/sable_island.geojson`), zeroing any edge intersecting non-ocean pixels:

$$W_{\text{marine}, uv} = \begin{cases} W_{uv} & \text{if } u, v \in \mathcal{M} \text{ and } \overline{\mathbf{c}_u \mathbf{c}_v} \subset \Omega_{\text{ocean}}, \\ 0 & \text{otherwise}, \end{cases}$$

and masking enclosed inland pockets smaller than $2\%$ of the primary ocean component.
- **Shoreline Polygon Clipping (`clip_polygons_to_ocean!`)**: Retracts individual coastal hexagon vertices radially toward their marine centroid $(\mathbf{c}_u)$ wherever the vertex ray enters non-ocean pixels, ensuring rendered choropleth cells conform to the coastline.

## 2.2. Surveyed vs. Unsurveyed Habitat Suitability

Within the Scotian Shelf trawl survey domain (`inputs/sppoly.jld2`), empirical monthly and climatological HSI fields (`inputs/hsi.zarr`) supply local suitability $H_u \in [0, 1]$. Across CFA 19 and the Southern Gulf of St. Lawrence (`inputs/cfagulf.dat`), comparable trawl-derived HSI surfaces are unavailable in the Maritimes store. Assigning a uniform baseline $H_u = 0.50$ across all in-depth (`25-350 m`) marine units inside `cfagulf.dat` yields $\Delta H_{uv} = H_v - H_u = 0$ between unsurveyed neighbours, so local transitions reduce to unbiased marine diffusion ($A_{uv}(H) = T_{\text{diff}, uv}$) rather than funneling trajectories along artificial boundary artifacts. Out-of-depth marine units under `depth_barrier_mode = :hsi_only` receive $H_u \le h_{\text{floor}} = 0.01$, discouraging traversal over shallow banks (`< 25 m`) or deep abyssal slopes (`> 350 m`) without severing topological reachability.

---

# 3. Statistical Movement Kernel and Conditioned Routing

## 3.1. Convex Mixture Markov Operator

At daily time resolution ($\delta t = 1/365.25\text{ yr}$), single-step movement on the marine graph is governed by the sparse row-stochastic transition matrix $\mathbf{P} \in [0, 1]^{S \times S}$:

$$\mathbf{P} = (1 - \rho)\bigl[(1 - \alpha)\mathbf{T}_{\text{diff}} + \alpha \mathbf{A}(H)\bigr] + \rho \mathbf{I},$$

where $T_{\text{diff}, uv} = W_{\text{marine}, uv} / \sum_{w} W_{\text{marine}, uw}$ is the isotropic topological random walk and $A_{uv}(H) = W_{\text{marine}, uv}\exp(\gamma(H_v - H_u)) / \sum_w W_{\text{marine}, uw}\exp(\gamma(H_w - H_u))$ is the softmax habitat-taxis kernel. Parameters $(v, D, \gamma)$ are sampled in `Turing.jl` under:

$$\alpha = \frac{v}{v + D + \epsilon}, \qquad \rho = \frac{1}{1 + v + D + \epsilon}, \qquad \mathcal{L}(v, D, \gamma) = \prod_{i=1}^{N} \bigl[\mathbf{P}^{k_i}\bigr]_{u_{\text{rel}, i},\, u_{\text{rec}, i}},$$

with separate stratified fits (`fit_datasource_stratified_kernels`) isolating fishery-independent acoustic detections (`OTN`) from effort-weighted commercial pot recaptures (`BIO`).

## 3.2. Conditioned Markov Bridges and Locally Smoothed All-Event Trajectories

Given an observed transition from $u_{\text{rel}}$ to $u_{\text{rec}}$ over $K$ daily steps, the conditioned Markov bridge samples interior states sequentially from the exact forward-backward conditional distribution:

$$\mathbb{P}(X_\tau = v \mid X_{\tau-1} = u, X_K = u_{\text{rec}}) = \frac{P_{uv}\, b_{K-\tau+1}(v)}{\sum_{w} P_{uw}\, b_{K-\tau+1}(w)}, \qquad \mathbf{b}_{r+1} = \frac{\mathbf{P}\mathbf{b}_r}{\|\mathbf{P}\mathbf{b}_r\|_1}, \quad \mathbf{b}_1 = \mathbf{e}_{u_{\text{rec}}}.$$

In homogeneous-HSI domains ($\Delta H = 0$), this operator samples genuine two-dimensional diffusive Brownian bridges across open marine shelves; across heterogeneous Scotian Shelf basins, positive $\gamma$ concentrates bridge mass along cold, high-suitability troughs.

To visualize movement structure across the complete encounter dataset ($N \approx 5{,}066$ events) without discrete $60^\circ$ hexagonal corner artifacts, `reconstruct_all_event_trajectories` shares the backward cascade $\{\mathbf{b}_r\}$ across events terminating at each recapture unit and applies two-pass Chaikin quadratic B-spline subdivision (`smooth_marine_polyline`), accepting each corner shortcut $\overline{\mathbf{b}_{\text{in}}\mathbf{a}_{\text{out}}}$ only when all interior raster substeps remain strictly oceanic (`data == 0`). Events are stratified by elapsed time at liberty (`< 1 month`, `1 mo - 1 yr`, `1 - 2 yr`, `2 - 3 yr`, `3 - 4 yr`, `>= 4 yr`) and rendered with interactive click-to-highlight path isolation.

---

# 4. Spatial Orientation, Corridor Canalization, and Stock Connectivity

## 4.1. Discretized $100 \times 100\text{ km}$ Compass Roses and Gradient Taxis

Shelf-wide circular averages obscure local topographic steering around Cape Breton, Sable Bank, and Emerald Basin. `compute_spatial_bearing_roses` partitions non-stationary displacements into $100 \times 100\text{ km}$ planar cells, evaluating sector resultant length $\bar{R}_c = \|\frac{1}{n_c}\sum_{j \in c} (\cos\theta_j, \sin\theta_j)\|_2$, circular mean bearing $\bar{\theta}_c$, von Mises concentration $\hat{\kappa}_c$, and Rayleigh uniformity $Z_c = n_c \bar{R}_c^2$ ($p_c$), anchoring each SVG polar rose at the nearest occupied marine unit centroid on the map.

Complementing geographic compass bearings, `analyze_gradient_taxis_projections` estimates local spatial gradients $\nabla f(u)$ ($f \in \{z, T, \text{HSI}\}$) across adjacent marine centroids via inverse-distance weighted least squares and projects unit displacement vectors $\hat{\mathbf{d}}_i$ onto $\nabla f(u_{\text{rel}, i}) / \|\nabla f(u_{\text{rel}, i})\|_2$ to test downslope ($\cos\psi_z > 0$), contour-following ($|\cos\psi_z| < \sin 25^\circ$), warmward ($\cos\psi_T$), and up-suitability ($\cos\psi_h$) movement.

## 4.2. Outgoing Step Entropy, Canalization, and Kirchhoff Pinch-Points

At each marine unit $u$ with marine degree $d_u \ge 2$, normalized non-self step entropy $H(u) = -(\log d_u)^{-1} \sum_{v \ne u} \tilde{P}_{uv}\log\tilde{P}_{uv} \in [0, 1]$ measures local directional dispersion, while normalized interior bridge passage $\tilde{C}(u) = \min(1, C(u) / P_{99}(C))$ measures empirical transit load. Their joint canalization index:

$$K(u) = \sqrt{\tilde{C}(u)}\,\bigl(1 - 0.5\,H(u)\bigr) \in [0, 1],$$

together with structural bottleneck centrality $B(u) = C_{\text{domain}}(u) / \max(1, d_u)$ and steady-state Kirchhoff current density $J(u)$ ($\mathbf{L}_C\mathbf{V} = \mathbf{I}_{\text{ext}}$), distinguishes constrained bathymetric corridors (Cabot Strait, St. Anns / Laurentian margin, and cross-shelf gullies) from broad diffusive banks and Gulf plains.

## 4.3. Annual ($\mathbf{P}^{365}$) Management-Area Connectivity

Single-day transition probabilities ($\mathbf{P}^1$) across broad administrative polygons are dominated by $> 99.9\%$ self-retention near interior cells and provide negligible demographic contrast. Promoting the daily operator to a 1-year horizon ($\mathbf{P}^{365}$) and weighting source units by local habitat suitability $H_u$ yields the annual regional exchange matrix across N-ENS, S-ENS, 4X, and CFA 19 / Southern Gulf:

$$M_{ab}^{(1\text{ yr})} = \frac{\sum_{u \in R_a} H_u \sum_{v \in R_b} [\mathbf{P}^{365}]_{uv}}{\sum_{u \in R_a} H_u},$$

propagated across posterior MCMC draws to report 95% credible intervals on annual retention and cross-boundary exchange.

---

# 5. Reproducible Execution and Artifacts

```julia
using TethysPath

# Execute full snow crab inference, diagnostics, and dashboard export
cfg = load_config(config_path = "configs/snowcrab.toml")
res = run_movement_analysis(cfg)
```

All outputs are persisted to `lakehouse/TethysPath/`, including Zarr checkpoints (`movement_checkpoint.zarr`, `movement_results_checkpoint.zarr`), `movement_dashboard.html`, `movement_stats.toml`, and diagnostic CSV summaries (`acoustic_vs_trap_inference.csv`, `spatial_bearing_roses_100km.csv`, `gradient_taxis_summary.csv`, `corridor_canalization_summary.csv`, `stock_connectivity_summary.csv`, `stock_connectivity_credible_intervals.csv`, and `movement_path_metrics.csv`).

---

# References

::: {#refs}
:::
