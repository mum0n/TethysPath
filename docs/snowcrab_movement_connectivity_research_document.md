---
title: >
  Inferring Snow Crab (*Chionoecetes opilio*) Movement, Dispersal, and
  Spatiotemporal Connectivity in Atlantic Canada: A Mechanistic Bayesian
  Advection-Diffusion and Probabilistic Network Framework
subtitle: >
  Consolidated Research Specification, Technical Pipeline Reference, and
  Empirical Synthesis for TethysPath.jl
author: "TethysPath Group"
date: "2026-11-27"
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
bibliography: references.bib
---

# Abstract

Spatial redistribution, habitat-directed taxis, and cross-boundary demographic exchange determine how benthic marine populations respond to exploitation and ocean warming. Snow crab (*Chionoecetes opilio*) supports major crustacean fisheries across the Scotian Shelf (NAFO Divisions 4VWX) and Southern Gulf of St. Lawrence (NAFO Division 4T), yet stock assessments remain partitioned into static Crab Fishing Areas (N-ENS, S-ENS, 4X, and CFA 19 / Southern Gulf) that treat administrative boundaries as demographically closed. This document consolidates the ecological framing, simplified model structure, underlying assumptions and weaknesses, empirical results, management implications, and technical pipeline reference of `TethysPath.jl`.

Across $N = 5{,}066$ mark-recapture and acoustic telemetry encounter intervals ($3{,}489$ unique individuals) mapped onto $3{,}487$ shoreline-clipped 5 km marine hexagonal units (`25-350 m` bathymetric band), movement is modeled by a single daily discrete-space Markov transition matrix $\mathbf{P} \in [0, 1]^{S \times S}$ decomposing each daily step into three mutually exclusive outcomes: staying in place ($\rho$), diffusing isotropically to a marine neighbour ($(1 - \rho)(1 - \alpha)$), or moving preferentially up local Habitat Suitability Index (HSI) gradients ($(1 - \rho)\alpha$, weighted by $\exp(\gamma\Delta\text{HSI})$). Pooled inference yields strong habitat-guided advection ($\hat{\alpha} = 0.821$, 95% CI $[0.764, 0.856]$; $\hat{\gamma} = 4.42$, $[3.74, 4.98]$) and high daily site residence ($\hat{\rho} = 0.733$, $[0.625, 0.836]$). Stratifying fishery-independent Ocean Tracking Network acoustic detections (`n = 1,559`, `84.4%` stationary within 5 km) from effort-weighted commercial pot-trap recaptures (`n = 3,507`, `20.0%` stationary) isolates gear-sampling bias while preserving consistent habitat-directed advection ($\hat{\alpha} = 0.860$ vs. $0.822$). Empirically, `62.7%` of non-stationary displacements align along local isobaths, cross-slope movement reverses seasonally from winter upslope ($\overline{\cos\psi_z} = -0.218$, $p = 0.003$) to summer-autumn downslope ($p < 10^{-5}$), and `14` of `19` spatial $100 \times 100\text{ km}$ compass bearing sectors show significant local topographic steering ($p < 0.05$) obscured by shelf-wide circular averages ($\bar{R} = 0.113$). Annual ($\mathbf{P}^{365}$) HSI-weighted transition matrices indicate asymmetric cross-boundary exchange—including `13.3%` (`[11.1%, 16.0%]`) annual export from N-ENS into S-ENS, `4.6%` (`[4.0%, 5.7%]`) from N-ENS into CFA 19 / Southern Gulf, and `16.3%` (`[12.9%, 19.4%]`) from 4X into S-ENS—highlighting both the demographic openness of management zones and the structural limitations that must be addressed in next-generation spatial assessments.

---

# 1. Introduction

## 1.1. Ecological Context and Management Scale

The snow crab (*Chionoecetes opilio*, Brachyura: Oregoniidae) is a cold-water stenotherm distributed across continental shelves of the North Pacific, Bering Sea, Arctic Ocean, and Northwest Atlantic [@Comeau_et_al_1998; @Choi_2011]. Across the Scotian Shelf (NAFO Divisions 4VWX), the Laurentian Channel, and the Southern Gulf of St. Lawrence (NAFO Division 4T), snow crab serves both as a dominant benthic predator-scavenger following the early-1990s demersal finfish regime shift [@Choi_et_al_2004; @Choi_et_al_2005a; @Bundy_et_al_2004] and as the foundation of regional crustacean fisheries [@Choi_Zisserson_2011; @Cadigan_et_al_2017; @DFO_2022_SnowCrab].

Benthic distribution, metabolism, and reproductive timing are constrained by bottom temperature and seabed geomorphology [@Foyle_et_al_1989; @Kuhn_Choi_2011; @Choi_et_al_2022]:

- **Thermal constraints**: Peak post-settlement densities occupy $-1.0^\circ\text{C}$ to $3.0^\circ\text{C}$ within the Cold Intermediate Layer (CIL) [@Choi_2011; @Choi_et_al_2022]. Bottom temperatures exceeding $6.0\text{--}7.0^\circ\text{C}$ elevate maintenance respiration and reduce scope for growth [@Foyle_et_al_1989], while embryonic brooding shifts from a 1-year cycle in warmer waters ($> 2.0^\circ\text{C}$) to a 2-year cycle below $1.0^\circ\text{C}$ [@Sainte_Marie_1993; @Webb_et_al_2007; @Kuhn_Choi_2011].
- **Bathymetric and substrate structure**: Shallow offshore banks (`< 60 m`; Banquereau, Sable Island, and Western Banks) separate cold depositional basins (`150-300 m`; Emerald, Misaine, and Roseway Basins) linked by cross-shelf troughs, submarine canyons, and the Laurentian Channel margin [@Zwanenburg_et_al_2002; @Choi_et_al_2005a]. Early benthic instars settle on structurally complex gravel and cobble substrates [@Sainte_Marie_Lafrance_2002; @Comeau_et_al_1998], whereas adolescents and adults redistribute across deeper mud and silt basins [@Comeau_Conan_1992].
- **Ontogeny and sexual dimorphism**: Determinate growth ends in a terminal molt to morphometric maturity [@Comeau_Conan_1992; @Elner_Beninger_1995], followed by carapace condition (CC) progression from clean hard shell (CC 2) through mature hard shell (CC 3-4) to senescent shell (CC 5). Large hard-shelled males undertake seasonal foraging and mating redistributions along basin slopes, whereas egg-bearing females maintain narrower spatial ranges [@Sainte_Marie_Hazel_1992; @Tremblay_et_al_2007; @Boudreau_et_al_2011].

Despite continuous benthic habitat across deep channels, commercial management partitions the resource into discrete administrative zones: Northern Eastern Nova Scotia (N-ENS; CFAs 20-22), Southern Eastern Nova Scotia (S-ENS; CFAs 23-24), Southwestern Nova Scotia (CFA 4X), and Southern Gulf of St. Lawrence / Western Cape Breton (CFA 19 / Area 12) [@Choi_2020; @DFO_2022_SnowCrab; @Cadigan_et_al_2017]. Treating these polygons as closed populations risks misattributing immigration or emigration to local recruitment or fishing mortality.

## 1.2. Separating Empirical Observables from Model Representations

Quantitative movement analysis requires keeping empirical observations distinct from model-dependent constructs:

- **Empirical observables**: Mark/release and recapture coordinates, elapsed time between mark and recapture ($\Delta t$), carapace width (CW), carapace condition (CC), sex/maturity state, and observation modality—specifically fishery-independent Ocean Tracking Network (`OTN`) acoustic receiver detections versus commercial pot-trap (`BIO`) tag returns.
- **Model-dependent representations**: Discrete-time Markov transition probabilities $\mathbf{P}$, conditioned forward-backward Markov-bridge trajectories and interior transit corridors, locally smoothed marine splines, 1-year effective dispersal area $A_{\text{eff}}(u)$, corridor canalization $K(u)$, interior Kirchhoff circuit current density $J^{\text{int}}(u)$, and multi-step regional exchange matrices $\mathbf{P}^{365}$.

Because mark-recapture data supply endpoints without intermediate positional fixes, unconditioned chord-to-path "tortuosity" ratios, endpoint-inferred behavioural bouts, and deterministic shortest-path (`A*`) routes conflate geometric lattice artifacts with animal behaviour. `TethysPath.jl` replaces those heuristics with a single coherent probabilistic movement operator on a shoreline-constrained marine graph, paired with direct empirical gradient projections and modality-stratified Bayesian inference.

---

# 2. Data and Marine Spatial Discretization

## 2.1. Telemetry and Habitat Inputs

The Scotian Shelf and Southern Gulf configuration (`configs/snowcrab.toml`) ingests three primary inputs consolidated under `inputs/`:

- **Mark-recapture and acoustic encounters (`inputs/tagging.jld2`)**: Yields $N = 5{,}066$ valid marine encounter intervals across $3{,}489$ unique individuals after excluding $302$ records with inverted dates (recapture preceding release) and $17$ records with unreachable coordinate/datum errors. Records partition into $n = 1{,}559$ acoustic receiver detections (`OTN`, $173$ tags) and $n = 3{,}507$ commercial pot-trap recaptures (`BIO`, $3{,}316$ tags).
- **Survey Habitat Suitability Index (`inputs/hsi.zarr`, `inputs/sppoly.jld2`)**: Supplies $324$ monthly and climatological HSI slices across the surveyed Scotian Shelf footprint.
- **Management region and shoreline boundaries**: Regional polygons (`inputs/cfanorth.dat`, `inputs/cfasouth.dat`, `inputs/cfa4x.dat`, `inputs/cfagulf.dat`) define `N-ENS`, `S-ENS`, `4X`, and `CFA19 / S.Gulf`, supplemented by `inputs/sable_island.geojson` and the GMT/basemap `1.25`-arcminute (`"h"`) global ocean/lake raster.

## 2.2. Shoreline-Constrained 5 km Hexagonal Lattice

Discretizing Maritime Canada at 5 km hexagonal circumradius ($8{,}003$ initial units before depth and footprint pruning; $3{,}487$ active marine units retained within `25-350 m`) requires strict topological constraints to prevent artificial overland shortcuts across Canso Causeway, Chignecto Isthmus, Prince Edward Island, Sable Island, and Bras d'Or Lake in central Cape Breton:

- **Signed elevation transfer**: Bathymetric elevation $z_{\text{elev}} \in \mathbb{R}$ (positive on land, negative in water, from ETOPO 2022) is area-transferred onto fine hexagons prior to zero-clamping so terrestrial elevations do not average with adjacent deep channels into artificial shallow marine cells.
- **High-resolution ocean/lake raster masking (`land_mask_from_global_mask`)**: Evaluates the `1.25`-arcminute (`~1.6-2.3 km`) `"h"` global mask, classifying all non-zero raster codes (`1` = terrestrial land, `>= 2` = inland lakes and ponds) as non-marine (`true`) and screening interior half-radius polygon vertices (`2,225 / 8,003` units masked).
- **Overland edge severing and enclosed-pocket exclusion (`sever_raster_land_edges!`, `mask_enclosed_inland_pockets!`)**: Tests 49 interior checkpoints along every adjacency chord $\overline{\mathbf{c}_u\mathbf{c}_v}$ and zeros edges intersecting non-ocean pixels (`15` overland/isthmus edges severed):

$$W_{\text{marine}, uv} = \begin{cases} W_{uv} & \text{if } u, v \in \mathcal{M} \text{ and } \overline{\mathbf{c}_u\mathbf{c}_v} \subset \Omega_{\text{ocean}}, \\ 0 & \text{otherwise}, \end{cases}$$

followed by masking enclosed inland lake/lagoon components smaller than $2\%$ of the primary marine component (`105` enclosed units masked, eliminating Bras d'Or Lake shortcuts through Cape Breton).
- **Radial shoreline clipping (`clip_polygons_to_ocean!`)**: Retracts individual coastal hexagon vertices radially toward their marine centroid wherever a vertex ray enters non-ocean pixels (`196` coastal hexagons clipped), guaranteeing zero polygon vertices protruding onto land.

## 2.3. Survey Suitability vs. Unsurveyed Diffusive Domains

Inside the Scotian Shelf trawl survey footprint (`1,250` active surveyed units), empirical HSI fields $H_u \in [0, 1]$ drive habitat-gradient taxis. Across CFA 19 and the Southern Gulf of St. Lawrence (`2,237` active in-depth units inside `inputs/cfagulf.dat`), compatible Maritimes trawl-derived HSI surfaces do not exist. Rather than extrapolating narrow edge channels, unsurveyed in-depth marine units receive a homogeneous suitability baseline ($H_u = 0.50$). Because $\Delta H_{uv} = H_v - H_u = 0$ across homogeneous-HSI neighbours, the transition kernel reduces locally to two-dimensional marine diffusion across the Gulf shelf. Out-of-depth marine units (`depth < 25 m` or `depth > 350 m`) under `depth_barrier_mode = :hsi_only` retain graph connectivity with HSI floored at `hsi_ood_floor = 0.01`, discouraging traversal across warm shallow banks or abyssal slopes without artificially fragmenting contiguous shelf basins.

---

# 3. Model Structure

At its core, `TethysPath.jl` is a **three-parameter discrete-space, discrete-time Markov chain** defined on the marine hexagonal graph $\mathcal{G} = (\mathcal{M}, \mathcal{E})$ at a daily clock ($\delta t = 1/365.25\text{ yr}$). Every downstream output—trajectories, corridors, step entropy, bottlenecks, and 1-year regional exchange—is derived directly from this single fitted transition matrix $\mathbf{P}$ rather than from separate statistical sub-models.

## 3.1. The Daily Movement Rule

Consider a snow crab located in marine cell $u \in \{1, \dots, S\}$ on day $t$, with navigable marine neighbours $\mathcal{N}(u) = \{v : W_{\text{marine}, uv} > 0\}$. Between day $t$ and day $t + 1$, exactly one of three mutually exclusive events occurs:

1. **Stay in place** with probability $\rho \in (0, 1)$ (local site residence).
2. **Diffuse randomly** with probability $(1 - \rho)(1 - \alpha)$, stepping uniformly to any adjacent marine cell $v \in \mathcal{N}(u)$ with probability $T_{\text{diff}, uv} = 1 / |\mathcal{N}(u)|$.
3. **Move along the habitat gradient** with probability $(1 - \rho)\alpha$, stepping to neighbour $v \in \mathcal{N}(u)$ with softmax probability weighted by the local suitability difference $\Delta H_{uv} = H_v - H_u$:

$$A_{uv}(H) = \frac{W_{\text{marine}, uv}\,\exp\bigl(\gamma (H_v - H_u)\bigr)}{\sum_{w \in \mathcal{N}(u)} W_{\text{marine}, uw}\,\exp\bigl(\gamma (H_w - H_u)\bigr)}.$$

Combining these three events defines the $S \times S$ row-stochastic daily transition matrix $\mathbf{P}$ ($\sum_v P_{uv} = 1$):

$$\mathbf{P} = \underbrace{\rho\,\mathbf{I}}_{\text{stay}} \;+\; (1 - \rho)\Bigl[\,\underbrace{(1 - \alpha)\,\mathbf{T}_{\text{diff}}}_{\text{random diffusion}} \;+\; \underbrace{\alpha\,\mathbf{A}(H)}_{\text{habitat-guided taxis}}\,\Bigr].$$

This formulation is the discrete-graph analogue of a continuous Advection-Diffusion-Reaction (ADR) partial differential equation [@Turchin_1998; @Wikle_2003], guaranteeing non-negativity, exact probability conservation, and zero overland leakage on arbitrary irregular coastlines. When $\gamma > 0$, movement biases toward higher-suitability neighbours; when $\gamma = 0$ or across homogeneous-HSI waters ($\Delta H_{uv} = 0$), $\mathbf{A}(H) = \mathbf{T}_{\text{diff}}$ and movement simplifies automatically to an unbiased two-dimensional marine random walk.

## 3.2. Parameterization, Priors, and Multi-Day Observation Model

To keep priors physically interpretable and ensure $\alpha \in [0, 1]$ and $\rho \in (0, 1]$ without constrained simplex transforms, `Turing.jl` [@Gelman_et_al_2013] samples three unconstrained or truncated latent quantities $(v, D, \gamma)$:

$$v \sim \mathcal{N}_{[0, 0.95]}(0.3, 0.2^2), \qquad D \sim \mathcal{N}_{[0, \infty)}(0.1, 0.2^2), \qquad \gamma \sim \mathcal{N}(1.0, 1.0^2),$$

mapped deterministically to the kernel probabilities via `movement_alpha_rho`:

$$\alpha = \frac{v}{v + D + \epsilon}, \qquad \rho = \frac{1}{1 + v + D + \epsilon}, \qquad \epsilon = 10^{-12}.$$

| Symbol | Domain / Units | Physical Meaning |
|:-------|:---------------|:-----------------|
| $S$ | $3{,}487$ cells (`5 km` hex) | Active marine units in the `25-350 m` habitat band |
| $\delta t$ | $1 / 365.25\text{ yr}$ (`1 day`) | Discrete transition clock of $\mathbf{P}$ |
| $H_u$ | $[0, 1]$ (dimensionless) | Local Habitat Suitability Index (`0.50` in unsurveyed `cfagulf.dat`, `<= 0.01` out-of-depth) |
| $v, D$ | $[0, 0.95]$, $[0, \infty)$ | Latent advective and diffusive rate parameters |
| $\alpha$ | $[0, 1]$ | Fraction of departing movement governed by habitat gradient vs. isotropic diffusion |
| $\rho$ | $(0, 1]$ | Daily probability of remaining in the current 5 km cell |
| $\gamma$ | $\mathbb{R}$ | Softmax sensitivity to $\Delta H_{uv} = H_v - H_u$ (inverse temperature $1/\gamma$ in `TethysBio`) |
| $w_{\text{eff}}(u)$ | $[0.05, 1]$ | Smoothed relative commercial pot-trap effort weight (`BIO` stratum only) |

For an individual marked in cell $u_{\text{rel}, i}$ and recaptured $k_i$ days later in cell $u_{\text{rec}, i}$, summing over all possible unobserved intermediate paths of length $k_i$ reduces analytically to the $(u_{\text{rel}, i}, u_{\text{rec}, i})$ entry of the $k_i$-step matrix power $\mathbf{P}^{k_i}$ (computed by sparse forward vector iteration $\mathbf{p}_{t+1} = \mathbf{P}^\top \mathbf{p}_t$ via `kstep_transition_cache`):

$$u_{\text{rec}, i} \mid u_{\text{rel}, i}, k_i \;\sim\; \operatorname{Categorical}\!\Bigl(\bigl[\mathbf{P}^{k_i}\bigr]_{u_{\text{rel}, i},\, :}\Bigr), \qquad \log \mathcal{L}(v, D, \gamma) = \sum_{i=1}^{N} \log \bigl[\mathbf{P}^{k_i}\bigr]_{u_{\text{rel}, i},\, u_{\text{rec}, i}}.$$

In the modality-stratified analysis (`fit_datasource_stratified_kernels!`), acoustic detections (`OTN`) use the unweighted transition likelihood above, whereas commercial trap recaptures (`BIO`) weight destination probabilities by the local trap-effort offset $w_{\text{eff}}(u)$ to account for non-uniform fleet coverage:

$$\mathbb{P}_{\text{bio}}(Y = u_{\text{rec}} \mid X_0 = u_{\text{rel}}, k) = \frac{w_{\text{eff}}(u_{\text{rec}})\, [\mathbf{P}^k]_{u_{\text{rel}}, u_{\text{rec}}}}{\sum_{u=1}^S w_{\text{eff}}(u)\, [\mathbf{P}^k]_{u_{\text{rel}}, u}}.$$

Posterior draws are generated via prior-scaled random-walk Metropolis-Hastings (`movement_sampler`, proposal width `mh_proposal_scale * prior_sd`), and model fit is evaluated by posterior predictive Kullback-Leibler divergence $\mathrm{KL}(p_{\text{obs}} \parallel p_{\text{pred}}^{(s)})$ [@Kullback_Leibler_1951].

## 3.3. Unified Derived Operators (From $\mathbf{P}$ to Trajectories, Corridors, and Connectivity)

Once $\mathbf{P}$ is estimated, every spatial diagnostic follows directly from the same operator:

```text
                    Fitted Daily Transition Matrix P (S x S)
                                       |
       +---------------+---------------+---------------+---------------+
       |               |               |               |               |
       v               v               v               v               v
  Conditioned      1-Yr Effective   Structural      Annual Matrix   Forward IBM
  Markov Bridge    Dispersal Area   Bottlenecks     P^365 (HSI-     Agent Space
  & Corridors      A_eff(u) & K(u)  B(u) & Circuit  Weighted)       Use (:agent)
```

- **Conditioned Markov bridges and interior transit corridors (`predict_bridge_path`)**: Given observed endpoints $(u_{\text{rel}}, u_{\text{rec}})$ separated by $K$ steps, backward reachability vectors $\mathbf{b}_{r+1} = \mathbf{P}\mathbf{b}_r / \|\mathbf{P}\mathbf{b}_r\|_1$ ($\mathbf{b}_1 = \mathbf{e}_{u_{\text{rec}}}$) yield the exact conditional forward step distribution:

$$\mathbb{P}(X_\tau = v \mid X_{\tau-1} = u, X_K = u_{\text{rec}}) = \frac{P_{uv}\, b_{K-\tau+1}(v)}{\sum_w P_{uw}\, b_{K-\tau+1}(w)}, \qquad q_\tau(u) \propto (\mathbf{e}_{u_{\text{rel}}}^\top \mathbf{P}^\tau)_u\, (\mathbf{P}^{K-\tau}\mathbf{e}_{u_{\text{rec}}})_u.$$

  To construct **Cumulative Markov Bridge Transit Corridors**, the endpoints ($\tau = 0$ and $\tau = K$) are excluded before averaging interior marginals $q_\tau(u)$ across $\tau = 1, \dots, K-1$ and summing over sampled trajectories. Excluding endpoints prevents release and recapture fishing concentrations from masking the in-transit probability cloud.
- **Loop-free 5-knot marine splines (`reconstruct_all_event_trajectories`, `smooth_marine_polyline`)**: To visualize overall predicted trajectories across all $N = 5{,}066$ encounters without random-walk loops or $60^\circ$ hexagonal zig-zags, each conditioned bridge undergoes Lawler loop erasure [@Lawler_1980], three-pass marine-guarded moving-average relaxation, selection of five arc-length control knots (subdivided only when a chord intersects land), and centripetal ($\alpha = 0.5$) Catmull-Rom cubic spline interpolation [@Yuksel_et_al_2011] with raster shoreline damping.
- **1-year effective dispersal area (`compute_effective_dispersal_area`)**: Single-step outgoing entropy $H(u)$ saturates near $1.0$ across homogeneous-HSI shelf zones (`99%` of marine units within $[0.862, 1.000]$) and is therefore replaced as a standalone map by the 365-day effective dispersal footprint, computed via binary matrix exponentiation of the marine submatrix $\mathbf{P}_m^{365}$:

$$H_{365}(u) = -\sum_{v \in \mathcal{M}} [\mathbf{P}^{365}]_{uv}\log[\mathbf{P}^{365}]_{uv}, \qquad A_{\text{eff}}(u) = \exp\!\bigl(H_{365}(u)\bigr)\,A_{\text{cell}}, \qquad r_{\text{eff}}(u) = \sqrt{\frac{A_{\text{eff}}(u)}{\pi}}.$$

  Here $A_{\text{eff}}(u)$ ($\text{km}^2$) measures the effective marine area explored over one year by an individual starting in cell $u$, directly separating locally retentive coastal embayments from broad open-shelf mixing zones.
- **Corridor canalization and structural bottlenecks (`compute_corridor_canalization`)**: For marine units with degree $d_u \ge 2$, normalized non-self step entropy $H(u) \in [0, 1]$ and interior bridge transit density $\tilde{C}(u) = \min(1, C(u)/P_{99}(C))$ define:

$$H(u) = -\frac{1}{\log d_u}\sum_{v \ne u}\tilde{P}_{uv}\log\tilde{P}_{uv}, \qquad K(u) = \sqrt{\tilde{C}(u)}\,\bigl(1 - 0.5\,H(u)\bigr), \qquad B(u) = \frac{C_{\text{domain}}(u)}{\max(1, d_u)}.$$

- **Interior $\hat{\gamma}$-calibrated Kirchhoff circuit current density (`current_density_map`)**: Models the marine graph as a resistive network $\mathbf{L}_C\mathbf{V} = \mathbf{I}_{\text{ext}}$ [@McRae_et_al_2008] with symmetric edge conductances calibrated to the MCMC-fitted habitat gradient $\hat{\gamma}$ on the 12-month mean HSI:

$$C_{uv} = W_{\text{marine}, uv}\,\exp\!\left(\frac{\hat{\gamma}}{2}(H_u + H_v)\right), \qquad J_i^{\text{int}} = \sum_{(u,v)} w_{uv}\!\left(\mathbb{I}[i \notin \{u, v\}]\,\frac{1}{2}\sum_{j \sim i}\bigl|I_{ij}^{(u,v)}\bigr|\right).$$

  Unique non-stationary marine origin-destination pairs receive log-damped weights $w_{uv} = 1 + \ln(n_{uv})$, endpoint injection spikes ($J_u = J_v = 1$) are excluded (`include_endpoints = false`) so transit bottlenecks are not drowned out by tagging ports, and each connected component Laplacian is Cholesky-factorized once. Robust pinch-points are defined as marine units falling in the upper 5% (`top_quantile = 0.95`) of $J^{\text{int}}$ in $\ge 80\%$ of interannual HSI realizations, displayed on a median-anchored logarithmic colour scale (`log_p99`).
- **1-year ($\mathbf{P}^{365}$) HSI-weighted regional connectivity (`compute_stock_connectivity_matrix`)**: Promotes $\mathbf{P}$ to 365 daily steps and aggregates across management polygons $R_a, R_b$ weighted by source suitability $H_u$:

$$M_{ab}^{(1\text{ yr})} = \frac{\sum_{u \in R_a} H_u \sum_{v \in R_b} [\mathbf{P}^{365}]_{uv}}{\sum_{u \in R_a} H_u}, \qquad \sum_b M_{ab}^{(1\text{ yr})} = 1.$$

- **Model-free empirical orientation diagnostics**: Complementing $\mathbf{P}$, `compute_spatial_bearing_roses` evaluates $100 \times 100\text{ km}$ circular mean bearings $\bar{\theta}_c$, resultant lengths $\bar{R}_c$, and Rayleigh uniformity $Z_c = n_c \bar{R}_c^2$ [@Mardia_Jupp_2000], while `analyze_gradient_taxis_projections` projects empirical unit displacement vectors $\hat{\mathbf{d}}_i$ directly onto local planar gradients $(\nabla z, \nabla T, \nabla\text{HSI})$.

---

# 4. Model Assumptions and Weaknesses

Every simplifying assumption in Section 3 carries concrete diagnostic consequences that bound how inferred quantities should be interpreted:

## 4.1. Behavioural and Dynamical Assumptions

- **First-order Markov memorylessness ($\kappa = 0$ during MCMC)**: The fitted likelihood assumes daily transitions depend solely on the current cell $u$, omitting directional persistence (heading memory), individual home-range attraction, and latent behavioural state switching (e.g., resting vs. directed seasonal migration). Forward persistence $\kappa > 0$ is supported only in optional post-fit `:agent` projections.
- **Time-homogeneous kernel during parameter estimation**: Although monthly `monthly_hsi` slices are used during Phase 4 trajectory reconstruction and Phase 2b agent projection, Phase 2 MCMC evaluates $\mathbf{P}^{k_i}$ using the static climatological mean `hsi_vec`. Because empirical cross-slope movement reverses between winter and summer-autumn (Section 5.2), a time-homogeneous operator averages over opposing seasonal forcings and likely understates seasonal directedness.
- **Population-pooled movement parameters**: Fitting a single $(\alpha, \rho, \gamma)$ across all individuals pools mature males, mature egg-bearing females, and adolescents across carapace conditions (CC 1-5). Given established sexual dimorphism and terminal-molt behavioural differences [@Sainte_Marie_Hazel_1992; @Comeau_Conan_1992], pooled parameters represent a sample-weighted mixture (dominated by legal-sized commercial males) rather than a universal species constant.

## 4.2. Observational and Statistical Weaknesses

- **Spatial sampling bias and imperfect effort correction**: Acoustic detections (`OTN`) are confined to fixed receiver gates, producing `84.4%` apparent within-cell stationarity at 5 km resolution, whereas commercial pot recaptures (`BIO`) occur only where and when fishing fleets deploy traps. In the `BIO` likelihood, the effort proxy $\mathbf{w}_{\text{eff}}$ is smoothed from observed tag-capture density rather than independent commercial logbook trap-haul effort; unfished habitats with zero tag returns can therefore remain partially confounded with low visitation probability.
- **Conditioning on observed recaptures without explicit mortality or tag loss**: The primary categorical likelihood models destination cell conditional on recapture, implicitly assuming spatially uniform natural mortality, fishing mortality, tag shedding, and reporting rates across management areas. Regions with higher harvest exploitation rates naturally generate more recaptures per unit resident abundance, which can bias inferred net directional flow toward heavily fished zones unless separated via the optional multistate Cormack-Jolly-Seber (`:recapture`) model.
- **Low MCMC effective sample size (`ESS ~ 3`)**: Prior-scaled isotropic random-walk Metropolis-Hastings achieved stable acceptance (`0.54-0.62`) over `300-500` draws, but strong posterior correlation among $(v, D, \gamma)$ induces high autocorrelation. Posterior means are stable across runs, yet 95% credible intervals may understate tail uncertainty until adaptive-covariance proposals or gradient-based sampling (`NUTS` over cached sparse sensitivities) replace isotropic random-walk updates.

## 4.3. Spatial and Environmental Limitations

- **5 km spatial resolution floor**: Displacements shorter than ~5 km within a single hexagon register as stationary (`0` hops), compressing fine-scale foraging variability into $\rho$.
- **Homogeneous suitability in unsurveyed waters (`cfagulf.dat`)**: Assigning uniform $H_u = 0.50$ across CFA 19 and the Southern Gulf prevents artificial boundary-channel artifacts and restores two-dimensional diffusion, but assumes spatially uniform benthic quality across `2,237` Gulf units where fine-scale thermal and substrate gradients certainly exist.
- **Absence of direct thermal-gradient fields in the baseline survey-HSI run**: Because `configs/snowcrab.toml` sets `use_geodata_habitat = false` to preserve Maritimes trawl-survey `hsi.zarr`, direct bottom-temperature taxis ($\nabla T$) is not evaluated (`n_thermal = 0`). Thermal forcing enters only indirectly through its empirical correlation with survey HSI.

---

# 5. Empirical Results Sketch (Scotian Shelf and Southern Gulf Snow Crab)

All summary quantities below derive from the full snow crab pipeline execution (`configs/snowcrab.toml`; $3{,}487$ marine units, `0` land/lake defects on the `1.25`-arcminute `"h"` mask, $N = 5{,}066$ encounters across $3{,}489$ tags).

## 5.1. Pooled and Modality-Stratified Kernel Parameters

| Stratum | Encounters (Tags) | % Stationary | Med. / Mean Disp. (km) | $\hat{\alpha}$ [95% CI] | $\hat{\rho}$ [95% CI] | $\hat{\gamma}$ [95% CI] | $D_{\text{eff}}$ ($\text{km}^2\,\text{yr}^{-1}$) |
|:--------|:------------------|:-------------|:-----------------------|:------------------------|:----------------------|:------------------------|:-------------------------------------------------|
| Pooled All | 5,066 (3,489) | 39.8% | 8.64 / 17.00 | 0.821 [0.764, 0.856] | 0.733 [0.625, 0.836] | 4.42 [3.74, 4.98] | 328.3 |
| Acoustic Array (`OTN`) | 1,559 (173) | 84.4% | 0.00 / 2.49 | 0.860 [0.835, 0.879] | 0.698 [0.657, 0.769] | 2.72 [1.32, 3.54] | 288.2 |
| Commercial Trap (`BIO`, effort-offset) | 3,507 (3,316) | 20.0% | 14.59 / 23.45 | 0.822 [0.804, 0.839] | 0.636 [0.591, 0.692] | 2.10 [1.85, 2.35] | 444.7 |

Posterior predictive calibration across $20$ MCMC draws yields $\mathrm{KL}(p_{\text{obs}} \parallel p_{\text{pred}}) = 0.765\text{ nats}$ (`95% CI [0.718, 0.835]`). Calibrated `TethysBio` coupling parameters exported to `movement_stats.toml` report a 95th-percentile annual movement speed of $214.4\text{ km yr}^{-1}$ (`max_move_m_per_year = 214,391 m/yr`), median observed speed of $17.4\text{ km yr}^{-1}$, and softmax temperature $1/\hat{\gamma} = 0.226$.

## 5.2. Bathymetric and Habitat-Gradient Taxis

| Stratum | $n_{\text{bathy}}$ | Downslope $\overline{\cos\psi_z}$ ($p$) | Mean Downslope (km) | Along-Isobath % | $n_{\text{HSI}}$ | Up-HSI $\overline{\cos\psi_h}$ ($p$) |
|:--------|:-------------------|:----------------------------------------|:--------------------|:----------------|:-----------------|:-------------------------------------|
| All Non-Stationary | 3,049 | $+0.076$ ($3.2 \times 10^{-9}$) | $+1.27$ | 62.7% | 2,114 | $+0.262$ ($2.6 \times 10^{-67}$) |
| Q1 (Winter) | 104 | $-0.218$ ($0.0030$) | $-15.01$ | 54.7% | 22 | $+0.370$ ($0.0184$) |
| Q2 (Spring) | 932 | $+0.038$ ($0.1054$) | $-2.12$ | 62.3% | 788 | $+0.269$ ($1.1 \times 10^{-27}$) |
| Q3 (Summer) | 1,358 | $+0.096$ ($3.2 \times 10^{-7}$) | $+2.24$ | 64.9% | 1,130 | $+0.229$ ($1.3 \times 10^{-28}$) |
| Q4 (Autumn) | 655 | $+0.135$ ($1.6 \times 10^{-6}$) | $+6.68$ | 59.7% | 174 | $+0.431$ ($5.1 \times 10^{-16}$) |

- **Contour-following and seasonal bathymetric reversal**: Nearly two-thirds (`62.7%`) of non-stationary movements track local isobaths rather than crossing steep depth gradients. Cross-slope components reverse seasonally from winter upslope displacement (`-15.0 km`, $p = 0.003$) to summer-autumn downslope redistribution (`+2.24` to `+6.68 km`, $p < 10^{-5}$).
- **Persistent up-suitability taxis**: Movement toward higher survey HSI is positive and statistically decisive across all four seasons and legal male size classes (`CW 95-110 mm`: $+0.265$; `110-125 mm`: $+0.248$; `>= 125 mm`: $+0.196$; all $p < 10^{-6}$).

## 5.3. Spatially Discretized 100 km Compass Bearing Structure

While shelf-wide circular averaging yields a low resultant length ($\bar{R} = 0.113$), `14` of `19` spatial $100 \times 100\text{ km}$ sectors reject circular uniformity ($p < 0.05$). Opposing local regimes—which cancel when pooled—align cleanly with regional bathymetry:

- **Southern Gulf / Magdalen Shallows**: Strong northward orientation at `47.4°N, 63.3°W` ($\bar{\theta} = 3.0^\circ$, $\bar{R} = 0.653$, $n = 60$, $p < 10^{-4}$) and `47.1°N, 62.5°W` ($\bar{\theta} = 359.1^\circ$, $\bar{R} = 0.502$, $n = 208$, $p < 10^{-4}$), paired with southward return flow further north (`48.2°N, 62.2°W`, $\bar{\theta} = 185.1^\circ$, $p = 0.0012$).
- **Cabot Strait and Northeastern Cape Breton**: Southwestward to westward steering along the Laurentian Channel margin (`46.9°N, 60.1°W`: $\bar{\theta} = 237.2^\circ$, $\bar{R} = 0.582$; `47.3°N, 60.7°W`: $\bar{\theta} = 226.5^\circ$, $\bar{R} = 0.362$; `46.4°N, 59.7°W`: $\bar{\theta} = 272.7^\circ$, $\bar{R} = 0.301$; all $p < 10^{-4}$).
- **Eastern and Outer Scotian Shelf**: East-southeast movement off Chedabucto / Canso (`45.4°N, 60.8°W`: $\bar{\theta} = 104.5^\circ$, $\bar{R} = 0.321$, $n = 331$, $p < 10^{-4}$) converging with northwestward movement from the outer shelf (`45.2°N, 58.4°W`: $\bar{\theta} = 308.9^\circ$, $\bar{R} = 0.436$, $p = 2 \times 10^{-5}$).

## 5.4. Markov Bridge Transit Corridors, Canalization, and 1-Year Effective Dispersal

Complementing individual smoothed trajectories, three probabilistic spatial fields characterize where snow crab travel in transit, how tightly movement is channelled, and how broadly populations mix over an annual cycle:

| Spatial Diagnostic | Active Units ($> 0$) | Median | 90th Pct | 99th Pct | Maximum | Flagged Hotspots |
|:-------------------|:---------------------|:-------|:---------|:---------|:--------|:-----------------|
| Markov Bridge Transit Corridors $C(u)$ | 2,848 / 3,487 | 0.00691 | 0.1845 | 0.8078 | 5.561 | 256 Core Corridors ($\ge P_{92}$) |
| Corridor Canalization $K(u)$ | 3,069 / 3,487 | 0.0296 | 0.1802 | 0.4326 | 1.000 | 318 Conduits ($\ge P_{90}$) |
| Directed Flux Bottlenecks $B(u)$ | 1,226 / 3,487 | 0.0000 | 0.2842 | 1.636 | 7.245 | 123 Bottlenecks ($B \ge 0.742$) |
| Interior Circuit Current $J^{\text{int}}(u)$ ($\hat{\gamma} = 4.42$) | 3,171 / 3,487 | 0.00102 | 0.00774 | 0.02015 | 0.06694 | 149 Robust Pinch-Points ($P \ge 0.80$, top 5%) |
| 1-Yr Effective Dispersal $A_{\text{eff}}(u)$ ($\text{km}^2$) | 3,487 / 3,487 | 25,036 | 35,739 | 43,720 | 46,310 | — (continuous linear scale) |
| Forward Agent Space-Use (visit prob.) | 1,778 / 3,487 | 0.005 | 0.035 | 0.085 | 0.285 | — (unconditioned `:agent` rollout) |

- **Cumulative Markov Bridge Transit Corridors ($C(u)$)**: Summing interior bridge occupancy across sampled tags ($\tau \in \{1, \dots, K-1\}$, excluding endpoints) maps the full probabilistic in-transit cloud rather than single deterministic routes. Across the `3,487` marine units, `2,848` carry positive transit density (`median = 0.00691`, `p99 = 0.8078`, `max = 5.561`), and `256` core corridor cells ($\ge 92\text{nd}$ percentile of positive density) trace continuous shelf-scale pathways linking eastern Cape Breton, Misaine/Emerald Basins, and the Southern Gulf. Because bridge density is conditioned on observed mark-recapture endpoints, bright corridors reflect both favourable habitat connectivity and spatial tagging coverage; dark regions may reflect either habitat avoidance or sparse bracketing endpoints.
- **Corridor Canalization ($K(u)$)**: Weighting bridge density by directional focus $(1 - 0.5\,H(u))$ isolates `318` narrow, habitat-guided conduits (`median = 0.0296`, `p99 = 0.4326`, `max = 1.000`) from broad diffusive basins where crabs spread isotropically.
- **1-Year Effective Dispersal Area ($A_{\text{eff}}(u)$)**: Unlike 1-step outgoing entropy (which saturated near `1.0` across `99%` of the shelf), propagating $\mathbf{P}_m^{365}$ resolves more than an order of magnitude in spatial mixing (`3,194.4` to `46,310.3 km²`, corresponding to `48.7` to `706.5` effective 5 km cells and equivalent circular radii $r_{\text{eff}}$ of `31.9` to `121.4 km`; `p10 = 10,729.5 km²`, `median = 25,036.4 km²`, `p90 = 35,739.4 km²`). Confined coastal embayments, fjords, and narrow straits exhibit tight annual retention ($r_{\text{eff}} \approx 32\text{--}58\text{ km}$), whereas open shelf plains support broad annual mixing ($r_{\text{eff}} \approx 107\text{--}121\text{ km}$).

## 5.5. Interior Kirchhoff Circuit Current Density, Structural Bottlenecks, and Pinch-Points

Electrical circuit flow provides an all-paths connectivity perspective complementary to Markov-bridge corridors. Initial uncalibrated runs using weak conductance contrast (`conductance_power = 1.0`, $e^1 \approx 2.7\times$ across $[0, 1]$ HSI) and endpoint current injections ($J_u = J_v = 1.0$) produced a visually flat shelf field (`max / median ~ 94×`) in which `317` pinch-point markers clustered on tagging ports rather than transit chokepoints. Three methodological refinements resolved this degeneracy:

- **Habitat-calibrated conductance ($\hat{\gamma} = 4.42$)**: Scaling edge conductance as $C_{uv} = W_{\text{marine}, uv}\exp(\frac{\hat{\gamma}}{2}(H_u + H_v))$ ($e^{\hat{\gamma}} \approx 83\times$ contrast between poor and optimal habitat) focuses multi-route electrical current along the cold depositional basins and shelf troughs inferred from telemetry.
- **Endpoint exclusion and log-damped pair weighting**: Computing pure interior transit current (`include_endpoints = false`) across unique non-stationary marine origin-destination pairs weighted by $1 + \ln(n_{uv})$ removes port-level delta spikes and prevents a single heavily fished area from dominating the network.
- **Logarithmic scaling and sharper pinch-point criterion**: Mapping interior current (`p25 = 0.00024`, `median = 0.00102`, `p75 = 0.00399`, `p90 = 0.00774`, `p99 = 0.02015`, `max = 0.06694`) on a median-anchored logarithmic stretch (`log_p99`, colourbar midpoint `0.0041` aligned with `p75`) reveals both primary inter-basin highways and secondary shelf routes across `3,171 / 3,487` active marine units. Robust pinch-points ($P \ge 0.80$ across interannual HSI draws at `top_quantile = 0.95`) sharpen from `317` to `149` localized chokepoints along Cabot Strait, the St. Anns / Laurentian slope margin, and inter-basin shelf gullies, corroborating the `123` degree-normalized structural bottlenecks ($B(u) \ge 0.742$) and `180` dominant net directed flux arrows derived from Markov-bridge routing.

## 5.6. Annual ($\mathbf{P}^{365}$) Regional Connectivity

| From \ To (1-Yr $\mathbf{P}^{365}$) | N-ENS [95% CI] | S-ENS [95% CI] | 4X [95% CI] | CFA19 / S.Gulf [95% CI] |
|:------------------------------------|:---------------|:---------------|:------------|:------------------------|
| **N-ENS** | 0.820 [0.783, 0.849] | 0.133 [0.111, 0.160] | 0.000 [0.000, 0.000] | 0.046 [0.040, 0.057] |
| **S-ENS** | 0.006 [0.005, 0.008] | 0.955 [0.946, 0.964] | 0.039 [0.031, 0.046] | < 0.001 [0.000, 0.000] |
| **4X** | 0.000 [0.000, 0.000] | 0.163 [0.129, 0.194] | 0.837 [0.806, 0.871] | 0.000 [0.000, 0.000] |
| **CFA19 / S.Gulf** | 0.008 [0.005, 0.010] | < 0.001 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.992 [0.990, 0.995] |

---

# 6. Discussion and Implications

## 6.1. Observation Modality vs. Biological Movement

Contrasting `OTN` acoustic arrays with `BIO` commercial pot recaptures demonstrates why unstratified mark-recapture summaries must be interpreted cautiously. Acoustic receivers record frequent within-cell detections (`84.4%` stationary at 5 km resolution), pulling pooled residence upward, whereas commercial trap recaptures span multi-year intervals across broader fishing grounds (`20.0%` stationary, $D_{\text{eff}} = 444.7\text{ km}^2\,\text{yr}^{-1}$). Yet after incorporating an empirical trap-effort offset in the `BIO` likelihood, both modalities infer strongly directed habitat advection ($\hat{\alpha} \in [0.82, 0.86]$, $\hat{\gamma} > 2.0$), indicating that habitat-guided movement along cold bathymetric basins is a robust biological signal rather than a gear artifact.

## 6.2. Implications for Stock Assessment and Fisheries Management

- **Asymmetric source-sink bias in closed-area assessments**: Annual transition estimates indicate that N-ENS exports `13.3%` southward to S-ENS and `4.6%` (`127` observed cross-boundary tags) into CFA 19 / Southern Gulf, while 4X exports `16.3%` into S-ENS against `3.9%` return inflow. Closed-population biomass models applied separately to N-ENS or 4X will absorb net emigration into elevated apparent natural mortality or underestimated productivity, whereas S-ENS (`95.5%` retention plus net immigration from both flanks) receives an unmodeled demographic subsidy.
- **Operational coupling to spatially explicit assessment (`TethysBio`)**: Exporting calibrated dispersal scales (`max_move_m_per_year = 214.4 km/yr`, softmax temperature `0.226`) in `movement_stats.toml` allows spatial stock-assessment models to replace ad-hoc diffusion priors with empirically bounded movement operators.
- **Climate warming and corridor constriction**: Because `62.7%` of displacements follow bathymetric contours and track high-suitability cold troughs, warming of shallow banks (`< 60 m`) and contraction of the Cold Intermediate Layer are expected to compress snow crab redistribution into the `256` core Markov-bridge corridor cells, `123` structural bottleneck units, and `149` robust interior Kirchhoff pinch-points (Cabot Strait, St. Anns / Laurentian slope, and cross-shelf gullies). Protecting bottleneck corridors (including spatial refugia such as the St. Anns Bank Marine Protected Area) and avoiding concentrated harvest inside narrow inter-basin conduits preserves regional re-colonization capacity.
- **Local retention vs. shelf-scale mixing**: The 1-year effective dispersal footprint ($A_{\text{eff}}$ spanning `3,194` to `46,310 km²`, $r_{\text{eff}} = 32\text{--}121\text{ km}$) demonstrates that management vulnerability is spatially heterogeneous even within a single CFA: local depletion in low-$A_{\text{eff}}$ embayments and coastal channels recovers more slowly via immigration than depletion on high-$A_{\text{eff}}$ open shelf plains.
- **Telemetry and tagging survey design**: Combining fishery-independent acoustic receiver gates positioned across identified pinch-points with commercial tagging paired to independent logbook trap-haul effort resolves the structural trade-off between `OTN` spatial coverage and `BIO` effort confounding, and directly enables joint survival-detection-movement separation (`:recapture`).

---

# 7. Conclusions

1. A single three-parameter daily Markov operator ($\alpha, \rho, \gamma$) on a `1.25`-arcminute shoreline-constrained 5 km marine lattice unifies trajectory reconstruction, interior Markov-bridge transit corridors (`256` core corridor cells), 1-year effective dispersal footprints (`3,194–46,310 km²`), interior $\hat{\gamma}$-calibrated Kirchhoff pinch-point detection (`149` robust pinch-points), and annual regional exchange without overland or inland-lake routing artifacts (`0` land/lake defects across `3,487` marine units and `5,066` encounters).
2. Empirical snow crab movement on the Scotian Shelf and Southern Gulf is strongly habitat-directed ($\hat{\alpha} \approx 0.82$, $\hat{\gamma} \approx 4.4$, up-HSI $\overline{\cos\psi_h} = +0.262$), predominantly contour-following (`62.7%` along-isobath), seasonally reversible across depth gradients (winter upslope vs. summer-autumn downslope), and locally steered by regional topography (`14 / 19` significant $100\text{ km}$ bearing sectors).
3. Administrative Crab Fishing Areas are asymmetrically coupled at annual timescales (`N-ENS -> S-ENS: 13.3%`, `N-ENS -> CFA 19 / S.Gulf: 4.6%`, `4X -> S-ENS: 16.3%`). Future work should prioritize time-varying seasonal MCMC likelihoods, independent commercial effort offsets, adaptive-covariance or gradient-based samplers to raise MCMC `ESS`, and dedicated Southern Gulf HSI surfaces.

---

# Appendix A: TethysPath Pipeline, Configuration, and Checkpoint Reference

## A.1. Six-Phase Execution Architecture

```text
[Phase 1] load_movement_data              -- Telemetry ingestion, 1.25' land/lake mask, edge severing,
                                             shoreline polygon clipping, homogeneous cfagulf.dat HSI
[Phase 2] fit_movement_models             -- Pooled & stratified OTN vs. BIO (effort-offset) RW-MH MCMC
[Phase 3] extract_transition_kernels      -- Posterior P_kernel, movement_checkpoint.zarr, :agent projection
[Phase 4] reconstruct_paths_and_diagnostics -- Conditioned Markov bridges (:bridge), corridors, bottlenecks B(u)
[Phase 5] compute_advanced_diagnostics /  -- Kirchhoff circuit flow (:circuit), canalization K(u), 1-yr P^365
          execute_validation_analyses        regional connectivity, KL PPC (:validation), :connectivity, :recapture
[Phase 6] export_dashboards               -- 5-knot spline trajectories (all events), 100 km roses, gradient taxis,
                                             movement_dashboard.html + .data.zz, movement_stats.toml, results Zarr
```

## A.2. CLI Usage and Key Configuration Parameters (`TethysPathConfig`)

```powershell
# Full snow crab pipeline from Phase 1
julia --project=. TethysPathRun.jl --config=configs/snowcrab.toml

# Resume from Phase 3 MCMC checkpoint (skip Phases 1-3)
julia --project=. TethysPathRun.jl --config=configs/snowcrab.toml --resume=true

# Regenerate HTML dashboard, .data.zz sidecar, and Phase 6 tables from full results checkpoint
julia --project=. TethysPathRun.jl --config=configs/snowcrab.toml --figures-only=true
```

| Parameter | Default (`configs/snowcrab.toml`) | Description |
|:----------|:----------------------------------|:------------|
| `model_modes` | `[:telemetry, :agent]` | Active inference/simulation modes (`:telemetry`, `:telemetry_and_survey`, `:agent`) |
| `land_mask_resolution` / `grid_minutes` | `"h"` / `1.25` | GMT/basemap ocean/lake raster tier and arc-minute spacing |
| `hex_radius_km` / `depth_range` | `5.0` / `[25.0, 350.0]` | Hexagonal lattice circumradius (km) and benthic depth window (m) |
| `depth_barrier_mode` / `hsi_ood_floor` | `:hsi_only` / `0.01` | Preserves marine connectivity while penalizing out-of-depth units via HSI |
| `path_methods` / `max_paths` | `[:bridge]` / `300` | Conditioned Markov-bridge (`:bridge`) or Viterbi (`:viterbi`) routing |
| `n_samples` / `n_warmup` / `mh_proposal_scale` | `500` / `100` / `0.05` | MCMC posterior draw count, warmup, and prior-scaled RW-MH step width |
| `diagnostics` | `[:circuit, :bottlenecks, :validation]` | Post-fit diagnostics (plus optional `:bayesian_ensemble`, `:connectivity`, `:recapture`) |
| `fpt_step_years` | `1 / 365.25` | Duration of one transition kernel step in years |
| `region_labels` / `region_polygon_files` | `["N-ENS", "S-ENS", "4X", "CFA19 / S.Gulf"]` | Management polygons (`inputs/cfa*.dat`) for annual connectivity |
| `output_dir` / `render_html` | `"../lakehouse/TethysPath"` / `true` | Lakehouse output store for Zarr checkpoints, CSVs, TOML, and dashboard |

## A.3. Lakehouse Artifacts and Zarr Checkpoint Inspection

Pipeline outputs written to `../lakehouse/TethysPath/` include:

- `movement_checkpoint.zarr` (Phases 1-3) and `movement_results_checkpoint.zarr` (Phases 1-6).
- `movement_dashboard.html` (`~77 KB` shell) and `movement_dashboard.data.zz` (`~704 KB` RFC 1950 zlib-compressed JSON sidecar; see [github_pages.md](github_pages.md) for local HTTP and GitHub Pages `gh-pages` deployment). The dashboard opens on **Smoothed Trajectories** (all $5{,}066$ events as loop-free `5`-knot marine splines colour-coded by time between mark and recapture, with green Mark/Release points, gold Recapture stars, movement-direction arrowheads, left-docked legends/controls, and click-map highlighting defaulting to `Nearest Marks / Releases`), followed by **Paths & HSI**, **Corridors**, **Directed Flux & Bottlenecks**, **Corridor Canalization**, **1-Yr Effective Dispersal** ($A_{\text{eff}}(u) = \exp(H(\mathbf{P}^{365}_{u,\cdot})) A_{\text{cell}}$ in $\text{km}^2$), **Compass Bearing Roses (100 km)**, **Circuit Flow** (interior $\hat{\gamma}$-calibrated Kirchhoff current on a `log_p99` scale with robust top-5% pinch-points), **Agent Space-Use**, **Tessellation & Depth**, **Path Analysis & Diagnostics**, and **Posterior Predictive (PPC)**. Every map panel includes a concise general-audience interpretation block, an interactive hover-over colour-scale readout (matching linear, square-root, or logarithmic transforms), and a per-panel background tessellation toggle; every report card includes a plain-language caption.
- `movement_stats.toml`, `acoustic_vs_trap_inference.csv`, `gradient_taxis_summary.csv`, `spatial_bearing_roses_100km.csv`, `corridor_canalization_summary.csv`, `effective_dispersal_1yr.csv`, `stock_connectivity_summary.csv`, `stock_connectivity_credible_intervals.csv`, `movement_path_metrics.csv`, `movement_agent_space_use.csv`, and `posterior_predictive_summary.txt`.

Inspecting stored Zarr checkpoints in Julia without re-running MCMC:

```julia
using TethysPath, Tethys, DataFrames, Statistics, SparseArrays

store_dir = Tethys.lakehouse_dir("TethysPath")
rc = Tethys.read_data(joinpath(store_dir, "movement_results_checkpoint.zarr"))
loaded      = TethysPath._zarr_checkpoint_restore_loaded(rc["loaded"])
kernels     = rc["kernels"]
path_res    = rc["path_res"]
diagnostics = rc["diagnostics"]
validation  = rc["validation"]

n_units       = loaded.mesh.n_units
P             = sparse(Float64.(kernels.P_kernel))
n_nonzero     = count(!iszero, P)
sample_paths  = collect(values(path_res.paths))
n_bottlenecks = count(path_res.domain_bottlenecks.bottleneck_mask)
n_pinch       = count(diagnostics.circuit.pinch_mask)
ppc_summary   = validation.posterior_predictive.summary
```

---

# References

::: {#refs}
:::
