### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 3f2c8a41-6b57-4d1e-9a83-2c5b7e1d0f44
# ╠═╡
using Pkg

# ╔═╡ 7a3f0c15-9b42-4e8d-a6f1-2b8c5d0e7a93
# ╠═╡
Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ c1000001-0001-0001-0001-000000000002
# ╠═╡
using Base64
using DataFrames
using Statistics
using SparseArrays
using UUIDs

# ╔═╡ 8e2b6d47-3f19-4c7b-8d5a-1f4c9b2e0a36
# ╠═╡
using GeoMotion
const MovementAnalysis = GeoMotion

# ╔═╡ 7f3c9e52-4b18-4d6a-8e05-2c9b1d6a3f47
# ╠═╡
using JLD2

# ╔═╡ c1000001-0001-0001-0001-000000000200
# ╠═╡
using PlutoUI

# ╔═╡ 9d4e1b72-8c05-4a63-b7e1-6f0d3a8c5e21
# ╠═╡
md"""
# GeoMotion — Results Viewer

This notebook loads **stored** pipeline results and displays them. It does **not** run any
models on initiation: the expensive Bayesian fitting, kernel construction, path
reconstruction, and diagnostics are all skipped unless you explicitly rerun them.

## How it works

1. The notebook looks for a stored results checkpoint
   (`output/movement_results_checkpoint.jld2`) written by a previous pipeline run.
2. If found, it loads `loaded`, `kernels`, `path_res`, `diagnostics`, `validation`,
   `agent_trajectories`, and `agent_space_use` and renders visualisations from them.
3. If **not found**, it stops with a clear error telling you to rerun the pipeline first.

No MCMC sampling, no kernel extraction, no path reconstruction, and no HTML dashboard
generation happens in this notebook. All figures are produced in-notebook from the
stored results.

## Checkpoint compatibility

Both shipped checkpoints (`output/movement_results_checkpoint.jld2` and
`output/movement_checkpoint.jld2`) store Julia types under the `GeoMotion.*` module
name and load with **no JLD2 reconstruction warnings**. The
`const MovementAnalysis = GeoMotion` alias below is kept only as a legacy bridge for
older checkpoints written before the package was renamed; it is not required for the
migrated files. The field mappings used by this notebook against the checkpoint
(`loaded.mesh.n_units`, `kernels.P_kernel` row-stochastic, `path_res.paths`,
`diagnostics.circuit`) are regression-tested in the `:numerics` test segment
(`test/test_numerics.jl`). To re-save a pre-rename checkpoint under the new module
name, run `scripts/migrate_checkpoint.jl`.
"""

# ╔═╡ 2b7f5d93-1e48-4c0a-8f6d-9a1c4e7b0d35
# ╠═╡
md"""
## Loading stored results

The results checkpoint is written by `run_movement_analysis` at the end of a successful
pipeline run. It contains every panel's output, so this notebook can render all
visualisations without recomputing anything.
"""

# ╔═╡ 5c1d8e64-2a79-4b5f-9e07-3d6a1c8b4f29
# ╠═╡
cd(joinpath(@__DIR__, ".."))

# ╔═╡ e1000001-0001-0001-0001-000000000100
# ╠═╡
output_dir = "output"
results_checkpoint = joinpath(output_dir, "movement_results_checkpoint.jld2")
intermediate_checkpoint = joinpath(output_dir, "movement_checkpoint.jld2")

if isfile(results_checkpoint)
    @info "Loading stored results from: $results_checkpoint"
    rc = JLD2.load(results_checkpoint)
    loaded       = rc["loaded"]
    kernels      = rc["kernels"]
    path_res     = rc["path_res"]
    diagnostics  = rc["diagnostics"]
    validation   = rc["validation"]
    agent_trajectories = get(rc, "agent_trajectories", nothing)
    agent_space_use    = get(rc, "agent_space_use", nothing)
    connectivity_assessment = get(validation, :connectivity_assessment, nothing)
    recapture_separation    = get(validation, :recapture_separation, nothing)
    results_source = "results_checkpoint"
elseif isfile(intermediate_checkpoint)
    @warn "Only the intermediate checkpoint found (no full results). Some visualisations will be unavailable."
    ic = JLD2.load(intermediate_checkpoint)
    loaded       = ic["loaded"]
    fitted       = ic["fitted"]
    kernels      = ic["kernels"]
    path_res     = nothing
    diagnostics  = nothing
    validation   = nothing
    agent_trajectories = nothing
    agent_space_use    = nothing
    connectivity_assessment = nothing
    recapture_separation    = nothing
    results_source = "intermediate_checkpoint"
else
    connectivity_assessment = nothing
    recapture_separation    = nothing
    @bind show_run_snippet PlutoUI.CheckBox(default = false)
    if show_run_snippet
        md"""
        ## Stored results not found

        No pipeline results are available to visualise. The notebook searched for:

        - `$(abspath(results_checkpoint))`
        - `$(abspath(intermediate_checkpoint))`

        ### Run the pipeline first (toggle to show commands)

        **Via CLI:**
        ```bash
        julia --project=. GeoMotionRun.jl --config=configs/snowcrab.toml
        ```

        **Via Julia session:**
        ```julia
        using Pkg
        Pkg.activate(".")
        using GeoMotion
        params = GeoMotion.load_config(config_path = "configs/snowcrab.toml")
        GeoMotion.run_movement_analysis(params)
        ```

        After the pipeline completes, reopen this notebook and the stored results will load automatically.
        """
    else
        md"""
        ## Stored results not found

        No pipeline results are available to visualise. The notebook searched for:

        - `$(abspath(results_checkpoint))`
        - `$(abspath(intermediate_checkpoint))`

        _Click the checkbox above to show the commands to run the pipeline._

        After the pipeline completes, reopen this notebook and the stored results will load automatically.
        """
    end
end

# ╔═╡ e1000001-0001-0001-0001-000000000101
# ╠═╡
md"""
### Results loaded from: `$(results_source)`

| Artefact | Loaded |
|---|---|
| `loaded` (ingestion & lattice) | $(!isnothing(loaded)) |
| `kernels` (transition matrix) | $(!isnothing(kernels)) |
| `path_res` (paths & corridors) | $(!isnothing(path_res)) |
| `diagnostics` (circuit theory) | $(!isnothing(diagnostics)) |
| `validation` (PPC & metrics) | $(!isnothing(validation)) |
| `agent_trajectories` | $(!isnothing(agent_trajectories)) |
| `agent_space_use` | $(!isnothing(agent_space_use)) |
| `connectivity_assessment` (first passage) | $(!isnothing(connectivity_assessment)) |
| `recapture_separation` (multistate CJS) | $(!isnothing(recapture_separation)) |
"""

# ╔═╡ e1000001-0001-0001-0001-000000000102
# ╠═╡
md"""
## Panel 1 — Ingestion & Lattice Summary

`loaded` contains the mark-recapture observations, the hexagonal lattice, the neighbour
weight matrix `W`, the HSI field, and the land/water mask.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000103
# ╠═╡
begin
    n_units   = loaded.mesh.n_units
    n_spatial = loaded.n_spatial
    n_obs     = nrow(loaded.obs_df)
    n_marine  = isnothing(loaded.land_mask) ? n_units : count(!, loaded.land_mask)
    n_land    = isnothing(loaded.land_mask) ? 0 : count(loaded.land_mask)

    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Metric</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Spatial units (cells)</td><td style="border:1px solid #ccc; padding:6px;">$(n_units)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Marine units</td><td style="border:1px solid #ccc; padding:6px;">$(n_marine)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Land units</td><td style="border:1px solid #ccc; padding:6px;">$(n_land)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Observations (mark-recapture pairs)</td><td style="border:1px solid #ccc; padding:6px;">$(n_obs)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Depth range</td><td style="border:1px solid #ccc; padding:6px;">$(isnothing(loaded.parsed_depth_range) ? "none" : loaded.parsed_depth_range)</td></tr>
    </table>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000104
# ╠═╡
md"""
### Observation sample

First 10 mark-recapture records from `loaded.obs_df`:
"""

# ╔═╡ e1000001-0001-0001-0001-000000000105
# ╠═╡
loaded.obs_df[1:min(10, nrow(loaded.obs_df)), :]

# ╔═╡ e1000001-0001-0001-0001-000000000106
# ╠═╡
md"""
## Panel 2 — Fitted Transition Kernel Parameters

`kernels` contains the posterior estimates for the dispersal exponent \$\\alpha\$, residence
probability \$\\rho\$, and habitat gradient responsiveness \$\\gamma\$, plus the full
\$n_{units} \\times n_{units}\$ transition matrix `P_kernel`.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000107
# ╠═╡
begin
    alpha_hat = kernels.alpha_hat
    rho_hat   = kernels.rho_hat
    gamma_hat = kernels.gamma_hat

    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Parameter</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Posterior Estimate</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Dispersal exponent (\$\\alpha\$)</td><td style="border:1px solid #ccc; padding:6px;">$(round(Float64(alpha_hat); digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Residence probability (\$\\rho\$)</td><td style="border:1px solid #ccc; padding:6px;">$(round(Float64(rho_hat); digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Habitat gradient (\$\\gamma\$)</td><td style="border:1px solid #ccc; padding:6px;">$(round(Float64(gamma_hat); digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Transition matrix size</td><td style="border:1px solid #ccc; padding:6px;">$(size(kernels.P_kernel))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Transition matrix non-zeros</td><td style="border:1px solid #ccc; padding:6px;">$(count(!iszero, kernels.P_kernel))</td></tr>
    </table>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000108
# ╠═╡
md"""
### Transition matrix row statistics

Each row of `P_kernel` sums to 1 (a conservative Markov jump process). The distribution of
row maximums shows how concentrated movement is on the most likely destination.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000109
# ╠═╡
begin
    P = kernels.P_kernel
    row_maxs = [maximum(P[i, :]) for i in 1:size(P, 1)]
    row_ents = [-sum(P[i, :] .* log.(max.(P[i, :], 1e-12))) for i in 1:size(P, 1)]

    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Statistic</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Mean row maximum (top destination prob)</td><td style="border:1px solid #ccc; padding:6px;">$(round(mean(row_maxs); digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Median row maximum</td><td style="border:1px solid #ccc; padding:6px;">$(round(median(row_maxs); digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Mean row entropy (nats)</td><td style="border:1px solid #ccc; padding:6px;">$(round(mean(row_ents); digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Rows with all probability on self (\$\\rho \\approx 1\$)</td><td style="border:1px solid #ccc; padding:6px;">$(count(>(0.99), row_maxs)) / $(length(row_maxs))</td></tr>
    </table>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000110
# ╠═╡
md"""
## Panel 3 — Paths & Corridors

`path_res` contains A* least-cost paths, stochastic path ensembles, corridor visitation
heatmaps, and domain-wide bottlenecks reconstructed from the fitted kernel.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000111
# ╠═╡
if !isnothing(path_res)
    n_paths       = length(path_res.paths)
    n_stochastic  = isnothing(path_res.stochastic_paths) ? 0 : length(path_res.stochastic_paths)
    n_corridors   = isnothing(path_res.corridors) ? 0 : length(path_res.corridors)
    n_bottlenecks = isnothing(path_res.domain_bottlenecks) ? 0 : (hasproperty(path_res.domain_bottlenecks, :bottleneck_mask) ? count(path_res.domain_bottlenecks.bottleneck_mask) : length(path_res.domain_bottlenecks))

    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Metric</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Deterministic A* paths</td><td style="border:1px solid #ccc; padding:6px;">$(n_paths)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Stochastic path ensembles</td><td style="border:1px solid #ccc; padding:6px;">$(n_stochastic)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Corridor visitation entries</td><td style="border:1px solid #ccc; padding:6px;">$(n_corridors)</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Domain bottlenecks</td><td style="border:1px solid #ccc; padding:6px;">$(n_bottlenecks)</td></tr>
    </table>
    """)
else
    md"""
    **No path reconstruction results available.** The intermediate checkpoint only stores
    `loaded`, `fitted`, and `kernels`. Run the full pipeline to produce `path_res`.
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000112
# ╠═╡
md"""
### Sample reconstructed paths

First few A* paths (cell index sequences):
"""

# ╔═╡ e1000001-0001-0001-0001-000000000113
# ╠═╡
if !isnothing(path_res) && !isempty(path_res.paths)
    paths = collect(values(path_res.paths))
    display_paths = paths[1:min(5, length(paths))]
    [HTML("""<div style='font-family: monospace; font-size: 0.9em;'>Path $(i): $(join(p, " → "))</div>""") for (i, p) in enumerate(display_paths)]
else
    md"*No paths available.*"
end

# ╔═╡ e1000001-0001-0001-0001-000000000114
# ╠═╡
md"""
## Panel 4 — Circuit Diagnostics

`diagnostics.circuit` contains electrical-circuit theory results: current density through
the resistor network, pinch points, and effective resistance.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000115
# ╠═╡
if !isnothing(diagnostics) && !isnothing(diagnostics.circuit)
    circ = diagnostics.circuit
    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Metric</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Circuit computed</td><td style="border:1px solid #ccc; padding:6px;">Yes</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Pinch points</td><td style="border:1px solid #ccc; padding:6px;">$(count(circ.pinch_mask))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Pinch threshold</td><td style="border:1px solid #ccc; padding:6px;">$(round(circ.threshold; digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Robust bottlenecks</td><td style="border:1px solid #ccc; padding:6px;">$(circ.n_robust)</td></tr>
    </table>
    """)
else
    md"""
    **No circuit diagnostics available.** Run the pipeline with `:circuit` in `diagnostics`
    to produce these results.
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000116
# ╠═╡
md"""
## Panel 5 — Validation & Posterior Predictive Checks

`validation` contains posterior predictive check results: Brier score, KL divergence, and
credible intervals comparing observed transitions against simulated replicates from the
fitted kernel.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000117
# ╠═╡
if !isnothing(validation)
    pp = validation.posterior_predictive
    ss = pp.summary
    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Metric</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Brier score (mean)</td><td style="border:1px solid #ccc; padding:6px;">$(round(ss.brier_mean; digits=6))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Brier score (95% CI)</td><td style="border:1px solid #ccc; padding:6px;">[$(round(ss.brier_lower_ci; digits=6)), $(round(ss.brier_upper_ci; digits=6))]</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">KL divergence (mean)</td><td style="border:1px solid #ccc; padding:6px;">$(round(ss.kl_mean; digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">KL divergence (95% CI)</td><td style="border:1px solid #ccc; padding:6px;">[$(round(ss.kl_lower_ci; digits=4)), $(round(ss.kl_upper_ci; digits=4))]</td></tr>
    </table>
    """)
else
    md"""
    **No validation results available.** Run the pipeline with `:validation` in `diagnostics`
    to produce these results.
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000118
# ╠═╡
md"""
## Panel 6 — Agent Projections

`agent_trajectories` and `agent_space_use` contain results from the individual-based
agent model projecting synthetic agents forward from observed release sites.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000119
# ╠═╡
if !isnothing(agent_space_use)
    HTML("""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="text-align:left; border:1px solid #ccc; padding:6px;">Metric</th><th style="text-align:left; border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Agent trajectories</td><td style="border:1px solid #ccc; padding:6px;">$(isnothing(agent_trajectories) ? "n/a" : length(agent_trajectories))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Spatial units visited</td><td style="border:1px solid #ccc; padding:6px;">$(count(>(0), agent_space_use.visit_probability))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Max visit probability</td><td style="border:1px solid #ccc; padding:6px;">$(round(maximum(agent_space_use.visit_probability); digits=4))</td></tr>
    </table>
    """)
else
    md"""
    **No agent projection results available.** Run the pipeline with `:agent` in
    `model_modes` to produce these results.
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000120
# ╠═╡
md"""
## Interactive Map — Cell Centroids & Paths

A lightweight Leaflet.js map (loaded from CDN, no Julia plotting package required)
showing the hexagonal lattice centroids, colored by HSI when available, with land cells
masked out. If path results are loaded, the first few A\* paths are drawn as polylines.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000121
# ╠═╡
function _leaflet_map_html(
    loaded, path_res;
    height::Int = 500,
    unit_colors = nothing,   # Vector{Int}: categorical per-cell color (palette)
    unit_labels = nothing,   # Vector{String}: per-cell label suffix
    unit_scale  = nothing,   # Vector{Float64} in [0, 1]: continuous viridis ramp
)
    # Resolve centroids (lon, lat)
    cents = nothing
    if hasproperty(loaded.mesh, :centroids_lonlat) && !isempty(loaded.mesh.centroids_lonlat)
        cents = loaded.mesh.centroids_lonlat
    elseif hasproperty(loaded.mesh, :centroids) && !isempty(loaded.mesh.centroids)
        cents = loaded.mesh.centroids
    end
    cents === nothing && return HTML("<div style='color:#94a3b8;padding:16px;'>No centroid coordinates available for mapping.</div>")

    n_units = length(cents)
    map_id = "leaflet-map-" * string(uuid4())
    # Center the map on the centroid mean
    c_lon = mean([c[1] for c in cents])
    c_lat = mean([c[2] for c in cents])

    # Build per-cell style: HSI color (viridis-like), land = grey
    hsi = hasproperty(loaded, :hsi_vec) ? loaded.hsi_vec : nothing
    land  = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing

    # Simple viridis-like color ramp
    function viridis(t::Float64)
        t = clamp(t, 0.0, 1.0)
        # Approximate viridis endpoints -> mid
        r = round(Int, 68 + (253 - 68) * t)
        g = round(Int, 1 + (231 - 1) * t)
        b = round(Int, 84 + (169 - 84) * t)
        "@$(string(r, base=16, pad=2))$(string(g, base=16, pad=2))$(string(b, base=16, pad=2))"
    end

    # Categorical palette for region/community overlays
    palette = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2",
               "#65a30d", "#db2777", "#7c3aed", "#ea580c"]
    pal(i::Int) = palette[(i - 1) % length(palette) + 1]

    # Build markers JSON array
    markers = []
    for i in 1:n_units
        lon, lat = Float64(cents[i][1]), Float64(cents[i][2])
        isnan(lon) || isnan(lat) && continue
        is_land = land !== nothing && land[i]
        if unit_scale !== nothing && i <= length(unit_scale)
            color = is_land ? "#9ca3af" : viridis(Float64(unit_scale[i]))
        elseif unit_colors !== nothing && i <= length(unit_colors)
            color = is_land ? "#9ca3af" : pal(Int(unit_colors[i]))
        else
            color = is_land ? "#9ca3af" : (hsi !== nothing ? viridis(Float64(hsi[i])) : "#3b82f6")
        end
        opacity = is_land ? 0.4 : 0.8
        radius = is_land ? 4 : 6
        suffix = (unit_labels !== nothing && i <= length(unit_labels)) ?
            " ($(unit_labels[i]))" : ""
        push!(markers, Dict(
            "lat" => lat, "lon" => lon,
            "color" => color, "opacity" => opacity,
            "radius" => radius,
            "label" => "Cell $i$(suffix) $(is_land ? "(land)" : "")"
        ))
    end

    # Build paths polylines JSON array
    paths = []
    if !isnothing(path_res) && !isempty(path_res.paths)
        for (pi, p) in enumerate(collect(values(path_res.paths))[1:min(8, length(path_res.paths))])
            coords = []
            for ci in p
                if 1 <= ci <= n_units
                    push!(coords, [Float64(cents[ci][2]), Float64(cents[ci][1])]) # [lat, lon]
                end
            end
            length(coords) > 1 && push!(paths, coords)
        end
    end

    # Serialize markers/paths to JSON-ish strings
    markers_js = join(["{lat: $(m["lat"]), lon: $(m["lon"]), color: \"$(m["color"])\", opacity: $(m["opacity"]), radius: $(m["radius"]), label: \"$(m["label"])\"}" for m in markers], ",")
    paths_js = join(["[" * join(["[ $(c[1]), $(c[2]) ]" for c in path], ",") * "]" for path in paths], ",")

    HTML("""
    <div id="$(map_id)" style="width:100%; height:$(height)px; border:1px solid #334155; border-radius:8px;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
      #$(map_id) .leaflet-container { width: 100%; height: $(height)px; }
    </style>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      (function() {
        var mapDiv = document.getElementById('$(map_id)');
        if (!mapDiv) return;
        var map = L.map(mapDiv).setView([$(c_lat), $(c_lon)], 7);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
          maxZoom: 10,
          attribution: '© OpenStreetMap contributors, © CARTO'
        }).addTo(map);

        var markers = [$(markers_js)];
        markers.forEach(function(m) {
          L.circleMarker([m.lat, m.lon], {
            radius: m.radius,
            color: m.color,
            fillColor: m.color,
            fillOpacity: m.opacity,
            weight: 1
          }).bindPopup(m.label).addTo(map);
        });

        var paths = [$(paths_js)];
        paths.forEach(function(coords) {
          L.polyline(coords, {color: '#ef4444', weight: 2, opacity: 0.7}).addTo(map);
        });

        map.fitBounds(markers.map(function(m) { return [m.lat, m.lon]; }));
      })();
    </script>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000122
# ╠═╡
_leaflet_map_html(loaded, path_res)

# ╔═╡ e1000001-0001-0001-0001-000000000127
# ╠═╡
md"""
## Panel 7 — First-Passage Connectivity Assessment

Stored under `validation.connectivity_assessment` when `:connectivity` is in the pipeline's
`diagnostics` list. The fitted unit-level transition kernel is coarse-grained onto the
configured regions, then summarised by:

- the row-stochastic **region transition matrix** `P_reg`,
- **mean first-passage times** (exact, in steps and years),
- **first-arrival probabilities** and **median arrival times** over the configured horizon,
- **corridor visit rates** (expected visits per unit under uniform walkers),
- a **community partition** (greedy modularity on the symmetrised region matrix), and
- **minimum cuts** in units of expected crossers per step.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000128
# ╠═╡
# Small HTML-table builder shared by the connectivity / recapture panels.
conn_cell_style = "border:1px solid #ccc; padding:4px 8px; font-size:13px;"
function conn_matrix_html(M, rowlab, collab; digits::Int = 3, fmt = nothing)
    celltxt(v) = fmt !== nothing ? fmt(v) :
        (isfinite(v) ? string(round(Float64(v); digits)) : "∞")
    hdr = "<tr><th style=''></th>" * join(
        ["<th style='$(conn_cell_style)'>$c</th>" for c in collab]) * "</tr>"
    rows = ["<tr><td style='$(conn_cell_style)'><b>$r</b></td>" *
            join(["<td style='$(conn_cell_style)'>$(celltxt(M[i, j]))</td>"
                  for j in eachindex(collab)]) * "</tr>"
            for (i, r) in enumerate(rowlab)]
    return "<table style='border-collapse: collapse; font-family: sans-serif;'>" *
           hdr * join(rows) * "</table>"
end

if isnothing(connectivity_assessment)
    md"""_No connectivity assessment available — run the pipeline with `:connectivity`
    in `diagnostics` to produce it._"""
else
    conn = connectivity_assessment
    lab = conn.region_labels
    step_txt = round(conn.step_years; digits = 4)
    HTML("""
    <div style="display:flex; gap:36px; flex-wrap:wrap;">
      <div>
        <div style="font-weight:600; margin-bottom:4px;">Region transition matrix P<sub>reg</sub></div>
        <div style="color:#64748b; font-size:12px; margin-bottom:6px;">row = from, col = to; per kernel step (1 step = $(step_txt) yr)</div>
        $(conn_matrix_html(conn.P_reg, lab, lab))
      </div>
      <div>
        <div style="font-weight:600; margin-bottom:6px;">Mean first passage time (years)</div>
        $(conn_matrix_html(conn.mfpt_years, lab, lab; digits = 2))
      </div>
    </div>
    <div style="display:flex; gap:36px; flex-wrap:wrap; margin-top:18px;">
      <div>
        <div style="font-weight:600; margin-bottom:6px;">P(first arrival by step $(conn.horizon))</div>
        $(conn_matrix_html(conn.arrival_prob, lab, lab))
      </div>
      <div>
        <div style="font-weight:600; margin-bottom:6px;">Median first-arrival time (years)</div>
        $(conn_matrix_html(conn.median_years, lab, lab;
             fmt = v -> isnan(v) ? "—" : string(round(v; digits = 2))))
      </div>
    </div>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000129
# ╠═╡
if isnothing(connectivity_assessment)
    md"""_No connectivity assessment available._"""
else
    conn = connectivity_assessment
    lab = conn.region_labels
    ord = sortperm(conn.corridor_norm; rev = true)
    top = ord[1:min(10, length(ord))]
    corridor_rows = join(["<tr><td style='$(conn_cell_style)'>Unit $(i)</td>
        <td style='$(conn_cell_style)'>$(round(conn.corridor_visits[i]; digits = 2))</td>
        <td style='$(conn_cell_style)'>$(round(conn.corridor_norm[i]; digits = 3))</td></tr>"
        for i in top])
    comm_rows = join(["<tr><td style='$(conn_cell_style)'>$(lab[r])</td>
        <td style='$(conn_cell_style)'>$(conn.region_sizes[r])</td>
        <td style='$(conn_cell_style)'><b>C$(conn.community_labels[r])</b></td></tr>"
        for r in 1:conn.n_regions])
    wp = conn.weakest_pair
    wp_txt = wp === nothing ? "—" :
        "Region $(lab[wp.source]) → Region $(lab[wp.sink]): capacity " *
        "$(round(wp.capacity; digits = 3)) expected crossers / step; cut edges " *
        join(["$(lab[e[1]]) → $(lab[e[2]])" for e in wp.cut_edges], ", ")
    ccut = conn.community_cut
    cc_txt = ccut === nothing ? "—" :
        "capacity $(round(ccut.capacity; digits = 3)) expected crossers / step; cut edges " *
        join(["$(lab[e[1]]) → $(lab[e[2]])" for e in ccut.cut_edges], ", ")
    HTML("""
    <div style="display:flex; gap:36px; flex-wrap:wrap;">
      <div>
        <div style="font-weight:600; margin-bottom:6px;">Top corridor units (horizon = $(conn.corridor_horizon) steps)</div>
        <table style='border-collapse: collapse; font-family: sans-serif;'>
          <tr><th style='$(conn_cell_style)'>Unit</th><th style='$(conn_cell_style)'>Visits</th><th style='$(conn_cell_style)'>Relative</th></tr>
          $(corridor_rows)
        </table>
      </div>
      <div>
        <div style="font-weight:600; margin-bottom:6px;">Community partition (greedy modularity)</div>
        <table style='border-collapse: collapse; font-family: sans-serif;'>
          <tr><th style='$(conn_cell_style)'>Region</th><th style='$(conn_cell_style)'>Units</th><th style='$(conn_cell_style)'>Community</th></tr>
          $(comm_rows)
        </table>
      </div>
    </div>
    <div style="margin-top:14px; font-family: sans-serif; font-size:13px;">
      <div><b>Weakest ordered region pair:</b> $(wp_txt)</div>
      <div><b>Community separating cut:</b> $(cc_txt)</div>
      <div style="color:#64748b; margin-top:4px;">Modularity = $(round(conn.modularity; digits = 4)) with $(conn.n_communities) communities.</div>
    </div>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000130
# ╠═╡
md"""
### Region and community overlay maps

Each lattice cell is colored by the region it belongs to (`conn.region_map`), then by its
modularity community. Land cells stay grey. Hovering a marker shows the cell's assignment.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000131
# ╠═╡
if isnothing(connectivity_assessment)
    md"""_No connectivity assessment available._"""
else
    conn = connectivity_assessment
    n = loaded.n_spatial
    region_of = [Int(conn.region_map[i]) for i in 1:n]
    _leaflet_map_html(loaded, path_res;
        unit_colors = region_of,
        unit_labels = ["Region $(conn.region_labels[r])" for r in region_of])
end

# ╔═╡ e1000001-0001-0001-0001-000000000132
# ╠═╡
if isnothing(connectivity_assessment)
    md"""_No connectivity assessment available._"""
else
    conn = connectivity_assessment
    n = loaded.n_spatial
    region_of = [Int(conn.region_map[i]) for i in 1:n]
    comm_of = [Int(conn.community_labels[region_of[i]]) for i in 1:n]
    _leaflet_map_html(loaded, path_res;
        unit_colors = comm_of,
        unit_labels = ["Community C$(comm_of[i])" for i in 1:n])
end

# ╔═╡ e1000001-0001-0001-0001-000000000133
# ╠═╡
md"""
### Corridor usage overlay

Cell shading follows the relative corridor visit rate (`corridor_norm`, max-normalized over
the corridor horizon): darker/greener cells absorb more expected walker visits per step.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000134
# ╠═╡
if isnothing(connectivity_assessment)
    md"""_No connectivity assessment available._"""
else
    conn = connectivity_assessment
    _leaflet_map_html(loaded, path_res;
        unit_scale  = Float64.(conn.corridor_norm),
        unit_labels = ["$(round(conn.corridor_norm[i]; digits = 3)) rel. visits"
                       for i in 1:loaded.n_spatial])
end

# ╔═╡ e1000001-0001-0001-0001-000000000135
# ╠═╡
md"""
## Panel 8 — Recapture Separation: Survival × Detection × Movement

Stored under `validation.recapture_separation` when `:recapture` is in the pipeline's
`diagnostics` list. A multistate CJS model (Arnason–Schwarz) is fitted to the aggregated
capture histories by adaptive random-walk Metropolis, separating **apparent survival**
`phi` from **detection** `pdet` and between-region **movement** `psi` per year. The
posterior movement matrix is compared against the transition kernel coarse-grained to the
regions and promoted to the capture interval.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000136
# ╠═╡
if isnothing(recapture_separation)
    md"""_No recapture separation available — run the pipeline with `:recapture`
    in `diagnostics` to produce it._"""
else
    rcs = recapture_separation
    lab = rcs.region_labels
    det_rows = join(["<tr><td style='$(conn_cell_style)'>$(lab[r])</td>
        <td style='$(conn_cell_style)'>$(round(rcs.detection_mean[r]; digits = 3))</td>
        <td style='$(conn_cell_style)'>[$(round(rcs.detection_lower[r]; digits = 3)), $(round(rcs.detection_upper[r]; digits = 3))]</td>
        <td style='$(conn_cell_style)'>$(round(rcs.survival_mean[r]; digits = 3))</td>
        <td style='$(conn_cell_style)'>[$(round(rcs.survival_lower[r]; digits = 3)), $(round(rcs.survival_upper[r]; digits = 3))]</td></tr>"
        for r in 1:rcs.n_regions])
    occ_txt = join([string(y) for y in rcs.occasions], ", ")
    HTML("""
    <table style='border-collapse: collapse; font-family: sans-serif;'>
      <tr>
        <th style='$(conn_cell_style)'>Region</th>
        <th style='$(conn_cell_style)'>P(annual detection)</th>
        <th style='$(conn_cell_style)'>95% CI</th>
        <th style='$(conn_cell_style)'>Apparent survival</th>
        <th style='$(conn_cell_style)'>95% CI</th>
      </tr>
      $(det_rows)
    </table>
    <div style="color:#64748b; font-size:13px; margin-top:8px;">
      $(rcs.n_tags) tags · $(rcs.n_segments) release segments · $(rcs.n_capture_events) capture
      events · occasions $(occ_txt) · sampler: $(rcs.n_samples) draws
      (thin $(rcs.thin), warmup $(rcs.n_warmup), acceptance $(round(rcs.acceptance_rate; digits = 2)))
    </div>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000137
# ╠═╡
if isnothing(recapture_separation)
    md"""_No recapture separation available._"""
else
    rcs = recapture_separation
    lab = rcs.region_labels
    mvk_rows = join(["<tr><td style='$(conn_cell_style)'>$(lab[i]) → $(lab[j])</td>
        <td style='$(conn_cell_style)'>$(round(rcs.psi_mean[i, j]; digits = 3))</td>
        <td style='$(conn_cell_style)'>[$(round(rcs.psi_lower[i, j]; digits = 3)), $(round(rcs.psi_upper[i, j]; digits = 3))]</td>
        <td style='$(conn_cell_style)'>$(rcs.kernel_yearly === nothing ? "—" : string(round(rcs.kernel_yearly[i, j]; digits = 3)))</td></tr>"
        for i in 1:rcs.n_regions, j in 1:rcs.n_regions])
    m_txt = rcs.kernel_yearly === nothing ? "n/a" : string(rcs.n_kernel_steps)
    HTML("""
    <table style='border-collapse: collapse; font-family: sans-serif;'>
      <tr>
        <th style='$(conn_cell_style)'>Movement</th>
        <th style='$(conn_cell_style)'>CJS posterior mean</th>
        <th style='$(conn_cell_style)'>95% CI</th>
        <th style='$(conn_cell_style)'>Kernel-implied annual (m = $(m_txt) steps)</th>
      </tr>
      $(mvk_rows)
    </table>
    <div style="color:#64748b; font-size:13px; margin-top:8px;">
      Disagreement between the CJS movement column and the kernel column flags
      detection-related bias: the CJS estimates movement *between observed captures*, while
      the kernel integrates movement at every step of the interval.
    </div>
    """)
end

# ╔═╡ e1000001-0001-0001-0001-000000000123
# ╠═╡
md"""
## Optional Plots

If `Plots` is installed in this environment, the cells below generate in-notebook figures.
Otherwise the tables above provide the complete numerical summary.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000124
# ╠═╡
md"""
## Notes

- **No models run on notebook load.** This notebook only reads stored results. To regenerate
  results, run `julia --project=. GeoMotionRun.jl --config=configs/snowcrab.toml` first.
- **No HTML dashboards.** The package no longer contains dashboard code
  (`src/dashboards.jl` was deleted; `export_dashboards` is a no-op shim). Visualisations
  are produced in-notebook from the stored results.
- **Results source:** `$(results_source)` — if this is `intermediate_checkpoint`, some panels
  (paths, diagnostics, validation, agents) will be unavailable because the intermediate
  checkpoint only stores `loaded`, `fitted`, and `kernels`.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000125
# ╠═╡
global plots_available = try
    using Plots
    true
catch
    false
end
if plots_available
    # Parameter bar chart
    params_df = DataFrame(
        Parameter = ["Dispersal (\\alpha)", "Residence (\\rho)", "Habitat (\\gamma)"],
        Value     = [Float64(kernels.alpha_hat), Float64(kernels.rho_hat), Float64(kernels.gamma_hat)]
    )
    bar(params_df.Parameter, params_df.Value;
        title = "Posterior Parameter Estimates",
        ylabel = "Estimate",
        legend = false,
        color = [:steelblue, :darkgreen, :darkorange])
else
    md"""
    `Plots` is not installed in this environment. Install it to see in-notebook figures:

    ```julia
    using Pkg
    Pkg.add("Plots")
    ```

    The HTML tables above provide the complete numerical summary of all stored results.
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000126
# ╠═╡
begin
    global plots_available = try
        using Plots
        true
    catch
        false
    end
    plots_available
end

# ╔═╡ Cell order:
# ╠═3f2c8a41-6b57-4d1e-9a83-2c5b7e1d0f44
# ╠═7a3f0c15-9b42-4e8d-a6f1-2b8c5d0e7a93
# ╠═c1000001-0001-0001-0001-000000000002
# ╠═8e2b6d47-3f19-4c7b-8d5a-1f4c9b2e0a36
# ╠═9d4e1b72-8c05-4a63-b7e1-6f0d3a8c5e21
# ╠═2b7f5d93-1e48-4c0a-8f6d-9a1c4e7b0d35
# ╠═5c1d8e64-2a79-4b5f-9e07-3d6a1c8b4f29
# ╠═7f3c9e52-4b18-4d6a-8e05-2c9b1d6a3f47
# ╠═c1000001-0001-0001-0001-000000000200
# ╠═e1000001-0001-0001-0001-000000000100
# ╠═e1000001-0001-0001-0001-000000000101
# ╠═e1000001-0001-0001-0001-000000000102
# ╠═e1000001-0001-0001-0001-000000000103
# ╠═e1000001-0001-0001-0001-000000000104
# ╠═e1000001-0001-0001-0001-000000000105
# ╠═e1000001-0001-0001-0001-000000000106
# ╠═e1000001-0001-0001-0001-000000000107
# ╠═e1000001-0001-0001-0001-000000000108
# ╠═e1000001-0001-0001-0001-000000000109
# ╠═e1000001-0001-0001-0001-000000000110
# ╠═e1000001-0001-0001-0001-000000000111
# ╠═e1000001-0001-0001-0001-000000000112
# ╠═e1000001-0001-0001-0001-000000000113
# ╠═e1000001-0001-0001-0001-000000000114
# ╠═e1000001-0001-0001-0001-000000000115
# ╠═e1000001-0001-0001-0001-000000000116
# ╠═e1000001-0001-0001-0001-000000000117
# ╠═e1000001-0001-0001-0001-000000000118
# ╠═e1000001-0001-0001-0001-000000000119
# ╠═e1000001-0001-0001-0001-000000000120
# ╠═e1000001-0001-0001-0001-000000000121
# ╠═e1000001-0001-0001-0001-000000000122
# ╠═e1000001-0001-0001-0001-000000000123
# ╠═e1000001-0001-0001-0001-000000000124
# ╠═e1000001-0001-0001-0001-000000000125
# ╠═e1000001-0001-0001-0001-000000000126
# ╠═e1000001-0001-0001-0001-000000000127
# ╠═e1000001-0001-0001-0001-000000000128
# ╠═e1000001-0001-0001-0001-000000000129
# ╠═e1000001-0001-0001-0001-000000000130
# ╠═e1000001-0001-0001-0001-000000000131
# ╠═e1000001-0001-0001-0001-000000000132
# ╠═e1000001-0001-0001-0001-000000000133
# ╠═e1000001-0001-0001-0001-000000000134
# ╠═e1000001-0001-0001-0001-000000000135
# ╠═e1000001-0001-0001-0001-000000000136
# ╠═e1000001-0001-0001-0001-000000000137
