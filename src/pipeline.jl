const _DepthRangeArg = Union{
    Nothing,
    Tuple{<:Real, <:Real},
    AbstractVector{<:Real},
    AbstractString,
}


using Arrow
using DataFrames

function _resolve_output_dir(output_dir::AbstractString)::String
    s = strip(String(output_dir))
    if isempty(s) || replace(s, '\\' => '/') == "../lakehouse/TethysPath"
        return movement_store_root()
    end
    return String(output_dir)
end

"""
    _require_r_backend(what) -> Nothing

Raise an error naming the missing object when an R-backed code path is reached
without `RCall` installed.

R is an optional dependency. `TethysPathRCallExt` is loaded automatically
when `RCall` and a working R are both present; this stub only fires when they are
not, and it names the specific input that needed them rather than surfacing an
`UndefVarError` from deep inside a loader.
"""
function _require_r_backend(what::AbstractString)
    Base.get_extension(@__MODULE__, :TethysPathRCallExt) === nothing && throw(
        ArgumentError(
            "$what requires the optional `RCall` backend, which is not loaded. " *
            "Install it with `import Pkg; Pkg.add(\"RCall\")` and ensure an R " *
            "installation with the `qs` and `arrow` R packages is configured. " *
            "Alternatively supply the data in Arrow, CSV, GeoJSON, or shapefile " *
            "form, which need no R."
        ),
    )
    return nothing
end

"""
    _r_ipc_convert(filepath, temp_ipc)

Convert a tabular R file to an Arrow IPC file, in R.

Implemented by `TethysPathRCallExt`. This stub exists so the package loads
without `RCall`.
"""
function _r_ipc_convert(filepath::AbstractString, temp_ipc::AbstractString)
    _require_r_backend("Reading R data ($filepath)")
    return nothing
end

"""
    _r_load_object(path)

Load an R object (`.rds` / `.RData`) for region-polygon extraction.

Implemented by `TethysPathRCallExt`. This stub exists so the package loads
without `RCall`.
"""
function _r_load_object(path::AbstractString)
    _require_r_backend("Reading R region polygons ($path)")
    return nothing
end

"""
    r_to_ipc(filepath::AbstractString)

Read tabular R data by utilizing an R session to convert it to a temporary 
Arrow IPC (Feather v2) file, which is then parsed into Julia.

This method achieves extremely high performance by writing the exact in-memory 
representation to disk. Julia reads the bytes into memory and constructs the 
DataFrame pointing directly at those bytes (zero-copy parsing), avoiding 
both memory duplication and deserialization CPU overhead.

# Analytical Assumptions
- **Tabular Data Requirement**: IPC exclusively supports tabular data. The R object 
  stored in the file must be a `data.frame`, `tibble`, or `data.table`.
- **Single Object extraction**: For `.rda` files, this extracts only the first 
  variable loaded alphabetically.
- **Garbage Collection**: The raw bytes of the file are held in a Julia byte array. 
  These bytes will remain in memory until the returned DataFrame is garbage collected.

# Arguments
- `filepath::AbstractString`: The path to the R data file.

# Returns
- A `DataFrame` containing the deserialized tabular data.

# Dependencies
- Julia packages: `RCall`, `Arrow`, `DataFrames`.
- R packages: `qs` and `arrow` must be installed.
"""
function r_to_ipc(filepath::AbstractString)
    if !isfile(filepath)
        error("File not found: ", filepath)
    end
    
    # Generate a temporary file path for the IPC intermediate
    temp_ipc = tempname() * ".arrow"
    
    try
        # The R conversion lives in TethysPathRCallExt so that the package
        # loads without an R installation.
        _r_ipc_convert(filepath, temp_ipc)

        # Read raw bytes into memory to avoid Windows file-locking issues from
        # memory-mapping the file directly on disk.
        ipc_bytes = read(temp_ipc)
        
        # Construct the Arrow Table from the in-memory byte array, and wrap it in a 
        # DataFrame without copying the underlying columns (zero-copy parsing).
        return DataFrame(Arrow.Table(ipc_bytes), copycols=false)
        
    finally
        # Safely delete the temporary file
        if isfile(temp_ipc)
            rm(temp_ipc)
        end
    end
end




# Ingest empirical telemetry data, construct 20 km hexagonal mesh, block
# terrestrial barriers, and infill unobserved marine HSI values


"""
    load_movement_dataset(;
        radius_km = 15.0,
        time_interval = :daily,
        crs = nothing,
        datum = WGS84Latest,
        ref_doy = 182,
        verbose = true,
        pre_mapped = nothing,
        data_dir = nothing
    ) -> NamedTuple

High-level convenience pipeline for empirical mark-recapture datasets
movement, telemetry, and environmental suitability data across Atlantic Canada.
Loads empirical mark-recapture encounters from JLD2 storage, constructs a unified
planar hexagonal domain grid covering the full extent of the
St. Lawrence, classifies and severs terrestrial land barriers, infills unobserved
marine Habitat Suitability Index (HSI) values via screened graph-Laplacian
Dirichlet diffusion, snaps telemetry observations to navigable marine units,
and stratifies event pairs into 4 biological demographic categories.

# Arguments
- `radius_km::Real`: Hexagon circumradius in kilometers (default `15.0`).
- `time_interval::Symbol`: Time discretization step (`:daily`, `:monthly`, `:weekly`).
- `crs`: Target coordinate reference system (defaults to local tangent projection).
- `datum`: Reference ellipsoid datum (defaults to `WGS84Latest`).
- `ref_doy::Int`: Annual survey reference day-of-year (default `182`).
- `verbose::Bool`: Enable progress and summary console output (default `true`).
- `pre_mapped`: Optional pre-computed mesh NamedTuple to bypass regeneration.
- `tagging_file`, `hsi_file`, `sppoly_file`, `surveydata_file`: dataset inputs.
  Defaults to the repository directory `docs/movement/data`.

# Returns
- `NamedTuple` containing:
  - `tagging::DataFrame`: Filtered telemetry records snapped to marine units.
  - `mesh::NamedTuple`: Full-domain grid with centroids and polygons.
  - `W::SparseMatrixCSC{Float64, Int}`: Adjacency matrix with land severed.
  - `hsi_vec::Vector{Float64}`: Infilled full-domain spatial HSI vector.
  - `monthly_hsi::Matrix{Float64}`: Monthly discretized HSI matrix.
  - `month_lookup::Dict`: Mapping of `(year, month)` to column indices.
  - `years::Vector{Int}`: Survey year span.
  - `obs::DataFrame`: Extracted release-recapture event pairs.
  - `group_lookup::Dict`: Stratum string to integer ID mapping.
  - `land_mask::Vector{Bool}`: Terrestrial barrier indicator vector.
"""
function load_movement_dataset(;
    radius_km     :: Real    = 15.0,
    time_interval :: Symbol  = :daily,
    crs                      = nothing,
    datum                    = WGS84Latest,
    ref_doy       :: Int     = 182,
    verbose       :: Bool    = true,
    pre_mapped               = nothing,
    data_dir      :: Union{Nothing, AbstractString} = nothing,
    tagging_file  :: Union{Nothing, AbstractString} = nothing,
    hsi_file      :: Union{Nothing, AbstractString} = nothing,
    sppoly_file   :: Union{Nothing, AbstractString} = nothing
)::NamedTuple

    dir = if data_dir !== nothing
        data_dir
    elseif isdir(joinpath(@__DIR__, "..", "inputs"))
        normpath(joinpath(@__DIR__, "..", "inputs"))
    elseif isdir(joinpath(@__DIR__, "inputs"))
        joinpath(@__DIR__, "inputs")
    else
        normpath(joinpath(@__DIR__, "..", "..", "docs", "movement", "inputs"))
    end

    actual_tagging = if tagging_file !== nothing
        tagging_file
    else
        # Fallback to defaults
        t_jld = joinpath(dir, "tagging.jld2")
        t_rdz = joinpath(dir, "tagging.rdz")
        t_rds = joinpath(dir, "tagging.rds")
        if isfile(t_rdz)
            t_rdz
        elseif isfile(t_rds)
            t_rds
        else
            t_jld
        end
    end

    isfile(actual_tagging) || error("Telemetry file not found: $actual_tagging")

    ext = lowercase(splitext(actual_tagging)[2])
    tagging = if ext == ".jld2"
        loaded_tag = JLD2.load(actual_tagging)
        loaded_tag isa AbstractDict ? (haskey(loaded_tag, "tagging") ? loaded_tag["tagging"] : first(values(loaded_tag))) : loaded_tag
    elseif ext in (".rdz", ".rds", ".rda", ".rdata", ".qs")
        r_to_ipc(actual_tagging)
    else
        error("Unsupported tagging file format: $ext")
    end

    actual_hsi = if hsi_file !== nothing
        hsi_file
    else
        h_zarr = joinpath(dir, "hsi.zarr")
        isdir(h_zarr) ? h_zarr : joinpath(dir, "hsi.jld2")
    end

    actual_sppoly = if sppoly_file !== nothing
        sppoly_file
    else
        joinpath(dir, "sppoly.jld2")
    end

    return prepare_movement_data(
        tagging;
        hsi_file      = ispath(actual_hsi) ? actual_hsi : nothing,
        sppoly_file   = isfile(actual_sppoly) ? actual_sppoly : nothing,
        radius_km     = radius_km,
        time_interval = time_interval,
        land_polygons = :none,
        crs           = crs,
        datum         = datum,
        ref_doy       = ref_doy,
        verbose       = verbose,
        pre_mapped    = pre_mapped
    )
end

# sc_data = load_movement_dataset(radius_km = 20.0, verbose = true)

# println("Active mark-recapture records: ", nrow(sc_data.obs))
# println("Domain units: ", sc_data.mesh.n_units, 
#         " (marine: ", count(!, sc_data.land_mask), 
#         ", land: ", sum(sc_data.land_mask), ")")


# =============================================================================
# Parameter Functions
# =============================================================================

"""
    movement_parameters_default() -> TethysPathConfig

The single canonical entry point for all analysis parameters.

Every code path -- library call, CLI invocation, or TOML file -- resolves to this
struct. A TOML file is applied as an overlay on top of these defaults (see
`load_config`), so a parameter absent from a config file simply keeps the default
declared in `src/config.jl`. A dataset is selected by its input files, not by a
separate code path: pass `tagging_file` (and the HSI and spatial-unit files that
accompany it) to analyse real data, and leave them unset for synthetic data.
"""
movement_parameters_default() = TethysPathConfig()

# =============================================================================
# Private Helpers
# =============================================================================

"""
    _parse_depth_range(depth_range) -> Union{Tuple{Float64,Float64}, Nothing}

Parses a depth range specification into a `(min, max)` float tuple, or
`nothing` when no depth constraint is desired. Accepts `Tuple`,
`AbstractVector`, or a comma/colon-separated `AbstractString` (e.g.
`"50,500"` or `"50:500"`).
"""
function _parse_depth_range(
    depth_range::_DepthRangeArg
)::Union{Tuple{Float64,Float64}, Nothing}
    depth_range === nothing && return nothing
    if depth_range isa AbstractString
        sep   = occursin(":", depth_range) ? ":" : ","
        parts = Base.split(depth_range, sep)
        length(parts) == 2 || throw(ArgumentError(
            "Invalid depth_range \"$depth_range\". Expected \"min,max\"."
        ))
        return (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2])))
    elseif length(depth_range) >= 2
        return (Float64(depth_range[1]), Float64(depth_range[2]))
    end
    return nothing
end

"""
    _extract_domain_depths(mesh, data, resharded_depths) -> Vector{Float64}

Extracts or synthesizes bathymetric depth values (m) for all mesh units.
Prioritizes: resharded fine hexagonal depths > loaded dataset depths >
mesh depth vector > synthetic bathymetric contours.
"""
function _extract_domain_depths(mesh, data, resharded_depths)::Vector{Float64}
    n = mesh.n_units
    if !isnothing(resharded_depths) && length(resharded_depths) == n
        return Float64.(resharded_depths)
    elseif hasproperty(data, :depth_vec) &&
           !isnothing(data.depth_vec) &&
           length(data.depth_vec) == n
        return Float64.(data.depth_vec)
    elseif hasproperty(mesh, :depth_vec) &&
           !isnothing(mesh.depth_vec) &&
           length(mesh.depth_vec) == n
        return Float64.(mesh.depth_vec)
    elseif hasproperty(mesh, :centroids_km) &&
           !isnothing(mesh.centroids_km) &&
           length(mesh.centroids_km) == n
        return [
            150.0 + 60.0 * sin(mesh.centroids_km[s][1] / 40.0) +
            40.0 * cos(mesh.centroids_km[s][2] / 40.0)
            for s in 1:n
        ]
    elseif hasproperty(mesh, :centroids_lonlat) &&
           !isnothing(mesh.centroids_lonlat) &&
           length(mesh.centroids_lonlat) == n
        return [
            150.0 + 60.0 * sin(
                (mesh.centroids_lonlat[s][1] + 63.0) * 5.0
            ) + 40.0 * cos(
                (mesh.centroids_lonlat[s][2] - 44.0) * 5.0
            )
            for s in 1:n
        ]
    else
        return fill(150.0, n)
    end
end
"""
    _resolve_hsi_for_time(loaded, t_decimal) -> Vector{Float64}

Returns the spatial HSI vector appropriate for decimal year `t_decimal`.

If `loaded.monthly_hsi` is non-empty and `loaded.month_lookup` contains an
entry for the (year, month) inferred from `t_decimal`, the corresponding
column of `monthly_hsi` is returned.  Otherwise the climatological mean
`loaded.hsi_vec` is returned as a fallback.

Decimal year `t_decimal` is decomposed as:
```
year  = floor(Int, t_decimal)
month = clamp(ceil(Int, (t_decimal - year) * 12), 1, 12)
```

# Arguments
- `loaded`: NamedTuple from `load_movement_data` (must contain `hsi_vec`,
  `monthly_hsi`, `month_lookup`).
- `t_decimal`: Release time in decimal years (e.g. 2018.583 ≈ August 2018).

# Returns
`Vector{Float64}` of length `n_spatial`.
"""
function _resolve_hsi_for_time(loaded, t_decimal::Real)::Vector{Float64}
    mhsi = loaded.monthly_hsi
    mlup = loaded.month_lookup
    if !isempty(mhsi) && !isempty(mlup)
        yr  = floor(Int, t_decimal)
        mo  = clamp(ceil(Int, (t_decimal - yr) * 12), 1, 12)
        col = get(mlup, (yr, mo), nothing)
        if col !== nothing && 1 <= col <= size(mhsi, 2)
            return mhsi[:, col]
        end
    end
    return loaded.hsi_vec
end

"""
    _bridge_depth_disconnected_basins(
        W_depth    :: SparseMatrixCSC,
        W_marine   :: SparseMatrixCSC,
        land_mask  :: BitVector,
        depth_mask :: BitVector
    ) -> SparseMatrixCSC

Augments the depth-constrained adjacency `W_depth` with minimal local bridge
edges that allow movement between depth-valid basins separated only by
out-of-depth marine transit zones.

## Algorithm

For each out-of-depth marine node `t` (i.e. `!land_mask[t] && depth_mask[t]`),
collect the set of its in-depth marine neighbours `N_t` in `W_marine`. Add a
symmetric bridge edge between every pair `(u, v) ∈ N_t × N_t` that is not
already adjacent in `W_depth`. Bridge weight equals the minimum non-zero entry
along column `t` of `W_marine`, preserving the adjacency scale.

This local one-hop strategy adds at most `deg(t)*(deg(t)-1)/2` edges per
transit node and is O(nnz(W_marine)) overall — no global BFS needed, no
O(n²) component-pair enumeration.
"""
function _bridge_depth_disconnected_basins(
    W_depth   ::SparseMatrixCSC{Float64, Int},
    W_marine  ::SparseMatrixCSC{Float64, Int},
    land_mask ::BitVector,
    depth_mask::BitVector
)::SparseMatrixCSC{Float64, Int}
    S           = size(W_depth, 1)
    depth_valid = BitVector(.!land_mask .& .!depth_mask)   # in-depth marine
    transit     = BitVector(.!land_mask .&  depth_mask)     # out-of-depth marine

    extra_I = Int[]
    extra_J = Int[]
    extra_V = Float64[]

    # Iterate over each transit node t; find its in-depth marine neighbours
    for t in findall(transit)
        # Collect in-depth neighbours of t in W_marine
        indepth_nbrs = Int[]
        w_min = Inf
        for ptr in W_marine.colptr[t]:(W_marine.colptr[t+1]-1)
            v = W_marine.rowval[ptr]
            if depth_valid[v]
                push!(indepth_nbrs, v)
            end
            w = W_marine.nzval[ptr]
            w > 0.0 && (w_min = min(w_min, w))
        end
        length(indepth_nbrs) < 2 && continue   # no bridge needed
        bridge_w = isinf(w_min) ? 1.0 : w_min

        # Add bridge between every pair of in-depth neighbours not yet adjacent
        for ii in 1:length(indepth_nbrs)
            u = indepth_nbrs[ii]
            for jj in (ii+1):length(indepth_nbrs)
                v = indepth_nbrs[jj]
                # Check whether u-v already adjacent in W_depth
                already = false
                for ptr in W_depth.colptr[u]:(W_depth.colptr[u+1]-1)
                    W_depth.rowval[ptr] == v && (already = true; break)
                end
                already && continue
                push!(extra_I, u); push!(extra_J, v); push!(extra_V, bridge_w)
                push!(extra_I, v); push!(extra_J, u); push!(extra_V, bridge_w)
            end
        end
    end

    isempty(extra_I) && return W_depth
    W_aug = W_depth + sparse(extra_I, extra_J, extra_V, S, S)
    dropzeros!(W_aug)
    return W_aug
end

"""
    _is_graph_reachable(W, start_node, end_node, max_k) -> Bool

Breadth-first search (BFS) on adjacency matrix `W` verifying that
`end_node` is reachable from `start_node` within `max_k` hops.
Returns `false` if either endpoint is isolated by a barrier.
"""
function _is_graph_reachable(
    W          ::AbstractMatrix{<:Real},
    start_node ::Int,
    end_node   ::Int,
    max_k      ::Int
)::Bool
    start_node == end_node && return true
    max_k <= 0             && return false
    S = size(W, 1)
    (!(1 <= start_node <= S) || !(1 <= end_node <= S)) && return false

    W_sp     = W isa SparseMatrixCSC ? W : sparse(W)
    visited  = falses(S)
    frontier = Int[start_node]
    visited[start_node] = true

    for _ in 1:max_k
        next_frontier = Int[]
        for u in frontier
            for ptr in W_sp.colptr[u]:(W_sp.colptr[u + 1] - 1)
                v = W_sp.rowval[ptr]
                v == end_node && return true
                if !visited[v]
                    visited[v] = true
                    push!(next_frontier, v)
                end
            end
        end
        frontier = next_frontier
        isempty(frontier) && break
    end
    return visited[end_node]
end

"""
    reconnect_severed_components(keep, W, centroids; max_bridge_cells = 64) -> (keep, n_link, n_unlinked)

Re-add the fewest possible units needed to make the retained set a single connected
body, and report what could not be connected.

# Why this exists
Pruning the mesh to a footprint can leave it in pieces. The domain filter keeps the
footprint and the units holding observation endpoints; everything between them may
be dropped, so a lobe of the domain joined to the rest only through pruned cells
comes away as its own component. The mark-recapture likelihood then conditions on a
transition that cannot happen, and the dashboard draws islands of ocean.

On the snow crab dataset that left 30 components around a 1,381-unit main body,
including a 39-unit lobe *inside* the declared footprint whose nearest retained
neighbour was two cells away.

# Method
Bridge through the lattice rather than inventing edges. Cost is 0 for a unit already
retained and 1 for one that would have to come back, so a 0-1 BFS from the largest
component returns the cheapest set of cells that reattaches each stranded component.
The cells on that path are added to `keep`; the hexagon adjacency in `W` then carries
the new connections with no synthetic edges and no invented distances.

`max_bridge_cells` bounds the cost. A component further away than that is reported
as unlinked rather than bridged through a long ribbon of re-added cells, which would
silently enlarge the domain by hundreds of units to reach one observation.

# Arguments
- `keep`: boolean vector over mesh units; the units currently retained.
- `W`: marine adjacency, the same graph the likelihood will use.
- `centroids`: `(lon, lat)` per unit, used only to break ties deterministically.
- `max_bridge_cells`: most cells that may be re-added to reattach one component.

# Returns
`(keep, n_link, n_unlinked)`: the amended mask, the number of units added, and the
number of stranded units that stayed stranded.
"""
function reconnect_severed_components(
    keep::AbstractVector{Bool},
    W::AbstractMatrix{<:Real},
    centroids::AbstractVector;
    navigable_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    max_bridge_cells::Int = 32
)::Tuple{BitVector, Int, Int}
    n = length(keep)
    keep = BitVector(keep)
    isempty(keep) && return (keep, 0, 0)

    # Adjacency lists, restricted strictly to navigable marine units so bridging
    # can never traverse terrestrial land, inland lakes, or severed barriers.
    nbrs = [Int[] for _ in 1:n]
    rows, cols = findnz(sparse(W))
    for (r, c) in zip(rows, cols)
        if navigable_mask !== nothing && (!navigable_mask[r] || !navigable_mask[c])
            continue
        end
        push!(nbrs[r], Int(c))
    end

    retained = findall(keep)
    n_link = 0
    n_unlinked = 0

    # Repeat: each pass attaches whatever a single 0-1 BFS can reach within budget.
    for _ in 1:(n + 1)
        comp = _connected_components(keep, nbrs)
        sizes = [length(c) for c in comp]
        isempty(sizes) && break
        main_k = argmax(sizes)
        main_comp = comp[main_k]

        # 0-1 BFS out of the main component over the whole lattice.
        dist = fill(typemax(Int), n)
        prev = zeros(Int, n)
        q = Int[]
        for u in main_comp
            dist[u] = 0
            push!(q, u)
        end
        head = 1
        while head <= length(q)
            u = q[head]; head += 1
            du = dist[u]
            for v in nbrs[u]
                c = keep[v] ? 0 : 1
                if du + c < dist[v]
                    dist[v] = du + c
                    prev[v] = u
                    push!(q, v)
                end
            end
        end

        progressed = false
        # Recomputed per pass and overwritten, not accumulated: a component that
        # stays stranded is seen on every iteration, and summing those sightings
        # reported the same units once per pass.
        pass_unlinked = 0
        for k in eachindex(comp)
            k == main_k && continue
            # Cheapest cell on a path from this component back to the main body.
            best_u = 0; best_d = typemax(Int)
            for u in comp[k]
                (keep[u] && dist[u] < best_d) || continue
                best_d = dist[u]; best_u = u
            end
            if best_u == 0 || best_d > max_bridge_cells
                pass_unlinked += length(comp[k])
                continue
            end
            # Walk `prev` back to the main component, re-adding pruned cells.
            v = best_u
            while v != 0 && dist[v] > 0
                if !keep[v]
                    keep[v] = true
                    n_link += 1
                end
                v = prev[v]
            end
            progressed = true
        end
        n_unlinked = pass_unlinked
        progressed || break
    end

    return (keep, n_link, n_unlinked)
end

"""
    _connected_components(keep, nbrs) -> Vector{Vector{Int}}

Connected components of the subgraph induced on `keep`, using adjacency list `nbrs`.
Only edges between two retained units count, so a component is genuinely connected
rather than connected through cells that are about to be removed.
"""
function _connected_components(
    keep::AbstractVector{Bool},
    nbrs::AbstractVector{<:AbstractVector{<:Integer}}
)::Vector{Vector{Int}}
    n = length(keep)
    seen = falses(n)
    comps = Vector{Vector{Int}}()
    for i in 1:n
        # Skip unless this unit is retained and not yet claimed. Written as a skip
        # on the negation, because `cond && continue` skips only when `cond` holds:
        # the inverse guard starts components at pruned cells and walks out of them.
        (!keep[i] || seen[i]) && continue
        comp = Int[]
        seen[i] = true
        q = [i]
        head = 1
        while head <= length(q)
            u = q[head]; head += 1
            push!(comp, u)
            for v in nbrs[u]
                (keep[v] && !seen[v]) || continue
                seen[v] = true
                push!(q, v)
            end
        end
        push!(comps, comp)
    end
    return comps
end

"""
    _snap_pruned_endpoints!(obs_df, keep, centroids) -> (n_rel, n_rec)

Repoint every observation endpoint that `keep` marks for deletion at the nearest
**retained** unit, in place. Returns the number of release and recapture endpoints
moved.

# Why this exists
The mesh prune reindexes observations with `old_to_new[r]`, where `old_to_new` is
`zeros(Int, n_spatial)` and only retained units get a nonzero entry. Nothing
checked the endpoint against the prune mask first, so an observation sitting on a
unit that the prune deleted was silently reindexed to **0**.

The units that reach that state are the ones the endpoint remap earlier in Phase 1c
does not cover. That remap flags endpoints with `out_bv`, which in the default
`:hsi_only` mode is the land-only barrier, whereas the prune mask is
`.!out_of_depth .& .!land_mask` — strictly stronger. An endpoint on an
out-of-depth *marine* unit therefore passed the remap unchanged, failed the
"observation points landing on them" protection in the mask pass (which requires
`keep_mesh_mask`), and was pruned.

Downstream this surfaced far from its cause, as a `KeyError: key (0, 729) not found`
out of `kstep_transition_cache`: the cache skips release indices outside
`1:S`, so `(0, 729)` was never built, and the likelihood then indexed it anyway.

Snapping to the nearest retained unit rather than dropping the observation matches
what Phase 1c already does for out-of-range endpoints, and displaces an endpoint by
at most one cell. Dropping instead would change the fitted dataset without being
counted anywhere.

# Arguments
- `obs_df`: observation table with `:release` and `:recapture` columns.
- `keep`: boolean vector over the *current* mesh units; `false` means "being pruned".
- `centroids`: `(lon, lat)` pairs, one per current mesh unit, or `nothing` when the
  mesh carries no centroids. Without them no snapping is possible and the endpoints
  are left alone for the caller's range check to reject.

# Notes
Endpoint indices are assumed to be in `1:length(keep)`, which
[`_is_graph_reachable`](@ref) guarantees for every observation that survives the
Phase 1c reachability filter: it returns `false` for an out-of-range endpoint, and
those rows are dropped before the prune runs. An out-of-range endpoint here is a
caller error and is reported rather than clamped.
"""
function _snap_pruned_endpoints!(
    obs_df    ::DataFrame,
    keep      ::AbstractVector{Bool},
    centroids ::Union{Nothing, AbstractVector}
)::Tuple{Int, Int}
    (isempty(keep) || centroids === nothing || nrow(obs_df) == 0) && return (0, 0)

    retained = findall(keep)
    isempty(retained) && error(
        "Mesh prune retained no units, so observation endpoints cannot be " *
        "reindexed. Widen depth_range or supply a spatial-domain polygon."
    )
    length(centroids) == length(keep) || error(
        "Endpoint snapping needs one centroid per mesh unit: got " *
        "$(length(centroids)) centroids for $(length(keep)) units."
    )
    retained_cents = centroids[retained]

    n_rel = 0
    n_rec = 0
    for (col, is_rel) in ((:release, true), (:recapture, false))
        hasproperty(obs_df, col) || error(
            "Endpoint snapping requires a `$(col)` column on the observation table."
        )
        units = obs_df[!, col]
        for (i, u) in pairs(units)
            (1 <= u <= length(keep)) || error(
                "Observation row $i has $(col) = $u, outside 1:$(length(keep)). " *
                "Endpoint indices must be valid mesh units before the prune reindex."
            )
        end
        stale_idx = findall(u -> !keep[u], units)
        isempty(stale_idx) && continue
        lons = [Float64(centroids[units[i]][1]) for i in stale_idx]
        lats = [Float64(centroids[units[i]][2]) for i in stale_idx]
        obs_df[stale_idx, col] = retained[map_to_units(lons, lats, retained_cents)]
        if is_rel
            n_rel = length(stale_idx)
        else
            n_rec = length(stale_idx)
        end
    end
    return (n_rel, n_rec)
end

"""
    _resolve_centroids(mesh, n_spatial)
        -> (cents_planar, cents_lonlat, cents_mesh)

Resolves planar (km) and geographic (lon/lat) centroid vectors from a
mesh object, returning `(planar, lonlat, preferred_for_mapping)`.
"""
function _resolve_centroids(mesh, n_spatial)
    cents_km = hasproperty(mesh, :centroids_km) &&
               !isnothing(mesh.centroids_km) &&
               length(mesh.centroids_km) == n_spatial
    cents_ll = hasproperty(mesh, :centroids_lonlat) &&
               !isnothing(mesh.centroids_lonlat) &&
               length(mesh.centroids_lonlat) == n_spatial
    cents_c  = hasproperty(mesh, :centroids) && !isnothing(mesh.centroids)

    cents_planar = cents_km ? mesh.centroids_km :
                   cents_ll ? mesh.centroids_lonlat :
                   cents_c  ? mesh.centroids : nothing
    cents_lonlat = cents_ll ? mesh.centroids_lonlat :
                   cents_c  ? mesh.centroids : nothing
    cents_mesh   = cents_lonlat !== nothing ? cents_lonlat : cents_planar
    return (cents_planar, cents_lonlat, cents_mesh)
end

# =============================================================================
# Phase 1: Data Ingestion
# =============================================================================

"""
    load_movement_data(params) -> NamedTuple

Phase 1 of the pipeline. Loads or simulates a spatial telemetry dataset,
optionally reshards the domain to a finer regular hexagonal lattice using
LibGEOS, ingests 3D hydrodynamic bathymetry, and enforces depth-range
traversal barriers.

# Arguments
- `params`: Configuration NamedTuple from `movement_parameters_*`.
  Relevant keys: `data_source`, `reshard_hex`, `hex_radius_km`,
  `use_hydrodynamics`, `depth_range`, `seed`, `verbose`.

# Returns
`NamedTuple` with fields: `data`, `mesh`, `W`, `hsi_vec`, `obs_df`,
`survey_df`, `group_map`, `land_mask`, `n_spatial`, `resharded_hydro`,
`resharded_depths`, `parsed_depth_range`.
"""
function load_movement_data(params)::NamedTuple
    verbose = params.verbose

    verbose && println("=" ^ 72)
    verbose && println("  TethysPath Pipeline")
    verbose && println("=" ^ 72)

    # -- 1a. Load or generate the dataset -----------------------------------
    # The dataset is identified by the input files themselves: when a tagging
# file is configured, that file -- together with whichever HSI and
# spatial-unit files accompany it -- *is* the dataset. With no tagging file
# there is nothing to read, so a synthetic dataset is generated instead.
# There is no dataset name to match on, and no silent fallback: a configured
# file that cannot be read is an error, not a reason to swap in simulated
# data and report a successful run of the wrong analysis.
data = if isnothing(params.tagging_file)
    verbose && println(
        "\n[Phase 1] No tagging file configured; generating a synthetic dataset"
    )
    generate_movement_data()
else
    verbose && println("\n[Phase 1] Ingesting dataset from $(params.tagging_file)")
    isfile(params.tagging_file) || error(
        "Configured tagging_file does not exist: $(params.tagging_file)"
    )
    load_movement_dataset(
        radius_km = params.hex_radius_km,
        verbose = verbose,
        tagging_file = params.tagging_file,
        hsi_file = params.hsi_file,
        sppoly_file = params.sppoly_file,
    )
end
    mesh      = data.mesh
    W         = data.W
    hsi_vec   = data.hsi_vec
    obs_df    = data.obs

    # Delimit domain extent using sppoly bounds when available
    sppoly_path = if !isnothing(params.sppoly_file) && isfile(params.sppoly_file)
        params.sppoly_file
    else
        nothing
    end
    sppoly_bounds = extract_sppoly_bounds(sppoly_path)
    if sppoly_bounds !== nothing && verbose
        # Reported as a union, because that is what it now is. This line used to
        # say "Delimiting southern/south-western domain", describing a clamp that
        # shrank the mesh *inside* the footprint and left the footprint's own
        # western and southern slivers unmeshed. It was the reason the rendered
        # domain did not cover the spatial-unit file.
        println(
            "  Mesh extent unioned with sppoly bounds: " *
            "lon $(round(sppoly_bounds[1]; digits=4))..$(round(sppoly_bounds[3]; digits=4)), " *
            "lat $(round(sppoly_bounds[2]; digits=4))..$(round(sppoly_bounds[4]; digits=4))"
        )
    end

    # The analysis domain is an explicit bounding box when configured, and
    # otherwise the extent of the input data plus a small padding so the mesh
    # is not clipped flush against the outermost detections.
    domain_bbox = resolve_bbox(
        params.bbox, params.bbox_padding_deg,
        hasproperty(obs_df, :lon) ? Float64[obs_df.lon...] : Float64[],
        hasproperty(obs_df, :lat) ? Float64[obs_df.lat...] : Float64[];
        sppoly_bounds = sppoly_bounds
    )
    verbose && println(
        "  Mesh bounding box (W, S, E, N) = " *
        "($(domain_bbox[1]), $(domain_bbox[2]), $(domain_bbox[3]), $(domain_bbox[4]))"
    )

    # User-supplied land polygons are *additional* to whatever `land_source`
    # provides, not a replacement for it.
    #
    # They used to be read only when `land_source === :polygons`, in which case the
    # global raster was skipped entirely. That made it impossible to say "this
    # coastline is right, and this one feature is also land": choosing the polygons
    # meant opting out of every continent and island on the map. `land_source`
    # selects the base mask; the polygon files add to it.
    land_polys = if isempty(params.land_polygon_files)
        nothing
    else
        rings = reduce(vcat, read_polygon_file.(params.land_polygon_files))
        params.verbose && println(
            "  Land polygons: $(length(params.land_polygon_files)) file(s), " *
            "$(length(rings)) ring(s) added to the land mask."
        )
        rings
    end
    params.land_source === :polygons && land_polys === nothing && error(
        "land_source = :polygons requires land_polygon_files"
    )
    survey_df = if !isnothing(params.surveydata_file)
    load_survey_data(params.surveydata_file; verbose = verbose)
elseif hasproperty(data, :survey_df)
    data.survey_df
else
    nothing
end
    group_map = hasproperty(data, :group_lookup) ?
                data.group_lookup : Dict(1 => "All")

    region_map = nothing
    conn_region_labels = nothing
    bottom_temp = nothing
    land_mask = hasproperty(data, :land_mask) ? data.land_mask : nothing
    n_spatial = mesh.n_units

    # Time-varying HSI, taken now rather than just before the return, so the
    # resharding block below can rescale it onto the fine mesh alongside
    # `hsi_vec` and `W`.
    monthly_hsi  = hasproperty(data, :monthly_hsi) ? data.monthly_hsi : Matrix{Float64}(undef, 0, 0)
    month_lookup = hasproperty(data, :month_lookup) ? data.month_lookup : Dict{Tuple{Int,Int}, Int}()
    years_vec    = hasproperty(data, :years) ? data.years : Int[]

    if verbose
        println("  Spatial mesh units : $n_spatial")
        if land_mask !== nothing
            println("    Marine    : $(count(!, land_mask))")
            println("    Land      : $(sum(land_mask))")
        end
        println("  Mark-recapture obs : $(nrow(obs_df))")
        println("  Biological groups  : $(length(group_map))")
    end

    # -- 1b. LibGEOS Hexagonal Resharding & Hydrodynamics --------------------
    resharded_hydro  = nothing
    resharded_depths = nothing

    if params.reshard_hex || params.use_hydrodynamics
        verbose && println(
            "\n[Phase 1b] Resharding to fine hexagons via LibGEOS..."
        )
        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        min_lon, max_lon = extrema(cents_lon)
        min_lat, max_lat = extrema(cents_lat)



        # `source` was unreachable from configuration, so a measured depth grid
        # could not be used even if one existed on disk. It is passed through
        # untouched otherwise: with nothing configured the loader fabricates a
        # shelf, and `is_synthetic` downstream is what disables depth rules.
        bathy = load_open_bathymetry(;
            source        = params.bathymetry_source === nothing ?
                                :synthetic : params.bathymetry_source,
            lon_range     = (min_lon - 0.2, max_lon + 0.2),
            lat_range     = (min_lat - 0.2, max_lat + 0.2),
            resolution_deg = 0.02
        )
        hydro = extract_hydrodynamic_dataset(bathy;
            depth_levels = [0.0, 25.0, 50.0, 100.0, 175.0],
            times        = [1.0]
        )
        fine_mesh = build_hex_mesh(
            bathy.grid_lon, bathy.grid_lat;
            radius_km = Float64(params.hex_radius_km)
        )
        verbose && println(
            "  Fine hexagonal units: $(fine_mesh.n_units) " *
            "(radius = $(params.hex_radius_km) km)"
        )

        P_transfer       = compute_network_transfer_matrix(
            bathy.au, fine_mesh; method = :area_weighted
        )
        resharded_hydro  = reshard_spatial_field(P_transfer, hydro)
        # Transfer signed elevation rather than zero-clamped depth so coastal cells
        # predominantly on positive-elevation land are not assigned artificial marine depth.
        resharded_elev = hasproperty(bathy, :elevation) ?
            reshard_spatial_field(P_transfer, vec(bathy.elevation)) :
            .-reshard_spatial_field(P_transfer, bathy.depths)
        resharded_depths = Float64[e >= 0.0 ? 0.0 : -Float64(e) for e in resharded_elev]
        signed_depth_for_land = Float64[-Float64(e) for e in resharded_elev]
        bottom_temp = if !(hasproperty(bathy, :is_synthetic) && bathy.is_synthetic)
            try
                hydro_woa = Tethys.fetch_boundary_hydrography(
                    :woa23;
                    lon_range = (min_lon, max_lon),
                    lat_range = (min_lat, max_lat),
                    month     = 0,
                    verbose   = false
                )
                Float64[
                    hydro_woa.T(
                        Float64(fine_mesh.centroids_lonlat[u][1]),
                        Float64(fine_mesh.centroids_lonlat[u][2]),
                        -abs(Float64(resharded_depths[u])),
                        0.0
                    )
                    for u in 1:fine_mesh.n_units
                ]
            catch
                hasproperty(resharded_hydro, :temperature) ?
                    Float64.(resharded_hydro.temperature) : nothing
            end
        elseif hasproperty(resharded_hydro, :temperature)
            Float64.(resharded_hydro.temperature)
        else
            nothing
        end

        mesh      = fine_mesh
        W         = fine_mesh.W
        fine_polys = hasproperty(fine_mesh, :polygons_lonlat) ?
            fine_mesh.polygons_lonlat : nothing
        land_mask = identify_land_units(
            fine_mesh.centroids_lonlat;
            polygons        = fine_polys,
            depth           = signed_depth_for_land,
            depth_threshold = 2.0,
            land_polygons   = land_polys,
        )
        if params.land_source === :landmask
            land_mask .|= land_mask_from_global_mask(
                fine_mesh.centroids_lonlat;
                polygons     = fine_polys,
                resolution   = params.land_mask_resolution,
                grid_minutes = params.land_mask_grid_minutes,
            )
            verbose && println(
                "  Land/lake units: $(count(land_mask)) / $(fine_mesh.n_units) " *
                "(global $(params.land_mask_grid_minutes)-min '$(params.land_mask_resolution)' mask)"
            )
        end

        P_orig_to_fine = compute_network_transfer_matrix(
            data.mesh, fine_mesh; method = :area_weighted
        )
        hsi_fine = if params.use_geodata_habitat
            verbose && println("  Evaluating real habitat suitability via Tethys...")
            build_geodata_habitat_suitability(
                fine_mesh.centroids_lonlat;
                polygons = fine_polys,
                lon_range = (min_lon, max_lon),
                lat_range = (min_lat, max_lat),
                depths = resharded_depths,
                bathymetry_source = params.bathymetry_source === nothing ? :etopo2022 : params.bathymetry_source,
                temperature_source = params.temperature_source,
                month = params.temperature_month,
                verbose = verbose
            )
        elseif hasproperty(data, :hsi_vec) && !isnothing(data.hsi_vec) &&
                      length(data.hsi_vec) == data.mesh.n_units
            reshard_spatial_field(P_orig_to_fine, data.hsi_vec)
        elseif resharded_hydro !== nothing && hasproperty(resharded_hydro, :hsi)
            resharded_hydro.hsi
        else
            fill(0.5, fine_mesh.n_units)
        end

        W, hsi_vec = apply_land_barrier(
            fine_mesh.W, hsi_fine, land_mask
        )

        sever_land_crossing_edges!(
            W, fine_mesh.centroids_lonlat; land_polygons = land_polys
        )
        if params.land_source === :landmask
            n_sev_ras = sever_raster_land_edges!(
                W, fine_mesh.centroids_lonlat;
                resolution   = params.land_mask_resolution,
                grid_minutes = params.land_mask_grid_minutes
            )
            verbose && n_sev_ras > 0 && println(
                "  Severed $(div(n_sev_ras, 2)) overland/isthmus adjacency edges."
            )
        end

        n_enclosed = mask_enclosed_inland_pockets!(land_mask, W)
        if n_enclosed > 0
            W, hsi_vec = apply_land_barrier(W, hsi_vec, land_mask)
            verbose && println(
                "  Masked $n_enclosed enclosed inland/lake pocket unit(s)."
            )
        end

        if params.land_source === :landmask && fine_polys !== nothing
            n_clip = clip_polygons_to_ocean!(
                fine_polys, fine_mesh.centroids_lonlat;
                land_mask    = land_mask,
                resolution   = params.land_mask_resolution,
                grid_minutes = params.land_mask_grid_minutes
            )
            verbose && n_clip > 0 && println(
                "  Clipped $n_clip coastal hexagon boundary polygons to ocean shoreline."
            )
        end

        if !isempty(monthly_hsi) && size(monthly_hsi, 1) != fine_mesh.n_units
            monthly_hsi = reshard_spatial_field(P_orig_to_fine, monthly_hsi)
            for col in axes(monthly_hsi, 2)
                monthly_hsi[land_mask, col] .= 0.0
            end
            verbose && println(
                "  Rescaled monthly HSI to $(size(monthly_hsi, 1)) x " *
                "$(size(monthly_hsi, 2)) fine-mesh units"
            )
        elseif !isempty(monthly_hsi) && size(monthly_hsi, 1) == fine_mesh.n_units
            for col in axes(monthly_hsi, 2)
                monthly_hsi[land_mask, col] .= 0.0
            end
        end
        n_spatial  = fine_mesh.n_units

        # Remap mark-recapture observations directly from GPS coordinates onto active
        # open-marine units so coastal releases never bounce through inland cells.
        obs_df       = copy(obs_df)
        fine_cents   = fine_mesh.centroids_lonlat
        marine_mask  = .!land_mask .& (vec(sum(W; dims = 2)) .> 0)
        marine_units = let mu = findall(marine_mask)
            isempty(mu) ? collect(1:n_spatial) : mu
        end
        marine_cents = fine_cents[marine_units]

        has_gps = hasproperty(obs_df, :lon) && hasproperty(obs_df, :lat) &&
                  hasproperty(obs_df, :rec_lon) && hasproperty(obs_df, :rec_lat)
        rel_lons = has_gps ? Float64.(obs_df.lon) :
            [Float64(data.mesh.centroids_lonlat[r][1]) for r in obs_df.release]
        rel_lats = has_gps ? Float64.(obs_df.lat) :
            [Float64(data.mesh.centroids_lonlat[r][2]) for r in obs_df.release]
        rec_lons = has_gps ? Float64.(obs_df.rec_lon) :
            [Float64(data.mesh.centroids_lonlat[r][1]) for r in obs_df.recapture]
        rec_lats = has_gps ? Float64.(obs_df.rec_lat) :
            [Float64(data.mesh.centroids_lonlat[r][2]) for r in obs_df.recapture]

        rel_sub = map_to_units(rel_lons, rel_lats, marine_cents)
        rec_sub = map_to_units(rec_lons, rec_lats, marine_cents)
        obs_df.release   = marine_units[rel_sub]
        obs_df.recapture = marine_units[rec_sub]

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            survey_df       = copy(survey_df)
            orig_surv       = [data.mesh.centroids_lonlat[s]
                                for s in survey_df.s_idx]
            surv_sub        = map_to_units(
                [c[1] for c in orig_surv],
                [c[2] for c in orig_surv],
                marine_cents
            )
            survey_df.s_idx = marine_units[surv_sub]
            survey_df.depth = resharded_depths[survey_df.s_idx]
        end
        verbose && println("  Resharding complete.")
    end

    # -- 1b-ii. Adaptive Multiresolution Hexagonal Mesh ----------------------
    if params.adaptive_mesh
        verbose && println(
            "\n[Phase 1b-ii] Constructing adaptive multiresolution domain..."
        )
        cents_lon = [Float64(c[1]) for c in mesh.centroids_lonlat]
        cents_lat = [Float64(c[2]) for c in mesh.centroids_lonlat]
        rc = params.coarse_radius_km
        rf = params.fine_radius_km
        mesh = construct_adaptive_multiresolution_domain(
            cents_lon, cents_lat;
            coarse_radius_km = rc,
            fine_radius_km   = rf,
            land_polygons    = land_mask !== nothing ? land_mask : :none
        )
        W = mesh.W
        land_mask = mesh.land_mask
        n_spatial = mesh.n_units
        hsi_vec = ones(Float64, n_spatial)

        # Remap release/recapture locations to nearest multiresolution units
        cents_xy = [[c[1], c[2]] for c in mesh.centroids_km]
        tree_m = KDTree(hcat(cents_xy...))
        orig_rel_c = [data.mesh.centroids_km[r] for r in obs_df.release]
        orig_rec_c = [data.mesh.centroids_km[r] for r in obs_df.recapture]
        obs_df.release = [
            knn(tree_m, [c[1], c[2]], 1)[1][1] for c in orig_rel_c
        ]
        obs_df.recapture = [
            knn(tree_m, [c[1], c[2]], 1)[1][1] for c in orig_rec_c
        ]
        verbose && println(
            "  Multiresolution mesh: $n_spatial units " *
            "(Fine: $(count(mesh.is_fine)))"
        )
    end

    # Resolve management-region polygons on the active pre-prune mesh so every fine
    # hexagonal unit (including CFA 19 and Southern Gulf) receives its region label.
    if !isempty(params.region_polygon_files) && hasproperty(mesh, :centroids_lonlat)
        labels = isempty(params.region_labels) ?
            ["Region $i" for i in eachindex(params.region_polygon_files)] :
            params.region_labels
        polys = load_region_polygons(params.region_polygon_files, labels)
        region_map = region_map_from_polygons(mesh.centroids_lonlat, polys)
        explicit = isnothing(params.region_map) ? Dict{Int, Int}() : params.region_map
        for (unit, r) in explicit
            1 <= unit <= length(region_map) && (region_map[unit] = r)
        end
        n_r = maximum(region_map; init = 0)
        n_r >= 1 || error(
            "No mesh unit falls inside any region polygon; check region_polygon_files."
        )
        conn_region_labels = isempty(params.region_labels) ?
            ["Region $i" for i in 1:n_r] : params.region_labels
        verbose && println(
            "  Regions: $(length(conn_region_labels)) from polygon files; " *
            "$(count(>(0), region_map)) / $(length(region_map)) units assigned"
        )
    elseif !isnothing(params.region_map) && !isempty(params.region_map)
        region_map = ones(Int, n_spatial)
        for (unit, r) in params.region_map
            1 <= unit <= n_spatial && (region_map[unit] = r)
        end
        n_r = maximum(region_map)
        conn_region_labels = isempty(params.region_labels) ?
            ["Region $i" for i in 1:n_r] : params.region_labels
    end

    # -- 1c. Depth Range Traversal Barrier -----------------------------------
    parsed_depth_range = _parse_depth_range(params.depth_range)

    if parsed_depth_range !== nothing
        min_d, max_d = parsed_depth_range
        verbose && println(
            "\n[Phase 1c] Depth range requested: [$min_d, $max_d] m"
        )
        depths_vec   = _extract_domain_depths(mesh, data, resharded_depths)

        # A synthetic bathymetry must not decide which units exist.
        #
        # `load_open_bathymetry` defaults to `source = :synthetic` and no caller
        # ever overrides it, so the depth field driving `depth_range` is fabricated
        # -- and its geometry is a generic shelf whose slope runs southwest-to-
        # northeast in normalised window coordinates, which puts its "abyss" over
        # the southwest. On the snow crab window that fabricates >350 m water across
        # the southwestern half of the `sppoly` footprint, so applying the depth
        # window to it deleted the footprint's southern and western lobes: the
        # rendered domain fell to ~12% of the footprint, 0% below 45.5 N, while
        # looking like a tessellation bug.
        #
        # A fabricated field cannot support a spatial restriction, so when it is in
        # use the depth window does not prune. The domain is then footprint and land,
        # which is the best-supported answer available. Reported, not silent.
        synthetic_depths = hasproperty(bathy, :is_synthetic) && bathy.is_synthetic
        if synthetic_depths
            out_of_depth = falses(length(depths_vec))
            @error """
                  Bathymetry is SYNTHETIC (load_open_bathymetry defaulted to \
                  source=:synthetic), so depth is not used to restrict the spatial \
                  domain or to sever connectivity: the domain is the configured \
                  footprint minus land, and depth_barrier_mode has no effect. \
                  Supply a measured depth grid to restrict by depth.
                  """ maxlog = 1
            verbose && println(
                "  Depth window NOT applied: bathymetry is synthetic, not measured."
            )
        else
            out_of_depth = BitVector([d < min_d || d > max_d for d in depths_vec])
        end

        if verbose
            n_allowed = count(!, out_of_depth)
            if synthetic_depths
                # Not "all units in range": no unit was tested. Printing a pass
                # count here read as though the depth window had been applied.
                println(
                    "  Depth window not evaluated: no measured bathymetry."
                )
            else
                println(
                    "  Units in depth range : $n_allowed / $(length(depths_vec))"
                )
            end
        end

        # depth_barrier_mode controls how out-of-depth marine nodes are treated:
        #
        #   :hsi_only (default)
        #       W uses land-only barrier (full marine connectivity preserved).
        #       Out-of-depth nodes have HSI clamped to a small floor value so
        #       the habitat-gradient term strongly discourages transit through
        #       them without creating structural disconnections.
        #
        #   :hard
        #       W uses depth+land combined barrier (hard structural constraint).
        #       Observations spanning disconnected depth-corridor basins are
        #       dropped. Use when depth imposes a true physical barrier.
        #
        #   :bridge
        #       W uses depth+land combined barrier, then adds sparse bridge
        #       edges wherever two in-depth basins share a marine transit
        #       corridor (shallow/deep). Avoids drops but can generate many
        #       extra edges when basins are numerous.
        depth_mode = params.depth_barrier_mode

        # Land-only W: always needed for reachability checks and :hsi_only mode
        land_only_bv = land_mask !== nothing ?
            BitVector(land_mask) : falses(n_spatial)
        W_marine_only, _ = apply_land_barrier(W, hsi_vec, land_only_bv)

        if depth_mode == :hsi_only
            # Structural W: land barrier only
            W, hsi_vec = apply_land_barrier(W, hsi_vec, land_only_bv)
            # Encode depth preference via HSI floor for out-of-depth nodes
            hsi_floor  = params.hsi_ood_floor
            hsi_vec[out_of_depth .& .!land_only_bv] .= min.(
                hsi_vec[out_of_depth .& .!land_only_bv], hsi_floor
            )
            land_mask = land_only_bv   # structural mask = land only
            verbose && println(
                "  Mode :hsi_only -- depth preference via HSI floor ($hsi_floor);"*
                " W connectivity uses land-only barrier."
            )
        else
            combined_barrier = land_mask !== nothing ?
                               BitVector(land_mask .| out_of_depth) :
                               out_of_depth
            W, hsi_vec = apply_land_barrier(W, hsi_vec, combined_barrier)
            land_mask  = combined_barrier

            if depth_mode == :bridge
                W_pre = W
                land_only_snap = land_only_bv
                W = _bridge_depth_disconnected_basins(
                    W, W_marine_only, land_only_snap, out_of_depth
                )
                n_bridge_edges = div(nnz(W) - nnz(W_pre), 2)
                verbose && n_bridge_edges > 0 && println(
                    "  Added $n_bridge_edges bridge edges through out-of-depth " *
                    "marine transit zones."
                )
            end
        end

        # Identify in-depth (or in-mode) valid units for endpoint remapping
        # In :hsi_only mode valid = marine (land_mask = land-only)
        # In :hard/:bridge mode valid = in-depth marine
        valid_units = if depth_mode == :hsi_only
            findall(.!land_only_bv)
        else
            findall(.!land_mask)
        end
        n_obs_orig = nrow(obs_df)

        if !isempty(valid_units)
            cents_raw = hasproperty(mesh, :centroids_lonlat) ?
                mesh.centroids_lonlat :
                hasproperty(mesh, :centroids) ? mesh.centroids : nothing

            obs_df = copy(obs_df)
            obs_df[!, :release_orig]   = copy(obs_df.release)
            obs_df[!, :recapture_orig] = copy(obs_df.recapture)

            if cents_raw !== nothing
                valid_cents = cents_raw[valid_units]

                # Combined barrier for endpoint-out-of-range check
                out_bv = depth_mode == :hsi_only ? land_only_bv : land_mask

                bad_rel_mask = [out_bv[r] for r in obs_df.release]
                bad_rec_mask = [out_bv[r] for r in obs_df.recapture]
                n_remapped_rel = count(bad_rel_mask)
                n_remapped_rec = count(bad_rec_mask)

                if n_remapped_rel > 0
                    bad_rel_idx   = findall(bad_rel_mask)
                    bad_rel_units = obs_df.release[bad_rel_idx]
                    lons = [Float64(cents_raw[u][1]) for u in bad_rel_units]
                    lats = [Float64(cents_raw[u][2]) for u in bad_rel_units]
                    local_idx = map_to_units(lons, lats, valid_cents)
                    obs_df.release[bad_rel_idx] = valid_units[local_idx]
                end

                if n_remapped_rec > 0
                    bad_rec_idx   = findall(bad_rec_mask)
                    bad_rec_units = obs_df.recapture[bad_rec_idx]
                    lons = [Float64(cents_raw[u][1]) for u in bad_rec_units]
                    lats = [Float64(cents_raw[u][2]) for u in bad_rec_units]
                    local_idx = map_to_units(lons, lats, valid_cents)
                    obs_df.recapture[bad_rec_idx] = valid_units[local_idx]
                end

                if verbose && (n_remapped_rel + n_remapped_rec) > 0
                    println(
                        "  Remapped $n_remapped_rel release / " *
                        "$n_remapped_rec recapture endpoints to nearest " *
                        "in-depth marine unit."
                    )
                end
            end
        end

        # Reachability check: use land-only W so depth-mode differences don't
        # matter here; if truly unreachable through any marine path the
        # observation is a coordinate error or genuine isolation.
        W_reach = W_marine_only   # land-only W for checking
        n_k0    = count(r -> r.k <= 0 && r.release != r.recapture, eachrow(obs_df))
        reachable_mask = [
            _is_graph_reachable(W_reach, r.release, r.recapture, r.k)
            for r in eachrow(obs_df)
        ]
        n_dropped = count(.!reachable_mask)

        if n_dropped > 0
            dropped_df = obs_df[.!reachable_mask, :]
            if verbose
                println(
                    "  Dropped $n_dropped obs unreachable via any marine route " *
                    "(likely coordinate/datum errors):"
                )
                cents_raw = hasproperty(mesh, :centroids_lonlat) ?
                    mesh.centroids_lonlat : nothing
                for r in eachrow(dropped_df)
                    rel_coord = cents_raw !== nothing ?
                        "($(round(cents_raw[r.release_orig][1]; digits=4)), " *
                        "$(round(cents_raw[r.release_orig][2]; digits=4)))" :
                        "node $(r.release_orig)"
                    rec_coord = cents_raw !== nothing ?
                        "($(round(cents_raw[r.recapture_orig][1]; digits=4)), " *
                        "$(round(cents_raw[r.recapture_orig][2]; digits=4)))" :
                        "node $(r.recapture_orig)"
                    println(
                        "    tagid=$(r.tagid)  k=$(r.k)" *
                        "  rel=$rel_coord  rec=$rec_coord"
                    )
                end
            end
        end
        obs_df = obs_df[reachable_mask, :]

        nrow(obs_df) == 0 && error(
            "Depth constraint [$min_d, $max_d] m excluded all " *
            "mark-recapture events. Widen depth_range."
        )

        if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
            # In :hsi_only mode survey units don't need depth filtering
            if depth_mode != :hsi_only
                survey_df = survey_df[
                    [!land_mask[s] for s in survey_df.s_idx], :
                ]
            end
        end

        # Prune out-of-depth and unreachable/boundary-exceeded units from the final mesh
        # so that downstream network operations and interactive dashboards do not retain
        # an oversized, empty rectangular domain.
        keep_mesh_mask = .!out_of_depth .& .!land_mask

        cents_tmp = hasproperty(mesh, :centroids_lonlat) ?
            mesh.centroids_lonlat : mesh.centroids

        # Domain footprint. When a spatial-unit file is configured, its areal
        # units *are* the study domain and nothing outside them is retained.
        #
        # The read used to be wrapped in `try ... catch e; end` with an empty body,
        # so a file this reader could not interpret left `domain_geom === nothing`
        # and every unit below was kept. The failure mode was invisible in the log
        # and the rendered domain became the whole mesh extent. A configured file
        # that cannot be read is an error, like every other configured input here.
        domain_geom = read_domain_polygon(params.sppoly_file)
        in_domain = geometries_in_domain(domain_geom, cents_tmp)
        in_regions = region_map !== nothing && length(region_map) == n_spatial ?
            BitVector(region_map .> 0) : falses(n_spatial)

        # Where survey HSI estimates do not exist outside the sppoly footprint
        # (such as CFA 19 and the Southern Gulf of St. Lawrence), assign a
        # spatially and temporally homogeneous HSI (0.50) across in-depth marine
        # units rather than boundary-extrapolated gradients or 1-cell bridges.
        if domain_geom !== nothing
            unsurveyed_marine = .!in_domain .& .!land_mask
            if any(unsurveyed_marine)
                hsi_vec[unsurveyed_marine] .= 0.50
                if depth_mode == :hsi_only
                    ood_unsurveyed = unsurveyed_marine .& out_of_depth
                    hsi_vec[ood_unsurveyed] .= min.(
                        hsi_vec[ood_unsurveyed], params.hsi_ood_floor
                    )
                end
                if !isempty(monthly_hsi) && size(monthly_hsi, 1) == n_spatial
                    for col in axes(monthly_hsi, 2)
                        monthly_hsi[unsurveyed_marine, col] .= hsi_vec[unsurveyed_marine]
                    end
                end
                verbose && println(
                    "  Assigned homogeneous HSI = 0.50 across " *
                    "$(count(unsurveyed_marine .& .!out_of_depth)) in-depth unsurveyed " *
                    "marine units outside sppoly footprint."
                )
            end
        end

        final_mask = keep_mesh_mask .& (in_domain .| in_regions)

        n_outside_domain = count(keep_mesh_mask .& .!(in_domain .| in_regions))
        if domain_geom === nothing && !any(in_regions)
            verbose && println(
                "  No domain footprint configured; the mesh extent is the domain."
            )
        else
            verbose && println(
                "  Domain + region footprint: $(count(final_mask)) / $n_spatial " *
                "in-depth marine units inside ($(count(in_domain .& keep_mesh_mask)) " *
                "surveyed, $(count(.!in_domain .& in_regions .& keep_mesh_mask)) " *
                "homogeneous regional), $n_outside_domain outside."
            )
        end

        # Add the units that observation endpoints actually occupy, wherever they
        # are. The domain is the union of the declared footprint and the data, not
        # the footprint alone: an animal released in the Southern Gulf or the Bay of
        # Fundy outside the mapped strata still has to be represented, and on this
        # dataset 2,827 of 5,064 events have at least one endpoint outside the
        # footprint. Dropping them would discard most of the mark-recapture data;
        # relocating them would invent a release position, which the endpoint snap
        # below was doing for more than half of all endpoints.
        #
        # The addition is bounded by the depth and land tests, so it cannot bring
        # back shallow or terrestrial cells, and the count is reported so the grown
        # domain is visible rather than implied.
        n_rel_out = 0
        n_rec_out = 0
        added = falses(n_spatial)
        for r in eachrow(obs_df)
            if 1 <= r.release <= n_spatial && keep_mesh_mask[r.release]
                if !in_domain[r.release]
                    added[r.release] = true
                    n_rel_out += 1
                end
                final_mask[r.release] = true
            end
            if 1 <= r.recapture <= n_spatial && keep_mesh_mask[r.recapture]
                if !in_domain[r.recapture]
                    added[r.recapture] = true
                    n_rec_out += 1
                end
                final_mask[r.recapture] = true
            end
        end

        (n_rel_out + n_rec_out) > 0 && verbose && println(
            "  Extended domain by $(count(added)) units holding observation " *
            "endpoints outside the footprint " *
            "($n_rel_out release / $n_rec_out recapture endpoints)."
        )

        keep_mesh_mask .= final_mask

        n_pruned = count(!, keep_mesh_mask)
        if n_pruned > 0 && count(keep_mesh_mask) > 0
            verbose && println(
                "  Pruning $n_pruned out-of-depth/out-of-bounds units from domain " *
                "($(count(keep_mesh_mask)) units retained)..."
            )

            # Reconnect severed marine components strictly through in-depth, non-land
            # marine units; never re-add land, inland lakes, or shallow coastal mudflats.
            nav_bridge = .!land_mask .& .!out_of_depth
            keep_mesh_mask, n_link, n_unlinked = reconnect_severed_components(
                keep_mesh_mask, W, cents_tmp; navigable_mask = nav_bridge
            )
            if n_unlinked > 0
                # Retain only the primary connected marine body (plus any secondary
                # marine basin with >= 25 units) rather than leaving disconnected singletons.
                nbrs_pr = [Int[] for _ in 1:n_spatial]
                rw_p, cl_p = findnz(sparse(W))
                for (r_p, c_p) in zip(rw_p, cl_p)
                    (!land_mask[r_p] && !land_mask[c_p]) && push!(nbrs_pr[r_p], Int(c_p))
                end
                comps_pr = _connected_components(keep_mesh_mask, nbrs_pr)
                if !isempty(comps_pr)
                    max_c = maximum(length, comps_pr)
                    for comp_k in comps_pr
                        if length(comp_k) < min(25, max_c)
                            for u_drop in comp_k
                                keep_mesh_mask[u_drop] = false
                            end
                        end
                    end
                end
            end
            if verbose && (n_link + n_unlinked) > 0
                println(
                    "  Reconnected the pruned domain: $n_link marine unit(s) added as " *
                    "bridges, $n_unlinked isolated unit(s) outside marine connectivity."
                )
            end

            # Repoint endpoints the prune is about to delete at the nearest retained
            # unit, before the reindex below. The "observation points landing on
            # them" pass above only protects an endpoint that already satisfies
            # `keep_mesh_mask`, so an endpoint on a pruned unit used to reach
            # `old_to_new` unchanged -- and `old_to_new` maps it to 0. See
            # `_snap_pruned_endpoints!`.
            n_snap_rel, n_snap_rec = _snap_pruned_endpoints!(
                obs_df, keep_mesh_mask, cents_tmp
            )
            if verbose && (n_snap_rel + n_snap_rec) > 0
                println(
                    "  Snapped $n_snap_rel release / $n_snap_rec recapture " *
                    "endpoints off pruned units to the nearest retained unit."
                )
            end

            old_to_new = zeros(Int, n_spatial)
            new_idx = 0
            for i in 1:n_spatial
                if keep_mesh_mask[i]
                    new_idx += 1
                    old_to_new[i] = new_idx
                end
            end

            mesh = prune_mesh(mesh, keep_mesh_mask)
            W = W[keep_mesh_mask, keep_mesh_mask]
            hsi_vec = hsi_vec[keep_mesh_mask]
            if !isempty(monthly_hsi) && size(monthly_hsi, 1) == n_spatial
                monthly_hsi = monthly_hsi[keep_mesh_mask, :]
            end
            if land_mask !== nothing && length(land_mask) == n_spatial
                land_mask = land_mask[keep_mesh_mask]
            end
            if resharded_depths !== nothing && length(resharded_depths) == n_spatial
                resharded_depths = resharded_depths[keep_mesh_mask]
            end
            if bottom_temp !== nothing && length(bottom_temp) == n_spatial
                bottom_temp = bottom_temp[keep_mesh_mask]
            end
            if region_map !== nothing && length(region_map) == n_spatial
                region_map = region_map[keep_mesh_mask]
            end
            if resharded_hydro !== nothing
                resharded_hydro = reshard_spatial_field(
                    spdiagm(0 => ones(Float64, count(keep_mesh_mask))),
                    resharded_hydro
                )
            end

            # Re-index observations to pruned mesh units
            obs_df = copy(obs_df)
            obs_df.release = [old_to_new[r] for r in obs_df.release]
            obs_df.recapture = [old_to_new[r] for r in obs_df.recapture]

            # Checked, not assumed. `old_to_new` is 0 for every pruned unit, so an
            # endpoint that escaped the snap above would surface much later as
            # `KeyError: key (0, k) not found` from the k-step transition cache in
            # Phase 2, with nothing pointing back at this line.
            new_n = count(keep_mesh_mask)
            bad = [
                (i, c, obs_df[i, c]) for c in (:release, :recapture)
                for i in 1:nrow(obs_df) if !(1 <= obs_df[i, c] <= new_n)
            ]
            isempty(bad) || error(
                "Mesh reindex left $(length(bad)) observation endpoints outside " *
                "1:$new_n (first: row $(bad[1][1]) $(bad[1][2]) = $(bad[1][3])). " *
                "Endpoint snapping before the prune should have made this " *
                "impossible; it is a bug in this reindex, not in the input data."
            )

            if !isnothing(survey_df) && hasproperty(survey_df, :s_idx)
                survey_df = copy(survey_df)
                keep_surv = [keep_mesh_mask[s] for s in survey_df.s_idx]
                survey_df = survey_df[keep_surv, :]
                survey_df.s_idx = [old_to_new[s] for s in survey_df.s_idx]
            end

            n_spatial = mesh.n_units
        end
    end

    # Standalone tessellation_polygons.html is no longer written during full runs;
    # the tessellation & depth view is included as a tab in movement_dashboard.html.

    return (
        data               = data,
        mesh               = mesh,
        W                  = W,
        hsi_vec            = hsi_vec,
        monthly_hsi        = monthly_hsi,
        month_lookup       = month_lookup,
        years              = years_vec,
        obs_df             = obs_df,
        survey_df          = survey_df,
        group_map          = group_map,
        region_labels      = conn_region_labels,
        region_map         = region_map,
        land_mask          = land_mask,
        n_spatial          = n_spatial,
        resharded_hydro    = resharded_hydro,
        resharded_depths   = resharded_depths,
        bottom_temp        = bottom_temp,
        parsed_depth_range = parsed_depth_range,
    )
end

"""
    tessellation_preview(loaded, params) -> String

Render the tessellated spatial domain, open it in a browser, and return the file
path.

Backs `--tessellation-only`. Deliberately independent of `params.render_html`:
the point of the flag is to look at the mesh, so gating it on whether HTML output
is enabled would let `--render-html=false --tessellation-only` quietly show
nothing at all.

The figure is written as the same self-contained HTML every other dashboard uses and the
platform launcher opens it. When no launcher is available (headless session, unknown
platform) the file is still written and the path is returned, so the map is never lost.

# Arguments
- `loaded`: output of [`load_movement_data`](@ref).
- `params`: configuration struct. Uses `output_dir`, `species_name`, `dark_mode`
  and `verbose`.

# Returns
Path to the rendered map.
"""
function _render_tessellation_html(
    mesh::NamedTuple,
    output_file::AbstractString;
    title::AbstractString = "Tessellated Spatial Domain",
    depth = nothing,
    hsi = nothing,
    dark_mode::Bool = false
)::String
    mkpath(dirname(output_file))
    polys = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat :
            (hasproperty(mesh, :polygons) ? mesh.polygons : nothing)
    vals = hsi !== nothing ? Float64.(hsi) :
           (depth !== nothing ? Float64.(depth) : ones(Float64, mesh.n_units))
    label = hsi !== nothing ? "HSI" : (depth !== nothing ? "Depth (m)" : "Unit")

    # Delegate to TethysPlots if already loaded in the session
    tp_id = Base.PkgId(Base.UUID("0e2c882d-e859-4e84-bf15-b0abe2d12c71"), "TethysPlots")
    if haskey(Base.loaded_modules, tp_id) && polys !== nothing
        tp = Base.loaded_modules[tp_id]
        m = tp.leaflet_choropleth(
            polys, vals;
            title = String(title),
            colorbar_label = label,
            dark_mode = dark_mode
        )
        return tp.save_html(m, output_file)
    end

    # Lightweight self-contained Leaflet HTML fallback
    cents = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat :
            (hasproperty(mesh, :centroids) ? mesh.centroids : Tuple{Float64, Float64}[])
    isempty(cents) && error("Mesh has no centroids_lonlat or polygons_lonlat to render.")
    c_lon = mean(Float64(c[1]) for c in cents)
    c_lat = mean(Float64(c[2]) for c in cents)
    v_fin = filter(isfinite, vals)
    v_min = isempty(v_fin) ? 0.0 : minimum(v_fin)
    v_max = isempty(v_fin) ? 1.0 : maximum(v_fin)
    span = max(v_max - v_min, 1e-9)

    poly_items = String[]
    if polys !== nothing
        for i in eachindex(polys)
            ring = polys[i]
            length(ring) < 3 && continue
            t = i <= length(vals) && isfinite(vals[i]) ? (vals[i] - v_min) / span : 0.0
            t = max(0.0, min(1.0, t))
            r = round(Int, 68 + 185 * t)
            g = round(Int, 1 + 230 * t)
            b = round(Int, 84 + 85 * (1.0 - t))
            hex = "#" * string(r, base = 16, pad = 2) *
                  string(g, base = 16, pad = 2) * string(b, base = 16, pad = 2)
            coords = join(["[$(Float64(pt[2])),$(Float64(pt[1]))]" for pt in ring], ",")
            push!(poly_items, "{\"c\":[$coords],\"k\":\"$hex\",\"v\":$(round(vals[min(i, length(vals))]; digits=3))}")
        end
    end

    ocean_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
    gray_url = dark_mode ?
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}" :
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"

    html = """
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>$title</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.css"/>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.js"></script>
    <style>html,body,#map{margin:0;padding:0;width:100%;height:100%;background:#cbd5e1;}</style>
    </head><body><div id="map"></div><script>
    var map = L.map('map').setView([$c_lat, $c_lon], 6);
    var ocean = L.tileLayer('$ocean_url', {maxZoom: 13, attribution: 'Tiles &copy; Esri'}).addTo(map);
    var gray = L.tileLayer('$gray_url', {maxZoom: 13, attribution: 'Tiles &copy; Esri'});
    L.control.layers({"Esri Ocean (Keyless)": ocean, "Esri Gray Canvas": gray}).addTo(map);
    var polys = [$(join(poly_items, ","))];
    polys.forEach(function(p) {
      L.polygon(p.c, {color: p.k, fillColor: p.k, fillOpacity: 0.6, weight: 0.8})
       .bindTooltip("$label: " + p.v).addTo(map);
    });
    </script></body></html>
    """
    write(output_file, html)
    return String(output_file)
end

function _open_in_browser(path::AbstractString)::Bool
    try
        if Sys.iswindows()
            run(`cmd /c start "" $(abspath(path))`; wait = false)
            return true
        elseif Sys.isapple()
            run(`open $(abspath(path))`; wait = false)
            return true
        elseif Sys.islinux()
            run(`xdg-open $(abspath(path))`; wait = false)
            return true
        end
    catch
    end
    return false
end

function tessellation_preview(loaded, params)::String
    out_dir = _resolve_output_dir(params.output_dir)
    mkpath(out_dir)
    tess_path = joinpath(out_dir, "tessellation_polygons.html")
    _render_tessellation_html(
        loaded.mesh, tess_path;
        title = "$(params.species_name) Tessellated Spatial Domain",
        depth = loaded.resharded_depths,
        hsi = loaded.hsi_vec,
        dark_mode = params.dark_mode
    )

    opened = _open_in_browser(tess_path)
    if params.verbose
        println("  Tessellation: $(loaded.mesh.n_units) units")
        println("  Grid polygon map: $tess_path")
        println(opened ? "  Opened in the default browser." :
                        "  No browser launcher available; open the file above.")
    end
    return abspath(tess_path)
end

"""
    _tessellation_only_result(loaded, tess_path, params) -> NamedTuple

Build the truncated result returned by a `--tessellation-only` run.

Same keys as a full [`run_movement_analysis`](@ref) return, with `nothing` for
everything that requires a fitted kernel. A caller reading `P_kernel` or `paths`
from this value is asking for something the run was told not to compute; a typed
key set makes that explicit instead of surfacing as a missing field.

# Arguments
- `loaded`: output of [`load_movement_data`](@ref).
- `tess_path`: path to the rendered map.
- `params`: configuration struct. Unused; kept so the call site reads uniformly
  with the other exits.
"""
function _tessellation_only_result(loaded, tess_path, params)::NamedTuple
    return (
        data               = loaded.data,
        mesh               = loaded.mesh,
        models             = nothing,
        chains             = nothing,
        P_kernel           = nothing,
        paths              = nothing,
        corridors          = nothing,
        stochastic_paths   = nothing,
        domain_bottlenecks = nothing,
        circuit            = nothing,
        validation_analyses = nothing,
        agent_trajectories = nothing,
        agent_space_use    = nothing,
        movement_stats     = nothing,
        phenology          = nothing,
        trait_models       = nothing,
        parameters         = nothing,
        depth_range        = loaded.parsed_depth_range,
        panel_skips       = copy(PANEL_SKIPS),
        tessellation_only  = true,
        tessellation_path  = tess_path,
    )
end

# =============================================================================
# Phase 2: Model Fitting
# =============================================================================

"""
    movement_sampler(prior_scales, params)

The MCMC sampler for the movement models, as a random walk in linked space with
diagonal covariance `prior_scales .^ 2 .* params.mh_proposal_scale`.

# Why not plain `MH()`
`MH()` in Turing 0.49 is documented as drawing proposals **from the model prior**.
That is unusable here: with 5,294 observations the posterior is far sharper than
the prior, so once the chain has any likelihood at all, a fresh prior draw almost
never beats it. Measured on the snow crab data, `MH()` accepts **0.0%** of
proposals and returns a single distinct value across 200 draws -- the reported
"posterior" is the initialisation, not a fit. Two different models frozen at the
same seeded starting point report *identical* parameters, which is how the bug
surfaced.

# Why a random walk and not NUTS
`NUTS` works (acc = 1.0, 200/200 distinct draws) and adapts its metric
automatically, which is the better long-term answer. On this problem it is ~15x
slower per draw than the random walk, because each of ~2^10 leapfrog steps
rebuilds the transition kernel. That cost is a tuning question, not a correctness
one; the proposal scale is exposed as `mh_proposal_scale` so it can be tuned
without editing code, and convergence diagnostics are tracked in `todo.md`.

`prior_scales` are the priors' standard deviations in linked space, so the proposal
is a fraction of the prior width rather than an arbitrary number.
"""
function movement_sampler(prior_scales::Vector{Float64}, params)
    s = params.mh_proposal_scale
    s > 0 || throw(ArgumentError("mh_proposal_scale must be positive, got $s"))
    return MH(Matrix(Diagonal((s .* prior_scales) .^ 2)))
end

# The three population parameters and their prior scales, in declaration order:
# mu_velocity ~ N(0.3, 0.2) truncated, mu_diffusion ~ N(0.1, 0.2) truncated,
# mu_gamma ~ N(1.0, 1.0). Shared by the discrete and continuous-time models, which
# differ only in their likelihood.
const POPULATION_PRIOR_SCALES = [0.2, 0.2, 1.0]

"""
    report_chain_health(label, chain, params) -> Bool

Print the two numbers that distinguish a chain that moved from one that did not:
the acceptance rate, and how many distinct values the retained draws actually took.

This exists because the alternative is invisible. `MH()` accepted 0.0% of proposals
and returned a single distinct value across 200 draws, and the run reported a
posterior mean as if it meant something. A distinct-value count near 1 is the
signature of a frozen chain and cannot be mistaken for a tight posterior once
stated.

Returns `false` when the chain looks frozen, so callers can escalate.
"""
function report_chain_health(label::AbstractString, chain, params)::Bool
    verbose = params.verbose

    n = 0
    acc = Float64[]
    for k in keys(chain)
        nm = string(k)
        vals = try
            Array(chain[k])
        catch
            continue
        end
        occursin("accept", nm) || (n = max(n, length(vals)); continue)
        m = skipmissing(vec(vals))
        isempty(m) || push!(acc, mean(m))
    end
    n == 0 && return true
    a = isempty(acc) ? NaN : mean(acc)

    # How many distinct values one sampled parameter actually took. This is the
    # signal that catches a frozen chain even when the reported acceptance is
    # misleading, so it is worth computing from whichever parameter is available
    # rather than assuming a name.
    n_distinct = 0
    for k in keys(chain)
        occursin("Parameter", string(k)) || continue
        vals = try
            vec(Array(chain[k]))
        catch
            continue
        end
        all(isfinite, vals) || continue
        n_distinct = length(unique(round.(vals; digits = 9)))
        break
    end

    # Effective sample size (autocorrelation-based) for the first sampled parameter.
    ess_val = NaN
    for k in keys(chain)
        occursin("Parameter", string(k)) || continue
        vals = try
            vec(Array(chain[k]))
        catch
            continue
        end
        all(isfinite, vals) || continue
        ess_val = _effective_sample_size(vals)
        break
    end

    healthy = (isnan(a) || a > 0.01) && n_distinct > 1
    if healthy
        verbose && println(
            "  $label: acceptance=$(isnan(a) ? "n/a" : string(round(a; digits = 3))), " *
            "distinct values=$n_distinct/$n, ESS=$(isnan(ess_val) ? "n/a" : string(round(ess_val; digits = 1)))")
    else
        @warn "$label chain looks frozen" acceptance = a n_distinct = n_distinct n_draws = n
    end
    if !isnan(ess_val) && ess_val < max(10.0, 0.1 * n)
        @warn "$label chain has low effective sample size" ess = ess_val n_draws = n
    end
    return healthy
end

function _effective_sample_size(x::AbstractVector{<:Real})::Float64
    n = length(x)
    n < 2 && return Float64(n)
    xc = x .- mean(x)
    # Autocorrelation via FFT for speed.
    acf = zeros(Float64, n)
    # lag 0
    c0 = dot(xc, xc)
    c0 == 0 && return Float64(n)
    acf[1] = 1.0
    for lag in 1:(n-1)
        acf[lag+1] = dot(xc[1:n-lag], xc[lag+1:n]) / c0
    end
    # Truncate at first negative autocorrelation (Geyer initial monotone).
    tau = 1.0
    for lag in 1:(n-1)
        if acf[lag+1] <= 0
            break
        end
        tau += 2 * acf[lag+1]
    end
    return n / tau
end

"""
    fit_movement_models(loaded, params) -> NamedTuple

Phase 2 of the pipeline. Fits Bayesian movement models via Turing MCMC.

Approaches to fit (set via `params.model_modes`):

- `"telemetry"`: Pure categorical mark-recapture transition likelihood, fitted at
    the population level. recapture ~ Categorical(P^k[release, :]) for free
    parameters `mu_velocity`, `mu_diffusion`, `mu_gamma`; `alpha` and `rho` are
    derived from the first two by `movement_alpha_rho`.
- `"telemetry_and_survey"`: **the survey density likelihood is not implemented** —
    `counts` and `depths` are accepted and ignored, so this reduces to the
    telemetry model and emits a warning.
- `"agent"`: Agent-based movement simulation; no MCMC.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `params`: Configuration struct. Relevant keys: `model_modes`,
  `n_samples`, `seed`, `verbose`.

# Returns
`NamedTuple` with `models::Dict` and `chains::Dict`.

# On burn-in
Every `sample` call passes `num_warmup = params.n_warmup`, which AbstractMCMC
turns into `discard_initial` -- the first `n_warmup` states of the walk are
thrown away and only the following `n_samples` are kept.

This was missing, and it mattered more than it looked. Without it the returned
"posterior" was the *opening segment* of the random walk, dominated by the
initial draw and therefore by the prior. Two different models, seeded the same
way, then reported near-identical parameters because both were mostly reporting
their priors. `n_warmup` was a documented, configurable field that nothing read.

# On reporting whether the chain moved
A frozen chain produces a posterior mean indistinguishable from a converged one, and
that is how the `MH()` problem stayed hidden for so long: two different models
reported identical parameters, and nothing in the output said why. `report_chain_health`
prints acceptance and the number of distinct posterior values, and escalates to a
warning when the chain barely moved, so the failure is visible in the run log
rather than only inferable from the results.
"""
function fit_movement_models(loaded, params)::NamedTuple
    verbose   = params.verbose
    params.n_warmup >= 0 || throw(ArgumentError(
        "n_warmup must be non-negative, got $(params.n_warmup)"))
    params.n_samples >= 1 || throw(ArgumentError(
        "n_samples must be at least 1, got $(params.n_samples)"))
    # `model_modes` is the request; `effective_model_modes` is what survives
    # contact with the data. Survey-dependent approaches are dropped when no
    # readable survey file is configured, so asking for one is not an error --
    # it simply is not run.
    mode_set = Set(effective_model_modes(params))
    for m in skipped_modes(params)
        verbose && println(
            "  Note: :$(m) not run -- no readable surveydata_file configured."
        )
    end
      fit_tel       = :telemetry in mode_set
      fit_joint     = :telemetry_and_survey in mode_set
      rng       = MersenneTwister(params.seed)
      models    = Dict{Symbol, Any}()
      chains    = Dict{Symbol, Any}()

      # -- Pure Telemetry Model ------------------------------------------------
    if fit_tel
        verbose && println("\n[Phase 2] Fitting Pure Telemetry model...")
        verbose && println(
            "  population-level: mu_velocity, mu_diffusion, mu_gamma"
        )
        verbose && println("  recapture ~ Categorical(P^k[release, :])")
        obs_df     = loaded.obs_df
        releases   = Int.(obs_df.release)
        recaptures = Int.(obs_df.recapture)
        ks         = Int.(round.(obs_df.k))
        tagids = obs_df.tagid
        # `size(loaded.W, 1)` is the support of every `Categorical` the likelihood
        # builds, so it is the exact precondition. A stale index here means the mesh
        # prune reindexed an observation onto a deleted unit; named here rather than
        # as a bare KeyError from the k-step cache inside the model.
        validate_mark_recapture_indices(releases, recaptures, size(loaded.W, 1))
        m_tel = pure_telemetry_turing_model(
            releases, recaptures, ks,
            loaded.W, loaded.hsi_vec, loaded.land_mask
        )
        models[:telemetry] = m_tel
        spl = movement_sampler(POPULATION_PRIOR_SCALES, params)
        verbose && println("  Sampling $(params.n_samples) draws " *
                          " (discarding $(params.n_warmup) warmup)...")
        chn = sample(rng, m_tel, spl, params.n_samples; num_warmup = params.n_warmup, progress = false)
        chains[:telemetry] = chn
        report_chain_health("telemetry", chn, params)
        verbose && println("  Pure telemetry model complete.")

        # Separate fishery-independent Acoustic (OTN) vs. Commercial Trap (BIO) inference
        if hasproperty(obs_df, :datasource)
            ds_low = String[lowercase(strip(string(v))) for v in obs_df.datasource]
            idx_otn = findall(==("otn"), ds_low)
            idx_bio = findall(==("bio"), ds_low)
            if length(idx_otn) >= 10 && length(idx_bio) >= 10
                fit_datasource_stratified_kernels!(
                    models, chains, loaded, params, idx_otn, idx_bio, rng
                )
            end
        end
    end

# -- Joint Survey + Telemetry Model --------------------------------------
    if fit_joint && !isnothing(loaded.survey_df)
        verbose && println(
            "\n[Phase 3] Fitting Joint Survey + Telemetry model..."
        )
        verbose && println("  density ~ NegBin(exp(eta_s), r)")

        obs_df     = loaded.obs_df
        releases   = Int.(obs_df.release)
        recaptures = Int.(obs_df.recapture)
        ks         = Int.(round.(obs_df.k))
        tagids = obs_df.tagid
        survey_df  = loaded.survey_df
        counts     = Int.(round.(survey_df.density))
        depths     = hasproperty(survey_df, :depth) ?
                     Float64.(survey_df.depth) : zeros(Float64, length(counts))

        survey_units = hasproperty(survey_df, :s_idx) ?
                       Int.(survey_df.s_idx) :
                       collect(1:length(counts))

        validate_mark_recapture_indices(releases, recaptures, size(loaded.W, 1))
        m_joint = joint_survey_telemetry_turing_model(
            counts, depths, survey_units,
            releases, recaptures, ks,
                        loaded.W, loaded.hsi_vec, loaded.land_mask
        )
        models[:telemetry_and_survey] = m_joint
        # The joint models declare three survey coefficients in addition to the
        # population parameters (see todo.md 1.6). They are prior-only, but the
        # proposal still has to span them or those coordinates cannot move.
        spl = movement_sampler(
            vcat([5.0, 2.0, 1.0], POPULATION_PRIOR_SCALES), params)
        verbose && println("  Sampling $(params.n_samples) draws " *
                          " (discarding $(params.n_warmup) warmup)...")
        chn_j = sample(rng, m_joint, spl, params.n_samples; num_warmup = params.n_warmup, progress = false)
                chains[:telemetry_and_survey] = chn_j
        report_chain_health("telemetry_and_survey", chn_j, params)
        verbose && println("  Joint model complete.")
    end

    return (models = models, chains = chains)
end

function _empirical_trap_effort_weights(
    obs_df::DataFrame,
    idx_bio::Vector{Int},
    W::SparseMatrixCSC,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Float64}
    S = size(W, 1)
    counts = zeros(Float64, S)
    for i in idx_bio
        u1 = Int(obs_df.release[i])
        u2 = Int(obs_df.recapture[i])
        1 <= u1 <= S && (counts[u1] += 1.0)
        1 <= u2 <= S && (counts[u2] += 1.0)
    end
    rows = rowvals(W)
    smoothed = Vector{Float64}(undef, S)
    for u in 1:S
        (land_mask !== nothing && land_mask[u]) && (smoothed[u] = 0.05; continue)
        nb_sum = counts[u]
        nb_cnt = 1
        for ptr in nzrange(W, u)
            v = rows[ptr]
            v == u && continue
            (land_mask !== nothing && land_mask[v]) && continue
            nb_sum += counts[v]
            nb_cnt += 1
        end
        smoothed[u] = nb_sum / nb_cnt
    end
    cmax = maximum(smoothed; init = 0.0)
    cmax <= 0.0 && return ones(Float64, S)
    return Float64[0.05 + 0.95 * (smoothed[u] / cmax) for u in 1:S]
end

"""
    fit_datasource_stratified_kernels(loaded, params) -> NamedTuple

Fits separate Bayesian movement transition kernels for fishery-independent OTN
acoustic receiver detections (`datasource == "otn"`) and commercial pot-trap
recaptures (`datasource == "bio"`, incorporating a smoothed pot-haul effort offset).
"""
function fit_datasource_stratified_kernels!(
    models::Dict{Symbol, Any},
    chains::Dict{Symbol, Any},
    loaded::NamedTuple,
    params,
    idx_otn::Vector{Int},
    idx_bio::Vector{Int},
    rng::AbstractRNG = MersenneTwister(params.seed + 101)
)
    verbose = params.verbose
    obs_df  = loaded.obs_df
    n_s     = size(loaded.W, 1)
    spl     = movement_sampler(POPULATION_PRIOR_SCALES, params)
    n_sub   = max(10, min(params.n_samples, 300))
    n_warm  = max(0, min(params.n_warmup, 80))

    verbose && println(
        "\n[Phase 2-DS] Fitting stratified Acoustic (OTN, n=$(length(idx_otn))) vs. " *
        "Commercial Trap (BIO, n=$(length(idx_bio))) models..."
    )

    rel_otn = Int.(obs_df.release[idx_otn])
    rec_otn = Int.(obs_df.recapture[idx_otn])
    ks_otn  = max.(1, Int.(round.(obs_df.k[idx_otn])))
    validate_mark_recapture_indices(rel_otn, rec_otn, n_s)
    m_otn = pure_telemetry_turing_model(
        rel_otn, rec_otn, ks_otn, loaded.W, loaded.hsi_vec, loaded.land_mask
    )
    models[:telemetry_otn] = m_otn
    chn_otn = sample(rng, m_otn, spl, n_sub; num_warmup = n_warm, progress = false)
    chains[:telemetry_otn] = chn_otn
    report_chain_health("telemetry_otn (acoustic)", chn_otn, params)

    effort_bio = _empirical_trap_effort_weights(
        obs_df, idx_bio, sparse(loaded.W), loaded.land_mask
    )
    rel_bio = Int.(obs_df.release[idx_bio])
    rec_bio = Int.(obs_df.recapture[idx_bio])
    ks_bio  = max.(1, Int.(round.(obs_df.k[idx_bio])))
    validate_mark_recapture_indices(rel_bio, rec_bio, n_s)
    m_bio = pure_telemetry_turing_model(
        rel_bio, rec_bio, ks_bio, loaded.W, loaded.hsi_vec, loaded.land_mask, effort_bio
    )
    models[:telemetry_bio] = m_bio
    chn_bio = sample(rng, m_bio, spl, n_sub; num_warmup = n_warm, progress = false)
    chains[:telemetry_bio] = chn_bio
    report_chain_health("telemetry_bio (commercial trap)", chn_bio, params)
    return (otn = chn_otn, bio = chn_bio)
end

function fit_datasource_stratified_kernels(loaded::NamedTuple, params)
    models = Dict{Symbol, Any}()
    chains = Dict{Symbol, Any}()
    obs_df = loaded.obs_df
    hasproperty(obs_df, :datasource) || throw(ArgumentError(
        "obs_df has no :datasource column for acoustic vs. trap stratification."
    ))
    ds_low = String[lowercase(strip(string(v))) for v in obs_df.datasource]
    idx_otn = findall(==("otn"), ds_low)
    idx_bio = findall(==("bio"), ds_low)
    (isempty(idx_otn) || isempty(idx_bio)) && throw(DomainError(
        (length(idx_otn), length(idx_bio)),
        "Both 'otn' and 'bio' records are required for stratified fitting."
    ))
    fit_datasource_stratified_kernels!(
        models, chains, loaded, params, idx_otn, idx_bio, MersenneTwister(params.seed + 101)
    )
    return (models = models, chains = chains)
end

"""
    _error_note(e, limit::Int = 200) -> String

A one-line, length-capped rendering of an exception, for use in the `catch` notes
that let an optional dashboard panel fail without ending the run.

Interpolating an exception directly is unsafe here. A `FieldError` or
`MethodError` embeds the full type of every argument, and those types carry the
entire loaded dataset -- so a single missing field on a 7517-unit mesh produced
about 200 KB of type signature on one line, burying the actual message and every
other line of output. Truncating keeps the useful part, which is the exception
type and the reason it failed.
"""
_error_note(e, limit::Int = 200) =
    (s = first(sprint(showerror, e), limit); length(sprint(showerror, e)) > limit ? s * " …" : s)

"""
    PANEL_SKIPS::Vector{String}

Every optional dashboard panel that failed during the last run, as `"name: reason"`.

The export phase wraps roughly twenty-five panels in `try`/`catch` so that one
bad panel cannot end a run that has already spent minutes sampling. That policy
is right, but it was also *silent*: each catch only printed a note, and gated
that print on `verbose`, so under the default quiet settings a panel could fail
every time and nothing in the output said so.

That is not hypothetical. Three real defects hid in exactly these blocks -- an
unbound `hydro`, a `_sample_column` call with a group argument the pooled model
no longer has, and a read of the `group` column after it had been removed. Each
one made a whole panel vanish permanently while the run reported success. The
notes only surfaced because a verbose end-to-end run was printed by hand.

Recording the failures here makes a missing panel an observable fact. It is
module-level rather than threaded through the panel helpers so that panels in
different functions are all captured without changing any of their signatures;
it is diagnostic state only, reset at the start of each run, and never read by
the model.
"""
const PANEL_SKIPS = String[]

function _record_panel_skip(name::AbstractString, e)::Nothing
    push!(PANEL_SKIPS, "$name: $(_error_note(e))")
    return nothing
end

"""
    report_panel_skips(verbose::Bool) -> Int

Print a consolidated summary of `PANEL_SKIPS` and return how many were recorded.
Called once at the end of a run so that silent panel loss is visible by default.
"""
function report_panel_skips(verbose::Bool = true)::Int
    n = length(PANEL_SKIPS)
    n == 0 && return 0
    println("\n  $(n) optional panel(s) did not render:")
    for s in PANEL_SKIPS
        println("    - $s")
    end
    return n
end

# =============================================================================
# Phase 3: Kernel Construction
# =============================================================================

"""
    posterior_kernel_draws(chain) -> (velocity, diffusion, gamma)

Reduce a fitted posterior chain to the three per-draw series the transition
kernel is parameterised by.

The models are population-level, so the kernel parameters are the fitted `mu_*`
draws with the models' own bounds applied:

    velocity_i  = clamp(mu_velocity[i],  0, 0.95)
    diffusion_i = max(mu_diffusion[i],   0)
    gamma_i     = mu_gamma[i]

`sigma_*` and `z_*` are read **if present** and folded in, so a chain from an
older hierarchical model still reduces correctly, but their absence is an
ordinary outcome rather than a warning. There is no `alpha`/`rho` derivation
here: it lives in `movement_alpha_rho`, shared with the models, because a model
and the code reporting its posterior must not each re-derive the same transform.

Two details of the chain layout are handled here rather than at the call site,
because both fail silently if missed:

  * Keys are wrapped (`Parameter(mu_velocity)`, `Extra(:accepted)`), so a plain
    string comparison matches nothing and the bookkeeping columns as well.
  * A vector-valued parameter reads back as one vector per draw nested in a
    trailing singleton axis, which needs unwrapping before it is numeric.
"""
function posterior_kernel_draws(chain)
    n_draws = if chain isa AbstractDict
        max(1, maximum(v -> v isa AbstractArray ? size(v, 1) : 1, values(chain); init = 1))
    else
        max(1, size(Array(chain), 1))
    end

    # Chain keys are not plain names: the sampler wraps each one as
    # `Parameter(name)` or `Extra(name)` (or sanitized `Parameter_name_` in Zarr).
    function parname(key)
        s = string(key)
        open = findfirst('(', s)
        close = findlast(')', s)
        if open !== nothing && close !== nothing && close > open
            inner = s[(open + 1):(close - 1)]
            return startswith(inner, ':') ? inner[2:end] : inner
        end
        if startswith(s, "Parameter_")
            return chop(s[11:end]; tail = endswith(s, '_') ? 1 : 0)
        end
        return s
    end
    lookup = Dict{String,Any}(parname(k) => k for k in keys(chain))

    "Posterior draws of a scalar parameter, one per draw, or `nothing` if absent."
    function scalar_draws(name::String)
        key = get(lookup, name, nothing)
        key === nothing && return nothing
        d = vec(Array(chain[key]))
        length(d) == n_draws || return nothing
        return Float64.(d)
    end

    "Per-individual effects as an n_draws x n_individuals matrix, or nothing."
    function effect_draws(name::String)
        key = get(lookup, name, nothing)
        key === nothing && return nothing
        a = Array(chain[key])
        # A vector-valued parameter comes back as one vector per draw inside a
        # trailing singleton axis; unwrap the individual axis first.
        if eltype(a) <: AbstractVector
            cols = [Float64.(collect(v)) for v in a]
            isempty(cols) && return nothing
            length(unique(length.(cols))) == 1 || return nothing
            # One row per draw, one column per individual. `hcat` puts the draws
            # in columns, so the result is transposed to the layout used below.
            return permutedims(reduce(hcat, cols))
        end
        ndims(a) == 2 || return nothing
        return Float64.(a)
    end

    # A missing `mu_*` is a real problem: the kernel cannot be built without it.
    mu_v = scalar_draws("mu_velocity")
    mu_d = scalar_draws("mu_diffusion")
    mu_g = scalar_draws("mu_gamma")
    missing_mus = String[]
    mu_v === nothing && push!(missing_mus, "mu_velocity")
    mu_d === nothing && push!(missing_mus, "mu_diffusion")
    mu_g === nothing && push!(missing_mus, "mu_gamma")
    isempty(missing_mus) || error(
        "posterior_kernel_draws: chain is missing " * join(missing_mus, ", ") *
        ", so no transition kernel can be built from it. Chain keys were: " *
        join(sort!(collect(keys(lookup))), ", ")
    )

    # `sigma_*` and `z_*` are optional: present in a chain from an older
    # hierarchical model, absent from the current population-level one.
    sg_v = scalar_draws("sigma_velocity")
    sg_d = scalar_draws("sigma_diffusion")
    sg_g = scalar_draws("sigma_gamma")
    z_v  = effect_draws("z_velocity")
    z_d  = effect_draws("z_diffusion")
    z_g  = effect_draws("z_gamma")

    "Fold optional dispersion and individual effects in, then apply the model's clamp."
    function combine(mu, sigma, z, lo::Float64, hi::Float64)
        (sigma === nothing || z === nothing || size(z, 1) != length(mu)) && return mu
        col_m = reshape(mu, :, 1)
        col_s = reshape(sigma, :, 1)
        return vec(mean(clamp.(col_m .+ col_s .* z, lo, hi); dims = 2))
    end

    velocity  = combine(mu_v, sg_v, z_v, 0.0, 0.95)
    diffusion = combine(mu_d, sg_d, z_d, 0.0, Inf)
    # Gamma is unbounded in the model, so it is averaged without a clamp.
    gamma = (sg_g === nothing || z_g === nothing || size(z_g, 1) != length(mu_g)) ?
        copy(mu_g) :
        vec(mean(reshape(mu_g, :, 1) .+ reshape(sg_g, :, 1) .* z_g; dims = 2))

    return velocity, diffusion, gamma
end

"""
    extract_transition_kernels(loaded, fitted, params) -> NamedTuple

Phase 3 of the pipeline. Extracts the posterior transition kernel and parameter
summaries from the fitted MCMC chains.

Single-group model: all individuals share one parameter set. To fit group-specific
movement, call `run_movement_analysis` once per group with that group's data and
combine the results externally; nothing here borrows parameters across groups.

# Returns
\$NamedTuple\$ with:
- \$P_kernel\$: the posterior-mean stochastic transition kernel
- \$grp_name_lookup\$: Dict mapping group index -> label (empty for single group)
- \$alpha_hat\$, \$rho_hat\$, \$gamma_hat\$: posterior mean advection, residence, HSI
- \$alpha_samples\$, \$rho_samples\$, \$gamma_samples\$: posterior draws, one per draw
- \$active_chain\$: the chain used
"""
function extract_transition_kernels(loaded, fitted, params)::NamedTuple
    verbose = params.verbose
    chains  = fitted.chains

    # Choose the first available chain (prefer joint > telemetry)
    active_chain = if haskey(chains, :telemetry)
        chains[:telemetry]
    elseif haskey(chains, :telemetry_and_survey)
        chains[:telemetry_and_survey]
    elseif !isempty(chains)
        first(values(chains))
    else
        nothing
    end

    # Every downstream phase indexes the kernel and reads alpha/rho/gamma as
    # scalars, so there is no meaningful "empty" result to hand back. Naming the
    # modes that were requested makes the cause obvious: a survey-dependent mode
    # with no survey file is skipped, and could be the only one asked for.
    active_chain === nothing && error(
        "no fitted model produced a posterior chain; requested modes were " *
        join(string.(keys(chains)), ", ", " and ") *
        ". Check that a mode with a readable input was requested -- the " *
        "survey-dependent approaches need `surveydata_file`."
    )

    verbose && println("\n[Phase 3] Extracting transition kernels from posterior...")

    verbose && println("  Extracting posterior means and draws from the active chain...")

    active_chain = _get_active_chain(chains)
    if active_chain === nothing
        # Every downstream phase indexes the kernel and reads alpha/rho/gamma as
        # scalars, so there is no meaningful "empty" result to hand back. Naming
        # the modes that were asked for makes the cause obvious: a survey-dependent
        # mode with no survey file is skipped, and could be the only one requested.
        error("no fitted model produced a posterior chain; requested modes were " *
              join(string.(keys(chains)), ", ", " and ") *
              " (asked for: " * join(string.(get(loaded, :model_modes, Symbol[])), ", ") * ")")
    end

    v_draws, d_draws, g_draws = posterior_kernel_draws(active_chain)

    # `alpha`/`rho` come from the same helper the models use. This line used to
    # re-derive them and got `rho` wrong -- `1/(v+d)` instead of `1/(1+v+d)` --
    # so the reported and propagated residence parameter was a different quantity
    # from the one the likelihood was fitted with. See todo.md 1.1.
    alpha_samples, rho_samples = movement_alpha_rho(v_draws, d_draws)
    gamma_samples = g_draws

    g_mean    = mean(g_draws)
    alpha_hat = mean(alpha_samples)
    rho_hat   = mean(rho_samples)

    # No label-seeded or config-seeded overrides.
    # Parameters are reported exactly as fitted.
    if verbose
        println("\n[Phase 3] Posterior parameters:")
        println("  alpha=$(round(alpha_hat; digits=4)), " *
                "rho=$(round(rho_hat; digits=4)), " *
                "gamma=$(round(g_mean; digits=4))")
    end

# Habitat-coupled residency. `gamma` biases *which* neighbour is chosen;
      # residence decides whether the animal moves at all. With a zero coupling
      # the fitted scalar rho applies everywhere, which is what the posterior
      # actually estimated. A positive coupling makes residency unit-specific
      # from local habitat advantage, which is an extra assumption the telemetry
      # did not fit, so it is opt-in and reported below when active.
      beta = Float64(params.rest_advantage)
      residency = if beta == 0.0
          rho_hat
      else
          adv = local_hsi_advantage(
              loaded.hsi_vec, loaded.W; form = params.rest_advantage_form)
          residency_from_advantage(rho_hat, adv, beta, params.rest_advantage_form)
      end

      # Build the single transition kernel from posterior means
      P_kernel = construct_stochastic_transition_kernel(
          loaded.W, loaded.hsi_vec;
          gamma     = g_mean,
          residence = residency,
          advection = alpha_hat,
          land_mask = loaded.land_mask,
          coupled_residency = beta != 0.0,
      )

      if verbose && beta != 0.0
          println("  Habitat-coupled residency active: beta=$(beta), " *
                  "form=:$(params.rest_advantage_form), " *
                  "rho range $(round(minimum(residency); digits = 3))..$(round(maximum(residency); digits = 3))")
      end

    datasource_summary = _summarize_datasource_posteriors(
        loaded, chains, params, alpha_samples, rho_samples, gamma_samples
    )

    return (
        P_kernel           = P_kernel,
        grp_name_lookup    = Dict{Int, String}(),
        alpha_hat          = alpha_hat,
        rho_hat            = rho_hat,
        gamma_hat          = g_mean,
        G                  = 1,
        alpha_samples      = alpha_samples,
        rho_samples        = rho_samples,
        gamma_samples      = gamma_samples,
        active_chain       = active_chain,
        datasource_summary = datasource_summary,
    )
end

function _summarize_datasource_posteriors(
    loaded::NamedTuple,
    chains::Dict,
    params,
    pooled_a::AbstractVector{<:Real},
    pooled_r::AbstractVector{<:Real},
    pooled_g::AbstractVector{<:Real}
)
    obs_df = loaded.obs_df
    step_yr = _step_years(params, loaded)
    mean_step_km = _mean_neighbor_step_km(loaded)
    cents_ll = hasproperty(loaded.mesh, :centroids_lonlat) ?
        loaded.mesh.centroids_lonlat : nothing

    function _emp_metrics(idxs::Vector{Int})
        isempty(idxs) && return (0, 0, 0.0, 0.0, 0.0)
        n_o = length(idxs)
        n_t = length(unique(obs_df.tagid[idxs]))
        disps = Float64[]
        n_stat = 0
        for i in idxs
            u1 = Int(obs_df.release[i])
            u2 = Int(obs_df.recapture[i])
            u1 == u2 && (n_stat += 1)
            if cents_ll !== nothing && 1 <= u1 <= length(cents_ll) && 1 <= u2 <= length(cents_ll)
                c1, c2 = cents_ll[u1], cents_ll[u2]
                push!(disps, haversine_distance(
                    Float64(c1[1]), Float64(c1[2]), Float64(c2[1]), Float64(c2[2])
                ) / 1000.0)
            else
                push!(disps, 0.0)
            end
        end
        return (n_o, n_t, 100.0 * n_stat / n_o, median(disps), mean(disps))
    end

    function _row_from_draws(lbl::String, idxs::Vector{Int}, a_v, r_v, g_v)
        n_o, n_t, pct_st, med_d, mn_d = _emp_metrics(idxs)
        am, rm, gm = mean(a_v), mean(r_v), mean(g_v)
        alo, ahi = length(a_v) >= 2 ? (quantile(a_v, 0.025), quantile(a_v, 0.975)) : (am, am)
        rlo, rhi = length(r_v) >= 2 ? (quantile(r_v, 0.025), quantile(r_v, 0.975)) : (rm, rm)
        glo, ghi = length(g_v) >= 2 ? (quantile(g_v, 0.025), quantile(g_v, 0.975)) : (gm, gm)
        v_eff = (1.0 - rm) * mean_step_km / step_yr
        d_eff = (1.0 - rm) * (1.0 - am) * (mean_step_km^2) / (4.0 * step_yr)
        return (
            stratum          = lbl,
            n_obs            = n_o,
            n_tags           = n_t,
            pct_stationary   = pct_st,
            median_disp_km   = med_d,
            mean_disp_km     = mn_d,
            alpha_mean       = am,
            alpha_ci_lo      = alo,
            alpha_ci_hi      = ahi,
            rho_mean         = rm,
            rho_ci_lo        = rlo,
            rho_ci_hi        = rhi,
            gamma_mean       = gm,
            gamma_ci_lo      = glo,
            gamma_ci_hi      = ghi,
            eff_speed_km_yr  = v_eff,
            eff_diff_km2_yr  = d_eff,
        )
    end

    rows = NamedTuple[]
    push!(rows, _row_from_draws("Pooled All", collect(1:nrow(obs_df)), pooled_a, pooled_r, pooled_g))

    if hasproperty(obs_df, :datasource)
        ds_low = String[lowercase(strip(string(v))) for v in obs_df.datasource]
        for (ckey, dkey, lbl) in [
            (:telemetry_otn, "otn", "Acoustic Array (OTN)"),
            (:telemetry_bio, "bio", "Commercial Trap (BIO, effort-offset)")
        ]
            idxs = findall(==(dkey), ds_low)
            isempty(idxs) && continue
            if haskey(chains, ckey) && chains[ckey] !== nothing
                vd, dd, gd = posterior_kernel_draws(chains[ckey])
                as, rs = movement_alpha_rho(vd, dd)
                push!(rows, _row_from_draws(lbl, idxs, as, rs, gd))
            end
        end
    end

    return DataFrame(rows)
end

# =============================================================================
# Phase 4: Path Reconstruction & Bottleneck Detection
# =============================================================================

"""
    reconstruct_paths_and_diagnostics(loaded, kernels, params) -> NamedTuple

Phase 4 of the pipeline. Reconstructs individual movement trajectories and
Markov bridge corridor heatmaps, computes optional stochastic least-cost
path ensembles, and assembles domain-wide Bayesian posterior averages for
bottleneck detection:

    F_domain(u, v) = sum_i w_i E_i(u, v)     [directed migration flux]
    C_domain(u)    = sum_i w_i rho_i(u)       [nodal transit density]
    B(u) = C_domain(u) / max(1, deg_marine(u))  [bottleneck index]

# Arguments
- `loaded`: Output of `load_movement_data`.
- `kernels`: Output of `extract_transition_kernels`.
- `params`: Relevant keys: `max_paths`, `path_methods`, `smooth_paths`,
  `compute_stochastic`, `compute_bottlenecks`, `n_stochastic_draws`,
  `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with: `paths`, `corridors`, `stochastic_paths`,
`domain_bottlenecks`, `cents_planar`, `cents_lonlat`, `cents_mesh`.
"""
function _dedup_consecutive_nodes(seq::AbstractVector{<:Integer})::Vector{Int}
    isempty(seq) && return Int[]
    out = Int[Int(seq[1])]
    for k in 2:length(seq)
        u = Int(seq[k])
        u != out[end] && push!(out, u)
    end
    return out
end

function _select_stratified_tags(
    obs_df::DataFrame,
    max_paths::Int,
    cents_lonlat
)::Vector{String}
    all_tags = String.(unique(obs_df.tagid))
    n_total = length(all_tags)
    n_sample = min(max(1, max_paths), n_total)
    n_sample >= n_total && return all_tags

    # Rank tags by (release_unit, max_displacement_km) so an evenly spaced stride
    # samples uniformly across all release regions and movement regimes.
    tag_meta = Dict{String, Tuple{Int, Float64}}()
    for row in eachrow(obs_df)
        tid = string(row.tagid)
        u_rel = Int(row.release)
        u_rec = Int(row.recapture)
        d_km = if u_rel != u_rec && cents_lonlat !== nothing &&
                  1 <= u_rel <= length(cents_lonlat) && 1 <= u_rec <= length(cents_lonlat)
            c1 = cents_lonlat[u_rel]
            c2 = cents_lonlat[u_rec]
            haversine_distance(
                Float64(c1[1]), Float64(c1[2]), Float64(c2[1]), Float64(c2[2])
            ) / 1000.0
        else
            0.0
        end
        prev = get(tag_meta, tid, (u_rel, -1.0))
        if d_km > prev[2]
            tag_meta[tid] = (prev[2] < 0.0 ? u_rel : prev[1], d_km)
        end
    end

    sorted_tags = sort(all_tags; by = t -> get(tag_meta, t, (0, 0.0)))
    idxs = unique(round.(Int, range(1, n_total; length = n_sample)))
    return sorted_tags[idxs]
end

function _interval_mean_hsi(
    loaded,
    t_rel::Real,
    k_steps::Int,
    step_yr::Float64
)::Vector{Float64}
    isnan(t_rel) && return loaded.hsi_vec
    mhsi = loaded.monthly_hsi
    mlup = loaded.month_lookup
    (isempty(mhsi) || isempty(mlup)) && return loaded.hsi_vec

    dur_yr = max(0.0, k_steps * step_yr)
    n_mo = clamp(ceil(Int, dur_yr * 12.0) + 1, 1, 24)
    acc = zeros(Float64, size(mhsi, 1))
    n_hit = 0
    for m_idx in 0:(n_mo - 1)
        t_cur = Float64(t_rel) + (m_idx / 12.0)
        yr = floor(Int, t_cur)
        mo = clamp(Int(fld((t_cur - yr) * 12.0, 1.0)) + 1, 1, 12)
        col = get(mlup, (yr, mo), nothing)
        if col !== nothing && 1 <= col <= size(mhsi, 2)
            acc .+= @view(mhsi[:, col])
            n_hit += 1
        end
    end
    n_hit == 0 && return _resolve_hsi_for_time(loaded, t_rel)
    acc ./= n_hit
    return acc
end

function _sparse_bridge_transit_density(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Int;
    land_mask = nothing
)::Vector{Float64}
    S = size(P, 1)
    out = zeros(Float64, S)
    (!(1 <= release <= S) || !(1 <= recapture <= S) || release == recapture) && return out

    K = clamp(k, 2, 365)
    P_sp = P isa SparseMatrixCSC ? P : sparse(Float64.(P))
    Pt_sp = sparse(P_sp')

    f_vecs = Vector{Vector{Float64}}(undef, K + 1)
    f0 = zeros(Float64, S)
    f0[release] = 1.0
    f_vecs[1] = f0
    for tau in 1:K
        nxt = Pt_sp * f_vecs[tau]
        s = sum(nxt)
        s > 0.0 && (nxt ./= s)
        f_vecs[tau + 1] = nxt
    end

    b_vecs = Vector{Vector{Float64}}(undef, K + 1)
    bK = zeros(Float64, S)
    bK[recapture] = 1.0
    b_vecs[K + 1] = bK
    for tau in K:-1:1
        prv = P_sp * b_vecs[tau + 1]
        s = sum(prv)
        s > 0.0 && (prv ./= s)
        b_vecs[tau] = prv
    end

    # Average interior bridge marginals tau = 1..(K-1) so pinned release/recapture
    # delta spikes at tau = 0 and tau = K do not mask transit corridors.
    n_int = 0
    buf = zeros(Float64, S)
    for tau in 1:(K - 1)
        col = tau + 1
        csum = 0.0
        @inbounds for u in 1:S
            val = (land_mask !== nothing && Bool(land_mask[u])) ?
                0.0 : f_vecs[col][u] * b_vecs[col][u]
            buf[u] = val
            csum += val
        end
        if csum > 1e-30
            out .+= buf ./ csum
            n_int += 1
        end
    end
    n_int > 0 && (out ./= n_int)
    return out
end

function reconstruct_paths_and_diagnostics(
    loaded, kernels, params
)::NamedTuple
    verbose     = params.verbose
    obs_df      = loaded.obs_df
    W           = loaded.W
    hsi_vec     = loaded.hsi_vec
    land_mask   = loaded.land_mask
    n_spatial   = loaded.n_spatial
    P_kernel    = kernels.P_kernel
    G           = 1
    step_yr     = _step_years(params, loaded)

    cents_planar, cents_lonlat, cents_mesh =
        _resolve_centroids(loaded.mesh, n_spatial)

    all_tags    = unique(obs_df.tagid)
    sample_tags = _select_stratified_tags(obs_df, params.max_paths, cents_lonlat)
    n_sample    = length(sample_tags)

    verbose && println(
        "\n[Phase 4] Reconstructing $(join(string.(params.path_methods), ", ")) trajectories " *
        "for $n_sample / $(length(all_tags)) spatially stratified individuals..."
    )

    land_polys = isempty(params.land_polygon_files) ? nothing :
        reduce(vcat, read_polygon_file.(params.land_polygon_files))

    P_seg_cache = Dict{Tuple{Int, Int, Int}, SparseMatrixCSC{Float64, Int}}()
    P_base_sp = P_kernel isa SparseMatrixCSC ? P_kernel : sparse(Float64.(P_kernel))

    reconstructed_paths     = Dict{String, Vector{Int}}()
    reconstructed_corridors = Dict{String, Vector{Float64}}()
    stochastic_paths        = Dict{String, Any}()
    forward_ibm_paths       = Dict{String, Vector{Int}}()

    _group_param(x, g::Int) = x isa AbstractVector ?
        ((1 <= g <= length(x)) ? x[g] : throw(BoundsError(x, g))) : x

    chosen_method = isempty(params.path_methods) ? :bridge : first(params.path_methods)

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue

        grp = hasproperty(sub_obs, :group) ? Int(first(sub_obs.group)) : 1
        P_k = _group_param(P_kernel, grp)
        P_k_sp = P_k isa SparseMatrixCSC ? P_k : P_base_sp

        full_path = if params.hmm_smoothing && nrow(sub_obs) > 1
            times = Int[1]
            cum_t = 1
            locs = Any[sub_obs.release[1]]
            for r in eachrow(sub_obs)
                cum_t += max(1, r.k)
                push!(times, cum_t)
                push!(locs, r.recapture)
            end
            res_hmm = viterbi_hmm_path_smoothing(
                times, locs, P_k_sp;
                mesh = loaded.mesh, land_mask = land_mask
            )
            _dedup_consecutive_nodes(res_hmm.path)
        else
            fpath = Int[sub_obs.release[1]]
            for (seg_idx, row) in enumerate(eachrow(sub_obs))
                k_seg = max(1, Int(row.k))
                t_rel = hasproperty(row, :rel_time) ? Float64(row.rel_time) : NaN
                hsi_row = if params.dynamic_kernels && !isnan(t_rel)
                    _interval_mean_hsi(loaded, t_rel, k_seg, step_yr)
                elseif !isnan(t_rel)
                    _resolve_hsi_for_time(loaded, t_rel)
                else
                    hsi_vec
                end

                P_seg = if hsi_row === hsi_vec
                    P_k_sp
                else
                    yr_k = isnan(t_rel) ? 0 : round(Int, t_rel * 12.0)
                    mo_span = clamp(ceil(Int, k_seg * step_yr * 12.0), 1, 24)
                    get!(P_seg_cache, (yr_k, mo_span, grp)) do
                        a_hat = _group_param(kernels.alpha_hat, grp)
                        r_hat = _group_param(kernels.rho_hat, grp)
                        g_hat = _group_param(kernels.gamma_hat, grp)
                        sparse(construct_stochastic_transition_kernel(
                            W, hsi_row;
                            gamma     = g_hat,
                            residence = r_hat,
                            advection = a_hat,
                            land_mask = land_mask
                        ))
                    end
                end

                rng_seg = MersenneTwister(hash((string(tid), seg_idx, params.seed)))
                seg = predict_path(
                    P_seg, row.release, row.recapture, k_seg;
                    centroids = cents_mesh,
                    method    = chosen_method,
                    land_mask = land_mask,
                    rng       = rng_seg
                )
                if isempty(seg)
                    verbose && println(
                        "  $tid segment $seg_idx/$(nrow(sub_obs)): no marine route " *
                        "from $(row.release) to $(row.recapture); dropped."
                    )
                elseif length(seg) >= 2
                    append!(fpath, seg[2:end])
                end
            end
            _dedup_consecutive_nodes(fpath)
        end

        if params.smooth_paths && cents_mesh !== nothing && land_polys !== nothing
            full_path = smooth_marine_path(
                full_path, cents_mesh;
                land_polygons = land_polys,
                centroids_lonlat = cents_lonlat
            )
        end
        reconstructed_paths[string(tid)] = full_path

        # Unconditioned forward IBM comparison trajectory
        fwd_ibm = simulate_forward_ibm(
            P_k_sp, first(sub_obs).release, min(sum(sub_obs.k), 365);
            land_mask = land_mask
        )
        forward_ibm_paths[string(tid)] = _dedup_consecutive_nodes(fwd_ibm)

        # Markov-bridge interior transit corridor across non-stationary segments
        corr_accum = zeros(Float64, n_spatial)
        n_active_seg = 0
        for row in eachrow(sub_obs)
            row.release == row.recapture && continue
            k_seg = max(2, Int(row.k))
            t_rel = hasproperty(row, :rel_time) ? Float64(row.rel_time) : NaN
            hsi_seg = !isnan(t_rel) ? _interval_mean_hsi(loaded, t_rel, k_seg, step_yr) : hsi_vec
            P_c = if hsi_seg === hsi_vec
                P_k_sp
            else
                yr_k = round(Int, t_rel * 12.0)
                mo_span = clamp(ceil(Int, k_seg * step_yr * 12.0), 1, 24)
                get(P_seg_cache, (yr_k, mo_span, grp), P_k_sp)
            end
            corr_accum .+= _sparse_bridge_transit_density(
                P_c, row.release, row.recapture, k_seg;
                land_mask = land_mask
            )
            n_active_seg += 1
        end
        if n_active_seg > 0
            corr_accum ./= n_active_seg
        end
        reconstructed_corridors[string(tid)] = corr_accum
    end

    if verbose
        n_multi = count(p -> length(p) >= 2, values(reconstructed_paths))
        max_hops = isempty(reconstructed_paths) ? 0 : maximum(length, values(reconstructed_paths)) - 1
        println("  Reconstructed $n_sample trajectories ($n_multi non-stationary, max $max_hops marine hex hops).")
    end

    # -- 4b. Domain-Wide Posterior Averaging & Bottleneck Detection ----------
    domain_bottlenecks = nothing

    if wants_diagnostic(params, :bottlenecks) && cents_planar !== nothing
        verbose && println("\n[Phase 4b] Domain bottleneck detection...")
        n_obs = nrow(obs_df)

        # Group all non-stationary marine observations into unique (release, recapture)
        # origin-destination pairs so every shelf sector is represented without a single
        # pot haul swamping shelf-wide corridors.
        pair_counts = Dict{Tuple{Int, Int}, Int}()
        for i in 1:n_obs
            u_rel = Int(obs_df.release[i])
            u_rec = Int(obs_df.recapture[i])
            u_rel == u_rec && continue
            if land_mask !== nothing && (land_mask[u_rel] || land_mask[u_rec])
                continue
            end
            pair_counts[(u_rel, u_rec)] = get(pair_counts, (u_rel, u_rec), 0) + 1
        end

        unique_pairs = sort!(collect(keys(pair_counts)))
        max_od_pairs = max(350, params.max_paths)
        eval_pairs = if length(unique_pairs) <= max_od_pairs
            unique_pairs
        else
            p_idxs = unique(round.(Int, range(1, length(unique_pairs); length = max_od_pairs)))
            unique_pairs[p_idxs]
        end

        verbose && println(
            "  Aggregating $(length(eval_pairs)) / $(length(unique_pairs)) unique " *
            "origin-destination movement pairs across $n_obs observations..."
        )

        C_domain = zeros(Float64, n_spatial)
        C_events = zeros(Float64, max(1, length(eval_pairs)), n_spatial)
        F_domain = spzeros(Float64, n_spatial, n_spatial)

        # Consolidate bottleneck and directed-flux routing onto the MCMC-fitted
        # posterior transition kernel (-log P[u, v] edge costs) rather than an
        # uncalibrated heuristic friction surface.
        a_s = _sample_column(hasproperty(kernels, :alpha_samples) ? kernels.alpha_samples : nothing)
        r_s = _sample_column(hasproperty(kernels, :rho_samples) ? kernels.rho_samples : nothing)
        g_s = _sample_column(hasproperty(kernels, :gamma_samples) ? kernels.gamma_samples : nothing)
        n_post = min(length(a_s), length(r_s), length(g_s))
        n_b_draws = n_post >= 2 ? min(n_post, 12) : 1
        P_b_draws = Vector{SparseMatrixCSC{Float64, Int}}(undef, n_b_draws)
        if n_b_draws > 1
            d_idxs = unique(round.(Int, range(1, n_post; length = n_b_draws)))
            P_b_draws = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(d_idxs))
            for (k_i, di) in enumerate(d_idxs)
                P_b_draws[k_i] = sparse(construct_stochastic_transition_kernel(
                    W, hsi_vec;
                    gamma     = g_s[di],
                    residence = r_s[di],
                    advection = a_s[di],
                    land_mask = land_mask
                ))
            end
        else
            P_b_draws[1] = P_base_sp
        end
        n_b_draws = length(P_b_draws)

        for (e_idx, (u_rel, u_rec)) in enumerate(eval_pairs)
            w_pair = 1.0 + log(Float64(pair_counts[(u_rel, u_rec)]))
            cp = zeros(Float64, n_spatial)
            for (d_i, P_d) in enumerate(P_b_draws)
                rng_b = MersenneTwister(hash((u_rel, u_rec, d_i, params.seed)))
                p_m = predict_bridge_path(
                    P_d, u_rel, u_rec, nothing;
                    land_mask = land_mask,
                    rng       = rng_b
                )
                p_m = _dedup_consecutive_nodes(p_m)
                for u in p_m
                    1 <= u <= n_spatial && (cp[u] += 1.0 / n_b_draws)
                end
                for step_i in 1:(length(p_m) - 1)
                    u_a, u_b = p_m[step_i], p_m[step_i + 1]
                    if 1 <= u_a <= n_spatial && 1 <= u_b <= n_spatial && u_a != u_b
                        F_domain[u_a, u_b] += w_pair / n_b_draws
                    end
                end
            end
            # Down-weight pinned release/recapture endpoints (0.25) relative to interior
            # transit conduit nodes (1.0) so pot locations do not masquerade as choke-points.
            cp[u_rel] *= 0.25
            cp[u_rec] *= 0.25
            C_domain .+= w_pair .* cp
            C_events[e_idx, :] .= w_pair .* cp
        end

        # Structural bottleneck index: B(u) = C(u) / deg_marine(u)
        deg_marine = [
            count(
                j -> W[i, j] > 0 &&
                     (land_mask === nothing || !land_mask[j]),
                1:n_spatial
            )
            for i in 1:n_spatial
        ]
        bottleneck_score = zeros(Float64, n_spatial)
        bottleneck_se    = zeros(Float64, n_spatial)
        n_ev = length(eval_pairs)

        for u in 1:n_spatial
            is_marine = land_mask === nothing ? true : !land_mask[u]
            if is_marine && deg_marine[u] > 0
                bottleneck_score[u] = C_domain[u] / deg_marine[u]
                if n_ev > 1
                    scores_u = [C_events[e, u] / deg_marine[u] for e in 1:n_ev]
                    bottleneck_se[u] = std(scores_u) / sqrt(n_ev)
                end
            end
        end

        marine_idx = land_mask === nothing ?
                     collect(1:n_spatial) : findall(!, land_mask)
        pos_scores = filter(>(0.0), bottleneck_score[marine_idx])
        b_thresh   = isempty(pos_scores) ? 0.0 : quantile(pos_scores, 0.90)
        bottleneck_mask = (bottleneck_score .>= b_thresh) .&
                          (bottleneck_score .> 0.0) .&
                          (land_mask === nothing ?
                           trues(n_spatial) : .!land_mask)

        if verbose
            println(
                "  Peak transit density: " *
                "$(round(maximum(C_domain); digits=2)) tag equiv."
            )
            println("  Active directed edges: $(nnz(F_domain))")
            println(
                "  Top-10% bottleneck threshold: " *
                "$(round(b_thresh; digits=4))"
            )
            println(
                "  Critical bottleneck units: " *
                "$(sum(bottleneck_mask)) / $n_spatial"
            )
        end

        directed_flux = cents_lonlat !== nothing ?
            compute_directed_flux_field(F_domain, cents_lonlat; top_k = 180) : nothing

        domain_bottlenecks = (
            transit_density  = C_domain,
            edge_flux        = F_domain,
            bottleneck_score = bottleneck_score,
            bottleneck_se    = bottleneck_se,
            bottleneck_mask  = bottleneck_mask,
            threshold        = b_thresh,
            directed_flux    = directed_flux,
        )
    end

    return (
        paths              = reconstructed_paths,
        corridors          = reconstructed_corridors,
        stochastic_paths   = stochastic_paths,
        forward_ibm_paths  = forward_ibm_paths,
        domain_bottlenecks = domain_bottlenecks,
        cents_planar       = cents_planar,
        cents_lonlat       = cents_lonlat,
        cents_mesh         = cents_mesh,
    )
end

# =============================================================================
# Phase 5: Advanced Diagnostics (Circuit Theory)
# =============================================================================

"""
    compute_advanced_diagnostics(loaded, path_results, params) -> NamedTuple

Phase 5 of the pipeline. Optionally computes:

**Circuit Theory** (`:circuit` in `params.diagnostics`):
- Multi-pair electrical current density I = C nabla V across all
  mark-recapture source-sink pairs.
- Identifies ecological pinch-points (top 10% current density).
- Posterior circuit inference via Monte Carlo HSI sampling:
    HSI_draw ~ N(hsi_mean, hsi_se^2)
  over `n_stochastic_draws * 2` replicates.

# Arguments
- `loaded`: Output of `load_movement_data`.
- `path_results`: Output of `reconstruct_paths_and_diagnostics`.
- `params`: Relevant keys: `diagnostics`,
  `n_stochastic_draws`, `hsi_se`, `seed`, `verbose`.

# Returns
`NamedTuple` with `circuit` (or `nothing` if not computed).
"""
function compute_advanced_diagnostics(
    loaded, path_results, params;
    kernels = nothing
)::NamedTuple
    verbose   = params.verbose
    W         = loaded.W
    hsi_vec   = loaded.hsi_vec
    land_mask = loaded.land_mask
    n_spatial = loaded.n_spatial
    obs_df    = loaded.obs_df
    sources_v = Int.(obs_df.release)
    sinks_v   = Int.(obs_df.recapture)
    hsi_se_v  = fill(params.hsi_se, n_spatial)

    # Group non-stationary marine observations into unique (release, recapture)
    # pairs with log-damped weights 1 + log(count) so a single heavily fished
    # port does not swamp shelf-wide circuit corridors.
    pair_counts = Dict{Tuple{Int, Int}, Int}()
    for (u_rel, u_rec) in zip(sources_v, sinks_v)
        u_rel == u_rec && continue
        if 1 <= u_rel <= n_spatial && 1 <= u_rec <= n_spatial &&
           (land_mask === nothing || (!land_mask[u_rel] && !land_mask[u_rec]))
            pair_counts[(u_rel, u_rec)] = get(pair_counts, (u_rel, u_rec), 0) + 1
        end
    end
    if !isempty(pair_counts)
        u_pairs = sort!(collect(keys(pair_counts)))
        pair_srcs = Int[p[1] for p in u_pairs]
        pair_snks = Int[p[2] for p in u_pairs]
        pair_wts  = Float64[1.0 + log(Float64(pair_counts[p])) for p in u_pairs]
    else
        pair_srcs = sources_v
        pair_snks = sinks_v
        pair_wts  = nothing
    end

    # Calibrate conductance exponent to the MCMC-fitted habitat gradient gamma
    # so edge conductance reflects inferred habitat selectivity rather than a
    # nearly isotropic sheet.
    cond_pow = if kernels !== nothing && hasproperty(kernels, :gamma_hat) &&
                  kernels.gamma_hat !== nothing
        clamp(Float64(mean(kernels.gamma_hat)), 1.0, 6.0)
    else
        3.5
    end

    circuit_res::Union{NamedTuple, Nothing} = nothing

    # -- Circuit Theory ------------------------------------------------------
    if wants_diagnostic(params, :circuit)
        verbose && println(
            "\n[Phase 5a] Computing Circuit Theory Current Density..."
        )
        try
            # Circuit theory models steady-state DC current flux (infinite-horizon
            # random walk potential), which cannot evaluate high-frequency non-stationary
            # time steps directly. When dynamic seasonal HSI is configured or available,
            # Conductance is evaluated on the annual time-averaged HSI field:
            #     \bar{H}_i = (1 / 12) \sum_{m=1}^{12} H_i(m)
            hsi_circuit = if hasproperty(loaded, :monthly_hsi) && !isempty(loaded.monthly_hsi) &&
                             size(loaded.monthly_hsi, 1) == n_spatial
                vec(mean(loaded.monthly_hsi; dims = 2))
            elseif params.dynamic_kernels && params.use_geodata_habitat &&
                   hasproperty(loaded.mesh, :centroids_lonlat) &&
                   loaded.mesh.centroids_lonlat !== nothing
                verbose && println("  Averaging dynamic seasonal HSI across 12-month annual cycle for circuit theory...")
                cents_ll = loaded.mesh.centroids_lonlat
                min_lon = minimum(c[1] for c in cents_ll)
                max_lon = maximum(c[1] for c in cents_ll)
                min_lat = minimum(c[2] for c in cents_ll)
                max_lat = maximum(c[2] for c in cents_ll)
                mesh_polys = (params.spatial_averaging != "centroid" &&
                              hasproperty(loaded.mesh, :polygons_lonlat)) ?
                             loaded.mesh.polygons_lonlat : nothing
                hsi_annual_mat = build_dynamic_geodata_hsi_series(
                    cents_ll;
                    polygons           = mesh_polys,
                    lon_range          = (min_lon, max_lon),
                    lat_range          = (min_lat, max_lat),
                    months             = 1:12,
                    temperature_source = params.temperature_source,
                    verbose            = false
                )
                vec(mean(hsi_annual_mat; dims = 2))
            else
                hsi_vec
            end

            cur_dens, _, _, _ = current_density_map(
                W, pair_srcs, pair_snks;
                weights           = pair_wts,
                hsi               = hsi_circuit,
                conductance_power = cond_pow,
                land_mask         = land_mask,
                include_endpoints = false
            )
            p_mask, p_score, p_thresh = identify_ecological_pinchpoints(
                cur_dens; top_quantile = 0.95
            )
            verbose && println(
                "  Current density (interior, gamma=$(round(cond_pow; digits=2))) range: " *
                "[$(round(minimum(cur_dens); digits=5)), " *
                "$(round(maximum(cur_dens); digits=5))]"
            )
            verbose && println(
                "  Critical pinch-points (top 5%): " *
                "$(sum(p_mask)) / $n_spatial"
            )

            # Propagate spatially coherent interannual HSI realizations from
            # loaded.monthly_hsi rather than i.i.d. cell-level white noise.
            annual_hsi_samples = if hasproperty(loaded, :monthly_hsi) &&
                                    size(loaded.monthly_hsi, 1) == n_spatial &&
                                    size(loaded.monthly_hsi, 2) >= 24
                n_yrs = div(size(loaded.monthly_hsi, 2), 12)
                n_sel = min(n_yrs, 10)
                yr_idxs = unique(round.(Int, range(1, n_yrs; length = n_sel)))
                mat_yr = Matrix{Float64}(undef, n_spatial, length(yr_idxs))
                for (ci, yi) in enumerate(yr_idxs)
                    c1 = (yi - 1) * 12 + 1
                    c2 = min(yi * 12, size(loaded.monthly_hsi, 2))
                    mat_yr[:, ci] = vec(mean(@view(loaded.monthly_hsi[:, c1:c2]); dims = 2))
                end
                mat_yr
            else
                nothing
            end

            stoch_circuit = if annual_hsi_samples !== nothing
                posterior_circuit_inference(
                    W, pair_srcs, pair_snks;
                    weights           = pair_wts,
                    hsi_samples       = annual_hsi_samples,
                    conductance_power = cond_pow,
                    land_mask         = land_mask,
                    include_endpoints = false,
                    top_quantile      = 0.95,
                    seed              = params.seed
                )
            else
                posterior_circuit_inference(
                    W, pair_srcs, pair_snks;
                    weights           = pair_wts,
                    hsi_mean          = hsi_circuit,
                    hsi_se            = hsi_se_v,
                    n_draws           = min(params.n_stochastic_draws, 12),
                    conductance_power = cond_pow,
                    land_mask         = land_mask,
                    include_endpoints = false,
                    top_quantile      = 0.95,
                    seed              = params.seed
                )
            end
            robust_mask, _, n_robust = identify_stochastic_pinchpoints(
                stoch_circuit; prob_threshold = 0.80
            )
            verbose && println(
                "  Robust pinch-points (P >= 80%): $n_robust / $n_spatial"
            )

            circuit_res = (
                current_density   = cur_dens,
                pinch_mask        = p_mask,
                pinch_score       = p_score,
                threshold         = p_thresh,
                stochastic        = stoch_circuit,
                robust_mask       = robust_mask,
                n_robust          = n_robust,
                interior_only     = true,
                conductance_power = cond_pow,
            )
        catch e
            _record_panel_skip("Circuit computation", e);             verbose && println("  (Circuit computation note: $(_error_note(e)))")
        end
    end

    # The trailing comma is required: a one-element parenthesised named tuple
    # without it collapses to the value itself, so `(circuit = nothing)` would
    # return a bare `nothing` and `diagnostics.circuit` would be unreachable.
    return (circuit = circuit_res,)
end

# =============================================================================
# Phase 6: Dashboard Export
# =============================================================================

# General movement ecology metrics (compute_movement_statistics,
# analyze_seasonal_movement_phenology, model_trait_movement_associations,
# export_movement_summary_csv) and standalone SVG/HTML dashboard exporters
# (export_movement_posterior_dashboard, export_movement_flow_dashboard,
# export_movement_summary_dashboard) have been consolidated into src/movement.jl
# and are exported by the core TethysPath module.


"""
    export_dashboards(loaded, kernels, path_results, diagnostics, params)

Phase 6 of the pipeline. Exports interactive Leaflet HTML dashboards to
`params.output_dir`. Individual dashboards are silently skipped on
rendering errors so the pipeline is never aborted. Exports:

- Movement path trajectories and empirical displacement vectors.
- Interactive two-click migration corridor explorer.
- Hydrodynamic stratification dashboard (when domain was resharded).
- Circuit current density and posterior pinch-point maps.
- Domain-wide bottleneck conduit heatmap.
"""
function _ramp_hex_color(t::Float64, cmap::Symbol = :viridis)::String
    u = clamp(isfinite(t) ? t : 0.0, 0.0, 1.0)
    r, g, b = if cmap === :plasma
        (round(Int, 13 + 227 * u),
         round(Int, 8 + 240 * (u^1.6)),
         round(Int, 135 + 100 * sin(π * u) - 102 * u))
    elseif cmap === :inferno || cmap === :magma
        (round(Int, 10 + 242 * (u^0.85)),
         round(Int, 7 + 215 * (u^1.8)),
         round(Int, 38 + 140 * sin(π * u)))
    else
        # viridis-like perceptual ramp
        (round(Int, 68 + 185 * (u^1.3)),
         round(Int, 1 + 230 * (u^0.85)),
         round(Int, 84 + 90 * sin(π * u) - 45 * u))
    end
    rc = clamp(r, 0, 255)
    gc = clamp(g, 0, 255)
    bc = clamp(b, 0, 255)
    return "#" * string(rc, base = 16, pad = 2) *
           string(gc, base = 16, pad = 2) * string(bc, base = 16, pad = 2)
end

function _js_escape(s::AbstractString)::String
    return replace(replace(replace(String(s), '\\' => "\\\\"), '"' => "\\\""), '\n' => " ")
end

function _encode_field_layer_json(
    id::AbstractString,
    tab_label::AbstractString,
    title::AbstractString,
    field_label::AbstractString,
    marine_indices::Vector{Int},
    field_vals::AbstractVector{<:Real};
    cmap::Symbol = :viridis,
    scale_mode::Symbol = :linear,
    tracks_key::AbstractString = "none",
    highlight_mask = nothing,
    highlight_label::AbstractString = "Critical Node",
    description::AbstractString = "",
    scale_hint::AbstractString = ""
)::String
    m_vals = Float64[
        u <= length(field_vals) && isfinite(field_vals[u]) ? Float64(field_vals[u]) : 0.0
        for u in marine_indices
    ]
    v_min = isempty(m_vals) ? 0.0 : minimum(m_vals)
    v_max = isempty(m_vals) ? 1.0 : maximum(m_vals)

    pos_vals = filter(>(1e-12), m_vals)
    v_ref = if (scale_mode === :sqrt_p99 || scale_mode === :log_p99) && !isempty(pos_vals)
        max(quantile(pos_vals, 0.99), v_min + 1e-9)
    else
        max(v_max, v_min + 1e-9)
    end
    span_ref = max(v_ref - v_min, 1e-9)
    v_scale = if scale_mode === :log_p99 && !isempty(pos_vals)
        max(median(pos_vals) - v_min, span_ref * 1e-3, 1e-9)
    else
        1.0
    end
    log_denom = scale_mode === :log_p99 ? max(log1p(span_ref / v_scale), 1e-9) : 1.0

    cols_vec = Vector{String}(undef, length(m_vals))
    ops_vec  = Vector{String}(undef, length(m_vals))
    for (idx, v) in enumerate(m_vals)
        if scale_mode === :log_p99
            if v <= 1e-12
                cols_vec[idx] = "\"" * _ramp_hex_color(0.0, cmap) * "\""
                ops_vec[idx]  = "0.06"
            else
                u_norm = clamp(log1p(max(0.0, v - v_min) / v_scale) / log_denom, 0.0, 1.0)
                cols_vec[idx] = "\"" * _ramp_hex_color(u_norm, cmap) * "\""
                ops_vec[idx]  = string(round(0.24 + 0.62 * u_norm; digits = 2))
            end
        elseif scale_mode === :sqrt_p99
            if v <= 1e-9
                cols_vec[idx] = "\"" * _ramp_hex_color(0.0, cmap) * "\""
                ops_vec[idx]  = "0.06"
            else
                u_norm = clamp(sqrt(max(0.0, v - v_min) / span_ref), 0.0, 1.0)
                cols_vec[idx] = "\"" * _ramp_hex_color(u_norm, cmap) * "\""
                ops_vec[idx]  = string(round(0.22 + 0.58 * u_norm; digits = 2))
            end
        else
            u_norm = clamp((v - v_min) / max(v_max - v_min, 1e-9), 0.0, 1.0)
            cols_vec[idx] = "\"" * _ramp_hex_color(u_norm, cmap) * "\""
            ops_vec[idx]  = "0.60"
        end
    end
    cols_str = join(cols_vec, ",")
    ops_str  = join(ops_vec, ",")
    vals_str = join([string(round(v; digits = 5)) for v in m_vals], ",")

    hi_idx = Int[]
    if highlight_mask !== nothing
        for (k, u) in enumerate(marine_indices)
            if u <= length(highlight_mask) && Bool(highlight_mask[u])
                push!(hi_idx, k - 1) # 0-based index into marine cells
            end
        end
    end
    hi_str = join(string.(hi_idx), ",")
    grad_stops = join(["$(_ramp_hex_color(s, cmap)) $(round(Int, s * 100))%" for s in 0.0:0.2:1.0], ", ")
    v_mid_disp = if scale_mode === :log_p99
        v_min + v_scale * expm1(0.5 * log_denom)
    elseif scale_mode === :sqrt_p99
        v_min + 0.25 * span_ref
    else
        0.5 * (v_min + v_ref)
    end
    scale_str = scale_mode === :log_p99 ? "log_p99" :
                (scale_mode === :sqrt_p99 ? "sqrt_p99" : "linear")

    return "{" *
        "\"id\":\"$(_js_escape(id))\"," *
        "\"tab\":\"$(_js_escape(tab_label))\"," *
        "\"title\":\"$(_js_escape(title))\"," *
        "\"label\":\"$(_js_escape(field_label))\"," *
        "\"desc\":\"$(_js_escape(description))\"," *
        "\"scale\":\"$scale_str\"," *
        "\"vscale\":$(round(v_scale; sigdigits=5))," *
        "\"scaleHint\":\"$(_js_escape(scale_hint))\"," *
        "\"vmin\":$(round(v_min; digits=4))," *
        "\"vmid\":$(round(v_mid_disp; digits=4))," *
        "\"vmax\":$(round(v_ref; digits=4))," *
        "\"grad\":\"$grad_stops\"," *
        "\"tracks\":\"$tracks_key\"," *
        "\"hiLabel\":\"$(_js_escape(highlight_label))\"," *
        "\"hi\":[$hi_str]," *
        "\"k\":[$cols_str]," *
        "\"o\":[$ops_str]," *
        "\"v\":[$vals_str]" *
        "}"
end

function _ppc_inner_html(ppc::NamedTuple)::String
    function _trace_svg_tab(vals::Vector{Float64}, label::String, color::String; width::Int = 420, height::Int = 175)::String
        N = length(vals)
        N < 2 && return "<p style='color:#64748b;font-size:12px;'>Insufficient draws</p>"
        vmin, vmax = minimum(vals), maximum(vals)
        abs(vmax - vmin) < 1e-14 && (vmax = vmin + 1.0)
        pad_l, pad_r, pad_t, pad_b = 52, 14, 18, 30
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b
        y_ax = height - pad_b
        io = IOBuffer()
        write(io, "<svg width=\"$width\" height=\"$height\" xmlns=\"http://www.w3.org/2000/svg\">")
        write(io, "<line x1=\"$pad_l\" y1=\"$y_ax\" x2=\"$(width - pad_r)\" y2=\"$y_ax\" stroke=\"#64748b\" stroke-width=\"1\"/>")
        write(io, "<line x1=\"$pad_l\" y1=\"$pad_t\" x2=\"$pad_l\" y2=\"$y_ax\" stroke=\"#64748b\" stroke-width=\"1\"/>")
        pts = String[]
        for i in 1:N
            px = pad_l + ((i - 1) / (N - 1)) * pw
            py = y_ax - ((vals[i] - vmin) / (vmax - vmin)) * ph
            push!(pts, (i == 1 ? "M" : "L") * " $(round(px; digits=1)) $(round(py; digits=1))")
        end
        write(io, "<path d=\"$(join(pts, " "))\" fill=\"none\" stroke=\"$color\" stroke-width=\"1.8\"/>")
        write(io, "<text x=\"$(pad_l + pw ÷ 2)\" y=\"$(height - 6)\" text-anchor=\"middle\" fill=\"#475569\" font-size=\"11\">$label</text>")
        write(io, "<text x=\"$(pad_l - 4)\" y=\"$y_ax\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"10\" font-family=\"monospace\">$(round(vmin; digits=4))</text>")
        write(io, "<text x=\"$(pad_l - 4)\" y=\"$(pad_t + 8)\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"10\" font-family=\"monospace\">$(round(vmax; digits=4))</text>")
        write(io, "</svg>")
        return String(take!(io))
    end

    function _dist_svg_tab(obs::Vector{Float64}, pred::Vector{Float64}; width::Int = 820, height::Int = 190)::String
        N = length(obs)
        N < 1 && return "<p>No distribution</p>"
        active_units = findall(i -> obs[i] > 0.0 || (i <= length(pred) && pred[i] > 1e-6), 1:N)
        isempty(active_units) && (active_units = collect(1:min(N, 60)))
        sort!(active_units, by = u -> (obs[u], u <= length(pred) ? pred[u] : 0.0), rev = true)
        n_show = min(length(active_units), 60)
        u_sel = active_units[1:n_show]
        obs_s = obs[u_sel]
        pred_s = [u <= length(pred) ? pred[u] : 0.0 for u in u_sel]
        vmax = max(maximum(obs_s), maximum(pred_s), 1e-14)
        pad_l, pad_r, pad_t, pad_b = 12, 12, 16, 26
        bw = max(2, (width - pad_l - pad_r) ÷ n_show)
        ph = height - pad_t - pad_b
        y_ax = height - pad_b
        io = IOBuffer()
        write(io, "<svg width=\"$width\" height=\"$height\" xmlns=\"http://www.w3.org/2000/svg\">")
        for i in 1:n_show
            x0 = pad_l + (i - 1) * bw
            h_o = round(Int, (obs_s[i] / vmax) * ph)
            h_p = round(Int, (pred_s[i] / vmax) * ph)
            write(io, "<rect x=\"$x0\" y=\"$(y_ax - h_o)\" width=\"$(max(1, bw - 1))\" height=\"$h_o\" fill=\"#0284c7\" opacity=\"0.78\"/>")
            write(io, "<rect x=\"$x0\" y=\"$(y_ax - h_p)\" width=\"$(max(1, bw - 1))\" height=\"$h_p\" fill=\"#e11d48\" opacity=\"0.48\"/>")
        end
        write(io, "<line x1=\"$pad_l\" y1=\"$y_ax\" x2=\"$(width - pad_r)\" y2=\"$y_ax\" stroke=\"#64748b\" stroke-width=\"1\"/>")
        write(io, "<text x=\"$(width ÷ 2)\" y=\"$(height - 6)\" text-anchor=\"middle\" fill=\"#475569\" font-size=\"11\">Top $n_show Recapture Units — Blue: Observed, Red: Posterior Mean Predicted</text>")
        write(io, "</svg>")
        return String(take!(io))
    end

    kl = Float64.(ppc.kl_divergences)
    obs_d = Float64.(ppc.observed_dist)
    pred_d = hasproperty(ppc, :predicted_dist_mean) ? Float64.(ppc.predicted_dist_mean) : Float64[]
    sm = ppc.summary
    svg_k = _trace_svg_tab(kl, "KL Divergence across Posterior Draws", "#e11d48")
    svg_d = _dist_svg_tab(obs_d, pred_d)

    return """
    <div class="grid">
      <div class="card">
        <h2>Posterior Predictive Summary ($(sm.n_observations) observations, $(sm.n_draws) draws)</h2>
        <p class="card-desc">Measures how closely the fitted movement model reproduces observed recapture locations across the marine grid. Lower Kullback-Leibler (KL) divergence indicates closer agreement between simulated and observed recapture distributions.</p>
        <table>
          <tr><th>Metric</th><th>Mean</th><th>SD</th><th>95% Credible Interval</th></tr>
          <tr><td>KL Divergence (nats)</td><td>$(round(sm.kl_mean; digits=4))</td><td>$(round(sm.kl_sd; digits=4))</td><td>[$(round(sm.kl_lower_ci; digits=4)), $(round(sm.kl_upper_ci; digits=4))]</td></tr>
        </table>
      </div>
      <div class="card"><h2>KL Divergence Trace</h2><p class="card-desc">Variation in model-data mismatch across posterior parameter draws; a narrow, stable trace indicates consistent predictive fit.</p>$svg_k</div>
    </div>
    <div class="card" style="overflow-x:auto;"><h2>Observed vs Predicted Recapture Distribution</h2><p class="card-desc">Compares observed recapture counts (blue) with posterior-mean model predictions (red) across the 60 most frequently recaptured marine units; overlapping bars indicate well-calibrated spatial redistribution.</p>$svg_d</div>
    """
end

function _svg_histogram(
    vals::AbstractVector{<:Real},
    xlabel::AbstractString,
    color::AbstractString = "#3b82f6";
    nbins::Int = 20,
    width::Int = 360,
    height::Int = 190,
    clip_quantile::Union{Nothing, Float64} = nothing
)::String
    v = Float64[Float64(x) for x in vals if isfinite(x)]
    isempty(v) && return "<p style='color:#64748b;font-size:12px;'>No finite data</p>"
    vmin = minimum(v)
    vmax_raw = maximum(v)
    n_tail = 0
    vmax = if clip_quantile !== nothing && 0.5 < clip_quantile < 1.0 && length(v) >= 10
        q_hi = quantile(v, clip_quantile)
        if q_hi > vmin + 1e-12 && vmax_raw > q_hi * 1.25
            n_tail = count(>(q_hi), v)
            q_hi
        else
            vmax_raw
        end
    else
        vmax_raw
    end
    if vmax - vmin < 1e-12
        vmin -= 0.5
        vmax += 0.5
    end
    edges = collect(range(vmin, vmax; length = nbins + 1))
    counts = zeros(Int, nbins)
    for x in v
        x > vmax && continue
        b = clamp(searchsortedlast(edges, x), 1, nbins)
        counts[b] += 1
    end
    cmax = max(1, maximum(counts))
    pad_l, pad_r, pad_t, pad_b = 40, 14, 16, 34
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    bw = pw / nbins
    y_ax = height - pad_b

    io = IOBuffer()
    write(io, "<svg width=\"$width\" height=\"$height\" xmlns=\"http://www.w3.org/2000/svg\">")
    for b in 1:nbins
        bh = (counts[b] / cmax) * ph
        x0 = pad_l + (b - 1) * bw
        y0 = y_ax - bh
        write(io, "<rect x=\"$(round(x0 + 0.8; digits=1))\" y=\"$(round(y0; digits=1))\" " *
                  "width=\"$(round(max(1.0, bw - 1.6); digits=1))\" height=\"$(round(bh; digits=1))\" " *
                  "fill=\"$color\" fill-opacity=\"0.82\" rx=\"1.5\"/>")
    end
    # Mean vertical line (clamped to display window when tail-clipped)
    mu = mean(v)
    mx = pad_l + clamp((mu - vmin) / (vmax - vmin), 0.0, 1.0) * pw
    write(io, "<line x1=\"$(round(mx; digits=1))\" y1=\"$pad_t\" x2=\"$(round(mx; digits=1))\" y2=\"$y_ax\" " *
              "stroke=\"#ef4444\" stroke-width=\"1.6\" stroke-dasharray=\"4,3\"/>")
    write(io, "<line x1=\"$pad_l\" y1=\"$y_ax\" x2=\"$(width - pad_r)\" y2=\"$y_ax\" stroke=\"#64748b\" stroke-width=\"1\"/>")
    write(io, "<line x1=\"$pad_l\" y1=\"$pad_t\" x2=\"$pad_l\" y2=\"$y_ax\" stroke=\"#64748b\" stroke-width=\"1\"/>")
    write(io, "<text x=\"$pad_l\" y=\"$(y_ax + 13)\" fill=\"#64748b\" font-size=\"10\" font-family=\"monospace\">$(round(vmin; digits=2))</text>")
    vmax_lbl = n_tail > 0 ? "$(round(vmax; digits=1))+" : "$(round(vmax; digits=2))"
    write(io, "<text x=\"$(width - pad_r)\" y=\"$(y_ax + 13)\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"10\" font-family=\"monospace\">$vmax_lbl</text>")
    write(io, "<text x=\"$(pad_l - 4)\" y=\"$(pad_t + 8)\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"10\" font-family=\"monospace\">$cmax</text>")
    tail_note = n_tail > 0 ? ", $n_tail > $(round(vmax; digits=0))" : ""
    write(io, "<text x=\"$(pad_l + pw ÷ 2)\" y=\"$(height - 5)\" text-anchor=\"middle\" fill=\"#475569\" font-size=\"11\">$xlabel (mean=$(round(mu; digits=2))$tail_note)</text>")
    write(io, "</svg>")
    return String(take!(io))
end

function _svg_polar_rose(bearings_deg::AbstractVector{<:Real}; size_px::Int = 200)::String
    b_clean = Float64[mod(Float64(b), 360.0) for b in bearings_deg if isfinite(b)]
    isempty(b_clean) && return "<p style='color:#64748b;font-size:12px;'>No bearings</p>"
    n_sec = 12
    sec_w = 360.0 / n_sec
    counts = zeros(Int, n_sec)
    for b in b_clean
        idx = clamp(floor(Int, b / sec_w) + 1, 1, n_sec)
        counts[idx] += 1
    end
    cmax = max(1, maximum(counts))
    cx = size_px / 2
    cy = size_px / 2
    rmax = size_px * 0.38

    io = IOBuffer()
    write(io, "<svg width=\"$size_px\" height=\"$size_px\" xmlns=\"http://www.w3.org/2000/svg\">")
    for frac in (0.5, 1.0)
        write(io, "<circle cx=\"$cx\" cy=\"$cy\" r=\"$(round(rmax * frac; digits=1))\" fill=\"none\" stroke=\"#cbd5e1\" stroke-width=\"0.8\" stroke-dasharray=\"2,2\"/>")
    end
    write(io, "<line x1=\"$cx\" y1=\"$(cy - rmax - 6)\" x2=\"$cx\" y2=\"$(cy + rmax + 6)\" stroke=\"#cbd5e1\" stroke-width=\"0.8\"/>")
    write(io, "<line x1=\"$(cx - rmax - 6)\" y1=\"$cy\" x2=\"$(cx + rmax + 6)\" y2=\"$cy\" stroke=\"#cbd5e1\" stroke-width=\"0.8\"/>")
    for s in 1:n_sec
        counts[s] == 0 && continue
        r = (counts[s] / cmax) * rmax
        a1 = deg2rad((s - 1) * sec_w - 90.0)
        a2 = deg2rad(s * sec_w - 90.0)
        x1, y1 = cx + r * cos(a1), cy + r * sin(a1)
        x2, y2 = cx + r * cos(a2), cy + r * sin(a2)
        write(io, "<path d=\"M $cx $cy L $(round(x1; digits=1)) $(round(y1; digits=1)) A $(round(r; digits=1)) $(round(r; digits=1)) 0 0 1 $(round(x2; digits=1)) $(round(y2; digits=1)) Z\" fill=\"#0ea5e9\" fill-opacity=\"0.72\" stroke=\"#0284c7\" stroke-width=\"1\"/>")
    end
    write(io, "<text x=\"$cx\" y=\"13\" text-anchor=\"middle\" fill=\"#475569\" font-size=\"10\" font-weight=\"600\">N</text>")
    write(io, "<text x=\"$(size_px - 8)\" y=\"$(cy + 4)\" text-anchor=\"end\" fill=\"#475569\" font-size=\"10\" font-weight=\"600\">E</text>")
    write(io, "<text x=\"$cx\" y=\"$(size_px - 4)\" text-anchor=\"middle\" fill=\"#475569\" font-size=\"10\" font-weight=\"600\">S</text>")
    write(io, "<text x=\"8\" y=\"$(cy + 4)\" fill=\"#475569\" font-size=\"10\" font-weight=\"600\">W</text>")
    write(io, "</svg>")
    return String(take!(io))
end

function _html_escape(s::AbstractString)::String
    s1 = replace(String(s), "&" => "&amp;")
    s2 = replace(s1, "<" => "&lt;")
    return replace(s2, ">" => "&gt;")
end

function _path_analysis_inner_html(
    kernels::NamedTuple,
    paths_rich::Vector,
    mov_stats,
    phenology,
    trait_models;
    bathy_mig = nothing,
    disp_diag = nothing,
    bio_strat = nothing,
    spatial_roses = nothing,
    taxis_proj = nothing,
    validation = nothing
)::String
    a_draws = _sample_column(hasproperty(kernels, :alpha_samples) ? kernels.alpha_samples : nothing)
    r_draws = _sample_column(hasproperty(kernels, :rho_samples) ? kernels.rho_samples : nothing)
    g_draws = _sample_column(hasproperty(kernels, :gamma_samples) ? kernels.gamma_samples : nothing)

    ci_txt(v) = length(v) >= 2 ?
        "[$(round(quantile(v, 0.025); digits=4)), $(round(quantile(v, 0.975); digits=4))]" : "—"

    disps  = mov_stats !== nothing ? mov_stats.net_displacement_km : Float64[]
    plens  = mov_stats !== nothing ? mov_stats.path_length_km : Float64[]
    speeds = Float64[p.speed_km_per_year for p in paths_rich if isfinite(p.speed_km_per_year)]

    pheno_df = if phenology !== nothing && hasproperty(phenology, :monthly_df)
        phenology.monthly_df
    elseif phenology !== nothing && hasproperty(phenology, :monthly)
        phenology.monthly
    else
        DataFrame()
    end
    pheno_rows = ""
    if nrow(pheno_df) > 0
        for r in eachrow(pheno_df)
            m_disp = hasproperty(r, :mean_disp_km) ? r.mean_disp_km : 0.0
            med_disp = hasproperty(r, :median_disp_km) ? r.median_disp_km : m_disp
            m_dz = hasproperty(r, :mean_dz_m) ? r.mean_dz_m : 0.0
            cnt = hasproperty(r, :n_events) ? r.n_events : 0
            cnt == 0 && continue
            pheno_rows *= "<tr><td>$(_html_escape(string(r.month_name)))</td><td>$cnt</td>" *
                          "<td>$(round(Float64(m_disp); digits=2))</td>" *
                          "<td>$(round(Float64(med_disp); digits=2))</td>" *
                          "<td>$(round(Float64(m_dz); digits=2))</td></tr>"
        end
    end

    trait_rows = ""
    trait_lbl = (trait_models !== nothing && hasproperty(trait_models, :trait_name)) ?
        _html_escape(string(trait_models.trait_name)) : "trait"
    if trait_models !== nothing && hasproperty(trait_models, :models) && !isempty(trait_models.models)
        for mname in sort!(collect(keys(trait_models.models)))
            mres = trait_models.models[mname]
            sl = hasproperty(mres, :beta_1) ? round(Float64(mres.beta_1); digits = 4) : 0.0
            r2 = hasproperty(mres, :r2) ? round(Float64(mres.r2); digits = 4) : 0.0
            pv = hasproperty(mres, :p_val) ? round(Float64(mres.p_val); digits = 4) : 1.0
            n_f = hasproperty(mres, :n_obs) ? string(mres.n_obs) : "—"
            trait_rows *= "<tr><td>$(_html_escape(mname))</td><td>$n_f</td>" *
                          "<td>$sl</td><td>$r2</td><td>$pv</td></tr>"
        end
    end

    alpha_card = isempty(a_draws) ? "" :
        "<div class=\"card\"><h2>Posterior Advection (&alpha;) Draws</h2>" *
        "<p class=\"card-desc\">Posterior distribution of the habitat-directed movement weight &alpha;; values near 1 indicate movement strongly follows habitat suitability gradients rather than pure random diffusion.</p>" *
        _svg_histogram(a_draws, "alpha", "#0891b2") * "</div>"
    pheno_card = isempty(pheno_rows) ? "" :
        "<div class=\"card\"><h2>Seasonal Movement Phenology (by Release Month)</h2>" *
        "<p class=\"card-desc\">Observed straight-line displacement and depth change (&Delta;z, negative = shallower, positive = deeper) grouped by month of release to highlight seasonal migration timing.</p>" *
        "<table><tr><th>Month</th><th>Events</th><th>Mean Disp. (km)</th>" *
        "<th>Med. Disp. (km)</th><th>Mean &Delta;z (m)</th></tr>" *
        pheno_rows * "</table></div>"
    trait_card = isempty(trait_rows) ? "" :
        "<div class=\"card\"><h2>Trait-Movement OLS Associations (Predictor: $trait_lbl)</h2>" *
        "<p class=\"card-desc\">Linear regressions testing whether individual body size ($trait_lbl) explains variation in displacement, path length, or speed; low R&sup2; and high p-values indicate weak size dependence.</p>" *
        "<table><tr><th>Response</th><th>n</th><th>Slope (&beta;&#8321;)</th>" *
        "<th>R&sup2;</th><th>p-value</th></tr>" *
        trait_rows * "</table></div>"
    extra_grid = (isempty(pheno_card) && isempty(trait_card)) ? "" :
        "<div class=\"grid\">$pheno_card$trait_card</div>"

    # Separate Acoustic (OTN) vs. Commercial Trap (BIO) inference card
    ds_card = ""
    if hasproperty(kernels, :datasource_summary) && kernels.datasource_summary !== nothing
        ds_df = kernels.datasource_summary isa DataFrame ?
            kernels.datasource_summary : DataFrame(kernels.datasource_summary)
        if nrow(ds_df) >= 2
            ds_rows = ""
            for r in eachrow(ds_df)
                ds_rows *= "<tr><td><b>$(_html_escape(string(r.stratum)))</b></td>" *
                    "<td>$(r.n_obs) ($(r.n_tags))</td>" *
                    "<td>$(round(Float64(r.pct_stationary); digits=1))%</td>" *
                    "<td>$(round(Float64(r.median_disp_km); digits=1)) / $(round(Float64(r.mean_disp_km); digits=1))</td>" *
                    "<td>$(round(Float64(r.alpha_mean); digits=3)) [$(round(Float64(r.alpha_ci_lo); digits=3)), $(round(Float64(r.alpha_ci_hi); digits=3))]</td>" *
                    "<td>$(round(Float64(r.rho_mean); digits=3)) [$(round(Float64(r.rho_ci_lo); digits=3)), $(round(Float64(r.rho_ci_hi); digits=3))]</td>" *
                    "<td>$(round(Float64(r.gamma_mean); digits=2))</td>" *
                    "<td>$(round(Float64(r.eff_speed_km_yr); digits=1)) / $(round(Float64(r.eff_diff_km2_yr); digits=0))</td></tr>"
            end
            ds_card = "<div class=\"card\" style=\"margin-bottom:18px;overflow-x:auto;\">" *
                "<h2>Acoustic Array (OTN) vs. Commercial Trap (BIO) Transition Kernel Inference</h2>" *
                "<p class=\"card-desc\">Compares movement parameters estimated from fishery-independent acoustic receivers (OTN, fine-scale local residency) against commercial trap recaptures (BIO, broader shelf dispersal adjusted for fishing effort).</p>" *
                "<table><tr><th>Data Source</th><th>Obs (Tags)</th><th>% Stat.</th>" *
                "<th>Med / Mean Disp (km)</th><th>&alpha; [95% CI]</th><th>&rho; [95% CI]</th>" *
                "<th>&gamma;</th><th>v<sub>eff</sub> (km/yr) / D<sub>eff</sub> (km&sup2;/yr)</th></tr>" *
                "$ds_rows</table></div>"
        end
    end

    # Compass bearing roses are placed at their geographic coordinates on the map
    # rather than rendered as a tabular grid inside the report pane.

    # Bathymetric, thermal (when available), & habitat gradient taxis vector projections
    taxis_card = ""
    if taxis_proj !== nothing && hasproperty(taxis_proj, :n_obs) && taxis_proj.n_obs > 0
        t_parts = (taxis_proj.overall, taxis_proj.by_season, taxis_proj.by_cw)
        has_thermal = any(
            df_p -> nrow(df_p) > 0 && hasproperty(df_p, :mean_warmward_cos) &&
                    any(x -> isfinite(Float64(x)), df_p.mean_warmward_cos),
            t_parts
        )
        trows = ""
        for df_part in t_parts
            nrow(df_part) == 0 && continue
            for r in eachrow(df_part)
                fmt_num(x, d=3) = isfinite(Float64(x)) ? string(round(Float64(x); digits=d)) : "—"
                therm_cells = has_thermal ?
                    "<td>$(fmt_num(r.mean_warmward_cos, 3)) (p=$(fmt_num(r.thermal_p, 3)))</td>" *
                    "<td>$(fmt_num(r.mean_delta_T_C, 2))</td>" : ""
                trows *= "<tr><td>$(_html_escape(string(r.stratum)))</td><td>$(r.n_bathy)</td>" *
                    "<td>$(fmt_num(r.mean_downslope_cos, 3)) (p=$(fmt_num(r.bathy_p, 3)))</td>" *
                    "<td>$(fmt_num(r.mean_downslope_km, 2))</td>" *
                    "<td>$(isfinite(Float64(r.along_isobath_frac)) ? string(round(100*Float64(r.along_isobath_frac); digits=1)) * "%" : "—")</td>" *
                    "$therm_cells<td>$(fmt_num(r.mean_hsi_upgrad_cos, 3))</td></tr>"
            end
        end
        t_title = has_thermal ?
            "Bathymetric (&nabla;z), Thermal (&nabla;T), &amp; Habitat (&nabla;HSI) Taxis Vector Projections" :
            "Bathymetric (&nabla;z) &amp; Habitat (&nabla;HSI) Taxis Vector Projections"
        t_desc = has_thermal ?
            "Positive cosines denote net movement downslope (deeper), warmward (higher bottom T), and up-HSI gradient; along-isobath % measures contour-following perpendicular to &nabla;z." :
            "Positive cosines denote net movement downslope (deeper) and toward higher Habitat Suitability (up-&nabla;HSI); along-isobath % measures contour-following perpendicular to seabed slope."
        therm_hdr = has_thermal ? "<th>Warmward cos&psi;<sub>T</sub></th><th>Mean &Delta;T (&deg;C)</th>" : ""
        taxis_card = "<div class=\"card\" style=\"margin-bottom:18px;overflow-x:auto;\">" *
            "<h2>$t_title</h2>" *
            "<p class=\"card-desc\">$t_desc</p>" *
            "<table><tr><th>Stratum</th><th>n</th><th>Downslope cos&psi;<sub>z</sub></th>" *
            "<th>Downslope (km)</th><th>Along-Isobath</th>$therm_hdr" *
            "<th>Up-HSI cos&psi;<sub>h</sub></th></tr>$trows</table></div>"
    end

    # Multi-horizon MSD scaling & null-model LPPD comparison
    disp_card = ""
    if disp_diag !== nothing && hasproperty(disp_diag, :bins_df) && nrow(disp_diag.bins_df) > 0
        drows = ""
        for r in eachrow(disp_diag.bins_df)
            drows *= "<tr><td>$(_html_escape(string(r.horizon)))</td><td>$(r.n_obs)</td>" *
                     "<td>$(round(r.emp_disp_km; digits=1))</td><td>$(round(r.pred_disp_km; digits=1))</td>" *
                     "<td>$(round(r.emp_msd_km2; digits=0))</td><td>$(round(r.pred_msd_km2; digits=0))</td></tr>"
        end
        beta_s = isfinite(disp_diag.msd_beta) ? string(round(disp_diag.msd_beta; digits=3)) : "—"
        dlppd_s = isfinite(disp_diag.delta_lppd) ? string(round(disp_diag.delta_lppd; digits=1)) : "—"
        gain_s = isfinite(disp_diag.mean_ll_gain) ? string(round(disp_diag.mean_ll_gain; digits=4)) : "—"
        disp_card = "<div class=\"card\"><h2>Multi-Horizon Dispersal Scaling &amp; Null-Model Test</h2>" *
                    "<p class=\"card-desc\">Checks how far crabs spread as time between mark and recapture increases. Exponent &beta; &lt; 1 indicates sub-diffusive (constrained) spread; positive &Delta;LPPD confirms the habitat-guided model outperforms a uniform random walk.</p>" *
                    "<p style=\"font-size:12.5px;margin:0 0 10px 0;\">" *
                    "<b>MSD exponent &beta;:</b> $beta_s &nbsp;|&nbsp; " *
                    "<b>&Delta;LPPD (fitted vs. isotropic null):</b> $dlppd_s ($gain_s / obs, n = $(disp_diag.n_obs))</p>" *
                    "<table><tr><th>Time Interval</th><th>n</th><th>Emp. Disp (km)</th><th>Pred. Disp (km)</th>" *
                    "<th>Emp. MSD (km&sup2;)</th><th>Pred. MSD (km&sup2;)</th></tr>$drows</table></div>"
    end

    # Bathymetric / ontogenetic migration card
    bathy_card = ""
    if bathy_mig !== nothing && hasproperty(bathy_mig, :n_obs) && bathy_mig.n_obs > 0
        brows = ""
        if nrow(bathy_mig.by_size) > 0
            for r in eachrow(bathy_mig.by_size)
                brows *= "<tr><td>Size: $(_html_escape(string(r.stratum)))</td><td>$(r.n)</td>" *
                         "<td>$(round(r.mean_dz_m; digits=2))</td><td>$(round(r.median_dz_m; digits=2))</td>" *
                         "<td>$(round(r.pct_deeper; digits=1))%</td></tr>"
            end
        end
        if nrow(bathy_mig.by_cc) > 0
            for r in eachrow(bathy_mig.by_cc)
                brows *= "<tr><td>Shell: $(_html_escape(string(r.stratum)))</td><td>$(r.n)</td>" *
                         "<td>$(round(r.mean_dz_m; digits=2))</td><td>$(round(r.median_dz_m; digits=2))</td>" *
                         "<td>$(round(r.pct_deeper; digits=1))%</td></tr>"
            end
        end
        bathy_card = "<div class=\"card\"><h2>Bathymetric &amp; Ontogenetic Migration (&Delta;z = z<sub>rec</sub> &minus; z<sub>rel</sub>)</h2>" *
                     "<p class=\"card-desc\">Net change in seabed depth between mark and recapture by carapace width and shell condition; positive &Delta;z indicates movement into deeper water, negative into shallower water.</p>" *
                     "<p style=\"font-size:12.5px;margin:0 0 10px 0;\">" *
                     "<b>Overall mean &Delta;z:</b> $(round(bathy_mig.mean_dz_m; digits=2)) m " *
                     "(t = $(round(bathy_mig.t_stat; digits=2)), p = $(round(bathy_mig.p_val; digits=4))) &nbsp;|&nbsp; " *
                     "<b>Deeper (&gt;5 m):</b> $(round(bathy_mig.pct_deeper; digits=1))% &nbsp;|&nbsp; " *
                     "<b>Shallower (&lt;&minus;5 m):</b> $(round(bathy_mig.pct_shallower; digits=1))%</p>" *
                     "<table><tr><th>Stratum</th><th>n</th><th>Mean &Delta;z (m)</th>" *
                     "<th>Median &Delta;z (m)</th><th>% Deeper</th></tr>$brows</table></div>"
    end

    # Biological-state stratification (shell condition & size class)
    bio_card = ""
    if bio_strat !== nothing && (nrow(bio_strat.by_cc) > 0 || nrow(bio_strat.by_cw) > 0)
        srows = ""
        for r in eachrow(bio_strat.by_cc)
            srows *= "<tr><td>$(_html_escape(string(r.stratum)))</td><td>$(r.n_obs)</td><td>$(round(r.pct_stationary; digits=1))%</td>" *
                     "<td>$(round(r.median_disp_km; digits=1))</td><td>$(round(r.mean_disp_km; digits=1))</td>" *
                     "<td>$(round(r.mean_speed_yr; digits=1))</td></tr>"
        end
        for r in eachrow(bio_strat.by_cw)
            srows *= "<tr><td>CW $(_html_escape(string(r.stratum)))</td><td>$(r.n_obs)</td><td>$(round(r.pct_stationary; digits=1))%</td>" *
                     "<td>$(round(r.median_disp_km; digits=1))</td><td>$(round(r.mean_disp_km; digits=1))</td>" *
                     "<td>$(round(r.mean_speed_yr; digits=1))</td></tr>"
        end
        bio_card = "<div class=\"card\"><h2>Biological-State Mobility Stratification (All Observations)</h2>" *
                   "<p class=\"card-desc\">Observed movement distance, speed, and site fidelity (% recaptured in the same 5 km cell) broken down by shell hardness and carapace width.</p>" *
                   "<table><tr><th>Stratum</th><th>n</th><th>% Stationary</th><th>Med. Disp (km)</th>" *
                   "<th>Mean Disp (km)</th><th>Mean Speed (km/yr)</th></tr>$srows</table></div>"
    end

    # 1-Year (P^365) HSI-weighted regional management-area connectivity matrix
    conn_card = ""
    if validation !== nothing && hasproperty(validation, :connectivity_matrix) &&
       validation.connectivity_matrix !== nothing && validation.connectivity_matrix.n_regions >= 2
        cm = validation.connectivity_matrix
        lbls = [_html_escape(string(l)) for l in cm.region_labels]
        hdr = "<tr><th>From \\ To (1-Yr P<sup>365</sup>)</th>" * join(["<th>$l</th>" for l in lbls], "") * "</tr>"
        crows = ""
        for i in 1:cm.n_regions
            cells = join([
                "<td>$(round(cm.connectivity_matrix[i, j]; digits=4)) (n=$(cm.flow_counts[i, j]))</td>"
                for j in 1:cm.n_regions
            ], "")
            crows *= "<tr><td><b>$(lbls[i])</b></td>$cells</tr>"
        end
        conn_card = "<div class=\"card\"><h2>1-Year HSI-Weighted Regional Transition Probabilities</h2>" *
                    "<p class=\"card-desc\">Model-predicted probability that a crab starting in a row management area is located in a column area after one year (diagonal = annual self-retention; n = observed tag counts).</p>" *
                    "<table>$hdr$crows</table></div>"
    end

    adv_grid_1 = (isempty(disp_card) && isempty(bathy_card)) ? "" :
        "<div class=\"grid\">$disp_card$bathy_card</div>"
    adv_grid_2 = (isempty(bio_card) && isempty(conn_card)) ? "" :
        "<div class=\"grid\">$bio_card$conn_card</div>"

    svg_disp = _svg_histogram(disps, "Net Displacement (km)", "#2563eb")
    svg_plen = _svg_histogram(plens, "Cumulative Bridge Length (km)", "#7c3aed")
    svg_spd  = _svg_histogram(speeds, "Speed (km/yr)", "#d97706"; clip_quantile = 0.96)

    disp_summary = isempty(disps) ? "—" : "$(round(mean(disps); digits=2)) / $(round(median(disps); digits=2))"
    plen_summary = isempty(plens) ? "—" : "$(round(mean(plens); digits=2)) / $(round(median(plens); digits=2))"
    spd_summary  = isempty(speeds) ? "—" : "$(round(mean(speeds); digits=1)) / $(round(quantile(speeds, 0.95); digits=1))"

    return """
    <div class="grid">
      <div class="card">
        <h2>Fitted Transition Kernel Parameters (Pooled)</h2>
        <p class="card-desc">Core daily movement parameters estimated across all tags: &alpha; is the weight on habitat-guided drift vs. random diffusion, &rho; is the daily probability of remaining in the same 5 km cell, and &gamma; controls how strongly crabs climb habitat gradients.</p>
        <table>
          <tr><th>Parameter</th><th>Posterior Mean</th><th>95% CI</th></tr>
          <tr><td>Advection weight (&alpha;)</td><td>$(round(Float64(mean(kernels.alpha_hat)); digits=4))</td><td>$(ci_txt(a_draws))</td></tr>
          <tr><td>Residence probability (&rho;)</td><td>$(round(Float64(mean(kernels.rho_hat)); digits=4))</td><td>$(ci_txt(r_draws))</td></tr>
          <tr><td>Habitat gradient (&gamma;)</td><td>$(round(Float64(mean(kernels.gamma_hat)); digits=4))</td><td>$(ci_txt(g_draws))</td></tr>
        </table>
      </div>
      <div class="card">
        <h2>Markov-Bridge Trajectory Summary ($(length(paths_rich)) sampled tracks)</h2>
        <p class="card-desc">Compares straight-line displacement between release and recapture against total reconstructed in-water path length and implied annual travel speed.</p>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Mean / Median net displacement (km)</td><td>$disp_summary</td></tr>
          <tr><td>Mean / Median Markov-bridge path length (km)</td><td>$plen_summary</td></tr>
          <tr><td>Mean / 95th-pct annual speed (km/yr)</td><td>$spd_summary</td></tr>
        </table>
      </div>
    </div>
    $ds_card
    $taxis_card
    <div class="grid">
      <div class="card"><h2>Net Displacement Distribution (km)</h2><p class="card-desc">Straight-line distance between mark and recapture across sampled tags (red dashed line = mean); most crabs are recaptured locally with a long tail of shelf-scale movers.</p>$svg_disp</div>
      <div class="card"><h2>Markov-Bridge Cumulative Length (km)</h2><p class="card-desc">Total marine path length accumulated along sampled daily-step bridges; larger values than net displacement reflect meandering and back-and-forth foraging.</p>$svg_plen</div>
      <div class="card"><h2>Annual Movement Speed (km/yr)</h2><p class="card-desc">Net displacement divided by elapsed time between mark and recapture; right-skewed tail reflects fast short-interval movers.</p>$svg_spd</div>
      $alpha_card
    </div>
    $extra_grid
    $adv_grid_1
    $adv_grid_2
    """
end

function _serialize_tracks_json(paths_vec::Vector; max_draw::Int = 300)::String
    palette = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#a855f7", "#ec4899"]
    items = String[]
    drawn = 0
    for idx in eachindex(paths_vec)
        drawn >= max_draw && break
        p = paths_vec[idx]
        (length(p.coords) < 2 || p.total_dist_km <= 0.05) && continue
        drawn += 1
        pts = join(["[$(round(Float64(c[2]); digits=5)),$(round(Float64(c[1]); digits=5))]" for c in p.coords], ",")
        col = palette[mod1(drawn, length(palette))]
        n_hops = length(p.coords) - 1
        info = _js_escape(
            "<b>Tag $(p.tagid)</b> ($n_hops marine hops)<br>" *
            "Bridge path length: $(round(p.total_dist_km; digits=1)) km<br>" *
            "Net displacement: $(round(p.displacement_km; digits=1)) km<br>" *
            "Speed: $(round(p.speed_km_per_year; digits=1)) km/yr | Mean HSI: $(round(p.mean_hsi; digits=3))"
        )
        push!(items, "{\"c\":[$pts],\"k\":\"$col\",\"p\":\"$info\"}")
    end
    return "[" * join(items, ",") * "]"
end

function _svg_map_polar_rose(
    counts::AbstractVector{<:Integer},
    mean_bearing_deg::Real,
    r_bar::Real,
    rayleigh_p::Real;
    size_px::Int = 78
)::String
    n_sec = length(counts)
    n_sec == 0 && return ""
    sec_w = 360.0 / n_sec
    cmax = max(1, maximum(counts))
    cx = size_px / 2.0
    cy = size_px / 2.0
    rmax = size_px * 0.40
    sig = rayleigh_p < 0.05
    petal_fill = sig ? "#0284c7" : "#64748b"
    petal_stroke = sig ? "#0369a1" : "#475569"

    io = IOBuffer()
    write(io, "<svg width=\"$size_px\" height=\"$size_px\" viewBox=\"0 0 $size_px $size_px\" xmlns=\"http://www.w3.org/2000/svg\">")
    write(io, "<circle cx=\"$cx\" cy=\"$cy\" r=\"$(round(rmax + 3.0; digits=1))\" fill=\"rgba(255,255,255,0.88)\" stroke=\"#334155\" stroke-width=\"1.1\"/>")
    write(io, "<circle cx=\"$cx\" cy=\"$cy\" r=\"$(round(rmax * 0.5; digits=1))\" fill=\"none\" stroke=\"#94a3b8\" stroke-width=\"0.6\" stroke-dasharray=\"2,2\"/>")
    write(io, "<line x1=\"$cx\" y1=\"$(round(cy - rmax; digits=1))\" x2=\"$cx\" y2=\"$(round(cy + rmax; digits=1))\" stroke=\"#94a3b8\" stroke-width=\"0.6\"/>")
    write(io, "<line x1=\"$(round(cx - rmax; digits=1))\" y1=\"$cy\" x2=\"$(round(cx + rmax; digits=1))\" y2=\"$cy\" stroke=\"#94a3b8\" stroke-width=\"0.6\"/>")

    for s in 1:n_sec
        counts[s] == 0 && continue
        r = max(2.0, (counts[s] / cmax) * rmax)
        a1 = deg2rad((s - 1) * sec_w - 90.0)
        a2 = deg2rad(s * sec_w - 90.0)
        x1, y1 = cx + r * cos(a1), cy + r * sin(a1)
        x2, y2 = cx + r * cos(a2), cy + r * sin(a2)
        write(io, "<path d=\"M $cx $cy L $(round(x1; digits=1)) $(round(y1; digits=1)) A $(round(r; digits=1)) $(round(r; digits=1)) 0 0 1 $(round(x2; digits=1)) $(round(y2; digits=1)) Z\" fill=\"$petal_fill\" fill-opacity=\"0.78\" stroke=\"$petal_stroke\" stroke-width=\"0.8\"/>")
    end

    if isfinite(mean_bearing_deg) && r_bar > 0.02
        ang = deg2rad(Float64(mean_bearing_deg) - 90.0)
        r_vec = (0.25 + 0.72 * min(1.0, Float64(r_bar))) * rmax
        tx = cx + r_vec * cos(ang)
        ty = cy + r_vec * sin(ang)
        write(io, "<line x1=\"$cx\" y1=\"$cy\" x2=\"$(round(tx; digits=1))\" y2=\"$(round(ty; digits=1))\" stroke=\"#dc2626\" stroke-width=\"2.0\"/>")
        write(io, "<circle cx=\"$(round(tx; digits=1))\" cy=\"$(round(ty; digits=1))\" r=\"2.3\" fill=\"#dc2626\"/>")
    end
    write(io, "</svg>")
    return String(take!(io))
end

function _serialize_map_roses_json(spatial_roses)::String
    (spatial_roses === nothing || !hasproperty(spatial_roses, :cells) ||
     isempty(spatial_roses.cells)) && return "[]"
    items = String[]
    for c in spatial_roses.cells
        lat = round(Float64(c.center_lat); digits = 5)
        lon = round(Float64(c.center_lon); digits = 5)
        brg = Float64(c.mean_bearing_deg)
        r_bar = Float64(c.mean_vector_length)
        pval = Float64(c.rayleigh_p)
        svg = _js_escape(_svg_map_polar_rose(c.sector_counts, brg, r_bar, pval; size_px = 76))
        popup = _js_escape(
            "<b>100 &times; 100 km Sector: $(_html_escape(c.sector_label))</b><br>" *
            "Events (n): <b>$(c.n_events)</b> &middot; Mean disp.: <b>$(round(c.mean_disp_km; digits=1)) km</b><br>" *
            "Mean bearing &theta;&#772;: <b>$(round(brg; digits=1))&deg;</b> &middot; Resultant R&#772;: <b>$(round(r_bar; digits=3))</b><br>" *
            "von Mises &kappa;: <b>$(round(c.kappa; digits=2))</b> &middot; Rayleigh Z: <b>$(round(c.rayleigh_z; digits=1))</b> (p = $(round(pval; digits=4)))"
        )
        push!(items, "{\"lat\":$lat,\"lon\":$lon,\"svg\":\"$svg\",\"p\":\"$popup\"}")
    end
    return "[" * join(items, ",") * "]"
end

function _serialize_flux_arrows_json(domain_bottlenecks)::String
    (domain_bottlenecks === nothing ||
     !hasproperty(domain_bottlenecks, :directed_flux) ||
     domain_bottlenecks.directed_flux === nothing) && return "[]"
    df = domain_bottlenecks.directed_flux
    edges = hasproperty(df, :top_edges) ? df.top_edges : nothing
    edges === nothing && return "[]"
    rows_iter = edges isa DataFrame ? eachrow(edges) : edges
    (edges isa DataFrame ? nrow(edges) == 0 : isempty(edges)) && return "[]"
    max_f = maximum(Float64(e.net_flux) for e in rows_iter)
    max_f <= 0.0 && return "[]"
    items = String[]
    for e in rows_iter
        nf = Float64(e.net_flux)
        w = round(1.2 + 3.2 * sqrt(nf / max_f); digits = 2)
        op = round(0.45 + 0.45 * sqrt(nf / max_f); digits = 2)
        info = _js_escape("Net Flux: Unit $(e.u) -> Unit $(e.v) ($(round(nf; digits=2)))")
        push!(items,
            "{\"a\":[$(round(Float64(e.u_lat); digits=5)),$(round(Float64(e.u_lon); digits=5))]," *
            "\"b\":[$(round(Float64(e.v_lat); digits=5)),$(round(Float64(e.v_lon); digits=5))]," *
            "\"w\":$w,\"o\":$op,\"p\":\"$info\"}"
        )
    end
    return "[" * join(items, ",") * "]"
end

function _serialize_smooth_events_json(smooth_events::AbstractVector)::Tuple{String, String}
    bin_counts = zeros(Int, length(LIBERTY_BIN_SPECS))
    items = Vector{String}(undef, length(smooth_events))
    for (idx, ev) in enumerate(smooth_events)
        b = clamp(Int(ev.liberty_bin), 1, length(LIBERTY_BIN_SPECS))
        bin_counts[b] += 1
        pts = join([
            "[$(round(Float64(pt[2]); digits=5)),$(round(Float64(pt[1]); digits=5))]"
            for pt in ev.coords
        ], ",")
        yrs = round(Float64(ev.liberty_days) / 365.25; digits = 2)
        days_r = round(Float64(ev.liberty_days); digits = 1)
        disp_r = round(Float64(ev.displacement_km); digits = 1)
        plen_r = round(Float64(ev.path_length_km); digits = 1)
        # Precompute directional arrowhead anchor (~62% cumulative arc length)
        n_pts = length(ev.coords)
        ah_idx = 0
        if n_pts >= 2 && (ev.coords[1] != ev.coords[end])
            seg_l = zeros(Float64, n_pts)
            for k_p in 2:n_pts
                seg_l[k_p] = seg_l[k_p - 1] + hypot(
                    Float64(ev.coords[k_p][1]) - Float64(ev.coords[k_p - 1][1]),
                    Float64(ev.coords[k_p][2]) - Float64(ev.coords[k_p - 1][2])
                )
            end
            if seg_l[end] > 1e-5
                ah_idx = clamp(searchsortedfirst(seg_l, 0.62 * seg_l[end]), 2, n_pts) - 1
            end
        end
        info = _js_escape(
            "<b>Tag $(_html_escape(ev.tagid))</b> ($(_html_escape(ev.datasource)), Event #$(ev.event_id))<br>" *
            "Time between mark &amp; recapture: <b>$(days_r) d</b> ($(yrs) yr &middot; $(_html_escape(ev.liberty_label)))<br>" *
            "Net displacement: <b>$(disp_r) km</b> &middot; Spline trajectory: <b>$(plen_r) km</b><br>" *
            "Mark/release epoch: $(ev.rel_time_str)"
        )
        items[idx] = "{\"i\":$(ev.event_id),\"b\":$b,\"k\":\"$(ev.color)\"," *
                     "\"d\":$disp_r,\"l\":$plen_r,\"t\":$days_r,\"a\":$ah_idx,\"c\":[$pts],\"p\":\"$info\"}"
    end
    bins_json = "[" * join([
        "{\"b\":$(sp.idx),\"lbl\":\"$(_js_escape(sp.label))\",\"k\":\"$(sp.color)\",\"n\":$(bin_counts[sp.idx])}"
        for sp in LIBERTY_BIN_SPECS
    ], ",") * "]"
    return ("[" * join(items, ",") * "]", bins_json)
end

function _export_consolidated_dashboard_html(
    filepath::AbstractString,
    loaded,
    kernels,
    path_results,
    diagnostics,
    params,
    validation,
    paths_rich::Vector,
    ag_paths::Vector,
    agent_space_use,
    mov_stats,
    phenology,
    trait_models;
    bathy_mig = nothing,
    disp_diag = nothing,
    bio_strat = nothing,
    spatial_roses = nothing,
    taxis_proj = nothing,
    smooth_events = NamedTuple[],
    eff_dispersal = nothing
)::String
    mkpath(dirname(filepath))
    mesh = loaded.mesh
    polys = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat :
            (hasproperty(mesh, :polygons) ? mesh.polygons : nothing)
    cents = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat :
            (hasproperty(mesh, :centroids) ? mesh.centroids : Tuple{Float64, Float64}[])
    isempty(cents) && error("Mesh has no geographic centroids for HTML export.")

    n_units = length(cents)
    land_mask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing

    marine_indices = Int[]
    shared_cells_json = String[]

    for i in 1:n_units
        is_land = land_mask !== nothing && i <= length(land_mask) && Bool(land_mask[i])
        is_land && continue
        lat = round(Float64(cents[i][2]); digits = 5)
        lon = round(Float64(cents[i][1]); digits = 5)
        has_poly = polys !== nothing && i <= length(polys) && length(polys[i]) >= 3
        push!(marine_indices, i)
        if has_poly
            coords = join(["[$(round(Float64(pt[2]); digits=5)),$(round(Float64(pt[1]); digits=5))]" for pt in polys[i]], ",")
            push!(shared_cells_json, "{\"u\":$i,\"pt\":[$lat,$lon],\"c\":[$coords]}")
        else
            push!(shared_cells_json, "{\"u\":$i,\"pt\":[$lat,$lon]}")
        end
    end

    c_lon = round(mean(Float64(cents[u][1]) for u in marine_indices); digits = 4)
    c_lat = round(mean(Float64(cents[u][2]) for u in marine_indices); digits = 4)

    hsi_field = hasproperty(loaded, :hsi_vec) && length(loaded.hsi_vec) >= n_units ?
        loaded.hsi_vec : fill(0.5, n_units)

    # Assemble map tab definitions (Smoothed Trajectories first as the default tab)
    map_layers_json = String[]

    # Tab 1 (Default): Smoothed All-Event Mark-Recapture Trajectories
    if !isempty(smooth_events)
        sm_bg = (hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing &&
                 length(loaded.resharded_depths) >= n_units) ? loaded.resharded_depths : hsi_field
        push!(map_layers_json, _encode_field_layer_json(
            "smoothed_all", "Smoothed Trajectories",
            "$(params.species_name) — Smoothed Predicted Trajectories ($(length(smooth_events)) Events by Time Between Mark and Recapture)",
            "Time Between Mark and Recapture", marine_indices, sm_bg;
            cmap = :viridis, scale_mode = :linear, tracks_key = "smooth",
            description = "Smoothed in-water routes linking each crab's mark/release (green dot) to its recapture (gold star), colour-coded by elapsed time. Arrowheads show movement direction; click the map to highlight nearby trajectories.",
            scale_hint = "Colours group events by elapsed time between mark and recapture."
        ))
    end

    # Tab 2: Paths & HSI
    push!(map_layers_json, _encode_field_layer_json(
        "paths", "Paths & HSI",
        "$(params.species_name) — Reconstructed Trajectories & Habitat Suitability",
        "Habitat Suitability (HSI)", marine_indices, hsi_field;
        cmap = params.cmap, scale_mode = :linear, tracks_key = "recon",
        description = "Background shading shows Habitat Suitability Index (0 = poor, 1 = optimal; 0.50 baseline in unsurveyed zones). Overlaid lines show sampled in-water movement paths connecting mark (green) and recapture (star) sites.",
        scale_hint = "0 = unsuitable habitat, 0.5 = neutral baseline, 1 = peak suitability."
    ))

    # Tab 2: Corridors
    corr_vec = zeros(Float64, n_units)
    if hasproperty(path_results, :corridors) && path_results.corridors !== nothing && !isempty(path_results.corridors)
        for cv in values(path_results.corridors)
            if cv isa AbstractVector && length(cv) >= n_units
                for i in 1:n_units
                    corr_vec[i] += Float64(cv[i])
                end
            elseif cv isa AbstractMatrix && size(cv, 1) == n_units
                if size(cv, 2) == n_units
                    # Recover nodal intensity from legacy outer-product corridor matrix
                    for i in 1:n_units
                        corr_vec[i] += sqrt(max(0.0, Float64(cv[i, i])))
                    end
                else
                    col_rng = size(cv, 2) >= 3 ? (2:(size(cv, 2) - 1)) : (1:size(cv, 2))
                    for col in col_rng
                        for i in 1:n_units
                            corr_vec[i] += Float64(cv[i, col]) / length(col_rng)
                        end
                    end
                end
            end
        end
    end
    if maximum(corr_vec; init = 0.0) <= 1e-12 && mov_stats !== nothing &&
       length(mov_stats.residence_time_per_patch) >= n_units
        for i in 1:n_units
            corr_vec[i] = Float64(mov_stats.residence_time_per_patch[i])
        end
    end
    corr_pos = filter(>(0.0), corr_vec[marine_indices])
    corr_hi_thresh = isempty(corr_pos) ? Inf : quantile(corr_pos, 0.92)
    corr_hi_mask = [corr_vec[u] >= corr_hi_thresh && corr_vec[u] > 0.0 for u in 1:n_units]
    push!(map_layers_json, _encode_field_layer_json(
        "corridors", "Corridors",
        "$(params.species_name) — Cumulative Markov Bridge Transit Corridors",
        "Corridor Transit Density", marine_indices, corr_vec;
        cmap = :inferno, scale_mode = :log_p99, tracks_key = "recon",
        highlight_mask = corr_hi_mask, highlight_label = "Core Corridor",
        description = "Shows where tagged crab most likely passed in transit between release and recapture (endpoints excluded). Brighter cells mark heavier traffic and yellow dots flag the top 8% core corridors; intensity reflects both movement routes and tagging coverage.",
        scale_hint = "Higher values = greater cumulative in-transit probability across sampled tags (log scale)."
    ))

    # Tab 3: Directed Flux & Structural Bottlenecks
    if hasproperty(path_results, :domain_bottlenecks) && path_results.domain_bottlenecks !== nothing
        db = path_results.domain_bottlenecks
        push!(map_layers_json, _encode_field_layer_json(
            "bottlenecks", "Directed Flux & Bottlenecks",
            "$(params.species_name) — Net Directed Migration Flux & Structural Bottlenecks B(u)",
            "Bottleneck Index B(u)", marine_indices, db.bottleneck_score;
            cmap = :plasma, scale_mode = :log_p99, tracks_key = "flux",
            highlight_mask = db.bottleneck_mask, highlight_label = "Bottleneck",
            description = "Highlights narrow chokepoints where crab transit concentrates through few open marine neighbours (bright cells and yellow dots). Blue arrows show net dominant movement direction and strength.",
            scale_hint = "Higher B(u) = heavier transit squeezed through fewer marine connections (log scale)."
        ))
    end

    # Tab 3b: Spatial Corridor Canalization & 1-Year Effective Dispersal Area
    if validation !== nothing && hasproperty(validation, :corridor_canalization) &&
       validation.corridor_canalization !== nothing
        cc = validation.corridor_canalization
        push!(map_layers_json, _encode_field_layer_json(
            "canalization", "Corridor Canalization",
            "$(params.species_name) — Spatial Corridor Canalization Index K(u) & Conduits",
            "Canalization Index K(u)", marine_indices, cc.canalization_index;
            cmap = :inferno, scale_mode = :sqrt_p99, tracks_key = "none",
            highlight_mask = cc.conduit_mask, highlight_label = "Conduit",
            description = "Separates narrow, habitat-guided movement conduits (bright values and yellow dots) from broad, undirected spreading (dark values) by combining transit volume with directional focus.",
            scale_hint = "0 = broad diffuse spread; 1 = tightly focused high-traffic conduit."
        ))
    end
    if eff_dispersal !== nothing && hasproperty(eff_dispersal, :effective_area_km2) &&
       length(eff_dispersal.effective_area_km2) >= n_units
        push!(map_layers_json, _encode_field_layer_json(
            "effective_dispersal", "1-Yr Effective Dispersal",
            "$(params.species_name) — 1-Year Effective Dispersal Area A_eff(u) = exp(H(P^$(eff_dispersal.horizon_steps))) * A_cell",
            "1-Yr Effective Dispersal Area (km²)", marine_indices, eff_dispersal.effective_area_km2;
            cmap = :viridis, scale_mode = :linear, tracks_key = "none",
            description = "Estimates the effective ocean area (km²) that crabs starting in each cell spread across over one year. Lower values (dark) indicate stronger local retention; higher values (yellow) indicate broader open-shelf mixing.",
            scale_hint = "Effective 1-year spread footprint in km² (smaller = tighter retention)."
        ))
    end
    if spatial_roses !== nothing && hasproperty(spatial_roses, :cells) && !isempty(spatial_roses.cells)
        rose_bg = (hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing &&
                   length(loaded.resharded_depths) >= n_units) ? loaded.resharded_depths : hsi_field
        rose_bg_lbl = (hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing) ?
            "Bathymetric Depth (m)" : "Habitat Suitability (HSI)"
        push!(map_layers_json, _encode_field_layer_json(
            "bearing_roses", "Compass Bearing Roses (100 km)",
            "$(params.species_name) — Spatially Discretized 100 x 100 km Compass Bearing Roses & Rayleigh Tests",
            rose_bg_lbl, marine_indices, rose_bg;
            cmap = :viridis, scale_mode = :linear, tracks_key = "roses",
            description = "Each compass rose summarizes observed movement directions within a 100 km block. Blue petals indicate statistically consistent directional travel (Rayleigh p < 0.05), grey indicates no preferred direction, and the red needle marks mean bearing.",
            scale_hint = "Background shading shows local seabed depth (or HSI)."
        ))
    end

    # Tab 4: Circuit Current Density
    if diagnostics !== nothing && hasproperty(diagnostics, :circuit) && diagnostics.circuit !== nothing
        circ = diagnostics.circuit
        pmask = hasproperty(circ, :robust_mask) ? circ.robust_mask : circ.pinch_mask
        push!(map_layers_json, _encode_field_layer_json(
            "circuit", "Circuit Flow",
            "$(params.species_name) — Electrical Circuit Current Density & Pinch-Points",
            "Interior Current Density", marine_indices, circ.current_density;
            cmap = :plasma, scale_mode = :log_p99, tracks_key = "none",
            highlight_mask = pmask, highlight_label = "Pinch-Point",
            description = "Models the ocean as an electrical resistor network calibrated to the fitted habitat selectivity (release/recapture endpoints excluded). Bright areas carry the strongest multi-route transit flow, and yellow dots mark robust top-5% pinch-points.",
            scale_hint = "Higher current = stronger domain-wide transit flow through this cell (log scale)."
        ))
    end

    # Tab 5: Forward Agent Space-Use
    if agent_space_use !== nothing && hasproperty(agent_space_use, :visit_probability)
        push!(map_layers_json, _encode_field_layer_json(
            "agents", "Agent Space-Use",
            "$(params.species_name) — Forward-Projected Agent Visitation Probability",
            "Visit Probability", marine_indices, agent_space_use.visit_probability;
            cmap = :viridis, scale_mode = :log_p99, tracks_key = "agent",
            description = "Shows where simulated crabs accumulate when released and stepped forward freely under the fitted movement model (without conditioning on recapture locations), revealing natural aggregation zones.",
            scale_hint = "Higher values = greater simulated visitation probability (log scale)."
        ))
    end

    # Tab 6: Tessellation & Depth
    depth_field = hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing &&
                  length(loaded.resharded_depths) >= n_units ?
        loaded.resharded_depths : hsi_field
    depth_label = (hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing) ?
        "Bathymetric Depth (m)" : "Habitat Suitability (HSI)"
    push!(map_layers_json, _encode_field_layer_json(
        "tessellation", "Tessellation & Depth",
        "$(params.species_name) — Spatial Hexagonal Tessellation ($(length(marine_indices)) Marine Units)",
        depth_label, marine_indices, depth_field;
        cmap = :viridis, scale_mode = :linear, tracks_key = "none",
        description = "Displays the 5 km marine hexagonal mesh used by the movement model, clipped to the coastline and coloured by seabed depth (metres) to show the underwater terrain.",
        scale_hint = "Seabed depth in metres across each marine hexagonal cell."
    ))

    recon_tracks_js = _serialize_tracks_json(paths_rich)
    agent_tracks_js = _serialize_tracks_json(ag_paths)
    flux_arrows_js  = _serialize_flux_arrows_json(
        hasproperty(path_results, :domain_bottlenecks) ? path_results.domain_bottlenecks : nothing
    )
    map_roses_js    = _serialize_map_roses_json(spatial_roses)
    smooth_events_js, interval_bins_js = _serialize_smooth_events_json(smooth_events)

    payload_json = "{" *
        "\"shared_cells\":[$(join(shared_cells_json, ","))]," *
        "\"map_layers\":[$(join(map_layers_json, ","))]," *
        "\"flux_arrows\":$flux_arrows_js," *
        "\"map_roses\":$map_roses_js," *
        "\"smooth_events\":$smooth_events_js," *
        "\"interval_bins\":$interval_bins_js," *
        "\"track_sets\":{\"recon\":$recon_tracks_js,\"agent\":$agent_tracks_js," *
        "\"flux\":[],\"roses\":[],\"smooth\":[],\"none\":[]}" *
        "}"
    sidecar_path = replace(String(filepath), r"\.html?$"i => "") * ".data.zz"
    sidecar_name = basename(sidecar_path)
    sidecar_name_js = _js_escape(sidecar_name)
    raw_bytes = Vector{UInt8}(codeunits(payload_json))
    compressed_bytes = transcode(ZlibCompressor(level = 7), raw_bytes)
    write(sidecar_path, compressed_bytes)
    analysis_html = _path_analysis_inner_html(
        kernels, paths_rich, mov_stats, phenology, trait_models;
        bathy_mig = bathy_mig, disp_diag = disp_diag, bio_strat = bio_strat,
        spatial_roses = spatial_roses, taxis_proj = taxis_proj,
        validation = validation
    )
    has_ppc = validation !== nothing && hasproperty(validation, :posterior_predictive) &&
              validation.posterior_predictive !== nothing
    ppc_html = has_ppc ? _ppc_inner_html(validation.posterior_predictive) : ""
    ppc_tab_btn = has_ppc ?
        "<button class=\"tab-btn\" data-kind=\"report\" data-target=\"ppc-pane\">Posterior Predictive (PPC)</button>" : ""
    ppc_pane_div = has_ppc ?
        "<div id=\"ppc-pane\" class=\"report-pane\"><div class=\"wrap\">$ppc_html</div></div>" : ""

    dark = params.dark_mode
    bg = dark ? "#0f172a" : "#f1f5f9"
    nav_bg = dark ? "#020617" : "#0f172a"
    card = dark ? "#1e293b" : "#ffffff"
    fg = dark ? "#e2e8f0" : "#1e293b"
    muted = dark ? "#94a3b8" : "#64748b"
    bdr = dark ? "#334155" : "#cbd5e1"
    panel_bg = dark ? "rgba(15,23,42,0.92)" : "rgba(255,255,255,0.94)"
    ocean_canvas = dark ? "#0c2340" : "#dbeafe"

    html = """
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$(_js_escape(params.species_name)) — TethysPath Movement Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.css"
          onerror="this.onerror=null;this.href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';"/>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.js"
            onerror="var s=document.createElement('script');s.src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';document.head.appendChild(s);"></script>
    <style>
      * { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; width: 100%; height: 100%; background: $bg; color: $fg;
                   font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; overflow: hidden; }
      .topbar { height: 48px; background: $nav_bg; color: #f8fafc; display: flex; align-items: center;
                padding: 0 16px; gap: 6px; border-bottom: 1px solid #1e293b; overflow-x: auto; white-space: nowrap; }
      .brand { font-weight: 700; font-size: 14px; margin-right: 12px; color: #38bdf8; letter-spacing: 0.02em; }
      .tab-btn { background: transparent; color: #cbd5e1; border: 1px solid transparent; padding: 6px 12px;
                 border-radius: 6px; font-size: 12.5px; font-weight: 500; cursor: pointer; transition: all 0.15s; }
      .tab-btn:hover { background: rgba(255,255,255,0.08); color: #ffffff; }
      .tab-btn.active { background: #0284c7; color: #ffffff; font-weight: 600; }
      .stage { position: relative; width: 100%; height: calc(100% - 48px); }
      #map-pane { width: 100%; height: 100%; position: relative; }
      #map { width: 100%; height: 100%; background: $ocean_canvas; }
      .report-pane { display: none; width: 100%; height: 100%; overflow-y: auto; padding: 24px; background: $bg; }
      .wrap { max-width: 1200px; margin: 0 auto; }
      h2 { font-size: 15.5px; margin: 0 0 12px 0; border-bottom: 1px solid $bdr; padding-bottom: 6px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin-bottom: 18px; }
      .card { background: $card; border: 1px solid $bdr; border-radius: 10px; padding: 16px; }
      .card-desc { font-size: 12px; color: $muted; line-height: 1.42; margin: -6px 0 10px 0; }
      table { width: 100%; border-collapse: collapse; font-size: 13px; }
      th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid $bdr; }
      th { color: $muted; font-weight: 600; }
      .left-dock { position: absolute; top: 12px; left: 12px; z-index: 1000; width: 294px;
                   max-height: calc(100% - 24px); overflow-y: auto; display: flex;
                   flex-direction: column; gap: 10px; }
      .info-panel { background: $panel_bg; color: $fg; padding: 10px 13px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2); border: 1px solid $bdr; }
      .info-panel h3 { margin: 0 0 4px 0; font-size: 13.5px; line-height: 1.3; }
      .info-panel .sub { font-size: 11.5px; color: $muted; line-height: 1.35; }
      .info-panel .desc { font-size: 11.5px; color: $fg; line-height: 1.42; margin-top: 7px;
                          padding-top: 7px; border-top: 1px solid $bdr; }
      .legend-box { background: $panel_bg; color: $fg; padding: 10px 13px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 12px; border: 1px solid $bdr; }
      #legend-grad-wrap { position: relative; }
      .grad-bar { height: 12px; width: 100%; border-radius: 3px; margin: 6px 0 4px 0; cursor: crosshair; }
      .scale-tooltip { display: none; position: absolute; left: 0; right: 0; bottom: calc(100% + 6px);
                       background: #0f172a; color: #f8fafc; padding: 6px 8px; border-radius: 6px;
                       font-size: 11px; line-height: 1.35; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                       pointer-events: none; z-index: 1200; }
      .tess-toggle-row { display: flex; align-items: center; gap: 6px; margin-top: 8px;
                         padding-top: 7px; border-top: 1px solid $bdr; font-size: 11.5px; cursor: pointer; }
      .grad-labels { display: flex; justify-content: space-between; font-family: monospace; font-size: 11px; }
      .swatch-list { display: none; flex-direction: column; gap: 4px; margin-top: 6px; font-size: 11.5px; }
      .swatch-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
      .swatch-left { display: flex; align-items: center; gap: 6px; }
      .swatch-line { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
      .symbol-legend { display: none; flex-direction: column; gap: 5px; margin-top: 8px;
                       padding-top: 8px; border-top: 1px solid $bdr; font-size: 11.5px; }
      .symbol-legend-hdr { font-weight: 600; font-size: 11px; color: $muted; text-transform: uppercase;
                           letter-spacing: 0.03em; margin-bottom: 1px; }
      .smooth-dock { display: none; background: $panel_bg; color: $fg; padding: 11px 13px;
                     border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.22); font-size: 12px;
                     border: 1px solid $bdr; }
      .smooth-dock h4 { margin: 0 0 8px 0; font-size: 12.5px; color: #0284c7; letter-spacing: 0.02em; }
      .smooth-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 7px; }
      .smooth-chk-row { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; font-size: 11.5px; cursor: pointer; }
      .smooth-input, .smooth-select { background: $card; color: $fg; border: 1px solid $bdr;
                                      border-radius: 5px; padding: 3px 6px; font-size: 12px; }
      .smooth-btn { background: #0284c7; color: #ffffff; border: none; border-radius: 5px;
                    padding: 4px 9px; font-size: 11.5px; font-weight: 600; cursor: pointer; }
      .smooth-btn:hover { background: #0369a1; }
      .smooth-stat { margin-top: 6px; padding-top: 6px; border-top: 1px solid $bdr;
                     font-size: 11px; color: $muted; line-height: 1.38; }
      .rose-marker { background: transparent; border: none; }
      .rose-marker svg { filter: drop-shadow(0 2px 4px rgba(15,23,42,0.35)); transition: transform 0.15s; }
      .rose-marker:hover svg { transform: scale(1.18); }
      #data-loader-overlay { position: fixed; inset: 0; z-index: 2500; display: flex;
                             align-items: center; justify-content: center;
                             background: rgba(15, 23, 42, 0.84); backdrop-filter: blur(5px); }
      #data-loader-overlay.hidden { display: none; }
      .loader-card { background: $card; color: $fg; border: 1px solid $bdr; border-radius: 10px;
                     padding: 20px 24px; max-width: 440px; width: 90%;
                     box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45); text-align: center; }
      .loader-title { font-size: 15px; font-weight: 600; color: #0284c7; margin-bottom: 8px; }
      .loader-status { font-size: 12.5px; color: $muted; line-height: 1.45; margin-bottom: 12px; }
      .loader-btn { display: inline-block; padding: 8px 14px; border-radius: 6px;
                    border: 1px solid #0284c7; background: rgba(2, 132, 199, 0.14);
                    color: $fg; font-size: 12.5px; font-weight: 600; cursor: pointer; }
      .loader-btn:hover { background: rgba(2, 132, 199, 0.25); }
    </style></head><body>
    <div class="topbar" id="tab-bar">
      <span class="brand">TethysPath · $(_js_escape(params.species_name))</span>
      <span id="map-tab-buttons" style="display:flex;gap:5px;"></span>
      <button class="tab-btn" data-kind="report" data-target="analysis-pane">Path Analysis &amp; Diagnostics</button>
      $ppc_tab_btn
    </div>

    <div class="stage">
      <div id="map-pane">
        <div id="map"></div>
        <div id="left-dock" class="left-dock">
          <div class="info-panel">
            <h3 id="map-title">Loading...</h3>
            <div class="sub" id="map-subtitle"></div>
            <div class="desc" id="map-desc" style="display:none;"></div>
          </div>
          <div class="legend-box">
            <div id="legend-label" style="font-weight:600;">Value</div>
            <div id="legend-grad-wrap">
              <div id="legend-scale-tip" class="scale-tooltip"></div>
              <div id="legend-bar" class="grad-bar"></div>
              <div class="grad-labels">
                <span id="legend-min">0</span>
                <span id="legend-mid">0.5</span>
                <span id="legend-max">1</span>
              </div>
            </div>
            <div id="legend-swatches" class="swatch-list"></div>
            <div id="legend-symbols" class="symbol-legend">
              <div class="symbol-legend-hdr">Markers &amp; Orientation</div>
              <div class="swatch-left">
                <svg width="16" height="14" viewBox="0 0 16 14"><circle cx="8" cy="7" r="4.2" fill="#22c55e" stroke="#064e3b" stroke-width="1.3"/></svg>
                <span>Mark / Release (point)</span>
              </div>
              <div class="swatch-left">
                <svg width="16" height="14" viewBox="0 0 16 14"><polygon points="8,1 9.9,5.0 14.3,5.5 11.0,8.5 11.9,12.8 8,10.6 4.1,12.8 5.0,8.5 1.7,5.5 6.1,5.0" fill="#f59e0b" stroke="#78350f" stroke-width="0.9"/></svg>
                <span>Recapture (star)</span>
              </div>
              <div class="swatch-left">
                <svg width="16" height="14" viewBox="0 0 16 14"><line x1="1" y1="7" x2="11" y2="7" stroke="#0284c7" stroke-width="1.8"/><polygon points="15,7 8.5,3.5 10.3,7 8.5,10.5" fill="#0284c7"/></svg>
                <span>Direction of Movement (arrowhead)</span>
              </div>
            </div>
            <label class="tess-toggle-row" title="Toggle the 5 km marine hexagonal tessellation overlay on or off in this panel">
              <input type="checkbox" id="toggle-tessellation-chk" checked/>
              <span>Show background tessellation</span>
            </label>
          </div>
          <div id="smooth-controls-dock" class="smooth-dock">
            <h4>Trajectory Controls &amp; Selection</h4>
            <div class="smooth-row">
              <span>Highlight Paths (<i>n</i>)</span>
              <div style="display:flex;gap:5px;align-items:center;">
                <input type="number" id="smooth-n-input" class="smooth-input" style="width:62px;text-align:center;" min="1" max="500" value="15"/>
                <button type="button" id="smooth-clear-btn" class="smooth-btn">Clear</button>
              </div>
            </div>
            <div class="smooth-row">
              <span>Click Criterion</span>
              <select id="smooth-mode-select" class="smooth-select" style="width:152px;">
                <option value="release" selected>Nearest Marks / Releases</option>
                <option value="any">Area Pass-Through</option>
                <option value="recapture">Nearest Recaptures</option>
              </select>
            </div>
            <div class="smooth-row">
              <span>Time Interval</span>
              <select id="smooth-bin-select" class="smooth-select" style="width:152px;">
                <option value="0">All Intervals</option>
                <option value="1">&lt; 1 month</option>
                <option value="2">1 mo - 1 yr</option>
                <option value="3">1 - 2 yr</option>
                <option value="4">2 - 3 yr</option>
                <option value="5">3 - 4 yr</option>
                <option value="6">&ge; 4 yr</option>
              </select>
            </div>
            <div class="smooth-row">
              <span>Base Opacity</span>
              <input type="range" id="smooth-alpha-slider" min="5" max="85" value="28" style="width:110px;"/>
              <span id="smooth-alpha-val" style="font-family:monospace;font-size:11px;width:32px;text-align:right;">28%</span>
            </div>
            <label class="smooth-chk-row">
              <input type="checkbox" id="smooth-show-endpoints" checked/>
              <span>Show Mark (point) &amp; Recapture (star)</span>
            </label>
            <label class="smooth-chk-row">
              <input type="checkbox" id="smooth-show-arrows" checked/>
              <span>Show Movement Direction Arrowheads</span>
            </label>
            <div id="smooth-status" class="smooth-stat">
              Click anywhere on the map to highlight the <i>n</i> nearest smoothed trajectories (press Esc to clear).
            </div>
          </div>
        </div>
      </div>

      <div id="analysis-pane" class="report-pane">
        <div class="wrap">$analysis_html</div>
      </div>
      $ppc_pane_div
    </div>

    <div id="data-loader-overlay">
      <div class="loader-card">
        <div class="loader-title">Loading Movement Dashboard Payload</div>
        <div class="loader-status" id="loader-status">Fetching $(_html_escape(sidecar_name))...</div>
        <div id="loader-picker" style="display:none;">
          <input type="file" id="sidecar-file-input" accept=".zz,.bin,.gz" style="display:none;"/>
          <label for="sidecar-file-input" class="loader-btn">Select $(_html_escape(sidecar_name))</label>
        </div>
      </div>
    </div>

    <script>
    const SIDECAR_FILE = "$sidecar_name_js";
    let SHARED_CELLS = [];
    let MAP_LAYERS = [];
    let FLUX_ARROWS = [];
    let MAP_ROSES = [];
    let SMOOTH_EVENTS = [];
    let INTERVAL_BINS = [];
    let TRACK_SETS = {recon: [], agent: [], flux: [], roses: [], smooth: [], none: []};

    function bootDashboard(payload) {
      if (typeof L === 'undefined') {
        setTimeout(function() { bootDashboard(payload); }, 80);
        return;
      }
      SHARED_CELLS = payload.shared_cells || [];
      MAP_LAYERS = payload.map_layers || [];
      FLUX_ARROWS = payload.flux_arrows || [];
      MAP_ROSES = payload.map_roses || [];
      SMOOTH_EVENTS = payload.smooth_events || [];
      INTERVAL_BINS = payload.interval_bins || [];
      TRACK_SETS = payload.track_sets || TRACK_SETS;

      const map = L.map('map', {preferCanvas: true, zoomControl: false}).setView([$c_lat, $c_lon], 6);
      L.control.zoom({position: 'topright'}).addTo(map);

      // Keyless public basemaps compatible with local file:// protocol (no API keys required)
      const esriOcean = L.layerGroup([
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
          maxZoom: 13, attribution: 'Tiles &copy; Esri — GEBCO, NOAA, CHS, OSU'
        }),
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}', {
          maxZoom: 13, opacity: 0.75
        })
      ]).addTo(map);

      const esriPhysical = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 8, attribution: 'Tiles &copy; Esri — US National Park Service'
      });

      const esriGray = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/$(dark ? "World_Dark_Gray_Base" : "World_Light_Gray_Base")/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 14, attribution: 'Tiles &copy; Esri'
      });

      const esriImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 15, attribution: 'Tiles &copy; Esri'
      });

      const offlineBlank = L.layerGroup();

      // Optional coordinate graticule (unchecked by default; no land polygons drawn)
      const graticuleLayer = L.layerGroup();
      for (let lon = -72; lon <= -52; lon += 2) {
        L.polyline([[39, lon], [51, lon]], {color: '#94a3b8', weight: 0.5, opacity: 0.35, dashArray: '3,4', interactive: false}).addTo(graticuleLayer);
      }
      for (let lat = 40; lat <= 50; lat += 2) {
        L.polyline([[lat, -72], [lat, -52]], {color: '#94a3b8', weight: 0.5, opacity: 0.35, dashArray: '3,4', interactive: false}).addTo(graticuleLayer);
      }

      const fieldLayer = L.layerGroup().addTo(map);
      const smoothBgLayer = L.layerGroup().addTo(map);
      const smoothHiLayer = L.layerGroup().addTo(map);
      const smoothPinLayer = L.layerGroup().addTo(map);
      const trackLayer = L.layerGroup().addTo(map);
      const fluxLayer = L.layerGroup().addTo(map);
      const roseLayer = L.layerGroup().addTo(map);
      const endpointLayer = L.layerGroup().addTo(map);
      const highlightLayer = L.layerGroup().addTo(map);
      const smoothCanvas = L.canvas({padding: 0.5});

      // Custom pixel-space 5-point StarMarker and tangent ArrowHeadMarker on Leaflet Canvas
      const StarMarker = L.CircleMarker.extend({
        _updatePath: function() {
          const rdr = this._renderer;
          if (!rdr || !rdr._ctx || !rdr._drawing || this._empty()) return;
          const ctx = rdr._ctx;
          const p = this._point;
          const R = Math.max(this._radius, 2.2);
          const r = R * 0.44;
          ctx.beginPath();
          for (let k = 0; k < 10; k++) {
            const rad = (k % 2 === 0) ? R : r;
            const ang = -Math.PI * 0.5 + k * (Math.PI / 5.0);
            const x = p.x + rad * Math.cos(ang);
            const y = p.y + rad * Math.sin(ang);
            if (k === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          rdr._fillStroke(ctx, this);
        }
      });

      const ArrowHeadMarker = L.CircleMarker.extend({
        _updatePath: function() {
          const rdr = this._renderer;
          if (!rdr || !rdr._ctx || !rdr._drawing || this._empty() || !this._map) return;
          const p1 = this._map.latLngToLayerPoint(this.options.pPrev);
          const p2 = this._map.latLngToLayerPoint(this.options.pNext);
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          if (dx * dx + dy * dy < 0.25) return;
          const ang = Math.atan2(dy, dx);
          const ctx = rdr._ctx;
          const p = this._point;
          const sz = Math.max(this._radius, 2.5);
          const ca = Math.cos(ang), sa = Math.sin(ang);
          ctx.beginPath();
          ctx.moveTo(p.x + ca * sz * 0.88, p.y + sa * sz * 0.88);
          ctx.lineTo(p.x - ca * sz * 0.72 - sa * sz * 0.62, p.y - sa * sz * 0.72 + ca * sz * 0.62);
          ctx.lineTo(p.x - ca * sz * 0.28, p.y - sa * sz * 0.28);
          ctx.lineTo(p.x - ca * sz * 0.72 + sa * sz * 0.62, p.y - sa * sz * 0.72 - ca * sz * 0.62);
          ctx.closePath();
          rdr._fillStroke(ctx, this);
        }
      });

      L.control.layers(
        {
          "Esri Ocean Basemap (No API Key)": esriOcean,
          "Esri Physical Relief (No API Key)": esriPhysical,
          "Esri Gray Canvas (No API Key)": esriGray,
          "Esri Satellite Imagery (No API Key)": esriImagery,
          "Blank Ocean Canvas": offlineBlank
        },
        {
          "Hexagonal Marine Field": fieldLayer,
          "Smoothed All-Event Trajectories": smoothBgLayer,
          "Highlighted Smoothed Trajectories": smoothHiLayer,
          "Trajectories": trackLayer,
          "Net Directed Flux Arrows": fluxLayer,
          "100 km Compass Bearing Roses": roseLayer,
          "Mark / Release & Recapture Endpoints": endpointLayer,
          "Critical Nodes (Bottlenecks / Pinch / Corridors)": highlightLayer,
          "Lat/Lon Graticule": graticuleLayer
        },
        {position: 'topright', collapsed: true}
      ).addTo(map);

      // Populate time-interval legend swatches
      const swatchContainer = document.getElementById('legend-swatches');
      INTERVAL_BINS.forEach(function(b) {
        const row = document.createElement('div');
        row.className = 'swatch-row';
        row.innerHTML = '<span class="swatch-left"><span class="swatch-line" style="background:' +
          b.k + ';"></span>' + b.lbl + '</span><span style="font-family:monospace;">' + b.n + '</span>';
        swatchContainer.appendChild(row);
      });

      const leftDock = document.getElementById('left-dock');
      if (leftDock && L.DomEvent) {
        L.DomEvent.disableClickPropagation(leftDock);
        L.DomEvent.disableScrollPropagation(leftDock);
      }

      const hiStarIcon = L.divIcon({
        className: '',
        html: '<svg width="16" height="16" viewBox="0 0 24 24" style="display:block;filter:drop-shadow(0 1px 2px rgba(0,0,0,0.5));"><polygon points="12,1.8 15.1,8.5 22.4,9.3 17.0,14.2 18.5,21.4 12,17.7 5.5,21.4 7.0,14.2 1.6,9.3 8.9,8.5" fill="#f59e0b" stroke="#78350f" stroke-width="1.6"/></svg>',
        iconSize: [16, 16],
        iconAnchor: [8, 8]
      });

      let activeLayerCfg = null;
      let smoothPolylines = [];
      let smoothRelDots = [];
      let smoothRecStars = [];
      let smoothDirArrows = [];
      let smoothBuilt = false;
      let smoothBaseAlpha = 0.28;
      let smoothBinFilter = 0;
      let smoothClickMode = 'release';
      let smoothHighlightN = 15;
      let smoothShowEndpoints = true;
      let smoothShowArrows = true;
      let smoothSelectedSet = null;
      let smoothLastClick = null;

      function ensureSmoothPolylines() {
        if (smoothBuilt) return;
        smoothBuilt = true;
        SMOOTH_EVENTS.forEach(function(ev) {
          const isMoving = ev.c.length >= 2 && (ev.c[0][0] !== ev.c[ev.c.length - 1][0] || ev.c[0][1] !== ev.c[ev.c.length - 1][1]);
          let shp;
          if (isMoving) {
            shp = L.polyline(ev.c, {
              renderer: smoothCanvas,
              color: ev.k,
              weight: 1.6,
              opacity: smoothBaseAlpha,
              interactive: false
            });
          } else {
            shp = L.circleMarker(ev.c[0], {
              renderer: smoothCanvas,
              radius: 2.0,
              stroke: false,
              fillColor: ev.k,
              fillOpacity: smoothBaseAlpha,
              interactive: false
            });
          }
          smoothPolylines.push(shp);

          const relDot = L.circleMarker(ev.c[0], {
            renderer: smoothCanvas,
            radius: 2.2,
            color: '#064e3b',
            weight: 0.6,
            opacity: smoothBaseAlpha * 0.75,
            fillColor: '#22c55e',
            fillOpacity: smoothBaseAlpha * 0.85,
            interactive: false
          });
          smoothRelDots.push(relDot);

          const recStar = new StarMarker(ev.c[ev.c.length - 1], {
            renderer: smoothCanvas,
            radius: 3.8,
            color: '#78350f',
            weight: 0.6,
            opacity: smoothBaseAlpha * 0.85,
            fillColor: '#f59e0b',
            fillOpacity: Math.min(0.92, smoothBaseAlpha * 1.15),
            interactive: false
          });
          smoothRecStars.push(recStar);

          if (isMoving && ev.a >= 1 && ev.a < ev.c.length) {
            const pA = ev.c[ev.a - 1];
            const pB = ev.c[ev.a];
            const midPt = [0.5 * (pA[0] + pB[0]), 0.5 * (pA[1] + pB[1])];
            const arr = new ArrowHeadMarker(midPt, {
              renderer: smoothCanvas,
              pPrev: pA,
              pNext: pB,
              radius: 4.4,
              stroke: false,
              fillColor: ev.k,
              fillOpacity: Math.min(0.92, smoothBaseAlpha * 1.45),
              interactive: false
            });
            smoothDirArrows.push(arr);
          } else {
            smoothDirArrows.push(null);
          }
        });
      }

      function haversineKm(lat1, lon1, lat2, lon2) {
        const toRad = Math.PI / 180.0;
        const dLat = (lat2 - lat1) * toRad;
        const dLon = (lon2 - lon1) * toRad;
        const a = Math.sin(dLat * 0.5) * Math.sin(dLat * 0.5) +
                  Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) *
                  Math.sin(dLon * 0.5) * Math.sin(dLon * 0.5);
        return 12742.0 * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0.0, 1.0 - a)));
      }

      function minDistToPolylineKm(lat, lon, pts) {
        const cosLat = Math.cos(lat * Math.PI / 180.0);
        const ky = 111.132;
        const kx = 111.320 * cosLat;
        if (pts.length === 1) {
          const dx0 = (pts[0][1] - lon) * kx;
          const dy0 = (pts[0][0] - lat) * ky;
          return Math.sqrt(dx0 * dx0 + dy0 * dy0);
        }
        let best2 = Infinity;
        for (let k = 1; k < pts.length; k++) {
          const ax = (pts[k - 1][1] - lon) * kx;
          const ay = (pts[k - 1][0] - lat) * ky;
          const bx = (pts[k][1] - lon) * kx;
          const by = (pts[k][0] - lat) * ky;
          const dx = bx - ax;
          const dy = by - ay;
          const len2 = dx * dx + dy * dy;
          const t = len2 > 1e-12 ? Math.max(0.0, Math.min(1.0, -(ax * dx + ay * dy) / len2)) : 0.0;
          const px = ax + t * dx;
          const py = ay + t * dy;
          const d2 = px * px + py * py;
          if (d2 < best2) best2 = d2;
        }
        return Math.sqrt(best2);
      }

      function refreshSmoothDisplay() {
        if (!activeLayerCfg || activeLayerCfg.tracks !== 'smooth') return;
        ensureSmoothPolylines();
        smoothHiLayer.clearLayers();
        const hasCluster = smoothSelectedSet !== null && smoothSelectedSet.size > 0;
        const dimAlpha = Math.max(0.02, smoothBaseAlpha * 0.14);
        let nVisible = 0;

        for (let idx = 0; idx < SMOOTH_EVENTS.length; idx++) {
          const ev = SMOOTH_EVENTS[idx];
          const shp = smoothPolylines[idx];
          const relDot = smoothRelDots[idx];
          const recStar = smoothRecStars[idx];
          const dirArr = smoothDirArrows[idx];
          const binOk = (smoothBinFilter === 0 || ev.b === smoothBinFilter);
          if (!binOk) {
            if (shp.options.fillOpacity !== undefined && shp.setRadius) shp.setStyle({fillOpacity: 0.0});
            else shp.setStyle({opacity: 0.0});
            relDot.setStyle({opacity: 0.0, fillOpacity: 0.0});
            recStar.setStyle({opacity: 0.0, fillOpacity: 0.0});
            if (dirArr) dirArr.setStyle({fillOpacity: 0.0});
            continue;
          }
          nVisible++;
          const isSel = hasCluster && smoothSelectedSet.has(ev.i);
          const op = hasCluster ? (isSel ? 0.0 : dimAlpha) : smoothBaseAlpha;
          if (shp.options.fillOpacity !== undefined && shp.setRadius) {
            shp.setStyle({fillOpacity: op});
          } else {
            shp.setStyle({opacity: op, weight: hasCluster ? 1.1 : 1.7});
          }

          const epOp = smoothShowEndpoints ? op : 0.0;
          relDot.setStyle({opacity: epOp * 0.75, fillOpacity: epOp * 0.85});
          recStar.setStyle({opacity: epOp * 0.85, fillOpacity: Math.min(0.92, epOp * 1.15)});
          if (dirArr) {
            const arOp = smoothShowArrows ? (hasCluster ? (isSel ? 0.0 : dimAlpha) : Math.min(0.92, smoothBaseAlpha * 1.45)) : 0.0;
            dirArr.setStyle({fillOpacity: arOp});
          }

          if (isSel) {
            const isMoving = ev.c.length >= 2 && (ev.c[0][0] !== ev.c[ev.c.length - 1][0] || ev.c[0][1] !== ev.c[ev.c.length - 1][1]);
            if (isMoving) {
              L.polyline(ev.c, {color: '#0f172a', weight: 5.0, opacity: 0.68, interactive: false}).addTo(smoothHiLayer);
              L.polyline(ev.c, {color: ev.k, weight: 3.2, opacity: 0.96})
               .bindTooltip(ev.p, {sticky: true})
               .bindPopup(ev.p)
               .addTo(smoothHiLayer);
              if (ev.a >= 1 && ev.a < ev.c.length) {
                const pA = ev.c[ev.a - 1];
                const pB = ev.c[ev.a];
                const midPt = [0.5 * (pA[0] + pB[0]), 0.5 * (pA[1] + pB[1])];
                new ArrowHeadMarker(midPt, {
                  renderer: smoothCanvas,
                  pPrev: pA,
                  pNext: pB,
                  radius: 7.5,
                  color: '#0f172a',
                  weight: 1.5,
                  opacity: 0.95,
                  fillColor: ev.k,
                  fillOpacity: 1.0,
                  interactive: false
                }).addTo(smoothHiLayer);
              }
            }
            L.circleMarker(ev.c[0], {radius: 4.8, color: '#064e3b', fillColor: '#22c55e', fillOpacity: 0.96, weight: 1.5})
             .bindTooltip('<b>Mark / Release</b><br>' + ev.p)
             .bindPopup('<b>Mark / Release</b><br>' + ev.p)
             .addTo(smoothHiLayer);
            L.marker(ev.c[ev.c.length - 1], {icon: hiStarIcon})
             .bindTooltip('<b>Recapture</b><br>' + ev.p)
             .bindPopup('<b>Recapture</b><br>' + ev.p)
             .addTo(smoothHiLayer);
          }
        }

        const statusEl = document.getElementById('smooth-status');
        if (!hasCluster && statusEl) {
          statusEl.innerHTML = 'Showing <b>' + nVisible + '</b> spline trajectories. Click anywhere on the map to highlight <i>n</i> paths (Esc to clear).';
        }
      }

      function selectSmoothAtLatLng(latlng) {
        if (!activeLayerCfg || activeLayerCfg.tracks !== 'smooth' || SMOOTH_EVENTS.length === 0) return;
        const nIn = document.getElementById('smooth-n-input');
        const parsed = parseInt(nIn.value, 10);
        const nPick = Math.max(1, Math.min(500, Number.isFinite(parsed) ? parsed : 15));
        smoothHighlightN = nPick;
        nIn.value = nPick;

        const cands = [];
        for (let idx = 0; idx < SMOOTH_EVENTS.length; idx++) {
          const ev = SMOOTH_EVENTS[idx];
          if (smoothBinFilter !== 0 && ev.b !== smoothBinFilter) continue;
          let dKm;
          if (smoothClickMode === 'release') {
            dKm = haversineKm(latlng.lat, latlng.lng, ev.c[0][0], ev.c[0][1]);
          } else if (smoothClickMode === 'recapture') {
            const pEnd = ev.c[ev.c.length - 1];
            dKm = haversineKm(latlng.lat, latlng.lng, pEnd[0], pEnd[1]);
          } else {
            dKm = minDistToPolylineKm(latlng.lat, latlng.lng, ev.c);
          }
          cands.push({ev: ev, distKm: dKm});
        }
        if (cands.length === 0) return;
        cands.sort(function(a, b) { return a.distKm - b.distKm; });
        const picked = cands.slice(0, Math.min(nPick, cands.length));
        const maxDistKm = picked[picked.length - 1].distKm;

        smoothLastClick = latlng;
        smoothSelectedSet = new Set(picked.map(function(x) { return x.ev.i; }));

        smoothPinLayer.clearLayers();
        L.circle([latlng.lat, latlng.lng], {
          radius: Math.max(800.0, maxDistKm * 1000.0),
          color: '#facc15',
          weight: 1.5,
          dashArray: '5,5',
          fillColor: '#facc15',
          fillOpacity: 0.06,
          interactive: false
        }).addTo(smoothPinLayer);
        L.circleMarker([latlng.lat, latlng.lng], {
          radius: 5.5,
          color: '#ffffff',
          weight: 2.0,
          fillColor: '#facc15',
          fillOpacity: 0.95,
          interactive: false
        }).addTo(smoothPinLayer);

        let sumDisp = 0.0;
        let sumDays = 0.0;
        picked.forEach(function(x) {
          sumDisp += x.ev.d;
          sumDays += x.ev.t;
        });
        const meanDisp = (sumDisp / picked.length).toFixed(1);
        const meanYr = (sumDays / picked.length / 365.25).toFixed(2);

        refreshSmoothDisplay();
        const statusEl = document.getElementById('smooth-status');
        if (statusEl) {
          statusEl.innerHTML = '<b>Highlighted: ' + picked.length + ' paths</b> (within ' +
            maxDistKm.toFixed(1) + ' km)<br>Mean disp: <b>' + meanDisp +
            ' km</b> &middot; Mean time between mark &amp; recapture: <b>' + meanYr + ' yr</b>';
        }
      }

      function clearSmoothSelection() {
        smoothSelectedSet = null;
        smoothLastClick = null;
        smoothPinLayer.clearLayers();
        smoothHiLayer.clearLayers();
        refreshSmoothDisplay();
      }

      map.on('click', function(ev) {
        if (activeLayerCfg && activeLayerCfg.tracks === 'smooth') {
          selectSmoothAtLatLng(ev.latlng);
        }
      });

      document.getElementById('smooth-n-input').addEventListener('change', function() {
        if (smoothLastClick !== null) selectSmoothAtLatLng(smoothLastClick);
      });
      document.getElementById('smooth-mode-select').addEventListener('change', function(ev) {
        smoothClickMode = ev.target.value;
        if (smoothLastClick !== null) selectSmoothAtLatLng(smoothLastClick);
      });
      document.getElementById('smooth-bin-select').addEventListener('change', function(ev) {
        smoothBinFilter = parseInt(ev.target.value, 10) || 0;
        if (smoothLastClick !== null) selectSmoothAtLatLng(smoothLastClick);
        else refreshSmoothDisplay();
      });
      document.getElementById('smooth-alpha-slider').addEventListener('input', function(ev) {
        smoothBaseAlpha = (parseFloat(ev.target.value) || 28.0) / 100.0;
        document.getElementById('smooth-alpha-val').textContent = ev.target.value + '%';
        refreshSmoothDisplay();
      });
      document.getElementById('smooth-show-endpoints').addEventListener('change', function(ev) {
        smoothShowEndpoints = !!ev.target.checked;
        refreshSmoothDisplay();
      });
      document.getElementById('smooth-show-arrows').addEventListener('change', function(ev) {
        smoothShowArrows = !!ev.target.checked;
        refreshSmoothDisplay();
      });
      document.getElementById('smooth-clear-btn').addEventListener('click', function() {
        clearSmoothSelection();
      });
      document.addEventListener('keydown', function(ev) {
        if (ev.key === 'Escape' && activeLayerCfg && activeLayerCfg.tracks === 'smooth') {
          clearSmoothSelection();
        }
      });

      // Instantiate marine cell shapes once and recolor on tab switch
      const cellShapes = [];
      const boundsPts = [];
      SHARED_CELLS.forEach(function(cell) {
        let shp;
        if (cell.c) {
          shp = L.polygon(cell.c, {color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.60, opacity: 0.35, weight: 0.5}).addTo(fieldLayer);
          boundsPts.push(cell.c[0]);
        } else {
          shp = L.circleMarker(cell.pt, {radius: 4, color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.70, weight: 0.5}).addTo(fieldLayer);
          boundsPts.push(cell.pt);
        }
        cellShapes.push(shp);
      });
      if (boundsPts.length > 1) {
        map.fitBounds(boundsPts, {paddingTopLeft: [314, 24], paddingBottomRight: [24, 24]});
      }

      const tessVisibleByTab = {};
      const tessChk = document.getElementById('toggle-tessellation-chk');
      const legendBar = document.getElementById('legend-bar');
      const legendTip = document.getElementById('legend-scale-tip');

      function setTessellationVisibility(visible) {
        if (visible) {
          if (!map.hasLayer(fieldLayer)) map.addLayer(fieldLayer);
        } else {
          if (map.hasLayer(fieldLayer)) map.removeLayer(fieldLayer);
        }
        if (tessChk) tessChk.checked = visible;
      }

      if (tessChk) {
        tessChk.addEventListener('change', function(ev) {
          const on = !!ev.target.checked;
          if (activeLayerCfg) tessVisibleByTab[activeLayerCfg.id] = on;
          setTessellationVisibility(on);
        });
      }
      map.on('overlayadd', function(ev) {
        if (ev.layer === fieldLayer) {
          if (activeLayerCfg) tessVisibleByTab[activeLayerCfg.id] = true;
          if (tessChk) tessChk.checked = true;
        }
      });
      map.on('overlayremove', function(ev) {
        if (ev.layer === fieldLayer) {
          if (activeLayerCfg) tessVisibleByTab[activeLayerCfg.id] = false;
          if (tessChk) tessChk.checked = false;
        }
      });

      if (legendBar && legendTip) {
        legendBar.addEventListener('mousemove', function(ev) {
          if (!activeLayerCfg || activeLayerCfg.tracks === 'smooth') return;
          const rect = legendBar.getBoundingClientRect();
          if (rect.width <= 0) return;
          const u = Math.max(0.0, Math.min(1.0, (ev.clientX - rect.left) / rect.width));
          const vmin = Number(activeLayerCfg.vmin) || 0.0;
          const vmax = Number(activeLayerCfg.vmax) || 1.0;
          const isLog = activeLayerCfg.scale === 'log_p99';
          const isSqrt = activeLayerCfg.scale === 'sqrt_p99';
          let val;
          if (isLog) {
            const vs = Math.max(Number(activeLayerCfg.vscale) || 1e-6, 1e-12);
            val = vmin + vs * Math.expm1(u * Math.log1p(Math.max(0.0, vmax - vmin) / vs));
          } else if (isSqrt) {
            val = vmin + u * u * (vmax - vmin);
          } else {
            val = vmin + u * (vmax - vmin);
          }
          const av = Math.abs(val);
          const fmtVal = av >= 100 ? val.toFixed(1) : (av >= 1 ? val.toFixed(2) : val.toFixed(5));
          const modeTxt = isLog ? 'Logarithmic scale (capped at 99th pct)' :
            (isSqrt ? 'Square-root scale (capped at 99th pct)' : 'Linear scale');
          const hintTxt = activeLayerCfg.scaleHint ? ('<br>' + activeLayerCfg.scaleHint) : '';
          legendTip.innerHTML = '<b>' + activeLayerCfg.label + ': ' + fmtVal + '</b><br>' + modeTxt + hintTxt;
          legendTip.style.display = 'block';
        });
        legendBar.addEventListener('mouseleave', function() {
          legendTip.style.display = 'none';
        });
      }

      function applyMapLayer(cfg) {
        activeLayerCfg = cfg;
        document.getElementById('map-title').textContent = cfg.title;
        const trks = TRACK_SETS[cfg.tracks] || [];
        let sub = 'Marine units: ' + SHARED_CELLS.length;
        if (trks.length > 0) sub += ' · Trajectories: ' + trks.length;
        if (cfg.tracks === 'smooth' && SMOOTH_EVENTS.length > 0) {
          sub += ' · Events: ' + SMOOTH_EVENTS.length + ' (click map to highlight n paths)';
        }
        if (cfg.tracks === 'roses' && MAP_ROSES.length > 0) sub += ' · 100 km Compass Roses: ' + MAP_ROSES.length + ' (blue = Rayleigh p < 0.05, red vector = mean bearing)';
        if (cfg.hi.length > 0) sub += ' · ' + cfg.hiLabel + 's: ' + cfg.hi.length;
        document.getElementById('map-subtitle').textContent = sub;

        const descEl = document.getElementById('map-desc');
        if (descEl) {
          if (cfg.desc && cfg.desc.length > 0) {
            descEl.textContent = cfg.desc;
            descEl.style.display = 'block';
          } else {
            descEl.textContent = '';
            descEl.style.display = 'none';
          }
        }

        document.getElementById('legend-label').textContent = cfg.label;
        const gradWrap = document.getElementById('legend-grad-wrap');
        if (cfg.tracks === 'smooth') {
          gradWrap.style.display = 'none';
          if (legendTip) legendTip.style.display = 'none';
          document.getElementById('legend-swatches').style.display = 'flex';
          document.getElementById('legend-symbols').style.display = 'flex';
          document.getElementById('smooth-controls-dock').style.display = 'block';
        } else {
          gradWrap.style.display = 'block';
          const scaleDesc = cfg.scale === 'log_p99' ? 'Logarithmic scale (capped at 99th pct)' :
            (cfg.scale === 'sqrt_p99' ? 'Square-root scale (capped at 99th pct)' : 'Linear scale');
          gradWrap.title = cfg.label + ' — ' + scaleDesc + (cfg.scaleHint ? ('. ' + cfg.scaleHint) : '');
          document.getElementById('legend-swatches').style.display = 'none';
          document.getElementById('legend-symbols').style.display = 'none';
          document.getElementById('smooth-controls-dock').style.display = 'none';
          document.getElementById('legend-bar').style.background = 'linear-gradient(to right, ' + cfg.grad + ')';
          document.getElementById('legend-min').textContent = cfg.vmin;
          document.getElementById('legend-mid').textContent = cfg.vmid;
          document.getElementById('legend-max').textContent = cfg.vmax;
        }

        for (let i = 0; i < cellShapes.length; i++) {
          const col = cfg.k[i] || '#3b82f6';
          let fop = (cfg.o && cfg.o[i] !== undefined) ? cfg.o[i] : 0.60;
          let sop = fop < 0.12 ? 0.12 : 0.42;
          if (cfg.tracks === 'smooth') {
            fop = 0.28;
            sop = 0.20;
          } else if (cfg.tracks === 'roses') {
            fop = 0.25;
            sop = 0.18;
          }
          const val = cfg.v[i];
          cellShapes[i].setStyle({color: col, fillColor: col, fillOpacity: fop, opacity: sop});
          cellShapes[i].unbindTooltip();
          if (cfg.tracks !== 'smooth') {
            cellShapes[i].bindTooltip('Unit ' + SHARED_CELLS[i].u + ' · ' + cfg.label + ': ' + val);
          }
        }

        const defaultTess = (cfg.tracks !== 'smooth');
        const showTess = (tessVisibleByTab[cfg.id] !== undefined) ? tessVisibleByTab[cfg.id] : defaultTess;
        setTessellationVisibility(showTess);

        trackLayer.clearLayers();
        fluxLayer.clearLayers();
        roseLayer.clearLayers();
        endpointLayer.clearLayers();
        smoothBgLayer.clearLayers();
        smoothHiLayer.clearLayers();
        smoothPinLayer.clearLayers();
        if (cfg.tracks === 'smooth') {
          ensureSmoothPolylines();
          for (let k = 0; k < SMOOTH_EVENTS.length; k++) {
            smoothBgLayer.addLayer(smoothPolylines[k]);
            smoothBgLayer.addLayer(smoothRelDots[k]);
            smoothBgLayer.addLayer(smoothRecStars[k]);
            if (smoothDirArrows[k]) smoothBgLayer.addLayer(smoothDirArrows[k]);
          }
          if (smoothLastClick !== null) {
            selectSmoothAtLatLng(smoothLastClick);
          } else {
            refreshSmoothDisplay();
          }
        } else if (cfg.tracks === 'flux') {
          FLUX_ARROWS.forEach(function(f) {
            L.polyline([f.a, f.b], {color: '#38bdf8', weight: f.w, opacity: f.o})
             .bindTooltip(f.p).addTo(fluxLayer);
            const tipLat = f.a[0] + 0.82 * (f.b[0] - f.a[0]);
            const tipLon = f.a[1] + 0.82 * (f.b[1] - f.a[1]);
            L.circleMarker([tipLat, tipLon], {
              radius: Math.max(2.2, f.w * 0.85), color: '#0284c7',
              fillColor: '#e0f2fe', fillOpacity: 0.95, weight: 1
            }).bindTooltip(f.p).addTo(fluxLayer);
          });
        } else if (cfg.tracks === 'roses') {
          MAP_ROSES.forEach(function(r) {
            const icon = L.divIcon({
              html: r.svg,
              className: 'rose-marker',
              iconSize: [76, 76],
              iconAnchor: [38, 38]
            });
            L.marker([r.lat, r.lon], {icon: icon})
             .bindTooltip(r.p, {direction: 'top', offset: [0, -34]})
             .bindPopup(r.p)
             .addTo(roseLayer);
          });
        }
        trks.forEach(function(t) {
          L.polyline(t.c, {color: t.k, weight: 2.2, opacity: 0.88}).bindPopup(t.p).addTo(trackLayer);
          L.circleMarker(t.c[0], {radius: 3.6, color: '#064e3b', fillColor: '#22c55e', fillOpacity: 0.94, weight: 1.1})
           .bindPopup('<b>Mark / Release</b><br>' + t.p).addTo(endpointLayer);
          L.marker(t.c[t.c.length - 1], {icon: hiStarIcon})
           .bindPopup('<b>Recapture</b><br>' + t.p).addTo(endpointLayer);
        });

        highlightLayer.clearLayers();
        cfg.hi.forEach(function(idx) {
          const cell = SHARED_CELLS[idx];
          if (!cell) return;
          L.circleMarker(cell.pt, {radius: 4.5, color: '#eab308', fillColor: '#fde047', fillOpacity: 0.85, weight: 1.5})
           .bindTooltip(cfg.hiLabel + ' · Unit ' + cell.u + ' (' + cfg.v[idx] + ')').addTo(highlightLayer);
        });
      }

      // Build map tab buttons dynamically from MAP_LAYERS
      const btnWrap = document.getElementById('map-tab-buttons');
      MAP_LAYERS.forEach(function(cfg, idx) {
        const b = document.createElement('button');
        b.className = 'tab-btn' + (idx === 0 ? ' active' : '');
        b.textContent = cfg.tab;
        b.dataset.kind = 'map';
        b.dataset.layerIdx = idx;
        btnWrap.appendChild(b);
      });

      document.getElementById('tab-bar').addEventListener('click', function(ev) {
        const btn = ev.target.closest('.tab-btn');
        if (!btn) return;
        document.querySelectorAll('.tab-btn').forEach(function(el) { el.classList.remove('active'); });
        btn.classList.add('active');

        const mapPane = document.getElementById('map-pane');
        document.querySelectorAll('.report-pane').forEach(function(el) { el.style.display = 'none'; });

        if (btn.dataset.kind === 'map') {
          mapPane.style.display = 'block';
          map.invalidateSize();
          applyMapLayer(MAP_LAYERS[parseInt(btn.dataset.layerIdx, 10)]);
        } else {
          mapPane.style.display = 'none';
          const target = document.getElementById(btn.dataset.target);
          if (target) target.style.display = 'block';
        }
      });

      if (MAP_LAYERS.length > 0) {
        applyMapLayer(MAP_LAYERS[0]);
      }
      document.getElementById('data-loader-overlay').classList.add('hidden');
    }

    async function decodeZlibBuffer(arrayBuf, statusEl) {
      if (typeof DecompressionStream === 'undefined') {
        throw new Error('Browser does not support DecompressionStream.');
      }
      const stream = new Blob([arrayBuf]).stream().pipeThrough(new DecompressionStream('deflate'));
      const text = await new Response(stream).text();
      if (statusEl) {
        statusEl.textContent = 'Parsing JSON (' + (text.length / 1e6).toFixed(1) + ' MB)...';
        await new Promise(function(r) { setTimeout(r, 0); });
      }
      return JSON.parse(text);
    }

    async function loadSidecarAndBoot() {
      const statusEl = document.getElementById('loader-status');
      const pickerEl = document.getElementById('loader-picker');
      const fileInput = document.getElementById('sidecar-file-input');

      fileInput.addEventListener('change', async function(ev) {
        const file = ev.target.files && ev.target.files[0];
        if (!file) return;
        statusEl.textContent = 'Reading ' + file.name + ' (' + (file.size / 1e6).toFixed(2) + ' MB)...';
        try {
          const buf = await file.arrayBuffer();
          statusEl.textContent = 'Decompressing ' + file.name + '...';
          await new Promise(function(r) { setTimeout(r, 0); });
          const payload = await decodeZlibBuffer(buf, statusEl);
          bootDashboard(payload);
        } catch (err) {
          statusEl.textContent = 'Failed to load ' + file.name + ': ' + err.message;
          pickerEl.style.display = 'block';
        }
      });

      if (window.location.protocol === 'file:') {
        statusEl.textContent =
          'Local file:// mode: browsers block automatic fetch of ' + SIDECAR_FILE +
          '. Select ' + SIDECAR_FILE + ' below, or serve over HTTP/HTTPS (e.g. GitHub Pages) for auto-loading.';
        pickerEl.style.display = 'block';
        return;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(function() { controller.abort(); }, 15000);
      try {
        statusEl.textContent = 'Fetching ' + SIDECAR_FILE + '...';
        const resp = await fetch('./' + SIDECAR_FILE, {signal: controller.signal});
        clearTimeout(timeoutId);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const buf = await resp.arrayBuffer();
        statusEl.textContent = 'Decompressing ' + SIDECAR_FILE + ' (' + (buf.byteLength / 1e6).toFixed(2) + ' MB)...';
        await new Promise(function(r) { setTimeout(r, 0); });
        const payload = await decodeZlibBuffer(buf, statusEl);
        bootDashboard(payload);
      } catch (err) {
        clearTimeout(timeoutId);
        const msg = err.name === 'AbortError' ? 'request timed out after 15 s' : err.message;
        statusEl.textContent =
          'Automatic fetch of ' + SIDECAR_FILE + ' failed (' + msg + '). Select ' + SIDECAR_FILE + ' directly below.';
        pickerEl.style.display = 'block';
      }
    }

    loadSidecarAndBoot();
    </script></body></html>
    """
    write(filepath, html)
    return String(filepath)
end

function _enrich_obs_traits_from_tagging!(loaded, params)::Nothing
    (loaded === nothing || !hasproperty(loaded, :obs_df) || !(loaded.obs_df isa DataFrame)) && return nothing
    obs_df = loaded.obs_df
    nrow(obs_df) == 0 && return nothing
    has_cw = hasproperty(obs_df, :cw) &&
             any(v -> !ismissing(v) && v isa Real && isfinite(Float64(v)), obs_df.cw)
    has_cw && return nothing

    tfile = hasproperty(params, :tagging_file) ? params.tagging_file : nothing
    (tfile === nothing || !isfile(tfile)) && return nothing

    try
        ext = lowercase(splitext(tfile)[2])
        ext == ".jld2" || return nothing
        raw = JLD2.load(tfile)
        tdf = raw isa AbstractDict ? (haskey(raw, "tagging") ? raw["tagging"] : first(values(raw))) : raw
        tdf isa DataFrame || return nothing

        cw_map    = Dict{String, Float64}()
        chela_map = Dict{String, Float64}()
        wgt_map   = Dict{String, Float64}()
        z_map     = Dict{String, Float64}()
        cc_map    = Dict{String, String}()
        sex_map   = Dict{String, String}()
        mat_map   = Dict{String, String}()

        for r in eachrow(tdf)
            ismissing(r.tagid) && continue
            tid = string(r.tagid)
            if hasproperty(r, :cw) && !ismissing(r.cw) && isfinite(Float64(r.cw))
                haskey(cw_map, tid) || (cw_map[tid] = Float64(r.cw))
            end
            if hasproperty(r, :chela) && !ismissing(r.chela) && isfinite(Float64(r.chela))
                haskey(chela_map, tid) || (chela_map[tid] = Float64(r.chela))
            end
            if hasproperty(r, :wgt) && !ismissing(r.wgt) && isfinite(Float64(r.wgt))
                haskey(wgt_map, tid) || (wgt_map[tid] = Float64(r.wgt))
            end
            if hasproperty(r, :z) && !ismissing(r.z) && isfinite(Float64(r.z))
                haskey(z_map, tid) || (z_map[tid] = Float64(r.z))
            end
            if hasproperty(r, :cc) && !ismissing(r.cc)
                c = strip(string(r.cc))
                (!isempty(c) && !(lowercase(c) in ("missing", "none")) && !haskey(cc_map, tid)) && (cc_map[tid] = c)
            end
            if hasproperty(r, :sex) && !ismissing(r.sex)
                s = strip(string(r.sex))
                (!isempty(s) && lowercase(s) != "missing" && !haskey(sex_map, tid)) && (sex_map[tid] = s)
            end
            if hasproperty(r, :mat) && !ismissing(r.mat)
                m = strip(string(r.mat))
                (!isempty(m) && lowercase(m) != "missing" && !haskey(mat_map, tid)) && (mat_map[tid] = m)
            end
        end

        tids = string.(obs_df.tagid)
        obs_df[!, :cw]    = Union{Float64, Missing}[get(cw_map, t, missing) for t in tids]
        obs_df[!, :chela] = Union{Float64, Missing}[get(chela_map, t, missing) for t in tids]
        obs_df[!, :wgt]   = Union{Float64, Missing}[get(wgt_map, t, missing) for t in tids]
        obs_df[!, :z]     = Union{Float64, Missing}[get(z_map, t, missing) for t in tids]
        obs_df[!, :cc]    = String[get(cc_map, t, "unknown") for t in tids]
        if hasproperty(obs_df, :sex)
            obs_df[!, :sex] = String[get(sex_map, t, "unknown") for t in tids]
        end
        if hasproperty(obs_df, :mat)
            obs_df[!, :mat] = String[get(mat_map, t, "unknown") for t in tids]
        end
    catch
    end
    return nothing
end

function _ensure_region_map(loaded, params)
    (loaded === nothing || !hasproperty(loaded, :mesh)) && return loaded
    rfiles = hasproperty(params, :region_polygon_files) ? params.region_polygon_files : String[]
    isempty(rfiles) && return loaded
    has_rmap = hasproperty(loaded, :region_map) && loaded.region_map !== nothing &&
               length(loaded.region_map) == loaded.n_spatial && maximum(loaded.region_map; init = 0) >= 2
    has_rmap && return loaded

    cents = hasproperty(loaded.mesh, :centroids_lonlat) ? loaded.mesh.centroids_lonlat : nothing
    cents === nothing && return loaded
    labels = (hasproperty(params, :region_labels) && !isempty(params.region_labels)) ?
        String.(params.region_labels) : ["Region $i" for i in eachindex(rfiles)]
    polys = load_region_polygons(rfiles, labels)
    rmap = region_map_from_polygons(cents, polys)
    maximum(rmap; init = 0) >= 1 || return loaded
    return merge(loaded, (region_map = rmap, region_labels = labels))
end

function _step_years(params, loaded = nothing)::Float64
    if loaded !== nothing && hasproperty(loaded, :data) && loaded.data !== nothing &&
       hasproperty(loaded.data, :step_years) && isfinite(loaded.data.step_years) &&
       loaded.data.step_years > 0
        return Float64(loaded.data.step_years)
    elseif loaded !== nothing && hasproperty(loaded, :obs_df) &&
           loaded.obs_df isa DataFrame && hasproperty(loaded.obs_df, :k) &&
           nrow(loaded.obs_df) > 0 && maximum(loaded.obs_df.k) > 120
        # Daily mark-recapture step count (k in days)
        return 1.0 / 365.25
    elseif hasproperty(params, :fpt_step_years) && params.fpt_step_years !== nothing &&
           isfinite(params.fpt_step_years) && params.fpt_step_years > 0
        return Float64(params.fpt_step_years)
    elseif hasproperty(params, :time_interval)
        ti = params.time_interval
        return ti == :weekly ? (1.0 / 52.0) : (ti == :daily ? (1.0 / 365.25) : (1.0 / 12.0))
    end
    return 1.0 / 12.0
end

function export_dashboards(
    loaded, kernels, path_results, diagnostics, params,
    validation = nothing,
    agent_trajectories = nothing,
    agent_space_use = nothing
)::Union{NamedTuple, Nothing}
    out_dir = _resolve_output_dir(params.output_dir)
    mkpath(out_dir)
    verbose = params.verbose
    verbose && println("\n[Phase 6] Exporting path analyses and consolidated dashboard to: $out_dir")

    _enrich_obs_traits_from_tagging!(loaded, params)
    step_yr = _step_years(params, loaded)

    paths_rich = try
        _build_paths_rich(path_results, loaded; step_years = step_yr)
    catch e
        _record_panel_skip("Build paths_rich", e)
        NamedTuple[]
    end

    mov_stats = nothing
    phenology = nothing
    trait_models = nothing
    written_files = String[]

    if !isempty(paths_rich)
        try
            mov_stats = compute_movement_statistics(paths_rich, path_results, loaded; params = params)
            csv_path = joinpath(out_dir, "movement_path_metrics.csv")
            export_movement_summary_csv(csv_path, paths_rich, mov_stats, loaded.obs_df)
            push!(written_files, csv_path)
            verbose && println("  Path metrics CSV: $csv_path")
        catch e
            _record_panel_skip("Path metrics CSV", e)
        end
        try
            if mov_stats !== nothing
                phenology = analyze_seasonal_movement_phenology(loaded.obs_df, mov_stats, loaded)
            end
        catch e
            _record_panel_skip("Seasonal phenology", e)
        end
        try
            if mov_stats !== nothing
                trait_models = model_trait_movement_associations(loaded.obs_df, mov_stats, loaded)
            end
        catch e
            _record_panel_skip("Trait-movement associations", e)
        end
    end

    if agent_space_use !== nothing
        try
            su_csv = joinpath(out_dir, "movement_agent_space_use.csv")
            su_df = agent_space_use isa DataFrame ? agent_space_use : DataFrame(agent_space_use)
            CSV.write(su_csv, su_df)
            push!(written_files, su_csv)
            verbose && println("  Agent space-use CSV: $su_csv")
        catch e
            _record_panel_skip("Agent space-use CSV", e)
        end
    end

    if params.render_html
        try
            ag_paths = NamedTuple[]
            n_spatial = loaded.n_spatial
            hsi_field = hasproperty(loaded, :hsi_vec) && length(loaded.hsi_vec) >= n_spatial ?
                loaded.hsi_vec : fill(0.5, n_spatial)
            if agent_trajectories isa DataFrame && hasproperty(agent_trajectories, :tagid) &&
               hasproperty(agent_trajectories, :mesh_unit) &&
               hasproperty(loaded.mesh, :centroids_lonlat)
                cents_ll = loaded.mesh.centroids_lonlat
                for (ai, sub) in enumerate(groupby(agent_trajectories, :tagid))
                    ai > 80 && break
                    nodes = Int[Int(u) for u in sub.mesh_unit if 1 <= Int(u) <= length(cents_ll)]
                    length(nodes) < 2 && continue
                    coords = Tuple{Float64, Float64}[(Float64(cents_ll[u][1]), Float64(cents_ll[u][2])) for u in nodes]
                    tot = sum(haversine_distance(coords[k-1][1], coords[k-1][2], coords[k][1], coords[k][2]) / 1000.0 for k in 2:length(coords))
                    disp = haversine_distance(coords[1][1], coords[1][2], coords[end][1], coords[end][2]) / 1000.0
                    push!(ag_paths, (
                        tagid = string(first(sub.tagid)),
                        coords = coords,
                        total_dist_km = tot,
                        displacement_km = disp,
                        speed_km_per_year = disp / max(length(nodes) * step_yr, 1e-4),
                        mean_hsi = mean(hsi_field[u] for u in nodes)
                    ))
                end
            end

            bathy_mig = try
                analyze_bathymetric_migration(loaded.obs_df, loaded)
            catch e
                _record_panel_skip("Bathymetric migration", e)
                nothing
            end
            disp_diag = try
                evaluate_dispersal_scaling_and_null(loaded, kernels; step_years = step_yr)
            catch e
                _record_panel_skip("Dispersal scaling & null model", e)
                nothing
            end
            bio_strat = try
                analyze_biological_state_stratification(loaded.obs_df, loaded; step_years = step_yr)
            catch e
                _record_panel_skip("Biological-state stratification", e)
                nothing
            end
            spatial_roses = try
                sr = compute_spatial_bearing_roses(loaded.obs_df, loaded; grid_km = 100.0, min_events = 8)
                r_csv = export_spatial_bearing_roses_csv(sr, out_dir)
                push!(written_files, r_csv)
                sr
            catch e
                _record_panel_skip("Spatial 100 km bearing roses", e)
                nothing
            end
            taxis_proj = try
                tp = analyze_gradient_taxis_projections(loaded.obs_df, loaded)
                t_csv = export_gradient_taxis_csv(tp, out_dir)
                push!(written_files, t_csv)
                tp
            catch e
                _record_panel_skip("Gradient taxis projections", e)
                nothing
            end
            if hasproperty(kernels, :datasource_summary) && kernels.datasource_summary !== nothing
                try
                    ds_df = kernels.datasource_summary isa DataFrame ?
                        kernels.datasource_summary : DataFrame(kernels.datasource_summary)
                    if nrow(ds_df) > 0
                        ds_csv = joinpath(out_dir, "acoustic_vs_trap_inference.csv")
                        CSV.write(ds_csv, ds_df)
                        push!(written_files, ds_csv)
                    end
                catch e
                    _record_panel_skip("Acoustic vs trap CSV", e)
                end
            end

            smooth_events = try
                reconstruct_all_event_trajectories(
                    loaded, kernels;
                    step_years   = step_yr,
                    seed         = params.seed,
                    resolution   = params.land_mask_resolution,
                    grid_minutes = params.land_mask_grid_minutes
                )
            catch e
                _record_panel_skip("Smoothed all-event trajectories", e)
                NamedTuple[]
            end

            eff_dispersal = try
                ed = compute_effective_dispersal_area(loaded, kernels; step_years = step_yr)
                ed_csv = export_effective_dispersal_csv(ed, loaded, out_dir)
                push!(written_files, ed_csv)
                ed
            catch e
                _record_panel_skip("1-Yr effective dispersal area", e)
                nothing
            end

            dash_html = joinpath(out_dir, "movement_dashboard.html")
            dash_sidecar = joinpath(out_dir, "movement_dashboard.data.zz")
            _export_consolidated_dashboard_html(
                dash_html, loaded, kernels, path_results, diagnostics, params,
                validation, paths_rich, ag_paths, agent_space_use,
                mov_stats, phenology, trait_models;
                bathy_mig = bathy_mig, disp_diag = disp_diag, bio_strat = bio_strat,
                spatial_roses = spatial_roses, taxis_proj = taxis_proj,
                smooth_events = smooth_events, eff_dispersal = eff_dispersal
            )
            push!(written_files, dash_html)
            isfile(dash_sidecar) && push!(written_files, dash_sidecar)
            if verbose
                cmp_mb = isfile(dash_sidecar) ? round(filesize(dash_sidecar) / 1e6; digits = 2) : 0.0
                println("  Consolidated multi-tab dashboard: $dash_html (sidecar: $dash_sidecar, $(cmp_mb) MB compressed)")
            end
        catch e
            _record_panel_skip("Consolidated HTML dashboard", e)
            verbose && println("  (Dashboard export skipped: $(_error_note(e)))")
        end
    end

    return (
        movement_stats = mov_stats,
        phenology = phenology,
        trait_models = trait_models,
        files = written_files
    )
end


# =============================================================================
# Phase 5c: Validation Mark-Recapture Analyses
# =============================================================================

"""
    execute_validation_analyses(loaded, fitted, kernels, params) ->
        Union{NamedTuple, Nothing}

Executes validation post-processing analyses for mark-recapture telemetry:
1. Spatial corridor entropy and canalization (`compute_corridor_canalization`)
2. 1-year HSI-weighted regional stock connectivity (`compute_stock_connectivity_matrix`)
3. Posterior predictive checks (`posterior_predictive_check`)
4. Optional full Bayesian ensemble trajectory and corridor propagation
   (`reconstruct_paths_bayesian_ensemble`)
5. First-passage connectivity assessment (`compute_connectivity_assessment`
   under diagnostic `:connectivity`), skipped loudly when fewer than two
   regions resolve
6. Multistate recapture separation of survival x detection x movement
   (`run_recapture_diagnostics` under diagnostic `:recapture`), skipped
   loudly when the histories cannot identify the joint model
"""
function execute_validation_analyses(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params
)::Union{NamedTuple, Nothing}
    # The ensemble, the connectivity assessment and the recapture separation
    # are their own diagnostics listed separately in `diagnostics`, so none of
    # them is gated on `:validation` as well. Gating them together with
    # `validation` meant a request like `--diagnostics=connectivity` on its
    # own returned here and did nothing, without a word of output.
    wants_validation  = wants_diagnostic(params, :validation)
    wants_ensemble    = wants_diagnostic(params, :bayesian_ensemble)
    wants_conn        = wants_diagnostic(params, :connectivity)
    wants_recap       = wants_diagnostic(params, :recapture)
    (wants_validation || wants_ensemble || wants_conn || wants_recap) ||
        return nothing
    verbose = params.verbose
    out_dir = _resolve_output_dir(params.output_dir)
    mkpath(out_dir)

    pa = wants_validation ?
        (verbose && println("\n[Phase 5c] Running Validation Mark-Recapture Analyses...");
         run_validation_analyses(loaded, fitted, kernels, params, out_dir)) :
        nothing

    conn_res = if wants_conn
        verbose && println(
            "\n[Phase 5e] First-passage connectivity assessment..."
        )
        try
            compute_connectivity_assessment(loaded, kernels, params;
                output_dir = out_dir)
        catch e
            _record_panel_skip("Connectivity assessment", e)
            verbose && println(
                "  (Connectivity assessment skipped: $(_error_note(e)))"
            )
            nothing
        end
    else
        nothing
    end

    recap_res = if wants_recap
        verbose && println(
            "\n[Phase 5f] Multistate recapture separation " *
            "(survival x detection x movement)..."
        )
        try
            run_recapture_diagnostics(loaded, kernels, params;
                output_dir = out_dir)
        catch e
            _record_panel_skip("Recapture separation", e)
            verbose && println(
                "  (Recapture separation skipped: $(_error_note(e)))"
            )
            nothing
        end
    else
        nothing
    end

    ensemble_res = if wants_ensemble
        verbose && println(
            "\n[Phase 5d] Running Bayesian ensemble path & corridor propagation..."
        )
        reconstruct_paths_bayesian_ensemble(loaded, fitted, params)
    else
        nothing
    end

    # `pa` is `nothing` when only a non-validation diagnostic was requested,
    # so each field is taken from it only when the validation analyses
    # actually ran.
    return (
        corridor_canalization    = isnothing(pa) ? nothing : pa.corridor_canalization,
        connectivity_matrix      = isnothing(pa) ? nothing : pa.connectivity_matrix,
        connectivity_uncertainty = isnothing(pa) ? nothing : pa.connectivity_uncertainty,
        posterior_predictive     = isnothing(pa) ? nothing : pa.posterior_predictive,
        summary_file             = isnothing(pa) ? nothing : pa.summary_file,
        bayesian_ensemble        = ensemble_res,
        connectivity_assessment  = conn_res,
        recapture_separation     = recap_res,
    )
end

# =============================================================================
# Orchestrator
# =============================================================================

"""
    run_movement_analysis(params = movement_parameters_default()) -> NamedTuple

Executes the complete TethysPath movement analysis pipeline. All six phases
are called in sequence:

1. `load_movement_data`                 -- data ingestion & depth barriers
2. `fit_movement_models`                -- Bayesian MCMC model fitting
3. `extract_transition_kernels`         -- posterior kernel construction
4. `reconstruct_paths_and_diagnostics` -- paths, corridors, bottlenecks
5. `compute_advanced_diagnostics`      -- circuit theory
     6. `export_dashboards`                 -- interactive Leaflet HTML maps

# Arguments
- `params`: Configuration NamedTuple. Start from a parameter preset and
  merge individual overrides:

      # Generic simulation
      run_movement_analysis()

      # Snow crab with custom depth range
      p = merge(movement_parameters_snowcrab(),
                (depth_range = (80.0, 400.0), n_samples = 500))
      run_movement_analysis(p)

# Returns
`NamedTuple` with fields: `data`, `models`, `chains`, `P_kernel`,
`paths`, `corridors`, `stochastic_paths`, `domain_bottlenecks`,
`circuit`, `parameters`, `depth_range`.

With `params.tessellation_only` the run stops after the mesh: the tessellation map
is written and opened, and the return has `nothing` for every fitted quantity plus
`tessellation_only = true` and `tessellation_path`.
"""
_sanitize_zarr_key(k) = replace(strip(string(k)), r"[^A-Za-z0-9._-]+" => "_")

function _zarr_checkpoint_sanitize(x)
    if x === nothing || x isa Bool || x isa Number || x isa AbstractString || x isa Symbol
        return x
    elseif x isa Dates.AbstractTime
        return string(x)
    elseif x isa Function || x isa Type || x isa Module
        return string(x)
    elseif x isa Turing.DynamicPPL.Model
        return "DynamicPPL.Model($(nameof(x.f)))"
    elseif x isa SparseMatrixCSC
        return Matrix(x)
    elseif x isa AbstractRange
        return collect(x)
    elseif x isa AbstractArray{<:Real} || x isa AbstractArray{Bool} ||
           x isa AbstractArray{Symbol} || x isa AbstractArray{<:AbstractString}
        return x
    elseif x isa AbstractVector && !isempty(x) &&
           all(v -> v isa Tuple && all(e -> e isa Real, v), x)
        k = length(first(x))
        mat = Matrix{Float64}(undef, length(x), k)
        for (i, pt) in enumerate(x)
            for j in 1:k
                mat[i, j] = Float64(pt[j])
            end
        end
        return mat
    elseif x isa AbstractVector && !isempty(x) &&
           all(v -> v isa AbstractVector && all(e -> e isa Tuple && length(e) >= 2 &&
                                                e[1] isa Real && e[2] isa Real, v), x)
        indptr = Vector{Int64}(undef, length(x) + 1)
        indptr[1] = 1
        xs = Float64[]
        ys = Float64[]
        for (i, ring) in enumerate(x)
            for pt in ring
                push!(xs, Float64(pt[1]))
                push!(ys, Float64(pt[2]))
            end
            indptr[i + 1] = length(xs) + 1
        end
        return (indptr = indptr, x = xs, y = ys)
    elseif x isa AbstractVector && !isempty(x) &&
           all(v -> v isa AbstractVector && eltype(v) <: Real, x)
        return x
    elseif x isa AbstractVector
        return Dict{String, Any}(string(i) => _zarr_checkpoint_sanitize(v) for (i, v) in enumerate(x))
    elseif x isa Tuple
        return Dict{String, Any}(string(i) => _zarr_checkpoint_sanitize(v) for (i, v) in enumerate(x))
    elseif x isa AbstractSet
        return [_zarr_checkpoint_sanitize(v) for v in x]
    elseif x isa DataFrame
        out = DataFrame()
        for col in names(x)
            v = x[!, col]
            S = nonmissingtype(eltype(v))
            if v isa AbstractArray{Missing} || S <: Union{Real, Bool, AbstractString, Symbol}
                out[!, col] = v
            else
                out[!, col] = [ismissing(e) ? missing : string(e) for e in v]
            end
        end
        return out
    elseif x isa AbstractDict
        return Dict{String, Any}(
            _sanitize_zarr_key(k) => _zarr_checkpoint_sanitize(v) for (k, v) in pairs(x)
        )
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(_zarr_checkpoint_sanitize, values(x)))
    elseif hasmethod(keys, Tuple{typeof(x)}) && hasmethod(Array, Tuple{typeof(x)}) &&
           occursin("Chain", string(typeof(x)))
        # MCMCChains / FlexiChains: extract parameter draws as a Dict{String, Vector{Float64}}
        d = Dict{String, Any}()
        for k in keys(x)
            try
                d[_sanitize_zarr_key(k)] = vec(Float64.(Array(x[k])))
            catch
            end
        end
        return d
    elseif fieldcount(typeof(x)) > 0
        fnames = fieldnames(typeof(x))
        return NamedTuple{fnames}(Tuple(_zarr_checkpoint_sanitize(getfield(x, f)) for f in fnames))
    else
        return string(x)
    end
end

function _restore_zarr_centroids(c)
    if c isa AbstractMatrix && size(c, 2) >= 2
        return [(Float64(c[i, 1]), Float64(c[i, 2])) for i in 1:size(c, 1)]
    end
    return c
end

function _restore_zarr_polygons(p)
    if p isa NamedTuple && haskey(p, :indptr) && haskey(p, :x) && haskey(p, :y)
        n = length(p.indptr) - 1
        return [
            [(Float64(p.x[k]), Float64(p.y[k])) for k in p.indptr[i]:(p.indptr[i + 1] - 1)]
            for i in 1:n
        ]
    end
    return p
end

function _restore_zarr_mesh(m)
    m isa NamedTuple || return m
    ks = keys(m)
    vs = map(ks) do k
        v = m[k]
        if k in (:centroids, :centroids_km, :centroids_lonlat)
            _restore_zarr_centroids(v)
        elseif k in (:polygons, :polygons_km, :polygons_lonlat)
            _restore_zarr_polygons(v)
        elseif k in (:W, :W_raw) && v isa AbstractMatrix
            sparse(Float64.(v))
        else
            v
        end
    end
    return NamedTuple{ks}(vs)
end

function _restore_zarr_month_lookup(ml)
    ml isa AbstractDict || return ml
    out = Dict{Tuple{Int, Int}, Int}()
    for (k, v) in pairs(ml)
        if k isa Tuple{Int, Int}
            out[k] = Int(v)
        else
            nums = [parse(Int, m.match) for m in eachmatch(r"-?\d+", string(k))]
            if length(nums) >= 2
                out[(nums[1], nums[2])] = Int(v)
            end
        end
    end
    return out
end

function _zarr_checkpoint_restore_loaded(loaded)
    loaded isa NamedTuple || return loaded
    ks = keys(loaded)
    vs = map(ks) do k
        v = loaded[k]
        if k === :mesh
            _restore_zarr_mesh(v)
        elseif k === :W && v isa AbstractMatrix
            sparse(Float64.(v))
        elseif k === :month_lookup
            _restore_zarr_month_lookup(v)
        elseif k === :data && v isa NamedTuple
            _zarr_checkpoint_restore_loaded(v)
        elseif k === :parsed_depth_range && v isa AbstractDict &&
               haskey(v, "1") && haskey(v, "2")
            (Float64(v["1"]), Float64(v["2"]))
        else
            v
        end
    end
    return NamedTuple{ks}(vs)
end

function _zarr_checkpoint_restore_fitted(fitted)
    fitted isa NamedTuple || return fitted
    models_r = haskey(fitted, :models) && fitted.models isa AbstractDict ?
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(fitted.models)) :
        get(fitted, :models, Dict{Symbol, Any}())
    chains_r = haskey(fitted, :chains) && fitted.chains isa AbstractDict ?
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(fitted.chains)) :
        get(fitted, :chains, Dict{Symbol, Any}())
    return (models = models_r, chains = chains_r)
end

function run_movement_analysis(
    params = movement_parameters_default()
)::NamedTuple

    out_dir = _resolve_output_dir(params.output_dir)
    mkpath(out_dir)

    # Panel failures are collected per run, so a previous run's dead panels are
    # never reported against this one.
    empty!(PANEL_SKIPS)

    results_checkpoint = joinpath(out_dir, "movement_results_checkpoint.zarr")

    # --figures-only: load the full results checkpoint and re-render dashboards
    # without re-running any computation.
    if params.figures_only
        ispath(results_checkpoint) || error(
            "[figures-only] Full results checkpoint not found: $results_checkpoint\n" *
            "Run the pipeline at least once without --figures-only to create it."
        )
        params.verbose && println(
            "\n[figures-only] Loading results checkpoint: $results_checkpoint"
        )
        rc = Tethys.read_data(results_checkpoint)
        loaded_r             = _zarr_checkpoint_restore_loaded(rc["loaded"])
        _enrich_obs_traits_from_tagging!(loaded_r, params)
        loaded_r             = _ensure_region_map(loaded_r, params)
        kernels_r            = rc["kernels"]
        path_res_r           = rc["path_res"]
        diagnostics_r        = rc["diagnostics"]
        validation_r         = rc["validation"]
        agent_trajectories_r = rc["agent_trajectories"]
        agent_space_use_r    = get(rc, "agent_space_use", nothing)

        # Force render_html on so figures are actually written.
        params_render = params isa TethysPathConfig ?
            apply_overrides(params, Dict{Symbol, Any}(:render_html => true)) :
            merge(params, (render_html = true,))

        # Refresh circuit diagnostics if checkpoint predates interior-only gamma-calibrated circuit
        has_new_circ = diagnostics_r !== nothing && hasproperty(diagnostics_r, :circuit) &&
                       diagnostics_r.circuit !== nothing &&
                       hasproperty(diagnostics_r.circuit, :interior_only) &&
                       Bool(diagnostics_r.circuit.interior_only)
        if !has_new_circ && wants_diagnostic(params_render, :circuit)
            params_render.verbose && println("\n[figures-only] Refreshing interior-only gamma-calibrated circuit flow...")
            diagnostics_r = compute_advanced_diagnostics(
                loaded_r, path_res_r, params_render; kernels = kernels_r
            )
        end

        params_render.verbose && println("\n[figures-only] Regenerating dashboards...")
        dash_r = export_dashboards(
            loaded_r, kernels_r, path_res_r, diagnostics_r,
            params_render, validation_r, agent_trajectories_r,
            agent_space_use_r
        )
        try
            step_yr = _step_years(params_render, loaded_r)
            tb_stats = compute_tethysbio_movement_stats(
                loaded_r, kernels_r, path_res_r;
                step_years = step_yr,
                species = params_render.species_name
            )
            write_tethysbio_movement_stats(tb_stats; store_dir = out_dir)
        catch e
            _record_panel_skip("TethysBio movement stats export", e)
        end
        report_panel_skips()
        params_render.verbose && println("\n[figures-only] Done.")

        return (
            data               = loaded_r.data,
            models             = nothing,
            chains             = nothing,
            P_kernel           = kernels_r.P_kernel,
            paths              = path_res_r.paths,
            corridors          = path_res_r.corridors,
            stochastic_paths   = path_res_r.stochastic_paths,
            domain_bottlenecks = path_res_r.domain_bottlenecks,
            circuit            = diagnostics_r.circuit,
            validation_analyses = validation_r,
            agent_trajectories = agent_trajectories_r,
            movement_stats     = nothing,
            phenology          = nothing,
            trait_models       = nothing,
            parameters         = (
                alpha     = kernels_r.alpha_hat,
                residence = kernels_r.rho_hat,
                gamma     = kernels_r.gamma_hat,
            ),
            depth_range        = loaded_r.parsed_depth_range,
        )
    end

    checkpoint_file = joinpath(out_dir, "movement_checkpoint.zarr")
    resume_from_checkpoint = params.resume_from_checkpoint

    loaded, fitted, kernels = if resume_from_checkpoint && ispath(checkpoint_file)
        if params.verbose
            println("\n[Checkpoint] Resuming from existing checkpoint: $checkpoint_file")
        end
        data = Tethys.read_data(checkpoint_file)

        loaded_r = _zarr_checkpoint_restore_loaded(data["loaded"])
        _enrich_obs_traits_from_tagging!(loaded_r, params)
        loaded_r = _ensure_region_map(loaded_r, params)

        # --tessellation-only is honoured here too; see the note on the load branch.
        if params.tessellation_only
            return _tessellation_only_result(
                loaded_r, tessellation_preview(loaded_r, params), params,
            )
        end

        (loaded_r, _zarr_checkpoint_restore_fitted(data["fitted"]), data["kernels"])
    else
        loaded_  = load_movement_data(params)

        # --tessellation-only: the mesh exists, so show it and stop. Placed here
        # rather than after `kernels` so nothing downstream of fitting runs -- the
        # checkpoint below, path reconstruction, and the diagnostics all need a
        # fitted kernel, and Phase 3 errors outright when no chain was produced.
        #
        # Honoured on the resume branch too (above). Leaving it ignored whenever a
        # checkpoint happened to exist would make the flag mean different things on
        # different runs.
        params.tessellation_only && return _tessellation_only_result(
            loaded_, tessellation_preview(loaded_, params), params,
        )

        fitted_  = fit_movement_models(loaded_, params)
        kernels_ = extract_transition_kernels(loaded_, fitted_, params)
        
        if params.verbose
            println("\n[Checkpoint] Saving intermediate states to: $checkpoint_file")
        end
        Tethys.write_data(
            checkpoint_file,
            _zarr_checkpoint_sanitize(
                Dict("loaded" => loaded_, "fitted" => fitted_, "kernels" => kernels_)
            );
            backend = :zarr,
            overwrite = true
        )
        
        (loaded_, fitted_, kernels_)
    end
    
    agent_trajectories = nothing
    agent_space_use = nothing
    if :agent in params.model_modes
        if params.verbose
            println("\n[Phase 2b] Projecting synthetic agents forward...")
        end
        n_sim_agents = params.n_agent_projections
        # Use observed release sites to start agents (column is :release, not :release_unit)
        release_nodes = Int.(loaded.obs_df.release)
        maximum(release_nodes) <= size(kernels.P_kernel, 1) || throw(BoundsError(
            kernels.P_kernel,
            "release unit $(maximum(release_nodes)) exceeds the fitted kernel",
        ))
        # The projection horizon is a modelling choice: how far to carry an untagged
        # animal forward. Telemetry does not identify it. The `k` column is a gap
        # ratio, delta_t over the nominal step, and it is 1 for every regularly
        # sampled interval -- so inheriting it blindly collapses every agent to a
        # single step and destroys the space-use question the projection exists
        # to answer. It is used only where it genuinely varies.
        durations = if hasproperty(loaded.obs_df, :k) && !isnothing(loaded.obs_df.k)
            k = max.(1, round.(Int, collect(loaded.obs_df.k)))
            (length(unique(k)) > 1 && any(>(1), k)) ? k : fill(params.agent_horizon, length(release_nodes))
        else
            fill(params.agent_horizon, length(release_nodes))
        end

        # _resolve_centroids returns (planar_km, lonlat, mesh_drawing). Headings
        # want one coordinate list in one space, so pick planar km when it exists
        # and fall back to lon/lat.
        agent_planar, agent_lonlat, _ =
            _resolve_centroids(loaded.mesh, size(kernels.P_kernel, 1))
        agent_centroids = agent_planar === nothing ? agent_lonlat : agent_planar

        # Dynamic vs static kernel selection for agent forward simulation.
        # Build 12 monthly sparse kernels and index each step by its true calendar month
        # rather than allocating thousands of dense matrices at 1 month per day.
        step_yr_ag = _step_years(params, loaded)
        P_agent_kernel = if params.dynamic_kernels &&
                            hasproperty(loaded, :monthly_hsi) &&
                            size(loaded.monthly_hsi, 1) == loaded.n_spatial &&
                            size(loaded.monthly_hsi, 2) >= 12
            max_horizon = maximum(durations; init = params.agent_horizon)
            n_cols_mh = size(loaded.monthly_hsi, 2)
            P_12 = Vector{SparseMatrixCSC{Float64, Int}}(undef, 12)
            for m in 1:12
                m_cols = collect(m:12:n_cols_mh)
                h_clim = vec(mean(@view(loaded.monthly_hsi[:, m_cols]); dims = 2))
                P_12[m] = sparse(construct_stochastic_transition_kernel(
                    loaded.W, h_clim;
                    gamma     = kernels.gamma_hat,
                    residence = kernels.rho_hat,
                    advection = kernels.alpha_hat,
                    land_mask = loaded.land_mask
                ))
            end
            start_mo = params.temperature_month > 0 ? params.temperature_month : 7
            P_seq = Vector{SparseMatrixCSC{Float64, Int}}(undef, max_horizon)
            for t in 1:max_horizon
                P_seq[t] = P_12[mod1(start_mo + floor(Int, (t - 1) * step_yr_ag * 12.0), 12)]
            end
            P_seq
        elseif params.dynamic_kernels && params.use_geodata_habitat &&
               agent_lonlat !== nothing
            max_horizon = maximum(durations; init = params.agent_horizon)
            min_lon = minimum(c[1] for c in agent_lonlat)
            max_lon = maximum(c[1] for c in agent_lonlat)
            min_lat = minimum(c[2] for c in agent_lonlat)
            max_lat = maximum(c[2] for c in agent_lonlat)
            mesh_polys = (params.spatial_averaging != "centroid" &&
                          hasproperty(loaded.mesh, :polygons_lonlat)) ?
                         loaded.mesh.polygons_lonlat : nothing
            hsi_12 = build_dynamic_geodata_hsi_series(
                agent_lonlat;
                polygons           = mesh_polys,
                lon_range          = (min_lon, max_lon),
                lat_range          = (min_lat, max_lat),
                months             = collect(1:12),
                temperature_source = params.temperature_source,
                verbose            = false
            )
            P_12 = Vector{SparseMatrixCSC{Float64, Int}}(undef, 12)
            for m in 1:12
                P_12[m] = sparse(construct_stochastic_transition_kernel(
                    loaded.W, @view(hsi_12[:, m]);
                    gamma     = kernels.gamma_hat,
                    residence = kernels.rho_hat,
                    advection = kernels.alpha_hat,
                    land_mask = loaded.land_mask
                ))
            end
            start_mo = params.temperature_month > 0 ? params.temperature_month : 7
            P_seq = Vector{SparseMatrixCSC{Float64, Int}}(undef, max_horizon)
            for t in 1:max_horizon
                P_seq[t] = P_12[mod1(start_mo + floor(Int, (t - 1) * step_yr_ag * 12.0), 12)]
            end
            P_seq
        else
            sparse(kernels.P_kernel)
        end

        agent_trajectories = forward_project_agents(
            release_nodes, durations;
            n_agents = n_sim_agents,
            transition_kernel = P_agent_kernel,
            centroids = agent_centroids,
            persistence = params.persistence,
            seed = params.seed,
        )
        # Expected space use under the projected kernel: how likely an untagged
        # animal is to reach each unit, and how long it stays there once it does.
        agent_space_use = forward_space_use(agent_trajectories, loaded.n_spatial)
        if params.verbose
            println("  Projected $(n_sim_agents) agents over $(params.agent_horizon) steps.")
            top = findmax(agent_space_use.visit_probability)
            println("  Highest projected use: unit $(top[2]) (p = $(round(top[1]; digits = 3)))")
        end
    end
    
    path_res    = reconstruct_paths_and_diagnostics(loaded, kernels, params)
    diagnostics = compute_advanced_diagnostics(loaded, path_res, params; kernels = kernels)
    validation  = execute_validation_analyses(loaded, fitted, kernels, params)
    dashboards  = export_dashboards(
        loaded, kernels, path_res, diagnostics, params, validation, agent_trajectories,
    agent_space_use
    )

    # Write a full results checkpoint so --figures-only can regenerate dashboards
    # without re-running any computation.  Written after every successful run so
    # it always reflects the most recent results.
    try
        params.verbose && println(
            "\n[Checkpoint] Saving full results to: $results_checkpoint"
        )
        Tethys.write_data(
            results_checkpoint,
            _zarr_checkpoint_sanitize(Dict(
                "loaded" => loaded,
                "kernels" => kernels,
                "path_res" => path_res,
                "diagnostics" => diagnostics,
                "validation" => validation,
                "agent_trajectories" => agent_trajectories,
                "agent_space_use" => agent_space_use
            ));
            backend = :zarr,
            overwrite = true
        )
    catch e
_record_panel_skip("Results checkpoint write", e); params.verbose && println(
"  (Results checkpoint write skipped: $(_error_note(e)))"
            )
    end

    try
        step_yr = _step_years(params, loaded)
        tb_stats = compute_tethysbio_movement_stats(
            loaded, kernels, path_res;
            step_years = step_yr,
            species = params.species_name
        )
        write_tethysbio_movement_stats(tb_stats; store_dir = out_dir)
    catch e
        _record_panel_skip("TethysBio movement stats export", e)
        params.verbose && println(
            "  (TethysBio movement stats export skipped: $(_error_note(e)))"
        )
    end

    # Reported unconditionally, not under `verbose`. A panel that fails on every
    # run must not be able to do so invisibly.
    report_panel_skips()

    if params.verbose
        println("\n" * "=" ^ 72)
        println("  Pipeline Completed Successfully!")
        println("=" ^ 72)
    end

    return (
        data               = loaded.data,
        models             = fitted.models,
        chains             = fitted.chains,
        P_kernel           = kernels.P_kernel,
        paths              = path_res.paths,
        corridors          = path_res.corridors,
        stochastic_paths   = path_res.stochastic_paths,
        domain_bottlenecks = path_res.domain_bottlenecks,
        circuit            = diagnostics.circuit,
        validation_analyses = validation,
        agent_trajectories = agent_trajectories,
          agent_space_use    = agent_space_use,
        movement_stats     = !isnothing(dashboards) && hasproperty(dashboards, :movement_stats) ?
                             dashboards.movement_stats : nothing,
        phenology          = !isnothing(dashboards) && hasproperty(dashboards, :phenology) ?
                             dashboards.phenology : nothing,
        trait_models       = !isnothing(dashboards) && hasproperty(dashboards, :trait_models) ?
                             dashboards.trait_models : nothing,
        parameters         = (
            alpha     = kernels.alpha_hat,
            residence = kernels.rho_hat,
            gamma     = kernels.gamma_hat,
        ),
depth_range        = loaded.parsed_depth_range,
          # Panels that failed this run, so a caller does not have to read stdout
          # to find out that an output is missing.
          panel_skips       = copy(PANEL_SKIPS),
      )
    end

