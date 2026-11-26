# =============================================================================
# Spatial data sources: domain extent, land classification, and region polygons.
#
# Nothing here knows about any particular study area. The domain comes from an
# explicit bounding box or, failing that, from the extent of the input data; land
# comes from a global land/sea mask or from bathymetry; regions of interest come
# from polygon files the user supplies. Species- and region-specific knowledge
# lives in the TOML configuration, not in this file.
# =============================================================================

using CSV
using DataFrames
using GeoDatasets
using GeoInterface
using GeoJSON
using JLD2
using JSON
using LibGEOS

# =============================================================================
# Bounding box
# =============================================================================

"""
    resolve_bbox(
        configured, padding_deg, lons, lats;
        sppoly_bounds = nothing
    ) -> NTuple{4, Float64}

Determine the analysis domain as `(west, south, east, north)` in degrees.

A configured `bbox` is authoritative and is only validated. When it is absent the
extent is derived from the supplied coordinates -- the telemetry positions and, if
available, the spatial-unit polygons -- and padded so the mesh is not clipped
flush against the outermost detections.

When `sppoly_bounds` is provided as `(min_lon, min_lat, max_lon, max_lat)`, the
extent becomes the **union** of the padded data extent and those bounds. Union,
not clamp: the spatial-unit file declares the study domain, so a part of it that
reaches further out than the telemetry has to be meshed. Taking `max` on the west
and south edges instead pulled the extent inside the footprint and left the
footprint's own outer slivers with no cells, which is not recoverable downstream.
"""
function resolve_bbox(
    configured::Union{Nothing, AbstractVector{<:Real}},
    padding_deg::Real,
    lons::AbstractVector{<:Real},
    lats::AbstractVector{<:Real};
    sppoly_bounds::Union{Nothing, NTuple{4, Float64}} = nothing
)::NTuple{4, Float64}

    if configured !== nothing
        length(configured) == 4 || throw(ArgumentError(
            "bbox must be [west, south, east, north]; got $(length(configured)) value(s)"
        ))
        w, s, e, n = (Float64(x) for x in configured)
        all(isfinite, (w, s, e, n)) || throw(ArgumentError(
            "bbox must be finite; got ($w, $s, $e, $n)"
        ))
        (w < e && s < n) || throw(ArgumentError(
            "bbox must satisfy west < east and south < north; got ($w, $s, $e, $n)"
        ))
        return (w, s, e, n)
    end

    isempty(lons) && throw(ArgumentError(
        "Cannot derive a bounding box: no bbox configured and no coordinates available."
    ))

    w = minimum(lons) - padding_deg
    e = maximum(lons) + padding_deg
    s = minimum(lats) - padding_deg
    n = maximum(lats) + padding_deg

    if sppoly_bounds !== nothing
        w = min(w, sppoly_bounds[1])
        s = min(s, sppoly_bounds[2])
        e = max(e, sppoly_bounds[3])
        n = max(n, sppoly_bounds[4])
    end

    # Clamp to the valid geographic range; padding must not run off the globe.
    return (max(w, -180.0), max(s, -90.0), min(e, 180.0), min(n, 90.0))
end

"""
    extract_sppoly_bounds(path) -> Union{Nothing, NTuple{4, Float64}}

Extract spatial domain bounding coordinates `(min_lon, min_lat, max_lon, max_lat)`
from an `sppoly` JLD2 or table file. Looks for `:centroid_lon`/`:centroid_lat`,
`:lon`/`:lat`, or reads polygon rings if present.
"""
function extract_sppoly_bounds(
    path::Union{Nothing, AbstractString}
)::Union{Nothing, NTuple{4, Float64}}
    isnothing(path) && return nothing
    isfile(path) || return nothing
    try
        f = JLD2.jldopen(path, "r")
        obj = if haskey(f, "sppoly")
            f["sppoly"]
        elseif haskey(f, "au")
            f["au"]
        else
            first(values(f))
        end
        close(f)

        if obj isa DataFrame
            lons = if hasproperty(obj, :centroid_lon)
                obj.centroid_lon
            elseif hasproperty(obj, :lon)
                obj.lon
            else
                nothing
            end
            lats = if hasproperty(obj, :centroid_lat)
                obj.centroid_lat
            elseif hasproperty(obj, :lat)
                obj.lat
            else
                nothing
            end
            if lons !== nothing && lats !== nothing && !isempty(lons)
                return (
                    Float64(minimum(lons)), Float64(minimum(lats)),
                    Float64(maximum(lons)), Float64(maximum(lats))
                )
            end
        elseif hasproperty(obj, :lon) && hasproperty(obj, :lat)
            return (
                Float64(minimum(obj.lon)), Float64(minimum(obj.lat)),
                Float64(maximum(obj.lon)), Float64(maximum(obj.lat))
            )
        end
    catch
    end

    # Fallback to general polygon reading
    try
        rings = read_polygon_file(path)
        all_pts = Tuple{Float64, Float64}[]
        for ring in rings
            append!(all_pts, ring)
        end
        return bbox_of(all_pts)
    catch
        return nothing
    end
end

"""
    bbox_of(points) -> Union{Nothing, NTuple{4, Float64}}

Smallest `(west, south, east, north)` containing the given `(lon, lat)` points,
or `nothing` when there are none.
"""
function bbox_of(points)::Union{Nothing, NTuple{4, Float64}}
    isempty(points) && return nothing
    lons = [Float64(p[1]) for p in points]
    lats = [Float64(p[2]) for p in points]
    return (minimum(lons), minimum(lats), maximum(lons), maximum(lats))
end

# =============================================================================
# Land classification
# =============================================================================

"""
    _basemap_grid_token(grid_minutes) -> String

Render a grid spacing the way matplotlib's basemap names its mask files, so
`5.0` becomes `5` and `10.0` becomes `10`. Whole-minute grids carry no decimal
part in the filename; the fractional grids keep their point (`1.25`, `2.5`).
"""
function _basemap_grid_token(grid_minutes::Real)
    g = Float64(grid_minutes)
    g > 0 || throw(ArgumentError("land mask grid spacing must be positive, got $g"))
    return isinteger(g) ? string(Int(g)) : string(g)
end

const _BASEMAP_MASK_URL = "https://raw.githubusercontent.com/matplotlib/basemap/" *
                          "v1.2.2rel/lib/mpl_toolkits/basemap/data"

# Resolutions basemap ships, and the grids each one is published at.
const _BASEMAP_RESOLUTIONS = Set(['c', 'l', 'i', 'h', 'f'])
const _BASEMAP_GRIDS = (1.25, 2.5, 5.0, 10.0)

# Parsed masks, keyed by (resolution, grid token), so a run pays the gunzip once.
const _LAND_MASK_CACHE = Dict{Tuple{Char,String},Tuple{Vector{Float64},Vector{Float64},Matrix{UInt8}}}()

"""
    _basemap_mask_data(resolution, grid_minutes) -> (lon, lat, data)

Load a basemap land/sea mask as a gzipped, headerless, row-major `UInt8` raster
where 0 is ocean, 1 is land and 2 is lake.

This reads the file directly rather than going through
`GeoDatasets.landseamask`, for two reasons: that function derives its download URL
by interpolating the grid verbatim, so a `Float64` `5.0` requests
`lsmask_5.0min_l.bin` while the published file is `lsmask_5min_l.bin` -- the
mismatched request 404s, and the 14-byte error page it leaves behind then fails
to decompress with a `ZlibError`. The same call also rejects a grid passed as a
string, so the filename cannot be corrected through its API. The indexing below
reproduces GeoDatasets' `nearest_point` arithmetic exactly, so classification is
unchanged.
"""
function _basemap_mask_data(resolution::AbstractString, grid_minutes::Real)
    res = resolution[1]
    res in _BASEMAP_RESOLUTIONS || throw(ArgumentError(
        "land_mask_resolution must be one of c, l, i, h, f; got \"$resolution\""
    ))
    grid = Float64(grid_minutes)
    any(g -> isapprox(g, grid; atol = 1e-9), _BASEMAP_GRIDS) || throw(ArgumentError(
        "land_mask_grid_minutes must be one of 1.25, 2.5, 5.0, 10.0; got $grid"
    ))

    token = _basemap_grid_token(grid)
    key = (res, token)
    haskey(_LAND_MASK_CACHE, key) && return _LAND_MASK_CACHE[key]

    file = joinpath(_basemap_data_dir(), "lsmask_$(token)min_$(res).bin")
    raw = _read_gzip_bytes(file, token, res)

    arcsec = round(Int, grid * 60)
    nlon = length(-180 * 3600 + arcsec / 2:arcsec:180 * 3600 - arcsec / 2)
    nlat = length(-90 * 3600 + arcsec / 2:arcsec:90 * 3600 - arcsec / 2)
    length(raw) == nlon * nlat || throw(ArgumentError(
        "land mask $(basename(file)) holds $(length(raw)) bytes but a " *
        "$(nlon)x$(nlat) raster needs $(nlon * nlat)"
    ))

    lon = (-180 * 3600 + arcsec / 2:arcsec:180 * 3600 - arcsec / 2) ./ 3600
    lat = (-90 * 3600 + arcsec / 2:arcsec:90 * 3600 - arcsec / 2) ./ 3600
    data = reshape(raw, length(lon), length(lat))

    _LAND_MASK_CACHE[key] = (collect(lon), collect(lat), data)
    return _LAND_MASK_CACHE[key]
end

"Directory GeoDatasets keeps its bundled basemap data in."
function _basemap_data_dir()
    dir = joinpath(dirname(dirname(pathof(GeoDatasets))), "data")
    return dir
end

"""
    _read_gzip_bytes(file, token, resolution) -> Vector{UInt8}

Return the decompressed contents of `file`, downloading it into place if absent.

A previous failed request can leave a short error page where the mask belongs, so
a file that is implausibly small is treated as absent and fetched again.
"""
function _read_gzip_bytes(file::AbstractString, token::AbstractString, resolution::Char)
    if !isfile(file) || filesize(file) <= 1024
        url = "$_BASEMAP_MASK_URL/$(basename(file))"
        @info "Fetching global land/sea mask" resolution token url
        mkpath(dirname(file))
        download(url, file)
    end
    return open(GzipDecompressorStream, file) do io
        read(io)
    end
end

function _is_raster_non_ocean(
    lon::Vector{Float64},
    lat::Vector{Float64},
    data::Matrix{UInt8},
    lon0::Float64,
    dlon::Float64,
    lat0::Float64,
    dlat::Float64,
    x::Float64,
    y::Float64
)::Bool
    i = mod1(1 + round(Int, (x - lon0) / dlon), length(lon))
    j = min(length(lat), max(1, 1 + round(Int, (y - lat0) / dlat)))
    # In GMT/basemap lsmask: 0 = ocean; 1 = land, 2 = lake, 3 = island_in_lake, 4 = pond.
    return data[i, j] != 0x00
end

"""
    land_mask_from_global_mask(centroids_lonlat; polygons, resolution, grid_minutes) -> BitVector

Classify each unit as non-marine (`true`) using the GMT/basemap land/sea/lake mask
where `0` is ocean and `>= 1` denotes terrestrial land or inland lakes. When `polygons`
are supplied, units whose half-radius interior points overlap land/lakes at >= 2
vertices are also masked so coastal hexagons do not sit inland.
"""
function land_mask_from_global_mask(
    centroids_lonlat::AbstractVector;
    polygons::Union{Nothing, AbstractVector} = nothing,
    resolution::AbstractString = "l",
    grid_minutes::Real = 5.0
)::BitVector
    lon, lat, data = _basemap_mask_data(resolution, grid_minutes)
    lon0, dlon = first(lon), (last(lon) - first(lon)) / (length(lon) - 1)
    lat0, dlat = first(lat), (last(lat) - first(lat)) / (length(lat) - 1)
    n = length(centroids_lonlat)
    mask = falses(n)
    has_polys = polygons !== nothing && length(polygons) == n

    for k in 1:n
        cx, cy = Float64(centroids_lonlat[k][1]), Float64(centroids_lonlat[k][2])
        if _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat, cx, cy)
            mask[k] = true
            continue
        end
        if has_polys && length(polygons[k]) >= 3
            ring = polygons[k]
            n_v = length(ring) - (ring[1] == ring[end] ? 1 : 0)
            n_hit = 0
            for vi in 1:n_v
                vx, vy = Float64(ring[vi][1]), Float64(ring[vi][2])
                mx = 0.5 * (cx + vx)
                my = 0.5 * (cy + vy)
                if _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat, mx, my)
                    n_hit += 1
                    if n_hit >= 2
                        mask[k] = true
                        break
                    end
                end
            end
        end
    end
    return mask
end

"""
    sever_raster_land_edges!(W, centroids_lonlat; resolution="h", grid_minutes=1.25, n_substeps=49)

Sever symmetric adjacency edges `(u, v)` in `W` whose straight-line geographic chord
intersects non-ocean pixels (`data != 0x00`, covering both terrestrial land and inland
lakes) on the GMT/basemap raster:

    W_marine[u, v] = 0  if exists s in [0.02, 0.98] : mask((1 - s)*c_u + s*c_v) != 0

Inputs:
- `W`: Square `S x S` spatial adjacency matrix (mutated in place).
- `centroids_lonlat`: Length-`S` vector of `(lon, lat)` centroids in decimal degrees.
- `resolution`: GMT/basemap coastline tier (`"c"`, `"l"`, `"i"`, `"h"`, or `"f"`).
- `grid_minutes`: Raster spacing in arc-minutes (`1.25`, `2.5`, `5.0`, or `10.0`).
- `n_substeps`: Number of interior chord checkpoints (`>= 3`).

Outputs:
Integer count of directed matrix entries zeroed (`2` per undirected edge severed).
"""
function sever_raster_land_edges!(
    W::AbstractMatrix{<:Real},
    centroids_lonlat::AbstractVector;
    resolution::AbstractString = "h",
    grid_minutes::Real = 1.25,
    n_substeps::Int = 49
)::Int
    S = size(W, 1)
    length(centroids_lonlat) == S || throw(DimensionMismatch(
        "centroids_lonlat length ($(length(centroids_lonlat))) != W size ($S)."
    ))
    lon, lat, data = _basemap_mask_data(resolution, grid_minutes)
    lon0, dlon = first(lon), (last(lon) - first(lon)) / (length(lon) - 1)
    lat0, dlat = first(lat), (last(lat) - first(lat)) / (length(lat) - 1)
    fracs = range(0.02, 0.98; length = max(3, n_substeps))

    severed = 0
    if W isa SparseMatrixCSC
        rows = rowvals(W)
        for u in 1:S
            x1, y1 = Float64(centroids_lonlat[u][1]), Float64(centroids_lonlat[u][2])
            for ptr in nzrange(W, u)
                v = rows[ptr]
                v <= u && continue
                x2, y2 = Float64(centroids_lonlat[v][1]), Float64(centroids_lonlat[v][2])
                dx = x2 - x1
                dy = y2 - y1
                hit = false
                for s in fracs
                    if _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat,
                                            x1 + s * dx, y1 + s * dy)
                        hit = true
                        break
                    end
                end
                if hit
                    W[u, v] = 0.0
                    W[v, u] = 0.0
                    severed += 2
                end
            end
        end
        severed > 0 && dropzeros!(W)
    else
        for u in 1:S
            x1, y1 = Float64(centroids_lonlat[u][1]), Float64(centroids_lonlat[u][2])
            for v in (u + 1):S
                (W[u, v] == 0.0 && W[v, u] == 0.0) && continue
                x2, y2 = Float64(centroids_lonlat[v][1]), Float64(centroids_lonlat[v][2])
                dx = x2 - x1
                dy = y2 - y1
                for s in fracs
                    if _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat,
                                            x1 + s * dx, y1 + s * dy)
                        W[u, v] = 0.0
                        W[v, u] = 0.0
                        severed += 2
                        break
                    end
                end
            end
        end
    end
    return severed
end

"""
    clip_polygons_to_ocean!(polygons_lonlat, centroids_lonlat; land_mask=nothing, resolution="h", grid_minutes=1.25)

Retract coastal polygon vertices radially toward their marine centroid `(c_x, c_y)`
whenever a ray from centroid to vertex `(v_x, v_y)` enters terrestrial land or an
inland lake on the global raster (`s_safe = 0.92 * s_last_ocean`).

Inputs:
- `polygons_lonlat`: Vector of polygon vertex rings `[(lon, lat), ...]` in degrees (mutated in place).
- `centroids_lonlat`: Vector of unit centroids `(lon, lat)` in decimal degrees.
- `land_mask`: Optional boolean vector marking non-marine units (`true` skipped).
- `resolution`: GMT/basemap coastline tier (`"h"` default).
- `grid_minutes`: Raster spacing in arc-minutes (`1.25` default).

Outputs:
Integer count of coastal marine polygons whose vertex rings were modified.
"""
function clip_polygons_to_ocean!(
    polygons_lonlat::AbstractVector,
    centroids_lonlat::AbstractVector;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    resolution::AbstractString = "h",
    grid_minutes::Real = 1.25
)::Int
    n = min(length(polygons_lonlat), length(centroids_lonlat))
    lon, lat, data = _basemap_mask_data(resolution, grid_minutes)
    lon0, dlon = first(lon), (last(lon) - first(lon)) / (length(lon) - 1)
    lat0, dlat = first(lat), (last(lat) - first(lat)) / (length(lat) - 1)

    n_clipped = 0
    for u in 1:n
        (land_mask !== nothing && u <= length(land_mask) && land_mask[u]) && continue
        ring = polygons_lonlat[u]
        length(ring) < 3 && continue
        cx, cy = Float64(centroids_lonlat[u][1]), Float64(centroids_lonlat[u][2])
        _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat, cx, cy) && continue
        is_closed = ring[1] == ring[end]
        n_v = length(ring) - (is_closed ? 1 : 0)
        new_ring = Vector{Tuple{Float64, Float64}}(undef, length(ring))
        changed = false

        for k in 1:n_v
            vx, vy = Float64(ring[k][1]), Float64(ring[k][2])
            dx = vx - cx
            dy = vy - cy
            last_ocean_s = 0.0
            hit_land = false
            for s in 0.02:0.02:1.00
                if _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat,
                                        cx + s * dx, cy + s * dy)
                    hit_land = true
                    break
                else
                    last_ocean_s = s
                end
            end
            if hit_land
                s_safe = 0.92 * last_ocean_s
                rx = round(cx + s_safe * dx; digits = 5)
                ry = round(cy + s_safe * dy; digits = 5)
                if _is_raster_non_ocean(lon, lat, data, lon0, dlon, lat0, dlat, rx, ry)
                    rx = round(cx; digits = 5)
                    ry = round(cy; digits = 5)
                end
                new_ring[k] = (rx, ry)
                changed = true
            else
                new_ring[k] = (vx, vy)
            end
        end
        if is_closed
            new_ring[end] = new_ring[1]
        end
        if changed
            polygons_lonlat[u] = new_ring
            n_clipped += 1
        end
    end
    return n_clipped
end

"""
    mask_enclosed_inland_pockets!(land_mask, W; min_component_fraction=0.02, min_total_marine=200)

Mark small disconnected or semi-enclosed inland water components (e.g., fjord-lakes or
lagoons isolated after overland edge severing) as non-marine (`true`) in `land_mask`
whenever their size falls below `max(15, round(Int, min_component_fraction * S_max))`,
where `S_max` is the size of the largest connected marine component.

Inputs:
- `land_mask`: Length-`S` boolean mask (`true` = non-marine, mutated in place).
- `W`: Symmetric `S x S` sparse marine adjacency matrix.
- `min_component_fraction`: Minimum component size relative to `S_max` in `(0, 1)`.
- `min_total_marine`: Minimum active marine count required before pocket pruning applies.

Outputs:
Integer count of units newly marked `true` in `land_mask`.
"""
function mask_enclosed_inland_pockets!(
    land_mask::AbstractVector{Bool},
    W::SparseMatrixCSC;
    min_component_fraction::Float64 = 0.02,
    min_total_marine::Int = 200
)::Int
    n = length(land_mask)
    count(!, land_mask) < min_total_marine && return 0

    rows = rowvals(W)
    visited = falses(n)
    comps = Vector{Vector{Int}}()
    for i in 1:n
        (land_mask[i] || visited[i]) && continue
        comp = Int[]
        visited[i] = true
        q = Int[i]
        head = 1
        while head <= length(q)
            u = q[head]
            head += 1
            push!(comp, u)
            for ptr in nzrange(W, u)
                v = rows[ptr]
                (land_mask[v] || visited[v] || W[v, u] <= 0.0) && continue
                visited[v] = true
                push!(q, v)
            end
        end
        push!(comps, comp)
    end
    isempty(comps) && return 0
    max_sz = maximum(length, comps)
    min_keep = max(15, round(Int, min_component_fraction * max_sz))
    n_masked = 0
    for c in comps
        if length(c) < min_keep
            for u in c
                land_mask[u] = true
                n_masked += 1
            end
        end
    end
    return n_masked
end

"""
    land_mask_from_polygon_files(files, centroids_lonlat) -> BitVector

Classify centroids against user-supplied land polygons, one polygon ring per
element of `files` (or every ring found within each file).
"""
function land_mask_from_polygon_files(
    files::AbstractVector{<:AbstractString},
    centroids_lonlat::AbstractVector
)::BitVector
    rings = reduce(vcat, read_polygon_file.(files))
    return BitVector([
        any(r -> point_in_ring(Float64(c[1]), Float64(c[2]), r), rings)
        for c in centroids_lonlat
    ])
end

"""
    point_in_ring(x::Real, y::Real, ring) -> Bool

Even-odd ray casting. `ring` is a closed or unclosed sequence of `(x, y)` vertices.
"""
function point_in_ring(x::Real, y::Real, ring)::Bool
    return Tethys.point_in_polygon(x, y, ring)
end

# =============================================================================
# Region-of-interest polygons
# =============================================================================

"""
    read_polygon_file(path) -> Vector{Vector{NTuple{2, Float64}}}

Read polygon rings from a region or land definition file. Format is chosen by
extension:

  - `.csv`          long format with `lon`/`x` and `lat`/`y` columns, and
                    optional `region`/`id`/`ring` column separating polygons
  - `.jld2`         a `DataFrame` in the same long format, or a vector of rings
  - `.rds`, `.RData`, `.rdata`  read through R; a list of rings or a data frame
                    in the same long format
  - `.shp`          an ESRI shapefile read with `Shapefile`
  - `.geojson`, `.json`  a FeatureCollection, read with `GeoJSON`

Every ring is returned as a vector of `(lon, lat)` vertices. Rings are not closed
for you; the containment test does not require closure.
"""
function read_polygon_file(path::AbstractString)::Vector{Vector{NTuple{2, Float64}}}
    isfile(path) || throw(ArgumentError("Polygon file not found: $path"))
    ext = lowercase(splitext(path)[2])

    if ext in (".csv", ".dat", ".txt")
        return _rings_from_table(CSV.read(path, DataFrame))
    elseif ext == ".jld2"
        return _rings_from_object(JLD2.load(path), path)
    elseif ext in (".rds", ".rdata")
        return _rings_from_object(_read_rdata(path), path)
    elseif ext == ".shp"
        return _rings_from_shapefile(path)
    elseif ext in (".geojson", ".json")
        return _rings_from_geojson(path)
    end

    throw(ArgumentError(
        "Unsupported polygon file '$path'. Supported extensions: " *
        ".csv, .dat, .txt, .jld2, .rds, .RData, .shp, .geojson, .json"
    ))
end

# -- shared table interpretation ---------------------------------------------

const LON_NAMES = (:lon, :long, :longitude, :x)
const LAT_NAMES = (:lat, :latitude, :y)
const GROUP_NAMES = (:region, :region_id, :id, :ring, :polygon, :part, :feature)

# Columns that hold well-known-text geometry. `wkt_planar` is deliberately absent:
# those coordinates are in a projected CRS (UTM km), so testing lon/lat against them
# would answer a different question than the caller asked.
const WKT_COLUMN_NAMES = (:wkt_geo, :wkt_lonlat, :wkt, :geometry, :geom)

# Variable names that identify a spatial-unit table inside a multi-variable file.
# A JLD2 file written with named variables loads as a `Dict`, whose iteration order
# is unspecified, so "the first DataFrame or vector in the file" picks whichever
# variable happens to come first -- for the shipped `data/sppoly.jld2` that is
# `wkt_planar`, a vector of WKT *strings*, which is not a list of rings.
const UNIT_TABLE_NAMES = ("sppoly", "units", "polygons", "features", "regions")

function _coord_column(df::DataFrame, candidates, what::AbstractString)
    syms = Symbol.(propertynames(df))
    for c in candidates
        c in syms && return c
    end
    throw(ArgumentError(
        "Polygon table is missing a $what column; looked for " *
        join(candidates, ", ") * " among " * join(Symbol.(propertynames(df)), ", ")
    ))
end

"""
    _rings_from_table(df) -> Vector{Vector{NTuple{2,Float64}}}

Interpret a long-format polygon table: one vertex per row, with optional group
columns separating one polygon from the next. When no group column is present the
whole table is treated as a single ring.
"""
function _rings_from_table(df::DataFrame)::Vector{Vector{NTuple{2, Float64}}}
    nrow(df) == 0 && return Vector{Vector{NTuple{2, Float64}}}()
    loncol = _coord_column(df, LON_NAMES, "longitude")
    latcol = _coord_column(df, LAT_NAMES, "latitude")

    syms = Symbol.(propertynames(df))
    groupcol = findfirst(c -> c in GROUP_NAMES, syms)
    isnothing(groupcol) && return [Vector{NTuple{2, Float64}}([
        (Float64(df[i, loncol]), Float64(df[i, latcol])) for i in 1:nrow(df)
    ])]

    rings = Vector{Vector{NTuple{2, Float64}}}()
    sub = df[!, groupcol]
    for key in unique(sub)
        idx = findall(isequal(key), sub)
        # Preserve file order within each group so ring vertices stay sequential.
        sort!(idx)
        ring = Vector{NTuple{2, Float64}}()
        for i in idx
            push!(ring, (Float64(df[i, loncol]), Float64(df[i, latcol])))
        end
        push!(rings, ring)
    end
    return rings
end

"""
    parse_wkt_geometry(wkt) -> LibGEOS.AbstractGeometry

Parse one WKT geometry with LibGEOS/GEOS.

GEOS already does WKT parsing, ring ordering, and validity checking, and it raises
on malformed input. Re-parsing the text here would buy nothing and would get
`POLYGON` vs `MULTIPOLYGON` vs `GEOMETRYCOLLECTION` nesting wrong.
"""
parse_wkt_geometry(wkt::AbstractString) = LibGEOS.readgeom(String(wkt))

"""
    _ring_points(ring) -> Vector{Tuple{Float64, Float64}}

Coordinates of one ring as `(lon, lat)` pairs.

`GeoInterface` represents a ring either as a list of `(x, y)` points or as a flat
interleaved coordinate vector, depending on the backend, so both are accepted
rather than assuming one.
"""
function _ring_points(ring)
    isempty(ring) && return Tuple{Float64, Float64}[]
    if first(ring) isa Real
        return [(Float64(ring[2i - 1]), Float64(ring[2i])) for i in 1:(length(ring) ÷ 2)]
    end
    return [(Float64(p[1]), Float64(p[2])) for p in ring]
end

"""
    geometry_rings(geom) -> Vector{Vector{NTuple{2, Float64}}}

Flatten a (Multi)Polygon or GeometryCollection into its rings, in lon/lat order,
via `GeoInterface`. Interior rings are returned alongside exteriors: a caller that
wants a union does not care about them, and one that wants holes can have them.
"""
function geometry_rings(geom)
    rings = Vector{Vector{NTuple{2, Float64}}}()
    _collect_rings!(rings, geom)
    return rings
end

function _collect_rings!(rings, geom)
    trait = GeoInterface.geomtrait(geom)
    if trait isa GeoInterface.PolygonTrait
        for ring in GeoInterface.coordinates(geom)
            pts = _ring_points(ring)
            # A closed ring repeats its first point last, so 3 distinct vertices
            # is the floor for a polygon.
            length(pts) >= 4 && push!(rings, pts)
        end
    elseif trait isa Union{GeoInterface.MultiPolygonTrait, GeoInterface.GeometryCollectionTrait}
        for part in GeoInterface.getgeom(geom)
            _collect_rings!(rings, part)
        end
    end
    return rings
end

"""
    dissolve_geometries(geoms) -> LibGEOS.AbstractGeometry

Merge polygon geometries into one, dropping the shared boundaries between
adjacent parts.

Unioning pairwise rather than accumulating left-deep matters: 707 areal units
dissolve in ~0.06 s as a balanced tree, against ~707 full recomputations for a
single accumulator. The result is a single geometry, so a containment test is one
predicate rather than a scan over parts.

# Arguments
- `geoms`: nonempty vector of geometries, e.g. one per areal unit.
"""
function dissolve_geometries(geoms::AbstractVector)
    isempty(geoms) && throw(ArgumentError(
        "dissolve_geometries needs at least one geometry, got none."
    ))
    level = collect(LibGEOS.AbstractGeometry, geoms)
    while length(level) > 1
        nxt = LibGEOS.AbstractGeometry[]
        for i in 1:2:length(level)
            push!(nxt,
                  i + 1 <= length(level) ? LibGEOS.union(level[i], level[i + 1]) : level[i])
        end
        level = nxt
    end
    return level[1]
end

"""
    geometry_contains(geom, lon, lat) -> Bool

True when the point lies inside `geom`. Boundary points count as inside: a mesh
unit centroid sitting exactly on a footprint edge belongs to the domain.
"""
geometry_contains(geom, lon::Real, lat::Real) =
    LibGEOS.covers(geom, LibGEOS.Point(Float64(lon), Float64(lat)))

"""
    geometries_in_domain(geom, centroids_lonlat) -> BitVector

Membership of each `(lon, lat)` centroid in `geom`, or all-`true` when `geom` is
`nothing` (no footprint configured, so no spatial restriction).
"""
function geometries_in_domain(geom, centroids_lonlat)::BitVector
    geom === nothing && return trues(length(centroids_lonlat))
    return BitVector([
        geometry_contains(geom, Float64(c[1]), Float64(c[2])) for c in centroids_lonlat
    ])
end

"""
    _first_present(df, candidates) -> Union{Symbol, Nothing}

First column of `df` whose name appears in `candidates`, or `nothing`.
"""
function _first_present(df::DataFrame, candidates)::Union{Symbol, Nothing}
    syms = Symbol.(propertynames(df))
    for c in candidates
        c in syms && return c
    end
    return nothing
end

# -- JLD2 / RData -------------------------------------------------------------

"""
    polygon_from_rings(rings) -> LibGEOS.AbstractGeometry

Build a polygon from coordinate rings: the first ring is the exterior, any further
rings are holes. The inverse of [`geometry_rings`](@ref) for the shapes a
long-format vertex table produces.

# Notes
Vertex tables are often unclosed, and GEOS rejects a linear ring whose last point
is not its first, so rings are closed here rather than requiring every input file
to pre-close them. Rings with fewer than three distinct vertices are dropped: GEOS
rejects them too, and one stray degenerate row should not fail the whole domain.
"""
function polygon_from_rings(rings)
    closed = Vector{Vector{Float64}}[]
    for ring in rings
        pts = [[Float64(pt[1]), Float64(pt[2])] for pt in ring]
        isempty(pts) && continue
        pts[1] == pts[end] || push!(pts, copy(pts[1]))
        length(pts) >= 4 || continue
        push!(closed, pts)
    end
    isempty(closed) && throw(ArgumentError(
        "polygon_from_rings found no usable ring; a polygon needs at least three " *
        "distinct vertices."
    ))
    return LibGEOS.Polygon(closed)
end

"""
    _geometries_from_unit_table(obj, path) -> Vector{LibGEOS.AbstractGeometry}

Read a spatial-unit table as a list of polygon geometries.

The file shipped as `data/sppoly.jld2` is the reason the WKT branch exists. It is
an areal-unit definition -- 707 units, each with `wkt_geo` in EPSG:4326 -- stored
as a `Dict` of named variables rather than as a ring list. It could not be read at
all before: the reader looked for a vector of rings, found `wkt_planar`, and threw
`Unsupported vector payload`. The caller in `src/pipeline.jl` caught that and
carried on with no domain polygon, which is how the rendered domain became the
whole mesh extent instead of the study footprint.

A table with no geometry column is read as long-format vertices, which
[`_rings_from_table`](@ref) already understands.
"""
function _geometries_from_unit_table(obj, path::AbstractString)
    if obj isa DataFrame
        wktcol = _first_present(obj, WKT_COLUMN_NAMES)
        wktcol === nothing || return [parse_wkt_geometry(string(w)) for w in obj[!, wktcol]]
        return [polygon_from_rings(_rings_from_table(obj))]
    end

    if obj isa AbstractString
        return [parse_wkt_geometry(obj)]
    end

    if obj isa AbstractVector
        isempty(obj) && return LibGEOS.AbstractGeometry[]
        all(v -> v isa AbstractString, obj) &&
            return [parse_wkt_geometry(string(v)) for v in obj]
        throw(ArgumentError(
            "Unsupported vector payload in $path; expected a list of WKT geometries."
        ))
    end

    throw(ArgumentError(
        "Unsupported payload in $path of type $(typeof(obj)); expected a table " *
        "with a WKT geometry column or long-format lon/lat vertices, a WKT " *
        "string, or a list of WKT geometries."
    ))
end

"""
    _unit_table_from_object(obj, path) -> Union{Nothing, Any}

Pick the spatial-unit table out of a loaded file. `nothing` when the payload holds
no table, so the caller can fall back to the long-format vertex reader.
"""
function _unit_table_from_object(obj, path::AbstractString)
    obj isa DataFrame && return obj
    obj isa AbstractDict || return obj
    # A JLD2 file written with named variables loads as a Dict, whose iteration
    # order is unspecified. Prefer a table that is *named* like a spatial-unit
    # table over whatever happens to come first.
    for name in UNIT_TABLE_NAMES
        haskey(obj, name) && return obj[name]
    end
    for v in values(obj)
        v isa DataFrame && return v
    end
    return nothing
end

function _rings_from_unit_table(obj, path::AbstractString)
    if obj isa DataFrame
        wktcol = _first_present(obj, WKT_COLUMN_NAMES)
        wktcol === nothing || return reduce(
            vcat, geometry_rings(parse_wkt_geometry(string(w))) for w in obj[!, wktcol];
            init = Vector{Vector{NTuple{2, Float64}}}(),
        )
        return _rings_from_table(obj)
    end

    if obj isa AbstractString
        return geometry_rings(parse_wkt_geometry(obj))
    end

    if obj isa AbstractVector
        isempty(obj) && return Vector{Vector{NTuple{2, Float64}}}[]
        if all(v -> v isa AbstractString, obj)
            return reduce(
                vcat, geometry_rings(parse_wkt_geometry(string(v))) for v in obj;
                init = Vector{Vector{NTuple{2, Float64}}}(),
            )
        end
        all(v -> v isa Union{Tuple, AbstractVector}, obj) || throw(ArgumentError(
            "Unsupported vector payload in $path; expected a list of polygon " *
            "rings, a list of WKT geometries, or a vector of coordinates."
        ))
        return [Vector{NTuple{2, Float64}}(_as_ring(r)) for r in obj]
    end

    if hasproperty(obj, :polygons)
        throw(ArgumentError(
            "$path exposes unit polygons in a projected CRS, so they cannot be " *
            "tested against lon/lat. Supply the unit footprints as WKT in " *
            "EPSG:4326 (a `wkt_geo` column or field) instead."
        ))
    end

    throw(ArgumentError(
        "Unsupported payload in $path of type $(typeof(obj)); expected a DataFrame " *
        "in long format or with a WKT geometry column, a WKT string, or a list of " *
        "polygon rings."
    ))
end

function _rings_from_object(obj, path::AbstractString)
    table = _unit_table_from_object(obj, path)
    table === nothing && throw(ArgumentError(
        "No DataFrame or polygon list found in $path; expected region " *
        "vertices with longitude/latitude columns"
    ))
    return _rings_from_unit_table(table, path)
end

"""
    read_domain_polygon(path) -> Union{Nothing, LibGEOS.AbstractGeometry}

Read a domain footprint as a single dissolved geometry, or `nothing` when `path`
is absent or unreadable.

Used by the pipeline to decide which mesh units are inside the study area. The
units are dissolved first so that one `covers` call per unit answers "inside any
part", which is both faster and better defined than testing each part: a centroid
in the gap between two adjacent areal units is inside the domain, and GEOS
resolves that boundary case.
"""
function read_domain_polygon(path::Union{Nothing, AbstractString})
    isnothing(path) && return nothing
    isfile(path) || throw(ArgumentError("Domain polygon file does not exist: $path"))

    ext = lowercase(splitext(path)[2])
    obj = if ext == ".jld2"
        JLD2.load(path)
    elseif ext in (".rda", ".rdata", ".rds")
        _read_rdata(path)
    else
        throw(ArgumentError(
            "Unsupported domain polygon format \"$ext\"; expected .jld2, .rda, " *
            ".rdata or .rds."
        ))
    end

    table = _unit_table_from_object(obj, path)
    table === nothing && throw(ArgumentError(
        "No spatial-unit table found in $path; expected a table with WKT " *
        "footprints (a `wkt_geo` column) or long-format lon/lat vertices."
    ))
    geoms = _geometries_from_unit_table(table, path)
    isempty(geoms) && throw(ArgumentError("$path contains no polygon geometry."))
    return dissolve_geometries(geoms)
end

function _as_ring(r)
    out = Vector{NTuple{2, Float64}}()
    isempty(r) && return out
    # A single vertex given as [x, y].
    if all(v -> v isa Real, r) && length(r) >= 2
        push!(r, (Float64(r[1]), Float64(r[2])))
        return out
    end
    for v in r
        (v isa Tuple || v isa AbstractVector) && length(v) >= 2 &&
            push!(out, (Float64(v[1]), Float64(v[2])))
    end
    return out
end

function _read_rdata(path::AbstractString)
    # Implemented by TethysPathRCallExt; the base stub raises a clear error
    # when RCall is not installed.
    return _r_load_object(path)
end

# -- vector formats -----------------------------------------------------------

function _rings_from_shapefile(path::AbstractString)
    handle = Shapefile.Handle(path)
    rings = Vector{NTuple{2, Float64}}[]
    for shp in handle.shapes
        geom = GeoInterface.geometry(shp)
        coords = GeoInterface.coordinates(geom)
        for part in _flatten_coords(coords)
            push!(rings, Vector{NTuple{2, Float64}}(part))
        end
    end
    return rings
end

function _rings_from_geojson(path::AbstractString)
    # `Vector{NTuple{2, Float64}}[]` is an *empty* vector whose elements are rings.
    # The extra nesting here (`Vector{Vector{NTuple{2, Float64}}}[]`) declares rings
    # to be lists of rings, so every `push!` of a single ring then failed to convert.
    # This path had never run: it is reached only by a `.geojson` polygon file.
    return _collect_geojson(JSON.parsefile(path), Vector{NTuple{2, Float64}}[])
end

function _collect_geojson(node, rings)
    node isa AbstractDict || return rings
    t = get(node, "type", "")
    if t == "Polygon"
        # coordinates: [ring, ring, ...], each a list of [lon, lat] positions.
        for ring in get(node, "coordinates", [])
            push!(rings, _flatten_coords(ring))
        end
    elseif t == "MultiPolygon"
        # coordinates: [polygon, ...], each polygon = [ring, ring, ...]
        for polygon in get(node, "coordinates", [])
            for ring in polygon
                push!(rings, _flatten_coords(ring))
            end
        end
    elseif t == "Feature"
        _collect_geojson(get(node, "geometry", nothing), rings)
    elseif t == "FeatureCollection"
        for f in get(node, "features", [])
            _collect_geojson(f, rings)
        end
    elseif t == "GeometryCollection"
        for g in get(node, "geometries", [])
            _collect_geojson(g, rings)
        end
    end
    return rings
end

# Flatten a GeoJSON coordinate nest down to a flat sequence of (x, y) pairs.
function _flatten_coords(c)
    out = Vector{NTuple{2, Float64}}()
    _walk_coords(c, out)
    return out
end

function _walk_coords(c, out)
    if c isa AbstractVector
        if !isempty(c) && all(v -> v isa Real, c) && length(c) >= 2
            push!(out, (Float64(c[1]), Float64(c[2])))
        else
            for v in c
                _walk_coords(v, out)
            end
        end
    end
    return out
end

# =============================================================================
# Region assignment
# =============================================================================

"""
    region_map_from_polygons(centroids_lonlat, region_polygons) -> Vector{Int}

Assign each unit to the first region whose polygons contain its centroid, or `0`
when it falls in none. Index `i` corresponds to `region_polygons[i]`.
"""
function region_map_from_polygons(
    centroids_lonlat::AbstractVector,
    region_polygons::AbstractVector
)::Vector{Int}
    map = zeros(Int, length(centroids_lonlat))
    for (i, c) in enumerate(centroids_lonlat)
        x, y = Float64(c[1]), Float64(c[2])
        for (j, rings) in enumerate(region_polygons)
            any(r -> point_in_ring(x, y, r), rings) && (map[i] = j; break)
        end
    end
    return map
end

"""
    load_region_polygons(files, labels) -> Vector{Vector{Vector{NTuple{2,Float64}}}}

Read one polygon file per region label. The returned vector is parallel to
`labels`, so `result[i]` holds the rings of `labels[i]`.
"""
function load_region_polygons(
    files::AbstractVector{<:AbstractString},
    labels::AbstractVector{<:AbstractString}
)::Vector{Vector{Vector{NTuple{2, Float64}}}}
    length(files) == length(labels) || throw(ArgumentError(
        "region_polygon_files has $(length(files)) entries but region_labels has $(length(labels)); supply exactly one polygon file per region."
    ))
    return [read_polygon_file(f) for f in files]
end