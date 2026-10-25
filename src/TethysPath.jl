"""
    TethysPath

Spatial animal movement, trajectory reconstruction, and ecological connectivity analysis engine, integrated with Tethys.jl.
"""
module TethysPath

using LinearAlgebra
using SparseArrays
using Statistics
using StatsBase
using Distributions
import Distributions: Normal, Categorical, truncated, MvNormal
using Random
using Dates
using DataFrames
using Printf
using CodecZlib
using CSV
using Arrow

using Graphs
using NearestNeighbors
using LibGEOS
using CoordRefSystems
using Unitful

using FFTW

using JLD2
using Turing
using Tethys

# Source modules in logical dependency order
include("config.jl")
include("spatial_sources.jl")
include("spatial_utils.jl")
include("turing_models.jl")
include("persistence.jl")
include("assimilation.jl")
include("movement.jl")
include("connectivity.jl")
include("recapture.jl")
include("circuit.jl")
include("pipeline.jl")
include("agent_movement.jl")

# -----------------------------------------------------------------------------
# Public API Exports
# -----------------------------------------------------------------------------

export
    # Spatial Mesh & Partitioning Utilities
    build_hex_mesh_planar,
    build_hex_mesh_clipped,
    lonlat_to_xy_km,
    xy_km_to_lonlat,
    map_to_units,
    map_point_to_units,
    assign_spatial_units,
    get_polygon_area,
    mask_raster_to_mesh,
    sample_bathymetry_to_mesh,
    generate_synthetic_shelf_bathymetry,
    compute_hydrodynamics,
    extract_flow_interpolator,
    create_planar_projection,
    forward_project,
    inverse_project,

    # Domain & Topographic Spatial Pruning
    prune_mesh,
    prune_mesh_by_boundary,
    prune_mesh_by_landmask,
    prune_mesh_by_depth,
    build_mesh_spatial_index,
    point_in_polygon_libgeos,
    points_in_polygon_batch,
    spatial_weights_matrix,
    polygon_adjacency_matrix,
    polygon_centroids_and_areas,
    hex_vertices_planar,
    bounding_box_filter,
    filter_dataframe_by_domain,
    validate_spatial_dataframe,

    # Probabilistic Turing Models
    GaussianObservationModel,
    fit_state_space_trajectory,
    fit_kernel_markov,

    # Biological Habitat & Kernel Models
    build_habitat_suitability_index,
    build_transition_kernel,
    simulate_trajectories,
    build_dynamic_transition_kernel,
    build_geodata_habitat_suitability,
    build_dynamic_geodata_hsi_series,

    # Least-Cost & Stochastic Routing
    dijkstra_least_cost_path,
    astar_corridor_corridors,
    effective_distance_matrix,

    # Circuit Theory Connectivity
    conductance_matrix,
    solve_current_map,
    commute_distance_matrix,
    resistance_covariance_matrix,

    # Configuration
    TethysPathConfig,
    load_config,
    save_config,
    create_argparse_settings,
    effective_model_modes,
    skipped_modes,
    wants_diagnostic,

    # Assimilation
    movement_store_root,
    movement_store_status,
    assimilate_movement_inputs,
    read_assimilated_movement_inputs,

    # Pipeline Orchestration
    movement_parameters_default,
    load_movement_data,
    load_movement_dataset,
    load_survey_data,

    # Spatial data sources
    resolve_bbox,
    bbox_of,
    land_mask_from_global_mask,
    land_mask_from_polygon_files,
    read_polygon_file,
    read_domain_polygon,
    geometries_in_domain,
    geometry_contains,
    dissolve_geometries,
    tessellation_preview,
    point_in_ring,
    region_map_from_polygons,
    load_region_polygons,
    fit_movement_models,
    extract_transition_kernels,
    reconstruct_paths_and_diagnostics,
    compute_advanced_diagnostics,
    execute_validation_analyses,
    run_movement_analysis,
    PANEL_SKIPS,
    report_panel_skips,
    julia_main,
    TethysPath

include("cli.jl")

function __init__()
    Random.seed!(42)
end

end # module TethysPath
