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
using Tethys: point_in_polygon, polygons_intersect
import TethysBio
using TethysBio: build_monthly_hsi_matrix, load_hsi_jld2

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
    build_hex_mesh,
    build_hex_mesh_clipped,
    lonlat_to_xy_km,
    xy_km_to_lonlat,
    haversine_distance,
    coordinate_space_of,
    bearing_deg,
    map_to_units,
    map_point_to_units,
    assign_spatial_units,
    get_polygon_area,
    project_forward,
    project_inverse,

    # Domain & Topographic Spatial Pruning
    prune_mesh,
    spatial_weights_matrix,

    # Probabilistic Turing Models & Kernels
    pure_telemetry_turing_model,
    build_sparse_transition_kernel,
    kstep_transition_cache,
    movement_alpha_rho,
    local_hsi_advantage,
    residency_from_advantage,
    construct_stochastic_transition_kernel,
    build_geodata_habitat_suitability,
    build_dynamic_geodata_hsi_series,
    model_trait_movement_associations,
    compute_directed_flux_field,
    analyze_bathymetric_migration,
    evaluate_dispersal_scaling_and_null,
    analyze_biological_state_stratification,
    validate_mark_recapture_indices,

    # Agent Movement & Forward Projection
    simulate_agent_trajectories,
    forward_project_agents,
    forward_space_use,

    # Probabilistic Routing & Spatial Diagnostics
    predict_bridge_path,
    compute_spatial_bearing_roses,
    analyze_gradient_taxis_projections,
    compute_corridor_canalization,
    fit_datasource_stratified_kernels,

    # Circuit Theory Connectivity
    build_circuit_laplacian,
    solve_circuit_voltage,
    effective_resistance_matrix,
    current_density_map,

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
    compute_tethysbio_movement_stats,
    write_tethysbio_movement_stats,

    # Pipeline Orchestration
    movement_parameters_default,
    load_movement_data,
    load_movement_dataset,

    # Spatial data sources
    resolve_bbox,
    bbox_of,
    land_mask_from_global_mask,
    sever_raster_land_edges!,
    clip_polygons_to_ocean!,
    mask_enclosed_inland_pockets!,
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
    classify_liberty_interval,
    smooth_marine_polyline,
    reconstruct_all_event_trajectories,
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
