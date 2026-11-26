include("test_fixtures.jl")

@testset "Diagnostics, Circuits, Visualisation & Lakehouse" begin
    # Conditioned Markov-Bridge & Viterbi Routing
    W_brg = spzeros(4, 4)
    W_brg[1, 2] = 1.0; W_brg[2, 1] = 1.0
    W_brg[1, 3] = 1.0; W_brg[3, 1] = 1.0
    W_brg[2, 4] = 1.0; W_brg[4, 2] = 1.0
    W_brg[3, 4] = 1.0; W_brg[4, 3] = 1.0
    P_brg = construct_stochastic_transition_kernel(W_brg, [0.5, 0.5, 0.5, 0.5];
        gamma = 1.0, residence = 0.2, advection = 0.5)
    path_brg = predict_bridge_path(P_brg, 1, 4, 4)
    @test length(path_brg) == 5
    @test first(path_brg) == 1 && last(path_brg) == 4
    path_vit = TethysPath.predict_path(P_brg, 1, 4, 4; method = :viterbi)
    @test first(path_vit) == 1 && last(path_vit) == 4

    # Circuit Theory Diagnostics
    W_circ = spzeros(3, 3)
    W_circ[1, 2] = 1.0; W_circ[2, 1] = 1.0
    W_circ[2, 3] = 1.0; W_circ[3, 2] = 1.0
    L, C = build_circuit_laplacian(W_circ)
    @test size(L) == (3, 3)
    @test isapprox(sum(L, dims = 2), zeros(3, 1); atol = 1e-10)
    R_eff = effective_resistance_matrix(L)
    @test R_eff[1, 3] > R_eff[1, 2]

    # Posterior Path Ensemble Panel
    # Interactive visualisations and Leaflet maps were removed from the package
    # (results inspection is documented in docs/TethysPathNotes.md).

    # Trait-Movement Associations OLS (without endpoint efficiency/tortuosity)
    obs_df = DataFrame(tag_id = 1:10, carapace_width = [95.0, 102.0, 108.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0])
    mov_stats = (net_displacement_km = [10.0, 12.0, 15.0, 18.0, 20.0, 24.0, 28.0, 30.0, 33.0, 36.0],
                 speed_km_per_year = [20.0, 24.0, 30.0, 36.0, 40.0, 48.0, 56.0, 60.0, 66.0, 72.0],
                 path_length_km = [14.0, 16.0, 20.0, 25.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0])
    assoc = model_trait_movement_associations(obs_df, mov_stats, (au = nothing,))
    @test haskey(assoc.models, "displacement")
    @test haskey(assoc.models, "speed_km_yr")
    @test !haskey(assoc.models, "tortuosity")
    @test assoc.models["displacement"].r2 > 0.9

    # Failed panels recording
    sk = TethysPath.PANEL_SKIPS
    empty!(sk)
    TethysPath._record_panel_skip("Widget", ErrorException("boom"))
    @test length(sk) == 1
    @test TethysPath.report_panel_skips() == 1
    empty!(sk)

    # Input assimilation
    tmp_assim = joinpath(TEST_TMP, "assimilation_$(UUIDs.uuid4())")
    input_dir = joinpath(tmp_assim, "data")
    store_dir = joinpath(tmp_assim, "store")
    mkpath(input_dir)
    tagging = DataFrame(aid = [1, 2, 3], lon = [-1.0, 0.0, 1.0], lat = [44.0, 45.0, 46.0])
    hsi_assim = (sims = reshape(collect(0.1:0.1:(0.1 * 6 * 2 * 3)), 6, 2, 3), years = [1999, 2000], auids = collect(1:6))
    sppoly = DataFrame(AUID = collect(1:4), centroid_lon = 1.0:4.0)
    JLD2.save(joinpath(input_dir, "tagging.jld2"), "tagging", tagging)
    JLD2.save(joinpath(input_dir, "hsi.jld2"), "sims", hsi_assim.sims, "years", hsi_assim.years, "auids", hsi_assim.auids)
    JLD2.save(joinpath(input_dir, "sppoly.jld2"), "graph", Graphs.SimpleGraph(4), "sppoly", sppoly)

    r_assim = TethysPath.assimilate_movement_inputs(input_dir = input_dir, store_dir = store_dir, verbose = false)
    @test r_assim.written == ["tagging", "hsi", "sppoly"]
    back_assim = TethysPath.read_assimilated_movement_inputs(store_dir = store_dir)
    @test size(back_assim.tagging, 1) == 3

    # TethysBio movement statistics generation & store persistence
    W_tb = sparse([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0])
    loaded_tb = (
        W = W_tb,
        mesh = (centroids_km = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)], n_units = 3),
        land_mask = falses(3)
    )
    kernels_tb = (alpha_hat = 0.4, rho_hat = 0.6, gamma_hat = 2.0, P_kernel = W_tb)
    stats_tb = compute_tethysbio_movement_stats(loaded_tb, kernels_tb; step_years = 1.0 / 12.0)
    @test isapprox(stats_tb.movement_statistics.mean_step_km, 10.0; atol = 1e-10)
    @test isapprox(stats_tb.movement_statistics.effective_speed_km_per_year, 0.4 * 10.0 * 12.0; atol = 1e-10)
    @test isapprox(stats_tb.tethysbio_parameters.movement_softmax_temperature, 0.5; atol = 1e-10)
    @test stats_tb.tethysbio_parameters.use_directed_movement === true
    @test stats_tb.tethysbio_parameters.max_move_m_per_year > 0.0

    toml_path = write_tethysbio_movement_stats(stats_tb; store_dir = store_dir)
    @test isfile(toml_path)
    parsed_toml = TethysPath.TOML.parsefile(toml_path)
    @test parsed_toml["producer"] == "TethysPath"
    @test isapprox(parsed_toml["tethysbio_parameters"]["movement_softmax_temperature"], 0.5; atol = 1e-10)

    # Directed flux, bathymetric migration, dispersal scaling, biological stratification
    F_test = sparse([0.0 5.0 0.0; 1.0 0.0 4.0; 0.0 1.0 0.0])
    cents_ll_test = [(-63.0, 44.0), (-62.5, 44.2), (-62.0, 44.4)]
    flux_res = compute_directed_flux_field(F_test, cents_ll_test; top_k = 10)
    @test flux_res.n_directed == 2
    @test isapprox(flux_res.top_edges.net_flux[1], 4.0; atol = 1e-10)

    obs_diag = DataFrame(
        tagid = ["t1", "t2", "t3", "t4"],
        release = [1, 1, 2, 2],
        recapture = [2, 3, 3, 2],
        k = [30, 120, 200, 60],
        rel_time = [2015.2, 2015.5, 2015.8, 2015.3],
        cw = Union{Float64, Missing}[90.0, 115.0, 130.0, 105.0],
        cc = ["2", "3", "4", "3"]
    )
    loaded_diag = (
        obs_df = obs_diag,
        W = W_tb,
        hsi_vec = [0.4, 0.7, 0.9],
        land_mask = falses(3),
        n_spatial = 3,
        resharded_depths = [80.0, 120.0, 180.0],
        mesh = (centroids_lonlat = cents_ll_test, n_units = 3)
    )
    bathy_res = analyze_bathymetric_migration(obs_diag, loaded_diag)
    @test bathy_res.n_obs == 4
    @test bathy_res.mean_dz_m > 0.0
    @test nrow(bathy_res.by_cc) == 3

    bio_res = analyze_biological_state_stratification(obs_diag, loaded_diag; step_years = 1.0 / 365.25)
    @test nrow(bio_res.by_cc) == 3
    @test nrow(bio_res.by_cw) == 4

    P_diag = construct_stochastic_transition_kernel(W_tb, loaded_diag.hsi_vec;
        gamma = 1.0, residence = 0.7, advection = 0.5, land_mask = falses(3))
    disp_res = evaluate_dispersal_scaling_and_null(
        loaded_diag, (P_kernel = P_diag, rho_hat = 0.7); step_years = 1.0 / 365.25
    )
    @test disp_res.n_obs == 4
    @test isfinite(disp_res.delta_lppd)
    @test nrow(disp_res.bins_df) >= 2

    # Effort-weight domain validation in pure_telemetry_turing_model
    @test_throws DomainError pure_telemetry_turing_model(
        [1, 2], [2, 3], [5, 5], W_tb, [0.4, 0.7, 0.9], falses(3), [1.0, 0.0, 1.0]
    )()

    # Stock connectivity with partial polygon coverage (0 = Unassigned units)
    loaded_conn0 = merge(loaded_diag, (
        region_map = [1, 2, 0],
        region_labels = ["N-ENS", "S-ENS"]
    ))
    kernels_conn0 = (
        P_kernel = P_diag,
        alpha_hat = 0.5, rho_hat = 0.7, gamma_hat = 1.0,
        alpha_samples = [0.45, 0.50, 0.55],
        rho_samples = [0.65, 0.70, 0.75],
        gamma_samples = [0.9, 1.0, 1.1]
    )
    conn0 = TethysPath.compute_stock_connectivity_matrix(loaded_conn0, kernels_conn0)
    @test conn0.n_regions == 3
    @test conn0.horizon_steps >= 300
    @test conn0.region_labels == ["N-ENS", "S-ENS", "Unassigned"]
    @test isapprox(sum(conn0.connectivity_matrix[1, :]), 1.0; atol = 1e-6)
    @test conn0.connectivity_matrix[1, 2] > 0.01
    conn_unc0 = TethysPath.compute_connectivity_credible_intervals(
        loaded_conn0, (models = Dict(), chains = Dict()), kernels_conn0, (;), conn0.region_map
    )
    @test size(conn_unc0.connectivity_mean) == (3, 3)
    @test isapprox(sum(conn_unc0.connectivity_mean[1, :]), 1.0; atol = 1e-6)

    # Spatial corridor entropy & canalization (replaces path_credible_intervals)
    canal_res = compute_corridor_canalization(
        loaded_conn0, (models = Dict(), chains = Dict()), kernels_conn0,
        (verbose = false, max_paths = 20)
    )
    @test length(canal_res.canalization_index) == 3
    @test all(x -> 0.0 <= x <= 1.0, canal_res.step_entropy)
    @test all(x -> 0.0 <= x <= 1.0, canal_res.canalization_index)
    # Homogeneous HSI yields maximum local step entropy (= 1.0) on interior node
    P_hom = sparse(construct_stochastic_transition_kernel(
        W_tb, [0.5, 0.5, 0.5]; gamma = 1.5, residence = 0.4, advection = 0.6, land_mask = falses(3)
    ))
    ent_hom = TethysPath._unit_step_entropy(P_hom, falses(3))
    @test isapprox(ent_hom[2], 1.0; atol = 1e-10)

    # Spatially discretized 100 x 100 km compass bearing roses & Rayleigh test
    obs_roses = DataFrame(
        tagid = ["r$i" for i in 1:12],
        release = fill(1, 12),
        recapture = fill(3, 12),
        k = fill(30, 12),
        rel_time = fill(2018.4, 12),
        cw = Union{Float64, Missing}[100.0 + i for i in 1:12],
        datasource = vcat(fill("otn", 6), fill("bio", 6))
    )
    loaded_roses = merge(loaded_diag, (
        obs_df = obs_roses,
        bottom_temp = [1.5, 2.5, 3.8],
        mesh = (
            centroids_lonlat = cents_ll_test,
            centroids_km = [(0.0, 0.0), (40.0, 22.0), (80.0, 44.0)],
            center_lon = -62.5,
            center_lat = 44.2,
            n_units = 3
        )
    ))
    roses_100 = compute_spatial_bearing_roses(obs_roses, loaded_roses; grid_km = 100.0, min_events = 5)
    @test nrow(roses_100.cells_df) == 1
    @test roses_100.cells[1].mean_vector_length > 0.99
    @test roses_100.cells[1].rayleigh_p < 0.01

    # Bathymetric, thermal, & habitat gradient taxis vector projections
    taxis_res = analyze_gradient_taxis_projections(obs_roses, loaded_roses)
    @test taxis_res.n_obs == 12
    @test nrow(taxis_res.overall) == 1
    @test taxis_res.overall.mean_downslope_cos[1] > 0.9
    @test taxis_res.overall.mean_warmward_cos[1] > 0.9

    # Stratified Acoustic (OTN) vs. Commercial Trap (BIO) inference
    ds_fit = fit_datasource_stratified_kernels(
        loaded_roses,
        TethysPathConfig(; n_samples = 12, n_warmup = 4, verbose = false, seed = 7)
    )
    @test haskey(ds_fit.chains, :telemetry_otn)
    @test haskey(ds_fit.chains, :telemetry_bio)

    # High-resolution marine/lake raster masking, overland edge severing, & shoreline clipping
    test_pts = [
        (-60.843, 45.640),  # Bras d'Or / Cape Breton interior (non-ocean)
        (-60.563, 46.112),  # Central Cape Breton / Bras d'Or channel (non-ocean)
        (-60.000, 44.500),  # Offshore Scotian Shelf (open ocean)
        (-59.500, 44.800),  # Offshore Scotian Shelf (open ocean)
    ]
    lm_hi = land_mask_from_global_mask(test_pts; resolution = "h", grid_minutes = 1.25)
    @test lm_hi[1] === true
    @test lm_hi[2] === true
    @test lm_hi[3] === false
    @test lm_hi[4] === false

    edge_pts = [
        (-61.60, 45.75),  # St. Georges Bay (west of Canso Causeway)
        (-61.15, 45.40),  # Chedabucto Bay open water (south of Isle Madame)
        (-60.75, 45.30),  # Offshore Canso Bank (open ocean)
    ]
    W_land_edge = sparse([1, 2, 2, 3], [2, 1, 3, 2], ones(4), 3, 3)
    n_sev = sever_raster_land_edges!(W_land_edge, edge_pts; resolution = "h", grid_minutes = 1.25)
    @test n_sev >= 2
    @test W_land_edge[1, 2] == 0.0 && W_land_edge[2, 1] == 0.0
    @test W_land_edge[2, 3] == 1.0 && W_land_edge[3, 2] == 1.0

    # Coastal polygon clipping retracts land-overlapping vertices onto open ocean
    coast_cent = [(-61.15, 45.40)]
    coast_poly = [[
        (-61.15, 45.55),  # protrudes northward onto Isle Madame / Cape Breton
        (-61.02, 45.42),
        (-61.05, 45.36),
        (-61.22, 45.37),
        (-61.15, 45.55),
    ]]
    n_clip = clip_polygons_to_ocean!(coast_poly, coast_cent; resolution = "h", grid_minutes = 1.25)
    @test n_clip == 1
    lon_m, lat_m, d_m = TethysPath._basemap_mask_data("h", 1.25)
    lon0_m = first(lon_m)
    dlon_m = (last(lon_m) - first(lon_m)) / (length(lon_m) - 1)
    lat0_m = first(lat_m)
    dlat_m = (last(lat_m) - first(lat_m)) / (length(lat_m) - 1)
    @test all(pt -> !TethysPath._is_raster_non_ocean(
        lon_m, lat_m, d_m, lon0_m, dlon_m, lat0_m, dlat_m, pt[1], pt[2]
    ), coast_poly[1])

    # Map-placed SVG compass bearing roses JSON & absence of bearing-rose table in report HTML
    roses_json_str = TethysPath._serialize_map_roses_json(roses_100)
    @test occursin("\"svg\":", roses_json_str)
    @test occursin("\"lat\":", roses_json_str)
    @test occursin("\"lon\":", roses_json_str)
    rep_html = TethysPath._path_analysis_inner_html(
        kernels_conn0, NamedTuple[], nothing, nothing, nothing;
        spatial_roses = roses_100, taxis_proj = taxis_res
    )
    @test !occursin("Compass Bearing Roses", rep_html)

    # Liberty interval classification, marine Chaikin smoothing, & all-event trajectories
    @test classify_liberty_interval(12.0).bin == 1
    @test classify_liberty_interval(120.0).bin == 2
    @test classify_liberty_interval(500.0).bin == 3
    @test classify_liberty_interval(900.0).bin == 4
    @test classify_liberty_interval(1200.0).bin == 5
    @test classify_liberty_interval(1800.0).bin == 6
    @test_throws DomainError classify_liberty_interval(-1.0)

    sm_open = smooth_marine_polyline(
        [
            (-60.35, 44.25),
            (-60.15, 44.38),
            (-60.08, 44.48),
            (-60.15, 44.38),  # deliberate backtrack/loop cycle
            (-59.95, 44.46),
            (-59.75, 44.40),
        ];
        n_knots = 5, check_raster = true, resolution = "h", grid_minutes = 1.25
    )
    @test length(sm_open) > 5
    @test first(sm_open) == (-60.35, 44.25)
    @test last(sm_open) == (-59.75, 44.40)
    # Loop erasure + centripetal spline ensures strictly monotonic eastward progression here
    @test issorted([pt[1] for pt in sm_open])
    @test all(pt -> !TethysPath._is_raster_non_ocean(
        lon_m, lat_m, d_m, lon0_m, dlon_m, lat0_m, dlat_m, pt[1], pt[2]
    ), sm_open)

    all_evs = reconstruct_all_event_trajectories(
        loaded_diag, (P_kernel = P_diag,);
        step_years = 1.0 / 365.25, seed = 42, n_knots = 5, check_raster = false
    )
    @test length(all_evs) == nrow(obs_diag)
    @test all(ev -> length(ev.coords) >= 2, all_evs)
    sm_json, bins_json = TethysPath._serialize_smooth_events_json(all_evs)
    @test occursin("\"a\":", sm_json)
    @test !occursin("liberty", lowercase(sm_json))
    @test occursin("< 1 month", bins_json)

    # Paired HTML shell + zlib-compressed .data.zz sidecar for GitHub Pages deployment
    tmp_dash = joinpath(tmp_assim, "movement_dashboard.html")
    tmp_zz   = joinpath(tmp_assim, "movement_dashboard.data.zz")
    TethysPath._export_consolidated_dashboard_html(
        tmp_dash, loaded_diag, kernels_conn0,
        (paths = Dict("t1" => [1, 2]), corridors = Dict(), domain_bottlenecks = nothing),
        (circuit = nothing,), TethysPathConfig(; verbose = false),
        nothing, NamedTuple[], NamedTuple[], nothing, nothing, nothing, nothing;
        smooth_events = all_evs
    )
    @test isfile(tmp_dash)
    @test isfile(tmp_zz)
    raw_sidecar = String(TethysPath.CodecZlib.transcode(
        TethysPath.CodecZlib.ZlibDecompressor, read(tmp_zz)
    ))
    @test occursin("\"shared_cells\":", raw_sidecar)
    @test occursin("\"smooth_events\":", raw_sidecar)

    rm(tmp_assim; recursive = true, force = true)
end
