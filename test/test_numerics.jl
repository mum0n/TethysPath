# ==============================================================================
# Numerical Invariants, Coordinate-Space Properties & NaN/Inf Robustness
# ==============================================================================
# Property-style tests for the numerical primitives and regression guards for
# the "degrade loudly" rule (FIGURE_CONVENTIONS.md). Nothing here depends on
# production data except the checkpoint segment near the bottom, which is
# conditional on the file existing.
# ==============================================================================

include("test_fixtures.jl")

@testset "Numerical Invariants & Robustness" begin

    # --------------------------------------------------------------------------
    @testset "Haversine distance properties" begin
        R_equator = 6378137.0  # documented default radius (WGS84 equatorial)

        # Zero-length steps measure zero, whatever the coordinates.
        @test haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0
        @test haversine_distance(45.7, -52.3, 45.7, -52.3) == 0.0

        # Symmetry, including pairs whose shortest arc crosses the antimeridian.
        pts = [(0.0, 0.0), (-60.1, 46.5), (179.9, 44.5), (12.3, -52.7)]
        for p1 in pts, p2 in pts
            d12 = haversine_distance(p1[1], p1[2], p2[1], p2[2])
            d21 = haversine_distance(p2[1], p2[2], p1[1], p1[2])
            @test d12 ≈ d21 rtol = 1e-12
        end

        # One degree of arc is 2πR/360 ≈ 111.32 km at the equator (either axis).
        one_deg_lon = haversine_distance(0.0, 0.0, 1.0, 0.0)
        one_deg_lat = haversine_distance(0.0, 0.0, 0.0, 1.0)
        @test one_deg_lon ≈ one_deg_lat rtol = 1e-12
        @test one_deg_lon ≈ 2π * R_equator / 360.0 rtol = 1e-12
        @test one_deg_lon / 1000.0 ≈ 111.3195 atol = 1e-3

        # Antipodal pairs measure half the circumference.
        @test haversine_distance(0.0, 0.0, 180.0, 0.0) ≈ π * R_equator rtol = 1e-9

        # Documented NaN passthrough (docstring contract, not an error).
        @test isnan(haversine_distance(NaN, 0.0, 1.0, 0.0))
        @test isnan(haversine_distance(0.0, Inf, 1.0, 0.0))
    end

    # --------------------------------------------------------------------------
    @testset "Planar tangent-frame properties (:planar_km)" begin
        center = (-60.1, 46.5)

        # Round trip is exact to numerical precision on the equirectangular frame.
        for (dlo, dla) in ((0.0, 0.0), (2.0, 1.5), (-0.5, -1.0), (5.0, -3.0))
            lon = center[1] + dlo
            lat = center[2] + dla
            x, y = lonlat_to_xy_km(lon, lat;
                center_lon = center[1], center_lat = center[2])
            lon2, lat2 = xy_km_to_lonlat(x, y;
                center_lon = center[1], center_lat = center[2])
            @test lon2 ≈ lon atol = 1e-6
            @test lat2 ≈ lat atol = 1e-6
        end

        # LongitudeContract: 1 degree of longitude scales by cos(lat) in the
        # tangent frame, using R = 6371.0 km (mean radius — see
        # FIGURE_CONVENTIONS.md for the radius-convention note).
        x1deg, _ = lonlat_to_xy_km(center[1] + 1.0, center[2];
            center_lon = center[1], center_lat = center[2])
        @test x1deg ≈ 6371.0 * cosd(46.5) * (π / 180.0) atol = 1e-6

        # Latitude scale is radius-proportional at any latitude.
        _, y1deg = lonlat_to_xy_km(center[1], center[2] + 1.0;
            center_lon = center[1], center_lat = center[2])
        @test y1deg ≈ 6371.0 * (π / 180.0) atol = 1e-6
    end

    # --------------------------------------------------------------------------
    @testset "Coordinate-space classification & bearings" begin
        # Empty input degrades to the conservative planar default.
        @test coordinate_space_of(Tuple{Float64, Float64}[]) === :planar_km

        # A NW-Atlantic-style degree box is recognised as geographic...
        @test coordinate_space_of([(-70.0, 40.0), (-55.0, 47.0)]) === :geographic
        # ...and small metric-scale coordinates are not misread as degrees.
        @test coordinate_space_of([(0.0, 0.0), (3.0, 4.0)]) === :planar_km

        # Cardinal bearings at the reference meridian.
        @test bearing_deg((0.0, 0.0), (0.0, 1.0); coord_space = :geographic) ≈ 0.0
        @test bearing_deg((0.0, 0.0), (1.0, 0.0); coord_space = :geographic) ≈ 90.0
        @test bearing_deg((0.0, 0.0), (0.0, -1.0); coord_space = :geographic) ≈ 180.0
        @test bearing_deg((0.0, 0.0), (-1.0, 0.0); coord_space = :geographic) ≈ 270.0

        # Zero-length steps are NaN by contract (callers preserve heading).
        @test isnan(bearing_deg((5.0, 5.0), (5.0, 5.0); coord_space = :planar_km))

        # heading_bin: NaN maps to bin 1 (documented: callers must resolve the
        # self-transition *before* binning), bins wrap and clamp in range.
        @test TethysPath.heading_bin(0.0, 4) == 1
        @test TethysPath.heading_bin(90.0, 4) == 2
        @test TethysPath.heading_bin(NaN, 4) == 1
        @test TethysPath.heading_bin(359.9999, 4) == 4
    end

    # --------------------------------------------------------------------------
    @testset "Transition-kernel invariants" begin
        W = spzeros(5, 5)
        for i in 1:4
            W[i, i+1] = 1.0
            W[i+1, i] = 1.0
        end
        hsi = [0.2, 0.4, 0.6, 0.8, 1.0]

        # Extreme and boundary parameter values all yield
        # row-stochastic, non-negative kernels.
        for (g, r, a) in ((0.0, 0.1, 0.0), (0.0, 0.9, 1.0), (10.0, 0.5, 0.5),
                          (-10.0, 0.5, 0.5), (1.0, 0.5, 0.5), (1e3, 0.95, 0.99))
            P = build_sparse_transition_kernel(W, hsi, g, r, a, nothing)
            @test all(P .>= 0.0)
            @test all(isapprox.(vec(sum(P; dims = 2)), 1.0; atol = 1e-9))
        end

        # Habitat-coupled per-unit residency (the derived pipeline path):
        # composed through local_hsi_advantage -> residency_from_advantage the
        # kernel stays row-stochastic.
        adv = local_hsi_advantage(hsi, W; form = :log_ratio)
        rho_v = residency_from_advantage(0.2, adv, 0.8, :exp_log_ratio)
        P_c = build_sparse_transition_kernel(W, hsi, 1.0, rho_v, 0.5, nothing)
        @test all(P_c .>= 0.0)
        @test all(isapprox.(vec(sum(P_c; dims = 2)), 1.0; atol = 1e-9))

        # Pipeline-facing dense constructor invariants.
        P_d = construct_stochastic_transition_kernel(
            W, hsi; gamma = 1.0, residence = 0.2, advection = 0.5)
        @test P_d isa Matrix{Float64}
        @test all(P_d .>= 0.0)
        @test all(isapprox.(vec(sum(P_d; dims = 2)), 1.0; atol = 1e-9))

        # Derived (not fitted) parameter mapping stays inside clamped range.
        for v in (0.0, 0.05, 1.0, 5.0), d in (0.0, 0.05, 1.0, 5.0)
            a2, r2 = movement_alpha_rho(v, d)
            @test 0.0 <= a2 <= 1.0
            @test 0.01 <= r2 <= 0.95
        end
        a2, r2 = movement_alpha_rho(1.0, 0.0)
        @test a2 ≈ 1.0 atol = 1e-5   # pure advection when diffusion is zero
        a2, r2 = movement_alpha_rho(0.0, 1.0)
        @test a2 ≈ 0.0 atol = 1e-5   # pure diffusion when velocity is zero

        # NaN HSI at a MARINE unit must degrade loudly (regression guard):
        # previously a NaN slipped the comparisons and silently dropped the
        # whole transition row, leaving the kernel non-stochastic.
        h_nan = copy(hsi)
        h_nan[3] = NaN
        @test_throws DomainError build_sparse_transition_kernel(
            W, h_nan, 1.0, 0.2, 0.5, nothing)
        @test_throws DomainError construct_stochastic_transition_kernel(W, h_nan)

        # NaN HSI at a LAND unit is tolerated: land units short-circuit to a
        # self-loop before the habitat value is ever read.
        h_land = copy(hsi)
        h_land[1] = NaN
        lm = falses(5)
        lm[1] = true
        P_land = construct_stochastic_transition_kernel(
            W, h_land; land_mask = lm)
        @test P_land[1, 1] == 1.0
        @test all(isapprox.(vec(sum(P_land; dims = 2)), 1.0; atol = 1e-9))
    end

    # --------------------------------------------------------------------------
    @testset "A* and circuit NaN/Inf robustness" begin
        coords = [(0.0, 0.0), (1.0, 1.0), (1.0, -1.0), (2.0, 0.0)]
        Wq = spzeros(4, 4)
        Wq[1, 2] = 1.0; Wq[2, 1] = 1.0
        Wq[1, 3] = 1.0; Wq[3, 1] = 1.0
        Wq[2, 4] = 1.0; Wq[4, 2] = 1.0
        Wq[3, 4] = 1.0; Wq[4, 3] = 1.0

        # Baseline: a connected path is found.
        p = astar_least_cost_path(coords, Wq, 1, 4)
        @test first(p) == 1 && last(p) == 4

        # Non-finite resistance surfaces degrade loudly, not "the search silently
        # returns arbitrary paths".
        @test_throws DomainError astar_least_cost_path(
            coords, Wq, 1, 4; hsi = fill(NaN, 4))
        @test_throws DomainError astar_least_cost_path(
            coords, Wq, 1, 4; resistance = [1.0, 2.0, Inf, 1.0])

        # Non-finite adjacency weights degrade loudly.
        W_nan = copy(Wq)
        W_nan[1, 2] = NaN
        @test_throws DomainError astar_least_cost_path(coords, W_nan, 1, 4)
        @test_throws DomainError build_circuit_laplacian(W_nan)

        # Circuit: NaN HSI at a marine unit is rejected; at a land unit it is
        # tolerated because land edges are severed before the value is read.
        @test_throws DomainError build_circuit_laplacian(Wq; hsi = fill(NaN, 4))
        lm_land = [true, false, false, false]
        h_mixed = [NaN, 0.4, 0.6, 0.3]
        L_ok, C_ok = build_circuit_laplacian(Wq; hsi = h_mixed, land_mask = lm_land)
        @test size(L_ok) == (4, 4)
        @test all(isfinite, nonzeros(C_ok))

        # Out-of-range HSI also rejects (existing contract, now NaN-proof).
        @test_throws DomainError build_circuit_laplacian(Wq; hsi = [1.2, 0.4, 0.6, 0.3])
    end

    # --------------------------------------------------------------------------
    @testset "Circuit diagnostics on disconnected graphs" begin
        # Nodes 1-2-3 connected as a chain; node 4 is an isolated island.
        Wd = spzeros(4, 4)
        Wd[1, 2] = 1.0; Wd[2, 1] = 1.0
        Wd[2, 3] = 1.0; Wd[3, 2] = 1.0

        Ld, Cd = build_circuit_laplacian(Wd)
        O = effective_resistance_matrix(Ld; W = Wd)

        # Unit conductances: R(1,2) = 1, R(1,3) = 2 through the chain.
        @test O[1, 2] ≈ 1.0 atol = 1e-6
        @test O[1, 3] ≈ 2.0 atol = 1e-6
        @test O[1, 3] > O[1, 2]

        # Disconnected pairs fall back to the documented sentinel.
        @test O[1, 4] == 1e6 && O[4, 1] == 1e6
        @test O[4, 4] == 0.0
        @test O ≈ transpose(O)

        # Unreachable source: documented loud contract — Inf resistance, zero
        # potentials, zero currents (no silent "finite but garbage" current).
        V, R_eff, E = solve_circuit_voltage(Ld, Cd, 4, 1)
        @test R_eff == Inf
        @test all(iszero, V)
        @test iszero(E)

        # Source == sink is the exact zero-resistance case.
        V2, R2, _ = solve_circuit_voltage(Ld, Cd, 2, 2)
        @test R2 == 0.0
        @test all(iszero, V2)

        # Kirchhoff's current law at the intermediate node: net outflow −
        # inflow vanishes, and point-to-point resistance matches (1,3) = 2.
        V3, R3, E3 = solve_circuit_voltage(Ld, Cd, 1, 3)
        E3d = Array(E3)
        @test (sum(E3d[2, :]) - sum(E3d[:, 2])) ≈ 0.0 atol = 1e-9
        @test R3 ≈ 2.0 atol = 1e-6
        @test V3[1] ≈ 2.0 atol = 1e-6
        @test V3[2] ≈ 1.0 atol = 1e-6
        @test V3[3] == 0.0

        # current_density_map silently skips pairs broken by disconnection:
        # an island event must not fabricate current anywhere.
        J_valid, I_tot, _, _ = current_density_map(Wd, [1], [3])
        @test J_valid[1] ≈ 1.0 atol = 1e-9   # source carries I_total
        @test J_valid[3] ≈ 1.0 atol = 1e-9   # sink carries I_total
        @test J_valid[2] ≈ 1.0 atol = 1e-9   # through-flux at the midpoint
        @test J_valid[4] == 0.0

        J_island, _, _, _ = current_density_map(Wd, [4], [3])
        @test all(iszero, J_island)

        # Malformed event tables are rejected loudly.
        @test_throws DimensionMismatch current_density_map(Wd, [1], [3, 4])
    end

    # --------------------------------------------------------------------------
    @testset "Notebook field-mapping contract" begin
        # Regression guard for the notebook's field mappings against the
        # checkpoint schema written by src/pipeline.jl. The textual contract
        # entries must stay in lock-step with notebooks/TethysPathNotes.jl;
        # the execution checks below run only when the checkpoint exists.
        nb_path = joinpath(@__DIR__, "..", "notebooks", "TethysPathNotes.jl")
        notebook_checks = [
            ("loaded.mesh.n_units", "mesh size read via loaded.mesh, not loaded"),
            ("count(!iszero,", "sparse/dense kernel active counting"),
            ("collect(values(path_res.paths))", "Dict paths sampling"),
            ("bottleneck_mask", "domain bottleneck counting"),
            ("pinch_mask", "circuit pinch metrics"),
            ("posterior_predictive", "validation summary entry point"),
            ("MovementAnalysis = TethysPath", "legacy checkpoint load alias"),
        ]
        if isfile(nb_path)
            nb_text = read(nb_path, String)
            for (needle, why) in notebook_checks
                @test occursin(needle, nb_text) || @info(
                    "notebook contract entry moved: $needle ($why)"
                )
            end
        end

        # Checkpoint-conditional invariants: kernel rows stay stochastic and
        # non-negative in the stored artifact itself.
        cpath = joinpath(@__DIR__, "..", "output", "movement_results_checkpoint.jld2")
        if isfile(cpath)
            required_keys = ("loaded", "kernels", "path_res", "diagnostics",
                "validation", "agent_trajectories")
            optional_keys = ("agent_space_use",)  # added by the current writer;
                                                  # absent in the pre-Phase-4b file
            K, top_keys, path_res, diagnostics = JLD2.jldopen(cpath) do f
                f["kernels"], keys(f), f["path_res"], f["diagnostics"]
            end
            for key in required_keys
                @test key in top_keys
            end
            for key in optional_keys
                if !(key in top_keys)
                    @info "checkpoint predates $key; regenerate to add it"
                else
                    @test true
                end
            end

            P = K isa AbstractDict ? K["P_kernel"] : K.P_kernel
            @test all(isapprox.(vec(sum(P; dims = 2)), 1.0; atol = 1e-6))
            @test minimum(P) >= 0.0

            # path_res.paths is a Dict{String, Vector{Int}} of at least the
            # observed two-cell paths (the notebook samples its values).
            @test !isempty(path_res.paths)
            for tpl in values(path_res.paths)
                @test length(tpl) >= 2
            end

            # Circuit diagnostics carry the renamed fields the notebook and
            # tests both consume. `propertynames` is the right probe here:
            # JLD2 reconstructs NamedTuples as wrappers whose *struct field*
            # is :fields, while propertynames reports the logical names.
            circ_names = propertynames(diagnostics.circuit)
            for fname in (:current_density, :pinch_mask, :pinch_score,
                          :threshold, :stochastic, :robust_mask, :n_robust)
                @test fname in circ_names
            end
        end
    end

end  # outer testset
