# =============================================================================
# Segment: connectivity
# First Passage, Communities, Minimum Cuts & Corridor Usage
#
# Every quantity here is checked against an analytic value on a toy graph,
# so a regression shows up as a wrong *number*, not just a wrong type.
# =============================================================================

include("test_fixtures.jl")

@testset "Connectivity: aggregation, first passage, communities, cuts" begin
    @testset "Region partition resolution" begin
        loaded_stub = (n_spatial = 4,)

        # Explicit argument wins over everything.
        amap, alabels = GeoMotion.resolve_region_partition(
            loaded_stub, (;); region_map = [1, 1, 2, 2])
        @test amap == [1, 1, 2, 2]
        @test alabels == ["Region 1", "Region 2"]

        # params.region_map fills over a region-1 background.
        pmap, plabels = GeoMotion.resolve_region_partition(
            loaded_stub, (; region_map = Dict(2 => 2, 3 => 2, 4 => 2)))
        @test pmap == [1, 2, 2, 2]
        @test length(plabels) == 2

        # Unassigned units (0) become a trailing "Unassigned" region.
        umap, ulabels = GeoMotion.resolve_region_partition(
            loaded_stub, (;); region_map = [0, 1, 1, 2])
        @test umap == [3, 1, 1, 2]
        @test ulabels == ["Region 1", "Region 2", "Unassigned"]

        # Label count mismatch is a loud error.
        @test_throws ArgumentError GeoMotion.resolve_region_partition(
            loaded_stub, (;); region_map = [1, 1, 2, 2],
            region_labels = ["A", "B", "C"])
    end

    @testset "Region transition matrix (mass-weighted coarse grain)" begin
        # Two regions of two units each; 45% into the sibling unit, 5% into
        # each unit of the other region => rows sum to 1.
        P4 = [
            0.45 0.45 0.05 0.05
            0.45 0.45 0.05 0.05
            0.05 0.05 0.45 0.45
            0.05 0.05 0.45 0.45
        ]
        P_reg = GeoMotion.region_transition_matrix(P4, [1, 1, 2, 2])
        # Uniform mass over region 1: P_reg[1,2] = sum over v in region 2 of
        # P[u, v] for a single u (rows are identical) = 0.10.
        @test P_reg[1, 2] ≈ 0.10 atol = 1e-12
        @test vec(sum(P_reg; dims = 2)) ≈ [1.0, 1.0] atol = 1e-12

        @test_throws ArgumentError GeoMotion.region_transition_matrix(
            P4, [1, 1, 2])        # map does not cover all units
        @test_throws DomainError GeoMotion.region_transition_matrix(
            P4, [0, 1, 2, 2])     # zero label
    end

    @testset "Mean first-passage times (exact analytic)" begin
        # Two-state symmetric chain: MFPT from either state to the other is 2.
        P2 = [0.5 0.5; 0.5 0.5]
        M2 = GeoMotion.mean_first_passage_times(P2)
        @test M2[1, 2] ≈ 2.0 atol = 1e-12
        @test M2[2, 1] ≈ 2.0 atol = 1e-12
        @test M2[1, 1] == 0.0 && M2[2, 2] == 0.0

        # Deterministic ring 1 -> 2 -> 3 -> 1.
        Pring = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]
        Mring = GeoMotion.mean_first_passage_times(Pring)
        @test Mring[1, 2] ≈ 1.0
        @test Mring[1, 3] ≈ 2.0
        @test Mring[2, 3] ≈ 1.0
        @test Mring[2, 1] ≈ 2.0
        @test Mring[3, 1] ≈ 1.0
        @test Mring[3, 2] ≈ 2.0

        # Bad kernels are rejected, not silently solved.
        @test_throws DomainError GeoMotion.mean_first_passage_times(
            [0.9 0.9; 0.5 0.5])   # rows sum to 1.8
        @test_throws DomainError GeoMotion.mean_first_passage_times(
            [NaN 1.0; 0.5 0.5])
        @test_throws ArgumentError GeoMotion.mean_first_passage_times(
            ones(3, 2))
    end

    @testset "First-arrival CDFs (absorbing forward propagation)" begin
        # Two-state meta-stable kernel: 10% out-migration per step.
        P2 = [0.9 0.1; 0.1 0.9]
        res = GeoMotion.first_arrival_cdfs(P2; horizon = 20)
        # First arrival at 2 from 1: F(tau) = 1 - 0.9^tau.
        @test res.cdf[1, 2, 1] ≈ 0.1 atol = 1e-12
        @test res.cdf[1, 2, 2] ≈ 1 - 0.9^2 atol = 1e-12
        @test res.cdf[1, 2, 20] ≈ 1 - 0.9^20 atol = 1e-12
        @test res.arrival_prob[2, 1] ≈ 1 - 0.9^20 atol = 1e-12
        # Diagonal is not a first-passage target.
        @test res.arrival_prob[1, 1] == 0.0
        # Median: smallest tau with 1 - 0.9^tau >= 0.5 => tau = 7.
        @test res.median_steps[1, 2] == 7.0
        # Corrupted kernels rejected.
        @test_throws DomainError GeoMotion.first_arrival_cdfs(
            [0.8 0.1; 0.1 0.9]; horizon = 5)
        @test_throws DomainError GeoMotion.first_arrival_cdfs(P2; horizon = 0)
    end

    @testset "Corridor visit rate (mass conservation)" begin
        P4 = [
            0.45 0.45 0.05 0.05
            0.45 0.45 0.05 0.05
            0.05 0.05 0.45 0.45
            0.05 0.05 0.45 0.45
        ]
        corridor = GeoMotion.corridor_visit_rate(P4; horizon = 10)
        # Every step carries unit mass, so expected visits sum to the horizon.
        @test sum(corridor.visits) ≈ 10.0 atol = 1e-9
        @test maximum(corridor.normalized) ≈ 1.0 atol = 1e-12
        @test all(corridor.normalized .>= 0.0)
        @test_throws DomainError GeoMotion.corridor_visit_rate(
            P4; horizon = 0)
    end

    @testset "Connectivity communities (greedy modularity)" begin
        # Two triangles coupled by one weak bridge (0.1): the greedy sweep
        # must split into the two cliques.
        W = zeros(6, 6)
        triangle = (a, b, c) -> begin
            W[a, b] = 1.0; W[b, a] = 1.0
            W[b, c] = 1.0; W[c, b] = 1.0
            W[a, c] = 1.0; W[c, a] = 1.0
        end
        triangle(1, 2, 3)
        triangle(4, 5, 6)
        W[3, 4] = 0.1
        W[4, 3] = 0.1
        comm = GeoMotion.connectivity_communities(W)
        @test comm.labels == [1, 1, 1, 2, 2, 2]
        @test comm.n_communities == 2
        @test comm.modularity > 0.3

        # Adaptive sweep on the two-triangle graph stops at two communities.
        comm_adaptive = GeoMotion.connectivity_communities(W; k = 0)
        @test comm_adaptive.n_communities == 2

        # An almost-fused pair of regions merges adaptively (positive gain).
        Wfused = [0.0 0.9; 0.9 0.0]
        comm_fused = GeoMotion.connectivity_communities(Wfused; k = 0)
        @test comm_fused.n_communities == 1

        # Guards.
        @test_throws DomainError GeoMotion.connectivity_communities(
            zeros(1, 1))
        @test_throws DomainError GeoMotion.connectivity_communities(
            [0.0 0.1; 0.1 0.0]; k = 5)
    end

    @testset "Region minimum cuts (Edmonds-Karp)" begin
        # Directed chain with a single saturated bottleneck edge.
        C = zeros(4, 4)
        C[1, 2] = 3.0
        C[2, 3] = 1.0
        C[3, 4] = 3.0
        cut = GeoMotion.region_min_cut(C, [1], [4])
        @test cut.capacity ≈ 1.0 atol = 1e-9
        @test cut.cut_edges == [(2, 3)]
        @test cut.source_side == [true, true, false, false]

        # Two parallel outlet edges: both must appear in the cut.
        C2 = zeros(4, 4)
        C2[1, 2] = 3.0
        C2[2, 3] = 1.0
        C2[2, 4] = 2.0
        cut2 = GeoMotion.region_min_cut(C2, [1], [3, 4])
        @test cut2.capacity ≈ 3.0 atol = 1e-9
        # The min cut returned is the FIRST saturated bridging edge (1, 2);
        # edges (2, 3) and (2, 4) are an alternative cut of the same value.
        @test cut2.cut_edges == [(1, 2)]

        # Guards: overlapping terminals, NaNs, outside range.
        @test_throws ArgumentError GeoMotion.region_min_cut(
            C2, [1], [1, 4])
        @test_throws DomainError GeoMotion.region_min_cut(
            fill(NaN, 2, 2), [1], [2])
        @test_throws ArgumentError GeoMotion.region_min_cut(
            C2, [0], [2])
    end

    @testset "Connectivity assessment orchestrator" begin
        P4 = [
            0.45 0.45 0.05 0.05
            0.45 0.45 0.05 0.05
            0.05 0.05 0.45 0.45
            0.05 0.05 0.45 0.45
        ]
        loaded_stub = (n_spatial = 4,)
        kernels_stub = (P_kernel = P4,)

        res = GeoMotion.compute_connectivity_assessment(
            loaded_stub, kernels_stub;
            region_map = [1, 1, 2, 2],
            horizon = 20,
            step_years = 1.0,
            communities_k = 2,
            corridor_horizon = 10,
        )
        @test res.P_reg[1, 2] ≈ 0.10 atol = 1e-12
        @test res.mfpt_steps[1, 2] ≈ 10.0 atol = 1e-9
        @test res.mfpt_years ≈ res.mfpt_steps atol = 1e-12
        @test res.arrival_prob[1, 2] ≈ 1 - 0.9^20 atol = 1e-12
        @test res.median_years[1, 2] == 7.0
        @test res.community_labels == [1, 2]
        @test res.weakest_pair !== nothing
        @test res.weakest_pair.capacity ≈ 0.2 atol = 1e-9
        @test res.community_cut !== nothing
        @test res.community_cut.capacity ≈ 0.2 atol = 1e-9
        @test sum(res.corridor_visits) ≈ 10.0 atol = 1e-9

        # Single region is a loud configuration error, not a trivial map.
        @test_throws DomainError GeoMotion.compute_connectivity_assessment(
            loaded_stub, kernels_stub; region_map = [1, 1, 1, 1])

        # Kernel/grid size mismatch is loud.
        @test_throws ArgumentError GeoMotion.compute_connectivity_assessment(
            (n_spatial = 3,), kernels_stub; region_map = [1, 1, 2])

        # CSV export smoke test against the scratch directory.
        out_dir = joinpath(TEST_TMP, "connectivity_csv")
        loaded_with_mesh = (
            n_spatial = 4,
            mesh = (centroids_lonlat =
                [(0.0, 45.0), (1.0, 45.0), (0.0, 46.0), (1.0, 46.0)],),
        )
        res_csv = GeoMotion.compute_connectivity_assessment(
            loaded_with_mesh, kernels_stub;
            region_map = [1, 1, 2, 2],
            horizon = 20,
            step_years = 1.0,
            communities_k = 2,
            corridor_horizon = 10,
            output_dir = out_dir,
        )
        @test isfile(joinpath(out_dir, "connectivity_region_matrix.csv"))
        @test isfile(joinpath(out_dir,
            "connectivity_first_passage_summary.csv"))
        @test isfile(joinpath(out_dir, "connectivity_communities.csv"))
        @test isfile(joinpath(out_dir, "connectivity_corridor_usage.csv"))
        @test isfile(joinpath(out_dir, "connectivity_min_cut.csv"))
    end
end
