# =============================================================================
# Segment: recapture
# Multistate Recapture Separation: Survival x Detection x Movement
#
# The machinery is checked against exactly constructed histories (known
# counts), and the sampler is checked on synthetic data generated from a
# *known* parameter set, so the posterior must recover the truth within wide
# but honest tolerances.
# =============================================================================

include("test_fixtures.jl")

@testset "Recapture separation: histories, likelihood, sampler" begin
    @testset "History building (exact counts)" begin
        # region_map = [1, 2, 1]: unit 1 -> region 1, unit 2 -> region 2,
        # unit 3 -> region 1.
        mapping = [1, 2, 1]
        df = DataFrame(;
            tagid = ["A", "A", "A", "B", "C", "D"],
            tag = [0, 1, 2, 0, 0, 0],
            time = [2020.1, 2022.3, 2022.7, 2020.5, 2021.0, 2021.0],
            s_idx = [1, 3, 2, 1, 2, 1],
            is_dead = [false, false, false, false, false, false],
        )
        # Tag A: release 2020.1 (unit 1), recaptures 2022.3 (unit 3) and
        #        2022.7 (unit 2) -- two detections in the same occasion (2022),
        #        so only the FIRST one (unit 3, region 1) counts.
        # Tag B: release only (2020.5) -> never-recaptured in (region 1, 2020).
        # Tag C: release only (2021.0, unit 2) -> never in (region 2, 2021).
        # Tag D: release only (2021.0, unit 1) -> never in (region 1, 2021).
        hist = GeoMotion.build_recapture_histories(df; region_map = mapping)

        @test hist.occasions == [2020, 2021, 2022]
        @test hist.n_tags == 4
        # Six valid event rows (A's 2022.7 row is a same-occasion duplicate
        # that is collapsed during binning but still counted as a valid row).
        @test hist.n_events == 6

        # A's unique next-capture outcome: (region 1, 2020) -> (region 1, 2022).
        @test hist.groups[(1, 2020)][(1, 2022)] == 1
        # B, C and D each contribute exactly one never-recaptured count.
        @test hist.never[(1, 2020)] == 1   # B
        @test hist.never[(2, 2021)] == 1   # C
        @test hist.never[(1, 2021)] == 1   # D
        # A's last capture (2022) also accrues a never count.
        @test hist.never[(1, 2022)] == 1
        @test hist.n_segments == 5   # A 2 + B 1 + C 1 + D 1

        # Dead recoveries are not detections: add one and re-check.
        df_dead = DataFrame(;
            tagid = ["A", "A", "A", "B", "C", "D", "C"],
            tag = [0, 1, 2, 0, 0, 0, 3],
            time = [2020.1, 2022.3, 2022.7, 2020.5, 2021.0, 2021.0, 2021.5],
            s_idx = [1, 3, 2, 1, 2, 1, 2],
            is_dead = [false, false, false, false, false, false, true],
        )
        hist_dead = GeoMotion.build_recapture_histories(df_dead;
            region_map = mapping)
        @test hist_dead.n_events == 6   # the dead row is excluded
        @test hist_dead.n_tags == 4
    end

    @testset "Likelihood sanity on known parameters" begin
        # Single-region CJS with two post-release occasions.
        mapping = [1]
        df = DataFrame(;
            tagid = ["A", "A", "B", "B", "C", "D"],
            tag = [0, 1, 0, 1, 0, 0],
            time = [2020.0, 2021.0, 2020.0, 2022.0, 2021.0, 2022.0],
            s_idx = [1, 1, 1, 1, 1, 1],
            is_dead = [false, false, false, false, false, false],
        )
        hist = GeoMotion.build_recapture_histories(df; region_map = mapping)
        psi_1 = [1.0;;]

        ll_good = GeoMotion.recapture_loglikelihood(
            [0.8], [0.5], psi_1, hist)
        ll_bad = GeoMotion.recapture_loglikelihood(
            [0.05], [0.05], psi_1, hist)
        @test isfinite(ll_good)
        @test isfinite(ll_bad)
        # A plausible model must beat an implausible one on CJS-structured
        # histories with both recaptures and gaps.
        @test ll_good > ll_bad

        # An impossible model (detection = 1 after an observed gap) is -Inf.
        ll_impossible = GeoMotion.recapture_loglikelihood(
            [1.0], [1.0], psi_1, hist)
        @test ll_impossible == -Inf
    end

    @testset "Sampler recovers known survival x detection" begin
        # Generate data from the known process:
        #   apparent survival phi = 0.8 per year, detection p = 0.5 per year,
        #   all survivors move deterministically region 1 -> region 2.
        rng = Random.MersenneTwister(11)
        phi_true = 0.8
        p_true = 0.5
        years_all = [2021, 2022, 2023]
        tagid = String[]
        tag_ord = Int[]
        times = Float64[]
        units = Int[]
        is_dead_col = Bool[]
        n_tags_generated = 0
        for i in 1:60
            n_tags_generated += 1
            name = "T$i"
            push!(tagid, name); push!(tag_ord, 0)
            push!(times, 2021.0); push!(units, 1)
            push!(is_dead_col, false)
            alive = true
            tags_seen = 0
            for yr in 2022:2023
                alive = alive && (rand(rng) < phi_true)
                if alive && rand(rng) < p_true
                    tags_seen += 1
                    push!(tagid, name); push!(tag_ord, tags_seen)
                    push!(times, Float64(yr) + 0.5)
                    push!(units, 2)
                    push!(is_dead_col, false)
                end
            end
        end
        df_gen = DataFrame(;
            tagid = tagid,
            tag = tag_ord,
            time = times,
            s_idx = units,
            is_dead = is_dead_col,
        )
        kernels_stub = (P_kernel = [0.0 1.0; 0.0 1.0],)
        params_stub = (; n_samples = 150, seed = 5, fpt_step_years = 1.0 / 12.0)

        # A two-occasion table cannot identify survival vs detection; the
        # orchestrator must reject it loudly rather than return confident
        # numbers.
        df_two = DataFrame(;
            tagid = ["A", "B", "C", "A"],
            tag = [0, 0, 0, 1],
            time = [2021.0, 2021.0, 2022.0, 2022.0],
            s_idx = [1, 1, 2, 2],
            is_dead = [false, false, false, false],
        )
        @test_throws DomainError GeoMotion.run_recapture_diagnostics(
            (n_spatial = 2, data = (; tagging = df_two)), kernels_stub;
            region_map = [1, 2])

        full_loaded = (n_spatial = 2, data = (; tagging = df_gen))
        result = GeoMotion.run_recapture_diagnostics(
            full_loaded, kernels_stub, params_stub;
            region_map = [1, 2],
            n_warmup = 400,
        )
        @test result.n_tags == n_tags_generated
        @test result.occasions == [2021, 2022, 2023]

        # Posterior recovers the known process within honest tolerances.
        # Detection is only identifiable in region 2, where detections occur:
        # released tags survive/move to region 2 before any later capture
        # occasion, so no post-release detection can ever fall in region 1 and
        # its parameter is (correctly) squeezed toward 0 by the "never seen
        # again" bucket. Assert the identified quantity, and that the
        # unidentified one stays dominated by it.
        @test result.detection_mean[2] > 0.25
        @test result.detection_mean[2] < 0.9
        @test result.detection_mean[1] < result.detection_mean[2]
        @test result.survival_mean[1] > 0.55
        @test result.survival_mean[1] < 0.98
        # Movement: survivors release in region 1 and are next seen in 2.
        @test result.psi_mean[1, 2] > 0.7
        @test result.psi_mean[2, 1] < 0.3

        # Kernel comparison: one kernel step = 1/12 year, so m = 12 steps per
        # capture interval; the kernel sends everything to unit 2, hence
        # both regions move to region 2 with probability 1.
        @test result.n_kernel_steps == 12
        @test result.kernel_yearly !== nothing
        @test result.kernel_yearly[1, 2] ≈ 1.0 atol = 1e-12

        # Reproducibility: same seed, same posterior mean.
        result_repeat = GeoMotion.run_recapture_diagnostics(
            full_loaded, kernels_stub, params_stub;
            region_map = [1, 2],
            n_warmup = 400,
        )
        @test result_repeat.survival_mean[1] == result.survival_mean[1]
        @test result_repeat.detection_mean[1] == result.detection_mean[1]

        # CSV export smoke test.
        out_dir = joinpath(TEST_TMP, "recapture_csv")
        result_csv = GeoMotion.run_recapture_diagnostics(
            full_loaded, kernels_stub, params_stub;
            region_map = [1, 2],
            n_warmup = 400,
            output_dir = out_dir,
        )
        @test isfile(joinpath(out_dir, "recapture_separation_summary.csv"))
        @test isfile(joinpath(out_dir, "recapture_movement_vs_kernel.csv"))
    end
end
