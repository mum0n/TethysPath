# =============================================================================
# Multistate capture-recapture separation
#
# The one-step kernel folds three processes into "mobility": whether the
# animal moved, whether observers could have seen it if it did, and whether it
# was still alive. This layer separates them with a multistate
# Cormack-Jolly-Seber model (Arnason-Schwarz parameterisation) fitted directly
# on the tagging table's detection histories, region-mapped the same way as
# the connectivity assessment:
#
#     apparent survival   phi_r   per region per year,
#     detection/recovery  p_r     per region per year,
#     between-region move psi_rs  per year for survivors.
#
# "Apparent" is deliberate: an animal that leaves the study area is
# indistinguishable from a dead one, which is the standard CJS caveat and the
# reason phi is a lower bound on true survival.
#
# The likelihood is the per-tag multistate recursion over capture occasions
# (calendar years of the detection times), aggregated into release-year x
# outcome-table form; the fit itself is a seeded adaptive random-walk
# Metropolis sampler, matching the package's prior-scaled random-walk
# convention (see `movement_sampler` for why plain Turing `MH()` is avoided).
#
# Identifiability: phi and p are jointly identified only with >= 3 capture
# occasions, and movement between a pair of regions with no observed passages
# falls back on the uniform prior rather than on data. The orchestrator
# rejects short histories loudly instead of returning confident nonsense.
# =============================================================================

"""
    build_recapture_histories(tag_df; region_map) -> NamedTuple

Turn the tagging event table into multistate capture histories.

Input requirements (checked loudly):

- columns `tagid` (Any, stringified), `time` (decimal year or anything
  `Real`-like), and a spatial unit column: `:s_idx` (preferred), `:unit` or
  `:unit_idx`;
- rows flagged `is_dead == true` are dropped when the column exists (a dead
  recovery confirms the last live position but is not a live detection);
- rows with missing/non-finite time or unit are dropped with no silent error.

Each tag's events are sorted by time, binned to integer capture occasions
(floor of the decimal year) and collapsed to one event per tag per occasion
(the first detection of the year). Consecutive captures `(r0, i) -> (r, j)`
accumulate into an outcome table; a tag's last capture accrues a "never
recaptured again" count for its `(region, occasion)` group. A tag with a
single detection contributes only that never-count, which is correct CJS
conditioning.

Returns `groups` (`Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int},Int}}` keyed by
`(release_region, release_occasion)` to outcome
`(region, occasion) => count`), `never` (`Dict{Tuple{Int,Int},Int}`),
`occasions` (sorted distinct capture years), `n_tags` and `n_segments`.
"""
function build_recapture_histories(
    tag_df::DataFrame;
    region_map::Vector{Int}
)
    nrow(tag_df) > 0 || throw(DomainError("tagging table is empty"))
    hasproperty(tag_df, :tagid) || throw(ArgumentError(
        "tagging table has no :tagid column"))
    hasproperty(tag_df, :time) || throw(ArgumentError(
        "tagging table has no :time column (decimal year expected)"))

    events = copy(tag_df)

    # Drop dead recoveries: they are terminal confirmations, not detections.
    if hasproperty(events, :is_dead)
        keep = BitVector(undef, nrow(events))
        for i in 1:nrow(events)
            dead = events[i, :is_dead]
            keep[i] = !(dead === true || dead === 1.0 || dead === 1)
        end
        events = events[keep, :]
    end

    # Resolve the spatial unit column.
    unit_col = nothing
    for candidate in (:s_idx, :unit, :unit_idx)
        if hasproperty(events, candidate)
            unit_col = candidate
            break
        end
    end
    unit_col === nothing && throw(ArgumentError(
        "tagging table has no spatial unit column (:s_idx, :unit or :unit_idx)" *
        ", so detections cannot be mapped to regions"))

    # Valid rows only: finite time, known unit in range.
    n = length(region_map)
    keep = BitVector(undef, nrow(events))
    for i in 1:nrow(events)
        row = events[i, :]
        t = row[:time]
        u = row[unit_col]
        ok = (t isa Real && isfinite(Float64(t)) &&
              u isa Integer && 1 <= u <= n)
        if !ok && u isa Real && !((u isa Integer))
            # Accept whole-number floats for the unit column.
            ui = trunc(Int, u)
            ok = (t isa Real && isfinite(Float64(t)) && isinteger(u) &&
                  1 <= ui <= n)
        end
        keep[i] = ok
    end
    events = events[keep, :]
    nrow(events) > 0 || throw(DomainError(
        "no valid tagging rows after filtering (need finite :time and " *
        ":$(unit_col) within 1..$n)"))

    # Group by tag, sort by time (ties in original order), bin to years,
    # collapse same-occasion repeats, then emit segments.
    events.tagid = String.(events.tagid)
    event_times = Float64.(events[:, :time])
    event_units = Int[trunc(Int, Float64(u)) for u in events[:, unit_col]]
    event_tags = events[:, :tagid]

    order = sortperm(1:nrow(events); by = i -> (event_tags[i], event_times[i]))
    tagged_events = Dict{String, Vector{Int}}()
    for i in order
        tag = event_tags[i]
        push!(get!(tagged_events, tag, Int[]), i)
    end

    groups = Dict{Tuple{Int,Int}, Dict{Tuple{Int,Int},Int}}()
    never = Dict{Tuple{Int,Int},Int}()
    occasions_set = Set{Int}()
    n_tags = 0
    n_segments = 0

    for (tag, idx_list) in tagged_events
        isempty(idx_list) && continue
        n_tags += 1
        # Binned capture occasions for this tag, deduplicated.
        years = Int[]
        units_of_occasion = Int[]
        for i in idx_list
            yr = floor(Int, event_times[i])
            if isempty(years) || yr > years[end]
                push!(years, yr)
                push!(units_of_occasion, event_units[i])
            end
            # Same-occasion repeats (event_times equal after flooring) are
            # skipped: the first detection of the year fixes the occasion.
        end
        for yr in years
            push!(occasions_set, yr)
        end
        for k in 1:length(years)
            r0 = region_map[units_of_occasion[k]]
            key = (r0, years[k])
            if k < length(years)
                rnext = region_map[units_of_occasion[k + 1]]
                outcome_key = (rnext, years[k + 1])
                outcomes = get!(groups, key, Dict{Tuple{Int,Int},Int}())
                outcomes[outcome_key] = get(outcomes, outcome_key, 0) + 1
            else
                never[key] = get(never, key, 0) + 1
            end
            n_segments += 1
        end
    end

    occasions = sort!(collect(occasions_set))
    return (
        groups     = groups,
        never      = never,
        occasions  = occasions,
        n_tags     = n_tags,
        n_segments = n_segments,
        n_events   = nrow(events),
    )
end

# -----------------------------------------------------------------------------
# Multistate CJS likelihood
# -----------------------------------------------------------------------------

"""
    recapture_loglikelihood(phi, pdet, psi, hist) -> Float64

Log-likelihood of the multistate CJS model on aggregated histories.

`phi` (apparent survival per region per year), `pdet` (detection probability
per region) and `psi` (R x R row-stochastic between-region movement matrix
for survivors) act between consecutive capture occasions. For a segment
released in region `r0` at occasion `i`, the recursion tracks the probability
of being alive and undetected in each region, multiplying by survival and
movement each interval and by `p_r` when a capture at occasion `j` in region
`r` is the recorded outcome. The "never seen again" bucket is the probability
complement of every observed capture at later occasions: undetected
survivors *and* animals that died in between, so each segment's categories
partition the unit simplex (a proper multinomial).

Returns `-Inf` for impossible parameter sets (an observed outcome with zero
probability) rather than erroring, so a sampler can reject the proposal.
"""
function recapture_loglikelihood(phi, pdet, psi, hist)
    R = length(phi)
    occasions = hist.occasions
    # A release occasion may have zero observed next captures (all its
    # segments are "never"), so the key set is the union of both maps.
    all_keys = Set(keys(hist.groups))
    union!(all_keys, keys(hist.never))
    total = 0.0
    for key in all_keys
        r0 = key[1]
        y0 = key[2]
        outcomes = get(hist.groups, key, nothing)
        i0 = findfirst(==(y0), occasions)
        i0 === nothing && return -Inf
        alive = zeros(Float64, R)
        alive[r0] = 1.0
        ll = 0.0
        outcome_total = 0.0
        for idx in (i0 + 1):length(occasions)
            y = occasions[idx]
            grown = alive .* phi
            # vec(...) keeps `alive` a plain Vector: transpose(grown) * psi on
            # a Vector yields an adjoint wrapper that turns into a 1 x R
            # Matrix once multiplied elementwise, which silently degenerates
            # the state recursion afterwards.
            alive_next = vec(transpose(grown) * psi)
            # Mass detected this occasion, whether or not the data record one:
            # everything the segment does NOT detect is its "never" bucket.
            outcome_total += sum(alive_next .* pdet)
            for r in 1:R
                cnt = outcomes === nothing ? 0 :
                    get(outcomes, (r, y), 0)
                if cnt > 0
                    prob_hit = alive_next[r] * pdet[r]
                    if prob_hit <= 0.0
                        return -Inf
                    end
                    ll += cnt * log(prob_hit)
                end
            end
            alive = alive_next  # implicit undetected conditioning below
            for r in 1:R
                alive[r] *= (1.0 - pdet[r])
            end
        end
        never_cnt = get(hist.never, key, 0)
        if never_cnt > 0
            # "Never recaptured again" is everything not observed at any later
            # occasion: undetected survivors AND animals that died in between.
            never_prob = 1.0 - outcome_total
            if never_prob <= 0.0
                return -Inf
            end
            ll += never_cnt * log(never_prob)
        end
        total += ll
    end
    return total
end

# -----------------------------------------------------------------------------
# Parameter transforms and sampler
# -----------------------------------------------------------------------------

_logit(x::Float64) = log(x / (1.0 - x))
_inv_logit(x::Real) = 1.0 / (1.0 + exp(-x))

"""
    _recapture_decode(x, R) -> (pdet, phi, psi, prior_log)

Map the unconstrained parameter vector to (detection, survival, movement)
matrices plus the log prior density implied by flat Beta(1,1) priors on
`pdet`/`phi` and uniform stick-weights for `psi`.

- `x[1:R]` are logits of `pdet`,
- `x[R+1:2R]` are logits of `phi`,
- the remaining `R * (R - 1)` entries are per-row stick logits for `psi`
  (row `r` has `R - 1` of them; none when `R == 1`).

Flat Beta(1,1) priors in the logit domain carry the density `s * (1 - s)`,
and uniform sticks (the GEM construction of `Dirichlet(1, ..., 1)`) carry the
same form per stick; those Jacobian terms are returned in `prior_log`.
"""
function _recapture_decode(x::Vector{Float64}, R::Int)
    lo = 1e-9
    hi = 1.0 - 1e-9

    pdet = [clamp(_inv_logit(x[i]), lo, hi) for i in 1:R]
    phi = [clamp(_inv_logit(x[R + i]), lo, hi) for i in 1:R]

    psi = Matrix{Float64}(undef, R, R)
    prior_log = 0.0
    for i in 1:R
        prior_log += log(pdet[i]) + log(1.0 - pdet[i])
        prior_log += log(phi[i]) + log(1.0 - phi[i])
        if R == 1
            psi[i, 1] = 1.0
            continue
        end
        base = 2 * R + 1 + (i - 1) * (R - 1)
        remaining = 1.0
        for k in 1:(R - 1)
            raw = clamp(_inv_logit(x[base + k - 1]), 1e-12, 1.0 - 1e-12)
            prior_log += log(raw) + log(1.0 - raw)
            share = remaining * raw
            psi[i, k] = share
            remaining -= share
        end
        psi[i, R] = remaining
    end
    valid_psi = true
    for entry in psi
        if !isfinite(entry) || entry < 0.0
            valid_psi = false
        end
    end
    !valid_psi && return (pdet, phi, psi, -Inf)
    return (pdet, phi, psi, prior_log)
end

"""
    _recapture_initial_point(hist, R) -> Vector{Float64}

Starting point: `p = 0.5`, `phi = 0.8`, and uniform movement rows
(`psi = 1/R`, whose stick decomposition is `s_k = 1 / (R - k + 1)`).
"""
function _recapture_initial_point(hist, R::Int)
    x = zeros(Float64, 2 * R + max(R * (R - 1), 0))
    for i in 1:R
        x[i] = _logit(0.5)
        x[R + i] = _logit(0.8)
    end
    if R > 1
        for i in 1:R
            base = 2 * R + 1 + (i - 1) * (R - 1)
            for k in 1:(R - 1)
                x[base + k - 1] = _logit(1.0 / (R - k + 1))
            end
        end
    end
    return x
end

"""
    fit_recapture_cjs(hist; n_regions = nothing, n_warmup = 300,
                      n_samples = 200, thin = 3, seed = 42) -> NamedTuple

Fit the multistate CJS model by adaptive coordinate-wise random-walk
Metropolis: each coordinate's proposal standard deviation adapts toward a
0.44 acceptance rate during warmup, then freezes; post-warmup draws are
thinned by `thin`. Seeded, so runs are reproducible.

Returns posterior draws as *decoded* matrices (draws x regions) for
`pdet_draws`, `phi_draws`, `psi_draws` (draws x (R * R), row-major), plus
`acceptance_rate`, iteration bookkeeping and `n_regions`.
"""
function fit_recapture_cjs(
    hist;
    n_regions::Union{Nothing, Int} = nothing,
    n_warmup::Int = 300,
    n_samples::Int = 200,
    thin::Int = 3,
    seed::Int = 42
)
    n_warmup >= 1 || throw(DomainError(n_warmup, "n_warmup must be >= 1"))
    n_samples >= 1 || throw(DomainError(n_samples, "n_samples must be >= 1"))
    thin >= 1 || throw(DomainError(thin, "thin must be >= 1"))

    R = n_regions
    if R === nothing
        # Regions may appear only in "never" buckets (a region whose animals
        # are never recaptured elsewhere), so scan both maps.
        R = 1
        for key in keys(hist.groups)
            local k = key[1]
            k > R && (R = k)
        end
        for key in keys(hist.never)
            local k = key[1]
            k > R && (R = k)
        end
    end
    R >= 1 || throw(DomainError(R, "no regions resolved"))
    dim = 2 * R + max(R * (R - 1), 0)

    x = _recapture_initial_point(hist, R)
    current_logpost = _recapture_target(x, hist, R)
    isfinite(current_logpost) || throw(DomainError(
        current_logpost, "initial point has non-finite log posterior"))

    rng = Random.MersenneTwister(seed)
    proposal_sd = fill(0.2, dim)
    accept_window = zeros(Int, dim)
    attempt_window = zeros(Int, dim)
    adapt_every = 50
    # Coordinate-wise Metropolis crawls along the strongly anti-correlated
    # phi/p ridge of the CJS likelihood; a joint full-vector move breaks the
    # ridge and is adapted on its own acceptance statistics.
    joint_sd = 0.1
    joint_accept_window = 0
    joint_attempt_window = 0

    total_iters = n_warmup + n_samples * thin
    pdet_draws = Matrix{Float64}(undef, n_samples, R)
    phi_draws = Matrix{Float64}(undef, n_samples, R)
    psi_draws = Matrix{Float64}(undef, n_samples, R * R)
    n_accept_total = 0
    n_total_total = 0
    kept = 0

    for iter in 1:total_iters
        for c in 1:dim
            proposal = copy(x)
            proposal[c] += proposal_sd[c] * randn(rng)
            proposal_logpost = _recapture_target(proposal, hist, R)
            attempt_window[c] += 1
            n_total_total += 1
            log_alpha = proposal_logpost - current_logpost
            if log(rand(rng)) < log_alpha
                x = proposal
                current_logpost = proposal_logpost
                accept_window[c] += 1
                n_accept_total += 1
            end
        end

        # Joint full-vector move alongside the coordinate sweep.
        joint_proposal = copy(x)
        for c in 1:dim
            joint_proposal[c] += joint_sd * randn(rng)
        end
        joint_logpost = _recapture_target(joint_proposal, hist, R)
        joint_attempt_window += 1
        n_total_total += 1
        if log(rand(rng)) < joint_logpost - current_logpost
            x = joint_proposal
            current_logpost = joint_logpost
            joint_accept_window += 1
            n_accept_total += 1
        end

        if iter <= n_warmup && iter % adapt_every == 0
            for c in 1:dim
                rate = accept_window[c] / max(attempt_window[c], 1)
                proposal_sd[c] = clamp(
                    proposal_sd[c] * exp(rate - 0.44), 1e-4, 10.0)
                accept_window[c] = 0
                attempt_window[c] = 0
            end
            joint_rate = joint_accept_window / max(joint_attempt_window, 1)
            joint_sd = clamp(joint_sd * exp(joint_rate - 0.44), 1e-4, 2.0)
            joint_accept_window = 0
            joint_attempt_window = 0
        end

        if iter > n_warmup && (iter - n_warmup) % thin == 0
            kept += 1
            # Store the DECODED parameters, not the raw unconstrained ones:
            # consumers read probabilities and simplex rows straight out of
            # the draw matrices.
            pdet_d, phi_d, psi_d, _ = _recapture_decode(x, R)
            pdet_draws[kept, :] .= pdet_d
            phi_draws[kept, :] .= phi_d
            psi_draws[kept, :] .= vec(transpose(psi_d))
        end
    end
    kept == n_samples || throw(DomainError(kept,
        "sampler produced fewer draws than requested"))

    return (
        pdet_draws       = pdet_draws,
        phi_draws        = phi_draws,
        psi_draws        = psi_draws,
        n_regions        = R,
        acceptance_rate  = n_accept_total / max(n_total_total, 1),
        n_warmup         = n_warmup,
        n_samples        = n_samples,
        thin             = thin,
    )
end

# Internal: full log posterior in the unconstrained parameterisation.
function _recapture_target(x::Vector{Float64}, hist, R::Int)
    pdet, phi, psi, prior_log = _recapture_decode(x, R)
    if !isfinite(prior_log)
        return -Inf
    end
    ll = recapture_loglikelihood(phi, pdet, psi, hist)
    if !isfinite(ll)
        return -Inf
    end
    return prior_log + ll
end

# -----------------------------------------------------------------------------
# Posterior summaries
# -----------------------------------------------------------------------------

function _posterior_summary(draws::Vector{Float64})
    q025 = quantile(draws, 0.025)
    qmean = mean(draws)
    q975 = quantile(draws, 0.975)
    return (mean = qmean, lower = q025, upper = q975)
end

# =============================================================================
# Diagnostics orchestrator
# =============================================================================

"""
    run_recapture_diagnostics(loaded, kernels, params = (;);
        output_dir = nothing, region_map = nothing, region_labels = nothing,
        n_warmup = 300, n_samples = nothing, thin = 3, seed = nothing) -> NamedTuple

Separate apparent survival, detection/recovery probability and
between-region movement from the tagging histories, then compare the fitted
between-region movement matrix against the movement the transition kernel
implies at the same timescale.

Guards (all loud `DomainError`s recorded by the pipeline skippers):

- a tagging table with the required columns must be present;
- at least **3** capture occasions (calendar years) are required, because
  fewer cannot identify survival and detection jointly;
- at least one region must resolve.

The kernel comparison expresses the region-level kernel at the capture
interval: `P_yearly = P_reg^m` where `m = round(Int, 1 / step_years)` comes
from `params.fpt_step_years` (or the matching keyword). When the kernel step
is not a clean sub-multiple of a year, the comparison is reported as
`nothing` rather than approximated silently.

When `output_dir` is a `String`, writes
`recapture_separation_summary.csv` (per region: detection and survival
posteriors) and `recapture_movement_vs_kernel.csv` (movement posteriors vs
kernel-implied annual transition probabilities). The returned artifact is
plain vectors/matrices and round-trips through JLD2.
"""
function run_recapture_diagnostics(
    loaded::NamedTuple,
    kernels::NamedTuple,
    params = (;);
    output_dir::Union{Nothing, String} = nothing,
    region_map::Union{Nothing, Vector{Int}} = nothing,
    region_labels::Union{Nothing, Vector{String}} = nothing,
    n_warmup::Int = 300,
    n_samples::Union{Nothing, Int} = nothing,
    thin::Int = 3,
    seed::Union{Nothing, Int} = nothing
)
    data = loaded.data
    (hasproperty(data, :tagging) && data.tagging !== nothing) || throw(
        DomainError("no tagging table in loaded.data"))

    partition, labels = resolve_region_partition(
        loaded, params;
        region_map = region_map,
        region_labels = region_labels,
    )
    R = length(labels)

    hist = build_recapture_histories(data.tagging; region_map = partition)
    n_occasions = length(hist.occasions)
    n_occasions >= 3 || throw(DomainError(n_occasions,
        "recapture separation needs >= 3 capture occasions (got $n_occasions); " *
        "with fewer, apparent survival and detection probability are not " *
        "jointly identifiable"))

    n_samples_val = n_samples
    if n_samples_val === nothing && hasproperty(params, :n_samples)
        n_samples_val = Int(params.n_samples)
    end
    n_samples_val === nothing && (n_samples_val = 200)

    seed_val = seed
    if seed_val === nothing && hasproperty(params, :seed)
        seed_val = Int(params.seed)
    end
    seed_val === nothing && (seed_val = 42)

    fit = fit_recapture_cjs(hist;
        n_regions = R,
        n_warmup = n_warmup,
        n_samples = n_samples_val,
        thin = thin,
        seed = seed_val,
    )

    pdet_mean = Float64[]
    pdet_lower = Float64[]
    pdet_upper = Float64[]
    phi_mean = Float64[]
    phi_lower = Float64[]
    phi_upper = Float64[]
    for r in 1:R
        sp = _posterior_summary(vec(fit.pdet_draws[:, r]))
        push!(pdet_mean, sp.mean)
        push!(pdet_lower, sp.lower)
        push!(pdet_upper, sp.upper)
        sph = _posterior_summary(vec(fit.phi_draws[:, r]))
        push!(phi_mean, sph.mean)
        push!(phi_lower, sph.lower)
        push!(phi_upper, sph.upper)
    end

    psi_mean = zeros(Float64, R, R)
    psi_lower = zeros(Float64, R, R)
    psi_upper = zeros(Float64, R, R)
    for i in 1:R
        for j in 1:R
            col = fit.psi_draws[:, (i - 1) * R + j]
            sp = _posterior_summary(vec(col))
            psi_mean[i, j] = sp.mean
            psi_lower[i, j] = sp.lower
            psi_upper[i, j] = sp.upper
        end
    end

    # Kernel-implied annual movement at the capture interval, when the kernel
    # step is a clean sub-multiple of a year.
    step_years_val = 0.0
    if hasproperty(params, :fpt_step_years)
        step_years_val = Float64(params.fpt_step_years)
    end
    kernel_yearly = nothing
    n_kernel_steps = nothing
    P_kernel = kernels.P_kernel
    if step_years_val > 0.0 && !(P_kernel isa AbstractVector)
        m_ratio = 1.0 / step_years_val
        m = round(Int, m_ratio)
        if m >= 1 && abs(m_ratio - m) < 1e-6
            P_reg = region_transition_matrix(P_kernel, partition)
            kernel_yearly = P_reg^m
            n_kernel_steps = m
        end
    end

    result = (
        region_labels    = labels,
        n_regions        = R,
        occasions        = hist.occasions,
        n_tags           = hist.n_tags,
        n_segments       = hist.n_segments,
        n_capture_events = hist.n_events,
        detection_mean   = pdet_mean,
        detection_lower  = pdet_lower,
        detection_upper  = pdet_upper,
        survival_mean    = phi_mean,
        survival_lower   = phi_lower,
        survival_upper   = phi_upper,
        psi_mean         = psi_mean,
        psi_lower        = psi_lower,
        psi_upper        = psi_upper,
        kernel_yearly    = kernel_yearly,
        n_kernel_steps   = n_kernel_steps,
        acceptance_rate  = fit.acceptance_rate,
        n_samples        = fit.n_samples,
        n_warmup         = fit.n_warmup,
        thin             = fit.thin,
    )

    if output_dir !== nothing
        mkpath(output_dir)
        export_recapture_separation_summary(
            joinpath(output_dir, "recapture_separation_summary.csv"),
            result,
        )
        export_recapture_movement_vs_kernel(
            joinpath(output_dir, "recapture_movement_vs_kernel.csv"),
            result,
        )
    end
    return result
end

# =============================================================================
# CSV exporters
# =============================================================================

"""
    export_recapture_separation_summary(path, result)

Write per-region posterior summaries of detection probability and apparent
survival (mean, 2.5th and 97.5th percentiles).
"""
function export_recapture_separation_summary(path::AbstractString,
                                             result::NamedTuple)
    region = String[]
    detection_mean = Float64[]
    detection_lower = Float64[]
    detection_upper = Float64[]
    survival_mean = Float64[]
    survival_lower = Float64[]
    survival_upper = Float64[]
    for r in 1:result.n_regions
        push!(region, result.region_labels[r])
        push!(detection_mean, result.detection_mean[r])
        push!(detection_lower, result.detection_lower[r])
        push!(detection_upper, result.detection_upper[r])
        push!(survival_mean, result.survival_mean[r])
        push!(survival_lower, result.survival_lower[r])
        push!(survival_upper, result.survival_upper[r])
    end
    CSV.write(path, DataFrame(;
        region = region,
        detection_prob_mean = detection_mean,
        detection_prob_lower = detection_lower,
        detection_prob_upper = detection_upper,
        apparent_survival_mean = survival_mean,
        apparent_survival_lower = survival_lower,
        apparent_survival_upper = survival_upper,
    ))
    return path
end

"""
    export_recapture_movement_vs_kernel(path, result)

Write the between-region movement posterior next to the kernel-implied
annual transition probability for the same pair (empty kernel column when no
timescale comparison was possible).
"""
function export_recapture_movement_vs_kernel(path::AbstractString,
                                             result::NamedTuple)
    R = result.n_regions
    from_region = String[]
    to_region = String[]
    psi_posterior_mean = Float64[]
    psi_posterior_lower = Float64[]
    psi_posterior_upper = Float64[]
    kernel_annual = Vector{Union{Missing,Float64}}()
    for i in 1:R
        for j in 1:R
            push!(from_region, result.region_labels[i])
            push!(to_region, result.region_labels[j])
            push!(psi_posterior_mean, result.psi_mean[i, j])
            push!(psi_posterior_lower, result.psi_lower[i, j])
            push!(psi_posterior_upper, result.psi_upper[i, j])
            if result.kernel_yearly === nothing
                push!(kernel_annual, missing)
            else
                push!(kernel_annual, result.kernel_yearly[i, j])
            end
        end
    end
    CSV.write(path, DataFrame(;
        from_region = from_region,
        to_region = to_region,
        recapture_movement_mean = psi_posterior_mean,
        recapture_movement_lower = psi_posterior_lower,
        recapture_movement_upper = psi_posterior_upper,
        kernel_annual = kernel_annual,
    ))
    return path
end
