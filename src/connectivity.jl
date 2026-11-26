# =============================================================================
# Time-resolved connectivity assessment
#
# This layer answers the questions the one-step kernel and the steady-state
# circuit analysis cannot: how fast, and with what probability, an animal
# resident in one region *arrives* in another. Everything is computed by
# forward-propagating the fitted transition kernel, so results are on the
# kernel's own clock: one "step" is the discrete interval the kernel was built
# for (a day for daily ingestion, a month for monthly). The caller converts
# steps to years through `step_years`.
#
# All arrival quantities are first-passage aware (a walk is absorbed at the
# target), which distinguishes "has never been to the target" from "passes
# through". Region-level quantities are mass-weighted coarse grainings of the
# unit-level kernel, not pairwise cell averages.
# =============================================================================

"""
    resolve_region_partition(loaded, params = (;); region_map = nothing,
                             region_labels = nothing) -> (region_map, labels)

Resolve the unit-to-region partition used by every connectivity assessment.

Priority (highest first):

1. an explicit `region_map::Vector{Int}` argument,
2. `params.region_map` (a `Dict{Int,Int}` from the configuration), which
   overrides the entries of the units it lists on a region-1 background,
3. `loaded.region_map`, which is only non-`nothing` when region polygons were
   configured at load time,
4. a single region spanning every unit (the historical fallback).

Units that no source assigned (a `0` from partial polygon coverage, or an
unlisted unit under `params.region_map`) are collected into a final
"Unassigned" region instead of dropped, so the partition covers the whole
domain: animals can be there and dropping them would silently understate
connectivity.

Returns `(region_map::Vector{Int}, labels::Vector{String})` with contiguous
region indices `1..R` and one label per region.
"""
function resolve_region_partition(
    loaded::NamedTuple,
    params = (;);
    region_map::Union{Nothing, Vector{Int}} = nothing,
    region_labels::Union{Nothing, Vector{String}} = nothing
)
    n = loaded.n_spatial
    n >= 1 || throw(ArgumentError("empty spatial domain"))

    resolved = nothing
    if region_map !== nothing
        resolved = copy(region_map)
    elseif params !== nothing && hasproperty(params, :region_map) &&
           params.region_map isa AbstractDict && !isempty(params.region_map)
        entries = collect(params.region_map)
        all(unit -> 1 <= unit[1] <= n, entries) || throw(ArgumentError(
            "params.region_map lists unit indices outside 1..$n"
        ))
        all(reg -> reg[2] > 0, entries) || throw(ArgumentError(
            "params.region_map values must be positive region indices"
        ))
        base = ones(Int, n)
        for pair in entries
            base[pair[1]] = pair[2]
        end
        resolved = base
    elseif hasproperty(loaded, :region_map) && loaded.region_map !== nothing
        resolved = collect(loaded.region_map)
    else
        resolved = ones(Int, n)
    end
    length(resolved) == n || throw(ArgumentError(
        "region_map length $(length(resolved)) does not match n_spatial $n"
    ))

    # Unassigned units (0, or gaps under a sparse params.region_map) become a
    # trailing region of their own rather than being dropped.
    used = Set{Int}()
    for r in resolved
        r > 0 && push!(used, r)
    end
    isempty(used) && throw(ArgumentError(
        "region resolution found no positively labelled units"
    ))
    unassigned_id = 0
    if any(==(0), resolved)
        unassigned_id = maximum(used) + 1
        for i in eachindex(resolved)
            if resolved[i] == 0
                resolved[i] = unassigned_id
            end
        end
    end

    # Relabel to contiguous 1..R so the region-level matrices are dense.
    present = sort!(unique(resolved))
    remap = Dict{Int,Int}(old => new for (new, old) in enumerate(present))
    final_map = Int[remap[r] for r in resolved]
    n_regions = length(present)

    labels = nothing
    if region_labels !== nothing
        if length(region_labels) != n_regions
            throw(ArgumentError(
                "region_labels has $(length(region_labels)) entries but the " *
                "partition has $n_regions regions"
            ))
        end
        labels = String[String(l) for l in region_labels]
    else
        try_labels = nothing
        if params !== nothing && hasproperty(params, :region_labels) && !isempty(params.region_labels)
            try_labels = params.region_labels
        elseif hasproperty(loaded, :region_labels) && loaded.region_labels isa AbstractVector &&
               !isempty(loaded.region_labels)
            try_labels = loaded.region_labels
        end
        if try_labels isa AbstractVector && length(try_labels) == n_regions
            labels = String[String(l) for l in try_labels]
        else
            labels = String[]
            for id in present
                if id == unassigned_id
                    push!(labels, "Unassigned")
                elseif try_labels isa AbstractVector && 1 <= id <= length(try_labels)
                    push!(labels, String(try_labels[id]))
                else
                    push!(labels, "Region $id")
                end
            end
        end
    end

    return final_map, labels
end

# -----------------------------------------------------------------------------
# Region-level transition matrix
# -----------------------------------------------------------------------------

"""
    region_transition_matrix(P, region_map) -> Matrix{Float64}

Coarse-grain the unit-level row-stochastic kernel `P` to region level with
*uniform* source mass inside each region:

    P_reg[r, s] = ( sum over u in r, v in s of P[u, v] ) / (number of units in r)

This is the row-stochastic analogue of averaging; it differs from the
pairwise-cell-mean and renormalised `connectivity_matrix` of
`compute_stock_connectivity_matrix`. A row of `P_reg` answers "where does the
next step go, on average, for an animal uniformly distributed over region r".

Rows are renormalised afterwards to guard against floating-point drift.
Throws `DomainError` when the region map does not cover every unit of `P`,
and `ArgumentError` when `P` is not square.
"""
function region_transition_matrix(P, region_map::Vector{Int})
    size(P, 1) == size(P, 2) || throw(ArgumentError(
        "P must be square, got $(size(P, 1)) x $(size(P, 2))"
    ))
    n = size(P, 1)
    (length(region_map) == n) || throw(ArgumentError(
        "region_map length $(length(region_map)) does not cover P's $n units"
    ))
    n_regions = maximum(region_map)
    n_regions >= 1 || throw(DomainError(n_regions, "region_map is empty"))
    minimum(region_map) >= 1 || throw(DomainError(
        minimum(region_map), "region_map must label units from 1 upward"
    ))

    P_reg = zeros(Float64, n_regions, n_regions)
    counts = zeros(Int, n_regions)
    if P isa SparseMatrixCSC
        rows = rowvals(P)
        vals = nonzeros(P)
        for v in 1:n
            rv = region_map[v]
            counts[rv] += 1
            for ptr in nzrange(P, v)
                u = rows[ptr]
                P_reg[region_map[u], rv] += vals[ptr]
            end
        end
    else
        for u in 1:n
            r = region_map[u]
            for v in 1:n
                P_reg[r, region_map[v]] += P[u, v]
            end
            counts[r] += 1
        end
    end
    for r in 1:n_regions
        counts[r] > 0 || throw(DomainError(
            r, "region $r has zero units in region_map"
        ))
        P_reg[r, :] ./= counts[r]
    end
    for r in 1:n_regions
        row_total = sum(P_reg[r, :])
        if row_total > 0.0
            P_reg[r, :] ./= row_total
        end
    end
    return P_reg
end

# -----------------------------------------------------------------------------
# Stochasticity validation shared by the Markov solvers
# -----------------------------------------------------------------------------

function _check_row_stochastic(P::AbstractMatrix{<:Real}; tol::Float64 = 1e-6)
    size(P, 1) == size(P, 2) || throw(ArgumentError(
        "P must be square, got $(size(P, 1)) x $(size(P, 2))"
    ))
    all(isfinite, P) || throw(DomainError("P contains NaN/Inf"))
    worst = 0.0
    for i in 1:size(P, 1)
        dev = abs(sum(P[i, :]) - 1.0)
        if dev > worst
            worst = dev
        end
    end
    if worst > tol
        throw(DomainError(worst,
            "P is not row-stochastic (max row-sum deviation)"))
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Exact mean first-passage times
# -----------------------------------------------------------------------------

"""
    mean_first_passage_times(P) -> Matrix{Float64}

Exact mean first-passage-time matrix (in kernel steps) of a discrete-time
Markov chain: `M[i, j]` is the expected number of steps for a walk started at
`i` to first reach `j`, on a chain that is absorbed at `j` on arrival.
Diagonal entries are 0 by first-passage convention.

For each target `j` the system `(I - Q_j) m = 1` over the remaining states is
solved exactly (dense LU), where `Q_j` is `P` with its `j` row and column
deleted. The cost is O(K^4) in the number of regions K, small here.

Throws `DomainError` when rows of `P` are not probabilities (all entries
finite, each row summing to 1 within 1e-6); a corrupted kernel must fail
loudly rather than produce travel times that merely look numerical.
"""
function mean_first_passage_times(P::AbstractMatrix{<:Real})
    _check_row_stochastic(P)
    n = size(P, 1)
    mfpt = zeros(Float64, n, n)
    for j in 1:n
        keep = Int[]
        for k in 1:n
            k != j && push!(keep, k)
        end
        isempty(keep) && continue
        Q = P[keep, keep]
        sol = (I - Q) \ ones(Float64, length(keep))
        for idx in 1:length(keep)
            mfpt[keep[idx], j] = sol[idx]
        end
        mfpt[j, j] = 0.0
    end
    return mfpt
end

# -----------------------------------------------------------------------------
# First-arrival probabilities (absorbing forward propagation)
# -----------------------------------------------------------------------------

"""
    first_arrival_cdfs(P; horizon = 100) -> NamedTuple

For every ordered region pair `(s, t)`, the distribution of the *first*
passage time from `s` to `t` over `1..horizon` kernel steps, computed by
forward-propagating one transient vector per source with absorption at each
fiber's target.

Returns:

- `cdf::Array{Float64,3}`, size (K, K, horizon): `cdf[s, t, tau]` is
  P(first arrival at `t` by step `tau`). The diagonal is forced to 0 (a
  self-arrival is a return time, not a first passage here).
- `arrival_prob::Matrix{Float64}`: `cdf[:, :, horizon]`, the matrix of
  within-horizon arrival probabilities.
- `median_steps::Matrix{Float64}`: the smallest `tau` with
  `cdf[s, t, tau] >= 0.5`, `NaN` when the target is not reached within the
  horizon (diagonal `NaN`).

Throws `DomainError` when `horizon < 1` and when `P` is not row-stochastic
(1e-6 tolerance).
"""
function first_arrival_cdfs(P::AbstractMatrix{<:Real}; horizon::Int = 100)
    horizon >= 1 || throw(DomainError(horizon, "horizon must be >= 1"))
    _check_row_stochastic(P)
    K = size(P, 1)

    Pd = Matrix{Float64}(P)
    cdf = zeros(Float64, K, K, horizon)

    for t in 1:K
        transient = Matrix{Float64}(I, K, K)   # column s: mass from source s
        P_absorbed = copy(Pd)
        P_absorbed[:, t] .= 0.0                # absorb mass entering t
        for tau in 1:horizon
            for s in 1:K
                arrival = 0.0
                for u in 1:K
                    arrival += Pd[u, t] * transient[u, s]
                end
                # Cumulative CDF slot: carry the running total through tau.
                carried = tau == 1 ? 0.0 : cdf[s, t, tau - 1]
                cdf[s, t, tau] = carried + arrival
            end
            transient = transpose(P_absorbed) * transient
        end
        for tau in 1:horizon
            cdf[t, t, tau] = 0.0
        end
    end

    arrival_prob = cdf[:, :, horizon]
    median_steps = fill(NaN, K, K)
    for s in 1:K
        for t in 1:K
            if s == t
                continue
            end
            for tau in 1:horizon
                if cdf[s, t, tau] >= 0.5
                    median_steps[s, t] = Float64(tau)
                    break
                end
            end
        end
    end

    return (
        cdf          = cdf,
        arrival_prob = arrival_prob,
        median_steps = median_steps,
        horizon      = horizon,
    )
end

# -----------------------------------------------------------------------------
# Domain-wide corridor usage
# -----------------------------------------------------------------------------

"""
    corridor_visit_rate(P; horizon = 50) -> NamedTuple

Expected visit counts per unit, per kernel step, for a random walker that
starts uniformly over the domain and moves on `P` for `horizon` steps:

    visits_u = (1/K) * sum over s, tau of (P^tau)[s, u]

`visits` is the raw expected-visit count; `normalized` divides by its maximum
so it can drive colour scales. Units that funnel domain-wide flow stand out
here even when no individual pairwise analysis would flag them.
"""
function corridor_visit_rate(P::AbstractMatrix{<:Real}; horizon::Int = 50)
    horizon >= 1 || throw(DomainError(horizon, "horizon must be >= 1"))
    _check_row_stochastic(P)
    n = size(P, 1)

    visits = zeros(Float64, n)
    mass = ones(Float64, n) ./ n
    for tau in 1:horizon
        next_mass = P' * mass
        for u in 1:n
            visits[u] += next_mass[u]
        end
        mass = next_mass
    end
    peak = maximum(visits)
    normalized = fill(0.0, n)
    if peak > 0.0
        normalized .= visits ./ peak
    end
    return (
        visits     = visits,
        normalized = normalized,
        horizon    = horizon,
    )
end

# -----------------------------------------------------------------------------
# Communities (greedy modularity on the region graph)
# -----------------------------------------------------------------------------

"""
    connectivity_communities(weights; k = 0) -> NamedTuple

Partition the regions into stable connectivity clusters by greedy modularity
maximisation on the symmetrised, diagonal-zeroed weight matrix
(`A = (W + W') / 2`). Starting from singletons, the pair of communities whose
merge gains the most modularity is merged repeatedly: until no positive gain
remains when `k == 0`, otherwise exactly `k` communities (`k >= 2`).

Requires `k >= 2` regions. Returns `(labels, modularity, n_communities)`,
with `labels[i]` the community of region `i`, relabelled contiguous from 1.
"""
function connectivity_communities(weights::AbstractMatrix{<:Real}; k::Int = 0)
    n = size(weights, 1)
    size(weights, 1) == size(weights, 2) || throw(ArgumentError(
        "weights must be square, got $(size(weights, 1)) x $(size(weights, 2))"
    ))
    n >= 2 || throw(DomainError(n, "community detection needs >= 2 regions"))
    k >= 0 || throw(DomainError(k, "k must be 0 (adaptive) or >= 2"))
    if k > 0 && k > n
        throw(DomainError(k, "k exceeds the number of regions"))
    end
    all(isfinite, weights) || throw(DomainError("weights contains NaN/Inf"))
    minimum(weights) >= 0 || throw(DomainError("weights must be nonnegative"))

    Wm = Matrix{Float64}(weights)
    A = 0.5 .* (Wm .+ transpose(Wm))
    for i in 1:n
        A[i, i] = 0.0
    end
    two_m = sum(A)
    if two_m <= 0.0
        return (
            labels        = collect(1:n),
            modularity    = 0.0,
            n_communities = n,
        )
    end

    deg = vec(sum(A; dims = 2))
    modularity_of(community) = begin
        total = 0.0
        for i in 1:n
            for j in 1:n
                if community[i] == community[j]
                    total += A[i, j] - deg[i] * deg[j] / two_m
                end
            end
        end
        return total / two_m
    end

    community = collect(1:n)
    target = k == 0 ? 1 : k
    current = modularity_of(community)

    while length(Set(community)) > target
        groups = sort!(unique(community))
        best_gain = 0.0
        best_pair = nothing
        for ai in 1:length(groups)
            for aj in (ai + 1):length(groups)
                gi = groups[ai]
                gj = groups[aj]
                trial = Int[c == gj ? gi : c for c in community]
                gain = modularity_of(trial) - current
                if gain > best_gain
                    best_gain = gain
                    best_pair = (gi, gj)
                end
            end
        end
        if best_pair === nothing
            break
        end
        community = Int[c == best_pair[2] ? best_pair[1] : c for c in community]
        current = current + best_gain
    end

    order = sort!(unique(community))
    remap = Dict{Int,Int}(old => new for (new, old) in enumerate(order))
    labels = Int[remap[c] for c in community]
    return (
        labels        = labels,
        modularity    = current,
        n_communities = length(order),
    )
end

# -----------------------------------------------------------------------------
# Minimum cuts (Edmonds-Karp max-flow on the regional capacity graph)
# -----------------------------------------------------------------------------

"""
    region_min_cut(capacity, sources, sinks) -> NamedTuple

Minimum cut of a dense directed capacity graph, by Edmonds-Karp max-flow
between a super-source feeding all `sources` and a super-sink draining all
`sinks` (both with an effectively unbounded connecting capacity).

Returns:

- `capacity`: the max-flow value, i.e. the total capacity crossing the
  minimum cut. Units are whatever the caller encoded (expected crossers per
  step, relative flux, ...).
- `cut_edges::Vector{Tuple{Int,Int}}`: the saturated region-to-region edges
  `(u, v)` with `u` on the source side and `v` on the sink side.
- `source_side::Vector{Bool}`: regions reachable from the sources in the
  final residual graph.

`sources` and `sinks` must be non-empty, disjoint and valid region indices.
"""
function region_min_cut(capacity::AbstractMatrix{<:Real},
                        sources::Vector{Int},
                        sinks::Vector{Int})
    n = size(capacity, 1)
    isempty(sources) && throw(ArgumentError("sources must be non-empty"))
    isempty(sinks) && throw(ArgumentError("sinks must be non-empty"))
    all(r -> 1 <= r <= n, sources) || throw(ArgumentError(
        "sources outside 1..$n"))
    all(r -> 1 <= r <= n, sinks) || throw(ArgumentError("sinks outside 1..$n"))
    for r in sources
        if r in sinks
            throw(ArgumentError(
                "a region cannot be both a source and a sink of the same cut"))
        end
    end
    all(isfinite, capacity) || throw(DomainError("capacity contains NaN/Inf"))
    minimum(capacity) >= 0 || throw(DomainError("capacity must be nonnegative"))

    BIG = 1.0e9
    total = n + 2
    SOURCE = n + 1
    SINK = n + 2
    cap = zeros(Float64, total, total)
    for i in 1:n
        for j in 1:n
            cap[i, j] = Float64(capacity[i, j])
        end
    end
    for s in sources
        cap[SOURCE, s] = BIG
    end
    for t in sinks
        cap[t, SINK] = BIG
    end

    residual = copy(cap)
    parent = zeros(Int, total)

    function find_augmenting_path()
        for i in 1:total
            parent[i] = 0
        end
        parent[SOURCE] = SOURCE
        queue = Int[SOURCE]
        head = 1
        found = false
        while head <= length(queue) && !found
            u = queue[head]
            head += 1
            for v in 1:total
                if parent[v] == 0 && residual[u, v] > 1e-12
                    parent[v] = u
                    push!(queue, v)
                    if v == SINK
                        found = true
                        break
                    end
                end
            end
        end
        return found
    end

    flow = 0.0
    while find_augmenting_path()
        bottleneck = Inf
        v = SINK
        while v != SOURCE
            u = parent[v]
            bottleneck = min(bottleneck, residual[u, v])
            v = u
        end
        v = SINK
        while v != SOURCE
            u = parent[v]
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck
            v = u
        end
        flow += bottleneck
        if flow >= BIG / 2
            break   # super-source saturated: the cut severs every path
        end
    end

    reached = fill(false, total)
    reached[SOURCE] = true
    queue = Int[SOURCE]
    head = 1
    while head <= length(queue)
        u = queue[head]
        head += 1
        for v in 1:total
            if !reached[v] && residual[u, v] > 1e-12
                reached[v] = true
                push!(queue, v)
            end
        end
    end

    cut_edges = Tuple{Int,Int}[]
    for u in 1:n
        if reached[u]
            for v in 1:n
                if !reached[v] && cap[u, v] > 1e-12
                    push!(cut_edges, (u, v))
                end
            end
        end
    end
    source_side = Bool[reached[i] for i in 1:n]
    return (
        capacity    = flow,
        cut_edges   = cut_edges,
        source_side = source_side,
    )
end

# =============================================================================
# Assessment orchestrator
# =============================================================================

"""
    compute_connectivity_assessment(loaded, kernels, params = (;);
        output_dir = nothing, region_map = nothing, region_labels = nothing,
        horizon = nothing, step_years = nothing, communities_k = 2,
        corridor_horizon = 50) -> NamedTuple

First-passage connectivity assessment over the resolved regions. Requires at
least 2 regions and throws a loud `DomainError` otherwise (the pipeline
records it with `_record_panel_skip`).

Computed from the fitted unit-level kernel `kernels.P_kernel`:

- the row-stochastic region transition matrix (`P_reg`),
- the exact mean first-passage-time matrix (`mfpt_steps`, `mfpt_years`),
- first-arrival CDFs over `horizon` steps (`arrival_prob`, `median_steps`,
  `median_years`),
- domain-wide corridor visit rates,
- a community partition (`communities_k`, or adaptive when `k = 0`),
- the globally weakest ordered region-pair cut and, when the partition has at
  least 2 communities, the cut separating the two largest communities, in
  units of expected crossers per step (`region_sizes * P_reg`).

`horizon` and `step_years` default to `params.fpt_horizon_steps` and
`params.fpt_step_years`. When `output_dir` is a `String` the assessment also
writes `connectivity_region_matrix.csv`,
`connectivity_first_passage_summary.csv`, `connectivity_communities.csv`,
`connectivity_corridor_usage.csv` and `connectivity_min_cut.csv`. Every
artifact field is a plain Array or primitive, so the result round-trips
through JLD2.
"""
function compute_connectivity_assessment(
    loaded::NamedTuple,
    kernels::NamedTuple,
    params = (;);
    output_dir::Union{Nothing, String} = nothing,
    region_map::Union{Nothing, Vector{Int}} = nothing,
    region_labels::Union{Nothing, Vector{String}} = nothing,
    horizon::Union{Nothing, Int} = nothing,
    step_years::Union{Nothing, Float64} = nothing,
    communities_k::Union{Nothing, Int} = nothing,
    corridor_horizon::Int = 50
)
    partition, labels = resolve_region_partition(
        loaded, params;
        region_map = region_map,
        region_labels = region_labels,
    )
    n_units = loaded.n_spatial
    n_regions = length(labels)
    if n_regions < 2
        throw(DomainError(n_regions,
            "connectivity assessment needs >= 2 regions (no region_polygon_files " *
            "or region_map configured?)"))
    end
    region_sizes = Int[count(==(r), partition) for r in 1:n_regions]

    P_kernel = kernels.P_kernel
    if P_kernel isa AbstractVector
        throw(ArgumentError(
            "connectivity assessment: a dynamic kernel vector is not supported; " *
            "pass P_kernel as a single (annual) kernel"))
    end
    size(P_kernel, 1) == n_units || throw(ArgumentError(
        "kernel size $(size(P_kernel, 1)) does not match n_spatial $n_units"
    ))

    P_reg = region_transition_matrix(P_kernel, partition)

    horizon_val = horizon
    if horizon_val === nothing && hasproperty(params, :fpt_horizon_steps)
        horizon_val = Int(params.fpt_horizon_steps)
    end
    horizon_val === nothing && (horizon_val = 100)

    step_years_val = step_years
    if step_years_val === nothing && hasproperty(params, :fpt_step_years)
        step_years_val = Float64(params.fpt_step_years)
    end
    step_years_val === nothing && (step_years_val = 1.0 / 12.0)
    step_years_val > 0.0 || throw(DomainError(step_years_val,
        "step_years must be positive"))

    communities_k_val = communities_k
    if communities_k_val === nothing && hasproperty(params, :n_communities)
        communities_k_val = Int(params.n_communities)
    end
    communities_k_val === nothing && (communities_k_val = 2)

    arrivals = first_arrival_cdfs(P_reg; horizon = horizon_val)
    mfpt_steps = mean_first_passage_times(P_reg)
    mfpt_years = mfpt_steps .* step_years_val
    median_years = arrivals.median_steps .* step_years_val
    corridor = corridor_visit_rate(P_kernel; horizon = corridor_horizon)
    comm = connectivity_communities(P_reg; k = communities_k_val)

    # Region capacities for the cut analysis: expected crossers per step given
    # a uniform walker distribution inside each source region
    # (region size x transition probability).
    cap = reshape(Float64.(region_sizes), :, 1) .* P_reg

    weakest = nothing
    for s in 1:n_regions
        for t in 1:n_regions
            if s == t
                continue
            end
            cut = region_min_cut(cap, [s], [t])
            if weakest === nothing || cut.capacity < weakest.capacity
                weakest = (
                    capacity  = cut.capacity,
                    cut_edges = cut.cut_edges,
                    source    = s,
                    sink      = t,
                )
            end
        end
    end

    community_cut = nothing
    if comm.n_communities >= 2
        group_a = findall(==(1), comm.labels)
        group_b = findall(==(2), comm.labels)
        cut = region_min_cut(cap, group_a, group_b)
        community_cut = (
            capacity  = cut.capacity,
            cut_edges = cut.cut_edges,
            source    = group_a,
            sink      = group_b,
        )
    end

    result = (
        region_map      = partition,
        region_labels   = labels,
        region_sizes    = region_sizes,
        n_regions       = n_regions,
        P_reg           = P_reg,
        horizon         = horizon_val,
        step_years      = step_years_val,
        mfpt_steps      = mfpt_steps,
        mfpt_years      = mfpt_years,
        arrival_prob    = arrivals.arrival_prob,
        median_steps    = arrivals.median_steps,
        median_years    = median_years,
        corridor_visits = corridor.visits,
        corridor_norm   = corridor.normalized,
        community_labels = comm.labels,
        modularity       = comm.modularity,
        n_communities    = comm.n_communities,
        weakest_pair     = weakest,
        community_cut    = community_cut,
        corridor_horizon = corridor_horizon,
    )

    if output_dir !== nothing
        mkpath(output_dir)
        export_connectivity_region_matrix(
            joinpath(output_dir, "connectivity_region_matrix.csv"), result)
        export_connectivity_first_passage_summary(
            joinpath(output_dir, "connectivity_first_passage_summary.csv"),
            result)
        export_connectivity_communities_csv(
            joinpath(output_dir, "connectivity_communities.csv"), result)
        export_connectivity_corridor_usage(
            joinpath(output_dir, "connectivity_corridor_usage.csv"),
            result, loaded)
        export_connectivity_min_cut(
            joinpath(output_dir, "connectivity_min_cut.csv"), result)
    end
    return result
end

# =============================================================================
# CSV exporters
# =============================================================================

"""
    export_connectivity_region_matrix(path, result)

Write the row-stochastic region transition matrix as
`source_region,sink_region,transition_prob` rows.
"""
function export_connectivity_region_matrix(path::AbstractString,
                                           result::NamedTuple)
    K = result.n_regions
    source_region = String[]
    sink_region = String[]
    transition_prob = Float64[]
    for i in 1:K
        for j in 1:K
            if i != j
                push!(source_region, result.region_labels[i])
                push!(sink_region, result.region_labels[j])
                push!(transition_prob, result.P_reg[i, j])
            end
        end
    end
    CSV.write(path, DataFrame(;
        source_region = source_region,
        sink_region = sink_region,
        transition_prob = transition_prob,
    ))
    return path
end

"""
    export_connectivity_first_passage_summary(path, result)

Write, for every ordered pair, the within-horizon arrival probability, the
median first-arrival step and its year equivalent. Pair medians that are `NaN`
(not reached within the horizon) are exported as empty cells so spreadsheets
do not confuse "slow" with "never".
"""
function export_connectivity_first_passage_summary(path::AbstractString,
                                                   result::NamedTuple)
    K = result.n_regions
    source_region = String[]
    sink_region = String[]
    arrival_prob = Float64[]
    median_arrival_steps = Vector{Union{Missing,Float64}}()
    median_arrival_years = Vector{Union{Missing,Float64}}()
    for i in 1:K
        for j in 1:K
            if i == j
                continue
            end
            push!(source_region, result.region_labels[i])
            push!(sink_region, result.region_labels[j])
            push!(arrival_prob, result.arrival_prob[i, j])
            if isnan(result.median_steps[i, j])
                push!(median_arrival_steps, missing)
                push!(median_arrival_years, missing)
            else
                push!(median_arrival_steps, result.median_steps[i, j])
                push!(median_arrival_years, result.median_years[i, j])
            end
        end
    end
    CSV.write(path, DataFrame(;
        source_region = source_region,
        sink_region = sink_region,
        arrival_prob = arrival_prob,
        median_arrival_steps = median_arrival_steps,
        median_arrival_years = median_arrival_years,
    ))
    return path
end

"""
    export_connectivity_communities_csv(path, result)

Write the community assignment of each region with its unit count.
"""
function export_connectivity_communities_csv(path::AbstractString,
                                             result::NamedTuple)
    rows_region = String[]
    rows_community = Int[]
    rows_units = Int[]
    for r in 1:result.n_regions
        push!(rows_region, result.region_labels[r])
        push!(rows_community, result.community_labels[r])
        push!(rows_units, result.region_sizes[r])
    end
    CSV.write(path, DataFrame(;
        region = rows_region,
        community = rows_community,
        units = rows_units,
    ))
    return path
end

"""
    export_connectivity_corridor_usage(path, result, loaded)

Write per-unit expected visit rates for the domain-wide corridor march,
including lon/lat centroids when the mesh exposes them.
"""
function export_connectivity_corridor_usage(path::AbstractString,
                                            result::NamedTuple,
                                            loaded::NamedTuple)
    n = loaded.n_spatial
    cents = nothing
    if hasproperty(loaded.mesh, :centroids_lonlat)
        cents = loaded.mesh.centroids_lonlat
    end
    units = Int[]
    lons = Vector{Union{Missing,Float64}}()
    lats = Vector{Union{Missing,Float64}}()
    region_col = String[]
    visits_col = Float64[]
    usage_col = Float64[]
    for i in 1:n
        push!(units, i)
        lon_missing = true
        lat_missing = true
        lon_val = 0.0
        lat_val = 0.0
        if cents !== nothing && length(collect(cents)) >= n
            c = collect(cents)[i]
            lon_val = Float64(c[1])
            lat_val = Float64(c[2])
            lon_missing = false
            lat_missing = false
        end
        if lon_missing
            push!(lons, missing)
        else
            push!(lons, lon_val)
        end
        if lat_missing
            push!(lats, missing)
        else
            push!(lats, lat_val)
        end
        push!(region_col, result.region_labels[result.region_map[i]])
        push!(visits_col, result.corridor_visits[i])
        push!(usage_col, result.corridor_norm[i])
    end
    CSV.write(path, DataFrame(;
        unit = units,
        lon = lons,
        lat = lats,
        region = region_col,
        visits_per_step = visits_col,
        usage_relative = usage_col,
    ))
    return path
end

"""
    export_connectivity_min_cut(path, result)

Write the minimum cut(s): one block for the globally weakest ordered pair and
one for the community-separating cut when available.
"""
function export_connectivity_min_cut(path::AbstractString, result::NamedTuple)
    cut_group = String[]
    edge = String[]
    cut_capacity = Float64[]
    edge_capacity = Float64[]
    function add_rows!(group::String, cut)
        if cut === nothing
            return
        end
        for pair in cut.cut_edges
            u = pair[1]
            v = pair[2]
            push!(cut_group, group)
            label = string(result.region_labels[u], "->", result.region_labels[v])
            push!(edge, label)
            push!(cut_capacity, cut.capacity)
            push!(edge_capacity, result.region_sizes[u] * result.P_reg[u, v])
        end
    end
    add_rows!("weakest_pair", result.weakest_pair)
    add_rows!("community_cut", result.community_cut)
    CSV.write(path, DataFrame(;
        cut_group = cut_group,
        edge = edge,
        cut_capacity = cut_capacity,
        edge_capacity = edge_capacity,
    ))
    return path
end
