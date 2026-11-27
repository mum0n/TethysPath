"""
    movement.jl

Advection-Diffusion-Reaction (ADR) spatiotemporal movement, spatial telemetry,
and particle trajectory simulation engine for the TethysPath package.

Version: v1.0.0
"""

"""
    TelemetryData <: AbstractMatrix{Float64}

Structured container for mark-recapture telemetry observation events. Supports categorical
group-stratified transition modeling while maintaining matrix indexing compatibility.

# Fields
- `releases`: Vector of unit indices at initial detection / release (1-indexed).
- `recaps`: Vector of unit indices at subsequent detection / recapture (1-indexed).
- `ks`: Elapsed discrete time intervals between consecutive detections.
- `groups`: Biological stratum / group identifiers (1-indexed integers).
- `covariates`: Individual-level continuous covariates.
- `G`: Total count of distinct biological groups.
- `max_k`: Maximum observed elapsed transition step count.
- `matrix`: `Matrix{Float64}` representation of size `(N, 4)`.
"""
struct TelemetryData <: AbstractMatrix{Float64}
    releases::Vector{Int}
    recaps::Vector{Int}
    ks::Vector{Int}
    covariates::Vector{Float64}
    max_k::Int
    matrix::Matrix{Float64}
end

Base.size(td::TelemetryData) = size(td.matrix)
Base.size(td::TelemetryData, d::Int) = size(td.matrix, d)
Base.getindex(td::TelemetryData, i::Int, j::Int) = td.matrix[i, j]
Base.getindex(td::TelemetryData, i::Int) = td.matrix[i]
Base.IndexStyle(::Type{TelemetryData}) = IndexLinear()

TelemetryData(input::Union{DataFrame, AbstractMatrix}; kwargs...) =
    _process_telemetry_data(input; kwargs...)

"""
    _process_telemetry_data(telemetry_input; mark_recapture_G=nothing)

Transforms input telemetry data (DataFrame or Matrix) into a validated `TelemetryData`
structure for movement modeling. Supports both longitudinal event sequences and
pre-aggregated transition event pairs.

The model is pooled, so no group index is derived and no demographic stratification
is applied. A `group`/`groups` column, or a fifth matrix column, is ignored; `sex`
and `mat` style covariates survive only if passed through `covariate`.
`mark_recapture_G` is accepted and ignored so that older call sites keep working.
"""
function _process_telemetry_data(telemetry_input; mark_recapture_G=nothing)
    if telemetry_input isa TelemetryData
        return telemetry_input
    elseif telemetry_input isa AbstractMatrix
        mat = Matrix{Float64}(telemetry_input)
        n_rows = size(mat, 1)
        rel = n_rows > 0 ? Int.(mat[:, 1]) : Int[]
        rec = n_rows > 0 && size(mat, 2) >= 2 ? Int.(mat[:, 2]) : Int[]
        ks  = n_rows > 0 && size(mat, 2) >= 3 ? Int.(mat[:, 3]) : Int[]
        cov = n_rows > 0 && size(mat, 2) >= 4 ? mat[:, 4] : zeros(Float64, n_rows)
        max_k_val = isempty(ks) ? 1 : maximum(ks)
        return TelemetryData(rel, rec, ks, cov, max_k_val, mat)

    elseif telemetry_input isa DataFrame
        df = telemetry_input
        has_rel = hasproperty(df, :release) || hasproperty(df, :releases)
        has_rec = hasproperty(df, :recapture) || hasproperty(df, :recaps)

        if has_rel && has_rec
            rel_col = hasproperty(df, :release) ? :release : :releases
            rec_col = hasproperty(df, :recapture) ? :recapture : :recaps
            k_col = hasproperty(df, :k) ? :k : (hasproperty(df, :ks) ? :ks : nothing)
            cov_col = hasproperty(df, :covariate) ? :covariate :
                (hasproperty(df, :individual_covariate) ? :individual_covariate : nothing)

            rel = Int.(df[!, rel_col])
            rec = Int.(df[!, rec_col])
            ks  = !isnothing(k_col) ? Int.(df[!, k_col]) : ones(Int, nrow(df))
            cov = !isnothing(cov_col) ? Float64.(df[!, cov_col]) : zeros(Float64, nrow(df))
            max_k_val = isempty(ks) ? 1 : maximum(ks)
            mat = hcat(Float64.(rel), Float64.(rec), Float64.(ks), cov)
            return TelemetryData(rel, rec, ks, cov, max_k_val, mat)
        end

        time_col = if hasproperty(df, :time)
            :time
        elseif hasproperty(df, :timestamp)
            :timestamp
        else
            _detect_time_column(df; allow_nothing=true)
        end

        tag_col = nothing
        for cand in [:tagid, :tag_id, :tag, :id, :individual_id, :animal_id]
            if hasproperty(df, cand)
                tag_col = cand
                break
            end
        end

        s_col = if hasproperty(df, :s_idx)
            :s_idx
        else
            _detect_spatial_unit_column(df; allow_nothing=true)
        end

        if isnothing(time_col) || isnothing(tag_col) || isnothing(s_col)
            error("Telemetry DataFrame must contain either event pairs " *
                  "(:release, :recapture) or longitudinal observations " *
                  "(tag, spatial unit, and time).")
        end

        releases = Int[]
        recaps = Int[]
        ks = Int[]
        covariates = Float64[]

        cov_col = hasproperty(df, :individual_covariate) ? :individual_covariate :
            (hasproperty(df, :covariate) ? :covariate : nothing)

        _t_val(v) = v isa Dates.TimeType ?
            Float64(Dates.datetime2epochms(DateTime(v))) / 86400000.0 : Float64(v)

        gdf = groupby(df, tag_col)
        for sub_df in gdf
            if nrow(sub_df) < 2
                continue
            end
            sub_sorted = sort(sub_df, [order(time_col)])

            for i in 1:(nrow(sub_sorted) - 1)
                row_rel = sub_sorted[i, :]
                row_rec = sub_sorted[i+1, :]
                push!(releases, Int(getproperty(row_rel, s_col)))
                push!(recaps, Int(getproperty(row_rec, s_col)))
                t_rel = _t_val(getproperty(row_rel, time_col))
                t_rec = _t_val(getproperty(row_rec, time_col))
                push!(ks, max(1, round(Int, t_rec - t_rel)))
                cov_val = !isnothing(cov_col) ? Float64(getproperty(row_rel, cov_col)) : 0.0
                push!(covariates, cov_val)
            end
        end

        n_events = length(releases)
        mat = if n_events > 0
            hcat(Float64.(releases), Float64.(recaps), Float64.(ks), covariates)
        else
            Matrix{Float64}(undef, 0, 4)
        end
        max_k_val = isempty(ks) ? 1 : maximum(ks)
        return TelemetryData(releases, recaps, ks, covariates, max_k_val, mat)
    else
        error("Unsupported format for mark_recapture_data: $(typeof(telemetry_input)). " *
              "Expected DataFrame or Matrix.")
    end
end

"""
    simulate_correlated_density_vector(habitat_prob, rho_target, log_mu, sigma_resid;
                                        rng=Random.GLOBAL_RNG)

Simulates a spatial population density vector correlated with habitat suitability index (HSI).

# Arguments
- `habitat_prob::AbstractVector{<:Real}`: Habitat suitability probabilities for spatial units.
- `rho_target::Real`: Desired correlation between habitat suitability and log-density.
- `log_mu::Real`: Mean of the log-density field.
- `sigma_resid::Real`: Marginal residual standard deviation of log-density.
- `rng::AbstractRNG`: Random number generator.

# Returns
- `Vector{Float64}`: Simulated positive population density values.
"""
function simulate_correlated_density_vector(
    habitat_prob::AbstractVector{<:Real},
    rho_target::Real,
    log_mu::Real,
    sigma_resid::Real;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)
    n = length(habitat_prob)
    p_std = (habitat_prob .- mean(habitat_prob)) ./ (std(habitat_prob) + 1e-9)
    epsilon = randn(rng, n)
    log_n_signal = (rho_target .* p_std) .+ (sqrt(max(0.0, 1.0 - rho_target^2)) .* epsilon)
    log_n = log_mu .+ (log_n_signal .* sigma_resid)
    return exp.(log_n)
end

"""
    generate_ADR_simulation_bundle(domain_size, n_units, n_years, n_marks;
                                   area_method=:hexagonal, rng=Random.GLOBAL_RNG)

Generates a complete synthetic simulation bundle for joint Advection-Diffusion-Reaction (ADR)
population density surveys and individual mark-recapture telemetry.

# Arguments
- `domain_size::Real`: Spatial bounding box size (e.g. 1000.0 km).
- `n_units::Int`: Target number of spatial partitioning units.
- `n_years::Int`: Number of discrete temporal observation years.
- `n_marks::Int`: Number of tagged individuals released in mark-recapture telemetry.
- `area_method::Symbol`: Spatial grid method (`:hexagonal`, `:cvt`, `:voronoi`, `:grid`).
- `rng::AbstractRNG`: Random number generator instance.

# Returns
- `NamedTuple`:
  - `data::DataFrame`: Long-format observation table with columns `(:density, :unit_id, ...)`.
  - `telemetry_data::DataFrame`: Long-format telemetry observations with columns `(:tagid, ...)`.
  - `au::NamedTuple`: Spatial areal units object containing boundaries, centroids, and `W`.
  - `n_spatial::Int`: Number of spatial units.
  - `n_years::Int`: Number of temporal survey periods.
"""
function generate_ADR_simulation_bundle(
    domain_size::Real,
    n_units::Int,
    n_years::Int,
    n_marks::Int;
    area_method::Symbol = :cvt,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)
    # 1. Generate spatial partitioning using TethysPath's partitioning engine
    s_x_init = rand(rng, 1000) .* domain_size
    s_y_init = rand(rng, 1000) .* domain_size
    
    au = assign_spatial_units(
        s_x_init, s_y_init;
        area_method = area_method,
        target_units = n_units,
        exact_units = false
    )
    
    centroids = au.centroids
    n_spatial = length(centroids)
    cent_x = [c[1] for c in centroids]
    cent_y = [c[2] for c in centroids]

    # 2. Dynamic Spatiotemporal Habitat Suitability Index (HSI)
    habitat_p = zeros(Float64, n_spatial, n_years)
    center_x = domain_size / 2.0
    center_y = domain_size / 2.0
    
    p_init = [exp(-sqrt((cent_x[i] - center_x)^2 + (cent_y[i] - center_y)^2) / (domain_size
      / 3.0)) for i in 1:n_spatial]
    habitat_p[:, 1] = p_init ./ (maximum(p_init) + 1e-9)
    
    for t in 2:n_years
        noise = randn(rng, n_spatial) .* 0.05
        habitat_p[:, t] = min.(1.0, max.(0.01, habitat_p[:, t-1] .+ noise))
    end

    # 3. Simulate Spatiotemporal Density Observations
    density_n = zeros(Float64, n_spatial, n_years)
    for t in 1:n_years
        density_n[:, t] = simulate_correlated_density_vector(habitat_p[:, t], 0.75, 3.5,
          0.6; rng=rng)
    end

    # 4. Simulate Individual Mark-Recapture Telemetry Transitions
    telemetry_df = DataFrame(
        tagid = Int[],
        s_idx = Int[],
        time = Float64[],
        tag = Int[],
        individual_covariate = Float64[]
    )
    
    for i in 1:n_marks
        release_unit = rand(rng, 1:n_spatial)
        year_start = rand(rng, 1:max(1, n_years - 2))
        time_steps = rand(rng, 1:2)
        
        # Simulate physical dispersal distance
        d_travel = rand(rng, Uniform(domain_size / 20.0, domain_size / 2.5)) * time_steps
        dists = [abs(sqrt((cent_x[release_unit] - cent_x[j])^2 + (cent_y[release_unit] -
          cent_y[j])^2) - d_travel) for j in 1:n_spatial]
        recapture_unit = argmin(dists)
        
        ind_cov = randn(rng)

        # Release event (tag = 0)
        push!(telemetry_df, (tagid=i, s_idx=release_unit, time=Float64(year_start), tag=0,
          individual_covariate=ind_cov))
        # Recapture event (tag = 1)
        push!(telemetry_df, (tagid=i, s_idx=recapture_unit, time=Float64(year_start +
          time_steps), tag=1, individual_covariate=ind_cov))
    end

    # 5. Build Long-Format Survey DataFrame
    df = DataFrame()
    for t in 1:n_years
        temp_df = DataFrame(
            density = density_n[:, t],
            unit_id = 1:n_spatial,
            s_idx = 1:n_spatial,
            year = t,
            time_idx = t,
            s_x = cent_x,
            s_y = cent_y,
            habitat_p = habitat_p[:, t]
        )
        append!(df, temp_df)
    end

    return (
        data = df,
        telemetry_data = telemetry_df,
        au = au,
        n_spatial = n_spatial,
        n_years = n_years
    )
end

"""
    compute_velocity_field(prob_vec, grid_dim, strength; mode=:exponential)

Computes an advection velocity vector field from a spatial gradient of habitat suitability.

# Arguments
- `prob_vec::AbstractVector{<:Real}`: Spatial habitat suitability values.
- `grid_dim::Int`: Dimension of regular grid (for lattice geometries).
- `strength::Real`: Scaling factor for advection velocity.
- `mode::Symbol`: `:exponential` (relative gradient) or `:linear` (absolute gradient).

# Returns
- `NamedTuple`: `(vx = vec(vx), vy = vec(vy))` velocity components.
"""
function compute_velocity_field(
    prob_vec::AbstractVector{<:Real},
    grid_dim::Int,
    strength::Real;
    mode::Symbol = :exponential
)
    grid = reshape(prob_vec, grid_dim, grid_dim)
    rows, cols = size(grid)
    vx = zeros(Float64, rows, cols)
    vy = zeros(Float64, rows, cols)
    eps_val = 1e-6

    for r in 1:rows, c in 1:cols
        gx = (c == 1) ? (grid[r, 2] - grid[r, 1]) :
             (c == cols ? (grid[r, cols] - grid[r, cols-1]) : ((grid[r, c+1] - grid[r, c-1]) / 2.0))
        gy = (r == 1) ? (grid[2, c] - grid[1, c]) :
             (r == rows ? (grid[rows, c] - grid[rows-1, c]) : ((grid[r+1, c] - grid[r-1, c]) / 2.0))

        if mode == :exponential
            denom_x = sqrt(grid[r, c] * (grid[r, c] + eps_val))
            denom_y = sqrt(grid[r, c] * (grid[r, c] + eps_val))
            vx[r, c] = (gx / denom_x) * strength
            vy[r, c] = (gy / denom_y) * strength
        else
            vx[r, c] = gx * strength
            vy[r, c] = gy * strength
        end
    end
    return (vx = vec(vx), vy = vec(vy))
end

"""
    calculate_multistep_transition(Gamma_base::AbstractMatrix{T}, steps::Int) where T <: Real

Calculates the multi-step dispersal transition matrix via Markov matrix exponentiation:
\$\\Gamma^{(k)} = \\Gamma^k\$.
"""
function calculate_multistep_transition(Gamma_base::AbstractMatrix{T}, steps::Int) where T <: Real
    n_spatial = size(Gamma_base, 1)
    if steps < 1
        return Matrix{T}(I, n_spatial, n_spatial)
    end
    
    G_step = copy(Gamma_base)
    for i in 1:n_spatial
        row_sum = sum(G_step[i, :])
        if row_sum > 0
            G_step[i, :] ./= row_sum
        end
    end
    
    Gamma_k = G_step ^ steps
    
    for i in 1:n_spatial
        final_sum = sum(Gamma_k[i, :])
        if final_sum > 0
            Gamma_k[i, :] ./= final_sum
        end
    end
    
    return Gamma_k
end

"""
    simulate_posterior_trajectories(Gamma_base, start_units, n_steps, au_context;
                                   rho_persistence=0.0, rng=Random.GLOBAL_RNG)

Simulates individual movement trajectories from a stationary transition matrix \$\\Gamma\$,
with optional directional persistence (Correlated Random Walk / CRW).

# Arguments
- `Gamma_base::AbstractMatrix`: Transition probability matrix (\$S \\times S\$).
- `start_units::Vector{Int}`: Starting spatial unit indices for each tracked individual.
- `n_steps::Int`: Number of discrete forward movement steps.
- `au_context::NamedTuple`: Spatial units object containing `centroids`.
- `rho_persistence::Real`: Directional persistence parameter (\$\\rho \\ge 0\$).
- `rng::AbstractRNG`: Random number generator.

# Returns
- `Matrix{Int}`: Matrix of shape `(n_indiv, n_steps + 1)` with unit indices visited over time.
"""
function simulate_posterior_trajectories(
    Gamma_base::AbstractMatrix{T},
    start_units::Vector{Int},
    n_steps::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
) where T <: Real
    
    n_indiv = length(start_units)
    n_spatial = size(Gamma_base, 1)
    centroids = if hasproperty(au_context, :centroids)
        au_context.centroids
    elseif hasproperty(au_context, :centroids_km)
        au_context.centroids_km
    elseif hasproperty(au_context, :centroids_lonlat)
        au_context.centroids_lonlat
    else
        throw(ArgumentError(
            "au_context must contain :centroids, :centroids_km, or :centroids_lonlat."
        ))
    end
    paths = zeros(Int, n_indiv, n_steps + 1)
    paths[:, 1] = start_units
    
    G_sampling = copy(Gamma_base)
    for i in 1:n_spatial
        G_sampling[i, :] .= max.(0.0, G_sampling[i, :])
        rs = sum(G_sampling[i, :])
        if rs > 1e-12
            G_sampling[i, :] ./= rs
        else
            G_sampling[i, :] .= 0.0
            G_sampling[i, i] = 1.0
        end
    end
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for t in 2:(n_steps + 1)
            p_row = max.(0.0, vec(G_sampling[curr_node, :]))
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                norm_prev = norm(v_prev)
                if norm_prev > 1e-9
                    for j in 1:n_spatial
                        if j != curr_node && p_row[j] > 0.0
                            v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                            norm_cand = norm(v_cand)
                            if norm_cand > 1e-9
                                cos_theta = dot(v_prev, v_cand) / (norm_prev * norm_cand)
                                p_row[j] *= exp(rho_persistence * cos_theta)
                            end
                        end
                    end
                end
            end
            row_sum = sum(p_row)
            if row_sum > 1e-12
                p_row ./= row_sum
            else
                p_row .= 0.0
                p_row[curr_node] = 1.0
            end
            next_node = rand(rng, Categorical(p_row))
            paths[i, t] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

"""
    simulate_mechanistic_trajectories(Gamma_sequence, start_units, t_start, au_context;
                                      rho_persistence=0.0, n_years_sim=1,
                                      rng=Random.GLOBAL_RNG)

Simulates individual movement paths through a dynamic, non-stationary environment where transition
kernels \$\\Gamma_t\$ vary over time.
"""
function simulate_mechanistic_trajectories(
    Gamma_sequence::Vector{<:AbstractMatrix{T}},
    start_units::Vector{Int},
    t_start::Int,
    au_context::NamedTuple;
    rho_persistence::Real = 0.0,
    n_years_sim::Int = 1,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
) where T <: Real

    n_indiv = length(start_units)
    n_spatial = size(Gamma_sequence[1], 1)
    centroids = if hasproperty(au_context, :centroids)
        au_context.centroids
    elseif hasproperty(au_context, :centroids_km)
        au_context.centroids_km
    elseif hasproperty(au_context, :centroids_lonlat)
        au_context.centroids_lonlat
    else
        throw(ArgumentError(
            "au_context must contain :centroids, :centroids_km, or :centroids_lonlat."
        ))
    end
    max_available_time = length(Gamma_sequence)
    
    actual_steps = n_years_sim
    if t_start + n_years_sim > max_available_time
        actual_steps = max_available_time - t_start
    end

    paths = zeros(Int, n_indiv, actual_steps + 1)
    paths[:, 1] = start_units
    
    for i in 1:n_indiv
        curr_node = start_units[i]
        prev_node = 0
        for step in 1:actual_steps
            t_current = t_start + step - 1
            if t_current > max_available_time
                break
            end
            Gamma_t = Gamma_sequence[t_current]
            p_row = max.(0.0, vec(Gamma_t[curr_node, :]))
            if rho_persistence > 0.0 && prev_node != 0
                v_prev = [centroids[curr_node][d] - centroids[prev_node][d] for d in 1:2]
                norm_prev = norm(v_prev)
                if norm_prev > 1e-9
                    for j in 1:n_spatial
                        if j != curr_node && p_row[j] > 0.0
                            v_cand = [centroids[j][d] - centroids[curr_node][d] for d in 1:2]
                            norm_cand = norm(v_cand)
                            if norm_cand > 1e-9
                                cos_theta = dot(v_prev, v_cand) / (norm_prev * norm_cand)
                                p_row[j] *= exp(rho_persistence * cos_theta)
                            end
                        end
                    end
                end
            end
            row_sum = sum(p_row)
            if row_sum > 1e-12
                p_row ./= row_sum
            else
                p_row .= 0.0
                p_row[curr_node] = 1.0
            end
            next_node = rand(rng, Categorical(p_row))
            paths[i, step + 1] = next_node
            prev_node = curr_node
            curr_node = next_node
        end
    end
    return paths
end

"""
    compute_suitability_transition_kernel(
        suitability_vec, W;
        sensitivity=1.0, diffusion_weight=0.1, relationship=:exponential
    )

Generates a spatial Markov transition probability kernel based on local habitat
suitability index (HSI) differences and network topology:
- `:exponential` (default): \$\\Gamma_{ij} \\propto \\exp(\\beta \\cdot \\text{HSI}_j) + D_{\\text{weight}}\$
- `:linear`: \$\\Gamma_{ij} \\propto (1 + \\beta \\cdot \\text{HSI}_j) + D_{\\text{weight}}\$
- `:logistic`: \$\\Gamma_{ij} \\propto \\frac{1}{1 + \\exp(-\\beta \\cdot \\text{HSI}_j)} + D_{\\text{weight}}\$

# Arguments
- `suitability_vec::AbstractVector{<:Real}`: Habitat Suitability Index (HSI) values (\$S\$).
- `W::AbstractMatrix`: Spatial adjacency weights matrix (\$S \\times S\$).
- `sensitivity::Real`: Advective response sensitivity (\$\\beta \\ge 0\$).
- `diffusion_weight::Real`: Baseline isotropic dispersal weight (\$D_{\\text{weight}} \\ge 0\$).
- `relationship::Symbol`: Functional relationship (`:exponential`, `:linear`, `:logistic`).

# Returns
- `SparseMatrixCSC{Float64, Int}`: Row-stochastic Markov transition probability matrix.
"""
function compute_suitability_transition_kernel(
    suitability_vec::AbstractVector{T},
    W::AbstractMatrix;
    sensitivity::Real = 1.0,
    diffusion_weight::Real = 0.1,
    relationship::Symbol = :exponential
) where T <: Real

    n_spatial = length(suitability_vec)
    
    rows = rowvals(W)
    vals = nonzeros(W)
    
    I_vec = Int[]
    J_vec = Int[]
    V_vec = T[]
    
    # Pre-calculate estimated number of non-zeros
    sizehint!(I_vec, nnz(W) + n_spatial)
    sizehint!(J_vec, nnz(W) + n_spatial)
    sizehint!(V_vec, nnz(W) + n_spatial)
    
    for i in 1:n_spatial
        h_i = suitability_vec[i]
        bias_i = if relationship == :exponential
            exp(sensitivity * h_i)
        elseif relationship == :logistic
            1.0 / (1.0 + exp(-sensitivity * h_i))
        else # :linear
            max(0.01, 1.0 + sensitivity * h_i)
        end
        
        push!(I_vec, i)
        push!(J_vec, i)
        push!(V_vec, bias_i)

        for j_idx in nzrange(W, i)
            j = rows[j_idx]
            if i == j
                continue
            end
            h_j = suitability_vec[j]
            suitability_bias = if relationship == :exponential
                exp(sensitivity * h_j)
            elseif relationship == :logistic
                1.0 / (1.0 + exp(-sensitivity * h_j))
            else # :linear
                max(0.01, 1.0 + sensitivity * h_j)
            end
            edge_weight = vals[j_idx]
            
            push!(I_vec, i)
            push!(J_vec, j)
            push!(V_vec, (suitability_bias + diffusion_weight) * edge_weight)
        end
    end
    
    Gamma = sparse(I_vec, J_vec, V_vec, n_spatial, n_spatial)
    
    # Efficient row-stochastic normalization for CSC matrix
    row_sums = zeros(T, n_spatial)
    for i in 1:length(Gamma.nzval)
        row_sums[Gamma.rowval[i]] += Gamma.nzval[i]
    end
    
    for i in 1:length(Gamma.nzval)
        r = Gamma.rowval[i]
        if row_sums[r] > 1e-12
            Gamma.nzval[i] /= row_sums[r]
        end
    end
    
    # Fix rows with zero sum (if any)
    for r in 1:n_spatial
        if row_sums[r] <= 1e-12
            Gamma[r, r] = 1.0
        end
    end
    
    return Gamma
end

"""
    calculate_regional_connectivity(Gamma, strata_definition)

Aggregates fine-scale spatial unit transition probabilities into a macro-regional
connectivity matrix.
"""
function calculate_regional_connectivity(Gamma::AbstractMatrix, strata_definition::AbstractVector)
    n_units = size(Gamma, 1)
    unique_strata = unique(strata_definition)
    n_strata = length(unique_strata)
    strata_map = Dict(s => i for (i, s) in enumerate(unique_strata))
    
    C = zeros(Float64, n_strata, n_strata)

    if Gamma isa SparseMatrixCSC
        for j in 1:n_units
            to_stratum_idx = strata_map[strata_definition[j]]
            for p in Gamma.colptr[j]:(Gamma.colptr[j+1]-1)
                i = Gamma.rowval[p]
                v = Gamma.nzval[p]
                from_stratum_idx = strata_map[strata_definition[i]]
                C[from_stratum_idx, to_stratum_idx] += v
            end
        end
    else
        for j in 1:n_units
            to_stratum_idx = strata_map[strata_definition[j]]
            for i in 1:n_units
                from_stratum_idx = strata_map[strata_definition[i]]
                C[from_stratum_idx, to_stratum_idx] += Gamma[i, j]
            end
        end
    end

    for r in 1:n_strata
        row_sum = sum(C[r, :])
        if row_sum > 0
            C[r, :] ./= row_sum
        end
    end

    return C
end

"""
    plot_ad_ratio_distribution(advection_field, diffusion_field; mode=:plots)

Generates a diagnostic histogram of the Advection-to-Diffusion ratio across spatial units.
Only `mode = :plots` is implemented; `:leaflet` / `:html` raise, because the
  figure they would produce is not written. See FIGURE_CONVENTIONS.md.
"""
function plot_ad_ratio_distribution(advection_field::AbstractVector{<:Real},
  diffusion_field::AbstractVector{<:Real}; mode::Symbol=:plots)
    mode === :leaflet || mode === :html && error(
        "The interactive Leaflet form of this figure is not implemented; the " *
        "conventions it would need are in FIGURE_CONVENTIONS.md. Use mode = :plots.")
    # One implementation, in the Plots extension. The base previously carried a
    # line-for-line copy of it, which drifted; delegating means one place to fix.
    # The extension loads when `Plots` is present, so a clear error beats a
    # `UndefVarError` naming an internal the caller never wrote.
    ext = Base.get_extension(TethysPath, :TethysPathPlottingExt)
    ext === nothing && error(
        "plot_ad_ratio_distribution needs `Plots`. Add Plots to this environment " *
        "and the TethysPathPlottingExt extension supplies the figure.")
    return ext._ad_ratio_histogram_via_plots(advection_field, diffusion_field)
end

"""
    synthesize_adr_results(chain_or_res, sim_data, vel_vectors; au=sim_data.au)

Performs post-processing, parameter extraction, and diagnostic visualization
for a fitted Advection-Diffusion-Reaction movement model.

# Returns
- `NamedTuple`: Containing:
  - `parameters::NamedTuple`: Posterior mean estimates for velocity, diffusion, sigma.
  - `propagator_matrix::Matrix{Float64}`: Reconstructed forward propagator \$M_{\\text{prop}}\$.
  - `transition_matrix::Matrix{Float64}`: Reconstructed one-step Markov transition matrix.
  - `plots::NamedTuple`: Rendered diagnostic plots (`regional_connectivity`, etc.).
"""
function synthesize_adr_results(
    chain_or_res,
    sim_data::NamedTuple,
    vel_vectors::NamedTuple;
    au = sim_data.au
)
    # 1. Extract Posterior Parameter Estimates
    S_strength_mean = if hasproperty(chain_or_res, :effects) &&
      hasproperty(chain_or_res.effects, :velocity)
        chain_or_res.effects.velocity.mean
    elseif hasproperty(chain_or_res, :value) && :velocity_movement in names(chain_or_res,
      :parameters)
        mean(chain_or_res[:velocity_movement])
    else
        1.0
    end

    D_coeff_mean = if hasproperty(chain_or_res, :effects) &&
      hasproperty(chain_or_res.effects, :diffusion)
        chain_or_res.effects.diffusion.mean
    elseif hasproperty(chain_or_res, :value) && :diffusion_movement in names(chain_or_res,
      :parameters)
        mean(chain_or_res[:diffusion_movement])
    else
        0.5
    end

    # 2. Reconstruct Spatial Graph Operators
    W = au.W
    n_spatial = size(W, 1)
    
    # Graph Laplacian L = D_deg - W
    deg = vec(sum(W, dims=2))
    L = spdiagm(0 => deg) - W
    
    # Directed Advection Operator A
    W_dir = tril(W, -1)
    out_deg = vec(sum(W_dir, dims=2))
    D_inv = spdiagm(0 => 1.0 ./ (out_deg .+ 1e-9))
    A = D_inv * W_dir
    
    # Propagator M_prop = I - v*A - D*L
    # NOTE: Inverting M_prop to get a transition matrix assumes that the propagator 
    # defines a solvable linear system. However, for advection-diffusion on bounded domains, 
    # this may not be the intended transition kernel. The transition matrix should emerge 
    # directly from discretization, not from inverting a propagator.
    # Fix: Verify this is the intended mathematical model, or construct 
    # Gamma_mean directly from the advection/diffusion operators without inversion.

    M_prop_mean = Matrix(I(n_spatial) - (S_strength_mean .* A) - (D_coeff_mean .* L))
    Gamma_mean = inv(M_prop_mean)
    
    # Normalize rows of Gamma
    for i in 1:n_spatial
        rs = sum(Gamma_mean[i, :])
        if rs > 0
            Gamma_mean[i, :] ./= rs
        else
            Gamma_mean[i, i] = 1.0
        end
    end

    # 3. Visualization 1: Regional Connectivity Matrix
    cents_x = [c[1] for c in au.centroids]
    mid_x = (minimum(cents_x) + maximum(cents_x)) / 2.0
    groups = [x > mid_x ? "East" : "West" for x in cents_x]
    conn_mat = calculate_regional_connectivity(Gamma_mean, groups)
    
    plt_conn = Plots.heatmap(
        ["West", "East"], ["West", "East"], conn_mat,
        title = "Regional Transfer Probability Matrix",
        xlabel = "To Region", ylabel = "From Region", color = :viridis, clims = (0, 1)
    )

    # 4. Visualization 2: Advection-to-Diffusion Ratio Distribution
    advection_magnitude = sqrt.(vel_vectors.vx.^2 .+ vel_vectors.vy.^2) .* S_strength_mean
    diffusion_magnitude = fill(D_coeff_mean, n_spatial)
    plt_ad = plot_ad_ratio_distribution(advection_magnitude, diffusion_magnitude)

    # 5. Visualization 3: Path Simulation with & without Persistence
    random_starts = rand(1:n_spatial, min(4, n_spatial))
    n_sim_steps = 15
    paths_standard = simulate_posterior_trajectories(Gamma_mean, random_starts, n_sim_steps,
      au; rho_persistence=0.0)
    paths_persistent = simulate_posterior_trajectories(Gamma_mean, random_starts,
      n_sim_steps, au; rho_persistence=2.0)
    
    plt_comp = Plots.plot(layout=(1, 2), size=(900, 450), aspect_ratio=:equal)
    render_paths!(plt_comp[1], paths_standard; au=au, color=:crimson, lw=1.5, labels=["Mark
      $i" for i in 1:length(random_starts)])
    Plots.plot!(plt_comp[1], title="Standard Markovian Random Walk")
    
    render_paths!(plt_comp[2], paths_persistent; au=au, color=:navy, lw=1.5, labels=["Mark
      $i" for i in 1:length(random_starts)])
    Plots.plot!(plt_comp[2], title="Persistent Movement (\\rho=2.0)")

    # 6. Visualization 4: Dynamic Multi-Year Projections
    gamma_seq = [Gamma_mean for _ in 1:sim_data.n_years]
    dynamic_paths = simulate_mechanistic_trajectories(gamma_seq, random_starts, 1, au;
      rho_persistence=1.5, n_years_sim=min(4, sim_data.n_years - 1))
    
    plt_dyn = Plots.plot(aspect_ratio=:equal, title="Multi-Year Mechanistic Path
      Projections", legend=:outerright)
    render_paths!(plt_dyn, dynamic_paths; au=au, color=:darkgreen, lw=1.5)

    # 7. Visualization 5: Interactive Leaflet Dashboard
    dash_html = leaflet_movement_dashboard(
        (au = au, transition_matrix = Gamma_mean, opts = (hsi = nothing,),
         d_val = D_coeff_mean, v_val = S_strength_mean),
        paths_persistent;
        groups = groups,
        title = "ADR Movement Estimation Dashboard"
    )

    plots_bundle = (
        regional_connectivity = plt_conn,
        ad_ratio = plt_ad,
        paths_comparison = plt_comp,
        dynamic_paths = plt_dyn,
        leaflet_dashboard = dash_html
    )

    params_bundle = (
        velocity_mean = S_strength_mean,
        diffusion_mean = D_coeff_mean
    )

    return (
        parameters = params_bundle,
        propagator_matrix = M_prop_mean,
        transition_matrix = Gamma_mean,
        plots = plots_bundle
    )
end

# ==============================================================================
# SECTION: GENERAL TELEMETRY & POSTERIOR TRANSITION KERNEL UTILITIES
# ==============================================================================

"""
    validate_telemetry(df::DataFrame) -> nothing

Validates that a telemetry / mark-recapture DataFrame conforms to TethysPath movement
schema requirements and logical consistency constraints.

# Validation Checks
1. Presence of required identifier, coordinate, and detection type columns:
   `:tagid`, `:lon`, `:lat`, `:tag`, and either `:timestamp` or `:time`.
2. Valid detection codes: `:tag ≥ 0` (where `0` indicates initial release/marking,
   and `n ≥ 1` indicates the \$n\$-th recapture event).
3. Individual trajectory consistency: exactly one release event (`tag == 0`) per
   `:tagid`, and all recapture events chronologically follow the initial release.

Throws an informative `ArgumentError` if any check fails.
"""
function validate_telemetry(df::DataFrame)
    time_col = if hasproperty(df, :time)
        :time
    elseif hasproperty(df, :timestamp)
        :timestamp
    else
        nothing
    end

    if isnothing(time_col)
        throw(ArgumentError("Telemetry DataFrame must contain either :time or :timestamp."))
    end

    required = [:tagid, :lon, :lat, :tag]
    missing_cols = filter(c -> !hasproperty(df, c), required)
    if !isempty(missing_cols)
        throw(ArgumentError("Telemetry DataFrame missing required columns: $(missing_cols)"))
    end

    if any(df.tag .< 0)
        throw(ArgumentError("Column :tag must be ≥ 0 (0 = initial mark, n ≥ 1 = recaptures)."))
    end

    for sub in groupby(df, :tagid)
        release_rows = filter(r -> r.tag == 0, sub)
        if nrow(release_rows) != 1
            tid = first(sub.tagid)
            throw(ArgumentError("tagid=$(tid): expected exactly one release event (tag=0), " *
                                "found $(nrow(release_rows))."))
        end
        release_time = Float64(release_rows[1, time_col])
        recapture_times = Float64.(filter(r -> r.tag > 0, sub)[!, time_col])
        if !all(recapture_times .> release_time)
            tid = first(sub.tagid)
            throw(ArgumentError("tagid=$(tid): one or more recaptures precede or coincide with release."))
        end
    end
    return nothing
end

"""
    map_point_to_units(df::DataFrame, au::NamedTuple)::DataFrame

Projects observation coordinates `(:lon, :lat)` in a telemetry DataFrame to
their nearest active, navigable spatial unit centroids, returning a copy of `df`
with an appended `:s_idx` column.

When `au` contains `:land_mask` or `:W`, mapping is restricted strictly to active
water units (`!land_mask && degree(W) > 0`), ensuring observations near coastlines
or outside the survey domain snap to valid water units rather than land barriers.

# Mathematical Formulation
For each observation \$\\mathbf{x}_i = (\\text{lon}_i, \\text{lat}_i)\$:
\$s_i = \\arg\\min_{j \\in \\mathcal{M}} \\|\\mathbf{x}_i - \\mathbf{c}_j\\|_2\$
where \$\\mathcal{M} = \\{ j \\mid \\neg \\text{land\\_mask}[j] \\land \\text{deg}(j) > 0 \\}\$.
"""
function map_point_to_units(
    df::DataFrame, au::NamedTuple;
    target_col::Union{Symbol, AbstractString, Nothing} = nothing
)::DataFrame
    land_mask = hasproperty(au, :land_mask) ? au.land_mask : nothing
    W = hasproperty(au, :W) ? au.W : nothing
    if hasproperty(au, :centroids_km) &&
       hasproperty(au, :center_lon) &&
       hasproperty(au, :center_lat)
        return map_point_to_units(
            df, au.centroids_km, au.center_lon, au.center_lat;
            target_col = target_col, land_mask = land_mask, W = W
        )
    end
    cents = hasproperty(au, :centroids) ? au.centroids :
            (hasproperty(au, :centroids_lonlat) ? au.centroids_lonlat : nothing)
    if cents === nothing
        throw(ArgumentError("au must contain :centroids or :centroids_km."))
    end
    S = length(cents)
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end
    active_cents = cents[active_indices]
    col_x, col_y = _detect_xy_columns(df)
    mapped_local = map_to_units(df[!, col_x], df[!, col_y], active_cents)
    out = copy(df)
    mapped_s = [active_indices[k] for k in mapped_local]
    out[!, :s_idx] = mapped_s
    if !isnothing(target_col)
        out[!, Symbol(target_col)] = mapped_s
    end
    return out
end

"""
    time_steps_between(t_release::Real, t_recapture::Real)::Int

Converts continuous decimal-year temporal difference to discrete integer time steps.
Enforces a minimum interval of 1 step: \$\\Delta t = \\max(1, \\lfloor t_{\\text{rec}} - t_{\\text{rel}} \\rceil)\$.
"""
function time_steps_between(t_release::Real, t_recapture::Real)::Int
    return max(1, round(Int, t_recapture - t_release))
end

"""
    extract_scalar_param(chain, param_prefix::String)::Vector{Float64}

Extracts posterior samples for a scalar parameter identified by `param_prefix`
(e.g., `"velocity"` or `"diffusion"`), resolving variable name suffixes dynamically
across MCMC chains, FlexiChains, or DataFrames.

# Suffix Resolution Precedence
1. Suffix matching formula indices: `"\$(param_prefix)_s_idx_t_idx"`
2. Suffix matching module name: `"\$(param_prefix)_movement"`
3. Exact parameter name: `param_prefix`
4. Any sampled parameter starting with `param_prefix`
"""
function extract_scalar_param(chain, param_prefix::String)::Vector{Float64}
    if chain isa NamedTuple || chain isa AbstractDict
        for k in keys(chain)
            sk = string(k)
            if sk == param_prefix || sk == "$(param_prefix)_s_idx_t_idx" ||
               sk == "$(param_prefix)_movement" || startswith(sk, param_prefix)
                val = chain[k]
                return isa(val, AbstractVector) ? Float64.(val) : [Float64(val)]
            end
        end
    end

    p_names = if occursin("FlexiChain", string(typeof(chain)))
        string.(keys(chain))
    elseif hasmethod(names, Tuple{typeof(chain), Symbol})
        string.(names(chain, :parameters))
    elseif hasmethod(names, Tuple{typeof(chain)})
        string.(names(chain))
    elseif hasmethod(keys, Tuple{typeof(chain)})
        string.(keys(chain))
    else
        String[]
    end

    target = ""
    for candidate in [
        "$(param_prefix)_s_idx_t_idx",
        "$(param_prefix)_movement",
        param_prefix
    ]
        matched = _find_parameter(p_names, candidate, 1, false)
        if !isempty(matched)
            target = matched
            break
        end
    end

    if isempty(target)
        idx = findfirst(n -> startswith(n, param_prefix), p_names)
        if !isnothing(idx)
            target = p_names[idx]
        end
    end

    if isempty(target)
        error("Could not find parameter matching prefix '$(param_prefix)' in MCMC chain.")
    end

    return get_params_vector(chain, target, 1)[:, 1]
end


"""
    reconstruct_posterior_kernel(
        chain,
        W::AbstractMatrix;
        hsi::Union{Nothing, AbstractVector}=nothing,
        relationship::Symbol=:exponential
    )::Matrix{Float64}

Reconstructs the posterior-mean row-stochastic Markov transition matrix \$\\bar{\\mathbf{\\Gamma}}\$
across all MCMC samples.

# Mathematical Formulation
For each posterior sample \$s \\in \\{1, \\dots, N_s\\}\$:
\$\\mathbf{M}_s = \\mathbf{I} - v_s \\mathbf{A} - D_s \\mathbf{L}\$
\$\\mathbf{\\Gamma}_s = \\text{row\\_normalize}\\left(\\mathbf{M}_s^{-1}\\right)\$
\$\\bar{\\mathbf{\\Gamma}} = \\frac{1}{N_s} \\sum_{s=1}^{N_s} \\mathbf{\\Gamma}_s\$

where \$\\mathbf{L} = \\text{diag}(\\mathbf{W} \\mathbf{1}) - \\mathbf{W}\$ is the graph Laplacian,
and \$\\mathbf{A}\$ is the directed advection operator derived from \$\\nabla \\text{HSI}\$
(or spatial topology).

# Arguments
- `chain`: MCMC posterior chain (Turing `Chains` or `FlexiChain`).
- `W::AbstractMatrix`: Adjacency / spatial weights matrix (\$S \\times S\$).
- `hsi::Union{Nothing, AbstractVector}`: Optional Habitat Suitability Index vector of length \$S\$.
- `relationship::Symbol`: Functional form for HSI gradient: `:exponential`, `:logistic`, `:linear`.

# Returns
- `Matrix{Float64}`: Row-stochastic \$S \\times S\$ Markov transition probability matrix.
"""
function reconstruct_posterior_kernel(
    chain,
    W::AbstractMatrix;
    hsi::Union{Nothing, AbstractVector}=nothing,
    relationship::Symbol=:exponential
)::Matrix{Float64}
    S = size(W, 1)

    deg = vec(sum(W, dims=2))
    L = spdiagm(0 => deg) - W

    if !isnothing(hsi)
        hsi_vec = Float64.(hsi)
        W_dir = spzeros(Float64, S, S)
        rows = rowvals(W)
        vals = nonzeros(W)
        for i in 1:S, j_idx in nzrange(W, i)
            j = rows[j_idx]
            i == j && continue
            dh = hsi_vec[j] - hsi_vec[i]
            dh > 0.0 || continue
            W_dir[i, j] = if relationship == :exponential
                vals[j_idx] * exp(dh)
            elseif relationship == :logistic
                vals[j_idx] / (1.0 + exp(-4.0 * dh))
            else
                vals[j_idx] * dh
            end
        end
        out_deg = vec(sum(W_dir, dims=2))
        D_inv = spdiagm(0 => [od > 1e-12 ? 1.0 / od : 0.0 for od in out_deg])
        A_base = Matrix(D_inv * W_dir)
    else
        W_dir = tril(W, -1)
        out_deg = vec(sum(W_dir, dims=2))
        D_inv = spdiagm(0 => 1.0 ./ (out_deg .+ 1e-9))
        A_base = Matrix(D_inv * W_dir)
    end

    L_dense = Matrix(L)
    I_S = Matrix{Float64}(I, S, S)

    v_samps = extract_scalar_param(chain, "velocity")
    d_samps = extract_scalar_param(chain, "diffusion")
    n_s = length(v_samps)

    Gamma_acc = zeros(Float64, S, S)
    for s in 1:n_s
        M_prop = I_S .- (v_samps[s] .* A_base) .- (d_samps[s] .* L_dense)
        Gamma_s = inv(M_prop)
        for i in 1:S
            Gamma_s[i, :] .= max.(0.0, Gamma_s[i, :])
            rs = sum(Gamma_s[i, :])
            if rs > 1e-12
                Gamma_s[i, :] ./= rs
            else
                Gamma_s[i, :] .= 0.0
                Gamma_s[i, i] = 1.0
            end
        end
        Gamma_acc .+= Gamma_s
    end
    Gamma_out = Gamma_acc ./ max(1, n_s)
    for i in 1:S
        Gamma_out[i, :] .= max.(0.0, Gamma_out[i, :])
        rs = sum(Gamma_out[i, :])
        if rs > 1e-12
            Gamma_out[i, :] ./= rs
        else
            Gamma_out[i, :] .= 0.0
            Gamma_out[i, i] = 1.0
        end
    end
    return Gamma_out
end


"""
    reshard_hsi_field(
        hsi_raw::AbstractVector{<:Real},
        au_dest::NamedTuple;
        au_src::Union{Nothing, NamedTuple} = nothing,
        hsi_coords::Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
        hsi_area_method::Symbol = :grid,
        domain_bbox::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
    )::Vector{Float64}

Reshards an input Habitat Suitability Index (HSI) vector or surface from an
arbitrary source spatial geometry onto the destination spatial grid
`au_dest` (defaulting to `:hexagonal`).

# Mathematical Formulation
Let ``\\mathbf{h}_{\\text{src}} \\in [0, 1]^{S_{\\text{src}}}`` be the source HSI values
and ``\\mathbf{P} \\in \\mathbb{R}^{S_{\\text{dest}} \\times S_{\\text{src}}}`` be the
spatial interpolation / area-overlap transfer matrix between `au_src` and `au_dest`:
```math
\\mathbf{h}_{\\text{dest}} = \\text{clamp}\\left(\\mathbf{P} \\, \\mathbf{h}_{\\text{src}}, \\, 0.0, \\, 1.0\\right)
```

# Arguments
- `hsi_raw::AbstractVector{<:Real}`: Raw source HSI values of length ``S_{\\text{src}}``.
- `au_dest::NamedTuple`: Destination spatial grid (e.g., hexagonal or CVT units).
- `au_src::Union{Nothing, NamedTuple}`: Explicit source spatial grid with `:centroids` and `:polygons`.
- `hsi_coords::Union{Nothing, Vector{Tuple{Float64, Float64}}}`: Explicit coordinate points for each source HSI unit.
- `hsi_area_method::Symbol`: Source grid geometry when constructing from coordinates/bbox (`:grid`, `:cvt`, `:voronoi`, `:hexagonal`). Default: `:grid`.
- `domain_bbox::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}}`: Optional domain bounding box `(min_lon, max_lon, min_lat, max_lat)`.

# Returns
- `Vector{Float64}`: Resharded HSI vector aligned with `au_dest.centroids` (length ``S_{\\text{dest}}``).
"""
function reshard_hsi_field(
    hsi_raw         :: AbstractVector{<:Real},
    au_dest         :: NamedTuple;
    au_src          :: Union{Nothing, NamedTuple} = nothing,
    hsi_coords      :: Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
    hsi_area_method :: Symbol = :grid,
    domain_bbox     :: Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
)::Vector{Float64}
    S_dest = length(au_dest.centroids)
    S_src = length(hsi_raw)

    # 1. If explicit source grid is provided
    if !isnothing(au_src) && length(au_src.centroids) == S_src
        hsi_dest = reshard_spatial_field(hsi_raw, au_src, au_dest)
        return clamp.(Vector{Float64}(hsi_dest), 0.0, 1.0)
    end

    # 2. If source and destination sizes match and no distinct geometry/coords specified
    if S_src == S_dest && isnothing(hsi_coords)
        return clamp.(Float64.(hsi_raw), 0.0, 1.0)
    end

    # 3. Determine source points for interpolation
    src_points = if !isnothing(hsi_coords) && length(hsi_coords) == S_src
        hsi_coords
    else
        # Construct synthetic regular grid over destination domain bounding box
        dest_xs = [c[1] for c in au_dest.centroids]
        dest_ys = [c[2] for c in au_dest.centroids]
        min_x = !isnothing(domain_bbox) ? domain_bbox[1] : minimum(dest_xs)
        max_x = !isnothing(domain_bbox) ? domain_bbox[2] : maximum(dest_xs)
        min_y = !isnothing(domain_bbox) ? domain_bbox[3] : minimum(dest_ys)
        max_y = !isnothing(domain_bbox) ? domain_bbox[4] : maximum(dest_ys)

        pad_x = (max_x - min_x) * 0.05
        pad_y = (max_y - min_y) * 0.05
        grid_side = ceil(Int, sqrt(S_src))
        gx = range(min_x - pad_x, max_x + pad_x, length=grid_side)
        gy = range(min_y - pad_y, max_y + pad_y, length=grid_side)
        all_pts = Tuple{Float64, Float64}[(x, y) for x in gx for y in gy]
        all_pts[1:S_src]
    end

    # 4. Inverse Distance Weighting via TethysBio.reshard_hsi_field
    return TethysBio.reshard_hsi_field(hsi_raw, src_points, au_dest.centroids; k = min(4, S_src))
end

# ==============================================================================
# SECTION: TELEMETRY & MARK-RECAPTURE DATA INGESTION & PROCESSING
# ==============================================================================

"""
    haversine_distance(lon1::Real, lat1::Real, lon2::Real, lat2::Real; radius::Real=6378137.0)::Float64

Computes the great-circle geodesic distance in metres between two points `(lon1, lat1)`
and `(lon2, lat2)` in decimal degrees on a spherical Earth of radius `radius`
(default: the WGS84 *equatorial* radius `6378137.0 m`, i.e. ≈ `6378.137 km`).

!!! note "Radius convention"
    The default radius is the WGS84 **equatorial** radius, whereas
    `lonlat_to_xy_km` projects with the Earth **mean** radius `R = 6371.0 km`
    (≈ 0.11 % smaller). This is deliberate: the hex mesh and every stored
    checkpoint were built with the projection's mean radius, so numbers agree
    with themselves across releases. New code must not mix the two — mesh-
    consistent distances come from `lonlat_to_xy_km`, displacement statistics
    consistent with the checkpoints come from this function with its default.

Inputs are `:geographic` coordinates (decimal degrees). Non-finite inputs return
`NaN` rather than throwing; call sites that cannot tolerate `NaN` must check
explicitly.

# Mathematical Formula
\$\\Delta\\phi = \\text{deg2rad}(\\text{lat}_2 - \\text{lat}_1), \\quad \\Delta\\lambda = \\text{deg2rad}(\\text{lon}_2 - \\text{lon}_1)\$
\$a = \\sin^2\\left(\\frac{\\Delta\\phi}{2}\\right) + \\cos(\\text{lat}_1) \\cos(\\text{lat}_2) \\sin^2\\left(\\frac{\\Delta\\lambda}{2}\\right)\$
\$d = 2 R \\operatorname{atan2}\\left(\\sqrt{a}, \\sqrt{1-a}\\right)\$
"""
function haversine_distance(
    lon1::Real, lat1::Real, lon2::Real, lat2::Real;
    radius::Real=6378137.0
)::Float64
    if !isfinite(lon1) || !isfinite(lat1) || !isfinite(lon2) || !isfinite(lat2)
        return NaN
    end
    phi1 = deg2rad(Float64(lat1))
    phi2 = deg2rad(Float64(lat2))
    dphi = deg2rad(Float64(lat2 - lat1))
    dlam = deg2rad(Float64(lon2 - lon1))

    a = sin(dphi / 2.0)^2 + cos(phi1) * cos(phi2) * sin(dlam / 2.0)^2
    a = clamp(a, 0.0, 1.0)
    return 2.0 * radius * atan(sqrt(a), sqrt(1.0 - a))
end

# Tag-to-study grouping is dataset-specific and therefore supplied with the
# dataset (or precomputed into a `study` column) rather than hard-coded here.

"""
    filter_dead_tags(df::DataFrame; time_threshold_days::Real=30.0,
                     dist_threshold_meters::Real=50.0)::DataFrame

Identifies and flags acoustic telemetry records belonging to a terminal stationary period,
which indicates a deceased animal or shed tag (e.g. tag stationary within `dist_threshold_meters`
for \$\\ge\$ `time_threshold_days` continuously until the end of detections).

# Arguments
- `df::DataFrame`: Telemetry table with columns `:tagid`, `:lon`, `:lat`, and `:timestamp`
  (or `:time`).
- `time_threshold_days::Real`: Minimum duration in days of terminal non-movement (default 30.0).
- `dist_threshold_meters::Real`: Maximum radius in metres from terminal location (default 50.0).

# Returns
- `DataFrame`: A copy of `df` with an added boolean column `:is_dead`.
"""
function filter_dead_tags(
    df::DataFrame;
    time_threshold_days::Real=30.0,
    dist_threshold_meters::Real=50.0
)::DataFrame
    out = copy(df)
    time_col = hasproperty(out, :timestamp) ? :timestamp : :time
    if !hasproperty(out, time_col) || !hasproperty(out, :tagid) ||
       !hasproperty(out, :lon) || !hasproperty(out, :lat)
        error("DataFrame must contain :tagid, :lon, :lat, and :timestamp or :time.")
    end

    t_days = if eltype(out[!, time_col]) <: Dates.TimeType
        [Dates.value(Dates.Millisecond(Dates.DateTime(t) - Dates.DateTime(1970, 1, 1))) / 8.64e7 for t in out[!, time_col]]
    else
        Float64.(out[!, time_col]) .* 365.25
    end
    out._t_days = t_days

    sort!(out, [:tagid, :_t_days])

    is_dead_flags = zeros(Bool, nrow(out))
    gdf = groupby(out, :tagid)

    for sub in gdf
        n_obs = nrow(sub)
        if n_obs == 0
            continue
        end

        row_indices = parentindices(sub)[1]
        last_lon = Float64(sub.lon[end])
        last_lat = Float64(sub.lat[end])
        last_t = sub._t_days[end]

        dists_to_last = [haversine_distance(Float64(sub.lon[i]), Float64(sub.lat[i]), last_lon, last_lat) for i in 1:n_obs]
        days_to_last = [last_t - sub._t_days[i] for i in 1:n_obs]

        cummax_dist = zeros(Float64, n_obs)
        curr_max = 0.0
        for i in n_obs:-1:1
            d = isfinite(dists_to_last[i]) ? dists_to_last[i] : Inf
            curr_max = max(curr_max, d)
            cummax_dist[i] = curr_max
        end

        in_terminal_cluster = [cummax_dist[i] <= dist_threshold_meters for i in 1:n_obs]
        cluster_durations = [in_terminal_cluster[i] ? days_to_last[i] : 0.0 for i in 1:n_obs]
        max_cluster_dur = maximum(cluster_durations)

        for i in 1:n_obs
            if in_terminal_cluster[i] && max_cluster_dur >= time_threshold_days
                is_dead_flags[row_indices[i]] = true
            end
        end
    end

    out.is_dead = is_dead_flags
    select!(out, Not(:_t_days))
    return out
end

"""
    _parse_flexible_date(date_val)::Union{Date, Nothing}

Parse date strings, `DateTime`, `Date`, or numeric representations safely into `Date`.

# Handled Formats
- `Date`: Returned directly.
- `Dates.TimeType` (`DateTime`): Converted via `Date(date_val)`.
- `Real` (numeric):
  - Continuous decimal year: ``1900.0 \\le t \\le 2100.0``. The calendar year is
    ``y = \\lfloor t \\rfloor`` and the day of year is
    ``\\text{doy} = \\min(D_y, \\max(1, \\text{round}((t - y) \\cdot D_y) + 1))``,
    where ``D_y`` is 366 for leap years and 365 otherwise.
  - R date integer (days since 1970-01-01): ``0 \\le t \\le 60000``, converted via
    `Date(1970, 1, 1) + Day(round(Int, t))`.
  - Unix epoch timestamp (seconds since 1970-01-01): ``t \\ge 10^8``, converted via
    `Date(Dates.unix2datetime(t))`.
- `AbstractString`:
  - Direct ISO strings parseable via `tryparse(Date, s)` or `tryparse(DateTime, s)`.
  - Regex ISO format: `YYYY-MM-DD` or `YYYY/MM/DD` with optional time component.
  - Disambiguated `DD/MM/YYYY`, `MM/DD/YYYY`, `DD-MM-YYYY`, `MM-DD-YYYY`.
  - Numeric string representing decimal year or epoch seconds.

Returns `nothing` if the value is missing, empty, or unparseable. Range errors are
caught cleanly without clamping.
"""
function _parse_flexible_date(date_val)::Union{Date, Nothing}
    if ismissing(date_val) || isnothing(date_val)
        return nothing
    end
    if date_val isa Date
        return date_val
    elseif date_val isa Dates.TimeType
        return Date(date_val)
    elseif date_val isa Real
        isnan(date_val) && return nothing
        v = Float64(date_val)
        if 1900.0 <= v <= 2100.0
            # Continuous decimal year
            yr = floor(Int, v)
            frac = v - yr
            days_in_yr = Dates.isleapyear(yr) ? 366 : 365
            doy = round(Int, frac * days_in_yr) + 1
            if doy < 1 || doy > days_in_yr
                doy = (doy < 1) ? 1 : days_in_yr
            end
            return Date(yr, 1, 1) + Day(doy - 1)
        elseif 0.0 <= v <= 60000.0 && isinteger(v)
            # R Date representation (days since 1970-01-01)
            return Date(1970, 1, 1) + Day(round(Int, v))
        elseif v >= 1.0e8
            # Unix epoch timestamp in seconds
            return try
                Date(Dates.unix2datetime(v))
            catch
                nothing
            end
        end
        return nothing
    elseif date_val isa AbstractString
        s = strip(date_val)
        isempty(s) && return nothing

        # Direct tryparse
        d_try = tryparse(Date, s)
        d_try !== nothing && return d_try
        dt_try = tryparse(DateTime, s)
        dt_try !== nothing && return Date(dt_try)

        # Numeric string (e.g. "2004.63")
        num_try = tryparse(Float64, s)
        if num_try !== nothing
            return _parse_flexible_date(num_try)
        end

        # Match ISO format: YYYY-MM-DD or YYYY/MM/DD (with optional time component)
        m_iso = match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s].*)?$", s)
        if m_iso !== nothing
            y = parse(Int, m_iso.captures[1])
            m = parse(Int, m_iso.captures[2])
            d = parse(Int, m_iso.captures[3])
            if 1 <= m <= 12
                dim = Dates.daysinmonth(y, m)
                if 1 <= d <= dim
                    return Date(y, m, d)
                end
            end
            return nothing
        end

        # Match format: P1-P2-YYYY or P1/P2/YYYY
        m_p = match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})(?:[T\s].*)?$", s)
        if m_p !== nothing
            p1 = parse(Int, m_p.captures[1])
            p2 = parse(Int, m_p.captures[2])
            y  = parse(Int, m_p.captures[3])
            # Disambiguate day vs month without clamp
            if p1 > 12 && 1 <= p2 <= 12
                # p1 must be day, p2 must be month (DD/MM/YYYY)
                dim = Dates.daysinmonth(y, p2)
                return (1 <= p1 <= dim) ? Date(y, p2, p1) : nothing
            elseif 1 <= p1 <= 12 && p2 > 12
                # p1 must be month, p2 must be day (MM/DD/YYYY)
                dim = Dates.daysinmonth(y, p1)
                return (1 <= p2 <= dim) ? Date(y, p1, p2) : nothing
            elseif 1 <= p1 <= 12 && 1 <= p2 <= 12
                # Ambiguous: default to MM/DD/YYYY standard
                dim = Dates.daysinmonth(y, p1)
                return (1 <= p2 <= dim) ? Date(y, p1, p2) : nothing
            end
            return nothing
        end
    end
    return nothing
end

"""
    _date_to_decimal_year(d::Date)::Float64

Converts a `Date` to a continuous decimal year representation.
"""
function _date_to_decimal_year(d::Date)::Float64
    yr = Dates.year(d)
    doy = Dates.dayofyear(d)
    days_in_yr = Dates.isleapyear(yr) ? 366.0 : 365.0
    return Float64(yr) + (Float64(doy) - 1.0) / days_in_yr
end

_date_to_decimal_year(dt::Dates.TimeType)::Float64 = _date_to_decimal_year(Date(dt))



"""
    summarize_tag_activity(df::DataFrame)::DataFrame

Calculates summary statistics per individual tag from a telemetry / mark-recapture DataFrame.

# Computed Metrics per Tag
- `duration_days::Float64`: Elapsed days from first to last observation.
- `cw_change::Float64`: Net carapace width growth in mm (`last_cw - first_cw`).
- `cc_change::Float64`: Net carapace condition progression.
- `total_dist_m::Float64`: Cumulative physical track distance in metres across consecutive pings.
- `n_points::Int`: Total number of detection records.
"""
function summarize_tag_activity(df::DataFrame)::DataFrame
    time_col = _detect_time_column(df)
    if isnothing(time_col) || !hasproperty(df, :tagid)
        error("Input DataFrame must have :tagid and a date/time column (:timestamp, :time, or :date).")
    end

    gdf = groupby(df, :tagid)
    summary_rows = []

    for sub in gdf
        tid = first(sub.tagid)
        n_pts = nrow(sub)

        times_parsed = [_parse_flexible_date(t) for t in sub[!, time_col]]
        valid_times = filter(!isnothing, times_parsed)
        dur_days = if length(valid_times) > 1
            Float64(Dates.value(maximum(valid_times) - minimum(valid_times)))
        else
            0.0
        end

        cw_raw = hasproperty(sub, :cw) ? [tryparse(Float64, string(x)) for x in sub.cw] : nothing
        cw_vals = !isnothing(cw_raw) ? Float64[x for x in cw_raw if !isnothing(x) && isfinite(x)] : Float64[]
        cw_chg = length(cw_vals) > 1 ? cw_vals[end] - cw_vals[1] : NaN

        cc_raw = hasproperty(sub, :cc) ? [tryparse(Float64, string(c)) for c in sub.cc] : nothing
        cc_vals = !isnothing(cc_raw) ? Float64[x for x in cc_raw if !isnothing(x) && isfinite(x)] : Float64[]
        cc_chg = length(cc_vals) > 1 ? cc_vals[end] - cc_vals[1] : NaN

        tot_dist = 0.0
        if hasproperty(sub, :lon) && hasproperty(sub, :lat) && n_pts > 1
            for i in 1:(n_pts - 1)
                lon1, lat1 = Float64(sub.lon[i]), Float64(sub.lat[i])
                lon2, lat2 = Float64(sub.lon[i+1]), Float64(sub.lat[i+1])
                d = haversine_distance(lon1, lat1, lon2, lat2)
                if isfinite(d)
                    tot_dist += d
                end
            end
        end

        push!(summary_rows, (
            tagid = tid,
            duration_days = dur_days,
            cw_change = cw_chg,
            cc_change = cc_chg,
            total_dist_m = tot_dist,
            n_points = n_pts
        ))
    end

    return DataFrame(summary_rows)
end

"""
    sample_markov_bridge(Gamma::AbstractMatrix{<:Real},
                         u_start::Integer, u_end::Integer,
                         n_steps::Integer;
                         graph::Union{Nothing, SimpleGraph} = nothing,
                         W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                         hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
                         powers::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
                         rng::Random.AbstractRNG = Random.GLOBAL_RNG)::Vector{Int}

Samples a discrete Markov state-space trajectory ``z_0 = u_{\\text{start}} \\to z_1 \\to
\\dots \\to z_T = u_{\\text{end}}`` conditioned on fixed start and terminal endpoints under
the transition kernel ``\\boldsymbol{\\Gamma}`` while strictly respecting the marine domain
and coastline movement boundaries of adjacency graph ``G(V, E)``.

# Mathematical Formulation:
Let ``G(V, E)`` be the topological water graph where edges ``(i, j) \\in E`` connect
only physically adjacent water units across marine boundaries.
1. The shortest geodesic graph distance ``d_G(u_{\\text{start}}, u_{\\text{end}})`` is computed via
   ``A^*`` search. To prevent artificial jumps across land barriers or peninsulas,
   the effective trajectory step count satisfies:
   ```math
   T_{\\text{eff}} = \\max(n_{\\text{steps}}, d_G(u_{\\text{start}}, u_{\\text{end}}))
   ```
2. For each intermediate step ``t \\in \\{1, \\dots, T_{\\text{eff}}-1\\}``, given previous state
   ``z_{t-1} = i`` and destination ``u_{\\text{end}}``, transitions are sampled strictly from
   valid marine graph neighbors ``j \\in \\mathcal{N}(i) \\cup \\{i\\}`` via Chapman-Kolmogorov:
   ```math
   P(z_t = j \\mid z_{t-1} = i, z_T = u_{\\text{end}}) = \\frac{T_{ij} \\, (\\mathbf{T}^{T_{\\text{eff}}-t})_{j, u_{\\text{end}}}}{(\\mathbf{T}^{T_{\\text{eff}}-t+1})_{i, u_{\\text{end}}}}
   ```
   where ``T_{ij} > 0`` only if ``(i, j) \\in E`` or ``i = j``.

# Inputs:
- `Gamma`: ``S \\times S`` transition matrix.
- `u_start`: Initial spatial unit index ``1 \\le u_{\\text{start}} \\le S``.
- `u_end`: Terminal spatial unit index ``1 \\le u_{\\text{end}} \\le S``.
- `n_steps`: Requested discrete time steps ``T \\ge 1``.
- `graph`: Optional `SimpleGraph` topology for marine coastline path routing.
- `W`: Optional ``S \\times S`` sparse adjacency matrix.
- `hsi`: Optional spatial HSI suitability vector for directional advective weighting.
- `powers`: Precomputed matrix powers of the 1-step kernel.
- `rng`: Random number generator.

# Outputs:
- `path::Vector{Int}`: Sequence of visited spatial unit indices strictly on the marine graph.
"""
function sample_markov_bridge(
    Gamma::AbstractMatrix{<:Real},
    u_start::Integer,
    u_end::Integer,
    n_steps::Integer;
    graph::Union{Nothing, SimpleGraph} = nothing,
    W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    hsi::Union{Nothing, AbstractVector{<:Real}} = nothing,
    powers::Union{Nothing, Vector{<:AbstractMatrix}} = nothing,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{Int}
    u_s = Int(u_start)
    u_e = Int(u_end)
    S = size(Gamma, 1)

    if u_s == u_e
        return fill(u_s, max(2, n_steps + 1))
    end

    # If graph is provided, ensure path stays strictly on valid water edges
    if !isnothing(graph)
        sp = a_star(graph, u_s, u_e)
        sp_nodes = isempty(sp) ? [u_s, u_e] : vcat([src(e) for e in sp], [dst(last(sp))])
        d_min = length(sp_nodes) - 1

        effective_steps = max(n_steps, d_min)
        if effective_steps <= d_min
            return sp_nodes
        end

        # Build 1-step water transition matrix T if W is provided
        T_step = if !isnothing(W)
            T_mat = zeros(Float64, S, S)
            for i in 1:S
                nbrs = findall(>(0.0), W[i, :])
                if isempty(nbrs)
                    T_mat[i, i] = 1.0
                    continue
                end
                w_vec = if !isnothing(hsi)
                    [exp(0.5 * (Float64(hsi[j]) - Float64(hsi[i]))) for j in nbrs]
                else
                    ones(Float64, length(nbrs))
                end
                w_sum = sum(w_vec)
                w_norm = w_sum > 0 ? (w_vec ./ w_sum) : fill(1.0 / length(nbrs), length(nbrs))
                T_mat[i, i] = 0.35 # probability of remaining in cell during unit step
                for (idx, j) in enumerate(nbrs)
                    T_mat[i, j] = 0.65 * w_norm[idx]
                end
            end
            T_mat
        else
            Matrix{Float64}(Gamma)
        end

        # Precompute powers up to effective_steps
        P_mats = Vector{Matrix{Float64}}(undef, effective_steps)
        P_mats[1] = copy(T_step)
        for k in 2:effective_steps
            P_mats[k] = P_mats[k-1] * T_step
        end

        path = zeros(Int, effective_steps + 1)
        path[1] = u_s
        path[end] = u_e
        curr = u_s

        for t in 1:(effective_steps - 1)
            rem = effective_steps - t
            cand_nodes = findall(>(0.0), T_step[curr, :])
            weights = Float64[]
            for j in cand_nodes
                t_ij = T_step[curr, j]
                p_j_end = (rem == 1) ? T_step[j, u_e] : P_mats[rem][j, u_e]
                push!(weights, t_ij * p_j_end)
            end
            w_sum = sum(weights)
            if w_sum <= 1e-15
                sp_curr = a_star(graph, curr, u_e)
                next_node = !isempty(sp_curr) ? dst(first(sp_curr)) : curr
                path[t + 1] = next_node
                curr = next_node
            else
                probs = weights ./ w_sum
                r = rand(rng)
                cum = 0.0
                chosen = cand_nodes[end]
                for (idx, j) in enumerate(cand_nodes)
                    cum += probs[idx]
                    if r <= cum
                        chosen = j
                        break
                    end
                end
                path[t + 1] = chosen
                curr = chosen
            end
        end

        return path
    end

    # Fallback when no graph topology is supplied
    if n_steps <= 1
        return [u_s, u_e]
    end

    local P_mats_dense::Vector{Matrix{Float64}}
    if !isnothing(powers) && length(powers) >= n_steps
        P_mats_dense = [Matrix{Float64}(p) for p in powers[1:n_steps]]
    else
        P_mats_dense = Vector{Matrix{Float64}}(undef, n_steps)
        P_mats_dense[1] = Matrix{Float64}(Gamma)
        for k in 2:n_steps
            P_mats_dense[k] = P_mats_dense[k-1] * P_mats_dense[1]
        end
    end

    path = zeros(Int, n_steps + 1)
    path[1] = u_s
    path[end] = u_e
    curr = u_s

    for t in 1:(n_steps - 1)
        rem_steps = n_steps - t
        weights = zeros(Float64, S)
        for j in 1:S
            g_ij = Float64(Gamma[curr, j])
            p_j_end = (rem_steps == 1) ? Float64(Gamma[j, u_e]) : Float64(P_mats_dense[rem_steps][j, u_e])
            weights[j] = g_ij * p_j_end
        end
        w_sum = sum(weights)
        if w_sum <= 1e-15
            weights = Float64[Float64(Gamma[curr, j]) for j in 1:S]
            w_sum = sum(weights)
        end
        probs = weights ./ (w_sum > 0.0 ? w_sum : 1.0)

        r = rand(rng)
        cum = 0.0
        next_s = S
        for j in 1:S
            cum += probs[j]
            if r <= cum
                next_s = j
                break
            end
        end
        path[t + 1] = next_s
        curr = next_s
    end

    return path
end

"""
    reconstruct_mark_recapture_paths(tagging::DataFrame, result::NamedTuple;
                                     time_interval::Symbol = :monthly,
                                     max_paths::Union{Nothing, Int} = nothing,
                                     smooth_jitter::Bool = true,
                                     rng::Random.AbstractRNG = Random.GLOBAL_RNG)::Vector{NamedTuple}

Reconstructs the latent continuous state-space trajectory for every individual
mark-recapture event in `tagging` using an exact discrete Markov Bridge filter conditioned
on the true observed release and recapture coordinates under the fitted transition matrix
and HSI field. Trajectories strictly follow the marine water graph and avoid land crossing.

# Mathematical Formulation:
For an individual ``k`` with release observation at ``(\\text{lon}_0, \\text{lat}_0)``
at time ``t_0`` and recapture at ``(\\text{lon}_T, \\text{lat}_T)`` at time ``t_T``:
1. Coordinates are mapped to nearest discrete marine areal units ``u_0, u_T \\in \\mathcal{U}``.
   When observations lie beyond the domain, the last encountered boundary unit HSI probability is extended.
2. The elapsed time interval ``\\Delta t = \\max(1, \\text{round}(\\text{Int}, |t_T - t_0| / \\Delta t_{\\text{interval}}))``.
3. Conditional latent intermediate units ``(z_0 = u_0, z_1, \\dots, z_T = u_T)`` are sampled
   along connected marine graph edges.
4. Each discrete state ``z_t`` is projected to continuous geographic coordinates ``(\\text{lon}_t, \\text{lat}_t)``
   with boundary pinning at true release and recapture points.

# Inputs:
- `tagging`: Observation DataFrame containing `:tagid`, `:lon`, `:lat`, and `:time` or `:timestamp`.
- `result`: Model bundle containing `au`, `transition_matrix`, and `opts`.
- `time_interval`: Temporal step resolution (`:monthly`, `:weekly`, `:daily`).
- `max_paths`: Optional integer limit on number of individuals to reconstruct (default `nothing` for all).
- `smooth_jitter`: Whether to apply subtle spatial smoothing inside marine polygons.

# Outputs:
- `trajectories::Vector{NamedTuple}`: Reconstructed trajectories for all individuals.
"""
function reconstruct_mark_recapture_paths(
    tagging::DataFrame,
    result::NamedTuple;
    time_interval::Symbol = :monthly,
    max_paths::Union{Nothing, Int} = nothing,
    smooth_jitter::Bool = true,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG
)::Vector{NamedTuple}
    au = result.au
    Gamma = Matrix{Float64}(result.transition_matrix)
    S = au.n_units
    g = hasproperty(au, :graph) ? au.graph : nothing
    W = hasproperty(au, :W) ? au.W : nothing
    hsi_vec = hasproperty(result, :hsi) && !isnothing(result.hsi) ? result.hsi : nothing

    # Pre-extract geographic centroids of au in WGS84 degrees
    cents_deg = Tuple{Float64, Float64}[]
    wkt_str = hasproperty(au, :wkt) ? string(au.wkt) : ""
    is_utm = contains(lowercase(wkt_str), "utm") || (!isempty(au.centroids) && au.centroids[1][2] > 1000.0)

    for c in au.centroids
        if is_utm
            push!(cents_deg, utm_to_lonlat(c[1], c[2]; zone=20, is_km=true))
        else
            push!(cents_deg, (Float64(c[1]), Float64(c[2])))
        end
    end

    # Group observations by tag
    time_col = _detect_time_column(tagging)
    isnothing(time_col) && error("tagging DataFrame must have a date/time column (:timestamp, :time, or :date).")

    gdf = groupby(tagging, :tagid)
    tag_keys = collect(keys(gdf))
    n_tags_total = length(tag_keys)
    n_to_process = isnothing(max_paths) ? n_tags_total : min(n_tags_total, max_paths)

    # Unit step duration in decimal years
    dt_step = (time_interval == :weekly) ? (1.0 / 52.0) : ((time_interval == :daily) ? (1.0 / 365.0) : (1.0 / 12.0))

    trajectories = NamedTuple[]

    for i in 1:n_to_process
        sub_group = gdf[tag_keys[i]]
        order_cols = hasproperty(sub_group, :tag) ? [:tag, time_col] : [time_col]
        sub = sort(sub_group, order_cols)
        nrow(sub) < 2 && continue

        tid = string(first(sub.tagid))
        all_path_units = Int[]
        all_path_coords = Tuple{Float64, Float64}[]

        # Reconstruct piecewise between consecutive observation records
        for seg in 1:(nrow(sub) - 1)
            r1 = sub[seg, :]
            r2 = sub[seg + 1, :]

            lon1, lat1 = Float64(r1.lon), Float64(r1.lat)
            lon2, lat2 = Float64(r2.lon), Float64(r2.lat)
            t1 = Float64(r1[time_col])
            t2 = Float64(r2[time_col])

            # Find nearest spatial units in au strictly on navigable water
            au_lmask = hasproperty(au, :land_mask) ? au.land_mask : nothing
            u_start = _nearest_unit_index(
                lon1, lat1, cents_deg;
                land_mask = au_lmask, W = W
            )
            u_end = _nearest_unit_index(
                lon2, lat2, cents_deg;
                land_mask = au_lmask, W = W
            )

            elapsed_t = max(0.0, t2 - t1)
            n_steps = clamp(round(Int, elapsed_t / dt_step), 1, 60)

            # Sample water-constrained Markov bridge
            bridge = sample_markov_bridge(
                Gamma, u_start, u_end, n_steps;
                graph = g, W = W, hsi = hsi_vec, rng = rng
            )

            # Convert bridge units to continuous geographic coordinates
            for (step_idx, u_id) in enumerate(bridge)
                if seg > 1 && step_idx == 1
                    continue # Avoid duplicate point at segment boundary
                end

                push!(all_path_units, u_id)

                if step_idx == 1
                    push!(all_path_coords, (lon1, lat1))
                elseif step_idx == length(bridge)
                    push!(all_path_coords, (lon2, lat2))
                else
                    c_pt = cents_deg[u_id]
                    pt_lon, pt_lat = c_pt[1], c_pt[2]
                    if smooth_jitter
                        # Subtle perturbation inside marine polygon (approx 0.003 deg ~ 300 m)
                        jit_r = 0.003 * sqrt(rand(rng))
                        jit_theta = 2.0 * pi * rand(rng)
                        pt_lon += jit_r * cos(jit_theta)
                        pt_lat += jit_r * sin(jit_theta)
                    end
                    push!(all_path_coords, (pt_lon, pt_lat))
                end
            end
        end

        length(all_path_coords) < 2 && continue

        # Compute trajectory distance and net displacement (km)
        total_dist_km = 0.0
        for k in 2:length(all_path_coords)
            total_dist_km += haversine_distance(
                all_path_coords[k-1][1], all_path_coords[k-1][2],
                all_path_coords[k][1], all_path_coords[k][2]
            ) / 1000.0
        end

        net_disp_km = haversine_distance(
            all_path_coords[1][1], all_path_coords[1][2],
            all_path_coords[end][1], all_path_coords[end][2]
        ) / 1000.0

        dur_days = Float64((sub[end, time_col] - sub[1, time_col]) * 365.25)

        push!(trajectories, (
            tagid           = tid,
            coords          = all_path_coords,
            units           = all_path_units,
            n_steps         = length(all_path_coords) - 1,
            start_date      = string(sub[1, time_col]),
            end_date        = string(sub[end, time_col]),
            duration_days   = dur_days,
            total_dist_km   = total_dist_km,
            displacement_km = net_disp_km
        ))
    end

    return trajectories
end

function _nearest_unit_index(
    lon::Float64,
    lat::Float64,
    cents_deg::Vector{Tuple{Float64, Float64}};
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    W::Union{Nothing, AbstractMatrix} = nothing
)::Int
    S = length(cents_deg)
    active_mask = trues(S)
    if land_mask !== nothing
        active_mask .&= .!land_mask
    end
    if W !== nothing
        deg = vec(sum(W, dims=2))
        active_mask .&= (deg .> 0.0)
    end
    active_indices = findall(active_mask)
    if isempty(active_indices)
        active_indices = collect(1:S)
    end

    best_idx = first(active_indices)
    best_dist = Inf
    for i in active_indices
        c = cents_deg[i]
        d = (c[1] - lon)^2 + (c[2] - lat)^2
        if d < best_dist
            best_dist = d
            best_idx = i
        end
    end
    return best_idx
end



"""
    _build_A_ad(adj_rows, hsi, gamma_g, S)

AD-compatible directed adjacency build. Optimized for ForwardDiff.
"""
function _build_A_ad(adj_rows, hsi, gamma_g, S)
    # exp(gamma * (HSI_j - HSI_i)) / sum_k exp(gamma * (HSI_k - HSI_i)) simplifies
    # to exp(gamma * HSI_j) / sum_k exp(gamma * HSI_k) since HSI_i cancels.
    exp_hsi = exp.(gamma_g .* hsi)
    
    # Extract the promoted AD type (e.g., ForwardDiff.Dual) directly from the math
    T = eltype(exp_hsi)
    A = zeros(T, S, S)
    
    @inbounds for i in 1:S
        nbrs = adj_rows[i]
        isempty(nbrs) && continue
        
        sw = zero(T)
        for j in nbrs
            sw += exp_hsi[j]
        end
        
        sw <= 0 && continue
        
        for j in nbrs
            A[i, j] = exp_hsi[j] / sw
        end
    end
    
    return A
end


function _row_normalise(M, S)
    T = eltype(M)
    M_rect = max.(zero(T), M)
    s = sum(M_rect, dims=2)
    
    # Non-mutating division-by-zero protection
    s_safe = s .+ (s .== zero(T))
    
    # Broadcast normalization and uniform distribution injection for zero-rows
    return (M_rect ./ s_safe) .+ ((s .== zero(T)) ./ T(S))
end

# ── Directed adjacency and resolvent operator ──────────────────────────────────
 

"""
    compute_directed_adjacency(hsi, W; gamma=1.0) -> Matrix{Float64}

Construct the directed adjacency matrix A from the symmetric adjacency W and
habitat suitability values HSI. Row i has non-zero entries only at W-neighbours
of i, weighted by exp(γ HSI_j) and row-normalised:

```math
A[i,j] = \\frac{\\exp(\\gamma_i \\cdot \\text{HSI}_j)}{\\sum_{k \\in \\mathcal{N}(i)} \\exp(\\gamma_i \\cdot \\text{HSI}_k)}
```

γ > 0 biases movement towards higher HSI; γ = 0 gives the uniform random walk.

# Arguments
- `hsi::AbstractVector{<:Real}`: Habitat suitability per spatial unit (length S).
- `W::SparseMatrixCSC`: Symmetric binary adjacency matrix (S × S).
- `gamma::Union{Real, AbstractVector{<:Real}}`: Advection sensitivity to HSI gradient
  (default 1.0). May be a scalar or a spatially varying vector of length S.

# Returns
- `Matrix{Float64}`: Dense S × S matrix A with row-stochastic structure over W-neighbours.
"""
function compute_directed_adjacency(
    hsi::AbstractVector{<:Real},
    W::SparseMatrixCSC;
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0
)::SparseMatrixCSC{Float64, Int}
    S = size(W, 1)

    is_spatial = gamma isa AbstractVector
    if is_spatial
        n_g = length(gamma)
        if n_g != S && n_g != 1
            throw(DimensionMismatch(
                "gamma vector has length $n_g, but must be scalar or match spatial units S=$S."
            ))
        end
        if n_g == 1
            gamma_scalar = Float64(gamma[1])
            is_spatial = false
        end
    else
        gamma_scalar = Float64(gamma)
    end

    # Build sparse triplets -- same sparsity pattern as W
    nnz_W = nnz(W)
    I_out = Vector{Int}(undef, nnz_W)
    J_out = Vector{Int}(undef, nnz_W)
    V_out = Vector{Float64}(undef, nnz_W)
    ptr_out = 0

    if !is_spatial
        # Fast path: precompute exponents once for uniform scalar gamma
        exp_hsi = exp.(gamma_scalar .* Float64.(hsi))
        for i in 1:S
            col_start = W.colptr[i]
            col_end   = W.colptr[i+1] - 1
            col_start > col_end && continue

            sw = 0.0
            @inbounds for ptr in col_start:col_end
                sw += exp_hsi[W.rowval[ptr]]
            end
            sw <= 0.0 && continue

            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                ptr_out += 1
                I_out[ptr_out] = i
                J_out[ptr_out] = j
                V_out[ptr_out] = exp_hsi[j] / sw
            end
        end
    else
        # Spatially-varying gamma[i] per unit i
        for i in 1:S
            col_start = W.colptr[i]
            col_end   = W.colptr[i+1] - 1
            col_start > col_end && continue

            g_i = Float64(gamma[i])
            sw = 0.0
            @inbounds for ptr in col_start:col_end
                sw += exp(g_i * Float64(hsi[W.rowval[ptr]]))
            end
            sw <= 0.0 && continue

            @inbounds for ptr in col_start:col_end
                j = W.rowval[ptr]
                ptr_out += 1
                I_out[ptr_out] = i
                J_out[ptr_out] = j
                V_out[ptr_out] = exp(g_i * Float64(hsi[j])) / sw
            end
        end
    end

    return sparse(
        view(I_out, 1:ptr_out),
        view(J_out, 1:ptr_out),
        view(V_out, 1:ptr_out),
        S, S
    )
end


 
"""
    resolvent_transition(beta, D_diff, A, L, S) -> Matrix{Float64}

Compute the resolvent transition operator:

    Γ̄ = (I − β A − D L)^{-1}

where L is the symmetric graph Laplacian. Rows of Γ̄ are rectified (negative
entries set to zero) and row-normalised to form valid probability distributions.

Note: the inverse exists when ‖β A + D L‖ < 1 in an appropriate operator norm.
The NUTS priors (β < 0.95, D ≥ 0) are chosen to help ensure this, but
near-boundary samples may produce poorly conditioned M; the try/catch below
falls back to `M \\ I` in those cases.

# Arguments
- `beta`: Advective weight (scalar, 0 ≤ β < 1).
- `D_diff`: Diffusion coefficient (scalar, ≥ 0).
- `A::Matrix{Float64}`: Directed adjacency matrix (S × S, row-stochastic over nbrs).
- `L::Matrix{Float64}`: Graph Laplacian (S × S, positive semidefinite).
- `S::Int`: Number of spatial units.

# Returns
- Dense S × S matrix Γ̄ with row-stochastic rows.
"""
function resolvent_transition(
    beta::Real,
    D_diff::Real,
    A::AbstractMatrix{<:Real},
    L::AbstractMatrix{<:Real},
    S::Int
)::Matrix{Float64}

    # Lock types upfront for type-stability
    b = Float64(beta)
    d = Float64(D_diff)

    # Assemble M = I - beta*A - D*L
    M = Matrix(sparse(1.0I, S, S) - b .* A - d .* L)

    Gamma = try
        inv(M)
    catch e
        @warn "resolvent_transition: inv failed, falling back to M\\I: $(e)"
        M \ Matrix{Float64}(I, S, S)
    end

    # Rectify negative entries and accumulate row sums (column-major traversal)
    row_sums = zeros(Float64, S)
    @inbounds for j in 1:S
        for i in 1:S
            v = max(0.0, Gamma[i, j])
            Gamma[i, j] = v
            row_sums[i] += v
        end
    end

    # 4. Row-normalise using the accumulated sums (Column-Major)
    # Row-normalise using the accumulated sums (Column-Major)
    @inbounds for j in 1:S
        for i in 1:S
            if row_sums[i] > 0.0
                Gamma[i, j] /= row_sums[i]
            end
        end
    end

    return Gamma
end

"""
    resolvent_expected_visits(P::SparseMatrixCSC{Float64, Int}, target::Int; alpha::Float64=0.99)

Computes the target column of the resolvent transition matrix Γ = (I - α P)^-1.
This returns a vector `x` where `x[i]` is the expected number of (discounted) visits 
to `target` starting from `i`, representing an infinite-horizon reachability metric.
"""
function resolvent_expected_visits(P::SparseMatrixCSC{Float64, Int}, target::Int; alpha::Float64=0.99)
    S = size(P, 1)
    M = sparse(I, S, S) - alpha * P
    b = zeros(Float64, S)
    b[target] = 1.0
    x = try
        M \ b
    catch e
            @warn "resolvent_expected_visits: linear solve failed, returning uniform." exception = (e, catch_backtrace())
        fill(1.0, S)
    end
    return max.(x, 1e-15)
end

 

"""
    _powerm(M, k) -> Matrix

Compute M^k using iterative binary (repeated squaring) exponentiation.
Optimized to perform all multiplications in-place, reducing memory allocations 
to \$O(1)\$ regardless of \$k\$.
"""
function _powerm(M::AbstractMatrix{T}, k::Integer) where {T <: Real}
    n = size(M, 1)
    
    # Ensure type stability by matching the precision of M (e.g., Float64)
    OutType = float(T) 
    
    k <= 0 && return Matrix{OutType}(I, n, n)
    k == 1 && return Matrix{OutType}(M)

    # Preallocate active matrices
    R = Matrix{OutType}(I, n, n)
    B = Matrix{OutType}(M)
    
    # Preallocate temporary buffers for in-place multiplication
    R_tmp = similar(R)
    B_tmp = similar(B)

    # Iterative repeated squaring
    while k > 0
        if isodd(k)
            # R = R * B (in-place)
            mul!(R_tmp, R, B)
            R, R_tmp = R_tmp, R  # Swap references (zero cost)
        end
        
        k >>= 1 # Fast bitwise division by 2  ..  isodd(k) and k >>= 1 is faster than k % 2 != 0 and k ÷ 2.
        
        if k > 0
            # B = B * B (in-place)
            mul!(B_tmp, B, B)
            B, B_tmp = B_tmp, B  # Swap references (zero cost)
        end
    end
    
    return R
end

"""
    power_transition(Gamma::AbstractMatrix{Float64}, k::Integer) -> Matrix{Float64}

Computes the ``k``-step Markov transition probability matrix ``\\boldsymbol{\\Gamma}^k``
via repeated matrix squaring (binary exponentiation) and applies non-negativity
rectification and row-stochastic normalization to eliminate floating-point drift.

# Mathematical Formulation
For a discrete spatial Markov transition matrix
``\\boldsymbol{\\Gamma} \\in \\mathbb{R}^{S \\times S}``:
```math
\\boldsymbol{\\Gamma}^k = \\underbrace{\\boldsymbol{\\Gamma} \\times \\cdots \\times \\boldsymbol{\\Gamma}}_{k \\text{ times}}
```
Each entry ``[\\boldsymbol{\\Gamma}^k]_{ij} = \\mathbb{P}(X_{t+k} = j \\mid X_t = i)``
denotes the conditional probability that an individual transitions from node ``i`` to
node ``j`` after ``k`` time steps. Small negative entries arising from numerical
rounding are clamped to zero and rows are re-normalized:
```math
[\\boldsymbol{\\Gamma}^k]_{ij} \\leftarrow \\frac{\\max(0, [\\boldsymbol{\\Gamma}^k]_{ij})}{\\sum_{l=1}^S \\max(0, [\\boldsymbol{\\Gamma}^k]_{il})}
```

# Arguments
- `Gamma`: Row-stochastic one-step transition matrix (size ``S \\times S``).
- `k`: Discrete time step exponent (``k \\ge 1``).

# Returns
- `Matrix{Float64}`: Numerically stable row-stochastic ``k``-step transition matrix.
"""
function power_transition(Gamma::AbstractMatrix{Float64}, k::Integer)::Matrix{Float64}
    # Uses _powerm (exponentiation by squaring)
    Gk = _powerm(Gamma, max(1, k))
    S = size(Gk, 1)
    
    # Rectify negative entries and accumulate row sums (column-major traversal)
    row_sums = zeros(Float64, S)
    @inbounds for j in 1:S
        for i in 1:S
            v = max(0.0, Gk[i, j])
            Gk[i, j] = v
            row_sums[i] += v
        end
    end
    
    # 2. Row-normalise using the accumulated sums (Column-Major)
    @inbounds for j in 1:S
        for i in 1:S
            if row_sums[i] > 0.0
                Gk[i, j] /= row_sums[i]
            end
        end
    end
    
    return Gk
end


"""
    construct_stochastic_transition_kernel(W, hsi; gamma=1.0, residence=0.2, advection=0.5, spatial=false)

Constructs strictly row-stochastic, unconditionally non-negative discrete movement
transition matrices ``P`` over spatial graph ``W``:

```math
P = (1 - \\rho) \\left[ (1 - \\alpha) T_{\\text{diff}} + \\alpha A \\right] + \\rho I
```

# Arguments
- `W::SparseMatrixCSC`: Spatial adjacency matrix (size ``S \\times S``).
- `hsi::AbstractVector{<:Real}`: Habitat suitability values per unit (length ``S``).
- `gamma::Union{Real, AbstractVector{<:Real}}`: Sensitivity of directional advection to
  the habitat gradient (default 1.0). May be scalar or vector across groups or units.
- `residence::Union{Real, AbstractVector{<:Real}}`: Probability ``\\rho \\in [0, 1)`` of
  remaining in current unit (default 0.2). May be scalar or vector across groups or units.
- `advection::Union{Real, AbstractVector{<:Real}}`: Fraction ``\\alpha \\in [0, 1]`` of
  directed movement vs random diffusion (default 0.5). May be scalar or vector.
- `spatial::Bool`: When `true`, vector parameters of length ``S`` are interpreted as
  spatially varying per unit ``s \\in 1:S``, returning a single ``S \\times S`` matrix.
  When `false` (default), vector parameters of length ``G`` represent group-level
  parameters across ``G`` biological groups, returning a `Vector{Matrix{Float64}}`.

# Returns
- `Matrix{Float64}`: If all parameters are scalars (or `spatial=true`), dense ``S \\times S``
  row-stochastic transition probability matrix.
- `Vector{Matrix{Float64}}`: If any parameter is an `AbstractVector` (and `spatial=false`),
  vector of ``G`` dense ``S \\times S`` row-stochastic transition matrices.
"""
function construct_stochastic_transition_kernel(
    W::SparseMatrixCSC,
    hsi::AbstractVector{<:Real};
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
    residence::Union{Real, AbstractVector{<:Real}} = 0.2,
    advection::Union{Real, AbstractVector{<:Real}} = 0.5,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    coupled_residency::Bool = false
  )::Matrix{Float64}
    S = size(W, 1)
    if length(hsi) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), but hsi has length $(length(hsi))."
        ))
    end
    if !isnothing(land_mask) && length(land_mask) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), but land_mask has length $(length(land_mask))."
        ))
    end

    # `gamma` and `advection` are fitted parameters and must stay scalar. `residence`
    # may also be a per-unit vector, but only when the caller sets
    # `coupled_residency`, which asserts that the vector is *derived* -- one value
    # per unit computed from the fitted scalar rho and local habitat -- rather
    # than a second set of fitted parameters. Without that flag the guard still
    # rejects it, so the group-axis misuse it was built to stop cannot reappear
    # through the back door.
    residence_arg = if coupled_residency && residence isa AbstractVector
        residence
    else
        _pooled_scalar(residence, "residence")
    end

    P_sparse = build_sparse_transition_kernel(
        W, hsi,
        _pooled_scalar(gamma, "gamma"),
        residence_arg,
        _pooled_scalar(advection, "advection"),
        land_mask,
    )
    return Matrix{Float64}(P_sparse)
end

"""
    _pooled_scalar(p, name) -> Float64

Coerce a kernel parameter to a single scalar for the pooled model.

A length-1 vector is accepted and unwrapped, because that is unambiguous. A longer
vector used to select one kernel per demographic group; the pooled model has one
parameter set, so accepting it would silently discard the extra values. This
raises instead, naming the parameter, so the caller has to be made explicit.
"""
function _pooled_scalar(p, name::AbstractString)::Float64
    p isa AbstractVector || return Float64(p)
    length(p) == 1 && return Float64(first(p))
    throw(ArgumentError(
        "Kernel parameter `$name` has length $(length(p)). The movement model is " *
        "pooled and takes one value per parameter; per-group vectors are no longer " *
        "supported. Reduce `$name` to a single value (a mean, a fitted draw, or a " *
        "configured constant) before building the kernel."
    ))
end


"""
    _spatial_node_distance(c1, c2; coord_space = :unknown) -> Float64

Great-circle or planar distance between two node coordinates, in kilometres.

`coord_space` is authoritative whenever the caller knows it: `:geographic` uses
haversine on WGS84 degrees, `:planar_km` uses Euclidean distance on a local metric
frame already in kilometres, and `:unknown` falls back to conservative inference.
Range is not evidence of degrees — a planar frame in kilometres almost always
falls inside the `[-180, 180]` degree box, and treating it as longitude/latitude
collapses the distance to nearly zero.

Both branches return kilometres. `haversine_distance` returns **metres**, so the
conversion is required here; omitting it made calibrated effective resistance
1000x too large for any geographic mesh while labelling it kilometres.
"""
function _spatial_node_distance(c1, c2; coord_space::Symbol = :unknown)::Float64
  x1, y1 = Float64(c1[1]), Float64(c1[2])
  x2, y2 = Float64(c2[1]), Float64(c2[2])
  space = coord_space
  if space === :unknown
    space = _infer_coord_space((c1, c2)) ? :geographic : :planar_km
  end
  if space === :geographic
    return haversine_distance(x1, y1, x2, y2) / 1000.0
  end
  return sqrt((x1 - x2)^2 + (y1 - y2)^2)
end

"""
    _find_navigable_node(u::Int, centroids, land_mask, g::AbstractGraph) -> Int

Resolves an endpoint unit to the nearest topologically active and navigable
marine node in graph `g`. If `u` is already active (not in `land_mask` and has
degree > 0 in `g`), returns `u` directly. Otherwise finds the closest marine
unit in Euclidean space.
"""
function _find_navigable_node(
    u::Int,
    centroids,
    land_mask::Union{Nothing, AbstractVector{Bool}},
    g::AbstractGraph
)::Int
    S = nv(g)
    is_valid_u = (1 <= u <= S) &&
                 (land_mask === nothing || !land_mask[u]) &&
                 (degree(g, u) > 0)
    if is_valid_u
        return u
    end

    best_v = u
    min_dist = Inf
    for v in 1:S
        if (land_mask === nothing || !land_mask[v]) && degree(g, v) > 0
            d = centroids !== nothing ?
                _spatial_node_distance(centroids[u], centroids[v]) : abs(u - v)
            if d < min_dist
                min_dist = d
                best_v = v
            end
        end
    end
    if best_v != u
        @warn "Unit $u is not navigable (land or severed); mapped to " *
              "nearest marine unit $best_v."
    end
    return best_v
end

"""
    _exact_k_max_prob_path(P, u_start, u_end, k; land_mask = nothing, p_min = 1e-12)

Maximum-probability walk of exactly `k` transitions from `u_start` to `u_end`,
computed as a length-constrained Viterbi trellis in log-probability space.

Only transitions with `P[i, j] > p_min` are admissible, so every returned step
corresponds to a transition the model actually permits -- including a residence
self-loop, but only where `P[i, i] > 0`.

Returns an empty `Vector{Int}` when no such walk exists. It never pads a shorter
route with fabricated self-loops and never returns a route whose length differs
from the requested `k`.

This exists because the two fallbacks it replaces were both fabrications. When
release and recapture fell on the same unit, or when the goal turned out to be
unreachable, the caller returned `fill(release, k + 1)` -- a path claiming `k`
steps of movement while never leaving the cell, and asserting a residence
probability the fitted kernel may not assign. Reporting an honest "no such route"
is more useful than a plausible wrong one, so unsatisfiable requests are dropped
with a warning rather than silently padded.
"""
function _exact_k_max_prob_path(
    P::AbstractMatrix{<:Real},
    u_start::Int,
    u_end::Int,
    k::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    p_min::Real = 1e-12
)::Vector{Int}
    k < 0 && return Int[]
    k == 0 && return u_start == u_end ? [u_start] : Int[]

    S = size(P, 1)
    P_csc = P isa SparseMatrixCSC ? P : SparseMatrixCSC(P)

    delta = fill(-Inf, S, k + 1)
    psi   = zeros(Int, S, k + 1)
    delta[u_start, 1] = 0.0

    @inbounds for tau in 2:(k + 1)
        prev = tau - 1
        for j in 1:S
            (land_mask !== nothing && land_mask[j]) && continue
            best_val = -Inf
            best_prev = 0
            for ptr in P_csc.colptr[j]:(P_csc.colptr[j + 1] - 1)
                i = P_csc.rowval[ptr]
                (land_mask !== nothing && land_mask[i]) && continue
                p_ij = Float64(P_csc.nzval[ptr])
                p_ij > p_min || continue
                prev_val = delta[i, prev]
                isfinite(prev_val) || continue
                score = prev_val + log(p_ij)
                if score > best_val
                    best_val = score
                    best_prev = i
                end
            end
            if best_prev != 0
                delta[j, tau] = best_val
                psi[j, tau]   = best_prev
            end
        end
    end

    isfinite(delta[u_end, k + 1]) || return Int[]

    path = zeros(Int, k + 1)
    path[k + 1] = u_end
    @inbounds for tau in (k + 1):-1:2
        prev = psi[path[tau], tau]
        prev == 0 && return Int[]
        path[tau - 1] = prev
    end
    path[1] == u_start || return Int[]
    return path
end

function _bfs_marine_shortest_path(
    P_sp::SparseMatrixCSC{<:Real},
    u_start::Int,
    u_end::Int,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing;
    p_min::Float64 = 1e-15
)::Vector{Int}
    S = size(P_sp, 1)
    u_start == u_end && return Int[u_start]
    (land_mask !== nothing && (land_mask[u_start] || land_mask[u_end])) && return Int[u_start]

    Pt = SparseMatrixCSC(P_sp')
    rows_t = rowvals(Pt)
    vals_t = nonzeros(Pt)

    prev = zeros(Int, S)
    visited = falses(S)
    visited[u_start] = true
    queue = Int[u_start]
    head = 1

    while head <= length(queue)
        u = queue[head]
        head += 1
        for ptr in nzrange(Pt, u)
            Float64(vals_t[ptr]) > p_min || continue
            v = rows_t[ptr]
            (v == u || visited[v]) && continue
            (land_mask !== nothing && land_mask[v]) && continue
            visited[v] = true
            prev[v] = u
            if v == u_end
                path = Int[u_end]
                cur = u_end
                while cur != u_start
                    cur = prev[cur]
                    cur == 0 && return Int[u_start]
                    pushfirst!(path, cur)
                end
                return path
            end
            push!(queue, v)
        end
    end
    return Int[u_start]
end

"""
    predict_bridge_path(
        P::AbstractMatrix{<:Real},
        release::Int,
        recapture::Int,
        k::Union{Nothing, Int} = nothing;
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        rng::Union{Nothing, AbstractRNG} = nothing,
        p_min::Real = 1e-15
    ) -> Vector{Int}

Sample a conditioned discrete-time Markov-bridge trajectory connecting `release`
to `recapture` under row-stochastic transition kernel `P`.

At intermediate step ``\\tau \\in \\{1, \\dots, K\\}``, transitions from ``u`` to ``v``
are drawn with probability proportional to ``P_{u, v} \\, b_{\\tau+1}(v)``, where
``b_{\\tau} = P \\, b_{\\tau+1}`` (with ``b_{K+1} = e_{\\text{recapture}}``) is the
L1-scaled backward reachability vector. In homogeneous-HSI regions where
``\\Delta\\text{HSI} = 0``, the bridge reduces to a 2D diffusive Brownian bridge
across navigable marine cells rather than collapsing onto a shortest-path line.
"""
function predict_bridge_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    rng::Union{Nothing, AbstractRNG} = nothing,
    p_min::Real = 1e-15
)::Vector{Int}
    S = size(P, 1)
    size(P, 2) == S || throw(ArgumentError("Transition matrix P must be square, got $(size(P))."))
    (1 <= release <= S && 1 <= recapture <= S) || throw(ArgumentError(
        "Release ($release) and recapture ($recapture) must lie within 1:$S."
    ))

    P_sp = P isa SparseMatrixCSC ? P : sparse(Float64.(P))
    all(isfinite, nonzeros(P_sp)) || throw(DomainError(
        P, "Transition matrix P contains NaN or Inf entries."
    ))

    if release == recapture
        (k === nothing || k <= 0) && return Int[release]
        if Float64(P_sp[release, release]) > Float64(p_min) &&
           (land_mask === nothing || !land_mask[release])
            return fill(release, min(k, 365) + 1)
        end
        return _exact_k_max_prob_path(P_sp, release, recapture, min(k, 365);
                                      land_mask = land_mask, p_min = p_min)
    end

    bfs_path = _bfs_marine_shortest_path(P_sp, release, recapture, land_mask;
                                         p_min = Float64(p_min))
    length(bfs_path) < 2 && return Int[release]
    min_hops = length(bfs_path) - 1

    K = if k === nothing
        clamp(max(2, 2 * min_hops), min_hops, 180)
    else
        k <= 0 && return Int[release]
        max(min_hops, min(k, 365))
    end

    b_vecs = Vector{Vector{Float64}}(undef, K + 1)
    b_end = zeros(Float64, S)
    b_end[recapture] = 1.0
    b_vecs[K + 1] = b_end

    for tau in K:-1:1
        prv = P_sp * b_vecs[tau + 1]
        if land_mask !== nothing
            @inbounds for u in 1:S
                land_mask[u] && (prv[u] = 0.0)
            end
        end
        s = sum(prv)
        s > 0.0 && (prv ./= s)
        b_vecs[tau] = prv
    end

    b_vecs[1][release] > 0.0 || return bfs_path

    Pt = SparseMatrixCSC(P_sp')
    rows_t = rowvals(Pt)
    vals_t = nonzeros(Pt)
    local_rng = rng === nothing ?
        MersenneTwister(hash((release, recapture, K, 0x54657468))) : rng

    path = Vector{Int}(undef, K + 1)
    path[1] = release
    curr = release
    nbr_nodes = Int[]
    nbr_weights = Float64[]

    for tau in 1:K
        empty!(nbr_nodes)
        empty!(nbr_weights)
        b_next = b_vecs[tau + 1]
        tot_w = 0.0
        for ptr in nzrange(Pt, curr)
            p_uv = Float64(vals_t[ptr])
            p_uv > Float64(p_min) || continue
            v = rows_t[ptr]
            (land_mask !== nothing && land_mask[v]) && continue
            w = p_uv * b_next[v]
            if w > 0.0
                push!(nbr_nodes, v)
                push!(nbr_weights, w)
                tot_w += w
            end
        end
        if tot_w <= 0.0 || isempty(nbr_nodes)
            return bfs_path
        end
        u_draw = rand(local_rng) * tot_w
        csum = 0.0
        chosen = nbr_nodes[end]
        for idx in eachindex(nbr_nodes)
            csum += nbr_weights[idx]
            if u_draw <= csum
                chosen = nbr_nodes[idx]
                break
            end
        end
        curr = chosen
        path[tau + 1] = curr
    end

    path[end] = recapture
    return path
end

"""
    smooth_marine_path(
        path::Vector{Int},
        centroids::AbstractVector;
        land_polygons = nothing,
        centroids_lonlat = nothing
    ) -> Vector{Int}

Applies line-of-sight shortcutting ("string-pulling") to an animal trajectory
on a discrete mesh, removing artificial cell-to-cell hexagonal zig-zagging
while strictly preserving clearance around terrestrial land barriers (islands,
peninsulas, headlands).

# Mathematical Formulation
For path waypoints \$\\mathbf{w} = [u_1, u_2, \\dots, u_m]\$, the algorithm
casts a line of sight between non-adjacent waypoints \$u_i\$ and \$u_j\$
(\$j > i + 1\$). If the line segment:
```math
L(u_i, u_j) = \\{ (1 - t) \\mathbf{c}_{u_i} + t \\mathbf{c}_{u_j} \\mid t \\in [0, 1] \\}
```
does not intersect any terrestrial barrier polygon (evaluated via
`_line_crosses_polygon_or_in`), intermediate waypoints \$u_{i+1}, \\dots, u_{j-1}\$
are removed. If an intersection occurs, waypoints around the headland are kept.

# Arguments
- `path`: Ordered sequence of spatial unit indices.
- `centroids`: Spatial centroids coordinate vector matching unit indices.
- `land_polygons`: Terrestrial boundary polygons (defaults to `:default`).
- `centroids_lonlat`: Optional coordinates in degrees for polygon intersection.

# Returns
- `Vector{Int}`: Smoothed subset of waypoints with direct lines of sight.
"""
function smooth_marine_path(
    path::Vector{Int},
    centroids::AbstractVector;
    land_polygons = nothing,
    centroids_lonlat = nothing
)::Vector{Int}
    isempty(path) && return Int[]
    # Collapse consecutive stationary self-loops first.
    dedup = Int[path[1]]
    for k in 2:length(path)
        if path[k] != dedup[end]
            push!(dedup, path[k])
        end
    end
    if length(dedup) <= 2
        return dedup
    end

    # Only shortcut across non-adjacent cells when explicit terrestrial barrier
    # polygons are supplied; without barrier geometry, line-of-sight string-pulling
    # would collapse every trajectory to its two endpoints [u_1, u_end].
    polys = land_polygons in (:none, :false, false) ? nothing : land_polygons
    polys === nothing && return dedup

    coords_check = centroids_lonlat !== nothing ? centroids_lonlat : centroids
    is_planar = abs(coords_check[1][1]) > 180.0 || abs(coords_check[1][2]) > 90.0

    if is_planar
        @warn "smooth_marine_path: centroids appear to be planar coordinates " *
              "while land_polygons are in geographic degrees; provide " *
              "centroids_lonlat to enable barrier-respecting smoothing." maxlog = 1
        return dedup
    end

    smoothed = Int[dedup[1]]
    curr_idx = 1
    n_pts = length(dedup)

    while curr_idx < n_pts
        furthest_idx = curr_idx + 1
        for look_idx in n_pts:-1:(curr_idx + 2)
            p_curr = (
                Float64(coords_check[dedup[curr_idx]][1]),
                Float64(coords_check[dedup[curr_idx]][2])
            )
            p_look = (
                Float64(coords_check[dedup[look_idx]][1]),
                Float64(coords_check[dedup[look_idx]][2])
            )
            if !_line_crosses_polygon_or_in(p_curr, p_look, polys)
                furthest_idx = look_idx
                break
            end
        end
        push!(smoothed, dedup[furthest_idx])
        curr_idx = furthest_idx
    end

    return smoothed
end

"""
    predict_path(P::AbstractMatrix{<:Real}, release::Int, recapture::Int,
                 k::Union{Nothing, Int} = nothing;
                 centroids = nothing, method = :bridge, land_mask = nothing,
                 rng = nothing) -> Vector{Int}

Predicts an individual trajectory between `release` and `recapture` using either
conditioned Markov-bridge sampling (`method = :bridge`, default) or exact fixed-horizon
Viterbi decoding (`method = :viterbi`).
"""
function predict_path(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    centroids = nothing,
    method::Symbol = :bridge,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    use_resolvent::Bool = false,
    rng::Union{Nothing, AbstractRNG} = nothing
)::Vector{Int}
    if method == :bridge || (method == :viterbi && k === nothing)
        return predict_bridge_path(
            P, release, recapture, k;
            land_mask = land_mask,
            rng = rng
        )
    elseif method == :viterbi
        S = size(P, 1)
        if !(1 <= release <= S) || !(1 <= recapture <= S)
            throw(ArgumentError(
                "Release unit ($release) and recapture unit ($recapture) must be within 1:$S."
            ))
        end
        if k < 1
            return [release]
        end
        if release == recapture
            return fill(release, k + 1)
        end
        if k == 1
            has_p = Float64(P[release, recapture]) > 1e-12
            no_land = land_mask === nothing ||
                      (!land_mask[release] && !land_mask[recapture])
            if has_p && no_land
                return [release, recapture]
            else
                return predict_bridge_path(
                    P, release, recapture, k;
                    land_mask = land_mask, rng = rng
                )
            end
        end

        P_csc = P isa SparseMatrixCSC ? P : SparseMatrixCSC(P)
        Pt_csc = SparseMatrixCSC(P_csc')

        delta = fill(-1e12, S, k + 1)
        psi   = zeros(Int, S, k + 1)
        delta[release, 1] = 0.0

        for tau in 2:(k + 1)
            prev_tau = tau - 1
            @inbounds for j in 1:S
                if land_mask !== nothing && land_mask[j]
                    continue
                end

                best_val = -Inf
                best_prev = 1

                col_start = Pt_csc.colptr[j]
                col_end   = Pt_csc.colptr[j+1] - 1

                for ptr in col_start:col_end
                    i = Pt_csc.rowval[ptr]
                    if land_mask !== nothing && land_mask[i]
                        continue
                    end

                    p_ij = Pt_csc.nzval[ptr]
                    if p_ij > 1e-15
                        score = delta[i, prev_tau] + log(p_ij)
                        if score > best_val
                            best_val = score
                            best_prev = i
                        end
                    end
                end

                if best_val > -Inf
                    delta[j, tau] = best_val
                    psi[j, tau]   = best_prev
                end
            end
        end

        if delta[recapture, k + 1] <= -1e11
            return predict_bridge_path(
                P, release, recapture, k;
                land_mask = land_mask, rng = rng
            )
        end

        path = zeros(Int, k + 1)
        path[k + 1] = recapture
        for tau in (k + 1):-1:2
            path[tau - 1] = psi[path[tau], tau]
        end
        return path
    else
        throw(ArgumentError("Unknown path method '$method'. Expected :bridge or :viterbi."))
    end
end

"""
    predict_steady_state_corridor(P::AbstractMatrix{<:Real}, release::Int, recapture::Int; alpha=0.99, land_mask=nothing) -> Vector{Float64}

Computes a time-independent (infinite-horizon) connectivity corridor representing the 
probability of ever visiting a spatial unit `x` en route from `release` to `recapture`.
This leverages the fast sparse resolvent transition method.
"""
function predict_steady_state_corridor(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int;
    alpha::Float64 = 0.99,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Float64}
    S = size(P, 1)
    P_sp = P isa SparseMatrixCSC ? P : SparseMatrixCSC(P)
    
    # 1. Expected visits to recapture from all x: (I - aP) x_to_v = e_v
    x_to_v = resolvent_expected_visits(P_sp, recapture; alpha=alpha)
    
    # 2. Expected visits to all x from release: (I - aP') u_to_x = e_u
    M_t = sparse(I, S, S) - alpha * P_sp'
    b_u = zeros(Float64, S)
    b_u[release] = 1.0
    u_to_x = try M_t \ b_u catch e fill(1.0, S) end
    u_to_x = max.(u_to_x, 0.0)
    
    corridor = zeros(Float64, S)
    for x in 1:S
        if land_mask !== nothing && land_mask[x]
            continue
        end
        corridor[x] = u_to_x[x] * x_to_v[x]
    end
    
    sum_c = sum(corridor)
    if sum_c > 0.0
        corridor ./= sum_c
    end
    return corridor
end


"""
    predict_corridor(P::AbstractMatrix{<:Real}, release::Int, recapture::Int, k::Int;
                     land_mask=nothing) -> Matrix{Float64}

Computes the Markov bridge probability distribution across all spatial units at each
intermediate time step ``\\tau \\in \\{0, 1, \\dots, k\\}``:

```math
\\mathbb{P}(X_\\tau = j \\mid X_0 = u_{\\text{rel}}, X_k = u_{\\text{rec}}) = 
\\frac{[P^\\tau]_{u_{\\text{rel}}, j} \\cdot [P^{k - \\tau}]_{j, u_{\\text{rec}}}}{[P^k]_{u_{\\text{rel}}, u_{\\text{rec}}}}
```

# Arguments
- `P`: Row-stochastic transition matrix (size ``S \\times S``).
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Total discrete time steps elapsed between release and recapture (``k \\ge 1``).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units). When provided,
  all probability mass on land units is strictly zeroed out and columns are renormalized
  over navigable marine units.

# Returns
- `Matrix{Float64}`: Array of size ``(S, k + 1)`` where column ``\\tau + 1`` gives the spatial
  probability distribution over all ``S`` units at time step ``\\tau``. Each column sums to 1.0.
"""
function predict_corridor(
    P::AbstractMatrix{<:Real},
    release::Int,
    recapture::Int,
    k::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    S = size(P, 1)
    if !(1 <= release <= S) || !(1 <= recapture <= S)
        error("Release unit ($release) and recapture unit ($recapture) must be within 1:$S.")
    end

    corridor = zeros(Float64, S, k + 1)

    if k <= 0
        corridor[release, 1] = 1.0
        return corridor
    end

    # Iterative sparse-vector multiplication instead of dense matrix powers
    # Forward probabilities: v_fwd[j, tau+1] = P(X_tau = j | X_0 = release)
    v_fwd = zeros(Float64, S, k + 1)
    v_fwd[release, 1] = 1.0
    P_t = P' # Transpose once for efficiency
    for tau in 1:k
        v_fwd[:, tau + 1] = P_t * v_fwd[:, tau]
    end

    # Backward probabilities: v_bwd[j, m+1] = P(X_k = recapture | X_{k-m} = j)
    v_bwd = zeros(Float64, S, k + 1)
    v_bwd[recapture, 1] = 1.0
    for m in 1:k
        v_bwd[:, m + 1] = P * v_bwd[:, m]
    end

    # Total likelihood of transitioning from release to recapture in k steps
    P_total = v_fwd[recapture, k + 1]

    if P_total <= 1e-15
        @warn "Recapture unit $recapture has near-zero reachability from release $release in $k steps."
        if land_mask !== nothing
            n_water = count(!, land_mask)
            w_val = n_water > 0 ? 1.0 / n_water : 1.0 / S
            for j in 1:S
                corridor[j, :] .= land_mask[j] ? 0.0 : w_val
            end
        else
            corridor[:, :] .= 1.0 / S
        end
        corridor[release, 1] = 1.0
        corridor[recapture, k + 1] = 1.0
        return corridor
    end

    # Markov bridge formula for each intermediate step tau = 0 .. k
    for tau in 0:k
        tau_idx = tau + 1
        rem_idx = (k - tau) + 1
        for j in 1:S
            prob_fwd = v_fwd[j, tau_idx]
            prob_bwd = v_bwd[j, rem_idx]
            corridor[j, tau_idx] = (prob_fwd * prob_bwd) / P_total
        end
        # Enforce land barrier: strictly zero probability on land
        if land_mask !== nothing
            for l in 1:S
                if land_mask[l]
                    corridor[l, tau_idx] = 0.0
                end
            end
        end
        # Normalize column
        col_sum = sum(view(corridor, :, tau_idx))
        if col_sum > 0.0
            corridor[:, tau_idx] ./= col_sum
        end
    end

    return corridor
end


"""
    predict_path(res::NamedTuple, release::Int, recapture::Int, k=nothing; kwargs...) -> Vector{Int}

Extracts the transition matrix and domain mesh from a fitted TethysPath result
NamedTuple and predicts an individual movement trajectory between release and
recapture nodes using Markov-bridge sampling (`:bridge`) or dynamic programming (`:viterbi`).
"""
function predict_path(
    res::NamedTuple,
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    group::Union{String, Symbol, Int} = 1,
    centroids = nothing,
    method::Symbol = :bridge,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    mask = land_mask !== nothing ? land_mask :
        (hasproperty(res, :land_mask) ? res.land_mask : nothing)
    cents = centroids !== nothing ? centroids :
        (hasproperty(res, :mesh) ? res.mesh : nothing)
    if hasproperty(res, :transition_matrices)
        tm = res.transition_matrices
        key = group isa Int ?
            (hasproperty(res, :group_lookup) ? res.group_lookup[group] : first(keys(tm))) :
            string(group)
        P = tm[key]
        return predict_path(
            P, release, recapture, k;
            centroids = cents,
            method = method,
            land_mask = mask
        )
    else
        throw(ArgumentError("Expected a result NamedTuple with field `:transition_matrices`."))
    end
end

"""
    predict_corridor(res::NamedTuple, release::Int, recapture::Int, k::Int; kwargs...) -> Matrix{Float64}

Extracts the group transition matrix from a fitted TethysPath result NamedTuple and
computes the space-time Markov bridge corridor matrix over ``k`` steps.

# Arguments
- `res`: Result NamedTuple containing `:transition_matrices`.
- `release`: Starting spatial unit index.
- `recapture`: Destination spatial unit index.
- `k`: Total discrete time steps (``k \\ge 1``).
- `group`: Demographic group identifier (string, symbol, or integer ID).
- `land_mask`: Optional boolean land mask vector.

# Returns
- `Matrix{Float64}`: State probability matrix of size ``S \\times (k + 1)``.
"""
function predict_corridor(
    res::NamedTuple, release::Int, recapture::Int, k::Int;
    group::Union{String, Symbol, Int} = 1,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)
    mask = land_mask !== nothing ? land_mask :
        (hasproperty(res, :land_mask) ? res.land_mask : nothing)
    if hasproperty(res, :transition_matrices)
        tm = res.transition_matrices
        key = group isa Int ?
            (hasproperty(res, :group_lookup) ? res.group_lookup[group] : first(keys(tm))) :
            string(group)
        P = tm[key]
        return predict_corridor(P, release, recapture, k; land_mask=mask)
    else
        error("Expected a result NamedTuple with field `:transition_matrices`.")
    end
end

"""
    predict_path(P_vec::AbstractVector{<:AbstractMatrix{<:Real}}, release::Int, recapture::Int, k::Int;
                 group::Union{Integer, Symbol, AbstractString} = 1,
                 land_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Vector{Int}

Group-aware overload for `predict_path` when given a vector of group-specific transition matrices.
Reconstructs the most likely sequence of spatial units (Viterbi path) for the specified group.

# Mathematical Formulation
Given group index ``g``, the dynamic programming Viterbi trellis identifies:
```math
\\mathbf{s}^* = \\arg\\max_{\\mathbf{s}} \\prod_{\\tau=1}^k [P_g]_{s_{\\tau-1}, s_\\tau}
```
subject to ``s_0 = u_{\\text{rel}}`` and ``s_k = u_{\\text{rec}}``.

# Arguments
- `P_vec`: Collection of row-stochastic transition matrices for each biological group.
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Number of discrete time intervals elapsed.
- `group`: 1-based integer index or group label.
- `land_mask`: Optional boolean mask of impermeable barrier units.

# Returns
- `Vector{Int}`: Sequence of ``k + 1`` spatial unit indices from `release` to `recapture`.
"""
function predict_path(
    P_vec::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int,
    k::Union{Nothing, Int} = nothing;
    group::Union{Integer, Symbol, AbstractString} = 1,
    centroids = nothing,
    method::Symbol = :bridge,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    if isempty(P_vec)
        throw(ArgumentError("P_vec transition kernel vector cannot be empty."))
    end
    g_idx = if group isa Integer
        Int(group)
    else
        parsed = tryparse(Int, string(group))
        parsed !== nothing ? parsed : 1
    end
    if !(1 <= g_idx <= length(P_vec))
        throw(ArgumentError("Group index $g_idx is out of bounds (1:$(length(P_vec)))."))
    end
    return predict_path(
        P_vec[g_idx], release, recapture, k;
        centroids = centroids,
        method = method,
        land_mask = land_mask
    )
end

"""
    predict_corridor(P_vec::AbstractVector{<:AbstractMatrix{<:Real}}, release::Int, recapture::Int, k::Int;
                     group::Union{Integer, Symbol, AbstractString} = 1,
                     land_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Matrix{Float64}

Group-aware overload for `predict_corridor` when given a vector of group-specific transition matrices.
Computes the Markov bridge probability distribution across spatial units for the specified group.

# Mathematical Formulation
Given group index ``g``, the Markov bridge probability at intermediate step ``\\tau`` is:
```math
\\mathbb{P}(X_\\tau = j \\mid X_0 = u_{\\text{rel}}, X_k = u_{\\text{rec}}) = 
\\frac{[P_g^\\tau]_{u_{\\text{rel}}, j} \\cdot [P_g^{k - \\tau}]_{j, u_{\\text{rec}}}}{[P_g^k]_{u_{\\text{rel}}, u_{\\text{rec}}}}
```

# Arguments
- `P_vec`: Collection of row-stochastic transition matrices for each biological group.
- `release`: Starting spatial unit index (1-indexed).
- `recapture`: Destination spatial unit index at step `k` (1-indexed).
- `k`: Number of discrete time intervals elapsed.
- `group`: 1-based integer index or group label.
- `land_mask`: Optional boolean mask of impermeable barrier units.

# Returns
- `Matrix{Float64}`: Array of size ``(S, k + 1)`` where column ``\\tau + 1`` gives the spatial
  probability distribution over all units at time step ``\\tau``.
"""
function predict_corridor(
    P_vec::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int,
    k::Int;
    group::Union{Integer, Symbol, AbstractString} = 1,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    if isempty(P_vec)
        throw(ArgumentError("P_vec transition kernel vector cannot be empty."))
    end
    g_idx = if group isa Integer
        Int(group)
    else
        parsed = tryparse(Int, string(group))
        parsed !== nothing ? parsed : 1
    end
    if !(1 <= g_idx <= length(P_vec))
        throw(ArgumentError("Group index $g_idx is out of bounds (1:$(length(P_vec)))."))
    end
    return predict_corridor(P_vec[g_idx], release, recapture, k; land_mask=land_mask)
end


# ==============================================================================
# Domain Grid, Autocorrelation Infilling & Land Barrier Engine
# ==============================================================================

# `point_in_polygon` methods are imported from `Tethys`.


# No study-area coastline is bundled with the package. Land is identified from the
# configured source -- a global land/sea raster, user-supplied polygons, or
# bathymetric depth -- so the same code serves any region.


"""
    identify_land_units(
        centroids::AbstractVector;
        land_polygons::Union{Nothing, Symbol, AbstractVector} = nothing,
        depth::Union{Nothing, AbstractVector{<:Real}} = nothing,
        depth_threshold::Real = 0.0
    ) -> Vector{Bool}

Classifies spatial partitioning units as terrestrial land (`true`) or navigable marine water
(`false`) using boundary polygons, bathymetric depth thresholds, or regional defaults.

# Mathematical Formulation
A spatial unit ``s \\in \\{1, \\dots, S\\}`` with centroid ``\\mathbf{c}_s = (x_s, y_s)`` is classified
as land if it satisfies either geometric boundary polygon inclusion or the bathymetric threshold:
```math
\\text{is\\_land}(s) = (\\exists p \\in \\mathcal{P} : \\mathbf{c}_s \\in p) \\lor (d_s \\le d_{\\text{thresh}})
```
where ``\\mathcal{P}`` is the set of land boundary polygons and ``d_s`` is the bathymetric depth.

# Arguments
- `centroids`: Vector of coordinate tuples `(x, y)` or `(lon, lat)` for all ``S`` units.
- `land_polygons`: Optional polygon or collection of polygons.
  - `nothing` or `:none`: no polygon criterion; land rests on `depth` alone.
  - `Vector{...}`: User-provided polygon rings or MultiPolygon coordinate vectors.
- `depth`: Optional bathymetric depth vector of length ``S`` (positive values indicate water depth).
- `depth_threshold`: Scalar threshold below which a unit is classified as land (default `0.0`).

# Returns
- `Vector{Bool}`: Boolean mask of length ``S`` where `true` indicates a terrestrial land unit.
"""
function identify_land_units(
    centroids::AbstractVector;
    polygons::Union{Nothing, AbstractVector} = nothing,
    land_polygons::Union{Nothing, Symbol, AbstractVector} = nothing,
    depth::Union{Nothing, AbstractVector{<:Real}} = nothing,
    depth_threshold::Real = 0.0,
    land_fraction_threshold::Real = 0.5
)::Vector{Bool}
    S = length(centroids)
    land_mask = falses(S)

    # 1. Depth thresholding
    if depth !== nothing
        if length(depth) != S
            throw(DimensionMismatch(
                "Depth vector length ($(length(depth))) must match centroids length ($S)."
            ))
        end
        for i in 1:S
            if isfinite(depth[i]) && depth[i] <= depth_threshold
                land_mask[i] = true
            end
        end
    end

    # 2. Polygon boundaries
    # `nothing` means "no polygon criterion"; land then rests entirely on
    # the depth vector, or on a mask supplied by the caller.
    polys = land_polygons in (:none, :false, false) ? nothing : land_polygons

    if polys !== nothing
        for i in 1:S
            land_mask[i] && continue
            c = centroids[i]
            x, y = Float64(c[1]), Float64(c[2])
            # Check centroid inclusion
            if point_in_polygon(x, y, polys)
                land_mask[i] = true
                continue
            end
            # Check unit polygon vertices if provided
            if polygons !== nothing && i <= length(polygons)
                u_poly = polygons[i]
                if length(u_poly) >= 3
                    n_verts = length(u_poly) - (u_poly[1] == u_poly[end] ? 1 : 0)
                    n_in = count(k -> point_in_polygon(u_poly[k][1], u_poly[k][2], polys), 1:n_verts)
                    if n_in >= ceil(Int, n_verts * land_fraction_threshold)
                        land_mask[i] = true
                    end
                end
            end
        end
    end

    return land_mask
end


"""
    apply_land_barrier(
        W::AbstractMatrix{<:Real},
        hsi::AbstractVector{<:Real},
        land_mask::AbstractVector{Bool}
    ) -> Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}

Enforces impassable terrestrial movement barriers on the spatial adjacency graph ``W`` and
habitat suitability vector ``\\mathbf{h}``, severing all topological connections across land.

# Mathematical Formulation
Let ``\\mathcal{L} = \\{l \\mid \\text{land\\_mask}[l] = \\text{true}\\}`` be the set of land units,
and ``\\mathcal{W} = \\{w \\mid \\text{land\\_mask}[w] = \\text{false}\\}`` be marine water units.
All directed and undirected edges incident to any land unit are strictly eliminated:
```math
W^{\\text{water}}_{ij} = \\begin{cases} 
W_{ij} & \\text{if } i \\in \\mathcal{W} \\land j \\in \\mathcal{W} \\\\
0 & \\text{otherwise}
\\end{cases}
```
Habitat suitability values on land are set to zero:
```math
h^{\\text{water}}_i = \\begin{cases}
h_i & \\text{if } i \\in \\mathcal{W} \\\\
0.0 & \\text{if } i \\in \\mathcal{L}
\\end{cases}
```
This guarantees that topological random walks (``T_{\\text{diff}}``) and directed advection (``A``)
have zero probability of transitioning onto or through land masses:
```math
P(X_{t+1} \\in \\mathcal{L} \\mid X_t \\in \\mathcal{W}) = 0
```

# Arguments
- `W`: Spatial graph adjacency matrix (size ``S \\times S``).
- `hsi`: Habitat suitability vector (length ``S``).
- `land_mask`: Boolean vector of length ``S`` (`true` for land units).

# Returns
- `Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}`:
  - `W_water`: Severed adjacency matrix with all land connections removed and zeros dropped.
  - `hsi_water`: Habitat suitability vector with land units zeroed out.
"""
function apply_land_barrier(
    W::AbstractMatrix{<:Real},
    hsi::AbstractVector{<:Real},
    land_mask::AbstractVector{Bool}
)::Tuple{SparseMatrixCSC{Float64, Int}, Vector{Float64}}
    S = size(W, 1)
    if length(hsi) != S || length(land_mask) != S
        throw(DimensionMismatch(
            "Dimension mismatch: W is $(S)x$(S), hsi has length $(length(hsi)), " *
            "and land_mask has length $(length(land_mask))."
        ))
    end

    W_water = copy(sparse(Float64.(W)))
    for l in findall(land_mask)
        W_water[l, :] .= 0.0
        W_water[:, l] .= 0.0
    end
    dropzeros!(W_water)

    hsi_water = copy(Float64.(hsi))
    hsi_water[land_mask] .= 0.0

    return (W_water, hsi_water)
end


function _extract_polygon_rings(polys)
    if polys isa AbstractVector && !isempty(polys)
        first_elem = first(polys)
        if first_elem isa Tuple || (first_elem isa AbstractVector && length(first_elem) == 2 && first_elem[1] isa Real)
            return [polys]
        elseif first_elem isa AbstractVector
            return polys
        end
    end
    return [polys]
end

function _line_crosses_polygon_or_in(p1, p2, polys)::Bool
    # Check midpoint inclusion in land polygon
    mid_x = (p1[1] + p2[1]) / 2.0
    mid_y = (p1[2] + p2[2]) / 2.0
    if point_in_polygon(mid_x, mid_y, polys)
        return true
    end

    # Orientation test for 2D line segment intersection
    function _ccw(A, B, C)
        return (C[2] - A[2]) * (B[1] - A[1]) > (B[2] - A[2]) * (C[1] - A[1])
    end

    function _seg_intersect(a1, a2, b1, b2)
        return (_ccw(a1, b1, b2) != _ccw(a2, b1, b2)) && (_ccw(a1, a2, b1) != _ccw(a1, a2, b2))
    end

    rings = _extract_polygon_rings(polys)
    for ring in rings
        if ring isa AbstractVector && length(ring) >= 3
            n_pts = length(ring)
            for k in 1:(n_pts - 1)
                e1 = (Float64(ring[k][1]), Float64(ring[k][2]))
                e2 = (Float64(ring[k + 1][1]), Float64(ring[k + 1][2]))
                if _seg_intersect(p1, p2, e1, e2)
                    return true
                end
            end
        end
    end
    return false
end

"""
    sever_land_crossing_edges!(
        W::AbstractMatrix{<:Real},
        centroids::AbstractVector;
        land_polygons = nothing
    ) -> Int

Inspects all non-zero edges ``(i, j)`` in the spatial adjacency matrix ``W`` and
severs any edge whose straight-line trajectory intersects a terrestrial land barrier polygon.

# Mathematical Formulation
For nodes ``i, j \\in \\mathcal{S}``, the trajectory line segment is:
```math
L_{ij} = \\{ (1 - t) \\mathbf{c}_i + t \\mathbf{c}_j \\mid t \\in [0, 1] \\}
```
If ``L_{ij} \\cap \\mathcal{P}_{\\text{land}} \\neq \\emptyset``, the edge is severed:
```math
W_{ij} \\leftarrow 0, \\quad W_{ji} \\leftarrow 0
```
preventing movement models from allowing transitions that cross overland barriers.

# Arguments
- `W`: Spatial graph adjacency matrix (size ``S \\times S``). Modified in-place if mutable.
- `centroids`: Centroids coordinate vector (length ``S``).
- `land_polygons`: Boundary polygons, or `nothing` to block nothing.

# Returns
- `Int`: Total number of directed edges severed.
"""
function sever_land_crossing_edges!(
    W::AbstractMatrix{<:Real},
    centroids::AbstractVector;
    land_polygons = nothing
)::Int
    # `nothing` means no polygon criterion; only explicit polygons block edges.
    polys = land_polygons in (:none, :false, false) ? nothing : land_polygons

    polys === nothing && return 0

    S = size(W, 1)
    if length(centroids) != S
        throw(DimensionMismatch(
            "centroids length ($(length(centroids))) must match W dimensions ($S)."
        ))
    end

    severed_count = 0
    if W isa SparseMatrixCSC
        rows = rowvals(W)
        for col in 1:S
            c_col = centroids[col]
            p_col = (Float64(c_col[1]), Float64(c_col[2]))
            for k in nzrange(W, col)
                row = rows[k]
                if row > col
                    c_row = centroids[row]
                    p_row = (Float64(c_row[1]), Float64(c_row[2]))
                    if _line_crosses_polygon_or_in(p_col, p_row, polys)
                        W[row, col] = 0.0
                        W[col, row] = 0.0
                        severed_count += 2
                    end
                end
            end
        end
        dropzeros!(W)
    else
        for i in 1:S
            c_i = centroids[i]
            p_i = (Float64(c_i[1]), Float64(c_i[2]))
            for j in (i + 1):S
                if W[i, j] != 0.0 || W[j, i] != 0.0
                    c_j = centroids[j]
                    p_j = (Float64(c_j[1]), Float64(c_j[2]))
                    if _line_crosses_polygon_or_in(p_i, p_j, polys)
                        W[i, j] = 0.0
                        W[j, i] = 0.0
                        severed_count += 2
                    end
                end
            end
        end
    end

    return severed_count
end

"""
    compact_marine_mesh(mesh, land_mask::AbstractVector{Bool}) -> NamedTuple

Extracts and compacts a spatial mesh into an active marine water sub-domain by
filtering out all terrestrial land units and reindexing spatial units from ``1`` to
``S_{\\text{water}}``.

# Mathematical Formulation
Given spatial unit set ``\\mathcal{S} = \\{1, \\dots, S\\}`` and boolean indicator
``\\text{land\\_mask}``, the navigable marine sub-domain is defined by:
```math
\\mathcal{M} = \\{ i \\in \\mathcal{S} \\mid \\neg \\text{land\\_mask}[i] \\}
```
The adjacency graph is subsetted to the induced sub-graph:
```math
W_{\\text{marine}} = W[\\mathcal{M}, \\mathcal{M}]
```
preserving all marine graph topology while reducing state dimensionality.

# Arguments
- `mesh`: Spatial mesh NamedTuple containing `:centroids`, `:polygons`, `:n_units`, and `:W`.
- `land_mask`: Boolean vector of length ``S`` (`true` for land units).

# Returns
- `NamedTuple`: Compacted marine mesh with updated `:n_units`, `:W`, `:centroids_lonlat`,
  `:polygons_lonlat`, and the mapping vector `:water_indices`.
"""
function compact_marine_mesh(mesh, land_mask::AbstractVector{Bool})::NamedTuple
    water_idx = findall(!, land_mask)
    S_water = length(water_idx)

    cents_ll = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat[water_idx] :
               (hasproperty(mesh, :centroids) ? mesh.centroids[water_idx] : Tuple{Float64, Float64}[])
    polys_ll = hasproperty(mesh, :polygons_lonlat) ? mesh.polygons_lonlat[water_idx] :
               (hasproperty(mesh, :polygons) ? mesh.polygons[water_idx] : Vector{Vector{Tuple{Float64, Float64}}}())
    cents_km = hasproperty(mesh, :centroids_km) ? mesh.centroids_km[water_idx] : nothing
    polys_km = hasproperty(mesh, :polygons_km) ? mesh.polygons_km[water_idx] : nothing

    W_sub = copy(mesh.W[water_idx, water_idx])
    if W_sub isa SparseMatrixCSC
        dropzeros!(W_sub)
    end

    res = Dict{Symbol, Any}(
        :n_units          => S_water,
        :centroids        => cents_ll,
        :centroids_lonlat => cents_ll,
        :polygons         => polys_ll,
        :polygons_lonlat  => polys_ll,
        :W                => W_sub,
        :water_indices    => water_idx
    )
    if cents_km !== nothing
        res[:centroids_km] = cents_km
    end
    if polys_km !== nothing
        res[:polygons_km] = polys_km
    end
    if hasproperty(mesh, :radius_km)
        res[:radius_km] = mesh.radius_km
    end
    if hasproperty(mesh, :areas_km2)
        res[:areas_km2] = mesh.areas_km2[water_idx]
    end

    return NamedTuple(res)
end


"""
    infill_spatial_hsi(
        hsi_raw::AbstractVector{<:Real},
        W::AbstractMatrix{<:Real},
        known_mask::AbstractVector{Bool};
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        centroids::Union{Nothing, AbstractVector} = nothing,
        rho::Real = 0.95,
        baseline_quantile::Real = 0.1,
        fallback_decay_km::Real = 50.0
    ) -> Vector{Float64}

Infills missing Habitat Suitability Index (HSI) values outside the primary survey domain
using a data-driven graph-Laplacian screened Dirichlet system constrained to marine water channels.

# Mathematical Formulation
The spatial domain is partitioned into known survey units ``\\mathcal{K}``, unknown marine units
``\\mathcal{U}``, and terrestrial land units ``\\mathcal{L}``. On the severed water graph
``W^{\\text{water}}``, the conditional expectation of unknown HSI values satisfies:
```math
(D_{\\mathcal{UU}} - \\rho W_{\\mathcal{UU}}) \\mathbf{h}_{\\mathcal{U}} = 
\\rho W_{\\mathcal{UK}} \\mathbf{h}_{\\mathcal{K}} + (1 - \\rho) D_{\\mathcal{UU}} \\mu_0 \\mathbf{1}
```
where:
- ``D_{\\mathcal{UU}} = \\operatorname{diag}\\left( \\sum_{j \\in \\mathcal{W}} W_{ij} \\right)_{i \\in \\mathcal{U}}``
  is the total degree of each unknown node in the water graph.
- ``W_{\\mathcal{UU}}`` is the internal adjacency submatrix among unknown water nodes.
- ``W_{\\mathcal{UK}}`` connects unknown water nodes to adjacent known survey nodes.
- ``\\rho \\in (0, 1)`` controls spatial autocorrelation strength (default 0.95).
- ``\\mu_0 = \\operatorname{quantile}(\\mathbf{h}_{\\mathcal{K}}, \\text{baseline\\_quantile})``
  is the empirical conservative baseline HSI for distant unsampled marine waters.

Because ``W^{\\text{water}}`` has all land edges severed, the Dirichlet diffusion flows strictly
around peninsulas and through marine straits (e.g. Cabot Strait), never jumping across land.
Isolated water components without graph connections to known units decay smoothly to ``\\mu_0``
or utilize an exponential spatial covariance fallback ``K(d) = \\exp(-d / \\ell)``.

# Arguments
- `hsi_raw`: Vector of observed HSI values (length ``S``).
- `W`: Spatial graph adjacency matrix (size ``S \\times S``).
- `known_mask`: Boolean vector of length ``S`` (`true` where HSI is observed).
- `land_mask`: Optional boolean vector of length ``S`` (`true` for land units).
- `centroids`: Optional coordinates vector for spatial distance fallback on disconnected units.
- `rho`: Spatial autocorrelation strength parameter in ``(0, 1)`` (default 0.95).
- `baseline`: Optional scalar baseline prior mean for unobserved regions. If `nothing`,
  defaults to empirical quantile `baseline_quantile` of known values.
- `baseline_quantile`: Empirical quantile of known HSI used as prior baseline (default 0.1).
- `fallback_decay_km`: Spatial correlation length scale in km for disconnected units (default 50.0).

# Returns
- `Vector{Float64}`: Infilled HSI vector of length ``S`` bounded in ``[0, 1]`` with land units set to 0.0.
"""
function infill_spatial_hsi(
    hsi_raw::AbstractVector{<:Real},
    W::AbstractMatrix{<:Real},
    known_mask::AbstractVector{Bool};
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    centroids::Union{Nothing, AbstractVector} = nothing,
    rho::Real = 0.95,
    baseline::Union{Nothing, Real} = nothing,
    baseline_quantile::Real = 0.1,
    fallback_decay_km::Real = 50.0
)::Vector{Float64}
    S = size(W, 1)
    if length(hsi_raw) != S || length(known_mask) != S
        throw(DimensionMismatch("hsi_raw, W, and known_mask dimensions must match."))
    end

    mask_l = land_mask === nothing ? falses(S) : land_mask
    hsi_out = copy(Float64.(hsi_raw))

    # Land units are set to 0.0
    hsi_out[mask_l] .= 0.0

    # Unknown water units
    unknown_idx = findall(.!known_mask .& .!mask_l)
    known_idx   = findall(known_mask .& .!mask_l)

    if isempty(unknown_idx)
        return hsi_out
    end

    # Baseline prior mean from known survey values or explicit baseline
    known_vals = filter(isfinite, hsi_out[known_idx])
    mu0 = if baseline !== nothing
        Float64(baseline)
    elseif isempty(known_vals)
        0.2
    else
        Float64(quantile(known_vals, baseline_quantile))
    end

    # Graph degree vector (summing only water connections)
    W_sp = sparse(W)
    D = vec(sum(W_sp, dims=2))

    W_uu = W_sp[unknown_idx, unknown_idx]
    W_uk = W_sp[unknown_idx, known_idx]
    D_uu = Diagonal(D[unknown_idx])

    rho_val = Float64(rho)
    (isfinite(rho_val) && 0.0 <= rho_val < 1.0) || throw(DomainError(
        rho, "CAR spatial autocorrelation rho must lie in [0.0, 1.0), got $rho"
    ))
    A_mat = D_uu - rho_val * W_uu
    b_vec = rho_val * (W_uk * hsi_out[known_idx]) + (1.0 - rho_val) * (D[unknown_idx] .* mu0)

    # For any isolated nodes (degree == 0), regularize diagonal to yield mu0
    for (k_sub, u) in enumerate(unknown_idx)
        if D[u] < 1e-12
            A_mat[k_sub, k_sub] = 1.0
            b_vec[k_sub] = mu0
        end
    end

    # Solve the screened Dirichlet system
    h_u = try
        A_mat \ b_vec
    catch e
                    @warn "infill_spatial_hsi: direct solve failed; using regularized solve." exception = (e, catch_backtrace())
        (A_mat + 1e-4 * I) \ b_vec
    end

    # Fallback distance decay for units with no direct path to survey area
    if centroids !== nothing && !isempty(known_idx)
        tree_known = KDTree(hcat([[Float64(c[1]), Float64(c[2])] for c in centroids[known_idx]]...))
        for (k_sub, u) in enumerate(unknown_idx)
            has_survey_conn = sum(W_uk[k_sub, :]) > 0.0
            if !has_survey_conn
                c_u = centroids[u]
                idx_nn, dists = knn(tree_known, [Float64(c_u[1]), Float64(c_u[2])], 1)
                nn_unit = known_idx[first(idx_nn)]
                d_val = first(dists)
                w_dist = exp(-d_val / fallback_decay_km)
                h_u[k_sub] = w_dist * hsi_out[nn_unit] + (1.0 - w_dist) * mu0
            end
        end
    end

    # Bound filled values in [0.0, 1.0]
    for (k_sub, u) in enumerate(unknown_idx)
        hsi_out[u] = clamp(h_u[k_sub], 0.0, 1.0)
    end

    return hsi_out
end


"""
    construct_full_movement_domain(
        lon_vec::AbstractVector{<:Real},
        lat_vec::AbstractVector{<:Real};
        radius_km::Real = 15.0,
        land_polygons = nothing,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest
    ) -> NamedTuple

Generates a unified data-driven spatial grid covering the complete movement domain
(including areas outside the core survey domain), identifies terrestrial land units, and
severs land edges to form contiguous marine movement channels.

# Mathematical Formulation
A regular planar hexagonal lattice of circumradius ``r`` is generated across the bounding
envelope of all spatial observations:
```math
\\Delta x = \\sqrt{3} r, \\quad \\Delta y = \\frac{3}{2} r
```
Centroids ``\\mathbf{c}_s`` are classified as land via `identify_land_units`. Land-water edges
in graph ``W`` are severed via `apply_land_barrier`, creating a topological state space where
marine passages remain fully connected while landmasses act as impenetrable barriers.

# Arguments
- `lon_vec, lat_vec`: Geographic coordinates encompassing the full observation domain.
- `radius_km`: Hexagon circumradius in km (default 15.0 km).
- `land_polygons`: Optional land boundary polygons (defaults to `:default`).
- `depth`: Optional depth vector matching units.
- `depth_threshold`: Bathymetric cutoff below which units are classified as land (default 0.0).
- `crs`: Coordinate reference system (default local tangent projection).
- `datum`: Reference ellipsoid datum (default `WGS84Latest`).
- `sppoly_bounds`: Optional `(min_lon, min_lat, max_lon, max_lat)` bounding coordinates
  of the spatial-unit file. The mesh extent becomes the **union** of these and the
  telemetry extent, so the declared domain is meshed in full on every edge.

# Returns
- `NamedTuple`:
  - `centroids_km`, `centroids_lonlat`: Centroid coordinates in km and degrees.
  - `polygons_km`, `polygons_lonlat`: Hexagonal cell boundary vertex rings.
  - `n_units::Int`: Total count of spatial units ``S``.
  - `W`: Cleaned adjacency matrix with land edges severed.
  - `W_raw`: Original geometric adjacency matrix prior to land barrier severing.
  - `land_mask`: Boolean vector of length ``S`` indicating land units.
  - `radius_km`: Hexagon radius in km.
  - `areas_km2`: Cell area per unit.
  - `center_lon`, `center_lat`: Projection origin.
"""
function construct_full_movement_domain(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    radius_km::Real = 15.0,
    land_polygons = nothing,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest,
    sppoly_bounds = nothing
)::NamedTuple
    lon_min, lon_max = extrema(lon_vec)
    lat_min, lat_max = extrema(lat_vec)

    if sppoly_bounds !== nothing
        # Union, not clamp. The spatial-unit file declares the study domain, so any
        # part of it reaching further out than the telemetry still has to be meshed.
        # This used to take `max` on the west and south edges, which did the opposite:
        # it pulled the extent *inside* the footprint and left the footprint's own
        # western and southern slivers with no cells at all. On the snow crab window
        # the footprint reaches lon -65.59 and lat 42.99 while the telemetry reaches
        # -65.42 and 43.02, so the clamp silently discarded both.
        #
        # Expanding the mesh cannot over-cover the domain: the footprint filter
        # trims it afterwards, whereas a too-small mesh can never be recovered.
        lon_min = min(lon_min, sppoly_bounds[1])
        lat_min = min(lat_min, sppoly_bounds[2])
        lon_max = max(lon_max, sppoly_bounds[3])
        lat_max = max(lat_max, sppoly_bounds[4])
    end

    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0

    # Planar bounding coordinates
    xy_sw = lonlat_to_xy_km(
        lon_min, lat_min;
        center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
    )
    xy_ne = lonlat_to_xy_km(
        lon_max, lat_max;
        center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
    )

    r = Float64(radius_km)
    dx = sqrt(3.0) * r
    dy = 1.5 * r

    x_min = min(xy_sw[1], xy_ne[1]) - r
    x_max = max(xy_sw[1], xy_ne[1]) + r
    y_min = min(xy_sw[2], xy_ne[2]) - r
    y_max = max(xy_sw[2], xy_ne[2]) + r

    row_min = floor(Int, y_min / dy)
    row_max = ceil(Int, y_max / dy)

    centroids_km = Tuple{Float64, Float64}[]
    for row in row_min:row_max
        yk = row * dy
        xoff = isodd(row) ? (dx / 2.0) : 0.0
        col_min = floor(Int, (x_min - xoff) / dx)
        col_max = ceil(Int, (x_max - xoff) / dx)
        for col in col_min:col_max
            xk = col * dx + xoff
            push!(centroids_km, (xk, yk))
        end
    end

    S = length(centroids_km)
    centroids_lonlat = [
        xy_km_to_lonlat(
            c[1], c[2];
            center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
        )
        for c in centroids_km
    ]

    # Flat-top hexagon vertices
    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)
    polygons_km     = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    polygons_lonlat = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)

    for i in 1:S
        cx, cy = centroids_km[i]
        verts_km = [(cx + r * cos(a), cy + r * sin(a)) for a in hex_angles]
        push!(verts_km, verts_km[1])
        polygons_km[i] = verts_km
        polygons_lonlat[i] = [
            xy_km_to_lonlat(
                v[1], v[2];
                center_lon=center_lon, center_lat=center_lat, crs=crs, datum=datum
            )
            for v in verts_km
        ]
    end

    # Adjacency graph W via KDTree
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = centroids_km[i][1]
        c_mat[2, i] = centroids_km[i][2]
    end
    tree = KDTree(c_mat)
    adj_thresh = sqrt(3.0) * r * 1.05

    rows_idx = Int[]
    cols_idx = Int[]
    for i in 1:S
        nbrs = inrange(tree, [centroids_km[i][1], centroids_km[i][2]], adj_thresh)
        for j in nbrs
            if j != i
                push!(rows_idx, i)
                push!(cols_idx, j)
            end
        end
    end
    W_init = sparse(rows_idx, cols_idx, ones(Float64, length(rows_idx)), S, S)
    W_init = max.(W_init, W_init')

    # Identify land units with area fraction check
    land_mask = identify_land_units(
        centroids_lonlat;
        polygons=polygons_lonlat,
        land_polygons=land_polygons,
        depth=depth,
        depth_threshold=depth_threshold
    )

    # Block land edges in W and topological land-crossing links
    W_water, _ = apply_land_barrier(W_init, zeros(Float64, S), land_mask)
    sever_land_crossing_edges!(W_water, centroids_lonlat; land_polygons=land_polygons)

    area_km2 = (3.0 * sqrt(3.0) / 2.0) * r^2

    return (
        centroids        = centroids_lonlat,
        centroids_km     = centroids_km,
        centroids_lonlat = centroids_lonlat,
        polygons         = polygons_lonlat,
        polygons_km      = polygons_km,
        polygons_lonlat  = polygons_lonlat,
        n_units          = S,
        W                = W_water,
        W_raw            = W_init,
        land_mask        = land_mask,
        radius_km        = r,
        areas_km2        = fill(area_km2, S),
        center_lon       = center_lon,
        center_lat       = center_lat
    )
end


"""
    prepare_movement_data(
        tagging::DataFrame;
        hsi_file::Union{Nothing, AbstractString} = nothing,
        sppoly_file::Union{Nothing, AbstractString} = nothing,
        radius_km::Real = 15.0,
        time_interval::Symbol = :monthly,
        land_polygons = nothing,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest,
        ref_doy::Int = 182,
        verbose::Bool = true
    ) -> NamedTuple

High-level end-to-end data preparation pipeline for individual animal movement and telemetry.
Builds the full-domain grid, classifies and blocks terrestrial land barriers, reshards
and autocorrelates HSI into external marine areas, maps telemetry observations, and extracts
mark-recapture transition events with biological group stratifications.

# Arguments
- `tagging::DataFrame`: Raw telemetry records with `:tagid`, `:lon`, `:lat`, `:time`, `:tag`.
- `hsi_file`: Path to environmental Habitat Suitability Index JLD2 file.
- `sppoly_file`: Path to survey spatial polygons JLD2 file defining known HSI units.
- `radius_km`: Spatial hexagon circumradius in km (default 15.0 km).
- `time_interval`: Discrete transition interval (`:monthly`, `:weekly`, `:biweekly`, `:daily`).
- `land_polygons`: Terrestrial boundary polygons (defaults to `:default`).
- `depth`: Optional depth vector matching units.
- `depth_threshold`: Bathymetric cutoff below which units are classified as land (default 0.0).
- `crs`: Target coordinate reference system (default local tangent projection).
- `datum`: Reference ellipsoid datum (default `WGS84Latest`).
- `ref_doy`: Annual reference survey day of year (default 182).
- `verbose`: Toggle progress logging to console.
- `pre_mapped`: Optional pre-computed mesh NamedTuple to turn off dynamic resharding.

# Returns
- `NamedTuple`:
  - `tagging`: Processed telemetry DataFrame with `:s_idx` and `:hsi`.
  - `mesh`: Full-domain grid NamedTuple.
  - `W`: Adjacency matrix with severed land boundaries.
  - `hsi_vec`: Infilled spatial HSI vector (length ``S``).
  - `monthly_hsi`: Monthly HSI matrix (size ``S \\times T``).
  - `month_lookup`: Mapping of `(year, month)` to column index.
  - `years`: Vector of survey years.
  - `obs`: Extracted mark-recapture event pairs DataFrame.
  - `group_lookup`: Biological stratum dictionary mapping.
  - `land_mask`: Boolean vector of length ``S`` denoting terrestrial barrier units.
"""
function prepare_movement_data(
    tagging::DataFrame;
    hsi_file::Union{Nothing, AbstractString} = nothing,
    sppoly_file::Union{Nothing, AbstractString} = nothing,
    radius_km::Real = 15.0,
    time_interval::Symbol = :daily,
    land_polygons = nothing,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest,
    ref_doy::Int = 182,
    verbose::Bool = true,
    pre_mapped = nothing
)::NamedTuple
    tag_df = copy(tagging)

    # Filter dead records
    if hasproperty(tag_df, :is_dead)
        filter!(:is_dead => d -> !coalesce(d, false), tag_df)
    end

    # Resolve time column (:timestamp, :time, :date, etc.)
    time_col = _detect_time_column(tag_df; allow_nothing = true)
    if time_col === nothing
        throw(ArgumentError(
            "tagging DataFrame must contain a date/time column (:timestamp, :time, or :date)."
        ))
    end

    # Parse timestamps systematically and compute continuous decimal years
    n_records = nrow(tag_df)
    parsed_dates = Vector{Union{Date, Nothing}}(undef, n_records)
    decimal_times = Vector{Union{Float64, Nothing}}(undef, n_records)

    raw_times = tag_df[!, time_col]
    for i in 1:n_records
        rt = raw_times[i]
        d = _parse_flexible_date(rt)
        if d !== nothing
            parsed_dates[i] = d
            decimal_times[i] = _date_to_decimal_year(d)
        elseif rt isa Real && !isnan(rt) && 1900.0 <= rt <= 2100.0
            decimal_times[i] = Float64(rt)
            yr = floor(Int, rt)
            frac = rt - yr
            diy = Dates.isleapyear(yr) ? 366 : 365
            doy = min(diy, max(1, round(Int, frac * diy) + 1))
            parsed_dates[i] = Date(yr, 1, 1) + Day(doy - 1)
        else
            parsed_dates[i] = nothing
            decimal_times[i] = nothing
        end
    end

    tag_df[!, :timestamp] = parsed_dates
    tag_df[!, :time]      = decimal_times

    # Filter invalid records (missing/non-finite coordinates, identifiers, or timestamps)
    filter!(r -> !ismissing(r.lon) && !ismissing(r.lat) &&
                 !ismissing(r.tagid) && !ismissing(r.tag) &&
                 r.timestamp !== nothing && r.time !== nothing &&
                 isfinite(Float64(r.lon)) && isfinite(Float64(r.lat)), tag_df)

    tag_df[!, :lon]       = Float64.(tag_df.lon)
    tag_df[!, :lat]       = Float64.(tag_df.lat)
    tag_df[!, :tag]       = Int.(tag_df.tag)
    tag_df[!, :tagid]     = string.(tag_df.tagid)
    tag_df[!, :timestamp] = Date.(tag_df.timestamp)
    tag_df[!, :time]      = Float64.(tag_df.time)

    # Require both release and recapture events
    valid_set = Set{String}()
    for sub in groupby(tag_df, :tagid)
        tags = sub.tag
        if any(==(0), tags) && any(>(0), tags)
            push!(valid_set, sub.tagid[1])
        end
    end
    filter!(:tagid => ∈(valid_set), tag_df)
    sort!(tag_df, [:tagid, :tag, :time])

    # 1. Full-domain grid with land barriers
    mesh = if pre_mapped !== nothing
        verbose && println("  [prepare] Using user pre-mapped domain mesh …")
        c_lon = hasproperty(pre_mapped, :center_lon) ?
            pre_mapped.center_lon : ((minimum(tag_df.lon) + maximum(tag_df.lon)) / 2.0)
        c_lat = hasproperty(pre_mapped, :center_lat) ?
            pre_mapped.center_lat : ((minimum(tag_df.lat) + maximum(tag_df.lat)) / 2.0)
        c_km = if hasproperty(pre_mapped, :centroids_km)
            pre_mapped.centroids_km
        elseif hasproperty(pre_mapped, :centroids_lonlat)
            [lonlat_to_xy_km(
                c[1], c[2];
                center_lon = c_lon, center_lat = c_lat,
                crs = crs, datum = datum
            ) for c in pre_mapped.centroids_lonlat]
        elseif hasproperty(pre_mapped, :centroids)
            [lonlat_to_xy_km(
                c[1], c[2];
                center_lon = c_lon, center_lat = c_lat,
                crs = crs, datum = datum
            ) for c in pre_mapped.centroids]
        else
            throw(ArgumentError(
                "pre_mapped must contain :centroids or :centroids_km."
            ))
        end
        n_u = hasproperty(pre_mapped, :n_units) ?
            pre_mapped.n_units : length(c_km)
        l_mask = hasproperty(pre_mapped, :land_mask) ?
            pre_mapped.land_mask : falses(n_u)
        c_ll = if hasproperty(pre_mapped, :centroids_lonlat)
            pre_mapped.centroids_lonlat
        elseif hasproperty(pre_mapped, :centroids)
            pre_mapped.centroids
        else
            [xy_km_to_lonlat(
                c[1], c[2];
                center_lon = c_lon, center_lat = c_lat,
                crs = crs, datum = datum
            ) for c in c_km]
        end
        (
            centroids_km     = c_km,
            centroids_lonlat = c_ll,
            n_units          = n_u,
            W                = pre_mapped.W,
            land_mask        = l_mask,
            center_lon       = c_lon,
            center_lat       = c_lat,
            radius_km        = hasproperty(pre_mapped, :radius_km) ?
                pre_mapped.radius_km : radius_km
        )
    else
        verbose && println("  [prepare] Constructing full movement domain (r=$(radius_km) km) …")
        sp_bounds = extract_sppoly_bounds(sppoly_file)
        construct_full_movement_domain(
            tag_df.lon, tag_df.lat;
            radius_km=radius_km, land_polygons=land_polygons,
            depth=depth, depth_threshold=depth_threshold,
            crs=crs, datum=datum,
            sppoly_bounds=sp_bounds
        )
    end
    verbose && println("    Total mesh units: $(mesh.n_units) " *
                       "(water: $(count(!, mesh.land_mask)), land: $(sum(mesh.land_mask)))")

    # 2. Map telemetry observations to hex units strictly on active water
    verbose && println("  [prepare] Mapping observations to domain units …")
    tag_df = map_point_to_units(
        tag_df, mesh.centroids_km,
        mesh.center_lon, mesh.center_lat;
        crs=crs, datum=datum,
        land_mask=mesh.land_mask, W=mesh.W
    )

    # 3. Load HSI and perform autocorrelation infilling outside core domain
    hsi_vec      = zeros(Float64, mesh.n_units)
    monthly_hsi  = Matrix{Float64}(undef, 0, 0)
    month_lookup = Dict{Tuple{Int, Int}, Int}()
    years_vec    = Int[]

    if hsi_file !== nothing && ispath(hsi_file)
        verbose && println("  [prepare] Loading and infilling HSI field …")
        h = TethysBio.load_hsi_bundle(hsi_file; ref_doy=ref_doy)
        years_vec    = h.years
        month_lookup = h.month_lookup
        S_src        = size(h.monthly_hsi, 1)

        src_coords_km = Tuple{Float64, Float64}[]
        known_mask = falses(mesh.n_units)

        if sppoly_file !== nothing && isfile(sppoly_file)
            au_src = try
                jldopen(sppoly_file, "r") do f
                    haskey(f, "au") ? f["au"] : nothing
                end
            catch err
                verbose && @warn "Failed reading :au from $sppoly_file: $err"
                nothing
            end

            if au_src !== nothing && hasproperty(au_src, :lon) && hasproperty(au_src, :lat)
                n_au = length(au_src.lon)
                src_coords_km = Vector{Tuple{Float64, Float64}}(undef, n_au)
                for i in 1:n_au
                    src_coords_km[i] = lonlat_to_xy_km(
                        au_src.lon[i], au_src.lat[i];
                        center_lon=mesh.center_lon, center_lat=mesh.center_lat,
                        crs=crs, datum=datum
                    )
                end

                # Identify mesh units lying within the known HSI survey domain
                tree_src = KDTree(hcat([[Float64(c[1]), Float64(c[2])] for c in src_coords_km]...))
                survey_radius_thresh = 1.5 * radius_km
                for i in 1:mesh.n_units
                    if !mesh.land_mask[i]
                        _, dists = knn(tree_src, [mesh.centroids_km[i][1], mesh.centroids_km[i][2]], 1)
                        if first(dists) <= survey_radius_thresh
                            known_mask[i] = true
                        end
                    end
                end
            end
        end

        n_mo = size(h.monthly_hsi, 2)
        if !isempty(src_coords_km) && n_mo > 0
            monthly_hsi = zeros(Float64, mesh.n_units, n_mo)
            for m in 1:n_mo
                raw_m = reshard_hsi_field(
                    h.monthly_hsi[:, m], (centroids=mesh.centroids_km,); hsi_coords=src_coords_km
                )
                monthly_hsi[:, m] = infill_spatial_hsi(
                    raw_m, mesh.W, known_mask;
                    land_mask=mesh.land_mask, centroids=mesh.centroids_km
                )
            end
            hsi_vec = vec(mean(monthly_hsi, dims=2))
        else
            base_vec = fill(0.2, mesh.n_units)
            hsi_vec = infill_spatial_hsi(
                base_vec, mesh.W, known_mask;
                land_mask=mesh.land_mask, centroids=mesh.centroids_km
            )
        end

        if !isempty(month_lookup) && size(monthly_hsi, 1) == mesh.n_units
            tag_df[!, :hsi] = match_telemetry_closest_month_hsi(
                tag_df, monthly_hsi, month_lookup, years_vec
            )
            # Climatology fallback: replace NaN observations
            # (missing year/month) with the spatial mean HSI
            n_nan = 0
            hsi_col = tag_df.hsi
            s_col   = tag_df.s_idx
            for i in eachindex(hsi_col)
                if isnan(hsi_col[i])
                    s = s_col[i]
                    hsi_col[i] = (1 <= s <= length(hsi_vec)) ?
                                 hsi_vec[s] : 0.5
                    n_nan += 1
                end
            end
            n_nan > 0 && verbose && println(
                "  [prepare] HSI climatology fallback applied " *
                "to $n_nan / $(nrow(tag_df)) observations."
            )
        else
            tag_df[!, :hsi] = fill(0.5, nrow(tag_df))
        end
    else
        verbose && println("  [prepare] No HSI file provided; using uniform marine HSI = 0.5 …")
        hsi_vec = [mesh.land_mask[i] ? 0.0 : 0.5 for i in 1:mesh.n_units]
        tag_df[!, :hsi] = fill(0.5, nrow(tag_df))
    end

    # 4. Extract mark-recapture event pairs
    dt_map = (monthly=1.0/12.0, weekly=1.0/52.0, biweekly=1.0/26.0, daily=1.0/365.25, raw=1.0)
    dt = hasproperty(dt_map, time_interval) ? getproperty(dt_map, time_interval) : 1.0 / 12.0

    has_sex    = hasproperty(tag_df, :sex)
    has_mat    = hasproperty(tag_df, :mat)
    has_cw     = hasproperty(tag_df, :cw)
    has_chela  = hasproperty(tag_df, :chela)
    has_cc     = hasproperty(tag_df, :cc)
    has_wgt    = hasproperty(tag_df, :wgt)
    has_z      = hasproperty(tag_df, :z)
    has_dsrc   = hasproperty(tag_df, :datasource)

    sorted_df = sort(tag_df, [:tagid, :tag, :time])
    n_rows = nrow(sorted_df)

    tagids    = sorted_df.tagid
    times     = sorted_df.time
    s_idxs    = sorted_df.s_idx
    lons      = sorted_df.lon
    lats      = sorted_df.lat
    sexes_col = has_sex   ? sorted_df.sex   : nothing
    mats_col  = has_mat   ? sorted_df.mat   : nothing
    cw_col    = has_cw    ? sorted_df.cw    : nothing
    chela_col = has_chela ? sorted_df.chela : nothing
    cc_col    = has_cc    ? sorted_df.cc    : nothing
    wgt_col   = has_wgt   ? sorted_df.wgt   : nothing
    z_col     = has_z     ? sorted_df.z     : nothing
    dsrc_col  = has_dsrc  ? sorted_df.datasource : nothing

    _opt_str(col, idx) = (col === nothing || ismissing(col[idx])) ?
        "unknown" :
        (s = strip(string(col[idx])); isempty(s) || lowercase(s) == "missing" ? "unknown" : s)
    _opt_flt(col, i1, i2) = if col === nothing
        missing
    elseif !ismissing(col[i1]) && isfinite(Float64(col[i1]))
        Float64(col[i1])
    elseif !ismissing(col[i2]) && isfinite(Float64(col[i2]))
        Float64(col[i2])
    else
        missing
    end

    RecordType = NamedTuple{
        (:tagid, :release, :recapture, :k, :rel_time, :rec_time,
         :sex, :mat, :cc, :datasource, :lon, :lat, :rec_lon, :rec_lat,
         :cw, :chela, :wgt, :z),
        Tuple{String, Int, Int, Int, Float64, Float64,
              String, String, String, String, Float64, Float64, Float64, Float64,
              Union{Float64, Missing}, Union{Float64, Missing},
              Union{Float64, Missing}, Union{Float64, Missing}}
    }
    pair_records = Vector{RecordType}(undef, 0)
    n_inverted = 0

    if n_rows >= 2
        sizehint!(pair_records, n_rows)
        for i in 2:n_rows
            if tagids[i] == tagids[i-1]
                Δt = times[i] - times[i-1]
                if Δt < 0.0
                    n_inverted += 1
                    continue
                end
                k = max(1, round(Int, Δt / dt))
                sx_val = _opt_str(sexes_col, i - 1)
                sx_val == "unknown" && (sx_val = _opt_str(sexes_col, i))
                mt_val = _opt_str(mats_col, i - 1)
                mt_val == "unknown" && (mt_val = _opt_str(mats_col, i))
                cc_val = _opt_str(cc_col, i - 1)
                cc_val == "unknown" && (cc_val = _opt_str(cc_col, i))
                ds_rec = lowercase(_opt_str(dsrc_col, i))
                ds_rel = lowercase(_opt_str(dsrc_col, i - 1))
                ds_val = ds_rec != "unknown" ? ds_rec : ds_rel
                push!(pair_records, (
                    tagid      = string(tagids[i-1]),
                    release    = s_idxs[i-1],
                    recapture  = s_idxs[i],
                    k          = k,
                    rel_time   = Float64(times[i-1]),
                    rec_time   = Float64(times[i]),
                    sex        = sx_val,
                    mat        = mt_val,
                    cc         = cc_val,
                    datasource = ds_val,
                    lon        = lons[i-1],
                    lat        = lats[i-1],
                    rec_lon    = lons[i],
                    rec_lat    = lats[i],
                    cw         = _opt_flt(cw_col, i - 1, i),
                    chela      = _opt_flt(chela_col, i - 1, i),
                    wgt        = _opt_flt(wgt_col, i - 1, i),
                    z          = _opt_flt(z_col, i - 1, i),
                ))
            end
        end
    end
    if n_inverted > 0 && verbose
        println(
            "  [prepare] Skipped $n_inverted mark-recapture pair(s) " *
            "with inverted dates (recapture before release)."
        )
    end

    obs = DataFrame(pair_records)

    # 5. Assign biological groupings (case-insensitive maturity/sex matching)
    n_obs = nrow(obs)
    labels = Vector{String}(undef, n_obs)
    if n_obs > 0
        obs_sexes = obs[!, :sex]
        obs_mats  = obs[!, :mat]
        @inbounds for i in 1:n_obs
            sx = uppercase(strip(string(obs_sexes[i])))
            mt = lowercase(strip(string(obs_mats[i])))
            if mt == "immature"
                labels[i] = "immature"
            elseif mt == "mature" && sx == "M"
                labels[i] = "male"
            elseif mt == "mature" && sx == "F"
                labels[i] = "female"
            else
                labels[i] = "unknown"
            end
        end
    end

    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
    group_ids = Vector{Int}(undef, n_obs)
    @inbounds for i in 1:n_obs
        group_ids[i] = group_lookup[labels[i]]
    end
    obs[!, :group] = group_ids

    return (
        tagging      = tag_df,
        mesh         = mesh,
        W            = mesh.W,
        hsi_vec      = hsi_vec,
        monthly_hsi  = monthly_hsi,
        month_lookup = month_lookup,
        years        = years_vec,
        obs          = obs,
        group_lookup = group_lookup,
        land_mask    = mesh.land_mask,
        step_years   = dt,
    )
end




# =============================================================================
# Movement Ecology Outputs & Trait Integration
# =============================================================================

"""
    _build_paths_rich(path_results, loaded; step_years = 1.0 / 12.0) -> Vector{NamedTuple}

Assemble coordinate-annotated trajectory records from `path_results.paths` and
`loaded` (`mesh.centroids_lonlat`, `hsi_vec`, `obs_df`) for summary statistics,
CSV exports, and Leaflet path visualizations.
"""
function _build_paths_rich(
    path_results::NamedTuple,
    loaded::NamedTuple;
    step_years::Real = 1.0 / 12.0
)
    paths_dict = hasproperty(path_results, :paths) ? path_results.paths : Dict()
    isempty(paths_dict) && return NamedTuple[]

    mesh = loaded.mesh
    cents = if hasproperty(mesh, :centroids_lonlat) && !isempty(mesh.centroids_lonlat)
        mesh.centroids_lonlat
    elseif hasproperty(mesh, :centroids) && !isempty(mesh.centroids)
        mesh.centroids
    elseif hasproperty(mesh, :centroids_km) && !isempty(mesh.centroids_km)
        mesh.centroids_km
    else
        Tuple{Float64, Float64}[]
    end
    n_units = length(cents)
    n_units == 0 && return NamedTuple[]
    is_geo = _infer_coord_space(cents)
    hsi = hasproperty(loaded, :hsi_vec) ? loaded.hsi_vec : nothing
    obs_df = hasproperty(loaded, :obs_df) ? loaded.obs_df : DataFrame()

    # Pre-index obs_df metadata by tagid
    obs_meta = Dict{String, NamedTuple}()
    if nrow(obs_df) > 0 && hasproperty(obs_df, :tagid)
        for sub in groupby(obs_df, :tagid)
            tid = string(first(sub.tagid))
            dur_days = if hasproperty(sub, :duration_days)
                sum(Float64(coalesce(v, 0.0)) for v in sub.duration_days)
            elseif hasproperty(sub, :rel_time) && hasproperty(sub, :rec_time)
                t0 = minimum(Float64(v) for v in sub.rel_time if !ismissing(v) && isfinite(v); init = NaN)
                t1 = maximum(Float64(v) for v in sub.rec_time if !ismissing(v) && isfinite(v); init = NaN)
                isfinite(t0) && isfinite(t1) && t1 > t0 ? (t1 - t0) * 365.25 :
                    sum(max(1.0, Float64(r)) for r in sub.k)
            elseif hasproperty(sub, :k)
                # In obs_df, k stores integer elapsed days (or step count when small)
                sum(max(1.0, Float64(r)) for r in sub.k)
            else
                Float64(step_years) * 365.25
            end
            r_dt = hasproperty(sub, :release_date) ? string(first(sub.release_date)) :
                   (hasproperty(sub, :rel_time) ? string(round(Float64(first(sub.rel_time)); digits = 3)) : "N/A")
            c_dt = hasproperty(sub, :recapture_date) ? string(last(sub.recapture_date)) :
                   (hasproperty(sub, :rec_time) ? string(round(Float64(last(sub.rec_time)); digits = 3)) :
                    (hasproperty(sub, :rel_time) ? string(round(Float64(first(sub.rel_time)) + dur_days / 365.25; digits = 3)) : "N/A"))
            obs_meta[tid] = (
                duration_days = max(1.0, dur_days),
                release_date = r_dt,
                recapture_date = c_dt
            )
        end
    end

    # Note: haversine_distance returns meters (WGS84 radius = 6_378_137 m), so convert to km.
    step_dist(a, b) = is_geo ?
        haversine_distance(Float64(a[1]), Float64(a[2]), Float64(b[1]), Float64(b[2])) / 1000.0 :
        hypot(Float64(b[1]) - Float64(a[1]), Float64(b[2]) - Float64(a[2]))

    out = NamedTuple[]
    for (tid_raw, seq) in pairs(paths_dict)
        tid = string(tid_raw)
        valid_nodes = Int[Int(u) for u in seq if 1 <= Int(u) <= n_units]
        isempty(valid_nodes) && continue
        coords = Tuple{Float64, Float64}[
            (Float64(cents[u][1]), Float64(cents[u][2])) for u in valid_nodes
        ]
        tot_km = 0.0
        for i in 2:length(coords)
            tot_km += step_dist(coords[i - 1], coords[i])
        end
        disp_km = length(coords) >= 2 ? step_dist(coords[1], coords[end]) : 0.0
        meta = get(obs_meta, tid, (
            duration_days = max(1.0, (length(valid_nodes) - 1) * Float64(step_years) * 365.25),
            release_date = "N/A",
            recapture_date = "N/A"
        ))
        dur_yr = max(meta.duration_days / 365.25, 1e-4)
        spd_yr = disp_km / dur_yr
        m_hsi = hsi !== nothing ?
            mean(Float64(hsi[u]) for u in valid_nodes if 1 <= u <= length(hsi)) : 0.5
        push!(out, (
            tagid = tid,
            path = valid_nodes,
            coords = coords,
            total_dist_km = tot_km,
            displacement_km = disp_km,
            duration_days = meta.duration_days,
            speed_km_per_year = spd_yr,
            mean_hsi = m_hsi,
            release_date = meta.release_date,
            recapture_date = meta.recapture_date,
        ))
    end
    sort!(out; by = p -> p.tagid)
    return out
end

"""
    compute_movement_statistics(paths_rich, path_results, loaded; params = nothing)

Computes movement ecology metrics across reconstructed individual trajectories:
- Net displacement (km): spherical Haversine distance between release and recapture.
- Cumulative marine path length (km) and annual speed (km/yr).
- Directional bias: mean bearing ``\\bar{\\theta}``, mean resultant length ``\\bar{R}``,
  von Mises ``\\hat{\\kappa}``, and Rayleigh uniformity test.
- Patch residence time: dwelling days allocated across spatial units.
"""
function _circular_bearing_stats(bearings_deg::AbstractVector{<:Real})
    C_sum = 0.0
    S_sum = 0.0
    n_valid = 0
    for b in bearings_deg
        isfinite(b) || continue
        r = deg2rad(Float64(b))
        C_sum += cos(r)
        S_sum += sin(r)
        n_valid += 1
    end
    N_circ = max(1, n_valid)
    C_bar  = C_sum / N_circ
    S_bar  = S_sum / N_circ
    R_bar  = min(1.0, hypot(C_bar, S_bar))

    mean_dir_deg = n_valid > 0 ? mod(rad2deg(atan(S_bar, C_bar)), 360.0) : NaN
    circ_var     = 1.0 - R_bar
    circ_sd_deg  = rad2deg(sqrt(max(0.0, -2.0 * log(max(1e-10, R_bar)))))

    kappa_hat = if R_bar < 0.53
        2.0 * R_bar + R_bar^3 + (5.0 / 6.0) * R_bar^5
    elseif R_bar < 0.85
        -0.4 + 1.39 * R_bar + 0.43 / (1.0 - R_bar)
    else
        denom = R_bar^3 - 4.0 * R_bar^2 + 3.0 * R_bar
        denom > 1e-6 ? 1.0 / denom : 100.0
    end

    rayleigh_z = n_valid * R_bar^2
    rayleigh_p = if n_valid >= 3
        p_raw = exp(-rayleigh_z) * (
            1.0 + (2.0 * rayleigh_z - rayleigh_z^2) / (4.0 * n_valid) -
            (24.0 * rayleigh_z - 132.0 * rayleigh_z^2 + 76.0 * rayleigh_z^3 -
             9.0 * rayleigh_z^4) / (288.0 * n_valid^2)
        )
        max(0.0, min(1.0, p_raw))
    else
        1.0
    end

    return (
        n_valid            = n_valid,
        kappa              = kappa_hat,
        mean_vector_length = R_bar,
        mean_bearing_deg   = mean_dir_deg,
        circular_variance  = circ_var,
        circular_sd_deg    = circ_sd_deg,
        rayleigh_z         = rayleigh_z,
        rayleigh_p         = rayleigh_p,
    )
end

function compute_movement_statistics(
    paths_rich::Vector{<:NamedTuple},
    path_results::NamedTuple,
    loaded::NamedTuple;
    params = nothing
)::NamedTuple
    n_spatial = loaded.n_spatial

    net_disp_v = Float64[p.displacement_km for p in paths_rich]
    path_len_v = Float64[p.total_dist_km for p in paths_rich]
    speed_yr_v = Float64[
        hasproperty(p, :speed_km_per_year) ? Float64(p.speed_km_per_year) : 0.0
        for p in paths_rich
    ]
    bearings_v = Float64[]

    for p in paths_rich
        has_moved = p.displacement_km > 0.01 && length(p.coords) >= 2
        if has_moved
            d_lon = (p.coords[end][1] - p.coords[1][1]) *
                    cosd(0.5 * (p.coords[end][2] + p.coords[1][2]))
            d_lat = p.coords[end][2] - p.coords[1][2]
            brg = atand(d_lon, d_lat)
            brg < 0.0 && (brg += 360.0)
            push!(bearings_v, brg)
        else
            push!(bearings_v, NaN)
        end
    end

    directional_bias = _circular_bearing_stats(bearings_v)

    residence_time = zeros(Float64, n_spatial)
    paths_dict = hasproperty(path_results, :paths) ?
        path_results.paths : Dict{String, Vector{Int}}()
    for p in paths_rich
        node_seq = if hasproperty(p, :path)
            p.path
        elseif haskey(paths_dict, string(p.tagid))
            paths_dict[string(p.tagid)]
        elseif haskey(paths_dict, Symbol(p.tagid))
            paths_dict[Symbol(p.tagid)]
        else
            Int[]
        end
        if !isempty(node_seq)
            dur = hasproperty(p, :duration_days) ?
                Float64(p.duration_days) : Float64(length(node_seq) - 1)
            time_per_step = dur / max(1, length(node_seq) - 1)
            for node in node_seq
                if 1 <= node <= n_spatial
                    residence_time[node] += time_per_step
                end
            end
        end
    end

    summary_df = DataFrame(
        tagid             = String[string(p.tagid) for p in paths_rich],
        displacement_km   = net_disp_v,
        total_dist_km     = path_len_v,
        speed_km_per_year = speed_yr_v,
        bearing_deg       = bearings_v,
    )

    return (
        summary_df               = summary_df,
        net_displacement_km      = net_disp_v,
        path_length_km           = path_len_v,
        speed_km_per_year        = speed_yr_v,
        bearings_deg             = bearings_v,
        directional_bias         = directional_bias,
        residence_time_per_patch = residence_time,
    )
end

"""
    analyze_seasonal_movement_phenology(obs_df, mov_stats, loaded)

Aggregates individual movement displacement, speed, and bathymetric shift across
release months and seasonal quarters.
"""
function analyze_seasonal_movement_phenology(
    obs_df::DataFrame,
    mov_stats::NamedTuple,
    loaded::NamedTuple
)::NamedTuple
    n_obs = nrow(obs_df)
    n_obs == 0 && return (monthly_df = DataFrame(), quarterly_df = DataFrame(), peak_month = "N/A")

    months_v = Vector{Int}(undef, n_obs)
    for (idx, row) in enumerate(eachrow(obs_df))
        m = if hasproperty(row, :month) && !ismissing(row.month)
            Int(row.month)
        elseif hasproperty(row, :date) && !ismissing(row.date)
            Dates.month(Date(string(row.date)))
        elseif hasproperty(row, :rel_time) && !ismissing(row.rel_time) && isfinite(Float64(row.rel_time))
            t = Float64(row.rel_time)
            yr = floor(Int, t)
            min(12, max(1, Int(fld((t - yr) * 12.0, 1.0)) + 1))
        else
            6
        end
        months_v[idx] = min(12, max(1, m))
    end

    tag_disp = Dict{String, Float64}()
    if hasproperty(mov_stats, :summary_df) && mov_stats.summary_df isa DataFrame
        for r in eachrow(mov_stats.summary_df)
            tag_disp[string(r.tagid)] = Float64(r.displacement_km)
        end
    end

    cents_ll = (hasproperty(loaded, :mesh) && loaded.mesh !== nothing &&
                hasproperty(loaded.mesh, :centroids_lonlat)) ?
               loaded.mesh.centroids_lonlat : nothing
    depths = (hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing) ?
        Float64.(loaded.resharded_depths) : Float64[]

    n_stat = length(mov_stats.net_displacement_km)
    disp_all = Vector{Float64}(undef, n_obs)
    dz_all   = fill(NaN, n_obs)
    for (i, row) in enumerate(eachrow(obs_df))
        if n_stat == n_obs
            disp_all[i] = Float64(mov_stats.net_displacement_km[i])
        else
            tid = hasproperty(row, :tagid) ? string(row.tagid) : ""
            if cents_ll !== nothing && hasproperty(row, :release) && hasproperty(row, :recapture) &&
               1 <= row.release <= length(cents_ll) && 1 <= row.recapture <= length(cents_ll)
                c1 = cents_ll[row.release]
                c2 = cents_ll[row.recapture]
                disp_all[i] = haversine_distance(
                    Float64(c1[1]), Float64(c1[2]), Float64(c2[1]), Float64(c2[2])
                ) / 1000.0
            elseif haskey(tag_disp, tid)
                disp_all[i] = tag_disp[tid]
            else
                disp_all[i] = 0.0
            end
        end
        if !isempty(depths) && hasproperty(row, :release) && hasproperty(row, :recapture) &&
           1 <= row.release <= length(depths) && 1 <= row.recapture <= length(depths)
            dz_all[i] = depths[row.recapture] - depths[row.release]
        end
    end

    m_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_rows = NamedTuple[]
    for m in 1:12
        idxs = findall(==(m), months_v)
        c = length(idxs)
        m_disp = c > 0 ? mean(disp_all[idxs]) : 0.0
        med_disp = c > 0 ? median(disp_all[idxs]) : 0.0
        dz_sub = Float64[dz_all[k] for k in idxs if isfinite(dz_all[k])]
        m_dz = !isempty(dz_sub) ? mean(dz_sub) : 0.0
        push!(monthly_rows, (
            month          = m,
            month_name     = m_names[m],
            n_events       = c,
            mean_disp_km   = m_disp,
            median_disp_km = med_disp,
            mean_dz_m      = m_dz,
        ))
    end
    monthly_df = DataFrame(monthly_rows)

    q_names = ["Q1 (Winter)", "Q2 (Spring)", "Q3 (Summer)", "Q4 (Autumn)"]
    quarterly_rows = NamedTuple[]
    for q in 1:4
        q_m = ((q - 1) * 3 + 1):(q * 3)
        idxs = findall(m -> m in q_m, months_v)
        c = length(idxs)
        q_disp = c > 0 ? mean(disp_all[idxs]) : 0.0
        med_disp = c > 0 ? median(disp_all[idxs]) : 0.0
        dz_sub = Float64[dz_all[k] for k in idxs if isfinite(dz_all[k])]
        q_dz = !isempty(dz_sub) ? mean(dz_sub) : 0.0
        push!(quarterly_rows, (
            quarter        = q,
            quarter_name   = q_names[q],
            n_events       = c,
            mean_disp_km   = q_disp,
            median_disp_km = med_disp,
            mean_dz_m      = q_dz,
        ))
    end
    quarterly_df = DataFrame(quarterly_rows)

    peak_row = argmax(monthly_df.mean_disp_km)
    peak_month = monthly_df.month_name[peak_row]

    return (
        monthly_df   = monthly_df,
        quarterly_df = quarterly_df,
        peak_month   = peak_month,
    )
end

"""
    model_trait_movement_associations(obs_df, mov_stats, loaded)

Fits Ordinary Least Squares (OLS) regression models assessing relationships
between biological traits (carapace width, chela height, weight) and physical
movement metrics (net displacement, annual speed, cumulative path length).
"""
function model_trait_movement_associations(
    obs_df::DataFrame,
    mov_stats::NamedTuple,
    loaded::NamedTuple
)::NamedTuple
    n_paths = length(mov_stats.net_displacement_km)
    n_paths < 3 && return (trait_name = "none", trait_vals = Float64[], models = Dict{String, NamedTuple}())

    candidates = [:carapace_width, :cw, :chela, :size, :wgt, :weight, :length, :size_mm]
    found_col = nothing
    for c in candidates
        if hasproperty(obs_df, c)
            col_vals = obs_df[!, c]
            if count(v -> !ismissing(v) && v isa Real && isfinite(Float64(v)), col_vals) >= 3
                found_col = c
                break
            end
        end
    end

    if found_col === nothing
        return (trait_name = "none", trait_vals = Float64[], models = Dict{String, NamedTuple}())
    end
    trait_name = string(found_col)

    raw_traits = fill(NaN, n_paths)
    if hasproperty(mov_stats, :summary_df) && mov_stats.summary_df isa DataFrame &&
       hasproperty(obs_df, :tagid)
        trait_by_tag = Dict{String, Float64}()
        for row in eachrow(obs_df)
            v = row[found_col]
            if !ismissing(v) && v isa Real && isfinite(Float64(v))
                tid = string(row.tagid)
                haskey(trait_by_tag, tid) || (trait_by_tag[tid] = Float64(v))
            end
        end
        for (i, r) in enumerate(eachrow(mov_stats.summary_df))
            i > n_paths && break
            raw_traits[i] = get(trait_by_tag, string(r.tagid), NaN)
        end
    elseif nrow(obs_df) == n_paths
        for i in 1:n_paths
            v = obs_df[i, found_col]
            if !ismissing(v) && v isa Real && isfinite(Float64(v))
                raw_traits[i] = Float64(v)
            end
        end
    end

    valid_idx = findall(isfinite, raw_traits)
    if length(valid_idx) < 3
        return (trait_name = trait_name, trait_vals = Float64[], models = Dict{String, NamedTuple}())
    end

    trait_vec = raw_traits[valid_idx]
    results = Dict{String, NamedTuple}()
    targets = Tuple{String, Vector{Float64}}[
        ("displacement", Float64.(mov_stats.net_displacement_km)),
    ]
    if hasproperty(mov_stats, :speed_km_per_year) && length(mov_stats.speed_km_per_year) == n_paths
        push!(targets, ("speed_km_yr", Float64.(mov_stats.speed_km_per_year)))
    end
    if hasproperty(mov_stats, :path_length_km) && length(mov_stats.path_length_km) == n_paths
        push!(targets, ("path_length", Float64.(mov_stats.path_length_km)))
    end

    for (t_name, y_full) in targets
        pair_mask = [isfinite(raw_traits[k]) && isfinite(y_full[k]) for k in 1:n_paths]
        idxs = findall(pair_mask)
        length(idxs) < 3 && continue
        x_v = raw_traits[idxs]
        y_v = y_full[idxs]
        n_fit = length(idxs)

        x_bar = mean(x_v)
        y_bar = mean(y_v)
        ss_xx = sum((x - x_bar)^2 for x in x_v)
        ss_yy = sum((y - y_bar)^2 for y in y_v)
        ss_xy = sum((x_v[i] - x_bar) * (y_v[i] - y_bar) for i in 1:n_fit)

        beta_1 = ss_xx > 1e-10 ? ss_xy / ss_xx : 0.0
        beta_0 = y_bar - beta_1 * x_bar
        residuals = [y_v[i] - (beta_0 + beta_1 * x_v[i]) for i in 1:n_fit]
        sse = sum(r^2 for r in residuals)
        r2 = ss_yy > 1e-10 ? max(0.0, min(1.0, 1.0 - sse / ss_yy)) : 0.0

        df_deg = max(1, n_fit - 2)
        s_err = sqrt(sse / df_deg)
        se_beta1 = ss_xx > 1e-10 ? s_err / sqrt(ss_xx) : 1.0
        t_stat = se_beta1 > 1e-10 ? beta_1 / se_beta1 : 0.0
        p_val = max(0.0, min(1.0, 2.0 * ccdf(TDist(df_deg), abs(t_stat))))

        results[t_name] = (
            trait_name = trait_name,
            n_obs      = n_fit,
            beta_0     = beta_0,
            beta_1     = beta_1,
            se_beta_1  = se_beta1,
            t_stat     = t_stat,
            p_val      = p_val,
            r2         = r2,
        )
    end

    return (
        trait_name = trait_name,
        trait_vals = trait_vec,
        models     = results,
    )
end

"""
    compute_directed_flux_field(F_domain, centroids_lonlat; top_k = 180) -> NamedTuple

Extracts the antisymmetric net directed migration flux ``J(u, v) = F(u, v) - F(v, u)``
and per-node resultant transport vectors from a directed edge-flux matrix `F_domain`.
"""
function compute_directed_flux_field(
    F_domain::AbstractMatrix{<:Real},
    centroids_lonlat::AbstractVector;
    top_k::Int = 180
)::NamedTuple
    S = size(F_domain, 1)
    F_sp = F_domain isa SparseMatrixCSC ? F_domain : sparse(Float64.(F_domain))
    J_mat = F_sp - sparse(F_sp')
    dropzeros!(J_mat)

    rows, cols, vals = findnz(J_mat)
    edges = NamedTuple{(:u, :v, :net_flux, :u_lon, :u_lat, :v_lon, :v_lat),
                       Tuple{Int, Int, Float64, Float64, Float64, Float64, Float64}}[]
    vx = zeros(Float64, S)
    vy = zeros(Float64, S)

    for (u, v, jval) in zip(rows, cols, vals)
        jval <= 1e-9 && continue
        (1 <= u <= length(centroids_lonlat) && 1 <= v <= length(centroids_lonlat)) || continue
        ulon, ulat = Float64(centroids_lonlat[u][1]), Float64(centroids_lonlat[u][2])
        vlon, vlat = Float64(centroids_lonlat[v][1]), Float64(centroids_lonlat[v][2])
        push!(edges, (u = u, v = v, net_flux = Float64(jval),
                      u_lon = ulon, u_lat = ulat, v_lon = vlon, v_lat = vlat))
        vx[u] += jval * (vlon - ulon)
        vy[u] += jval * (vlat - ulat)
    end

    sort!(edges; by = e -> e.net_flux, rev = true)
    top_slice = length(edges) > top_k ? edges[1:top_k] : edges
    top_df = isempty(top_slice) ?
        DataFrame(u = Int[], v = Int[], net_flux = Float64[],
                  u_lon = Float64[], u_lat = Float64[], v_lon = Float64[], v_lat = Float64[]) :
        DataFrame(top_slice)
    node_mag = [hypot(vx[i], vy[i]) for i in 1:S]

    return (
        net_flux_matrix = J_mat,
        top_edges       = top_df,
        vx              = vx,
        vy              = vy,
        node_magnitude  = node_mag,
        n_directed      = length(edges),
    )
end

"""
    analyze_bathymetric_migration(obs_df, loaded) -> NamedTuple

Quantifies bathymetric displacement ``\\Delta z = z_{\\text{rec}} - z_{\\text{rel}}`` (m)
across all mark-recapture observations, stratified by carapace-width class, shell
condition (`cc`), and release season, with a two-sided t-test against ``\\Delta z = 0``.
"""
function analyze_bathymetric_migration(
    obs_df::DataFrame,
    loaded::NamedTuple
)::NamedTuple
    n_obs = nrow(obs_df)
    depths = hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing ?
        Float64.(loaded.resharded_depths) : Float64[]
    (n_obs == 0 || isempty(depths)) && return (
        n_obs = 0, mean_dz_m = NaN, median_dz_m = NaN, pct_deeper = 0.0,
        pct_shallower = 0.0, t_stat = 0.0, p_val = 1.0,
        by_size = DataFrame(), by_cc = DataFrame(), by_season = DataFrame()
    )

    dz_v = Float64[]
    cw_v = Float64[]
    cc_v = String[]
    q_v  = String[]

    q_names = ["Q1 (Winter)", "Q2 (Spring)", "Q3 (Summer)", "Q4 (Autumn)"]
    has_cw = hasproperty(obs_df, :cw)
    has_cc = hasproperty(obs_df, :cc)
    has_rt = hasproperty(obs_df, :rel_time)

    for r in eachrow(obs_df)
        u1, u2 = Int(r.release), Int(r.recapture)
        (1 <= u1 <= length(depths) && 1 <= u2 <= length(depths)) || continue
        d1, d2 = depths[u1], depths[u2]
        (isfinite(d1) && isfinite(d2)) || continue
        push!(dz_v, d2 - d1)
        push!(cw_v, has_cw && !ismissing(r.cw) && isfinite(Float64(r.cw)) ? Float64(r.cw) : NaN)
        cc_raw = has_cc && !ismissing(r.cc) ? strip(string(r.cc)) : "unknown"
        push!(cc_v, isempty(cc_raw) || lowercase(cc_raw) in ("missing", "none") ? "unknown" : cc_raw)
        if has_rt && !ismissing(r.rel_time) && isfinite(Float64(r.rel_time))
            t = Float64(r.rel_time)
            mo = clamp(Int(fld((t - floor(Int, t)) * 12.0, 1.0)) + 1, 1, 12)
            push!(q_v, q_names[div(mo - 1, 3) + 1])
        else
            push!(q_v, "Unknown")
        end
    end

    n_valid = length(dz_v)
    n_valid == 0 && return (
        n_obs = 0, mean_dz_m = NaN, median_dz_m = NaN, pct_deeper = 0.0,
        pct_shallower = 0.0, t_stat = 0.0, p_val = 1.0,
        by_size = DataFrame(), by_cc = DataFrame(), by_season = DataFrame()
    )

    mu_dz = mean(dz_v)
    sd_dz = n_valid > 1 ? std(dz_v) : 0.0
    t_stat = (n_valid > 1 && sd_dz > 1e-10) ? mu_dz / (sd_dz / sqrt(n_valid)) : 0.0
    p_val = n_valid > 1 ? clamp(2.0 * ccdf(TDist(n_valid - 1), abs(t_stat)), 0.0, 1.0) : 1.0

    size_bins = [("< 95 mm", 0.0, 95.0), ("95-110 mm", 95.0, 110.0),
                 ("110-125 mm", 110.0, 125.0), (">= 125 mm", 125.0, Inf)]
    size_rows = NamedTuple[]
    for (lbl, lo, hi) in size_bins
        idxs = findall(i -> isfinite(cw_v[i]) && cw_v[i] >= lo && cw_v[i] < hi, 1:n_valid)
        isempty(idxs) && continue
        sub = dz_v[idxs]
        push!(size_rows, (stratum = lbl, n = length(sub),
                          mean_dz_m = mean(sub), median_dz_m = median(sub),
                          pct_deeper = 100.0 * count(>(5.0), sub) / length(sub)))
    end

    cc_groups = [("CC 2 (New/Soft)", Set(["2"])),
                 ("CC 3 (Clean Hard)", Set(["3"])),
                 ("CC 4-5 (Old/Mossy)", Set(["4", "5"]))]
    cc_rows = NamedTuple[]
    for (lbl, cset) in cc_groups
        idxs = findall(i -> cc_v[i] in cset, 1:n_valid)
        isempty(idxs) && continue
        sub = dz_v[idxs]
        push!(cc_rows, (stratum = lbl, n = length(sub),
                        mean_dz_m = mean(sub), median_dz_m = median(sub),
                        pct_deeper = 100.0 * count(>(5.0), sub) / length(sub)))
    end

    season_rows = NamedTuple[]
    for ql in q_names
        idxs = findall(==(ql), q_v)
        isempty(idxs) && continue
        sub = dz_v[idxs]
        push!(season_rows, (stratum = ql, n = length(sub),
                            mean_dz_m = mean(sub), median_dz_m = median(sub),
                            pct_deeper = 100.0 * count(>(5.0), sub) / length(sub)))
    end

    return (
        n_obs         = n_valid,
        mean_dz_m     = mu_dz,
        median_dz_m   = median(dz_v),
        pct_deeper    = 100.0 * count(>(5.0), dz_v) / n_valid,
        pct_shallower = 100.0 * count(<(-5.0), dz_v) / n_valid,
        t_stat        = t_stat,
        p_val         = p_val,
        by_size       = DataFrame(size_rows),
        by_cc         = DataFrame(cc_rows),
        by_season     = DataFrame(season_rows),
    )
end

"""
    evaluate_dispersal_scaling_and_null(loaded, kernels; step_years = 1/365.25) -> NamedTuple

Evaluates multi-horizon mean-squared displacement ``\\text{MSD}(\\Delta t)`` across
liberty-duration bins (empirical vs. fitted kernel prediction), estimates the empirical
MSD power-law scaling exponent ``\\beta`` (``\\text{MSD} \\propto \\Delta t^\\beta``),
and compares log-predictive density (``\\Delta\\text{LPPD}``) of the fitted habitat-advection
kernel against a null isotropic diffusion kernel (``\\alpha = 0, \\gamma = 0``).
"""
function evaluate_dispersal_scaling_and_null(
    loaded::NamedTuple,
    kernels::NamedTuple;
    step_years::Real = 1.0 / 365.25
)::NamedTuple
    obs_df = loaded.obs_df
    n_obs = nrow(obs_df)
    cents_ll = hasproperty(loaded.mesh, :centroids_lonlat) ?
        loaded.mesh.centroids_lonlat : nothing
    (n_obs == 0 || cents_ll === nothing) && return (
        bins_df = DataFrame(), msd_beta = NaN,
        lppd_fitted = NaN, lppd_null = NaN, delta_lppd = NaN,
        mean_ll_gain = NaN, n_obs = 0
    )

    releases   = Int.(obs_df.release)
    recaptures = Int.(obs_df.recapture)
    ks         = max.(1, Int.(round.(obs_df.k)))
    S          = loaded.n_spatial

    P_fit = kernels.P_kernel isa SparseMatrixCSC ?
        kernels.P_kernel : sparse(Float64.(kernels.P_kernel))
    P_null = build_sparse_transition_kernel(
        loaded.W, loaded.hsi_vec, 0.0, Float64(kernels.rho_hat), 0.0, loaded.land_mask
    )

    cache_fit  = kstep_transition_cache(sparse(P_fit'), releases, ks)
    cache_null = kstep_transition_cache(sparse(P_null'), releases, ks)

    lppd_fit  = 0.0
    lppd_null = 0.0
    obs_disp  = Vector{Float64}(undef, n_obs)
    pred_disp = Vector{Float64}(undef, n_obs)
    pred_msd  = Vector{Float64}(undef, n_obs)
    days_v    = Float64.(ks) .* (Float64(step_years) * 365.25)

    # Precompute distance vector from each unique release node to all S units
    rel_dist_cache = Dict{Int, Vector{Float64}}()
    for u in unique(releases)
        c1 = cents_ll[u]
        lon1, lat1 = Float64(c1[1]), Float64(c1[2])
        dvec = Vector{Float64}(undef, S)
        @inbounds for j in 1:S
            c2 = cents_ll[j]
            dvec[j] = haversine_distance(lon1, lat1, Float64(c2[1]), Float64(c2[2])) / 1000.0
        end
        rel_dist_cache[u] = dvec
    end

    for n in 1:n_obs
        u, v, k = releases[n], recaptures[n], ks[n]
        p_f = cache_fit[(u, k)]
        p_0 = cache_null[(u, k)]
        lppd_fit  += log(max(p_f[v], 1e-15))
        lppd_null += log(max(p_0[v], 1e-15))

        dvec = rel_dist_cache[u]
        obs_disp[n] = dvec[v]
        ed = 0.0
        ed2 = 0.0
        @inbounds for j in 1:S
            pj = p_f[j]
            pj > 0.0 || continue
            dj = dvec[j]
            ed  += pj * dj
            ed2 += pj * dj * dj
        end
        pred_disp[n] = ed
        pred_msd[n]  = ed2
    end

    bin_specs = [("1-90 d", 0.0, 90.0), ("91-180 d", 90.0, 180.0),
                 ("181-365 d", 180.0, 365.0), ("366-730 d", 365.0, 730.0),
                 ("> 730 d", 730.0, Inf)]
    rows = NamedTuple[]
    log_t = Float64[]
    log_m = Float64[]
    for (lbl, lo, hi) in bin_specs
        idxs = findall(i -> days_v[i] > lo && days_v[i] <= hi, 1:n_obs)
        isempty(idxs) && continue
        emp_d   = mean(obs_disp[idxs])
        emp_msd = mean(obs_disp[i]^2 for i in idxs)
        mod_d   = mean(pred_disp[idxs])
        mod_msd = mean(pred_msd[idxs])
        mt      = mean(days_v[idxs])
        if emp_msd > 1e-6 && mt > 0.0
            push!(log_t, log(mt))
            push!(log_m, log(emp_msd))
        end
        push!(rows, (
            horizon      = lbl,
            n_obs        = length(idxs),
            mean_days    = mt,
            emp_disp_km  = emp_d,
            pred_disp_km = mod_d,
            emp_msd_km2  = emp_msd,
            pred_msd_km2 = mod_msd,
        ))
    end

    msd_beta = if length(log_t) >= 2
        xt, xm = mean(log_t), mean(log_m)
        sxx = sum((x - xt)^2 for x in log_t)
        sxx > 1e-10 ? sum((log_t[i] - xt) * (log_m[i] - xm) for i in eachindex(log_t)) / sxx : NaN
    else
        NaN
    end

    return (
        bins_df      = DataFrame(rows),
        msd_beta     = msd_beta,
        lppd_fitted  = lppd_fit,
        lppd_null    = lppd_null,
        delta_lppd   = lppd_fit - lppd_null,
        mean_ll_gain = (lppd_fit - lppd_null) / n_obs,
        n_obs        = n_obs,
    )
end

"""
    analyze_biological_state_stratification(obs_df, loaded; step_years = 1/365.25) -> NamedTuple

Stratifies mark-recapture movement metrics (stationary fraction, displacement, annual
mobility rate, and bathymetric shift) across carapace shell condition (`cc`) and
carapace-width (`cw`) classes.
"""
function analyze_biological_state_stratification(
    obs_df::DataFrame,
    loaded::NamedTuple;
    step_years::Real = 1.0 / 365.25
)::NamedTuple
    n_obs = nrow(obs_df)
    cents_ll = hasproperty(loaded.mesh, :centroids_lonlat) ?
        loaded.mesh.centroids_lonlat : nothing
    depths = hasproperty(loaded, :resharded_depths) && loaded.resharded_depths !== nothing ?
        Float64.(loaded.resharded_depths) : Float64[]
    (n_obs == 0 || cents_ll === nothing) && return (by_cc = DataFrame(), by_cw = DataFrame())

    has_cw = hasproperty(obs_df, :cw)
    has_cc = hasproperty(obs_df, :cc)

    disp_km = Vector{Float64}(undef, n_obs)
    spd_yr  = Vector{Float64}(undef, n_obs)
    stat_v  = falses(n_obs)
    dz_v    = fill(NaN, n_obs)
    cw_v    = fill(NaN, n_obs)
    cc_v    = fill("unknown", n_obs)

    for (i, r) in enumerate(eachrow(obs_df))
        u1, u2 = Int(r.release), Int(r.recapture)
        stat_v[i] = (u1 == u2)
        c1, c2 = cents_ll[u1], cents_ll[u2]
        d = u1 == u2 ? 0.0 :
            haversine_distance(Float64(c1[1]), Float64(c1[2]), Float64(c2[1]), Float64(c2[2])) / 1000.0
        disp_km[i] = d
        dur_yr = max(Float64(r.k) * Float64(step_years), 1.0 / 365.25)
        spd_yr[i] = d / dur_yr
        if !isempty(depths) && 1 <= u1 <= length(depths) && 1 <= u2 <= length(depths)
            dz_v[i] = depths[u2] - depths[u1]
        end
        if has_cw && !ismissing(r.cw) && isfinite(Float64(r.cw))
            cw_v[i] = Float64(r.cw)
        end
        if has_cc && !ismissing(r.cc)
            s = strip(string(r.cc))
            (!isempty(s) && !(lowercase(s) in ("missing", "none"))) && (cc_v[i] = s)
        end
    end

    function _summarize_stratum(lbl::String, idxs::Vector{Int})
        d_sub = disp_km[idxs]
        s_sub = spd_yr[idxs]
        dz_sub = Float64[dz_v[k] for k in idxs if isfinite(dz_v[k])]
        return (
            stratum        = lbl,
            n_obs          = length(idxs),
            pct_stationary = 100.0 * count(k -> stat_v[k], idxs) / length(idxs),
            median_disp_km = median(d_sub),
            mean_disp_km   = mean(d_sub),
            mean_speed_yr  = mean(s_sub),
            mean_dz_m      = isempty(dz_sub) ? NaN : mean(dz_sub),
        )
    end

    cc_rows = NamedTuple[]
    for (lbl, cset) in [("CC 2 (New/Soft)", Set(["2"])),
                        ("CC 3 (Clean Hard)", Set(["3"])),
                        ("CC 4-5 (Old/Mossy)", Set(["4", "5"]))]
        idxs = findall(i -> cc_v[i] in cset, 1:n_obs)
        !isempty(idxs) && push!(cc_rows, _summarize_stratum(lbl, idxs))
    end

    cw_rows = NamedTuple[]
    for (lbl, lo, hi) in [("< 95 mm", 0.0, 95.0), ("95-110 mm", 95.0, 110.0),
                          ("110-125 mm", 110.0, 125.0), (">= 125 mm", 125.0, Inf)]
        idxs = findall(i -> isfinite(cw_v[i]) && cw_v[i] >= lo && cw_v[i] < hi, 1:n_obs)
        !isempty(idxs) && push!(cw_rows, _summarize_stratum(lbl, idxs))
    end

    return (by_cc = DataFrame(cc_rows), by_cw = DataFrame(cw_rows))
end

"""
    export_movement_summary_csv(filepath, paths_rich, mov_stats, obs_df)

Exports per-individual movement trajectory metrics to a standardized CSV file.
"""
function export_movement_summary_csv(
    filepath::String,
    paths_rich::Vector{<:NamedTuple},
    mov_stats::NamedTuple,
    obs_df::DataFrame
)::String
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        write(io, "tagid,release_date,recapture_date,start_lon,start_lat," *
                  "end_lon,end_lat,duration_days,total_dist_km,displacement_km," *
                  "speed_km_per_year,velocity_km_day,mean_hsi,bearing_deg\n")
        for (i, p) in enumerate(paths_rich)
            coords = hasproperty(p, :coords) ? p.coords : Tuple{Float64, Float64}[]
            c_start = length(coords) >= 1 ? coords[1] : (0.0, 0.0)
            c_end   = length(coords) >= 1 ? coords[end] : (0.0, 0.0)
            dur = hasproperty(p, :duration_days) ? Float64(p.duration_days) : 1.0
            tot_d = hasproperty(p, :total_dist_km) ? Float64(p.total_dist_km) : 0.0
            disp_d = hasproperty(p, :displacement_km) ? Float64(p.displacement_km) : 0.0
            spd_yr = hasproperty(p, :speed_km_per_year) ? Float64(p.speed_km_per_year) :
                     (dur > 0.1 ? disp_d / (dur / 365.25) : 0.0)
            vel = dur > 0.1 ? disp_d / dur : 0.0
            r_date = hasproperty(p, :release_date) ? string(p.release_date) : "N/A"
            c_date = hasproperty(p, :recapture_date) ? string(p.recapture_date) : "N/A"
            m_hsi = hasproperty(p, :mean_hsi) ? Float64(p.mean_hsi) : 0.5
            bearing_val = i <= length(mov_stats.bearings_deg) && isfinite(mov_stats.bearings_deg[i]) ?
                mov_stats.bearings_deg[i] : 0.0

            write(io, string(
                p.tagid, ",",
                r_date, ",",
                c_date, ",",
                round(c_start[1]; digits=4), ",",
                round(c_start[2]; digits=4), ",",
                round(c_end[1]; digits=4), ",",
                round(c_end[2]; digits=4), ",",
                round(dur; digits=1), ",",
                round(tot_d; digits=2), ",",
                round(disp_d; digits=2), ",",
                round(spd_yr; digits=2), ",",
                round(vel; digits=3), ",",
                round(m_hsi; digits=3), ",",
                round(bearing_val; digits=1), "\n"
            ))
        end
    end
    return filepath
end

# =============================================================================
# Posterior Uncertainty Propagation & Validation Analyses
# =============================================================================

"""
    _sample_column(samples) -> Vector{Float64}

Return a parameter's posterior draws as a plain vector.

`extract_transition_kernels` returns one draw series per parameter, not one column
per demographic group, so these arrive as a vector of length `n_draws`. A matrix
is still accepted, in which case its single pooled column is used: the models fit
one unstratified population, so only column 1 carries estimated values.
"""
function _sample_column(samples)::Vector{Float64}
    samples === nothing && return Float64[]
    if samples isa AbstractVector
        return Float64[Float64(v) for v in samples]
    end
    size(samples, 2) == 0 && return Float64[]
    return Float64[Float64(v) for v in view(samples, :, 1)]
end

"""
    _posterior_param_draws(kernels, name::Symbol, default::Float64) -> Matrix{Float64}

Return the posterior draws of a kernel parameter as an `n_draws x 1` matrix.

The trailing axis is retained so that callers can keep indexing draws uniformly,
but it has length one: the model is pooled and fits a single parameter set, so
there is no group axis to read.
"""
function _posterior_param_draws(kernels::NamedTuple, name::Symbol, default::Float64)
    n_draws = if hasproperty(kernels, :alpha_samples) && !isempty(kernels.alpha_samples)
        size(kernels.alpha_samples, 1)
    elseif hasproperty(kernels, name) && !isempty(getproperty(kernels, name))
        size(getproperty(kernels, name), 1)
    else
        1
    end
    out = zeros(Float64, n_draws, 1)
    raw = hasproperty(kernels, name) ? getproperty(kernels, name) : nothing
    col = _sample_column(raw)
    if isempty(col)
        out[:, 1] .= default
    else
        n = min(n_draws, length(col))
        out[1:n, 1] .= col[1:n]
    end
    return out
end

"""
    compute_corridor_canalization(loaded, fitted, kernels, params = (;)) -> NamedTuple

Quantifies spatial corridor entropy, effective lateral bridge spread (km), and
migratory canalization across the marine mesh by propagating conditioned Markov
bridges over posterior kernel draws.

For each marine unit ``u``:
- `step_entropy[u]`: normalized outgoing transition Shannon entropy in ``[0, 1]``
  (``1`` = isotropic 2D diffusion as in homogeneous-HSI zones; ``0`` = single-edge funnel).
- `transit_density[u]`: posterior mean interior Markov-bridge occupancy across
  unique non-stationary origin-destination pairs.
- `effective_spread_km[u]`: transit-weighted characteristic 2D corridor width
  ``\\sqrt{\\exp(H(q_\\tau)) \\, A_{\\text{cell}}}`` (km) of bridges passing through ``u``.
- `canalization_index[u]`: normalized composite index in ``[0, 1]`` isolating narrow,
  low-entropy, high-flux conduits from broad diffusive sheets.
"""
function _unit_step_entropy(
    P_sp::SparseMatrixCSC{<:Real},
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Float64}
    S = size(P_sp, 1)
    Pt = SparseMatrixCSC(P_sp')
    rows_t = rowvals(Pt)
    vals_t = nonzeros(Pt)
    ent = zeros(Float64, S)

    for u in 1:S
        (land_mask !== nothing && land_mask[u]) && continue
        off_sum = 0.0
        deg_off = 0
        for ptr in nzrange(Pt, u)
            v = rows_t[ptr]
            v == u && continue
            (land_mask !== nothing && land_mask[v]) && continue
            p = Float64(vals_t[ptr])
            if p > 1e-15
                off_sum += p
                deg_off += 1
            end
        end
        if deg_off <= 1 || off_sum <= 1e-15
            ent[u] = 0.0
            continue
        end
        h_val = 0.0
        for ptr in nzrange(Pt, u)
            v = rows_t[ptr]
            v == u && continue
            (land_mask !== nothing && land_mask[v]) && continue
            p = Float64(vals_t[ptr])
            if p > 1e-15
                q = p / off_sum
                h_val -= q * log(q)
            end
        end
        ent[u] = max(0.0, min(1.0, h_val / log(Float64(deg_off))))
    end
    return ent
end

"""
    compute_corridor_canalization(loaded, fitted, kernels, params = (;)) -> NamedTuple

Evaluate normalized outgoing marine step entropy `H(u)` and interior Markov-bridge
corridor canalization `K(u)` across the spatial domain:

    H(u) = -(1 / log d_u) * sum_{v != u} P_tilde[u, v] * log P_tilde[u, v]  in [0, 1]
    K(u) = sqrt(min(1, C(u) / P_99(C))) * (1 - 0.5 * H(u))                  in [0, 1]

Inputs:
- `loaded`: Loaded domain NamedTuple containing `obs_df`, `W`, `hsi_vec`, `land_mask`, and `mesh`.
- `fitted`: Fitted MCMC container `(models, chains)`.
- `kernels`: Extracted transition kernel bundle containing `P_kernel` and posterior draws.
- `params`: Configuration struct or NamedTuple (`max_paths`, `verbose`).

Outputs:
NamedTuple with `transit_density`, `transit_sd`, `step_entropy`, `effective_spread_km`,
`canalization_index`, and `conduit_mask` (units in the upper decile of positive `K(u)`).
"""
function compute_corridor_canalization(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params = (;)
)::NamedTuple
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    land_mask = loaded.land_mask
    verbose   = hasproperty(params, :verbose) ? Bool(params.verbose) : false
    max_p     = hasproperty(params, :max_paths) ? Int(params.max_paths) : 300

    a_samples = _posterior_param_draws(kernels, :alpha_samples, 0.3)
    r_samples = _posterior_param_draws(kernels, :rho_samples, 0.5)
    g_samples = _posterior_param_draws(kernels, :gamma_samples, 1.0)
    n_draws   = size(a_samples, 1)

    n_eval = min(n_draws, 8)
    eval_indices = unique(round.(Int, range(1, n_draws; length = n_eval)))

    P_draws = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(eval_indices))
    for (di, draw) in enumerate(eval_indices)
        alpha_val = max(0.0, min(1.0, Float64(a_samples[draw, 1])))
        rho_val   = max(0.0, min(0.9999, Float64(r_samples[draw, 1])))
        gamma_val = Float64(g_samples[draw, 1])
        P_draws[di] = sparse(construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = gamma_val,
            residence = rho_val,
            advection = alpha_val,
            land_mask = land_mask
        ))
    end

    step_entropy = zeros(Float64, n_spatial)
    for P_d in P_draws
        step_entropy .+= _unit_step_entropy(P_d, land_mask)
    end
    step_entropy ./= length(P_draws)

    cell_area_km2 = if hasproperty(loaded.mesh, :areas_km2) &&
                       length(loaded.mesh.areas_km2) == n_spatial
        mean(loaded.mesh.areas_km2)
    elseif hasproperty(loaded.mesh, :radius_km)
        1.5 * sqrt(3.0) * Float64(loaded.mesh.radius_km)^2
    else
        65.0
    end

    pair_k = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for row in eachrow(obs_df)
        u_rel = Int(row.release)
        u_rec = Int(row.recapture)
        u_rel == u_rec && continue
        (1 <= u_rel <= n_spatial && 1 <= u_rec <= n_spatial) || continue
        (land_mask !== nothing && (land_mask[u_rel] || land_mask[u_rec])) && continue
        prev = get(pair_k, (u_rel, u_rec), (0, 0))
        pair_k[(u_rel, u_rec)] = (prev[1] + 1, prev[2] + max(2, Int(row.k)))
    end

    all_pairs = sort!(collect(keys(pair_k)))
    eval_pairs = if length(all_pairs) <= max_p
        all_pairs
    else
        idxs = unique(round.(Int, range(1, length(all_pairs); length = max_p)))
        all_pairs[idxs]
    end

    verbose && println(
        "  Corridor canalization: $(length(eval_pairs)) OD pairs x $(length(P_draws)) draws"
    )

    transit_by_draw = zeros(Float64, n_spatial, length(P_draws))
    spread_num      = zeros(Float64, n_spatial)
    spread_den      = zeros(Float64, n_spatial)

    for (di, P_d) in enumerate(P_draws)
        Pt_d = SparseMatrixCSC(P_d')
        col_transit = @view(transit_by_draw[:, di])

        for (u_rel, u_rec) in eval_pairs
            cnt, k_sum = pair_k[(u_rel, u_rec)]
            w_pair = 1.0 + log(Float64(cnt))
            K = min(180, max(2, round(Int, k_sum / cnt)))

            f_vecs = Vector{Vector{Float64}}(undef, K + 1)
            f0 = zeros(Float64, n_spatial)
            f0[u_rel] = 1.0
            f_vecs[1] = f0
            for tau in 1:K
                nxt = Pt_d * f_vecs[tau]
                s = sum(nxt)
                s > 0.0 && (nxt ./= s)
                f_vecs[tau + 1] = nxt
            end

            b_vecs = Vector{Vector{Float64}}(undef, K + 1)
            bK = zeros(Float64, n_spatial)
            bK[u_rec] = 1.0
            b_vecs[K + 1] = bK
            for tau in K:-1:1
                prv = P_d * b_vecs[tau + 1]
                s = sum(prv)
                s > 0.0 && (prv ./= s)
                b_vecs[tau] = prv
            end

            buf = zeros(Float64, n_spatial)
            n_int = max(1, K - 1)
            for tau in 1:(K - 1)
                c_idx = tau + 1
                csum = 0.0
                @inbounds for u in 1:n_spatial
                    val = (land_mask !== nothing && land_mask[u]) ?
                        0.0 : f_vecs[c_idx][u] * b_vecs[c_idx][u]
                    buf[u] = val
                    csum += val
                end
                csum > 1e-30 || continue
                buf ./= csum

                h_slice = 0.0
                @inbounds for u in 1:n_spatial
                    q = buf[u]
                    q > 1e-15 && (h_slice -= q * log(q))
                end
                w_eff_km = sqrt(exp(h_slice) * cell_area_km2)
                scale_w = w_pair / n_int
                @inbounds for u in 1:n_spatial
                    q = buf[u]
                    if q > 1e-12
                        wq = scale_w * q
                        col_transit[u] += wq
                        spread_num[u]  += wq * w_eff_km
                        spread_den[u]  += wq
                    end
                end
            end
        end
    end

    transit_mean = vec(mean(transit_by_draw; dims = 2))
    transit_sd   = size(transit_by_draw, 2) >= 2 ?
        vec(std(transit_by_draw; dims = 2)) : zeros(Float64, n_spatial)

    default_spread = sqrt(cell_area_km2)
    eff_spread_km = Vector{Float64}(undef, n_spatial)
    for u in 1:n_spatial
        eff_spread_km[u] = spread_den[u] > 1e-12 ?
            (spread_num[u] / spread_den[u]) : default_spread
    end

    active_spreads = Float64[
        eff_spread_km[u] for u in 1:n_spatial
        if transit_mean[u] > 1e-9 && (land_mask === nothing || !land_mask[u])
    ]
    med_spread = isempty(active_spreads) ? max(1.0, default_spread) :
        max(1.0, median(active_spreads))
    c_max = maximum(transit_mean; init = 0.0)

    raw_canal = zeros(Float64, n_spatial)
    if c_max > 0.0
        for u in 1:n_spatial
            (land_mask !== nothing && land_mask[u]) && continue
            rel_c = sqrt(transit_mean[u] / c_max)
            dir_focus = 1.0 - 0.5 * step_entropy[u]
            lat_focus = 1.0 / (1.0 + eff_spread_km[u] / med_spread)
            raw_canal[u] = rel_c * dir_focus * lat_focus
        end
    end
    k_max = maximum(raw_canal; init = 0.0)
    canalization = k_max > 0.0 ? (raw_canal ./ k_max) : raw_canal

    pos_k = filter(>(0.0), canalization)
    k_thresh = isempty(pos_k) ? Inf : quantile(pos_k, 0.90)
    conduit_mask = BitVector([
        canalization[u] >= k_thresh && canalization[u] > 0.0 &&
        (land_mask === nothing || !land_mask[u])
        for u in 1:n_spatial
    ])

    return (
        transit_density     = transit_mean,
        transit_sd          = transit_sd,
        step_entropy        = step_entropy,
        effective_spread_km = eff_spread_km,
        canalization_index  = canalization,
        conduit_mask        = conduit_mask,
        threshold           = isfinite(k_thresh) ? k_thresh : 0.0,
        n_pairs             = length(eval_pairs),
        n_draws             = length(P_draws),
    )
end

"""
    compute_effective_dispersal_area(loaded, kernels; horizon_steps=nothing, step_years=1/365.25) -> NamedTuple

Compute the multi-step (default 1-year, `K = round(Int, 1 / step_years)`) destination
Shannon entropy and effective dispersal area (Hill number of order 1) for every active
marine unit under the fitted transition operator `P`:

    H_K(u)     = -sum_v [P^K]_{u,v} * log([P^K]_{u,v})      (nats)
    N_eff(u)   = exp(H_K(u))                                (effective marine cells)
    A_eff(u)   = N_eff(u) * A_cell                          (km^2)
    r_eff(u)   = sqrt(A_eff(u) / pi)                        (km)

Inputs:
- `loaded`: Domain NamedTuple (`W`, `mesh`, `land_mask`, optional `region_map`, `region_labels`).
- `kernels`: Transition kernel NamedTuple containing `P_kernel`.
- `horizon_steps`: Positive integer step horizon (`nothing` defaults to `max(1, round(Int, 1 / step_years))`).
- `step_years`: Duration of one transition kernel step in years (`1 / 365.25` default).

Outputs:
NamedTuple `(horizon_steps, cell_area_km2, entropy_nats, effective_cells, effective_area_km2, effective_radius_km)`.
"""
function compute_effective_dispersal_area(
    loaded::NamedTuple,
    kernels::NamedTuple;
    horizon_steps::Union{Nothing, Int} = nothing,
    step_years::Real = 1.0 / 365.25
)::NamedTuple
    P_raw = kernels.P_kernel isa AbstractVector ? kernels.P_kernel[1] : kernels.P_kernel
    P_sp = P_raw isa SparseMatrixCSC ? P_raw : sparse(Float64.(P_raw))
    S = size(P_sp, 1)
    land_mask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing
    marine_idx = land_mask === nothing ? collect(1:S) : findall(!, land_mask)
    n_m = length(marine_idx)

    K = horizon_steps === nothing ? max(1, round(Int, 1.0 / max(Float64(step_years), 1e-6))) : Int(horizon_steps)
    K >= 1 || throw(DomainError(K, "horizon_steps must be >= 1."))

    # Estimate hexagonal cell area A_cell = (sqrt(3)/2) * d_med^2 from median marine edge length
    mesh = loaded.mesh
    cents = if hasproperty(mesh, :centroids_lonlat) && !isempty(mesh.centroids_lonlat)
        mesh.centroids_lonlat
    elseif hasproperty(mesh, :centroids) && !isempty(mesh.centroids)
        mesh.centroids
    else
        mesh.centroids_km
    end
    is_geo = _infer_coord_space(cents)
    W_sp = loaded.W isa SparseMatrixCSC ? loaded.W : sparse(Float64.(loaded.W))
    rows_w = rowvals(W_sp)
    edge_dists = Float64[]
    for u in marine_idx
        c1 = cents[u]
        for ptr in nzrange(W_sp, u)
            v = rows_w[ptr]
            v <= u && continue
            (land_mask !== nothing && land_mask[v]) && continue
            c2 = cents[v]
            d_km = is_geo ?
                haversine_distance(Float64(c1[1]), Float64(c1[2]), Float64(c2[1]), Float64(c2[2])) / 1000.0 :
                hypot(Float64(c2[1]) - Float64(c1[1]), Float64(c2[2]) - Float64(c1[2]))
            (isfinite(d_km) && d_km > 1e-6) && push!(edge_dists, d_km)
        end
    end
    d_med = isempty(edge_dists) ? 8.660254 : median(edge_dists)
    cell_area_km2 = 0.5 * sqrt(3.0) * d_med * d_med

    entropy_nats = zeros(Float64, S)
    eff_cells = zeros(Float64, S)
    eff_area = zeros(Float64, S)
    eff_radius = zeros(Float64, S)
    n_m == 0 && return (
        horizon_steps       = K,
        cell_area_km2       = cell_area_km2,
        entropy_nats        = entropy_nats,
        effective_cells     = eff_cells,
        effective_area_km2  = eff_area,
        effective_radius_km = eff_radius,
    )

    P_m = Matrix{Float64}(P_sp[marine_idx, marine_idx])
    for i in 1:n_m
        rs = 0.0
        @inbounds for j in 1:n_m
            rs += P_m[i, j]
        end
        if rs > 0.0
            inv_rs = 1.0 / rs
            @inbounds for j in 1:n_m
                P_m[i, j] *= inv_rs
            end
        else
            P_m[i, i] = 1.0
        end
    end

    # Binary matrix exponentiation: res = P_m^K
    res = Matrix{Float64}(I, n_m, n_m)
    base = P_m
    exp_rem = K
    while exp_rem > 0
        if (exp_rem & 1) == 1
            res = res * base
        end
        exp_rem >>= 1
        if exp_rem > 0
            base = base * base
        end
    end

    for (mi, u) in enumerate(marine_idx)
        rs = 0.0
        @inbounds for j in 1:n_m
            v_p = res[mi, j]
            v_p > 0.0 && (rs += v_p)
        end
        if rs <= 0.0
            entropy_nats[u] = 0.0
            eff_cells[u] = 1.0
        else
            inv_rs = 1.0 / rs
            h_u = 0.0
            @inbounds for j in 1:n_m
                q = res[mi, j] * inv_rs
                if q > 1e-15
                    h_u -= q * log(q)
                end
            end
            h_u = max(0.0, h_u)
            entropy_nats[u] = h_u
            eff_cells[u] = exp(h_u)
        end
        eff_area[u] = eff_cells[u] * cell_area_km2
        eff_radius[u] = sqrt(eff_area[u] / pi)
    end

    return (
        horizon_steps       = K,
        cell_area_km2       = cell_area_km2,
        entropy_nats        = entropy_nats,
        effective_cells     = eff_cells,
        effective_area_km2  = eff_area,
        effective_radius_km = eff_radius,
    )
end

function export_effective_dispersal_csv(
    eff::NamedTuple,
    loaded::NamedTuple,
    out_dir::AbstractString
)::String
    mkpath(out_dir)
    out_file = joinpath(out_dir, "effective_dispersal_1yr.csv")
    cents = hasproperty(loaded.mesh, :centroids_lonlat) ?
        loaded.mesh.centroids_lonlat : loaded.mesh.centroids
    lmask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing
    rmap  = hasproperty(loaded, :region_map) ? loaded.region_map : nothing
    rlbls = hasproperty(loaded, :region_labels) ? loaded.region_labels : String[]

    open(out_file, "w") do io
        write(io, "unit,lon,lat,region,horizon_steps,entropy_nats,effective_cells,effective_area_km2,effective_radius_km\n")
        for u in eachindex(eff.effective_area_km2)
            (lmask !== nothing && lmask[u]) && continue
            lon_u = round(Float64(cents[u][1]); digits = 5)
            lat_u = round(Float64(cents[u][2]); digits = 5)
            reg_lbl = if rmap !== nothing && u <= length(rmap) &&
                         1 <= rmap[u] <= length(rlbls)
                rlbls[rmap[u]]
            else
                "Unassigned"
            end
            write(io, string(
                u, ",", lon_u, ",", lat_u, ",", reg_lbl, ",",
                eff.horizon_steps, ",",
                round(eff.entropy_nats[u]; digits = 4), ",",
                round(eff.effective_cells[u]; digits = 2), ",",
                round(eff.effective_area_km2[u]; digits = 1), ",",
                round(eff.effective_radius_km[u]; digits = 2), "\n"
            ))
        end
    end
    return out_file
end

"""
    export_corridor_canalization_csv(canal, loaded, output_dir, params = (;)) -> String

Exports spatial corridor entropy, effective lateral spread, and canalization metrics
to `corridor_canalization_summary.csv`.
"""
function export_corridor_canalization_csv(
    canal::NamedTuple,
    loaded::NamedTuple,
    output_dir::AbstractString,
    params = (;)
)::String
    mkpath(output_dir)
    out_file = joinpath(output_dir, "corridor_canalization_summary.csv")
    n = loaded.n_spatial
    cents = hasproperty(loaded.mesh, :centroids_lonlat) ?
        loaded.mesh.centroids_lonlat : nothing
    part, rlabels = resolve_region_partition(loaded, params)

    open(out_file, "w") do io
        write(io, "unit,lon,lat,region,transit_density,transit_sd,step_entropy," *
                  "effective_spread_km,canalization_index,is_conduit\n")
        for u in 1:n
            (loaded.land_mask !== nothing && loaded.land_mask[u]) && continue
            lon_u = cents !== nothing ? round(Float64(cents[u][1]); digits = 5) : 0.0
            lat_u = cents !== nothing ? round(Float64(cents[u][2]); digits = 5) : 0.0
            reg_lbl = rlabels[part[u]]
            write(io, string(
                u, ",", lon_u, ",", lat_u, ",", reg_lbl, ",",
                round(canal.transit_density[u]; digits = 5), ",",
                round(canal.transit_sd[u]; digits = 5), ",",
                round(canal.step_entropy[u]; digits = 4), ",",
                round(canal.effective_spread_km[u]; digits = 2), ",",
                round(canal.canalization_index[u]; digits = 4), ",",
                canal.conduit_mask[u], "\n"
            ))
        end
    end
    return out_file
end

"""
    compute_spatial_bearing_roses(obs_df, loaded; grid_km = 100.0, min_events = 8) -> NamedTuple

Partition non-stationary marine displacements into `grid_km x grid_km` planar sectors
and evaluate directional bearing histograms, circular mean bearing `theta_bar` (degrees
clockwise from north), resultant vector length `R_bar` in `[0, 1]`, von Mises
concentration `kappa`, and small-sample Rayleigh uniformity statistics `Z = n * R_bar^2`.
Each sector anchor is snapped to the nearest occupied marine unit centroid.

Inputs:
- `obs_df`: Mark-recapture DataFrame with `release` and `recapture` unit indices.
- `loaded`: Domain NamedTuple containing `mesh` (`centroids_lonlat`, `centroids_km`) and `land_mask`.
- `grid_km`: Planar cell width in kilometres (`> 0`).
- `min_events`: Minimum non-stationary marine displacements required to report a sector (`>= 1`).

Outputs:
NamedTuple `(cells, cells_df, grid_km)` containing sector records and a summary DataFrame.
"""
function compute_spatial_bearing_roses(
    obs_df::DataFrame,
    loaded::NamedTuple;
    grid_km::Real = 100.0,
    min_events::Int = 8
)::NamedTuple
    g_km = Float64(grid_km)
    g_km > 0.0 || throw(DomainError(grid_km, "grid_km must be positive."))
    min_events >= 1 || throw(DomainError(min_events, "min_events must be >= 1."))

    mesh = loaded.mesh
    cents_ll = hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat : nothing
    cents_km = hasproperty(mesh, :centroids_km) ? mesh.centroids_km : nothing
    (cents_ll === nothing || isempty(cents_ll)) && return (
        grid_km = g_km, cells_df = DataFrame(), cells = NamedTuple[]
    )
    n_spatial = length(cents_ll)
    c_lon = hasproperty(mesh, :center_lon) ? Float64(mesh.center_lon) :
        mean(Float64(c[1]) for c in cents_ll)
    c_lat = hasproperty(mesh, :center_lat) ? Float64(mesh.center_lat) :
        mean(Float64(c[2]) for c in cents_ll)

    cell_events = Dict{Tuple{Int, Int}, Vector{Tuple{Float64, Float64, Float64, Float64, Int}}}()
    has_raw_ll = hasproperty(obs_df, :lon) && hasproperty(obs_df, :lat) &&
                 hasproperty(obs_df, :rec_lon) && hasproperty(obs_df, :rec_lat)
    lmask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing

    for row in eachrow(obs_df)
        u_rel = Int(row.release)
        u_rec = Int(row.recapture)
        (1 <= u_rel <= n_spatial && 1 <= u_rec <= n_spatial) || continue
        (lmask !== nothing && (lmask[u_rel] || lmask[u_rec])) && continue

        lon1, lat1, lon2, lat2 = if has_raw_ll &&
                isfinite(Float64(row.lon)) && isfinite(Float64(row.lat)) &&
                isfinite(Float64(row.rec_lon)) && isfinite(Float64(row.rec_lat))
            (Float64(row.lon), Float64(row.lat), Float64(row.rec_lon), Float64(row.rec_lat))
        else
            (Float64(cents_ll[u_rel][1]), Float64(cents_ll[u_rel][2]),
             Float64(cents_ll[u_rec][1]), Float64(cents_ll[u_rec][2]))
        end

        d_km = haversine_distance(lon1, lat1, lon2, lat2) / 1000.0
        d_km > 0.25 || continue

        d_lon = (lon2 - lon1) * cosd(0.5 * (lat1 + lat2))
        d_lat = lat2 - lat1
        brg = mod(atand(d_lon, d_lat), 360.0)

        x_km, y_km = if cents_km !== nothing && length(cents_km) == n_spatial
            (Float64(cents_km[u_rel][1]), Float64(cents_km[u_rel][2]))
        else
            lonlat_to_xy_km(lon1, lat1; center_lon = c_lon, center_lat = c_lat)
        end
        gx = floor(Int, x_km / g_km)
        gy = floor(Int, y_km / g_km)
        vec_cell = get!(cell_events, (gx, gy)) do
            Tuple{Float64, Float64, Float64, Float64, Int}[]
        end
        push!(vec_cell, (brg, d_km, lon1, lat1, u_rel))
    end

    cells_out = NamedTuple[]
    n_sec = 12
    sec_w = 360.0 / n_sec

    for ((gx, gy), evs) in sort!(collect(pairs(cell_events)); by = kv -> (-length(kv[2]), kv[1]))
        length(evs) >= min_events || continue
        brgs  = Float64[e[1] for e in evs]
        disps = Float64[e[2] for e in evs]
        raw_lon = mean(e[3] for e in evs)
        raw_lat = mean(e[4] for e in evs)

        # Snap sector rose anchor to the nearest occupied marine unit centroid so a
        # curved bay or island-straddling sector never places a rose marker on land.
        best_u = evs[1][5]
        best_d = Inf
        for e in evs
            u = e[5]
            (lmask !== nothing && lmask[u]) && continue
            ulon, ulat = Float64(cents_ll[u][1]), Float64(cents_ll[u][2])
            d = hypot((ulon - raw_lon) * cosd(raw_lat), ulat - raw_lat)
            if d < best_d
                best_d = d
                best_u = u
            end
        end
        m_lon = Float64(cents_ll[best_u][1])
        m_lat = Float64(cents_ll[best_u][2])
        cs = _circular_bearing_stats(brgs)

        counts = zeros(Int, n_sec)
        for b in brgs
            idx = min(n_sec, max(1, floor(Int, mod(b, 360.0) / sec_w) + 1))
            counts[idx] += 1
        end
        ew_lbl = gx >= 0 ? "+$(Int(round((gx + 0.5) * g_km)))E" :
                           "$(Int(round((gx + 0.5) * g_km)))W"
        ns_lbl = gy >= 0 ? "+$(Int(round((gy + 0.5) * g_km)))N" :
                           "$(Int(round((gy + 0.5) * g_km)))S"
        sector_id = "$(round(m_lat; digits=1))N, $(round(abs(m_lon); digits=1))W ($ew_lbl,$ns_lbl)"

        push!(cells_out, (
            gx                 = gx,
            gy                 = gy,
            sector_label       = sector_id,
            center_lon         = m_lon,
            center_lat         = m_lat,
            n_events           = length(evs),
            mean_disp_km       = mean(disps),
            mean_bearing_deg   = cs.mean_bearing_deg,
            mean_vector_length = cs.mean_vector_length,
            kappa              = cs.kappa,
            rayleigh_z         = cs.rayleigh_z,
            rayleigh_p         = cs.rayleigh_p,
            sector_counts      = counts,
            bearings_deg       = brgs,
        ))
    end

    cells_df = isempty(cells_out) ? DataFrame() : DataFrame([
        (
            gx                 = c.gx,
            gy                 = c.gy,
            sector_label       = c.sector_label,
            center_lon         = round(c.center_lon; digits = 4),
            center_lat         = round(c.center_lat; digits = 4),
            n_events           = c.n_events,
            mean_disp_km       = round(c.mean_disp_km; digits = 2),
            mean_bearing_deg   = round(c.mean_bearing_deg; digits = 1),
            mean_vector_length = round(c.mean_vector_length; digits = 4),
            kappa              = round(c.kappa; digits = 3),
            rayleigh_z         = round(c.rayleigh_z; digits = 2),
            rayleigh_p         = round(c.rayleigh_p; digits = 5),
        )
        for c in cells_out
    ])

    return (
        grid_km  = g_km,
        cells_df = cells_df,
        cells    = cells_out,
    )
end

function export_spatial_bearing_roses_csv(
    roses::NamedTuple,
    output_dir::AbstractString
)::String
    mkpath(output_dir)
    out_file = joinpath(output_dir, "spatial_bearing_roses_100km.csv")
    if nrow(roses.cells_df) > 0
        CSV.write(out_file, roses.cells_df)
    else
        write(out_file, "gx,gy,sector_label,center_lon,center_lat,n_events," *
                        "mean_disp_km,mean_bearing_deg,mean_vector_length," *
                        "kappa,rayleigh_z,rayleigh_p\n")
    end
    return out_file
end

"""
    analyze_gradient_taxis_projections(obs_df, loaded) -> NamedTuple

Projects observed individual displacement vectors onto local planar spatial gradients
of bathymetric depth (``\\nabla z``, downslope vs. upslope and along-isobath),
bottom temperature (``\\nabla T``, warmward vs. coldward), and habitat suitability
(``\\nabla\\text{HSI}``), stratified by release season and carapace width (`cw`).
"""
function _local_planar_gradients(
    field::AbstractVector{<:Real},
    W::SparseMatrixCSC,
    cents_km::AbstractVector,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Tuple{Vector{Float64}, Vector{Float64}}
    S = length(field)
    gx = fill(NaN, S)
    gy = fill(NaN, S)
    rows = rowvals(W)

    for u in 1:S
        (land_mask !== nothing && land_mask[u]) && continue
        fu = Float64(field[u])
        isfinite(fu) || continue
        xu, yu = Float64(cents_km[u][1]), Float64(cents_km[u][2])

        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        sxf = 0.0
        syf = 0.0
        n_nb = 0
        for ptr in nzrange(W, u)
            v = rows[ptr]
            v == u && continue
            (land_mask !== nothing && land_mask[v]) && continue
            fv = Float64(field[v])
            isfinite(fv) || continue
            dx = Float64(cents_km[v][1]) - xu
            dy = Float64(cents_km[v][2]) - yu
            df = fv - fu
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
            sxf += dx * df
            syf += dy * df
            n_nb += 1
        end
        det_m = sxx * syy - sxy * sxy
        tr_m  = sxx + syy
        if n_nb >= 2 && det_m > 1e-6 * tr_m * tr_m && det_m > 1e-8
            gx[u] = (syy * sxf - sxy * syf) / det_m
            gy[u] = (sxx * syf - sxy * sxf) / det_m
        elseif n_nb >= 1 && tr_m > 1e-6
            # Minimum-norm directional gradient along 1D channels or coastal tips
            gx[u] = sxf / tr_m
            gy[u] = syf / tr_m
        end
    end
    return (gx, gy)
end

"""
    analyze_gradient_taxis_projections(obs_df, loaded) -> NamedTuple

Project unit empirical displacement vectors `d_hat_i` onto local planar gradients of
bathymetric depth (`nabla z`), bottom temperature (`nabla T`), and habitat suitability
(`nabla HSI`) estimated via weighted least squares across adjacent marine centroids:

    cos(psi_f, i) = dot(d_hat_i, nabla f(u_rel, i)) / ||nabla f(u_rel, i)||_2  in [-1, 1]

Inputs:
- `obs_df`: Mark-recapture DataFrame (`release`, `recapture`, optional `rel_time`, `cw`).
- `loaded`: Domain NamedTuple containing `mesh`, `W`, `land_mask`, `resharded_depths`,
  `bottom_temp`, and `hsi_vec`.

Outputs:
NamedTuple `(n_obs, overall, by_season, by_cw, gradients)` reporting mean directional
cosines, downslope projections (km), along-isobath fractions (`|cos psi_z| < sin 25 deg`),
and two-sided t-test p-values across seasonal and size-class strata.
"""
function analyze_gradient_taxis_projections(
    obs_df::DataFrame,
    loaded::NamedTuple
)::NamedTuple
    n_obs = nrow(obs_df)
    mesh  = loaded.mesh
    W_sp  = loaded.W isa SparseMatrixCSC ? loaded.W : sparse(Float64.(loaded.W))
    n_s   = loaded.n_spatial
    lmask = loaded.land_mask

    cents_km = if hasproperty(mesh, :centroids_km) && length(mesh.centroids_km) == n_s
        mesh.centroids_km
    elseif hasproperty(mesh, :centroids_lonlat) && length(mesh.centroids_lonlat) == n_s
        c_lon = mean(Float64(c[1]) for c in mesh.centroids_lonlat)
        c_lat = mean(Float64(c[2]) for c in mesh.centroids_lonlat)
        [lonlat_to_xy_km(Float64(c[1]), Float64(c[2]); center_lon = c_lon, center_lat = c_lat)
         for c in mesh.centroids_lonlat]
    else
        Tuple{Float64, Float64}[]
    end
    (n_obs == 0 || isempty(cents_km)) && return (
        n_obs = 0, overall = DataFrame(), by_season = DataFrame(), by_cw = DataFrame()
    )

    depths = (hasproperty(loaded, :resharded_depths) &&
              loaded.resharded_depths !== nothing &&
              length(loaded.resharded_depths) == n_s) ?
             Float64.(loaded.resharded_depths) : Float64[]
    temps = if hasproperty(loaded, :bottom_temp) &&
               loaded.bottom_temp !== nothing &&
               length(loaded.bottom_temp) == n_s
        Float64.(loaded.bottom_temp)
    elseif hasproperty(loaded, :resharded_hydro) &&
           loaded.resharded_hydro !== nothing &&
           hasproperty(loaded.resharded_hydro, :temperature) &&
           length(loaded.resharded_hydro.temperature) == n_s
        Float64.(loaded.resharded_hydro.temperature)
    else
        Float64[]
    end
    hsi = (hasproperty(loaded, :hsi_vec) && length(loaded.hsi_vec) == n_s) ?
        Float64.(loaded.hsi_vec) : Float64[]

    gz_x, gz_y = !isempty(depths) ?
        _local_planar_gradients(depths, W_sp, cents_km, lmask) : (fill(NaN, n_s), fill(NaN, n_s))
    gt_x, gt_y = !isempty(temps) ?
        _local_planar_gradients(temps, W_sp, cents_km, lmask) : (fill(NaN, n_s), fill(NaN, n_s))
    gh_x, gh_y = !isempty(hsi) ?
        _local_planar_gradients(hsi, W_sp, cents_km, lmask) : (fill(NaN, n_s), fill(NaN, n_s))

    q_names = ["Q1 (Winter)", "Q2 (Spring)", "Q3 (Summer)", "Q4 (Autumn)"]
    has_cw = hasproperty(obs_df, :cw)
    has_rt = hasproperty(obs_df, :rel_time)

    cos_z   = Float64[]
    proj_z  = Float64[]
    iso_rat = Float64[]
    cos_t   = Float64[]
    proj_t  = Float64[]
    dt_end  = Float64[]
    cos_h   = Float64[]
    proj_h  = Float64[]
    rec_q   = String[]
    rec_cw  = Float64[]

    for row in eachrow(obs_df)
        u1 = Int(row.release)
        u2 = Int(row.recapture)
        (1 <= u1 <= n_s && 1 <= u2 <= n_s && u1 != u2) || continue
        dx = Float64(cents_km[u2][1]) - Float64(cents_km[u1][1])
        dy = Float64(cents_km[u2][2]) - Float64(cents_km[u1][2])
        d_mag = hypot(dx, dy)
        d_mag > 0.25 || continue
        ux = dx / d_mag
        uy = dy / d_mag

        mz = hypot(gz_x[u1], gz_y[u1])
        cz = (isfinite(mz) && mz > 1e-4) ? (ux * gz_x[u1] + uy * gz_y[u1]) / mz : NaN
        pz = isfinite(cz) ? d_mag * cz : NaN
        ir = isfinite(cz) ? sqrt(max(0.0, 1.0 - cz * cz)) : NaN

        mt = hypot(gt_x[u1], gt_y[u1])
        ct = (isfinite(mt) && mt > 1e-5) ? (ux * gt_x[u1] + uy * gt_y[u1]) / mt : NaN
        pt = isfinite(ct) ? d_mag * ct : NaN
        dte = (!isempty(temps) && isfinite(temps[u1]) && isfinite(temps[u2])) ?
            (temps[u2] - temps[u1]) : NaN

        mh = hypot(gh_x[u1], gh_y[u1])
        ch = (isfinite(mh) && mh > 1e-4) ? (ux * gh_x[u1] + uy * gh_y[u1]) / mh : NaN
        ph = isfinite(ch) ? d_mag * ch : NaN

        q_lbl = if has_rt && !ismissing(row.rel_time) && isfinite(Float64(row.rel_time))
            t = Float64(row.rel_time)
            mo = min(12, max(1, Int(fld((t - floor(Int, t)) * 12.0, 1.0)) + 1))
            q_names[cld(mo, 3)]
        else
            "All"
        end
        cw_val = (has_cw && !ismissing(row.cw) && isfinite(Float64(row.cw))) ?
            Float64(row.cw) : NaN

        push!(cos_z, cz)
        push!(proj_z, pz)
        push!(iso_rat, ir)
        push!(cos_t, ct)
        push!(proj_t, pt)
        push!(dt_end, dte)
        push!(cos_h, ch)
        push!(proj_h, ph)
        push!(rec_q, q_lbl)
        push!(rec_cw, cw_val)
    end

    function _ttest_mean(vals::Vector{Float64})
        v = Float64[x for x in vals if isfinite(x)]
        n = length(v)
        n == 0 && return (0, NaN, 0.0, 1.0)
        mu = mean(v)
        n < 3 && return (n, mu, 0.0, 1.0)
        se = std(v) / sqrt(n)
        t_val = se > 1e-12 ? mu / se : 0.0
        p_val = max(0.0, min(1.0, 2.0 * ccdf(TDist(n - 1), abs(t_val))))
        return (n, mu, t_val, p_val)
    end

    function _summarize_taxis(group_lbl::String, idxs::Vector{Int})
        nz, muz_c, tz, pz = _ttest_mean(cos_z[idxs])
        _, muz_km, _, _   = _ttest_mean(proj_z[idxs])
        _, mu_iso, _, _   = _ttest_mean(iso_rat[idxs])
        nt, mut_c, tt, pt = _ttest_mean(cos_t[idxs])
        _, mut_km, _, _   = _ttest_mean(proj_t[idxs])
        _, mu_dt, _, _    = _ttest_mean(dt_end[idxs])
        nh, muh_c, th, ph = _ttest_mean(cos_h[idxs])
        return (
            stratum              = group_lbl,
            n_bathy              = nz,
            mean_downslope_cos   = muz_c,
            mean_downslope_km    = muz_km,
            along_isobath_frac   = mu_iso,
            bathy_t              = tz,
            bathy_p              = pz,
            n_thermal            = nt,
            mean_warmward_cos    = mut_c,
            mean_warmward_km     = mut_km,
            mean_delta_T_C       = mu_dt,
            thermal_t            = tt,
            thermal_p            = pt,
            n_hsi                = nh,
            mean_hsi_upgrad_cos  = muh_c,
            hsi_t                = th,
            hsi_p                = ph,
        )
    end

    n_mov = length(cos_z)
    overall_df = n_mov > 0 ?
        DataFrame([_summarize_taxis("All Non-Stationary", collect(1:n_mov))]) : DataFrame()

    season_rows = NamedTuple[]
    for qn in q_names
        idxs = findall(==(qn), rec_q)
        !isempty(idxs) && push!(season_rows, _summarize_taxis(qn, idxs))
    end
    season_df = DataFrame(season_rows)

    cw_rows = NamedTuple[]
    for (lbl, lo, hi) in [("CW < 95 mm", 0.0, 95.0), ("CW 95-110 mm", 95.0, 110.0),
                          ("CW 110-125 mm", 110.0, 125.0), ("CW >= 125 mm", 125.0, Inf)]
        idxs = findall(i -> isfinite(rec_cw[i]) && rec_cw[i] >= lo && rec_cw[i] < hi, 1:n_mov)
        !isempty(idxs) && push!(cw_rows, _summarize_taxis(lbl, idxs))
    end
    cw_df = DataFrame(cw_rows)

    return (
        n_obs     = n_mov,
        overall   = overall_df,
        by_season = season_df,
        by_cw     = cw_df,
    )
end

function export_gradient_taxis_csv(
    taxis::NamedTuple,
    output_dir::AbstractString
)::String
    mkpath(output_dir)
    out_file = joinpath(output_dir, "gradient_taxis_summary.csv")
    dfs = DataFrame[]
    for df in (taxis.overall, taxis.by_season, taxis.by_cw)
        nrow(df) > 0 && push!(dfs, df)
    end
    if !isempty(dfs)
        CSV.write(out_file, vcat(dfs...))
    else
        write(out_file, "stratum,n_bathy,mean_downslope_cos,mean_downslope_km," *
                        "along_isobath_frac,bathy_t,bathy_p,n_thermal," *
                        "mean_warmward_cos,mean_warmward_km,mean_delta_T_C," *
                        "thermal_t,thermal_p,n_hsi,mean_hsi_upgrad_cos,hsi_t,hsi_p\n")
    end
    return out_file
end

"""
    reconstruct_paths_bayesian_ensemble(loaded, fitted, params = (;)) -> NamedTuple

Reconstructs paths and corridor probabilities by iterating directly over
Turing MCMC posterior samples, capturing full parameter uncertainty.
"""
function reconstruct_paths_bayesian_ensemble(
    loaded::NamedTuple,
    fitted::NamedTuple,
    params = (;)
)::NamedTuple
    verbose   = params.verbose
    obs_df    = loaded.obs_df
    W         = loaded.W
    hsi_vec   = loaded.hsi_vec
    land_mask = loaded.land_mask
    n_spatial = loaded.n_spatial

    chains    = fitted.chains
    active_chn = haskey(chains, :telemetry) ? chains[:telemetry] :
                 haskey(chains, :telemetry_and_survey) ?
                 chains[:telemetry_and_survey] :
                 first(values(chains))

    cents_mesh = if hasproperty(loaded, :mesh) &&
                    hasproperty(loaded.mesh, :centroids_planar)
        loaded.mesh.centroids_planar
    elseif hasproperty(loaded, :centroids_planar)
        loaded.centroids_planar
    else
        [Float64[0.0, 0.0] for _ in 1:n_spatial]
    end

    vel_draws, diff_draws, gam_draws =
        TethysPath.posterior_kernel_draws(active_chn)

    chn_mat_v = reshape(vel_draws, :, 1)
    chn_mat_d = reshape(diff_draws, :, 1)
    chn_mat_g = reshape(gam_draws, :, 1)

    n_draws_chain = size(chn_mat_v, 1)
    req_ensemble  = params.n_ensemble
    n_ensemble    = min(req_ensemble, n_draws_chain)
    draw_indices  = round.(Int, range(1, n_draws_chain, length = n_ensemble))

    all_tags    = unique(obs_df.tagid)
    max_p       = params.max_paths
    sample_tags = all_tags[1:min(max_p, length(all_tags))]
    path_method = first(params.path_methods)

    ensemble_corridors = Dict{String, Matrix{Float64}}()
    ensemble_paths     = Dict{String, Vector{Vector{Int}}}()

    verbose && println(
        "[Ensemble Phase] Running path prediction over ",
        "$n_ensemble MCMC posterior samples..."
    )

    P_draws = Any[]
    for d_idx in draw_indices
        alpha_arr, rho_arr = movement_alpha_rho(
            [chn_mat_v[d_idx, 1]], [chn_mat_d[d_idx, 1]])
        gamma_val = Float64(chn_mat_g[d_idx, 1])

        P_draw = construct_stochastic_transition_kernel(
            W, hsi_vec;
            gamma     = gamma_val,
            residence = rho_arr,
            advection = alpha_arr,
            land_mask = land_mask
        )
        push!(P_draws, P_draw)
    end

    for tid in sample_tags
        sub_obs = filter(:tagid => ==(tid), obs_df)
        isempty(sub_obs) && continue
        first_row = first(sub_obs)

        k_steps = max(1, first_row.k)
        accumulated_corridor = zeros(Float64, n_spatial, k_steps + 1)
        sample_path_collection = Vector{Int}[]

        for (i, d_idx) in enumerate(draw_indices)
            P_draw = P_draws[i]
            rng_draw = MersenneTwister(hash((string(tid), d_idx)))

            corr_draw = predict_corridor(
                P_draw, first_row.release, first_row.recapture, k_steps;
                land_mask = land_mask
            )
            accumulated_corridor .+= corr_draw

            path_draw = predict_path(
                P_draw, first_row.release, first_row.recapture, k_steps;
                centroids = cents_mesh,
                method    = path_method,
                land_mask = land_mask,
                rng       = rng_draw
            )
            push!(sample_path_collection, path_draw)
        end

        ensemble_corridors[string(tid)] = accumulated_corridor ./ n_ensemble
        ensemble_paths[string(tid)]     = sample_path_collection
    end

    return (
        ensemble_corridors = ensemble_corridors,
        ensemble_paths     = ensemble_paths
    )
end

"""
    annual_region_transition_matrix(
        P, region_map;
        hsi_vec = nothing, land_mask = nothing, horizon_steps = 365
    ) -> Matrix{Float64}

Computes the 1-year (``K = \\text{horizon\\_steps}``) HSI-weighted regional
transition probability matrix ``C_{\\text{1yr}}[r, s]`` by forward-propagating
habitat-weighted initial regional occupancy ``p_r^{(0)}`` under ``(P^\\top)^K``.
"""
function annual_region_transition_matrix(
    P::AbstractMatrix{<:Real},
    region_map::Vector{Int};
    hsi_vec::Union{Nothing, AbstractVector{<:Real}} = nothing,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    horizon_steps::Int = 365
)::Matrix{Float64}
    size(P, 1) == size(P, 2) || throw(ArgumentError(
        "P must be square, got $(size(P, 1)) x $(size(P, 2))."
    ))
    horizon_steps >= 1 || throw(DomainError(
        horizon_steps, "horizon_steps must be >= 1."
    ))
    n = size(P, 1)
    length(region_map) == n || throw(ArgumentError(
        "region_map length $(length(region_map)) != $n."
    ))
    n_regions = maximum(region_map)
    n_regions >= 1 || throw(DomainError(n_regions, "region_map is empty."))
    minimum(region_map) >= 1 || throw(DomainError(
        minimum(region_map), "region_map must label units from 1 upward."
    ))

    Pt_sp = P isa SparseMatrixCSC ? SparseMatrixCSC(P') : sparse(Float64.(P'))
    C_yr = zeros(Float64, n_regions, n_regions)

    for r in 1:n_regions
        p = zeros(Float64, n)
        n_in_r = 0
        for u in 1:n
            region_map[u] == r || continue
            n_in_r += 1
            (land_mask !== nothing && land_mask[u]) && continue
            w_h = (hsi_vec !== nothing && u <= length(hsi_vec) && isfinite(hsi_vec[u])) ?
                max(Float64(hsi_vec[u]), 1e-3) : 1.0
            p[u] = w_h
        end
        n_in_r > 0 || throw(DomainError(r, "region $r has zero units in region_map."))
        s0 = sum(p)
        if s0 <= 0.0
            for u in 1:n
                region_map[u] == r && (p[u] = 1.0 / n_in_r)
            end
        else
            p ./= s0
        end

        for _ in 1:horizon_steps
            p = Pt_sp * p
            sp = sum(p)
            sp > 0.0 && (p ./= sp)
        end

        for v in 1:n
            C_yr[r, region_map[v]] += p[v]
        end
        rs = sum(@view(C_yr[r, :]))
        rs > 0.0 && (C_yr[r, :] ./= rs)
    end

    return C_yr
end

function _annual_horizon_steps(loaded, params)::Int
    step_yr = if hasproperty(params, :fpt_step_years) &&
                 params.fpt_step_years !== nothing &&
                 isfinite(params.fpt_step_years) && params.fpt_step_years > 0
        Float64(params.fpt_step_years)
    elseif hasproperty(loaded, :data) && loaded.data !== nothing &&
           hasproperty(loaded.data, :step_years) && loaded.data.step_years > 0
        Float64(loaded.data.step_years)
    elseif hasproperty(loaded, :obs_df) && loaded.obs_df isa DataFrame &&
           hasproperty(loaded.obs_df, :k) && nrow(loaded.obs_df) > 0 &&
           maximum(loaded.obs_df.k) > 120
        1.0 / 365.25
    else
        1.0 / 365.25
    end
    return max(1, round(Int, 1.0 / step_yr))
end

"""
    compute_stock_connectivity_matrix(loaded, kernels, params = (;);
                                      region_labels = nothing, region_map = nothing)

Derives the 1-year (``P^{K_{\\text{yr}}}``) HSI-weighted regional stock transition
probability matrix across management regions alongside empirical transition counts.
"""
function compute_stock_connectivity_matrix(
    loaded::NamedTuple,
    kernels::NamedTuple,
    params = (;);
    region_labels::Union{Vector{String}, Nothing} = nothing,
    region_map::Union{Vector{Int}, Nothing} = nothing
)::NamedTuple
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    P_kernel  = kernels.P_kernel

    region_map, region_labels = resolve_region_partition(
        loaded, params; region_map = region_map, region_labels = region_labels
    )
    n_regions = maximum(region_map)
    horizon_yr = _annual_horizon_steps(loaded, params)

    flow_counts = zeros(Int, n_regions, n_regions)
    for row in eachrow(obs_df)
        rel = Int(row.release)
        rec = Int(row.recapture)
        (1 <= rel <= n_spatial) || throw(BoundsError(region_map, rel))
        (1 <= rec <= n_spatial) || throw(BoundsError(region_map, rec))
        rel_r = region_map[rel]
        rec_r = region_map[rec]
        (1 <= rel_r <= n_regions && 1 <= rec_r <= n_regions) || continue
        flow_counts[rel_r, rec_r] += 1
    end

    P_k = P_kernel isa AbstractVector ? P_kernel[1] : P_kernel
    hsi_v = hasproperty(loaded, :hsi_vec) ? loaded.hsi_vec : nothing
    lmask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing
    connectivity_matrix = annual_region_transition_matrix(
        P_k, Int.(region_map);
        hsi_vec = hsi_v, land_mask = lmask, horizon_steps = horizon_yr
    )

    flow_rates = connectivity_matrix .* (flow_counts ./ (sum(flow_counts) + 1e-10))

    return (
        connectivity_matrix = connectivity_matrix,
        flow_counts         = flow_counts,
        flow_rates          = flow_rates,
        region_labels       = region_labels,
        region_map          = region_map,
        n_regions           = n_regions,
        horizon_steps       = horizon_yr,
    )
end

"""
    compute_connectivity_credible_intervals(loaded, fitted, kernels, params = (;),
                                            region_map = nothing)

Propagates MCMC posterior uncertainty to derive 95% credible intervals for
1-year HSI-weighted inter-region transition probabilities.
"""
function compute_connectivity_credible_intervals(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params = (;),
    region_map::Union{Vector{Int}, Nothing} = nothing
)::NamedTuple
    land_mask  = loaded.land_mask
    horizon_yr = _annual_horizon_steps(loaded, params)

    region_map, _ = resolve_region_partition(loaded, params; region_map = region_map)
    n_regions = maximum(region_map)

    a_samples = _posterior_param_draws(kernels, :alpha_samples, 0.3)
    r_samples = _posterior_param_draws(kernels, :rho_samples, 0.5)
    g_samples = _posterior_param_draws(kernels, :gamma_samples, 1.0)
    n_draws   = size(a_samples, 1)

    n_eval_draws = min(n_draws, 30)
    eval_indices = unique(round.(Int, range(1, n_draws, length=n_eval_draws)))
    connectivity_samples = Matrix{Float64}[]
    hsi_v = hasproperty(loaded, :hsi_vec) ? loaded.hsi_vec : nothing

    for draw in eval_indices
        alpha_val = max(0.0, min(1.0, Float64(a_samples[draw, 1])))
        rho_val   = max(0.0, min(0.9999, Float64(r_samples[draw, 1])))
        gamma_val = Float64(g_samples[draw, 1])

        P_draw = sparse(construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = gamma_val,
            residence = rho_val,
            advection = alpha_val,
            land_mask = land_mask
        ))
        conn_mat = annual_region_transition_matrix(
            P_draw, region_map;
            hsi_vec = hsi_v, land_mask = land_mask, horizon_steps = horizon_yr
        )
        push!(connectivity_samples, conn_mat)
    end

    conn_mean = mean(connectivity_samples)
    conn_lower = [quantile([s[i, j] for s in connectivity_samples], 0.025)
                  for i in 1:n_regions, j in 1:n_regions]
    conn_upper = [quantile([s[i, j] for s in connectivity_samples], 0.975)
                  for i in 1:n_regions, j in 1:n_regions]

    return (
        connectivity_mean    = conn_mean,
        connectivity_lower   = conn_lower,
        connectivity_upper   = conn_upper,
        connectivity_samples = connectivity_samples,
        n_regions            = n_regions,
        horizon_steps        = horizon_yr,
    )
end

"""
    export_connectivity_matrix(conn, output_dir) -> String

Exports 1-year regional stock transition probabilities and empirical flow counts.
"""
function export_connectivity_matrix(conn, output_dir)::String
    mkpath(output_dir)
    summary_file = joinpath(output_dir, "stock_connectivity_summary.csv")
    open(summary_file, "w") do io
        write(io, "source_region,sink_region,annual_transition_prob,flow_count,flow_rate\n")
        for r in 1:conn.n_regions, s in 1:conn.n_regions
            write(io, string(
                conn.region_labels[r], ",",
                conn.region_labels[s], ",",
                round(conn.connectivity_matrix[r, s]; digits=6), ",",
                conn.flow_counts[r, s], ",",
                round(conn.flow_rates[r, s]; digits=6), "\n"
            ))
        end
    end
    return summary_file
end

"""
    export_connectivity_uncertainty(conn_unc, output_dir, region_labels) -> String

Exports 1-year regional transition credible interval bounds to CSV.
"""
function export_connectivity_uncertainty(conn_unc, output_dir, region_labels)::String
    mkpath(output_dir)
    unc_file = joinpath(output_dir, "stock_connectivity_credible_intervals.csv")
    open(unc_file, "w") do io
        write(io, "source_region,sink_region,mean_annual_prob,lower_ci,upper_ci\n")
        for r in 1:conn_unc.n_regions, s in 1:conn_unc.n_regions
            write(io, string(
                region_labels[r], ",",
                region_labels[s], ",",
                round(conn_unc.connectivity_mean[r, s]; digits=6), ",",
                round(conn_unc.connectivity_lower[r, s]; digits=6), ",",
                round(conn_unc.connectivity_upper[r, s]; digits=6), "\n"
            ))
        end
    end
    return unc_file
end

"""
    posterior_predictive_check(loaded, fitted, kernels, params = (;)) -> NamedTuple

Performs Bayesian posterior predictive validation by simulating mark-recapture
events and evaluating Kullback-Leibler divergence and displacement calibration.
"""
function _month_key(t::Real)::String
    yr = floor(Int, t)
    mo = min(12, max(1, ceil(Int, (t - yr) * 12)))
    return "$(yr)-$(lpad(string(mo), 2, '0'))"
end
_month_key(::Missing) = "missing"
_month_key(x) = "all"

function posterior_predictive_check(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params = (;)
)::NamedTuple
    obs_df    = loaded.obs_df
    n_spatial = loaded.n_spatial
    land_mask = loaded.land_mask
    seed      = params.seed

    a_samples = _posterior_param_draws(kernels, :alpha_samples, 0.3)
    r_samples = _posterior_param_draws(kernels, :rho_samples, 0.5)
    g_samples = _posterior_param_draws(kernels, :gamma_samples, 1.0)
    n_draws   = size(a_samples, 1)

    observed_recaptures = obs_df.recapture
    observed_dist = zeros(n_spatial)
    for rec in observed_recaptures
        1 <= rec <= n_spatial && (observed_dist[rec] += 1)
    end
    observed_dist ./= max(1.0, sum(observed_dist))

    kl_divergences = Float64[]
    simulated_recapture_dists = Vector{Int}[]
    rng = MersenneTwister(seed)

    n_eval_draws = min(n_draws, 20)
    eval_indices = unique(round.(Int, range(1, n_draws, length=n_eval_draws)))
    ks_true = max.(1, Int.(round.(obs_df.k)))

    for draw in eval_indices
        alpha_val = max(0.0, min(1.0, Float64(a_samples[draw, 1])))
        rho_val   = max(0.0, min(0.9999, Float64(r_samples[draw, 1])))
        gamma_val = Float64(g_samples[draw, 1])

        P_k = sparse(construct_stochastic_transition_kernel(
            loaded.W, loaded.hsi_vec;
            gamma     = gamma_val,
            residence = rho_val,
            advection = alpha_val,
            land_mask = land_mask
        ))

        PT = sparse(P_k')
        kcache = kstep_transition_cache(PT, Int.(obs_df.release), ks_true)

        simulated_recaptures = Int[]
        for n in 1:nrow(obs_df)
            rel = Int(obs_df.release[n])
            kk  = ks_true[n]

            if 1 <= rel <= n_spatial
                prob_vec = kcache[(rel, kk)]
                s = sum(prob_vec)
                if s > 0.0
                    prob_vec = prob_vec ./ s
                    u_samp = rand(rng)
                    c_sum = 0.0
                    sampled_rec = rel
                    for j in 1:n_spatial
                        c_sum += prob_vec[j]
                        if u_samp <= c_sum
                            sampled_rec = j
                            break
                        end
                    end
                    push!(simulated_recaptures, sampled_rec)
                else
                    push!(simulated_recaptures, rel)
                end
            end
        end

        sim_dist = zeros(n_spatial)
        for rec in simulated_recaptures
            1 <= rec <= n_spatial && (sim_dist[rec] += 1)
        end
        sim_dist ./= max(1.0, sum(sim_dist))
        push!(simulated_recapture_dists, simulated_recaptures)

        kl = 0.0
        for i in 1:n_spatial
            if observed_dist[i] > 1e-10 && sim_dist[i] > 1e-10
                kl += observed_dist[i] * log(observed_dist[i] / sim_dist[i])
            end
        end
        push!(kl_divergences, kl)
    end

    summary = (
        n_draws        = length(kl_divergences),
        n_observations = length(observed_recaptures),
        kl_mean        = mean(kl_divergences),
        kl_sd          = std(kl_divergences),
        kl_lower_ci    = quantile(kl_divergences, 0.025),
        kl_upper_ci    = quantile(kl_divergences, 0.975),
    )

    predicted_dist_mean = zeros(n_spatial)
    for sim_recs in simulated_recapture_dists
        sim_dist = zeros(n_spatial)
        for rec in sim_recs
            1 <= rec <= n_spatial && (sim_dist[rec] += 1)
        end
        sim_dist ./= max(1.0, sum(sim_dist))
        predicted_dist_mean .+= sim_dist
    end
    predicted_dist_mean ./= max(1.0, length(simulated_recapture_dists))

    return (
        kl_divergences            = kl_divergences,
        observed_recaptures       = observed_recaptures,
        observed_dist             = observed_dist,
        simulated_recapture_dists = simulated_recapture_dists,
        predicted_dist_mean       = predicted_dist_mean,
        summary                   = summary,
    )
end

"""
    export_posterior_predictive_check(ppc, output_dir) -> String

Exports posterior predictive validation diagnostics to a text summary.
"""
function export_posterior_predictive_check(ppc, output_dir)::String
    mkpath(output_dir)
    summary_file = joinpath(output_dir, "posterior_predictive_summary.txt")
    open(summary_file, "w") do f
        write(f, "Posterior Predictive Check Summary\n")
        write(f, "=" ^ 50 * "\n\n")
        write(f, "Observations: $(ppc.summary.n_observations)\n")
        write(f, "Evaluated MCMC Draws: $(ppc.summary.n_draws)\n\n")
        write(f, "KL Divergence (nats): $(round(ppc.summary.kl_mean; digits=6)) " *
                 "[$(round(ppc.summary.kl_lower_ci; digits=6)), " *
                 "$(round(ppc.summary.kl_upper_ci; digits=6))]\n")
    end
    return summary_file
end

"""
    plot_posterior_predictive_check(ppc, output_dir) -> String

Base stub for `Plots`-backed posterior predictive diagnostics (implemented in
`TethysPathPlottingExt`).
"""
function plot_posterior_predictive_check(ppc, output_dir)::String
    ext = Base.get_extension(TethysPath, :TethysPathPlottingExt)
    ext === nothing && throw(ArgumentError(
        "plot_posterior_predictive_check requires `Plots` to be loaded."
    ))
    return TethysPath.plot_posterior_predictive_check(ppc, output_dir)
end

"""
    run_validation_analyses(loaded, fitted, kernels, params, output_dir) -> NamedTuple

Consolidated validation post-processing pipeline executing:
- Spatial corridor entropy and canalization (`corridor_canalization_summary.csv`)
- 1-year HSI-weighted regional stock connectivity (`stock_connectivity_summary.csv`)
- Posterior predictive checks (`posterior_predictive_summary.txt`)
"""
function run_validation_analyses(
    loaded::NamedTuple,
    fitted::NamedTuple,
    kernels::NamedTuple,
    params,
    output_dir::String
)::NamedTuple
    mkpath(output_dir)
    verbose = params.verbose

    verbose && println("\n--- Running Validation Analyses ---")
    canal = compute_corridor_canalization(loaded, fitted, kernels, params)
    export_corridor_canalization_csv(canal, loaded, output_dir, params)

    conn = compute_stock_connectivity_matrix(loaded, kernels, params)
    export_connectivity_matrix(conn, output_dir)

    conn_unc = compute_connectivity_credible_intervals(
        loaded, fitted, kernels, params, conn.region_map
    )
    export_connectivity_uncertainty(conn_unc, output_dir, conn.region_labels)

    ppc = posterior_predictive_check(loaded, fitted, kernels, params)
    summary_file = export_posterior_predictive_check(ppc, output_dir)

    return (
        corridor_canalization    = canal,
        connectivity_matrix      = conn,
        connectivity_uncertainty = conn_unc,
        posterior_predictive     = ppc,
        summary_file             = summary_file,
    )
end

# =============================================================================
# Adaptive Multiresolution Hexagonal Mesh & Routing
# =============================================================================

"""
    construct_adaptive_multiresolution_domain(
        lon_vec::AbstractVector{<:Real},
        lat_vec::AbstractVector{<:Real};
        coarse_radius_km::Real = 25.0,
        fine_radius_km::Real = 8.0,
        refine_zones = nothing,
        land_polygons = nothing,
        refine_buffer_km::Real = 20.0,
        depth = nothing,
        depth_threshold::Real = 0.0,
        crs = nothing,
        datum = WGS84Latest
    )::NamedTuple

Constructs a hierarchical dual-resolution hexagonal spatial mesh. Coarse
hexagons of radius `coarse_radius_km` cover regional/offshore waters, while
high-priority zones (coastal zones, steep bathymetric transitions, or telemetry
focal areas) are adaptively refined into sub-hexagons of radius `fine_radius_km`.

# Process Description
1. Bounding coordinates of `(lon_vec, lat_vec)` are projected to planar
   coordinates using `lonlat_to_xy_km`.
2. A base regular hexagonal lattice with spacing dx = sqrt(3)*rc, dy = 1.5*rc
   is initialized.
3. Coarse cells whose centroids lie within `refine_buffer_km` of land barriers,
   shallow bathymetry, or user-provided `refine_zones` are flagged for
   refinement.
4. Each flagged cell is subdivided into sub-hexagons of radius `fine_radius_km`.
5. The cross-resolution spatial adjacency graph W connects:
   - Fine-fine neighbor pairs within sqrt(3) * rf * 1.08
   - Coarse-coarse neighbor pairs within sqrt(3) * rc * 1.08
   - Coarse-fine boundary interface pairs within (rc + rf) * 0.95

# Arguments
- `lon_vec`, `lat_vec`: Spatial coordinate extents.
- `coarse_radius_km`: Circumradius for offshore/regional cells (km).
- `fine_radius_km`: Circumradius for refined/coastal cells (km).
- `refine_zones`: Optional collection of focus coordinates (lon, lat).
- `land_polygons`: Optional terrestrial barrier polygons.
- `refine_buffer_km`: Proximity distance triggering refinement (km).
- `depth`: Optional bathymetry depth values.
- `depth_threshold`: Depth cutoff for land masking.
- `crs`, `datum`: Geodetic projection parameters.

# Returns
`NamedTuple` with fields:
- `centroids_km`: Projected coordinates (km) of all cells.
- `centroids_lonlat`: Geographic coordinates (lon, lat) of all cells.
- `polygons_km`, `polygons_lonlat`: 6-vertex boundary outlines.
- `W`: Cross-resolution sparse spatial adjacency matrix.
- `areas_km2`: Area of each unit in square kilometers.
- `is_fine`: Boolean indicator vector identifying fine-scale units.
- `radius_km`: Cell radius per unit.
- `center_lon`, `center_lat`: Projection center coordinates.
- `n_units`: Total spatial units S.
- `land_mask`: Boolean land mask vector.
"""
function construct_adaptive_multiresolution_domain(
    lon_vec::AbstractVector{<:Real},
    lat_vec::AbstractVector{<:Real};
    coarse_radius_km::Real = 25.0,
    fine_radius_km::Real = 8.0,
    refine_zones = nothing,
    land_polygons = nothing,
    refine_buffer_km::Real = 20.0,
    depth = nothing,
    depth_threshold::Real = 0.0,
    crs = nothing,
    datum = WGS84Latest
)::NamedTuple
    rc = Float64(coarse_radius_km)
    rf = min(Float64(fine_radius_km), rc * 0.75)

    # 1. Base coarse domain
    base_domain = construct_full_movement_domain(
        lon_vec, lat_vec;
        radius_km = rc,
        land_polygons = land_polygons,
        depth = depth,
        depth_threshold = depth_threshold,
        crs = crs,
        datum = datum
    )

    c_km = base_domain.centroids_km
    c_ll = base_domain.centroids_lonlat
    S_coarse = length(c_km)
    l_mask_coarse = base_domain.land_mask

    # 2. Identify coarse units requiring refinement
    refine_mask = falses(S_coarse)
    buf_sq = refine_buffer_km^2

    for i in 1:S_coarse
        if l_mask_coarse[i]
            refine_mask[i] = true
            continue
        end
        for j in 1:S_coarse
            if l_mask_coarse[j]
                dx = c_km[i][1] - c_km[j][1]
                dy = c_km[i][2] - c_km[j][2]
                if (dx^2 + dy^2) <= buf_sq
                    refine_mask[i] = true
                    break
                end
            end
        end
    end

    if refine_zones !== nothing
        for pt in refine_zones
            px, py = pt[1], pt[2]
            for i in 1:S_coarse
                refine_mask[i] && continue
                dx = c_ll[i][1] - px
                dy = c_ll[i][2] - py
                if (dx^2 + dy^2) <= (refine_buffer_km / 111.0)^2
                    refine_mask[i] = true
                end
            end
        end
    end

    # 3. Assemble multiresolution cells
    final_cents_km = Tuple{Float64, Float64}[]
    final_is_fine = Bool[]
    final_radius = Float64[]

    for i in 1:S_coarse
        if !refine_mask[i]
            push!(final_cents_km, c_km[i])
            push!(final_is_fine, false)
            push!(final_radius, rc)
        end
    end

    dx_f = sqrt(3.0) * rf
    dy_f = 1.5 * rf
    hex_bound = rc * 0.95

    for i in 1:S_coarse
        !refine_mask[i] && continue
        cx_c, cy_c = c_km[i]

        n_sub_x = ceil(Int, rc / dx_f)
        n_sub_y = ceil(Int, rc / dy_f)

        for ry in -n_sub_y:n_sub_y
            yk = cy_c + ry * dy_f
            xoff = isodd(abs(ry)) ? (dx_f / 2.0) : 0.0
            for rx in -n_sub_x:n_sub_x
                xk = cx_c + rx * dx_f + xoff
                dist_sq = (xk - cx_c)^2 + (yk - cy_c)^2
                if dist_sq <= hex_bound^2
                    too_close = false
                    for ex in final_cents_km
                        if ((xk - ex[1])^2 + (yk - ex[2])^2) < (0.6 * dx_f)^2
                            too_close = true
                            break
                        end
                    end
                    if !too_close
                        push!(final_cents_km, (xk, yk))
                        push!(final_is_fine, true)
                        push!(final_radius, rf)
                    end
                end
            end
        end
    end

    S = length(final_cents_km)
    c_lon = base_domain.center_lon
    c_lat = base_domain.center_lat

    final_cents_ll = [
        xy_km_to_lonlat(
            c[1], c[2];
            center_lon = c_lon, center_lat = c_lat, crs = crs, datum = datum
        )
        for c in final_cents_km
    ]

    hex_angles = (30.0 .+ 60.0 .* (0:5)) .* (π / 180.0)
    final_polys_km = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    final_polys_ll = Vector{Vector{Tuple{Float64, Float64}}}(undef, S)
    final_areas = zeros(Float64, S)

    for i in 1:S
        cx, cy = final_cents_km[i]
        rad = final_radius[i]
        final_areas[i] = (3.0 * sqrt(3.0) / 2.0) * (rad^2)
        v_km = [(cx + rad * cos(a), cy + rad * sin(a)) for a in hex_angles]
        push!(v_km, v_km[1])
        final_polys_km[i] = v_km
        final_polys_ll[i] = [
            xy_km_to_lonlat(
                p[1], p[2];
                center_lon = c_lon, center_lat = c_lat, crs = crs, datum = datum
            )
            for p in v_km
        ]
    end

    # 4. Cross-resolution spatial adjacency graph W
    c_mat = Matrix{Float64}(undef, 2, S)
    for i in 1:S
        c_mat[1, i] = final_cents_km[i][1]
        c_mat[2, i] = final_cents_km[i][2]
    end
    kdtree = KDTree(c_mat)

    rows_w = Int[]
    cols_w = Int[]

    for i in 1:S
        r_i = final_radius[i]
        is_f_i = final_is_fine[i]
        max_search = sqrt(3.0) * max(r_i, rc) * 1.15
        nbrs = inrange(kdtree, [c_mat[1, i], c_mat[2, i]], max_search)

        for j in nbrs
            j <= i && continue
            r_j = final_radius[j]
            is_f_j = final_is_fine[j]
            dx = c_mat[1, i] - c_mat[1, j]
            dy = c_mat[2, i] - c_mat[2, j]
            dist = sqrt(dx^2 + dy^2)

            thresh = if is_f_i && is_f_j
                sqrt(3.0) * rf * 1.08
            elseif !is_f_i && !is_f_j
                sqrt(3.0) * rc * 1.08
            else
                ((r_i + r_j) * 0.95)
            end

            if dist <= thresh
                push!(rows_w, i)
                push!(cols_w, j)
                push!(rows_w, j)
                push!(cols_w, i)
            end
        end
    end

    W = sparse(rows_w, cols_w, fill(1.0, length(rows_w)), S, S)

    # 5. Land mask for multiresolution units
    land_mask = identify_land_units(
        final_cents_ll;
        polygons = final_polys_ll,
        land_polygons = land_polygons,
        depth = depth,
        depth_threshold = depth_threshold
    )

    return (
        centroids_km     = final_cents_km,
        centroids_lonlat = final_cents_ll,
        polygons_km     = final_polys_km,
        polygons_lonlat = final_polys_ll,
        W                = W,
        areas_km2        = final_areas,
        is_fine          = final_is_fine,
        radius_km        = final_radius,
        center_lon       = c_lon,
        center_lat       = c_lat,
        n_units          = S,
        land_mask        = land_mask
    )
end

# =============================================================================
# Time-Varying Dynamic Environmental Transition Kernels
# =============================================================================

"""
    construct_dynamic_transition_kernels(
        W::SparseMatrixCSC,
        hsi_temporal;
        gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
        residence::Union{Real, AbstractVector{<:Real}} = 0.2,
        advection::Union{Real, AbstractVector{<:Real}} = 0.5,
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        num_steps::Union{Nothing, Integer} = nothing
    )::Vector{Matrix{Float64}}

Constructs a sequence of time-varying row-stochastic transition probability
matrices [P^{(1)}, ..., P^{(T)}] driven by temporally evolving habitat
suitability index (HSI) fields (e.g. seasonal temperature shifts or currents).

# Process Description
At each time step t in {1, ..., T}:
A^{(t)}_{ij} is proportional to W_{ij} * exp(gamma_t * (h_{t,j} - h_{t,i}))
P^{(t)}_{ij} = (1 - rho_t) * [alpha_t * A^{(t)}_{ij} +
    (1 - alpha_t) * T^{diff}_{ij}] + rho_t * delta_{ij}

# Arguments
- `W`: Sparse spatial adjacency matrix (S x S).
- `hsi_temporal`: Habitat suitability field over time:
  - `AbstractMatrix{<:Real}`: S x T matrix where column t is hsi at step t.
  - `AbstractVector{<:AbstractVector{<:Real}}`: Vector of T HSI vectors.
  - `Function`: Callable `(t::Int) -> Vector{Float64}`.
- `gamma`: Advection sensitivity parameter.
- `residence`: Residence probability rho in [0, 1).
- `advection`: Advection weight alpha in [0, 1].
- `land_mask`: Optional boolean vector denoting terrestrial units.
- `num_steps`: Number of steps when `hsi_temporal` is a function.

# Returns
`Vector{Matrix{Float64}}`: Vector of T dense S x S transition matrices.
"""
function construct_dynamic_transition_kernels(
    W::SparseMatrixCSC,
    hsi_temporal;
    gamma::Union{Real, AbstractVector{<:Real}} = 1.0,
    residence::Union{Real, AbstractVector{<:Real}} = 0.2,
    advection::Union{Real, AbstractVector{<:Real}} = 0.5,
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    num_steps::Union{Nothing, Integer} = nothing
)::Vector{Matrix{Float64}}
    S = size(W, 1)

    hsi_list = if hsi_temporal isa AbstractMatrix
        T_steps = size(hsi_temporal, 2)
        if size(hsi_temporal, 1) != S && size(hsi_temporal, 2) == S
            [Float64.(hsi_temporal[t, :]) for t in 1:size(hsi_temporal, 1)]
        else
            [Float64.(hsi_temporal[:, t]) for t in 1:T_steps]
        end
    elseif hsi_temporal isa AbstractVector{<:AbstractVector}
        [Float64.(v) for v in hsi_temporal]
    elseif isa(hsi_temporal, Function)
        T_steps = num_steps !== nothing ? Int(num_steps) : 12
        [Float64.(hsi_temporal(t)) for t in 1:T_steps]
    else
        throw(ArgumentError(
            "hsi_temporal must be an S x T Matrix, a Vector of Vectors, or a Function."
        ))
    end

    T = length(hsi_list)
    T > 0 || throw(ArgumentError("hsi_temporal must have at least one time step."))

    P_kernels = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        g_t = gamma isa AbstractVector ? gamma[min(t, length(gamma))] : gamma
        r_t = residence isa AbstractVector ? (
            residence[min(t, length(residence))]
        ) : residence
        a_t = advection isa AbstractVector ? (
            advection[min(t, length(advection))]
        ) : advection

        P_kernels[t] = construct_stochastic_transition_kernel(
            W, hsi_list[t];
            gamma = g_t,
            residence = r_t,
            advection = a_t,
            land_mask = land_mask
        )
    end

    return P_kernels
end

"""
    predict_dynamic_path(
        P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
        release::Int,
        recapture::Int;
        centroids = nothing,
        method::Symbol = :probabilistic,
        land_mask = nothing
    )::Vector{Int}

Reconstructs an individual animal movement trajectory through time-varying
transition kernels [P^{(1)}, ..., P^{(K)}].

# Process Description
At step tau in {1, ..., K}:
Next node is selected proportional to P^{(tau)}_{ij} * B_{tau}(j, u_rec),
where B_tau = prod_{t=tau+1}^K P^{(t)} represents target reachability.

# Arguments
- `P_kernels`: Vector of K transition matrices across consecutive time steps.
- `release`: Release node index (1 <= release <= S).
- `recapture`: Recapture node index (1 <= recapture <= S).
- `centroids`: Optional spatial node coordinates for reachability fallback.
- `method`: Simulation/optimization mode (`:viterbi`, `:deterministic`, or `:probabilistic`).
  `:viterbi` computes the global most probable trajectory via exact dynamic programming.
- `land_mask`: Optional land mask.

# Returns
`Vector{Int}`: Predicted path sequence of length K + 1.
"""
function predict_dynamic_path(
    P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int;
    centroids = nothing,
    method::Symbol = :viterbi,
    land_mask = nothing
)::Vector{Int}
    K = length(P_kernels)
    K >= 1 || throw(ArgumentError("P_kernels must contain >= 1 matrix."))
    S = size(P_kernels[1], 1)

    (1 <= release <= S && 1 <= recapture <= S) ||
        throw(ArgumentError(
            "Release ($release) or recapture ($recapture) out of range 1:$S"
        ))

    if release == recapture || K == 0
        return fill(release, K + 1)
    end

    if method == :viterbi
        # Exact non-stationary Viterbi dynamic programming across time-varying P_kernels
        delta = fill(-1e12, S, K + 1)
        psi = zeros(Int, S, K + 1)
        delta[release, 1] = 0.0

        for tau in 1:K
            P_tau = P_kernels[tau] isa SparseMatrixCSC ?
                P_kernels[tau] : SparseMatrixCSC(P_kernels[tau])
            Pt_tau = SparseMatrixCSC(P_tau')

            t_next = tau + 1
            @inbounds for j in 1:S
                if land_mask !== nothing && land_mask[j]
                    continue
                end

                best_val = -Inf
                best_prev = 1
                col_start = Pt_tau.colptr[j]
                col_end   = Pt_tau.colptr[j + 1] - 1

                for ptr in col_start:col_end
                    i = Pt_tau.rowval[ptr]
                    if land_mask !== nothing && land_mask[i]
                        continue
                    end
                    p_ij = Pt_tau.nzval[ptr]
                    if p_ij > 1e-15
                        score = delta[i, tau] + log(Float64(p_ij))
                        if score > best_val
                            best_val = score
                            best_prev = i
                        end
                    end
                end

                if best_val > -Inf
                    delta[j, t_next] = best_val
                    psi[j, t_next]   = best_prev
                end
            end
        end

        if delta[recapture, K + 1] > -1e11
            path = zeros(Int, K + 1)
            path[K + 1] = recapture
            for t_step in (K + 1):-1:2
                path[t_step - 1] = psi[path[t_step], t_step]
            end
            return path
        end
        # Fallback to bridge-conditioned selection if recapture unreachable at exact K
    end

    B_mats = Vector{Matrix{Float64}}(undef, K)
    B_mats[K] = Matrix{Float64}(I, S, S)
    for tau in (K - 1):-1:1
        B_mats[tau] = B_mats[tau + 1] * Matrix{Float64}(P_kernels[tau + 1])
    end

    path = Vector{Int}(undef, K + 1)
    path[1] = release
    curr = release

    for tau in 1:K
        P_curr = P_kernels[tau]
        B_next = B_mats[tau]

        weights = zeros(Float64, S)
        for j in 1:S
            weights[j] = Float64(P_curr[curr, j]) * Float64(B_next[j, recapture])
        end

        tot_w = sum(weights)
        if tot_w > 1e-14
            weights ./= tot_w
            if method == :deterministic
                curr = argmax(weights)
            else
                r = rand()
                cum_w = 0.0
                next_node = recapture
                for j in 1:S
                    cum_w += weights[j]
                    if r <= cum_w
                        next_node = j
                        break
                    end
                end
                curr = next_node
            end
        else
            if centroids !== nothing && length(centroids) == S
                best_nbr = curr
                best_d = Inf
                for j in 1:S
                    if Float64(P_curr[curr, j]) > 0.0
                        dx = centroids[j][1] - centroids[recapture][1]
                        dy = centroids[j][2] - centroids[recapture][2]
                        d = sqrt(dx^2 + dy^2)
                        if d < best_d
                            best_d = d
                            best_nbr = j
                        end
                    end
                end
                curr = best_nbr
            else
                curr = recapture
            end
        end
        path[tau + 1] = curr
    end

    path[end] = recapture
    return path
end

"""
    predict_dynamic_corridor(
        P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
        release::Int,
        recapture::Int;
        land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
    )::Matrix{Float64}

Computes the dynamic Markov bridge probability matrix across time-varying
transition kernels [P^{(1)}, ..., P^{(K)}].

# Process Description
Forward propagation: f_0 = e_{release}, f_tau = f_{tau-1} * P^{(tau)}
Backward propagation: b_K = e_{recapture}, b_tau = P^{(tau+1)} * b_{tau+1}
Visitation intensity:
C_j = (1 / (K + 1)) * sum_tau (f_tau(j) * b_tau(j)) / (f_tau . b_tau)

# Returns
`Matrix{Float64}`: Dense S x S normalized corridor intensity matrix.
"""
function predict_dynamic_corridor(
    P_kernels::AbstractVector{<:AbstractMatrix{<:Real}},
    release::Int,
    recapture::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Matrix{Float64}
    K = length(P_kernels)
    K >= 1 || throw(ArgumentError("P_kernels must contain >= 1 matrix."))
    S = size(P_kernels[1], 1)

    (1 <= release <= S && 1 <= recapture <= S) ||
        throw(ArgumentError(
            "Release ($release) or recapture ($recapture) out of range 1:$S"
        ))

    # Sparse per-step L1-normalized forward-backward propagation (exact Markov bridge
    # marginals without floating-point underflow on long multi-hundred-step horizons).
    P_sp = [P isa SparseMatrixCSC ? P : sparse(Float64.(P)) for P in P_kernels]
    f_vecs = Vector{Vector{Float64}}(undef, K + 1)
    f0 = zeros(Float64, S)
    f0[release] = 1.0
    f_vecs[1] = f0
    for tau in 1:K
        nxt = P_sp[tau]' * f_vecs[tau]
        s = sum(nxt)
        s > 0.0 && (nxt ./= s)
        f_vecs[tau + 1] = nxt
    end

    b_vecs = Vector{Vector{Float64}}(undef, K + 1)
    bK = zeros(Float64, S)
    bK[recapture] = 1.0
    b_vecs[K + 1] = bK
    for tau in K:-1:1
        prv = P_sp[tau] * b_vecs[tau + 1]
        s = sum(prv)
        s > 0.0 && (prv ./= s)
        b_vecs[tau] = prv
    end

    corridor = zeros(Float64, S, K + 1)
    for tau in 0:K
        col = tau + 1
        @inbounds for u in 1:S
            corridor[u, col] = (land_mask !== nothing && land_mask[u]) ?
                0.0 : f_vecs[col][u] * b_vecs[col][u]
        end
        csum = sum(@view(corridor[:, col]))
        if csum > 1e-30
            corridor[:, col] ./= csum
        else
            corridor[release, col] += 0.5
            corridor[recapture, col] += 0.5
        end
    end

    return corridor
end

# =============================================================================
# Multi-Segment Hidden Markov Model (HMM) Viterbi Smoothing
# =============================================================================

"""
    viterbi_hmm_path_smoothing(
        obs_times::AbstractVector{<:Integer},
        obs_locations::AbstractVector,
        P_kernels;
        mesh = nothing,
        sigma_obs_km::Real = 15.0,
        land_mask = nothing
    )::NamedTuple

Performs global trajectory decoding for animals with multi-stage recaptures,
acoustic detections, or irregular time intervals using a Hidden Markov Model
(HMM) with exact log-space Viterbi dynamic programming.

# Process Description
- Latent states: z_t in {1, ..., S} for t in 1:T.
- Transitions: P(z_t = j | z_{t-1} = i) = P^{(t)}_{ij}.
- Emissions: Gaussian spatial likelihood for observed coordinates/units;
  log b_j(missing) = 0.0 for unobserved intermediate time steps.
- Recursion: delta_t(j) = max_i [delta_{t-1}(i) + log P^{(t)}_{ij}] + log b_j(y_t).

# Arguments
- `obs_times`: Ascending vector of integer time steps [t_1, ..., t_M].
- `obs_locations`: Locations at each observation step (unit indices or (lon, lat)).
- `P_kernels`: Transition kernel matrix (S x S) or Vector of matrices.
- `mesh`: Spatial mesh NamedTuple containing `:centroids_lonlat` or `:centroids_km`.
- `sigma_obs_km`: Standard deviation of spatial observation error (km).
- `land_mask`: Optional boolean vector denoting terrestrial units.

# Returns
`NamedTuple` with fields:
- `path`: Decoded optimal spatial unit sequence z*_1, ..., z*_T.
- `log_likelihood`: Maximum path log-likelihood.
- `time_steps`: Vector 1:T.
- `obs_mask`: Boolean vector of length T indicating observed time steps.
"""
function viterbi_hmm_path_smoothing(
    obs_times::AbstractVector{<:Integer},
    obs_locations::AbstractVector,
    P_kernels;
    mesh = nothing,
    sigma_obs_km::Real = 15.0,
    land_mask = nothing
)::NamedTuple
    length(obs_times) == length(obs_locations) ||
        throw(ArgumentError("obs_times and obs_locations must have identical length."))
    issorted(obs_times) ||
        throw(ArgumentError("obs_times must be sorted in ascending order."))

    T = obs_times[end]
    T >= 1 || throw(ArgumentError("Maximum observation time must be >= 1."))

    S = if P_kernels isa AbstractMatrix
        size(P_kernels, 1)
    elseif P_kernels isa AbstractVector{<:AbstractMatrix}
        size(P_kernels[1], 1)
    else
        throw(ArgumentError("P_kernels must be a Matrix or Vector of Matrices."))
    end

    cents_ll = if mesh !== nothing
        if hasproperty(mesh, :centroids_lonlat)
            mesh.centroids_lonlat
        elseif hasproperty(mesh, :centroids)
            mesh.centroids
        else
            nothing
        end
    else
        nothing
    end

    log_B = zeros(Float64, S, T)
    obs_mask = falses(T)
    var_obs = Float64(sigma_obs_km)^2
    log_norm = -0.5 * log(2.0 * π * var_obs)

    for (idx, t) in enumerate(obs_times)
        obs_mask[t] = true
        loc = obs_locations[idx]

        if loc isa Integer
            target_u = Int(loc)
            for j in 1:S
                if j == target_u
                    log_B[j, t] = 0.0
                elseif cents_ll !== nothing
                    d_km = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        cents_ll[target_u][1], cents_ll[target_u][2]
                    ) / 1000.0
                    log_B[j, t] = -0.5 * (d_km^2) / var_obs + log_norm
                else
                    log_B[j, t] = -50.0
                end
            end
        elseif loc isa Tuple || loc isa AbstractVector
            lon_obs, lat_obs = Float64(loc[1]), Float64(loc[2])
            for j in 1:S
                if cents_ll !== nothing
                    d_km = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        lon_obs, lat_obs
                    ) / 1000.0
                    log_B[j, t] = -0.5 * (d_km^2) / var_obs + log_norm
                else
                    log_B[j, t] = 0.0
                end
            end
        end
    end

    if land_mask !== nothing
        for j in 1:S
            if land_mask[j]
                log_B[j, :] .= -1e9
            end
        end
    end

    log_P_fn = (step::Int) -> begin
        P_mat = if P_kernels isa AbstractMatrix
            P_kernels
        else
            P_kernels[min(step, length(P_kernels))]
        end
        log_mat = Matrix{Float64}(undef, S, S)
        for c in 1:S
            for r in 1:S
                val = Float64(P_mat[r, c])
                log_mat[r, c] = val > 1e-15 ? log(val) : -1e9
            end
        end
        return log_mat
    end

    delta = Matrix{Float64}(undef, S, T)
    psi = zeros(Int, S, T)

    init_prob = -log(S)
    for j in 1:S
        delta[j, 1] = init_prob + log_B[j, 1]
    end

    for t in 2:T
        log_P = log_P_fn(t - 1)
        for j in 1:S
            best_val = -Inf
            best_i = 1
            for i in 1:S
                cand = delta[i, t - 1] + log_P[i, j]
                if cand > best_val
                    best_val = cand
                    best_i = i
                end
            end
            delta[j, t] = best_val + log_B[j, t]
            psi[j, t] = best_i
        end
    end

    best_last = argmax(delta[:, T])
    best_ll = delta[best_last, T]

    path = Vector{Int}(undef, T)
    path[T] = best_last
    for t in (T - 1):-1:1
        path[t] = psi[path[t + 1], t + 1]
    end

    return (
        path           = path,
        log_likelihood = best_ll,
        time_steps     = collect(1:T),
        obs_mask       = obs_mask
    )
end

"""
    forward_backward_state_probabilities(
        obs_times::AbstractVector{<:Integer},
        obs_locations::AbstractVector,
        P_kernels;
        mesh = nothing,
        sigma_obs_km::Real = 15.0,
        land_mask = nothing
    )::Matrix{Float64}

Computes marginal posterior state probabilities gamma_t(i) = P(z_t = i | y_{1:T})
across all spatial mesh units and discrete time steps using the scaled
Baum-Welch forward-backward algorithm.

# Process Description
Forward: alpha_t(j) = (sum_i alpha_{t-1}(i) * P^{(t)}_{ij}) * b_j(y_t)
Backward: beta_t(i) = sum_j P^{(t+1)}_{ij} * b_j(y_{t+1}) * beta_{t+1}(j)
Posterior: gamma_t(j) = (alpha_t(j) * beta_t(j)) / sum_k (alpha_t(k) * beta_t(k))

# Returns
`Matrix{Float64}`: Dense S x T matrix where column t is the posterior
occupancy distribution across all spatial units at time step t.
"""
function forward_backward_state_probabilities(
    obs_times::AbstractVector{<:Integer},
    obs_locations::AbstractVector,
    P_kernels;
    mesh = nothing,
    sigma_obs_km::Real = 15.0,
    land_mask = nothing
)::Matrix{Float64}
    T = obs_times[end]
    S = P_kernels isa AbstractMatrix ? size(P_kernels, 1) : size(P_kernels[1], 1)

    cents_ll = if mesh !== nothing
        hasproperty(mesh, :centroids_lonlat) ? mesh.centroids_lonlat : (
            hasproperty(mesh, :centroids) ? mesh.centroids : nothing
        )
    else
        nothing
    end

    B = ones(Float64, S, T)
    var_obs = Float64(sigma_obs_km)^2

    for (idx, t) in enumerate(obs_times)
        loc = obs_locations[idx]
        if loc isa Integer
            u_t = Int(loc)
            for j in 1:S
                if j == u_t
                    B[j, t] = 1.0
                elseif cents_ll !== nothing
                    d = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        cents_ll[u_t][1], cents_ll[u_t][2]
                    ) / 1000.0
                    B[j, t] = exp(-0.5 * (d^2) / var_obs)
                else
                    B[j, t] = 1e-12
                end
            end
        elseif loc isa Tuple || loc isa AbstractVector
            lon_obs, lat_obs = Float64(loc[1]), Float64(loc[2])
            for j in 1:S
                if cents_ll !== nothing
                    d = haversine_distance(
                        cents_ll[j][1], cents_ll[j][2],
                        lon_obs, lat_obs
                    ) / 1000.0
                    B[j, t] = exp(-0.5 * (d^2) / var_obs)
                end
            end
        end
    end

    if land_mask !== nothing
        for j in 1:S
            if land_mask[j]
                B[j, :] .= 0.0
            end
        end
    end

    alpha = Matrix{Float64}(undef, S, T)
    c_scales = Vector{Float64}(undef, T)

    alpha[:, 1] .= (1.0 / S) .* B[:, 1]
    c_scales[1] = sum(alpha[:, 1])
    if c_scales[1] > 0.0
        alpha[:, 1] ./= c_scales[1]
    end

    for t in 2:T
        P_curr = P_kernels isa AbstractMatrix ? P_kernels : (
            P_kernels[min(t - 1, length(P_kernels))]
        )
        for j in 1:S
            s = 0.0
            for i in 1:S
                s += alpha[i, t - 1] * Float64(P_curr[i, j])
            end
            alpha[j, t] = s * B[j, t]
        end
        c_scales[t] = sum(alpha[:, t])
        if c_scales[t] > 0.0
            alpha[:, t] ./= c_scales[t]
        end
    end

    beta = Matrix{Float64}(undef, S, T)
    beta[:, T] .= 1.0

    for t in (T - 1):-1:1
        P_next = P_kernels isa AbstractMatrix ? P_kernels : (
            P_kernels[min(t, length(P_kernels))]
        )
        for i in 1:S
            s = 0.0
            for j in 1:S
                s += Float64(P_next[i, j]) * B[j, t + 1] * beta[j, t + 1]
            end
            beta[i, t] = s
        end
        if c_scales[t + 1] > 0.0
            beta[:, t] ./= c_scales[t + 1]
        end
    end

    gamma = Matrix{Float64}(undef, S, T)
    for t in 1:T
        gamma[:, t] = alpha[:, t] .* beta[:, t]
        tot = sum(gamma[:, t])
        if tot > 0.0
            gamma[:, t] ./= tot
        else
            gamma[:, t] .= 1.0 / S
        end
    end

    return gamma
end


"""
    generate_movement_data(; radius_km=8.0, time_interval=:monthly, crs=nothing,
                           datum=WGS84Latest, domain_km=60.0, n_tags=50, n_steps=3,
                           center_lon=-60.0, center_lat=46.0, seed=42) -> NamedTuple

Generates synthetic animal movement and mark-recapture telemetry datasets mapped over a
planar hexagonal spatial mesh. Simulates individual movement trajectories via discrete
Markov transitions across mesh units, assigns demographic attributes, and classifies
individuals into canonical biological groups matching `load_movement_data`:
- `"female"`: Mature females (`mat == "mature"`, `sex == "F"`)
- `"male"`: Mature males (`mat == "mature"`, `sex == "M"`)
- `"immature"`: Immature individuals (`mat == "immature"`)
- `"unknown"`: Unclassified or missing demographic observations

# Arguments
- `radius_km::Real = 8.0`: Hexagonal cell circumradius in kilometers.
- `time_interval::Symbol = :monthly`: Temporal step discretization interval
  (`:monthly`, `:weekly`, `:biweekly`, `:daily`, or `:raw`).
- `crs = nothing`: Coordinate reference system for spatial projections.
- `datum = WGS84Latest`: Geographic geodetic datum.
- `domain_km::Real = 60.0`: Spatial domain width and height in kilometers.
- `n_tags::Int = 50`: Total number of tagged individuals simulated.
- `n_steps::Int = 3`: Number of consecutive movement observation steps per individual.
- `center_lon::Real = -60.0`: Geographic center longitude of the simulated domain.
- `center_lat::Real = 46.0`: Geographic center latitude of the simulated domain.
- `seed::Int = 42`: Random seed for reproducible spatial and demographic simulation.

# Returns
A `NamedTuple` with fields:
- `tagging::DataFrame`: Raw synthetic telemetry event records with coordinates and units.
- `mesh::NamedTuple`: Planar hexagonal mesh containing centroids, polygons, and `W`.
- `W::SparseMatrixCSC`: Adjacency matrix of the spatial mesh.
- `hsi_vec::Vector{Float64}`: Domain Habitat Suitability Index (HSI) vector.
- `monthly_hsi::Matrix{Float64}`: Monthly dynamic HSI fields (empty if static).
- `month_lookup::Dict`: Mapping from `(year, month)` to monthly HSI column index.
- `years::Vector{Int}`: Observed survey years.
- `obs::DataFrame`: Extracted consecutive mark-recapture event pairs with biological
  group classifications (`:group` column with 1-based indices).
- `survey_df::DataFrame`: Synthetic spatial survey density observations.
- `group_lookup::Dict{String, Int}`: Dictionary mapping group names to integer IDs.
"""
function generate_movement_data(;
    radius_km     :: Real    = 8.0,
    time_interval :: Symbol  = :monthly,
    crs                      = nothing,
    datum                    = WGS84Latest,
    domain_km     :: Real    = 60.0,
    n_tags        :: Int     = 50,
    n_steps       :: Int     = 3,
    center_lon    :: Real    = -60.0,
    center_lat    :: Real    = 46.0,
    seed          :: Int     = 42
)::NamedTuple
    
    # Internal categorical draw — avoids importing Distributions in this file
    function _sample_categorical(p::AbstractVector{Float64}, rng::AbstractRNG)::Int
        u    = rand(rng)
        csum = 0.0
        for (i, pi) in enumerate(p)
            csum += pi
            csum >= u && return i
        end
        return length(p)
    end

    rng  = MersenneTwister(seed)
    half = Float64(domain_km) / 2.0

    n_grid = max(10, round(Int, domain_km / radius_km * 2))
    xs_g   = range(-half, half, length=n_grid)
    ys_g   = range(-half, half, length=n_grid)
    
    # Pre-allocate grid coordinate vectors
    n_pts = length(xs_g) * length(ys_g)
    grid_lon = Vector{Float64}(undef, n_pts)
    grid_lat = Vector{Float64}(undef, n_pts)
    
    idx = 1
    for y in ys_g, x in xs_g
        lon, lat = xy_km_to_lonlat(x, y; 
                        center_lon=center_lon, center_lat=center_lat, 
                        crs=crs, datum=datum)
        grid_lon[idx] = lon
        grid_lat[idx] = lat
        idx += 1
    end

    # Pass the CRS formatting down to the mesh generator
    mesh = build_hex_mesh(grid_lon, grid_lat; 
               radius_km=radius_km, crs=crs, datum=datum)
    S = mesh.n_units

    # Construct the true kernel directly using row sums of the sparse matrix
    row_sums = sum(mesh.W, dims=2)
    kernel   = zeros(Float64, S, S)
    for i in 1:S
        rs = row_sums[i]
        if rs > 0
            @views kernel[i, :] .= mesh.W[i, :] ./ rs
        else
            kernel[i, i] = 1.0
        end
    end

    # Biological demographic simulation covering female, male, immature, and unknown
    # Guarantee the 4 canonical groups appear in simulated tagging dataset
    canonical_demographics = [
        ("F", "mature"),        # -> female
        ("M", "mature"),        # -> male
        ("M", "immature"),      # -> immature
        ("unknown", "unknown")  # -> unknown
    ]
    pool_demographics = [
        ("F", "mature"),
        ("M", "mature"),
        ("M", "immature"),
        ("F", "immature"),
        ("unknown", "mature"),
        ("unknown", "unknown")
    ]
    pool_weights = [0.35, 0.35, 0.12, 0.08, 0.05, 0.05]

    sexes = Vector{String}(undef, n_tags)
    mats  = Vector{String}(undef, n_tags)
    for i in 1:n_tags
        if i <= length(canonical_demographics)
            sx, mt = canonical_demographics[i]
        else
            w_idx = _sample_categorical(pool_weights, rng)
            sx, mt = pool_demographics[w_idx]
        end
        sexes[i] = sx
        mats[i]  = mt
    end

    t0_dt = Date(2020, 1, 1)
    
    total_records = n_tags * (n_steps + 1)
    records = Vector{NamedTuple{
        (:tagid, :lon, :lat, :tag, :timestamp, :time, :sex, :mat, :is_dead, :s_idx),
        Tuple{String, Float64, Float64, Int, DateTime, Float64, String, String, Bool, Int}
    }}(undef, total_records)
    
    row_idx = 1
    for i in 1:n_tags
        s_cur   = rand(rng, 1:S)
        tid_str = string(i)
        
        (lon_r, lat_r) = mesh.centroids_lonlat[s_cur]
        records[row_idx] = (
            tagid     = tid_str,
            lon       = lon_r,
            lat       = lat_r,
            tag       = 0,
            timestamp = DateTime(t0_dt),
            time      = _date_to_decimal_year(t0_dt),
            sex       = sexes[i],
            mat       = mats[i],
            is_dead   = false,
            s_idx     = s_cur
        )
        row_idx += 1

        for step in 1:n_steps
            p_row = kernel[s_cur, :]
            s_cur = _sample_categorical(p_row, rng)
            t_dt  = t0_dt + Month(step)
            
            (lon_r, lat_r) = mesh.centroids_lonlat[s_cur]
            records[row_idx] = (
                tagid     = tid_str,
                lon       = lon_r,
                lat       = lat_r,
                tag       = step,
                timestamp = DateTime(t_dt),
                time      = _date_to_decimal_year(t_dt),
                sex       = sexes[i],
                mat       = mats[i],
                is_dead   = false,
                s_idx     = s_cur
            )
            row_idx += 1
        end
    end
  
    tagging = DataFrame(records)
    true_kernel = kernel
    tagging = map_point_to_units(tagging, mesh.centroids_km,
                  mesh.center_lon, mesh.center_lat; crs=crs, datum=datum)
    
    # Generate realistic spatial bathymetry and habitat suitability gradient
    depth_vec = [150.0 + 60.0 * sin(mesh.centroids_km[s][1] / 40.0) + 
                 40.0 * cos(mesh.centroids_km[s][2] / 40.0) for s in 1:mesh.n_units]
    temp_vec  = [3.0 + 1.5 * cos(mesh.centroids_km[s][1] / 50.0) for s in 1:mesh.n_units]
    
    # HSI peaks in optimal thermal/depth window
    hsi_raw = [exp(-((depth_vec[s] - 170.0) / 45.0)^2 - ((temp_vec[s] - 2.5) / 1.5)^2) 
               for s in 1:mesh.n_units]
    hsi_min, hsi_max = extrema(hsi_raw)
    hsi_vec = (hsi_raw .- hsi_min) ./ max(1e-6, hsi_max - hsi_min) .* 0.8 .+ 0.1

    monthly_hsi  = Matrix{Float64}(undef, 0, 0)
    month_lookup = Dict{Tuple{Int, Int}, Int}()
    years_vec    = Int[]

    # Time-interval step conversion mapping
    dt_map = (monthly=1.0/12.0, weekly=1.0/52.0, biweekly=1.0/26.0, daily=1.0/365.25, raw=1.0)
    dt = hasproperty(dt_map, time_interval) ? getproperty(dt_map, time_interval) : 1.0 / 12.0

    has_sex = hasproperty(tagging, :sex)
    has_mat = hasproperty(tagging, :mat)

    # 1. Sort globally upfront
    sorted_df = sort(tagging, [:tagid, :time])
    n_rows = nrow(sorted_df)

    # Return empty DataFrame immediately if not enough rows to form a pair
    if n_rows < 2
        obs = DataFrame(tagid=String[], release=Int[], recapture=Int[], 
                        k=Int[], sex=String[], mat=String[], group=Int[])
        survey_df = DataFrame(s_idx=Int[], t_idx=Int[], density=Int[], depth=Float64[], temp=Float64[])
        default_group_lookup = Dict{String, Int}(
            "female"   => 1,
            "immature" => 2,
            "male"     => 3,
            "unknown"  => 4
        )
        return (
            tagging      = tagging,
            mesh         = mesh,
            W            = mesh.W,
            hsi_vec      = hsi_vec,
            monthly_hsi  = monthly_hsi,
            month_lookup = month_lookup,
            years        = years_vec,
            obs          = obs,
            survey_df    = survey_df,
            group_lookup = default_group_lookup
        )
    end

    # 2. Extract columns to local vectors for type stability
    tagids    = sorted_df.tagid
    times     = sorted_df.time
    s_idxs    = sorted_df.s_idx
    lons      = sorted_df.lon
    lats      = sorted_df.lat
    sexes_col = has_sex ? sorted_df.sex : nothing
    mats_col  = has_mat ? sorted_df.mat : nothing

    RecordType = NamedTuple{
        (:tagid, :release, :recapture, :k, :rel_time, :sex, :mat, :lon, :lat),
        Tuple{String, Int, Int, Int, Float64, String, String, Float64, Float64}
    }
    
    pair_records = Vector{RecordType}(undef, 0)
    sizehint!(pair_records, n_rows)

    # 3. Single-pass flat loop for consecutive pairs
    for i in 2:n_rows
        if tagids[i] == tagids[i-1]
            Δt = times[i] - times[i-1]
            k  = max(1, round(Int, Δt / dt))
            
            s_str = has_sex ? string(sexes_col[i-1]) : "unknown"
            m_str = has_mat ? string(mats_col[i-1])  : "unknown"

            push!(pair_records, (
                tagid     = string(tagids[i-1]),
                release   = s_idxs[i-1],
                recapture = s_idxs[i],
                k         = k,
                rel_time  = Float64(times[i-1]),
                sex       = s_str,
                mat       = m_str,
                lon       = Float64(lons[i-1]),
                lat       = Float64(lats[i-1])
            ))
        end
    end

    obs = DataFrame(pair_records)

    # 4. Assign 3-tier biological groupings matching load_movement_data()
    n_obs = nrow(obs)
    labels = Vector{String}(undef, n_obs)
    
    if n_obs > 0
        obs_sexes = obs[!, :sex]
        obs_mats  = obs[!, :mat]

        @inbounds for i in 1:n_obs
            sx = string(obs_sexes[i])
            mt = string(obs_mats[i])

            if mt == "immature" || mt == "imm"
                labels[i] = "immature"
            elseif (mt == "mature" || mt == "mat") && (sx == "M" || sx == "male")
                labels[i] = "male"
            elseif (mt == "mature" || mt == "mat") && (sx == "F" || sx == "female")
                labels[i] = "female"
            else
                labels[i] = "unknown"
            end
        end
    end
    
    unique_labels = sort!(unique(labels))
    group_lookup  = Dict{String, Int}(lbl => i for (i, lbl) in enumerate(unique_labels))
        
    group_ids = Vector{Int}(undef, n_obs)
    @inbounds for i in 1:n_obs
        group_ids[i] = group_lookup[labels[i]]
    end

    obs[!, :group] = group_ids

    # 5. Generate synthetic survey density observations for Option 3 joint modeling
    mu_density = exp.(1.5 .+ 2.0 .* hsi_vec .- 0.005 .* (depth_vec .- 170.0))
    density_counts = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + mu_density[s]))) 
                      for s in 1:mesh.n_units]
    survey_df = DataFrame(
        s_idx   = collect(1:mesh.n_units),
        t_idx   = ones(Int, mesh.n_units),
        density = density_counts,
        depth   = depth_vec,
        temp    = temp_vec
    )
  
    return (
        tagging      = tagging,
        mesh         = mesh,
        W            = mesh.W,
        hsi_vec      = hsi_vec,
        monthly_hsi  = monthly_hsi,
        month_lookup = month_lookup,
        years        = years_vec,
        obs          = obs,
        survey_df    = survey_df,
        depth_vec    = depth_vec,
        group_lookup = group_lookup
    )
end

"""
    simulate_forward_ibm(P, start_node, steps; land_mask)

Simulate an unconstrained forward Individual-Based Model (IBM) random walk
on the transition kernel `P`.
"""
function simulate_forward_ibm(
    P::AbstractMatrix{<:Real},
    start_node::Int,
    steps::Int;
    land_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)::Vector{Int}
    path = Int[start_node]
    node = start_node
    for _ in 1:steps
        row = copy(vec(P[node, :]))
        land_mask !== nothing && (row[land_mask] .= 0.0)
        rs = sum(row)
        rs <= 0.0 && break
        row ./= rs
        node = rand(Categorical(row))
        push!(path, node)
    end
    return path
end

const LIBERTY_BIN_SPECS = (
    (idx = 1, label = "< 1 month",   color = "#38bdf8", min_days = 0.0,         max_days = 30.4375),
    (idx = 2, label = "1 mo - 1 yr", color = "#10b981", min_days = 30.4375,     max_days = 365.25),
    (idx = 3, label = "1 - 2 yr",    color = "#eab308", min_days = 365.25,      max_days = 730.50),
    (idx = 4, label = "2 - 3 yr",    color = "#f97316", min_days = 730.50,      max_days = 1095.75),
    (idx = 5, label = "3 - 4 yr",    color = "#ef4444", min_days = 1095.75,     max_days = 1461.00),
    (idx = 6, label = ">= 4 yr",     color = "#a855f7", min_days = 1461.00,     max_days = Inf),
)

"""
    classify_liberty_interval(days::Real) -> NamedTuple

Classify elapsed mark-recapture time at liberty (`days >= 0`) into six mutually
exclusive intervals (`< 1 month`, `1 mo - 1 yr`, `1 - 2 yr`, `2 - 3 yr`, `3 - 4 yr`,
`>= 4 yr`). Raises a `DomainError` if `days` is negative or non-finite.

Inputs:
- `days`: Elapsed duration between release and recapture in days (`>= 0`).

Outputs:
NamedTuple `(bin, label, color)` with integer bin index in `1:6`, interval label,
and hex colour string.
"""
function classify_liberty_interval(days::Real)
    d = Float64(days)
    (isfinite(d) && d >= 0.0) || throw(DomainError(
        days, "Time at liberty (days) must be finite and non-negative."
    ))
    for spec in LIBERTY_BIN_SPECS
        if d < spec.max_days
            return (bin = spec.idx, label = spec.label, color = spec.color)
        end
    end
    last_spec = LIBERTY_BIN_SPECS[end]
    return (bin = last_spec.idx, label = last_spec.label, color = last_spec.color)
end

function _catmull_rom_centripetal_pt(
    p0::Tuple{Float64, Float64},
    p1::Tuple{Float64, Float64},
    p2::Tuple{Float64, Float64},
    p3::Tuple{Float64, Float64},
    tau::Float64
)::Tuple{Float64, Float64}
    dt01 = max(1e-5, sqrt(hypot(p1[1] - p0[1], p1[2] - p0[2])))
    dt12 = max(1e-5, sqrt(hypot(p2[1] - p1[1], p2[2] - p1[2])))
    dt23 = max(1e-5, sqrt(hypot(p3[1] - p2[1], p3[2] - p2[2])))
    t0 = 0.0
    t1 = dt01
    t2 = t1 + dt12
    t3 = t2 + dt23
    t = t1 + tau * dt12

    a1x = ((t1 - t) * p0[1] + (t - t0) * p1[1]) / (t1 - t0)
    a1y = ((t1 - t) * p0[2] + (t - t0) * p1[2]) / (t1 - t0)
    a2x = ((t2 - t) * p1[1] + (t - t1) * p2[1]) / (t2 - t1)
    a2y = ((t2 - t) * p1[2] + (t - t1) * p2[2]) / (t2 - t1)
    a3x = ((t3 - t) * p2[1] + (t - t2) * p3[1]) / (t3 - t2)
    a3y = ((t3 - t) * p2[2] + (t - t2) * p3[2]) / (t3 - t2)

    b1x = ((t2 - t) * a1x + (t - t0) * a2x) / (t2 - t0)
    b1y = ((t2 - t) * a1y + (t - t0) * a2y) / (t2 - t0)
    b2x = ((t3 - t) * a2x + (t - t1) * a3x) / (t3 - t1)
    b2y = ((t3 - t) * a2y + (t - t1) * a3y) / (t3 - t1)

    cx = ((t2 - t) * b1x + (t - t1) * b2x) / (t2 - t1)
    cy = ((t2 - t) * b1y + (t - t1) * b2y) / (t2 - t1)
    return (cx, cy)
end

"""
    smooth_marine_polyline(coords; n_knots=5, samples_per_span=6, passes=3, check_raster=true, resolution="h", grid_minutes=1.25)

Construct a loop-free, synoptic centripetal Catmull-Rom cubic spline trajectory from a
discrete marine waypoint sequence `[(lon, lat), ...]`:

1. Erases self-intersecting loops and backtrack cycles (`p_i -> ... -> p_i`).
2. Relaxes hexagonal staircase jitter via `passes` iterations of `[0.25, 0.50, 0.25]`
   interior averaging guarded against non-ocean pixels.
3. Selects `n_knots` (default `5`) evenly spaced arc-length control points, subdividing
   any control chord that intersects land or inland lakes until all chords are oceanic.
4. Evaluates a centripetal (`alpha = 0.5`) Catmull-Rom spline (`samples_per_span` points
   per span), damping toward the verified marine chord wherever coastal overshoot would
   graze non-ocean pixels.

Inputs:
- `coords`: Sequence of `(lon, lat)` or planar `(x, y)` coordinate tuples.
- `n_knots`: Target number of arc-length spline control knots (`5` default, `>= 2`).
- `samples_per_span`: Interpolated curve segments per knot span (`6` default, `>= 2`).
- `passes`: Pre-spline moving-average relaxation passes (`3` default).
- `check_raster`: Verify ocean clearance on the GMT/basemap raster (`true` default).
- `resolution`: Raster coastline tier (`"h"` default).
- `grid_minutes`: Raster cell spacing in arc-minutes (`1.25` default).

Outputs:
`Vector{Tuple{Float64, Float64}}` of loop-free spline coordinates preserving exact endpoints.
"""
function smooth_marine_polyline(
    coords::AbstractVector;
    n_knots::Int = 5,
    samples_per_span::Int = 6,
    passes::Int = 3,
    check_raster::Bool = true,
    resolution::AbstractString = "h",
    grid_minutes::Real = 1.25
)::Vector{Tuple{Float64, Float64}}
    raw = Tuple{Float64, Float64}[
        (round(Float64(p[1]); digits = 5), round(Float64(p[2]); digits = 5))
        for p in coords
    ]
    length(raw) <= 2 && return raw
    if raw[1] == raw[end]
        return Tuple{Float64, Float64}[raw[1], raw[end]]
    end

    # 1. Erase vertex loops and backtracks (A -> ... -> A)
    cur = Tuple{Float64, Float64}[]
    seen = Dict{Tuple{Float64, Float64}, Int}()
    for pt in raw
        prev_idx = get(seen, pt, 0)
        if prev_idx > 0
            for k_del in (prev_idx + 1):length(cur)
                delete!(seen, cur[k_del])
            end
            resize!(cur, prev_idx)
        else
            push!(cur, pt)
            seen[pt] = length(cur)
        end
    end
    (length(cur) < 3 || passes <= 0) && return cur

    lon_m, lat_m, data_m, lon0, dlon, lat0, dlat = if check_raster
        lo, la, dm = _basemap_mask_data(resolution, grid_minutes)
        (lo, la, dm,
         first(lo), (last(lo) - first(lo)) / (length(lo) - 1),
         first(la), (last(la) - first(la)) / (length(la) - 1))
    else
        (Float64[], Float64[], zeros(UInt8, 0, 0), 0.0, 1.0, 0.0, 1.0)
    end

    seg_ocean(a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64}) = begin
        !check_raster && return true
        _is_raster_non_ocean(lon_m, lat_m, data_m, lon0, dlon, lat0, dlat, a[1], a[2]) && return false
        _is_raster_non_ocean(lon_m, lat_m, data_m, lon0, dlon, lat0, dlat, b[1], b[2]) && return false
        dx = b[1] - a[1]
        dy = b[2] - a[2]
        for s in 0.05:0.05:0.95
            if _is_raster_non_ocean(lon_m, lat_m, data_m, lon0, dlon, lat0, dlat,
                                    a[1] + s * dx, a[2] + s * dy)
                return false
            end
        end
        return true
    end

    # 2. Relax discrete hexagonal zig-zags while preserving marine clearance
    rlx = copy(cur)
    m_r = length(rlx)
    for _ in 1:passes
        nxt = copy(rlx)
        for j in 2:(m_r - 1)
            cand = (
                round(0.25 * rlx[j - 1][1] + 0.50 * rlx[j][1] + 0.25 * rlx[j + 1][1]; digits = 5),
                round(0.25 * rlx[j - 1][2] + 0.50 * rlx[j][2] + 0.25 * rlx[j + 1][2]; digits = 5),
            )
            if seg_ocean(nxt[j - 1], cand) && seg_ocean(cand, rlx[j + 1])
                nxt[j] = cand
            end
        end
        rlx = nxt
    end

    # 3. Select n_knots evenly spaced by cumulative arc length, subdividing only if
    #    a control chord crosses land or an inland lake.
    cum_s = zeros(Float64, m_r)
    for j in 2:m_r
        cum_s[j] = cum_s[j - 1] + hypot(rlx[j][1] - rlx[j - 1][1], rlx[j][2] - rlx[j - 1][2])
    end
    cum_s[end] <= 1e-8 && return Tuple{Float64, Float64}[rlx[1], rlx[end]]

    k_target = min(max(2, n_knots), m_r)
    knot_idx = Int[1]
    if k_target > 2
        for q in 1:(k_target - 2)
            s_q = (q / (k_target - 1)) * cum_s[end]
            idx_q = clamp(searchsortedlast(cum_s, s_q), 2, m_r - 1)
            idx_q > knot_idx[end] && push!(knot_idx, idx_q)
        end
    end
    push!(knot_idx, m_r)

    r_ptr = 1
    while r_ptr < length(knot_idx)
        i_a = knot_idx[r_ptr]
        i_b = knot_idx[r_ptr + 1]
        if i_b > i_a + 1 && !seg_ocean(rlx[i_a], rlx[i_b])
            insert!(knot_idx, r_ptr + 1, div(i_a + i_b, 2))
        else
            r_ptr += 1
        end
    end

    knots = rlx[knot_idx]
    n_k = length(knots)
    n_k <= 2 && return knots

    # 4. Evaluate centripetal Catmull-Rom cubic spline across knots
    n_sub = max(2, samples_per_span)
    p_first = (2.0 * knots[1][1] - knots[2][1], 2.0 * knots[1][2] - knots[2][2])
    p_last  = (2.0 * knots[n_k][1] - knots[n_k - 1][1], 2.0 * knots[n_k][2] - knots[n_k - 1][2])

    out = Tuple{Float64, Float64}[knots[1]]
    span_pts = Vector{Tuple{Float64, Float64}}(undef, n_sub)

    for j in 1:(n_k - 1)
        p0 = j == 1 ? p_first : knots[j - 1]
        p1 = knots[j]
        p2 = knots[j + 1]
        p3 = j + 1 == n_k ? p_last : knots[j + 2]

        accepted = false
        for blend in (1.0, 0.5)
            ok = true
            prev_p = out[end]
            for s_i in 1:n_sub
                tau = s_i / n_sub
                sp_p = s_i == n_sub ? p2 : _catmull_rom_centripetal_pt(p0, p1, p2, p3, tau)
                lin_x = (1.0 - tau) * p1[1] + tau * p2[1]
                lin_y = (1.0 - tau) * p1[2] + tau * p2[2]
                cand_p = s_i == n_sub ? p2 : (
                    round(blend * sp_p[1] + (1.0 - blend) * lin_x; digits = 5),
                    round(blend * sp_p[2] + (1.0 - blend) * lin_y; digits = 5),
                )
                if !seg_ocean(prev_p, cand_p)
                    ok = false
                    break
                end
                span_pts[s_i] = cand_p
                prev_p = cand_p
            end
            if ok
                append!(out, span_pts)
                accepted = true
                break
            end
        end
        if !accepted
            push!(out, p2)
        end
    end
    return out
end

function _reverse_bfs_hops(
    P_sp::SparseMatrixCSC{Float64, Int},
    target::Int,
    land_mask::Union{Nothing, AbstractVector{Bool}},
    p_min::Float64 = 1e-15
)::Tuple{Vector{Int}, Vector{Int}}
    S = size(P_sp, 1)
    # Column `v` of `P_sp` stores all `u` with `P_sp[u, v] > 0` (incoming marine steps).
    dist = fill(typemax(Int), S)
    next_hop = zeros(Int, S)
    (land_mask !== nothing && land_mask[target]) && return (dist, next_hop)
    dist[target] = 0
    next_hop[target] = target
    queue = Int[target]
    head = 1
    rows = rowvals(P_sp)
    vals = nonzeros(P_sp)
    while head <= length(queue)
        v = queue[head]
        head += 1
        d_next = dist[v] + 1
        for ptr in nzrange(P_sp, v)
            vals[ptr] > p_min || continue
            u = rows[ptr]
            (land_mask !== nothing && land_mask[u]) && continue
            if dist[u] == typemax(Int)
                dist[u] = d_next
                next_hop[u] = v
                push!(queue, u)
            end
        end
    end
    return (dist, next_hop)
end

"""
    reconstruct_all_event_trajectories(loaded, kernels; kwargs...) -> Vector{NamedTuple}

Reconstruct and locally smooth a conditioned Markov-bridge trajectory for every
mark-recapture event row in `loaded.obs_df` by sharing L1-normalized backward
reachability cascades `b_{r+1} = P * b_r / ||P * b_r||_1` across events terminating at
each unique recapture unit.

Inputs:
- `loaded`: Domain NamedTuple (`obs_df`, `mesh`, `land_mask`).
- `kernels`: Transition kernel NamedTuple (`P_kernel`).
- `step_years`: Duration of one kernel step in years (`1 / 365.25` default).
- `seed`: Deterministic RNG seed for per-event bridge draws.
- `smooth_passes`: Chaikin B-spline subdivision passes (`2` default).
- `check_raster`: Enforce ocean-raster validation (`nothing` auto-detects lon/lat space).
- `resolution`, `grid_minutes`: Global land/lake mask parameters (`"h"`, `1.25`).

Outputs:
`Vector{NamedTuple}` of length `nrow(loaded.obs_df)` containing `event_id`, `tagid`,
`datasource`, `rel_time_str`, `liberty_days`, `liberty_bin`, `liberty_label`, `color`,
`displacement_km`, `path_length_km`, and smoothed `coords`.
"""
function reconstruct_all_event_trajectories(
    loaded::NamedTuple,
    kernels::NamedTuple;
    step_years::Real = 1.0 / 365.25,
    seed::Int = 42,
    n_knots::Int = 5,
    smooth_passes::Int = 3,
    check_raster::Union{Nothing, Bool} = nothing,
    resolution::AbstractString = "h",
    grid_minutes::Real = 1.25
)
    obs_df = loaded.obs_df
    nrow(obs_df) == 0 && return NamedTuple[]

    mesh = loaded.mesh
    cents = if hasproperty(mesh, :centroids_lonlat) && !isempty(mesh.centroids_lonlat)
        mesh.centroids_lonlat
    elseif hasproperty(mesh, :centroids) && !isempty(mesh.centroids)
        mesh.centroids
    else
        mesh.centroids_km
    end
    S = length(cents)
    is_geo = _infer_coord_space(cents)
    do_raster = check_raster === nothing ? is_geo : Bool(check_raster)
    land_mask = hasproperty(loaded, :land_mask) ? loaded.land_mask : nothing

    P_sp = kernels.P_kernel isa SparseMatrixCSC ?
        kernels.P_kernel : sparse(Float64.(kernels.P_kernel))
    Pt = SparseMatrixCSC(P_sp')
    rows_t = rowvals(Pt)
    vals_t = nonzeros(Pt)

    step_days = max(1e-3, Float64(step_years) * 365.25)
    n_ev = nrow(obs_df)

    # Precompute elapsed days and group event indices by recapture node for shared
    # backward reachability cascades.
    ev_days = Vector{Float64}(undef, n_ev)
    by_rec = Dict{Int, Vector{Int}}()
    has_tpair = hasproperty(obs_df, :rel_time) && hasproperty(obs_df, :rec_time)
    has_dur = hasproperty(obs_df, :duration_days)

    for i in 1:n_ev
        d_i = if has_tpair && !ismissing(obs_df.rel_time[i]) && !ismissing(obs_df.rec_time[i])
            t0 = Float64(obs_df.rel_time[i])
            t1 = Float64(obs_df.rec_time[i])
            (isfinite(t0) && isfinite(t1) && t1 >= t0) ?
                max(0.5, (t1 - t0) * 365.25) : max(1.0, Float64(obs_df.k[i]) * step_days)
        elseif has_dur && !ismissing(obs_df.duration_days[i]) && isfinite(Float64(obs_df.duration_days[i]))
            max(0.5, Float64(obs_df.duration_days[i]))
        else
            max(1.0, Float64(obs_df.k[i]) * step_days)
        end
        ev_days[i] = d_i
        u_rec = Int(obs_df.recapture[i])
        push!(get!(by_rec, u_rec, Int[]), i)
    end

    step_dist_km(a, b) = is_geo ?
        haversine_distance(Float64(a[1]), Float64(a[2]), Float64(b[1]), Float64(b[2])) / 1000.0 :
        hypot(Float64(b[1]) - Float64(a[1]), Float64(b[2]) - Float64(a[2]))

    results = Vector{NamedTuple}(undef, n_ev)
    nbr_nodes = Int[]
    nbr_weights = Float64[]

    for (u_rec, idxs) in by_rec
        dist_hops, next_hop = _reverse_bfs_hops(P_sp, u_rec, land_mask)
        ev_K = zeros(Int, length(idxs))
        K_max = 0
        for (loc_j, i) in enumerate(idxs)
            u_rel = Int(obs_df.release[i])
            raw_k = max(1, round(Int, ev_days[i] / step_days))
            if u_rel == u_rec
                ev_K[loc_j] = 0
            elseif 1 <= u_rel <= S && dist_hops[u_rel] < typemax(Int)
                mh = dist_hops[u_rel]
                K_i = clamp(max(mh, min(raw_k, round(Int, 1.35 * mh) + 2)), max(2, mh), 110)
                ev_K[loc_j] = K_i
                K_max = max(K_max, K_i)
            end
        end

        b_rem = Vector{Vector{Float64}}(undef, K_max + 1)
        b0 = zeros(Float64, S)
        (1 <= u_rec <= S) && (b0[u_rec] = 1.0)
        b_rem[1] = b0
        for r in 1:K_max
            nxt = P_sp * b_rem[r]
            if land_mask !== nothing
                @inbounds for u in 1:S
                    land_mask[u] && (nxt[u] = 0.0)
                end
            end
            s = sum(nxt)
            s > 0.0 && (nxt ./= s)
            b_rem[r + 1] = nxt
        end

        for (loc_j, i) in enumerate(idxs)
            u_rel = Int(obs_df.release[i])
            K_i = ev_K[loc_j]
            rng_i = MersenneTwister(hash((i, string(obs_df.tagid[i]), seed)))

            seq = if K_i >= 2 && 1 <= u_rel <= S && b_rem[K_i + 1][u_rel] > 0.0
                p_nodes = Vector{Int}(undef, K_i + 1)
                p_nodes[1] = u_rel
                curr = u_rel
                ok = true
                for tau in 1:K_i
                    empty!(nbr_nodes)
                    empty!(nbr_weights)
                    b_next = b_rem[K_i - tau + 1]
                    tot_w = 0.0
                    for ptr in nzrange(Pt, curr)
                        p_uv = Float64(vals_t[ptr])
                        p_uv > 1e-15 || continue
                        v = rows_t[ptr]
                        (land_mask !== nothing && land_mask[v]) && continue
                        w = p_uv * b_next[v]
                        if w > 0.0
                            push!(nbr_nodes, v)
                            push!(nbr_weights, w)
                            tot_w += w
                        end
                    end
                    if tot_w <= 0.0 || isempty(nbr_nodes)
                        ok = false
                        break
                    end
                    u_draw = rand(rng_i) * tot_w
                    csum = 0.0
                    chosen = nbr_nodes[end]
                    for ni in eachindex(nbr_nodes)
                        csum += nbr_weights[ni]
                        if u_draw <= csum
                            chosen = nbr_nodes[ni]
                            break
                        end
                    end
                    curr = chosen
                    p_nodes[tau + 1] = curr
                end
                if ok
                    p_nodes[end] = u_rec
                    p_nodes
                else
                    # Reconstruct shortest marine hop chain via next_hop
                    fb = Int[u_rel]
                    c_u = u_rel
                    while c_u != u_rec && 1 <= next_hop[c_u] <= S && length(fb) <= S
                        c_u = next_hop[c_u]
                        push!(fb, c_u)
                    end
                    fb
                end
            elseif u_rel != u_rec && 1 <= u_rel <= S && dist_hops[u_rel] < typemax(Int)
                fb = Int[u_rel]
                c_u = u_rel
                while c_u != u_rec && 1 <= next_hop[c_u] <= S && length(fb) <= S
                    c_u = next_hop[c_u]
                    push!(fb, c_u)
                end
                fb
            else
                Int[clamp(u_rel, 1, S)]
            end

            # Lawler loop erasure removes random-walk cycles and backtracks
            loop_free = Int[]
            seen_u = Dict{Int, Int}()
            for u_s in seq
                prev_pos = get(seen_u, u_s, 0)
                if prev_pos > 0
                    for k_del in (prev_pos + 1):length(loop_free)
                        delete!(seen_u, loop_free[k_del])
                    end
                    resize!(loop_free, prev_pos)
                else
                    push!(loop_free, u_s)
                    seen_u[u_s] = length(loop_free)
                end
            end

            raw_coords = Tuple{Float64, Float64}[
                (round(Float64(cents[u][1]); digits = 5),
                 round(Float64(cents[u][2]); digits = 5))
                for u in loop_free
            ]
            if length(raw_coords) == 1
                push!(raw_coords, raw_coords[1])
            end
            sm_coords = smooth_marine_polyline(
                raw_coords;
                n_knots      = n_knots,
                passes       = smooth_passes,
                check_raster = do_raster,
                resolution   = resolution,
                grid_minutes = grid_minutes
            )

            plen_km = 0.0
            for k_c in 2:length(sm_coords)
                plen_km += step_dist_km(sm_coords[k_c - 1], sm_coords[k_c])
            end
            c_rel = cents[clamp(u_rel, 1, S)]
            c_rec = cents[clamp(u_rec, 1, S)]
            disp_km = step_dist_km(c_rel, c_rec)
            lib = classify_liberty_interval(ev_days[i])
            ds_str = hasproperty(obs_df, :datasource) ?
                uppercase(strip(string(obs_df.datasource[i]))) : "TAG"
            t_rel_s = hasproperty(obs_df, :rel_time) && isfinite(Float64(obs_df.rel_time[i])) ?
                string(round(Float64(obs_df.rel_time[i]); digits = 2)) : "N/A"

            results[i] = (
                event_id        = i,
                tagid           = string(obs_df.tagid[i]),
                datasource      = ds_str,
                rel_time_str    = t_rel_s,
                liberty_days    = ev_days[i],
                liberty_bin     = lib.bin,
                liberty_label   = lib.label,
                color           = lib.color,
                displacement_km = disp_km,
                path_length_km  = plen_km,
                coords          = sm_coords,
            )
        end
    end

    return results
end
