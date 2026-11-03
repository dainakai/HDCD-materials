using Pkg
const PROJECT_DIR = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(PROJECT_DIR)

using CSV
using DataFrames
using JSON
using LinearAlgebra
using Printf
using ProgressMeter
using SavitzkyGolay
using Statistics

Base.@kwdef mutable struct Options
    run_dir::String = joinpath(PROJECT_DIR, "runs", "C000H001S0011")
    input_dir::String = ""
    output_dir::String = ""
    degree::Int = 2
    alpha::Float64 = 0.0
    lambda0::Float64 = 1e-6
    lambda_other::Float64 = 1e-4
    selection::String = "auto"
    degree_grid::Vector{Int} = [1, 2]
    lambda_other_grid::Vector{Float64} = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    alpha_grid::Vector{Float64} = [0.0, 1e-4, 1e-2]
    tune_max_trajectories::Int = 512
    tune_min_points::Int = 8
    tune_em_max_iter::Int = 100
    tune_relative_tolerance::Float64 = 0.01
    pi_init::Float64 = 0.05
    sigma2z0_init::Float64 = NaN
    sigma2z1_init::Float64 = NaN
    outlier_variance_multiplier::Float64 = 50.0
    em_tol::Float64 = 5e-5
    em_max_iter::Int = 100
    sg_window::Int = 7
    sg_order::Int = 2
    file_limit::Int = 0
    force::Bool = false
end

struct Trajectory
    filename::String
    frames::Vector{Int}
    times::Vector{Float64}
    xc::Vector{Float64}
    yc::Vector{Float64}
    z::Vector{Float64}
    diameter_px::Vector{Float64}
end

Base.@kwdef mutable struct ZGMMState
    pi_out::Float64
    sigma2_normal::Float64
    sigma2_outlier::Float64
    betas::Vector{Vector{Float64}}
    post_out_prob::Vector{Vector{Float64}}
    time_scale::Float64
    prior_mean::Vector{Float64}
    iterations::Int = 0
    final_diff::Float64 = Inf
    converged::Bool = false
end

function usage()
    println("""
    Usage:
      julia src/tracking/laptrack/zsmooth_new.jl [options]

    Input:
      Reads <run-dir>/trajectory3d/path_*.csv with columns:
      frame, xc, yc, slice, diameter_px

    Output:
      Writes <run-dir>/smoothedtrajs/path_*.csv with columns:
      frame, x, vx, y, vy, z, vz, diameter_px, postOutProb
      Also writes <run-dir>/smoothedtrajs_zgmm_params.csv with final Z-GMM parameters.

    Options:
      --run-dir=PATH          Run directory. Default: runs/C000H001S0011
      --input-dir=PATH        Input trajectory3d directory.
      --output-dir=PATH       Output smoothedtrajs directory.
      --selection=MODE        Hyperparameter selection: auto or fixed. Default: auto
      --degree=N              Fixed-mode polynomial degree. Default: 2
      --alpha=X               Fixed-mode degree-weighted regularization. Default: 0
      --lambda0=X             Constant coefficient prior precision. Default: 1e-6
      --lambda-other=X        Fixed-mode non-constant coefficient precision. Default: 1e-4
      --degree-grid=LIST      Auto-mode degrees. Default: 1,2
      --lambda-other-grid=LIST
                              Auto-mode base precisions. Default: 1e-8,1e-6,1e-4,1e-2,1e-1,1
      --alpha-grid=LIST       Auto-mode degree-weighted precisions for degree >= 2.
                              Default: 0,1e-4,1e-2
      --tune-max-trajectories=N
                              Length-stratified trajectory sample size. Default: 512
      --tune-min-points=N     Minimum points per tuning trajectory. Default: 8
      --tune-em-max-iter=N    EM iteration cap per tuning candidate. Default: 100
      --tune-relative-tolerance=X
                              Prefer a lower degree within this fraction of the best score.
                              Default: 0.01
      --pi-init=X             Initial outlier fraction. Default: 0.05
      --sigma2z0-init=X       Initial normal-noise variance. Default: from residuals
      --sigma2z1-init=X       Initial outlier-noise variance. Default: sigma2z0 * 50
      --outlier-var-mult=X    Fallback multiplier for outlier variance. Default: 50
      --em-tol=X              EM convergence tolerance. Default: 5e-5
      --em-max-iter=N         Maximum EM iterations. Default: 100
      --sg-window=N           Savitzky-Golay window size for x/y. Default: 7
      --sg-order=N            Savitzky-Golay polynomial order for x/y. Default: 2
      --file-limit=N          Process only first N files. Default: 0 (all files)
      --force                 Remove existing path_*.csv in output-dir before writing
      -h, --help              Show this help
    """)
end

function parse_int_list(value::AbstractString)::Vector{Int}
    values = [parse(Int, strip(item)) for item in split(value, ",") if !isempty(strip(item))]
    isempty(values) && error("Expected a non-empty comma-separated integer list")
    return values
end

function parse_float_list(value::AbstractString)::Vector{Float64}
    values = [parse(Float64, strip(item)) for item in split(value, ",") if !isempty(strip(item))]
    isempty(values) && error("Expected a non-empty comma-separated numeric list")
    return values
end

function parse_bool(value::AbstractString)
    lowered = lowercase(value)
    if lowered in ("1", "true", "yes", "y", "on")
        return true
    elseif lowered in ("0", "false", "no", "n", "off")
        return false
    end
    error("Invalid boolean value: $value")
end

function parse_args(args)::Options
    opt = Options()
    for arg in args
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif arg == "--force"
            opt.force = true
        elseif startswith(arg, "--force=")
            opt.force = parse_bool(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--run-dir=")
            opt.run_dir = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--input-dir=")
            opt.input_dir = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--output-dir=")
            opt.output_dir = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--selection=")
            opt.selection = lowercase(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--degree=")
            opt.degree = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--alpha=")
            opt.alpha = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lambda0=")
            opt.lambda0 = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lambda-other=")
            opt.lambda_other = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--degree-grid=")
            opt.degree_grid = parse_int_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--lambda-other-grid=")
            opt.lambda_other_grid = parse_float_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--alpha-grid=")
            opt.alpha_grid = parse_float_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tune-max-trajectories=")
            opt.tune_max_trajectories = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tune-min-points=")
            opt.tune_min_points = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tune-em-max-iter=")
            opt.tune_em_max_iter = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tune-relative-tolerance=")
            opt.tune_relative_tolerance = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--pi-init=")
            opt.pi_init = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sigma2z0-init=")
            opt.sigma2z0_init = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sigma2z1-init=")
            opt.sigma2z1_init = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--outlier-var-mult=")
            opt.outlier_variance_multiplier = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--em-tol=")
            opt.em_tol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--em-max-iter=")
            opt.em_max_iter = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sg-window=")
            opt.sg_window = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sg-order=")
            opt.sg_order = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--file-limit=")
            opt.file_limit = parse(Int, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg")
        end
    end
    return opt
end

function normalize_options!(opt::Options)
    opt.run_dir = abspath(opt.run_dir)
    isempty(opt.input_dir) && (opt.input_dir = joinpath(opt.run_dir, "trajectory3d"))
    isempty(opt.output_dir) && (opt.output_dir = joinpath(opt.run_dir, "smoothedtrajs"))
    opt.input_dir = abspath(opt.input_dir)
    opt.output_dir = abspath(opt.output_dir)

    opt.degree >= 0 || error("--degree must be non-negative")
    opt.alpha >= 0.0 || error("--alpha must be non-negative")
    opt.lambda0 > 0.0 || error("--lambda0 must be positive")
    opt.lambda_other >= 0.0 || error("--lambda-other must be non-negative")
    0.0 < opt.pi_init < 1.0 || error("--pi-init must be between 0 and 1")
    opt.outlier_variance_multiplier > 1.0 || error("--outlier-var-mult must be greater than 1")
    opt.em_tol > 0.0 || error("--em-tol must be positive")
    opt.em_max_iter > 0 || error("--em-max-iter must be positive")
    opt.sg_window > 0 || error("--sg-window must be positive")
    opt.sg_order >= 0 || error("--sg-order must be non-negative")
    opt.file_limit >= 0 || error("--file-limit must be non-negative")
    opt.selection in ("auto", "fixed") || error("--selection must be auto or fixed")
    all(degree -> degree >= 0, opt.degree_grid) || error("--degree-grid values must be non-negative")
    all(value -> value >= 0.0, opt.lambda_other_grid) ||
        error("--lambda-other-grid values must be non-negative")
    all(value -> value >= 0.0, opt.alpha_grid) || error("--alpha-grid values must be non-negative")
    opt.tune_max_trajectories > 0 || error("--tune-max-trajectories must be positive")
    opt.tune_min_points > 1 || error("--tune-min-points must be greater than 1")
    opt.tune_em_max_iter > 0 || error("--tune-em-max-iter must be positive")
    opt.tune_relative_tolerance >= 0.0 || error("--tune-relative-tolerance must be non-negative")
    return opt
end

function is_path_csv(name::AbstractString)
    return occursin(r"^path_[0-9]+\.csv$", name)
end

function collect_path_files(input_dir::AbstractString, file_limit::Integer)
    isdir(input_dir) || error("Input directory not found: $input_dir")
    files = sort(collect(joinpath(input_dir, name) for name in readdir(input_dir) if is_path_csv(name)))
    isempty(files) && error("No path_*.csv files found in $input_dir")
    if file_limit > 0 && length(files) > file_limit
        resize!(files, file_limit)
    end
    return files
end

function prepare_output_dir!(opt::Options)
    mkpath(opt.output_dir)
    existing = filter(is_path_csv, readdir(opt.output_dir))
    if !isempty(existing)
        if !opt.force
            error("Output directory already contains $(length(existing)) path_*.csv file(s): $(opt.output_dir). Use --force to overwrite.")
        end
        for name in existing
            rm(joinpath(opt.output_dir, name); force=true)
        end
    end
end

function require_columns!(df::DataFrame, file::AbstractString)
    required = (:frame, :xc, :yc, :slice, :diameter_px)
    missing_columns = setdiff(required, Tuple(Symbol.(names(df))))
    isempty(missing_columns) ||
        error("$(basename(file)) is missing required column(s): $(join(string.(missing_columns), ", "))")
end

function load_trajectory(file::AbstractString)::Trajectory
    df = CSV.read(file, DataFrame)
    require_columns!(df, file)
    frames = Int.(df[!, :frame])
    return Trajectory(
        basename(file),
        frames,
        Float64.(frames),
        Float64.(df[!, :xc]),
        Float64.(df[!, :yc]),
        Float64.(df[!, :slice]),
        Float64.(df[!, :diameter_px]),
    )
end

function load_trajectories(files::Vector{String})
    trajs = Trajectory[]
    p = Progress(length(files), 1, "Loading trajectory3d CSVs... ")
    for file in files
        traj = load_trajectory(file)
        if !isempty(traj.frames)
            push!(trajs, traj)
        end
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
    isempty(trajs) && error("No non-empty trajectories were loaded.")
    return trajs
end

@inline function time_span(times::AbstractVector{<:Real})::Float64
    isempty(times) && return 1.0
    span = Float64(last(times) - first(times))
    return iszero(span) ? 1.0 : span
end

function max_time_span(trajs::Vector{Trajectory})::Float64
    spans = Float64[]
    for traj in trajs
        span = time_span(traj.times)
        if isfinite(span) && span > 0.0
            push!(spans, span)
        end
    end
    return isempty(spans) ? 1.0 : maximum(spans)
end

function scaled_relative_times(traj::Trajectory, time_scale::Real)::Vector{Float64}
    isempty(traj.times) && return Float64[]
    t0 = Float64(first(traj.times))
    return Float64.((traj.times .- t0) ./ Float64(time_scale))
end

function design_matrix(times::AbstractVector{<:Real}, degree::Integer)::Matrix{Float64}
    n = length(times)
    phi = Matrix{Float64}(undef, n, degree + 1)
    for k in 0:degree
        phi[:, k + 1] = Float64.(times) .^ k
    end
    return phi
end

function regularization_matrix(degree::Integer, alpha::Real, lambda0::Real, lambda_other::Real)::Matrix{Float64}
    diag_values = Vector{Float64}(undef, degree + 1)
    diag_values[1] = Float64(lambda0)
    for k in 1:degree
        diag_values[k + 1] = Float64(lambda_other) + Float64(alpha) * k^2
    end
    return Diagonal(diag_values) |> Matrix
end

function stable_logsumexp2(a::Float64, b::Float64)::Float64
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function log_gaussian_pdf(x::Real, mu::Real, sigma2::Real)::Float64
    sigma2 <= 0.0 && return -Inf
    return -0.5 * log(2 * pi * sigma2) - 0.5 * (Float64(x) - Float64(mu))^2 / sigma2
end

function z_ridge_fit(phi::Matrix{Float64}, z::Vector{Float64},
                     prior_precision::Matrix{Float64}, prior_mean::Vector{Float64};
                     weights::Union{Nothing,Vector{Float64}}=nothing)::Vector{Float64}
    if weights === nothing
        A = phi' * phi + prior_precision
        b = phi' * z + prior_precision * prior_mean
    else
        weighted_phi = phi .* weights
        A = phi' * weighted_phi + prior_precision
        b = phi' * (weights .* z) + prior_precision * prior_mean
    end
    return cholesky(Hermitian(A)) \ b
end

function initial_z_betas(trajs::Vector{Trajectory}, degree::Integer,
                         prior_precision::Matrix{Float64}, time_scale::Float64)
    zero_prior = zeros(Float64, degree + 1)
    betas = Vector{Vector{Float64}}(undef, length(trajs))
    for (i, traj) in enumerate(trajs)
        tau = scaled_relative_times(traj, time_scale)
        phi = design_matrix(tau, degree)
        betas[i] = z_ridge_fit(phi, traj.z, prior_precision, zero_prior)
    end
    return betas
end

function mean_beta(betas::Vector{Vector{Float64}})::Vector{Float64}
    isempty(betas) && return Float64[]
    mu = zeros(Float64, length(first(betas)))
    for beta in betas
        mu .+= beta
    end
    return mu ./ length(betas)
end

function residual_variance(trajs::Vector{Trajectory}, betas::Vector{Vector{Float64}},
                           degree::Integer, time_scale::Float64)::Float64
    residuals_sq_sum = 0.0
    n = 0
    for (traj, beta) in zip(trajs, betas)
        phi = design_matrix(scaled_relative_times(traj, time_scale), degree)
        residuals = traj.z .- phi * beta
        residuals_sq_sum += sum(abs2, residuals)
        n += length(residuals)
    end
    return max(1e-9, residuals_sq_sum / max(n, 1))
end

function initialize_z_gmm(trajs::Vector{Trajectory}, opt::Options)::ZGMMState
    time_scale = max_time_span(trajs)
    prior_precision = regularization_matrix(opt.degree, opt.alpha, opt.lambda0, opt.lambda_other)

    betas0 = initial_z_betas(trajs, opt.degree, prior_precision, time_scale)
    prior_mean = mean_beta(betas0)
    betas = Vector{Vector{Float64}}(undef, length(trajs))
    for (i, traj) in enumerate(trajs)
        phi = design_matrix(scaled_relative_times(traj, time_scale), opt.degree)
        betas[i] = z_ridge_fit(phi, traj.z, prior_precision, prior_mean)
    end

    sigma2_0 = isfinite(opt.sigma2z0_init) ? opt.sigma2z0_init : residual_variance(trajs, betas, opt.degree, time_scale)
    sigma2_0 = max(1e-9, sigma2_0)
    sigma2_1 = isfinite(opt.sigma2z1_init) ? opt.sigma2z1_init : sigma2_0 * opt.outlier_variance_multiplier
    sigma2_1 = max(sigma2_0 * 1.000001, sigma2_1)

    post = [zeros(Float64, length(traj.z)) for traj in trajs]
    return ZGMMState(opt.pi_init, sigma2_0, sigma2_1, betas, post, time_scale, prior_mean, 0, Inf, false)
end

function update_posteriors!(state::ZGMMState, trajs::Vector{Trajectory}, degree::Integer)
    total_resp = 0.0
    total_points = 0
    for (i, traj) in enumerate(trajs)
        phi = design_matrix(scaled_relative_times(traj, state.time_scale), degree)
        pred = phi * state.betas[i]
        resp = state.post_out_prob[i]
        for j in eachindex(traj.z)
            log_p0 = log(1.0 - state.pi_out + 1e-12) +
                     log_gaussian_pdf(traj.z[j], pred[j], state.sigma2_normal)
            log_p1 = log(state.pi_out + 1e-12) +
                     log_gaussian_pdf(traj.z[j], pred[j], state.sigma2_outlier)
            denom = stable_logsumexp2(log_p0, log_p1)
            resp[j] = clamp(exp(log_p1 - denom), 0.0, 1.0)
        end
        total_resp += sum(resp)
        total_points += length(resp)
    end
    return total_resp, total_points
end

function update_betas!(state::ZGMMState, trajs::Vector{Trajectory}, opt::Options,
                       prior_precision::Matrix{Float64})
    for (i, traj) in enumerate(trajs)
        phi = design_matrix(scaled_relative_times(traj, state.time_scale), opt.degree)
        resp = state.post_out_prob[i]
        weights = (1.0 .- resp) ./ state.sigma2_normal .+ resp ./ state.sigma2_outlier
        state.betas[i] = z_ridge_fit(phi, traj.z, prior_precision, state.prior_mean; weights=weights)
    end
end

function update_variances!(state::ZGMMState, trajs::Vector{Trajectory}, degree::Integer)
    sum_resp = 0.0
    sum_normal = 0.0
    sum_resp_sq = 0.0
    sum_normal_sq = 0.0

    for (i, traj) in enumerate(trajs)
        phi = design_matrix(scaled_relative_times(traj, state.time_scale), degree)
        pred = phi * state.betas[i]
        residual_sq = (traj.z .- pred).^2
        resp = state.post_out_prob[i]
        normal_weight = 1.0 .- resp
        sum_resp += sum(resp)
        sum_normal += sum(normal_weight)
        sum_resp_sq += sum(resp .* residual_sq)
        sum_normal_sq += sum(normal_weight .* residual_sq)
    end

    state.sigma2_normal = max(1e-9, sum_normal_sq / max(sum_normal, 1e-9))
    state.sigma2_outlier = max(state.sigma2_normal * 1.000001, sum_resp_sq / max(sum_resp, 1e-9))
end

function fit_z_gmm!(trajs::Vector{Trajectory}, opt::Options; verbose::Bool=true)::ZGMMState
    state = initialize_z_gmm(trajs, opt)
    prior_precision = regularization_matrix(opt.degree, opt.alpha, opt.lambda0, opt.lambda_other)

    if verbose
        @printf("Z smoothing setup: trajectories=%d points=%d degree=%d alpha=%.6g lambda_other=%.6g time_scale=%.6g\n",
                length(trajs), sum(length(traj.z) for traj in trajs), opt.degree, opt.alpha,
                opt.lambda_other, state.time_scale)
        @printf("Initial Z-GMM: pi=%.6g sigma2_normal=%.6g sigma2_outlier=%.6g\n",
                state.pi_out, state.sigma2_normal, state.sigma2_outlier)
    end

    diff = Inf
    iterations = 0
    converged = false
    progress = verbose ? Progress(opt.em_max_iter, 1, "Running Z GMM EM... ") : nothing
    for iter in 1:opt.em_max_iter
        iterations = iter
        old_pi = state.pi_out
        old_s0 = state.sigma2_normal
        old_s1 = state.sigma2_outlier
        old_beta_norm = mean(norm.(state.betas))

        total_resp, total_points = update_posteriors!(state, trajs, opt.degree)
        total_points > 0 || error("No points available for Z smoothing.")
        state.pi_out = clamp(total_resp / total_points, 1e-9, 1.0 - 1e-9)

        update_betas!(state, trajs, opt, prior_precision)
        update_variances!(state, trajs, opt.degree)

        beta_norm = mean(norm.(state.betas))
        diff = maximum((
            abs(state.pi_out - old_pi) / (abs(old_pi) + 1e-8),
            abs(state.sigma2_normal - old_s0) / (abs(old_s0) + 1e-8),
            abs(state.sigma2_outlier - old_s1) / (abs(old_s1) + 1e-8),
            abs(beta_norm - old_beta_norm) / (abs(old_beta_norm) + 1e-8),
        ))

        if verbose
            ProgressMeter.next!(progress; showvalues=[
                (:iter, iter),
                (:diff, diff),
                (:pi, state.pi_out),
                (:sigma2_normal, state.sigma2_normal),
                (:sigma2_outlier, state.sigma2_outlier),
            ])
        end
        if diff <= opt.em_tol
            converged = true
            break
        end
    end
    verbose && ProgressMeter.finish!(progress)

    state.iterations = iterations
    state.final_diff = diff
    state.converged = converged

    if !converged && verbose
        @warn "Z GMM EM reached max iterations without satisfying tolerance" diff=diff tol=opt.em_tol
    end

    update_posteriors!(state, trajs, opt.degree)
    if verbose
        @printf("Final Z-GMM: pi=%.6g sigma2_normal=%.6g sigma2_outlier=%.6g diff=%.6g iterations=%d converged=%s\n",
                state.pi_out, state.sigma2_normal, state.sigma2_outlier, diff,
                state.iterations, string(state.converged))
    end
    return state
end

function select_tuning_trajectories(trajs::Vector{Trajectory}, opt::Options)::Vector{Trajectory}
    eligible = [traj for traj in trajs if length(traj.z) >= opt.tune_min_points]
    isempty(eligible) &&
        error("No trajectories have at least $(opt.tune_min_points) points for auto tuning")
    sort!(eligible; by=traj -> (length(traj.z), traj.filename))
    if length(eligible) <= opt.tune_max_trajectories
        return eligible
    end
    count = opt.tune_max_trajectories
    indices = unique(round.(Int, range(1, length(eligible); length=count)))
    return eligible[indices]
end

function weighted_quantile(values::Vector{Float64}, weights::Vector{Float64}, q::Real)::Float64
    length(values) == length(weights) || error("values and weights must have equal lengths")
    isempty(values) && return NaN
    order = sortperm(values)
    total_weight = sum(weights)
    total_weight > 0.0 || return NaN
    target = clamp(Float64(q), 0.0, 1.0) * total_weight
    cumulative = 0.0
    for index in order
        cumulative += weights[index]
        if cumulative >= target
            return values[index]
        end
    end
    return values[last(order)]
end

function press_metrics(state::ZGMMState, trajs::Vector{Trajectory}, opt::Options)
    errors = Float64[]
    score_weights = Float64[]
    track_maes = Float64[]
    leverage_sum = 0.0
    leverage_count = 0

    prior_precision = regularization_matrix(opt.degree, opt.alpha, opt.lambda0, opt.lambda_other)
    for (i, traj) in enumerate(trajs)
        phi = design_matrix(scaled_relative_times(traj, state.time_scale), opt.degree)
        resp = state.post_out_prob[i]
        fit_weights = (1.0 .- resp) ./ state.sigma2_normal .+ resp ./ state.sigma2_outlier
        A = phi' * (phi .* fit_weights) + prior_precision
        solved = cholesky(Hermitian(A)) \ transpose(phi)
        leverage = fit_weights .* vec(sum(phi .* transpose(solved); dims=2))
        residual = traj.z .- phi * state.betas[i]
        denominator = max.(1.0 .- leverage, 1e-6)
        loo_error = abs.(residual ./ denominator)
        inlier_weight = clamp.(1.0 .- resp, 1e-6, 1.0)
        keep = isfinite.(loo_error) .& isfinite.(inlier_weight) .& (inlier_weight .> 0.0)
        any(keep) || continue
        kept_errors = loo_error[keep]
        kept_weights = inlier_weight[keep]
        append!(errors, kept_errors)
        append!(score_weights, kept_weights)
        push!(track_maes, sum(kept_errors .* kept_weights) / sum(kept_weights))
        leverage_sum += sum(leverage[keep])
        leverage_count += count(keep)
    end
    isempty(errors) && error("No finite PRESS residuals were produced during auto tuning")

    total_weight = sum(score_weights)
    weighted_mae = sum(errors .* score_weights) / total_weight
    weighted_rmse = sqrt(sum((errors .^ 2) .* score_weights) / total_weight)
    p95 = weighted_quantile(errors, score_weights, 0.95)
    p99 = weighted_quantile(errors, score_weights, 0.99)
    clipped = min.(errors, p99)
    trimmed_rmse_99 = sqrt(sum((clipped .^ 2) .* score_weights) / total_weight)
    return Dict{String,Any}(
        "objective" => trimmed_rmse_99,
        "objective_name" => "inlier_weighted_press_trimmed_rmse_99",
        "press_weighted_mae" => weighted_mae,
        "press_weighted_rmse" => weighted_rmse,
        "press_weighted_p95" => p95,
        "press_weighted_p99" => p99,
        "press_track_macro_mae" => mean(track_maes),
        "mean_leverage" => leverage_sum / max(leverage_count, 1),
        "scored_tracks" => length(track_maes),
        "scored_points" => length(errors),
    )
end

function tuning_candidates(opt::Options)
    candidates = NamedTuple{(:degree, :alpha, :lambda_other),Tuple{Int,Float64,Float64}}[]
    for degree in sort(unique(opt.degree_grid))
        alphas = degree >= 2 ? sort(unique(opt.alpha_grid)) : [0.0]
        for lambda_other in sort(unique(opt.lambda_other_grid))
            for alpha in alphas
                push!(candidates, (
                    degree=degree,
                    alpha=alpha,
                    lambda_other=lambda_other,
                ))
            end
        end
    end
    return candidates
end

function tune_hyperparameters!(trajs::Vector{Trajectory}, opt::Options)
    tuning_trajs = select_tuning_trajectories(trajs, opt)
    lengths = length.(getfield.(tuning_trajs, :z))
    candidates = Dict{String,Any}[]
    @printf(
        "Auto-tuning Z smoothing: sample_trajectories=%d sample_points=%d candidates=%d\n",
        length(tuning_trajs),
        sum(lengths),
        length(tuning_candidates(opt)),
    )

    for (index, spec) in enumerate(tuning_candidates(opt))
        candidate_opt = deepcopy(opt)
        candidate_opt.selection = "fixed"
        candidate_opt.degree = spec.degree
        candidate_opt.alpha = spec.alpha
        candidate_opt.lambda_other = spec.lambda_other
        candidate_opt.em_max_iter = min(opt.em_max_iter, opt.tune_em_max_iter)
        started = time()
        candidate = Dict{String,Any}(
            "index" => index,
            "degree" => spec.degree,
            "alpha" => spec.alpha,
            "lambda_other" => spec.lambda_other,
        )
        try
            state = fit_z_gmm!(tuning_trajs, candidate_opt; verbose=false)
            merge!(candidate, press_metrics(state, tuning_trajs, candidate_opt))
            candidate["status"] = "passed"
            candidate["em_iterations"] = state.iterations
            candidate["em_converged"] = state.converged
            candidate["em_final_diff"] = state.final_diff
            candidate["sigma_z_normal"] = sqrt(state.sigma2_normal)
            candidate["pi_out"] = state.pi_out
        catch error
            candidate["status"] = "failed"
            candidate["error"] = sprint(showerror, error)
            candidate["objective"] = nothing
        end
        candidate["elapsed_sec"] = time() - started
        push!(candidates, candidate)
        @printf(
            "Auto-tune candidate %d/%d: degree=%d alpha=%.6g lambda_other=%.6g status=%s objective=%s elapsed=%.3fs\n",
            index,
            length(tuning_candidates(opt)),
            spec.degree,
            spec.alpha,
            spec.lambda_other,
            candidate["status"],
            candidate["objective"] === nothing ? "NA" : @sprintf("%.6g", candidate["objective"]),
            candidate["elapsed_sec"],
        )
    end

    passed = [candidate for candidate in candidates if candidate["status"] == "passed"]
    isempty(passed) && error("All Z-smoothing auto-tune candidates failed")
    converged_candidates = [candidate for candidate in passed if candidate["em_converged"]]
    isempty(converged_candidates) &&
        error("No Z-smoothing auto-tune candidate converged within $(opt.tune_em_max_iter) iterations")
    minimum_objective = minimum(candidate["objective"] for candidate in converged_candidates)
    near_optimal_threshold = minimum_objective * (1.0 + opt.tune_relative_tolerance)
    near_optimal = [
        candidate for candidate in converged_candidates
        if candidate["objective"] <= near_optimal_threshold
    ]
    sort!(near_optimal; by=candidate -> (
        candidate["degree"],
        candidate["objective"],
        -candidate["lambda_other"],
        -candidate["alpha"],
    ))
    selected = first(near_optimal)
    opt.degree = selected["degree"]
    opt.alpha = selected["alpha"]
    opt.lambda_other = selected["lambda_other"]
    @printf(
        "Auto-tune selected: degree=%d alpha=%.6g lambda_other=%.6g objective=%.6g minimum=%.6g tolerance=%.3g\n",
        opt.degree,
        opt.alpha,
        opt.lambda_other,
        selected["objective"],
        minimum_objective,
        opt.tune_relative_tolerance,
    )
    return Dict{String,Any}(
        "selection" => "auto",
        "objective_name" => selected["objective_name"],
        "selection_rule" => "lowest degree within relative tolerance, then minimum objective",
        "minimum_objective" => minimum_objective,
        "relative_tolerance" => opt.tune_relative_tolerance,
        "near_optimal_threshold" => near_optimal_threshold,
        "converged_candidates" => length(converged_candidates),
        "sample" => Dict(
            "selection_method" => "length_stratified_deterministic",
            "eligible_trajectories" => count(traj -> length(traj.z) >= opt.tune_min_points, trajs),
            "sampled_trajectories" => length(tuning_trajs),
            "sampled_points" => sum(lengths),
            "minimum_track_points" => minimum(lengths),
            "median_track_points" => median(lengths),
            "maximum_track_points" => maximum(lengths),
        ),
        "grid" => Dict(
            "degree" => sort(unique(opt.degree_grid)),
            "lambda_other" => sort(unique(opt.lambda_other_grid)),
            "alpha_for_degree_ge_2" => sort(unique(opt.alpha_grid)),
        ),
        "candidates" => candidates,
        "selected" => selected,
    )
end

function fixed_selection_summary(opt::Options)
    return Dict{String,Any}(
        "selection" => "fixed",
        "selected" => Dict(
            "degree" => opt.degree,
            "alpha" => opt.alpha,
            "lambda_other" => opt.lambda_other,
        ),
        "candidates" => Any[],
    )
end

function z_smoothed(state::ZGMMState, traj::Trajectory, beta::Vector{Float64}, degree::Integer)
    phi = design_matrix(scaled_relative_times(traj, state.time_scale), degree)
    return phi * beta
end

function z_velocity(traj::Trajectory, beta::Vector{Float64}, time_scale::Float64)
    degree = length(beta) - 1
    velocity = zeros(Float64, length(traj.times))
    degree <= 0 && return velocity

    tau = scaled_relative_times(traj, time_scale)
    for k in 1:degree
        velocity .+= k * beta[k + 1] .* (tau .^ (k - 1))
    end
    return velocity ./ time_scale
end

function effective_sg_params(n::Integer, requested_window::Integer, requested_order::Integer; deriv::Integer=0)
    n > 1 || return nothing
    w = min(requested_window, n)
    iseven(w) && (w -= 1)
    w < 3 && return nothing
    order = min(requested_order, w - 2)
    order < deriv && return nothing
    return w, order
end

function smooth_savgol(values::Vector{Float64}, requested_window::Integer, requested_order::Integer; deriv::Integer=0)
    params = effective_sg_params(length(values), requested_window, requested_order; deriv=deriv)
    params === nothing && return deriv == 0 ? copy(values) : zeros(Float64, length(values))
    w, order = params
    return Float64.(savitzky_golay(values, w, order; deriv=deriv, rate=1.0).y)
end

function smooth_savgol_with_velocity(values::Vector{Float64}, requested_window::Integer, requested_order::Integer)
    smoothed = smooth_savgol(values, requested_window, requested_order; deriv=0)
    deriv_params = effective_sg_params(length(values), requested_window, requested_order; deriv=1)
    if deriv_params === nothing
        return smoothed, zeros(Float64, length(values))
    end
    w, order = deriv_params
    velocity = Float64.(savitzky_golay(values, w, order; deriv=1, rate=1.0).y)
    return smoothed, velocity
end

function save_zgmm_params(state::ZGMMState, opt::Options)
    outpath = joinpath(opt.run_dir, "smoothedtrajs_zgmm_params.csv")
    params = DataFrame(
        parameter=[
            "pi_out",
            "sigma2_z0",
            "sigma2_z1",
            "sigma_z0",
            "sigma_z1",
            "time_scale",
            "degree",
            "alpha",
            "lambda0",
            "lambda_other",
            "selection_auto",
            "sg_window",
            "sg_order",
            "em_tol",
            "em_max_iter",
            "em_iterations",
            "em_final_diff",
            "em_converged",
        ],
        value=[
            state.pi_out,
            state.sigma2_normal,
            state.sigma2_outlier,
            sqrt(state.sigma2_normal),
            sqrt(state.sigma2_outlier),
            state.time_scale,
            Float64(opt.degree),
            opt.alpha,
            opt.lambda0,
            opt.lambda_other,
            opt.selection == "auto" ? 1.0 : 0.0,
            Float64(opt.sg_window),
            Float64(opt.sg_order),
            opt.em_tol,
            Float64(opt.em_max_iter),
            Float64(state.iterations),
            state.final_diff,
            state.converged ? 1.0 : 0.0,
        ],
    )
    CSV.write(outpath, params)
    return outpath
end

function save_tuning_summary!(summary::Dict{String,Any}, state::ZGMMState, opt::Options)
    summary["selected"] = merge(
        Dict{String,Any}(
            "degree" => opt.degree,
            "alpha" => opt.alpha,
            "lambda_other" => opt.lambda_other,
        ),
        get(summary, "selected", Dict{String,Any}()),
    )
    summary["final_fit"] = Dict(
        "trajectories" => length(state.betas),
        "degree" => opt.degree,
        "alpha" => opt.alpha,
        "lambda0" => opt.lambda0,
        "lambda_other" => opt.lambda_other,
        "pi_out" => state.pi_out,
        "sigma_z_normal" => sqrt(state.sigma2_normal),
        "sigma_z_outlier" => sqrt(state.sigma2_outlier),
        "time_scale" => state.time_scale,
        "em_iterations" => state.iterations,
        "em_final_diff" => state.final_diff,
        "em_converged" => state.converged,
    )
    summary["input_dir"] = opt.input_dir
    summary["output_dir"] = opt.output_dir
    outpath = joinpath(opt.run_dir, "zsmooth_tuning_summary.json")
    open(outpath, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
    return outpath
end

function save_smoothed!(trajs::Vector{Trajectory}, state::ZGMMState, opt::Options)
    prepare_output_dir!(opt)
    p = Progress(length(trajs), 1, "Writing smoothed trajectories... ")
    for (i, traj) in enumerate(trajs)
        x, vx = smooth_savgol_with_velocity(traj.xc, opt.sg_window, opt.sg_order)
        y, vy = smooth_savgol_with_velocity(traj.yc, opt.sg_window, opt.sg_order)
        z = z_smoothed(state, traj, state.betas[i], opt.degree)
        vz = z_velocity(traj, state.betas[i], state.time_scale)
        outdf = DataFrame(
            frame=traj.frames,
            x=x,
            vx=vx,
            y=y,
            vy=vy,
            z=z,
            vz=vz,
            diameter_px=traj.diameter_px,
            postOutProb=state.post_out_prob[i],
        )
        CSV.write(joinpath(opt.output_dir, traj.filename), outdf)
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
end

function main(args=ARGS)
    opt = normalize_options!(parse_args(args))
    files = collect_path_files(opt.input_dir, opt.file_limit)
    @printf("input_dir=%s files=%d\n", opt.input_dir, length(files))
    @printf("output_dir=%s\n", opt.output_dir)
    trajs = load_trajectories(files)
    tuning_summary = opt.selection == "auto" ?
        tune_hyperparameters!(trajs, opt) :
        fixed_selection_summary(opt)
    state = fit_z_gmm!(trajs, opt)
    save_smoothed!(trajs, state, opt)
    params_path = save_zgmm_params(state, opt)
    tuning_path = save_tuning_summary!(tuning_summary, state, opt)
    @printf("params=%s\n", params_path)
    @printf("tuning_summary=%s\n", tuning_path)
    @printf("done files=%d output_dir=%s\n", length(trajs), opt.output_dir)
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
