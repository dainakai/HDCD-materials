using Pkg

const PROJECT_DIR = @__DIR__
Pkg.activate(PROJECT_DIR)

using CSV
using CUDA
using CUDA.CUFFT
using DataFrames
using Images
using JSON
using LinearAlgebra
using ParticleHolography
using ProgressMeter
using Statistics

Base.@kwdef struct RenderParams
    out_dir::String = joinpath(PROJECT_DIR, "..", "..", "synth", "collision_benchmark_260715")
    datlen::Int = 1024
    propagation_datlen::Int = 1536
    dx_um::Float32 = 10.0f0
    wavelength_um::Float32 = 0.6328f0
    background_gray::Float32 = 77.5f0
    c002_front_distance_um::Float32 = 74900.0f0
    c001_front_distance_um::Float32 = 96350.0f0
    device::Int = parse(Int, get(ENV, "CUDA_DEVICE", "1"))
    frames::Int = 0
    frame_start::Int = 0
    frame_end::Int = 0
    summary_name::String = "render_waveprop_summary.json"
end

mutable struct PropagationContext
    wave_a::CuArray{ComplexF32,2}
    wave_b::CuArray{ComplexF32,2}
    fft_arr::CuArray{ComplexF32,2}
    transfer_fft_order::CuArray{ComplexF32,2}
    sqrt_fft_order::CuArray{Float32,2}
    intensity::CuArray{Float32,2}
    crop_start::Int
    crop_end::Int
    fft_plan
    ifft_plan
end

function parse_args(args)
    params = RenderParams()
    out_dir = params.out_dir
    device = params.device
    frames = params.frames
    frame_start = params.frame_start
    frame_end = params.frame_end
    summary_name = params.summary_name
    for arg in args
        if startswith(arg, "--out-dir=")
            out_dir = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--device=")
            device = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--frames=")
            frames = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--frame-start=")
            frame_start = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--frame-end=")
            frame_end = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--summary-name=")
            summary_name = split(arg, "=", limit=2)[2]
        else
            error("Unknown argument: $arg")
        end
    end
    frame_start >= 0 || error("--frame-start must be non-negative")
    frame_end == 0 || frame_end > frame_start ||
        error("--frame-end must be zero or greater than --frame-start")
    return RenderParams(;
        out_dir=abspath(out_dir),
        device=device,
        frames=frames,
        frame_start=frame_start,
        frame_end=frame_end,
        summary_name=summary_name,
    )
end

function select_cuda_device!(device_index::Integer)
    devices = collect(CUDA.devices())
    length(devices) >= device_index ||
        error("CUDA device $device_index was requested, but only $(length(devices)) device(s) are visible")
    CUDA.device!(devices[device_index])
    @info "Using CUDA device" index=device_index name=CUDA.name(CUDA.device())
end

function transfer_kernel!(tf, sqrt_fft_order, z_um::Float32, wavelength_um::Float32, n::Int)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= n && col <= n
        phase = Float32(2π) * z_um / wavelength_um * sqrt(sqrt_fft_order[row, col])
        tf[row, col] = ComplexF32(cos(phase), sin(phase))
    end
    return nothing
end

function apply_disk_kernel!(wave, cx::Float32, cy::Float32, radius::Float32,
                            min_col::Int32, min_row::Int32, box::Int32, n::Int32)
    lx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ly = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if lx <= box && ly <= box
        col = min_col + lx - Int32(1)
        row = min_row + ly - Int32(1)
        if Int32(1) <= col <= n && Int32(1) <= row <= n
            dx = Float32(col - 1) - cx
            dy = Float32(row - 1) - cy
            if dx * dx + dy * dy <= radius * radius
                wave[row, col] = ComplexF32(0.0f0, 0.0f0)
            end
        end
    end
    return nothing
end

function abs2_center_crop_to_gray_kernel!(out, wave, background_gray::Float32, n::Int, crop_start::Int)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= n && col <= n
        src_row = crop_start + row - 1
        src_col = crop_start + col - 1
        val = background_gray * abs2(wave[src_row, src_col])
        out[row, col] = clamp(val, 0.0f0, 255.0f0)
    end
    return nothing
end

function make_context(params::RenderParams)
    fieldlen = params.propagation_datlen
    fieldlen >= params.datlen ||
        error("propagation_datlen must be at least datlen")
    iseven(fieldlen - params.datlen) ||
        error("propagation_datlen - datlen must be even for a centered crop")
    wave_a = CUDA.fill(ComplexF32(1.0f0, 0.0f0), fieldlen, fieldlen)
    wave_b = similar(wave_a)
    fft_arr = similar(wave_a)
    transfer_fft_order = similar(wave_a)
    d_sqr = cu_transfer_sqrt_arr(fieldlen, params.wavelength_um, params.dx_um)
    sqrt_fft_order = CUFFT.ifftshift(d_sqr.data)
    intensity = CUDA.zeros(Float32, params.datlen, params.datlen)
    crop_start = (fieldlen - params.datlen) ÷ 2 + 1
    crop_end = crop_start + params.datlen - 1
    fft_plan = CUFFT.plan_fft(wave_a)
    ifft_plan = CUFFT.plan_ifft(wave_a)
    return PropagationContext(
        wave_a,
        wave_b,
        fft_arr,
        transfer_fft_order,
        sqrt_fft_order,
        intensity,
        crop_start,
        crop_end,
        fft_plan,
        ifft_plan,
    )
end

function propagate!(out, input, z_um::Float32, ctx::PropagationContext, params::RenderParams)
    if abs(z_um) < 1.0f-6
        out .= input
        return nothing
    end
    fieldlen = size(input, 1)
    threads = (32, 8)
    blocks = cld.((fieldlen, fieldlen), threads)
    @cuda threads=threads blocks=blocks transfer_kernel!(
        ctx.transfer_fft_order,
        ctx.sqrt_fft_order,
        z_um,
        params.wavelength_um,
        fieldlen,
    )
    LinearAlgebra.mul!(ctx.fft_arr, ctx.fft_plan, input)
    ctx.fft_arr .*= ctx.transfer_fft_order
    LinearAlgebra.mul!(out, ctx.ifft_plan, ctx.fft_arr)
    return nothing
end

function apply_disk!(wave, cx_px::Float32, cy_px::Float32, diameter_um::Float32,
                     ctx::PropagationContext, params::RenderParams)
    radius = max(0.5f0, diameter_um / (2.0f0 * params.dx_um))
    field_cx = cx_px + Float32(ctx.crop_start - 1)
    field_cy = cy_px + Float32(ctx.crop_start - 1)
    margin = Int32(ceil(radius) + 2)
    min_col = Int32(floor(field_cx) + 1 - margin)
    min_row = Int32(floor(field_cy) + 1 - margin)
    box = Int32(2 * margin + 1)
    threads = (16, 16)
    blocks = cld.((Int(box), Int(box)), threads)
    @cuda threads=threads blocks=blocks apply_disk_kernel!(
        wave,
        field_cx,
        field_cy,
        radius,
        min_col,
        min_row,
        box,
        Int32(size(wave, 1)),
    )
    return nothing
end

function save_gray_png(path::AbstractString, d_img::CuArray{Float32,2})
    img = Array(d_img)
    img01 = clamp.(img ./ 255.0f0, 0.0f0, 1.0f0)
    save(path, Gray.(img01))
    return Dict(
        "min" => Float64(minimum(img)),
        "max" => Float64(maximum(img)),
        "mean" => Float64(mean(img)),
        "std" => Float64(std(img)),
    )
end

function wave_to_gray_png!(path::AbstractString, wave::CuArray{ComplexF32,2},
                           ctx::PropagationContext, params::RenderParams)
    threads = (32, 8)
    blocks = cld.((params.datlen, params.datlen), threads)
    @cuda threads=threads blocks=blocks abs2_center_crop_to_gray_kernel!(
        ctx.intensity,
        wave,
        params.background_gray,
        params.datlen,
        ctx.crop_start,
    )
    return save_gray_png(path, ctx.intensity)
end

function apply_particles_to_wave!(rows::DataFrame, ctx::PropagationContext, params::RenderParams)
    sort!(rows, :z_um, rev=true)
    ctx.wave_a .= ComplexF32(1.0f0, 0.0f0)
    current_z = Float32(rows.z_um[1])
    first_plane = true

    for row in eachrow(rows)
        z = Float32(row.z_um)
        if first_plane
            current_z = z
            first_plane = false
        else
            dz = current_z - z
            if abs(dz) >= 1.0f-6
                propagate!(ctx.wave_b, ctx.wave_a, dz, ctx, params)
                ctx.wave_a, ctx.wave_b = ctx.wave_b, ctx.wave_a
                current_z = z
            end
        end
        apply_disk!(ctx.wave_a, Float32(row.x_px), Float32(row.y_px), Float32(row.diameter_um), ctx, params)
    end
    return current_z
end

function render_frame!(rows::DataFrame, out_path::AbstractString, front_distance_um::Float32,
                       ctx::PropagationContext, params::RenderParams)
    current_z = apply_particles_to_wave!(rows, ctx, params)
    propagate!(ctx.wave_b, ctx.wave_a, front_distance_um + current_z, ctx, params)
    return wave_to_gray_png!(out_path, ctx.wave_b, ctx, params)
end

function render_frame_pair!(rows::DataFrame, c001_path::AbstractString, c002_path::AbstractString,
                            ctx::PropagationContext, params::RenderParams)
    current_z = apply_particles_to_wave!(rows, ctx, params)
    if params.c002_front_distance_um <= params.c001_front_distance_um
        near_path = c002_path
        far_path = c001_path
        near_front = params.c002_front_distance_um
        camera_gap = params.c001_front_distance_um - params.c002_front_distance_um
        near_key = "C002"
        far_key = "C001"
    else
        near_path = c001_path
        far_path = c002_path
        near_front = params.c001_front_distance_um
        camera_gap = params.c002_front_distance_um - params.c001_front_distance_um
        near_key = "C001"
        far_key = "C002"
    end

    propagate!(ctx.wave_b, ctx.wave_a, near_front + current_z, ctx, params)
    near_stats = wave_to_gray_png!(near_path, ctx.wave_b, ctx, params)
    propagate!(ctx.wave_a, ctx.wave_b, camera_gap, ctx, params)
    far_stats = wave_to_gray_png!(far_path, ctx.wave_a, ctx, params)
    return Dict(near_key => near_stats, far_key => far_stats)
end

function frame_file(frame::Integer)
    return lpad(string(frame + 1), 6, '0') * ".png"
end

function render_dataset(params::RenderParams)
    select_cuda_device!(params.device)
    labels_path = joinpath(params.out_dir, "labels", "particles.csv")
    labels = CSV.read(labels_path, DataFrame)
    if params.frames > 0
        labels = labels[labels.frame .< params.frames, :]
    end
    labels = labels[labels.frame .>= params.frame_start, :]
    if params.frame_end > 0
        labels = labels[labels.frame .< params.frame_end, :]
    end
    grouped = groupby(labels, :frame)

    out_dirs = Dict(
        "C001" => joinpath(params.out_dir, "01_bkgrem", "C001H001S0011"),
        "C002" => joinpath(params.out_dir, "01_bkgrem", "C002H001S0011"),
    )
    foreach(mkpath, values(out_dirs))
    ctx = make_context(params)

    stats = Dict{String, Any}(
        "generator" => "sequential_transmission_mask_angular_spectrum",
        "cuda_device_name" => CUDA.name(CUDA.device()),
        "cuda_visible_devices" => get(ENV, "CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_one_based_index" => params.device,
        "datlen" => params.datlen,
        "propagation_datlen" => params.propagation_datlen,
        "object_mask_canvas" => "particles are drawn on a 1536x1536 propagation plane; label x_px/y_px coordinates are shifted by +256 px so the nominal 1024x1024 observation plane is centered",
        "propagation_canvas" => "the 1536x1536 complex wavefront is carried through every particle plane and both camera planes without an intermediate crop",
        "saved_crop" => "central 1024x1024 pixels of the 1536x1536 propagated field",
        "dx_um" => Float64(params.dx_um),
        "wavelength_um" => Float64(params.wavelength_um),
        "background_gray" => Float64(params.background_gray),
        "frames_rendered" => length(grouped),
        "frame_start" => params.frame_start,
        "frame_end_exclusive" => params.frame_end,
        "camera_stats" => Dict{String, Any}(),
    )
    stats["camera_stats"]["C001"] = Vector{Dict{String, Any}}()
    stats["camera_stats"]["C002"] = Vector{Dict{String, Any}}()
    stats["camera_pair_rendering"] = Dict(
        "mode" => "shared_particle_wave_then_camera_gap_propagation",
        "c001_front_distance_um" => Float64(params.c001_front_distance_um),
        "c002_front_distance_um" => Float64(params.c002_front_distance_um),
        "camera_gap_um" => Float64(abs(params.c001_front_distance_um - params.c002_front_distance_um)),
    )
    prog = Progress(length(grouped); desc="Rendering C001/C002")
    for subdf in grouped
        frame = Int(subdf.frame[1])
        file = frame_file(frame)
        frame_stats = render_frame_pair!(
            DataFrame(subdf),
            joinpath(out_dirs["C001"], file),
            joinpath(out_dirs["C002"], file),
            ctx,
            params,
        )
        if frame in (0, length(grouped) - 1)
            for camera_id in ["C001", "C002"]
                push!(
                    stats["camera_stats"][camera_id],
                    merge(Dict("frame" => frame, "file" => file), frame_stats[camera_id]),
                )
            end
        end
        next!(prog)
    end
    open(joinpath(params.out_dir, "metadata", params.summary_name), "w") do io
        JSON.print(io, stats, 2)
        println(io)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    params = parse_args(ARGS)
    render_dataset(params)
end
