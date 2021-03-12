module NNO4F

using LinearAlgebra
using Random
using Statistics
using Printf
using PyPlot

using  Flux
import Flux.Optimise.update!
import Flux: gradient
import Flux.Losses.crossentropy
using  SetIntersectionProjection
using  InvertibleNetworks
using  Distributions
import Images
using CUDA

include("DataSetUtils.jl")
include("LossFunctions.jl")
include("NetworkOutputSampling.jl")
include("Train.jl")
include("PlotUtils.jl")
include("AugmentDataLabel.jl")

export TrOpts

mutable struct TrOpts
           eval_every      ::Int
           alpha           ::Union{AbstractFloat,Vector{AbstractFloat},Array{Any}}
           batchsize       ::Int
           use_gpu         ::Bool
           lossf
           lossg
           active_channels :: Array{Int}
           flip_dims       ::Array{Int}
           permute_dims    ::Array{Int}
           maxiter
           opt
           #active_z_slice :: Int
end

end #end module
