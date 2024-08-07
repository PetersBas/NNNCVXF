module NNNCVXF

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
using Parameters
using ChainRules

include("DataSetUtils.jl")
include("LossFunctions.jl")
include("NetworkOutputSampling.jl")
include("Train.jl")
include("Train_AD.jl")
include("PlotUtils.jl")
include("AugmentDataLabel.jl")
include("ConstraintSetupHelperFunctions.jl")

export TrOpts, Log

@with_kw  mutable struct TrOpts
           eval_every      ::Int  = 5
           alpha           ::Union{AbstractFloat,Vector{AbstractFloat},Array{Any}} = []
           batchsize       ::Int  = 1
           use_gpu         ::Bool = true
           lossf           ::Any  = []
           lossg           ::Any  = []
           active_channels ::Array{Int}  = [1]
           flip_dims       ::Array{Int}  = []
           permute_dims    ::Array{Int}  = []
           maxiter         ::Int         = 10
           opt             ::Any         = Flux.ADAM(1f-3)
           rand_grad_perc_zero  ::Float32     = 0f0
end

@with_kw mutable struct Log
  train     = []
  val       = []
  dc2_train = []
  dc2_val   = []
  IoU_train = []
  IoU_val   = []
end

end #end module
