using NNNCVXF #(https://github.com/PetersBas/NNNCVXF)
using PyPlot
using SetIntersectionProjection #(https://github.com/slimgroup/SetIntersectionProjection.jl)
using InvertibleNetworks #(https://github.com/slimgroup/InvertibleNetworks.jl)
using LinearAlgebra
using Flux
import Flux: gradient
import Flux.Losses.crossentropy

# We use the CamVid data prepared by:
# https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

use_gpu = true

cd("/home/bpete23/Results/") #working directory
basepath = "/home/bpete23/Data"
#basepath = "/Users/BasP/Downloads/"

nr_output_chan = 3*16
train_data, train_labels, val_data, val_labels = LoadCamvid(basepath,nr_output_chan)

active_channels = [1,2]

n_pix_per_class = 8
output_samples_train = Vector{Array{Float32,2}}(undef,length(train_data))
output_samples_val   = Vector{Array{Float32,2}}(undef,length(val_data))
for i=1:length(output_samples_train)
  output_samples_train[i] = SparseClassSamples(train_labels[i],active_channels,n_pix_per_class)
end
for i=1:length(output_samples_val)
  output_samples_val[i] = SparseClassSamples(val_labels[i],active_channels,n_pix_per_class)
end


mutable struct grid #grid structure for discrete differential operators
    n
    d
end
compgrid = grid(size(train_labels[1])[1:2], (1, 1))

noise_factor = 0.33
TV_list_train = GetTVvalues(grid,train_labels,noise_factor,compgrid)

#set up one projector per example
options=PARSDMM_options()
options.FL=Float32
options=default_PARSDMM_options(options,options.FL)
options.maxit = 1000

#using these (below) more accurate projections onto the anisotropic total-variation constraint set_type
#give slightly better segmentation reguls, but it takes significantly more computational time
options.feas_tol     = 0.0001
options.obj_tol      = 0.0001
options.evol_rel_tol = 0.00001

BLAS.set_num_threads(4)

constraint = Vector{SetIntersectionProjection.set_definitions}()

m_min        = 0.0
m_max        = 1f0#dummy value that will be replaced in the loop below true_TV_list[i]
set_type     = "l1"
TD_OP        = "TV"
app_mode     = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

#set up constraints, precompute some things and define projector
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,compgrid,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,compgrid,options)

P_sub_list = Vector{Any}(undef,length(train_data))
P_train    = Vector{Any}(undef,length(train_data))
P_val      = Vector{Any}(undef,length(val_data))
for i=1:length(train_data)
  P_sub_list[i] = Vector{Any}(undef,1)
  P_sub_list[i][1] = x-> project_l1_Duchi!(x,Float32(TV_list_train[i]))
end
for i=1:length(train_data)
  P_train[i] = Vector{Any}(undef,2)
  function prj(x)
    os=size(x)
    (x,dummy1,dummy2,dymmy3) = PARSDMM(deepcopy(vec(x)), AtA, TD_OP, set_Prop, P_sub, compgrid, options)
    return reshape(x,os)
  end
  P_train[i][1] = x -> prj(x)
  P_train[i][2] = x -> x #don't apply any projection to 2nd channel (not necessary for good results in this example)
end

#No projection for validation set
for i=1:length(val_data)
  P_val[i] = Vector{Any}(undef,2)
  P_val[i][1] = x-> x
  P_val[i][2] = x-> x
end

#Identity mapping (in place of the projector in case we don't want to project output for training data)
P_I = Vector{Any}(undef,length(train_data))
for i=1:length(train_data)
  P_I[i] = Vector{Any}(undef,2)
  P_I[i][1] = x-> x
  P_I[i][2] = x-> x
end

architecture = ((0, 48), (0, 48), (0, 48),(0, 48), (0, 48), (0, 48),(0, 48),(0, 48),(0, 48),(0,48),(0, 48),(0,48),(0, 48),(0, 48))

k = 3   # kernel size
s = 1   # stride
p = 1   # padding
n_chan_in = size(train_data[1],3)
#(nx, ny)  = size(train_data[1])[1:2]
α = 0.3^2 #artificial time-step in the nonlinear telegraph equation discretization that is the neural network
#HN = H = NetworkHyperbolic(nx,ny,n_chan_in,1,architecture; α)
HN = H = NetworkHyperbolic(n_chan_in,architecture; α)

# define loss functions for the labels and corresponding gradients
CESM   = (x,y)-> crossentropy(softmax(x,dims=3), y; dims=3, agg=sum,ϵ=eps(x[1]))
wCESM  = (x,y,c)-> crossentropy(softmax(x,dims=3), y; dims=3, agg=x->sum(c .* x),ϵ=eps(x[1]))
gCESM  = (x,y)-> gradient(CESM,x,y)
gwCESM = (x,y,c)-> gradient(wCESM,x,y,c)


########## EXPERIMENT: Point annotations only #################################
TrainOptions = TrOpts()
TrainOptions.eval_every      = 50
TrainOptions.batchsize       = 4
TrainOptions.use_gpu         = use_gpu
TrainOptions.lossf           = wCESM
TrainOptions.lossg           = gwCESM
TrainOptions.active_channels = [1,2]
TrainOptions.flip_dims       = []
TrainOptions.permute_dims    = []

alpha   = [0f0 0f0]
maxiter = [1200 800]
opt     = [Flux.ADAM(1f-3) Flux.ADAM(1f-4)]

  #set all data and labels to GPU
  if TrainOptions.use_gpu==true
    HN         = H |> gpu
    val_data   = val_data|>gpu
    train_data = train_data|>gpu
  end

  logs = Log()

  for i=1:length(maxiter)
    TrainOptions.alpha   = alpha[i]
    TrainOptions.opt     = opt[i]
    TrainOptions.maxiter = maxiter[i]
    logs = Train(HN,logs,TrainOptions,train_data,val_data,train_labels,val_labels,P_I,output_samples_train,output_samples_val)
  end

  if TrainOptions.use_gpu==true
    train_data  = train_data |>cpu
    val_data    = val_data |>cpu
    HN          = HN |>cpu
  end

  if isdir("camvid_unconstrained") == true
    cd("camvid_unconstrained")
  else
    mkdir("camvid_unconstrained")
    cd("camvid_unconstrained")
  end

  PlotCamvidLossUnconstrained(logs,TrainOptions.eval_every)

  #plot training result
  plt_ind = 10
  tag     = "train"
  PlotDataLabelPredictionCamvid(plt_ind,train_data,train_labels,HN,TrainOptions.active_channels,tag)

  #plot validation result
  plt_ind = 3
  tag     = "validation"
  PlotDataLabelPredictionCamvid(plt_ind,val_data,val_labels,HN,TrainOptions.active_channels,tag)

  cd("..")

########## EXPERIMENT: point_annotations_+_TV_constraints #####################
  #set all data and labels to GPU
  if use_gpu==true
    HN         = H |> gpu
    val_data   = val_data|>gpu
    train_data = train_data|>gpu
  end

  TrainOptions = TrOpts()
  TrainOptions.eval_every      = 50
  TrainOptions.batchsize       = 4
  TrainOptions.use_gpu         = true
  TrainOptions.lossf           = wCESM
  TrainOptions.lossg           = gwCESM
  TrainOptions.active_channels = [1,2]
  TrainOptions.flip_dims       = []
  TrainOptions.permute_dims    = []

  alpha   = [0f0 0f0 1f-5 5f-5 1f-4 5f-4 1f-3 5f-3 1f-2]
  maxiter = [1200 800 400 400 400 400 800 800 800]
  opt     = [Flux.ADAM(1f-3) Flux.ADAM(1f-4) Flux.ADAM(1f-4) Flux.ADAM(1f-4) Flux.ADAM(1f-4) Flux.ADAM(1f-4) Flux.ADAM(1f-4) Flux.ADAM(1f-4) Flux.ADAM(1f-4)]

  logs = Log()

  for i=1:2
    TrainOptions.alpha   = alpha[i]
    TrainOptions.opt     = opt[i]
    TrainOptions.maxiter = maxiter[i]
    logs = Train(HN,logs,TrainOptions,train_data,val_data,train_labels,val_labels,P_I,output_samples_train,output_samples_val)
  end
  for i=3:length(maxiter)
    TrainOptions.alpha   = alpha[i]
    TrainOptions.opt     = opt[i]
    TrainOptions.maxiter = maxiter[i]
    logs = Train(HN,logs,TrainOptions,train_data,val_data,train_labels,val_labels,P_train,output_samples_train,output_samples_val)
  end

  if use_gpu==true
    train_data  = train_data |>cpu
    val_data    = val_data |>cpu
    HN          = HN |>cpu
  end

  if isdir("camvid_constrained") == true
    cd("camvid_constrained")
  else
    mkdir("camvid_constrained")
    cd("camvid_constrained")
  end

  PlotCamvidLossConstrained(logs,TrainOptions.eval_every)

  #plot training result
  plt_ind = 10
  tag     = "train"
  PlotDataLabelPredictionCamvid(plt_ind,train_data,train_labels,HN,TrainOptions.active_channels,tag)

  #plot validation result
  plt_ind = 3
  tag     = "validation"
  PlotDataLabelPredictionCamvid(plt_ind,val_data,val_labels,HN,TrainOptions.active_channels,tag)

  cd("..")
