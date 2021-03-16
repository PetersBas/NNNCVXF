using PyPlot, MAT, Printf
using LinearAlgebra, Random, Statistics
using NNNCVXF #(https://github.com/PetersBas/NNNCVXF)
using Flux
using InvertibleNetworks #(https://github.com/slimgroup/InvertibleNetworks.jl)
using SetIntersectionProjection  #(https://github.com/slimgroup/SetIntersectionProjection.jl)

# Download data at:
# https://rslab.ut.ac.ir/documents/81960329/82034892/Hyperspectral_Change_Datasets.zip
# from https://rslab.ut.ac.ir/data ("the USA Dataset")


use_gpu = true

cd("/home/bpete23/Results/") #working directory
data_dir = "/home/bpete23/Data"

file = matopen(joinpath(data_dir,"USA_Change_Dataset.mat"));
data = zeros(Float32,307,241,154,2,1)
labels = zeros(Float32,307,241,154,2,1)

data[:,:,:,1,1] = read(file,"T1")
data[:,:,:,2,1] = read(file,"T2")

labels[:,:,66,1,1] = read(file,"Binary")
labels[:,:,66,2,1] = 1f0 .- labels[:,:,66,1,1]

#cut data so we can divide by 2 sufficiently many times
data = data[1:304,1:240,1:152,:,:]
labels = labels[1:304,1:240,1:152,:,:]

data = data[1:2:end,1:2:end,2:2:end,:,:]
labels = labels[1:2:end,1:2:end,2:2:end,:,:]

data = data[1:100,:,:,:,:]
labels = labels[1:100,:,:,:,:]

#is this normalized properly?>
#looks like incorrect, check initial vlaues
for i=1:size(data,4)
    data[:,:,:,i,1] .= data[:,:,:,i,1] .- mean(data[:,:,:,i,1]);
    #data[:,:,:,i,1] .= data[:,:,:,i,1] .- minimum(data[:,:,:,i,1]);
    data[:,:,:,i,1] .= data[:,:,:,i,1] ./ maximum(data[:,:,:,i,1]);
end

data = repeat(data,outer=(1,1,1,8,1))

n                  = size(labels)
active_channels    = [1,2]
#n_samples_per_iter = 40


#change data to a vector of examples
dataL    = Vector{Array{Float32,5}}(undef,1)
dataL[1] = data
data     = deepcopy(dataL)
dataL    = []

labelsL   = Vector{Any}(undef,1)
labelsL[1] = labels
labels     = deepcopy(labelsL)
labelsL    = []

#labels_V = Vector{Any}(undef,1)


architecture = ((0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12))
#architecture = ((0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12),(0,12))
k = 3   # kernel size
s = 1   # stride
p = 1   # padding
n_chan_in = size(data[1],4)
#(nx, ny)  = size(data[1])[1:2]
α = 0.2^2 #artificial time-step in the nonlinear telegraph equation discretization that is the neural network
#HN = H = NetworkHyperbolic(nx,ny,n_chan_in,1,architecture; α)
HN = H = NetworkHyperbolic3D(n_chan_in,architecture; α)

#set up one projector per channel to enforce constraints
#input comes as a tensor (matrix in 2D, 3D array for 3d data)
#output should be same size as input
pos_inds = findall(labels[1][:,:,33,1,1].>0)
pos_inds_select = shuffle(pos_inds)[1:20]

function Proj_bounds_card_ch1(input,pos_inds_select,max_card_pos)
  proj=deepcopy(input)
  proj[pos_inds_select] .= 1.1f0
  proj = reshape(project_cardinality!(vec(proj),round(Int,max_card_pos*prod(size(proj)[1:2]))),size(proj)[1:2])
  proj = reshape(project_bounds!(vec(proj),0.0f0,1.0f0),size(proj)[1:2])
  return proj
end

function Proj_bounds_card_ch2(input,pos_inds_select,max_card_neg)
  proj=deepcopy(input)
  proj[pos_inds_select] .= 0.0f0
  proj = reshape(project_cardinality!(vec(proj),round(Int,max_card_neg*prod(size(proj)[1:2]))),size(proj)[1:2])
  #proj = reshape(project_bounds!(vec(proj),0.0,1.0),size(x)[1:2])
  return proj
end


P    = Vector{Vector{Any}}(undef,1)
P[1] = Vector{Any}(undef,2)
P[1][1] = x -> Proj_bounds_card_ch1(x,pos_inds_select,0.45)
P[1][2] = x -> Proj_bounds_card_ch2(x,pos_inds_select,0.75)

#set up the training options
TrainOptions = TrOpts()
TrainOptions.eval_every      = 5
TrainOptions.batchsize       = 1
TrainOptions.use_gpu         = use_gpu
TrainOptions.lossf           = []
TrainOptions.lossg           = []
TrainOptions.active_channels = [1,2]
TrainOptions.flip_dims       = []
TrainOptions.permute_dims    = []

TrainOptions.alpha   = 1f0
TrainOptions.maxiter = 20#300
TrainOptions. opt    = Flux.Momentum(1f-4,0.9)

active_z_slice = 33

if use_gpu==true
  HN   = H |> gpu  #move network to use_gpu
  data = data|>gpu #move data to gpu
end

logs = Log() #create structure of logs of misfits

#fill unused data/labels with empty arrays
val_data      = [[]]
train_labels  = [[]]
val_labels    = [[]]
output_samples_train = [[]]
output_samples_val   = [[]]

#train the network; returns the loss, network parameters are updated inplace so HN is updated after training
logs = Train(HN,logs,TrainOptions,data,val_data,train_labels,val_labels,P,output_samples_train,output_samples_val,active_z_slice)

#plot the distance function per iteration
PlotHyperspectralLossConstrained(logs,TrainOptions.eval_every)

#plot data and results
PlotDataLabelPredictionHyperspectral(1,data,labels,HN,TrainOptions.active_channels,active_z_slice,pos_inds_select,"train")
