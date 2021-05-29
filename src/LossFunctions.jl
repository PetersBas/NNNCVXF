export Dist2Set, LossTotal, IoU, g_CQ, MultipleSplitFeasCQ_fun_grad

function Dist2Set(input,P,TrOpts)
  #active_z_slice is only used for hyperspectral imaging where we map a 3D or 4D
  #data volume to a 2D map of the earth. this 2D map is located at slice number
  #active_z_slice in a x-y-z-nchan-n_ex tensor

  #initialize some stuff
  size_input = size(input)
  pen_grad   = zeros(Float32,size(input))
  pen_value  = 0f0

  #use softmax of final network state for constraining
  #the output for segmentation problems (don't use softmax for non-linear regression)
  #input      = softmax(Array(input)[:,:,TrainOpts.channels,:],dims=3)

  input      = softmax(input[:,:,TrOpts.active_channels,:],dims=3)

  for j in TrOpts.active_channels #loop over channels, each channel has a (different) corresponding projector
    input_slice        = input[:,:,j,1]
    pen_value_slice    = 0f0 #
    P_input            = P[j](input_slice) #project onto constraint set (can be an intersection)
    pen_value_slice    = 0.5f0*norm(vec(P_input)-vec(input_slice))^2 #squared point-to-set distance functino
    pen_grad[:,:,j,1] .= input_slice .- P_input # gradient of point-to-set distance function

    pen_value = pen_value + pen_value_slice #accumulate penalty value (loss) over channels
  end #end channels loop

  #at this point,the softmax of the network output has been projected onto the intersection. Now we need to 'complete' the gradient
  #via the last part of the chain-rule for distance-function-squared and softmax
  pen_grad[:,:,TrOpts.active_channels,1] .= ∇softmax(pen_grad[:,:,TrOpts.active_channels,1], input[:,:,TrOpts.active_channels,1]; dims=3)

  return pen_value, pen_grad
end

function g_CQ(input,TD_OP,P_sub,alpha_CQ,TrOpts)

  #initialize some stuff
  size_input = size(input)
  pen_grad   = zeros(Float32,size(input))
  pen_value  = 0f0

  #use softmax of final network state for constraining
  #the output for segmentation problems (don't use softmax for non-linear regression)
  input      = softmax(input[:,:,TrOpts.active_channels,:],dims=3)

  for j in TrOpts.active_channels #loop over channels, each channel has a (different) corresponding projector
    input_slice        = input[:,:,j,1]
    pen_value_slice    = 0f0 #
    #P_input            = P[j](input_slice) #project onto constraint set (can be an intersection)
    #pen_value_slice    = 0.5f0*norm(vec(P_input)-vec(input_slice))^2 #squared point-to-set distance functino
    f_temp,g_temp      = MultipleSplitFeasCQ_fun_grad(vec(deepcopy(input_slice)),TD_OP[j],P_sub[j],alpha_CQ[j])
    pen_grad[:,:,j,1] .= reshape(g_temp,size(input_slice)) # gradient of point-to-set distance function

    pen_value = pen_value + f_temp #accumulate penalty value (loss) over channels
  end #end channels loop

  #at this point,the softmax of the network output has been projected onto the intersection. Now we need to 'complete' the gradient
  #via the last part of the chain-rule for distance-function-squared and softmax
  pen_grad[:,:,TrOpts.active_channels,1] .= ∇softmax(pen_grad[:,:,TrOpts.active_channels,1], input[:,:,TrOpts.active_channels,1]; dims=3)

  return pen_value, pen_grad
end

function MultipleSplitFeasCQ_fun_grad(x::AbstractVector{T},A,P_sub,alpha::AbstractVector) where T

# returns the gradient of the sum of squared distance functions for the split-feasibility
#problem: A_i x = y_i , y_i \in C_i
# f = \sum_{i=1}^p \alpha_i/2 \| P_C_i (A_i x) - A_i x \|_2^2
# g = \sum_{i=1}^p \alpha_i/2 A_i^t (A_i x - P_C_i (A_i x) )

#serial implementation
#(each projection or mat-vec can still use multiple threads)

f    = zeros(T,length(alpha))#allocate function value per constraint/linear operator
ng   = zeros(T,length(alpha))#allocate norm of gradient per constraint/linear operator
grad = zeros(T,length(x))

for i = 1:length(A)
  y     = A[i]*x
  PcAx  = P_sub[i](deepcopy(y))
  f[i]  = norm(PcAx - y,2).^2
  grad .= grad .+ alpha[i].*A[i]'*(y.-PcAx)
end

return sum(f),grad
end

function LossTotal(HN,TrOpts,X0::AbstractArray{T, N},label,P,image_weights,active_z_slice::Int) where {T, N}
  #loss for hyperspectral imaging with 5d input
  alpha=TrOpts.alpha; use_gpu=TrOpts.use_gpu; lossf=TrOpts.lossf; lossg=TrOpts.lossg; flip_dims=TrOpts.flip_dims; permute_dims=TrOpts.permute_dims; active_channels=TrOpts.active_channels;

    if (isempty(flip_dims) && isempty(permute_dims)) == false
      X0, label, image_weights = AugmentDataLabel(X0, label, image_weights,TrOpts)#optional: augment data
    end

    Y_curr, Y_new, lgdet = HN.forward(X0,X0)
    if use_gpu == true
      Y_new = Y_new|>cpu
    end

    if isempty(label)==false
        lval         = lossf(Y_new[:,:,active_z_slice,active_channels,1],label[:,:,active_z_slice,active_channels,1],image_weights)
        (grad,dummy) = lossg(Y_new[:,:,active_z_slice,active_channels,1],label[:,:,active_z_slice,active_channels,1],image_weights)
      else
        lval = 0.0
    end

  if alpha>0
    dc2,dc2_grad = Dist2Set(Y_new[:,:,active_z_slice,:,:],P,TrOpts)


    #grad  = grad + alpha*dc2_grad[:,:,active_channels,1]
  else
    dc2 = 0f0
  end

  if isempty(label)==false
      grad  = grad + alpha*dc2_grad[:,:,active_z_slice,active_channels,1]
  elseif isempty(label)==true
      n = size(dc2_grad)
      #dc2_grad = reshape(dc2_grad,n[1],n[2],1,n[3],1)
      grad  = alpha*dc2_grad[:,:,active_channels,1]
  end

   ΔY_curr= zeros(Float32,size(Y_new))
   ΔY_curr[:,:,active_z_slice,active_channels,1] .= grad
   ΔY_new = zeros(Float32,size(Y_new))

   if use_gpu == true
     Y_new   = Y_new|>gpu
     ΔY_curr = ΔY_curr|>gpu
     ΔY_new  = ΔY_new|>gpu
   end
   ΔY_curr, ΔY_new, Y_curr, Y_new = HN.backward(ΔY_curr, ΔY_new, Y_curr, Y_new)

   return lval, dc2
end

function LossTotal(HN,TrOpts,X0::AbstractArray{T, N},label,P,image_weights,active_z_slice::Array{Any,1},P_mode::String,TD_OP,P_sub,alpha_CQ) where {T, N}
    alpha=TrOpts.alpha; use_gpu=TrOpts.use_gpu; lossf=TrOpts.lossf; lossg=TrOpts.lossg; active_channels=TrOpts.active_channels;

    Y_curr, Y_new, lgdet = HN.forward(X0,X0)
    if use_gpu == true
      Y_new = Y_new|>cpu
    end

    #initialize gradient
    if N==4
      grad = zeros(Float32,size(Y_new[:,:,active_channels,1]))
    elseif N==5
      grad = zeros(Float32,size(Y_new[:,:,:,active_channels,1]))
    end

    if isempty(label)==false
      if N==4
        lval         = TrOpts.lossf(Y_new[:,:,active_channels,1],label,image_weights)
        (grad,dummy) = TrOpts.lossg(Y_new[:,:,active_channels,1],label,image_weights)
      elseif N==5
        lval         = TrOpts.lossf(Y_new[:,:,:,active_channels,1],label,image_weights)
        (grad,dummy) = TrOpts.lossg(Y_new[:,:,:,active_channels,1],label,image_weights)
      end
    else
      lval = 0.0
    end

  if (alpha[1][1]>0 || alpha_CQ[1][1]>0)==true
    if P_mode == "Proj_intersection"
      dc2,dc2_grad = Dist2Set(Y_new,P,active_channels)
      dc2_grad .= alpha.*dc2_grad
    elseif P_mode == "g_CQ"
      dc2,dc2_grad = g_CQ(Y_new,TD_OP,P_sub,alpha_CQ,TrOpts)
    end

    if N==4
      grad  = grad + dc2_grad[:,:,active_channels,1]
    elseif N==5
      grad  = grad + dc2_grad[:,:,:,active_channels,1]
    end
  else
    dc2 = 0f0
  end

  #optional: randomly set parts of the gradient to zero
  if TrOpts.rand_grad_perc_zero!=0.0
    rand_mask = CartesianIndices(size(grad)[1:end-2])
    rand_mask = shuffle(vec(rand_mask))[1:round(Int,TrOpts.rand_grad_perc_zero*prod(size(rand_mask)))]
    if N==4;grad[rand_mask,:,:].=0f0;elseif N==5;grad[rand_mask,:,:,:].=0f0;end;
  end

   ΔY_curr= zeros(Float32,size(Y_new))
   if N==4
     ΔY_curr[:,:,active_channels,1] .= grad
   elseif N==5
     ΔY_curr[:,:,:,active_channels,1] .= grad
   end
   ΔY_new = zeros(Float32,size(Y_new))

   if use_gpu == true
     Y_new   = Y_new|>gpu
     ΔY_curr = ΔY_curr|>gpu
     ΔY_new  = ΔY_new|>gpu
   end
   ΔY_curr, ΔY_new, Y_curr, Y_new = HN.backward(ΔY_curr, ΔY_new, Y_curr, Y_new)

   return lval, dc2
end

function IoU(HN,data,labels)
threshold = 0.65
IoU_pos = zeros(length(data))
IoU_neg = zeros(length(data))

  for i=1:length(data)
    #prediction = CAE(ϕ,KnetArray(data[i]),h)
    ~, prediction, ~ = HN.forward(data[i],data[i])
    prediction = prediction |> cpu
    prediction[:,:,1:2,1].=softmax(prediction[:,:,1:2,1],dims=3);


    pred_thres = zeros(Int,size(prediction)[1:2])
    pos_inds   = findall(prediction[:,:,1] .> threshold)
    pred_thres[pos_inds] .= 1

    #plot pixel accuracy per time-slice
    prediction_for_acc = pred_thres[5:end-4,5:end-4]
    mask_for_acc       = labels[i][5:end-4,5:end-4,1]

    pos_pred_inds = findall(prediction_for_acc.==1)
    neg_pred_inds = findall(prediction_for_acc.==0)

    true_pos_inds = findall(mask_for_acc .== 1)
    true_neg_inds = findall(mask_for_acc .== 0)

    #IoU
    IoU_pos[i] = length(intersect(pos_pred_inds,true_pos_inds))/length(union(pos_pred_inds,true_pos_inds))
    IoU_neg[i] = length(intersect(neg_pred_inds,true_neg_inds))/length(union(neg_pred_inds,true_neg_inds))
  end

return IoU_pos, IoU_neg
end

#Loss for applications other than Hyperspectral. Uses projection onto intersection.
function LossTotal(HN,TrOpts,X0::AbstractArray{T, N},label,P,image_weights,active_z_slice::Array{Any,1}) where {T, N}
    alpha=TrOpts.alpha; use_gpu=TrOpts.use_gpu; lossf=TrOpts.lossf; lossg=TrOpts.lossg; active_channels=TrOpts.active_channels;

    Y_curr, Y_new, lgdet = HN.forward(X0,X0)
    if use_gpu == true
      Y_new = Y_new|>cpu
    end

    #initialize gradient
    if N==4
      grad = zeros(Float32,size(Y_new[:,:,active_channels,1]))
    elseif N==5
      grad = zeros(Float32,size(Y_new[:,:,:,active_channels,1]))
    end

    if isempty(label)==false
      if N==4
        lval         = lossf(Y_new[:,:,active_channels,1],label,image_weights)
        (grad,dummy) = lossg(Y_new[:,:,active_channels,1],label,image_weights)
      elseif N==5
        lval         = lossf(Y_new[:,:,:,active_channels,1],label,image_weights)
        (grad,dummy) = lossg(Y_new[:,:,:,active_channels,1],label,image_weights)
      end
    else
      lval = 0.0
    end
    # # #take equal number of random pixels from each class for the gradient
    # pos_inds = findall(label[:,:,1,1].==1)
    # neg_inds = findall(label[:,:,1,1].==0)
    # npix = min(length(pos_inds),length(neg_inds))
    # npix = round(Int,npix/2)
    # pos_inds_select = shuffle(pos_inds)[1:npix]
    # neg_inds_select = shuffle(neg_inds)[1:npix]
    # random_mask = zeros(Float32,size(grad))
    # random_mask[pos_inds_select,1:2,1].=1
    # random_mask[neg_inds_select,1:2,1].=1

    #grad .= grad.*random_mask

  if alpha>0
    #if P_mode == "Proj_intersection"
      dc2,dc2_grad = Dist2Set(Y_new,P,TrOpts)
    #elseif P_mode == "g_CQ"
    #  dc2,dc2_grad = g_CQ(Y_new,TD_OP,P_sub,alpha_CQ,active_channels)
    #end
    # if (norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) > 10f0
    #   @warn "(norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) > 10f0"
    # elseif (norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) < 0.1f0
    #   @warn "(norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) < 0.1f0"
    # end
    if N==4
      grad  = grad + alpha*dc2_grad[:,:,active_channels,1]
    elseif N==5
      grad  = grad + alpha*dc2_grad[:,:,:,active_channels,1]
    end
  else
    dc2 = 0f0
  end

  #optional: randomly set parts of the gradient to zero
  if TrOpts.rand_grad_perc_zero!=0.0
    rand_mask = CartesianIndices(size(grad)[1:end-2])
    rand_mask = shuffle(vec(rand_mask))[1:round(Int,TrOpts.rand_grad_perc_zero*prod(size(rand_mask)))]
    if N==4;grad[rand_mask,:,:].=0f0;elseif N==5;grad[rand_mask,:,:,:].=0f0;end;
  end

   ΔY_curr= zeros(Float32,size(Y_new))
   if N==4
     ΔY_curr[:,:,active_channels,1] .= grad
   elseif N==5
     ΔY_curr[:,:,:,active_channels,1] .= grad
   end
   ΔY_new = zeros(Float32,size(Y_new))

   if use_gpu == true
     Y_new   = Y_new|>gpu
     ΔY_curr = ΔY_curr|>gpu
     ΔY_new  = ΔY_new|>gpu
   end
   ΔY_curr, ΔY_new, Y_curr, Y_new = HN.backward(ΔY_curr, ΔY_new, Y_curr, Y_new)

   return lval, dc2
end
