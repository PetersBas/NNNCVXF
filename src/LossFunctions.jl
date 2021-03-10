export Dist2Set, LossTotal, IoU

function Dist2Set(input,P,active_channels)
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

  input      = softmax(input[:,:,active_channels,:],dims=3)

  for j in active_channels #loop over channels, each channel has a (different) corresponding projector
    input_slice        = input[:,:,j,1]
    pen_value_slice    = 0f0 #
    P_input            = P[j](input_slice) #project onto constraint set (can be an intersection)
    pen_value_slice    = 0.5f0*norm(vec(P_input)-vec(input_slice))^2 #squared point-to-set distance functino
    pen_grad[:,:,j,1] .= input_slice .- P_input # gradient of point-to-set distance function

    pen_value = pen_value + pen_value_slice #accumulate penalty value (loss) over channels
  end #end channels loop

  #at this point,the softmax of the network output has been projected onto the intersection. Now we need to 'complete' the gradient
  #via the last part of the chain-rule for distance-function-squared and softmax
  pen_grad[:,:,active_channels,1] .= ∇softmax(pen_grad[:,:,active_channels,1], input[:,:,active_channels,1]; dims=3)

  return pen_value, pen_grad
end

function g_CQ(input,TD_OP,P_sub,alpha_CQ,active_channels)
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

  input      = softmax(input[:,:,active_channels,:],dims=3)

  for j in active_channels #loop over channels, each channel has a (different) corresponding projector
    input_slice        = input[:,:,j,1]
    pen_value_slice    = 0f0 #
    #P_input            = P[j](input_slice) #project onto constraint set (can be an intersection)
    #pen_value_slice    = 0.5f0*norm(vec(P_input)-vec(input_slice))^2 #squared point-to-set distance functino
    f_temp,g_temp = MultipleSplitFeasCQ_fun_grad(vec(deepcopy(input_slice)),TD_OP[j],P_sub[j],alpha)
    pen_grad[:,:,j,1] .= g_temp # gradient of point-to-set distance function

    pen_value = pen_value + f_temp #accumulate penalty value (loss) over channels
  end #end channels loop

  #at this point,the softmax of the network output has been projected onto the intersection. Now we need to 'complete' the gradient
  #via the last part of the chain-rule for distance-function-squared and softmax
  pen_grad[:,:,active_channels,1] .= ∇softmax(pen_grad[:,:,active_channels,1], input[:,:,active_channels,1]; dims=3)

  return pen_value, pen_grad
end

function LossTotal(HN,alpha,use_gpu,X0::AbstractArray{T, N},label,P,image_weights,lossf,lossg,active_channels,active_z_slice::Int,flip_dims,permute_dims) where {T, N}
  #loss for hyperspectral imaging with 5d input
    if (isempty(flip_dims) && isempty(permute_dims)) == false
      X0, label, image_weights = AugmentDataLabel(X0, label, image_weights,flip_dims,permute_dims)#optional: augment data
    end

    Y_curr, Y_new, lgdet = HN.forward(X0,X0)

    if isempty(label)==false
        lval         = lossf(Y_new[:,:,active_z_slice,active_channels,1],label,image_weights)
        (grad,dummy) = lossg(Y_new[:,:,active_z_slice,active_channels,1],label,image_weights)
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
    if use_gpu == true
      Y_new = Y_new|>cpu
    end
    dc2,dc2_grad = Dist2Set(Y_new[:,:,active_z_slice,:,:],P,active_channels)
    # if (norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) > 10f0
    #   @warn "(norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) > 10f0"
    # elseif (norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) < 0.1f0
    #   @warn "(norm(alpha*dc2_grad[:,:,active_channels,1])/norm(grad)) < 0.1f0"
    # end

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

function LossTotal(HN,alpha,use_gpu,X0::AbstractArray{T, N},label,P,image_weights,lossf,lossg,active_channels,active_z_slice::Array{Any,1},flip_dims,permute_dims,P_mode::String,TD_OP,P_sub,alpha_CQ) where {T, N}

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
    if P_mode == "Proj_intersection"
      dc2,dc2_grad = Dist2Set(Y_new,P,active_channels)
    elseif P_mode == "g_CQ"
      dc2,dc2_grad = g_CQ(Y_new,TD_OP,P_sub,alpha_CQ,active_channels)
    end
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

function LossTotal(HN,alpha,use_gpu,X0::AbstractArray{T, N},label,P,image_weights,lossf,lossg,active_channels,active_z_slice::Array{Any,1},flip_dims,permute_dims) where {T, N}

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
      dc2,dc2_grad = Dist2Set(Y_new,P,active_channels)
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
