export Train_AD#, TrainStatusPrint

function Train_AD(K,loss,TrOpts,train_data,val_data,train_labels,val_labels,image_weights_train,image_weights_val,active_z_slice=[])
  flip_dims=TrOpts.flip_dims; permute_dims=TrOpts.permute_dims; active_channels = TrOpts.active_channels
  f_train = zeros(TrOpts.maxiter); f_val = zeros(TrOpts.maxiter)
  for j=1:TrOpts.maxiter
    #training set
    rand_ind = randperm(length(train_data))[1]
    if (isempty(flip_dims) && isempty(permute_dims)) == false
      X0, label, image_weights = AugmentDataLabel(train_data[rand_ind], train_labels[rand_ind], image_weights_train[rand_ind],flip_dims,permute_dims)#optional: augment data
    else
      X0            = deepcopy(train_data[rand_ind])
      label         = deepcopy(train_labels[rand_ind])
      image_weights = deepcopy(image_weights_train[rand_ind])
    end

  #optional: randomly set parts of the gradient to zero
  rand_mask = ones(Float32,size(image_weights))
  if TrOpts.rand_grad_perc_zero!=0.0
    rand_inds = CartesianIndices(size(image_weights))
    rand_inds = shuffle(vec(rand_inds))[1:round(Int,TrOpts.rand_grad_perc_zero*prod(size(rand_inds)))]
    rand_mask[rand_inds] .= 0f0
    #rand_mask = reshape(rand_mask,size(image_weights))
  end

    # Evaluate objective and gradients
    if isempty(active_z_slice)==false
      out = Flux.withgradient(loss, X0, K, label[:,:,active_z_slice,active_channels,1]|>gpu, image_weights.*rand_mask|>gpu);
    else
      out = Flux.withgradient(loss, X0, K, label[:,:,active_channels,1]|>gpu, image_weights.*rand_mask|>gpu);
    end
      f_train[j] = out[1]
    for h = 1:length(K)
      Flux.update!(TrOpts.opt, K[h], out[2][2][h])#grad_K = out[2][2]
    end

    #evaluate
    if mod(j, TrOpts.eval_every) == 0 || j==1
      if isempty(active_z_slice)==false
        out = Flux.withgradient(loss, train_data[1], K, train_labels[1][:,:,active_z_slice,active_channels,1]|>gpu, image_weights_val[1]|>gpu);
      else
        out = Flux.withgradient(loss, train_data[1], K, train_labels[1][:,:,active_channels,1]|>gpu, image_weights_val[1]|>gpu);
      end
      f_val[j] = out[1]
      println([f_train[j], f_val[j]])
    end
  end

  return K, f_train, f_val
  #return logs
end
