export Train

function Train(HN,LossFunction,alpha,batchsize,use_gpu,train_data,val_data,train_labels,val_labels,P,image_weights_train,image_weights_val,lossf,lossg,active_channels,flip_dims,maxiter,opt,active_z_slice)
  fval_train   = zeros(Float32,maxiter)
  fval_val     = zeros(Float32,maxiter)
  dc2val_train = zeros(Float32,maxiter)
  dc2val_val   = zeros(Float32,maxiter)
  IoU_hist_train = zeros(Float32,maxiter,2)
  IoU_hist_val   = zeros(Float32,maxiter,2)
  counterprint = 1

  for j=1:maxiter
    for k=1:batchsize
      rand_ind = randperm(length(train_data))[1]
      # Evaluate objective and gradients
      fval, dc2val = LossTotal(HN,alpha,use_gpu,train_data[rand_ind],train_labels[rand_ind],P[rand_ind],image_weights_train[rand_ind],lossf,lossg,active_channels,active_z_slice)
    end
    for p in get_params(HN)
        update!(opt, p.data, p.grad)
        #update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(HN)

    if mod(j, 50) == 0
      if isempty(train_labels)==false
        ioupos_train,iouneg_train = IoU(HN,train_data,train_labels)
        ioupos_train = ioupos_train[ioupos_train.>0.0]
        IoU_hist_train[counterprint,:] = [mean(ioupos_train) mean(iouneg_train)]
      end
      fvalepoch_train = 0.0
      dvalepoch_train = 0.0
      for i=1:length(train_data)
        f,d = LossTotal(HN,alpha,use_gpu,train_data[i],train_labels[i],P[i],image_weights_train[i],lossf,lossg,active_channels,active_z_slice)
        fvalepoch_train = fvalepoch_train + f
        dvalepoch_train = dvalepoch_train + d
      end
      fval_train[counterprint]   = fvalepoch_train/length(train_data)
      dc2val_train[counterprint] = dvalepoch_train/length(train_data)

      #validation data/labels
      if isempty(val_labels)==false
        ioupos_val,iouneg_val = IoU(HN,val_data,val_labels)
        ioupos_val=ioupos_val[ioupos_val.>0.0]
        IoU_hist_val[counterprint,:] = [mean(ioupos_val) mean(iouneg_val)]
      end
      fvalepoch_val = 0.0
      dvalepoch_val = 0.0
      for i=1:length(val_data)
        f,d = LossTotal(HN,0f0,use_gpu,val_data[i],val_labels[i],P[i],image_weights_val[i],lossf,lossg,active_channels,active_z_slice)
        fvalepoch_val = fvalepoch_val + f
        dvalepoch_val = dvalepoch_val + d
      end
      fval_val[counterprint]   = fvalepoch_val/length(val_data)
      dc2val_val[counterprint] = dvalepoch_val/length(val_data)
      print("Iteration: ", j, "; ftrain = ", fval_train[counterprint], "; dtrain = ", dc2val_train[counterprint], "; fval = ", fval_val[counterprint], "; dval = ", dc2val_val[counterprint], ";  IoUtrain:", IoU_hist_train[counterprint,:] , ";  IoUval:" , IoU_hist_val[counterprint,:], "\n")
      counterprint = counterprint + 1
      clear_grad!(HN)
    end

  end

  fval_train   = fval_train[1:counterprint-1]
  fval_val     = fval_val[1:counterprint-1]
  dc2val_train = dc2val_train[1:counterprint-1]
  dc2val_val   = dc2val_val[1:counterprint-1]
  IoU_hist_train = IoU_hist_train[1:counterprint-1,:]
  IoU_hist_val   = IoU_hist_val[1:counterprint-1,:]

  return fval_train, fval_val, dc2val_train, dc2val_val, IoU_hist_train, IoU_hist_val
end
