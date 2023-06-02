export Train, TrainStatusPrint

function Train(HN,logs,TrOpts,train_data,val_data,train_labels,val_labels,P,image_weights_train,image_weights_val,active_z_slice::Union{Array{Any,1},Int}=[],P_val=[])

  for j=1:TrOpts.maxiter
    for k=1:TrOpts.batchsize
      rand_ind = randperm(length(train_data))[1]
      # Evaluate objective and gradients
      fval, dc2val = LossTotal(HN,TrOpts,train_data[rand_ind],train_labels[rand_ind],P[rand_ind],image_weights_train[rand_ind],active_z_slice)
    end
    for p in get_params(HN)
        update!(TrOpts.opt, p.data, p.grad)
        #update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(HN)

    if mod(j, TrOpts.eval_every) == 0 || j==1
      if isempty(train_labels[1])==false
        ioupos_train,iouneg_train = IoU(HN,train_data,train_labels)
        ioupos_train   = ioupos_train[ioupos_train.>0.0]
        logs.IoU_train = cat(logs.IoU_train,[mean(ioupos_train) mean(iouneg_train)],dims=1)
      end
      fvalepoch_train = 0.0
      dvalepoch_train = 0.0
      for i=1:length(train_data)
        f,d = LossTotal(HN,TrOpts,train_data[i],train_labels[i],P[i],image_weights_train[i],active_z_slice)
        fvalepoch_train = fvalepoch_train + f
        dvalepoch_train = dvalepoch_train + d
      end
      logs.train     = vcat(logs.train,fvalepoch_train/length(train_data))
      logs.dc2_train = vcat(logs.dc2_train,dvalepoch_train/length(train_data))

      #validation data/labels
      if isempty(val_labels[1])==false && isempty(P_val)==false
        ioupos_val,iouneg_val = IoU(HN,val_data,val_labels)
        ioupos_val    = ioupos_val[ioupos_val.>0.0]
        logs.IoU_val  = cat(logs.IoU_val,[mean(ioupos_val) mean(iouneg_val)],dims=1)

        fvalepoch_val = 0.0
        dvalepoch_val = 0.0
        for i=1:length(val_data)
          #f,d = LossTotal(HN,TrOpts,val_data[i],val_labels[i],P[i],image_weights_val[i],active_z_slice)
          f,d = LossTotal(HN,TrOpts,val_data[i],val_labels[i],P_val[i],image_weights_val[i],active_z_slice)
          fvalepoch_val = fvalepoch_val + f
          dvalepoch_val = dvalepoch_val + d
        end
        logs.val     = vcat(logs.val,fvalepoch_val/length(val_data))
        logs.dc2_val = vcat(logs.dc2_val,dvalepoch_val/length(val_data))
      end

      TrainStatusPrint(j,logs)
      #print("Iteration: ", j, "; ftrain = ", logs.train[end], "; dtrain = ", logs.dc2_train[end], "; fval = ", logs.val[end], "; dval = ", logs.dc2_val[end], ";  IoUtrain:", logs.IoU_train[end,:] , ";  IoUval:" , logs.IoU_val[end,:], "\n")
      clear_grad!(HN)
    end

  end

  return logs
end

function Train(HN,logs,TrOpts,train_data,val_data,train_labels,val_labels,P,image_weights_train,image_weights_val,active_z_slice,P_val,P_mode::String,TD_OP,P_sub,alpha_CQ,TD_OP_val,P_sub_val,alpha_CQ_val)

  for j=1:TrOpts.maxiter
    for k=1:TrOpts.batchsize
      rand_ind = randperm(length(train_data))[1]
      # Evaluate objective and gradients
      #println(string("example-",rand_ind))
      fval, dc2val = LossTotal(HN,TrOpts,train_data[rand_ind],train_labels[rand_ind],P[rand_ind],image_weights_train[rand_ind],active_z_slice,P_mode,TD_OP[rand_ind],P_sub[rand_ind],alpha_CQ[rand_ind])
    end
    for p in get_params(HN)
        update!(TrOpts.opt, p.data, p.grad)
        #update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(HN)

    if mod(j, TrOpts.eval_every) == 0 || j==1
      if isempty(train_labels[1])==false
        ioupos_train,iouneg_train = IoU(HN,train_data,train_labels)
        ioupos_train   = ioupos_train[ioupos_train.>0.0]
        logs.IoU_train = cat(logs.IoU_train,[mean(ioupos_train) mean(iouneg_train)],dims=1)
      end
      fvalepoch_train = 0.0
      dvalepoch_train = 0.0
      for i=1:length(train_data)
        #println(string("example train-",i))
        f,d = LossTotal(HN,TrOpts,train_data[i],train_labels[i],P[i],image_weights_train[i],active_z_slice,P_mode,TD_OP[i],P_sub[i],alpha_CQ[i])
        fvalepoch_train = fvalepoch_train + f
        dvalepoch_train = dvalepoch_train + d
      end
      logs.train       = vcat(logs.train,fvalepoch_train/length(train_data))
      logs.dc2_train   = vcat(logs.dc2_train,dvalepoch_train/length(train_data))

      #validation data/labels
      if isempty(val_labels[1])==false || isempty(P_sub_val)==false
        ioupos_val,iouneg_val = IoU(HN,val_data,val_labels)
        ioupos_val=ioupos_val[ioupos_val.>0.0]
        logs.IoU_val  = cat(logs.IoU_val,[mean(ioupos_val) mean(iouneg_val)],dims=1)

        fvalepoch_val = 0.0
        dvalepoch_val = 0.0
        for i=1:length(val_data)
          #PI = x -> x#no projector for validation
          #println(string("example val-",i))
          f,d = LossTotal(HN,TrOpts,val_data[i],val_labels[i],P[i],image_weights_val[i],active_z_slice,P_mode,TD_OP_val[i],P_sub_val[i],alpha_CQ_val[i])
          fvalepoch_val = fvalepoch_val + f
          dvalepoch_val = dvalepoch_val + d
        end
        logs.val     = vcat(logs.val,fvalepoch_val/length(val_data))
        logs.dc2_val = vcat(logs.dc2_val,dvalepoch_val/length(val_data))
      end

      TrainStatusPrint(j,logs)
      #print("Iteration: ", j, "; ftrain = ", logs.train[end], "; dtrain = ", logs.dc2_train[end], "; fval = ", logs.val[end], "; dval = ", logs.dc2_val[end], ";  IoUtrain:", logs.IoU_train[end,:] , ";  IoUval:" , logs.IoU_val[end,:], "\n")
      clear_grad!(HN)
    end

  end

  return logs
end

function TrainStatusPrint(iteration,logs)
  printstring=string("Iteration: ", iteration)
  if isempty(logs.train)==false
    printstring = printstring * string("; ftrain = ", logs.train[end])
  end
  if isempty(logs.dc2_train)==false
    printstring = printstring * string("; dtrain = ", logs.dc2_train[end])
  end
  if isempty(logs.val)==false
    printstring = printstring * string("; fval = ", logs.val[end])
  end
  if isempty(logs.dc2_val)==false
     printstring = printstring * string("; dval = ", logs.dc2_val[end])
   end
   if isempty(logs.IoU_train)==false
     printstring = printstring * string(";  IoUtrain:", logs.IoU_train[end,:])
   end
   if isempty(logs.IoU_val)==false
     printstring = printstring * string(";  IoUval:" , logs.IoU_val[end,:])
   end
   println(printstring)
end
