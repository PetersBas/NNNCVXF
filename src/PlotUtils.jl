export PlotCamvidLossUnconstrained, PlotCamvidLossConstrained, PlotDataLabelPredictionCamvid

function PlotCamvidLossUnconstrained(fval_train,fval_val,IoU_hist_train,IoU_hist_val)

  f_log_train   = [];
  f_log_val     = [];
  # dc2_log_train = [];
  # dc2_log_val   = [];
  IoU_log_train = [[] []];
  IoU_log_val   = [[] []];

  for i=1:length(fval_train)
    f_log_train     = vcat(f_log_train,fval_train[i])
    f_log_val       = vcat(f_log_val,fval_val[i])
    # dc2_log_train = vcat(dc2_log_train,fval_train[i])
    # dc2_log_val   = vcat(dc2_log_val,fval_train[i])
    IoU_log_train   = vcat(IoU_log_train,IoU_hist_train[i])
    IoU_log_val     = vcat(IoU_log_val,IoU_hist_val[i])
  end

  iter_ax = range(0,step=50,length=length(fval_train))

  figure(figsize=(16,7));
  subplot(1,2,1);plot(iter_ax,f_log_train,label="train");title("Training loss labels");
  subplot(1,2,1);plot(iter_ax,f_log_val,"--",label="validation");legend();title("Training loss labels");
  subplot(1,2,2);plot(iter_ax,IoU_log_train[:,1],label="class1 - train");title("IoU");
  subplot(1,2,2);plot(iter_ax,IoU_log_train[:,2],label="class2 - train");legend();title("IoU");
  subplot(1,2,2);plot(iter_ax,IoU_log_val[:,1],"--",label="class1 - validation");title("IoU");
  subplot(1,2,2);plot(iter_ax,IoU_log_val[:,2],"--",label="class2 - validation");legend();title("IoU");
  xlabel("Iteration number");
  savefig("loss_unconstrained.png")

end

function PlotCamvidLossConstrained(fval_train,fval_val,IoU_hist_train,IoU_hist_val,dc2val_train)

  f_log_train   = [];
  f_log_val     = [];
  dc2_log_train = [];
  #dc2_log_val   = [];
  IoU_log_train = [[] []];
  IoU_log_val   = [[] []];

  for i=1:length(fval_train)
    f_log_train     = vcat(f_log_train,fval_train[i])
    f_log_val       = vcat(f_log_val,fval_val[i])
    dc2_log_train   = vcat(dc2_log_train,dc2val_train[i])
    #dc2_log_val   = vcat(dc2_log_val,fval_train[i])
    IoU_log_train   = vcat(IoU_log_train,IoU_hist_train[i])
    IoU_log_val     = vcat(IoU_log_val,IoU_hist_val[i])
  end

  iter_ax = range(0,step=50,length=length(dc2val_train))
  figure(figsize=(16,7));
  subplot(1,3,1);semilogy(iter_ax,dc2_log_train,label="train");legend();title("Squared distance-to-set");
  subplot(1,3,2);plot(iter_ax,f_log_train,label="train");title("Training loss labels");
  subplot(1,3,2);plot(iter_ax,f_log_val,"--",label="validation");legend();title("Training loss labels");
  subplot(1,3,3);plot(iter_ax,IoU_log_train[:,1],label="class1 - train");title("IoU");
  subplot(1,3,3);plot(iter_ax,IoU_log_train[:,2],label="class2 - train");legend();title("IoU");
  subplot(1,3,3);plot(iter_ax,IoU_log_val[:,1],"--",label="class1 - validation");title("IoU");
  subplot(1,3,3);plot(iter_ax,IoU_log_val[:,2],"--",label="class2 - validation");legend();title("IoU");
  xlabel("Iteration number");
  savefig("loss_constrained.png")

end

function PlotDataLabelPredictionCamvid(plt_ind::Int,data,label,HN,active_channels,tag::String)

  #Plot data
  figure(figsize=(5,4))
  imshow(data[plt_ind][:,:,1:3,1])
  title(string("Data - ",tag))
  savefig(string(tag,"data.png"),bbox="tight")

  #predict
  p1, prediction, ~ = HN.forward(data[plt_ind], data[plt_ind])
  prediction[:,:,active_channels,1].=softmax(prediction[:,:,active_channels,1],dims=3);

  close("all")
  vmi = 0.0
  vma = 1.0
  figure(figsize=(6,4))
  subplot(2,2,1);imshow(Array(prediction)[3:end-2,3:end-2,1,1],vmin=vmi,vmax=vma);PyPlot.title("Prediction - class 1");colorbar()
  subplot(2,2,2);imshow(Array(prediction)[3:end-2,3:end-2,2,1],vmin=vmi,vmax=vma);PyPlot.title("Prediction - class 2");colorbar()
  subplot(2,2,3);imshow(label[plt_ind][3:end-2,3:end-2,1,1],vmin=vmi,vmax=vma);PyPlot.title("Label - class 1");colorbar()
  subplot(2,2,4);imshow(label[plt_ind][3:end-2,3:end-2,2,1],vmin=vmi,vmax=vma);PyPlot.title("Label - class 2");colorbar()
  savefig(string(tag,"_prediction_channels.png"))

  #thresholded prediction
  pred_thres = zeros(Int,size(prediction)[1:2])
  pos_inds = findall(prediction[:,:,1,1] .> 0.65)#prediction[:,:,2,1])
  pred_thres[pos_inds] .= 1
  figure(figsize=(5,4));
  imshow(Array(pred_thres)[3:end-2,3:end-2],vmin=vmi,vmax=vma);PyPlot.title(string("Prediction - ",tag));#xlabel("x");ylabel("y")
  savefig(string(tag,"_prediction.png"))#,bbox="tight")

  #Plot data+prediction
  figure(figsize=(5,4))
  imshow(data[plt_ind][3:end-2,3:end-2,1:3,1])
  imshow(Array(pred_thres)[3:end-2,3:end-2],alpha=0.45)
  title(string("Image+Prediction - ",tag))
  savefig(string(tag,"data_plus_pred.png"))
  return
end
