export PlotCamvidLossUnconstrained, PlotCamvidLossConstrained, PlotDataLabelPredictionCamvid, PlotHyperspectralLossConstrained, PlotDataLabelPredictionHyperspectral

function PlotHyperspectralLossConstrained(logs,eval_every)

  iter_ax = range(0,step=eval_every,length=length(logs.dc2_train))

  figure(figsize=(5,4));
  semilogy(iter_ax,logs.dc2_train);title("Squared distance to set");xlabel("Iteration")
  tight_layout()
  savefig("dc2_hyperspectral.png")
end

function PlotDataLabelPredictionHyperspectral(plt_ind::Int,data,label,HN,active_channels,active_z_slice,pos_inds_select,neg_inds_select,tag::String)

  #predict
  p1, prediction, ~ = HN.forward(data[plt_ind], data[plt_ind])
  prediction = prediction |> cpu
  prediction[:,:,active_z_slice,active_channels,1].=softmax(prediction[:,:,active_z_slice,active_channels,1],dims=3);

  prediction = prediction |> cpu

  close("all")
  vmi = 0.0
  vma = 1.0
  figure(figsize=(6,4))
  subplot(2,2,1);imshow(Array(prediction)[3:end-2,3:end-2,active_z_slice,1,1],vmin=vmi,vmax=vma);PyPlot.title("Prediction - class 1");colorbar()
  subplot(2,2,2);imshow(Array(prediction)[3:end-2,3:end-2,active_z_slice,2,1],vmin=vmi,vmax=vma);PyPlot.title("Prediction - class 2");colorbar()
  subplot(2,2,3);imshow(label[plt_ind][3:end-2,3:end-2,active_z_slice,1,1],vmin=vmi,vmax=vma);PyPlot.title("Label - class 1");colorbar()
  subplot(2,2,4);imshow(label[plt_ind][3:end-2,3:end-2,active_z_slice,2,1],vmin=vmi,vmax=vma);PyPlot.title("Label - class 2");colorbar()
  savefig(string(tag,"_prediction_channels.png"))

  #thresholded prediction
  pred_thres = zeros(Int,size(prediction)[1:2])
  pos_inds   = findall(prediction[:,:,active_z_slice,1,1] .> prediction[:,:,active_z_slice,2,1])
  pred_thres[pos_inds] .= 1
  pred_thres2 = zeros(Int,size(prediction)[1:2])
  pos_inds    = findall(prediction[:,:,active_z_slice,1,1] .< prediction[:,:,active_z_slice,2,1])
  pred_thres2[pos_inds] .= 1
  figure(figsize=(5,4));
  imshow(Array(pred_thres)[3:end-2,3:end-2],vmin=vmi,vmax=vma);PyPlot.title(string("Prediction"));#xlabel("x");ylabel("y")
  tight_layout()
  savefig(string(tag,"_prediction.png"))#,bbox="tight")

  println(length(findall(pred_thres.>0))/prod(size(pred_thres)[1:2]))

  figure(figsize=(5,4));
  imshow(label[plt_ind][:,:,33,1,1],vmin=vmi,vmax=vma);PyPlot.title(string("Labels"));#xlabel("x");ylabel("y")
  for i=1:length(pos_inds_select)
      PyPlot.scatter(pos_inds_select[i][2],pos_inds_select[i][1],c="r")
  end
  if isempty(neg_inds_select)==false
    for i=1:length(neg_inds_select)
        PyPlot.scatter(neg_inds_select[i][2],neg_inds_select[i][1],c="w")
    end
  end
  tight_layout()
  savefig(string(tag,"_labels.png"))#,bbox="tight")

  figure(figsize=(5,4));
  imshow(Array(pred_thres)[3:end-2,3:end-2]-label[plt_ind][:,:,33,1,1][3:end-2,3:end-2],vmin=-1,vmax=1);PyPlot.title(string("Difference"));#xlabel("x");ylabel("y")
  tight_layout()
  savefig(string(tag,"_error.png"))#,bbox="tight")

  figure(figsize=(5,4));
  imshow(Array(pred_thres2)[3:end-2,3:end-2]-label[plt_ind][:,:,33,1,1][3:end-2,3:end-2],vmin=-1,vmax=1);PyPlot.title(string("Difference"));#xlabel("x");ylabel("y")
  tight_layout()
  savefig(string(tag,"_error2.png"))#,bbox="tight")

  #Plot data+prediction
  figure(figsize=(9,5))
  subplot(1,2,1);
  imshow(data[plt_ind][3:end-2,3:end-2,40,1,1],cmap="Greys")
  imshow(Array(pred_thres)[3:end-2,3:end-2],alpha=0.30)
  title(string("Data T1 + Prediction - ",tag))
  subplot(1,2,2);
  imshow(data[plt_ind][3:end-2,3:end-2,40,2,1],cmap="Greys")
  imshow(Array(pred_thres)[3:end-2,3:end-2],alpha=0.30)
  title(string("Data T2 + Prediction - ",tag))
  tight_layout()
  savefig(string(tag,"data_plus_pred.png"))

  figure();
  imshow(data[plt_ind][:,:,40,1,1]);title("Slice of T1 data");savefig("data_surface_T1.png")
  imshow(data[plt_ind][:,:,40,2,1]);title("Slice of T2 data");savefig("data_surface_T2.png")

  #plot data cube
  #-----------------------
  close("all")
  index_cube = CartesianIndices(size(data[plt_ind])[1:3]);
  slice1 = vec(index_cube[:,:,end])
  slice1x = zeros(length(slice1)); slice1y = zeros(length(slice1));slice1z = zeros(length(slice1))
  slice1_values1 = zeros(length(slice1));
  slice1_values2 = zeros(length(slice1));
  for i=1:length(slice1)
    slice1x[i] = slice1[i][1]
    slice1y[i] = slice1[i][2]
    slice1z[i] = slice1[i][3]
    slice1_values1[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],1,1]
    slice1_values2[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],2,1]
  end
  figure()
  scatter3D(slice1x,slice1y,slice1z,c=slice1_values1,vmin=-0.2,vmax=0.5);title("Time 1")
  #scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5);title("Time 2")


  slice1 = vec(index_cube[:,1,:])
  slice1x = zeros(length(slice1)); slice1y = zeros(length(slice1));slice1z = zeros(length(slice1))
  slice1_values1 = zeros(length(slice1));
  slice1_values2 = zeros(length(slice1));
  for i=1:length(slice1)
    slice1x[i] = slice1[i][1]
    slice1y[i] = slice1[i][2]
    slice1z[i] = slice1[i][3]
    slice1_values1[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],1,1]
    slice1_values2[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],2,1]
  end
  scatter3D(slice1x,slice1y,slice1z,c=slice1_values1,vmin=-0.2,vmax=0.5)
  #subplot(1,2,2);scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5);

  slice1 = vec(index_cube[end,:,:])
  slice1x = zeros(length(slice1)); slice1y = zeros(length(slice1));slice1z = zeros(length(slice1))
  slice1_values1 = zeros(length(slice1));
  slice1_values2 = zeros(length(slice1));
  for i=1:length(slice1)
    slice1x[i] = slice1[i][1]
    slice1y[i] = slice1[i][2]
    slice1z[i] = slice1[i][3]
    slice1_values1[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],1,1]
    slice1_values2[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],2,1]
  end
  scatter3D(slice1x,slice1y,slice1z,c=slice1_values1,vmin=-0.2,vmax=0.5)
  #subplot(1,2,2);scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5)
  tight_layout()
  savefig("data3D_T1.png",bbox="tight")
  #-----------------------

  #-----------------------
  close("all")
  index_cube = CartesianIndices(size(data[plt_ind])[1:3]);
  slice1 = vec(index_cube[:,:,end])
  slice1x = zeros(length(slice1)); slice1y = zeros(length(slice1));slice1z = zeros(length(slice1))
  slice1_values1 = zeros(length(slice1));
  slice1_values2 = zeros(length(slice1));
  for i=1:length(slice1)
    slice1x[i] = slice1[i][1]
    slice1y[i] = slice1[i][2]
    slice1z[i] = slice1[i][3]
    slice1_values1[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],1,1]
    slice1_values2[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],2,1]
  end
  figure()
  scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5);title("Time 2")
  #scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5);title("Time 2")


  slice1 = vec(index_cube[:,1,:])
  slice1x = zeros(length(slice1)); slice1y = zeros(length(slice1));slice1z = zeros(length(slice1))
  slice1_values1 = zeros(length(slice1));
  slice1_values2 = zeros(length(slice1));
  for i=1:length(slice1)
    slice1x[i] = slice1[i][1]
    slice1y[i] = slice1[i][2]
    slice1z[i] = slice1[i][3]
    slice1_values1[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],1,1]
    slice1_values2[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],2,1]
  end
  scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5)
  #subplot(1,2,2);scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5);

  slice1 = vec(index_cube[end,:,:])
  slice1x = zeros(length(slice1)); slice1y = zeros(length(slice1));slice1z = zeros(length(slice1))
  slice1_values1 = zeros(length(slice1));
  slice1_values2 = zeros(length(slice1));
  for i=1:length(slice1)
    slice1x[i] = slice1[i][1]
    slice1y[i] = slice1[i][2]
    slice1z[i] = slice1[i][3]
    slice1_values1[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],1,1]
    slice1_values2[i] = data[plt_ind][slice1[i][1],slice1[i][2],slice1[i][3],2,1]
  end
  scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5)
  #subplot(1,2,2);scatter3D(slice1x,slice1y,slice1z,c=slice1_values2,vmin=-0.2,vmax=0.5)
  tight_layout()
  savefig("data3D_T2.png")
  #-----------------------

end

function PlotCamvidLossUnconstrained(logs,eval_every)

  iter_ax = range(0,step=eval_every,length=length(logs.train))

  figure(figsize=(13,4));
  subplot(1,2,1);plot(iter_ax,logs.train,label="train");title("Training loss labels");
  subplot(1,2,1);plot(iter_ax,logs.val,"--",label="validation");legend();title("Training loss labels");
  xlabel("Iteration number");
  subplot(1,2,2);plot(iter_ax,logs.IoU_train[:,1],label="class1 - train");title("IoU");
  subplot(1,2,2);plot(iter_ax,logs.IoU_train[:,2],label="class2 - train");legend();title("IoU");
  subplot(1,2,2);plot(iter_ax,logs.IoU_val[:,1],"--",label="class1 - validation");title("IoU");
  subplot(1,2,2);plot(iter_ax,logs.IoU_val[:,2],"--",label="class2 - validation");legend();title("IoU");
  xlabel("Iteration number");
  tight_layout()
  savefig("loss_unconstrained.png")

end

function PlotCamvidLossConstrained(logs,eval_every)

  iter_ax = range(0,step=eval_every,length=length(logs.dc2_train))
  figure(figsize=(13,4));
  subplot(1,3,1);semilogy(iter_ax,logs.dc2_train,label="train");legend();title("Squared distance-to-set");
  xlabel("Iteration number");
  subplot(1,3,2);plot(iter_ax,logs.train,label="train");title("Training loss labels");
  subplot(1,3,2);plot(iter_ax,logs.val,"--",label="validation");legend();title("Training loss labels");
  xlabel("Iteration number");
  subplot(1,3,3);plot(iter_ax,logs.IoU_train[:,1],label="class1 - train");title("IoU");
  subplot(1,3,3);plot(iter_ax,logs.IoU_train[:,2],label="class2 - train");legend();title("IoU");
  subplot(1,3,3);plot(iter_ax,logs.IoU_val[:,1],"--",label="class1 - validation");title("IoU");
  subplot(1,3,3);plot(iter_ax,logs.IoU_val[:,2],"--",label="class2 - validation");legend();title("IoU");
  xlabel("Iteration number");
  tight_layout()
  savefig("loss_constrained.png")

end

function PlotDataLabelPredictionCamvid(plt_ind::Int,data,label,HN,active_channels,tag::String)

  #Plot data
  figure(figsize=(5,4))
  imshow(data[plt_ind][:,:,1:3,1])
  title(string("Data - ",tag))
  tight_layout()
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
  tight_layout()
  savefig(string(tag,"_prediction_channels.png"))

  #thresholded prediction
  pred_thres = zeros(Int,size(prediction)[1:2])
  pos_inds = findall(prediction[:,:,1,1] .> 0.65)#prediction[:,:,2,1])
  pred_thres[pos_inds] .= 1
  figure(figsize=(5,4));
  imshow(Array(pred_thres)[3:end-2,3:end-2],vmin=vmi,vmax=vma);PyPlot.title(string("Prediction - ",tag));#xlabel("x");ylabel("y")
  tight_layout()
  savefig(string(tag,"_prediction.png"))#,bbox="tight")

  #Plot data+prediction
  figure(figsize=(5,4))
  imshow(data[plt_ind][3:end-2,3:end-2,1:3,1])
  imshow(Array(pred_thres)[3:end-2,3:end-2],alpha=0.45)
  title(string("Image+Prediction - ",tag))
  tight_layout()
  savefig(string(tag,"data_plus_pred.png"))
  return
end
