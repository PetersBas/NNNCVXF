export PlotCamvidLossUnconstrained, PlotCamvidLossConstrained

function PlotCamvidLossUnconstrained(fval_train,fval_val,IoU_hist_train,IoU_hist_val)


  iter_ax = range(0,step=50,length=length(fval_train))

  figure(figsize=(16,7));
  subplot(1,2,1);plot(iter_ax,fval_train,label="train");title("Training loss labels");
  subplot(1,2,1);plot(iter_ax,fval_val,"--",label="validation");legend();title("Training loss labels");
  subplot(1,2,2);plot(iter_ax,IoU_hist_train[:,1],label="class1 - train");title("IoU");
  subplot(1,2,2);plot(iter_ax,IoU_hist_train[:,2],label="class2 - train");legend();title("IoU");
  subplot(1,2,2);plot(iter_ax,IoU_hist_val[:,1],"--",label="class1 - validation");title("IoU");
  subplot(1,2,2);plot(iter_ax,IoU_hist_val[:,2],"--",label="class2 - validation");legend();title("IoU");
  xlabel("Iteration number");
  savefig("loss_unconstrained.png")

end

function PlotCamvidLossConstrained(fval_train,fval_val,IoU_hist_train,IoU_hist_val,dc2val_train)

  iter_ax = range(0,step=50,length=length(dc2val_train))
  figure(figsize=(16,7));
  subplot(1,3,1);semilogy(iter_ax,dc2val_train,label="train");legend();title("Squared distance-to-set");
  subplot(1,3,2);plot(iter_ax,fval_train,label="train");title("Training loss labels");
  subplot(1,3,2);plot(iter_ax,fval_val,"--",label="validation");legend();title("Training loss labels");
  subplot(1,3,3);plot(iter_ax,IoU_hist_train[:,1],label="class1 - train");title("IoU");
  subplot(1,3,3);plot(iter_ax,IoU_hist_train[:,2],label="class2 - train");legend();title("IoU");
  subplot(1,3,3);plot(iter_ax,IoU_hist_val[:,1],"--",label="class1 - validation");title("IoU");
  subplot(1,3,3);plot(iter_ax,IoU_hist_val[:,2],"--",label="class2 - validation");legend();title("IoU");
  xlabel("Iteration number");
  savefig("loss_constrained.png")

end
