export SparseClassSamples

function SparseClassSamples(label::AbstractArray{T, 4},active_channels,n_pix_per_class::Int) where T
  #get a few randomly selected pixels per class per image as the point-annotations
  cw = zeros(Float32,size(label)[1:2])
  for i in active_channels
    class_i_ind = findall(label[:,:,i,1].==1)
    if length(class_i_ind)>0
      class_i_ind_select = shuffle(class_i_ind)[1:n_pix_per_class]
      cw[class_i_ind_select] .= 1.0f0
    end
  end
  return cw
end
