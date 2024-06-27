export SparseClassSamples, SparseClassBoundarySamples

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

function SparseClassBoundarySamples(comp_grid, label::AbstractArray{T, 4},active_channels,n_pix_per_class::Int) where T
  #get a few randomly selected pixels per class per image as the point-annotations
  #sample close to the boundary of the object (intended for 2 classes)
  TD_OP, ~, ~, ~, ~ = get_TD_operator(comp_grid,"TV",Float32)

  cw = zeros(Float32,size(label)[1:2])
  for i in active_channels
    diff_im = TD_OP'*(TD_OP*vec(label[:,:,i]))
    diff_im = reshape(diff_im,comp_grid.n)
    boundary = findall(diff_im .!= 0)
    boundary_im = 0 .* label[:,:,i,1]; boundary_im[boundary] .= 1
    class_i_ind = findall(boundary_im.*label[:,:,i,1].==1)
    if length(class_i_ind)>0
      class_i_ind_select = shuffle(class_i_ind)[1:n_pix_per_class]
      cw[class_i_ind_select] .= 1.0f0
    end
  end
  return cw
end

function SparseClassSamples(label::AbstractArray{T, 5},active_channels,n_pix_per_class::Int) where T
  #get a few randomly selected pixels per class per image as the point-annotations
  cw = zeros(Float32,size(label)[1:3])
  for i in active_channels
    class_i_ind = findall(label[:,:,:,i,1].==1)
    if length(class_i_ind)>0
      class_i_ind_select = shuffle(class_i_ind)[1:n_pix_per_class]
      cw[class_i_ind_select] .= 1.0f0
    end
  end
  return cw
end

function SparseClassSamples(label::AbstractArray{T, 5},active_channels,n_pix_per_class::Int,active_z_slice) where T
  #get a few randomly selected pixels per class per image as the point-annotations
  cw = zeros(Float32,size(label)[1:2])
  for i in active_channels
    class_i_ind = findall(label[:,:,active_z_slice,i,1].==1)
    if length(class_i_ind)>0
      class_i_ind_select = shuffle(class_i_ind)[1:n_pix_per_class]
      cw[class_i_ind_select] .= 1.0f0
    end
  end
  return cw
end