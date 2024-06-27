export AugmentDataLabel

function AugmentDataLabel(data::AbstractArray{T,N}, label, image_weights,flip_dims,permute_dims) where {T,N}

  for i in flip_dims
    if rand() > 0.5
      data          = reverse(data,dims=i)
      if isempty(label) == false
        label         = reverse(label,dims=i)
      end
      if isempty(image_weights) == false
        image_weights = reverse(image_weights,dims=i)
      end
    end
  end

  if isempty(permute_dims)==false
    if rand() > 0.5
      permute_dims_augment       = shuffle(permute_dims)
      permutation                = range(1,stop=N,step=1); permutation = convert(Vector{Any},permutation)
      permutation[permute_dims] .=  permute_dims_augment

      data          = permutedims(data,permutation)
      if isempty(label) == false
        println(permutation)
        label         = permutedims(label,permutation)
      end
      if isempty(image_weights) == false
        image_weights = permutedims(image_weights,permutation)
      end
    end
  end

  return data, label, image_weights
end

function CreateAugmentedDatasetExplicit()

#for simple data augmentation, add permuted/flipped versions of data and label tiles
data_transp   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
labels_transp = Vector{Array{Float32,2}}(undef,length(labels))
interest_patches_transp     = deepcopy(interest_patches)
non_interest_patches_transp = deepcopy(non_interest_patches)
geo_transp        = deepcopy(geo)
#
data_fliplr   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
labels_fliplr = Vector{Array{Float32,2}}(undef,length(labels))
interest_patches_lr     = deepcopy(interest_patches)
non_interest_patches_lr = deepcopy(non_interest_patches)
geo_lr        = deepcopy(geo)
#
data_flipud   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
labels_flipud = Vector{Array{Float32,2}}(undef,length(labels))
interest_patches_ud     = deepcopy(interest_patches)
non_interest_patches_ud = deepcopy(non_interest_patches)
geo_flipud        = deepcopy(geo)
#
data_fliplr_transp   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
labels_fliplr_transp = Vector{Array{Float32,2}}(undef,length(labels))
interest_patches_lr_transp     = deepcopy(interest_patches)
non_interest_patches_lr_transp = deepcopy(non_interest_patches)
geo_fliplr_transp        = deepcopy(geo)
# #
data_flipud_transp   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
labels_flipud_transp = Vector{Array{Float32,2}}(undef,length(labels))
interest_patches_ud_transp     = deepcopy(interest_patches)
non_interest_patches_ud_transp = deepcopy(non_interest_patches)
geo_flipud_transp        = deepcopy(geo)
# #
# data_flipudlr   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
# labels_flipudlr = Vector{Array{Float32,2}}(undef,length(labels))
# interest_patches_udlr     = deepcopy(interest_patches)
# non_interest_patches_udlr = deepcopy(non_interest_patches)
# geo_udlr        = deepcopy(geo)
# # #
# data_flipudlr_transp   = Vector{Union{Array{Float32,4},Array{Float32,3}}}(undef,length(data))
# labels_flipudlr_transp = Vector{Array{Float32,2}}(undef,length(labels))
# interest_patches_udlr_transp     = deepcopy(interest_patches)
# non_interest_patches_udlr_transp = deepcopy(non_interest_patches)
# geo_udlr_transp        = deepcopy(geo)

for i=1:length(data)
    #1
    data_transp[i]   = permutedims(data[i],(2,1,3))
    labels_transp[i] = permutedims(labels[i],(2,1))
    geo_transp[i] = permutedims(geo[i],(2,1))
    for j=1:length(interest_patches[i])
        interest_patches_transp[i][j] = permutedims(interest_patches[i][j],(2,1))
    end
    for j=1:length(non_interest_patches[i])
        non_interest_patches_transp[i][j] = permutedims(non_interest_patches[i][j],(2,1))
    end

    #2
    data_fliplr[i]   = reverse(data[i],dims=2)
    labels_fliplr[i] = reverse(labels[i],dims=2)
    geo_lr[i] = reverse(geo[i],dims=2)
    for j=1:length(interest_patches_lr[i])
        interest_patches_lr[i][j] = reverse(interest_patches_lr[i][j],dims=2)
    end
    for j=1:length(non_interest_patches_lr[i])
        non_interest_patches_lr[i][j] = reverse(non_interest_patches_lr[i][j],dims=2)
    end

    #3
    data_flipud[i]   = reverse(data[i],dims=1)
    labels_flipud[i] = reverse(labels[i],dims=1)
    geo_flipud[i] = reverse(geo[i],dims=1)
    for j=1:length(interest_patches_ud[i])
        interest_patches_ud[i][j] = reverse(interest_patches_ud[i][j],dims=1)
    end
    for j=1:length(non_interest_patches_ud[i])
        non_interest_patches_ud[i][j] = reverse(non_interest_patches_ud[i][j],dims=1)
    end

    #4
    data_flipud_transp[i]   = permutedims(reverse(data[i],dims=1),(2,1,3))
    labels_flipud_transp[i] = permutedims(reverse(labels[i],dims=1),(2,1))
    geo_flipud_transp[i] = permutedims(reverse(geo[i],dims=1),(2,1))
    for j=1:length(interest_patches_ud_transp[i])
        interest_patches_ud_transp[i][j] = permutedims(reverse(interest_patches_ud_transp[i][j],dims=1),(2,1))
    end
    for j=1:length(non_interest_patches_ud_transp[i])
        non_interest_patches_ud_transp[i][j] = permutedims(reverse(non_interest_patches_ud_transp[i][j],dims=1),(2,1))
    end

    #5
    data_fliplr_transp[i]   = permutedims(reverse(data[i],dims=2),(2,1,3))
    labels_fliplr_transp[i] = permutedims(reverse(labels[i],dims=2),(2,1))
    geo_fliplr_transp[i] = permutedims(reverse(geo[i],dims=2),(2,1))
    for j=1:length(interest_patches_lr_transp[i])
        interest_patches_lr_transp[i][j] = permutedims(reverse(interest_patches_lr_transp[i][j],dims=2),(2,1))
    end
    for j=1:length(non_interest_patches_lr_transp[i])
        non_interest_patches_lr_transp[i][j] = permutedims(reverse(non_interest_patches_lr_transp[i][j],dims=2),(2,1))
    end

    #6
    # data_flipudlr[i]   = reverse(data[i],dims=(1,2))
    # labels_flipudlr[i] = reverse(labels[i],dims=(1,2))
    # geo_flipudlr[i] = reverse(geo[i],dims=(1,2))
    # for j=1:length(interest_patches_udlr[i])
    #     interest_patches_udlr[i][j] = reverse(interest_patches_udlr[i][j],dims=(1,2))
    # end
    # for j=1:length(non_interest_patches_udlr[i])
    #     non_interest_patches_udlr[i][j] = reverse(non_interest_patches_udlr[i][j],dims=(1,2))
    # end

    # #7
    # data_flipudlr_transp[i]   = permutedims(reverse(data[i],dims=(1,2)),(2,1,3))
    # labels_flipudlr_transp[i] = permutedims(reverse(labels[i],dims=(1,2)),(2,1))
    #geo_flipudlr_transp[i] = permutedims(reverse(geo[i],dims=(1,2)),(2,1))

    # for j=1:length(interest_patches_udlr_transp[i])
    #     interest_patches_udlr_transp[i][j] = permutedims(reverse(interest_patches_udlr_transp[i][j],dims=(1,2)),(2,1))
    # end
    # for j=1:length(non_interest_patches_udlr_transp[i])
    #     non_interest_patches_udlr_transp[i][j] = permutedims(reverse(non_interest_patches_udlr_transp[i][j],dims=(1,2)),(2,1))
    # end

end

data = vcat(data,data_transp,data_fliplr,data_flipud,data_fliplr_transp,data_flipud_transp)#,data_flipudlr,data_flipudlr_transp)
labels = vcat(labels,labels_transp,labels_fliplr,labels_flipud,labels_flipud_transp,labels_fliplr_transp)#,labels_flipudlr,labels_flipudlr_transp)
geo = vcat(geo,geo_transp,geo_lr,geo_flipud,geo_flipud_transp,geo_fliplr_transp)#,labels_flipudlr,labels_flipudlr_transp)
interest_patches = vcat(interest_patches,interest_patches_transp,interest_patches_lr,interest_patches_ud, interest_patches_ud_transp, interest_patches_lr_transp)#, interest_patches_udlr, interest_patches_udlr_transp)
non_interest_patches = vcat(non_interest_patches, non_interest_patches_transp, non_interest_patches_lr, non_interest_patches_ud, non_interest_patches_ud_transp, non_interest_patches_lr_transp)#, non_interest_patches_udlr, non_interest_patches_udlr_transp)

max_card_patch = repeat(max_card_patch,6)
min_card_patch = repeat(min_card_patch,6)
max_card_patch_non_interest = repeat(max_card_patch_non_interest,6)
min_card_patch_non_interest = repeat(min_card_patch_non_interest,6)


# plot_ind=1
# figure(figsize=(12,4))
# imshow(non_interest_map .+ 2.0*non_interest_map_val .+ 3.0.*interest_map .+ 4.0.*interest_map_val)
# savefig("all_train_val_areas_interest_noninterest.png",bbox="tight")

# plot_ind=1
# figure(figsize=(12,4))
# imshow(non_interest_map .+ 2.0*non_interest_map_val .+ interest_map .+ 2.0.*interest_map_val)
# savefig("dall_train_val_areas.png",bbox="tight")
######
######



end
