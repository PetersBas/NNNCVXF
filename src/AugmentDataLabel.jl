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
        label         = permutedims(label,permutation)
      end
      if isempty(image_weights) == false
        image_weights = permutedims(image_weights,permutation)
      end
    end
  end

  return data, label, image_weights
end
