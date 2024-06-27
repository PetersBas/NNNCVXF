export LoadCamvid, LoadCamvidFull, LoadBearVideo, LoadBearImages, GetTVvalues

"""
Obtain the anisotropic total-variation per image. If the noise factor is larger than zero,
the function return a number from the uniform distribution [trueTV * (1-noise_factor) - trueTV * (1+noise_factor)]
"""
function GetTVvalues(grid,labels,noise_factor,compgrid,active_z_slice=[1:compgrid.n[3]])

  (TV,dummy1,dummy2,dummy3) = get_TD_operator(compgrid,"TV",Float32)

  true_TV_list = zeros(Float32,length(labels))

  for i=1:length(labels)
    if length(size(labels[i]))<=4
      true_TV_list[i] = norm(TV*vec(labels[i][:,:,1]),1)
    elseif length(size(labels[i]))==5
      true_TV_list[i] = norm(TV*vec(labels[i][:,:,active_z_slice,1,1]),1)
    end
  end

    #add 'noise' to true tv list
  if noise_factor>0.0
    for i=1:length(labels)
      if true_TV_list[i] > 0.0
        d = Uniform(max((1-noise_factor)*true_TV_list[i],0.1),max((1+noise_factor)*true_TV_list[i],0.2))
        true_TV_list[i] = rand(d,1)[1]
      elseif true_TV_list[i] == 0.0
        true_TV_list[i] = 0.1
      end
    end
  end

  return true_TV_list
end

function LoadCamvid(basepath::String,nr_output_chan::Int)

  current_dir = pwd()

  #set data directories
  data_dir_train  = joinpath(basepath,"Camvid/train/")
  data_dir_val    = joinpath(basepath,"Camvid/val/")
  label_dir_train = joinpath(basepath,"Camvid/trainannot/")
  label_dir_val   = joinpath(basepath,"Camvid/valannot/")

  #read file names
  all_train_data = readdir(data_dir_train);
  all_val_data   = readdir(data_dir_val);

  all_train_labels = readdir(label_dir_train);
  all_val_labels   = readdir(label_dir_val);

  #allocate tensors for data/label storage
  train_data     = zeros(Float32,3,360,480,62);
  val_data       = zeros(Float32,3,360,480,10);

  train_labels   = zeros(Float32,360,480,1,62);
  val_labels     = zeros(Float32,360,480,1,10);

  #read training data (hardcoded for getting the first video only)
  cd(data_dir_train)
  for i=1:62#length(all_train_data)
      temp_img = Images.load(all_train_data[i])
      train_data[1:3,:,:,i] = Images.channelview(temp_img)
      println(i)
   end

  #permute to x-y-n_chan-n_examples
  train_data    = permutedims(train_data,[2,3,1,4]);

  #repeat data to increase channel count explicitly for input-output
  repeat_factor = Int(nr_output_chan/3)
  train_data    = repeat(train_data,outer=[1,1,repeat_factor,1])

  #read val data
  cd(data_dir_val)
  for i=1:10#length(all_val_data)
      temp_img = Images.load(all_val_data[i])
      val_data[1:3,:,:,i] = Images.channelview(temp_img)
      println(i)
   end
  val_data = permutedims(val_data,[2,3,1,4]);;
  val_data = repeat(val_data,outer=[1,1,repeat_factor,1])

  #read training labels
  cd(label_dir_train)
  counter = 1
  for i=1:62#length(all_train_labels)
      temp_img = Images.load(all_train_labels[i])
      train_labels[:,:,1,i] = Images.channelview(temp_img)
      println(i)
   end

   #read validation labels
   cd(label_dir_val)
   counter = 1
   for i=1:10#length(all_val_labels)
       temp_img = Images.load(all_val_labels[i])
       val_labels[:,:,1,i] = Images.channelview(temp_img)
       println(i)
    end

    #reduce resolution and make sure each dimension can be divided by 2 at least 2 times (to be able to take wavelet transforms)
    train_labels = train_labels[1:2:end,1:2:end,:,:]; train_labels = train_labels[1:176,:,:,:];
    val_labels   = val_labels[1:2:end,1:2:end,:,:];   val_labels   = val_labels[1:176,:,:,:];
    val_data     = val_data[1:2:end,1:2:end,:,:];     val_data     = val_data[1:176,:,:,:];
    train_data   = train_data[1:2:end,1:2:end,:,:];   train_data   = train_data[1:176,:,:,:];

    #change labels to 1-hot encodings
    n_class             = length(unique(train_labels))
    class_values        = sort(unique(train_labels))
    train_labels_OneHot = zeros(Int,size(train_labels,1),size(train_labels,2),length(class_values),size(train_labels,4));
    for i=1:length(class_values)
        println(i)
        class_indices = findall(train_labels[:,:,1,:] .== class_values[i])
        class_image = zeros(Int,size(train_labels)[1],size(train_labels)[2],size(train_labels)[4])
        class_image[class_indices] .= 1
        train_labels_OneHot[:,:,i,:] .= class_image
    end

    val_labels_OneHot = zeros(Int,size(val_labels,1),size(val_labels,2),length(class_values),size(val_labels,4));
    for i=1:length(class_values)
        class_indices = findall(val_labels[:,:,1,:] .== class_values[i])
        class_image = zeros(Int,size(val_labels)[1],size(val_labels)[2],size(val_labels)[4])
        class_image[class_indices] .= 1
        val_labels_OneHot[:,:,i,:] .= class_image
    end

    #print % of pixels in each class for the first video
    println("% of pixels in each class, for first video in CamVid dataset")
    for i=1:12;
      println("class: ",i)
      println(sum(train_labels_OneHot[:,:,i,:])/prod(size(train_labels_OneHot[:,:,1,:])).*100);
    end

    # selected_classes = [1, 2, 4, 5, 6, 9, 12]
    # combine_classes = [3,7,8,10,11]
    #
    # val_labels_OneHot[:,:,[combine_classes[1]],:]  .= sum(val_labels_OneHot[:,:,combine_classes,:],dims=3)
    # train_labels_OneHot[:,:,[combine_classes[1]],:].= sum(train_labels_OneHot[:,:,combine_classes,:],dims=3)

    #class 9 corresponds to vehicles, 11 to bikers
    selected_classes=[9]
    #val_labels_OneHot = val_labels_OneHot[:,:,selected_classes,:]
    train_labels_OneHot = train_labels_OneHot[:,:,selected_classes,:]

    #add the second channel (1 - first channel) (this is unnecessary if n_classes > 3)
    train_labels_OneHot = repeat(train_labels_OneHot,outer=[1,1,2,1])
    train_labels_OneHot[:,:,2,:] .= 1 .- train_labels_OneHot[:,:,1,:]

    #create a vector with one label per entry
    train_labels   = Vector{Any}(undef,size(train_data)[end])
    val_labels     = Vector{Any}(undef,size(val_data)[end])

    for i=1:length(train_labels)
      train_labels[i] = train_labels_OneHot[:,:,:,i]
    #   train_labels[i] = zeros(Float32,size(train_labels_OneHot,1),size(train_labels_OneHot,2),2,1)
    #   train_labels[i][:,:,1,1] .= 1f0 .*train_labels_OneHot[:,:,9,i]
    #   train_labels[i][:,:,2,1] .= 1f0 .- train_labels_OneHot[:,:,9,i]
    end

    for i=1:length(val_labels)
      val_labels[i] = val_labels_OneHot[:,:,:,i]
    #   val_labels[i] = zeros(Float32,size(val_labels_OneHot,1),size(val_labels_OneHot,2),2,1)
    #   val_labels[i][:,:,1,1] .= val_labels_OneHot[:,:,9,i]
    #   val_labels[i][:,:,2,1] .= 1f0 .- val_labels_OneHot[:,:,9,i]
    end

    #change data to a vector of examples
    dataL    = Vector{Array{Float32,4}}(undef,size(train_data,4))
    datavalL = Vector{Array{Float32,4}}(undef,size(val_data,4))
    for i=1:length(dataL)
        dataL[i] = train_data[:,:,:,[i]]
    end
    for i=1:length(datavalL)
        datavalL[i] = val_data[:,:,:,[i]]
    end
    train_data = deepcopy(dataL)
    dataL      = []
    val_data   = deepcopy(datavalL)
    datavalL   = []

    cd(current_dir) #switch back to current working dir

    #repartition training and validation:
    val_data   = deepcopy(train_data[47:end])
    train_data = train_data[1:46]

    val_labels = deepcopy(train_labels[47:end])
    train_labels = train_labels[1:46]


    return train_data, train_labels, val_data, val_labels
end #end function

function LoadCamvidFull(basepath::String,nr_output_chan::Int)

  current_dir = pwd()

  #set data directories
  data_dir_train  = joinpath(basepath,"Camvid/train/")
  data_dir_val    = joinpath(basepath,"Camvid/val/")
  label_dir_train = joinpath(basepath,"Camvid/trainannot/")
  label_dir_val   = joinpath(basepath,"Camvid/valannot/")

  #read file names
  all_train_data = readdir(data_dir_train);
  all_val_data   = readdir(data_dir_val);

  all_train_labels = readdir(label_dir_train);
  all_val_labels   = readdir(label_dir_val);

  #allocate tensors for data/label storage
  train_data     = zeros(Float32,3,360,480,length(all_train_data));
  val_data       = zeros(Float32,3,360,480,length(all_val_data));

  train_labels   = zeros(Float32,360,480,1,length(all_train_data));
  val_labels     = zeros(Float32,360,480,1,length(all_val_data));

  #read training data (hardcoded for getting the first video only)
  cd(data_dir_train)
  for i=1:length(all_train_data)
      temp_img = Images.load(all_train_data[i])
      train_data[1:3,:,:,i] = Images.channelview(temp_img)
      println(i)
   end

  #permute to x-y-n_chan-n_examples
  train_data    = permutedims(train_data,[2,3,1,4]);

  #repeat data to increase channel count explicitly for input-output
  repeat_factor = Int(nr_output_chan/3)
  train_data    = repeat(train_data,outer=[1,1,repeat_factor,1])

  #read val data
  cd(data_dir_val)
  for i=1:length(all_val_data)
      temp_img = Images.load(all_val_data[i])
      val_data[1:3,:,:,i] = Images.channelview(temp_img)
      println(i)
   end
  val_data = permutedims(val_data,[2,3,1,4]);;
  val_data=repeat(val_data,outer=[1,1,16,1])

  #read training labels
  cd(label_dir_train)
  counter = 1
  for i=1:length(all_train_labels)
      temp_img = Images.load(all_train_labels[i])
      train_labels[:,:,1,i] = Images.channelview(temp_img)
      println(i)
   end

   #read validation labels
   cd(label_dir_val)
   counter = 1
   for i=1:length(all_val_labels)
       temp_img = Images.load(all_val_labels[i])
       val_labels[:,:,1,i] = Images.channelview(temp_img)
       println(i)
    end

    #reduce resolution and make sure each dimension can be divided by 2 at least 2 times (to be able to take wavelet transforms)
    train_labels = train_labels[1:2:end,1:2:end,:,:]; train_labels = train_labels[1:176,:,:,:];
    val_labels   = val_labels[1:2:end,1:2:end,:,:];   val_labels   = val_labels[1:176,:,:,:];
    val_data     = val_data[1:2:end,1:2:end,:,:];     val_data     = val_data[1:176,:,:,:];
    train_data   = train_data[1:2:end,1:2:end,:,:];   train_data   = train_data[1:176,:,:,:];

    #change labels to 1-hot encodings
    n_class             = length(unique(train_labels))
    class_values        = sort(unique(train_labels))
    train_labels_OneHot = zeros(Int,size(train_labels,1),size(train_labels,2),length(class_values),size(train_labels,4));
    for i=1:length(class_values)
        println(i)
        class_indices = findall(train_labels[:,:,1,:] .== class_values[i])
        class_image = zeros(Int,size(train_labels)[1],size(train_labels)[2],size(train_labels)[4])
        class_image[class_indices] .= 1
        train_labels_OneHot[:,:,i,:] .= class_image
    end

    val_labels_OneHot = zeros(Int,size(val_labels,1),size(val_labels,2),length(class_values),size(val_labels,4));
    for i=1:length(class_values)
        class_indices = findall(val_labels[:,:,1,:] .== class_values[i])
        class_image = zeros(Int,size(val_labels)[1],size(val_labels)[2],size(val_labels)[4])
        class_image[class_indices] .= 1
        val_labels_OneHot[:,:,i,:] .= class_image
    end

    #print % of pixels in each class for the first video
    println("% of pixels in each class, for first video in CamVid dataset")
    for i=1:12;
      println("class: ",i)
      println(sum(train_labels_OneHot[:,:,i,:])/prod(size(train_labels_OneHot[:,:,1,:])).*100);
    end

    # selected_classes = [1, 2, 4, 5, 6, 9, 12]
    # combine_classes = [3,7,8,10,11]
    #
    # val_labels_OneHot[:,:,[combine_classes[1]],:]  .= sum(val_labels_OneHot[:,:,combine_classes,:],dims=3)
    # train_labels_OneHot[:,:,[combine_classes[1]],:].= sum(train_labels_OneHot[:,:,combine_classes,:],dims=3)

    #class 9 corresponds to vehicles
    selected_classes=[9]
    #val_labels_OneHot = val_labels_OneHot[:,:,selected_classes,:]
    train_labels_OneHot = train_labels_OneHot[:,:,selected_classes,:]

    #add the second channel (1 - first channel) (this is unnecessary if n_classes > 3)
    train_labels_OneHot = repeat(train_labels_OneHot,outer=[1,1,2,1])
    train_labels_OneHot[:,:,2,:] .= 1 .- train_labels_OneHot[:,:,1,:]

    #create a vector with one label per entry
    train_labels   = Vector{Any}(undef,size(train_data)[end])
    val_labels     = Vector{Any}(undef,size(val_data)[end])

    for i=1:length(train_labels)
      train_labels[i] = train_labels_OneHot[:,:,:,i]
    #   train_labels[i] = zeros(Float32,size(train_labels_OneHot,1),size(train_labels_OneHot,2),2,1)
    #   train_labels[i][:,:,1,1] .= 1f0 .*train_labels_OneHot[:,:,9,i]
    #   train_labels[i][:,:,2,1] .= 1f0 .- train_labels_OneHot[:,:,9,i]
    end

    for i=1:length(val_labels)
      val_labels[i] = val_labels_OneHot[:,:,:,i]
    #   val_labels[i] = zeros(Float32,size(val_labels_OneHot,1),size(val_labels_OneHot,2),2,1)
    #   val_labels[i][:,:,1,1] .= val_labels_OneHot[:,:,9,i]
    #   val_labels[i][:,:,2,1] .= 1f0 .- val_labels_OneHot[:,:,9,i]
    end

    #change data to a vector of examples
    dataL    = Vector{Array{Float32,4}}(undef,size(train_data,4))
    datavalL = Vector{Array{Float32,4}}(undef,size(val_data,4))
    for i=1:length(dataL)
        dataL[i] = train_data[:,:,:,[i]]
    end
    for i=1:length(datavalL)
        datavalL[i] = val_data[:,:,:,[i]]
    end
    train_data = deepcopy(dataL)
    dataL      = []
    val_data   = deepcopy(datavalL)
    datavalL   = []

    cd(current_dir) #switch back to current working dir

    # #repartition training and validation:
    # val_data   = deepcopy(train_data[47:end])
    # train_data = train_data[1:46]

    # val_labels = deepcopy(train_labels[47:end])
    # train_labels = train_labels[1:46]


    return train_data, train_labels, val_data, val_labels
end #end function

function LoadBearImages(basepath::String,nr_output_chan::Int)

  current_dir = pwd()

  #set data directories
  data_dir_train  = joinpath(basepath,"bear_images/")
  data_dir_val    = joinpath(basepath,"bear_images/")
  label_dir_train = joinpath(basepath,"bear_annotations/")
  label_dir_val   = joinpath(basepath,"bear_annotations/")

  #read file names (limit to 77 images, otherwise there's a corrupted image)
  all_train_data = readdir(data_dir_train)[1:77];
  all_val_data   = readdir(data_dir_val)[1:77];

  all_train_labels = readdir(label_dir_train)[1:77];
  all_val_labels   = readdir(label_dir_val)[1:77];

  #allocate tensors for data/label storage
  train_data     = zeros(Float32,3,480,854,length(all_train_data));
  val_data       = zeros(Float32,3,480,854,length(all_val_data));

  train_labels   = zeros(Float32,480,854,1,length(all_train_data));
  val_labels     = zeros(Float32,480,854,1,length(all_val_data));

  #read training data (hardcoded for getting the first video only)
  cd(data_dir_train)
  for i=1:length(all_train_data)
      temp_img = Images.load(all_train_data[i])
      train_data[1:3,:,:,i] = Images.channelview(temp_img)
      println(i)
   end

  #permute to x-y-n_chan-n_examples
  train_data    = permutedims(train_data,[2,3,1,4]);

  #repeat data to increase channel count explicitly for input-output
  repeat_factor = Int(nr_output_chan/3)
  train_data    = repeat(train_data,outer=[1,1,repeat_factor,1])

  #read val data
  cd(data_dir_val)
  for i=1:length(all_val_data)
      temp_img = Images.load(all_val_data[i])
      val_data[1:3,:,:,i] = Images.channelview(temp_img)
      println(i)
   end
  val_data = permutedims(val_data,[2,3,1,4]);;
  val_data = repeat(val_data,outer=[1,1,repeat_factor,1])

  #read training labels
  cd(label_dir_train)
  counter = 1
  for i=1:length(all_train_labels)
      temp_img = Images.load(all_train_labels[i])
      train_labels[:,:,1,i] = Images.channelview(temp_img)
      println(i)
   end

   #read validation labels
   cd(label_dir_val)
   counter = 1
   for i=1:length(all_val_labels)
       temp_img = Images.load(all_val_labels[i])
       val_labels[:,:,1,i] = Images.channelview(temp_img)
       println(i)
    end

    #reduce resolution and make sure each dimension can be divided by 2 at least 2 times (to be able to take wavelet transforms)
    train_labels = train_labels[1:2:end,1:2:end,:,:]; train_labels = train_labels[:,1:424,:,:];
    val_labels   = val_labels[1:2:end,1:2:end,:,:];   val_labels   = val_labels[:,1:424,:,:];
    val_data     = val_data[1:2:end,1:2:end,:,:];     val_data     = val_data[:,1:424,:,:];
    train_data   = train_data[1:2:end,1:2:end,:,:];   train_data   = train_data[:,1:424,:,:];

    #change labels to 1-hot encodings
    #class 1 corresponds to the object/anomaly (bear)
    #add the second channel (1 - first channel) (this is unnecessary if n_classes > 3)
    train_labels_OH = repeat(train_labels,outer=[1,1,2,1])
    train_labels_OH[:,:,2,:] .= 1 .- train_labels_OH[:,:,1,:]
    val_labels_OH = repeat(val_labels,outer=[1,1,2,1])
    val_labels_OH[:,:,2,:] .= 1 .- val_labels_OH[:,:,1,:]

    #create a vector with one label per entry
    train_labels   = Vector{Any}(undef,size(train_data)[end])
    val_labels     = Vector{Any}(undef,size(val_data)[end])

    for i=1:length(train_labels)
      train_labels[i] = train_labels_OH[:,:,:,i]
    #   train_labels[i] = zeros(Float32,size(train_labels_OneHot,1),size(train_labels_OneHot,2),2,1)
    #   train_labels[i][:,:,1,1] .= 1f0 .*train_labels_OneHot[:,:,9,i]
    #   train_labels[i][:,:,2,1] .= 1f0 .- train_labels_OneHot[:,:,9,i]
    end

    for i=1:length(val_labels)
      val_labels[i] = val_labels_OH[:,:,:,i]
    #   val_labels[i] = zeros(Float32,size(val_labels_OneHot,1),size(val_labels_OneHot,2),2,1)
    #   val_labels[i][:,:,1,1] .= val_labels_OneHot[:,:,9,i]
    #   val_labels[i][:,:,2,1] .= 1f0 .- val_labels_OneHot[:,:,9,i]
    end

    #change data to a vector of examples
    dataL    = Vector{Array{Float32,4}}(undef,size(train_data,4))
    datavalL = Vector{Array{Float32,4}}(undef,size(val_data,4))
    for i=1:length(dataL)
        dataL[i] = train_data[:,:,:,[i]]
    end
    for i=1:length(datavalL)
        datavalL[i] = val_data[:,:,:,[i]]
    end
    train_data = deepcopy(dataL)
    dataL      = []
    val_data   = deepcopy(datavalL)
    datavalL   = []

    cd(current_dir) #switch back to current working dir

    # #repartition training and validation:
    # val_data   = deepcopy(train_data[47:end])
    # train_data = train_data[1:46]

    # val_labels = deepcopy(train_labels[47:end])
    # train_labels = train_labels[1:46]


    return train_data, train_labels, val_data, val_labels
end #end function

function LoadBearVideo(basepath::String,nr_output_chan::Int)

  current_dir = pwd()

  #set data directories
  video_dir  = joinpath(basepath,"bear_images/")
  label_dir = joinpath(basepath,"bear_annotations/")

  #read file names
  all_video_files = readdir(video_dir)
  all_label_files = readdir(label_dir)


  #allocate tensors for data/label storage
  video  = zeros(Float32,3,480,854,77,1)
  mask   = zeros(Float32,480,854,77,2,1)
  all_video_files = all_video_files[1:77]
  all_label_files = all_label_files[1:77]

  #read training data
  cd(video_dir)
  for i=1:length(all_video_files)
      temp_img = Images.load(all_video_files[i])
      video[1:3,:,:,i,1] = Images.channelview(temp_img)
   end

  #permute to x-y-z-n_chan-n_examples
  video = permutedims(video,[2,3,4,1,5]);

  #repeat data to increase channel count explicitly for input-output
  repeat_factor = Int(nr_output_chan/3)
  video    = repeat(video,outer=[1,1,1,repeat_factor,1])


  #read training labels
  cd(label_dir)
  for i=1:length(all_label_files)
      temp_img = Images.load(all_label_files[i])
      mask[:,:,i,1,1] = Images.channelview(temp_img)
   end

   mask[:,:,:,2,1] = 1f0 .- mask[:,:,:,1,1]

   #subsample (not in a smart way..)
   video = video[1:2:end,1:2:end,:,:,:]
   mask = mask[1:2:end,1:2:end,:,:,:]

   #cut video so we can divide by 2 sufficiently many times
   video = video[:,1:424,1:72,:,:]
   mask  = mask[:,1:424,1:72,:,:]

    #create a vector with one label per entry
    data   = Vector{Array{Float32,5}}(undef,1)
    data_V = Vector{Array{Float32,5}}(undef,1)
    labels   = Vector{Array{Float32,5}}(undef,1)
    labels_V = Vector{Array{Float32,5}}(undef,1)

    data[1] = video
    data_V[1] = video
    labels[1] = mask
    labels_V[1] = mask

    cd(current_dir) #switch back to current working dir

    return data, labels, data_V, labels_V
end #end function
