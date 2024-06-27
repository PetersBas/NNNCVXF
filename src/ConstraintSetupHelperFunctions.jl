export geomap2boundaries, fill_geomap_bound_constraint, addTVconstraint, addAreaConstraint, SetDataConstraints, GetPatches, CheckPatches, filter_small_patches, Proj_Hist_Bbox_ch1, Proj_Hist_Bbox_ch2, addMonotonicityconstraint

function geomap2boundaries(geomap,buffer)

    boundaries = [zeros(Float32,1,size(geomap,2));diff(geomap,dims=1)] .+ [zeros(Float32,size(geomap,1),1) diff(geomap,dims=2)]
    #boundaries = diff(geological_maps[:,:,8],dims=1)
    boundary_mask = zeros(Float32,size(boundaries))
    for i = 1:size(boundary_mask,1)
        for j = 1:size(boundary_mask,2)
            if any(boundaries[max(1,i-buffer):min(i+buffer,size(boundaries,1)),max(1,j-buffer):min(j+buffer,size(boundaries,2))] .!= 0) 
                boundary_mask[i,j] = 1
            end
        end
    end

    return boundary_mask
end

function fill_geomap_bound_constraint(i,boundary_mask,boundary_inds,val_mask,geomap,alpha_CQ,A_CQ,P_sub_CQ,alpha_CQ_val,A_CQ_val,P_sub_CQ_val)
  #define bound constraints
    #ch1
    lb_ch1 = zeros(Float32,size(boundary_mask))
    ub_ch1 = ones(Float32,size(boundary_mask))
    geo_unit_inds = findall(geomap.==1)
    other_inds    = findall(geomap.==0)
    lb_ch1[geo_unit_inds] .= 0.55f0#where the geo unit of interest is present, assign 0.55 as minimum
    ub_ch1[other_inds]    .= 0.45f0#where the geo unit of non-interest is present, assign 0.45 as maximum

    #ch2
    lb_ch2 = zeros(Float32,size(boundary_mask))
    ub_ch2 = ones(Float32,size(boundary_mask))
    ub_ch2[geo_unit_inds] .= 0.45f0#where the geo unit of interest is present, assign 0.45 as maximum
    lb_ch2[other_inds]    .= 0.55f0#where the geo unit of non-interest is present, assign 0.55 as minimum

    #don't use boundary areas, leave them to be decided
    lb_ch1[boundary_inds] .= 0f0;  
    lb_ch2[boundary_inds] .= 0f0;  
    ub_ch1[boundary_inds] .= 1f0;  
    ub_ch2[boundary_inds] .= 1f0;  

    #use validation mask to split into training and validation
    val_inds = findall(val_mask .== 1)
    train_inds = findall(val_mask .== 0)

    lb_ch1_val = deepcopy(lb_ch1); lb_ch1_val[train_inds] .= 0f0
    lb_ch2_val = deepcopy(lb_ch2); lb_ch2_val[train_inds] .= 0f0
    ub_ch1_val = deepcopy(ub_ch1); ub_ch1_val[train_inds] .= 1f0
    ub_ch2_val = deepcopy(ub_ch2); ub_ch2_val[train_inds] .= 1f0

    lb_ch1[val_inds] .= 0f0;  
    lb_ch2[val_inds] .= 0f0;  
    ub_ch1[val_inds] .= 1f0;  
    ub_ch2[val_inds] .= 1f0;  

    lb_ch1 = vec( lb_ch1)
    lb_ch2 = vec( lb_ch2)
    ub_ch1 = vec( ub_ch1)
    ub_ch2 = vec( ub_ch2)

    lb_ch1_val = vec( lb_ch1_val)
    lb_ch2_val = vec( lb_ch2_val)
    ub_ch1_val = vec( ub_ch1_val)
    ub_ch2_val = vec( ub_ch2_val)

    alpha_CQ[i][1][1] = 1f0
    alpha_CQ[i][2][1] = 1f0
    alpha_CQ_val[i][1][1] = 1f0
    alpha_CQ_val[i][2][1] = 1f0

    A_CQ[i][1][1] = []
    A_CQ[i][2][1] = []
    A_CQ_val[i][1][1] = []
    A_CQ_val[i][2][1] = []

    P_sub_CQ[i][1][1] = x -> project_bounds!(deepcopy(x),deepcopy(lb_ch1),deepcopy(ub_ch1))
    P_sub_CQ[i][2][1] = x -> project_bounds!(deepcopy(x),deepcopy(lb_ch2),deepcopy(ub_ch2))
    P_sub_CQ_val[i][1][1] = x -> project_bounds!(deepcopy(x),deepcopy(lb_ch1_val),deepcopy(ub_ch1_val))
    P_sub_CQ_val[i][2][1] = x -> project_bounds!(deepcopy(x),deepcopy(lb_ch2_val),deepcopy(ub_ch2_val))


    return alpha_CQ,A_CQ,P_sub_CQ,alpha_CQ_val,A_CQ_val,P_sub_CQ_val
end

#for Camvid example
function GetPatches(labels, mask, use_label_chan, r, label_indicator, min_card, max_card,active_z_slice=[])

    total_interest_map = Vector{Array{Float32,2}}(undef,length(labels))
    interest_patches   = Vector{Any}(undef,length(labels))
    max_card_patch     = Vector{Any}(undef,length(labels))
    min_card_patch     = Vector{Any}(undef,length(labels))


    for j=1:length(labels)
        if isempty(active_z_slice)==false
            labels_temp = deepcopy(mask[j].*labels[j][:,:,active_z_slice,use_label_chan])  
        else
            labels_temp = deepcopy(mask[j].*labels[j][:,:,use_label_chan])
        end

        label_loc = findall(labels_temp .== label_indicator)
        total_interest_map[j] = zeros(Float32,size(labels[j])[1:2])
        interest_patches[j]   = Vector{Vector{CartesianIndex{2}}}(undef,length(label_loc))
        max_card_patch[j]     = Vector{Float32}(undef,length(label_loc)) #give as a ratio
        min_card_patch[j]     = Vector{Float32}(undef,length(label_loc)) #give as a ratio
        for i=1:length(label_loc)

            #label_loc[i]=label_loc[i]-CartesianIndex(1,1) #introduce error
            
            temp_patch = zeros(Float32,size(labels[j])[1:2])
            l_b = label_loc[i][2] - r; l_b = max.(l_b,8)
            l_e = label_loc[i][2] + r; l_e = min.(l_e,size(labels[j])[2]-8)
            h_b = label_loc[i][1] - r; h_b = max.(h_b,8)
            h_e = label_loc[i][1] + r; h_e = min.(h_e,size(labels[j])[1]-8)

            #patches need to be at least size 1, check and modify to comply
            if h_b>h_e
                println("patch setup error")
                h_e = h_e + (h_b-h_e + 1)
            end
            if h_b>h_e
                println("patch setup error, still")
                h_e = h_e + (h_b-h_e + 1)
            end
            if l_b>l_e
                println("patch setup error")
                l_e = l_e + (l_b-l_e + 1)
            end
            if l_b>l_e
                println("patch setup error, still")
                l_e = l_e + (l_b-l_e + 1)
            end

            temp_patch[h_b:h_e,l_b:l_e] .= 1
            #interest_patches[j][i] .= interest_patches[j][i].*(1 .- total_interest_map[j])
            total_interest_map[j] += temp_patch

            interest_patches[j][i] = findall(temp_patch .== 1)

            max_card_patch[j][i] = max_card
            min_card_patch[j][i] = min_card
        end
    end
    return total_interest_map, interest_patches, max_card_patch, min_card_patch
end

#for Bear video example
function GetPatches(data::Vector{Array{Float32, 5}}, labels, mask, use_label_chan, r, label_indicator, min_card, max_card)

    total_interest_map = Vector{Array{Float32,3}}(undef,length(data))
    interest_patches   = Vector{Any}(undef,length(data))
    max_card_patch     = Vector{Any}(undef,length(data))
    min_card_patch     = Vector{Any}(undef,length(data))


    for j=1:length(data)
        labels_temp = deepcopy(mask[j].*labels[j][:,:,:,use_label_chan])

        label_loc = findall(labels_temp .== label_indicator)

        total_interest_map[j] = zeros(Float32,size(data[j])[1:3])
        interest_patches[j]   = Vector{Vector{CartesianIndex{3}}}(undef,length(label_loc))
        max_card_patch[j]     = Vector{Float32}(undef,length(label_loc)) #give as a ratio
        min_card_patch[j]     = Vector{Float32}(undef,length(label_loc)) #give as a ratio
        for i=1:length(label_loc)
            

            temp_patch = zeros(Float32,size(data[j])[1:3])
            l_b = label_loc[i][2] - r; l_b = max.(l_b,8)
            l_e = label_loc[i][2] + r; l_e = min.(l_e,size(data[j])[2]-8)
            h_b = label_loc[i][1] - r; h_b = max.(h_b,8)
            h_e = label_loc[i][1] + r; h_e = min.(h_e,size(data[j])[1]-8)

            temp_patch[h_b:h_e,l_b:l_e,label_loc[i][3]] .= 1
            #interest_patches[j][i] .= interest_patches[j][i].*(1 .- total_interest_map[j])
            total_interest_map[j] += temp_patch

            interest_patches[j][i] = findall(temp_patch .== 1)

            max_card_patch[j][i] = max_card
            min_card_patch[j][i] = min_card
        end
    end
    return total_interest_map, interest_patches, max_card_patch, min_card_patch
end

function filter_small_patches(patches, max_card_patch, min_card_patch, threshold=40)
    #filter out patches that are very very small, so that we dont' use constraints on the general patch don't make much sense
    for j=1:length(patches)
        small_patches = []
        for i=1:length(patches[j])
            if length(patches[j][i])<threshold
                append!(small_patches,i)
            end
        end
        deleteat!(patches[j],small_patches)
        deleteat!(max_card_patch[j],small_patches)
        deleteat!(min_card_patch[j],small_patches)
    end

  return patches, max_card_patch, min_card_patch
end

function CheckPatches(patches)#,remove_empty_examples=false)
    empty_examples = []
    for i=1:length(patches)
        if isempty(patches[i]) == true
            println("warning: example has no patches")
            append!(empty_examples,i)
        else
            for k=1:length(patches[i])
                if isempty(patches[i][k]) == true
                    println("warning: example has an empty patch")
                    append!(empty_examples,i)
                end
            end
        end
    end
    # if remove_empty_examples == true
    #     delete_list = unique(delete_list)
    #     deleteat!(patches,delete_list)
    # end
    return patches, empty_examples
end

function Proj_Hist_Bbox_ch1(input,tl,rb,orig_size,min_card_patch)
    proj = reshape(deepcopy(input),orig_size)
    #max_area = round(Int,max_card_patch*prod(size(tl:rb)))
    min_area = round(Int,min_card_patch*prod(size(tl:rb)))
    #the last min_area entries are equal to 1, so the first N-max_are entries are equal to, say, 0.4
    lb_area = zeros(Float32,prod(size(tl:rb))); 
    ub_area = ones(Float32,prod(size(tl:rb)))
    lb_area[end-min_area+1:end] .= 0.75f0
    proj[tl:rb] .= reshape(project_histogram_relaxed!(vec(proj[tl:rb]),lb_area,ub_area),size(tl:rb))

    return vec(proj)
end

#tl:top-left
#rb:right-bottom
function Proj_Hist_Bbox_ch2(input,tl,rb,orig_size,max_card_patch)
    proj = reshape(deepcopy(input),orig_size)
    #max_area = round(Int,(1-min_card_patch)*prod(size(tl:rb)))
    max_area = round(Int,max_card_patch*prod(size(tl:rb)))
    lb_area = zeros(Float32,prod(size(tl:rb))); 
    ub_area = ones(Float32,prod(size(tl:rb)))
    ub_area[1:end-max_area] .= 0.4
    proj[tl:rb] .= reshape(project_histogram_relaxed!(vec(proj[tl:rb]),lb_area,ub_area),size(tl:rb))


    #proj[tl:rb] .= reshape(project_cardinality!(vec(proj[tl:rb]),upper),size(tl:rb))
    #proj[tl:rb] .= reshape(project_l1_Duchi!(vec(proj[tl:rb]),1f0*upper),size(tl:rb))
    #proj = reshape(project_bounds!(vec(proj),0.0f0,1.0f0),size(input)[1:2])
    return vec(proj)
end

#Define channel 1 to be the class of interest
function Proj_Card_Bbox_ch1(input,tl,rb,orig_size,max_card_patch)
    proj = reshape(deepcopy(input),orig_size)
    upper = round(Int,max_card_patch*prod(size(tl:rb)))
    proj[tl:rb] .= reshape(project_cardinality!(vec(proj[tl:rb]),upper),size(tl:rb))
    #proj[tl:rb] .= reshape(project_cardinality!(vec(proj[tl:rb]),upper),size(tl:rb))
    #proj[tl:rb] .= reshape(project_l1_Duchi!(vec(proj[tl:rb]),1f0*upper),size(tl:rb))
    #proj = reshape(project_bounds!(vec(proj),0.0f0,1.0f0),size(input)[1:2])
    return vec(proj)
end
  
  #define channel 2 as the class of other/non-interest
  function Proj_Card_Bbox_ch2(input,tl,rb,orig_size,min_card_patch)
    proj = reshape(deepcopy(input),orig_size)
    upper = round(Int,(1-min_card_patch)*prod(size(tl:rb)))
    proj[tl:rb] .= reshape(project_cardinality!(vec(proj[tl:rb]),upper),size(tl:rb))
    #proj[tl:rb] .= reshape(project_l1_Duchi!(vec(proj[tl:rb]),1f0*upper),size(tl:rb))
    return vec(proj)
end

function SetDataConstraints(labels, patches_c1, patches_c2, min_card_patch_c1, max_card_patch_c1, min_card_patch_c2, max_card_patch_c2)
#one entry per example
    #one entry per channel
        #within each channel: one entry per constraint set

        #length: nr of examples
        A_CQ     = Vector{Any}(undef,length(labels)) 
        P_sub_CQ = Vector{Any}(undef,length(labels)) #nr of examples
        alpha_CQ = Vector{Any}(undef,length(labels))
        for i=1:length(labels)
            #temp_map = sum(patches_c1[i])

            A_CQ[i]     = Vector{Any}(undef,2) #one entry per channel - per example
            P_sub_CQ[i] = Vector{Any}(undef,2)
            alpha_CQ[i] = Vector{Any}(undef,2)
            #nr_constraints_example = length(all_train_patches)

            for j=1:2 #initialize per channel
                A_CQ[i][j]     = Vector{Any}(undef,length(patches_c1[i])+length(patches_c2[i])) #within each channel: one entry per constraint set
                P_sub_CQ[i][j] = Vector{Any}(undef,length(patches_c1[i])+length(patches_c2[i])) #within each channel: one entry per constraint set
                alpha_CQ[i][j] = Vector{Any}(undef,length(patches_c1[i])+length(patches_c2[i])) #within each channel: one entry per constraint set
            end

            for k=1:length(patches_c1[i]) #loop over data patches (constraints) per example
                j=1 #channel/class 1   
                A_CQ[i][j][k]     = []
                P_sub_CQ[i][j][k] = x -> Proj_Hist_Bbox_ch1(x,minimum(patches_c1[i][k]),maximum(patches_c1[i][k]),size(labels[i])[1:2],min_card_patch_c1[i][k])
                alpha_CQ[i][j][k] = 1f0
                j=2 #channel/class 2 (dummy)  
                A_CQ[i][j][k]     = []
                P_sub_CQ[i][j][k] = x -> x
                alpha_CQ[i][j][k] = 1f0
            end
            #for the negative/non-interest/class-2 patches:
            for k=1+length(patches_c1[i]):length(patches_c1[i])+length(patches_c2[i])
                ko_ind = k - length(patches_c1[i])
                j=1 #channel/class 1 (dummy)
                A_CQ[i][j][k]     = []
                P_sub_CQ[i][j][k] = x -> x
                alpha_CQ[i][j][k] = 1f0
                j=2
                A_CQ[i][j][k]     = []#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(all_train_patches[k])),prod(size(all_train_patches[k])))
                P_sub_CQ[i][j][k] = x -> Proj_Hist_Bbox_ch1(x,minimum(patches_c2[i][ko_ind]),maximum(patches_c2[i][ko_ind]),size(labels[i])[1:2],min_card_patch_c2[i][ko_ind])
                alpha_CQ[i][j][k] = 1f0#/( sum(all_train_patches[k].*temp_map)/sum(all_train_patches[k]) ) 
            end

        end

    return A_CQ, P_sub_CQ, alpha_CQ
end

function addAreaConstraint(labels, A_CQ, P_sub_CQ, alpha_CQ, min_area_list, max_area_list, alpha_area=0.01f0)
    #add some global constraints on total area
    for i=1:length(labels)
        
        #channel 1
        push!( A_CQ[i][1], [])#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][1], 1.234) #add dummy entry
        lb_area = zeros(Float32,prod(size(labels[i])[1:2])); ub_area = ones(Float32,prod(size(labels[i])[1:2]))
        min_area = round(Int,0.0f0 * prod(size(labels[i])[1:2]))
        max_area = round(Int,max_area_list[i] * prod(size(labels[i])[1:2])) 
        ub_area[1:end-max_area] .= 0.25
        lb_area[end-min_area:end] .= 0.75
        P_sub_CQ[i][1][end] =  x -> project_histogram_relaxed!(vec(x),lb_area,ub_area)
        append!( alpha_CQ[i][1], alpha_area)

        #channel 2
        push!( A_CQ[i][2], [])#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][2], 1.234)#add dummy entry
        lb_area2 = zeros(Float32,prod(size(labels[i])[1:2])); ub_area2 = ones(Float32,prod(size(labels[i])[1:2]))
        max_area2 = round(Int,(1f0-min_area_list[i]) * prod(size(labels[i])[1:2])) #min for ki&iron: 0.05, for ps:17
        min_area2 = round(Int,0.0f0 * prod(size(labels[i])[1:2]))#round(Int,(1 - 0.025f0) * prod(size(data[i])[1:2]))
        lb_area2[end-min_area2:end] .= 0.75
        ub_area2[1:end-max_area2] .= 0.25
        P_sub_CQ[i][2][end] =  x -> project_histogram_relaxed!(vec(x),lb_area2,ub_area2)
        append!( alpha_CQ[i][2], alpha_area)


    end
    return A_CQ, P_sub_CQ, alpha_CQ
end

#add global constraints on TV
function addTVconstraint(compgrid, labels, A_CQ, P_sub_CQ, alpha_CQ, TV_list, alpha_TV=0.03f0)
    for i=1:length(labels)
    
        comp_grid = compgrid((1.0,1.0),(size(labels[i])[1], size(labels[i])[2]))
        TD_OP, ~, ~, ~, ~ = get_TD_operator(comp_grid,"TV",Float32)

        push!( A_CQ[i][1], TD_OP)#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][1], 1.234) #add dummy entry
        P_sub_CQ[i][1][end] =  x -> project_l1_Duchi!(vec(deepcopy(x)), TV_list[i])# 50000f0)#42286f0)#0.0125f0 * prod(size(data[i])[1:2]) ) 
        append!( alpha_CQ[i][1], alpha_TV)

        push!( A_CQ[i][2], TD_OP)#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][2], 1.234)#add dummy entry
        P_sub_CQ[i][2][end] =  x -> x#project_l1_Duchi!(vec(deepcopy(x)), 26850f0)#project_cardinality!(vec(x), prod(size(data[i])[1:2]) ) 
        append!( alpha_CQ[i][2], alpha_TV)

    end
    return A_CQ, P_sub_CQ, alpha_CQ
end

#add global constraints on the monotonicity, decreasing monotonically from left->right of the image. TD_OP is forward 1st order finite difference.
function addMonotonicityconstraint(compgrid, labels, A_CQ, P_sub_CQ, alpha_CQ, alpha_Mono=0.03f0)
    for i=1:length(labels)

        n = size(labels[i])
        comp_grid = compgrid((1.0,1.0),(n[1], n[2]))
        TD_OP, ~, ~, ~, ~ = get_TD_operator(comp_grid,"D_x",Float32)

        #channel 1
        push!( A_CQ[i][1], TD_OP)#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][1], 1.234) #add dummy entry
        P_sub_CQ[i][1][end] =  x -> project_bounds!(vec(deepcopy(x)), zeros(Float32,prod(n)),1f6.*ones(Float32,prod(n)))#
        append!( alpha_CQ[i][1], alpha_Mono)
        

        #channel 2 - apply this constraint to channel 1 only, so put dummy projection (identity) here
        push!( A_CQ[i][2], TD_OP)#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][2], 1.234)#add dummy entry
        P_sub_CQ[i][2][end] =  x -> x
        append!( alpha_CQ[i][2], alpha_Mono)

    end
    return A_CQ, P_sub_CQ, alpha_CQ
end