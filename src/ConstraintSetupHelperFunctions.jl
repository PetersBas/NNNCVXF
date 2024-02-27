export geomap2boundaries, fill_geomap_bound_constraint, addTVconstraint, addAreaConstraint, SetDataConstraints, GetPatches, filter_small_patches, Proj_Hist_Bbox_ch1, Proj_Hist_Bbox_ch2

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
function GetPatches(data, labels, mask, use_label_chan, r, label_indicator, min_card, max_card)

    total_interest_map = Vector{Array{Float32,2}}(undef,length(data))
    interest_patches   = Vector{Any}(undef,length(data))
    max_card_patch     = Vector{Any}(undef,length(data))
    min_card_patch     = Vector{Any}(undef,length(data))


    for j=1:length(data)
        labels_temp = deepcopy(mask[j].*labels[j][:,:,use_label_chan])

        label_loc = findall(labels_temp .== label_indicator)
        total_interest_map[j] = zeros(Float32,size(data[j])[1:2])
        interest_patches[j]   = Vector{Vector{CartesianIndex{2}}}(undef,length(label_loc))
        max_card_patch[j]     = Vector{Float32}(undef,length(label_loc)) #give as a ratio
        min_card_patch[j]     = Vector{Float32}(undef,length(label_loc)) #give as a ratio
        for i=1:length(label_loc)
            temp_patch = zeros(Float32,size(data[j])[1:2])
            l_b = label_loc[i][2] - r; l_b = max.(l_b,8)
            l_e = label_loc[i][2] + r; l_e = min.(l_e,size(data[j])[2]-8)
            h_b = label_loc[i][1] - r; h_b = max.(h_b,8)
            h_e = label_loc[i][1] + r; h_e = min.(h_e,size(data[j])[1]-8)

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

function SetDataConstraints(data, patches_c1, patches_c2, min_card_patch_c1, max_card_patch_c1, min_card_patch_c2, max_card_patch_c2)
#one entry per example
    #one entry per channel
        #within each channel: one entry per constraint set

        #length: nr of examples
        A_CQ     = Vector{Any}(undef,length(data)) 
        P_sub_CQ = Vector{Any}(undef,length(data)) #nr of examples
        alpha_CQ = Vector{Any}(undef,length(data))
        for i=1:length(data)
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
                P_sub_CQ[i][j][k] = x -> Proj_Hist_Bbox_ch1(x,minimum(patches_c1[i][k]),maximum(patches_c1[i][k]),size(data[i])[1:2],min_card_patch_c1[i][k])
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
                P_sub_CQ[i][j][k] = x -> Proj_Hist_Bbox_ch1(x,minimum(patches_c2[i][ko_ind]),maximum(patches_c2[i][ko_ind]),size(data[i])[1:2],min_card_patch_c2[i][ko_ind])
                alpha_CQ[i][j][k] = 1f0#/( sum(all_train_patches[k].*temp_map)/sum(all_train_patches[k]) ) 
            end

        end

    return A_CQ, P_sub_CQ, alpha_CQ
end

function addAreaConstraint(data, A_CQ, P_sub_CQ, alpha_CQ, min_area_list, max_area_list, alpha_area=0.01f0)
    #add some global constraints on total area
    for i=1:length(data)
        
        #channel 1
        push!( A_CQ[i][1], [])#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][1], 1.234) #add dummy entry
        lb_area = zeros(Float32,prod(size(data[i])[1:2])); ub_area = ones(Float32,prod(size(data[i])[1:2]))
        min_area = round(Int,0.0f0 * prod(size(data[i])[1:2]))
        max_area = round(Int,max_area_list[i] * prod(size(data[i])[1:2])) 
        ub_area[1:end-max_area] .= 0.25
        lb_area[end-min_area:end] .= 0.75
        P_sub_CQ[i][1][end] =  x -> project_histogram_relaxed!(vec(x),lb_area,ub_area)
        append!( alpha_CQ[i][1], alpha_area)

        #channel 2
        push!( A_CQ[i][2], [])#SparseMatrixCSC{Float32}(LinearAlgebra.I,prod(size(data[i])[1:2]),prod(size(data[i])[1:2])) )
        append!( P_sub_CQ[i][2], 1.234)#add dummy entry
        lb_area2 = zeros(Float32,prod(size(data[i])[1:2])); ub_area2 = ones(Float32,prod(size(data[i])[1:2]))
        max_area2 = round(Int,(1f0-min_area_list[i]) * prod(size(data[i])[1:2])) #min for ki&iron: 0.05, for ps:17
        min_area2 = round(Int,0.0f0 * prod(size(data[i])[1:2]))#round(Int,(1 - 0.025f0) * prod(size(data[i])[1:2]))
        lb_area2[end-min_area2:end] .= 0.75
        ub_area2[1:end-max_area2] .= 0.25
        P_sub_CQ[i][2][end] =  x -> project_histogram_relaxed!(vec(x),lb_area2,ub_area2)
        append!( alpha_CQ[i][2], alpha_area)


    end
    return A_CQ, P_sub_CQ, alpha_CQ
end

#add global constraints on TV
function addTVconstraint(compgrid, data, A_CQ, P_sub_CQ, alpha_CQ, TV_list, alpha_TV=0.03f0)
    for i=1:length(data)
    
        comp_grid = compgrid((1.0,1.0),(size(data[i])[1], size(data[i])[2]))
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