local patchExtraction = {}

assert(opt.patchDim==3 or opt.patchDim==2)
patchExtraction.isVol = opt.patchDim==3

if patchExtraction.isVol then
    
    --package.path = "itkslave/?.lua;" .. package.path
    local itkslave = require('itkslave.itkslave')
    
    --------------------------------
    -- samples a cubic patch (random side length, random position, random rotation; all uniform)
    -- [scale and rot may additionally get jittered in extractPatch()]
    function patchExtraction.samplePatch(input, template)
        local side = template and template.s or math.ceil(opt.patchSize * (opt.patchSampleGlobalTr and 1 or torch.uniform(opt.patchSampleMinScaleF, opt.patchSampleMaxScaleF)))
        side = math.min(side, input:size(1)-1, input:size(2)-1, input:size(3)-1)
        local z1 = math.ceil(torch.uniform(1e-2, input:size(1)-side))
        local x1 = math.ceil(torch.uniform(1e-2, input:size(3)-side))
        local y1 = math.ceil(torch.uniform(1e-2, input:size(2)-side))
        local rot = template and template.r or (opt.patchSampleGlobalTr and 0 or torch.uniform(-math.pi, math.pi) * opt.patchSampleRotMaxPercA)
        local axis = template and template.a or {torch.normal(0,1), torch.normal(0,1), torch.normal(0,1)}
        return {p={{z1,z1 + side-1}, {y1,y1 + side-1}, {x1,x1 + side-1}}, r=rot, s=side, a=axis}
    end
    
    function patchExtraction.samplePatchInbetween(idx1, idx2)
        local w = torch.uniform() --[0,1)
        local side = math.ceil( w * idx1.s + (1-w) * idx2.s )
        
        local pos = {{},{},{}}
        for i=1,3 do
            pos[i][1] = math.floor( w * idx1.p[i][1] + (1-w) * idx2.p[i][1] )
            pos[i][2] = pos[i][1] + side-1
        end
        
        local delta = idx1.r-idx2.r
        local rcomp = delta>math.pi and 2*math.pi or (delta<-math.pi and -2*math.pi or 0)
        local rot = w * idx1.r + (1-w) * (idx2.r + rcomp)
        
        idx1.a = idx2.a --! a bit hacky. in the general case we would need to do lerp on quaternions: vnl_quaternion ; http://physicsforgames.blogspot.fr/2010/02/quaternions.html
        return {p=pos, r=rot, s=side, a=idx1.a}, idx1
    end    
    
    --------------------------------
    -- moves the box by some random offset (but still keeps it within image)
    function patchExtraction.offsetBox(box, input)               
        local off = torch.Tensor(3):uniform(-opt.patchSampleSimulUnalignedOffset, opt.patchSampleSimulUnalignedOffset):round()
        for i=1,#box do
            if off[i]<0 then off[i] = math.max(1, box[i][1]+off[i]) - box[i][1] else off[i] = math.min(input:size(i), box[i][2]+off[i]) - box[i][2] end
            box[i][1] = box[i][1] + off[i]
            box[i][2] = box[i][2] + off[i]
        end
        --todo: jitter also transfpar?
        --todo: maybe different noise? gaussian or multimodal (uniform yes/now, if yes then gaussian)
        return box
    end

    --------------------------------
    -- Extracts a 3D patch from volume.
    -- Optionally performs randomized rotation and scaling (normal d). Surrounding data need to be available, doesn't do any zero-padding.
    -- Note that trilinear (?) interpolation introduces smoothing artifacts 
    function patchExtraction.extractPatch(input, indices, jitter)
        if indices.r~=0 or indices.s~=opt.patchSize or (jitter and (opt.patchJitterRotMaxPercA > 0 or opt.patchJitterMaxScaleF > 1)) then
            local patchCenter = {}
            for i=1,#indices.p do patchCenter[i] = (indices.p[i][2] + indices.p[i][1])/2 end
            local srcPatch
            
            -- sample rotation and scaling until we fit into the available space
            local ok = false
            local patchJitterRotMaxPercA = jitter and opt.patchJitterRotMaxPercA or 0
            local patchJitterMaxScaleF = jitter and opt.patchJitterMaxScaleF or 1
            local iters = jitter and 100 or 1
            local alpha, axis, sc = 0, {1,0,0}, 1
            for a=1,iters do
                alpha = indices.r + torch.uniform(-math.pi, math.pi) * patchJitterRotMaxPercA
                for d=1,3 do axis[d] = indices.a[d] + torch.normal(0, patchJitterRotMaxPercA + 1e-10) end
                if patchJitterMaxScaleF > 1 then
                    sc = torch.normal(1, (patchJitterMaxScaleF-1)/2) --in [1/f;f] with 95% prob
                    sc = math.max(math.min(sc, patchJitterMaxScaleF), 1/patchJitterMaxScaleF)
                end                  
                 
                --compute inverse transformation of a axis-aligned box centered at (0,0,0) with vertices at points like (1,1,1) 
                -- to get the source area, the bounding box of which we need to crop (defined as box, at least as big as destbox)
                local box = itkslave.getSourceBox(axis, alpha, sc)               
                
                --try to crop it  
                local srcIndices = {}
                ok = true
                for i=1,#indices.p do   
                    local mi = torch.cmin(box:min(1):squeeze(), -1) * (indices.p[i][2] - indices.p[i][1] +1)/2
                    local ma = torch.cmax(box:max(1):squeeze(), 1) * (indices.p[i][2] - indices.p[i][1] +1)/2                         
                    srcIndices[i] = {math.floor(patchCenter[i] + mi[i]), math.ceil(patchCenter[i] + ma[i])} 
                    if srcIndices[i][1]<1 or srcIndices[i][2]>input:size(i) then ok = false; break end
                end

                if ok then            
                    srcPatch = input[srcIndices]
                    break
                end
            end
            
            if not ok then return input[indices.p]:squeeze(), false end

            --transform the crop
            sc = sc * opt.patchSize/indices.s       
            local dstPatch = itkslave.transformVolume(axis, alpha, sc, {opt.patchSize,opt.patchSize,opt.patchSize}, srcPatch, opt.patchSampleGPU)
            --writePatches(input[indices.p],input[indices.p],indices.p[1][2],true,{0,0,0})
            return dstPatch, true
        
        else
            return input[indices.p]:squeeze(), true
        end    
    end
    
    function patchExtraction.plotPatches(out)
        image.display{image=out[1], zoom=2, legend='Input1', padding=1, nrow=math.ceil(math.sqrt(out[1]:size(1)))}  
        image.display{image=out[2], zoom=2, legend='Input2', padding=1, nrow=math.ceil(math.sqrt(out[1]:size(1)))}  
    end  



else
    --------------------------------
    -- samples oH x oW patch from random slice of random dimension (in case of volumetric input)
    function patchExtraction.samplePatch(oW, oH, oD, input)
        assert(oD <= 1) 
        if input:dim()==3 then
            local dim = math.ceil(torch.uniform(1e-2, 3))
            local sliceidx = math.ceil(torch.uniform(1e-2, input:size(dim)))
            local sizes = torch.totable(input:size())        
            table.remove(sizes, dim)
            local x1 = math.ceil(torch.uniform(1e-2, sizes[2]-oW))
            local y1 = math.ceil(torch.uniform(1e-2, sizes[1]-oH))                    
            local indices = {{y1,y1 + oH-1}, {x1,x1 + oW-1}}
            table.insert(indices,dim,{sliceidx, sliceidx})
            return indices
        else
            local x1 = math.ceil(torch.uniform(1e-2, input:size(2)-oW))
            local y1 = math.ceil(torch.uniform(1e-2, input:size(1)-oH))                
            return {{y1,y1 + oH-1}, {x1,x1 + oW-1}}
        end 
    end

    --------------------------------
    -- Extracts a 2D patch from volume.
    -- Optionally performs randomized rotation (uniform d) and scaling (normal d). Surrounding data need to be available, doesn't do any zero-padding.
    -- Note that bilinear interpolation introduces smoothing artifacts 
    function patchExtraction.extractPatch(input, indices, transfpar)
        if opt.patchSampleRotMaxPercA ~= 0 or opt.patchSampleMaxScaleF ~= 1 then 
            -- determine available space around patch
                --note: the inverse transformation idea used at teh 3D version is much nicer:)
            local availablePad = 1e10
            for i=1,#indices do
                if indices[i][1]~=indices[i][2] then
                    availablePad = math.min(availablePad, math.min(indices[i][1] - 1, input:size(i) - indices[i][2]))
                end
            end
            
            -- sample rotation and scaling until we fit into the available space
            local ok = false
            local alpha, sc, requiredPad = 0, 1, 0
            for a=1,100 do
                if transfpar then
                    alpha, sc = unpack(transfpar)
                else
                    alpha = torch.uniform(-math.pi, math.pi) * opt.patchSampleRotMaxPercA
                    if opt.patchSampleMaxScaleF > 1 then
                        sc = torch.normal(1, (opt.patchSampleMaxScaleF-1)/2) --in [1/f;f] with 95% prob
                        sc = math.max(math.min(sc, opt.patchSampleMaxScaleF), 1/opt.patchSampleMaxScaleF)
                    end     --sc has here the inverse meaning as in the 3D version:)
                    if opt.patchSampleMaxScaleF < -1 then
                        sc = torch.uniform(1, -opt.patchSampleMaxScaleF) --just upscale (pyramid mode)    
                    end
                    
                    transfpar = {alpha, sc}
                end            

                -- norm distance of box corner point to rot center (not tight, but ok)
                local rotFactor = math.max(math.abs(math.cos(alpha-math.pi/4)), math.abs(math.sin(alpha-math.pi/4))) / math.cos(math.pi/4)          
                requiredPad = math.ceil( opt.patchSize/2 * (sc*rotFactor - 1) )
                if requiredPad < availablePad then
                    ok = true
                    break
                end        
            end
            if not ok then return input[indices]:squeeze() end
        
            local patchEx = input[boxPad(indices, requiredPad, 0)]:squeeze()
            
            -- rotate & crop center
            if (alpha ~= 0) then
                patchEx = image.rotate(patchEx, alpha, 'bilinear')
                local s = math.ceil((patchEx:size(1) - sc*opt.patchSize)/2)
                local cidx = {s, s + math.floor(sc*opt.patchSize)-1}
                patchEx = patchEx[{cidx, cidx}]
            end
    
            -- scale
            if (sc ~= 1) then
                patchEx = image.scale(patchEx, opt.patchSize, opt.patchSize, 'bilinear')
            end    
    
            return patchEx, transfpar
        
        else
            return input[indices]:squeeze()
        end
    end
    
    function patchExtraction.plotPatches(out)
        image.display{image=out[1], zoom=2, legend='Input1', padding=1, nrow=1}
        image.display{image=out[2], zoom=2, legend='Input2', padding=1, nrow=1}  
    end    
end


return patchExtraction