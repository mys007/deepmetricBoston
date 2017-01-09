--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
package.path = "../myrock/?.lua;../?.lua;" .. package.path
require 'image'
require 'myutils'
paths.dofile('dataset.lua')
require 'util'
require 'torchzlib'
local pe = require 'patchExtraction'
local itkslave = require('itkslave.itkslave')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = os.getenv('HOME')..'/datasets/cache/donkeyModapairs'..opt.dataset..'.trainCache_s'..opt.trainSplit..'_'..opt.modality1..opt.modality2..'.t7'
local testCache = os.getenv('HOME')..'/datasets/cache/donkeyModapairs'..opt.dataset..'.testCache_'..opt.modality1..opt.modality2..'.t7'
local meanstdCache = os.getenv('HOME')..'/datasets/cache/donkeyModapairs'..opt.dataset..(opt.normalize01 and 'N01' or '')..'.meanstdCache_s'..opt.trainSplit..'_'..opt.modality1..opt.modality2..'.t7'
local datapath = os.getenv('HOME')..'/datasets/'..opt.dataset
local patchdir = os.getenv('HOME')..'/datasets/'..opt.dataset..'/volumes'
local modalitiesext = {opt.modality1..'.t7img.gz', opt.modality2..'.t7img.gz'}

local sampleSize, maxBlacks
if pe.isVol then
    sampleSize = {2, opt.patchSize, opt.patchSize, opt.patchSize}
    maxBlacks = sampleSize[2]*sampleSize[3]*sampleSize[4]*opt.patchSampleMaxBlacks
else
    sampleSize = {2, opt.patchSize, opt.patchSize}
    maxBlacks = sampleSize[2]*sampleSize[3]*opt.patchSampleMaxBlacks
end    

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- Check for existence of opt.data
if not os.execute('cd ' .. datapath) then
    error(("could not chdir to '%s'"):format(opt.data))
end

--------------------------------
local function loadImage(path, modality)
    assert(path~=nil)
    local input = string.ends(path,'t7img') and torch.load(path) or (string.ends(path,'t7img.gz') and torch.load(path):decompress() or image.load(path))
    if input:dim() == 2 then -- 1-channel image loaded as 2D tensor
    elseif input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
        input = input:view(input:size(2), input:size(3))
    elseif input:dim() == 3 and input:size(1) == 3 then -- 3-channel image
        input = input[1]
    end    

    if opt.normalize01 then    
        local vmask
        if opt.blackenInvalids and opt.dataset=='BITE' and modality=='US' then
            vmask = torch.gt(input,0)
        else
            vmask = torch.ge(input,0)
        end
        local valids = input[vmask]
        local mi, ma = valids:min(), valids:max()
        valids:csub(mi):div(ma-mi+1e-8)
        input:fill(-1e-3)
        input:maskedCopy(vmask, valids)
    end 

    return input
end

--------------------------------
-- only one modality is indexed by dataset. The other modalities can be found in patchdir
local function loadImagePair(path, traintime, doPos)
    assert(path~=nil) 
    --if opt.dataset=='BITE' and opt.biteSkipUStoMR and traintime then torch.manualSeed(torch.random(1,10000)) end
    --if opt.dataset=='BITE' and opt.biteSkipUStoMR and traintime then path = path:gsub('([0-9us]+)-MR','01us-MR'); torch.manualSeed(torch.random(1,1000)) end
    if opt.dataset=='BITE' and opt.biteSkipUStoMR then path = path:gsub('([0-9]+)-MR','%1us-MR') end
    if opt.dataset=='BITE' and opt.biteSkipMRtoUS then path = path:gsub('([0-9]+)us-MR','%1-MR') end
    if opt.dataset=='BITE' and opt.modality1~='MR' then path = paths.concat(patchdir, string.sub(paths.basename(path),1,-string.len(modalitiesext[1])-1)..modalitiesext[1]) end  
    local mod2path = paths.concat(patchdir, string.sub(paths.basename(path),1,-string.len(modalitiesext[1])-1)..modalitiesext[2])  

    if traintime then
        if opt.dictIXIPerturbed and opt.dictIXIPerturbed[paths.basename(mod2path)] then 
            mod2path = opt.dictIXIPerturbed[paths.basename(mod2path)] 
        elseif opt.patchSampleDeformProb>0 then
            -- deformed examples are precomputed, online bspline resampling would be too expensive
            if doPos and torch.bernoulli(opt.patchSampleDeformProb)==1 then
                path = paths.concat('/media/simonovm/Vast/datasets/IXI_def10-15', paths.basename(path):sub(3))
                mod2path = paths.concat('/media/simonovm/Vast/datasets/IXI_def10-15', paths.basename(mod2path):sub(3))
            end
            if not doPos and torch.bernoulli(opt.patchSampleDeformProb)==1 then
                path = paths.concat('/media/simonovm/Vast/datasets/IXI_def10-15', paths.basename(path):sub(3))
            end
            if not doPos and torch.bernoulli(opt.patchSampleDeformProb)==1 then
                mod2path = paths.concat('/media/simonovm/Vast/datasets/IXI_def10-15', paths.basename(mod2path):sub(3))
            end
        end
    end    
    
    local input1 = loadImage(path, opt.modality1)
    local input2 = loadImage(mod2path, opt.modality2)
    
    if opt.dataset=='BITE' and not opt.biteSkipUStoMR then    
        --special case for UStoMR in BITE: lot of black around, it can be cropped. So find just the informative part + margin around
        local input = opt.modality2=='US' and input2 or input1
        local range = {}
        for d=1,3 do
            local proj = input
            for k=1,3 do if k~=d then proj = proj:max(k) end end
            proj = torch.data(proj:squeeze():ge(1e-6))

            range[d] = {}
            for i=0,input:size(d)-1 do
                if proj[i]==1 then range[d][1] = math.max(1, i+1 - opt.patchSize * opt.patchSampleMaxScaleF); break end
            end 
            for i=input:size(d)-1,0,-1 do
                if proj[i]==1 then range[d][2] = math.min(input:size(d), i+1 + opt.patchSize * opt.patchSampleMaxScaleF); break end
            end     
        end
        input1 = input1[range]
        input2 = input2[range]
    end
    
    -- global per-image resample, instead of per-patch   
    if opt.patchSampleGlobalTr then
        local s = 1/torch.uniform(opt.patchSampleMinScaleF, opt.patchSampleMaxScaleF)
        local rot = torch.uniform(-math.pi, math.pi) * opt.patchSampleRotMaxPercA
        local axis = {torch.normal(0,1), torch.normal(0,1), torch.normal(0,1)}
        input1 = itkslave.transformVolume(axis, rot, s, {-1,-1,-1}, input1, false)
        input2 = itkslave.transformVolume(axis, rot, s, {-1,-1,-1}, input2, false)
    end
    
    return input1, input2
end

--------------------------------
local function extractPatch(input, indices, jitter)
    assert(input and indices)
    local out, ok = pe.extractPatch(input, indices, jitter)
    -- ignore invalid patches (partial registration, missing values in one patch; don't assume any default val) ---this is maybe too restrictive here, can handled by maxBlacks
    local allvalid = opt.blackenInvalidsFix or not torch.any(torch.lt(out,0))
    return out, allvalid and ok
end

--check patches for blackness before applying resampling (rot), saves a lot of time (esp. on BITE); returns ok
local function earlyPrune(input1, indices1, input2, indices2)
    if opt.patchSampleMaxBlacks==1 then return true end
    local crop = input1[indices1.p]
    local o1b = torch.lt(crop,1e-6):sum()/crop:numel() > opt.patchSampleMaxBlacks
    crop = input2[indices2.p]
    local o2b = torch.lt(crop,1e-6):sum()/crop:numel() > opt.patchSampleMaxBlacks       --fix 01/03 21:52  (before it was more strict on large patches, see below)
    --local o1b, o2b = torch.lt(input1[indices1.p],1e-6):sum()>maxBlacks, torch.lt(input2[indices2.p],1e-6):sum()>maxBlacks
    return not ( (opt.patchSampleBlackPairs and o1b and o2b) or (not opt.patchSampleBlackPairs and (o1b or o2b)) )
end

local function writePatches(out1,out2,i,doPos,extra)
    local function dispAndZoom(src, zoom)
        local img = image.toDisplayTensor{input=src, min=0, max=1, padding=1, nrow=math.ceil(math.sqrt(src:size(1)-3))}
        return image.scale(img, img:size(img:dim())*zoom, img:size(img:dim()-1)*zoom, 'simple')
    end

    local plotpath = '/home/simonovm/tmp/patches'
    image.save(plotpath..'/'..(doPos and 'p' or 'n').. i..extra[1]..'e'..extra[2]..'e'..extra[3]..'_1.png', dispAndZoom(out1/1000,2)) --/1000
    image.save(plotpath..'/'..(doPos and 'p' or 'n').. i..extra[1]..'e'..extra[2]..'e'..extra[3]..'_2.png', dispAndZoom(out2/1000,2)) --/1000
end

--------------------------------
local function processImagePair(dataset, path, nSamples, traintime)
    assert(traintime~=nil)
    
    --TODO:
    -- - write 2d code to use 1xHxW instead of HxW ... more portable then
    
    collectgarbage()
    local doPos = paths.basename(paths.dirname(path)) == 'pos'    
    local input1, input2 = loadImagePair(path, traintime, doPos)
    local oW, oH, oD
    if pe.isVol then oW, oH, oD = sampleSize[4], sampleSize[3], sampleSize[2] else oW, oH, oD = sampleSize[3], sampleSize[2], 1 end
    local nExtras = opt.regression == 'xyz' and 3 or (opt.regression == 'd' and 1 or 0)
    local extrainfo = torch.Tensor(nSamples, 1+nExtras):fill(0) --dim1: weight, other dims: regression target
    local output = pe.isVol and torch.Tensor(nSamples, 2, oD, oH, oW) or torch.Tensor(nSamples, 2, oW, oH)

    for s=1,nSamples do
        local out1, out2
        local ok = false
                    
        for a=1,1000 do
            local in1idx = pe.samplePatch(input1)
            
            if not doPos then 
                -- rejective sampling for neg position
               for b=1,200 do
                   local in2idx = pe.samplePatch(input2, opt.patchSampleNegSameTransf and in1idx or nil)
                   local closeness

                   if opt.patchSampleNegDist=='center' then
                        local meanside = (in1idx.p[2][2] - in1idx.p[2][1] + 1)/2 + (in2idx.p[2][2] - in2idx.p[2][1] + 1)/2
                        local reldist = boxCenterDistance(in1idx.p, in2idx.p) / (math.sqrt(2)*meanside) 
                        local distLimit = opt.patchSampleNegThres
                        ok = (distLimit>0 and reldist >= distLimit) or (distLimit<0 and reldist <= -distLimit and reldist > 0)
                        closeness = 1-distLimit         
                   elseif opt.patchSampleNegDist=='inter' then
                        local inter, union
                        if pe.isVol then
                            inter, union = boxIntersectionUnion(in1idx.p, in2idx.p)
                        else
                            local pad3d = sampleSize[2]/10/2
                            inter, union = boxIntersectionUnion(boxPad(in1idx.p, 0, pad3d), boxPad(in2idx.p, 0, pad3d))                                        
                        end
                        local iou = inter / union   --note: <1 if one box is included in other, even if centered. note: before 2016 we had just intersection (degrades slower)
                        local limit = opt.patchSampleNegThres
                        ok = (limit>0 and iou <= limit) or (limit<0 and iou >= -limit and iou < 1) or (limit==0 and iou < 1)
                        closeness = iou                     
                    else
                        assert(false)
                    end
 
                    if ok then
                        ok = earlyPrune(input1, in1idx, input2, in2idx)
                    end
                    if ok then                     
                        out2, ok = extractPatch(input2, in2idx, true)
                        if ok and opt.patchSampleDeformNegByProb and torch.bernoulli(math.max(closeness,0)/2)==1 then
                            -- sampling is optimization, deformation in far negatives doesn't matter anyhow. weakness: minimalistic mesh sizes doesn't preserve center point,
                            out2 = itkslave.jitterVolumeBsplines(3, torch.uniform(3,10), out2, opt.patchSampleGPU) -- so we don't use it before transformVolume
                            -- alternative thing: sample input2@in1idx sometimes and deform it
                        end
                        if opt.regression == 'xyz' then
                            for d=1,#in2idx.p do extrainfo[s][d+1] = (in1idx.p[d][1]-in2idx.p[d][1]) / (in1idx.p[d][2]-in1idx.p[d][1]+1) end 
                        elseif opt.regression == 'd' then 
                            extrainfo[s][2] = closeness
                        end
                        break
                    end
                end    
            else   
                ok = earlyPrune(input1, in1idx, input2, in1idx)
                if ok then       
                    out2, ok = extractPatch(input2, in1idx, true)
                end
                if opt.regression == 'd' then extrainfo[s][2] = 1 end                  
            end   
                     
            if ok then
                if traintime and opt.patchSampleSimulUnalignedOffset>0 then
                    in1idx.p = pe.offsetBox(in1idx.p, input1)
                end
                out1, ok = extractPatch(input1, in1idx)
            end      
                 
            -- ignore boring black patch pairs (they could be both similar and dissimilar)
            -- TODO: maybe uniform patches are bad, so check for std dev          
            if ok then
                local o1s, o2s = torch.lt(out1,1e-6):sum(), torch.lt(out2,1e-6):sum()
                local o1b, o2b
                if opt.patchSampleBlackByProb then
                    o1b = torch.bernoulli(o1s/maxBlacks)==1; o2b = torch.bernoulli(o2s/maxBlacks)==1
                else
                    o1b = o1s > maxBlacks; o2b = o2s > maxBlacks
                end

                if (opt.patchSampleBlackPairs and o1b and o2b) or (not opt.patchSampleBlackPairs and (o1b or o2b)) then
                    ok = false
                elseif opt.blackenInvalids then
                    local mask = torch.cmin(out1,out2):lt(0)
                    local ninv = mask:sum()
                    if ninv > maxBlacks then
                        ok = false
                    elseif ninv > 0 then
                        out1[mask] = 0
                        out2[mask] = 0
                    end
                end                
            end
            
            if ok then break end
        end
        
        --assert(ok, 'too many bad attemps, something went wrong with sampling from '..path)
        if not ok then print('too many bad attemps, something went wrong with sampling from '..path); return; end
        
        local out = output[s]
        out[1]:copy(out1)
        out[2]:copy(out2)
        
        -- mean/std
        for i=1,2 do -- channels/modalities
            if mean then out[i]:add(-mean[i]) end
            if std then out[i]:div(std[i]) end
        end      
        
        if opt.sampleWeighting=='grad' then
            local stats = itkslave.gradientStats(out1)
            local weakestExtreme1 = torch.cmax(stats:select(2,4), torch.abs(stats:select(2,3))):min()         
            stats = itkslave.gradientStats(out2)
            local weakestExtreme2 = torch.cmax(stats:select(2,4), torch.abs(stats:select(2,3))):min()       
            extrainfo[s][1] = math.min(weakestExtreme1, weakestExtreme2)
        elseif opt.sampleWeighting=='std' then
            extrainfo[s][1] = math.min(out1:std(), out2:std())
            if opt.dataset=='BITE' and opt.normalize01 then extrainfo[s][1] = extrainfo[s][1] + 0.01 end --min value for each sample, ~5x smaller than typical stds there
        end           
        
        -- optionally flip
        if traintime then
            for d=2,opt.patchDim+1 do
                if torch.uniform() > 0.5 then out = image.flip(out,d) end   
            end 
        end        
        
        if false and doPos then
            writePatches(out[1]*std[1]+mean[1],out[2]*std[2]+mean[2],s,doPos,{0,0,0})
            --pe.plotPatches(out)
        end  
    end

    return {d=output, e=extrainfo}
end

--------------------------------------------------------------------------------
-- function to load the image pair
local trainHook = function(self, path)
    if string.starts(opt.criterion,'tri') then
        return processImagePairTriplets(self, path, opt.numTSPatches, true)
    else
        return processImagePair(self, path, opt.numTSPatches, true)
    end
end

--------------------------------------------------------------------------------
-- function to load the image (seed set in a test-sample specific way, repeatable)
local hash = require 'hash'
local hashstate = hash.XXH64()

local testHook = function(self, path)

    --TODO: note that this uses same constraints on sampling

    local rngState = torch.getRNGState()
    torch.manualSeed(opt.seed-1 + hashstate:hash(path))
    local out
    if string.starts(opt.criterion,'tri') then
        out = processImagePairTriplets(self, path, opt.numTestSPatches, false)
    else
        out = processImagePair(self, path, opt.numTestSPatches, false)
    end  
    torch.setRNGState(rngState)
    return out
end

--------------------------------------------------------------------------------
--[[ Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
]]--

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleSize = sampleSize
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(datapath, 'train')},
      loadSize = {2, 256, 256},
      sampleSize = sampleSize,
      forceClasses = {[1] = 'pos', [2] = 'neg'}, --(dataLoader can't handle -1)
      split = opt.trainSplit,
      sameSplitPerm = true --valid data independent
   }
   torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook
trainLoader.sampleHookTest = testHook --(validation)
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

--[[ Section 2: Create a test data loader (testLoader),
   which can iterate over the test set--]]

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleSize = sampleSize
else
   print('Creating test metadata')
   testLoader = dataLoader{
      paths = {paths.concat(datapath, 'test')},
      loadSize = {2, 256, 256},
      sampleSize = sampleSize,
      split = 0,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   trainLoader.meanstd = meanstd
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 500
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples*opt.numTSPatches .. ' randomly sampled training images')
   local meanEstimate = {0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,2 do for k=1,img:size(1) do
         meanEstimate[j] = meanEstimate[j] + img[k][j]:mean()
      end end
   end
   for j=1,2 do
      meanEstimate[j] = meanEstimate[j] / (nSamples*opt.numTSPatches)
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples*opt.numTSPatches .. ' randomly sampled training images')
   local stdEstimate = {0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,2 do for k=1,img:size(1) do
         stdEstimate[j] = stdEstimate[j] + img[k][j]:std()
      end end
   end
   for j=1,2 do
      stdEstimate[j] = stdEstimate[j] / (nSamples*opt.numTSPatches)
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   trainLoader.meanstd = cache
   print('Time to estimate:', tm:time().real)
   
    do -- just check if mean/std look good now
       local testmean = 0
       local teststd = 0
       for i=1,100 do
          local img = trainLoader:sample(1)
          testmean = testmean + img:mean()
          teststd  = teststd + img:std()
       end
       print('Stats of 100 randomly sampled images after normalizing. Mean: ' .. testmean/100 .. ' Std: ' .. teststd/100)
    end      
end
print('Mean: ', mean[1], mean[2], 'Std:', std[1], std[2])
