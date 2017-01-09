require 'torch'
require 'paths'
local ffi = require 'ffi'

ffi.cdef[[
void getSourceBox(float axis1, float axis2, float axis3, float angle, float scale, THFloatTensor* dest);
void transformVolume(float axis1, float axis2, float axis3, float angle, float scale, float center1, float center2, float center3, int useGpu, THFloatTensor* src, THFloatTensor* dst);
void gradientStats(THFloatTensor* src, THFloatTensor* dest);
void setNumITKThreads(unsigned int n);
void readITKImage(const char * path, THFloatTensor* dest);
void createOpenCLContext();
void jitterVolumeBsplines(int nNodes, double maxParamMag, int useGpu, THGenerator *_generator, THFloatTensor* src, THFloatTensor* dst);
]]

local itkslave = {}

itkslave.C = ffi.load(paths.dirname(paths.thisfile())..'/build/libitkslave.so')
local C = itkslave.C

C.setNumITKThreads(1)

function itkslave.createOpenCLContext()
    C.createOpenCLContext()
end

function itkslave.getSourceBox(axis, angle, scale)
	local res = torch.FloatTensor()
	C.getSourceBox(axis[1], axis[2], axis[3], angle, scale, res:cdata())
	return res
end

function itkslave.transformVolume(axis, angle, scale, outsz, image, useGpu)
	local res = torch.FloatTensor()	
	if not image:isContiguous() then image = image:clone() end
	C.transformVolume(axis[1], axis[2], axis[3], angle, scale, outsz[1], outsz[2], outsz[3], useGpu and 1 or 0, image:cdata(), res:cdata())
	return res
end

function itkslave.jitterVolumeBsplines(nNodes, maxParamMag, image, useGpu)
    local res = torch.FloatTensor() 
    if not image:isContiguous() then image = image:clone() end
    local gen = ffi.typeof('THGenerator**')(torch._gen)[0]
    C.jitterVolumeBsplines(nNodes, maxParamMag, useGpu and 1 or 0, gen, image:cdata(), res:cdata())
    return res
end

function itkslave.gradientStats(image)
    local res = torch.FloatTensor() 
    if not image:isContiguous() then image = image:clone() end
    C.gradientStats(image:cdata(), res:cdata())
    return res
end

-- works but is up to 2x slower than compressed tensors produced in preproc/nii2torch pipeline
function itkslave.readITKImage(path)
    local res = torch.FloatTensor() 
    C.readITKImage(path, res:cdata())
    return res:transpose(2,3) --transpose to be consistent with bug(?) in preproc/nii2torch pipeline
end

return itkslave
