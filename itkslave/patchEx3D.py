#### original code, now it's reimplemted in C. But good for single-threaded experiments. Python can't work multithreaded.

import SimpleITK as sitk
import numpy as np

box = ((1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1))

# compute inverse transformation of a axis-aligned box centered at (0,0,0) with vertices at points like (1,1,1) 
def get_source_box(axis, angle, scale):	
	transform = sitk.Similarity3DTransform()
	transform.SetScale(1/scale)
	transform.SetRotation(axis, -angle)
	trbox = np.array([ transform.TransformPoint(p) for p in box], np.float32)
	return trbox	

	
def transform_volume(axis, angle, scale, center, img):
	#Transform wants to be given the inverse transform
	transform = sitk.Similarity3DTransform()
	transform.SetScale(1/scale)
	transform.SetRotation(axis, -angle)
	transform.SetCenter(center)
	
	#Do transform. Input size = result size
	imgitk = sitk.GetImageFromArray(img)
	result = sitk.Resample(imgitk, transform, sitk.sitkLinear, -1, sitk.sitkFloat32) #sitkLiner doesn't overshoot, unlike splines?
	#np.copyto(img, sitk.GetArrayFromImage(result))
	return sitk.GetArrayFromImage(result)


if __name__ == "__main__":
	#get_source_box(1,1,1, np.pi/4, 1.5)
	a = np.zeros((5,5,5), dtype=np.float32)  
	print(a)
	transform_volume((1, 1, 1), np.pi/4, 0.7, (2, 2, 2), a)
	print(a)
