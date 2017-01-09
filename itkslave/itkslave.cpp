#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <TH/TH.h>

extern "C" {
void getSourceBox(float axis1, float axis2, float axis3, float angle, float scale, THFloatTensor* dest);
void transformVolume(float axis1, float axis2, float axis3, float angle, float scale, float outsz1, float outsz2, float outsz3, int useGpu, THFloatTensor* src, THFloatTensor* dst);
void gradientStats(THFloatTensor* src, THFloatTensor* dest);
void setNumITKThreads(unsigned int n);
void readITKImage(const char * path, THFloatTensor* dest);
void createOpenCLContext();
void jitterVolumeBsplines(int nNodes, double maxParamMag, int useGpu, THGenerator *_generator, THFloatTensor* src, THFloatTensor* dst);
}

// General note: all coordinates/sizes/.. need to be flipped between Torch and ITK, they index their dimensions in the opposite way. But memory is indexed the same way:).


#include <itkSimilarity3DTransform.h>
#include <itkBSplineTransform.h>
#include <itkResampleImageFilter.h>

typedef float T;
typedef itk::Transform<T> TransformT;
typedef itk::Similarity3DTransform<T> Similarity3DTransformT;
typedef itk::Image<T,3> ImageT;


ImageT::Pointer resampleCPU(ImageT::Pointer srcimg, TransformT::Pointer transform, ImageT::SizeType sizenew, itk::Point<T,3> orinew);
ImageT::Pointer resampleGPU(ImageT::Pointer srcimg, TransformT::Pointer transform, ImageT::SizeType sizenew, itk::Point<T,3> orinew);


// compute inverse transformation of a axis-aligned box centered at (0,0,0) with vertices at points like (1,1,1) 
void getSourceBox(float axis1, float axis2, float axis3, float angle, float scale, THFloatTensor* dest)
{
	THFloatTensor_resize2d(dest, 8, 3);
	float* data = THFloatTensor_data(dest);
	
	itk::Vector<T,3> axis;
	axis[0] = axis3;
    axis[1] = axis2;
    axis[2] = axis1;
	
	Similarity3DTransformT::Pointer transform = Similarity3DTransformT::New();
	transform->SetScale(1.f/scale);
	transform->SetRotation (axis, -angle);
	
	itk::Point<T,3> p, pt;
	int pos = 0;
	for (int i=-1; i<=1; i+=2)
		for (int j=-1; j<=1; j+=2)	
			for (int k=-1; k<=1; k+=2)		
			{
			    p[0] = i;
				p[1] = j;
				p[2] = k;
				pt = transform->TransformPoint(p);
				data[pos++] = pt[2];
				data[pos++] = pt[1];
				data[pos++] = pt[0];
			}
}

// set image to the size of tensor and deep-copy the content
bool tensorToImage(THFloatTensor* tensor, ImageT::Pointer& image)
{
 	if (tensor->nDimension != 3) {
 		std::cerr << "Input tensor doesn't have 3 dims: " << tensor->nDimension << std::endl;
 		return false;
	}
	
	image = ImageT::New();
  	ImageT::IndexType start;
  	start.Fill(0);
 	ImageT::SizeType size;
 	size[0] = tensor->size[2];
 	size[1] = tensor->size[1];
 	size[2] = tensor->size[0];	 	
 	
  	image->SetRegions(ImageT::RegionType(start, size));
  	image->Allocate();
  	memcpy(image->GetBufferPointer(), THFloatTensor_data(tensor), THFloatTensor_nElement(tensor)*sizeof(float));
  	//TODO: or just sharing? get inspired by http://docs.mitk.org/2014.10/mitkITKImageImport_8txx_source.html
  	return true;
}

// set tensor to the size of image and deep-copy the content
bool imageToTensor(ImageT::Pointer image, THFloatTensor* tensor)
{
 	if (image->ImageDimension != 3) {
 		std::cerr << "Input image doesn't have 3 dims: " << image->ImageDimension << std::endl;
 		return false;
	}
	
	ImageT::SizeType size = image->GetLargestPossibleRegion().GetSize();
	THFloatTensor_resize3d(tensor, size[2], size[1], size[0]);

  	memcpy(THFloatTensor_data(tensor), image->GetBufferPointer(), THFloatTensor_nElement(tensor)*sizeof(float));
  	//TODO: or just sharing? get inspired by http://docs.mitk.org/2014.10/mitkITKImageImport_8txx_source.html
  	return true;
}

// resamples img by the given transformation
void transformVolume(float axis1, float axis2, float axis3, float angle, float scale, float outsz1, float outsz2, float outsz3, int useGpu, THFloatTensor* src, THFloatTensor* dst)
{
	ImageT::Pointer srcimg;
	if (!tensorToImage(src, srcimg))
		return;

	ImageT::SizeType size = srcimg->GetLargestPossibleRegion().GetSize();
	itk::Point<T,3> ori;
	for (int i=0; i<3; i++) ori[i] = -T(size[i])/2; //how to spend hours looking for a bug? negating f**ing uint without cast.
	srcimg->SetOrigin(ori);

    ImageT::SizeType sizenew;
    if (outsz1>0)
	{
    	sizenew[0] = outsz3;
    	sizenew[1] = outsz2;
    	sizenew[2] = outsz1;
	}
    else
    {
    	for (int i=0; i<3; i++) sizenew[i] = std::ceil(size[i] * scale);
    }

	itk::Point<T,3> orinew;
	for (int i=0; i<3; i++) orinew[i] = -T(sizenew[i])/2;

	itk::Vector<T,3> axis;
	axis[0] = axis3;
    axis[1] = axis2;
    axis[2] = axis1;

	//Transform wants to be given the inverse transform (and it's good to have center = origin, otherwise extra translations appear under scaling)
	Similarity3DTransformT::Pointer transform = Similarity3DTransformT::New();
	transform->SetScale(1.f/scale);
	transform->SetRotation (axis, -angle);

	ImageT::Pointer result;
	if (useGpu != 0)
		result = resampleGPU(srcimg, transform.GetPointer(), sizenew, orinew);
	else
		result = resampleCPU(srcimg, transform.GetPointer(), sizenew, orinew);

	if (!imageToTensor(result, dst))
		return;
}


void jitterVolumeBsplines(int nNodes, double maxParamMag, int useGpu, THGenerator *_generator, THFloatTensor* src, THFloatTensor* dst)
{
	ImageT::Pointer srcimg;
	if (!tensorToImage(src, srcimg))
		return;

	const int SplineOrder = 2;
	typedef itk::BSplineTransform<T,3,SplineOrder> TransformType;
	TransformType::Pointer transform = TransformType::New();

	ImageT::SpacingType   spacing = srcimg->GetSpacing();
	ImageT::PointType     origin = srcimg->GetOrigin();
	ImageT::DirectionType direction = srcimg->GetDirection();
	ImageT::RegionType 	  region = srcimg->GetBufferedRegion();
	ImageT::SizeType  	  size = region.GetSize();
	TransformType::PhysicalDimensionsType   physicalDimensions;
	TransformType::MeshSizeType             meshSize;

	for (unsigned int i = 0; i< 3; i++)
		physicalDimensions[i] = spacing[i] * static_cast<double>(size[i] - 1);
	meshSize.Fill(nNodes - SplineOrder);
	transform->SetTransformDomainOrigin(origin);
	transform->SetTransformDomainPhysicalDimensions(physicalDimensions);
	transform->SetTransformDomainMeshSize(meshSize);
	transform->SetTransformDomainDirection(direction);


	const unsigned int numberOfParameters = transform->GetNumberOfParameters();
	TransformType::ParametersType parameters(numberOfParameters);
	for (unsigned int i = 0; i < numberOfParameters; ++i)
		parameters[i] =  THRandom_uniform(_generator, -maxParamMag, maxParamMag);
	transform->SetParameters(parameters);

	ImageT::Pointer result;
	if (useGpu != 0)
		result = resampleGPU(srcimg, transform.GetPointer(), size, origin);
	else
		result = resampleCPU(srcimg, transform.GetPointer(), size, origin);

	if (!imageToTensor(result, dst))
		return;
}



#include <itkStatisticsImageFilter.h>
#include <itkGradientRecursiveGaussianImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

//computes gradient statistics (mean,std,min,max) of each image dimension
void gradientStats(THFloatTensor* src, THFloatTensor* dest)
{
	ImageT::Pointer srcimg;
	if (!tensorToImage(src, srcimg))
		return;

	typedef itk::GradientRecursiveGaussianImageFilter<ImageT> GradientImageFilterType;
	typedef GradientImageFilterType::OutputImageType OutputImageType;
	GradientImageFilterType::Pointer gradientFilter = GradientImageFilterType::New();
	gradientFilter->SetInput(srcimg);

	const typename ImageT::SpacingType& spacing = srcimg->GetSpacing();
	double maximumSpacing=0.0;
	for(unsigned int i=0; i<3; i++)
		if( spacing[i] > maximumSpacing )
			maximumSpacing = spacing[i];

	gradientFilter->SetSigma( maximumSpacing );
	gradientFilter->SetNormalizeAcrossScale( true );
	gradientFilter->SetUseImageDirection( false );
	gradientFilter->Update();

	typedef itk::Image<T> SliceT;
	typedef itk::VectorIndexSelectionCastImageFilter<OutputImageType,SliceT> SelectionFilterType;
	SelectionFilterType::Pointer componentExtractor = SelectionFilterType::New();
	componentExtractor->SetInput(gradientFilter->GetOutput());

	typedef itk::StatisticsImageFilter<SliceT> StatisticsImageFilterType;
	StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
	statisticsImageFilter->SetInput(componentExtractor->GetOutput());

	THFloatTensor_resize2d(dest, 3, 4);
	float* data = THFloatTensor_data(dest);

	for(unsigned int i=0; i<3; i++)
	{
		componentExtractor->SetIndex(i);
		componentExtractor->Update();
		statisticsImageFilter->Update();

		data[4*i] = statisticsImageFilter->GetMean();
		data[4*i+1] = statisticsImageFilter->GetSigma();
		data[4*i+2] = statisticsImageFilter->GetMinimum();
		data[4*i+3] = statisticsImageFilter->GetMaximum();
	}
}

void setNumITKThreads(unsigned int n)
{
	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(n);
}

//#include "itkImageFileReader.h" --somehow keeps qlua from terminating when it's finished, wow

void readITKImage(const char * path, THFloatTensor* dest)
{
	/*typedef itk::ImageFileReader<ImageT> ImageReaderType;
	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	imageReader->SetFileName(path);
	ImageT::Pointer image = imageReader->GetOutput();
	image->Update();
	if (!imageToTensor(image, dest))
		return;*/
}









typedef itk::ResampleImageFilter<ImageT,ImageT,T,T> ResampleImageFilterT;
static __thread ResampleImageFilterT* m_resampler = NULL; //C++<11 doesn't like tls objects

ImageT::Pointer resampleCPU(ImageT::Pointer srcimg, TransformT::Pointer transform, ImageT::SizeType sizenew, itk::Point<T,3> orinew)
{
	if (m_resampler == NULL)
	{
		typedef itk::LinearInterpolateImageFunction<ImageT, T> InterpolatorT;
		ResampleImageFilterT::Pointer imageResampleFilter = ResampleImageFilterT::New();
		imageResampleFilter->SetInterpolator(InterpolatorT::New());
		imageResampleFilter->Register(); m_resampler = imageResampleFilter;
	}

	//Do transform (complexity in size of output).
	ResampleImageFilterT::Pointer imageResampleFilter = ResampleImageFilterT::New();
	m_resampler->SetInput(srcimg);
	m_resampler->SetSize(sizenew);
	m_resampler->SetOutputSpacing(srcimg->GetSpacing());
	m_resampler->SetOutputOrigin(orinew);
	m_resampler->SetOutputDirection(srcimg->GetDirection());
	m_resampler->SetTransform(transform);
	m_resampler->SetDefaultPixelValue(-1);
	m_resampler->UpdateLargestPossibleRegion();
	m_resampler->Update();

	return m_resampler->GetOutput();
}


#include "itkGPUImage.h"
#include "itkGPUResampleImageFilter.h"
#include "itkGPUBSplineTransform.h"
#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUTransformCopier.h"
#include "itkGPUInterpolatorCopier.h"

struct OCLImageDims
{
 itkStaticConstMacro( Support1D, bool, true );
 itkStaticConstMacro( Support2D, bool, true );
 itkStaticConstMacro( Support3D, bool, true );
};
typedef itk::GPUImage<T,3> GPUImageType;
typedef typelist::MakeTypeList< T >::Type OCLImageTypes;
typedef itk::GPUTransformCopier< OCLImageTypes, OCLImageDims, TransformT, T > GPUTransformCopierType;
typedef itk::GPUResampleImageFilter<GPUImageType,GPUImageType,T> GPUResamplerType;

static __thread GPUImageType* m_gpuMovingImage = NULL; //C++<11 doesn't like tls objects
static __thread GPUTransformCopierType* m_gpuTransformCopier = NULL;
static __thread GPUResamplerType* m_gpuResampler = NULL;


//Must be called from main thread so there is no racing condition (it's a singleton). It's not mutex-protected but it seems to work in multithreading.
//...but use with care:)
void createOpenCLContext()
{
	itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
	if (!context->IsCreated ()) context->Create(itk::OpenCLContext::SingleMaximumFlopsDevice);
}

ImageT::Pointer resampleGPU(ImageT::Pointer srcimg, TransformT::Pointer transform, ImageT::SizeType sizenew, itk::Point<T,3> orinew)
{
	try
	{
		//slaughtered MyResampler below:)
		if (m_gpuResampler == NULL)
		{
			typename GPUImageType::Pointer gpuMovingImageS = GPUImageType::New();
			typename GPUTransformCopierType::Pointer gpuTransformCopierS = GPUTransformCopierType::New();
			typename GPUResamplerType::Pointer gpuResamplerS = GPUResamplerType::New();

			gpuMovingImageS->Register(); m_gpuMovingImage = gpuMovingImageS;
			gpuTransformCopierS->Register(); m_gpuTransformCopier = gpuTransformCopierS;
			gpuResamplerS->Register(); m_gpuResampler = gpuResamplerS;

			typedef itk::GPULinearInterpolateImageFunction<GPUImageType> GPULinearInterpolatorT;
			typename GPULinearInterpolatorT::Pointer gpuInterpolator = GPULinearInterpolatorT::New();
			m_gpuResampler->SetInterpolator(gpuInterpolator);
			m_gpuResampler->SetInput(m_gpuMovingImage);
		}

		m_gpuMovingImage->GraftITKImage(srcimg);
		m_gpuMovingImage->AllocateGPU();
		m_gpuMovingImage->GetGPUDataManager()->SetCPUBufferLock( true );
		m_gpuMovingImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
		m_gpuMovingImage->UpdateBuffers();

		m_gpuTransformCopier->SetInputTransform(transform);
		m_gpuTransformCopier->Update();
		m_gpuResampler->SetTransform(m_gpuTransformCopier->GetModifiableOutput());

		m_gpuResampler->SetGPUEnabled(true);
		m_gpuResampler->SetSize(sizenew);
		m_gpuResampler->SetDefaultPixelValue(-1);
		m_gpuResampler->SetOutputSpacing(srcimg->GetSpacing());
		m_gpuResampler->SetOutputOrigin(orinew);
		m_gpuResampler->SetOutputDirection(srcimg->GetDirection());
		m_gpuResampler->UpdateLargestPossibleRegion();
		m_gpuResampler->Update();
		m_gpuResampler->GetOutput()->UpdateBuffers();

		//explicit memory release necessary, some elastix bug imho
		m_gpuResampler->GetOutput()->GetGPUDataManager()->Initialize();
		m_gpuMovingImage->GetGPUDataManager()->Initialize();

		//TODO: there is still RAM leak... damn! Doesn't depend on multithread and isn't present with cpu resampling.
	}
	catch (const std::exception &exc) {
        std::cerr << exc.what();
	}

	return m_gpuResampler->GetOutput();
}

