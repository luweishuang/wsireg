"""Process fails with large WSIs due to an integer overflow error somewhere in ITK."""
import os
import sys
import cv2
import numpy as np
import SimpleITK as sitk


root = "../data/"
master_img = sitk.ReadImage(root + "OK1_1_32.jpg")
target_img = sitk.ReadImage(root + "NG1_1_32.jpg")

# Convert our data to some operable format.
fiximg = sitk.VectorIndexSelectionCast(master_img, 0, sitk.sitkFloat32)
movimg = sitk.VectorIndexSelectionCast(target_img, 0, sitk.sitkFloat32)

######### Object Oriented api ############
# Initialize transform calculator
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fiximg)
elastixImageFilter.SetMovingImage(movimg)

# Construct transformation queue
parameterMapVector = sitk.VectorOfParameterMap()
params = sitk.GetDefaultParameterMap("translation")
params['NumberOfResolutions'] = ['8',]
params['NumberOfSpatialSamples'] = ['16384',]
params['NumberOfSamplesForExactGradient'] = ['16384',]
params['MaximumNumberOfIterations'] = ['2000',]
# print(dict(params))
parameterMapVector.append(params)
# parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
# parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.PrintParameterMap()

# Run
elastixImageFilter.Execute()
transform = elastixImageFilter.GetTransformParameterMap()
result = elastixImageFilter.GetResultImage()

# Save results
fiximg = sitk.Cast(fiximg, sitk.sitkUInt8)
movimg = sitk.Cast(movimg, sitk.sitkUInt8)
result = sitk.Cast(result, sitk.sitkUInt8)

outfolder =  "./"

sitk.WriteImage(fiximg, os.path.join(outfolder, "fiximg_test.jpg"))
sitk.WriteImage(movimg, os.path.join(outfolder, "movimg_test.jpg"))
sitk.WriteImage(result, os.path.join(outfolder, "result.jpg"))
# cv2.imwrite(os.path.join(outfolder, "overlay.png"), overlay)

# Document experiment
with open(os.path.join(outfolder, "doc.txt"), "a") as doc:
    doctext = "\n%s: increased MaximumNumberOfIterations to hopefully increase convergence." % outfolder
    # doctext = "\n%s: sitk, default translation params, with the following increased: NumberOfResolutions, NumberOfSpatialSamples, NumberOfSamplesForExactGradient." % outfolder
    doc.write(doctext)
