"""Process fails with large WSIs due to an integer overflow error somewhere in ITK."""
import os
import sys
import cv2
import numpy as np
import SimpleITK as sitk


def sitk_reg(master_img, target_img):
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

    sitk.WriteImage(fiximg, "fiximg_test.jpg")
    sitk.WriteImage(movimg, "movimg_test.jpg")
    sitk.WriteImage(result, "result.jpg")
    # cv2.imwrite("fiximg1.jpg", fiximg)



if __name__ == "__main__":
    root = "../data/"
    master_img = sitk.ReadImage(root + "OK1_1_3.jpg")
    target_img = sitk.ReadImage(root + "NG1_1_3.jpg")
    sitk_reg(master_img, target_img)