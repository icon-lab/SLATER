import skimage
from skimage import measure
from PIL import Image
import glob
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt


def compute_psnr(generated_image, original_image):
    data1 = np.asarray( generated_image, dtype="float32" )    
    max1 = np.amax(data1)
    data1 /= max1

    data2 = np.asarray( original_image, dtype="float32" )
    max2 = np.amax(data2)
    data2 /= max2

    psnr = measure.compare_psnr(data1, data2)
    print('psnr: ' + str(psnr))
    return psnr

def compute_ssim(generated_image,original_image):
    data1 = np.asarray( generated_image, dtype="float32" )    
    max1 = np.amax(data1)
    data1 /= max1

    data2 = np.asarray( original_image, dtype="float32" )    
    max2 = np.amax(data2)
    data2 /= max2
    ssim = measure.compare_ssim(data1,data2)
    print('ssim: ' + str(ssim))
    return ssim                    