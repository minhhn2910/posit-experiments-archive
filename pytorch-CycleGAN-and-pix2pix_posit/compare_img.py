import sys
import os
from imageio import imread
import numpy as np
import cv2

import mmcv



def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def _to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = mmcv.bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def compare_img (img1, img2):
    assert (img1.size == img2.size)
    divider = img2
    divider [divider == 0 ] = 1.0
    return sum(np.abs((img1-img2)/divider))/float(img1.size)
def get_file_list (path):
    file_list  = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'fake' in file:
                file_list.append(os.path.join(r, file))
    return file_list

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = _to_y_channel(img1)
        img2 = _to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = _to_y_channel(img1)
        img2 = _to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def main(path1, path2, fake=False):

    # read images as 2D arrays (convert to grayscale for simplicity)
    #img1 = imread(file1).astype(float).flatten()
    #img2 = imread(file2).astype(float).flatten()
    # compare
    img_list1 = get_file_list(path1)
    img_list2 = get_file_list(path2)
    print (fake)
    if (fake):
        img_list1 = list(filter(lambda k: 'fake' in k, img_list1)) 
        img_list2 = list(filter(lambda k: 'fake' in k, img_list2)) 
    #print (img_list1[:])
    #print (img_list2[:10])
    print (len(img_list1)," ", len(img_list2))
    if(len(img_list1) > len(img_list2)):
        for item in img_list1:
            if item not in img_list2:
                #print ("bingo")
                img_list1.remove(item)
    print (len(img_list1)," ", len(img_list2))
    avg_rel_err=[]
    psnr = []
    ssim = []
    for i in range(len(img_list2)):
        img1 = imread(img_list1[i]).astype(float)
        img2 = imread(img_list2[i]).astype(float)
 
        avg_rel_err.append(compare_img(img1.flatten(),img2.flatten() ))
        
        psnr.append(calculate_psnr(img1,img2,crop_border=0))
        ssim.append(calculate_ssim(img1,img2,crop_border=0))
        
    #print ("avg relative err",avg_rel_err)
    
    print ("avg relative err ",sum(avg_rel_err)/float(len(avg_rel_err)))
    print ("avg psnr ",sum(psnr)/float(len(psnr)))
    print ("avg ssim ",sum(ssim)/float(len(ssim)))

if __name__ == '__main__':
    arguments = sys.argv[1:]
    #print (arguments)
    fake = False
    if (len(arguments) == 3):
        fake=True #only process image name with *fake*, use for cycle gan test output
    main(arguments[0],arguments[1], fake)
