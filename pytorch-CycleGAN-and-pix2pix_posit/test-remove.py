"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
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
def get_file_list (path_):
    file_list  = []

    # r=root, d=directories, f = files
    
    for r, d, f in os.walk(path_):
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

def compare_img_main(path1, path2, fake=False):

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
                print (item)
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
    return sum(ssim)/float(len(ssim))

def test_table (opt, table):
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options

    model.setup(opt,table=table)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

    return compare_img_main('results_ref/horse2zebra/test_latest/images/', web_dir +'/images/', fake=True)
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    print (opt.results_dir)
    
    original_table =  np.array([1.0/65536, 1.0/32768, 1.0/16384, 1.0/8192, 1.0/4096, 1.0/2048, 1.0/1024, 1.0/512, 1.0/256, 1.0/128,
               3.0/256, 1.0/64,  5.0/256 , 3.0/128,  7.0/256, 1.0/32, 9.0/256, 5.0/128, 3.0/64, 7.0/128,
               1.0/16,  9.0/128, 5.0/64, 3.0/32,    7.0/64,    1.0/8, 9.0/64, 3.0/16, 1.0/4, 3.0/8, 1.0/2, 1.0])
    remove_16 = [11, 9, 7, 5, 4, 2, 0, 3, 1, 6, 8, 14, 12, 22, 20, 10, 15, 16, 21, 13, 17, 19, 25, 18, 24, 23, 26, 29, 27, 30, 31, 28]
    remove_8 = [1, 2, 9, 4, 3, 5, 8, 7, 10, 6, 15, 13, 0, 14, 12, 11]
    remove_4 = [5, 7, 6, 4, 3, 2, 1, 0]
    original_table = np.delete(original_table,remove_16[:16])
    #print (test_table(opt, original_table))
    
    original_table = np.delete(original_table,remove_8[:8])
    #print (test_table(opt, original_table))
    original_table = np.delete(original_table,remove_4[:4])
    print (test_table(opt, original_table))
    exit(0)
    res_remove = []
    for i in range(len(original_table)):
        temp_table = np.copy(original_table)
        temp_table = np.delete(temp_table,i)
        res_remove.append(test_table(opt, temp_table))
        print (res_remove)
    
        

