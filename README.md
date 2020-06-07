# pytorch-posit
The old version of posit integration for pytorch.
Design: 
* Pytorch DNNs will have a series of operations  (convolution etc.) 
* After each operation, we intercept the output, do a posit convesion to low precision and convert back to float. 
* Posit wrapper for pytorch op: posit_cuda.cpp
* Main CUDA kernel for conversion:  posit_cuda_kernels.cu (modify posit exponent and wordlenth inside this file)
* Main operation: modified pytorch inference of cycle gan :  
  * /pytorch-CycleGAN-and-pix2pix_posit/models/networks.py line 428 for activation
  * /pytorch-CycleGAN-and-pix2pix_posit/models/base_model.py line 239 for weights 
