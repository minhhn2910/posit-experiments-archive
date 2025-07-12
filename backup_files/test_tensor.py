from torch.utils.cpp_extension import load
custom_module = load(name='posit_module', sources=['posit_cuda.cpp', 'posit_cuda_kernels.cu'])
import torch
from torch.utils.cpp_extension import load_inline
import numpy as np
np.set_printoptions(suppress=True)
from ctypes import *
#def ber_uniform_np (x_input,seeds_arr): #x_input is a np array XOR 1 = flip.
  #random.shuffle(seeds_arr)
#  seed_index = 0
#  for i in range (len(x_input)):
#    if (seed_index == len(seeds_arr)):
#        seed_index =0
#    bits = cast(pointer(c_float(x_input[i])), POINTER(c_int32)).contents.value
#    bits = bits ^ seeds_arr[seed_index]
#    x_input[i] = cast(pointer(c_int32(bits)), POINTER(c_float)).contents.value
#    seed_index = seed_index+1
np.random.seed(0)
a_np = np.random.rand(100)* 20 -10.0
#a_np = np.array([1.0]*32)
a = torch.tensor(a_np, dtype=torch.float)
print (a)
b = custom_module.posit_wrapper(a.cuda())
print ('------')
print (b)


#custom_module_improved = load(name='posit_module', sources=['posit_cuda.cpp', 'backup_posit_improved.cu'])
import qtorch
from qtorch.quant import posit_quantize
c = posit_quantize(a.cuda(), nsize=8, es=2, rounding="nearest")
#c = custom_module_improved.posit_wrapper(a.cuda())
print ('------')
print (c)
print ('------')
print (c - b)
exit()
