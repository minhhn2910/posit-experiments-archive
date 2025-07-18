from torch.utils.cpp_extension import load
custom_module = load(name='ber_uniform', sources=['flip_bit_cuda.cpp', 'flip_bit_cuda_kernel.cu'])
import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

from ctypes import *
def ber_uniform_np (x_input,seeds_arr): #x_input is a np array XOR 1 = flip.
  #random.shuffle(seeds_arr)
  seed_index = 0
  for i in range (len(x_input)):
    if (seed_index == len(seeds_arr)):
        seed_index =0
    bits = cast(pointer(c_float(x_input[i])), POINTER(c_int32)).contents.value
    bits = bits ^ seeds_arr[seed_index]
    x_input[i] = cast(pointer(c_int32(bits)), POINTER(c_float)).contents.value
    seed_index = seed_index+1
a_np = np.random.rand(100)* 2 -1.0
a = torch.tensor(a_np, dtype=torch.float)
b_np = np.array([2**22-1]*10)
#b_np = np.random.randint(32, 32768,size=100)
b = torch.tensor(b_np, dtype=torch.int)
print(a)
print(b)
print(a.size())
a = custom_module.ber_uniform(a.cuda(), b.cuda())
print ('------')
print (a)
print ('------')
ber_uniform_np (a_np,b_np)
print (a_np)

exit()

#getting binary representatation of num
INVBER=100
import random
random.seed()
total_elements = 384000
total_bits = total_elements * 32
ones = int(total_bits/INVBER)
zeroes = total_bits - ones
ones_list = ['1']*ones
zeroes_lists = ['0']*zeroes
ones_list.extend(zeroes_lists)
full_list = np.array(ones_list)
assert(len (full_list) == total_bits)
random.shuffle(full_list)
full_list = full_list.reshape(-1,32)
int_list = np.apply_along_axis(lambda x: int("".join(x),2), 1 , full_list)
seed_tensor = torch.tensor(int_list, dtype=torch.int)

from ttictoc import TicToc
t = TicToc() ## TicToc("name")
t.tic();

#print (full_list)
#reshape to int 32 bits

#print (int_list)
random.shuffle(int_list)
#ber_uniform_np(input_arr,int_list)
a = custom_module.ber_uniform(input_tensor.cuda(), seed_tensor.cuda())
t.toc();
print(str(t.elapsed) + " seconds ")
