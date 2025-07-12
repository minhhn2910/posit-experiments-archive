#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

#include <fstream>


//template <typename scalar_t>
__global__ void ber_uniform_cuda_kernel(
     float* input,
     int*  xor_val,
    size_t input_size,
    size_t seed_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    const int seed_index = index % seed_size;
    uint32_t temp = *(uint32_t*)&input[index];
   temp = temp^xor_val[seed_index];
    input[index] = *(float*) & temp;

  }
}


#define MASK 0xffffffff
// #define MASK 0xffffc000
// template <typename scalar_t>
__global__ void ber_asymmetric_cuda_kernel(
     float* input,
     int*  xor_val,
     int* temp_val,
    size_t input_size,
    size_t seed_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    const int seed_index = index % seed_size;
    uint32_t temp = *(uint32_t*)&input[index];
    temp = temp & MASK;
    if (xor_val[seed_index] != 0)
      atomicAdd (&temp_val[0],1);

    uint32_t temp1 = temp&xor_val[seed_index];
    if (temp1 != 0)
      atomicAdd (&temp_val[1],1);

    temp = temp^temp1;//^xor_val[seed_index];
    input[index] = *(float*) & temp;

  }
}

#define MASK_WEIGHT 0xffffffff
// #define MASK_WEIGHT 0xfff00000
//#define MASK_WEIGHT 0x00ff
__global__ void ber_asymmetric_weight_cuda_kernel(
     float* input,
     int*  xor_val,
     int* temp_val,
    size_t input_size,
    size_t seed_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    const int seed_index = index % seed_size;
    uint32_t temp = *(uint32_t*)&input[index];
    temp = temp & MASK_WEIGHT;
    bool flip_flag= false;
    uint32_t one_bits = __popc (temp);

    if (one_bits > 16){
      flip_flag = true;
      temp = ~temp;

    }

    if (xor_val[seed_index] != 0)
      atomicAdd (&temp_val[0],1);

    int temp1 = temp&xor_val[seed_index];
    if (temp1 != 0)
      atomicAdd (&temp_val[1],1);


    temp = temp^temp1;//^xor_val[seed_index];

    if(flip_flag)
        temp = ~temp;

    input[index] = *(float*) & temp;

  }
}
#define WORDLENGTH 32
#define MID 18
__global__ void ber_asymmetric_weight_cuda_kernel_newalgo(
     float* input,
     int*  xor_val,
     int* temp_val,
    size_t input_size,
    size_t seed_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    const int seed_index = index % seed_size;
    int32_t temp = *(int32_t*)&input[index];
    int32_t mask = ~((1<<MID) - 1);
    int32_t temp_masked = temp & mask;
    //encode exponent
       temp_masked = temp_masked^0x3f800000;
       temp= temp^0x3f800000;
    //
    bool flip_flag = false;

    uint32_t one_bits = __popc (temp_masked);
    if (one_bits > (WORDLENGTH-MID)/2){
      temp = temp | ((1<<MID) - 1);
      flip_flag = true;
      temp = ~temp;

    }
    else

    {
      temp = temp & (~((1<<MID) - 1));

    }
    if (xor_val[seed_index] != 0)
      atomicAdd (&temp_val[0],1);

    int temp1 = temp&xor_val[seed_index];
    if (temp1 != 0)
      atomicAdd (&temp_val[1],1);

    temp = temp^temp1;//^xor_val[seed_index];


    //secfphm_decode
    if(flip_flag)
      temp = ~temp;
    //decode exponent
     temp= temp^0x3f800000;
    //
    input[index] = *(float*) & temp;

  }
}
//#define WITHXOR
#define SECPHM
#define WORDLENGTH 32
#define MID_BITCOUNT 16
__global__ void ber_asymmetric_weight_cuda_kernel_bitcount(
     float* input,
     int*  xor_val,
     int* temp_val,
    size_t input_size,
    size_t seed_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < input_size) {
    const int seed_index = index % seed_size;
    int32_t temp = *(int32_t*)&input[index];
    int32_t mask = ~((1<<MID_BITCOUNT) - 1);
    int32_t temp_masked = temp & mask;
    //encode exponent
#ifdef WITHXOR
      temp_masked = temp_masked^0x3f800000;
       temp= temp^0x3f800000;
#endif
    bool flip_flag = false;
#ifdef SECPHM
    uint32_t one_bits = __popc (temp_masked);
    if (one_bits > (WORDLENGTH-MID_BITCOUNT)/2){
      temp = temp | ((1<<MID_BITCOUNT) - 1);
      flip_flag = true;
      temp = ~temp;

    }
    else


    {
      temp = temp & (~((1<<MID_BITCOUNT) - 1));

    }
#endif
    temp_val[index] = __popc(temp);


    //secfphm_decode
    if(flip_flag)
      temp = ~temp;
    //decode exponent
#ifdef WITHXOR
    temp= temp^0x3f800000;
#endif
    //
    input[index] = *(float*) & temp;

  }
}


torch::Tensor ber_uniform_cuda(
    torch::Tensor input,
    torch::Tensor xor_val) {


//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  int64_t xor_size = xor_val.size(0);
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];
  //std::cout<< " state_size "<< input_size<<"\n";
  //auto output = torch::ones_like(input);

  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

//  AT_DISPATCH_FLOATING_TYPES(input.type(), "ber_uniform_cuda", ([&] {
    ber_uniform_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
    xor_val.data<int>(),
        input_size,
      xor_size );
//  }));



  return input;
}


torch::Tensor ber_asymmetric_cuda(
    torch::Tensor input,
    torch::Tensor xor_val) {


//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  int64_t xor_size = xor_val.size(0);
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];
  //std::cout<< " state_size "<< input_size<<"\n";
  //auto temp_val = torch::zeros_like(input);
  torch::Tensor temp_val = torch::zeros(2,torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

  //AT_DISPATCH_FLOATING_TYPES(input.type(), "ber_uniform_cuda", ([&] {
    ber_asymmetric_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
    xor_val.data<int>(),
    temp_val.data<int>(),
        input_size,
      xor_size );
//  }));
  torch::Tensor cpu_tensor = temp_val.to(torch::kCPU);
  auto foo_a = cpu_tensor.accessor<int,1>();
//  std::cout<< " here "<<"\n";
  std::ofstream myfile;
  myfile.open ("/tmp/asymmetric_log.txt",std::ios_base::app);
  myfile << foo_a[0]<<","<< foo_a[1]<<","<<input_size<<"\n";
  myfile.close();

  return input;
}


torch::Tensor ber_asymmetric_weight_cuda(
    torch::Tensor input,
    torch::Tensor xor_val) {


//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  int64_t xor_size = xor_val.size(0);
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];
  //std::cout<< " state_size "<< input_size<<"\n";
  //auto temp_val = torch::zeros_like(input);
  torch::Tensor temp_val = torch::zeros(2,torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

  //AT_DISPATCH_FLOATING_TYPES(input.type(), "ber_uniform_cuda", ([&] {
    //ber_asymmetric_weight_cuda_kernel<<<blocks, threads>>>(
     ber_asymmetric_weight_cuda_kernel_newalgo<<<blocks, threads>>>(
    input.data<float>(),
    xor_val.data<int>(),
    temp_val.data<int>(),
        input_size,
      xor_size );
//  }));
  torch::Tensor cpu_tensor = temp_val.to(torch::kCPU);
  auto foo_a = cpu_tensor.accessor<int,1>();
  //printf ("weight \n");
//  std::cout<< " here "<<"\n";
  std::ofstream myfile;
  myfile.open ("/tmp/asymmetric_log.txt",std::ios_base::app); //overwrite
  myfile << foo_a[0]<<","<< foo_a[1]<<","<<input_size<<"\n";
  myfile.close();

  return input;
}



torch::Tensor ber_bitcount(
    torch::Tensor input,
    torch::Tensor xor_val) {


//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  int64_t xor_size = xor_val.size(0);
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];
  //std::cout<< " state_size "<< input_size<<"\n";
  //auto temp_val = torch::zeros_like(input);
  torch::Tensor temp_val = torch::zeros(input_size,torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

  //AT_DISPATCH_FLOATING_TYPES(input.type(), "ber_uniform_cuda", ([&] {
    //ber_asymmetric_weight_cuda_kernel<<<blocks, threads>>>(
     ber_asymmetric_weight_cuda_kernel_bitcount<<<blocks, threads>>>(
    input.data<float>(),
    xor_val.data<int>(),
    temp_val.data<int>(),
        input_size,
      xor_size );
//  }));
  torch::Tensor cpu_tensor = temp_val.to(torch::kCPU);
  auto foo_a = cpu_tensor.accessor<int,1>();
  double result = 0.0;
  for (int i = 0; i <foo_a.size(0); i ++)
  {
    result += foo_a[i];
    //printf("%d ", foo_a[i]);
  }
  //printf ("tensor size %d \n" ,cpu_tensor.size(0));

  //printf(" result: total bit count \%f \n", result);
  //printf ("result : % bit 1 %f \n ", result/float(foo_a.size(0)*32));
  //printf ("result : % bit 0 %f \n ", 1 - result/float(foo_a.size(0)*32));
  //printf ("weight \n");
//  std::cout<< " here "<<"\n";
  std::ofstream myfile;
  myfile.open ("/tmp/bitcount_log.txt",std::ios_base::app); //overwrite

  myfile << result<<","<< float(foo_a.size(0)*32)<<"\n";
  //myfile.close();

  return input;
}
