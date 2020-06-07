#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

template <typename scalar_t>
__global__ void ber_uniform_cuda_kernel(
     scalar_t* input,
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



torch::Tensor ber_uniform_cuda(
    torch::Tensor input,
    torch::Tensor xor_val) {


//  const auto state_size = input.size(0);
  int64_t input_size = 1;
  int64_t xor_size = xor_val.size(0);
  for (int i = 0 ; i< input.sizes().size();i++)
    input_size = input_size * input.sizes()[i];
  //std::cout<< " state_size "<< input_size<<"\n";
  auto output = torch::ones_like(input);

  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ber_uniform_cuda", ([&] {
    ber_uniform_cuda_kernel<scalar_t><<<blocks, threads>>>(
    input.data<scalar_t>(),
    xor_val.data<int>(),
        input_size,
      xor_size );
  }));

  return input;
}
