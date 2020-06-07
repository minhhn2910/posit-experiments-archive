
#include <vector>
#include <torch/extension.h>
#include <iostream>
// CUDA forward declarations

torch::Tensor ber_uniform_cuda(
    torch::Tensor input,
    torch::Tensor xor_val);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ber_uniform(
    torch::Tensor input,
    torch::Tensor xor_val) {
  CHECK_INPUT(input);
  CHECK_INPUT(xor_val);
  return ber_uniform_cuda(input, xor_val);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ber_uniform", &ber_uniform, "BER uniform (CUDA)");
}
