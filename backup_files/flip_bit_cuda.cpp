
#include <vector>
#include <torch/extension.h>
#include <iostream>
// CUDA forward declarations

torch::Tensor ber_uniform_cuda(
    torch::Tensor input,
    torch::Tensor xor_val);

torch::Tensor ber_asymmetric_cuda(
        torch::Tensor input,
        torch::Tensor xor_val);

torch::Tensor ber_asymmetric_weight_cuda(
        torch::Tensor input,
        torch::Tensor xor_val);

torch::Tensor ber_bitcount(
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

torch::Tensor ber_asymmetric(
    torch::Tensor input,
    torch::Tensor xor_val) {
  CHECK_INPUT(input);
  CHECK_INPUT(xor_val);
  return ber_asymmetric_cuda(input, xor_val);
}

torch::Tensor ber_asymmetric_weight(
    torch::Tensor input,
    torch::Tensor xor_val) {
  CHECK_INPUT(input);
  CHECK_INPUT(xor_val);
  return ber_asymmetric_weight_cuda(input, xor_val);
}

//#define BER_UNIFORM
torch::Tensor ber_wrapper(
    torch::Tensor input,
    torch::Tensor xor_val) {
  #ifdef BER_UNIFORM
    return ber_uniform(input, xor_val);
  #else
    return ber_asymmetric(input, xor_val);
  #endif
}
torch::Tensor ber_wrapper_weight(
    torch::Tensor input,
    torch::Tensor xor_val) {
    return ber_bitcount(input, xor_val);
/*
  #ifdef BER_UNIFORM
    return ber_uniform(input, xor_val);
  #else
    return ber_asymmetric_weight(input, xor_val);
  #endif
*/
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ber_wrapper", &ber_wrapper, "BER wrapper (CUDA)");
  m.def("ber_wrapper_weight", &ber_wrapper_weight, "BER wrapper weights (CUDA)");
}
