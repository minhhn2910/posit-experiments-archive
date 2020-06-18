
#include <vector>
#include <torch/extension.h>
#include <iostream>
// CUDA forward declarations

torch::Tensor posit_cuda(
    torch::Tensor input);
torch::Tensor posit_cuda_weight(
    torch::Tensor input);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor posit_interface(
    torch::Tensor input) {
  CHECK_INPUT(input);
//debug, for colleting histogram stats, disable posit conversion
 return posit_cuda(input);
  // return input;
}
torch::Tensor posit_interface_weight(
    torch::Tensor input) {
  CHECK_INPUT(input);
//debug, for colleting histogram stats, disable posit conversion
 return posit_cuda_weight(input);
  // return input;
}

//#define BER_UNIFORM
torch::Tensor posit_wrapper(
    torch::Tensor input) {

    return posit_interface(input);

}
torch::Tensor posit_wrapper_weight(
    torch::Tensor input) {
    return posit_interface_weight(input); // can define a separate format for weights

}

torch::Tensor posit_wrapper_compatible(
    torch::Tensor input,
    torch::Tensor dummy) {
    return posit_interface(input); // can define a separate format for weights

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("posit_wrapper", &posit_wrapper, "posit wrapper (CUDA)");
  m.def("ber_wrapper", &posit_wrapper_compatible, "posit wrapper (CUDA)");   //compatible purpose
  m.def("posit_wrapper_weight", &posit_wrapper_weight, "posit wrapper weights (CUDA)");
}
