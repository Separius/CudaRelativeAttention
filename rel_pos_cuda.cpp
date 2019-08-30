#include <torch/extension.h>
#include <vector>
#include <stdio.h>

// CUDA forward declarations

torch::Tensor relative_positioning_forward_2d_cuda(
    torch::Tensor logits, torch::Tensor r_h, torch::Tensor r_w, torch::Tensor mask,
    const int h_q, const int w_q, const int h_k, const int w_k, const bool use_mask);

std::vector<torch::Tensor> relative_positioning_backward_2d_cuda(
    torch::Tensor grad_out, const int h_q, const int w_q, const int h_k, const int w_k);

torch::Tensor relative_positioning_forward_3d_cuda(
    torch::Tensor logits, torch::Tensor r_t, torch::Tensor r_h, torch::Tensor r_w, torch::Tensor mask,
    const int t_q, const int h_q, const int w_q, const int t_k, const int h_k, const int w_k, const bool use_mask);

std::vector<torch::Tensor> relative_positioning_backward_3d_cuda(
    torch::Tensor grad_out, const int t_q, const int h_q, const int w_q, const int t_k, const int h_k, const int w_k);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor relative_positioning_forward_2d(
    torch::Tensor logits, torch::Tensor r_h, torch::Tensor r_w, torch::Tensor mask,
    const int h_q, const int w_q, const int h_k, const int w_k, const bool use_mask) {
  CHECK_INPUT(logits); // logits is N(==B*Nh), H_q*W_q, H_k*W_k
  CHECK_INPUT(r_h); // r_h is N(==B*Nh), H_q*W_q, H_k+H_q-1
  CHECK_INPUT(r_w); // r_w is N(==B*Nh), H_q*W_q, W_k+W_q-1
  CHECK_INPUT(mask); // mask is a bool tensor of N(==B*Nh), H_q*W_q, H_k*W_k or H_q*W_q, H_k*W_k OR 1, 1, 1
  return relative_positioning_forward_2d_cuda(logits, r_h, r_w, mask, h_q, w_q, h_k, w_k, use_mask);
}

std::vector<torch::Tensor> relative_positioning_backward_2d(
    torch::Tensor grad_out, const int h_q, const int w_q, const int h_k, const int w_k) {
    CHECK_INPUT(grad_out);
    auto grads = relative_positioning_backward_2d_cuda(grad_out, h_q, w_q, h_k, w_k);
    grads.insert(grads.begin(), grad_out.clone());
    return grads;
}

torch::Tensor relative_positioning_forward_3d(
    torch::Tensor logits, torch::Tensor r_t, torch::Tensor r_h, torch::Tensor r_w, torch::Tensor mask,
    const int t_q, const int h_q, const int w_q, const int t_k, const int h_k, const int w_k, const bool use_mask) {
  CHECK_INPUT(logits); // logits is B*Nh, Tq*Hq*Wq, Tq*Hk*Wk
  CHECK_INPUT(r_t); // r_t is B*Nh, Tq*Hq*Wq, Tk+Tq-1
  CHECK_INPUT(r_h); // r_h is B*Nh, Tq*Hq*Wq, Hk+Hq-1
  CHECK_INPUT(r_w); // r_w is B*Nh, Tq*Hq*Wq, Wk+Wq-1
  CHECK_INPUT(mask); // mask is a bool tensor of {B*Nh, Tq*Hq*Wq, Tk*Hk*Wk} or {Tq*Hq*Wq, Tk*Hk*Wk} or {1, 1, 1}
  return relative_positioning_forward_3d_cuda(logits, r_t, r_h, r_w, mask, t_q, h_q, w_q, t_k, h_k, w_k, use_mask);
}

std::vector<torch::Tensor> relative_positioning_backward_3d(
    torch::Tensor grad_out, const int t_q, const int h_q, const int w_q, const int t_k, const int h_k, const int w_k) {
    CHECK_INPUT(grad_out);
    auto grads = relative_positioning_backward_3d_cuda(grad_out, t_q, h_q, w_q, t_k, h_k, w_k);
    grads.insert(grads.begin(), grad_out.clone());
    return grads;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_2d", &relative_positioning_forward_2d, "adds 2d relative positioning logits to the main logits (cuda, forward)");
  m.def("backward_2d", &relative_positioning_backward_2d, "adds 2d relative positioning logits to the main logits (cuda, backward)");
  m.def("forward_3d", &relative_positioning_forward_3d, "adds 3d relative positioning logits to the main logits (cuda, forward)");
  m.def("backward_3d", &relative_positioning_backward_3d, "adds 3d relative positioning logits to the main logits (cuda, backward)");
}