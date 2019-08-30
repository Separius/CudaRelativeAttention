#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdio.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

namespace {

template <typename scalar_t>
__global__ void relative_positioning_forward_2d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> r_h,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> r_w,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> new_logits,
    const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> mask,
    const int h_q, const int w_q, const int h_k, const int w_k, const int mask_ndim,
    const bool use_shared_memory, const bool use_mask) {
  const int k = blockIdx.y; //N
  const int i = blockIdx.x; //Hq*Wq
  const int j = threadIdx.x; //Hk*Wk
  const int r_h_index = j/w_k - i/w_q + h_q - 1;
  const int r_w_index = j%w_k - i%w_q + w_q - 1;
  if(use_shared_memory){
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char _shared_memory_ptr[];
    scalar_t *_shared_memory = reinterpret_cast<scalar_t *>(_shared_memory_ptr);
    scalar_t* r_h_shared = _shared_memory;
    scalar_t* r_w_shared = &_shared_memory[h_k + h_q - 1];
    if(j < (h_k + h_q - 1))
      r_h_shared[j] = r_h[k][i][j];
    if(j < (w_k + w_q - 1))
      r_w_shared[j] = r_w[k][i][j];
    __syncthreads();
    new_logits[k][i][j] += r_h_shared[r_h_index] + r_w_shared[r_w_index];
  } else
    new_logits[k][i][j] += r_h[k][i][r_h_index] + r_w[k][i][r_w_index];
  if(use_mask)
    new_logits[k][i][j] += mask[mask_ndim == 2 ? 0 : k][i][j] ? -10000.0 : 0.0;
}

template <typename scalar_t>
__global__ void relative_positioning_forward_3d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> r_t,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> r_h,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> r_w,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> new_logits,
    const torch::PackedTensorAccessor<bool, 3, torch::RestrictPtrTraits, size_t> mask,
    const int t_q, const int h_q, const int w_q, const int t_k, const int h_k, const int w_k,
    const int mask_ndim, const bool use_shared_memory, const bool use_mask) {
  const int k = blockIdx.z;   // N
  const int i = blockIdx.y;   // Tq * Hq * Wq
  const int k_t = blockIdx.x; // Tk
  const int j = threadIdx.x;  // Hk * Wk
  const int q_t = i / (h_q * w_q);
  const int q_h = (i % (h_q * w_q)) / w_q;
  const int q_w = i % w_q;
  const int k_h = j / w_k;
  const int k_w = j % w_k;
  const int r_t_index = k_t - q_t + t_q - 1;
  const int r_h_index = k_h - q_h + h_q - 1;
  const int r_w_index = k_w - q_w + w_q - 1;
  const int l = k_t * h_k * w_k + j;
  if(use_shared_memory){
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char _shared_memory_ptr[];
    scalar_t *_shared_memory = reinterpret_cast<scalar_t *>(_shared_memory_ptr);
    scalar_t* r_h_shared = _shared_memory;
    scalar_t* r_w_shared = &_shared_memory[h_k + h_q - 1];
    if(j < (h_k + h_q - 1))
      r_h_shared[j] = r_h[k][i][j];
    if(j < (w_k + w_q - 1))
      r_w_shared[j] = r_w[k][i][j];
    __syncthreads();
    new_logits[k][i][l] += r_t[k][i][r_t_index] + r_h_shared[r_h_index] + r_w_shared[r_w_index] + ((mask[mask_ndim == 2 ? 0 : k][i][l] && use_mask) ? -10000.0 : 0.0);
  } else
    new_logits[k][i][l] += r_t[k][i][r_t_index] + r_h[k][i][r_h_index] + r_w[k][i][r_w_index] + ((mask[mask_ndim == 2 ? 0 : k][i][l] && use_mask) ? -10000.0 : 0.0);
}

template <typename scalar_t>
__global__ void relative_positioning_backward_2d_kernel_h(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_out,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_h_r,
    const int h_q, const int w_q, const int h_k, const int w_k) {
  //Wq * Hk threads(=> N * Hq blocks), each summing over Wk sequential elements and then place them in the right location
  const int k = blockIdx.y; //N
  const int i = blockIdx.x; //Hq
  const int l = threadIdx.y; //Wq
  const int h = threadIdx.x; //Hk
  scalar_t ans = 0.0;
  for(int w = 0; w < w_k; ++w)
    ans += grad_out[k][i * w_q + l][h * w_k + w];
  grad_h_r[k][i * w_q + l][h - i + h_q - 1] = ans;
}

template <typename scalar_t>
__global__ void relative_positioning_backward_2d_kernel_w(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_out,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_w_r,
    const int h_q, const int w_q, const int h_k, const int w_k) {
  //Wq * Hk threads(=> N * Wh blocks), each summing over Hk non-sequential elements and then place them in the right location
  const int k = blockIdx.y; //N
  const int i = blockIdx.x; //Hq
  const int l = threadIdx.y; //Wq
  const int w = threadIdx.x; //Wk
  scalar_t ans = 0.0;
  for(int h = 0; h < h_k; ++h)
    ans += grad_out[k][i * w_q + l][h * w_k + w];
  grad_w_r[k][i * w_q + l][w - l + w_q - 1] = ans;
}

template <typename scalar_t>
__global__ void relative_positioning_backward_2d_kernel_place(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> sum_out,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits,size_t> grad_x_r,
    const int w_q, const int x_q, const int mode) {
  const int k = blockIdx.y; //N
  const int i = blockIdx.x; //Hq
  const int l = threadIdx.y; //Wq
  const int x = threadIdx.x; //Hk(mode == 0) or Wk(mode == 1)
  //mode == 0; x_q == h_q
  //mode == 1; x_q == w_q
  int q_x;
  if(mode == 0)
    q_x = i;
  else
    q_x = l;
  grad_x_r[k][i * w_q + l][x - q_x + x_q - 1] = sum_out[k][i * w_q + l][x];
}

template <typename scalar_t>
__global__ void relative_positioning_backward_3d_kernel_place(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> sum_out,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits,size_t> grad_x_r,
    const int h_q, const int w_q, const int x_q, const int mode) {
  const int k = blockIdx.y; //N
  const int i = blockIdx.x; //Tq*Hq
  const int l = threadIdx.y; //Wq
  const int j = threadIdx.x; //Tk or Hk or Wk
  int q_x;
  if(mode == 0)
    q_x = i / h_q; // q_t
  else if(mode == 1)
    q_x = i % h_q; // q_h
  else
    q_x = l; // q_k
  grad_x_r[k][i * w_q + l][j - q_x + x_q - 1] = sum_out[k][i * w_q + l][j];
}

} // namespace

torch::Tensor relative_positioning_forward_2d_cuda(
    torch::Tensor logits, torch::Tensor r_h, torch::Tensor r_w, torch::Tensor mask,
    const int h_q, const int w_q, const int h_k, const int w_k, const bool use_mask) {
  auto new_logits = logits.clone();
  const dim3 blocks(h_q * w_q, logits.size(0));
  const int threads = h_k * w_k;
  const bool use_shared_memory = (w_k * h_k + 1) >= ((h_q + h_k) > (w_q + w_k) ? (h_q + h_k) : (w_q + w_k));
  const int mask_ndim = mask.ndimension();
  const int shared_memory_amount = use_shared_memory ? (h_k + h_q - 1 + w_k + w_q - 1) : 0;
  mask = mask_ndim == 2 ? mask.unsqueeze(0) : mask;
  AT_DISPATCH_FLOATING_TYPES(logits.type(), "relative_positioning_forward_2d_kernel", ([&] {
    relative_positioning_forward_2d_kernel<scalar_t><<<blocks, threads, shared_memory_amount*sizeof(scalar_t)>>>(
    r_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    r_w.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    new_logits.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
    h_q, w_q, h_k, w_k, mask_ndim, use_shared_memory, use_mask);
  }));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  return new_logits;
}

std::vector<torch::Tensor> relative_positioning_backward_2d_cuda(
    torch::Tensor grad_out, const int h_q, const int w_q, const int h_k, const int w_k) {
  const dim3 blocks(h_q, grad_out.size(0));
  auto grad_out_view = grad_out.view({grad_out.size(0), grad_out.size(1), h_k, w_k});

  auto grad_h_r = torch::zeros({grad_out.size(0), grad_out.size(1), h_k + h_q - 1},
                                torch::dtype(grad_out.dtype()).device(grad_out.device()));
  const dim3 threads_h(h_k, w_q);
  {
      AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "relative_positioning_backward_2d_kernel_h", ([&] {
      relative_positioning_backward_2d_kernel_h<scalar_t><<<blocks, threads_h>>>(
        grad_out.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_h_r.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        h_q, w_q, h_k, w_k);
      }));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
  }

  auto grad_w_r = torch::zeros({grad_out.size(0), grad_out.size(1), w_k + w_q - 1},
                                torch::dtype(grad_out.dtype()).device(grad_out.device()));
  const dim3 threads_w(w_k, w_q);
  {
      auto grad_out_sum_h = grad_out_view.sum({2});
      AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "relative_positioning_backward_2d_kernel_place_w", ([&] {
      relative_positioning_backward_2d_kernel_place<scalar_t><<<blocks, threads_w>>>(
        grad_out_sum_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_w_r.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        w_q, w_q, 1);
      }));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
  }
  return {grad_h_r, grad_w_r};
}

torch::Tensor relative_positioning_forward_3d_cuda(
    torch::Tensor logits, torch::Tensor r_t, torch::Tensor r_h, torch::Tensor r_w, torch::Tensor mask,
    const int t_q, const int h_q, const int w_q, const int t_k, const int h_k, const int w_k, const bool use_mask) {
  auto new_logits = logits.clone();
  const dim3 blocks(t_k, t_q * h_q * w_q, logits.size(0));
  const int threads = h_k * w_k;
  const bool use_shared_memory = (w_k * h_k + 1) >= ((h_q + h_k) > (w_q + w_k) ? (h_q + h_k) : (w_q + w_k));
  const int mask_ndim = mask.ndimension();
  const int shared_memory_amount = use_shared_memory ? (h_k + h_q - 1 + w_k + w_q - 1) : 0;
  mask = mask_ndim == 2 ? mask.unsqueeze(0) : mask;
  AT_DISPATCH_FLOATING_TYPES(logits.type(), "relative_positioning_forward_3d_kernel", ([&] {
    relative_positioning_forward_3d_kernel<scalar_t><<<blocks, threads, shared_memory_amount*sizeof(scalar_t)>>>(
    r_t.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    r_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    r_w.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    new_logits.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    mask.packed_accessor<bool, 3, torch::RestrictPtrTraits, size_t>(),
    t_q, h_q, w_q, t_k, h_k, w_k, mask_ndim, use_shared_memory, use_mask);
  }));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  return new_logits;
}

std::vector<torch::Tensor> relative_positioning_backward_3d_cuda(
    torch::Tensor grad_out, const int t_q, const int h_q, const int w_q,
    const int t_k, const int h_k, const int w_k) {
  const dim3 blocks(t_q * h_q, grad_out.size(0));
  auto grad_out_view = grad_out.view({grad_out.size(0), grad_out.size(1), t_k, h_k, w_k});

  auto grad_t_r = torch::zeros({grad_out.size(0), grad_out.size(1), t_k + t_q - 1},
                                torch::dtype(grad_out.dtype()).device(grad_out.device()));
  auto grad_h_r = torch::zeros({grad_out.size(0), grad_out.size(1), h_k + h_q - 1},
                                torch::dtype(grad_out.dtype()).device(grad_out.device()));
  auto grad_w_r = torch::zeros({grad_out.size(0), grad_out.size(1), w_k + w_q - 1},
                                torch::dtype(grad_out.dtype()).device(grad_out.device()));

  {
      auto grad_out_sum_w = grad_out_view.sum({4});
      {
          const dim3 threads_t(t_k, w_q);
          auto grad_out_sum_w_h = grad_out_sum_w.sum({3});
          AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "relative_positioning_backward_3d_kernel_place_t", ([&] {
          relative_positioning_backward_3d_kernel_place<scalar_t><<<blocks, threads_t>>>(
              grad_out_sum_w_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
              grad_t_r.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
              h_q, w_q, t_q, 0);
          }));
          gpuErrchk(cudaPeekAtLastError());
          gpuErrchk(cudaDeviceSynchronize());
      }
      {
          const dim3 threads_h(h_k, w_q);
          auto grad_out_sum_w_t = grad_out_sum_w.sum({2});
          AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "relative_positioning_backward_3d_kernel_place_t", ([&] {
          relative_positioning_backward_3d_kernel_place<scalar_t><<<blocks, threads_h>>>(
              grad_out_sum_w_t.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
              grad_h_r.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
              h_q, w_q, h_q, 1);
          }));
          gpuErrchk(cudaPeekAtLastError());
          gpuErrchk(cudaDeviceSynchronize());
      }
  }
  {
      const dim3 threads_w(w_k, w_q);
      auto grad_out_sum_h_t = grad_out_view.sum({2, 3});
      AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "relative_positioning_backward_3d_kernel_place_t", ([&] {
      relative_positioning_backward_3d_kernel_place<scalar_t><<<blocks, threads_w>>>(
          grad_out_sum_h_t.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          grad_w_r.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
          h_q, w_q, w_q, 2);
      }));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
  }

  return {grad_t_r, grad_h_r, grad_w_r};
}