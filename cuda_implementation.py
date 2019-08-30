import torch
from torch.utils.cpp_extension import load

rel_cuda = load('rel_cuda', ['rel_pos_cuda.cpp', 'rel_pos_cuda_kernel.cu'], verbose=True)


class RelativePositioning2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, r_h, r_w, h_q, w_q, h_k, w_k, mask):
        ctx.h_q = h_q
        ctx.w_q = w_q
        ctx.h_k = h_k
        ctx.w_k = w_k
        use_mask = not (mask is None)
        if not use_mask:
            mask = torch.zeros(1, 1, 1).to(logits).bool()
        return rel_cuda.forward_2d(logits, r_h, r_w, mask, h_q, w_q, h_k, w_k, use_mask)

    @staticmethod
    def backward(ctx, grad_output):
        h_q = ctx.h_q
        w_q = ctx.w_q
        h_k = ctx.h_k
        w_k = ctx.w_k
        grad_logits, grad_r_h, grad_r_w = rel_cuda.backward_2d(grad_output, h_q, w_q, h_k, w_k)
        return grad_logits, grad_r_h, grad_r_w, None, None, None, None, None


class RelativePositioning3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, r_t, r_h, r_w, t_q, h_q, w_q, t_k, h_k, w_k, mask):
        ctx.t_q = t_q
        ctx.h_q = h_q
        ctx.w_q = w_q
        ctx.t_k = t_k
        ctx.h_k = h_k
        ctx.w_k = w_k
        use_mask = not (mask is None)
        if not use_mask:
            mask = torch.zeros(1, 1, 1).to(logits).bool()
        return rel_cuda.forward_3d(logits, r_t, r_h, r_w, mask, t_q, h_q, w_q, t_k, h_k, w_k, use_mask)

    @staticmethod
    def backward(ctx, grad_output):
        t_q = ctx.t_q
        h_q = ctx.h_q
        w_q = ctx.w_q
        t_k = ctx.t_k
        h_k = ctx.h_k
        w_k = ctx.w_k
        grad_logits, grad_r_t, grad_r_h, grad_r_w = rel_cuda.backward_3d(grad_output, t_q, h_q, w_q, t_k, h_k, w_k)
        return grad_logits, grad_r_t, grad_r_h, grad_r_w, None, None, None, None, None, None, None


relative_positioning_2d = RelativePositioning2d.apply
relative_positioning_3d = RelativePositioning3d.apply
