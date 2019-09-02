import time

import torch
from tqdm import trange

from relative_embedding import EmbeddingPaddingMode, PositionEmbeddingType, KeyStartPosition
from relative_attention import RelativeAttention2d
from cuda_implementation import rel_cuda


def fused_call(q, k, rh, rw, uk, uh, uw, m, h, w, n):
    return rel_cuda.fuse_all_2d(q, k, rh, rw, uk, uh, uw, m, h, w, h, w, n)


def main():
    batch = 1
    num_heads = 1
    model_depth = 16
    h = 8
    w = 8
    max_relative_positions_past = h - 1
    max_relative_positions_future = h - 1
    num_runs = 400
    net = RelativeAttention2d(num_heads, model_depth, max_relative_positions_past, max_relative_positions_future,
                              heads_share_relative_embeddings=False,
                              embedding_padding_modes=EmbeddingPaddingMode.Extend,
                              position_embedding_types=PositionEmbeddingType.Hybrid,
                              key_start_positions=KeyStartPosition.BeforeQuery,
                              add_bias_to_query_for_relative_logits=True, add_bias_to_query_for_key_logit=True,
                              use_custom_cuda_kernel=True).cuda()
    q = torch.randn(batch, num_heads, h, w, model_depth // num_heads, device='cuda')
    k = torch.randn_like(q)
    m = (torch.randn(batch * num_heads, h * w, h * w) > 0).cuda()
    net(q, k)  # warmup
    start = time.time()
    for _ in trange(num_runs):
        net(q, k)
    non_fused = time.time() - start
    q = q.reshape(batch * num_heads, h * w, -1)
    k = k.reshape_as(q)
    rh = net.relative_embeddings[1](h).permute(0, 2, 1).contiguous()
    rw = net.relative_embeddings[0](w).permute(0, 2, 1).contiguous()
    uk = net.query_to_key_bias
    uh = net.relative_biases[1].permute(1, 0).contiguous()
    uw = net.relative_biases[0].permute(1, 0).contiguous()
    fused_call(q, k, rh, rw, uk, uh, uw, m, h, w, num_heads)  # warmup
    start = time.time()
    for _ in trange(num_runs):
        fused_call(q, k, rh, rw, uk, uh, uw, m, h, w, num_heads)
    fused = time.time() - start
    print(non_fused / fused)


if __name__ == '__main__':
    main()
