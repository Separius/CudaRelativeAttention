import numpy as np
import torch

from np_implementation import python_relative_att_nd
from pt_code import python_relative_att, pytorch_relative_att_fused
from pytorch_implementation import pytorch_relative_att_fused_nd


def compare_pytorch_np():
    for nd in (1, 2, 3):
        batch = 2
        num_heads = 3
        time = 4 if nd == 3 else 1
        height = 5 if nd > 1 else 1
        width = 6
        d = 7
        for i in range(100):
            q = torch.randn(batch, num_heads, time, height, width, d)
            k = torch.randn_like(q)
            for heads_share_relative_embedding in (True, False):
                time_key_relative_embeddings = torch.randn(num_heads, 2 * time - 1, d)
                height_key_relative_embeddings = torch.randn(num_heads, 2 * height - 1, d)
                width_key_relative_embeddings = torch.randn(num_heads, 2 * width - 1, d)
                if heads_share_relative_embedding:
                    time_key_relative_embeddings = time_key_relative_embeddings[0]
                    height_key_relative_embeddings = height_key_relative_embeddings[0]
                    width_key_relative_embeddings = width_key_relative_embeddings[0]
                torch_ans = pytorch_relative_att_fused_nd(q, k, time_key_relative_embeddings,
                                                          height_key_relative_embeddings,
                                                          width_key_relative_embeddings).numpy().squeeze()
                numpy_ans = python_relative_att_nd(q.numpy(), k.numpy(),
                                                   time_key_relative_embeddings.numpy(),
                                                   height_key_relative_embeddings.numpy(),
                                                   width_key_relative_embeddings.numpy()).squeeze()
                assert np.allclose(torch_ans, numpy_ans, atol=1.e-6), \
                    f'dim: {nd}, shared: {heads_share_relative_embedding}, ' \
                    f'error: {np.max(np.abs(torch_ans, numpy_ans))}, iter: {i}'


def compare_old():
    batch = 2
    num_heads = 3
    height = 5
    width = 6
    d = 7
    for i in range(100):
        q = torch.randn(batch, num_heads, height, width, d)
        k = torch.randn_like(q)
        height_key_relative_embeddings = torch.randn(num_heads, 2 * height - 1, d)
        width_key_relative_embeddings = torch.randn(num_heads, 2 * width - 1, d)
        torch_ans = pytorch_relative_att_fused(q, k, height_key_relative_embeddings,
                                               width_key_relative_embeddings).numpy()
        numpy_ans = python_relative_att(q.numpy(), k.numpy(),
                                        batch, num_heads, height, width,
                                        height_key_relative_embeddings,
                                        width_key_relative_embeddings,
                                        False)
        assert np.allclose(torch_ans, numpy_ans, atol=1.e-6), \
            f'error: {np.max(np.abs(torch_ans, numpy_ans))}, iter: {i}'


if __name__ == '__main__':
    compare_old()
    # compare_pytorch_np()
