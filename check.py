import torch
import numpy as np
from tqdm import trange

from np_implementation import python_relative_att_nd
from relative_embedding import EmbeddingPaddingMode, PositionEmbeddingType
from relative_attention import RelativeAttention1d, RelativeAttention2d, RelativeAttention3d

tf_is_available = True
try:
    from tf_code import TensorFlowTest
except:
    tf_is_available = False


def assert_equal(a, b):
    assert np.allclose(a, b, atol=1.e-5), np.max(np.abs(a - b))


def correctness_check_1d_basic():
    batch_size = 2
    num_heads = 3
    width = 4
    depth = 5
    for _ in trange(100, desc='1d correctness check'):
        for heads_share_relative_embedding in (True, False):
            for mask in [None, torch.randn(batch_size * num_heads, width, width) > 0]:
                net = RelativeAttention1d(num_heads, depth * num_heads, width,
                                          heads_share_relative_embeddings=heads_share_relative_embedding,
                                          embedding_padding_modes=EmbeddingPaddingMode.Zero,
                                          position_embedding_types=PositionEmbeddingType.Learned)
                q = torch.randn(batch_size, num_heads, width, depth)
                k = torch.randn_like(q)
                torch_ans = net(q, k, mask).detach().numpy()
                width_key_relative_embeddings = net.relative_embeddings[0](width).detach().numpy()
                np_ans = python_relative_att_nd(q.numpy(), k.numpy(), heads_share_relative_embedding,
                                                width_key_relative_embeddings, mask=mask)
                assert_equal(torch_ans, np_ans)


def correctness_check_2d():
    batch_size = 2
    num_heads = 3
    width = 4
    depth = 5
    height = 6
    for _ in trange(100, desc='2d correctness check'):
        for heads_share_relative_embedding in (True, False):
            for mask in [None, torch.randn(batch_size * num_heads, height * width, height * width) > 0]:
                net = RelativeAttention2d(num_heads, depth * num_heads, [width, height],
                                          heads_share_relative_embeddings=heads_share_relative_embedding,
                                          embedding_padding_modes=EmbeddingPaddingMode.Zero,
                                          position_embedding_types=PositionEmbeddingType.Learned)
                q = torch.randn(batch_size, num_heads, height, width, depth)
                k = torch.randn_like(q)
                net.use_custom_cuda_kernel = False
                torch_ans = net(q, k, mask).detach().numpy()
                width_key_relative_embeddings = net.relative_embeddings[0](width).detach().numpy()
                height_key_relative_embeddings = net.relative_embeddings[1](height).detach().numpy()
                np_ans = python_relative_att_nd(q.numpy(), k.numpy(), heads_share_relative_embedding,
                                                width_key_relative_embeddings, height_key_relative_embeddings,
                                                mask=mask)
                net = net.cuda()
                net.use_custom_cuda_kernel = True
                custom_ans = net(q.cuda(), k.cuda(), mask.cuda() if mask is not None else None).detach().cpu().numpy()
                assert_equal(torch_ans, np_ans)
                assert_equal(custom_ans, np_ans)


def correctness_check_tf():
    batch_size = 2
    num_heads = 3
    width = 4
    depth = 5
    height = 6
    for _ in trange(100, desc='tf correctness check'):
        for heads_share_relative_embedding in (True, False):
            q, k, tf_ans, height_key_relative_embeddings, width_key_relative_embeddings = TensorFlowTest().run(
                batch_size, height, width, num_heads, max(height, width), depth, heads_share_relative_embedding)
            if heads_share_relative_embedding:
                width_key_relative_embeddings = width_key_relative_embeddings.transpose(1, 0)
                height_key_relative_embeddings = height_key_relative_embeddings.transpose(1, 0)
            else:
                width_key_relative_embeddings = width_key_relative_embeddings.transpose(0, 2, 1)
                height_key_relative_embeddings = height_key_relative_embeddings.transpose(0, 2, 1)
            np_ans = python_relative_att_nd(q, k, heads_share_relative_embedding,
                                            width_key_relative_embeddings, height_key_relative_embeddings)
            assert_equal(tf_ans.reshape(np_ans.shape), np_ans)


def correctness_check_3d():
    batch_size = 1
    num_heads = 2
    width = 3
    depth = 4
    height = 5
    time = 6
    for _ in trange(100, desc='3d correctness check'):
        for heads_share_relative_embedding in (True, False):
            for mask in [None, torch.randn(batch_size * num_heads, time * height * width, time * height * width) > 0]:
                net = RelativeAttention3d(num_heads, depth * num_heads, [width, height, time],
                                          heads_share_relative_embeddings=heads_share_relative_embedding,
                                          embedding_padding_modes=EmbeddingPaddingMode.Zero,
                                          position_embedding_types=PositionEmbeddingType.Learned)
                q = torch.randn(batch_size, num_heads, time, height, width, depth)
                k = torch.randn_like(q)
                net.use_custom_cuda_kernel = False
                torch_ans = net(q, k, mask).detach().numpy()
                width_key_relative_embeddings = net.relative_embeddings[0](width).detach().numpy()
                height_key_relative_embeddings = net.relative_embeddings[1](height).detach().numpy()
                time_key_relative_embeddings = net.relative_embeddings[2](time).detach().numpy()
                np_ans = python_relative_att_nd(q.numpy(), k.numpy(), heads_share_relative_embedding,
                                                width_key_relative_embeddings, height_key_relative_embeddings,
                                                time_key_relative_embeddings, mask=mask)
                net = net.cuda()
                net.use_custom_cuda_kernel = True
                custom_ans = net(q.cuda(), k.cuda(), mask.cuda() if mask is not None else None).detach().cpu().numpy()
                assert_equal(torch_ans, np_ans)
                assert_equal(custom_ans, np_ans)


if __name__ == '__main__':
    correctness_check_1d_basic()
    correctness_check_2d()
    if tf_is_available:
        correctness_check_tf()
    else:
        print('skipping tf check')
    correctness_check_3d()
