import torch
import numpy as np
from tqdm import trange, tqdm

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


def must_fail(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
        assert False
    except:
        pass


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
                                          position_embedding_types=PositionEmbeddingType.Learned,
                                          add_bias_to_query_for_relative_logits=False,
                                          add_bias_to_query_for_key_logit=False)
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
                                          position_embedding_types=PositionEmbeddingType.Learned,
                                          add_bias_to_query_for_relative_logits=False,
                                          add_bias_to_query_for_key_logit=False)
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
                                          position_embedding_types=PositionEmbeddingType.Learned,
                                          add_bias_to_query_for_relative_logits=False,
                                          add_bias_to_query_for_key_logit=False)
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


def config_check():
    batch_size = 2
    num_heads = 3
    depth = 5
    width_q = 8
    height_q = 6
    time_q = 4
    config_tqdm = tqdm(total=832, desc='check configs')
    for n in range(3):
        for max_relative_positions_past in (3,):
            for max_relative_positions_future in (2, 5):
                for heads_share_relative_embedding in (True, False):
                    for embedding_padding_mode in range(3):
                        for position_embedding_type in range(3):
                            if position_embedding_type == 1 and embedding_padding_mode == 2:  # learned and extend
                                continue
                            if position_embedding_type == 0 and not heads_share_relative_embedding:  # fixed and !shared
                                continue
                            for key_start_position in range(2):
                                for add_bias_to_query_for_relative_logits in (True,):
                                    for add_bias_to_query_for_key_logit in (True,):
                                        net = RelativeAttention3d if n == 2 else (
                                            RelativeAttention2d if n == 1 else RelativeAttention1d)
                                        net = net(num_heads, num_heads * depth, max_relative_positions_past,
                                                  max_relative_positions_future, heads_share_relative_embedding,
                                                  embedding_padding_mode, position_embedding_type,
                                                  key_start_position, add_bias_to_query_for_relative_logits,
                                                  add_bias_to_query_for_key_logit)
                                        for use_custom in (True, False):
                                            if n != 0:
                                                net.use_custom_cuda_kernel = use_custom
                                                net = net.cuda() if use_custom else net.cpu()
                                                pass
                                            elif use_custom:
                                                continue
                                            if n == 0:
                                                q = torch.randn(batch_size, num_heads, width_q, depth)
                                            elif n == 1:
                                                q = torch.randn(batch_size, num_heads, width_q, height_q, depth)
                                            else:
                                                q = torch.randn(batch_size, num_heads, width_q, height_q, time_q, depth)
                                            if use_custom:
                                                q = q.cuda()
                                            for width_k in (width_q, width_q // 2, width_q * 2):
                                                if (n == 0 or not use_custom) and width_k != width_q:
                                                    continue
                                                if key_start_position == 0 and width_k == width_q // 2:  # before and q > k
                                                    continue
                                                for height_k in (height_q, height_q // 2, height_q * 2):
                                                    if (n == 0 or not use_custom) and height_k != height_q:
                                                        continue
                                                    if key_start_position == 0 and height_k == height_q // 2:  # before and q > k
                                                        continue
                                                    for time_k in (time_q,):
                                                        if (n == 0 or not use_custom) and time_k != time_q:
                                                            continue
                                                        if n == 1 and time_k != time_q:
                                                            continue
                                                        if key_start_position == 0 and time_k == time_q // 2:  # before and q > k
                                                            continue
                                                        if n == 0:
                                                            k = torch.randn(batch_size, num_heads, width_k, depth)
                                                        elif n == 1:
                                                            k = torch.randn(batch_size, num_heads, width_k, height_k,
                                                                            depth)
                                                        else:
                                                            k = torch.randn(batch_size, num_heads, width_k, height_k,
                                                                            time_k, depth)
                                                        if use_custom:
                                                            k = k.cuda()
                                                        net(q, k)
                                                        config_tqdm.update()


if __name__ == '__main__':
    # correctness_check_1d_basic()
    # correctness_check_2d()
    # if tf_is_available:
    #     correctness_check_tf()
    # else:
    #     print('skipping tf check')
    # correctness_check_3d()
    config_check()
