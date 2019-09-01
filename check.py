import time

import torch
import numpy as np
from tqdm import trange, tqdm

from cuda_implementation import relative_positioning_2d, relative_positioning_3d
from np_implementation import python_relative_att_nd
from relative_embedding import EmbeddingPaddingMode, PositionEmbeddingType, KeyStartPosition
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


def grad_check():
    for i in trange(10, desc='grad check'):
        for j in range(2):
            if i == 0: H = 1; B = 1; w_q = h_q = 1; w_k = h_k = 1; t_k = t_q = 1
            if i == 1: H = 1; B = 1; w_q = h_q = 2; w_k = h_k = 1; t_k = t_q = 2
            if i == 2: H = 1; B = 2; w_q = h_q = 1; w_k = h_k = 2; t_k = t_q = 4
            if i == 3: H = 1; B = 2; w_q = h_q = 4; w_k = h_k = 4; t_k = t_q = 1
            if i == 4: H = 2; B = 1; w_q = h_q = 8; w_k = h_k = 4; t_k = 1; t_q = 2
            if i == 5: H = 2; B = 1; w_q = h_q = 1; w_k = h_k = 8; t_k = 2; t_q = 1
            if i == 6: H = 2; B = 2; w_q = h_q = 8; w_k = h_k = 1; t_k = 4; t_q = 1
            if i == 7: H = 2; B = 2; w_q = 2; h_q = 4; w_k = 1; h_k = 2; t_k = 1; t_q = 4
            if i == 8: H = 4; B = 1; w_q = 2; h_q = 4; w_k = 4; h_k = 2; t_k = 2; t_q = 2
            if i == 9: H = 4; B = 2; w_q = 4; h_q = 2; w_k = 2; h_k = 1; t_k = 2; t_q = 4
            N = B * H
            if j == 0:
                t_k = t_q = 1
            logits = torch.randn(N, w_q * h_q * t_q, w_k * h_k * t_k).double().cuda().requires_grad_()
            r_w = torch.randn(N, w_q * h_q * t_q, w_q + w_k - 1).double().cuda().requires_grad_()
            r_h = torch.randn(N, w_q * h_q * t_q, h_q + h_k - 1).double().cuda().requires_grad_()
            for mi, mask in enumerate((None, (torch.randn(w_q * h_q * t_q, w_k * h_k * t_k) > 0).bool().cuda(),
                                       (torch.randn(N, w_q * h_q * t_q, w_k * h_k * t_k) > 0).bool().cuda())):
                if j == 0:
                    torch.autograd.gradcheck(relative_positioning_2d, (logits, r_h, r_w, h_q, w_q, h_k, w_k, mask))
                else:
                    r_t = torch.randn(N, w_q * h_q * t_q, t_q + t_k - 1).double().cuda().requires_grad_()
                    torch.autograd.gradcheck(relative_positioning_3d,
                                             (logits, r_t, r_h, r_w, t_q, h_q, w_q, t_k, h_k, w_k, mask))


def speed_check():
    model_depth = 64
    num_heads = 4
    width = 16
    height = 16
    time_ = 8
    num_runs = 1000
    speed_tqdm = tqdm([False, True, False, True], desc='speed check')
    shared_params = dict(max_relative_positions_future=None, heads_share_relative_embeddings=True,
                         embedding_padding_modes=EmbeddingPaddingMode.Extend,
                         position_embedding_types=PositionEmbeddingType.Hybrid,
                         key_start_positions=KeyStartPosition.BeforeQuery, add_bias_to_query_for_relative_logits=True,
                         add_bias_to_query_for_key_logit=True, use_custom_cuda_kernel=True)
    forward_speedup_2d = 'N/A'
    backward_speedup_2d = 'N/A'
    forward_speedup_3d = 'N/A'
    backward_speedup_3d = 'N/A'
    for i, backward in enumerate(speed_tqdm):
        if i // 2 == 0:
            B = 32
            net = RelativeAttention2d(num_heads, model_depth, [width, height], **shared_params).cuda()
        else:
            B = 16
            net = RelativeAttention3d(num_heads, model_depth, [width, height, time_], **shared_params).cuda()
        with torch.set_grad_enabled(backward):
            if i // 2 == 0:
                q = torch.randn(B, num_heads, height, width, model_depth // num_heads).cuda()
            else:
                q = torch.randn(B, num_heads, time_, height, width, model_depth // num_heads).cuda()
            k = torch.randn_like(q).cuda()
            if backward:
                q.requires_grad_()
                k.requires_grad_()
            for j in range(2):
                if j == 0:
                    net.use_custom_cuda_kernel = True
                else:
                    net.use_custom_cuda_kernel = False
                # warmup
                ans = net(q, k)
                if backward:
                    ans.mean().backward()
                start = time.time()
                for _ in trange(num_runs):
                    ans = net(q, k)
                    if backward:
                        ans.mean().backward()
                if j == 0:
                    custom = time.time() - start
                else:
                    default = time.time() - start
            del ans
            if i // 2 == 0:
                if backward:
                    backward_speedup_2d = default / custom
                else:
                    forward_speedup_2d = default / custom
            else:
                if backward:
                    backward_speedup_3d = default / custom
                else:
                    forward_speedup_3d = default / custom
            speed_tqdm.set_description(f'f2: {backward_speedup_2d}, b2: {forward_speedup_2d}, '
                                       f'f3:{forward_speedup_3d}, b3:{backward_speedup_3d}')


if __name__ == '__main__':
    # correctness_check_1d_basic()
    # correctness_check_2d()
    # if tf_is_available:
    #     correctness_check_tf()
    # else:
    #     print('skipping tf check')
    # correctness_check_3d()
    # config_check()
    # grad_check()
    if torch.cuda.is_available():
        speed_check()
        # speed_check_3d()
    else:
        print('skipping speed tests')
