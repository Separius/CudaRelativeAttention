import string

import numpy as np
import torch
import torch.nn as nn

from relative_embedding import DistanceEmbedding, PositionEmbeddingType, KeyStartPosition, EmbeddingPaddingMode
from cuda_implementation import relative_positioning_2d, relative_positioning_3d


class RelativeAttention(nn.Module):
    def __init__(self, n_dim, num_heads, model_depth, max_relative_positions_past,
                 max_relative_positions_future=None, heads_share_relative_embeddings=True,
                 embedding_padding_modes=EmbeddingPaddingMode.Extend,
                 position_embedding_types=PositionEmbeddingType.Hybrid,
                 key_start_positions=KeyStartPosition.BeforeQuery,
                 add_bias_to_query_for_relative_logits=True,  # the d term in transformer-xl(second equation in page 5)
                 add_bias_to_query_for_key_logit=True,  # the b term in transformer-xl(second equation in page 5)
                 use_custom_cuda_kernel=True):
        super().__init__()
        assert model_depth % num_heads == 0
        assert 1 <= n_dim <= 3
        self.use_custom_cuda_kernel = use_custom_cuda_kernel
        self.head_depth = model_depth // num_heads
        self.n_dimension = n_dim
        self.num_heads = num_heads
        max_relative_positions_past = self._get_list(max_relative_positions_past, int)
        if max_relative_positions_future is None:
            max_relative_positions_future = max_relative_positions_past
        else:
            max_relative_positions_future = self._get_list(max_relative_positions_future, int)
        heads_share_relative_embeddings = self._get_list(heads_share_relative_embeddings, bool)
        embedding_padding_modes = self._get_list(embedding_padding_modes, EmbeddingPaddingMode)
        position_embedding_types = self._get_list(position_embedding_types, PositionEmbeddingType)
        key_start_positions = self._get_list(key_start_positions, KeyStartPosition)
        add_bias_to_query_for_relative_logits = self._get_list(add_bias_to_query_for_relative_logits, bool)
        self.relative_biases = []
        for i in range(n_dim):
            new_param = nn.Parameter(torch.randn(self.head_depth, num_heads) * 0.01) \
                if add_bias_to_query_for_relative_logits[i] else None
            self.register_parameter('relative_bias_{}'.format(i + 1), new_param)
            self.relative_biases.append(new_param)
        if add_bias_to_query_for_key_logit:
            self.query_to_key_bias = nn.Parameter(torch.randn(num_heads, self.head_depth) * 0.01)
        else:
            self.register_parameter('query_to_key_bias', None)
        self.relative_embeddings = nn.ModuleList([DistanceEmbedding(self.head_depth, max_relative_positions_past[i],
                                                                    max_relative_positions_future[i], num_heads,
                                                                    heads_share_relative_embeddings[i],
                                                                    embedding_padding_modes[i],
                                                                    position_embedding_types[i],
                                                                    key_start_positions[i]) for i in range(n_dim)])

    def _get_list(self, optional_list, desired_class):
        if not isinstance(optional_list, (list, tuple)):
            obj_list = [optional_list] * self.n_dimension  # w, h, t
        else:
            obj_list = optional_list
        desired_list = []
        for obj in obj_list:
            if desired_class == int:
                if isinstance(obj, int):
                    desired_list.append(obj)
                else:
                    desired_list.append(int(obj))
            elif desired_class == bool:
                if isinstance(obj, bool):
                    desired_list.append(obj)
                else:
                    desired_list.append(bool(obj))
            else:  # enum cases
                if isinstance(obj, desired_class):
                    desired_list.append(obj)
                elif isinstance(obj, str):
                    desired_list.append(desired_class[obj])
                elif isinstance(obj, int):
                    desired_list.append(desired_class(obj))
                else:
                    raise ValueError(f'invalid input({obj}) for enum {desired_class}')
        return desired_list

    @staticmethod
    def relative_position_to_absolute_position(x):
        """Converts tensor from relative to aboslute indexing for local attention.
        Args:
            x: [batch (or batch*num_blocks), heads, length, 2 * length - 1]
        Returns:
            A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch, heads, length, 1), device=x.device, dtype=x.dtype)
        x = torch.cat([x, col_pad], dim=3)
        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        flat_x = x.reshape(batch, heads, length * 2 * length)
        flat_pad = torch.zeros((batch, heads, length - 1), device=x.device, dtype=x.dtype)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        # Reshape and slice out the padded elements.
        final_x = flat_x_padded.reshape(batch, heads, length + 1, 2 * length - 1)
        return final_x[:, :, :length, length - 1:]

    def forward(self, q, k, mask=None):
        raise NotImplementedError()

    @staticmethod
    def apply_mask(logits, mask):
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            return logits + mask.to(logits.dtype) * -10000.0
        return logits

    def get_logits(self, q, k):
        # q is (B, N, ..., d) and k is also (B, N, ..., d); Note that m,n,o are in the middle of alphabet
        # => logits with shape == (B * N, Sq, Sk)
        if self.query_to_key_bias is not None:
            # q is (B, N, ..., d) and bias is (N, d)
            q = q + self.query_to_key_bias.view(1, q.size(1), *([1] * (q.ndim - 3)), -1)
        return torch.einsum(
            'mn{q_dims}o, mn{k_dims}o -> mn{q_dims}{k_dims}'.format(q_dims=string.ascii_lowercase[:q.ndim - 3],
                                                                    k_dims=string.ascii_lowercase[::-1][:k.ndim - 3]),
            q, k).view(q.size(0) * q.size(1), np.prod(q.size()[2:-1]), np.prod(k.size()[2:-1]))


class RelativeAttention1d(RelativeAttention):
    def __init__(self, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future=None,
                 heads_share_relative_embeddings=True, embedding_padding_modes=EmbeddingPaddingMode.Extend,
                 position_embedding_types=PositionEmbeddingType.Hybrid,
                 key_start_positions=KeyStartPosition.BeforeQuery, add_bias_to_query_for_relative_logits=True,
                 add_bias_to_query_for_key_logit=True):
        super().__init__(1, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future,
                         heads_share_relative_embeddings, embedding_padding_modes, position_embedding_types,
                         key_start_positions, add_bias_to_query_for_relative_logits, add_bias_to_query_for_key_logit,
                         use_custom_cuda_kernel=False)

    def forward(self, q, k, mask=None):
        """forward function for RelativeAttention.
            Args:
                q: [batch, heads, Wq, d]
                k: [batch, heads, Wk, d]
                mask: Optional[binary tensor of shape [batch * heads or None, Wq, Wk]]
                        true to mask(add -10000.0) and false to attend
            Returns:
                logits: [batch * heads, Wq, Wk]
        """
        if self.use_custom_cuda_kernel:
            raise ValueError('can not use custom cuda kernel with 1d')
        if not q.size() == k.size():
            raise ValueError('RelativeAttention1d only supports self attention so q.size() == k.size()')
        batch, num_heads, width, _ = q.size()
        logits = self.get_logits(q, k)
        distance_logits = self.relative_embeddings[0](width, q, self.relative_biases[0])
        width_rel_logits = self.relative_position_to_absolute_position(distance_logits).view_as(logits)
        return self.apply_mask(logits + width_rel_logits, mask)


class RelativeAttention2d(RelativeAttention):
    def __init__(self, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future=None,
                 heads_share_relative_embeddings=True, embedding_padding_modes=EmbeddingPaddingMode.Extend,
                 position_embedding_types=PositionEmbeddingType.Hybrid,
                 key_start_positions=KeyStartPosition.BeforeQuery, add_bias_to_query_for_relative_logits=True,
                 add_bias_to_query_for_key_logit=True, use_custom_cuda_kernel=True):
        super().__init__(2, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future,
                         heads_share_relative_embeddings, embedding_padding_modes, position_embedding_types,
                         key_start_positions, add_bias_to_query_for_relative_logits, add_bias_to_query_for_key_logit,
                         use_custom_cuda_kernel)

    def forward(self, q, k, mask=None):
        """forward function for RelativeAttention.
            Args:
                q: [batch, heads, Hq, Wq, d]
                k: [batch, heads, Hk, Wk, d]
                mask: Optional[binary tensor of shape [batch * heads or None, Hq * Wq, Hk * Wk]]
                    true to mask(add -10000) and false to attend
            Returns:
                logits: [batch * heads, Hq * Wq, Hk * Wk]
        """
        batch, num_heads, height_q, width_q, _ = q.size()
        batch, num_heads, height_k, width_k, _ = k.size()
        logits = self.get_logits(q, k)
        wr = self.relative_embeddings[0](width_q, q, self.relative_biases[0], width_k)
        hr = self.relative_embeddings[1](height_q, q, self.relative_biases[1], height_k)
        if self.use_custom_cuda_kernel and torch.cuda.is_available() and q.is_cuda:
            xr_shape = (batch * num_heads, height_q * width_q, -1)
            return relative_positioning_2d(logits, hr.reshape(xr_shape), wr.reshape(xr_shape),
                                           height_q, width_q, height_k, width_k, mask)
        if not q.size() == k.size():
            raise ValueError('basic RelativeAttention2d only supports self attention so q.size() == k.size()')
        width_unmasked_rel_logits = self._compute_2d_relative_logits(wr, height_q, width_q,
                                                                     [0, 1, 2, 4, 3, 5]).view_as(logits)
        height_unmasked_rel_logits = self._compute_2d_relative_logits(hr.permute(0, 1, 3, 2, 4), width_q, height_q,
                                                                      [0, 1, 4, 2, 5, 3]).view_as(logits)
        return self.apply_mask(logits + width_unmasked_rel_logits + height_unmasked_rel_logits, mask)

    def _compute_2d_relative_logits(self, rel_logits, height, width, transpose_mask):
        batch, num_heads, _, _, _ = rel_logits.size()
        # collapse height and heads
        rel_logits = rel_logits.reshape(batch, num_heads * height, width, 2 * width - 1)
        rel_logits = self.relative_position_to_absolute_position(rel_logits)
        # shape it back for tiling
        rel_logits = rel_logits.reshape(batch, num_heads, height, 1, width, width)
        # tiling it height times
        rel_logits = rel_logits.expand(-1, -1, -1, height, -1, -1)
        # bringing it to the right shape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        return rel_logits.reshape(batch, num_heads, height * width, height * width)


class RelativeAttention3d(RelativeAttention):
    def __init__(self, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future=None,
                 heads_share_relative_embeddings=True, embedding_padding_modes=EmbeddingPaddingMode.Extend,
                 position_embedding_types=PositionEmbeddingType.Hybrid,
                 key_start_positions=KeyStartPosition.BeforeQuery, add_bias_to_query_for_relative_logits=True,
                 add_bias_to_query_for_key_logit=True, use_custom_cuda_kernel=True):
        super().__init__(3, num_heads, model_depth, max_relative_positions_past, max_relative_positions_future,
                         heads_share_relative_embeddings, embedding_padding_modes, position_embedding_types,
                         key_start_positions, add_bias_to_query_for_relative_logits, add_bias_to_query_for_key_logit,
                         use_custom_cuda_kernel)

    def forward(self, q, k, mask=None):
        """forward function for RelativeAttention.
            Args:
                q: [batch, heads, Tq, Hq, Wq, d]
                k: [batch, heads, Tk, Hk, Wk, d]
                mask: Optional[binary tensor of shape [batch * heads or None, Tq * Hq * Wq, Tk * Hk * Wk]]
                    true to mask(add -10000) and false to attend
            Returns:
                logits: [batch * heads, Tq * Hq * Wq, Tk * Hk * Wk]
        """
        batch, num_heads, time_q, height_q, width_q, _ = q.size()
        batch, num_heads, time_k, height_k, width_k, _ = k.size()
        logits = self.get_logits(q, k)
        wr = self.relative_embeddings[0](width_q, q, self.relative_biases[0], width_k)
        hr = self.relative_embeddings[1](height_q, q, self.relative_biases[1], height_k)
        tr = self.relative_embeddings[2](time_q, q, self.relative_biases[2], time_k)
        if self.use_custom_cuda_kernel and torch.cuda.is_available() and q.is_cuda:
            xr_shape = (batch * num_heads, time_q * height_q * width_q, -1)
            return relative_positioning_3d(logits, tr.reshape(xr_shape), hr.reshape(xr_shape), wr.reshape(xr_shape),
                                           time_q, height_q, width_q, time_k, height_k, width_k, mask)
        if not q.size() == k.size():
            raise ValueError('basic RelativeAttention3d only supports self attention so q.size() == k.size()')
        width_rel_logits = self._compute_3d_relative_logits(wr, [0, 1, 2, 3, 4, 5, 6, 7]).view_as(logits)
        height_rel_logits = self._compute_3d_relative_logits(hr.permute(0, 1, 2, 4, 3, 5),
                                                             [0, 1, 2, 4, 3, 5, 7, 6]).view_as(logits)
        time_rel_logits = self._compute_3d_relative_logits(tr.permute(0, 1, 4, 3, 2, 5),
                                                           [0, 1, 4, 3, 2, 7, 6, 5]).view_as(logits)
        return self.apply_mask(logits + width_rel_logits + height_rel_logits + time_rel_logits, mask)

    def _compute_3d_relative_logits(self, rel_logits, transpose_mask):
        # unmasked_rel_logits = (B,N,T,H,W,2*W-1) | (B,N,T,W,H,2*H-1) | (B,N,W,H,T,2*T-1) == (B,N,Z,Y,X,2*X-1)
        b, n, z, y, x, _ = rel_logits.size()
        # collapse height and heads
        rel_logits = rel_logits.reshape(b, n * z * y, x, 2 * x - 1)
        rel_logits = self.relative_position_to_absolute_position(rel_logits)
        # now it is B,N*Z*Y,X,X
        # shape it back for tiling
        rel_logits = rel_logits.reshape(b, n, z, y, x, 1, 1, x)
        # tiling it height times
        rel_logits = rel_logits.expand(-1, -1, -1, -1, -1, z, y, -1)
        # bringing it to the right shape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        return rel_logits.reshape(b, n, z * y * x, z * y * x)
