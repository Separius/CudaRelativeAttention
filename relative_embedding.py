from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingPaddingMode(Enum):
    Edge = 0
    Zero = 1
    Extend = 2  # only applicable with Fixed and Hybrid PositionEmbeddingTypes


class PositionEmbeddingType(Enum):
    Fixed = 0
    Learned = 1
    Hybrid = 2


class KeyStartPosition(Enum):
    """
    q1,q2,q3
    k1,k2,k3,k4,k5
    or
          q1,q2,q3
    k1,k2,k3,k4,k5
    """
    BeforeQuery = 0
    WithQuery = 1


class RelativeEmbedding(nn.Module):
    def __init__(self, depth, max_relative_position_past, max_relative_position_future, num_heads,
                 heads_share_relative_embedding, embedding_padding_mode, position_embedding_type, key_start_position):
        super().__init__()
        self.depth = depth
        self.max_relative_position_past = max_relative_position_past + 1  # count rel_dist==0 as past
        self.max_relative_position_future = max_relative_position_future
        self.heads_share_relative_embedding = heads_share_relative_embedding
        self.embedding_padding_mode = embedding_padding_mode
        self.position_embedding_type = position_embedding_type
        self.key_start_position = key_start_position
        if position_embedding_type == PositionEmbeddingType.Learned:
            assert embedding_padding_mode != EmbeddingPaddingMode.Extend
            if heads_share_relative_embedding:
                embedding_shape = (max_relative_position_past + max_relative_position_future, depth)
            else:
                embedding_shape = (num_heads, max_relative_position_past + max_relative_position_future, depth)
            self.embedding = nn.Parameter(torch.empty(embedding_shape))
            nn.init.normal_(self.embedding, mean=0, std=depth ** -0.5)
            self.last_past = max_relative_position_past
            self.last_future = max_relative_position_future
        else:
            self.register_buffer('_float_tensor', torch.FloatTensor(1))
            self.last_past = None
            self.last_future = None
            self.embedding = self.get_sinusoidal_embedding(self.max_relative_position_past,
                                                           self.max_relative_position_future)
            if position_embedding_type == PositionEmbeddingType.Fixed:
                assert heads_share_relative_embedding
            if position_embedding_type == PositionEmbeddingType.Hybrid:
                if heads_share_relative_embedding:
                    self.weight = nn.Parameter(torch.eye(depth))
                else:
                    self.weight = nn.Parameter(torch.eye(depth).unsqueeze(0).repeat(num_heads, 1, 1))


    def get_sinusoidal_embedding(self, past, future):
        if self.last_past is not None and past <= self.last_past and \
                self.last_future is not None and future <= self.last_future:
            emb = self.embedding.to(self._float_tensor)
        else:
            num_embeddings = past + future
            half_dim = self.depth // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            emb = (torch.arange(num_embeddings, dtype=torch.float) - past + 1).unsqueeze(1) * emb.unsqueeze(0)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
            if self.depth % 2 == 1:
                emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
            emb = emb.to(self._float_tensor)
            self.last_past = past
            self.last_future = future
        self.embedding = emb
        return self.embedding

    @staticmethod
    def matmul_with_relative_keys(query, distance_embedding, heads_share_relative_embedding):
        """Helper function for dot_product_unmasked_self_attention_relative_{2, 3}d.
        Args:
            query: [batch, heads, None or T, None or H, W, d]
            distance_embedding: [None or heads, length, d]
        Returns:
            res: [batch, heads, None or T, None or H, W, length]
        """
        dim_str = 'xyz'[:query.ndim - 3]
        head_str = '' if heads_share_relative_embedding else 'h'
        return torch.einsum(f'bh{dim_str}d,{head_str}md->bh{dim_str}m', query, distance_embedding)

    def get_distance_embedding(self, q_len, k_len):
        if self.key_start_position == KeyStartPosition.BeforeQuery:
            assert q_len <= k_len
            past = k_len
            future = q_len - 1
        else:
            past = q_len
            future = k_len - 1
        if self.position_embedding_type == PositionEmbeddingType.Learned:
            initial_embedding = self.embedding  # (Nh or None, max_past+max_future+1, depth)
        elif self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            initial_embedding = self.get_sinusoidal_embedding(past, future)
        else:
            initial_embedding = self.embedding.to(self._float_tensor)
        initial_embedding = self.prune_embedding(past, future, initial_embedding)
        if self.position_embedding_type == PositionEmbeddingType.Hybrid:
            if self.heads_share_relative_embedding:
                initial_embedding = torch.einsum('td, de -> te', initial_embedding, self.weight)
            else:
                initial_embedding = torch.einsum('td, hde -> hte', initial_embedding, self.weight)
        if self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            return initial_embedding
        past_pad = max(past - self.last_past, 0)
        future_pad = max(future - self.last_future, 0)
        if self.embedding_padding_mode == EmbeddingPaddingMode.Zero:
            pad_shape = (0, 0, past_pad, future_pad)
            if not self.heads_share_relative_embedding:
                pad_shape = pad_shape + (0, 0)
            return F.pad(initial_embedding, pad_shape, 'constant')
        # replicate padding does not work on 2d tensors, also it only supports padding in the last dimension
        pad_shape = (past_pad, future_pad)
        if self.heads_share_relative_embedding:
            return F.pad(initial_embedding.transpose(1, 0).unsqueeze(0), pad_shape, 'replicate')[0].transpose(1, 0)
        return F.pad(initial_embedding.transpose(1, 0), pad_shape, 'replicate').transpose(1, 0)

    def prune_embedding(self, past_len, future_len, embedding):
        # (Nh or None, max_past+max_future+1 or last_past+last_future+1, depth)
        return embedding[...,
               max(0, self.last_past - past_len):self.last_past + future_len, :]

    def forward(self, q_len, k_len=None, q=None):
        if k_len is None:
            k_len = q_len
        distance_embedding = self.get_distance_embedding(q_len, k_len)
        if q is None:
            return distance_embedding
        return self.matmul_with_relative_keys(q, distance_embedding, self.heads_share_relative_embedding)


if __name__ == '__main__':
    a = RelativeEmbedding(4, 3, 5, 2, True, EmbeddingPaddingMode.Zero,
                          PositionEmbeddingType.Learned, KeyStartPosition.WithQuery)
