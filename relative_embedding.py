import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingPaddingMode(Enum):
    Replication = 0
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
        self.max_relative_position_past = max_relative_position_past
        self.max_relative_position_future = max_relative_position_future
        self.heads_share_relative_embedding = heads_share_relative_embedding
        self.embedding_padding_mode = embedding_padding_mode
        self.position_embedding_type = position_embedding_type
        self.key_start_position = key_start_position
        if position_embedding_type == PositionEmbeddingType.Learned:
            assert embedding_padding_mode != EmbeddingPaddingMode.Extend
            if heads_share_relative_embedding:
                embedding_shape = (max_relative_position_past + max_relative_position_future + 1, depth)
            else:
                embedding_shape = (num_heads, max_relative_position_past + max_relative_position_future + 1, depth)
            self.embedding = nn.Parameter(torch.empty(embedding_shape))
            nn.init.normal_(self.embedding, mean=0, std=depth ** -0.5)
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
                    weight_shape = (depth, depth)
                else:
                    weight_shape = (num_heads, depth, depth)
                self.weight = nn.Parameter(torch.eye(weight_shape))

    def get_sinusoidal_embedding(self, past, future):
        if self.last_past is not None and past <= self.last_past and \
                self.last_future is not None and future <= self.last_future:
            emb = self.embedding.to(self._float_tensor)
        else:
            half_dim = self.depth // 2
            num_embeddings = past + future + 1
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            emb = (torch.arange(num_embeddings, dtype=torch.float) - past).unsqueeze(1) * emb.unsqueeze(0)
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
            past = k_len - 1
            future = q_len - 1
        else:
            past = q_len - 1
            future = k_len - 1
        if self.position_embedding_type == PositionEmbeddingType.Learned:
            initial_embedding = self.embedding  # (Nh or None, max_past+max_future+1, depth)
            based_on_last = False
        else:
            if self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
                based_on_last = True
            else:
                past = self.max_relative_position_past
                future = self.max_relative_position_future
                based_on_last = False
            initial_embedding = self.get_sinusoidal_embedding(past, future)
        initial_embedding = self.prune_embedding(past + 1, future + 1, initial_embedding, based_on_last)
        if self.position_embedding_type == PositionEmbeddingType.Hybrid:
            if self.heads_share_relative_embedding:
                initial_embedding = torch.einsum('td, de -> te', initial_embedding, self.weight)
            else:
                initial_embedding = torch.einsum('htd, hde -> hte', initial_embedding, self.weight)
        if self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            return initial_embedding
        if self.heads_share_relative_embedding:
            pad_shape = (0, 0, past - self.max_relative_position_past, future - self.max_relative_position_future)
        else:
            pad_shape = (0, 0, past - self.max_relative_position_past, future - self.max_relative_position_future, 0, 0)
        return F.pad(initial_embedding, pad_shape,
                     'replicate' if self.embedding_padding_mode == EmbeddingPaddingMode.Replication else 'constant')

    def prune_embedding(self, past_len, future_len, embedding, based_on_last):
        # (Nh or None, max_past+max_future+1 or last_past+last_future+1, depth)
        prev_past = self.last_past if based_on_last else self.max_relative_position_past
        prev_future = self.last_future if based_on_last else self.max_relative_position_future
        return embedding[..., max(0, prev_past - past_len + 1):prev_past + max(0, prev_future - future_len + 1), :]

    def forward(self, q_len, k_len=None, q=None):
        if k_len is None:
            k_len = q_len
        distance_embedding = self.get_distance_embedding(q_len, k_len)
        if q is None:
            return distance_embedding
        return self.matmul_with_relative_keys(q, distance_embedding, self.heads_share_relative_embedding)
