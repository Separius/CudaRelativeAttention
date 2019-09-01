import numpy as np
from relative_embedding import EmbeddingPaddingMode, PositionEmbeddingType, KeyStartPosition

class Tmp:
    def __init__(self, max_relative_position_past, max_relative_position_future,
                 embedding_padding_mode, position_embedding_type, key_start_position):
        super().__init__()
        self.max_relative_position_past = max_relative_position_past + 1  # count rel_dist==0 as past
        self.max_relative_position_future = max_relative_position_future
        self.embedding_padding_mode = embedding_padding_mode
        self.position_embedding_type = position_embedding_type
        self.key_start_position = key_start_position
        self.last_past = None
        self.last_future = None
        self.embedding = self.get_sinusoidal_embedding(self.max_relative_position_past,
                                                       self.max_relative_position_future)

    def get_sinusoidal_embedding(self, past, future):
        if self.last_past is not None and past <= self.last_past and \
                self.last_future is not None and future <= self.last_future:
            emb = self.embedding
        else:
            num_embeddings = past + future
            emb = np.arange(num_embeddings) - past + 1
            self.last_past = past
            self.last_future = future
        self.embedding = emb
        return self.embedding

    def get_distance_embedding(self, q_len, k_len):
        if self.key_start_position == KeyStartPosition.BeforeQuery:
            assert q_len <= k_len
            past = k_len
            future = q_len - 1
        else:
            past = q_len
            future = k_len - 1
        if self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            initial_embedding = self.get_sinusoidal_embedding(past, future)
        else:
            initial_embedding = self.embedding
        initial_embedding = self.prune_embedding(past, future, initial_embedding)
        if self.embedding_padding_mode == EmbeddingPaddingMode.Extend:
            return initial_embedding
        pad_shape = (max(past - self.last_past, 0), max(future - self.last_future, 0))
        return np.pad(initial_embedding, pad_shape,
                      'edge' if self.embedding_padding_mode == EmbeddingPaddingMode.Edge else 'constant')

    def prune_embedding(self, past_len, future_len, embedding):
        # (Nh or None, max_past+max_future+1 or last_past+last_future+1, depth)
        return embedding[max(0, self.last_past - past_len):self.last_past + future_len]

    def forward(self, q_len, k_len=None):
        if k_len is None:
            k_len = q_len
        return self.get_distance_embedding(q_len, k_len)