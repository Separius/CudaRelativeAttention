import numpy as np


def python_relative_att_nd(q, k, time_key_relative_embeddings,
                           height_key_relative_embeddings, width_key_relative_embeddings):
    """Relative attention computation in numpy.
    Args:
      q: [batch, heads, time, height, width, depth] tensor
      k: [batch, heads, time, height, width, depth] tensor
      time_key_relative_embeddings:
        with shape: [None or heads, 2 * time - 1, depth]
      height_key_relative_embeddings: a tensor of relative embeddings
        with shape: [None or heads, 2 * height - 1, depth]
      width_key_relative_embeddings: a tensor of relative embeddings
        with shape: [None or heads, 2 * width - 1, depth]
    Returns:
      att_output: A tensor with shape: [batch * heads, time * height * width, time * height * width]
    """
    batch, num_heads, time, height, width, _ = q.shape
    heads_share_relative_embedding = width_key_relative_embeddings.ndim == 2
    has_time = time > 1
    has_height = height > 1
    logits = np.zeros((batch, num_heads, height * width * time, height * width * time))
    for b in range(batch):
        for h in range(num_heads):
            for i in range(time * height * width):
                q_t = i // (height * width)
                q_h = (i - q_t * height * width) // width
                q_w = i % width
                for j in range(time * height * width):
                    k_t = j // (height * width)
                    k_h = (j - k_t * height * width) // width
                    k_w = j % width
                    logit = np.dot(q[b][h][q_t][q_h][q_w], k[b][h][k_t][k_h][k_w])

                    def calc_rel_logit(embedding, index):
                        if heads_share_relative_embedding:
                            return np.dot(q[b][h][q_t][q_h][q_w], embedding[index])
                        return np.dot(q[b][h][q_t][q_h][q_w], embedding[h][index])

                    width_rel_dist = k_w - q_w
                    width_rel_index = width - 1 + width_rel_dist
                    logit += calc_rel_logit(width_key_relative_embeddings, width_rel_index)
                    if has_height:
                        height_rel_dist = k_h - q_h
                        height_rel_index = height - 1 + height_rel_dist
                        logit += calc_rel_logit(height_key_relative_embeddings, height_rel_index)
                    if has_time:
                        time_rel_dist = k_t - q_t
                        time_rel_index = time - 1 + time_rel_dist
                        logit += calc_rel_logit(time_key_relative_embeddings, time_rel_index)
                    logits[b, h, i, j] = logit
    return np.reshape(logits, (batch, num_heads, time, height, width, time, height, width))
