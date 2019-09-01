import numpy as np


def python_relative_att_nd(q, k, heads_share_relative_embedding, width_key_relative_embeddings,
                           height_key_relative_embeddings=None, time_key_relative_embeddings=None, mask=None):
    """Relative attention computation in numpy.
    Args:
        q: [batch, heads, time or None, height or None, width, depth]
        k: [batch, heads, time or None, height or None, width, depth]
        width_key_relative_embeddings: [heads or None, depth, 2 * width - 1]
        height_key_relative_embeddings: Optional([heads or None, depth, 2 * height - 1])
        time_key_relative_embeddings: Optional([heads or None, depth, 2 * time - 1])
        mask: Optional([batch * heads or None, time * height * width, time * height * width])
    Returns:
        logits: [batch * num_heads, time * height * width, time * height * width]
    """
    if q.ndim == 6:
        batch, num_heads, time, height, width, _ = q.shape
    elif q.ndim == 5:
        batch, num_heads, height, width, _ = q.shape
        time = 1
        time_key_relative_embeddings = None
        q = np.expand_dims(q, 2)
        k = np.expand_dims(k, 2)
    else:
        batch, num_heads, width, _ = q.shape
        time = 1
        time_key_relative_embeddings = None
        height = 1
        height_key_relative_embeddings = None
        q = np.expand_dims(q, 2)
        k = np.expand_dims(k, 2)
        q = np.expand_dims(q, 2)
        k = np.expand_dims(k, 2)
    logits = np.zeros((batch * num_heads, time * height * width, time * height * width))
    for b in range(batch):
        for h in range(num_heads):
            for i in range(time * height * width):
                q_t = i // (height * width)
                q_h = (i % (height * width)) // width
                q_w = i % width
                for j in range(time * height * width):
                    k_t = j // (height * width)
                    k_h = (j % (height * width)) // width
                    k_w = j % width
                    logit = np.dot(q[b, h, q_t, q_h, q_w], k[b, h, k_t, k_h, k_w])

                    def x_rel_logit(embedding, x_rel_index):
                        if embedding is None:
                            return 0
                        if heads_share_relative_embedding:
                            return np.dot(q[b, h, q_t, q_h, q_w], embedding[:, x_rel_index])
                        return np.dot(q[b, h, q_t, q_h, q_w], embedding[h, :, x_rel_index])

                    logit += x_rel_logit(width_key_relative_embeddings, width - 1 + k_w - q_w)
                    logit += x_rel_logit(height_key_relative_embeddings, height - 1 + k_h - q_h)
                    logit += x_rel_logit(time_key_relative_embeddings, time - 1 + k_t - q_t)
                    if mask is not None:
                        if mask.ndim == 2:
                            logit += -10000.0 if mask[i, j] else 0.0
                        else:
                            logit += -10000.0 if mask[b * num_heads + h, i, j] else 0.0
                    logits[b * num_heads + h, i, j] = logit
    return np.reshape(logits, (batch * num_heads, time * height * width, time * height * width))
