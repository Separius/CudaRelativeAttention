import torch
import numpy as np


def python_relative_att(q, k, batch, num_heads, height, width,
                        height_key_relative_embeddings,
                        width_key_relative_embeddings,
                        heads_share_relative_embedding):
    """Relative attention computation in numpy.
    For query index (i,j) and key index (l, m) the logit is
    q_i k_j^T + q_i rh_{l-i}^T + q_i rw_{m-j}^T, where rh and ry are the set of
    relative embeddings in height and width spatial dimensions, respectively.
    Args:
      q: [batch, heads, height, width, depth] tensor
      k: [batch, heads, height, width, depth] tensor
      batch: int scalar
      num_heads: int scalar
      height: int scalar
      width: int scalar
      depth: int scalar
      height_key_relative_embeddings: a tensor of relative embeddings
      width_key_relative_embeddings: a tensor of relative embeddings
      heads_share_relative_embedding: a boolean
    Returns:
      att_output: A tensor
    """

    logits = np.zeros((batch, num_heads, height * width, height * width))
    for b in range(batch):
        for h in range(num_heads):
            for i in range(height * width):
                q_col = i % width
                q_row = int((i - q_col) / width)
                for j in range(height * width):
                    k_col = j % width
                    k_row = int((j - k_col) / width)
                    logit = np.dot(q[b][h][q_row][q_col], k[b][h][k_row][k_col])
                    width_rel_dist = k_col - q_col
                    width_rel_index = width - 1 + width_rel_dist
                    if heads_share_relative_embedding:
                        width_rel_logit = (
                            np.dot(q[b][h][q_row][q_col],
                                   width_key_relative_embeddings[width_rel_index]))
                    else:
                        width_rel_logit = (
                            np.dot(q[b][h][q_row][q_col],
                                   width_key_relative_embeddings[h][width_rel_index]))
                    height_rel_dist = k_row - q_row
                    height_rel_index = height - 1 + height_rel_dist
                    if heads_share_relative_embedding:
                        height_rel_logit = (
                            np.dot(q[b][h][q_row][q_col],
                                   height_key_relative_embeddings[height_rel_index]))
                    else:
                        height_rel_logit = (
                            np.dot(q[b][h][q_row][q_col],
                                   height_key_relative_embeddings[h][height_rel_index]))
                    logits[b, h, i, j] = logit + width_rel_logit + height_rel_logit
    return np.reshape(logits, (batch, num_heads, height, width, height, width))


def _relative_position_to_absolute_position_unmasked(x):
    """Converts tensor from relative to aboslute indexing for local attention.
    Args:
      x: a Tensor of shape [batch (or batch*num_blocks), heads,
                            length, 2 * length - 1]
    Returns:
      A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
    """
    x_shape = x.size()
    batch = x_shape[0]
    heads = x_shape[1]
    length = x_shape[2]
    # Concat columns of pad to shift from relative to absolute indexing.
    col_pad = torch.zeros((batch, heads, length, 1)).to(x)
    x = torch.cat([x, col_pad], dim=3)

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    flat_x = torch.reshape(x, [batch, heads, length * 2 * length])
    flat_pad = torch.zeros((batch, heads, length - 1)).to(x)
    flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)

    # Reshape and slice out the padded elements.
    final_x = torch.reshape(flat_x_padded, [batch, heads, length + 1, 2 * length - 1])
    final_x = final_x[:, :, :, length - 1:]
    final_x = final_x[:, :, :length, :]
    return final_x


def _matmul_with_relative_keys_2d(x, y, heads_share_relative_embedding):
    """Helper function for dot_product_unmasked_self_attention_relative_2d."""
    if heads_share_relative_embedding:
        ret = torch.einsum("bhxyd,md->bhxym", x, y)
    else:
        ret = torch.einsum("bhxyd,hmd->bhxym", x, y)
    return ret


def pytorch_relative_att_fused(q, k, height_key_relative_embeddings, width_key_relative_embeddings):
    # q is (B, N, H, W, d), k is (B, N, H, W, d)
    # hkr is (N, 2*H-1, d), wkr is (N, 2*W-1, d)
    _, num_heads, height, width, depth_k = q.size()
    # [batch, num_heads, query_length, memory_length]
    logits = torch.einsum('bnhwd, bnxyd -> bnhwxy', q, k).view(-1, num_heads, height * width, height * width)

    def _compute_2d_relative_logits(
            query, key_relative_embeddings, height, width,
            heads_share_relative_embedding, transpose_mask):
        """compute relative logits."""
        unmasked_rel_logits = _matmul_with_relative_keys_2d(
            query, key_relative_embeddings, heads_share_relative_embedding)
        # collapse height and heads
        unmasked_rel_logits = torch.reshape(unmasked_rel_logits,
                                            [-1, num_heads * height, width,
                                             2 * width - 1])
        unmasked_rel_logits = (
            _relative_position_to_absolute_position_unmasked(
                unmasked_rel_logits))
        # shape it back for tiling
        unmasked_rel_logits = torch.reshape(
            unmasked_rel_logits, [-1, num_heads, height, width, width])
        # tiling it height times
        unmasked_rel_logits = unmasked_rel_logits.unsqueeze(3)
        unmasked_rel_logits = unmasked_rel_logits.expand(-1, -1, -1, height, -1, -1)
        # bringing it to the right shape for adding to the logits.
        unmasked_rel_logits = unmasked_rel_logits.permute(transpose_mask)
        unmasked_rel_logits = torch.reshape(unmasked_rel_logits,
                                            [-1, num_heads, height * width,
                                             height * width])
        return unmasked_rel_logits

    # Relative logits in width dimension first.
    # [batch, heads, height, 2*width-1, 2*width-1]
    width_unmasked_rel_logits = _compute_2d_relative_logits(q, width_key_relative_embeddings,
                                                            height, width, False, [0, 1, 2, 4, 3, 5])
    logits = logits + width_unmasked_rel_logits
    # Relative logits in height dimension next. For ease, we transpose
    # height and width and repeat the above steps, and transpose to eventually
    # put the logits in their right positions.
    # [batch, heads, height, 2*height-1, 2*width-1]

    height_unmasked_rel_logits = _compute_2d_relative_logits(q.permute(0, 1, 3, 2, 4), height_key_relative_embeddings,
                                                             width, height, False, [0, 1, 4, 2, 5, 3])
    logits = logits + height_unmasked_rel_logits
    # reshape back the same spatial dimensions as q
    return torch.reshape(logits, [-1, num_heads, height, width, height, width])

def pytorch_relative_att(logits, h_r, w_r, num_heads, height, width):
    # logits is (B, N, H*W, H*W)
    # h_r is (B, N, H, W, 2*H-1), w_r is (B, N, H, W, 2*W-1)
    def _compute_2d_relative_logits(unmasked_rel_logits, height, width, transpose_mask):
        # collapse height and heads
        unmasked_rel_logits = torch.reshape(unmasked_rel_logits,
                                            [-1, num_heads * height, width,
                                             2 * width - 1])
        unmasked_rel_logits = (
            _relative_position_to_absolute_position_unmasked(
                unmasked_rel_logits))
        # shape it back for tiling
        unmasked_rel_logits = torch.reshape(
            unmasked_rel_logits, [-1, num_heads, height, width, width])
        # tiling it height times
        unmasked_rel_logits = unmasked_rel_logits.unsqueeze(3)
        unmasked_rel_logits = unmasked_rel_logits.expand(-1, -1, -1, height, -1, -1)
        # bringing it to the right shape for adding to the logits.
        unmasked_rel_logits = unmasked_rel_logits.permute(transpose_mask)
        unmasked_rel_logits = torch.reshape(unmasked_rel_logits,
                                            [-1, num_heads, height * width,
                                             height * width])
        return unmasked_rel_logits

    # Relative logits in width dimension first.
    # [batch, heads, height, 2*width-1, 2*width-1]
    width_unmasked_rel_logits = _compute_2d_relative_logits(w_r, height, width, [0, 1, 2, 4, 3, 5])
    logits = logits + width_unmasked_rel_logits
    # Relative logits in height dimension next. For ease, we transpose
    # height and width and repeat the above steps, and transpose to eventually
    # put the logits in their right positions.
    # [batch, heads, height, 2*height-1, 2*width-1]

    height_unmasked_rel_logits = _compute_2d_relative_logits(h_r.permute(0, 1, 3, 2, 4), width, height, [0, 1, 4, 2, 5, 3])
    logits = logits + height_unmasked_rel_logits
    # reshape back the same spatial dimensions as q
    return torch.reshape(logits, [-1, num_heads, height, width, height, width])