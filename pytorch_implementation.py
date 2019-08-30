import torch


def matmul_with_relative_keys_nd(x, y):
    """Helper function for dot_product_unmasked_self_attention_relative_nd."""
    if x.ndim == 4:  # 1d
        x_representation = 'bhx'
    elif x.ndim == 5:  # 2d
        x_representation = 'bhxy'
    else:  # 3d
        assert x.ndim == 6
        x_representation = 'bhxyz'
    h_representation = '' if y.ndim == 2 else 'h'
    return torch.einsum("{x}d,{h}md->{x}m".format(x=x_representation, h=h_representation), x, y)


def relative_position_to_absolute_position_unmasked(x):
    """Converts tensor from relative to absolute indexing for local attention.
    Args:
      x: a Tensor of shape [batch (or batch*num_blocks),
                            heads, length, 2 * length - 1]
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


def compute_nd_relative_logits(query, key_relative_embeddings, time, height, width, transpose_mask, num_heads):
    """compute relative logits."""
    unmasked_rel_logits = matmul_with_relative_keys_nd(query, key_relative_embeddings)
    # collapse height and heads
    unmasked_rel_logits = unmasked_rel_logits.reshape(-1, num_heads * time * height, width, 2 * width - 1)
    unmasked_rel_logits = relative_position_to_absolute_position_unmasked(unmasked_rel_logits)
    # shape it back for tiling
    unmasked_rel_logits = unmasked_rel_logits.reshape(-1, num_heads, time, height, width, 1, 1, width)
    # tiling it height times
    unmasked_rel_logits = unmasked_rel_logits.expand(-1, -1, -1, -1, -1, time, height, -1)
    # bringing it to the right shape for adding to the logits.
    unmasked_rel_logits = unmasked_rel_logits.permute(transpose_mask)
    unmasked_rel_logits = unmasked_rel_logits.reshape(-1, num_heads, time * height * width, time * height * width)
    return unmasked_rel_logits


def pytorch_relative_att_fused_nd(q, k, time_key_relative_embeddings, height_key_relative_embeddings,
                                  width_key_relative_embeddings):
    # q is (B, N, T, H, W, d), k is (B, N, T, H, W, d)
    # tkr is (N or None, 2*T-1, d), hkr is (N or None, 2*H-1, d), wkr is (N or None, 2*W-1, d)
    _, num_heads, time, height, width, depth_k = q.size()
    # [batch, num_heads, query_length, memory_length]
    logits = torch.einsum('bnthwd, bnxyzd -> bnthwxyz', q, k).view(-1, num_heads, time * height * width,
                                                                   time * height * width)

    # Relative logits in width dimension first.
    if width != 1:
        logits = logits + compute_nd_relative_logits(q, width_key_relative_embeddings, time,
                                                     height, width, [0, 1, 2, 3, 4, 5, 6, 7], num_heads)

    # Relative logits in height dimension next. For ease, we transpose
    # height and width and repeat the above steps, and transpose to eventually
    # put the logits in their right positions.
    if height != 1:
        logits = logits + compute_nd_relative_logits(q.permute(0, 1, 2, 4, 3, 5), height_key_relative_embeddings,
                                                     time, width, height, [0, 1, 2, 4, 3, 5, 7, 6], num_heads)

    # Relative logits in time dimension next.
    if time != 1:
        logits = logits + compute_nd_relative_logits(q.permute(0, 1, 4, 3, 2, 5), time_key_relative_embeddings,
                                                     width, height, time, [0, 1, 4, 3, 2, 7, 6, 5], num_heads)

    # reshape back the same spatial dimensions as q
    return logits.reshape(-1, num_heads, time, height, width, time, height, width)


def compute_2d_relative_logits(query, key_relative_embeddings, height, width, transpose_mask, num_heads):
    """compute relative logits."""
    unmasked_rel_logits = matmul_with_relative_keys_nd(query, key_relative_embeddings)
    # collapse height and heads
    unmasked_rel_logits = torch.reshape(unmasked_rel_logits, [-1, num_heads * height, width, 2 * width - 1])
    unmasked_rel_logits = relative_position_to_absolute_position_unmasked(unmasked_rel_logits)
    # shape it back for tiling
    unmasked_rel_logits = torch.reshape(unmasked_rel_logits, [-1, num_heads, height, width, width])
    # tiling it height times
    unmasked_rel_logits = unmasked_rel_logits.unsqueeze(3)
    unmasked_rel_logits = unmasked_rel_logits.expand(-1, -1, -1, height, -1, -1)
    # bringing it to the right shape for adding to the logits.
    unmasked_rel_logits = unmasked_rel_logits.permute(transpose_mask)
    return torch.reshape(unmasked_rel_logits, [-1, num_heads, height * width, height * width])


def pytorch_relative_att_fused_2d(q, k, height_key_relative_embeddings, width_key_relative_embeddings):
    # q is (B, N, H, W, d), k is (B, N, H, W, d)
    # hkr is (N, 2*H-1, d), wkr is (N, 2*W-1, d)
    _, num_heads, height, width, depth_k = q.size()
    # [batch, num_heads, query_length, memory_length]
    logits = torch.einsum('bnhwd, bnxyd -> bnhwxy', q, k).view(-1, num_heads, height * width, height * width)

    # Relative logits in width dimension first.
    # [batch, heads, height, 2*width-1, 2*width-1]
    width_unmasked_rel_logits = compute_2d_relative_logits(q, width_key_relative_embeddings,
                                                           height, width, [0, 1, 2, 4, 3, 5], num_heads)
    logits = logits + width_unmasked_rel_logits
    # Relative logits in height dimension next. For ease, we transpose
    # height and width and repeat the above steps, and transpose to eventually
    # put the logits in their right positions.
    # [batch, heads, height, 2*height-1, 2*width-1]

    height_unmasked_rel_logits = compute_2d_relative_logits(q.permute(0, 1, 3, 2, 4), height_key_relative_embeddings,
                                                            width, height, [0, 1, 4, 2, 5, 3], num_heads)
    logits = logits + height_unmasked_rel_logits
    # reshape back the same spatial dimensions as q
    return torch.reshape(logits, [-1, num_heads, height, width, height, width])
