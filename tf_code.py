import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
from tqdm import trange

tf.compat.v1.enable_eager_execution()

from cuda_implementation import relative_positioning_2d, relative_positioning_3d


def python_relative_att_nd(q, k, time_key_relative_embeddings, height_key_relative_embeddings,
                           width_key_relative_embeddings, heads_share_relative_embedding):
    """Relative attention computation in numpy.
    Args:
        q: [batch, heads, time or None, height or None, width, depth]
        k: [batch, heads, time or None, height or None, width, depth]
        time_key_relative_embeddings: [heads or None, 2 * time - 1, depth]
        height_key_relative_embeddings: [heads or None, 2 * height - 1, depth]
        width_key_relative_embeddings: [heads or None, 2 * width - 1, depth]
    Returns:
        logits: [batch * num_heads, time * height * width, time * height * width]
    """
    if q.ndim == 5:
        batch, num_heads, time, height, width, _ = q.shape
    elif q.ndim == 4:
        batch, num_heads, height, width, _ = q.shape
        time = 1
        time_key_relative_embeddings = None
    else:
        batch, num_heads, width, _ = q.shape
        time = 1
        time_key_relative_embeddings = None
        height = 1
        height_key_relative_embeddings = None
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
                    logit = np.dot(q[b][h][q_t][q_h][q_w], k[b][h][k_t][k_h][k_w])

                    def x_rel_logit(embedding, x_rel_index):
                        if embedding is None:
                            return 0
                        if heads_share_relative_embedding:
                            return np.dot(q[b][h][q_t][q_h][q_w], embedding[x_rel_index])
                        return np.dot(q[b][h][q_t][q_h][q_w], embedding[h][x_rel_index])

                    logit += x_rel_logit(width_key_relative_embeddings, width - 1 + k_w - q_w)
                    logit += x_rel_logit(height_key_relative_embeddings, height - 1 + k_h - q_h)
                    logit += x_rel_logit(time_key_relative_embeddings, time - 1 + k_t - q_t)
                    logits[b * num_heads + h, i, j] = logit
    return np.reshape(logits, (batch * num_heads, time * height * width, time * height * width))


class RelativeAttention(nn.Module):
    def __init__(self, num_heads, head_depth, n_dimension, use_custom_cuda_kernel,
                 max_relative_positions, heads_share_relative_embeddings):
        super().__init__()
        assert 1 <= n_dimension <= 3
        self.use_custom_cuda_kernel = use_custom_cuda_kernel
        self.n_dimension = n_dimension
        self.head_depth = head_depth
        if not isinstance(max_relative_positions, (list, tuple)):
            max_relative_positions = [max_relative_positions] * n_dimension  # w, h, t
        self.max_relative_positions = max_relative_positions
        self.num_heads = num_heads
        if not isinstance(heads_share_relative_embeddings, (list, tuple)):
            heads_share_relative_embeddings = [heads_share_relative_embeddings] * n_dimension  # w, h, t
        self.heads_share_relative_embeddings = heads_share_relative_embeddings
        initializer_stddev = head_depth ** -0.5
        self.relative_embeddings = nn.ParameterList()
        for max_relative_position, heads_share_relative_embedding in zip(max_relative_positions,
                                                                         heads_share_relative_embeddings):
            max_relative_position_unmasked = 2 * max_relative_position - 1
            if heads_share_relative_embedding:
                embedding_shape = (max_relative_position_unmasked, head_depth)
            else:
                embedding_shape = (num_heads, max_relative_position_unmasked, head_depth)
            self.relative_embeddings.append(nn.Parameter(torch.randn(embedding_shape) * initializer_stddev))

    def forward(self, q, k, mask=None):
        """forward function for RelativeAttention.
            Args:
                q: [batch, heads, None or T, None or H, W, d]
                k: [batch, heads, None or T, None or H, W, d]
                mask: Optional[binary tensor of shape [batch * heads or None, T*H*W, T*H*W]]
                    true to mask(add -10000) and false to attend
            Returns:
                logits: [batch * heads, T * H * W]
        """
        if self.use_custom_cuda_kernel and torch.cuda.is_available() and q.is_cuda and self.n_dimension != 1:
            if self.n_dimension == 2:
                _, _, height_q, width_q, _ = q.size()
                _, _, height_k, width_k, _ = k.size()
            else:
                _, _, time_q, height_q, width_q, _ = q.size()
                _, _, time_k, height_k, width_k, _ = k.size()
            wr = self.compute_rel_logits(q, width_q, 0)
            hr = self.compute_rel_logits(q, height_q, 1)
            if self.n_dimension == 2:
                return relative_positioning_2d(self.get_logits_2d(q, k), hr, wr,
                                               height_q, width_q, height_k, width_k, mask)
            else:
                tr = self.compute_rel_logits(q, time_q, 2)
                return relative_positioning_3d(self.get_logits_3d(q, k), tr, hr, wr, time_q,
                                               height_q, width_q, time_k, height_k, width_k, mask)
        else:
            assert q.size() == k.size(), 'basic RelativeAttention only supports self attention so q.size() == k.size()'
            if self.n_dimension == 2:
                logits = self.dot_product_self_attention_relative_2d(q, k)
            elif self.n_dimension == 3:
                logits = self.dot_product_self_attention_relative_3d(q, k)
            else:
                logits = self.dot_product_self_attention_relative_1d(q, k)
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                return logits + mask.to(logits.dtype) * -10000.0
            return logits

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

    def get_relative_embeddings(self, q_length, dim):
        """retrieve relative embeddings, sliced according to length.
        Args:
            q_length: an Integer, specifies the length of the input sequence for which
                this relative embedding is retrieved for.
            dim: an Integer, specifies the current dimension.
        Returns:
            a Tensor with shape [None or heads, length, depth]
        """
        pad_length = max(q_length - self.max_relative_positions[dim], 0)
        slice_offset = max(self.max_relative_positions[dim] - q_length, 0)
        if self.heads_share_relative_embedding[dim]:
            padded_relative_embeddings = F.pad(self.relative_embeddings[dim], (0, 0, pad_length, pad_length))
            used_relative_embeddings = padded_relative_embeddings[slice_offset:slice_offset + 2 * q_length - 1]
        else:
            padded_relative_embeddings = F.pad(self.relative_embeddings[dim], (0, 0, pad_length, pad_length, 0, 0))
            used_relative_embeddings = padded_relative_embeddings[:, slice_offset:slice_offset + 2 * q_length - 1]
        return used_relative_embeddings

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

    def compute_rel_logits(self, q, q_length, dim):
        return self.matmul_with_relative_keys(q, self.get_relative_embeddings(q_length, dim),
                                              self.heads_share_relative_embeddings[dim])

    def get_logits_3d(self, q, k):
        batch, num_heads, time, height, width, depth_k = q.size()
        return torch.einsum('bnthwd, bnxyzd -> bnthwxyz', q, k).reshape(batch, num_heads, time * height * width,
                                                                        time * height * width)

    def get_logits_2d(self, q, k):
        batch, num_heads, height, width, depth_k = q.size()
        return torch.einsum('bnhwd, bnxyd -> bnhwxy', q, k).reshape(batch, num_heads, height * width, height * width)

    def get_logits_1d(self, q, k):
        batch, num_heads, width, depth_k = q.size()
        return torch.einsum('bnwd, bnxd -> bnwx', q, k).reshape(batch, num_heads, width, width)

    def dot_product_self_attention_relative_3d(self, q, k):
        batch, num_heads, time, height, width, _ = q.size()
        logits = self.get_logits_3d(q, k)
        width_unmasked_rel_logits = self._compute_3d_relative_logits(self.compute_rel_logits(q, width, 0),
                                                                     [0, 1, 2, 3, 4, 5, 6, 7])
        height_unmasked_rel_logits = self._compute_3d_relative_logits(
            self.compute_rel_logits(q, height, 1).permute(0, 1, 2, 4, 3, 5), [0, 1, 2, 4, 3, 5, 7, 6])
        time_unmasked_rel_logits = self._compute_3d_relative_logits(
            self.compute_rel_logits(q, time, 2).permute(0, 1, 4, 3, 2, 5), [0, 1, 4, 3, 2, 7, 6, 5])
        logits = logits + width_unmasked_rel_logits + height_unmasked_rel_logits + time_unmasked_rel_logits
        # reshape back the same spatial dimensions as q
        return logits.reshape(batch * num_heads, time * height * width, time * height * width)

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

    def _compute_1d_relative_logits(self, rel_logits):
        batch, num_heads, width, _ = rel_logits.size()
        rel_logits = rel_logits.reshape(batch, num_heads, width, 2 * width - 1)
        rel_logits = self.relative_position_to_absolute_position(rel_logits)
        return rel_logits.reshape(batch, num_heads, width, width)

    def dot_product_self_attention_relative_2d(self, q, k):
        """Calculate relative position unmasked dot-product self-attention 2d.
        The attention calculation is augmented with learned representations for the
        relative position between each element in q and each element in k and v in
        height and width dimensions. for query index (i,j) and key index (l, m),
        the logit is q_i k_j^T + q_i rh_{l-i}^T + q_i rw_{m-j}^T, where rh and ry are
        the set of relative embeddings in height and width spatial dimensions,
        respectively.
        Args:
          q: a Tensor with shape [batch, heads, height, width, depth].
          k: a Tensor with shape [batch, heads, height, width, depth].
        Returns:
          logits: [batch * heads, height * width, height * width]
        """
        batch, num_heads, height, width, depth_k = q.size()
        logits = self.get_logits_2d(q, k)
        width_unmasked_rel_logits = self._compute_2d_relative_logits(
            self.compute_rel_logits(q, width, 0), height, width, [0, 1, 2, 4, 3, 5])
        height_unmasked_rel_logits = self._compute_2d_relative_logits(
            self.compute_rel_logits(q.permute(0, 1, 3, 2, 4), width, 0), width, height, [0, 1, 4, 2, 5, 3])
        logits = logits + width_unmasked_rel_logits + height_unmasked_rel_logits
        return logits.reshape(batch * num_heads, height * width, height * width)

    def dot_product_self_attention_relative_1d(self, q, k):
        """Calculate relative position unmasked dot-product self-attention 1d.
        Args:
          q: a Tensor with shape [batch, heads, width, depth].
          k: a Tensor with shape [batch, heads, width, depth].
        Returns:
          logits: [batch * heads, width, width]
        """
        batch, num_heads, width, depth_k = q.size()
        logits = self.get_logits_1d(q, k)
        width_unmasked_rel_logits = self._compute_1d_relative_logits(self.compute_rel_logits(q, width, 0))
        logits = logits + width_unmasked_rel_logits
        return logits.reshape(batch * num_heads, width, width)


class PytorchTest:
    def testDotProductUnMaskedAttentionRelative2d(self, batch, time, height, width, num_heads,
                                                  max_relative_position, depth, heads_share_relative_embedding):
        q = np.random.rand(batch, num_heads, height, width, depth).astype(np.float32)
        k = np.random.rand(batch, num_heads, height, width, depth).astype(np.float32)

        a = self.dot_product_unmasked_self_attention_relative_2d(
            torch.from_numpy(q),
            torch.from_numpy(k),
            None,
            max_relative_position=max_relative_position,
            heads_share_relative_embedding=heads_share_relative_embedding)

        res, height_key_relative_embeddings, width_key_relative_embeddings = a
        att_output = python_relative_att_2d(q, k, height_key_relative_embeddings.numpy(),
                                            width_key_relative_embeddings.numpy(), heads_share_relative_embedding)
        assert res.shape == (batch, num_heads, height, width, height, width)
        assert np.allclose(res.numpy(), att_output, atol=1.e-6), np.max(np.abs(res.numpy() - att_output))

        q = torch.from_numpy(q).cuda()
        k = torch.from_numpy(k).cuda()
        logits = torch.einsum('bnhwd, bnxyd -> bnhwxy', q, k).reshape(-1, height * width, height * width)
        height_key_relative_embeddings = self._matmul_with_relative_keys_2d(q, height_key_relative_embeddings.cuda(),
                                                                            heads_share_relative_embedding)
        h_r = height_key_relative_embeddings.reshape(-1, height * width, 2 * height - 1)
        width_key_relative_embeddings = self._matmul_with_relative_keys_2d(q, width_key_relative_embeddings.cuda(),
                                                                           heads_share_relative_embedding)
        w_r = width_key_relative_embeddings.reshape(-1, height * width, 2 * width - 1)
        new_res = relative_positioning_2d(logits, h_r, w_r, height, width, height, width, None).cpu().view_as(
            res).numpy()
        assert np.allclose(new_res, att_output, atol=1.e-6), np.max(np.abs(new_res - att_output))

        q = np.random.rand(batch, num_heads, time, height, width, depth).astype(np.float32)
        k = np.random.rand(batch, num_heads, time, height, width, depth).astype(np.float32)
        if heads_share_relative_embedding:
            t_r = np.random.rand(2 * time - 1, depth).astype(np.float32)
            h_r = np.random.rand(2 * height - 1, depth).astype(np.float32)
            w_r = np.random.rand(2 * width - 1, depth).astype(np.float32)
        else:
            t_r = np.random.rand(num_heads, 2 * time - 1, depth).astype(np.float32)
            h_r = np.random.rand(num_heads, 2 * height - 1, depth).astype(np.float32)
            w_r = np.random.rand(num_heads, 2 * width - 1, depth).astype(np.float32)
        att_output = python_relative_att_3d(q, k, t_r, h_r, w_r, heads_share_relative_embedding)

        q = torch.from_numpy(q).cuda()
        k = torch.from_numpy(k).cuda()
        logits = torch.einsum('bnthwd, bnxyzd -> bnthwxyz', q, k).reshape(-1, time * height * width,
                                                                          time * height * width)
        time_key_relative_embeddings = self._matmul_with_relative_keys_3d(q, torch.from_numpy(t_r).cuda(),
                                                                          heads_share_relative_embedding)
        t_r = time_key_relative_embeddings.reshape(-1, time * height * width, 2 * time - 1)
        height_key_relative_embeddings = self._matmul_with_relative_keys_3d(q, torch.from_numpy(h_r).cuda(),
                                                                            heads_share_relative_embedding)
        h_r = height_key_relative_embeddings.reshape(-1, time * height * width, 2 * height - 1)
        width_key_relative_embeddings = self._matmul_with_relative_keys_3d(q, torch.from_numpy(w_r).cuda(),
                                                                           heads_share_relative_embedding)
        w_r = width_key_relative_embeddings.reshape(-1, time * height * width, 2 * width - 1)

        res = self.dot_product_unmasked_self_attention_relative_3d(q, k, time_key_relative_embeddings,
                                                                   height_key_relative_embeddings,
                                                                   width_key_relative_embeddings).cpu().numpy()
        new_res = relative_positioning_3d(logits, t_r, h_r, w_r, time, height, width, time, height, width,
                                          None).cpu().numpy()
        assert np.allclose(res, att_output, atol=1.e-6), np.max(np.abs(res - att_output))
        assert np.allclose(new_res, att_output, atol=1.e-6), np.max(np.abs(new_res - att_output))


class TensorFlowTest:
    def evaluate(self, tensors):
        """Evaluates tensors and returns numpy values.
        Args:
          tensors: A Tensor or a nested list/tuple of Tensors.
        Returns:
          tensors numpy values.
        """
        return self._eval_helper(tensors)

    def _eval_tensor(self, tensor):
        if tensor is None:
            return None
        elif callable(tensor):
            return self._eval_helper(tensor())
        else:
            try:
                if sparse_tensor.is_sparse(tensor):
                    return sparse_tensor.SparseTensorValue(tensor.indices.numpy(),
                                                           tensor.values.numpy(),
                                                           tensor.dense_shape.numpy())
                elif ragged_tensor.is_ragged(tensor):
                    return ragged_tensor_value.RaggedTensorValue(
                        self._eval_tensor(tensor.values),
                        self._eval_tensor(tensor.row_splits))
                elif isinstance(tensor, ops.IndexedSlices):
                    return ops.IndexedSlicesValue(
                        values=tensor.values.numpy(),
                        indices=tensor.indices.numpy(),
                        dense_shape=tensor.dense_shape.numpy())
                return tensor.numpy()
            except AttributeError as e:
                ValueError("Unsupported type %s." % type(tensor)), e

    def _eval_helper(self, tensors):
        if tensors is None:
            return None
        return nest.map_structure(self._eval_tensor, tensors)

    def shape_list(self, x):
        """Return list of dims, statically where possible."""
        x = tf.convert_to_tensor(x)

        # If unknown rank, return dynamic shape
        if x.get_shape().dims is None:
            return tf.shape(x)

        static = x.get_shape().as_list()
        shape = tf.shape(x)

        ret = []
        for i, dim in enumerate(static):
            if dim is None:
                dim = shape[i]
            ret.append(dim)
        return ret

    def _matmul_with_relative_keys_2d(self, x, y, heads_share_relative_embedding):
        """Helper function for dot_product_unmasked_self_attention_relative_2d."""
        if heads_share_relative_embedding:
            ret = tf.einsum("bhxyd,md->bhxym", x, y)
        else:
            ret = tf.einsum("bhxyd,hmd->bhxym", x, y)
        return ret

    def _relative_position_to_absolute_position_unmasked(self, x):
        """Converts tensor from relative to aboslute indexing for local attention.
        Args:
          x: a Tensor of shape [batch (or batch*num_blocks), heads,
                                length, 2 * length - 1]
        Returns:
          A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
        """
        x_shape = self.shape_list(x)
        batch = x_shape[0]
        heads = x_shape[1]
        length = x_shape[2]
        # Concat columns of pad to shift from relative to absolute indexing.
        col_pad = tf.zeros((batch, heads, length, 1))
        x = tf.concat([x, col_pad], axis=3)

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        flat_x = tf.reshape(x, [batch, heads, length * 2 * length])
        flat_pad = tf.zeros((batch, heads, length - 1))
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)

        # Reshape and slice out the padded elements.
        final_x = tf.reshape(flat_x_padded, [batch, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :, length - 1:]
        final_x = final_x[:, :, :length, :]
        return final_x

    def dot_product_unmasked_self_attention_relative_2d(self, q, k, bias,
                                                        max_relative_position=None, name=None,
                                                        heads_share_relative_embedding=False,
                                                        add_relative_to_values=False):
        """Calculate relative position unmasked dot-product self-attention 2d.
        The attention calculation is augmented with learned representations for the
        relative position between each element in q and each element in k and v in
        height and width dimensions. for query index (i,j) and key index (l, m),
        the logit is q_i k_j^T + q_i rh_{l-i}^T + q_i rw_{m-j}^T, where rh and ry are
        the set of relative embeddings in height and width spatial dimensions,
        respectively.
        Args:
          q: a Tensor with shape [batch, heads, height, width, depth].
          k: a Tensor with shape [batch, heads, height, width, depth].
          bias: bias Tensor.
          max_relative_position: an integer the max relative embedding considered.
            Changing this invalidates checkpoints.
          dropout_rate: a floating point number.
          image_shapes: optional tuple of integer scalars.
          name: an optional string.
          make_image_summary: Whether to make an attention image summary.
          dropout_broadcast_dims:  an optional list of integers less than 4
            specifying in which dimensions to broadcast the dropout decisions.
            saves memory.
          heads_share_relative_embedding: a boolean indicating wheather to share
            relative embeddings between attention heads.
          add_relative_to_values: a boolean for adding relative embeddings to values.
        Returns:
          [batch, heads, height, width, depth] tensor, the output of attention.
          height_key_relative_embeddings: a 3d or 2d tensor, depending on head sharing
            settings, which are the relative embeddings for height.
          width_key_relative_embeddings: a 3d or 2d tensor, depending on head sharing
            settings, which are the relative embeddings for width.
        Raises:
          ValueError: if max_relative_position is not > 0.
        """
        if not max_relative_position:
            raise ValueError("Max relative position (%s) should be > 0 when using "
                             "relative self attention." % (max_relative_position))

        if add_relative_to_values:
            raise ValueError("Adding relative embeddings to values is not implemented")

        with tf.compat.v1.variable_scope(
                name,
                default_name="dot_product_self_attention_relative_v2",
                values=[q, k]):
            # This calculation only works for self attention.
            # q, k and v must therefore have the same shape.
            q.get_shape().assert_is_compatible_with(k.get_shape())

            (height, width) = (self.shape_list(q)[2],
                               self.shape_list(q)[3])
            k_shape = self.shape_list(k)
            num_heads = k_shape[1]
            depth_k = k_shape[-1]
            # flatten height width
            flatten_hw = lambda x, d: tf.reshape(x, [-1, num_heads, height * width, d])
            # [batch, num_heads, query_length, memory_length]
            logits = tf.matmul(flatten_hw(q, depth_k), flatten_hw(k, depth_k), transpose_b=True)

            def _compute_2d_relative_logits(
                    query, key_relative_embeddings, height, width,
                    heads_share_relative_embedding, transpose_mask):
                """compute relative logits."""
                unmasked_rel_logits = self._matmul_with_relative_keys_2d(
                    query, key_relative_embeddings, heads_share_relative_embedding)
                # collapse height and heads
                unmasked_rel_logits = tf.reshape(unmasked_rel_logits,
                                                 [-1, num_heads * height, width,
                                                  2 * width - 1])
                unmasked_rel_logits = (
                    self._relative_position_to_absolute_position_unmasked(
                        unmasked_rel_logits))
                # shape it back for tiling
                unmasked_rel_logits = tf.reshape(
                    unmasked_rel_logits, [-1, num_heads, height, width, width])
                # tiling it height times
                unmasked_rel_logits = tf.expand_dims(
                    unmasked_rel_logits, axis=3)
                unmasked_rel_logits = tf.tile(unmasked_rel_logits,
                                              [1, 1, 1, height, 1, 1])
                # bringing it to the right shape for adding to the logits.
                unmasked_rel_logits = tf.transpose(unmasked_rel_logits, transpose_mask)
                unmasked_rel_logits = tf.reshape(unmasked_rel_logits,
                                                 [-1, num_heads, height * width,
                                                  height * width])
                return unmasked_rel_logits

            # Relative logits in width dimension first.
            width_key_relative_embeddings = self.get_relative_embeddings_left_right(
                max_relative_position, width, depth_k, num_heads,
                heads_share_relative_embedding,
                "width_key_relative_embeddings")  # => [nH or None, 2*width - 1, depth_k]
            # [batch, heads, height, 2*width-1, 2*width-1]
            width_unmasked_rel_logits = _compute_2d_relative_logits(
                q, width_key_relative_embeddings, height, width,
                heads_share_relative_embedding, [0, 1, 2, 4, 3, 5])
            logits += width_unmasked_rel_logits
            # Relative logits in height dimension next. For ease, we transpose
            # height and width and repeat the above steps, and transpose to eventually
            # put the logits in their right positions.
            # [batch, heads, height, 2*height-1, 2*width-1]
            height_key_relative_embeddings = self.get_relative_embeddings_left_right(
                max_relative_position, height, depth_k, num_heads,
                heads_share_relative_embedding,
                "height_key_relative_embeddings")

            height_unmasked_rel_logits = _compute_2d_relative_logits(
                tf.transpose(q, [0, 1, 3, 2, 4]),
                height_key_relative_embeddings,
                width,
                height,
                heads_share_relative_embedding, [0, 1, 4, 2, 5, 3])
            logits += height_unmasked_rel_logits
            if bias is not None:
                logits += bias
            # reshape back the same spatial dimensions as q
            return (
                tf.reshape(logits, [-1, num_heads, height, width, height, width]),
                height_key_relative_embeddings,
                width_key_relative_embeddings)

    def get_relative_embeddings_left_right(self, max_relative_position, length, depth,
                                           num_heads, heads_share_relative_embedding, name):
        """Instantiate or retrieve relative embeddings, sliced according to length.
        Use for unmasked case where the relative attention looks both left and right.
        Args:
          max_relative_position: an Integer for the number of entries in the relative
            embedding, which corresponds to the max relative distance that is
            considered.
          length: an Integer, specifies the length of the input sequence for which
            this relative embedding is retrieved for.
          depth: an Integer, specifies the depth for relative embeddings.
          num_heads: an Integer, specifies the number of heads.
          heads_share_relative_embedding: a Boolean specifying if the relative
            embedding is shared across heads.
          name: a string giving the name of the embedding variables.
        Returns:
          a Tensor with shape [length, depth]
        """
        initializer_stddev = depth ** -0.5
        max_relative_position_unmasked = 2 * max_relative_position - 1
        if heads_share_relative_embedding:
            embedding_shape = (max_relative_position_unmasked, depth)
        else:
            embedding_shape = (num_heads, max_relative_position_unmasked, depth)
        relative_embeddings = tf.compat.v1.get_variable(
            name=name, shape=embedding_shape,
            initializer=tf.random_normal_initializer(stddev=initializer_stddev))
        # Pad first before slice to avoid using tf.cond.
        pad_length = tf.maximum(length - max_relative_position, 0)
        slice_start_position = tf.maximum(max_relative_position - length, 0)
        if heads_share_relative_embedding:
            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                [[pad_length, pad_length], [0, 0]])
            used_relative_embeddings = tf.slice(
                padded_relative_embeddings,
                [slice_start_position, 0], [2 * length - 1, -1])
        else:
            padded_relative_embeddings = tf.pad(
                relative_embeddings,
                [[0, 0], [pad_length, pad_length], [0, 0]])
            used_relative_embeddings = tf.slice(
                padded_relative_embeddings,
                [0, slice_start_position, 0], [-1, 2 * length - 1, -1])
        return used_relative_embeddings

    def testDotProductUnMaskedAttentionRelative2dSharedOneRow(
            self, batch, height, width, num_heads, max_relative_position, depth, heads_share_relative_embedding):
        q = np.random.rand(batch, num_heads, height, width, depth)
        k = np.random.rand(batch, num_heads, height, width, depth)

        a = self.dot_product_unmasked_self_attention_relative_2d(
            tf.constant(q, dtype=tf.float32),
            tf.constant(k, dtype=tf.float32),
            None,
            max_relative_position=max_relative_position,
            heads_share_relative_embedding=heads_share_relative_embedding)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        res, height_key_relative_embeddings, width_key_relative_embeddings = self.evaluate(a)
        att_output = python_relative_att_2d(q, k, height_key_relative_embeddings,
                                            width_key_relative_embeddings, heads_share_relative_embedding)
        assert res.shape == (batch, num_heads, height, width, height, width)
        assert np.allclose(res, att_output, atol=1.e-6), np.max(np.abs(res - att_output))


def main():
    for a in trange(2):
        if a == 0:
            a = PytorchTest()
        else:
            a = TensorFlowTest()
        for _ in trange(25):
            for heads_share_relative_embedding in (True, False):
                for params in (
                        (1, 1, 10, 12, 2, 6, 3),
                        (1, 4, 1, 12, 2, 6, 3),
                        (2, 2, 10, 1, 2, 6, 3),
                        (1, 2, 10, 12, 2, 1, 1),
                        (1, 4, 10, 12, 2, 2, 8),
                        (4, 1, 10, 12, 2, 12, 10),
                ):
                    a.testDotProductUnMaskedAttentionRelative2d(*params, heads_share_relative_embedding)


if __name__ == '__main__':
    main()

# TODO
#   x_q, x_k
#   speed test for 2d, 3d, 2d_new_fused?, 3d_new_fused?
#   backward test for 2d, 3d, 2d_new_fused?, 3d_new_fused?
#   memory test for forward and backward of all models
#   python wrapper for both mine and pytorch code
