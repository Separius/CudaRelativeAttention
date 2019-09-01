import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest

tf.compat.v1.enable_eager_execution()


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

    def run(self, batch, height, width, num_heads, max_relative_position, depth, heads_share_relative_embedding):
        q = np.random.rand(batch, num_heads, height, width, depth)
        k = np.random.rand(batch, num_heads, height, width, depth)

        a = self.dot_product_unmasked_self_attention_relative_2d(
            tf.constant(q, dtype=tf.float32),
            tf.constant(k, dtype=tf.float32),
            None, max_relative_position=max_relative_position,
            heads_share_relative_embedding=heads_share_relative_embedding)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        res, height_key_relative_embeddings, width_key_relative_embeddings = self.evaluate(a)
        return q, k, res, height_key_relative_embeddings, width_key_relative_embeddings

# TODO
#   speed test for {2d, 3d} * {forward, backward}
#   grad test for {2d, 3d}
#   memory test for {2d, 3d} * {forward, backward}
