# Efficient Relative Attention

## Reasoning
I was trying to use a relative position encoding in my 2d attention network
and there wasn't a good implementation for pytorch, so I decided to adopted the
tensor2tensor implementation into pytorch.
Furthermore our architectures, uses this operation at each layer, so I decided
to make it a bit more efficient by writing a custom **cuda** kernel. It's not
a general purpose kernel and it might be slower than vanilla pytorch code, it
depends on your GPU, your batch_size, query_size and query_dim, so profile it
on your settings before using it.

## How to profile
You can see how to profile it by checking the `speed_check()` and
`run_profiler()` function in `check.py`

## Further Improvements
I also tried to fuse the logit calculation in my kernel, but it was way too
slow, compared to cublas.

## Embedding Class
`DistanceEmbedding` class is an awesome wrapper for most of the common usages,
check it out! :))

## How to use
```py
net = RelativeAttention2d(num_heads, model_depth, [width, height],
                          heads_share_relative_embeddings=True,
                          embedding_padding_modes=EmbeddingPaddingMode.Extend,
                          position_embedding_types=PositionEmbeddingType.Fixed,
                          add_bias_to_query_for_relative_logits=True, # the D term in transformer-xl
                          add_bias_to_query_for_key_logit=True) # the B term in transformer-xl
```
