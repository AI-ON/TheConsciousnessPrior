import tensorflow as tf

def _rowwise_unsorted_segment_sum(values, indices, n):
    """UnsortedSegmentSum on each row.
    Args:
        values: a `Tensor` with shape `[batch_size, k]`.
        indices: an integer `Tensor` with shape `[batch_size, k]`.
        n: an integer.
    Returns:
        A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
    """
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])
