import tensorflow as tf

def encoder(image, FLAGS):
    """Encoder from raw pixels to an embedding."""
    pass

def representation(image_encoding, FLAGS):
    """Representation RNN (R)."""

    lstm = tf.contrib.rnn.LSTM(FLAGS.representation_dim)


