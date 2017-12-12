import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def encoder(image, is_train=True, reuse=False):
    """Encoder from raw pixels to an embedding."""
    # Adopting a simple DCGAN-inspired image encoder.
    num_filters = 32
    
    with tf.variable_scope("encoder", reuse=reuse):
        conv_0 = tf.layers.conv2d(image, 
                                 filters=num_filters, 
                                 kernel_size=(5,5), 
                                 strides=(2,2),
                                 activation=None)
        lrelu_0 = tf.nn.leaky_relu(conv_0, 0.2)
        conv_1 = tf.layers.conv2d(lrelu_0, 
                                 filters=num_filters*2, 
                                 kernel_size=(5,5), 
                                 strides=(2,2),
                                 activation=None)
        batch_norm_1 = tf.layers.batch_normalization(conv_1, training=is_train)
        lrelu_1 = tf.nn.leaky_relu(batch_norm_1, 0.2)
        conv_2 = tf.layers.conv2d(lrelu_1, 
                                 filters=num_filters*4, 
                                 kernel_size=(5,5), 
                                 strides=(2,2),
                                 activation=None)
        batch_norm_2 = tf.layers.batch_normalization(conv_2, training=is_train)
        lrelu_2 = tf.nn.leaky_relu(batch_norm_2, 0.2)
        conv_3 = tf.layers.conv2d(conv_2, 
                                 filters=num_filters*8, 
                                 kernel_size=(5,5), 
                                 strides=(2,2),
                                 activation=None)
        batch_norm_3 = tf.layers.batch_normalization(conv_3, training=is_train)
        lrelu_3 = tf.nn.leaky_relu(batch_norm_3, 0.2)

        # Flatten convolutional feature maps.
        flattened = tf.layers.flatten(lrelu_3)
        embedding = tf.layers.dense(flattened,
                                    units=FLAGS.representation_dim)
        return embedding


def representation_recurrence(image_encoding):
    """Recurrent component of representation."""
    pass


def representation(FLAGS):
    """Representation (R)"""
    pass    

