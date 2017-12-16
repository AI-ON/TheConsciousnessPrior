import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def encoder(image, is_train=True, reuse=False):
    """Encoder from raw pixels to an embedding.
    
    Input:

    Returns:

    """
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

        
def representation(images, is_train=True, rnn_type="lstm"):
    """Representation (R).  This encodes the frame using an encoder and
    maintains a recurrent computation over the sequence of images to maintain
    the full (unconscious) representation of the environment.
    
    Input:
        images:  Sequence of images. Shape: [batch_size, time_steps, image_dim,
        image_dim, channels]
        is_train:  (default=True) Whether the module is currently training.  

    Returns:
        outputs:  Full unconscious representation of the environment.
    """
    # Using the same representation dimension for the top level of the encoder
    # and the RNN.
    rnn_dim = FLAGS.representation_dim

    # Choose RNN based on the type. Right now it's either GRU or LSTM
    if (rnn_type == "lstm"):
        rnn = tf.contrib.rnn.BasicLSTMCell(rnn_dim)
    elif(rnn_type == "gru"):
        rnn = tf.contrib.rnn.GRUCell(rnn_dim)

    # Initial state for the representation RNN.
    initial_state = state = rnn.zero_state(FLAGS.batch_size, dtype=tf.float32) 
    
    # Unstack the images.
    images = tf.unstack(images, axis=1)

    with tf.variable_scope("representation") as rnn_rep:
        outputs = []
        # For simplicity, we're considering a statically unrolled RNN.  Dynamic 
        # RNNs may be considered later, as necessary for the tasks.
        for i, image in enumerate(images):
            if i > 0:
                rnn_rep.reuse_variables()
            embedding = encoder(image, is_train=is_train)

            # Recurrent computation.
            output, state = rnn(embedding, state)

            # Append to list. 
            outputs.append(output)
        
        outputs = tf.stack(outputs, axis=1)
        return outputs



