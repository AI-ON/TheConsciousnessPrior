import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def integrate_vectors(rep, z):
    """Integrate representation (h_t) with the noise (z_t).  Ensure that the
    output dimension is equal to the consciousness RNN dimension via a learned 
    linear layer."""
    if FLAGS.vector_integration == 'concat':
        combined = tf.concat([rep, z], axis=1)
        #combined = tf.expand_dims(concat, 0)
    
    elif FLAGS.vector_integration == 'outer_prod':
        outer_prod = tf.einsum('i,j->ij', rep, z)
        combined =  tf.reshape(outer_prod, shape=[-1])
        combined = tf.expand_dims(concat, 0)

    else:
        raise NotImplementedError

    output = tf.contrib.layers.fully_connected(combined,
             num_outputs=FLAGS.representation_dim,
             activation_fn=None)
    return output    


def select_conscious_elements(representation, c_rnn_out, is_train):
    """Function that produces the selected elements from the representation 
    Select the top-k elements in a sparse and noisy manner.  We employ the tools
    of Outrageously Large Neural Networks (https://arxiv.org/pdf/1701.06538.pdf)
    which were used for selecting the top-k experts.
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
    
    Args:

    Returns:

    """
    k = FLAGS.num_conscious_elements

    with tf.variable_scope("selection") as sel: 
        w_gate = tf.get_variable("w_gate", 
                [FLAGS.representation_dim, FLAGS.consciousness_dim], 
                tf.float32)
        w_noise = tf.get_variable("w_noise", 
                [FLAGS.representation_dim, FLAGS.consciousness_dim], 
                tf.float32)

        # Selection mechanism. 
        logits = tf.matmul(representation, w_gate)

        # Add noise to the element selection.
        noise_eps = 1e-2
        raw_noise_stddev = tf.matmul(representation, w_noise)
        noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + noise_eps) *
                tf.to_float(is_train))

        # Noisy logits.
        noisy_logits = logits + (tf.random_normal(tf.shape(logits)
            * noise_stddev) 
       
        # Select the top-k elements.
        top_logits, top_indices = tf.nn.top_k(noisy_logits, k)

        top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
        top_k_gates = tf.nn.softmax(top_k_logits)

        return

def consciousness(representations, is_train=True):
    """Consciousness (C) module.  This retrieves elements from the
    representations produced by R and produces a sparse conscious state."""
    
    rnn_dim = FLAGS.consciousness_dim
    
    # Choose RNN based on the type.
    if (FLAGS.rnn_type == "lstm"):
        rnn = tf.contrib.rnn.BasicLSTMCell(rnn_dim)
    elif(FLAGS.rnn_type == "gru"):
        rnn = tf.contrib.rnn.GRUCell(rnn_dim)
    else:
        raise NotImplementedError

    # Initial state for consciousness RNN.
    initial_state = state = rnn.zero_state(FLAGS.batch_size, dtype=tf.float32)

    # Unstack representations.
    representations = tf.unstack(representations, axis=1)

    with tf.variable_scope("consciousness") as c_rnn:
        # As initially diagrammed, the C-module is responsible for producing
        # the current conscious state as well as predicting future conscious
        # elements.
        current_elements = []
        future_elements = []

        for t, rnn_in in enumerate(representations):
            if t > 0:
                c_rnn.reuse_variables()

            # Recurrent computation.
            output, state = rnn(rnn_in, state)

            # Select current conscious elements and conscious elements to
            # predict.
            (a, b) = select_conscious_elements(output, rep, is_train)
            
            current_elements.append(b)
            future_elements.append(a)
       
            # Current conscious state should be a (key, value)-tuple.
            # TODO(liamfedus):  Retrieve corresponding elements of the 
            # representation.  Something like a gather in this simple 
            # case: B = tf.gather(rep, b) of scalars.  Need to generalize
            # later.

        # TODO(liamfedus):  Return the outputs.
        return 
