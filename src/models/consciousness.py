import tensorflow as tf
import utils.attention_utils as att

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


def generate_keys_to_predict(conscious_state):
    """Generate the keys of future conscious states to predict.  As an initial
    implementation, we will generate the keys from the current conscious state.
    
    Args:
        conscious_state:  Current conscious state.
            
    Returns:
        keys_to_predict: Keys to predict.
    """
    # Assume that we predict the same number of conscious elements.
    k = FLAGS.num_conscious_elements    
    con_dim = FLAGS.consciousness_dim

    with tf.variable_scope("pred_keys") as pred_keys:
        w_pred = tf.get_variable("w_pred",
                                 [con_dim, con_dim],
                                 tf.float32)

        # Generate 
        logits = tf.matmul(conscious_state, w_pred)

        # Extract the indices to predict.
        _, keys_to_predict = tf.nn.top_k(logits, k)
        return keys_to_predict


def select_conscious_elements(representation, is_train, noise_epsilon=1e-2):
    """Function that produces the selected elements from the representation 
    Select the top-k elements in a sparse and noisy manner.  We employ the tools
    of Outrageously Large Neural Networks (https://arxiv.org/pdf/1701.06538.pdf)
    which were used for selecting the top-k experts.
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py
    
    Args:

    Returns:

    """
    k = FLAGS.num_conscious_elements
    con_dim = FLAGS.consciousness_dim

    with tf.variable_scope("noisy_selection") as sel: 
        w_gate = tf.get_variable("w_gate", 
                [FLAGS.representation_dim, con_dim], 
                tf.float32)
        w_noise = tf.get_variable("w_noise", 
                [FLAGS.representation_dim, con_dim], 
                tf.float32)

        # Selection mechanism. 
        logits = tf.matmul(representation, w_gate)

        # Add noise to the element selection.
        raw_noise_stddev = tf.matmul(representation, w_noise)
        noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + noise_epsilon) *
                tf.to_float(is_train))

        # Noisy logits.
        noisy_logits = logits + (tf.random_normal(tf.shape(logits)) * noise_stddev) 
        
        # Select the top-k elements.
        top_logits, top_indices = tf.nn.top_k(noisy_logits, k)

        top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
        top_k_gates = tf.nn.softmax(top_k_logits)
    
        # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in
        # the positions corresponding to all but the k selected conscious 
        # elements.
        conscious_state = att._rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, con_dim)
        
        # Future keys to predict given the current conscious state.
        fut_keys = generate_keys_to_predict(conscious_state)
        return conscious_state, fut_keys


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
        conscious_states = []
        future_elements = []

        for t, rnn_in in enumerate(representations):
            if t > 0:
                c_rnn.reuse_variables()

            # Recurrent computation.
            output, state = rnn(rnn_in, state)

            # Select current conscious state (B,b) and future conscious elements to
            # predict (A).
            (conscious_state, to_predict_keys) = select_conscious_elements(output, is_train)
            
            conscious_states.append(conscious_state)
            future_elements.append(to_predict_keys)
      
        conscious_states = tf.stack(conscious_states, axis=1)
        future_elements = tf.stack(future_elements, axis=1)
        return conscious_states, future_elements
