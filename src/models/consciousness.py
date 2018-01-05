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


def select_conscious_elements(representation, c_rnn_out):
    """Function that produces the selected elements from the representation.""" 
    # TODO(liamfedus):What are principles that we want this attention mechanism
    # to follow? Use the processing of the conscious state RNN and the current 
    # representation to create a probability distribution over the unconscious 
    # elements.  Then sample particular elements.  Training through hard 
    # attention mechanisms may be done with methods like REBAR, RELAX, etc.
    pass


def consciousness(representations, is_train=True):
    """Consciousness (C) module.  This retrieves elements from the
    representations produced by R and produces a sparse conscious state."""
    
    rnn_dim = FLAGS.representation_dim
    
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

        for t, rep in enumerate(representations):
            if t > 0:
                c_rnn.reuse_variables()
        
            # Noise injected at each time-step.
            z_t = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dim],
                    dtype=tf.float32)
            
            # Input to C-RNN is a combination of the representation and the
            # noise.
            rnn_in = integrate_vectors(rep, z_t)    

            # Recurrent computation.
            output, state = rnn(rnn_in, state)

            # TODO(liamfedus): How do we want this computation to select 
            # particular elements from the representation?
            (a, b) = select_conscious_elements(output, rep)
            
            current_elements.append(b)
            future_elements.append(a)
       
            # Current conscious state should be a (key, value)-tuple.
            # TODO(liamfedus):  Retrieve corresponding elements of the 
            # representation.  Something like a gather in this simple 
            # case: B = tf.gather(rep, b) of scalars.  Need to generalize
            # later.

        # TODO(liamfedus):  Return the outputs.
        return 
