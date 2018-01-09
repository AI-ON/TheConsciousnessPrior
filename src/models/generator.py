import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def predict_values(gen_state, keys_to_predict):
    """Predict the future values for the given keys.
    
    Args:
        gen_state:  Output of the Generator RNN, [batch_size, 
            consciousness_dim].  
        keys_to_predict:  Keys to predict for the future conscious states, 
            [batch_size, k] where k is the number of num_conscious_elements

    Returns:
        pred_values: Values for the selected elements, [batch_size,
            consciousness_dim] 
    """
    # Convert the keys to a one-hot representation
    keys_one_hot = tf.one_hot(keys_to_predict, 
                              depth=FLAGS.consciousness_dim,
                              dtype=tf.float32)
    
    pred_values = tf.multiply(gen_state, keys_one_hot)
    return pred_values


def generator(conscious_states, keys_to_predict, is_train=True):
    """Predict the values of provided elements."""
   
    # Maintain the same dimension as the consciousness RNN.
    rnn_dim = FLAGS.consciousness_dim

    # Choose RNN based on the type.
    if (FLAGS.rnn_type == "lstm"):
        rnn = tf.contrib.rnn.BasicLSTMCell(rnn_dim)
    elif(FLAGS.rnn_type == "gru"):
        rnn = tf.contrib.rnn.GRUCell(rnn_dim)
    else:
        raise NotImplementedError

    # Unstack
    conscious_states = tf.unstack(conscious_states, axis=1)
    keys_to_predict = tf.unstack(keys_to_predict, axis=1)

    # Initial state for generator RNN.
    initial_state = state = rnn.zero_state(FLAGS.batch_size, dtype=tf.float32)

    with tf.variable_scope('generator') as g_rnn:
        # Generator module is responsible for predicting future conscious
        # states.
        conscious_state_predictions = []

        for t, (rnn_in, keys)  in enumerate(zip(conscious_states, keys_to_predict)):
            if t > 0:
                g_rnn.reuse_variables()

            # Recurrent computation.
            output, state = rnn(rnn_in, state)

            # Generate predictions.
            conscious_prediction = predict_values(output, keys)

            # Append to list.
            conscious_state_predictions.append(conscious_prediction)
        
        conscious_state_predictions = tf.stack(conscious_state_predictions,
                axis=1)
        return conscious_state_predictions 
