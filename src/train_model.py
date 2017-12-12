import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('time_steps', 5, 'Number of time steps to unroll model')
flags.DEFINE_integer('representation_dim', 128, 'Number of units in hidden units in RNN layer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.  '
                         'Must divide evenly into the dataset sizes.')

def create_graph():
    """Create computational graph."""
    pass


if __name__=="__main__":
    model = create_graph()
