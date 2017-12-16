import tensorflow as tf

from models.representation import *
from environments.billiards import Billiards

flags = tf.app.flags

flags.DEFINE_string("model_dir", "/tmp/cp", "Directory for Tensorboard" 
                    "summaries and videos.")
flags.DEFINE_string("env", "Billiards", "Name of environment. One of"
                    " ['Billiards.].")
flags.DEFINE_string("rnn_type", "lstm", "Choice of RNN")
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('training_steps', 2000, 'Number of steps to train model.')
flags.DEFINE_integer('time_steps', 5, 'Number of time steps to unroll model')
flags.DEFINE_integer('representation_dim', 128, 'Number of units in hidden units in RNN layer.')
flags.DEFINE_integer('image_dim', 128, 'Dimension of the image frames.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.  '
                         'Must divide evenly into the dataset sizes.')

FLAGS = flags.FLAGS

def create_env():
    if FLAGS.env == 'Billiards':
        return Billiards(T=FLAGS.time_steps)
    # TODO: Consider gym environments eventually.
    # env = gym.envs.make(FLAGS.env)


def generate_frames(env):
    """Generate frames from the environment"""
    env.make_frames(FLAGS.image_dim)
    frames = env.frames
    return frames    


def create_model():
    """Create the model."""
    # Global step.
    global_step = tf.Variable(0, name="global_step", trainable=False)
   
    # Placeholder.
    inputs = tf.placeholder(shape=[None, FLAGS.time_steps, FLAGS.image_dim,
        FLAGS.image_dim, 1], dtype=tf.float32, name="X")
    
    # Representation RNN.
    representations = representation(inputs, is_train=True, rnn_type=FLAGS.rnn_type)
    
    # Model containing the modules.
    model = {'inputs': inputs, 
            'R_RNN':  representations, 
            'global_step': global_step}
    print(model['R_RNN'].get_shape())
    return model


def create_loss(model):
    """Create the losses."""
    pass


def create_optimizer(loss):
    """Create the optimzer."""
    pass


def create_graph():
    """Create computational graph."""
    model = create_model()
    loss = create_loss(model)
    optimizer = create_optimizer(loss)

    
def train_model(frames, model):
    """Train the model."""
    pass    


if __name__=="__main__":
    env = create_env() 
    frames = generate_frames(env)
    create_graph()

