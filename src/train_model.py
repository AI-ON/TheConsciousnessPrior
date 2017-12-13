import tensorflow as tf

from models.representation import *

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('training_steps', 2000, 'Number of steps to train model.')
flags.DEFINE_integer('time_steps', 5, 'Number of time steps to unroll model')
flags.DEFINE_integer('representation_dim', 128, 'Number of units in hidden units in RNN layer.')
flags.DEFINE_integer('image_dim', 128, 'Dimension of the image frames.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.  '
                         'Must divide evenly into the dataset sizes.')

FLAGS = flags.FLAGS

def create_model():
    """Create the model."""
    # Dummy image for testing shape.  Our model will operate over a video
    # sequence.
    images = tf.random_normal(shape=[FLAGS.batch_size,
                                    FLAGS.time_steps,
                                    FLAGS.image_dim,
                                    FLAGS.image_dim, 3])
    
    representations = representation(images, is_train=True)

    model = representations
    print(model.get_shape())
    return model


def create_loss():
    pass


def create_optimizer():
    pass


def create_graph():
    """Create computational graph."""
    model = create_model()
    

def train_model():
    """Train the model."""
    pass


if __name__=="__main__":
    create_graph()

