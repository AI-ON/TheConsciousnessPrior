import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def generator(indices, is_train=True):
    """Predict the values of provided elements."""
    # TODO(liamfedus):  We seek the ability to predict future conscious states.
    # The Generator is passed a key corresponding to an element and 
    # predcits the value in the future.  With such limited context, this seems
    # like a poor construction, at the very least this should probably be 
    # recurrent.  Broader context beyond just the elements to predict may be 
    # necessary and/or valuable.
    pass
