"""Moving MNIST.

This class produces a batch of moving MNIST experiments
considered in the ICML2015 paper:
    Unsupervised Learning of Video Representations using LSTMs.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
from builtins import range

import numpy as np


def default_motion_fn(data_batch, data_pairs, config=None):
    """A default motion function for moving MNIST.

    Parameters
    ----------
    data_batch : numpy.ndarray
        the background data batch
        (batch_size x bg_size x bg_size x time_steps)
    data_pairs : numpy.ndarray
        the numpy array that takes the digits pairs
        (batch_size x num_digits x 28 x 28)
    config : dictionary
        configuration dictionary for extra configs
    """
    # facts about the target data
    batch_size = data_batch.shape[0]
    bg_size = data_batch.shape[1]
    time_steps = data_batch.shape[3]
    num_digits = data_pairs.shape[1]
    digit_shape = data_pairs.shape[2]
    bound_size = bg_size-digit_shape+1

    # load config
    # we use top left to coordinate the position
    x_vel = config["x_vel"]  # velocity range for x direction
    y_vel = config["y_vel"]  # velocity range for y direction
    # make sure you chose some reasonable range
    x_init = config["x_init"]  # range for sample initial x position
    y_init = config["y_init"]  # range for sample initial y position

    # the most intense way, looping everything
    for sample_idx in range(batch_size):
        # sample initial position (num_digits, 2)
        curr_pos = np.vstack((np.random.choice(y_init, num_digits),
                              np.random.choice(x_init, num_digits))).T
        # sample initial velocity
        vels = np.vstack((np.random.choice(y_vel*2, num_digits)-y_vel,
                          np.random.choice(x_vel*2, num_digits)-x_vel)).T

        # process every step
        for step in range(time_steps):
            # set the image at current position
            for num_idx in range(num_digits):
                data_batch[
                    sample_idx,
                    curr_pos[num_idx, 0]:curr_pos[num_idx, 0]+digit_shape,
                    curr_pos[num_idx, 1]:curr_pos[num_idx, 1]+digit_shape,
                    step] += data_pairs[sample_idx, num_idx]
            data_batch[sample_idx] = np.clip(data_batch[sample_idx], 0, 255)

            # update current position based on the velocity
            # while consider boundary issues
            for num_idx in range(num_digits):
                # consider x direction
                x_pos = curr_pos[num_idx, 0]+vels[num_idx, 0]
                y_pos = curr_pos[num_idx, 1]+vels[num_idx, 1]
                # free space at x direction
                if x_pos >= 0 and x_pos <= bound_size-1:
                    curr_pos[num_idx, 0] = x_pos
                elif x_pos < 0:
                    curr_pos[num_idx, 0] = -x_pos
                    vels[num_idx, 0] = -vels[num_idx, 0]
                elif x_pos > bound_size-1:
                    curr_pos[num_idx, 0] = 2*bound_size-x_pos-2
                    vels[num_idx, 0] = -vels[num_idx, 0]
                # free space at y direction
                if y_pos >= 0 and y_pos <= bound_size-1:
                    curr_pos[num_idx, 1] = y_pos
                elif y_pos < 0:
                    curr_pos[num_idx, 1] = -y_pos
                    vels[num_idx, 1] = -vels[num_idx, 1]
                elif y_pos > bound_size-1:
                    curr_pos[num_idx, 1] = 2*bound_size-y_pos-2
                    vels[num_idx, 1] = -vels[num_idx, 1]

    return data_batch


class MovingMNIST(object):
    "The moving MNIST generator."""
    def __init__(self, data_path, batch_size=64,
                 num_digits=2, time_steps=40,
                 motion_fn=None, motion_fn_config=None, bg_size=64):
        """Generate a batch of moving MNIST samples.

        The output of the class is a 4-D tensor:
            batch_size x bg_size x bg_size x num_steps

        Parameters
        ----------
        batch_size : int
            number of samples in this batch
        num_digits : int
            number of digits in the scene
        time_steps : int
            simulation steps
        motion_fn : function
            A function that controls a digit's movement.
        motion_fn_config : dictionary
            A dictionary that contains custom motion function configuration.
        bg_size : int
            the background size is bg_size x bg_size
        """
        self.batch_size = batch_size
        self.num_digits = num_digits
        self.time_steps = time_steps
        self.motion_fn = motion_fn
        self.motion_fn_config = motion_fn_config
        self.bg_size = bg_size
        self.data_path = data_path

        # load the dataset
        self.train_data, self.test_data = self.load_mnist()
        self.num_train_data = self.train_data[0].shape[0]
        self.num_test_data = self.test_data[0].shape[0]

    def load_mnist(self):
        """Load MNIST.

        We adopted Keras copy of MNSIT, the download address is here:
            https://s3.amazonaws.com/img-datasets/mnist.npz

        Download the dataset and put it somewhere in your file system.
        """
        f = np.load(self.data_path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()

        return (x_train, y_train), (x_test, y_test)

    def get_batch(self):
        """Get a batch of moving MNIST data."""
        # prepare the batch background
        data_batch = np.zeros(
            (self.batch_size, self.bg_size, self.bg_size,
             self.time_steps), dtype=np.uint8)

        # generate pairs of digits
        # make sure we have enough unique pairs
        assert self.num_digits*self.batch_size < self.num_train_data
        num_pairs = np.random.choice(
            self.num_train_data,
            self.num_digits*self.batch_size,
            replace=False)
        data_pairs = self.train_data[0][num_pairs].reshape(
            self.batch_size, self.num_digits, 28, 28)

        return self.motion_fn(data_batch, data_pairs,
                              self.motion_fn_config)
