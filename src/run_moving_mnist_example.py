"""Moving MNIST Example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from environments.moving_mnist import MovingMNIST
from environments.moving_mnist import default_motion_fn

# create moving MNIST environment
env = MovingMNIST(
        "./mnist.npz",
        motion_fn=default_motion_fn,
        motion_fn_config={"x_vel": 10, "y_vel": 10, "x_init": 100,
                          "y_init": 100},
        batch_size=2,
        bg_size=128,
        num_digits=5,
        time_steps=100)

# get a batch of data
data_batch = env.get_batch()

# visualise first sample
fig = plt.figure()


def update(t):
    plt.imshow(data_batch[0, :, :, t], cmap="gray")


anim = FuncAnimation(fig, update, frames=100, interval=100)
anim.save("moving_mnist.gif", writer="imagemagick")
