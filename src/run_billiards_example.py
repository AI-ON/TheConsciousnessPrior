"""Billiards Example."""
from __future__ import print_function
from builtins import range

import matplotlib.pyplot as plt

from environments.billiards import Billiards

env = Billiards()


env.make_frames(200)

frames = env.frames

plt.figure()
for t in range(frames.shape[0]):
    plt.imshow(frames[t], cmap="gray")
    plt.show()
