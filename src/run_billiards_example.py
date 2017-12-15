"""Billiards Example."""
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from environments.billiards import Billiards

resolution = 128
time_steps = 25

env = Billiards(T=time_steps)
env.make_frames(resolution)

frames = env.frames
num_frames = frames.shape[0]

fig = plt.figure()

def update(t):
    plt.imshow(frames[t], cmap="gray")

if __name__=="__main__":
    # FuncAnimation will call the 'update' function for each frame
    anim = FuncAnimation(fig, update, frames=time_steps, interval=200)
    anim.save('billiards.gif',writer='imagemagick');
