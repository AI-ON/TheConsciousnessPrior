"""An experimental Billiards environment.

The original code of this environment is at:
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar

We modified the above code and use as a toy case
to test the consciousness prior.
"""
from __future__ import print_function
from builtins import range

import numpy as np


class Billiards(object):
    """The Billiards class."""
    def __init__(self, T=128, n=2, r=None, m=None, eps=0.5, SIZE=10):
        """The Billiards Environment.

        # Parameters
            T: int
                the total time of the billiards movie
            n: int
                The number of balls? (to check)
            r: float

            m: float

            eps: float

            SIZE: int
                the size of bounding box is SIZE x SIZE
        """
        self.T = T
        self.SIZE = SIZE

        self.n = n
        self.r = np.array([1.2]*self.n) if r is None else r
        self.m = np.array([1]*self.n) if m is None else m

        self.X = np.zeros((self.T, self.n, 2), dtype=np.float)
        self.v = np.random.randn(self.n, 2)
        self.vnorm = self.norm(self.v)
        self.v /= self.vnorm*0.5

        self.x = self.config()

        self.eps = eps

        self.make_steps()

    def config(self):
        """Do configuration."""
        good_config = False
        while not good_config:
            x = 2+np.random.rand(self.n, 2)*8
            good_config = True
            for i in range(self.n):
                for z in range(2):
                    if x[i][z]-self.r[i] < 0:
                        good_config = False
                    if x[i][z]+self.r[i] > self.SIZE:
                        good_config = False

            # that's the main part.
            for i in range(self.n):
                for j in range(i):
                    if self.norm(x[i]-x[j]) < self.r[i]+self.r[j]:
                        good_config = False

        return x

    def norm(self, x):
        """Normalize input array.

        # Parameters
            x: numpy.ndarray
                the input array
        # Returns
            normed_x: float
                the normalized amplitude
        """
        return np.sqrt((x**2).sum())

    def new_speeds(self, m1, m2, v1, v2):
        new_v2 = (2*m1*v1+v2*(m2-m1))/(m1+m2)
        new_v1 = new_v2+(v2-v1)
        return new_v1, new_v2

    def make_steps(self):
        """Fill up X."""
        for t in range(self.T):
            for i in range(self.n):
                self.X[t, i] = self.x[i]

            for mu in range(int(1/self.eps)):
                for i in range(self.n):
                    self.x[i] += self.eps*self.v[i]

                for i in range(self.n):
                    for z in range(2):
                        if self.x[i][z]-self.r[i] < 0:  # want positive
                            self.v[i][z] = np.abs(self.v[i][z])
                        if self.x[i][z]+self.r[i] > self.SIZE:  # want negative
                            self.v[i][z] = -np.abs(self.v[i][z])

                for i in range(self.n):
                    for j in range(i):
                        if self.norm(
                                self.x[i]-self.x[j]) < self.r[i]+self.r[j]:
                            # the bouncing off part:
                            w = self.x[i]-self.x[j]
                            w = w/self.norm(w)

                            v_i = np.dot(w.transpose(), self.v[i])
                            v_j = np.dot(w.transpose(), self.v[j])

                            new_v_i, new_v_j = self.new_speeds(
                                self.m[i], self.m[j], v_i, v_j)

                            self.v[i] += w*(new_v_i - v_i)
                            self.v[j] += w*(new_v_j - v_j)

    def ar(self, x, y, z):
        return z/2+np.arange(x, y, z, dtype=np.float)

    def make_frames(self, res):
        """Make frames from X."""
        self.frames = np.zeros((self.T, res, res), dtype=np.float)

        [I, J] = np.meshgrid(self.ar(0, 1, 1./res)*self.SIZE,
                             self.ar(0, 1, 1./res)*self.SIZE)

        for t in range(self.T):
            for i in range(self.n):
                self.frames[t] += np.exp(
                    -(((I-self.X[t, i, 0])**2 +
                       (J-self.X[t, i, 1])**2)/(self.r[i]**2))**4)

            self.frames[t][self.frames[t] > 1] = 1

    def vectorize_frames(self, frames):
        """Vectorize frames."""
        return frames.reshap(self.T, frames.shape[1]**2)
