# ==================================================================
# Course: Understanding Deep Neural Networks
# Teacher: Zhang Yi
# Student: Wang Yaxuan
# ID:   2019141440341
#
# Lab 6 - Big Cat Recognition
# ====================================================================
import numpy as np

# define the activation function
def f(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid  function
df = lambda s: f(s) * (1 - f(s))


def bc(w, z, delta_next, beta):
    """
    :param w: shape [the Lth dim, the (L-1)th dim]
    :param z: shape [the (L-1)th dim, batch dimension]
    :param delta_next: [the Lth dim, batch dimension]
    :return: delta
    """
    # print("z_w.shape",w.shape)
    # print("z.shape",z.shape)
    # print("delta_next.shape",delta_next.shape)
    delta = (np.dot(w.T, delta_next) + beta) * df(z)
    return delta
