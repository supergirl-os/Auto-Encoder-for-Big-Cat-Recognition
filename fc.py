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


def fc(w, a):
    """
    :param w: shape [the Lth dim, the (L-1)th dim]
    :param a: shape [feature dim, batch dimension]
    :return: a_next, z_next
    """

    # % forward computing( in either vector form)
    # calculate net input
    z_next = np.dot(w, a)
    # calculate activation
    a_next = f(z_next)
    return a_next, z_next





