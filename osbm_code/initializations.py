import tensorflow as tf
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def matrix_weight_variable_truncated_normal(dim, name=""):

    initial = 0.01*tf.truncated_normal((dim, dim), mean=0., stddev=1., dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zerodiag_matrix_weight_variable_truncated_normal(dim, name=""):

    initial = 0.01*tf.truncated_normal((dim, dim), mean=0., stddev=1., dtype=tf.float32)
    initial = tf.linalg.set_diag(initial, tf.zeros(dim))
    return tf.Variable(initial, name=name)
