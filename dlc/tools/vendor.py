#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""This module includes some vendored patches or implementations from \
    external sources."""

import tensorflow as tf


def new_py_function(func, inp, Tout, name=None):
    # Taken from:
    # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    # See also:
    # https://github.com/tensorflow/tensorflow/issues/36278#issuecomment-781484858
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp,
                                                     flat_inp,
                                                     expand_composites=True)
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)

    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name,
    )
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec,
                                     Tout,
                                     expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out


def _dtype_to_tensor_spec(v):
    # Taken from:
    # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v


def _tensor_spec_to_dtype(v):
    # Taken from:
    # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    return v.dtype if isinstance(v, tf.TensorSpec) else v
