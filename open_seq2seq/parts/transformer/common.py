# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Define defaults for parameters

class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer(dtype=tf.float32),
                                 dtype=tf.float32)
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer(dtype=tf.float32),
                                dtype=tf.float32)
    self.built = True

  def call(self, x, epsilon=1e-6):
    '''
    dtype = x.dtype
    x = tf.cast(x=x, dtype=tf.float32)
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    result = norm_x * self.scale + self.bias
    z = tf.cast(x=result, dtype=dtype)
    #print(z)
    return z
    '''

    #x1 = tf.expand_dims(x, axis=2)
    dtype = x.dtype
    y1 = tf.layers.batch_normalization(
      #center=True,
      #scale=True,
      inputs=tf.cast(x, dtype=tf.float32),
      #inputs=x1,
      training=True,
      axis=-1,
      #momentum=0.95,
      #epsilon=epsilon,
      #epsilon=0.001,
    )
    #y2 = tf.squeeze(y1, axis=[2])
    y2 = tf.cast(y1, dtype=dtype)
    #print(y2)
    return y2

    '''
    return tf.contrib.layers.layer_norm(
      inputs=x,
      center=True,
      scale=True,
      activation_fn=None,
      trainable=True,
      begin_norm_axis=-1
    )
    '''

class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params, train):
    self.layer = layer
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    self.train = train

    # Create normalization layer
    self.layer_norm = LayerNormalization(params["hidden_size"])

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y
