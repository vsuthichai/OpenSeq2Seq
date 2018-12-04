# This code is heavily based on the code from MLPerf
# https://github.com/mlperf/reference/tree/master/translation/tensorflow/transformer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
import os
from .layer_norm_fused_layer import layer_norm_custom

#dir_path = os.path.dirname(os.path.realpath(__file__))
#loading the custom op library
#custom_module = tf.load_op_library(os.path.join(dir_path, 'layer_norm_fused_op.so'))

#This line is needed so TensorFlow can infer the shape of the output.
#This may not be required (may even raise error) if you are using newer version of TensorFlow.
#tf.RegisterShape("LayerNormCustom")(common_shapes.call_cpp_shape_fn)

#register gradients for auto-differentiation.
#@ops.RegisterGradient("LayerNormCustom")
#def _LayerNormCustomGrad(op, grad):
    #return [custom_module.layer_norm_backprop_custom(
            #op.inputs[0], grad, op.get_attr("epsilon"))]

#input_shape = [32,512,128]
#inputs = tf.random_normal(input_shape)
#normalized_output = custom_module.layer_norm_custom(inputs, epsilon=1e-6)

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
    dtype = x.dtype
    x = tf.cast(x=x, dtype=tf.float32)

    '''
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    result = norm_x * self.scale + self.bias
    z = tf.cast(x=result, dtype=dtype)
    #return z
    '''

    '''
    x1 = tf.expand_dims(x, axis=2)
    y1 = tf.layers.batch_normalization(
      inputs=x1,
      #inputs=tf.cast(x1, dtype=tf.float32),
      axis=-1,
      momentum=0.0,
      #epsilon=epsilon,
      #epsilon=0.001,
      center=True,
      scale=True,
      training=False,
      trainable=False,
    )
    y2 = tf.squeeze(y1, axis=[2])
    y2 = tf.cast(y2, dtype=dtype)
    '''

    a1 = layer_norm_custom(
        inputs=x, 
        center=True,
        scale=True,
        activation_fn=None,
        trainable=True,
        #epsilon=epsilon,
        #begin_norm_axis=-1
    )
    a1 = tf.cast(a1, dtype=dtype)
    return a1

    '''
    #x = tf.cast(x=x, dtype=tf.float32)
    result = tf.contrib.layers.layer_norm(
      inputs=x,
      center=True,
      scale=True,
      activation_fn=None,
      trainable=False,
      begin_norm_axis=-1
    )
    z1 = tf.cast(x=result, dtype=dtype)
    z1 = tf.Print(z1, [tf.shape(a1), a1, tf.shape(z1), z1, tf.shape(z), z])
    return z1
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
