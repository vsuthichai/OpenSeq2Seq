# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import os
import time
import sys

import numpy as np
import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, log_summaries_from_dict, \
                                     get_results_for_epoch

class PrintTensorHook(tf.train.SessionRunHook):
  """Session hook that prints training samples and prediction from time to time
  """
  def __init__(self, every_steps, model, tensor_name):
    super(PrintTensorHook, self).__init__()
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps)
    self._iter_count = 0
    self._global_step = None
    self._model = model
    # using only first GPU
    #output_tensors = model.get_output_tensors(0)
    #self._fetches = [
        #model.get_data_layer(0).input_tensors,
        #output_tensors,
    #]
    self.tensor_name = tensor_name

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()

  def before_run(self, run_context):
    session = run_context.session
    graph = session.graph
    tensor = graph.get_tensor_by_name(self.tensor_name)

    print(tensor)

    if self._timer.should_trigger_for_step(self._iter_count):
      return tf.train.SessionRunArgs([[tensor], self._global_step])
    return tf.train.SessionRunArgs([[], self._global_step])

  def after_run(self, run_context, run_values):
    results, step = run_values.results
    self._iter_count = step

    if not results:
      return
    self._timer.update_last_triggered_step(self._iter_count - 1)

    tensor_val = results[0]
    #np.set_printoptions(precision=20, floatmode='fixed')
    #print(type(tensor_val))
    #print(tensor_val.shape)
    #print(tensor_val.dtype)
    #print(tensor_val)
    np.save("/tmp/1-tf", tensor_val)
    sys.exit(0)

