# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import os
from enum import Enum
from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary, pad_vocab_to_eight
from open_seq2seq.data.text2text.t2t import _read_and_batch_from_files
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import PAD_ID


class SyntheticTextDataLayer(DataLayer):
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
      'source_file': str,
      'src_vocab_file': str,
      'tgt_vocab_file': str,
      'max_length': int,
      'shuffle': bool,
      'repeat': bool,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
      'use_targets': bool,
      'delimiter': str,
      'target_file': str,
      'map_parallel_calls': int,
      'prefetch_buffer_size': int,
      'pad_lengths_to_eight': bool,
      'pad_vocab_to_eight': bool,
      'shuffle_buffer_size': int,
      'special_tokens_already_in_vocab': bool,
      'use_start_token': bool,
    })

  def __init__(self, params, model, num_workers=1, worker_id=0):
    super(SyntheticTextDataLayer, self).__init__(params, model,
                                                num_workers, worker_id)
    self._batch_size = self.params['batch_size']
    self.source_file = self.params['source_file']
    self._use_targets = self.params.get('use_targets', True)
    if not self._use_targets:
      self.target_file = self.source_file
      if 'target_file' in self.params:
        print("WARNING: target file was specified but was "
              "ignored by data layer because 'use_targets'=False")
    else:
      self.target_file = self.params['target_file']
    self.src_vocab_file = self.params['src_vocab_file']
    self.tgt_vocab_file = self.params['tgt_vocab_file']
    self.max_len = self.params['max_length']
    self._delimiter = self.params.get('delimiter', ' ')
    self._map_parallel_calls = self.params.get('map_parallel_calls', 8)
    self._pad_lengths_to_eight = self.params.get('pad_lengths_to_eight', False)
    self._prefetch_buffer_size = self.params.get('prefetch_buffer_size',
                                                 tf.contrib.data.AUTOTUNE)
    self._shuffle_buffer_size = self.params.get('shuffle_buffer_size', -1)
    self._num_workers = num_workers
    self._worker_id = worker_id
    self._use_start_token = self.params.get('use_start_token', True)
    if self._pad_lengths_to_eight and not (self.params['max_length'] % 8 == 0):
      raise ValueError("If padding to 8 in data layer, then "
                       "max_length should be multiple of 8")

    def file_len(fname):
      with open(fname) as f:
        for i, l in enumerate(f):
          pass
      return i + 1

    self.dataset_size = file_len(self.source_file)
    special_tokens_already_in_vocab = self.params.get('special_tokens_already_in_vocab', True)

    # load source and target vocabularies to RAM
    self.src_seq2idx = load_pre_existing_vocabulary(
      self.src_vocab_file, min_idx=0 if special_tokens_already_in_vocab
      else SpecialTextTokens.UNK_ID.value + 1)
    self.tgt_seq2idx = load_pre_existing_vocabulary(
      self.tgt_vocab_file, min_idx=0 if special_tokens_already_in_vocab
      else SpecialTextTokens.UNK_ID.value + 1)

    if not special_tokens_already_in_vocab:
      # manually add special tokens
      # unknown symbol
      self.src_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.UNK_ID.value)] = \
        SpecialTextTokens.UNK_ID.value
      self.tgt_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.UNK_ID.value)] = \
        SpecialTextTokens.UNK_ID.value
      # sentence start
      self.src_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.S_ID.value)] = \
        SpecialTextTokens.S_ID.value
      self.tgt_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.S_ID.value)] = \
        SpecialTextTokens.S_ID.value
      # sentence end
      self.src_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.EOS_ID.value)] = \
        SpecialTextTokens.EOS_ID.value
      self.tgt_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.EOS_ID.value)] = \
        SpecialTextTokens.EOS_ID.value
      # padding
      self.src_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.PAD_ID.value)] = \
        SpecialTextTokens.PAD_ID.value
      self.tgt_seq2idx[
        SpecialTextTokens.to_string(SpecialTextTokens.PAD_ID.value)] = \
        SpecialTextTokens.PAD_ID.value

    if self.params.get('pad_vocab_to_eight', False):
      self.src_seq2idx = pad_vocab_to_eight(self.src_seq2idx)
      self.tgt_seq2idx = pad_vocab_to_eight(self.tgt_seq2idx)

    self.src_idx2seq = {idx: w for w, idx in self.src_seq2idx.items()}
    self.tgt_idx2seq = {idx: w for w, idx in self.tgt_seq2idx.items()}

    self.params['src_vocab_size'] = len(self.src_seq2idx)
    self.params['tgt_vocab_size'] = len(self.tgt_seq2idx)
    self.params['target_seq2idx'] = self.tgt_seq2idx
    self.params['source_seq2idx'] = self.src_seq2idx
    self.params['target_idx2seq'] = self.tgt_idx2seq
    self.params['source_idx2seq'] = self.src_idx2seq

    self._input_tensors = {}

  def _pad2eight(self, lst, do_pad_eight):
    if len(lst) % 8 == 0 or not do_pad_eight:
      return lst
    else:
      return lst + [SpecialTextTokens.PAD_ID.value] * (8 - len(lst) % 8)

  def _src_token_to_id(self, line):
    tokens = line.decode("utf-8").split(self._delimiter)
    if self._use_start_token:
      return np.array(self._pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.src_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")
    else:
      return np.array(self._pad2eight([self.src_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in
                                       tokens[:self.max_len - 2]] + \
                                      [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

  def _tgt_token_to_id(self, line):
    tokens = line.decode("utf-8").split(self._delimiter)
    if self._use_start_token:
      return np.array(self._pad2eight([SpecialTextTokens.S_ID.value] + \
             [self.tgt_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens[:self.max_len-2]] + \
             [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")
    else:
      return np.array(self._pad2eight([self.tgt_seq2idx.get(token, SpecialTextTokens.UNK_ID.value) for token in
                                       tokens[:self.max_len - 2]] + \
                                      [SpecialTextTokens.EOS_ID.value], self._pad_lengths_to_eight), dtype="int32")

  def build_graph(self):
    """
    _sources = tf.data.TextLineDataset(self.source_file)\
      .map(lambda line: tf.py_func(func=self._src_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _targets = tf.data.TextLineDataset(self.target_file) \
      .map(lambda line: tf.py_func(func=self._tgt_token_to_id, inp=[line],
                                   Tout=[tf.int32], stateful=False),
           num_parallel_calls=self._map_parallel_calls) \
      .map(lambda tokens: (tokens, tf.size(tokens)),
           num_parallel_calls=self._map_parallel_calls)

    _src_tgt_dataset = tf.data.Dataset.zip((_sources, _targets)).filter(
      lambda t1, t2: tf.logical_and(tf.less_equal(t1[1], self.max_len),
                                    tf.less_equal(t2[1], self.max_len))
    ).cache()

    if self._num_workers > 1:
      _src_tgt_dataset = _src_tgt_dataset\
        .shard(num_shards=self._num_workers, index=self._worker_id)

    if self.params['shuffle']:
      bf_size = self.get_size_in_samples() if self._shuffle_buffer_size == -1 \
                                           else self._shuffle_buffer_size
      _src_tgt_dataset = _src_tgt_dataset.shuffle(buffer_size=bf_size)
    else:
      _src_tgt_dataset = _src_tgt_dataset

    if self.params['repeat']:
      _src_tgt_dataset = _src_tgt_dataset.repeat()

    self.batched_dataset = _src_tgt_dataset.padded_batch(
      self._batch_size,
      padded_shapes=((tf.TensorShape([None]),
                      tf.TensorShape([])),
                     (tf.TensorShape([None]),
                      tf.TensorShape([]))),
      padding_values=(
      (SpecialTextTokens.PAD_ID.value,
       0),
      (SpecialTextTokens.PAD_ID.value,
       0))).prefetch(buffer_size=self._prefetch_buffer_size)

    self._iterator = self.batched_dataset.make_initializable_iterator()

    if self.params['mode'] == 'train' or self.params['mode'] == 'eval':
      t1, t2 = self.iterator.get_next()
      x, x_length = t1[0], t1[1]
      y, y_length = t2[0], t2[1]
      self._input_tensors['source_tensors'] = [x, x_length]
      self._input_tensors['target_tensors'] = [y, y_length]
    else:
      t1, _ = self.iterator.get_next()
      self._input_tensors['source_tensors'] = [t1[0], t1[1]]
    """
    def generate_batch():
      avg_len = 30

      while 1:
        src_batch = np.zeros((self._batch_size, self.max_len), dtype=np.int)
        src_batch[:, 0] = SpecialTextTokens.S_ID.value
        src_batch[:, 1:avg_len + 1] = np.random.randint(low=4, high=len(self.src_seq2idx) - 1, size=(self._batch_size, avg_len))
        src_batch[:, avg_len + 1] = SpecialTextTokens.EOS_ID.value

        trg_batch = np.zeros((self._batch_size, self.max_len), dtype=np.int)
        trg_batch[:, 0] = SpecialTextTokens.S_ID.value
        trg_batch[:, 1:avg_len + 1] = np.random.randint(low=4, high=len(self.src_seq2idx) - 1, size=(self._batch_size, avg_len))
        trg_batch[:, avg_len + 1] = SpecialTextTokens.EOS_ID.value

        yield ((src_batch, [avg_len + 2] * self._batch_size), (trg_batch, [avg_len + 2] * self._batch_size))

    dataset = tf.data.Dataset.from_generator(
      generate_batch,
      ((tf.int32, tf.int32), (tf.int32, tf.int32)),
      ((tf.TensorShape([None, None]), tf.TensorShape([None])), (tf.TensorShape([None, None]), tf.TensorShape([None])))
    )
    self._iterator = dataset.make_initializable_iterator()
    ((x, x_length), (y, y_length)) = self.iterator.get_next()
    self._input_tensors['source_tensors'] = [x, x_length]
    self._input_tensors['target_tensors'] = [y, y_length]


  def create_interactive_placeholders(self):
    self._text = tf.placeholder(dtype=tf.int32, shape=[self._batch_size, None])
    self._text_length = tf.placeholder(dtype=tf.int32, shape=[self._batch_size])

    self._input_tensors = {}
    self._input_tensors['source_tensors'] = [self._text, self._text_length]

  def create_feed_dict(self, model_in):
    """ Creates the feed dict for interactive infer

    Args:
      model_in (str): the string to be translated. Should be in bpe format.

    Returns:
      feed_dict (dict): Dictionary with values for the placeholders.
    """
    text = self._src_token_to_id(model_in)
    text_length = text.shape[0]

    text = np.reshape(text, [self._batch_size, -1])
    text_length = np.reshape(text_length, [self._batch_size])

    feed_dict = {
        self._text: text,
        self._text_length: text_length
    }
    return feed_dict

  def get_size_in_samples(self):
    return self.dataset_size

  @property
  def iterator(self):
    return self._iterator

  @property
  def input_tensors(self):
    return self._input_tensors

