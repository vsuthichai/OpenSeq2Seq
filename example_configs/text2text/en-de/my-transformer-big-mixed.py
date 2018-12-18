# pylint: skip-file
from __future__ import absolute_import, division, print_function
from open_seq2seq.models import Text2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.data.text2text.text2text_vs import TransformerDataLayer
from open_seq2seq.data.text2text.text2text_vs import ParallelTextDataLayer
from open_seq2seq.data.text2text.text2text_synthetic import SyntheticTextDataLayer
from open_seq2seq.data.text2text.text2text_padded import PaddedParallelTextDataLayer
from open_seq2seq.losses import PaddedCrossEntropyLossWithSmoothing
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import EOS_ID
from open_seq2seq.optimizers.lr_policies import transformer_policy
import tensorflow as tf

"""
This configuration file describes a variant of Transformer model from
https://arxiv.org/abs/1706.03762
"""

base_model = Text2Text
d_model = 1024
num_layers = 6

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
data_root = "scripts/fb_wmt16_en_de_bpe32k/"
#data_root = "scripts/alldata_en_dt/"

base_params = {
  "use_horovod": True,
  "num_gpus": 1, # when using Horovod we set number of workers with params to mpirun
  "batch_size_per_gpu": 256,  # this size is in sentence pairs, reduce it if you get OOM
  #"max_steps": 100000,
  "save_summaries_steps": 25,
  "print_loss_steps": 1,
  "print_samples_steps": None,
  "eval_steps": 100,
  "save_checkpoint_steps": 10000,
  "logdir": "Transformer-FP32-H-256",
  #"dtype": tf.float32, # to enable mixed precision, comment this line and uncomment two below lines
  "dtype": "mixed",
  "loss_scaling": "Backoff",
  "iter_size": 1,
  #"max_grad_norm": 1.0,
  "num_epochs": 30,

  "summaries": [
    "learning_rate",
    #"gradients",
    "gradient_norm",
    "global_gradient_norm",
    "variables",
    "variable_norm",
    "larc_summaries",
    "loss_scale"
  ],

  "optimizer": tf.contrib.opt.LazyAdamOptimizer,
  "optimizer_params": {
    "beta1": 0.9,
    "beta2": 0.98,
    "epsilon": 1e-09,
  },

  "lr_policy": transformer_policy,
  "lr_policy_params": {
    "learning_rate": 2.0,
    #"warmup_steps": 8000,
    #"warmup_steps": 30,
    "warmup_steps": 4000,
    "d_model": d_model,
  },

  "encoder": TransformerEncoder,
  "encoder_params": {
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    #"num_heads": 8,
    "num_heads": 16,
    "attention_dropout": 0.1,
    "filter_size": 4 * d_model,
    "relu_dropout": 0.3,
    #"layer_postprocess_dropout": 0.1,
    "layer_postprocess_dropout": 0.3,
    "pad_embeddings_2_eight": True,
    "remove_padding": False,
  },

  "decoder": TransformerDecoder,
  "decoder_params": {
    "layer_postprocess_dropout": 0.3,
    "num_hidden_layers": num_layers,
    "hidden_size": d_model,
    #"num_heads": 8,
    "num_heads": 16,
    "attention_dropout": 0.1,
    "relu_dropout": 0.3,
    "filter_size": 4 * d_model,
    "beam_size": 4,
    "alpha": 0.6,
    "extra_decode_length": 50,
    "EOS_ID": EOS_ID,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },

  "loss": PaddedCrossEntropyLossWithSmoothing,
  "loss_params": {
    "label_smoothing": 0.1,
  }
}

'''
train_params = {
  #"data_layer": SyntheticTextDataLayer,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "pad_vocab_to_eight": True,
    "src_vocab_file": data_root + "m_common.vocab",
    "tgt_vocab_file": data_root + "m_common.vocab",
    #"source_file": data_root + "train.clean.en.shuffled.BPE_common.32K.tok",
    "source_file": data_root + "train.en-de.en",
    #"target_file": data_root + "train.clean.de.shuffled.BPE_common.32K.tok",
    "target_file": data_root + "train.en-de.de",
    "delimiter": " ",
    "shuffle": True,
    #"shuffle_buffer_size": 25000,
    "shuffle_buffer_size": 1024,
    "repeat": True,
    "map_parallel_calls": 64,
    "max_length": 56,
  }
}
'''

train_params = {
  "data_layer": TransformerDataLayer,
  "data_layer_params": {
    "data_dir": data_root + "tfrecord/",
    "file_pattern": "wmt16-en-de*",
    "src_vocab_file": data_root + "m_common.vocab",
    "tgt_vocab_file": data_root + "m_common.vocab",
    "batch_size": 3584,
    "max_length": 256,
    "shuffle": True,
    "delimiter": ' ',
    "batch_in_tokens": True,
    "num_cpu_cores": 16,
    "repeat": 30,
    'pad_data_to_eight': True,
  },
}

eval_params = {
  "batch_size_per_gpu": 64,
  #"data_layer": TransformerDataLayer,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"m_common.vocab",
    "tgt_vocab_file": data_root+"m_common.vocab",
    #"source_file": data_root+"wmt13-en-de.src.BPE_common.32K.tok",
    "source_file": data_root+"valid.en-de.en",
    #"target_file": data_root+"wmt13-en-de.ref.BPE_common.32K.tok",
    "target_file": data_root+"valid.en-de.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
    },
}

infer_params = {
  "batch_size_per_gpu": 128,
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": data_root+"m_common.vocab",
    "tgt_vocab_file": data_root+"m_common.vocab",
    #"source_file": data_root+"wmt14-full-en-de.src.BPE_common.32K.tok",
    "source_file": data_root+"test.en-de.en",
    #"target_file": data_root+"wmt14-full-en-de.src.BPE_common.32K.tok",
    "target_file": data_root+"test.en-de.de",
    "delimiter": " ",
    "shuffle": False,
    "repeat": False,
    "max_length": 256,
  },
}
