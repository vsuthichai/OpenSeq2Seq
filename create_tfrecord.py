import tensorflow as tf
import argparse
import io
from enum import Enum

class SpecialTextTokens(Enum):
  PAD_ID = 0  # special padding token
  EOS_ID = 1  # special end of sentence token
  S_ID = 2  # special start of sentence token
  UNK_ID = 3  # out-of-vocabulary tokens will map there
  OUT_OF_BUCKET = 1234567890
  END_OF_CHOICE = -100

def token_to_id(vocab_dict, line, delimiter):
    tokens = line.split(delimiter)
    #print(tokens)
    return [SpecialTextTokens.S_ID.value] + \
           [vocab_dict.get(token, SpecialTextTokens.UNK_ID.value) for token in tokens] + \
           [SpecialTextTokens.EOS_ID.value]

def load_pre_existing_vocabulary(path, min_idx=0, read_chars=False):
  """Loads pre-existing vocabulary into memory.
  The vocabulary file should contain a token on each line with optional
  token count on the same line that will be ignored. Example::
    a 1234
    b 4321
    c 32342
    d
    e
    word 234
  Args:
    path (str): path to vocabulary.
    min_idx (int, optional): minimum id to assign for a token.
    read_chars (bool, optional): whether to read only the
        first symbol of the line.
  Returns:
     dict: vocabulary dictionary mapping tokens (chars/words) to int ids.
  """
  idx = min_idx
  vocab_dict = {}
  with io.open(path, newline='', encoding='utf-8') as f:
    for line in f:
      # ignoring empty lines
      if not line or line == '\n':
        continue
      if read_chars:
        token = line[0]
      else:
        token = line.rstrip().split('\t')[0]
      vocab_dict[token] = idx
      idx += 1
  return vocab_dict

def textline_generator(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line.strip()

def create_tfrecords(src_file, trg_file, vocab_dict):
    src_generator = textline_generator(src_file)
    ref_generator = textline_generator(trg_file)
    #samples_per_file = 45010
    # ~/github/OpenSeq2Seq/scripts/alldata_en_dt/tfrecord
    samples_per_file = 17676 #256 files
    samples_per_file = 8838 #512 files

    writer = None
    file_count = 0
    for i, sample in enumerate(zip(src_generator, ref_generator)):
        src_sample = token_to_id(vocab_dict, sample[0], ' ')
        ref_sample = token_to_id(vocab_dict, sample[1], ' ')
        feature = {
            'inputs': tf.train.Feature(int64_list=tf.train.Int64List(value=src_sample)),
            'targets': tf.train.Feature(int64_list=tf.train.Int64List(value=ref_sample))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        if writer == None:
            writer = tf.python_io.TFRecordWriter("/home/ubuntu/github/OpenSeq2Seq/scripts/alldata_en_dt/tfrecord/wmt16-en-de.%03d" % file_count)
            #writer = tf.python_io.TFRecordWriter("/home/ubuntu/github/OpenSeq2Seq/scripts/fb_wmt16_en_de_bpe32k/tfrecord/wmt16-en-de.%03d" % file_count)
            file_count += 1
        elif i % samples_per_file == 0:
            writer.close()
            writer = tf.python_io.TFRecordWriter("/home/ubuntu/github/OpenSeq2Seq/scripts/alldata_en_dt/tfrecord/wmt16-en-de.%03d" % file_count)
            #writer = tf.python_io.TFRecordWriter("/home/ubuntu/github/OpenSeq2Seq/scripts/fb_wmt16_en_de_bpe32k/tfrecord/wmt16-en-de.%03d" % file_count)
            file_count += 1

        writer.write(example.SerializeToString())

    if writer != None:
        writer.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', help="vocab_file", type=str)
    parser.add_argument('--src_file', help="source language dataset", type=str, required=True)
    parser.add_argument('--ref_file', help="reference language dataset", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    vocab_dict = load_pre_existing_vocabulary(args.vocab_file)
    create_tfrecords(args.src_file, args.ref_file, vocab_dict)

if __name__ == '__main__':
    main()

