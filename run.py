# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import contextlib
import os
import sys
import tensorflow as tf
from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
                                     create_logdir, create_model
from open_seq2seq.utils import train, infer, evaluate

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

def init_horovod(base_config, hvd_size):
  # Initilize Horovod
  if base_config['use_horovod']:
    import horovod.tensorflow as hvd
    hvd.init(comm=list(range(hvd_size)))
    if hvd.rank() == 0:
      deco_print("Using horovod")
  else:
    hvd = None

  return hvd

def start_coordinator(comm, size, iter_size):

  SLEEP_TIME = 0.0005  # in seconds

  count_limit = iter_size*size

  counter = 0
  notified = [0 for _ in range(size)]

  while True:
    for worker in range(size):
      if comm.Iprobe(source=worker):
        comm.recv(source=worker)    # message is dummy so no need for return value
  
        if counter >= count_limit - size:
          comm.send({'msg':'sync'}, dest=worker)   
          assert(notified[worker]==0)
          notified[worker] = 1 
          counter += 1

          if counter == count_limit:
            assert(sum(notified)==size)
            counter = 0
            notified = [0 for _ in range(size)]
            break
        else:
          # logic to determine duration based on counter && count_limit && iter_size
          if counter <= count_limit*1/2:
            duration = 4
          elif counter <= count_limit*3/4:
            duration = 2
          else:
            duration = 1

          comm.send({'msg':'continue', 'ctr': duration}, dest=worker)   
          counter += 1

    time.sleep(SLEEP_TIME)


 
def main(args, base_config, base_model, config_model, hvd):
  restore_best_checkpoint = base_config.get('restore_best_checkpoint', False)

  # Check logdir and create it if necessary
  if hvd is None or hvd.rank() == 0:
    checkpoint = check_logdir(args, base_config, restore_best_checkpoint)
  if args.enable_logs:
    if hvd is None or hvd.rank() == 0:
      old_stdout, old_stderr, stdout_log, stderr_log = create_logdir(
          args,
          base_config
      )
    base_config['logdir'] = os.path.join(base_config['logdir'], 'logs')

  if args.mode == 'train' or args.mode == 'train_eval' or args.benchmark:
    if hvd is None or hvd.rank() == 0:
      if checkpoint is None or args.benchmark:
        deco_print("Starting training from scratch")
      else:
        deco_print(
            "Restored checkpoint from {}. Resuming training".format(checkpoint),
        )
  elif args.mode == 'eval' or args.mode == 'infer':
    if hvd is None or hvd.rank() == 0:
      deco_print("Loading model from {}".format(checkpoint))

  # Create model and train/eval/infer
  with tf.Graph().as_default():
    model = create_model(args, base_config, config_module, base_model, hvd)
    if args.mode == "train_eval":
      train(model[0], model[1], debug_port=args.debug_port)
    elif args.mode == "train":
      train(model, None, debug_port=args.debug_port)
    elif args.mode == "eval":
      evaluate(model, checkpoint)
    elif args.mode == "infer":
      infer(model, checkpoint, args.infer_output_file, args.use_trt)

  if args.enable_logs and (hvd is None or hvd.rank() == 0):
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    stdout_log.close()
    stderr_log.close()

@contextlib.contextmanager
def profile_context(profile, hvd):
  if profile:
    with tf.contrib.tfprof.ProfileContext(
        "os2s_{}".format(hvd.rank()), trace_steps=range(100, 120), dump_steps=range(100, 120)) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling("op", opts, range(100, 120))
      #pctx.add_auto_profiling("scope", opts, [15])
      yield
  else:
    yield

if __name__ == '__main__':
  mpi_rank = comm.Get_rank()
  mpi_size = comm.Get_size()

  # Parse args and create config
  args, base_config, base_model, config_module = get_base_config(sys.argv[1:])

  if mpi_rank == mpi_size - 1: 
    start_coordinator(comm=comm, size=mpi_size-1, iter_size=base_config['iter_size'])
    sys.exit(0)

  if args.mode == "interactive_infer":
    raise ValueError(
        "Interactive infer is meant to be run from an IPython",
        "notebook not from run.py."
    )

  hvd = init_horovod(base_config, mpi_size-1)
  with profile_context(args.profile, hvd):
    main(args, base_config, base_model, config_module, hvd)

