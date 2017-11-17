"""Script to evaluate a dataset fold under a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from magenta.models.coconet import lib_evaluation
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_data
from magenta.models.coconet import lib_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', None,
                           'Path to the base directory for different datasets.')
tf.app.flags.DEFINE_string('fold', None, 'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string('fold_index', None, 'optionally, index of particular data point in fold to evaluate')
tf.app.flags.DEFINE_string('unit', None, 'note or frame or example')
tf.app.flags.DEFINE_integer('ensemble_size', 5, 'number of ensemble members to average')
tf.app.flags.DEFINE_bool('chronological', False, 'indicates evaluation should proceed in chronological order')
tf.app.flags.DEFINE_string('checkpoint', None, 'path to checkpoint file')

def main(argv):
  wmodel = lib_graph.load_checkpoint(FLAGS.checkpoint)

  evaluator = lib_evaluation.BaseEvaluator.make(FLAGS.unit, wmodel=wmodel,
                                                chronological=FLAGS.chronological)
  evaluator = lib_evaluation.EnsemblingEvaluator(evaluator, FLAGS.ensemble_size)

  # tf.app.flags parses known flags but passes unknown flags on. unfortunately
  # this gives us no way to tell whether the leftovers were meant to be flags
  # or positional arguments, as tf.app.flags might have consumed a `--` arg.
  for arg in argv[1:]:
    if arg.startswith("-") and "logtostderr" not in arg:
      raise ValueError("unknown flag: %s" % arg)
      break
    if "logtostderr" in arg:
      script_arg_idx = 2
    else:
      script_arg_idx = 1
  paths = argv[script_arg_idx:]
  if bool(paths) == bool(FLAGS.fold is not None):
    raise ValueError("Either --fold must be specified, or paths of npz files to load must be given, but not both.")
  if FLAGS.fold is not None:
    evaluate_fold(FLAGS.fold, evaluator, wmodel.hparams)
  if paths:
    evaluate_paths(paths, evaluator, wmodel.hparams)
  print ('Done')

def evaluate_fold(fold, evaluator, hparams):
  name = "eval_%s_%s%s_%s_ensemble%s_chrono%s" % (
    lib_util.timestamp(), fold, FLAGS.fold_index if FLAGS.fold_index is not None else "",
    FLAGS.unit, FLAGS.ensemble_size, FLAGS.chronological)
  save_path = "%s__%s" % (FLAGS.checkpoint, name)

  pianorolls = get_fold_pianorolls(fold, hparams)

  if False: # inspect data
    for pianoroll in pianorolls:
      min_pitch = pianoroll.argmax(axis=1).min()
      max_pitch = pianoroll.argmax(axis=1).max()
      pianoroll = pianoroll[:, min_pitch:max_pitch+1]
      T, P, I = pianoroll.shape
      lines = [[" " for _ in range(T)] for _ in range(P)]
      for t, p, i in np.transpose(np.nonzero(pianoroll)):
        lines[p][t] = "SATB"[i]
      print("yay:")
      print("\n".join("".join(line) for line in reversed(lines)))
      input()

  rval = lib_evaluation.evaluate(evaluator, pianorolls)
  np.savez_compressed("%s.npz" % save_path, **rval)

def evaluate_paths(paths, evaluator, hparams):
  for path in paths:
    name = "eval_%s_%s_ensemble%s_chrono%s" % (
      lib_util.timestamp(), FLAGS.unit, FLAGS.ensemble_size, FLAGS.chronological)
    save_path = "%s__%s" % (path, name)

    pianorolls = get_path_pianorolls(path)
    rval = lib_evaluation.evaluate(evaluator, pianorolls)
    np.savez_compressed("%s.npz" % save_path, **rval)

def get_fold_pianorolls(fold, hparams):
  dataset = lib_data.get_dataset(FLAGS.data_dir, hparams, fold)
  pianorolls = dataset.get_pianorolls()
  print('\nRetrieving pianorolls from %s set of %s dataset.\n' % (
      fold, hparams.dataset))
  print_statistics(pianorolls)
  if FLAGS.fold_index is not None:
    pianorolls = [pianorolls[int(FLAGS.fold_index)]]
  return pianorolls

def get_path_pianorolls(path):
  pianorolls = np.load(path)
  if isinstance(pianorolls, np.ndarray):
    print(pianorolls.shape)
  print_statistics(pianorolls)
  return pianorolls

def print_statistics(pianorolls):
  if isinstance(pianorolls, np.ndarray):
    print(pianorolls.shape)
  print('# of total pieces in evaluation set:', len(pianorolls))
  lengths = [len(roll) for roll in pianorolls]
  if len(np.unique(lengths)) > 1:
    print('lengths', np.sort(lengths))
  print('max_len', max(lengths))
  print('unique lengths', np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls)))
  print('shape', pianorolls[0].shape)

if __name__ == '__main__':
  tf.app.run()
