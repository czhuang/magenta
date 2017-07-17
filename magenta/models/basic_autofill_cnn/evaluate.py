"""Script to evaluate a dataset fold under a model."""
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

import lib.evaluation as evaluation
import retrieve_model_tools
import data_tools
import util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('fold', None, 'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string('fold_index', None, 'optionally, index of particular data point in fold to evaluate')
tf.app.flags.DEFINE_string('unit', None, 'note or frame or example')
tf.app.flags.DEFINE_integer('ensemble_size', 5, 'number of ensemble members to average')
tf.app.flags.DEFINE_bool('chronological', False, 'indicates evaluation should proceed in chronological order')

def main(argv):
  # FIXME don't steal flags from elsewhere
  hparam_updates = {'use_pop_stats': FLAGS.use_pop_stats}
  wmodel = retrieve_model_tools.retrieve_model(
    model_name=FLAGS.model_name, hparam_updates=hparam_updates)
  hparams = wmodel.hparams
  assert hparams.use_pop_stats == FLAGS.use_pop_stats

  evaluator = evaluation.BaseEvaluator.make(FLAGS.unit, wmodel=wmodel,
                                            chronological=FLAGS.chronological)
  evaluator = evaluation.EnsemblingEvaluator(evaluator, FLAGS.ensemble_size)

  paths = argv[1:]
  if bool(paths) == bool(FLAGS.fold is not None):
    raise ValueError("Either --fold must be specified, or paths of npz files to load must be given.")
  if FLAGS.fold is not None:
    evaluate_fold(evaluator, FLAGS.fold)
  if paths:
    evaluate_paths(evaluator, paths)

def evaluate_fold(evaluator):
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  name = "eval_%s_%s%s_%s_ensemble%s_chrono%s" % (
    timestamp, fold, FLAGS.fold_index if FLAGS.fold_index is not None else "",
    FLAGS.unit, FLAGS.ensemble_size, FLAGS.chronological)
  save_path = os.path.join(FLAGS.checkpoint_dir, name)

  print 'model_name', hparams.model_name
  print hparams.checkpoint_fpath

  pianorolls = get_fold_pianorolls(fold)
  rval = evaluation.evaluate(evaluator, pianorolls)
  np.savez_compressed("%s.npz" % save_path, **rval)

def evaluate_paths(evaluator):
  for path in paths:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = "eval_%s_%s_ensemble%s_chrono%s" % (
      timestamp, FLAGS.unit, FLAGS.ensemble_size, FLAGS.chronological)
    save_path = "%s__%s" % (path, name)

    pianorolls = get_path_pianoroll(path)
    rval = evaluation.evaluate(evaluator, pianorolls)
    np.savez_compressed("%s.npz" % save_path, **rval)

def get_fold_pianorolls(fold):
  pianorolls = data_tools.get_data_as_pianorolls(FLAGS.data_dir, hparams, fold)
  print '\nRetrieving pianorolls from %s set of %s dataset.\n' % (
      fold, hparams.dataset)
  print_statistics(pianorolls)
  if FLAGS.fold_index is not None:
    pianorolls = [pianorolls[int(FLAGS.fold_index)]]
  return pianorolls

def get_path_pianorolls(path):
  pianorolls = np.load(path)["pianorolls"]
  if isinstance(pianorolls, np.ndarray):
    print pianorolls.shape
  print_statistics(pianorolls)
  return pianorolls

def print_statistics(pianorolls):
  if isinstance(pianorolls, np.ndarray):
    print pianorolls.shape
  print '# of total pieces in evaluation set:', len(pianorolls)
  lengths = [len(roll) for roll in pianorolls]
  if len(np.unique(lengths)) > 1:
    print 'lengths', np.sort(lengths)
  print 'max_len', max(lengths)
  print 'unique lengths', np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls))
  print 'shape', pianorolls[0].shape

if __name__ == '__main__':
  tf.app.run()
