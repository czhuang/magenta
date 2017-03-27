from collections import defaultdict, OrderedDict
import os, sys, traceback
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import functools as ft
from scipy.misc import logsumexp

# TODO: don't import evaluate; it's a toplevel script
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, data_tools, util, evaluate

FLAGS = tf.app.flags.FLAGS

def main(argv):
  paths = argv[1:]

  # FIXME don't steal flags from elsewhere
  hparam_updates = {'use_pop_stats': FLAGS.use_pop_stats}
  wmodel = retrieve_model_tools.retrieve_model(
    model_name=FLAGS.model_name, hparam_updates=hparam_updates)
  hparams = wmodel.hparams
  assert hparams.use_pop_stats == FLAGS.use_pop_stats

  for path in paths:
    pianorolls = np.load(path)["pianorolls"]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = "eval_%s_%s_ensemble%s_chrono%s" % (
      timestamp, FLAGS.unit, FLAGS.ensemble_size, FLAGS.chronological)
    save_path = "%s__%s" % (path, name)

    if isinstance(pianorolls, np.ndarray):
      print pianorolls.shape
    print '# of total pieces in evaluation set:', len(pianorolls)
    lengths = [len(roll) for roll in pianorolls]
    if len(np.unique(lengths)) > 1:
      print 'lengths', np.sort(lengths)
    print 'max_len', max(lengths)
    print 'unique lengths', np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls))
    print 'shape', pianorolls[0].shape
  
    evaluator = evaluate.BaseEvaluator.make(FLAGS.unit, wmodel=wmodel, chronological=FLAGS.chronological)
    evaluator = evaluate.EnsemblingEvaluator(evaluator, FLAGS.ensemble_size)
    rval = evaluate.evaluate(evaluator, pianorolls)
  
    if True: # if this works then great!
      np.savez_compressed("%s.npz" % save_path, **rval)
    else:
      with gzip.open("%s.pkl.gz" % save_path, "wb") as file:
        pkl.dump(rval, file, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  tf.app.run()
