'''Evaluates trained model on given split of dataset.'''

from collections import defaultdict, OrderedDict
import os, sys, traceback
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import functools as ft
from scipy.misc import logsumexp

import mask_tools
import retrieve_model_tools
import data_tools
import util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('fold', None, 'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string('index', None, 'optionally, index of particular data point in fold to evaluate')
tf.app.flags.DEFINE_string('unit', None, 'note or frame or example')
tf.app.flags.DEFINE_integer('ensemble_size', 5, 'number of ensemble members to average')
tf.app.flags.DEFINE_bool('chronological', False, 'indicates evaluation should proceed in chronological order')

def main(unused_argv):
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  name = "eval_%s_%s%s_%s_ensemble%s_chrono%s" % (
    timestamp, FLAGS.fold, FLAGS.index if FLAGS.index is not None else "", FLAGS.unit, FLAGS.ensemble_size, FLAGS.chronological)
  save_path = os.path.join(FLAGS.checkpoint_dir, name)

  hparam_updates = {'use_pop_stats': FLAGS.use_pop_stats}
  wmodel = retrieve_model_tools.retrieve_model(
    model_name=FLAGS.model_name, hparam_updates=hparam_updates)
  hparams = wmodel.hparams
  assert hparams.use_pop_stats == FLAGS.use_pop_stats

  print 'model_name', hparams.model_name
  print hparams.checkpoint_fpath

  # TODO option to get pianorolls from command line arguments
  pianorolls = data_tools.get_data_as_pianorolls(FLAGS.data_dir, hparams, FLAGS.fold)
  print '\nRetrieving pianorolls from %s set of %s dataset.\n' % (
      FLAGS.fold, hparams.dataset)

  if isinstance(pianorolls, np.ndarray):
    print pianorolls.shape
  print '# of total pieces in evaluation set:', len(pianorolls)
  lengths = [len(roll) for roll in pianorolls]
  if len(np.unique(lengths)) > 1:
    print 'lengths', np.sort(lengths)
  print 'max_len', max(lengths)
  print 'unique lengths', np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls))
  print 'shape', pianorolls[0].shape

  if FLAGS.index is not None:
    pianorolls = [pianorolls[int(FLAGS.index)]]

  evaluator = BaseEvaluator.make(FLAGS.unit, wmodel=wmodel, chronological=FLAGS.chronological)
  evaluator = EnsemblingEvaluator(evaluator, FLAGS.ensemble_size)
  rval = evaluate(evaluator, pianorolls)

  if True: # if this works then great!
    np.savez_compressed("%s.npz" % save_path, **rval)
  else:
    with gzip.open("%s.pkl.gz" % save_path, "wb") as file:
      pkl.dump(rval, file, protocol=pkl.HIGHEST_PROTOCOL)

def evaluate(evaluator, pianorolls):
  example_losses = []
  unit_losses = []

  for pi, pianoroll in enumerate(pianorolls):
    start_time = time.time()

    unit_loss = -evaluator(pianoroll)
    example_loss = np.mean(unit_loss)

    example_losses.append(example_loss)
    unit_losses.append(unit_loss)

    duration = (time.time() - start_time) / 60.
    report(unit_loss, prefix="%i %5.2fmin " % (pi, duration))

    if np.isinf(example_loss):
      break

  report(example_losses, prefix="FINAL example-level ")
  report(unit_losses, prefix="FINAL unit-level ")

  rval = dict(example_losses=example_losses,
              unit_losses=unit_losses)
  rval.update(("example_%s" % k, v) for k, v in stats(example_losses).items())
  rval.update(("unit_%s" % k, v) for k, v in stats(flatcat(unit_losses)).items())
  return rval

def report(losses, prefix=""):
  print "%s loss %s" % (prefix, statstr(flatcat(losses)))

def stats(x):
  return dict(mean=np.mean(x), sem=np.std(x) / np.sqrt(len(x)),
              min=np.min(x), max=np.max(x),
              q1=np.percentile(x, 25), q2=np.percentile(x, 50), q3=np.percentile(x, 75))

def statstr(x):
  return "mean/sem: {mean:8.5f}+-{sem:8.5f} {min:.5f} < {q1:.5f} < {q2:.5f} < {q3:.5f} < {max:.5g}".format(**stats(x))

def flatcat(xs):
  return np.concatenate([x.flatten() for x in xs])

class BaseEvaluator(util.Factory):
  def __init__(self, wmodel, chronological):
    self.wmodel = wmodel
    self.chronological = chronological

    def predictor(xs, masks):
      input_data = [mask_tools.apply_mask_and_stack(x, mask) for x, mask in zip(xs, masks)]
      p = self.wmodel.sess.run(self.wmodel.model.predictions,
                               feed_dict={self.wmodel.model.input_data: input_data})
      return p
    self.predictor = RobustPredictor(predictor)

  @property
  def separate_instruments(self):
    return self.wmodel.hparams.separate_instruments

  def update_lls(self, lls, x, pxhat, t, d):
    # The code below assumes x is binary, so instead of x * log(px) which is inconveniently NaN
    # if both x and log(px) are zero, we can use where(x, log(px), 0). If x were not binary, we
    # would have to multiply by it.
    assert np.array_equal(x, x.astype(bool))
    index = ((np.arange(x.shape[0]), t, slice(None), d) if self.separate_instruments else
             (np.arange(x.shape[0]), t, d, slice(None)))
    lls[t, d] = np.log(np.where(x[index], pxhat[index], 1)).sum(axis=1)

class FrameEvaluator(BaseEvaluator):
  key = "frame"

  def __call__(self, pianoroll):
    T, P, I = pianoroll.shape
    assert self.separate_instruments or I == 1
    D = I if self.separate_instruments else P

    # compile a batch with each frame being an example
    B = T
    xs = np.tile(pianoroll[None], [B, 1, 1, 1])

    ts, ds = self.draw_ordering(T, D)

    # set up sequence of masks to predict the first (according to ordering) instrument for each frame
    mask = []
    mask_scratch = np.ones([T, P, I], dtype=np.float32)
    for j, (t, d) in enumerate(zip(ts, ds)):
      # when time rolls over, reveal the entire current frame for purposes of predicting the next one
      if j % D != 0:
        continue
      mask.append(mask_scratch.copy())
      mask_scratch[t, :, :] = 0
    assert np.allclose(mask_scratch, 0)
    del mask_scratch
    mask = np.array(mask)

    lls = np.zeros([T, D], dtype=np.float32)

    # we can't parallelize within the frame, as we need the predictions of some of the other
    # instruments. Hence we outer loop over the instruments and parallelize across frames.
    xs_scratch = xs.copy()
    for d_idx in range(D):
      # call out to the model to get predictions for the first instrument at each time step
      pxhats = self.predictor(xs_scratch, mask)

      t, d = ts[d_idx::D], ds[d_idx::D]
      assert len(t) == B and len(d) == B

      # write in predictions and update mask
      if self.separate_instruments:
        xs_scratch[np.arange(B), t, :, d] = np.eye(P)[np.argmax(pxhats[np.arange(B), t, :, d], axis=1)]
        mask[np.arange(B), t, :, d] = 0
        # every example in the batch sees one frame more than the previous
        assert np.allclose((1 - mask).sum(axis=(1, 2, 3)),
                           [(k * D + d_idx + 1) * P for k in range(mask.shape[0])])
      else:
        xs_scratch[np.arange(B), t, d, :] = pxhats[np.arange(B), t, d, :] > 0.5
        mask[np.arange(B), t, d, :] = 0
        # every example in the batch sees one frame more than the previous
        assert np.allclose((1 - mask).sum(axis=(1, 2, 3)),
                           [(k * D + d_idx + 1) * I for k in range(mask.shape[0])])

      self.update_lls(lls, xs, pxhats, t, d)

    # conjunction over notes within frames; frame is the unit of prediction
    return lls.sum(axis=1)

  def draw_ordering(self, T, D):
    o = np.arange(T, dtype=np.int32)
    if not self.chronological:
      np.random.shuffle(o)
    # random variable orderings within each time step
    o = o[:, None] * D + np.arange(D, dtype=np.int32)[None, :]
    for t in range(T):
      np.random.shuffle(o[t])
    o = o.reshape([T * D])
    ts, ds = np.unravel_index(o.T, dims=(T, D))
    return ts, ds

class NoteEvaluator(BaseEvaluator):
  key = "note"

  def __call__(self, pianoroll):
    T, P, I = pianoroll.shape
    assert self.separate_instruments or I == 1
    D = I if self.separate_instruments else P
  
    # compile a batch with an example for each variable
    B = T * D
    xs = np.tile(pianoroll[None], [B, 1, 1, 1])
  
    ts, ds = self.draw_ordering(T, D)
    assert len(ts) == B and len(ds) == B
  
    # set up sequence of masks, one for each variable
    mask = []
    mask_scratch = np.ones([T, P, I], dtype=np.float32)
    for j, (t, d) in enumerate(zip(ts, ds)):
      mask.append(mask_scratch.copy())
      if self.separate_instruments:
        mask_scratch[t, :, d] = 0
      else:
        mask_scratch[t, d, :] = 0
    assert np.allclose(mask_scratch, 0)
    del mask_scratch
    mask = np.array(mask)
    
    pxhats = self.predictor(xs, mask)

    lls = np.zeros([T, D], dtype=np.float32)
    self.update_lls(lls, xs, pxhats, ts, ds)
    return lls

  def draw_ordering(self, T, D):
    o = np.arange(T * D, dtype=np.int32)
    if not self.chronological:
      np.random.shuffle(o)
    ts, ds = np.unravel_index(o.T, dims=(T, D))
    return ts, ds

class ExampleEvaluator(BaseEvaluator):
  key = "example"

  def __call__(self, pianoroll):
    varwise_lls = self.notewise_evaluator(pianoroll)
    # conjunction across variables
    return np.sum(varwise_lls)

class EnsemblingEvaluator(BaseEvaluator):
  key = "_ensembling"

  def __init__(self, evaluator, ensemble_size):
    self.evaluator = evaluator
    self.ensemble_size = ensemble_size

  def __call__(self, pianoroll):
    lls = [self.evaluator(pianoroll) for _ in range(self.ensemble_size)]
    return logsumexp(lls, b=1. / len(lls), axis=0)

# adapts batch size in response to ResourceExhaustedErrors
class RobustPredictor(object):
  def __init__(self, predictor):
    self.predictor = predictor
    self.maxsize = None
    self.factor = 2

  def __call__(self, pianoroll, mask):
    if self.maxsize is not None and pianoroll.size > self.maxsize:
      return self.bisect(pianoroll, mask)
    try:
      return self.predictor(pianoroll, mask)
    except tf.errors.ResourceExhaustedError:
      if self.maxsize is None:
        self.maxsize = pianoroll.size
      self.maxsize = int(self.maxsize / self.factor)
      print "ResourceExhaustedError on batch of %s elements, lowering max size to %s" % (pianoroll.size, self.maxsize)
      return self.bisect(pianoroll, mask)

  def bisect(self, pianoroll, mask):
    i = int(len(pianoroll) / 2)
    if i == 0:
      raise ValueError('Batch size is zero!')
    return np.concatenate([self(pianoroll[:i], mask[:i]),
                           self(pianoroll[i:], mask[i:])],
                          axis=0)

if __name__ == '__main__':
  tf.app.run()
