"""Utilities that depend on Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

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
      print("ResourceExhaustedError on batch of %s elements, lowering max size to %s" % (pianoroll.size, self.maxsize))
      return self.bisect(pianoroll, mask)

  def bisect(self, pianoroll, mask):
    i = int(len(pianoroll) / 2)
    if i == 0:
      raise ValueError('Batch size is zero!')
    return np.concatenate([self(pianoroll[:i], mask[:i]),
                           self(pianoroll[i:], mask[i:])],
                          axis=0)

class WrappedModel(object):
  """A Wrapper for passing model related and other configs as one object."""

  def __init__(self, model, graph, hparams):
    self.model = model
    self.graph = graph
    self.hparams = hparams
