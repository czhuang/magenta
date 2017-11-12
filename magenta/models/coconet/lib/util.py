  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib, time, os, numbers, yaml
from datetime import datetime
import numpy as np


def sample_bernoulli(p, temperature):
  B, T, P, I = p.shape
  assert I == 1
  if temperature == 0.:
    sampled = p > 0.5
  else:
    axis = 3
    pp = np.concatenate((p, (1-p)), axis=3)
    logpp = np.log(pp)
    logpp /= temperature
    logpp -= logpp.max(axis=axis, keepdims=True)
    #p = np.where(logpp > 0, 
    #             1 / (1 + np.exp(-logpp)), 
    #             np.exp(logpp) / (np.exp(logpp) + 1))
    p = np.exp(logpp)
    p /= p.sum(axis=axis, keepdims=True)
    p = p[:, :, :, :1]
    print("%.5f < %.5f < %.5f < %.5f < %.5g" % (np.min(p), np.percentile(p, 25), np.percentile(p, 50), np.percentile(p, 75), np.max(p)))

    sampled = np.random.random(p.shape) < p
  return sampled

def softmax(p, axis=None, temperature=1):
  if axis is None:
    axis = p.ndim - 1
  if temperature == 0.:
    # NOTE: in case of multiple equal maxima, returns uniform distribution over them
    p = p == np.max(p, axis=axis, keepdims=True)
  else:
    oldp = p
    logp = np.log(p)
    logp /= temperature
    logp -= logp.max(axis=axis, keepdims=True)
    p = np.exp(logp)
  p /= p.sum(axis=axis, keepdims=True)
  if np.isnan(p).any():
    import pdb; pdb.set_trace()
  return p

def sample(p, axis=None, temperature=1, onehot=False):
  assert (p >= 0).all() # just making sure we don't put log probabilities in here

  if axis is None:
    axis = p.ndim - 1

  if temperature != 1:
    p = p ** (1. / temperature)
  cmf = p.cumsum(axis=axis)
  totalmasses = cmf[tuple(slice(None) if d != axis else slice(-1, None) for d in range(cmf.ndim))]
  u = np.random.random([p.shape[d] if d != axis else 1 for d in range(p.ndim)])
  i = np.argmax(u * totalmasses < cmf, axis=axis)

  return to_onehot(i, axis=axis, depth=p.shape[axis]) if onehot else i

def to_onehot(i, depth, axis=None):
  if axis is None:
    axis = i.ndim
  x = np.eye(depth)[i]
  if axis != i.ndim:
    # move new axis forward
    axes = list(range(i.ndim))
    axes.insert(axis, i.ndim)
    x = np.transpose(x, axes)
  assert np.allclose(x.sum(axis=axis), 1)
  return x

def sample_onehot(p, axis=None, temperature=1):
  return sample(p, axis=axis, temperature=temperature, onehot=True)

def deepsubclasses(klass):
  for subklass in klass.__subclasses__():
    yield subklass
    for subsubklass in deepsubclasses(subklass):
      yield subsubklass

class Factory(object):
  @classmethod
  def make(klass, key, *args, **kwargs):
    for subklass in deepsubclasses(klass):
      if subklass.key == key:
        return subklass(*args, **kwargs)
    else:
      raise KeyError("unknown %s subclass key %s" % (klass, key))


@contextlib.contextmanager
def timing(label, printon=True):
  if printon:
    print("enter %s" % label)
  start_time = time.time()
  yield
  time_taken = (time.time() - start_time) / 60.0
  if printon:
    print("exit %s (%.2fmin)" % (label, time_taken))


class AggregateMean(object):
  """Aggregates values for mean."""

  def __init__(self, name):
    self.name = name
    self.value = 0.
    self.total_counts = 0

  def add(self, value, counts=1):
    """Add an amount to the total and also increment the counts."""
    self.value += value
    self.total_counts += counts

  @property
  def mean(self):
    """Return the mean."""
    return self.value / self.total_counts

def timestamp():
  return datetime.now().strftime("%Y%m%d%H%M%S")

def get_rng(rng=None):
  if rng is None:
    return np.random
  if isinstance(rng, numbers.Integral):
    return np.random.RandomState(rng)
  else:
    return rng

@contextlib.contextmanager
def numpy_seed(seed):
  if seed is not None:
    prev_rng_state = np.random.get_state()
    np.random.seed(seed)
  yield
  if seed is not None:
    np.random.set_state(prev_rng_state)

def load_hparams(checkpoint_path):
  hparams_fpath = os.path.join(os.path.dirname(checkpoint_path), 'config')
  with open(hparams_fpath, 'r') as p:
    hparams = yaml.load(p)
  return hparams

def random_crop(x, length):
  leeway = len(x) - length
  start = np.random.randint(1 + max(0, leeway))
  x = x[start:start + length]
  return x


def batches(*xss, **kwargs):
  """Iterate over subsets of lists of examples

  Yields batches of the form `[xs[indices] for xs in xss]` where at each
  iteration `indices` selects a subset. Each index is only selected once.

  Args:
    *xss: lists of elements to batch
    size: number of elements per batch
    discard_remainder: if true, discard final short batch
    shuffle: if true, yield examples in randomly determined order
    shuffle_rng: seed or rng to determine random order

  Yields:
    A batch of the same structure as `xss`, but with `size` examples.
  """
  size = kwargs.get("size", 1)
  discard_remainder = kwargs.get("discard_remainder", True)
  shuffle = kwargs.get("shuffle", False)
  shuffle_rng = kwargs.get("shuffle_rng", None)

  shuffle_rng = get_rng(shuffle_rng)
  n = int(np.unique(list(map(len, xss))))
  assert all(len(xs) == len(xss[0]) for xs in xss)
  indices = np.arange(len(xss[0]))
  if shuffle:
    np.random.shuffle(indices)
  for start in range(0, n, size):
    batch_indices = indices[start:start + size]
    if len(batch_indices) < size and discard_remainder:
      break
    batch_xss = [xs[batch_indices] for xs in xss]
    yield batch_xss

def pad_and_stack(*xss):
  """Pad and stack lists of examples.

  Each argument `xss[i]` is taken to be a list of variable-length examples.
  The examples are padded to a common length and stacked into an array.
  Example lengths must match across the `xss[i]`.

  Args:
    *xss: lists of examples to stack

  Returns:
    A tuple `(yss, lengths)`. `yss` is a list of arrays of padded examples,
    each `yss[i]` corresponding to `xss[i]`. `lengths` is an array of example
    lengths.
  """
  yss = []
  lengths = list(map(len, xss[0]))
  for xs in xss:
    # example lengths must be the same across arguments
    assert lengths == list(map(len, xs))
    max_length = max(lengths)
    rest_shape = xs[0].shape[1:]
    ys = np.zeros((len(xs), max_length) + rest_shape, dtype=xs[0].dtype)
    for i in range(len(xs)):
      ys[i, :len(xs[i])] = xs[i]
    yss.append(ys)
  return list(map(np.asarray, yss)), np.asarray(lengths)

def identity(x):
  return x

def eqzip(*xss):
  xss = list(map(list, xss))
  lengths = list(map(len, xss))
  if not all(length == lengths[0] for length in lengths):
    raise ValueError("eqzip got iterables of unequal lengths %s" % lengths)
  return zip(*xss)

