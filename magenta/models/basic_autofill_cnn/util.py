import contextlib, time
import numpy as np

def softmax(p, axis=None, temperature=1):
  if axis is None:
    axis = p.ndim - 1
  if temperature == 0.:
    # NOTE: may have multiple equal maxima, normalized below
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
    print "%.5f < %.5f < %.5f < %.5f < %.5g" % (np.min(p), np.percentile(p, 25), np.percentile(p, 50), np.percentile(p, 75), np.max(p))

    sampled = np.random.random(p.shape) < p
  return sampled

def sample_onehot(p, axis=None, temperature=1):
  if axis is None:
    axis = p.ndim - 1

  if temperature != 1:
    p = softmax(np.log(p), axis=axis, temperature=temperature)
  else:
    # callers depend on us taking care of normalization
    p /= p.sum(axis=axis, keepdims=True)

  # temporary transpose/reshape to matrix
  if axis != p.ndim - 1:
    permutation = list(range(0, axis)) + list(range(axis + 1, p.ndim)) + [axis]
    p = np.transpose(p, permutation)
  pshape = p.shape
  p = p.reshape([-1, p.shape[-1]])

  if not np.allclose(p.sum(axis=1), 1):
    k = (1 - np.isclose(p.sum(axis=1), 1)).sum()
    n = p.sum(axis=1).size
    maxdev = abs(p.sum(axis=1) - 1).max()
    print ("warning: %i/%i (%.2f) pmfs don't quite sum to 1; max deviation: %g"
           % (k, n, k * 1. / n, maxdev))

  # sample in a loop i guess -_-
  x = np.zeros(p.shape, dtype=np.float32)
  for i in range(p.shape[0]):
    x[i, np.random.choice(p.shape[1], p=p[i])] = 1.
  
  # transpose/reshape back
  x = x.reshape(pshape)
  if axis != x.ndim - 1:
    x = np.transpose(x, permutation)

  assert np.allclose(x.sum(axis=axis), 1)
  return x


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
def timing(label):
  print "enter %s" % label
  start_time = time.time()
  yield
  time_taken = (time.time() - start_time) / 60.0
  print "exit %s (%.2fmin)" % (label, time_taken)


# unobtrusive structured logging of arbitrary values
class Bamboo(object):
  def __init__(self):
    self.root = Scope(label="root", items=[], subsample_factor=1)
    self.stack = [self.root]

  @contextlib.contextmanager
  def scope(self, label, subsample_factor=None):
    new_scope = Scope(label, subsample_factor=subsample_factor)
    self.stack[-1].log(new_scope)
    self.stack.append(new_scope)
    yield
    self.stack.pop()

  def log(self, **kwargs):
    self.stack[-1].log(kwargs)

  class Scope(object):
    def __init__(self, label, subsample_factor=None):
      self.label = label
      self.subsample_factor = 1 if subsample_factor is None else subsample_factor
      self.items = []
      self.i = 0

    def log(self, x):
      # append or overwrite such that we retain every `subsample_factor`th value and the last value
      item = (self.i, x)
      if self.i % self.subsample_factor == 1 or not self.items:
        self.items.append(item)
      else:
        self.items[-1] = item
      self.i += 1
