import contextlib, time, os
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
    print "%.5f < %.5f < %.5f < %.5f < %.5g" % (np.min(p), np.percentile(p, 25), np.percentile(p, 50), np.percentile(p, 75), np.max(p))

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
    print "enter %s" % label
  start_time = time.time()
  yield
  time_taken = (time.time() - start_time) / 60.0
  if printon:
    print "exit %s (%.2fmin)" % (label, time_taken)


# unobtrusive structured logging of arbitrary values
class Bamboo(object):
  def __init__(self):
    self.root = BambooScope("root", subsample_factor=1)
    self.stack = [self.root]

  @contextlib.contextmanager
  def scope(self, label, subsample_factor=None):
    new_scope = BambooScope(label, subsample_factor=subsample_factor)
    self.stack[-1].log(new_scope)
    self.stack.append(new_scope)
    yield
    self.stack.pop()

  def log(self, **kwargs):
    self.stack[-1].log(kwargs)

  def dump(self, path):
    dikt = {}
    def _compile_npz_dict(item, path):
      i, node = item
      if isinstance(node, BambooScope):
        for subitem in node.items:
          _compile_npz_dict(subitem, os.path.join(path, "%s_%s" % (i, node.label)))
      else:
        for k, v in node.items():
          dikt[os.path.join(path, "%s_%s" % (i, k))] = v
    _compile_npz_dict((0, self.root), "")
    np.savez_compressed(path, **dikt)

class BambooScope(object):
  def __init__(self, label, subsample_factor=None):
    self.label = label
    self.subsample_factor = 1 if subsample_factor is None else subsample_factor
    self.items = []
    self.i = 0

  def log(self, x):
    # append or overwrite such that we retain every `subsample_factor`th value and the last value
    item = (self.i, x)
    if (self.subsample_factor == 1 or
        self.i % self.subsample_factor == 1 or
        not self.items):
      self.items.append(item)
    else:
      self.items[-1] = item
    self.i += 1
