"""Logging the generation process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, contextlib
import numpy as np

# unobtrusive structured logging of arbitrary values

class NoLogger(object):
  def log(self, **kwargs):
    pass

  @contextlib.contextmanager
  def scope(self, *args, **kwargs):
    pass

class Logger(object):
  def __init__(self):
    self.root = Scope("root", subsample_factor=1)
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

  def dump(self, path):
    dikt = {}
    def _compile_npz_dict(item, path):
      i, node = item
      if isinstance(node, Scope):
        for subitem in node.items:
          _compile_npz_dict(subitem, os.path.join(path, "%s_%s" % (i, node.label)))
      else:
        for k, v in node.items():
          dikt[os.path.join(path, "%s_%s" % (i, k))] = v
    _compile_npz_dict((0, self.root), "")
    np.savez_compressed(path, **dikt)

class Scope(object):
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
