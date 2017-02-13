
import os
from copy import copy
from collections import defaultdict
import cPickle as pickle

import numpy as np
from scipy.io import loadmat
import pylab as plt


def prep_omniglot():
  fpath = '/Users/czhuang/packages/iwae/datasets/OMNIGLOT/chardata.mat'
  data = loadmat(fpath)
  # # of pixels by examples
  train = data['data']
  test = data['testdata']
  # targets by examples
  train_targets = data['target']
  test_targets = data['testtarget']

  num_classes = train_targets.shape[0]

  n = train.shape[-1]
  train_n = n / 3. * 2
  assert train_n % 1. == 0
  train_subsamples = np.zeros((784, train_n))
  valid_subsamples = np.zeros((784, n - train_n))
  train_subtargets = np.zeros((50, train_n))
  valid_subtargets = np.zeros((50, n - train_n))
  train_count = 0 
  valid_count = 0
  # Want to make sure balanced among classes.
  for class_idx in range(num_classes):
    inclusion = train_targets[class_idx]>0.5
    inclass_samples = train[:, inclusion]
    inclass_targets = train_targets[:, inclusion]
    n_class = inclass_samples.shape[1]
    n_valid = n_class / 3. 
    assert n_valid % 1. == 0
    n_train = n_class - n_valid
    random_inds = np.random.permutation(n_class)
    train_subsamples[:, train_count:train_count+n_train] = (
        inclass_samples[:, random_inds[:n_train]])
    train_subtargets[:, train_count:train_count+n_train] = (
        inclass_targets[:, random_inds[:n_train]])
    
    valid_subsamples[:, valid_count:valid_count+n_valid] = (
        inclass_samples[:, n_train:])
    valid_subtargets[:, valid_count:valid_count+n_valid] = (
        inclass_targets[:, n_train:])
    train_count += n_train
    valid_count += n_valid
  assert train_count == train_subsamples.shape[-1]
  assert valid_count == valid_subsamples.shape[-1]

  split_data = dict()
  split_data['train'] = train_subsamples.T.reshape((-1, 28, 28))
  split_data['test'] = test.T.reshape((-1, 28, 28))
  valid = valid_subsamples.T.reshape((-1, 28, 28))
  split_data['valid'] = np.random.random(valid.shape) < valid
  fname = 'omniglot-only_valid_binarized.npz'
  np.savez_compressed(fname, **split_data)


def check_omniglot():
  fname = 'omniglot-only_valid_binarized.npz'
  data = np.load(fname)
  for set_ in ['train', 'valid', 'test']:
    print set_, data[set_].shape

  for set_ in ['train', 'valid', 'test']:
    num_uniques = len(np.unique(data[set_]))
    print set_, num_uniques
    if num_uniques < 5:
      print np.unique(data[set_])
      fig, axes = plt.subplots(4,4)
      axes = np.ravel(axes)
      for i, ax in enumerate(axes):
        ax.imshow(data[set_][i], cmap='gray', 
                  interpolation='none')
      plt.savefig('check_%s.png' % (set_))


if __name__ == '__main__':
  #try:
  #  prep_omniglot()
  #except:
  #  import pdb; pdb.post_mortem()
  #prep_omniglot()   
  check_omniglot()   




