
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
  valid_prop = 10.

  train_subsamples = []
  valid_subsamples = []
  # Want to make sure balanced among classes.
  for class_idx in range(num_classes):
    inclusion = train_targets[class_idx]>0.5
    inclass_samples = train[:, inclusion]
    n_class = inclass_samples.shape[1]
    n_valid = int(np.ceil(n_class / 10.))
    n_train = n_class - n_valid
    random_inds = np.random.permutation(n_class)
    
    train_subsamples.append(
        inclass_samples[:, random_inds[:n_train]])
    
    valid_subsamples.append(
        inclass_samples[:, random_inds[n_train:]])
  
  train_subsamples = np.concatenate(train_subsamples, axis=-1)
  valid_subsamples = np.concatenate(valid_subsamples, axis=-1)
  assert train_subsamples.shape[-1] + valid_subsamples.shape[-1] == train.shape[-1]

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
  prep_omniglot()   
  check_omniglot()   




