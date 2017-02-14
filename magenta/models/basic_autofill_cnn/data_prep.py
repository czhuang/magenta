
import os
from copy import copy
from collections import defaultdict
import cPickle as pickle

import numpy as np
from scipy.io import loadmat
import pylab as plt


def load_omniglot():
  fpath = '/Users/czhuang/packages/iwae/datasets/OMNIGLOT/chardata.mat'
  data = loadmat(fpath)
  return data


def decipher_omniglot_targetchar():
  # Each of these groups are 5 instances (characters?) for each alphebet?
  data = load_omniglot()
  train = data['data']
  targetchar = data['targetchar']
  for i in np.unique(targetchar):
    boolinds = targetchar==i
    boolinds = np.reshape(boolinds, (-1))
    samples = train[:, boolinds].T
    print i, samples.shape
    samples = np.reshape(samples, (-1, 28, 28))
    plot_subsample(samples, 'train_targetchar_group_%d' % i)
    

def decipher_omniglot_target():
  # Each group is an alphebet group?
  data = load_omniglot()
  train = data['data']
  target = data['target']
  for i in range(target.shape[0]):
    # targets are one-hot vectors.
    inclusion = target[i]>0.5
    samples = train[:, inclusion].T
    samples = np.reshape(samples, (-1, 28, 28))
    plot_subsample(samples, 'train_target_group_%d' % i)


def prep_omniglot(sample=False):
  # FIXME: still might have the problem of wanting to separate who wrote what?
  data = load_omniglot()
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

  split = dict()
  split['train'] = train_subsamples.T.reshape((-1, 28, 28))
  split['valid'] = valid_subsamples.T.reshape((-1, 28, 28))
  split['test'] = test.T.reshape((-1, 28, 28))

  if sample:
    # Sample binarization.  
    split['train'] = np.random.random(split['train'].shape) < split['train']
    split['valid'] = np.random.random(split['valid'].shape) < split['valid']
    split['test'] = np.random.random(split['test'].shape) < split['test'] 
    fname = 'omniglot-all_sampled_binarization.npz'
  else:
    # Threshold binarization.
    split['train'] = split['train'] > 0.5
    split['valid'] = split['valid'] > 0.5
    split['test'] = split['test'] > 0.5
    fname = 'omniglot-all_threshold_binarized.npz'

  np.savez_compressed(fname, **split)
  return fname


def plot_subsample(xs, tag_fname):
  fig, axes = plt.subplots(10,10)
  axes = np.ravel(axes)
  print '# of examples', len(xs)
  rand_inds = np.random.choice(len(xs), size=100)
  for i, ax in enumerate(axes):
    ax.imshow(xs[rand_inds[i]], cmap='gray', 
              interpolation='none')
  plt.savefig('check_%s.png' % tag_fname)
  

def check_omniglot(fname):
  data = np.load(fname)
  for set_ in ['train', 'valid', 'test']:
    print set_, data[set_].shape

  fname_tag = fname.split('.npz')[0]

  for set_ in ['train', 'valid', 'test']:
    num_uniques = len(np.unique(data[set_]))
    print set_, num_uniques
    #if num_uniques < 5:
    #  print np.unique(data[set_])
    xs = data[set_]
    plot_subsample(xs, '%s-%s' % (fname_tag, set_))


if __name__ == '__main__':
  #try:
  #  prep_omniglot()
  #except:
  #  import pdb; pdb.post_mortem()
  fname = prep_omniglot(sample=True)   
  check_omniglot(fname)   
  #decipher_omniglot()
  #decipher_omniglot_target()


