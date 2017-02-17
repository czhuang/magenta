
import os
from copy import copy
from collections import defaultdict
import cPickle as pickle

import numpy as np
from scipy.io import loadmat
import pylab as plt


MAX_NUM_CHARS = 55
NUM_ALPHABETS = 50

NUM_VALID_EXAMPLES_TO_SPLIT = 3


def load_omniglot():
  fpath = '/Users/czhuang/packages/iwae/datasets/OMNIGLOT/chardata.mat'
  data = loadmat(fpath)
  return data


def decipher_omniglot_targetchar(set_='train'):
  # Each of these groups are 5 instances (characters?) for each alphebet?
  data = load_omniglot()

  if set_ == 'train':
    # for each group (750, 784)
    targetchar = data['targetchar']
    train = data['data']
  elif set_ == 'test':
    # for each group the size is (249, 784)
    targetchar = data['testtargetchar']
    train = data['testdata']
  else:
    assert False, 'Unrecognized set %s' % set_
  for i in np.unique(targetchar):
    boolinds = targetchar==i
    boolinds = np.reshape(boolinds, (-1))
    samples = train[:, boolinds].T
    print i, samples.shape
    samples = np.reshape(samples, (-1, 28, 28))
    plot_subsample(samples, 'fromBack-%s_targetchar_group_%d' % (set_, i), 
                   randomize=False)
    

def decipher_omniglot_target():
  # Each group is an alphebet group?
  data = load_omniglot()
  train = data['data']
  target = data['target']
  for i in range(target.shape[0]):
    # alphabets are one-hot vectors.
    alpha_inclusion = target[i]>0.5
    samples = train[:, alpha_inclusion].T
    samples = np.reshape(samples, (-1, 28, 28))
    plot_subsample(samples, 'train_target_group_%d' % i)


def prep_omniglot(sample=None):
  data = load_omniglot()
  # # of pixels by examples
  train = data['data']
  test = data['testdata']

  # One-hot column vector for which alphabet. 
  # 50 rows (alphabets) by # of training examples.
  train_alphabets = data['target']
  alphabet_counts = np.sum(train_alphabets, axis=1)

  # Char index in corresponding alphabet.
  train_chars = data['targetchar']
  
  num_alphabets = train_alphabets.shape[0]
  assert num_alphabets == NUM_ALPHABETS

  train_subsamples = []
  valid_subsamples = []
  # Want to make sure balanced among alphabets.
  for alpha_idx in range(num_alphabets):
    alpha_inclusion = train_alphabets[alpha_idx]==1
    in_alpha_samples = train[:, alpha_inclusion]
    in_alpha_chars = train_chars[:, alpha_inclusion]
    assert in_alpha_chars.shape[0] == 1 and in_alpha_chars.ndim == 2
    in_alpha_chars = np.ravel(in_alpha_chars)
    counts = in_alpha_samples.shape[1]
    for char_idx in np.unique(in_alpha_chars):
       char_inclusion = in_alpha_chars == char_idx
       in_alpha_char_samples = in_alpha_samples[:, char_inclusion]
       n_valid = int(np.ceil(np.sum(char_inclusion)/ 5.))
       permutation = np.random.permutation(in_alpha_char_samples.shape[-1])  
 
       train_subsamples.append(in_alpha_char_samples[:, permutation[n_valid:]])
       valid_subsamples.append(in_alpha_char_samples[:, permutation[:n_valid]])
  
  train_subsamples = np.concatenate(train_subsamples, axis=-1)
  valid_subsamples = np.concatenate(valid_subsamples, axis=-1)
  assert train_subsamples.shape[-1] + valid_subsamples.shape[-1] == (
      train.shape[-1])

  split = dict()
  split['train'] = train_subsamples.T.reshape((-1, 28, 28))
  split['valid'] = valid_subsamples.T.reshape((-1, 28, 28))
  split['test'] = test.T.reshape((-1, 28, 28))

  if sample is None:
    fname = 'omniglot-all_real.npz'
  elif sample:
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


def plot_subsample(xs, tag_fname, randomize=True):
  fig, axes = plt.subplots(10,10)
  axes = np.ravel(axes)
  print '# of examples', len(xs)
  if randomize:
    chosen_inds = np.random.choice(len(xs), size=100)
  else:
    chosen_inds = np.arange(100)
  for i, ax in enumerate(axes):
    ax.imshow(xs[chosen_inds[i]], cmap='gray', 
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
  fname = prep_omniglot(sample=None)   
  check_omniglot(fname)   
  #decipher_omniglot()
  #decipher_omniglot_target()
  #decipher_omniglot_targetchar('train')
  #decipher_omniglot_targetchar('test')

