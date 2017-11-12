"""Classes for datasets and batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import lib.mask
import lib.pianoroll
import lib.util


class Dataset(lib.util.Factory):
  def __init__(self, basepath, hparams, fold):
    self.basepath = basepath
    self.hparams = hparams
    self.fold = fold

    if self.shortest_duration != self.hparams.quantization_level:
      raise ValueError('The data has a temporal resolution of shortest '
                       'duration=%r, requested=%r' %
                       (self.shortest_duration,
                        self.hparams.quantization_level))

    # Update the default pitch ranges in hparams to reflect that of dataset.
    hparams.pitch_ranges = [self.min_pitch, self.max_pitch]
    hparams.shortest_duration = self.shortest_duration
    self.encoder = lib.pianoroll.get_pianoroll_encoder_decoder(hparams)
    self.data = np.load(os.path.join(self.basepath, "%s.npz" % self.name))[fold]

  @property
  def name(self):
    return self.hparams.dataset

  @property
  def num_examples(self):
    return len(self.data)

  @property
  def num_pitches(self):
    return self.max_pitch + 1 - self.min_pitch

  def get_sequences(self):
    return self.data

  def get_pianorolls(self, sequences=None):
    if sequences is None:
      sequences = self.get_sequences()
    return list(map(self.encoder.encode, sequences))

  def get_featuremaps(self, sequences=None):
    if sequences is None:
      sequences = self.get_sequences()

    pianorolls = []
    masks = []

    for sequence in sequences:
      pianoroll = self.encoder.encode(sequence)
      pianoroll = lib.util.random_crop(pianoroll, self.hparams.crop_piece_len)
      mask = lib.mask.get_mask(
          self.hparams.maskout_method, pianoroll.shape,
          separate_instruments=self.hparams.separate_instruments,
          blankout_ratio=self.hparams.corrupt_ratio)
      pianorolls.append(pianoroll)
      masks.append(mask)
  
    (pianorolls, masks), lengths = lib.util.pad_and_stack(pianorolls, masks)
    assert pianorolls.ndim == 4 and masks.ndim == 4
    assert pianorolls.shape == masks.shape
    return Batch(pianorolls=pianorolls, masks=masks, lengths=lengths)

  def update_hparams(self, hparams):
    for key in "num_instruments num_pitches min_pitch max_pitch qpm".split():
      setattr(hparams, key, getattr(self, key))

def get_dataset(basepath, hparams, fold):
  return Dataset.make(hparams.dataset, basepath, hparams, fold)

class Nottingham(Dataset):
  key = "Nottingham"
  min_pitch = 21
  max_pitch = 108
  shortest_duration = 0.25
  num_instruments = 9

class MuseData(Dataset):
  key = "MuseData"
  min_pitch = 21
  max_pitch = 108
  shortest_duration = 0.25
  num_instruments = 14

class PianoMidiDe(Dataset):
  key = "PianoMidiDe"
  min_pitch = 21
  max_pitch = 108
  shortest_duration = 0.25
  num_instruments = 12

class Jsb16thSeparated(Dataset):
  key = "Jsb16thSeparated"
  min_pitch = 36
  max_pitch = 81
  shortest_duration = 0.125
  num_instruments = 4
  qpm = 60

class Batch(object):
  keys = set("pianorolls masks lengths".split())

  def __init__(self, **kwargs):
    assert set(kwargs.keys()) == self.keys
    self.features = kwargs

  def get_feed_dict(self, placeholders):
    assert set(placeholders.keys()) == self.keys
    return dict((placeholders[key], self.features[key])
                for key in self.keys)

  def batches(self, **batches_kwargs):
    keys, values = list(zip(*list(self.features.items())))
    for batch in lib.util.batches(*values, **batches_kwargs):
      yield Batch(**dict(lib.util.eqzip(keys, batch)))
