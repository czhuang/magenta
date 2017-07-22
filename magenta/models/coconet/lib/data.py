"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

import lib.mask
import lib.pianoroll
import lib.util

DATASET_PARAMS = {
    'Nottingham': {
        'pitch_range': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 9},
    'MuseData': {
        'pitch_range': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 14},
    'Piano-midi.de': {
        'pitch_range': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 12,
        'batch_size': 12},
    'jsb-chorales-16th-instrs_separated': {
        'pitch_range': [36, 81], 'shortest_duration': 0.125,
        'num_instruments': 4, 'qpm': 60},
}


class Dataset(object):
  def __init__(self, basepath, hparams, fold):
    self.basepath = basepath
    self.hparams = hparams
    self.fold = fold

    if self.params['shortest_duration'] != self.hparams.quantization_level:
      raise ValueError('The data has a temporal resolution of shortest '
                       'duration=%r, requested=%r' %
                       (self.params['shortest_duration'],
                        self.hparams.quantization_level))

    self.encoder = lib.pianoroll.PianorollEncoderDecoder(
        shortest_duration=self.params['shortest_duration'],
        min_pitch=self.min_pitch,
        max_pitch=self.max_pitch,
        separate_instruments=self.hparams.separate_instruments,
        num_instruments=self.hparams.num_instruments,
        quantization_level=self.hparams.quantization_level)

    self.data = np.load(os.path.join(self.basepath, "%s.npz" % self.name))[fold]

  @property
  def name(self):
    return self.hparams.dataset_name

  @property
  def num_examples(self):
    return len(self.data)

  @property
  def num_pitches(self):
    return self.max_pitch + 1 - self.min_pitch

  @property
  def max_pitch(self):
    return self.params["pitch_range"][1]

  @property
  def min_pitch(self):
    return self.params["pitch_range"][0]

  @property
  def params(self):
    return DATASET_PARAMS[self.name]

  def get_sequences(self):
    return self.data

  def get_pianorolls(self, sequences=None):
    if sequences is None:
      sequences = self.get_sequences()
    encoder = self.get_encoder()
    return list(map(encoder.encode, sequences))

  def get_featuremaps(self, sequences=None):
    """Return input and output pairs of masked out and full pianorolls.
  
    Args:
      sequences: A list of NoteSequences. If not given, the full dataset
          is used.
  
    Returns:
      input_data: A 4D matrix with dimensions named
          (batch, time, pitch, masked_or_mask), interleaved with maskout
          pianorolls and masks.
      target: A 4D matrix of the original pianorolls with dimensions named
          (batch, time, pitch).
  
    Raises:
      DataProcessingError: If pianoroll is shorter than the desired crop_len, or
          if the inputs and targets have the wrong number of dimensions.
    """
    if sequences is None:
      sequences = self.get_sequences()

    input_data = []
    targets = []

    for sequence in sequences:
      pianoroll = self.encoder.encode(sequence)
      pianoroll = lib.util.random_crop(pianoroll, self.hparams.crop_piece_len)
      mask = lib.mask.get_mask(
          self.hparams.maskout_method, pianoroll.shape,
          separate_instruments=self.hparams.separate_instruments,
          blankout_ratio=self.hparams.corrupt_ratio)
      masked_pianoroll = lib.mask.apply_mask_and_stack(pianoroll, mask)
      input_data.append(masked_pianoroll)
      targets.append(pianoroll)
  
    (input_data, targets), lengths = lib.util.pad_and_stack(input_data, targets)
    assert input_data.ndim == 4 and targets.ndim == 4
    return input_data, targets, lengths
