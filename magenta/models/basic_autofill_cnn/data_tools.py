"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

import mask_tools
from pianorolls_lib import PianorollEncoderDecoder

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


def make_data_feature_maps(sequences, hparams, encoder):
  """Return input and output pairs of masked out and full pianorolls.

  Args:
    sequences: A list of NoteSequences.

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
  input_data = []
  targets = []
  for sequence in sequences:
    pianoroll = encoder.encode(sequence)
    pianoroll = util.random_crop(pianoroll, hparams.crop_piece_len)
    mask = mask_tools.get_mask(
        hparams.maskout_method, pianoroll.shape,
        separate_instruments=hparams.separate_instruments,
        blankout_ratio=hparams.corrupt_ratio)
    masked_pianoroll = mask_tools.apply_mask_and_stack(pianoroll, mask)
    input_data.append(masked_pianoroll)
    targets.append(pianoroll)

  (input_data, targets), lengths = util.pad_and_stack(input_data, targets)
  assert input_data.ndim == 4 and targets.ndim == 4
  return input_data, targets, lengths


def get_data_as_pianorolls(basepath, hparams, fold):
  seqs, encoder = get_data_and_update_hparams(
      basepath, hparams, fold, update_hparams=False, return_encoder=True)
  assert encoder.quantization_level == hparams.quantization_level
  return [encoder.encode(seq) for seq in seqs]


def get_data_and_update_hparams(basepath, hparams, fold, 
                                update_hparams=True, 
                                return_encoder=False):
  dataset_name = hparams.dataset
  params = DATASET_PARAMS[dataset_name]
  fpath = os.path.join(basepath, dataset_name+'.npz')
  data = np.load(fpath)
  seqs = data[fold]
  pitch_range = params['pitch_range']

  if update_hparams:
    hparams.num_pitches = pitch_range[1] - pitch_range[0] + 1
    hparams.update(params)

  if not return_encoder:
    return seqs

  if params['shortest_duration'] != hparams.quantization_level:
    raise ValueError('The data has a temporal resolution of shortest '
                     'duration=%r, requested=%r' %
                     (params['shortest_duration'], hparams.quantization_level))

  encoder = PianorollEncoderDecoder(
      shortest_duration=params['shortest_duration'],
      min_pitch=pitch_range[0],
      max_pitch=pitch_range[1],
      separate_instruments=hparams.separate_instruments,
      num_instruments=hparams.num_instruments,
      quantization_level=hparams.quantization_level)
  return seqs, encoder
