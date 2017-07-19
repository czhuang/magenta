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


def random_crop_pianoroll(pianoroll, length):
  leeway = len(pianoroll) - length
  start = np.random.randint(1 + max(0, leeway))
  pianoroll = pianoroll[start:start + length]
  return pianoroll


def pad_and_stack(*xss):
  """Pad and stack lists of examples.

  Each argument `xss[i]` is taken to be a list of variable-length examples.
  The examples are padded to a common length and stacked into an array.
  Example lengths must match across the `xss[i]`.

  Args:
    *xss: lists of examples to stack

  Returns:
    A tuple `(yss, lengths)`. `yss` is a list of arrays of padded examples,
    each `yss[i]` corresponding to `xss[i]`. `lengths` is an array of example
    lengths.
  """
  yss = []
  lengths = list(map(len, xss[0]))
  for xs in xss:
    # example lengths must be the same across arguments
    assert lengths == list(map(len, xs))
    max_length = max(lengths)
    rest_shape = xs[0].shape[1:]
    ys = np.zeros((len(xs), max_length,) + rest_shape, dtype=xs[0].dtype)
    for i in range(len(xs)):
      ys[i, :len(xs[i])] = xs[i]
    yss.append(ys)
  return yss, np.asarray(lengths)


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
    pianoroll = random_crop_pianoroll(pianoroll, hparams.crop_piece_len)
    mask_fn = getattr(mask_tools, 'get_%s_mask' % hparams.maskout_method)
    mask = mask_fn(pianoroll.shape,
                   separate_instruments=hparams.separate_instruments,
                   blankout_ratio=hparams.corrupt_ratio)
    if hparams.denoise_mode:
      masked_pianoroll = mask_tools.perturb_and_stack(pianoroll, mask)
    else:
      masked_pianoroll = mask_tools.apply_mask_and_stack(pianoroll, mask)
    input_data.append(masked_pianoroll)
    targets.append(pianoroll)

  (input_data, targets), lengths = pad_and_stack(input_data, targets)
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
      encode_silences=hparams.encode_silences,
      quantization_level=hparams.quantization_level)
  return seqs, encoder
