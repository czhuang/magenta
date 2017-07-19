"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

import mask_tools
from pianorolls_lib import PianorollEncoderDecoder

DATASET_PARAMS = {
    'Nottingham': {
        'pitch_ranges': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 9}, 
    'MuseData': {
        'pitch_ranges': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 14},
    'Piano-midi.de': {
        'pitch_ranges': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 12,
        'batch_size': 12},
    'jsb-chorales-16th-instrs_separated': {
        'pitch_ranges': [36, 81], 'shortest_duration': 0.125,
        'num_instruments': 4, 'qpm': 60},
}


class DataProcessingError(Exception):
  """Exception for when data does not meet the expected requirements."""
  pass


def random_crop_pianoroll(pianoroll,
                          crop_len,
                          start_crop_index=None):
  """Return a random crop in time of a pianoroll.

  Args:
    pianoroll: A 3D matrix, with time as the first axis.
    crop_len: The number of rows.

  Returns:
    A 3D matrix with the first axis cropped to the length of crop_len.

  Raises:
    DataProcessingError: If the pianoroll shorter than the desired crop_len.
  """
  if len(pianoroll) < crop_len:
    # TODO(annahuang): Pad pianoroll when too short, and add mask to loss.
    raise DataProcessingError(
        'Piece needs to be at least %d steps, currently %d steps.' %
        (crop_len, len(pianoroll)))
  if len(pianoroll) == crop_len:
    start_time_idx = 0
  elif start_crop_index is not None:
    start_time_idx = start_crop_index
  else:
    start_time_idx = np.random.randint(len(pianoroll) - crop_len)
  return pianoroll = pianoroll[start_time_idx:start_time_idx + crop_len]


def random_crop_pianoroll_pad(pianoroll,
                              crop_len,
                              start_crop_index=None):
  length = len(pianoroll)
  pad_length = crop_len - len(pianoroll)
  if pad_length > 0:
    pianoroll = np.pad(pianoroll, [(0, pad_length)] + [(0, 0)] * (pianoroll.ndim - 1), mode="constant")
  else:
    if start_crop_index is not None:
      start_time_idx = start_crop_index
    else:
      start_time_idx = np.random.randint(len(pianoroll) - crop_len + 1)
    pianoroll = pianoroll[start_time_idx:start_time_idx + crop_len]
  non_padded_length = length if length < crop_len else crop_len
  return pianoroll, non_padded_length


def make_data_feature_maps(sequences, hparams, encoder, start_crop_index=None):
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
  lengths = []
  seq_count = 0
  for sequence in sequences:
    if encoder is not None:
      pianoroll = encoder.encode(sequence)
    else:
      # For images, no encoder, already in pianoroll-like form.
      pianoroll = sequence
    try:
      if hparams.pad:
        cropped_pianoroll, length = random_crop_pianoroll_pad(
            pianoroll, hparams.crop_piece_len, start_crop_index)
      else:  
        cropped_pianoroll = random_crop_pianoroll(
            pianoroll, hparams.crop_piece_len, start_crop_index)
        length = hparams.crop_piece_len
    except DataProcessingError:
      tf.logging.warning('Piece shorter than requested crop length.')
      continue
    seq_count += 1
   
    # Get mask.
    T, P, I = cropped_pianoroll.shape
    unpadded_shape = length, P, I
    assert np.sum(cropped_pianoroll[length:, :, :]) == 0
    mask = getattr(mask_tools, 'get_%s_mask' % hparams.maskout_method)(
        unpadded_shape, separate_instruments=hparams.separate_instruments,
        blankout_ratio=hparams.corrupt_ratio)
    if not hparams.pad:
      assert mask.shape[0] == cropped_pianoroll.shape[0]
    if hparams.denoise_mode:
      # TODO: Denoise not yet supporting padding.
      masked_pianoroll = mask_tools.perturb_and_stack(cropped_pianoroll, mask)
    else:
      masked_pianoroll = mask_tools.apply_mask_and_stack(
          cropped_pianoroll, mask, hparams.pad)

    input_data.append(masked_pianoroll)
    targets.append(cropped_pianoroll)
    lengths.append(length)
    assert len(input_data) == seq_count
    assert len(input_data) == len(targets)
    assert len(input_data) == len(lengths)

  input_data = np.asarray(input_data)
  targets = np.asarray(targets)
  lengths = np.asarray(lengths)
  if not (input_data.ndim == 4 and targets.ndim == 4):
    print input_data.ndim, targets.ndim
    raise DataProcessingError('Input data or target dimensions incorrect.')
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
  
  separate_instruments = hparams.separate_instruments
  # TODO: Read dataset params from JSON file or the like.
  pitch_range = params['pitch_ranges']
  if '4part_Bach_chorales' in dataset_name:
    fpath = os.path.join(basepath, params['relative_path'], '%s.tfrecord' % fold)
    seqs = list(note_sequence_record_iterator(fpath))
  else:
    fpath = os.path.join(basepath, dataset_name+'.npz')
    data = np.load(fpath)
    seqs = data[fold]

  # Update hparams.
  if update_hparams:
    hparams.num_pitches = pitch_range[1] - pitch_range[0] + 1
    for key, value in params.iteritems():
      if hasattr(hparams, key): 
        setattr(hparams, key, value)
    #FIXME: just for debug.
    for key in params:
      if hasattr(hparams, key):
        assert getattr(hparams, key) == params[key], 'hparams did not get updated, %r!=%r' % (getattr(hparams, key), params[key])

  if not return_encoder:
    return seqs

  assert params['shortest_duration'] == hparams.quantization_level, 'The data has a temporal resolution of shortest duration=%r, requested=%r' % (params['shortest_duration'], hparams.quantization_level)
  encoder = PianorollEncoderDecoder(
      shortest_duration=params['shortest_duration'],
      min_pitch=pitch_range[0],
      max_pitch=pitch_range[1],
      separate_instruments=separate_instruments,
      num_instruments=hparams.num_instruments,
      encode_silences=hparams.encode_silences,
      quantization_level=hparams.quantization_level)
  return seqs, encoder
