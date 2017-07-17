"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

import mask_tools
from pianorolls_lib import PianorollEncoderDecoder

# Enumerations for data augmentation for durations.
KEEP_ORIGINAL_DURATIONS, HALF_TIME, DOUBLE_TIME = range(3)

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

# TODO: not included in this version.
IMAGE_DATASETS = ['MNIST', 'BinaryMNIST', 'OMNIGLOT']


class DataProcessingError(Exception):
  """Exception for when data does not meet the expected requirements."""
  pass


def random_double_or_halftime_pianoroll_from_note_sequence(
    sequence, augment_by_halfing_doubling_durations, encoder):
  if not augment_by_halfing_doubling_durations:
    return encoder.encode(sequence)
  assert False, 'Not yet supported in this version.'


def random_crop_pianoroll(pianoroll,
                          crop_len,
                          start_crop_index=None,
                          augment_by_transposing=False):
  """Return a random crop in time of a pianoroll.

  Args:
    pianoroll: A 3D matrix, with time as the first axis.
    crop_len: The number of rows.

  Returns:
    A 3D matrix with the first axis cropped to the length of crop_len.

  Raises:
    DataProcessingError: If the pianoroll shorter than the desired crop_len.
  """
  #print '\ncrop_len', crop_len
  #print pianoroll.shape
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

  cropped_pianoroll = pianoroll[start_time_idx:start_time_idx + crop_len]
  if not augment_by_transposing:
    return cropped_pianoroll

  # High exclusive, also it's asymmetic.
  random_shift = np.random.randint(-5, 7)
  #print 'cropped_pianoroll', cropped_pianoroll.shape
  #print 'sum of crop', np.sum(cropped_pianoroll)
  #print 'random_shift', random_shift
  if random_shift == 0:
    return cropped_pianoroll

  pitch_sum = np.sum(cropped_pianoroll, axis=(0, 2))
  #print np.max(np.where(pitch_sum)), np.min(np.where(pitch_sum))

  shifted_pianoroll = np.roll(cropped_pianoroll, random_shift, axis=2)
  # Check that there's actually no roll over.
  if random_shift > 0:
    # Even though high range here is exclusive, since checking if high rolled into low.
    num_events = np.sum(shifted_pianoroll[:, :random_shift - 1, :])
    #print 'num_events', num_events
    assert num_events == 0
  else:
    # random_shift is negative, hence don't need to add negative sign to random_shift.
    num_events = np.sum(shifted_pianoroll[:, random_shift - 1:, :])
    #print 'num_events', num_events
    assert num_events == 0
  return shifted_pianoroll


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
  maskout_method = hparams.maskout_method
  input_data = []
  targets = []
  lengths = []
  seq_count = 0
  for sequence in sequences:
    if encoder is not None:
      pianoroll = random_double_or_halftime_pianoroll_from_note_sequence(
          sequence, hparams.augment_by_halfing_doubling_durations, encoder)
    else:
      # For images, no encoder, already in pianoroll-like form.
      pianoroll = sequence
      if hparams.dataset == 'OMNIGLOT':
        #print 'binarizing images', pianoroll.shape
        pianoroll = np.random.random(pianoroll.shape) < pianoroll
    try:
      if hparams.pad:
        # TODO: Padding function does not support augment_by_transposing yet.
        cropped_pianoroll, length = random_crop_pianoroll_pad(
            pianoroll, hparams.crop_piece_len, start_crop_index)
        #if length != cropped_pianoroll.shape[0]:
        #  print length, 'padded to', cropped_pianoroll.shape[0]
      else:  
        cropped_pianoroll = random_crop_pianoroll(
            pianoroll, hparams.crop_piece_len, start_crop_index,
            hparams.augment_by_transposing)
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
  if hparams.dataset not in IMAGE_DATASETS:
    assert encoder.quantization_level == hparams.quantization_level
    return [encoder.encode(seq) for seq in seqs]

  if hparams.dataset == 'OMNIGLOT':
    prev_rng_state = np.random.get_state()
    np.random.seed(123)
    
    seqs = np.asarray(seqs)
    print 'binarizing images', seqs.shape
    seqs = np.random.random(seqs.shape) < seqs

    # restore main random stream
    np.random.set_state(prev_rng_state)
  return seqs


def get_image_data(dataset_name, fold, params):
  if dataset_name == 'MNIST':
    from tensorflow.examples.tutorials.mnist import input_data
    print 'Downloading or unpacking MNIST data...'
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if fold == 'valid':
      fold = 'validation'
    data = getattr(mnist, fold).images
    data = np.reshape(data, (-1, 28, 28, 1))
    print 'MNIST', data.shape
    return data
  elif dataset_name == 'BinaryMNIST':
    fpath = os.path.join(params['path'], 'binarized_mnist_%s.amat' % fold)
    print 'Loading BinaryMNIST data from', fpath
    with open(fpath) as f:
      lines = f.readlines()
    data = np.array([[int(i) for i in line.split()] for line in lines])    
    data = np.reshape(data, (-1, 28, 28, 1))
    print 'BinaryMNIST', data.shape
    return data
  elif dataset_name == 'OMNIGLOT':
    data = np.load(params['path'])[fold]
    data = np.reshape(data, (-1, 28, 28, 1))
    return data
  else:
    assert False, 'Dataset %s not yet supported.' % dataset_name


def get_data_and_update_hparams(basepath, hparams, fold, 
                                update_hparams=True, 
                                return_encoder=False):
  dataset_name = hparams.dataset
  params = DATASET_PARAMS[dataset_name]
  
  if dataset_name in IMAGE_DATASETS: 
    # for image datasets
    seqs = get_image_data(dataset_name, fold, params)
  else:
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
    if dataset_name not in IMAGE_DATASETS:
      hparams.num_pitches = pitch_range[1] - pitch_range[0] + 1
    for key, value in params.iteritems():
      if hasattr(hparams, key): 
        setattr(hparams, key, value)
    #FIXME: just for debug.
    for key in params:
      if hasattr(hparams, key):
        assert getattr(hparams, key) == params[key], 'hparams did not get updated, %r!=%r' % (getattr(hparams, key), params[key])
  if return_encoder and dataset_name not in IMAGE_DATASETS:
    assert params['shortest_duration'] == hparams.quantization_level, 'The data has a temporal resolution of shortest duration=%r, requested=%r' % (params['shortest_duration'], hparams.quantization_level)
    encoder = PianorollEncoderDecoder(
        shortest_duration=params['shortest_duration'],
        min_pitch=pitch_range[0],
        max_pitch=pitch_range[1],
        separate_instruments=separate_instruments,
        num_instruments=hparams.num_instruments,
        encode_silences=hparams.encode_silences,
        quantization_level=hparams.quantization_level)
  else:
    encoder = None
  
  if return_encoder:
    return seqs, encoder
  else:
    return seqs
