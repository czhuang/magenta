"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

from magenta.lib.note_sequence_io import note_sequence_record_iterator
from magenta.protobuf import music_pb2

from magenta.models.basic_autofill_cnn import mask_tools
from magenta.models.basic_autofill_cnn.mask_tools import MaskUseError
from magenta.models.basic_autofill_cnn.pianorolls_lib import PianorollEncoderDecoder

# Enumerations for data augmentation for durations.
KEEP_ORIGINAL_DURATIONS, HALF_TIME, DOUBLE_TIME = range(3)


class DataProcessingError(Exception):
  """Exception for when data does not meet the expected requirements."""
  pass


def random_double_or_halftime_pianoroll_from_note_sequence(
    sequence, augment_by_halfing_doubling_durations, encoder):
  if not augment_by_halfing_doubling_durations:
    return encoder.encode(sequence)

  durations = set(note.end_time - note.start_time for note in sequence.notes)
  longest_to_double = 4
  shortest_to_half = 0.5
  duration_augmentation_type = np.random.randint(3)
  if duration_augmentation_type == KEEP_ORIGINAL_DURATIONS:
    return encoder.encode(sequence)
  elif duration_augmentation_type == HALF_TIME:
    # Half time.
    for duration in list(durations):
      if duration < shortest_to_half:
        return encoder.encode(sequence)
    #print sequence.filename, sequence.id, sequence.collection_name
    return encoder.encode(sequence, duration_ratio=0.5)
  else:
    for duration in list(durations):
      if duration > longest_to_double:
        return encoder.encode(sequence)
    return encoder.encode(sequence, duration_ratio=2)


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


def make_data_feature_maps(sequences, config, encoder, start_crop_index=None):
  """Return input and output pairs of masked out and full pianorolls.

  Args:
    sequences: A list of NoteSequences.
    config: A PipelineConfig object that stores which mask out method to use
        and its enumerations. It also stores hyperparameters such as
        crop_piece_length which determines the width of the feature maps.

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
  maskout_method = config.maskout_method
  input_data = []
  targets = []
  maskout_border = config.maskout_border
  seq_count = 0
  for sequence in sequences:
    pianoroll = random_double_or_halftime_pianoroll_from_note_sequence(
        sequence, config.hparams.augment_by_halfing_doubling_durations, encoder)
    try:
      cropped_pianoroll = random_crop_pianoroll(
          pianoroll, config.hparams.crop_piece_len, start_crop_index,
          config.hparams.augment_by_transposing)
    except DataProcessingError:
      tf.logging.warning('Piece shorter than requested crop length.')
      continue
    seq_count += 1
    
    if maskout_method == config.RANDOM_INSTRUMENT:
      mask = mask_tools.get_random_instrument_mask(cropped_pianoroll.shape)
    elif maskout_method == config.RANDOM_PATCHES:
      mask = mask_tools.get_multiple_random_patch_mask(
          cropped_pianoroll.shape, maskout_border,
          config.initial_maskout_factor)
    elif maskout_method == config.RANDOM_PITCH_RANGE:
      # Only use when all instruments are collapsed in one pianoroll.
      if pianoroll.shape[-1] > 1:
        raise MaskUseError(
            'Only use when all instruments are represented in one pianoroll.')
    elif maskout_method == config.RANDOM_TIME_RANGE:
      mask = mask_tools.get_random_time_range_mask(cropped_pianoroll.shape,
                                                   maskout_border)
    elif maskout_method == config.RANDOM_MULTIPLE_INSTRUMENT_TIME:
      mask = mask_tools.get_multiple_random_instrument_time_mask(
          cropped_pianoroll.shape, maskout_border, config.num_maskout)
    elif config.hparams.denoise_mode or maskout_method == config.RANDOM_ALL_TIME_INSTRUMENT:
      mask = mask_tools.get_random_all_time_instrument_mask(
          cropped_pianoroll.shape, config.hparams.corrupt_ratio)
    elif maskout_method == config.RANDOM_EASY:
      mask = mask_tools.get_random_easy_mask(cropped_pianoroll.shape)
    elif maskout_method == config.RANDOM_MEDIUM:
      mask = mask_tools.get_random_medium_mask(cropped_pianoroll.shape)
    elif maskout_method == config.RANDOM_HARD:
      mask = mask_tools.get_random_hard_mask(cropped_pianoroll.shape)
    elif maskout_method == config.CHRONOLOGICAL_TI:
      mask = mask_tools.get_chronological_ti_mask(cropped_pianoroll.shape)
    elif maskout_method == config.CHRONOLOGICAL_IT:
      mask = mask_tools.get_chronological_it_mask(cropped_pianoroll.shape)
    elif maskout_method == config.FIXED_ORDER:
      mask = mask_tools.get_fixed_order_mask(cropped_pianoroll.shape)
    elif maskout_method == config.BALANCED:
      mask = mask_tools.get_balanced_mask(cropped_pianoroll.shape)
    elif maskout_method == config.NO_MASK:
      mask = mask_tools.get_no_mask(cropped_pianoroll.shape)
    else:
      raise ValueError('Mask method not supported.')
    
    if config.hparams.denoise_mode:
      masked_pianoroll = mask_tools.perturb_and_stack(cropped_pianoroll, mask)
    else:
      masked_pianoroll = mask_tools.apply_mask_and_stack(cropped_pianoroll, mask)
    input_data.append(masked_pianoroll)
    targets.append(cropped_pianoroll)
    assert len(input_data) == seq_count
    assert len(input_data) == len(targets)

  input_data = np.asarray(input_data)
  targets = np.asarray(targets)
  print '# of input_data', input_data.shape[0]
  print '# of targets', targets.shape[0]
  if not (input_data.ndim == 4 and targets.ndim == 4):
    print input_data.ndim, targets.ndim
    raise DataProcessingError('Input data or target dimensions incorrect.')
  return input_data, targets


def get_pianoroll_from_note_sequence_data(path, type_):
  """Retrieves NoteSequences from a TFRecord and returns piano rolls.

  Args:
    path: The absolute path to the TFRecord file.
    type_: The name of the TFRecord file which also specifies the type of data.

  Yields:
    3D binary numpy arrays.

  Raises:
    DataProcessingError: If the type_ specified is not one of train, test or
        valid.
  """
  if type_ not in ['train', 'test', 'valid']:
    raise DataProcessingError(
        'Data is grouped by train, test or valid. Please specify one.')
  fpath = os.path.join(path, '%s.tfrecord' % type_)
  encoder = PianorollEncoderDecoder()
  seq_reader = note_sequence_record_iterator(fpath)
  for seq in seq_reader:
    yield encoder.encode(seq)


def get_note_sequence_data(path, type_):
  """Retrieves NoteSequences from a TFRecord.

  Args:
    path: The absolute path to the TFRecord file.
    type_: The name of the TFRecord file which also specifies the type of data.

  Yields:
    NoteSequences.

  Raises:
    DataProcessingError: If the type_ specified is not one of train, test or
        valid.
  """
  if type_ not in ['train', 'test', 'valid']:
    raise DataProcessingError(
        'Data is grouped by train, test or valid. Please specify one.')
  fpath = os.path.join(path, '%s.tfrecord' % type_)
  print 'fpath', fpath
  #seq_reader = note_sequence_record_iterator(fpath)
  #for seq in seq_reader:
  #  yield seq
  reader = tf.python_io.tf_record_iterator(fpath)
  for serialized_sequence in reader:
    yield music_pb2.NoteSequence.FromString(serialized_sequence)
