"""Prepares data for basic_autofill_cnn model by blanking out pianorolls."""

import os

import numpy as np
import tensorflow as tf

from magenta.music.note_sequence_io import note_sequence_record_iterator
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
  assert isinstance(sequence, music_pb2.NoteSequence), 'No support for dataaugmentation on non-NoteSequence data yet.'
  durations = set(note.end_time - note.start_time for note in sequence.notes)
  longest_to_double = 4
  shortest_to_half = 0.25
  duration_augmentation_type = np.random.randint(3)
  if duration_augmentation_type == KEEP_ORIGINAL_DURATIONS:
    return encoder.encode(sequence)
  elif duration_augmentation_type == HALF_TIME:
    # Half time.
    if any(duration < shortest_to_half for duration in durations):
      return encoder.encode(sequence)
    #print sequence.filename, sequence.id, sequence.collection_name
    return encoder.encode(sequence, duration_ratio=0.5)
  else:
    if any(duration > longest_to_double for duration in durations):
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


def random_crop_pianoroll_pad(pianoroll,
                              crop_len,
                              start_crop_index=None):
  length = len(pianoroll)
  pad_length = crop_len - len(pianoroll)
  if pad_length > 0:
    pianoroll = np.pad(pianoroll, [(0, pad_length)] + [(0, 0)] * (pianoroll.ndim - 1), mode="constant")
  if start_crop_index is not None:
    start_time_idx = start_crop_index
  else:
    start_time_idx = np.random.randint(len(pianoroll) - crop_len + 1)
  pianoroll = pianoroll[start_time_idx:start_time_idx + crop_len]
  return pianoroll, length


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
  seq_count = 0
  for sequence in sequences:
    pianoroll = random_double_or_halftime_pianoroll_from_note_sequence(
        sequence, hparams.augment_by_halfing_doubling_durations, encoder)
    try:
      cropped_pianoroll = random_crop_pianoroll(
          pianoroll, hparams.crop_piece_len, start_crop_index,
          hparams.augment_by_transposing)
    except DataProcessingError:
      tf.logging.warning('Piece shorter than requested crop length.')
      continue
    seq_count += 1
   
    # Get mask.
    mask = getattr(mask_tools, 'get_%s_mask' % hparams.maskout_method)(
        cropped_pianoroll.shape, separate_instruments=hparams.separate_instruments,
        blankout_ratio=hparams.corrupt_ratio)
    #  raise ValueError('Mask method not supported.')
    
    if hparams.denoise_mode:
      masked_pianoroll = mask_tools.perturb_and_stack(cropped_pianoroll, mask)
    else:
      masked_pianoroll = mask_tools.apply_mask_and_stack(cropped_pianoroll, mask)
    input_data.append(masked_pianoroll)
    targets.append(cropped_pianoroll)
    assert len(input_data) == seq_count
    assert len(input_data) == len(targets)

  input_data = np.asarray(input_data)
  targets = np.asarray(targets)
  if not (input_data.ndim == 4 and targets.ndim == 4):
    print input_data.ndim, targets.ndim
    raise DataProcessingError('Input data or target dimensions incorrect.')
  return input_data, targets


#def get_pianoroll_from_note_sequence_data(path, type_, len_from_beginning=None):
#  """Retrieves NoteSequences from a TFRecord and returns piano rolls.
#
#  Args:
#    path: The absolute path to the TFRecord file.
#    type_: The name of the TFRecord file which also specifies the type of data.
#
#  Yields:
#    3D binary numpy arrays.
#
#  Raises:
#    DataProcessingError: If the type_ specified is not one of train, test or
#        valid.
#  """
#  if type_ not in ['train', 'test', 'valid']:
#    raise DataProcessingError(
#        'Data is grouped by train, test or valid. Please specify one.')
#  fpath = os.path.join(path, '%s.tfrecord' % type_)
#  encoder = PianorollEncoderDecoder()
#  seq_reader = note_sequence_record_iterator(fpath)
#  for seq in seq_reader:
#    pianoroll = encoder.encode(seq)
#    if len_from_beginning is None:
#      yield pianoroll
#    elif pianoroll.shape[0] >= len_from_beginning:
#      yield pianoroll[:len_from_beginning]
#    else:
#      continue


DATASET_PARAMS = {
    'Nottingham': {
        'pitch_ranges': [31, 93], 'shortest_duration': 0.25, 'num_instruments': 9}, 
    'MuseData': {
        'pitch_ranges': [21, 105], 'shortest_duration': 0.25, 'num_instruments': 14},
    'Piano-midi.de': {
        'pitch_ranges': [21, 108], 'shortest_duration': 0.25, 'num_instruments': 12,
        'batch_size': 12},
    'JSB_Chorales': {
        'pitch_ranges': [43, 96], 'shortest_duration': 0.5, 'num_instruments': 4,
        'crop_piece_len': 32},
    '4part_Bach_chorales': {
        'pitch_ranges': [36, 88], 'shortest_duration': 0.125, 
        'relative_path': 'bach/qbm120/instrs=4_duration=0.125_sep=True'}
}


def get_data_as_pianorolls(basepath, hparams, fold):
  seqs, encoder = get_data_and_update_hparams(
      basepath, hparams, fold, update_hparams=False, return_encoder=True)
  return [encoder.encode(seq) for seq in seqs]


def get_data_and_update_hparams(basepath, hparams, fold, 
                                update_hparams=True, return_encoder=True):
  """Returns NoteSequences for '4_part_JSB_Chorales' and list of lists for the rest, and updates dataset specific hparams."""
  dataset_name = hparams.dataset
  separate_instruments = hparams.separate_instruments
  # TODO: Read dataset params from JSON file or the like.
  params = DATASET_PARAMS[dataset_name]
  pitch_range = params['pitch_ranges']
  if dataset_name == '4part_Bach_chorales':
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
      if key in hparams.__dict__:
        setattr(hparams, key, value)
    #FIXME: just for debug
    for key in ['batch_size', 'num_instruments', 'crop_piece_len']:
      if key in params:
        assert getattr(hparams, key) == params[key], 'hparams did not get updated.'
  if return_encoder:
    encoder = PianorollEncoderDecoder(
        shortest_duration=params['shortest_duration'],
        min_pitch=pitch_range[0],
        max_pitch=pitch_range[1],
        separate_instruments=separate_instruments,
        num_instruments=hparams.num_instruments,
        encode_silences=hparams.encode_silences)
  
  if return_encoder:
    return seqs, encoder
  else:
    return seqs


#def get_note_sequence_data(path, type_):
#  """Retrieves NoteSequences from a TFRecord.
#
#  Args:
#    path: The absolute path to the TFRecord file.
#    type_: The name of the TFRecord file which also specifies the type of data.
#
#  Yields:
#    NoteSequences.
#
#  Raises:
#    DataProcessingError: If the type_ specified is not one of train, test or
#        valid.
#  """
#  if type_ not in ['train', 'test', 'valid']:
#    raise DataProcessingError(
#        'Data is grouped by train, test or valid. Please specify one.')
#  fpath = os.path.join(path, '%s.tfrecord' % type_)
#  print 'fpath', fpath
#  #seq_reader = note_sequence_record_iterator(fpath)
#  #for seq in seq_reader:
#  #  yield seq
#  reader = tf.python_io.tf_record_iterator(fpath)
#  for serialized_sequence in reader:
#    yield music_pb2.NoteSequence.FromString(serialized_sequence)
