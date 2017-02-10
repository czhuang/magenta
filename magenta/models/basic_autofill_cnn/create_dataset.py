r"""Creates pianoroll dataset from NoteSequence tfrecords.

Example usage:
  $ bazel run :create_dataset -- --input=/tmp/note_sequences.tfrecord \
      --output_dir=/tmp/tensors  --separate_instruments=False \
      --num_instruments_requested=4
"""

import os

 

import tensorflow as tf

from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework.tensor_util import make_tensor_proto

from magenta.models.basic_autofill_cnn.pianorolls_lib import PianorollEncoderDecoder

FLAGS = tf.app.flags.FLAGS
# '/ /is-d/home/annahuang/ttl=100d/BachChorales/instrs=4_wo_16th/all_note_sequences.tfrecord'
# TODO(annahuang): Set the default input and output_dir to None for opensource.
# BachChorales-prog.tfrecord has part info, that corresponds to instrument, and also reassigned 4 voice prog
tf.app.flags.DEFINE_string('input', '/u/huangche/data/bach/bach_chorale_note_sequences.tfrecord',
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir',
                           '/u/huangche/data/bach/qbm120',
                           'Directory to write training, validation, test '
                           'TFRecord files.')
tf.app.flags.DEFINE_bool('separate_instruments', True,
                         'If true, creates a pianoroll by intrument type, '
                         'which results in a 3D tensor per piece. If False, '
                         'encodes all instruments in one pianoroll, resulting '
                         'in just one pianoroll.')
tf.app.flags.DEFINE_float('shortest_duration_allowed', 0.125,
                          'Only include pieces that do not have note '
                          'durations shorter than the requested value. '
                          'For qpm=120, notated quarter note equals 0.5.')
tf.app.flags.DEFINE_float('quantization_level', 0.125, 'Quantization duration.'
                          'For qpm=120, notated quarter note equals 0.5.')
tf.app.flags.DEFINE_integer('num_instruments_requested', 4,
                            'Only includes pieces with the requested number of '
                            'instruments. To include all pieces regardless of '
                            'the number of instruments used, type 0.')


class FilterByLength(pipeline.Pipeline):
  """Filter NoteSequences to return those with a minimum length."""

  def __init__(self, len_requested):
    super(FilterByLength, self).__init__(
      input_type=music_pb2.NoteSequence, output_type=music_pb2.NoteSequence)
    self.len_requested = len_requested
    

class FilterByNumOfVoices(pipeline.Pipeline):
  """Filter NoteSequences to return those with desired number of instruments."""

  def __init__(self, num_instruments_requested=4):
    super(FilterByNumOfVoices, self).__init__(
        input_type=music_pb2.NoteSequence, output_type=music_pb2.NoteSequence)
    self.num_instruments_requested = num_instruments_requested

  def transform(self, note_sequence, attribute='part'):
    """Only return a NoteSequence if it has the desired number of voices."""
    if not hasattr(note_sequence.notes[0], attribute) and attribute != 'instrument':
      attribute = 'instrument'
    if (len(set(getattr(note, attribute) for note in note_sequence.notes)) ==
        self.num_instruments_requested):
      self._set_stats(
          self._make_stats('%s_voices' % self.num_instruments_requested))
      return [note_sequence]
    else:
      self._set_stats(
          self._make_stats('not_%s_voices' % self.num_instruments_requested))
      return []

  def _make_stats(self, increment_key):
    """Increment the filtered and unfiltered number of pieces by 1."""
    return [statistics.Counter(increment_key, 1)]


class FilterByShortestDurationAllowed(pipeline.Pipeline):
  """Filter NoteSequences to exclude pieces with too short durations."""

  def __init__(self, shortest_duration_allowed=0.25):
    super(FilterByShortestDurationAllowed, self).__init__(
        input_type=music_pb2.NoteSequence, output_type=music_pb2.NoteSequence)
    self.shortest_duration_allowed = shortest_duration_allowed

  def transform(self, note_sequence):
    """Only return a NoteSequence if it doesn't have too short durations."""
    for note in note_sequence.notes:
      if note.end_time - note.start_time < self.shortest_duration_allowed:
        self._set_stats(
          self._make_stats('with_too_short_durations'))
        return []
    self._set_stats(
          self._make_stats('all_durations_long_enough'))
    return [note_sequence]

  def _make_stats(self, increment_key):
    """Increment the filtered and unfiltered number of pieces by 1."""
    return [statistics.Counter(increment_key, 1)]


class ToPianorollByInstrumentType(pipeline.Pipeline):
  """Transforms a NoteSequence into pianorolls with one per instrument type.

  Args:
  shortest_duration: A float of the shortest duration in the corpus, or None
      in which case shortest_duration will be identified in
      PianorollEncoderDecoder by iterating through all NoteSequences.
  min_pitch: An integer giving the lowest pitch in the corpus, or None in
      which case min_pitch will be identified in PianorollEncoderDecoder by
      iterating through all NoteSequences.
  max_pitch: An integer giving the highest pitch in the corpus, or None in
       which case max_pitch will be identified in PianorollEncoderDecoder by
       iterating through all NoteSequences.
  sequence_iterator: A TFRecord iterator that iterates through
      NoteSequences.
  separate_instruments: A boolean to indicate whether to encode one instrument
      per pianoroll.
  """

  # TODO(annahuang): Take quantized sequence as input to support midi input.
  def __init__(self,
               shortest_duration=None,
               min_pitch=None,
               max_pitch=None,
               sequence_iterator=None,
               separate_instruments=True):
    super(ToPianorollByInstrumentType, self).__init__(
        input_type=music_pb2.NoteSequence, output_type=tensor_pb2.TensorProto)
    self.pianoroll_encoder_decoder = PianorollEncoderDecoder(
        shortest_duration, min_pitch, max_pitch, sequence_iterator,
        separate_instruments, quantization_level=FLAGS.quantization_level)

  def transform(self, sequence):
    """Transform NoteSequence into pianorolls with one per instrument type.

    Args:
      sequence: A NoteSequence.

    Returns:
      A list of 3D tensors, with each 2D slice representing an instrument
          or group of instruments of the same type.
    """
    pianoroll = self.pianoroll_encoder_decoder.encode(sequence)
    return [make_tensor_proto(pianoroll)]


def get_pipeline(num_instruments_requested, shortest_duration_allowed, 
                 separate_instruments):
  """Returns data processing pipeline for score-based NoteSequences.

  This pipeline iterates through a NoteSequence TFRecord to return pianorolls of
      NoteSequences with a specific number of instruments and with a specific
      shortest duration. Running the pipeline outputs NoteSequence protos in
      separate train, validation and test TFRecords.

  Args:
    num_instruments_requested: An integer specifying the number of instruments
        a piece should have in order to be included in the dataset.
    shortest_duration_allowed: A float specifying the smallest allowed note
        duration in a piece to be included in the dataset.
    separate_instruments: A boolean to indicate whether to encode one instrument
        per pianoroll.

  Returns:
    A DAGPipeline.
  """
  # TODO(annahuang): Check NoteSequence to see that it is actually from score.
  filter_by_num_instruments = FilterByNumOfVoices(num_instruments_requested)
  filter_by_shortest_duration = FilterByShortestDurationAllowed(
      shortest_duration_allowed)
  partitioner = pipelines_common.RandomPartition(music_pb2.NoteSequence,
                                                 ['train', 'valid', 'test'],
                                                 [0.6, 0.2])
  dag = {filter_by_num_instruments: dag_pipeline.DagInput(music_pb2.NoteSequence),
         filter_by_shortest_duration: filter_by_num_instruments,
         partitioner: filter_by_shortest_duration,
         dag_pipeline.DagOutput(): partitioner}
  return dag_pipeline.DAGPipeline(dag)


def run_pipeline_from_flags():
  """Run the data processing pipline serially, with configs from FLAGS."""
  output_dir_name = 'instrs=%d_duration=%.3f_sep=%s' % (
      FLAGS.num_instruments_requested, FLAGS.shortest_duration_allowed,
      FLAGS.separate_instruments)
  output_fpath = os.path.join(FLAGS.output_dir, output_dir_name)
  if not os.path.exists(output_fpath):
    os.mkdir(output_fpath)
  dataset_pipeline = get_pipeline(
      FLAGS.num_instruments_requested, FLAGS.shortest_duration_allowed,
      FLAGS.separate_instruments)
  pipeline.run_pipeline_serial(
      dataset_pipeline,
      pipeline.tf_record_iterator(FLAGS.input, dataset_pipeline.input_type),
      output_fpath)


def main(unused_argv):
  """Set up the data processing pipeline."""
  if unused_argv[1:]:
    raise AttributeError('Misspecified flags.')
  if FLAGS.input is None:
    tf.logging.fatal('No input was provided.')
  tf.logging.set_verbosity(tf.logging.INFO)
  #fpath = os.path.join(tf.resource_loader.get_data_files_path(), FLAGS.input)
  run_pipeline_from_flags()


if __name__ == '__main__':
  tf.app.run()
