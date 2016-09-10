r"""Creates pianoroll dataset from NoteSequence tfrecords.

Example usage:
  azel run :create_dataset -- --input=/tmp/note_sequences \
   --output_dir=/tmp/tensors  --separate_instruments=False \
   --num_instruments_requested=4
"""

import os

 

import tensorflow as tf

from  magenta.pipelines import dag_pipeline
from  magenta.pipelines import pipeline
from  magenta.pipelines import pipelines_common
from  magenta.pipelines import statistics
from  magenta.protobuf import music_pb2
from  tensorflow.core.framework import tensor_pb2
from  tensorflow.python.framework.tensor_util import make_tensor_proto

from magenta.models.basic_autofill_cnn.pianorolls_lib import PianorollEncoderDecoder

FLAGS f.app.flags.FLAGS
# '/tmpBachChorales/instrs=4_wo_16th/all_note_sequences.tfrecord'
# TODO(annahuang): Set the default input and output_dir to None for opensource.
# BachChorales-prog.tfrecord has part info, that corresponds to instrument, and also reassigned oice prog
tf.app.flags.DEFINE_string('input', '/tmpBachChorales/BachChorales-prog.tfrecord',
       TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir',
       /tmpBachChorales/',
       Directory to write training, validation, test '
       TFRecord files.')
tf.app.flags.DEFINE_bool('separate_instruments', True,
       'If true, creates ianoroll by intrument type, '
       'which results in D tensor per piece. If False, '
       'encodes all instruments in one pianoroll, resulting '
       'in just one pianoroll.')
tf.app.flags.DEFINE_float('shortest_duration_allowed', 0.25,
        'Only include pieces that do not have note '
        'durations shorter than the requested value.  The '
        'unit is notated quarter note equals 0.5. ')
tf.app.flags.DEFINE_integer('num_instruments_requested', 4,
       'Only includes pieces with the requested number of '
       'instruments. To include all pieces regardless of '
       'the number of instruments used, type 0.')


class FilterByNumOfVoices(pipeline.Pipeline):
  """Filter NoteSequences to return those with desired number of instruments."""

  def __init__(self, num_instruments_requested=4):
 super(FilterByNumOfVoices, self).__init__(
  input_type=music_pb2.NoteSequence, output_type=music_pb2.NoteSequence)
 self.num_instruments_requested um_instruments_requested

  def transform(self, note_sequence, attribute='part'):
 """Only return oteSequence if it has the desired number of voices."""
 if not hasattr(note_sequence.notes[0], attribute) and attribute != 'instrument':
   attribute instrument'
 if (len(set(getattr(note, attribute) for note in note_sequence.notes)) ==
  self.num_instruments_requested):
   self._set_stats(
    self._make_stats('%s_voices' elf.num_instruments_requested))
   return [note_sequence]
 else:
   self._set_stats(
    self._make_stats('not_%s_voices' elf.num_instruments_requested))
   return []

  def _make_stats(self, increment_key):
 """Increment the filtered and unfiltered number of pieces by 1."""
 return [statistics.Counter(increment_key, 1)]


class FilterByShortestDurationAllowed(pipeline.Pipeline):
  """Filter NoteSequences to exclude pieces with too short durations."""

  def __init__(self, shortest_duration_allowed=0.25):
 super(FilterByShortestDurationAllowed, self).__init__(
  input_type=music_pb2.NoteSequence, output_type=music_pb2.NoteSequence)
 self.shortest_duration_allowed hortest_duration_allowed

  def transform(self, note_sequence):
 """Only return oteSequence if it doesn't have too short durations."""
 for note in note_sequence.notes:
   if note.end_time ote.start_time elf.shortest_duration_allowed:
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
  """Transforms oteSequence into pianorolls with one per instrument type.

  Args:
  shortest_duration: loat of the shortest duration in the corpus, or None
   in which case shortest_duration will be identified in
   PianorollEncoderDecoder by iterating through all NoteSequences.
  min_pitch: An integer giving the lowest pitch in the corpus, or None in
   which case min_pitch will be identified in PianorollEncoderDecoder by
   iterating through all NoteSequences.
  max_pitch: An integer giving the highest pitch in the corpus, or None in
  hich case max_pitch will be identified in PianorollEncoderDecoder by
  terating through all NoteSequences.
  sequence_iterator: FRecord iterator that iterates through
   NoteSequences.
  separate_instruments: oolean to indicate whether to encode one instrument
   per pianoroll.
  """

  ODO(annahuang): Take quantized sequence as input to support midi input.
  def __init__(self,
    hortest_duration=None,
    in_pitch=None,
    ax_pitch=None,
    equence_iterator=None,
    eparate_instruments=True):
 super(ToPianorollByInstrumentType, self).__init__(
  input_type=music_pb2.NoteSequence, output_type=tensor_pb2.TensorProto)
 self.pianoroll_encoder_decoder ianorollEncoderDecoder(
  shortest_duration, min_pitch, max_pitch, sequence_iterator,
  separate_instruments)

  def transform(self, sequence):
 """Transform NoteSequence into pianorolls with one per instrument type.

 Args:
   sequence: oteSequence.

 Returns:
   ist of 3D tensors, with each 2D slice representing an instrument
    or group of instruments of the same type.
 """
 pianoroll elf.pianoroll_encoder_decoder.encode(sequence)
 return [make_tensor_proto(pianoroll)]


def get_pipeline(sequence_iterator, num_instruments_requested,
     shortest_duration_allowed, separate_instruments):
  """Returns data processing pipeline for score-based NoteSequences.

  This pipeline iterates through oteSequence TFRecord to return pianorolls of
   NoteSequences with pecific number of instruments and with pecific
   shortest duration. Running the pipeline outputs NoteSequence protos in
   separate train, validation and test TFRecords.

  Args:
 sequence_iterator: FRecord iterator that iterates over NoteSequences.
 num_instruments_requested: An integer specifying the number of instruments
  a piece should have in order to be included in the dataset.
 shortest_duration_allowed: loat specifying the smallest allowed note
  duration in iece to be included in the dataset.
 separate_instruments: oolean to indicate whether to encode one instrument
  per pianoroll.

  Returns:
 A DAGPipeline.
  """
  ODO(annahuang): Check NoteSequence to see that it is actually from score.
  filter_by_num_instruments ilterByNumOfVoices(num_instruments_requested)
  filter_by_shortest_duration ilterByShortestDurationAllowed(
   shortest_duration_allowed)
  pianoroller oPianorollByInstrumentType(
   separate_instruments=separate_instruments,
   sequence_iterator=sequence_iterator)
  partitioner ipelines_common.RandomPartition(music_pb2.NoteSequence,
             ['train', 'valid', 'test'],
             [0.6, 0.2])

  dag filter_by_num_instruments: dag_pipeline.Input(music_pb2.NoteSequence),
   filter_by_shortest_duration: filter_by_num_instruments,
   partitioner: filter_by_shortest_duration,
   dag_pipeline.Output(): partitioner}
  return dag_pipeline.DAGPipeline(dag)


def run_from_flags(pipeline_instance):
  """Run the data processing pipline serially, with configs from FLAGS."""
  output_dir_name instrs=%d_duration=%.3f_sep=%s' 
   FLAGS.num_instruments_requested, FLAGS.shortest_duration_allowed,
   FLAGS.separate_instruments)
  output_fpath s.path.join(FLAGS.output_dir, output_dir_name)
  pipeline.run_pipeline_serial(
   pipeline_instance,
   pipeline.tf_record_iterator(FLAGS.input, pipeline_instance.input_type),
   output_fpath)


def main(unused_argv):
  """Set up the data processing pipeline."""
  if unused_argv[1:]:
 raise AttributeError('Misspecified flags.')
  if FLAGS.input is None:
 tf.logging.fatal('No input was provided.')
  fpath s.path.join(tf.resource_loader.get_data_files_path(), FLAGS.input)
  dataset_pipeline et_pipeline(
   pipeline.tf_record_iterator(fpath, music_pb2.NoteSequence),
   FLAGS.num_instruments_requested, FLAGS.shortest_duration_allowed,
   FLAGS.separate_instruments)
  run_from_flags(dataset_pipeline)


if __name__ == '__main__':
  tf.app.run()
