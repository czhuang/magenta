"""One-line documentation for pipeline_tools module.

A detailed description of pipeline_tools.
"""

from collections import defaultdict

import os

 

import tensorflow as tf

from  magenta.pipelines import dag_pipeline
from  magenta.pipelines import pipeline
from  magenta.pipelines import pipelines_common
from magenta.models.basic_autofill_cnn.create_dataset import FilterByNumOfVoices
from magenta.models.basic_autofill_cnn.create_dataset import FilterByShortestDurationAllowed
from  magenta.protobuf import music_pb2
from  magenta.lib.note_sequence_io import NoteSequenceRecordWriter
from  magenta.lib.note_sequence_io import note_sequence_record_iterator
from magenta.models.basic_autofill_cnn import config_tools


def filter_by_num_of_voices(input_, output_fpath, num_instruments_requested,
       shortest_duration_allowed):
  filter_by_num_instruments ilterByNumOfVoices(num_instruments_requested)
  #dag filter_: dag_pipeline.Input(music_pb2.NoteSequence),
    dag_pipeline.Output(output_fpath): filter_}
  filter_by_shortest_duration ilterByShortestDurationAllowed(
   shortest_duration_allowed)
  partitioner ipelines_common.RandomPartition(music_pb2.NoteSequence,
             ['train', 'valid', 'test'],
             [0.6, 0.2])
  dag filter_by_num_instruments: dag_pipeline.Input(music_pb2.NoteSequence),
   filter_by_shortest_duration: filter_by_num_instruments,
   partitioner: filter_by_shortest_duration,
   dag_pipeline.Output(): partitioner}
  pipeline_instance ag_pipeline.DAGPipeline(dag)
  pipeline.run_pipeline_serial(
   pipeline_instance,
   pipeline.tf_record_iterator(input_, pipeline_instance.input_type),
   output_fpath)


def make_bach_chorales_with_4_voices_dataset():
  input_ /tmpBachChorales/BachChorales-prog.tfrecord'
  frecord will be appended to this
  output_fpath /tmpBachChorales/instrs=4_note_sequence'
  num_instruments_requested 
  shortest_duration_allowed .25
  filter_by_num_of_voices(input_, output_fpath, num_instruments_requested,
        shortest_duration_allowed)


def check_num_voices_encoded_by_part():
  input_ /tmpBachChorales/BachChorales-prog.tfrecord'
  seq_reader ote_sequence_record_iterator(input_)
  for seq in seq_reader:
 print set(getattr(note, 'part') for note in seq.notes)
 print set(getattr(note, 'instrument') for note in seq.notes)
 print set(getattr(note, 'program') for note in seq.notes)


def get_dataset_gross_stats(path):
  seqs_reader et_note_sequence_reader(path)
  piece_count 
  total_time 
  total_num_notes 
  stats efaultdict(float)
  for seq in seqs_reader:
 stats['piece_count'] += 1
 # Total time is with bpm=120, quarter notes as 0.5.
 stats['total_time'] += seq.total_time
 stats['total_num_notes'] += len(seq.notes)
  for key, val in stats.iteritems():
 print '%s: %d' key, val)
  return stats


def filter_pieces_with_16th_notes_within_4_voice_dataset():
  seqs_reader et_bach_chorales_with_4_voices_dataset()
  output_fpath 
  '/tmpBachChorales/instrs=4_wo_16th/all_note_sequences.tfrecord'
  pieces_with_16th_notes efaultdict(int)
  pieces_wo_16th_notes ]

  for seq in seqs_reader:
 without_16s rue
 for note in seq.notes:
   if note.end_time ote.start_time == 0.125:
  pieces_with_16th_notes[seq.filename] += 1
  without_16s alse
  break
 if without_16s:
   pieces_wo_16th_notes.append(seq)

  for key, val in pieces_with_16th_notes.iteritems():
 print key, val

  with NoteSequenceRecordWriter(output_fpath) as writer:
 for seq in pieces_wo_16th_notes:
   writer.write(seq)

  get_dataset_gross_stats(output_fpath)


#def check_no_16th_notes


def get_bach_chorales_with_4_voices_dataset():
  #fpath /tmpBachChorales/instrs=4_wo_16th/all_note_sequences.tfrecord'
  #fpath /usr/local/ /home/annahuang/magenta_tmp/note_sequences/instrs=4_wo_16th/all_note_sequences.tfrecord'
  fpath /usr/local/ /home/annahuang/magenta_tmp/note_sequence_data/instrs=4_duration=0.250_sep=True/train.tfrecord'
  #fpath /usr/local/ /home/annahuang/magenta_tmp/note_sequence_data/instrs=4_duration=0.250_sep=True/valid.tfrecord'
  #fpath /usr/local/ /home/annahuang/magenta_tmp/note_sequence_data/instrs=4_duration=0.250_sep=True/test.tfrecord'
  return get_note_sequence_reader(fpath)

def get_small_4_voice_dataset():
  path /usr/local/ /home/annahuang/magenta_tmp/note_sequence_data/instrs=4_duration=0.250_sep=True'
  for type_ in ['train', 'valid', 'test']:
 fpath s.path.join(path, '%s.tfrecord' ype_)
 seqs ist(get_note_sequence_reader(fpath))
 output_fpath s.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb', '%s.tfrecord' ype_)
 with NoteSequenceRecordWriter(output_fpath) as writer:
   for seq in seqs[:3]:
  print set(note.program for note in seq.notes)
  print set(note.part for note in seq.notes)
  writer.write(seq)


def get_path_small_bach_chorales_with_4_voices_dataset():
  return os.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb',
  '6_note_sequences_with_only_4_voices.tfrecord')

def make_small_bach_chorales_with_4_voices_dataset():
  seqs et_bach_chorales_with_4_voices_dataset()
  num_seq 
  seqs_to_save ]
  for i, seq in enumerate(seqs):
 if = num_seq:
   break
 seqs_to_save.append(seq)

  output_fpath et_path_small_bach_chorales_with_4_voices_dataset()
  print 'output_fpath', output_fpath
  with NoteSequenceRecordWriter(output_fpath) as writer:
 for seq in seqs_to_save:
   writer.write(seq)

def get_small_bach_chorales_with_4_voices_dataset():
  input_fpath et_path_small_bach_chorales_with_4_voices_dataset()
  return get_note_sequence_reader(input_fpath)


def check_small_bach_chorales_with_4_voices_dataset():
  input_fpath et_path_small_bach_chorales_with_4_voices_dataset()
  get_dataset_gross_stats(input_fpath)


def get_all_note_sequence_data():
  fpath /tmpBachChorales/BachChorales-prog.tfrecord'
  return get_note_sequence_reader(fpath)


def get_note_sequence_reader(fpath):
  reader f.python_io.tf_record_iterator(fpath)
  for serialized_sequence in reader:
 yield music_pb2.NoteSequence.FromString(serialized_sequence)


def main(argv):
  #make_bach_chorales_with_4_voices_dataset()
  #filter_pieces_with_16th_notes_within_4_voice_dataset()

  #output_fpath 
  #'/tmpBachChorales/instrs=4_wo_16th/all_note_sequences.tfrecord'
  #get_dataset_gross_stats(output_fpath)

  #input_ /tmpBachChorales/instrs=4_note_sequence.tfrecord'
  #get_dataset_gross_stats(input_)

  #make_small_bach_chorales_with_4_voices_dataset()
  #check_small_bach_chorales_with_4_voices_dataset()

  #make_bach_chorales_with_4_voices_dataset()
  #check_num_voices_encoded_by_part()

  get_small_4_voice_dataset()

if __name__ == '__main__':
  tf.app.run()

