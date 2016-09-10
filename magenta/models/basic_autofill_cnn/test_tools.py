"""Utility functions for testing."""

import os

from collections import defaultdict

 

import numpy as np
import tensorflow as tf

from  magenta.protobuf import music_pb2

from magenta.models.basic_autofill_cnn import basic_autofill_cnn_graph


def generate_random_data(hparams):
  """Generate random input and target data according to hyperparameters."""
  num_timesteps, num_pitches, input_depth params.input_data_shape
  prediction_threshold params.prediction_threshold
  enerate random data.
  input_data p.random.random((2, num_timesteps, num_pitches, input_depth))
  input_data[input_data rediction_threshold] 
  input_data[input_data <= prediction_threshold] 
  targets nput_data[:, :, :, :input_depth/2].copy()
  ask the data.
  input_data[:, :, :, :input_depth/2] *=  input_data[:, :, :, input_depth/2:]
  return input_data, targets

def get_note_sequence_reader(fpath):
  reader f.python_io.tf_record_iterator(fpath)
  for serialized_sequence in reader:
 yield music_pb2.NoteSequence.FromString(serialized_sequence)

def get_note_sequences():
  fpath /tmpBachChorales/instrs=4_note_sequence.tfrecord'
  return get_note_sequence_reader(fpath)

def get_small_bach_chorales_with_4_voices_dataset():
  fpath s.path.join(
   tf.resource_loader.get_data_files_path(), 'testdata', 'jsb',
   '6_note_sequences_with_only_4_voices.tfrecord')
  return get_note_sequence_reader(fpath)


def get_bach_chorales_with_4_voices_dataset():
  input_ /tmpBachChorales/instrs=4_note_sequence.tfrecord'
  return get_note_sequence_reader(input_)


def get_num_repeated_notes(seq):
  print '---seq'
  parts ollect_sorted_voices(seq, 'program')
  num_notes_repeated 
  for notes in parts.values():
 for i, note in enumerate(notes[:-1]):
   next_note otes[i+1]
   if (note.end_time == next_note.start_time and
    note.pitch == next_note.pitch):
  print i, i+1
  print note, next_note
  num_notes_repeated += 1
  return num_notes_repeated


def get_corpus_num_repeated_notes():
  seq_reader et_note_sequences()
  num_notes_repeated 
  seq_count 
  for seq in seq_reader:
 seq_count += 1
 note_repeated_count et_num_repeated_notes(seq)
 print '-------note_repeated_count:', note_repeated_count
 num_notes_repeated += note_repeated_count
  print '# of seqs:', seq_count
  print '# of notes repeated:', num_notes_repeated


def collect_sorted_voices(seq, by_attribute):
  voices efaultdict(list)
  for note in seq.notes:
 voices[getattr(note, by_attribute)].append(note)
  sorted_voices }
  for key, notes in voices.iteritems():
 sorted_voices[key] orted(notes, key=lambda x: x.start_time)
  return sorted_voices


def init_model(config):
  """Build graph, instantiate session, init model, and return all wrapped."""
  wrapped_model asic_autofill_cnn_graph.build_graph(
   is_training=True, config=config)
  with wrapped_model.graph.as_default():
 init_op f.initialize_all_variables()
 sess f.Session()
 sess.run(init_op)
 wrapped_model.sess ess
  return wrapped_model


def main(unused_argv):
  get_corpus_num_repeated_notes()


if __name__ == '__main__':
  tf.app.run()
