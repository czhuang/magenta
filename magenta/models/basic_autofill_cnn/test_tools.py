"""Utility functions for testing."""

import os

from collections import defaultdict

 

import numpy as np
import tensorflow as tf

from magenta.protobuf import music_pb2

from magenta.models.basic_autofill_cnn import basic_autofill_cnn_graph


def generate_random_data(hparams):
  """Generate random input and target data according to hyperparameters."""
  num_timesteps, num_pitches, input_depth = hparams.input_data_shape
  prediction_threshold = hparams.prediction_threshold
  # Generate random data.
  input_data = np.random.random((2, num_timesteps, num_pitches, input_depth))
  input_data[input_data > prediction_threshold] = 1
  input_data[input_data <= prediction_threshold] = 0
  targets = input_data[:, :, :, :input_depth / 2].copy()
  # Mask the data.
  input_data[:, :, :, :input_depth / 2] *= 1 - input_data[:, :, :, input_depth /
                                                          2:]
  return input_data, targets

def get_note_sequence_reader(fpath):
  reader = tf.python_io.tf_record_iterator(fpath)
  for serialized_sequence in reader:
    yield music_pb2.NoteSequence.FromString(serialized_sequence)


def get_note_sequences_():
  fpath = '/ /is-d/home/annahuang/ttl=100d/BachChorales/instrs=4_note_sequence.tfrecord'
  return get_note_sequence_reader(fpath)


def get_small_bach_chorales_with_4_voices_dataset():
  fpath = os.path.join(tf.resource_loader.get_data_files_path(), 'testdata',
                       'jsb', '6_note_sequences_with_only_4_voices.tfrecord')
  return get_note_sequence_reader(fpath)


#def get_four_part_sequences():
#  fpath = '/u/huangche/data/bach/bach_chorale_note_sequences.tfrecord'
#  seq_reader = get_note_sequence_reader(fpath)
#  seqs = list(seq_reader)
#  print '# of raw seqs', len(seqs)
#  four_part_seqs = []
#  for seq in seqs:
#    if set(note.part for note in seq.notes) == 4:
#      four_part_seqs.append(seq)
#  return four_part_seqs


def get_bach_chorales_with_4_voices_dataset():
  input_ = '/ /is-d/home/annahuang/ttl=100d/BachChorales/instrs=4_note_sequence.tfrecord'
  return get_note_sequence_reader(input_)


def get_num_repeated_notes(seq):
  print '---seq'
  parts = collect_sorted_voices(seq, 'program')
  num_notes_repeated = 0
  for notes in parts.values():
    for i, note in enumerate(notes[:-1]):
      next_note = notes[i + 1]
      if (note.end_time == next_note.start_time and
          note.pitch == next_note.pitch):
        print i, i + 1
        print note, next_note
        num_notes_repeated += 1
  return num_notes_repeated


def get_corpus_num_repeated_notes():
  seq_reader = get_note_sequences()
  num_notes_repeated = 0
  seq_count = 0
  for seq in seq_reader:
    seq_count += 1
    note_repeated_count = get_num_repeated_notes(seq)
    print '-------note_repeated_count:', note_repeated_count
    num_notes_repeated += note_repeated_count
  print '# of seqs:', seq_count
  print '# of notes repeated:', num_notes_repeated


def collect_sorted_voices(seq, by_attribute):
  voices = defaultdict(list)
  for note in seq.notes:
    voices[getattr(note, by_attribute)].append(note)
  sorted_voices = {}
  for key, notes in voices.iteritems():
    sorted_voices[key] = sorted(notes, key=lambda x: x.start_time)
  return sorted_voices


def init_model(config):
  """Build graph, instantiate session, init model, and return all wrapped."""
  wrapped_model = basic_autofill_cnn_graph.build_graph(
      is_training=True, config=config)
  with wrapped_model.graph.as_default():
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    wrapped_model.sess = sess
  return wrapped_model


def main(unused_argv):
  get_corpus_num_repeated_notes()


if __name__ == '__main__':
  tf.app.run()
