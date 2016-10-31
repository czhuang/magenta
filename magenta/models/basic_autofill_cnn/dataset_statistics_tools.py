"""One-line documentation for dataset_statistics_tools module.

A detailed description of dataset_statistics_tools.
"""
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import data_tools
#from magenta.models.basic_autofill_cnn import data_pipeline_tools
from magenta.models.basic_autofill_cnn import test_tools
from magenta.models.basic_autofill_cnn import pianorolls_lib

from magenta.lib.note_sequence_io import note_sequence_record_iterator
from magenta.lib.midi_io import sequence_proto_to_midi_file


def get_note_sequences():
  path = '/u/huangche/data/bach'
  fname = 'bach_chorale_note_sequences.tfrecord'
  return list(note_sequence_record_iterator(os.path.join(path, fname)))


def check_num_of_pieces_in_tfrecord():
  path = '/u/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True'
  fnames = ['train', 'valid', 'test']
  num_of_pieces = dict()
  for fname in fnames:
    fpath = os.path.join(path, fname + '.tfrecord')
    num_of_pieces[fname] = len(list(note_sequence_record_iterator(fpath)))
  print num_of_pieces.items()
  print np.sum(num_of_pieces.values())


def get_num_of_voices_hist(seqs=None):
  if seqs is None:
    seqs = get_note_sequences()
  voicing_counts = defaultdict(int)
  for seq in seqs:
    voicing = set(note.part for note in seq.notes)
    voicing_counts[tuple(voicing)] += 1
  for voicing, counts in voicing_counts.iteritems():
    print voicing, counts
  return voicing_counts


def get_4_voice_sequences(seqs=None):
  if seqs is None:
    seqs = get_note_sequences()
  four_voice_seqs = []
  for seq in seqs:
    if len(set(note.part for note in seq.notes)) == 4:
      four_voice_seqs.append(seq)
  return four_voice_seqs


def synth_start_of_note_sequences():
  path = '/data/lisatmp4/huangche/data/bach/midi/'
  seqs = get_4_voice_sequences()
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  synth_timesteps = 32
  short_seqs = []
  for seq in seqs:
    pianoroll = encoder.encode(seq)
    short_seq = encoder.decode(pianoroll[:synth_timesteps])
    fpath = os.path.join(path, seq.filename)
    
def get_duration_hist():
  seqs = get_note_sequences()
  get_num_of_voices_hist(seqs)
  seqs = get_4_voice_sequences(seqs)
  tempo_used = set()
  duration_counts = defaultdict(int)
  piece_with_such_duration = defaultdict(set)
  for seq in seqs:
   tempo_used.add(seq.tempos[0].qpm)
   for note in seq.notes:
     duration = note.end_time - note.start_time
     duration_counts[duration] += 1
     piece_with_such_duration[duration].add(seq.filename)
  sorted_durations = sorted(duration_counts, key=lambda x: x)
  for duration in sorted_durations:
    print duration, duration_counts[duration], len(piece_with_such_duration[duration])
  print tempo_used


#def check_tessitura_hist_per_voice():
#  ust checking range for now.
#  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
#  voice_tessitura efaultdict(list)
#  ggregate tessitura for each voice across pieces.
#  for seq in seqs_reader:
# voices est_tools.collect_sorted_voices(seq, 'program')
# #voices est_tools.collect_sorted_voices(seq, 'part')
# #print voices.keys()
# #assert sorted(voices.keys()) == range(4)
# assert set(voices.keys()) == set([74, 72, 69, 71])
# pitch_duration_by_voice efaultdict(list)
# for part_index, notes in voices.iteritems():
#   pitches ]
#   durations ]
#   for note in notes:
#  pitches.append(note.pitch)
#  durations.append(note.end_time ote.start_time)
#   ggregate itch as the approximate tessitura.
#   voice_tessitura[part_index].append(np.ceil(
#    np.average(pitches, weights=durations)))
#  rint historgram of tessitura.
#  for part_index in [74, 72, 69, 71]:
# pitches, counts p.unique(voice_tessitura[part_index], return_counts=True)
# print '\nPart %d:' art_index
# print 'pitches', pitches
# print 'counts', counts
# sorted_indices p.argsort(pitches)
#
# #for index in sorted_indices:
# #  print '(%d, %d)' pitches[index], counts[index]),
#
#
def check_tessitura_ordering_hist():
  print 'check_tessitura_ordering_hist'
  seqs = get_4_voice_sequences()
  print 'len(seqs)', len(seqs)
  ordering_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
  ordering_counts = defaultdict(int)
  # aggregate tessitura for each voice across pieces.
  pieces = []
  for seq in seqs:
    voices = test_tools.collect_sorted_voices(seq, 'part')
    pieces.append(voices)

  tessitura = defaultdict(list)
  for voices in pieces:
    for part_index, notes in voices.iteritems():
      pitches = []
      durations = []
      for note in notes:
        pitches.append(note.pitch)
        durations.append(note.end_time-note.start_time)
      #aggregate itch as the approximate tessitura.
      tessitura[part_index].append(np.ceil(
          np.average(pitches, weights=durations)))

  for part_index, tess in tessitura.items():
    print part_index, len(tess)
 
  # Check tessitura ordering.
  for top_index, bottom_index in ordering_pairs:
    for piece_index in range(len(pieces)):
      if (tessitura[top_index][piece_index] 
        > tessitura[bottom_index][piece_index]):
        ordering_counts[(top_index, bottom_index)] += 1

  for ordering_pair in ordering_pairs:
    print ordering_pair, ordering_counts[ordering_pair]
#
#
#def check_voices():
#  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
#  for seq in seqs_reader:
# voices est_tools.collect_sorted_voices(seq, 'program')
# print voices.keys()


def main(unused_argv):
  #get_duration_hist()
  #check_num_of_pieces_in_tfrecord()
  #check_tessitura_hist_per_voice()
  #check_voices()
  #check_tessitura_ordering_hist()
  synth_start_of_note_sequences()

if __name__ == '__main__':
  tf.app.run()

