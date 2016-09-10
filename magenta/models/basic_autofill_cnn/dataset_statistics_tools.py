"""One-line documentation for dataset_statistics_tools module.

A detailed description of dataset_statistics_tools.
"""

from collections import defaultdict

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import data_tools
from magenta.models.basic_autofill_cnn import data_pipeline_tools
from magenta.models.basic_autofill_cnn import test_tools


def get_duration_histogram():
  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
  duration_counts efaultdict(int)
  for seq in seqs_reader:
 for note in seq.notes:
   duration_counts[note.end_time ote.start_time] += 1
  sorted_durations orted(duration_counts, key=lambda x: x)
  for duration in sorted_durations:
 print duration, duration_counts[duration]


def check_tessitura_histogram_per_voice():
  ust checking range for now.
  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
  voice_tessitura efaultdict(list)
  ggregate tessitura for each voice across pieces.
  for seq in seqs_reader:
 voices est_tools.collect_sorted_voices(seq, 'program')
 #voices est_tools.collect_sorted_voices(seq, 'part')
 #print voices.keys()
 #assert sorted(voices.keys()) == range(4)
 assert set(voices.keys()) == set([74, 72, 69, 71])
 pitch_duration_by_voice efaultdict(list)
 for part_index, notes in voices.iteritems():
   pitches ]
   durations ]
   for note in notes:
  pitches.append(note.pitch)
  durations.append(note.end_time ote.start_time)
   ggregate itch as the approximate tessitura.
   voice_tessitura[part_index].append(np.ceil(
    np.average(pitches, weights=durations)))
  rint historgram of tessitura.
  for part_index in [74, 72, 69, 71]:
 pitches, counts p.unique(voice_tessitura[part_index], return_counts=True)
 print '\nPart %d:' art_index
 print 'pitches', pitches
 print 'counts', counts
 sorted_indices p.argsort(pitches)

 #for index in sorted_indices:
 #  print '(%d, %d)' pitches[index], counts[index]),


def check_tessitura_ordering_histogram():
  ust checking range for now.
  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
  ordering_pairs (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
  ordering_counts efaultdict(int)
  ggregate tessitura for each voice across pieces.
  for seq in seqs_reader:
 voices est_tools.collect_sorted_voices(seq, 'program')
 voice_indices 74, 72, 69, 71]
 assert set(voices.keys()) == set(voice_indices)

 tessitura efaultdict(list)
 for part_index, notes in voices.iteritems():
   pitches ]
   durations ]
   for note in notes:
  pitches.append(note.pitch)
  durations.append(note.end_time ote.start_time)
   ggregate itch as the approximate tessitura.
   tessitura[part_index].append(np.ceil(
    np.average(pitches, weights=durations)))
 # Check tessitura ordering.
 for top_index, bottom_index in ordering_pairs:
   top_program oice_indices[top_index]
   bottom_program oice_indices[bottom_index]
   if tessitura[top_program] essitura[bottom_program]:
  ordering_counts[(top_index, bottom_index)] += 1
  rint ordering count.
  for ordering_pair in ordering_pairs:
 print ordering_pair, ordering_counts[ordering_pair]


def check_voices():
  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
  for seq in seqs_reader:
 voices est_tools.collect_sorted_voices(seq, 'program')
 print voices.keys()


def main(unused_argv):
  get_duration_histogram()
  #check_tessitura_histogram_per_voice()
  #check_voices()
  #check_tessitura_ordering_histogram()


if __name__ == '__main__':
  tf.app.run()

