"""One-line documentation for check_midi module.

A detailed description of check_midi.
"""

import os

 

import numpy as np
import tensorflow as tf
from  magenta.lib import midi_io


def check_num_voices():
  fnames s.listdir(path)
  seqences ]
  num_instrs ]
  num_programs ]
  for fname in fnames:
 sequence idi_io.midi_to_sequence_proto(fname)
 num_instrs.append(len(set(note.instrument for note in sequence.notes)))
 num_programs.append(len(set(note.program for note in sequence.notes)))
 sequences.append(sequence)
  print '# of seq:', len(sequences)
  print num_instrs
  print num_programs

def main(argv):
  check_num_voices()


if __name__ == '__main__':
  tf.app.run()

