from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, gzip, cPickle as pkl
import pretty_midi
import numpy as np
import util


def main():
  for path in sys.argv[1:]:
    d = np.load(path)
    dirname = os.path.splitext(os.path.basename(path))[0]
    basepath = os.path.splitext(path)[0]
    if not os.path.isabs(path):
      basepath = os.path.join(os.getcwd(), basepath)
    if not os.path.exists(basepath):
      os.makedirs(basepath)
    for i, pianoroll in enumerate(d["pianorolls"]):
      midi_data = pianoroll_to_midi(pianoroll)
      midi_path = os.path.join(basepath, "%s_%i.midi" % (dirname, i))
      print(midi_path)
      midi_data.write(midi_path)


# NOTE: assumes four separate instruments ordered high to low
def pianoroll_to_midi(pianoroll, quantization_level=0.125, 
                      pitch_offset=36, qpm=60.):
  midi_data = pretty_midi.PrettyMIDI(initial_tempo=qpm)
  programs = [69, 70, 72, 71]
  # FIXME: currently 0.125 is 16th note, since qpm is qpm quarter note should be 1 instead of 0.5.
  duration = qpm / 60 * (quantization_level * 2)  
  T, P, I = pianoroll.shape
  for i in range(I):
    notes = []
    for p in range(P):
      for t in range(T):
        if pianoroll[t, p, i]:
          notes.append(pretty_midi.Note(velocity=100,
                                        pitch=pitch_offset + p,
                                        start=t * duration,
                                        end=(t + 1) * duration))
    notes = merge_held(notes)

    instrument = pretty_midi.Instrument(program=programs[i] - 1)
    instrument.notes.extend(notes)
    midi_data.instruments.append(instrument)
  return midi_data


def merge_held(notes):
  notes = list(notes)
  i = 1
  while i < len(notes):
    if (notes[i].pitch == notes[i - 1].pitch and
        notes[i].start == notes[i - 1].end):
      notes[i - 1].end = notes[i].end
      del notes[i]
    else:
      i += 1
  return notes


if __name__ == "__main__":
  main()
