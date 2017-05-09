import os, sys, gzip, cPickle as pkl
import numpy as np
from magenta.models.basic_autofill_cnn import util

def main():
  for path in sys.argv[1:]:
    d = np.load(path)
    for i, pianoroll in enumerate(d["pianorolls"]):
      midi_data = pianoroll_to_midi(pianoroll)
      midi_path = os.path.join(path, "%s_%i.midi" % (path, i))
      print midi_path
      midi_data.write(midi_path)

# NOTE: assumes four separate instruments ordered high to low
def pianoroll_to_midi(x):
  import pretty_midi
  midi_data = pretty_midi.PrettyMIDI()
  programs = [69, 70, 72, 71]
  pitch_offset = 36
  bpm = 120.
  duration = bpm / 60 / 16.
  T, P, I = x.shape
  for i in range(I):
    notes = []
    for p in range(P):
      for t in range(T):
        if x[t, p, i]:
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
