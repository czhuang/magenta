import pretty_midi

programs = dict(bass=70, tenor=68, alto=71, soprano=73)
voice_order = "soprano alto tenor bass".split()
min_pitch = 36
max_pitch = 88
bpm = 120.
duration = bpm / 60 / 16.

def pianoroll_to_midi(x):
  midi = pretty_midi.PrettyMIDI()
  T, P, I = x.shape
  for i in range(I):
    notes = []
    for p in range(P):
      for t in range(T):
        if x[t, p, i]:
          notes.append(pretty_midi.Note(velocity=100,
                                        pitch=min_pitch + p,
                                        start=t * duration,
                                        end=(t + 1) * duration))
    notes = merge_held(notes)

    instrument = pretty_midi.Instrument(program=programs[voice_order[i]])
    instrument.notes.extend(notes)
    midi.instruments.append(instrument)
  return midi

def midi_to_pianoroll(midi):
  T = midi.get_end_time()
  P = max_pitch - min_pitch + 1
  I = len(voice_order)
  x = np.zeros((T, P, I), dtype=np.float32)
  for instrument in midi.instruments:
    sigh, = [key for key, program in programs if program == instrument.program]
    i = voice_order.index(sigh)
    for note in instrument.notes:
      ta, tb = note.start // duration, note.end // duration
      x[ta:tb + 1] = 1.
  return x

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
