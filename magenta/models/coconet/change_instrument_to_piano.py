
import pretty_midi
import os

def to_piano(midi_fpath, target_path):
  midi = pretty_midi.PrettyMIDI(midi_fpath)
  target_midi = pretty_midi.PrettyMIDI(
      initial_tempo=30.)
  instrument = pretty_midi.Instrument(program=0)
#  for instr in midi.instruments:
#    instrument.notes.extend(instr.notes) 
  slow_rate = 2.
  for i, instr in enumerate(midi.instruments):
    for note in instr.notes:
      print note.start, note.end
      note.start = note.start * slow_rate
      note.end = note.end * slow_rate
      if i == 0:
        note.velocity = 90
      else:
        note.velocity = 70
      instrument.notes.append(note)

  target_midi.instruments.append(instrument)
  _, fname = os.path.split(midi_fpath)
  target_fpath = os.path.join(target_path, fname)
  target_midi.write(target_fpath)
  print 'get_tempo_changes', target_midi.get_tempo_changes()

def main():
  source_path = '/Users/czhuang/repos/coconet_samples/ode_to_joy-coconet/for_send'
  target_path = '/Users/czhuang/repos/coconet_samples/ode_to_joy-coconet/for_send/all_piano'
  for fname in os.listdir(source_path):
    source_fpath = os.path.join(source_path, fname)
    print source_fpath
    if os.path.isdir(source_fpath):
      continue
    to_piano(source_fpath, target_path)


if __name__ == '__main__':
  main()

