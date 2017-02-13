import numpy as np, sys, matplotlib.pyplot as plt, os

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

def pianoroll_to_midi(x):
  import pretty_midi
  midi_data = pretty_midi.PrettyMIDI()
  programs = [74, 72, 69, 71]
  # For sigmoid generation, all voices are on the piano.
  #programs = [1]  
  pitch_offset = 36
  bpm = 120.
  duration = bpm / 60 / 16.
  I, P, T = x.shape
  for i in range(I):
    notes = []
    for p in range(P):
      for t in range(T):
        if x[i, p, t]:
          notes.append(pretty_midi.Note(velocity=100,
                                        pitch=pitch_offset + p,
                                        start=t * duration,
                                        end=(t + 1) * duration))
    notes = merge_held(notes)

    instrument = pretty_midi.Instrument(program=programs[i] - 1)
    instrument.notes.extend(notes)
    midi_data.instruments.append(instrument)
  return midi_data


def plot_pianorolls(path, all_full=True):
  data = np.load(path)
  for j, pianoroll_batch in enumerate(data["pianorolls"]):
    #if j % 20 != 0 or j==0 or j==1:
    if j==0 or j==1:
      continue
    batch_size = len(pianoroll_batch)
    if batch_size == 20:
      m, n = 4, 5
    elif batch_size == 100:
      m, n = 10, 10
    elif batch_size == 70:
      m, n = 9, 9
    else:
      assert False
  
    fig, axes = plt.subplots(m, n)
    for i, (pianoroll, ax) in enumerate(zip(pianoroll_batch, axes.ravel())):
      pianoroll = pianoroll.T
      pianoroll_to_midi(pianoroll).write(
          "%s_step%i_ex%i.midi" % (os.path.splitext(path)[0], j, i))
      assert 0 <= pianoroll.min()
      assert pianoroll.max() <= 1
      if not all_full:
        assert np.allclose(pianoroll.sum(axis=1), 1)
      # max across instruments
      pianoroll = pianoroll.max(axis=0)
      ax.imshow(pianoroll, cmap="viridis", interpolation="none", vmin=0, vmax=1, aspect="auto", origin="lower")
      ax.set_axis_off()
  
    fig.suptitle("%s %i" % (os.path.basename(path)[:100], j))
    fig.set_size_inches(800 / fig.dpi, 600 / fig.dpi)
    plt.tight_layout()
    plt.subplots_adjust(hspace=.01, wspace=.01)
    #plt.show()
    plt.savefig("%s_%i.png" % (os.path.splitext(path)[0], j), bbox_inches="tight")
    plt.close(fig)
    print j


def main():
  path = sys.argv[1]
  print path
  plot_pianorolls(path)


if __name__ == "__main__":
  main()
