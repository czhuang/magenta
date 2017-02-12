
import os
from copy import copy
from collections import defaultdict
import cPickle as pickle

import numpy as np

from music21.pitch import Pitch

from recover_data_split import parse_split

# C1 here is equal to the usual C4.
OCTAVE_OFFSET = 3 
# Pitch map for music21.
PITCH_NAME_MAP = {'H': 'B', 'B': 'B-', 'H#': 'B#'}
PIECE_START_LINE = 9


def parse_from_text():
  folder = '/Users/czhuang/data_music/Nicolas_ICML_bach/bach-chorales/music'
  pieces = dict()
  pchs = set()
  for fname in os.listdir(folder):
    print fname
    with open(os.path.join(folder, fname), 'r') as p:
      lines = p.readlines()
    print '# of lines', len(lines)
    piece = []
    for line in lines[PIECE_START_LINE:]:
      step = []
      cols = line.split('\t')
      if len(cols) == 1:
        # Hold previous notes.
        piece.append(copy(piece[-1]))
        continue

      assert len(cols) == 7 
      for j, col in enumerate(cols[2:6]):
        if len(col) == 0:
          # Hold that part.
          step.append(piece[-1][j])
          continue
        elif col == 'P':
          # It's a rest.
          step.append(np.nan)
          continue
        assert len(col) in [3,4]
        pch = col[:2].strip()
        pch = pch if pch not in PITCH_NAME_MAP else PITCH_NAME_MAP[pch]
        if len(pch) == 2:
          if pch[1] == 'b':
            pch = pch[0] + '-'
        octave = int(col[2:]) + OCTAVE_OFFSET
        pchs.add(pch)
        midi = Pitch(pch + str(octave)).midi
        step.append(midi)
      print step 
      piece.append(step)
    print '\t\t\t# of steps', len(piece)
    pieces[fname] = piece
  print '\t\t\t\t\t\t# of pieces', len(pieces)
  print '# of diff pitch names', len(pchs), pchs 

  fname = 'bach-16th-all-priorwork.npz'
  np.savez_compressed(fname, **pieces)
  print 'Writing to ', fname


def format_as_nicolas():
  fname = 'bach-16th-all-priorwork.npz'
  pieces = np.load(fname)
  r_pieces = dict()
  for name, piece in pieces.iteritems():
    r_piece = []
    print '# of steps', len(piece)
    for step in piece:
      r_step = [part for part in step if step is not np.nan]
      r_piece.append(tuple(r_step))
    r_pieces[name] = r_piece
  print '# of pieces', len(r_pieces)
  print [len(piece) for piece in r_pieces.values()]
  fname = 'bach-16th-all-priorwork-nicolas_style.npz'
  np.savez_compressed(fname, **r_pieces)


def in_which_split(split_dict, num):
  for split_name, nums in split_dict.iteritems():
    if num in nums:
      return split_name
  return None


def split(fpath):
  split_dict = parse_split()
  split_data = defaultdict(list)
  pieces = np.load(fpath)
  count = 0
  pieces_not_included = []
  for name, piece in pieces.iteritems():
    num = name.split('bch')[1].split('.txt')[0]    
    split_name = in_which_split(split_dict, int(num))
    if split_name is None:
      pieces_not_included.append(int(num))
      continue
    split_data[split_name].append(piece)
    count += 1
    
  assert count == 382
  print 'WARNING: These pieces were not included somehow.', pieces_not_included

  output_fname = ''.join(fpath.split('-all'))
  np.savez_compressed(output_fname, **split_data)
  return output_fname


def convert_npz_to_pickle(fpath):
  data = np.load(fpath)
  r_data = dict()
  for name, entries in data.iteritems():
    r_data[name] = entries
  path, fname = os.path.split(fpath)
  output_fname = fname.split('.npz')[0] + '.pickle'
  with open(os.path.join(path, output_fname), 'wb') as p:
    pickle.dump(r_data, p)  
  

def run_split():
  fnames = ['data/bach-16th-all-priorwork-nicolas_style.npz',
            'data/bach-16th-all-priorwork.npz']
  for fname in fnames:
    split_fname = split(fname)
    convert_npz_to_pickle(split_fname)



if __name__ == '__main__':
  try:
#    parse_from_text()
#    format_as_nicolas()
    run_split()
  except:
    import pdb; pdb.post_mortem()


