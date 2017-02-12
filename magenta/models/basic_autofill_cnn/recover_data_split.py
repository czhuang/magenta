
import os
from collections import defaultdict

import numpy as np

def parse_map():
  fpath = '/Users/czhuang/magenta-autofill/magenta/models/basic_autofill_cnn/data/Konkordanzliste copy.txt'
  num2bwv = dict()
  with open(fpath, 'r') as p:
    lines = p.readlines()
  map_start_line = 7
  for line in lines[map_start_line:]:
    line = line.strip()
    if len(line) == 0:
      continue
    splits = line.split('\t')
    parts = []
    for part in splits:
      try:
        if part.strip() == '?':
          parts.append(None)
        else:
          part = int(part)
          parts.append(part)
      except:
        print 'failed', part
        if part == '244a':
          parts.append([[244, 'a']])
        else:
          pparts = [pp.strip() for pp in part.split(',')]
          print pparts
          parts.append([int(pparts[0]), int(pparts[1])])
        print parts
    num2bwv[parts[0]] = parts[1]
  assert len(num2bwv) == 389
  return num2bwv


def parse_split():
  fpath = '/Users/czhuang/magenta-autofill/magenta/models/basic_autofill_cnn/data/mysplit copy.txt'
  with open(fpath) as p:
    lines = p.readlines()
  key = None
  split = defaultdict(list)
  for i, line in enumerate(lines):
    print i, line
    if '[' in line:
      parts = line.split("'")      
      key = parts[1]
    parts = line.split("bach-chorales/music/bch")
    num = parts[1].split('.txt')[0]
    print num
    split[key].append(int(num))

  # TODO: check if there are overlaps
  counts = [len(nums) for nums in split.values()]
  assert np.sum(counts) == 382
  return split


def get_music21_bwvs():
  path = '/Users/czhuang/packages/music21/music21/corpus/bach'
  fnames = [fname for fname in os.listdir(path)]
  bwvs = []
  for fname in fnames:
    if 'choraleAnalyses' in fname:
      continue
    bwv = fname.split('bwv')[1].split('.mxl')[0]
    bwvs.append(bwv)
  print '# of bwvs:', len(bwvs)
  return bwvs


def recover_split():
  num2bwv = parse_map()
  split = parse_split()
  split_bwvs = dict()
  for set_, nums in split.iteritems():
    split_bwvs[set_] = [num2bwv[num] for num in nums]
  print split_bwvs
  for key, nums in split_bwvs.iteritems():
    print key, len(nums)
  bwvs = get_music21_bwvs()
  split_music21 = defaultdict(list)
  not_matched = defaultdict(list)
  for set_, bwvs in split_bwvs.iteritems():
    for bwv in bwvs:
      if str(bwv) in bwvs:
        split_music21[set_].append(bwv)
      else:
        not_matched[set_].append(bwv)
  print not_matched
 

if __name__ == '__main__':
  recover_split()
