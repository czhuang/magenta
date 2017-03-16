import os, sys
from collections import defaultdict
import cPickle as pickle
from datetime import datetime

import numpy as np
import pylab as plt

from plotgibbs_process import pianoroll_to_midi, plot_pianorolls

path = sys.argv[1]

def main():
  try:
    plot_pianorolls(path, all_full=True)
  except:
    import pdb; pdb.post_mortem()

if __name__ == '__main__':
  main()
