import os, sys
from collections import defaultdict
import cPickle as pickle
from datetime import datetime

import numpy as np
import pylab as plt

from plotgibbs_process import pianoroll_to_midi, plot_pianorolls


# Inpainting generations.
base_path = '/Users/czhuang/@coconet/new_generation'
base_path = '/Users/czhuang/@coconet/new_generation/npzs'
# Bernoulli
fname = 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker-BernoulliMasker----schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161119203302_1.50min.npz'
# Transition
fname = 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--transition---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161119234002_1.05min.npz'

# Transition with multiple rewrites.
fname = 'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-100--masker-BernoulliInpaintingMasker-context-kind-transition---schedule-YaoSchedule-pmin-0-1--pmax-0-9--alpha-0-7---sampler-IndependentSampler-temperature-1e-05--_20161121011940_1.60min.npz'
fname = 'transition_inpainting_independent100.npz'
fname = 'transition_inpainting_independent10.npz'

def main():
  fpath = os.path.join(base_path, fname)
  try:
    plot_pianorolls(fpath, all_full=True)
  except:
    import pdb; pdb.post_mortem()


if __name__ == '__main__':
  main()
