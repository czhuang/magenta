import os
import cPickle as pickle
from datetime import datetime

import numpy as np
import pylab as plt

#from magenta.models.basic_autofill_cnn import evaluation_tools
#from magenta.models.basic_autofill_cnn import retrieve_model_tools


TEST_MODE = False
NOTEWISE = True


def get_current_time_as_str():
  return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def get_fpath_wrapper(fname_tag='', file_type='png'):
  source_fpath = '/Tmp/huangche/compare_sampling/'
  fpath = os.path.join(source_fpath, 
                       '%s_%s.%s' % (fname_tag, get_current_time_as_str(), file_type))
  return fpath


def get_NADE_nll_mean_and_sem():
  #base_path_nade = '/Tmp/huangche/compare_sampling/nade_unzip/20161112215554_nade'
  #pianorolls = np.load(fpaths[name])['pianorolls']
  #assert pianorolls.shape == (1, 100, 32, 53, 4)
  #pianoroll = pianorolls[0]
  #model_name = 'balanced_by_scaling'
  #wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
  #wrapped_model.config.hparams.crop_piece_len = 32
  #if NOTEWISE:
  #  losses = evaluation_tools.compute_notewise_loss(wrapped_model, pianorolls)  
  if NOTEWISE:
    return 0.56509, 0.01071 
  else: 
    return 0.96849*4, 0.01253*4

# actual long run
if NOTEWISE:
  stats_fpaths = ['/Tmp/huangche/compare_sampling/stats_2016-11-15_13:56:11.pkl',
                  '/Tmp/huangche/compare_sampling/stats_2016-11-17_08:50:39.pkl']
else:
  stats_fpaths = ['/Tmp/huangche/compare_sampling/stats_2016-11-16_09:36:18.pkl']
if TEST_MODE:
  # short test run
  stats_fpaths = ['/Tmp/huangche/compare_sampling/stats_2016-11-14_21:31:41.pkl']

# Load pickled statistics.
lls_stats_by_method = dict()
for stats_fpath in stats_fpaths:
  print 'Reading from', stats_fpath
  with open(stats_fpath, 'rb') as p:
    lls_stats_by_method_partial = pickle.load(p)
  lls_stats_by_method.update(lls_stats_by_method_partial)

num_methods = len(lls_stats_by_method)
num_iters = len(lls_stats_by_method.values()[0])
iter_keys = lls_stats_by_method.values()[0].keys()
lls_means = np.zeros((num_methods, num_iters))
sorted_iters = sorted(lls_stats_by_method.values()[0].keys(), 
                      key=lambda x: int(x))

aggregated_lls_stats = dict()
for method_name, lls_stats_by_iter in lls_stats_by_method.items():
  lls_means = []
  lls_sem = []
  for eval_iter in sorted_iters:
    stats = lls_stats_by_iter[eval_iter]
    lls_means.append(stats[0])
    lls_sem.append(stats[2])
  aggregated_lls_stats[method_name] = (np.asarray(lls_means), np.asarray(lls_sem))

#method_names = ['sequential', 'independent', '50', '75', '99']
#method_names = lls_stats_by_method.keys()
# To enforce ordering
method_names = ['sequential', 'independent', '50', '75', '90', '95', '99']
if TEST_MODE:
  method_names = method_names[:2]

#aggregated_means = []
#aggregated_sems = []
#print aggregated_lls_stats.keys()
#for method_name in method_names:
#  stats = aggregated_lls_stats[method_name]
#  aggregated_means.append(stats[0])
#  aggregated_sems.append(stats[1])
#
#aggregated_means = np.asarray(aggregated_means)
#aggregated_sems = np.asarray(aggregated_sems)
#print aggregated_means.shape, aggregated_sems.shape

legend_names = {'sequential':'Contiguous(4timestep*16, covering50%)\n (sequential in block)',
                'independent': 'Yao (2014) annealed block size\n (independent in block)',
                '50': 'Bernoulli(0.5) (sequential in block)',
                '75': 'Bernoulli(0.25) (sequential in block)',
                '90': 'Bernoulli(0.1) (sequentail in block)',
                '95': 'Bernoulli(0.05) (sequentail in block)',
                '99': 'Bernoulli(0.01) (sequentail in block)'}
 

plt.figure()
for method_name in method_names:
  means, sems = aggregated_lls_stats[method_name] 
  if not NOTEWISE:
    means *= 4
    sems *= 4
  plt.errorbar(sorted_iters, means, yerr=sems, label='%s' % legend_names[method_name])
# Add Nade line.
nade_mean, nade_sem = get_NADE_nll_mean_and_sem()
plt.axhline(y=nade_mean, xmin=0, xmax=101, linewidth=1, color = 'k', label='NADE ancestral sampling')
plt.axhline(y=nade_mean+nade_sem, xmin=0, xmax=101, linewidth=0.5, color = 'k', linestyle='dotted')
plt.axhline(y=nade_mean-nade_sem, xmin=0, xmax=101, linewidth=0.5, color = 'k', linestyle='dotted')

plt.legend(prop={'size':10})
plt.xlim(-1, 101)
plt.xlabel('# of Gibbs steps')
plt.ylabel('Negative log-likelihood (NLL)')
title_str = "NLL for different Blocked-Gibbs setups" 
if NOTEWISE:
  title_str = "Note-wise " + title_str
else:
  title_str = "Chord-wise " + title_str
  
plt.title(title_str)

if NOTEWISE:
  plot_fpath = get_fpath_wrapper('nll_curve-notewise', 'png')
else:
  plot_fpath = get_fpath_wrapper('nll_curve-chordwise', 'png')
print 'Saving plot to', plot_fpath
plt.savefig(plot_fpath)


