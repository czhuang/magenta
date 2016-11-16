import os
import cPickle as pickle
from datetime import datetime

import numpy as np
import pylab as plt


def get_current_time_as_str():
  return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def get_fpath_wrapper(fname_tag='', file_type='png'):
  source_fpath = '/Tmp/huangche/compare_sampling/'
  fpath = os.path.join(source_fpath, 
                       '%s_%s.%s' % (fname_tag, get_current_time_as_str(), file_type))
  return fpath

TEST_MODE = False
NOTEWISE = True

# actual long run
stats_fpath = '/Tmp/huangche/compare_sampling/stats_2016-11-15_13:56:11.pkl'
if TEST_MODE:
  # short test run
  stats_fpath = '/Tmp/huangche/compare_sampling/stats_2016-11-14_21:31:41.pkl'
print 'Reading from', stats_fpath
# try reading it back in
with open(stats_fpath, 'rb') as p:
  lls_stats_by_method = pickle.load(p)

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
  aggregated_lls_stats[method_name] = (lls_means, lls_sem)

method_names = ['sequential', 'independent', '50', '75', '99']
if TEST_MODE:
  method_names = ['sequential', 'independent']

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

legend_names = {'sequential':'Contiguous(4timestep*16, covering50%) (sequential in block)',

                'independent': 'Yao (2014) annealed block size (independent in block)',
                '50': 'Bernoulli(50) (sequential in block)',
                '75': 'Bernoulli(75) (sequential in block)',
                '99': 'Bernoulli(99) (sequentail in block)'}

plt.figure()
for method_name in method_names:
  means, sems = aggregated_lls_stats[method_name] 
  plt.errorbar(sorted_iters, means, yerr=sems, label='%s' % legend_names[method_name])
plt.legend(prop={'size':10})
plt.xlim(-1, 101)
plt.xlabel('# of Gibbs steps')
plt.ylabel('Negative log-likelihood (NLL)')
plt.title("Note-wise NLL for different Blocked-Gibbs setups")
plot_fpath = get_fpath_wrapper('nll_curve-notewise', 'png')
print 'Saving plot to', plot_fpath
plt.savefig(plot_fpath)


