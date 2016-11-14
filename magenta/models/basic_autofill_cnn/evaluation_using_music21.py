"""Evaluate generated samples using music theory rules in Music21."""

import os
from collections import defaultdict
import numpy as np
from music21 import converter, midi, alpha

TEST_MODE = False
ANALYZE = True


#path = '/Users/czhuang/@coconet/gibbs50stepscomparison/step50'
#methods = ['BisectingSampler', 'IndependentSampler']
#
#path2 = '/Users/czhuang/@coconet/2016-11-07_02-36-02-balanced_by_scaling'
#methods2 = ['generate_gibbs_like']
#source_sets = [[path, methods], [path2, methods2]]
#methods = ['generate_gibbs_like', 'BisectingSampler', 'IndependentSampler']


base_path = '/Tmp/huangche/compare_sampling/gibbs40_unzip'
folders = ['20161111_sequentialgibbs', '20161111_bisectinggibbs', '20161111_independentgibbs']
tag_in_fname = 'step40'


base_path = '/Tmp/huangche/compare_sampling/gibbs10_unzip'
folders = "20161112135932_sequentialgibbs 20161112135932_bisectinggibbs  20161112135932_independentgibbs  ".split()
tag_in_fname = 'step10'


base_path = '/Tmp/huangche/compare_sampling/gibbs_2016111223_100steps_unzip/2016111223_100steps'
folders = "sequential bisecting  independent  ".split()
tag_in_fname = 'step100'

sampling_methods = ['SequentialSampler', 'BisectingSampler', 'IndependentSampler']


#base_path = '/Tmp/huangche/compare_sampling/nade_unzip'
#folders = ['20161112215554_nade']
#tag_in_fname = 'step0'
#sampling_methods = ['nade_Gibbs']

base_path = '/Tmp/huangche/compare_sampling/longestGibbs/2016-11-07_02:03:00-balanced_by_scaling'
folders = ['']
tag_in_fname = '0-empty'
sampling_methods = ['gibbs']

# TODO: need to fix the divide ratio b/c different number of measures
base_path = '/Tmp/huangche/compare_sampling/rewriting/2016-11-13_19:52:58-balanced_by_scaling'
folders = ['']
tag_in_fname = 'regenerate_voice_by_voice'
sampling_methods = ['regenerate_voice_by_voice']

base_path ='/Tmp/huangche/compare_sampling/20161112_100steps_505799_unzip/20161112_100steps_505799'
folders = ['', '', '']
# tag_in_fname = 'step100'
tag_in_fname = 'step10_'
sampling_methods = ['0-99--', '0-75--', '0-5--']


# Make source sets, for the second entry, expects a list of methods for each folder.
source_sets = [[os.path.join(base_path, folder), [sampling_methods[i]]] for i, folder in enumerate(folders)]
method_fpaths = defaultdict(list)
for source in source_sets:
    path = source[0]
    print path
    methods = source[1]
    print methods
    raw_fnames = os.listdir(path)
    print '# of all fnames', len(raw_fnames)
    print raw_fnames[0]
    fnames = [fname for fname in os.listdir(path) if '.midi' in fname and tag_in_fname in fname]
    print '# of fnames', len(fnames)
    for fname in fnames:
        for method in methods:
            if method in fname and 'original' not in fname:
              method_fpaths[method].append(os.path.join(path, fname))
              break

for key, values in method_fpaths.items():
    print key, len(values)

# make smaller set for testing
if TEST_MODE:
    num_test = 2 
    for method, values in method_fpaths.items():
        method_fpaths[method] = method_fpaths[method][:num_test]
for key, values in method_fpaths.items():
    print key, len(values)
  
  
  #rules = filter(lambda x:'identify' in x and '_identify' not in x, dir(alpha.theoryAnalysis.theoryAnalyzer))

rules = ['identifyParallelFifths',
 'identifyParallelOctaves',
 'identifyParallelUnisons',
 'identifyHiddenFifths',
 'identifyHiddenOctaves',
 'identifyImproperDissonantIntervals',
 'identifyImproperResolutions']

#rules = ['identifyParallelFifths',
# 'identifyImproperDissonantIntervals',
# 'identifyImproperResolutions']

if TEST_MODE:
    rules = ['identifyImproperDissonantIntervals']
grouped_rule_prefixes = ['Parallel', 'Hidden', 'ImproperDissonantIntervals',
                         'ImproperResolutions']

rnames = []
for rule in rules:
  name_part = rule.split('identify')[-1]
  rname = name_part[0].lower() + name_part[1:]
  rnames.append(rname)
if 'identifyImproperDissonantIntervals' in rules or 'identifyImproperResolutions' in rules:
  rnames.append('res')
print 'rnames', rnames
print '# of rules', len(rules)
print '# of groups', len(grouped_rule_prefixes)
  
  
if ANALYZE:
  
  def get_rule_violations(fpath):
      midi_fpath = os.path.join(path, fpath)
      try:
          sc = converter.parse(midi_fpath)
      except:
          print 'ERROR: Failed to parse %s', midi_fpath
          return defaultdict(int)
      for rule in rules:
          #print '\t\t', rule
          print '.',
          getattr(alpha.theoryAnalysis.theoryAnalyzer, rule)(sc)
      print
      #for rule, violations in sc.analysisData['ResultDict'].items():
      #    print '\t\t', rule, len(violations)
      return sc.analysisData['ResultDict']
  
  def get_aggregated_stats(fpaths):
      raw_results = [get_rule_violations(fpath) for fpath in fpaths]
      stats = defaultdict(int)
      for result in raw_results:
          for rule_name, violations in result.items():
              stats[rule_name] += len(violations)
      for rule, stat in stats.items():
          print '\t', rule, stat
      return raw_results, stats
  
  
  # In[97]:
  
  raw_results = dict()
  stats = dict()
  for method_name, fpaths in method_fpaths.items():
      print method_name
      raw_results[method_name], stats[method_name] = get_aggregated_stats(fpaths)
  for method_name, stat in stats.items():
      print method_name
      for rule, count in stat.items():
          print '\t', rule, count
  
  
  # In[107]:
  
  # divide by num of pieces
  print 'sampling_methods', sampling_methods
  # loop over rule first
  for rule in rnames:
      print rule, 
      for method in sampling_methods:
          print float(stats[method][rule]) / len(method_fpaths[method]) * 0.5,
      print
          
  import cPickle as pickle
  fpath = os.path.join(base_path, 'music21_analysis.pkl')
  pickle_dict = dict(raw_results=raw_results, stats=stats)
  print 'Writing to pickle', fpath
  with open(fpath, 'wb') as p:
    pickle.dump(pickle_dict, p)


import cPickle as pickle
# Reading from pickle...
fpath = os.path.join(base_path, 'music21_analysis.pkl')
print 'Reading from pickle', fpath
with open(fpath, 'rb') as p:
  pickle_dict = pickle.load(p)
  stats = pickle_dict['stats']
  raw_results = pickle_dict['raw_results']
  for rule in rnames: 
      print rule, 
      for method in sampling_methods:
          #print rule, method,
          #print float(stats[method][rule]), len(method_fpaths[method]),
          #print float(stats[method][rule]) / len(method_fpaths[method]),
          print float(stats[method][rule]) / len(method_fpaths[method]) * 0.5,
          #print
      print

# Aggregate individual counts
raw_aggregates_by_method = dict()
rules = stats[sampling_methods[-1]].keys()
for method in sampling_methods:
  raw_aggregates = defaultdict(list)
  for piece_violations in raw_results[method]:
    for rule in rnames:
      if rule not in piece_violations:
        raw_aggregates[rule].append(0)
      else:
        raw_aggregates[rule].append(len(piece_violations[rule]))
  raw_aggregates_by_method[method] = raw_aggregates

# Compute mean and standard error
stats_by_method = dict()
for method, method_violations in raw_aggregates_by_method.items():
  method_stats = dict()
  for rule, violation_counts in method_violations.items():
    # Since the counts are aggregated over two measures
    counts = np.asarray(violation_counts) * 0.5
    #print rule, method, np.sum(violation_counts), len(violation_counts), np.mean(counts)
    method_stats[rule] = (np.mean(counts), np.std(counts)/np.sqrt(len(counts)))
  stats_by_method[method] = method_stats

# Print out the result by rules, method
print 'sampling methods', sampling_methods
for rule in rules:
  print rule,
  for method in sampling_methods:
    if rule not in stats_by_method[method]:
      print '0.000 ()',
    else:
      mean, sem = stats_by_method[method][rule]
      print '%.3f (%.3f)[%.3f, %.3f],' % (mean, sem, mean-sem, mean+sem), 
  print    


