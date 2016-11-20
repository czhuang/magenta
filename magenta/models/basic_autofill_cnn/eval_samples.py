import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from magenta.models.basic_autofill_cnn import evaluation_tools
from magenta.models.basic_autofill_cnn import retrieve_model_tools 


TEST_MODE = False
NOTEWISE = True 
#eval_iters = range(0, 101, 5)
eval_iters = range(20)
eval_iters = range(2)

def get_current_time_as_str():
  return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def get_fpath_wrapper(fname_tag='', file_type='png'):
  source_fpath = '/Tmp/huangche/compare_sampling/'
  fpath = os.path.join(source_fpath, 
                       '%s_%s.%s' % (fname_tag, get_current_time_as_str(), file_type))
  return fpath
#
## 5 sets of samples, need to collect npz
#base_path_sequential = '/Tmp/huangche/compare_sampling/gibbs_2016111223_100steps_unzip/2016111223_100steps/sequential'
#base_path_nade = '/Tmp/huangche/compare_sampling/nade_unzip/20161112215554_nade'
#
#base_path ='/Tmp/huangche/compare_sampling/20161112_100steps_505799_unzip/20161112_100steps_505799'
#sampling_methods = ['0-5', '0-75', '0-99']
#
#paths = [base_path_sequential] + [base_path]*3 + [base_path_nade]
#fname_tag = [None] + sampling_methods + [None]
#
#model_name = 'balanced_by_scaling'
#set_names = ['sequential', '50', '75', '99', 'nade'] 
#fpaths = dict()
## retrieve the fnames of the npz
#for i, path in enumerate(paths):
#  fnames = os.listdir(path)
#  npz_matching_fnames = []
#  for fname in fnames:
#    if '.npz' in fname and (fname_tag[i] is None or fname_tag[i] in fname):
#      npz_matching_fnames.append(fname)
#  assert len(npz_matching_fnames) == 1
#  assert model_name in npz_matching_fnames[0] 
#  fpaths[set_names[i]] = os.path.join(path, npz_matching_fnames[0])
#
#print '.....check that this is right'
#for name in set_names:
#  print name, fpaths[name]    
#

model_name = 'balanced_by_scaling'
basepath = '/data/lisatmp4/huangche/compare_sampling/collect_npz'
fpaths = {'sequential':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-ContiguousMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112185008_284.97min.npz',
          '50':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112230525_251.30min.npz',
          '75':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161113031711_364.67min.npz',
          '99':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-99---sampler-SequentialSampler-temperature-1e-05--_20161113092212_488.05min.npz',
	  'independent':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-YaoSchedule-pmin-0-1--pmax-0-9--alpha-0-7---sampler-IndependentSampler-temperature-1e-05--_20161112233522_4.73min.npz'}

fpaths = {'90':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-9---sampler-SequentialSampler-temperature-1e-05--_20161116075839_520.12min.npz',
          '95':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-95---sampler-SequentialSampler-temperature-1e-05--_20161115225435_543.70min.npz'}


fpaths = {'sequential':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-ContiguousMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112185008_284.97min.npz',
          '50':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112230525_251.30min.npz',
          '75':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161113031711_364.67min.npz',
          '99':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-99---sampler-SequentialSampler-temperature-1e-05--_20161113092212_488.05min.npz',
          '90':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-9---sampler-SequentialSampler-temperature-1e-05--_20161116075839_520.12min.npz',
          '95':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-95---sampler-SequentialSampler-temperature-1e-05--_20161115225435_543.70min.npz'}


#          'nade':'fromscratch_balanced_by_scaling_init=nade_Gibbs-num-steps-0--masker-BernoulliMasker----schedule-ConstantSchedule-1-0---sampler-SequentialSampler-temperature-1e-05--_20161112215554_5.05min.npz',


fpaths = {'bernoulli75_inpainting':'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker-BernoulliMasker----schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161119203302_1.50min.npz',
          'transition_inpainting': 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--transition---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161119234002_1.05min.npz',
          'inner_voices_inpainting': 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--inner-voices---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161120013826_1.06min.npz',
          'tensor_inpainting': 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--tenor---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161120014019_0.58min.npz'}

set_names = fpaths.keys()

for name, path in fpaths.items():
  fpaths[name] = os.path.join(basepath, path)

if TEST_MODE:
  eval_iters = [0, 100]
  eval_iters = [2]
pianorolls_set = dict()
for name in set_names:
  print name, fpaths[name]
  pianorolls_by_iter = dict()
  for eval_iter in eval_iters:
    #pianorolls = np.load(fpaths[name])['pianorolls'][eval_iter]
    pianorolls = np.load(fpaths[name])['pianorolls']
    if 'inpainting' in name:
      assert pianorolls.shape[0] == 3
    else:
      assert pianorolls.shape[0] == 101
    pianorolls = pianorolls[eval_iter]
    if 'inpainting' in name:
      assert pianorolls.shape == (70, 32, 53, 4)
    else:
      assert pianorolls.shape == (100, 32, 53, 4)
    if TEST_MODE:
      pianorolls = pianorolls[:2, :2, :, :]  
      print pianorolls.shape
    pianorolls_by_iter[eval_iter] = pianorolls
  pianorolls_set[name] = pianorolls_by_iter
wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
wrapped_model.config.hparams.crop_piece_len = 32

lls_by_method = dict()
lls_stats_by_method = dict()
# not putting this before retrieving pianorolls from npz to make sure all is retrievable.
if TEST_MODE:
  wrapped_model.config.hparams.crop_piece_len = 2
  set_names = set_names[:2]
  pianorolls_set = {name:pianorolls_set[name] for name in set_names}
  
for name, pianorolls_by_iterations in pianorolls_set.items():
  print name
  lls_by_iter = dict()
  lls_stats_by_iter = dict()  
  for eval_iter, pianorolls in pianorolls_by_iterations.items():
    print 'eval_iter', eval_iter
    if NOTEWISE:
      losses = evaluation_tools.compute_notewise_loss(wrapped_model, pianorolls)
    else:
      losses = evaluation_tools.compute_chordwise_loss(wrapped_model, pianorolls)
    losses = np.asarray(losses)
    mean_losses = np.mean(losses)
    print 'losses.shape', losses.shape, 'pianorolls.shape', pianorolls.shape
    assert losses.shape[0] == 5 * pianorolls.shape[1] * pianorolls.shape[3]
    piece_means = np.mean(losses, axis=0)
    std_losses = np.std(piece_means)
    sem_losses = std_losses/np.sqrt(pianorolls.shape[0])
    lls_stats_by_iter[eval_iter] = (mean_losses, std_losses, sem_losses)
    lls_by_iter[eval_iter] = losses
  lls_by_method[name] = lls_by_iter
  lls_stats_by_method[name] = lls_stats_by_iter

loss_fpath = get_fpath_wrapper('losses', 'npz')

#flatten the nested dict to store in npz
lls_by_method_flatten = {'%s_%d'%(method_name, eval_iter):lls for method_name, lls_by_iters in lls_by_method.items() for eval_iter, lls in lls_by_iters.items()}
np.savez_compressed(loss_fpath, **lls_by_method_flatten)
#if TEST_MODE:
#  #np.savez_compressed(loss_fpath, seventyFive=lls['75'], fifty=lls['50'])
#  np.savez_compressed(loss_fpath, independent=lls['independent'])
#else:
#  #np.savez_compressed(loss_fpath, sequential=lls['sequential'], fifty=lls['50'], 
#  #                  seventyFive=lls['75'], nintyNine=lls['99'], nade=lls['nade'])  
#  np.savez_compressed(loss_fpath, independent=lls['independent'])
  
def write_results(set_names, lls_stats_by_iter):
  lines = ''
  print 'set_names', set_names
  print lls_stats_by_method.keys()
  for name in set_names:
    lls_stats_by_iter = lls_stats_by_method[name]
    lines += '\n, %s,' % (name)
    for eval_iter, lls_stats in lls_stats_by_iter.items():
      mean, std, sem = lls_stats
      lines += '%d: %.5f (%.5f) [%.5f, %.5f], ' % (eval_iter, mean, sem, mean-sem, mean+sem)
  lines += '\n'
  loss_stat_fpath = get_fpath_wrapper('loss_stats', 'txt')
  print 'Writing to', loss_stat_fpath
  with open(loss_stat_fpath, 'w') as p:
    p.writelines(lines)

# write out a txt file
write_results(set_names, lls_stats_by_iter)

# write out stats as a pickle so easier to make plot
import cPickle as pickle
stats_fpath = get_fpath_wrapper('stats', 'pkl')
print 'Writing to', stats_fpath
with open(stats_fpath, 'wb') as p:
  pickle.dump(lls_stats_by_method, p)

print 'Reading from', stats_fpath
# try reading it back in
with open(stats_fpath, 'rb') as p:
  lls_stats_by_method = pickle.load(p)

print 'after loading from pickle'
# write out a txt file
write_results(set_names, lls_stats_by_iter)

