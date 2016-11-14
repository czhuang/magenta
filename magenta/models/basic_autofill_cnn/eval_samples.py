import os

import numpy as np
import tensorflow as tf
from magenta.models.basic_autofill_cnn import evaluation_tools
from magenta.models.basic_autofill_cnn import retrieve_model_tools 


TEST_MODE = False
NOTEWISE = False

def get_fpath_wrapper(fname_tag='', file_type='png'):
  source_fpath = '/Tmp/huangche/compare_sampling/'
  fpath = os.path.join(source_fpath, 
                       '%s.%s' % (fname_tag, file_type))
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
set_names = ['sequential', '50', '75', '99', 'nade'] 
basepath = '/data/lisatmp4/huangche/compare_sampling/collect_npz'
fpaths = {'sequential':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-ContiguousMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112185008_284.97min.npz',
          '50':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161112230525_251.30min.npz',
          '75':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161113031711_364.67min.npz',
          '99':'fromscratch_balanced_by_scaling_init=independent_Gibbs-num-steps-100--masker-BernoulliMasker----schedule-ConstantSchedule-0-99---sampler-SequentialSampler-temperature-1e-05--_20161113092212_488.05min.npz',
          'nade':'fromscratch_balanced_by_scaling_init=nade_Gibbs-num-steps-0--masker-BernoulliMasker----schedule-ConstantSchedule-1-0---sampler-SequentialSampler-temperature-1e-05--_20161112215554_5.05min.npz'}

for name, path in fpaths.items():
  fpaths[name] = os.path.join(basepath, path)

print '.....check that this is right'
for name in set_names:
  print name, fpaths[name]    


pianorolls_set = dict()
for name in set_names:
  print name, fpaths[name]
  pianorolls_set[name] = np.load(fpaths[name])['pianorolls'][-1]
  print pianorolls_set[name].shape
  assert pianorolls_set[name].shape == (100, 32, 53, 4)
  if TEST_MODE:
    pianorolls_set[name] = pianorolls_set[name][:2]  
    print pianorolls_set[name].shape
wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
wrapped_model.config.hparams.crop_piece_len = 32

lls = dict()
lls_stats = dict()
if TEST_MODE:
  wrapped_model.config.hparams.crop_piece_len = 2
  set_names = ['50', '75']
  pianorolls_set = {'50':pianorolls_set['50'][:, :2, :, :],
                    '75':pianorolls_set['75'][:, :2, :, :]}
 
for name, pianorolls in pianorolls_set.items():
  print name
  if NOTEWISE:
    losses = evaluation_tools.compute_notewise_loss(wrapped_model, pianorolls)
  else:
    losses = evaluation_tools.compute_chordwise_loss(wrapped_model, pianorolls)
  losses = np.asarray(losses)
  mean_losses = np.mean(losses)
  print 'losses.shape', losses.shape
  assert losses.shape[0] == 5 * pianorolls.shape[1] * pianorolls.shape[3]
  piece_means = np.mean(losses, axis=0)
  std_losses = np.std(piece_means)
  sem_losses = std_losses/np.sqrt(pianorolls.shape[0])
  lls_stats[name] = (mean_losses, std_losses, sem_losses)
  lls[name] = losses


loss_fpath = get_fpath_wrapper('losses', 'npz')
if TEST_MODE:
  np.savez_compressed(loss_fpath, seventyFive=lls['75'], fifty=lls['50'])
else:
  np.savez_compressed(loss_fpath, sequential=lls['sequential'], fifty=lls['50'], 
                    seventyFive=lls['75'], nintyNine=lls['99'], nade=lls['nade'])  
  
# write out a txt file
lines = ''
for name in set_names:
  mean, std, sem = lls_stats[name]
  lines += '\n%s: %.5f (%.5f) [%.5f, %.5f]' % (name, mean, sem, mean-sem, mean+sem)
loss_stat_fpath = get_fpath_wrapper('loss_stats', 'txt')
with open(loss_stat_fpath, 'w') as p:
  p.writelines(lines)


