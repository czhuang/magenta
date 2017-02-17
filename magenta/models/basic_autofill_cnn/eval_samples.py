import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from magenta.models.basic_autofill_cnn import evaluation_tools
from magenta.models.basic_autofill_cnn import retrieve_model_tools 

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('generation_output_dir', None, 'Path to generated samples and also base fpath for outputting eval stats.')
tf.app.flags.DEFINE_bool('quick_eval', False, 'Only evaluate a subset of samples to get a quick estimate.')
tf.app.flags.DEFINE_bool('eval_intermed', False, 'Evaluate intermediate steps.')


if FLAGS.quick_eval:
  subsample_size = 4 


def get_current_time_as_str():
  return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def get_fpath_wrapper(fname_tag='', file_type='png', folder_name=None, path=None):
  assert folder_name is not None
  #source_fpath = '/Tmp/huangche/compare_sampling/'
  if path is None:
    path = FLAGS.generation_output_dir
  output_fpath = os.path.join(path, folder_name)
  if not os.path.exists(output_fpath):
    os.mkdir(output_fpath)
  fpath = os.path.join(
      output_fpath, '%s_%s.%s' % (
          fname_tag, get_current_time_as_str(), file_type))
  return fpath


def get_fpath_wrapper_alt(fname_tag='', file_type='png', fname=None):
  assert fname is not None
  # To extract the timestamp for generation.
  source_fpath = FLAGS.generation_output_dir
  generation_timestamp = fname.split('___')[-1].split('_')[0]
  fpath = os.path.join(
      source_fpath, '%s_%s_%s.%s' % (
          generation_timestamp, fname_tag, get_current_time_as_str(), file_type))
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

# This is the input basepath. 
#basepath = '/data/lisatmp4/huangche/compare_sampling/collect_npz'
basepath = '/data/lisatmp4/huangche/sigmoids'


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
          'harmonization_inpainting':'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--harmonization---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161120192512_1.47min.npz',
          'transition_inpainting': 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--transition---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161119234002_1.05min.npz',
          'inner_voices_inpainting': 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--inner-voices---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161120013826_1.06min.npz',
          'tensor_inpainting': 'fromscratch_balanced_by_scaling_init=bach_nade_Gibbs-num-steps-0--masker--tenor---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161120014019_0.58min.npz'}


fpaths = {'transition_inpainting_independent-100':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-100--masker-BernoulliInpaintingMasker-context-kind-transition---schedule-YaoSchedule-pmin-0-1--pmax-0-9--alpha-0-7---sampler-IndependentSampler-temperature-1e-05--_20161121020140_1.68min.npz'}


fpaths = {'transition_inpainting_sequential-30_bernoulli-25':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-transition---schedule-ConstantSchedule-0-25---sampler-SequentialSampler-temperature-1e-05--_20161121033900_8.04min.npz',
          'transition_inpainting_sequential-30_bernoulli-50': 'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-transition---schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161121032410_14.65min.npz',
          'transition_inpainting_sequential-30_bernoulli-75': 'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-transition---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161121030150_22.16min.npz',
          'transition_inpainting_sequential-30_bernoulli-90': 'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-transition---schedule-ConstantSchedule-0-9---sampler-SequentialSampler-temperature-1e-05--_20161121023609_25.49min.npz'}


fpaths = {'harmonization_inpainting_sequential-30_bernoulli-75':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-harmonization---schedule-ConstantSchedule-0-75---sampler-SequentialSampler-temperature-1e-05--_20161121134918_62.77min.npz',
          'harmonization_inpainting_sequential-30_bernoulli-90':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-harmonization---schedule-ConstantSchedule-0-9---sampler-SequentialSampler-temperature-1e-05--_20161121123327_75.65min.npz'}

fpaths = {'harmonization_inpainting_independent-100':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-100--masker-BernoulliInpaintingMasker-context-kind-harmonization---schedule-YaoSchedule-pmin-0-1--pmax-0-9--alpha-0-7---sampler-IndependentSampler-temperature-1e-05--_20161121155835_2.97min.npz',
          'harmonization_inpainting_sequential-30-bernoulli-25':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-harmonization---schedule-ConstantSchedule-0-25---sampler-SequentialSampler-temperature-1e-05--_20161121153459_23.41min.npz',
          'harmonization_inpainting_sequential-30-bernoulli-50':'fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-30--masker-BernoulliInpaintingMasker-context-kind-harmonization---schedule-ConstantSchedule-0-5---sampler-SequentialSampler-temperature-1e-05--_20161121145217_42.53min.npz'}

fpaths = {
    'sigmoid_bach_ind_848':'fromscratch_None_init=independent_Gibbs_num_steps_848__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0___20170108220211_12.27min.npz',
    'sigmoid_bach_nade':'fromscratch_None_init=sequential_Gibbs_num_steps_0__masker_None__schedule_None__sampler_None__20170109121655_24.81min.npz',
    'sigmoid_bach_independent_1696': '/data/lisatmp4/huangche/sigmoids/fromscratch_None_init=independent_Gibbs_num_steps_1696__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_0___20170109143550_24.55min.npz'}

# test sigmoid higher temperature
fpaths = {
    'sigmoid_ind_higher_temp': 'fromscratch_None_init=independent_Gibbs_num_steps_30__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_01___20170110000418_0.52min.npz'}

# eval BinarMNIST samples
fpaths = {
    'BinaryMNIST-iGibbs-temp1_wo_pop': 'fromscratch_None_init=independent_Gibbs_num_steps_784__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170207141152_10.44min.npz',
    'BinaryMNIST-nade-temp1_wo_pop': 'fromscratch_None_init=sequential_Gibbs_num_steps_0__masker_None__schedule_None__sampler_None__1.0_20170207141152_10.61min.npz',
    'BinaryMNIST-iGibbs-temp1_wo_pop-558_steps':'fromscratch_None_init=independent_Gibbs_num_steps_588__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170209021314_4.22min.npz',
    'BinaryMNIST-iGibbs-temp1_wo_pop-392_steps':'fromscratch_None_init=independent_Gibbs_num_steps_392__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170209021316_2.86min.npz'}

fpaths = {
    'BinaryMNIST-iGibbs-temp1_wo_pop': 'fromscratch_None_init=independent_Gibbs_num_steps_784__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170207141152_10.44min.npz'}

fpaths = {
    'BinaryMNIST-iGibbs-temp1_wo_pop-392_steps-yao_pmin_0.03': 'fromscratch_None_init=independent_Gibbs_num_steps_392__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_03__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170209111940_2.72min.npz'}

fpaths = {
    'BinaryMNIST-iGibbs-temp0.5-wo_pop-':'fromscratch_None_init=independent_Gibbs_num_steps_784__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_0_5___0.5_20170207163021_5.57min.npz'}

fpaths = {
    'binarymnist-iGibbs-temp1_random_init': 'fromscratch_None_init=random_Gibbs_num_steps_784__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170210222234_5.52min.npz'}


fpaths = {
    'omni-NADE-temp1-trainedBinary9h': '/data/lisatmp4/huangche/sigmoids/fromscratch_None_init=sequential_Gibbs_num_steps_0__masker_None__schedule_None__sampler_None__1.0_20170214122021_4.20min.npz',
    'omni-iGibbs-temp1-trainedBinary9h': '/data/lisatmp4/huangche/sigmoids/fromscratch_None_init=independent_Gibbs_num_steps_784__masker_BernoulliMasker____schedule_YaoSchedule_pmin_0_1__pmax_0_9__alpha_0_7___sampler_IndependentSampler_temperature_1_0___1.0_20170214124030_6.09min.npz'}


set_names = fpaths.keys()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for name, path in fpaths.items():
  fpaths[name] = os.path.join(basepath, path)

pianorolls_set = dict()
T = None
for name in set_names:
  print name, fpaths[name]
  pianorolls_steps = np.load(fpaths[name])['pianorolls']
  print 'shape', pianorolls_steps.shape
  pianorolls_by_iter = dict()
  S = len(pianorolls_steps)
  if FLAGS.eval_intermed:
    if FLAGS.eval_test_mode:
      eval_iters = [int(S/2), -1]
    else:
      #eval_iters = range(S)
      eval_iters = range(1, S, 3)

    print '\n# of eval iterations: %d\n' % len(eval_iters)
  else:
    eval_iters = [-1]
  for eval_iter in eval_iters:
    print 'before eval_iter indexing', pianorolls_steps.shape 
    pianorolls = pianorolls_steps[eval_iter]
    print 'after eval_iter indexing', pianorolls.shape 
    if 'inpainting' in name:
      assert pianorolls.shape[0] == 3 or pianorolls.shape[0] == 103 or pianorolls.shape[0] == 13 or pianorolls.shape[0] == 33
    else:
      pass # since might test diff # of steps.
      # assert pianorolls.shape[0] == 101
    if T is None:
      T = pianorolls.shape[1]
    elif T is not None and T != pianorolls.shape[1]:
      assert False, 'Pianorolls for different sample methods are of different shapes.'
    if 'inpainting' in name:
      assert pianorolls.shape == (70, 32, 53, 4)
    else:
      print pianorolls.ndim, pianorolls.shape
      assert pianorolls.ndim == 4
      #assert pianorolls.shape == (100, 32, 53, 4)
    if FLAGS.quick_eval:
      inds = np.random.choice(np.arange(pianorolls.shape[0]), size=subsample_size, replace=False)
      pianorolls = pianorolls[inds, :, :, :]
    pianorolls_by_iter[eval_iter] = pianorolls
  pianorolls_set[name] = pianorolls_by_iter

#lls_by_method = dict()
lls_stats_by_method = dict()
wrapped_model = None
for name, pianorolls_by_iterations in pianorolls_set.items():
  print name
  lls_by_iter = dict()
  lls_stats_by_iter = dict()  
  for eval_iter, pianorolls in pianorolls_by_iterations.items():
    print 'eval_iter', eval_iter
    sample_name = '%s-iter=%d' % (name, eval_iter) 
    mean_loss, sem_loss, N, wrapped_model, eval_fpath = (
        evaluation_tools.run(pianorolls, wrapped_model, 
                             sample_name=sample_name))
    
    #lls_by_iter[eval_iter] = losses
    lls_stats_by_iter[eval_iter] = (mean_loss, sem_loss, N)
  #lls_by_method[name] = lls_by_iter
  lls_stats_by_method[name] = lls_stats_by_iter

#loss_fpath = get_fpath_wrapper('losses', 'npz', timestamp)
#flatten the nested dict to store in npz
#lls_by_method_flatten = {'%s_%d'%(method_name, eval_iter):lls for method_name, lls_by_iters in lls_by_method.items() for eval_iter, lls in lls_by_iters.items()}
#np.savez_compressed(loss_fpath, **lls_by_method_flatten)
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
      mean, sem, N = lls_stats
      try:
        lines += '\n\t\t%d: %.5f (%.5f, N=%d) [%.5f, %.5f], ' % (eval_iter, mean, sem, N, mean-sem, mean+sem)
      except TypeError:
        lines += '\n\t\t%d: %r (%r, N=%r) [%r, %r], ' % (eval_iter, mean, sem, N, mean-sem, mean+sem)
 
  lines += '\n'
  loss_stat_fpath = get_fpath_wrapper(
    'loss_stats', 'txt', timestamp, path=eval_fpath)
  print 'Writing to', loss_stat_fpath
  print lines
  with open(loss_stat_fpath, 'w') as p:
    p.writelines(lines)

# write out a txt file
write_results(set_names, lls_stats_by_iter)

# write out stats as a pickle so easier to make plot
import cPickle as pickle
stats_fpath = get_fpath_wrapper('stats', 'pkl', timestamp, eval_fpath)
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

