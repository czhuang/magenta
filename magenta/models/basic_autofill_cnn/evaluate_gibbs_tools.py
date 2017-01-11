"""Tools for evaluating Gibbs sampling."""
import os, sys, traceback
from collections import defaultdict

import tensorflow as tf
import numpy as np
import pylab as plt

from magenta.models.basic_autofill_cnn import postprocess
from magenta.models.basic_autofill_cnn import evaluation_tools 
from magenta.models.basic_autofill_cnn import retrieve_model_tools 
from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn.generate_tools import AutofillStep


def get_fpath():
  # short test example
  path = '/Tmp/huangche/new_generation/2016-11-07_00:33:00-random_medium'
  fname = '0_generate_gibbs_like-0-4.73min-2016-11-07_00:33:00-random_medium-0-empty-None.npz'

  # real long example, 100*4 full gibbs iterations
#  fname = '0_generate_gibbs_like-0-156.86min-2016-11-07_02:03:00-balanced_by_scaling-0-empty-None.npz'
#  path = '/Tmp/huangche/new_generation/2016-11-07_02:03:00-balanced_by_scaling'
  return os.path.join(path, fname)


def get_fpath_wrapper(fname_tag='', file_type='png'):
  source_fpath = get_fpath()
  dirname = os.path.dirname(source_fpath)
  fname = os.path.basename(source_fpath).split('.')[0]
  fpath = os.path.join(dirname, 
                       '%s%s.%s' % (fname_tag, fname, file_type))
  return fpath


def get_process_pianorolls_from_bundle(generation_bundle):
  """For old pickle formats."""
  generated_seq, steps, original_seq = generation_bundle
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  generated_pianoroll = encoder.encode(generated_seq)

  num_steps = len(steps) #10  # len(steps)
  pianorolls = [step.generated_piece for step in steps][:num_steps]
  print np.sum(generated_pianoroll), np.sum(pianorolls[-1])
  for i, step in enumerate(steps[:num_steps]):
    print i, np.sum(step.generated_piece)
  return pianorolls


def get_process_pianorolls(fpath):
  with open(fpath, 'rb') as p:
    rolls = np.load(p)['arr_0']
  return rolls


def get_gibbs_step_pianorolls(pianorolls):
  T, _, I = pianorolls[0].shape
  # Can not collect periodically at num_blankouts b/c they might overlap
  # and then the number of steps are different.
  # Collect full pianorolls.
  pianorolls = [roll for roll in pianorolls if np.sum(roll) == T*I]
  print '# of full pianorolls', len(pianorolls)
  return pianorolls


def get_gibbs_step_pianorolls_wrapper():
  fpath = get_fpath()
  #generation_bundle = postprocess.retrieve_generation_bundle(fpath)
  #pianorolls = get_process_pianorolls(generation_bundle)
  pianorolls = get_process_pianorolls(fpath)
  return get_gibbs_step_pianorolls(pianorolls)


def plot_process():
  pianorolls = get_gibbs_step_pianorolls_wrapper()
  subplot_shape = [3,5]
  num_figures = len(pianorolls) / np.product(subplot_shape) + 1
  for fi in range(num_figures):
    fig, axess = plt.subplots(subplot_shape[0], subplot_shape[1])
    for r, axes in enumerate(axess.T):
      pi_r = fi * np.product(subplot_shape) + r * subplot_shape[0]
      if pi_r >= len(pianorolls):
        break
      for c, ax in enumerate(axes):
        pi = pi_r + c
        print 'pi', pi
        if pi >= len(pianorolls):
          print 'not plotting, pianorolls are all consumed'
          break
        summed_roll = np.clip(pianorolls[pi].sum(axis=2).T, 0, 1)
        ax.imshow(summed_roll, interpolation="none", cmap="bone", 
                aspect="auto", origin="lower")
        ax.set_title('Gibbs step %d' % pi )
      
      plt.tight_layout()
      plt.savefig(get_fpath_wrapper('subplot_process_fig-%d_' % fi, 'png'))


def evaluate_autocorrelation():
  pianorolls = get_gibbs_step_pianorolls_wrapper()
  T, _, I = pianorolls[0].shape
  num_rolls = len(pianorolls)
  #autocorrs = defaultdict(list)
  autocorrs = []
  #for i in range(num_rolls):
  #for interval in [1, 2, 5, 10, 20]:
  for i in range(num_rolls):  
    corrcoef = np.corrcoef(np.ravel(pianorolls[0]), np.ravel(pianorolls[i]))
    autocorrs.append(corrcoef[0,1])
  plt.figure()
  #for interval, corrcoefs in autocorrs.items():
  plt.plot(autocorrs)  #, label='interval %s steps' % interval)
  #plt.legend()
  plt.title('Autocorrelation across lag in Gibbs steps')
  plt.xlabel('lag in # of Gibbs steps')
  plt.ylabel('correlation coefficient')
  fpath = get_fpath_wrapper('plot_autocorrelation')
  plt.savefig(fpath)


def evaluate_llk_progress():
  pianorolls = get_gibbs_step_pianorolls_wrapper()
  fpath = get_fpath_wrapper('plot_llk_progress')
  T, _, I = pianorolls[0].shape
  model_name = 'random_medium' #balanced_by_scaling' #'random_medium'
  # Make sure the model used to evaluate is also the model used to generate.
  assert model_name in fpath 
  wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
  wrapped_model.config.hparams.crop_piece_len = pianorolls[0].shape[0]
  print 'piece_len', pianorolls[0].shape

  losses = evaluation_tools.compute_notewise_loss(wrapped_model, pianorolls, model_name)
  assert len(losses) ==  5 * T * I
  # losses, list
  loss_by_step = np.mean(np.asarray(losses), axis=0)
  print 'losses', loss_by_step[:10]
  loss_fpath = get_fpath_wrapper('loss_by_step', 'npz')
  np.savez_compressed(loss_fpath, loss_by_step)  
 
  plot_curve(loss_by_step, fpath) 
 

def plot_curve(curve, fpath):
  plt.plot(curve)
  plt.title('NLL against number of Gibbs step taken')
  plt.xlabel('# of Gibbs steps')
  plt.ylabel('Negative log-likelihood (NLL)')
  plt.savefig(fpath) 


def check_and_plot_loss():
  loss_fpath = get_fpath_wrapper('loss_by_step', 'npz')
  with open(loss_fpath, 'rb') as p:
    losses = np.load(p)['arr_0']
  plot_fpath = get_fpath_wrapper('check_plot_loss')
  plot_curve(losses, plot_fpath)


def main(argv):
  try:
    plot_process()
    #evaluate_autocorrelation()
  #  evaluate_llk_progress()   
  #  check_and_plot_loss()
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, KeyboardInterrupt):
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


if __name__ == '__main__':
  tf.app.run()
