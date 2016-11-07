"""Tools for evaluating Gibbs sampling."""
import os, sys, traceback

import tensorflow as tf
import numpy as np
import pylab as plt

from magenta.models.basic_autofill_cnn import postprocess
from magenta.models.basic_autofill_cnn import evaluation_tools 
from magenta.models.basic_autofill_cnn import retrieve_model_tools 
from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn.generate_tools import AutofillStep


def get_fpath():
  fname = '0_generate_gibbs_like-0-0.12min-2016-11-06_22:11:42-random_medium-0-empty-None.npz'
  path = '/Tmp/huangche/new_generation/2016-11-06_22:11:42-random_medium'
  
  fname = '0_generate_gibbs_like-0-4.73min-2016-11-07_00:33:00-random_medium-0-empty-None.npz'
  path = '/Tmp/huangche/new_generation/2016-11-07_00:33:00-random_medium'
  return os.path.join(path, fname)


def get_process_pianorolls_from_bundle(generation_bundle):
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


#  shape = generated_pianoroll.shape
#  print 'shape', shape
#  pianorolls_by_step = []
#  updated_pianoroll = np.zeros((shape))
#  first_full_idx = None
#  print 'np.product(shape)', np.product(shape)
#  for i, step in enumerate(steps):
#    idx = step.change_to_context[0]
#    updated_pianoroll[idx] = 1
#    print i, np.sum(step.generated_piece - updated_pianoroll)
#    assert np.sum(step.generated_piece - updated_pianoroll) == 0
#    pianorolls_by_step.append(updated_pianoroll.copy)
#    if np.sum(updated_pianoroll) == np.product(shape) and first_full_idx is None:
#      first_full_idx = i
#      print 'first_full_idx', first_full_idx
#  print np.sum(generated_pianoroll - updated_pianoroll)
#  assert np.sum(generated_pianoroll - updated_pianoroll) == 0
#  return pianorolls_by_step


def evaluate_llk_progress():
  fpath = get_fpath()
  #generation_bundle = postprocess.retrieve_generation_bundle(fpath)
  #pianorolls = get_process_pianorolls(generation_bundle)
  pianorolls = get_process_pianorolls(fpath)
  T, _, I = pianorolls[0].shape
  num_blankouts = 8 * 4
  for i, piano_roll in enumerate(pianorolls):
    if i % T * I == 0:
      print i
    if i % num_blankouts == 0:
      print i
    print '(',i, np.sum(piano_roll), ')',

  # Can not collect periodically at num_blankouts b/c they might overlap
  # and then the number of steps are different.
  # Collect full pianorolls.
  pianorolls = [roll for roll in pianorolls if np.sum(roll) == T*I]
  print '# of full pianorolls', len(pianorolls)

  model_name = 'random_medium'
  # Make sure the model used to evaluate is also the model used to generate.
  assert model_name in fpath 
  wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
  wrapped_model.config.hparams.crop_piece_len = pianorolls[0].shape[0]
  print 'piece_len', pianorolls[0].shape

  losses = evaluation_tools.compute_notewise_loss(wrapped_model, pianorolls, model_name)
  assert len(losses) ==  5 * T * I
  loss_by_step = np.mean(np.asarray(losses), axis=0)
  print 'losses', loss_by_step[:10]
  plot_curve(loss_by_step, fpath) 


def plot_curve(curve, source_fpath):
  plt.plot(curve)
  plt.title('NLL against number of Gibbs step taken')
  plt.xlabel('# of Gibbs steps')
  plt.ylabel('Negative log-likelihood (NLL)')

  dirname = os.path.dirname(source_fpath)
  fname = os.path.basename(source_fpath).split('.')[0]
  fpath = os.path.join(dirname, 'plot_%s.png' % fname)
  plt.savefig(fpath) 


def main(argv):
  try:
    evaluate_llk_progress()   
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, KeyboardInterrupt):
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


if __name__ == '__main__':
  tf.app.run()
