"""Tools for plotting training setup and results."""
from collections import namedtuple
import os

 

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
import tensorflow as tf

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import seed_tools
from  magenta.lib import midi_io

FLAGS f.app.flags.FLAGS
tf.app.flags.DEFINE_string('plot_dir', '/usr/local/ /home/annahuang/magenta_tmp/generated/',
       Path to the directory where plots will be stored.')

INSTR_ORDERING 1, 0, 2, 3]


def get_unique_output_path():
  path LAGS.output_dir
  timestamp_str onfig_tools.get_current_time_as_str()
  output_path s.path.join(path, timestamp_str)
  if not os.path.exists(output_path):
 os.makedirs(output_path)
  return output_path


def get_sequence():
  fpath / /src/cloud/annahuang/annahuang0-annahuang-basic_autofill_cnn-separate_voices-git5/magenta/models/basic_autofill_cnn/testdata/generated/sample1.midi'
  return midi_io.midi_file_to_sequence_proto(fpath)


def get_input():
  seq et_sequence()
  encoder ianorolls_lib.PianorollEncoderDecoder()
  pianoroll ncoder.encode(seq)
  return pianoroll

def set_axes_style(ax):
  ide tick marks.
  ax.tick_params(axis=u'both', which=u'both',length=0)
  ut show tick labels.
  ax.tick_params(axis='both', which='major', labelsize=7)
  ax.tick_params(axis='both', which='minor', labelsize=7)
  itch ticks.
  y_ticks 0, 30, 60]
  ax.yaxis.set_ticks(y_ticks)
  pitch_base 3
  ax.set_yticklabels([pitch_base+tick for tick in y_ticks])

  ime ticks.
  x_ticks 0, 1, 2]
  time_base 6
  ax.xaxis.set_ticks([time_base*tick for tick in x_ticks])
  ax.set_xticklabels([time_base*tick for tick in x_ticks])

  xes thinner.
  plt.setp(ax.spines.values(), linewidth=0.5)

def plot_summed_pianoroll(pianoroll, crop_length, ax):
  summed_pianoroll p.clip(pianoroll[:crop_length].sum(axis=2), 0, 1)
  plt.imshow(summed_pianoroll.T, aspect='equal', cmap='Greys',
    origin='lower', interpolation='none')
  set_axes_style(ax)
  plt.ylabel('pitch')
  plt.xlabel('time')

def plot_input():
  seq et_sequence()
  encoder ianorolls_lib.PianorollEncoderDecoder()
  pianoroll ncoder.encode(seq)
  crop_length 2  #64

  plt.figure(figsize=(6, 20))
  lot individual instruments.
  for n INSTR_ORDERING:
 ax lt.subplot(5, 1,  1)
 plt.imshow(pianoroll[:crop_length, :, i].T, aspect='equal', cmap='Greys',
    origin='lower', interpolation='none')
 set_axes_style(ax)
 if = len(INSTR_ORDERING) :
   ll but last
   ax.set_xticklabels(())
 if = 1:
   plt.ylabel('pitch')
 if = len(INSTR_ORDERING) :
   plt.xlabel('time')

  lot all instruments combined.
  ax lt.subplot(515)
  plot_summed_pianoroll(pianoroll, crop_length, ax)

  plt.savefig(os.path.join(get_unique_output_path(), 'input_vertical.png'),
     bbox_inches='tight')
  plt.close()


def plot_blankout():
  fpath et_unique_output_path()
  et multiple to get the one we want.
  pianoroll et_input()
  crop_length 2  #64
  pianoroll ianoroll[:crop_length, :, :]
  plot_horizontal rue

  ake colormap for target to be magenta.
  target_cmap olors.ListedColormap(['white', 'magenta'])
  bounds=[0, 0.5, 1]
  target_norm olors.BoundaryNorm(bounds, target_cmap.N)
  title_strs 'Masked inputs', 'Masks', 'Targets']

  for n range(200):
 blankouts ask_tools.get_multiple_random_instrument_time_mask(pianoroll.shape, 4, 4)
 input_data ask_tools.apply_mask_and_stack(pianoroll, blankouts)
 targets ianoroll lankouts
 print 'sum of masks:', np.sum(input_data[:, :, 4:])

 if plot_horizontal:
   plt.figure(figsize=(13, 20))
 plt.figure(figsize=(13, 20))
 # Plot individual instruments.
 for n INSTR_ORDERING:
   for type_ in range(3):
  print '\n', i, type_
  plot_idx   ype_

  # If mask, then instr_idx should be offset by num_instruments.
  instr_idx 
  if type_ == 1:
    instr_idx  
  print 'plot_idx, instr_idx', plot_idx , instr_idx
  ax lt.subplot(5, 3, plot_idx+1)
  if type_ in range(2):
    plt.imshow(input_data[:crop_length, :, instr_idx].T,
     spect='equal', cmap='Greys',
     rigin='lower', interpolation='none')
  else:
    plt.imshow(targets[:crop_length, :, instr_idx].T,
      interpolation='nearest', origin='lower',
      cmap=target_cmap, norm=target_norm)
  set_axes_style(ax)
  if = len(INSTR_ORDERING) :
    ll but last
    ax.set_xticklabels(())
  if = 1:
    plt.ylabel('pitch')
  if = len(INSTR_ORDERING) :
    plt.xlabel('time')
  if plot_idx in range(3):
    plt.title(title_strs[plot_idx])

 # Plot all instruments combined.  Just the pianoroll part.
 ax lt.subplot(5, 3, 13)
 summed_pianoroll p.clip(input_data[:, :, :4].sum(axis=2), 0, 1)
 plt.imshow(summed_pianoroll.T, aspect='equal', cmap='Greys',
    rigin='lower', interpolation='none')
 set_axes_style(ax)
 plt.ylabel('pitch')
 plt.xlabel('time')

 # Plot target part of all instruments combined.
 ax lt.subplot(5, 3, 15)
 summed_pianoroll p.clip(targets.sum(axis=2), 0, 1)
 plt.imshow(summed_pianoroll.T,
    nterpolation='nearest', origin='lower',
    map=target_cmap, norm=target_norm)
 set_axes_style(ax)
 plt.ylabel('pitch')
 plt.xlabel('time')

 plt.savefig(os.path.join(fpath, 'blankouts-%d.png' ),
    bbox_inches='tight')
 plt.close()


def plot_steps(autofill_steps, original_pianoroll):
  maskedout_instr_indices et()
  shape utofill_steps[0].prediction.shape
  already_generated_pianoroll p.zeros(shape)
  previous_already_generated_pianoroll lready_generated_pianoroll.copy()
  for i, step in enumerate(autofill_steps):
 plt.figure()
 axis lt.gca()

 # Update change.
 change_index tep.change_to_context[0]
 time_step, pitch, instr_idx hange_index
 maskedout_instr_indices.add(instr_idx)
 already_generated_pianoroll[change_index] 

 # TODO(annahuang): Seems to have lost the current instr original context

 # Mark the context that is conditioned on as darker.
 print 'maskedout_instr_indices', maskedout_instr_indices
 original_context p.delete(original_pianoroll, list(maskedout_instr_indices), 2)
 print original_pianoroll.shape
 for t, n zip(*np.where(original_context.sum(axis=2))):
   axis.add_patch(Rectangle((t 5,  .5), 1, 1,
        acecolor="grey", edgecolor="grey"))

 # Prediction for mask outs of current instrument.
 prediction_for_masked tep.prediction[:, :, instr_idx].T 
  1 revious_already_generated_pianoroll[:, :, instr_idx].T)
 #plt.imshow(prediction_for_masked, origin='lower',
 #   nterpolation='none', aspect='auto', cmap='summer')

 # Trying log scale.
 plt.imshow(np.log(prediction_for_masked), origin='lower',
    nterpolation='none', aspect='auto', cmap='summer')

 # Prediction on the current context.
 #predictions_for_context p.delete(
 # step.prediction.T, instr_idx, 2)
 #plt.imshow(prediction_for_masked, origin='lower',
 #   nterpolation='none', aspect='auto', cmap='summer', alpha=0.4)

 # Marks the already generated as magenta.
 for t, n zip(*np.where(previous_already_generated_pianoroll.sum(axis=2))):
   axis.add_patch(Rectangle((t 5,  .5), 1, 1,
        acecolor='magenta', edgecolor="magenta"))

 # Mark the current instrument original.
 current_context riginal_pianoroll[:, :, instr_idx]
 print current_context.shape
 for t, n zip(*np.where(current_context)):
   axis.add_patch(Rectangle((t 5,  .5), 1, 1,
        ill=None, edgecolor="grey"))

 # Marks the current change.
 axis.add_patch(Rectangle((time_step 5, pitch 5), 1, 1,
        fill=None, edgecolor="navy"))

 plt.colorbar()
 plt.title(repr(step.change_to_context))
 plt.savefig(os.path.join(output_path, 'run_id%s-iter_%d.png' run_id, i)))
 plt.close()

 previous_already_generated_pianoroll lready_generated_pianoroll.copy()


def main(argv):
  plot_input()
  #plot_blankout()



if __name__ == '__main__':
  tf.app.run()

