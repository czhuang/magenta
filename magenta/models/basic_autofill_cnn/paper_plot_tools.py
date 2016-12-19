from collections import namedtuple
import os
from datetime import datetime

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('plot_dir', '/u/huangche/plots/',
       'Path to the directory where plots will be stored.')


#--------------
# For plotting Bach chorale (prime piece) on poster
def get_sequence():
  from magenta.lib.note_sequence_io import note_sequence_record_iterator
  fpath = '/data/lisatmp4/huangche/data/bach/bwv103.6.tfrecord'
  return list(note_sequence_record_iterator(fpath))[0]

def plot_pianoroll_with_colored_voices_wrapper():
  seq = get_sequence()
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  pianoroll = encoder.encode(seq)[:64]
  T, P, I = pianoroll.shape
  print T, P, I
  # TODO: hack for one-off in pitch
  pianoroll = transpose_down_one(pianoroll)
  plt.figure()
  axis = plt.gca()
  pitch_lb=43
  pitch_ub=77
#  set_pianoroll_style(axis, T, pitch_lb=pitch_lb, pitch_ub=pitch_ub)
  plot_pianoroll_with_colored_voices(pianoroll, pitch_lb=pitch_lb, pitch_ub=pitch_ub)
  path = get_unique_output_path()
  fname_prefix = 'bach'
  plt.savefig(os.path.join(path, 'bach.png'),
     bbox_inches='tight')
  plt.savefig(os.path.join(path, 'bach.pdf'),
     bbox_inches='tight')
  plt.close()
#--------------

# Uitilities
def get_unique_output_path():
  path = FLAGS.plot_dir
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = os.path.join(path, timestamp)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return output_path

def tranpose_down_one(pianoroll):
  T, P, I = pianoroll.shape
  pianoroll = np.concatenate((pianoroll, np.zeros((T, 1, I))), axis=1)
  return np.roll(pianoroll, -1, axis=1)


CONTEXT_COLORS = np.array([[ 0.253935,  0.265254,  0.529983,  1.      ],
       [ 0.163625,  0.471133,  0.558148,  1.      ],
       [ 0.134692,  0.658636,  0.517649,  1.      ],
       [ 0.477504,  0.821444,  0.318195,  1.      ]])
GENERATED_COLORS = np.array([[  4.17642000e-01,   5.64000000e-04,   6.58390000e-01,
          1.00000000e+00],
       [  6.92840000e-01,   1.65141000e-01,   5.64522000e-01,
          1.00000000e+00],
       [  8.81443000e-01,   3.92529000e-01,   3.83229000e-01,
          1.00000000e+00],
       [  9.88260000e-01,   6.52325000e-01,   2.11364000e-01,
          1.00000000e+00]])
CONTEXT_COLORS = GENERATED_COLORS


def get_npz_data(fpath=None):
  if fpath is None:
    # For second main version paper plot on NADE.
    fpath = '/Tmp/huangche/new_generation/fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-0--masker-None--schedule-ConstantSchedule-None---sampler-SequentialSampler-temperature-1e-05--_20161121235937_1.03min.npz'
    # Later moved to /data/lisatmp4
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-0--masker-None--schedule-ConstantSchedule-None---sampler-SequentialSampler-temperature-1e-05--_20161121235937_1.03min.npz'
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps2maskerBernoulliInpaintingMaskercontextkindharmonizationscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161218173017_0.03min.npz'
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps3maskerBernoulliInpaintingMaskercontextkindharmonizationscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161218205028_0.04min.npz'
    # 100 steps independent Gibbs.
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps100maskerBernoulliInpaintingMaskercontextkindharmonizationscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161218213502_1.48min.npz'
  
  data = np.load(fpath)
  pianorolls = data["pianorolls"]
  predictions = data["predictions"]
  masks = data["masks"]
  print len(pianorolls), len(predictions), len(masks)
  print pianorolls.shape, predictions.shape, masks.shape
  S, B, T, P, I = pianorolls.shape
  
  # Take the first one in the batch
  batch_idx = 0
  rolls = pianorolls[:, batch_idx, :, :, :]
  masks = masks[:, batch_idx, :, :, :]
  predictions = predictions[:, batch_idx, :, :, :]
  return rolls, masks, predictions


def get_nade_pianoroll():
  # Second paper plot.
  fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-0--masker-None--schedule-ConstantSchedule-None---sampler-SequentialSampler-temperature-1e-05--_20161121235937_1.03min.npz'
  rolls, masks, predictions = get_npz_data(fpath)
  return rolls[-1]


# Plotting functions
def set_axes_style(ax, total_time, subplots, pitch_lb=None, pitch_ub=None):
  ax.tick_params(axis=u'both', which=u'both',length=0)
  if subplots:
    labelsize=4
  else:
    labelsize=7
  ax.tick_params(axis='both', which='major', labelsize=labelsize)
  ax.tick_params(axis='both', which='minor', labelsize=labelsize)
  pitch_base = 37  # the label is actually one above 36, since starts with 0, ends at 88
  #TODO: hack to fix one-off
  # DID NOT WORK
  #pitch_base = 36
  c_ticks = [12*i + 11 for i in range(4)] 
  y_ticks = [0] + c_ticks + [89-pitch_base]

  # blabel_pitch_base = pitch_base + 1  # the label is actually one above 36, since starts with 0, ends at 88
  # c_ticks = [12*i + label_pitch_base % 12 for i in range(num_pitches/12 + 1)] 
  # label_pitch_top = label_pitch_base + num_pitches
  # if label_pitch_top % 12 == 0:
  #   extra_top_tick = []
  # else:
  #   extra_top_tick = [num_pitches]
  # y_ticks = c_ticks + extra_top_tick
  # print extra_top_tick
  # print 'y_ticks', y_ticks
  ax.yaxis.set_ticks(y_ticks)
  ax.set_yticklabels([pitch_base+tick for tick in y_ticks])
  if pitch_lb is not None and pitch_ub is not None:
    ax.set_ylim([pitch_lb - pitch_base, pitch_ub - pitch_base])

  time_hop = 8
  x_ticks = range(total_time/time_hop + 1)
  ax.xaxis.set_ticks([time_hop*tick for tick in x_ticks])
  ax.set_xticklabels([time_hop*tick for tick in x_ticks])

  plt.setp(ax.spines.values(), linewidth=0.5)
  
  # Get black key positions
  b_ticks = [1, 3, 6, 8, 10]
  
  # TODO: hack to fix one-off
  # DID NOT WORK
  #c_ticks = np.asarray(c_ticks) - 1
  #b_ticks = np.asarray(b_ticks) - 1 
  return c_ticks, b_ticks

def set_pianoroll_style(axis, T, pitch_lb=36, pitch_ub=89, is_subplot=False):
  total_time = T
  # Setting axes styles.  
  c_positions, b_positions = set_axes_style(
      axis, T, is_subplot, pitch_lb, pitch_ub)
  # Turn off tick labels that was set in set_axes_style.
  axis.get_xaxis().set_visible(False)
  axis.get_yaxis().set_visible(False)
  axis.set_frame_on(False)

  # Alpha levels
  bc_alpha = 0.7  # between B, C line
  ef_alpha = 0.4  # between E, F line
  bk_alpha = 0.15  # black key alpha
 
  # Add lines for C.
  offset = 0.5
  c_positions = [c_positions[0] - 12] + c_positions
  b_positions = b_positions + [5]
  #print 'c_positions', c_positions
  for c_pos in c_positions:
    c_y_pos = c_pos-offset
    c_alpha = bc_alpha
    #print c_y_pos
    if c_pos > 0:
      axis.axhline(y=c_y_pos, xmin=0, xmax=total_time, color='royalblue', linewidth='1', alpha=c_alpha)
    # Add dash lines for black keys.
    for b_pos in b_positions:
      y_pos = c_y_pos + b_pos
      alpha = bk_alpha
      if b_pos == 5:
        alpha = ef_alpha
      else:
        y_pos += offset
      #print y_pos
      axis.axhline(y=y_pos, xmin=0, xmax=total_time, color='royalblue', linewidth='1', alpha=alpha)


def plot_pianoroll_with_colored_voices_with_style(ax, pianoroll, pitch_lb, pitch_ub, 
                                                  is_subplot=False, colors=CONTEXT_COLORS, 
                                                  imshow=False, plot_boxes=True, empty_boxes=False):
  T, P, I = pianoroll.shape
  #print T, P, I
  set_pianoroll_style(ax, T, pitch_lb=pitch_lb, pitch_ub=pitch_ub)
  plot_pianoroll_with_colored_voices(
      ax, pianoroll, colors=colors, imshow=imshow, plot_boxes=plot_boxes, empty_boxes=empty_boxes,
      ensure_aspect_equal=True)


# does not create a figure...to be called by others
def plot_pianoroll_with_colored_voices(axis, pianoroll, colors=CONTEXT_COLORS, 
                                       imshow=False, plot_boxes=True, empty_boxes=False, 
                                       ensure_aspect_equal=False):
  T, P, I = pianoroll.shape
  if imshow:
    axis.imshow(pianoroll.sum(axis=2).T, aspect='equal', cmap='Greys',
      origin='lower', interpolation='none')
  if ensure_aspect_equal:
    # Hack to keep the equal aspect ratio.
    axis.imshow(np.zeros_like(pianoroll).sum(axis=2).T, aspect='equal', cmap='Greys',
      origin='lower', interpolation='none')
  if plot_boxes:
    for i in range(I):
      if empty_boxes:
        for t, p in zip(*np.where(pianoroll[:, :, i])):
          axis.add_patch(Rectangle((t-.5, p-.5), 1, 1,
               facecolor='none', edgecolor=colors[i], alpha=0.5))
      else:  
        for t, p in zip(*np.where(pianoroll[:, :, i])):
          axis.add_patch(Rectangle((t-.5, p-.5), 1, 1,
               facecolor=colors[i], edgecolor='none'))


def plot_fancy_pianoroll(ax, T, pitch_lb, pitch_ub, roll, proll, prediction, context, step, is_subplot, plot_current_step=True):
  # with predictions

  # set style of pianoroll lines
  set_pianoroll_style(ax, T, pitch_lb=pitch_lb, pitch_ub=pitch_ub, is_subplot=True)

  # plot context
  plot_pianoroll_with_colored_voices(ax, context)
  
  # plot blankout
  #plot_pianoroll_with_colored_voices(ax, blankouts, empty_boxes=True)

  # plot prediction
  plot_pianoroll_with_colored_voices(ax, prediction, imshow=True, plot_boxes=False)

  # plot generated
  plot_pianoroll_with_colored_voices(ax, proll - context, colors=GENERATED_COLORS)

  if plot_current_step:
    # plot current step
    plot_pianoroll_with_colored_voices(ax, roll - proll, colors=GENERATED_COLORS, empty_boxes=True)

  ax.set_title('Step %d' % (step-2))


def plot_mask(ax, mask):
    """Show instrument by time mask."""
    T, P, I = mask.shape
    #print T, P, I
    mask = np.clip(mask.sum(axis=1), 0, 1)
    # Instrument by time
    ax.imshow(np.flipud(1-mask.T), aspect='equal', cmap='Greys', origin='lower', interpolation='none')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def get_process():
  rolls, masks, predictions = get_npz_data()
  S, T, P, I = rolls.shape
  original = rolls[0]
  context = rolls[1]
  blankouts = original - context
  #plot_steps = [2, int(S/8), int(S/4), int(S/2), S-1]  
  # Paper plot steps, on NADE, was for second version of paper.
  plot_steps = [2, 3, 6, 18, S-1]  
  
  is_subplot = True 
  figs, axes = plt.subplots(2, 3, figsize=(11, 6))
  axes = axes.ravel()
  pitch_lb=43 #36 #43
  pitch_ub=72 #89 #77
  for i in range(len(plot_steps)):
    ax = axes[i]
    step = plot_steps[i]
    roll = rolls[step]
    proll = rolls[step-1]
    prediction = predictions[step] 

    plot_fancy_pianoroll(
        ax, T, pitch_lb, pitch_ub, roll, proll, prediction, context, step, is_subplot)

  # Showing the original.
  ax = axes[-1]
  plot_pianoroll_with_colored_voices_with_style(ax, original, pitch_lb, pitch_ub, is_subplot=True)
  ax.set_title('Ground Truth')
 
  path = get_unique_output_path()
  fname_prefix = 'process'
  output_path = os.path.join(path, fname_prefix + '.png') 
  print 'Plotting to', output_path
  plt.savefig(output_path, bbox_inches='tight')
  plt.savefig(os.path.join(path, fname_prefix + '.pdf'), bbox_inches='tight')
  plt.close()


def plot_mask_prediction_realizations():
  rolls, masks, predictions = get_npz_data()
  S, T, P, I = rolls.shape
  print 'rolls.shape', rolls.shape
  original = rolls[0]
  context = rolls[1]
  blankouts = original - context
  plot_steps = [2, 3, 522, S-1]
  is_subplot = True
  n_rows = 4
  n_cols = 4
  sz_of_subplot = 4
  height = sz_of_subplot * n_rows
  width = sz_of_subplot * n_cols
  figs, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
  pitch_lb=43 #36 #43
  pitch_ub=72 #89 #77

  for i, step in enumerate(plot_steps):
    # plot mask
    ax = axes[i, 0]
    mask = masks[step]
    plot_mask(ax, mask) 

    # plot predictions
    ax = axes[i, 1]
    roll = rolls[step]
    proll = rolls[step-1]
    prediction = predictions[step] 
    
    # with context
    plot_pianoroll_with_colored_voices_with_style(ax, context, pitch_lb, pitch_ub, is_subplot)
    # with prediction
    plot_pianoroll_with_colored_voices(ax, prediction, imshow=True, plot_boxes=False)
    # with generated context
    plot_pianoroll_with_colored_voices(ax, proll*(1-mask), colors=GENERATED_COLORS)

    #plot_fancy_pianoroll(
    #    ax, T, pitch_lb, pitch_ub, roll, proll, prediction, context, step, is_subplot, 
    #    plot_current_step=False)

    # plot realizations
    ax = axes[i, 2]
    #print np.sum(roll)
    plot_pianoroll_with_colored_voices_with_style(ax, roll, pitch_lb, pitch_ub, is_subplot)
    ax.set_title('Step %d' % (step-2))

  # Shows NADE generated pianoroll.
  # this example was for the transition inpainting.
  #ax = axes[-2, 3]
  #nade_pianoroll = get_nade_pianoroll()
  #plot_pianoroll_with_colored_voices_with_style(ax, nade_pianoroll, pitch_lb, pitch_ub, is_subplot)
  #ax.set_title('NADE')

  # Showing the original.
  ax = axes[-1, 3]
  plot_pianoroll_with_colored_voices_with_style(ax, original, pitch_lb, pitch_ub, is_subplot)
  ax.set_title('Ground Truth')
 
  path = get_unique_output_path()
  fname_prefix = 'gibbs_process'
  output_path = os.path.join(path, fname_prefix + '.png') 
  print 'Plotting to', output_path
  plt.savefig(output_path, bbox_inches='tight')
  plt.savefig(os.path.join(path, fname_prefix + '.pdf'), bbox_inches='tight')
  plt.close()


def main(argv):
  # Paper plot for NADE process.
  # get_process()

  plot_mask_prediction_realizations()


if __name__ == '__main__':
  #tf.app.run()
  main(None)
