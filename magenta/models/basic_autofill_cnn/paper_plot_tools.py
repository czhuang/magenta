from collections import namedtuple
import os
from datetime import datetime

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
#import tensorflow as tf

#FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_string('plot_dir', '/u/huangche/plots/',
#       'Path to the directory where plots will be stored.')


#--------------
# For plotting Bach chorale (prime piece) on poster
def get_sequence():
  from magenta.music.note_sequence_io import note_sequence_record_iterator
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
  path = '/u/huangche/plots/'
  #path = FLAGS.plot_dir
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
#CONTEXT_COLORS = GENERATED_COLORS


def get_fpath(context_kind, sampling_method):
  # NADE samples.
  if context_kind == 'transition' and sampling_method == 'nade':
    # Transition, NADE, second main version paper plot.
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=bach_Gibbs-num-steps-0--masker-None--schedule-ConstantSchedule-None---sampler-SequentialSampler-temperature-1e-05--_20161121235937_1.03min.npz'
  elif context_kind == 'harmonization' and sampling_method == 'nade':
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=sequential_Gibbsnumsteps0maskerBernoulliInpaintingMaskercontextkindharmonizationscheduleConstantScheduleNonesamplerNone_20161220190944_0.00min.npz'

  # YAO samples.  
  elif context_kind == 'transition' and sampling_method == 'yao':
    # Transition, 100 steps independent Gibbs.
#    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps100maskerBernoulliInpaintingMaskercontextkindtransitionscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161220160427_1.41min.npz'

    # Transition, 64 steps of I. gibbs (to use the same number of model evaluations as NADE.
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps64maskerBernoulliInpaintingMaskercontextkindtransitionscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161220195113_0.90min.npz'

    # Take 2 of above.
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps64maskerBernoulliInpaintingMaskercontextkindtransitionscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161220200126_0.91min.npz'

    # Take 3 of above.
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps64maskerBernoulliInpaintingMaskercontextkindtransitionscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161220212451_0.90min.npz'

  elif context_kind == 'harmonization' and sampling_method == 'yao':
     # TODO: still ned to run one that uses the same number of model evaluations as NADE.
     # Harmonization, 100 steps independent Gibbs.
    fpath = '/data/lisatmp4/huangche/new_generated/fromscratch_balanced_by_scaling_init=independent_Gibbsnumsteps100maskerBernoulliInpaintingMaskercontextkindharmonizationscheduleYaoSchedulepmin01pmax09alpha07samplerIndependentSamplertemperature1e05_20161218213502_1.48min.npz'

  else:
    assert False
  return fpath


def get_npz_data(fpath=None, batch_idx=None):
  data = np.load(fpath)
  pianorolls = data["pianorolls"]
  predictions = data["predictions"]
  masks = data["masks"]
  print len(pianorolls), len(predictions), len(masks)
  print pianorolls.shape, predictions.shape, masks.shape
  S, B, T, P, I = pianorolls.shape
  
  if batch_idx is None:
    return pianorolls, masks, predictions
  rolls = pianorolls[:, batch_idx, :, :, :]
  masks = masks[:, batch_idx, :, :, :]
  predictions = predictions[:, batch_idx, :, :, :]
  return rolls, masks, predictions


def get_gibbs_process(context_kind, sampling_method, batch_idx):
  fpath = get_fpath(context_kind, sampling_method)  
  return get_npz_data(fpath, batch_idx)


def get_nade_pianorolls(context_kind, batch_idx):
  fpath = get_fpath(context_kind, 'nade')
  rolls, masks, predictions = get_npz_data(fpath, batch_idx)
  step_idx = len(rolls) - 2 
  return rolls[-1], step_idx


# Plotting functions
def set_axes_style(ax, total_time, subplots, pitch_lb=None, pitch_ub=None):
  ax.tick_params(axis=u'both', which=u'both',length=0)
  if subplots:
    labelsize=4
  else:
    labelsize=7
  ax.tick_params(axis='both', which='major', labelsize=labelsize)
  ax.tick_params(axis='both', which='minor', labelsize=labelsize)
  # For 4part plots:
  #pitch_base =  pitch_lb + 1 # the label is actually one above 36=37, since starts with 0, ends at 88
  # TODO: seems we can just use pitch_lb instead of pitch_base.
  pitch_base = pitch_lb
  #TODO: hack to fix one-off
  # DID NOT WORK
  #pitch_base = 36
  c_ticks = [12*i + 11 for i in range(4)] 
  y_ticks = [0] + c_ticks + [pitch_ub - pitch_base]

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
    ax.set_ylim([pitch_lb - pitch_base - 1, pitch_ub - pitch_base + 1])

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
                                       ensure_aspect_equal=False, **kwargs):
  T, P, I = pianoroll.shape
  if 'aspect_equal' in kwargs:
    aspect_equal = kwargs['aspect_equal']
    aspect = 'equal' if aspect_equal else 'auto'
  else:
    aspect = 'auto'
  if imshow:
    axis.imshow(pianoroll.sum(axis=2).T, aspect=aspect, cmap='Greys',
      origin='lower', interpolation='none')
  assert (aspect_equal and ensure_aspect_equal) or not ensure_aspect_equal
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


def plot_fancy_pianoroll(ax, T, pitch_lb, pitch_ub, roll, proll, prediction, context, step, is_subplot, plot_current_step=True, plot_blankouts=False):
  # with predictions

  # set style of pianoroll lines
  set_pianoroll_style(ax, T, pitch_lb=pitch_lb, pitch_ub=pitch_ub, is_subplot=True)

  # plot context
  plot_pianoroll_with_colored_voices(ax, context, **kwargs)
  
  # plot blankout
  # plot_pianoroll_with_colored_voices(ax, blankouts, empty_boxes=True, **kwargs)

  ## plot prediction
  #plot_pianoroll_with_colored_voices(ax, prediction, imshow=True, plot_boxes=False, **kwargs)

  ## plot generated
  #plot_pianoroll_with_colored_voices(ax, proll - context, colors=GENERATED_COLORS, **kwargs)

  if plot_current_step:
    # plot current step
    plot_pianoroll_with_colored_voices(ax, roll - proll, colors=GENERATED_COLORS, empty_boxes=True, **kwargs)

  ax.set_title('Step %d' % (step-2))


def plot_inspect_pianoroll(ax, T, pitch_lb, pitch_ub, ground_truth, roll, proll, prediction, context, step, is_subplot, plot_current_step=True, plot_blankouts=False, blankouts=None, **kwargs):

  # set style of pianoroll lines
  set_pianoroll_style(ax, T, pitch_lb=pitch_lb, pitch_ub=pitch_ub, is_subplot=True)

  # plot context
  plot_pianoroll_with_colored_voices(ax, context, **kwargs)
  
  # plot ground truth
  plot_pianoroll_with_colored_voices(
      ax, ground_truth, empty_boxes=True, colors=CONTEXT_COLORS, **kwargs)
  
  # plot blankout
  #if plot_blankouts:
  #  assert blankouts is not None
  #  plot_pianoroll_with_colored_voices(ax, blankouts, empty_boxes=True, **kwargs)

  # plot prediction
  plot_pianoroll_with_colored_voices(ax, prediction, imshow=True, plot_boxes=False, **kwargs)

  ## plot generated
  plot_pianoroll_with_colored_voices(ax, proll - context, colors=GENERATED_COLORS, **kwargs)

  if plot_current_step:
    # plot current step
    #plot_pianoroll_with_colored_voices(ax, roll - proll, colors=GENERATED_COLORS, empty_boxes=True, **kwargs)
    t, p = step
    color = GENERATED_COLORS[0]
    ax.add_patch(Rectangle((t-.5, p-.5), 1, 1,
                 facecolor=color, edgecolor=color))#, alpha=0.5))


def plot_mask(ax, mask, axes_visible=False):
    """Show instrument by time mask."""
    T, P, I = mask.shape
    #print T, P, I
    mask = np.clip(mask.sum(axis=1), 0, 1)
    # Instrument by time.
    ax.imshow(np.flipud(1-mask.T), aspect='equal', cmap='Greys', origin='lower', interpolation='none')
    ax.get_xaxis().set_visible(axes_visible)
    ax.get_yaxis().set_visible(axes_visible)


def plot_mask_colored(ax, mask, axes_visible=False):
    plot_mask(ax, mask, axes_visible)
    T, P, I = mask.shape
    colors = CONTEXT_COLORS[::-1]
    #mask = np.flipud(1-mask.T)
    mask = mask[:, 0, :]
    mask = np.fliplr(1 - mask)
    for i in range(I):
     for t in np.where(mask[:, i])[0]:
       ax.add_patch(Rectangle((t-.5, i-.5), 1, 1,
           facecolor=colors[i], edgecolor='none'))


def get_pitch_bounds(rolls, repr_lb=36, repr_ub=88):
  pitch_lb, pitch_ub = [127, 0]
  T, P, I = rolls[0].shape
  assert repr_ub - repr_lb + 1 == P
  for roll in rolls:
    roll = roll.sum(axis=-1)
    pitches = np.where(roll)[1] 
    pitch_lb = np.minimum(pitch_lb, np.min(pitches))
    pitch_ub = np.maximum(pitch_ub, np.max(pitches))
  pitch_lb += repr_lb
  pitch_ub += repr_lb + 2
  print pitch_lb, pitch_ub
  return pitch_lb, pitch_ub


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


def plot_all_mask_prediction_realizations():
  CONTEXT_KIND = 'transition'
  #CONTEXT_KIND = 'harmonization'
  path = get_unique_output_path()
  #rolls, masks, predictions = get_gibbs_process(CONTEXT_KIND, 'yao')
  rolls, masks, predictions = get_gibbs_process(CONTEXT_KIND, 'yao', None)
  nade_pianorolls, nade_step_idx = get_nade_pianorolls(CONTEXT_KIND, None)
  S, B, T, P, I = rolls.shape
  print rolls.shape, masks.shape, predictions.shape, nade_pianorolls.shape
#  for batch_idx in range(B):
  for batch_idx in [53]:
  #for batch_idx in range(1):
    plot_mask_prediction_realizations(
        rolls[:, batch_idx, :, :, :], masks[:, batch_idx, :, :, :], 
        predictions[:, batch_idx, :, :, :], nade_pianorolls[batch_idx, :, :, :],
        nade_step_idx, batch_idx, CONTEXT_KIND, path)


def plot_mask_prediction_realizations(rolls, masks, predictions,
                                      nade_pianoroll, nade_step_idx, 
                                      batch_idx, context_kind, path):
  ROW_FOR_TIME = True
  S, T, P, I = rolls.shape
  print 'rolls.shape', rolls.shape
  original = rolls[0]
  context = rolls[1]
  blankouts = original - context
  plot_steps = [2, 3, (S-1)/2, S-1]
  plot_steps = [2, 3, 16+2, S-1]
  is_subplot = True

  n_rows = 4
  n_cols = 4

  n_rows = len(plot_steps) + 1
  n_cols = 3
  if not ROW_FOR_TIME:
    temp = n_rows
    n_rows = n_cols
    n_cols = temp
  sz_of_subplot = 4
  height = sz_of_subplot * n_rows
  width = sz_of_subplot * n_cols
  figs, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))

  pitch_lb=43 #36 #43
  pitch_ub=72 #89 #77
  
  pitch_lb, pitch_ub = get_pitch_bounds([original, nade_pianoroll, rolls[-1]])

  for i, step in enumerate(plot_steps):
    # plot mask
    if ROW_FOR_TIME:
      ax = axes[i, 0]
    else:
      ax = axes[0, i]
    mask = masks[step]
    plot_mask_colored(ax, mask) 
    if i == 0:
      ax.set_title('Annealed mask\nfor independent Gibbs\nStep %d' % (step-2))
      ax.set_ylabel('Instrument')
    else:
      ax.set_title('Step %d' % (step-2))
  
    if i == len(plot_steps) - 1:
      ax.set_xlabel('Time')

    # plot predictions
    if ROW_FOR_TIME:
      ax = axes[i, 1]
    else:
      ax = axes[0, i]
    roll = rolls[step]
    proll = rolls[step-1]
    prediction = predictions[step] 
    
    # with context
    plot_pianoroll_with_colored_voices_with_style(ax, context, pitch_lb, pitch_ub, is_subplot)
    # with prediction
    plot_pianoroll_with_colored_voices(ax, prediction, imshow=True, plot_boxes=False)
    # with generated context
    plot_pianoroll_with_colored_voices(ax, proll*(1-mask), colors=GENERATED_COLORS)
    if i == 0:
      ax.set_title('Independent Gibbs\nStep %d (predictions)' % (step-2))
    else:
      ax.set_title('Step %d' % (step-2))

    # plot realizations
    if ROW_FOR_TIME: 
      ax = axes[i, 2]
    else:
      ax = axes[2, i]
    #print np.sum(roll)
    plot_pianoroll_with_colored_voices_with_style(ax, roll, pitch_lb, pitch_ub, is_subplot)
    if i == 0:
      ax.set_title('Independent Gibbs\nStep %d (sampled realizations)' % (step-2))
    elif i == len(plot_steps) - 1:
      ax.set_title('Independent Gibbs\nStep %d' % (step-2))
    else:
      ax.set_title('Step %d' % (step-2))

  # Shows NADE generated pianoroll.
  # this example was for the transition inpainting.
  if ROW_FOR_TIME:
    ax = axes[-1, 1]
  else:
    ax = axes[1, -1]
  plot_pianoroll_with_colored_voices_with_style(ax, nade_pianoroll, pitch_lb, pitch_ub, is_subplot)
  #ax.set_title('NADE\nStep %d' % (nade_step_idx - 1))
  ax.set_title('NADE')

  # Showing the original.
  if ROW_FOR_TIME:
    ax = axes[-1, 2]
  else:
    ax = axes[2, -1]
  plot_pianoroll_with_colored_voices_with_style(ax, original, pitch_lb, pitch_ub, is_subplot)
  ax.set_title('Ground Truth')

  # Remove axis for empty slot.
  if ROW_FOR_TIME:
    ax = axes[-1, 0]
  else:
    ax = axes[0, -1]
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
 
  fname_prefix = 'gibbs_%s-%d' % (context_kind, batch_idx) 
  if not ROW_FOR_TIME:
    fname_prefix += 'column_as_step'
  output_path = os.path.join(path, fname_prefix + '.png') 
  print 'Plotting to', output_path
  plt.savefig(output_path, bbox_inches='tight')
  plt.savefig(os.path.join(path, fname_prefix + '.pdf'), bbox_inches='tight')
  plt.close()


def main(argv):
  # Paper plot for NADE process.
  # get_process()

  plot_all_mask_prediction_realizations()


if __name__ == '__main__':
  #tf.app.run()
  main(None)
