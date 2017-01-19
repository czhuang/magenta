r"""Trains a convolution net model class for music autofill.

Example usage:
    $ bazel run :basic_autofill_cnn_train -- \
        --run_dir=/tmp/cnn_autofill_logs --log_progress \
        --maskout_method=random_patches \
        --hparams='{"batch_size":20, "num_layers":16, "num_filters":128}'
"""
import os
import time
import sys

import yaml

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import data_tools
from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import summary_tools
from magenta.models.basic_autofill_cnn.basic_autofill_cnn_graph import BasicAutofillCNNGraph
from magenta.models.basic_autofill_cnn.basic_autofill_cnn_graph import build_placeholders_initializers_graph
from magenta.models.basic_autofill_cnn.hparams_tools import Hyperparameters


# '/u/huangche/data/bach/instrs=4_duration=0.250_sep=True',
#    '/u/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True',
#    'input_dir', '/Tmp/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True',
FLAGS = tf.app.flags.FLAGS
# TODO(annahuang): Set the default input and output_dir to None for opensource.
tf.app.flags.DEFINE_string(
    'input_dir', '/data/lisatmp4/huangche/data/',
    'Path to the base directory for different datasets.')
tf.app.flags.DEFINE_string('run_dir', '/u/huangche/logs',
                           'Path to the directory where checkpoints and '
                           'summary events will be saved during training and '
                           'evaluation. Multiple runs can be stored within the '
                           'parent directory of `run_dir`. Point TensorBoard '
                           'to the parent directory of `run_dir` to see all '
                           'your runs.')
tf.app.flags.DEFINE_bool('log_progress', True,
                         'If False, do not log any checkpoints and summary'
                         'statistics.')

# Dataset.
tf.app.flags.DEFINE_string('dataset', '4part_Bach_chorales', '4part_JSB_Chorales,' 
                           ' JSB_Chorales, MuseData, Nottingham, Piano-midi.de')
# Later on have a lookup table for different datasets.
tf.app.flags.DEFINE_integer('num_instruments', 4, 
                        'Maximum number of instruments that appear in this dataset.  Use 0 if not separating instruments and hence does not matter how many there are.')
tf.app.flags.DEFINE_bool('separate_instruments', True,
                         'Separate instruments into different input feature'
                         'maps or not.')
tf.app.flags.DEFINE_integer('crop_piece_len', 64, 'The number of time steps included in a crop')
tf.app.flags.DEFINE_integer('encode_silences', False, 'Encode silence as the lowest pitch.')

# Model architecture.
tf.app.flags.DEFINE_string('model_name', None,
                           'A string specifying the name of the model.  The '
                           'choices are currently "PitchLocallyConnectedConvSpecs", '
                           '"PitchFullyConnectedConvSpecs", '
                           '"DeepStraightConvSpecs", and '
                           '"DeepStraightConvSpecsWithEmbedding".')
tf.app.flags.DEFINE_integer('num_layers', 64,
                            'The number of convolutional layers.')
tf.app.flags.DEFINE_integer('num_filters', 128,
                            'The number of filters for each convolutional '
                            'layer.')
# TODO(annahuang): Some are meant to be booleans.
tf.app.flags.DEFINE_integer('use_residual', 1,
                            '1 specifies use residual, while 0 specifies not '
                            'to.')
tf.app.flags.DEFINE_integer('batch_size', 20,
                            'The batch size training and validation the model.')

# Mask related.
tf.app.flags.DEFINE_string('maskout_method', 'balanced_by_scaling', 
                           "The choices include: 'bernoulli', "
                           "'random_patches', 'random_pitch_range',"
                           'random_time_range, random_multiple_instrument_time, '
                           'random_multiple_instrument_time,'
                           'random_easy, random_medium, random_hard,'
                           'chronological_ti, chronological_it, fixed_order, '
                           'balanced, and balanced_by_scaling (which '
                           'invokes gradient rescaling as per NADE).')
tf.app.flags.DEFINE_bool('mask_indicates_context', True, 
                         'Feed inverted mask into convnet so that zero-padding makes sense')
tf.app.flags.DEFINE_bool('optimize_mask_only', False, 'optimize masked predictions only')
tf.app.flags.DEFINE_bool('rescale_loss', True, 'Rescale loss based on context size.')
tf.app.flags.DEFINE_integer('patience', 5, 'Number of epochs to wait for improvement before decaying the learning rate.')

# Data Augmentation.
tf.app.flags.DEFINE_integer('augment_by_transposing', 0, 'If true during '
                            'training shifts each data point by a random '
                            'interval between -5 and 6 ')
tf.app.flags.DEFINE_integer('augment_by_halfing_doubling_durations', 0, 'If '
                            'true during training randomly chooses to double '
                            'or halve durations or stay the same.  The former '
                            'two options are only available if they do not '
                            'go outside of the original set of durations.')
# Denoise mode.
tf.app.flags.DEFINE_bool('denoise_mode', False, 'Instead of blankout, randomly add perturb noise.  Hence instead of inpainting, model learns to denoise.')
tf.app.flags.DEFINE_float('corrupt_ratio', 0.5, 'Ratio to blankout (or perturb in case of denoising).')
# Run parameters.
tf.app.flags.DEFINE_integer('num_epochs', 0,
                            'The number of epochs to train the model. Default '
                            'is 0, which means to run until terminated '
                            'manually.')
tf.app.flags.DEFINE_integer('save_model_secs', 30,
                            'The number of seconds between saving each '
                            'checkpoint.')


import contextlib
@contextlib.contextmanager
def pdb_post_mortem():
  try:
    yield
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
      import traceback
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


def run_epoch(supervisor,
              sess,
              m,
              raw_data,
              encoder,
              hparams,
              eval_op,
              experiment_type,
              epoch_count,
              best_validation_loss=None,
              best_model_saver=None):
  """Runs an epoch of training or evaluate the model on given data."""
  input_data, targets = data_tools.make_data_feature_maps(raw_data, hparams,
                                                          encoder)
  permutation = np.random.permutation(len(input_data))
  input_data = input_data[permutation]
  targets = targets[permutation]

  # TODO(annahuang): Leaves out last incomplete minibatch, needs wrap around.
  batch_size = m.batch_size
  num_batches = input_data.shape[0] // m.batch_size

  losses = summary_tools.AggregateMean('losses_%s' % experiment_type)
  losses_total = summary_tools.AggregateMean('losses_total_%s' %
                                             experiment_type)
  losses_mask = summary_tools.AggregateMean('losses_mask_%s' % experiment_type)
  losses_unmask = summary_tools.AggregateMean('losses_unmasked_%s' %
                                              (experiment_type))
  start_time = time.time()
  for step in range(num_batches):
    start_idx = step * batch_size
    end_idx = start_idx + batch_size
    x = input_data[start_idx:end_idx, :, :, :]
    y = targets[start_idx:end_idx, :, :]

    # Evaluate the graph and run back propagation.
    results = sess.run([m.predictions, m.loss, m.loss_total, m.loss_mask,
                        m.mask_size, m.mask, m.loss_unmask, m.unmask_size,
                        m.learning_rate, m._lossmask,
                        eval_op], {m.input_data: x,
                                   m.targets: y})

    (predictions, loss, loss_total, loss_mask, mask_size, mask, loss_unmask,
     unmask_size, learning_rate, lossmask, _) = results
    #print 'predictions', np.sum(predictions) / np.product(predictions.shape)
    #print 'mask_sizes', mask_size, unmask_size, '%.4f, %.4f, %.4f' % (loss, loss_total, loss_mask)
 
    # Aggregate performances.
    losses_total.add(loss_total, 1)
    # Multiply the mean loss_mask by mask_size for aggregation as the mask size
    # could be different for every batch.
    losses_mask.add(loss_mask * mask_size, mask_size)
    losses_unmask.add(loss_unmask * unmask_size, unmask_size)

    if hparams.optimize_mask_only:
      losses.add(loss * mask_size, mask_size)
    else:
      losses.add(loss, 1)

  # Collect run statistics.
  run_stats = dict()
  run_stats['loss_mask_%s' % experiment_type] = losses_mask.mean
  run_stats['loss_unmask_%s' % experiment_type] = losses_unmask.mean
  run_stats['loss_total_%s' % experiment_type] = losses_total.mean
  run_stats['loss_%s' % experiment_type] = losses.mean

  run_stats['perplexity_mask_%s' % experiment_type] = np.exp(losses_mask.mean)
  run_stats['perplexity_unmask_%s' % experiment_type] = (
      np.exp(losses_unmask.mean))
  run_stats['perplexity_total_%s' % experiment_type] = np.exp(losses_total.mean)
  run_stats['perplexity_%s' % experiment_type] = np.exp(losses.mean)
  run_stats['learning_rate'] = float(learning_rate)

  # Make summaries.
  if FLAGS.log_progress:
    summaries = tf.Summary()
    for stat_name, stat in run_stats.iteritems():
      value = summaries.value.add()
      value.tag = stat_name
      value.simple_value = stat
    supervisor.summary_computed(sess, summaries, epoch_count)

  # Checkpoint best model so far if running validation.
  if FLAGS.log_progress and experiment_type == 'valid' and (
      best_validation_loss > run_stats['loss_%s' % experiment_type]):
    tf.logging.info('Previous best validation loss: %.4f.' %
                    (best_validation_loss))
    best_validation_loss = run_stats['loss_%s' % experiment_type]
    save_path = os.path.join(FLAGS.run_dir, hparams.log_subdir_str,
                             '%s-best_model.ckpt' % hparams.name)
    # Saving the best model thusfar checkpoint.
    best_model_saver.save(sess, save_path)

    tf.logging.info('Storing best model so far with loss %.4f at %s.' %
                    (best_validation_loss, save_path))
    # TODO(annahuang): Remove printouts
    print 'Storing best model so far with loss %.4f at %s.' % (
        best_validation_loss, save_path)

  tf.logging.info('%s, epoch %d: perplexity, loss (mask): %.4f, %.4f, ' %
                  (experiment_type, epoch_count,
                   run_stats['perplexity_mask_%s' % experiment_type],
                   run_stats['loss_mask_%s' % experiment_type]))
  tf.logging.info('perplexity, loss (unmask): %.4f, %.4f, ' %
                  (run_stats['perplexity_unmask_%s' % experiment_type],
                   run_stats['loss_unmask_%s' % experiment_type]))
  tf.logging.info('perplexity, loss (total): %.4f, %.4f, ' %
                  (run_stats['perplexity_total_%s' % experiment_type],
                   run_stats['loss_total_%s' % experiment_type]))
  tf.logging.info('log lr: %.4f' % np.log2(run_stats['learning_rate']))
  tf.logging.info('time taken: %.4f' % (time.time() - start_time))

  # TODO(annahuang): Remove printouts.
  print '%s, epoch %d: real loss %.4f perplexity, loss (mask): %.4f, %.4f, ' % (
      experiment_type, epoch_count, run_stats['loss_%s' % experiment_type],
      run_stats['perplexity_mask_%s' % experiment_type],
      run_stats['loss_mask_%s' % experiment_type]),
  print 'perplexity, loss (unmask): %.4f, %.4f, ' % (
      run_stats['perplexity_unmask_%s' % experiment_type],
      run_stats['loss_unmask_%s' % experiment_type]),
  print 'perplexity, loss (total): %.4f, %.4f, ' % (
      run_stats['perplexity_total_%s' % experiment_type],
      run_stats['loss_total_%s' % experiment_type]),
  print 'log lr: %.4f' % np.log2(run_stats['learning_rate']),
  print 'time taken: %.4f' % (time.time() - start_time)

  return best_validation_loss


def main(unused_argv):
  """Builds the graph and then runs training and validation."""
  if FLAGS.input_dir is None:
    tf.logging.fatal('No input directory was provided.')

  print FLAGS.maskout_method, 'seperate', FLAGS.separate_instruments
  print 'Augmentation', FLAGS.augment_by_transposing, FLAGS.augment_by_halfing_doubling_durations

  # Load hyperparameter settings, configs, and data.
  hparams = Hyperparameters(
      dataset=FLAGS.dataset,
      num_instruments=FLAGS.num_instruments,
      separate_instruments=FLAGS.separate_instruments,
      crop_piece_len=FLAGS.crop_piece_len,
      model_name=FLAGS.model_name,
      num_layers=FLAGS.num_layers,
      num_filters=FLAGS.num_filters,
      encode_silences=FLAGS.encode_silences,
      use_residual=FLAGS.use_residual,
      batch_size=FLAGS.batch_size,
      maskout_method=FLAGS.maskout_method,
      mask_indicates_context=FLAGS.mask_indicates_context,
      optimize_mask_only=FLAGS.optimize_mask_only,
      rescale_loss=FLAGS.rescale_loss,
      patience=FLAGS.patience,
      augment_by_transposing=FLAGS.augment_by_transposing,
      augment_by_halfing_doubling_durations=FLAGS.
      augment_by_halfing_doubling_durations,
      denoise_mode=FLAGS.denoise_mode,
      corrupt_ratio=FLAGS.corrupt_ratio)
  
  # Get data.
  # TODO(annahuang): Use queues.
  train_data, pianoroll_encoder = data_tools.get_data_and_update_hparams(
      FLAGS.input_dir, hparams, 'train')
  valid_data = data_tools.get_data_and_update_hparams(
      FLAGS.input_dir, hparams, 'valid', return_encoder=False)
  print '# of train_data:', len(train_data)
  print '# of valid_data:', len(valid_data)

  # TODO(annahuang): Set this according to pitch range.
  best_validation_loss = np.inf

  # Build the graph and subsequently running it for train and validation.
  with tf.Graph().as_default() as graph:
    no_op = tf.no_op()

    # Builds input and target placeholders, initializer, and training graph.
    graph_objects = build_placeholders_initializers_graph(
        is_training=True, hparams=hparams)
    input_data, targets, initializer, m = graph_objects

    # Build validation graph, reusing the model parameters from training graph.
    with tf.variable_scope('model', reuse=True, initializer=initializer):
      mvalid = BasicAutofillCNNGraph(
          is_training=False,
          hparams=hparams,
          input_data=input_data,
          targets=targets)

    # Instantiate a supervisor and use it to start a managed session.
    saver = 0  # Use default saver from supervisor.
    if not FLAGS.log_progress:
      saver = None
    best_model_saver = tf.train.Saver()

    # Save hparam configs.
    logdir = os.path.join(FLAGS.run_dir, hparams.log_subdir_str)
    if not os.path.exists(logdir):
      os.mkdir(logdir)
    config_fpath = os.path.join(logdir, 'config')
    print 'Writing to', config_fpath
    with open(config_fpath, 'w') as p:
      yaml.dump(hparams, p)

    # Graph will be finalized after instantiating supervisor.
    sv = tf.train.Supervisor(
        graph=graph,
        logdir=logdir,
        saver=saver,
        summary_op=None,
        save_model_secs=FLAGS.save_model_secs)
    #with sv.managed_session('local') as sess:
    #with sv.managed_session() as sess:
    with sv.PrepareSession() as sess:
      epoch_count = 0
      time_since_improvement = 0
      true_time_since_improvement = 0
      while epoch_count < FLAGS.num_epochs or not FLAGS.num_epochs:
        if sv.should_stop():
          break
        # Run training.
        run_epoch(sv, sess, m, train_data, pianoroll_encoder, hparams,
                  m.train_op, 'train', epoch_count)

        # Run validation.
        if epoch_count % hparams.eval_freq == 0:
          new_best_validation_loss = run_epoch(sv, sess, mvalid, valid_data, pianoroll_encoder, hparams,
                                               no_op, 'valid', epoch_count, best_validation_loss,
                                               best_model_saver)
          if new_best_validation_loss < best_validation_loss:
            best_validation_loss = new_best_validation_loss
            time_since_improvement = 0
            true_time_since_improvement = 0
          else:
            time_since_improvement += 1
            true_time_since_improvement += 1
            if time_since_improvement > FLAGS.patience:
              sess.run(m.decay_op)
              time_since_improvement = 0
            if true_time_since_improvement > 5 * FLAGS.patience:
              break
        epoch_count += 1

    print "best validation loss", best_validation_loss
    return best_validation_loss


if __name__ == '__main__':
  with pdb_post_mortem():
    tf.app.run()
