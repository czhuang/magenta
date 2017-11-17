"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys

import yaml

import numpy as np
import tensorflow as tf

from magenta.models.coconet import lib_data
from magenta.models.coconet import lib_util
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_hparams


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', None,
                           'Path to the base directory for different datasets.')
tf.app.flags.DEFINE_string('log_dir', None,
                           'Path to the directory where checkpoints and '
                           'summary events will be saved during training and '
                           'evaluation. Multiple runs can be stored within the '
                           'parent directory of `log_dir`. Point TensorBoard '
                           'to the parent directory of `log_dir` to see all '
                           'your runs.')
tf.app.flags.DEFINE_bool('log_progress', True,
                         'If False, do not log any checkpoints and summary'
                         'statistics.')

# Dataset.
tf.app.flags.DEFINE_string('dataset', None,
                           'Choices: Jsb16thSeparated, MuseData, Nottingham, '
                           'PianoMidiDe')
tf.app.flags.DEFINE_float('quantization_level', 0.125, 'Quantization duration.'
                          'For qpm=120, notated quarter note equals 0.5.')

tf.app.flags.DEFINE_integer('num_instruments', 4,
                            'Maximum number of instruments that appear in this '
                            'dataset.  Use 0 if not separating instruments and '
                            'hence does not matter how many there are.')
tf.app.flags.DEFINE_bool('separate_instruments', True,
                         'Separate instruments into different input feature'
                         'maps or not.')
tf.app.flags.DEFINE_integer('crop_piece_len', 64, 'The number of time steps '
                            'included in a crop')

# Model architecture.
tf.app.flags.DEFINE_string('architecture', 'straight',
                           'Convnet style. Choices: straight')
tf.app.flags.DEFINE_integer('num_layers', 64,
                            'The number of convolutional layers.')
tf.app.flags.DEFINE_integer('num_filters', 128,
                            'The number of filters for each convolutional '
                            'layer.')
# TODO: Some are meant to be booleans.
tf.app.flags.DEFINE_bool('use_residual', True,
                         'Add residual connections or not.')
tf.app.flags.DEFINE_integer('batch_size', 20,
                            'The batch size for training and validating the model.')

# Mask related.
tf.app.flags.DEFINE_string('maskout_method', 'orderless', 
                           "The choices include: 'bernoulli' "
                           "and 'orderless' (which "
                           "invokes gradient rescaling as per NADE).")
tf.app.flags.DEFINE_bool('mask_indicates_context', True, 
                         'Feed inverted mask into convnet so that zero-padding makes sense')
tf.app.flags.DEFINE_bool('optimize_mask_only', False, 'optimize masked predictions only')
tf.app.flags.DEFINE_bool('rescale_loss', True, 'Rescale loss based on context size.')
tf.app.flags.DEFINE_integer('patience', 5, 'Number of epochs to wait for improvement before decaying the learning rate.')

tf.app.flags.DEFINE_float('corrupt_ratio', 0.5, 'Fraction of variables to mask out.')
# Run parameters.
tf.app.flags.DEFINE_integer('num_epochs', 0,
                            'The number of epochs to train the model. Default '
                            'is 0, which means to run until terminated '
                            'manually.')
tf.app.flags.DEFINE_integer('save_model_secs', 360,
                            'The number of seconds between saving each '
                            'checkpoint.')
tf.app.flags.DEFINE_integer('eval_freq', 5,
                            'The number of training iterations before validation.')
tf.app.flags.DEFINE_string('run_id', '', 'A run_id to add to directory names to avoid accidentally overwriting when testing same setups.') 


def estimate_popstats(sv, sess, m, dataset, hparams):
  print('Estimating population statistics...')
  tfbatchstats, tfpopstats = list(zip(*m.popstats_by_batchstat.items()))

  nepochs = 3
  nppopstats = [lib_util.AggregateMean("") for _ in tfpopstats]
  for _ in range(nepochs):
    batches = (dataset
               .get_featuremaps()
               .batches(size=m.batch_size, shuffle=True))
    for step, batch in enumerate(batches):
      feed_dict = batch.get_feed_dict(m.placeholders)
      npbatchstats = sess.run(tfbatchstats, feed_dict=feed_dict)
      for nppopstat, npbatchstat in zip(nppopstats, npbatchstats):
        nppopstat.add(npbatchstat)
  nppopstats = [nppopstat.mean for nppopstat in nppopstats]

  _print_popstat_info(tfpopstats, nppopstats)

  # update tfpopstat variables
  for j, (tfpopstat, nppopstat) in enumerate(zip(tfpopstats, nppopstats)):
    tfpopstat.load(nppopstat)

def run_epoch(supervisor,
              sess,
              m,
              dataset,
              hparams,
              eval_op,
              experiment_type,
              epoch_count):
  """Runs an epoch of training or evaluate the model on given data."""
  # reduce variance in validation loss by fixing the seed
  data_seed = 123 if experiment_type == "valid" else None
  with lib_util.numpy_seed(data_seed):
    batches = (dataset
               .get_featuremaps()
               .batches(size=m.batch_size, shuffle=True, shuffle_rng=data_seed))

  losses = lib_util.AggregateMean('losses')
  losses_total = lib_util.AggregateMean('losses_total')
  losses_mask = lib_util.AggregateMean('losses_mask')
  losses_unmask = lib_util.AggregateMean('losses_unmask')

  start_time = time.time()
  for step, batch in enumerate(batches):
    # Evaluate the graph and run back propagation.
    fetches = [m.loss, m.loss_total, m.loss_mask, m.loss_unmask,
               m.reduced_mask_size, m.reduced_unmask_size,
               m.learning_rate, eval_op]
    feed_dict = batch.get_feed_dict(m.placeholders)
    (loss, loss_total, loss_mask, loss_unmask,
     reduced_mask_size, reduced_unmask_size,
     learning_rate, _) = sess.run(fetches, feed_dict=feed_dict)

    # Aggregate performances.
    losses_total.add(loss_total, 1)
    # Multiply the mean loss_mask by reduced_mask_size for aggregation as the
    # mask size could be different for every batch.
    losses_mask.add(loss_mask * reduced_mask_size, reduced_mask_size)
    losses_unmask.add(loss_unmask * reduced_unmask_size, reduced_unmask_size)

    if hparams.optimize_mask_only:
      losses.add(loss * reduced_mask_size, reduced_mask_size)
    else:
      losses.add(loss, 1)

  # Collect run statistics.
  run_stats = dict()
  run_stats['loss_mask'] = losses_mask.mean
  run_stats['loss_unmask'] = losses_unmask.mean
  run_stats['loss_total'] = losses_total.mean
  run_stats['loss'] = losses.mean
  run_stats['learning_rate'] = float(learning_rate)

  # Make summaries.
  if FLAGS.log_progress:
    summaries = tf.Summary()
    for stat_name, stat in run_stats.iteritems():
      value = summaries.value.add()
      value.tag = "%s_%s" % (stat_name, experiment_type)
      value.simple_value = stat
    supervisor.summary_computed(sess, summaries, epoch_count)

  tf.logging.info('%s, epoch %d: loss (mask): %.4f, loss (unmask): %.4f, '
                  'loss (total): %.4f, log lr: %.4f, time taken: %.4f',
                  experiment_type, epoch_count,
                  run_stats['loss_mask'],
                  run_stats['loss_unmask'],
                  run_stats['loss_total'],
                  np.log2(run_stats['learning_rate']),
                  time.time() - start_time)

  return run_stats["loss"]


def main(unused_argv):
  """Builds the graph and then runs training and validation."""
  print('TensorFlow version:', tf.__version__)

  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.data_dir is None:
    tf.logging.fatal('No input directory was provided.')

  print(FLAGS.maskout_method, 'seperate', FLAGS.separate_instruments)

  hparams = _hparams_from_flags()
  
  # Get data.
  print('dataset:', FLAGS.dataset, FLAGS.data_dir)
  print('current dir:', os.path.curdir)
  train_data = lib_data.get_dataset(FLAGS.data_dir, hparams, "train")
  valid_data = lib_data.get_dataset(FLAGS.data_dir, hparams, "valid")
  print('# of train_data:', train_data.num_examples)
  print('# of valid_data:', valid_data.num_examples)
  if train_data.num_examples < hparams.batch_size:
    print("reducing batch_size to %i" % train_data.num_examples)
    hparams.batch_size = train_data.num_examples

  train_data.update_hparams(hparams)

  # Save hparam configs.
  logdir = os.path.join(FLAGS.log_dir, hparams.log_subdir_str)
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  config_fpath = os.path.join(logdir, 'config')
  print('Writing to', config_fpath)
  with open(config_fpath, 'w') as p:
    yaml.dump(hparams, p)

  # Build the graph and subsequently running it for train and validation.
  with tf.Graph().as_default():
    no_op = tf.no_op()

    # Build placeholders and training graph, and validation graph with reuse.
    m = lib_graph.build_graph(is_training=True, hparams=hparams)
    tf.get_variable_scope().reuse_variables()
    mvalid = lib_graph.build_graph(is_training=False, hparams=hparams)

    tracker = Tracker(label="validation loss",
                      patience=FLAGS.patience,
                      decay_op=m.decay_op,
                      save_path=os.path.join(FLAGS.log_dir,
                                             hparams.log_subdir_str,
                                             'best_model.ckpt'))

    # Graph will be finalized after instantiating supervisor.
    sv = tf.train.Supervisor(
        logdir=logdir,
        saver=tf.train.Supervisor.USE_DEFAULT if FLAGS.log_progress else None,
        summary_op=None,
        save_model_secs=FLAGS.save_model_secs)
    with sv.PrepareSession() as sess:
      epoch_count = 0
      while epoch_count < FLAGS.num_epochs or not FLAGS.num_epochs:
        if sv.should_stop():
          break

        # Run training.
        run_epoch(sv, sess, m, train_data, hparams,
                  m.train_op, 'train', epoch_count)

        # Run validation.
        if epoch_count % hparams.eval_freq == 0:
          estimate_popstats(sv, sess, m, train_data, hparams)
          loss = run_epoch(
              sv, sess, mvalid, valid_data, hparams,
              no_op, 'valid', epoch_count)
          tracker(loss, sess)
          if tracker.should_stop():
            break

        epoch_count += 1

    print("best", tracker.label, tracker.best)
    print("Done.")
    return tracker.best


class Tracker(object):
  def __init__(self, label, save_path, sign=-1, patience=5, decay_op=None):
    self.label = label
    self.sign = sign
    self.best = np.inf
    self.saver = tf.train.Saver()
    self.save_path = save_path
    self.patience = patience
    # NOTE: age is reset with decay, but true_age is not
    self.age = 0
    self.true_age = 0
    self.decay_op = decay_op

  def __call__(self, loss, sess):
    if self.sign * loss > self.sign * self.best:
      if FLAGS.log_progress:
        tf.logging.info('Previous best %s: %.4f.', self.label, self.best)
        tf.gfile.MakeDirs(os.path.dirname(self.save_path))
        self.saver.save(sess, self.save_path)
        tf.logging.info('Storing best model so far with loss %.4f at %s.' %
                        (loss, self.save_path))
      self.best = loss
      self.age = 0
      self.true_age = 0
    else:
      self.age += 1
      self.true_age += 1
      if self.age > self.patience:
        sess.run([self.decay_op])
        self.age = 0

  def should_stop(self):
    return self.true_age > 5 * self.patience


def _print_popstat_info(tfpopstats, nppopstats):
  mean_errors = []
  stdev_errors = []
  for j, (tfpopstat, nppopstat) in enumerate(zip(tfpopstats, nppopstats)):
    moving_average = tfpopstat.eval()
    if j % 2 == 0:
      mean_errors.append(abs(moving_average - nppopstat))
    else:
      stdev_errors.append(abs(np.sqrt(moving_average) - np.sqrt(nppopstat)))
  def flatmean(xs):
    return np.mean(np.concatenate([x.flatten() for x in xs]))
  print("average of pop mean/stdev errors: %g %g"
         % (flatmean(mean_errors), flatmean(stdev_errors)))
  print("average of batch mean/stdev: %g %g"
         % (flatmean(nppopstats[0::2]),
            flatmean([np.sqrt(ugh) for ugh in nppopstats[1::2]])))

def _hparams_from_flags():
  keys = ("""
      dataset quantization_level num_instruments separate_instruments
      crop_piece_len architecture num_layers num_filters use_residual
      batch_size maskout_method mask_indicates_context optimize_mask_only
      rescale_loss patience corrupt_ratio eval_freq run_id
      """.split())
  hparams = lib_hparams.Hyperparameters(**dict((key, getattr(FLAGS, key))
                                               for key in keys))
  return hparams

if __name__ == '__main__':
  tf.app.run()
