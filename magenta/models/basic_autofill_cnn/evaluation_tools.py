"""Evaluations for comparing against prior work."""
import os, sys, traceback
import numpy as np
import scipy.stats as stats
import tensorflow as tf

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, data_tools

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('fold', None, 'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string('kind', None, 'notewise or chordwise loss, or maxgreedy_notewise or mingreedy_notewise')
tf.app.flags.DEFINE_integer('num_crops', 5, 'number of random crops to consider')

def compute_maxgreedy_notewise_loss(wrapped_model, piano_rolls):
  return compute_greedy_notewise_loss(wrapped_model, piano_rolls, sign=+1)

def compute_mingreedy_notewise_loss(wrapped_model, piano_rolls):
  return compute_greedy_notewise_loss(wrapped_model, piano_rolls, sign=-1)

def compute_greedy_notewise_loss(wrapped_model, piano_rolls, sign):
  hparams = wrapped_model.hparams
  model = wrapped_model.model
  session = wrapped_model.sess

  losses = []
  def report():
    loss_mean = np.mean(losses)
    #loss_std = np.std(losses)
    loss_sem = stats.sem(losses)
    sys.stdout.write("%.5f+-%.5f" % (loss_mean, loss_sem))

  num_crops = 5
  for _ in range(num_crops):
    xs = np.array([data_tools.random_crop_pianoroll(x, hparams.crop_piece_len)
                   for x in piano_rolls], dtype=np.float32)

    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)

    for _ in range(T * I):
      input_data = [mask_tools.apply_mask_and_stack(x, m)
                    for x, m in zip(xs, mask)]
      p = session.run(model.predictions,
                      feed_dict={model.input_data: input_data})

      # determine vectors t, i by minimum entropy
      plogp = -np.where(p == 0, 0, p * np.log(p))
      # make entropy high at unmasked (given) places so those aren't selected
      plogp = np.where(mask, plogp, -sign * np.inf)
      entropies = plogp.sum(axis=2)
      # flatten time/instrument
      entropies = entropies.reshape([B, T * I])
      # find index of minimum entropy (vector of shape [batch_size])
      j = (sign * entropies).argmax(axis=1)

      t = j / I
      i = j % I

      loss = -np.where(xs[np.arange(B), t, :, i], np.log(p[np.arange(B), t, :, i]), 0).sum(axis=1)
      losses.append(loss)
      mask[np.arange(B), t, :, i] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      if len(losses) % 100 == 0:
        report()

      sys.stdout.write(".")
      sys.stdout.flush()
    assert np.allclose(mask, 0)
    report()
  sys.stdout.write("\n")
  return losses


def compute_chordwise_loss(wrapped_model, piano_rolls, separate_instruments=True, num_crops=5):
  hparams = wrapped_model.hparams
  model = wrapped_model.model
  session = wrapped_model.sess

  losses = []
  def report():
    loss_mean = np.mean(losses)
    #loss_std = np.std(losses)
    loss_sem = stats.sem(np.array(losses).flat)
    sys.stdout.write("%.5f+-%.5f" % (loss_mean, loss_sem))

  for _ in range(num_crops):
    xs = np.array([data_tools.random_crop_pianoroll(x, hparams.crop_piece_len)
                   for x in piano_rolls], dtype=np.float32)
    # working copy to fill in model's own predictions of chord notes
    xs_scratch = np.copy(xs)

    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)
    assert separate_instruments or (not separate_instruments and I == 1)

    # each example has its own ordering
    orders = np.ones([B, 1], dtype=np.int32) * np.arange(T, dtype=np.int32)[None, :]
    # random time orderings
    for i in range(B):
      np.random.shuffle(orders[i])
    if separate_instruments:
      # random instrument orderings within each time step
      D = I
    else:
      # random pitch orderings within each time step
      D = P
    orders = orders[:, :, None] * D + np.arange(D, dtype=np.int32)[None, None, :]
    for i in range(B):
      for t in range(T):
        np.random.shuffle(orders[i, t])
    orders = orders.reshape([B, T * D])

    for bahh, j in enumerate(orders.T):
      # NOTE: j is a vector with an index for each example in the batch
      t = j // D
      i = j % D

      # replace model predictions with ground truth when starting a fresh timestep
      if bahh % D == 0:
        xs_scratch = np.copy(xs)

      input_data = [mask_tools.apply_mask_and_stack(x, m)
                    for x, m in zip(xs_scratch, mask)]
      p = session.run(model.predictions,
                      feed_dict={model.input_data: input_data})
      if separate_instruments:
        loss = -np.where(xs_scratch[np.arange(B), t, :, i], np.log(p[np.arange(B), t, :, i]), 0).sum(axis=1)
      else:
        # Multiply by P so that we can frame wise loss mean later.
        loss = P * -np.where(xs_scratch[np.arange(B), t, i, 0], 
                         np.log(p[np.arange(B), t, i, 0]), 
                         np.log(1-p[np.arange(B), t, i, 0]))
      #if np.isinf(loss).any():
      #  import pdb; pdb.set_trace()
      losses.append(loss)

      # update xs_scratch to contain predictions
      if separate_instruments:
        xs_scratch[np.arange(B), t, :, i] = np.eye(P)[np.argmax(p[np.arange(B), t, :, i], axis=1)]
        mask[np.arange(B), t, :, i] = 0
      else:
        #FIXME: check this comparison
        xs_scratch[np.arange(B), t, i, 0] = np.where(p[np.arange(B), t, i, 0]>0.5, 1, 0)
        mask[np.arange(B), t, i, 0] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      #print
      #mask_tools.print_mask(mask[0])

      if len(losses) % 100 == 0:
        report()

      sys.stdout.write(".")
      sys.stdout.flush()
    assert np.allclose(mask, 0)
    report()
    print 'hparams.checkpoint_fpath', hparams.checkpoint_fpath
  sys.stdout.write("\n")
  return losses


def compute_notewise_loss(wrapped_model, piano_rolls, separate_instruments=True, num_crops=5):
  hparams = wrapped_model.hparams
  model = wrapped_model.model
  session = wrapped_model.sess

  losses = []
  def report():
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    sys.stdout.write("%.5f+-%.5f" % (loss_mean, loss_std))

  for _ in range(num_crops):
    xs = np.array([data_tools.random_crop_pianoroll(x, hparams.crop_piece_len)
                   for x in piano_rolls], dtype=np.float32)

    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)
    assert separate_instruments or (not separate_instruments and I == 1)

    # each example has its own ordering
    if separate_instruments:
      D = T * I
    else:
      D = T * P
    orders = np.ones([B, 1], dtype=np.int32) * np.arange(D, dtype=np.int32)
    # yuck
    for i in range(B):
      np.random.shuffle(orders[i])

    for j in orders.T:
      # NOTE: j is a vector with an index for each example in the batch
      if separate_instruments:
        t = j // I
        i = j % I
      else:
        t = j // P
        p = j % P
      input_data = [mask_tools.apply_mask_and_stack(x, m)
                    for x, m in zip(xs, mask)]
      preds = session.run(model.predictions,
                      feed_dict={model.input_data: input_data})
      if separate_instruments:
        loss = -np.where(xs[np.arange(B), t, :, i], np.log(preds[np.arange(B), t, :, i]), 0).sum(axis=1)
        #loss = -(np.log(p[np.arange(B), t, :, i]) * xs[np.arange(B), t, :, i]).sum(axis=1)
      else:
        loss = -np.where(xs[np.arange(B), t, p, 0], np.log(preds[np.arange(B), t, p, 0]), 0).sum()
        
      losses.append(loss)
      if separate_instruments:
        mask[np.arange(B), t, :, i] = 0
      else:
        mask[np.arange(B), t, p, 0] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      if len(losses) % 100 == 0:
        report()

      sys.stdout.write(".")
      sys.stdout.flush()
    assert np.allclose(mask, 0)
    report()
  sys.stdout.write("\n")
  return losses


def main(argv):
  try:
    print FLAGS.model_name, FLAGS.fold, FLAGS.kind, FLAGS.num_crops
    print FLAGS.checkpoint_dir
    fn = dict(notewise=compute_notewise_loss,
              chordwise=compute_chordwise_loss,
              mingreedy_notewise=compute_mingreedy_notewise_loss,
              maxgreedy_notewise=compute_maxgreedy_notewise_loss)[FLAGS.kind]
    # Retrieve model and hparams.
    wrapped_model = retrieve_model_tools.retrieve_model(model_name=FLAGS.model_name)
    hparams = wrapped_model.hparams
    print 'model_name', hparams.model_name
    print hparams.checkpoint_fpath
    # TODO: model_name in hparams is the conv spec class name, not retrieve model_name
    #assert wrapped_model.config.hparams.model_name == FLAGS.model_name
   
    # Get data to evaluate on. 
    piano_rolls = data_tools.get_data_as_pianorolls(FLAGS.input_dir, hparams, FLAGS.fold)
    print sorted(pianoroll.shape[0] for pianoroll in piano_rolls)
    
    # Evaluate!
    fn(wrapped_model, piano_rolls, hparams.separate_instruments, FLAGS.num_crops)
    print "%s done" % hparams.model_name
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, KeyboardInterrupt):
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


if __name__ == '__main__':
  tf.app.run()
