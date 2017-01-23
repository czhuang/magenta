"""Evaluations for comparing against prior work."""
import os, sys, traceback
import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, data_tools
from magenta.models.basic_autofill_cnn.data_tools import DataProcessingError


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('fold', None, 'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string('kind', None, 'notewise or chordwise loss, or maxgreedy_notewise or mingreedy_notewise')
tf.app.flags.DEFINE_integer('num_crops', 5, 'number of random crops to consider')
tf.app.flags.DEFINE_integer('evaluation_batch_size', 1000, 'Batch size for evaluation.')
# already defined in basic_autofill_cnn_train.py which comes in through retrieve_model_tools (!)
#tf.app.flags.DEFINE_integer('crop_piece_len', None, 'length of random crops (short pieces are padded in case of chordwise)')

class InfiniteLoss(Exception):
  pass

def sem(xs):
  return np.std(xs) / np.sqrt(np.asarray(xs).size)

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
    loss_sem = sem(losses)
    sys.stdout.write("%.5f+-%.5f" % (loss_mean, loss_sem))

  num_crops = 5
  crop_piece_len = FLAGS.crop_piece_len if FLAGS.crop_piece_len is not None else hparams.crop_piece_len
  for _ in range(num_crops):
    xs = np.array([data_tools.random_crop_pianoroll(x, crop_piece_len)
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


def compute_chordwise_loss(wrapped_model, piano_rolls, crop_piece_len, 
                           num_crops=5, **kwargs):
  
  hparams = wrapped_model.hparams
  model = wrapped_model.model
  session = wrapped_model.sess

  separate_instruments = hparams.separate_instruments
  print 'separate_instruments', separate_instruments

  losses = []
  def report(final=False):
    loss_mean = np.mean(losses)
    #loss_std = np.std(losses)
    loss_sem = sem(losses)
    print "%s%.5f+-%.5f" % ("FINAL " if final else "", loss_mean, loss_sem)

  for _ in range(num_crops):
    xs, lengths = list(zip(*[data_tools.random_crop_pianoroll_pad(x, crop_piece_len)
                             for x in piano_rolls]))
    xs = np.array(xs, dtype=np.float32)
    lengths = np.array(lengths, dtype=int)
    # working copy to fill in model's own predictions of chord notes
    xs_scratch = np.copy(xs)

    B, T, P, I = xs.shape
    print xs.shape
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

      # in both cases, loss is a vector over batch examples
      if separate_instruments:
        # batched loss at time/instrument pair, summed over pitches
        loss = I * -np.where(xs_scratch[np.arange(B), t, :, i], np.log(p[np.arange(B), t, :, i]), 0).sum(axis=1)
      else:
        # batched loss at time/pitch pair, single instrument
        # multiply by P to counteract the division by P when we take the mean loss;
        # we want to sum over pitches, not average.
        loss = P * -np.where(xs_scratch[np.arange(B), t, i, 0], 
                             np.log(p[np.arange(B), t, i, 0]), 
                             np.log(1-p[np.arange(B), t, i, 0]))

      # don't judge predictions of padded elements
      loss = np.where(t < lengths, loss, 0)

      losses.append(loss)

      print "%i: %.5f < %.5f < %.5f < %.5f < %.5g" % (bahh, np.min(loss), np.percentile(loss, 25), np.percentile(loss, 50), np.percentile(loss, 75), np.max(loss))

      if np.isinf(loss).any():
        report(final=True)
        raise InfiniteLoss()

      # update xs_scratch to contain predictions
      if separate_instruments:
        xs_scratch[np.arange(B), t, :, i] = np.eye(P)[np.argmax(p[np.arange(B), t, :, i], axis=1)]
        mask[np.arange(B), t, :, i] = 0
      else:
        #FIXME: check this comparison
        xs_scratch[np.arange(B), t, i, 0] = p[np.arange(B), t, i, 0] > 0.5
        mask[np.arange(B), t, i, 0] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      #print
      #mask_tools.print_mask(mask[0])

      if len(losses) % 100 == 0:
        report()
    assert np.allclose(mask, 0)
    report()
    print 'hparams.checkpoint_fpath', hparams.checkpoint_fpath
  report(final=True)
  return losses


def compute_notewise_loss(wrapped_model, piano_rolls, crop_piece_len, 
                          num_crops=5, imagewise=False, eval_batch_size=1000, **kwargs):
  hparams = wrapped_model.hparams
  model = wrapped_model.model
  session = wrapped_model.sess
  separate_instruments = hparams.separate_instruments

  losses = []
  def report(final=False):
    loss_mean = np.mean(losses)
    loss_sem = sem(losses)
    #loss_std = np.std(losses)
    if imagewise:
      pixel_count = crop_piece_len * hparams.input_shape[1]
      #FIXME: other image datasets will have diff dimensions
      assert 28*28 == pixel_count
      loss_mean *= pixel_count 
      loss_sem *= pixel_count
    print "%s%.5f+-%.5f" % ("FINAL " if final else "", loss_mean, loss_sem)
  
  for _ in range(num_crops):
    xs_all = np.array([data_tools.random_crop_pianoroll(x, crop_piece_len)
                   for x in piano_rolls], dtype=np.float32)
    B_all, T, P, I = xs_all.shape
    assert B_all % eval_batch_size == 0
    num_batches = B_all / eval_batch_size
    for bi in range(num_batches):
      start_idx = bi * eval_batch_size
      end_idx = (bi + 1) * eval_batch_size
      xs = xs_all[start_idx:end_idx]
      B = xs.shape[0]
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
          loss = -np.where(xs[np.arange(B), t, i, 0], 
                           np.log(preds[np.arange(B), t, i, 0]), 
                           np.log(1-preds[np.arange(B), t, i, 0]))
        losses.append(loss)
        if separate_instruments:
          mask[np.arange(B), t, :, i] = 0
        else:
          mask[np.arange(B), t, p, 0] = 0
        assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

        if len(losses) % 1 == 0:
          report()

        sys.stdout.write(".")
        sys.stdout.flush()
    assert np.allclose(mask, 0)
    report()
  sys.stdout.write("\n")
  return losses


def main(argv):
  try:
    print FLAGS.model_name, FLAGS.fold, FLAGS.kind, 
    print FLAGS.num_crops, FLAGS.crop_piece_len, FLAGS.evaluation_batch_size
    print FLAGS.checkpoint_dir
    fn = dict(notewise=compute_notewise_loss,
              chordwise=compute_chordwise_loss,
              imagewise=compute_notewise_loss,
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
    B, T, P, I = piano_rolls.shape
    print pianorolls.shape
    print np.unique(sorted(pianoroll.shape[0] for pianoroll in piano_rolls))

    print FLAGS.crop_piece_len, hparams.crop_piece_len
    crop_piece_len = FLAGS.crop_piece_len if FLAGS.crop_piece_len is not None else hparams.crop_piece_len
    if crop_piece_len != hparams.crop_piece_len:
      print 'WARNING: crop_piece_len %r,  hparams.crop_piece_len %r, mismatch' % (crop_piece_len, hparams.crop_piece_len)
    print 'crop_piece_len', crop_piece_len

    eval_batch_size = FLAGS.evaluation_batch_size if B > FLAGS.evaluation_batch_size else B
    if eval_batch_size != hparams.batch_size:
      print 'Using batch size %r for evaluation instead of %r' % (eval_batch_size, hparams.batch_size)
    print 'eval_batch_size', eval_batch_size    
    # Evaluate!
    try:
      #TODO: notewise does not take crop_piece_len yet
      fn(wrapped_model, piano_rolls, crop_piece_len, FLAGS.num_crops,
         FLAGS.kind == 'imagewise', eval_batch_size=eval_batch_size)
    except InfiniteLoss:
      print "infinite loss"
    print "%s done" % hparams.model_name
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, KeyboardInterrupt):
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


if __name__ == '__main__':
  tf.app.run()
