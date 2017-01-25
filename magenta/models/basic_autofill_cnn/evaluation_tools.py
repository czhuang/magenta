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

def store(losses, position, path):
  print 'Storing image losses to', path
  np.savez_compressed(path, losses=losses, current_position=position)

def report(losses, final=False):
  loss_mean = np.mean(losses)
  loss_sem = sem(losses)
  print "%s%.5f+-%.5f " % ("FINAL " if final else "", loss_mean, loss_sem)
  
def batches(xs, k):
  assert len(xs) % k == 0
  for a in range(0, len(xs), k):
    yield xs[a:a+k]

def pad(xs):
  shape = xs[0].shape[1:]
  # all should have equal shape in all but the first dimension
  assert all(x.shape[1:] == shape for x in xs)
  lengths = np.array(list(map(len, xs)), dtype=int)
  ys = np.zeros([len(xs), max(lengths)] + list(shape), dtype=np.float32)
  for i, x in enumerate(xs):
    ys[i, :lengths[i]] = x
  return ys, lengths

def compute_maxgreedy_notewise_loss(wrapped_model, pianorolls):
  return compute_greedy_notewise_loss(wrapped_model, pianorolls, sign=+1)

def compute_mingreedy_notewise_loss(wrapped_model, pianorolls):
  return compute_greedy_notewise_loss(wrapped_model, pianorolls, sign=-1)

def compute_greedy_notewise_loss(wrapped_model, pianorolls, sign):
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
                   for x in pianorolls], dtype=np.float32)

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


def compute_chordwise_loss(predictor, pianorolls, crop_piece_len, 
                           num_crops=5, batch_size=None, eval_data=None, eval_fpath=None, chronological=False, chronological_margin=0, separate_instruments=True, **kwargs):
  
  print 'separate_instruments', separate_instruments

  if batch_size is None:
    batch_size = len(pianorolls)

  assert eval_fpath is not None
  if eval_data is not None:
    losses = list(eval_data["losses"])
    crop_sofar, batch_sofar = eval_data["current_position"]
  else:
    losses = [] 
    crop_sofar, batch_sofar = 0, 0

  losses = []

  if chronological:
    # no point in doing multiple passes
    assert num_crops == 1

  for ci in range(num_crops)[crop_sofar:]:
    print 'crop idx', ci

    for bi, xs in list(enumerate(batches(pianorolls, batch_size)))[batch_sofar:]:
      xs, lengths = pad(xs)

      B, T, P, I = xs.shape
      mask = np.ones([B, T, P, I], dtype=np.float32)
      assert separate_instruments or I == 1
  
      # each example has its own ordering
      orders = np.ones([B, 1], dtype=np.int32) * np.arange(T, dtype=np.int32)[None, :]
      if not chronological:
        # random time orderings
        for i in range(B):
          np.random.shuffle(orders[i])
      # random variable orderings within each time step
      D = I if separate_instruments else P
      orders = orders[:, :, None] * D + np.arange(D, dtype=np.int32)[None, None, :]
      for i in range(B):
        for t in range(T):
          np.random.shuffle(orders[i, t])
      orders = orders.reshape([B, T * D])
  
      # working copy to fill in model's own predictions of chord notes
      xs_scratch = np.copy(xs)
  
      for bahh, j in enumerate(orders.T):
        # NOTE: j is a vector with an index for each example in the batch
        t = j // D
        i = j % D
  
        # replace model predictions with ground truth when starting a fresh timestep
        if bahh % D == 0:
          xs_scratch = np.copy(xs)

        # crop_piece_len is taken the indicate the maximum input length the model can deal with (in
        # terms of gpu memory).  if we cannot feed in the entire input, feed in a temporal crop
        # centered on the current timestep.

        # FIXME: put assertions
        if chronological:
          t0 = t - crop_piece_len - chronological_margin
        else:
          t0 = t - crop_piece_len / 2.,
        # restrict to valid indices
        t0 = np.astype(np.round(np.clip(t0, 0, T - crop_piece_len)), np.int32)

        cropped_xs_scratch = xs_scratch[:, t0:t0 + crop_piece_len]
        cropped_mask = mask[:, t0:t0 + crop_piece_len]
        # ensure resulting crop is the correct size. this can fail if all pieces in the batch are
        # shorter than crop_piece_len, so allow length of longest piece (= T) as well.
        assert cropped_xs_scratch[0].shape[0] in [crop_piece_len, T]
        p = predictor(cropped_xs_scratch, cropped_mask)
  
        # in both cases, loss is a vector over batch examples
        if separate_instruments:
          # batched loss at time/instrument pair, summed over pitches
          loss = -np.where(xs_scratch[np.arange(B), t, :, i],
                           np.log(p[np.arange(B), t - t0, :, i]),
                           0).sum(axis=1)
        else:
          # batched loss at time/pitch pair, single instrument
          loss = -np.where(xs_scratch[np.arange(B), t, i, 0], 
                           np.log(p[np.arange(B), t - t0, i, 0]), 
                           np.log(1-p[np.arange(B), t - t0, i, 0]))
  
        # at the end we take the mean of the losses. multiply by D because we want to sum over the D
        # axis (instruments or pitches), not average.
        loss *= D
  
        # don't judge predictions of padded elements
        loss = np.where(t < lengths, loss, 0)
  
        losses.append(loss)
  
        print "%i: %.5f < %.5f < %.5f < %.5f < %.5g" % (bahh, np.min(loss), np.percentile(loss, 25), np.percentile(loss, 50), np.percentile(loss, 75), np.max(loss))
  
        if np.isinf(loss).any():
          report(losses, final=True)
          raise InfiniteLoss()
  
        # update xs_scratch to contain predictions
        if separate_instruments:
          xs_scratch[np.arange(B), t, :, i] = np.eye(P)[np.argmax(p[np.arange(B), t, :, i], axis=1)]
          mask[np.arange(B), t, :, i] = 0
        else:
          xs_scratch[np.arange(B), t, i, 0] = p[np.arange(B), t, i, 0] > 0.5
          mask[np.arange(B), t, i, 0] = 0
        assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1
  
        #print
        #mask_tools.print_mask(mask[0])
  
        if len(losses) % 100 == 0:
          report(losses)
      assert np.allclose(mask, 0)
      report(losses)
      store(losses=losses, position=[ci, bi+1], path=eval_fpath)
    # After running possbily less # of batches b/c continuing from last logged point,
    # reset to 0.
    batch_sofar = 0
  report(final=True)
  return losses


def compute_notewise_loss(predictor, pianorolls, crop_piece_len, num_crops=5, eval_data=None, 
                          imagewise=False, separate_instruments=True,
                          eval_batch_size=1000, eval_fpath=None, **kwargs):
  assert eval_fpath is not None
  if eval_data is not None:
    losses = list(eval_data["losses"])
    crop_sofar, batch_sofar = eval_data["current_position"]
  else:
    losses = [] 
    crop_sofar, batch_sofar = 0, 0
  
  for ci in range(num_crops)[crop_sofar:]:
    print 'crop idx', ci
    xs_all = np.array([data_tools.random_crop_pianoroll(x, crop_piece_len)
                   for x in pianorolls], dtype=np.float32)
    for bi, xs in list(enumerate(batches(pianorolls, eval_batch_size)))[batch_sofar:]:
      print 'batch idx, started at this batch', bi, batch_sofar

      B, T, P, I = xs.shape
      mask = np.ones([B, T, P, I], dtype=np.float32)
      assert separate_instruments or I == 1

      # number of variables
      D = T * (I if separate_instruments else P)
      print 'D', D
      # each example has its own ordering
      orders = np.ones([B, 1], dtype=np.int32) * np.arange(D, dtype=np.int32)
      for i in range(B):
        np.random.shuffle(orders[i]) # yuck

      for j in orders.T:
        # NOTE: j is a vector with an index for each example in the batch
        if separate_instruments:
          t = j // I
          i = j % I
        else:
          t = j // P
          i = j % P
        preds = predictor(xs, mask)
        if separate_instruments:
          loss = -np.where(xs[np.arange(B), t, :, i], np.log(preds[np.arange(B), t, :, i]), 0).sum(axis=1)
          mask[np.arange(B), t, :, i] = 0
        else:
          loss = -np.where(xs[np.arange(B), t, i, 0], 
                           np.log(preds[np.arange(B), t, i, 0]), 
                           np.log(1-preds[np.arange(B), t, i, 0]))
          mask[np.arange(B), t, i, 0] = 0

        if imagewise:
          pixel_count = crop_piece_len * hparams.input_shape[1]
          #FIXME: other image datasets will have diff dimensions
          assert 28*28 == pixel_count
          loss *= pixel_count

        losses.append(loss)
        assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

        if len(losses) % 1 == 0:
          report(losses, final=False)

      assert np.allclose(mask, 0)
      report(losses, final=False)
      store(losses=losses, position=[ci, bi+1], path=eval_fpath)
    # After running possbily less # of batches b/c continuing from last logged point,
    # reset to 0.
    batch_sofar = 0
  report(losses, final=True)
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
    pianorolls = data_tools.get_data_as_pianorolls(FLAGS.input_dir, hparams, FLAGS.fold)
    B, T, P, I = pianorolls.shape
    pianorolls = pianorolls[:2]
    print pianorolls.shape
    print np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls))

    # Get folder for previous runs for this config.
    dir_name = '%s-%s-roll_shape=%r-num_crops=%r-crop_len=%r-eval_bs=%r' % (
        FLAGS.fold, FLAGS.kind, pianorolls.shape, FLAGS.num_crops, 
        FLAGS.crop_piece_len, FLAGS.evaluation_batch_size)

    # Check to see if there's previous evaluation losses.
    eval_path = os.path.join(FLAGS.checkpoint_dir, dir_name)
    eval_fpath = os.path.join(eval_path, 'evaluations.npz')
    print eval_fpath
    if not os.path.exists(eval_path):
      os.mkdir(eval_path)
      eval_data = None
      print '\nMade directory.  No previous log.\n'
    else: 
      if os.path.exists(eval_fpath):
        print '\nPrevious log exists.'
        print 'Loading'
        eval_data = np.load(eval_fpath)
      else:
        print '\nNo previous log.\n'
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
      def predictor(xs, masks):
        input_data = [mask_tools.apply_mask_and_stack(x, mask) for x, mask in zip(xs, masks)]
        p = session.run(model.predictions, feed_dict={model.input_data: input_data})
        return p

      fn(predictor, pianorolls, crop_piece_len, FLAGS.num_crops, eval_data,
         imagewise=FLAGS.kind == 'imagewise',
         separate_instruments=hparams.separate_instruments,
         eval_batch_size=eval_batch_size, eval_fpath=eval_fpath)
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
