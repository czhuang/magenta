"""Evaluations for comparing against prior work."""
from collections import defaultdict
import os, sys, traceback
import time
import numpy as np
import tensorflow as tf
import functools as ft

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, data_tools
from magenta.models.basic_autofill_cnn.data_tools import DataProcessingError


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('fold', None, 'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string('kind', None, 'notewise or chordwise loss, or maxgreedy_notewise or mingreedy_notewise')
tf.app.flags.DEFINE_integer('num_crops', 5, 'number of random crops to consider')
tf.app.flags.DEFINE_integer('evaluation_batch_size', 20, 'Batch size for evaluation.')
# already defined in basic_autofill_cnn_train.py which comes in through retrieve_model_tools (!)
tf.app.flags.DEFINE_integer('convnet_len', None, 'Length of convnet receptive field at input.')
tf.app.flags.DEFINE_bool('chronological', False, 'indicates chordwise evaluation should proceed in chronological order')
tf.app.flags.DEFINE_integer('chronological_margin', 0, 'right-hand margin for chronological evaluation to avoid convnet edge effects')
# TODO: haven't implemented "pitch" chronological for images.
tf.app.flags.DEFINE_bool('pitch_chronological', False, 'indicates chordwise evaluation should proceed in chronological order')
tf.app.flags.DEFINE_bool('log_eval_progress', False, 'Store the intermediate intra-frame pianorolls predictions and others.')
tf.app.flags.DEFINE_string('pad_mode', None, 'Mode for padding shorter sequences. Options are "none", "zeros" or "wrap".')
tf.app.flags.DEFINE_integer('eval_len', None, '(Crop) length of piece to evaluate.  0 means evaluate on whole piece.')
tf.app.flags.DEFINE_bool('eval_test_mode', False, 'If in test mode for evaluation.')


class InfiniteLoss(Exception):
  pass

# adapts batch size in response to ResourceExhaustedErrors
class RobustPredictor(object):
  def __init__(self, predictor):
    self.predictor = predictor
    self.Bmax = np.inf
    self.factor = 1.5

  def __call__(self, pianoroll, mask):
    if len(pianoroll) > self.Bmax:
      return self.bisect(pianoroll, mask)
    try:
      return self.predictor(pianoroll, mask)
    except tf.errors.ResourceExhaustedError:
      self.Bmax = int(len(pianoroll) / self.factor)
      print "ResourceExhaustedError on batch of %s, lowering max batch size to %s" % (len(pianoroll), self.Bmax)
      return self.bisect(pianoroll, mask)

  def bisect(self, pianoroll, mask):
    i = int(len(pianoroll) / self.factor)
    return np.concatenate([self(pianoroll[:i], mask[:i]),
                           self(pianoroll[i:], mask[i:])],
                          axis=0)

def sem(xs):
  return np.std(xs) / np.sqrt(np.asarray(xs).size)

def store(losses, position, path):
  print 'Storing losses to', path
  np.savez_compressed(path, losses=losses, current_position=position)

def report(losses, final=False, tag=''):
  #loss_mean = np.mean(losses)
  losses = np.concatenate(losses, axis=0)
  loss_mean = np.mean(losses)
  loss_sem = sem(losses)
  print "%.5f < %.5f < %.5f < %.5f < %.5g" % (np.min(losses), np.percentile(losses, 25), np.percentile(losses, 50), np.percentile(losses, 75), np.max(losses))
  print "\n\t\t\t\t\t\t\t\t\t\t\t\t%s%.5f+-%.5f \n" % (tag+" FINAL " if final else "", loss_mean, loss_sem)
  return loss_mean, loss_sem, np.asarray(losses).size
 
 
def batches(xs, k):
  #assert len(xs) % k == 0
  if len(xs) % k != 0:
    print 'WARNING: # of data points (%d) is not divisible by %d' % (len(xs), k)
  for a in range(0, len(xs), k):
    yield xs[a:a+k]


def pad(xs, chronological, requested_piece_len, pad_mode):
  if pad_mode == 'zeros':
    return pad_with_zeros(xs, chronological, requested_piece_len)
  elif pad_mode == 'wrap':
    return pad_with_wrap(xs, requested_piece_len)
  elif pad_mode == 'none':
    return np.asarray(xs), np.array([len(x) for x in xs])
  else: 
    assert False, 'Pad mode %s not yet supported.' % (pad_mode)
  

def pad_with_wrap(xs, requested_piece_len):
  lengths = np.array([len(x) for x in xs])
  pad_lengths = requested_piece_len - lengths
  ys = [np.pad(roll, [(0, pad_lengths[i])] + [(0, 0)] * (xs[0].ndim - 1), mode="wrap") for i, roll in enumerate(xs)]
  return np.asarray(ys), lengths


def pad_with_zeros(xs, chronological, requested_piece_len):
  lengths = np.array(list(map(len, xs)), dtype=int)
  padded_xs = []
  lengths = []
  for x in xs:
    x, len_ = data_tools.random_crop_pianoroll_pad(
      x, requested_piece_len)
    padded_xs.append(x)
    lengths.append(len_)
  return np.array(padded_xs), np.array(lengths)


def breakup_long_pieces(xs):
  num_pieces = len(xs)
  lens = [len(x) for x in xs]
  if len(np.unique(lens)) == 1:
    return xs, lens[0]
  max_len = max(lens)
  if max_len < 200:
    return xs, max_len

  sorted_lens = np.sort(lens)
  max_allowed_len = sorted_lens[-2]
  wrap_inds = [i for i, l in enumerate(lens) if l > max_allowed_len]

  # FIXME: just for debugging.
  lens_too_long = [lens[i] for i in wrap_inds]
  print 'max_allowed_len, and longest len', max_allowed_len, sorted_lens[-1]
  print 'lens_too_long', lens_too_long

  added_wraps = 0
  for i in wrap_inds:
    len_ = lens[i]
    num_wraps = int(np.ceil(len_ / float(max_allowed_len)))
    # if the last wrapped piece is shorter than the shortest piece, then assert.
    last_wrap_len = len_ % max_allowed_len
    if last_wrap_len < sorted_lens[0] * 2/3.:
      assert False, 'Wrapped piece len %d shorter than shortest piece len %d' % (last_wrap_len, sorted_lens[0])
    temp_x = xs[i]
    xs[i] = temp_x[:max_allowed_len]
    for w in range(1, num_wraps): 
      xs.append(temp_x[w*max_allowed_len:(w+1)*max_allowed_len])
      added_wraps += 1
  assert len(xs) == num_pieces + added_wraps
  return xs, max_allowed_len


def chordwise_ordering(B, T, D, chronological=False, pitch_chronological=False):
  # each example has its own ordering
  orders = np.ones([B, 1], dtype=np.int32) * np.arange(T, dtype=np.int32)[None, :]
  if not chronological:
    # random time orderings
    for i in range(B):
      np.random.shuffle(orders[i])
  # random variable orderings within each time step
  orders = orders[:, :, None] * D + np.arange(D, dtype=np.int32)[None, None, :]
  # TODO: Not sure if going from bottom pitch to top is the best yet.
  if not pitch_chronological:
    for i in range(B):
      for t in range(T):
        np.random.shuffle(orders[i, t])
  orders = orders.reshape([B, T * D])
  ts, ds = np.unravel_index(orders.T, dims=(T, D))
  return ts, ds

def notewise_ordering(B, T, D):
  # each example has its own ordering
  orders = np.ones([B, 1], dtype=np.int32) * np.arange(T * D, dtype=np.int32)
  for i in range(B):
    np.random.shuffle(orders[i]) # yuck
  ts, ds = np.unravel_index(orders.T, dims=(T, D))
  return ts, ds


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


def evaluation_loop(evaluator, pianorolls, num_crops=5, batch_size=None, eval_data=None, eval_fpath=None, chronological=None, log_eval_progress=None, **kwargs):
  assert batch_size is not None
  assert chronological is not None
  assert eval_fpath is not None 
  assert log_eval_progress is not None

  if eval_data is not None:
    losses = list(eval_data["losses"])
    crop_sofar, batch_sofar, t_sofar = eval_data["current_position"]
  else:
    losses = [] 
    crop_sofar, batch_sofar, t_sofar = 0, 0, 0
  print 'position from last time', crop_sofar, batch_sofar, t_sofar

  intermediates = defaultdict(list)

  for ci in range(num_crops)[crop_sofar:]:
    print 'crop idx', ci
    frame_losses = []
    for bi, xs in list(enumerate(batches(pianorolls, batch_size)))[batch_sofar:]:
      print 'batch idx, started at this batch', bi, batch_sofar
      start_time = time.time()
      frame_loss = []
      for i, (loss, inds, states) in enumerate(evaluator(xs, t_sofar)):
        if np.isinf(loss).any():
          # report losses before inf just for information
          report(loss)
          raise InfiniteLoss()
        losses.append(loss)
        frame_loss.append(loss)
        ts, ds, end_of_frame = inds
        t_print = ts[0] if len(np.unique(ts)) == 1 else -1
        d_print = ds[0] if len(np.unique(ds)) == 1 else -1
        if chronological:
          print "%i, %i, %i(%d): " % (ci, bi, t_print, d_print),
        else:
          print "%i, %i, %i(not chrono): " % (ci, bi, i),

        print "%.5f < %.5f < %.5f < %.5f < %.5g, \tstep mean: %.5f" % (
            np.min(loss), np.percentile(loss, 25), np.percentile(loss, 50), 
            np.percentile(loss, 75), np.max(loss), np.mean(loss))
        if end_of_frame:
          print 'time per frame: %.2f' % (time.time() - start_time)
          print 'Frame loss:'
          frame_mean, frame_sem, n = report(frame_loss)
          print 'Total loss:'
          report(losses)
          
          frame_losses.append([t_print, frame_mean, frame_sem])
          frame_loss = []
	  start_time = time.time()
          if chronological and t_print != 0 and t_print % 100 == 0:
            store(losses=losses, position=[ci, bi, t_print+1], path=eval_fpath)
        # Store states to intermediates
        if states is not None:
          for key, vals in states.iteritems():
            intermediates[key].extend(vals)         
      report(losses)
      store(losses=losses, position=[ci, bi+1, 0], path=eval_fpath)
    # After running possbily less # of batches b/c continuing from last logged point, reset to 0.
    batch_sofar = 0

  # Report how loss progresses over timestep
  print 'chronological', chronological
  for t, mean, sem in frame_losses:
    print '%d: %.5f+-%.5f' % (t, mean, sem)

  eval_stats = report(losses, final=True, tag=eval_fpath)

  # Store intermediates
  fpath = os.path.join(os.path.dirname(eval_fpath), 'intermediates' + ".npz")
  print "Writing to", fpath  
  np.savez_compressed(fpath, **intermediates)

  return eval_stats


def compute_chordwise_loss_batch(predictor, pianorolls, separate_instruments=True, log_eval_progress=False, **kwargs):
  predictor = RobustPredictor(predictor)

  def evaluator(pianorolls, t_sofar):
    states = defaultdict(list)

    for pianoroll in pianorolls:
      T, P, I = pianoroll.shape
      assert separate_instruments or I == 1
      D = I if separate_instruments else P
      B = T

      xs = np.tile(pianoroll[None], [B, 1, 1, 1])
  
      ts, ds = chordwise_ordering(1, T, D)
      assert ts.shape[1] == 1 and ds.shape[1] == 1
      ts, ds = ts[:, 0], ds[:, 0]
      ts, ds = ts[t_sofar * D:], ds[t_sofar * D:]
  
      # set up sequence of masks to predict the first (according to ordering) instrument for each frame
      mask = []
      mask_scratch = np.ones([T, P, I], dtype=np.float32)
      for j, (t, d) in enumerate(zip(ts, ds)):
        if j % D != 0:
          continue
    
        mask.append(mask_scratch.copy())
        # for predicting the next (according to ordering) frame, unmask the entire current frame
        mask_scratch[t, :, :] = 0
      del mask_scratch
      mask = np.array(mask)
  
      if log_eval_progress:
        states['xs'].append(xs)

      xs_scratch = xs.copy()
  
      # we can't parallelize within the frame, as we need the predictions of some of the other
      # instruments. Hence we outer loop over the instruments and parallelize across frames.
      for d_idx in range(D):
        # call out to the model to get predictions for the first instrument at each time step
        p = predictor(xs_scratch, mask)
  
        # write in predictions and update mask
        t, d = ts[d_idx::D], ds[d_idx::D]
        if separate_instruments:
          xs_scratch[np.arange(B), t, :, d] = np.eye(P)[np.argmax(p[np.arange(B), t, :, d], axis=1)]
          mask[np.arange(B), t, :, d] = 0
        else:
          xs_scratch[np.arange(B), t, d, :] = p[np.arange(B), t, d, :] > 0.5
          mask[np.arange(B), t, d, :] = 0
        # every example in the batch sees one frame more than the previous
        assert np.allclose((1 - mask).sum(axis=(1, 2, 3)),
                           [(k * D + d_idx + 1) * P for k in range(mask.shape[0])])
  
        # in both cases, loss is a vector over batch examples
        if separate_instruments:
          # batched loss at time/instrument pair, summed over pitches
          loss = -np.where(xs[np.arange(B), t, :, d],
                           np.log(p[np.arange(B), t, :, d]),
                           0).sum(axis=1)
        else:
          # batched loss at time/pitch pair, single instrument
          loss = -np.where(xs[np.arange(B), t, d, 0], 
                           np.log(p[np.arange(B), t, d, 0]), 
                           np.log(1 - p[np.arange(B), t, d, 0]))
    
        # at the end we take the mean of the losses. multiply by D because we want to sum over the D
        # axis (instruments or pitches), not average.
        loss *= D

        # Log states.
        if log_eval_progress:
          states["step"].append((t, d)) 
          states["loss"].append(loss) 
          states["predictions"].append(p)
          states["xs_scratch"].append(xs_scratch.copy())
          states["mask"].append(mask.copy())

        yield loss, (t, d, None), states
      assert np.allclose(mask[-1], 0)
  return evaluation_loop(evaluator, pianorolls, log_eval_progress=log_eval_progress, **kwargs)

def compute_chordwise_loss(predictor, pianorolls, convnet_len, eval_len,
                           chronological=False, chronological_margin=0, 
                           pitch_chronological=False,
                           separate_instruments=True, log_eval_progress=False, 
                           pad_mode=None,
                           **kwargs):
  print 'separate_instruments', separate_instruments

  def varwise_losses(xs, t_sofar):
    states = defaultdict()

    print 'before pad # of pieces:', len(xs) 
    xs, lengths = pad(xs, chronological, eval_len, pad_mode)
    print 'padded shape:', xs.shape
    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)
    
    #TODO: assuming one crop.
    if log_eval_progress:
      states['xs'].append(xs) 

    assert chronological or t_sofar == 0
    if chronological:
      # If starting evaluation mid-way, then should not mask out the before parts.
      mask[:, :t_sofar, :, :] = 0

    assert separate_instruments or I == 1
    D = I if separate_instruments else P

    ts, ds = chordwise_ordering(
        B, T, D, chronological=chronological, pitch_chronological=pitch_chronological)
    flattened_idx = t_sofar * D
    for j, (t, d) in enumerate(zip(ts[flattened_idx:], ds[flattened_idx:])):
      assert t.shape == (B,)
      assert d.shape == (B,)

      # working copy to fill in model's own predictions of chord notes.
      # replace model predictions with ground truth when starting a fresh timestep.
      if j % D == 0:
        xs_scratch = np.copy(xs)
        ground_truth_mask = np.copy(mask)

      if log_eval_progress:
        # To make sure indices line up, save the same for all D. 
        states["context"].append(xs_scratch * (1 - ground_truth_mask))  

      # Checks for debugging.
      if chronological:
        if D == I:
          assert len(np.unique(t)) == 1 
          unmasked_count = B * P * (t[0] * I + j % D)  
        else:
          unmasked_count = B * (t[0] * P + j % D)
        assert np.product(mask.shape) - np.sum(mask) == unmasked_count

      # FIXME: put more assertions
      if chronological:
        t0 = t - convnet_len + chronological_margin + 1
      else:
        t0 = t - convnet_len / 2.
      # restrict to valid indices
      t0 = np.round(np.clip(t0, 0, T - convnet_len)).astype(np.int32)
      slice_ = np.arange(B)[:, None], t0[:, None] + np.arange(convnet_len)[None, :]
      cropped_xs_scratch = xs_scratch[slice_]
      cropped_mask = mask[slice_]
      # ensure resulting crop is the correct size. this can fail if all pieces in the batch are
      # shorter than convnet_len, so allow length of longest piece (= T) as well.
      #assert cropped_xs_scratch[0].shape[0] in [convnet_len, T]
      # convnet_len is how much music the convnet sees at once.
      assert cropped_xs_scratch[0].shape[0] == convnet_len
      p = predictor(cropped_xs_scratch, cropped_mask)
  
      # update xs_scratch to contain predictions
      if separate_instruments:
        xs_scratch[np.arange(B), t, :, d] = np.eye(P)[np.argmax(p[np.arange(B), t - t0, :, d], axis=1)]
        mask[np.arange(B), t, :, d] = 0
      else:
        xs_scratch[np.arange(B), t, d, 0] = p[np.arange(B), t - t0, d, 0] > 0.5
        mask[np.arange(B), t, d, 0] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      # in both cases, loss is a vector over batch examples
      if separate_instruments:
        # batched loss at time/instrument pair, summed over pitches
        loss = -np.where(xs[np.arange(B), t, :, d],
                         np.log(p[np.arange(B), t - t0, :, d]),
                         0).sum(axis=1)
      else:
        # batched loss at time/pitch pair, single instrument
        loss = -np.where(xs[np.arange(B), t, d, 0], 
                         np.log(p[np.arange(B), t - t0, d, 0]), 
                         np.log(1-p[np.arange(B), t - t0, d, 0]))

      # at the end we take the mean of the losses. multiply by D because we want to sum over the D
      # axis (instruments or pitches), not average.
      loss *= D
  
      # don't judge predictions of padded elements
      loss = np.where(t < lengths, loss, 0)
      # reweight to account for number of valid losses
      loss *= 1 / (t < lengths).mean()
      
      # Log states.
      if log_eval_progress:
        states["step"].append((t, d)) 
        states["loss"].append(loss) 
        states["predictions"].append(p)
        states["xs_scratch"].append(xs_scratch.copy())
        states["mask"].append(mask.copy())

      if (j+1) % D == 0:
        end_of_frame = True
      else:
        end_of_frame = False
      yield loss, (t, d, end_of_frame), states
    assert np.allclose(mask, 0)
  return evaluation_loop(varwise_losses, pianorolls, chronological=chronological, convnet_len=convnet_len, log_eval_progress=log_eval_progress, **kwargs)


def compute_notewise_loss(predictor, pianorolls, convnet_len, eval_len,
                          chronological=False, chronological_margin=0, 
                          separate_instruments=True, imagewise=False, 
                          log_eval_progress=None, pad_mode=None,
                          **kwargs):
  #FIXME: not yet supporting chronological.
  assert not chronological, 'Not yet supporting chronological'
  def varwise_losses(xs, t_sofar):
    xs, lengths = pad(xs, chronological, eval_len, pad_mode)
    print 'padded shape:', xs.shape
    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)
    
    assert separate_instruments or I == 1
    D = I if separate_instruments else P

    ts, ds = notewise_ordering(B, T, D)
    for t, d in zip(ts, ds):
      assert t.shape == (B,)
      assert d.shape == (B,)

      preds = predictor(xs, mask)
      if separate_instruments:
        loss = -np.where(xs[np.arange(B), t, :, d], np.log(preds[np.arange(B), t, :, d]), 0).sum(axis=1)
        mask[np.arange(B), t, :, d] = 0
      else:
        loss = -np.where(xs[np.arange(B), t, d, 0], 
                         np.log(preds[np.arange(B), t, d, 0]), 
                         np.log(1-preds[np.arange(B), t, d, 0]))
        mask[np.arange(B), t, d, 0] = 0

      if imagewise:
        pixel_count = T * P
        #FIXME: other image datasets will have diff dimensions
        if not FLAGS.eval_test_mode:
          assert 28*28 == pixel_count 
        loss *= pixel_count

      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1
      yield loss, (t, d, False), None

  return evaluation_loop(varwise_losses, pianorolls, chronological=chronological, convnet_len=convnet_len, log_eval_progress=log_eval_progress, **kwargs)


def run(pianorolls=None, wrapped_model=None, sample_name=''):
  print FLAGS.model_name, FLAGS.fold, FLAGS.kind, 
  print FLAGS.num_crops, FLAGS.convnet_len, FLAGS.evaluation_batch_size
  print FLAGS.checkpoint_dir

  # Check for FLAG conflicts.
  if not FLAGS.chronological and FLAGS.pad_mode == 'wrap':
    assert False, 'For non chronological evaluation, padding can only be zero.'
  if FLAGS.chronological and FLAGS.pitch_chronological and FLAGS.num_crops != 1:
    assert False, 'For all chronological evaluation, num_crops should just be 1.'

  if FLAGS.chronological and FLAGS.eval_len != 0:
    assert False, 'If chronological and then should evaluate the whole piece.'

  fn = dict(notewise=compute_notewise_loss,
            chordwise=compute_chordwise_loss,
            chordwise_batch=compute_chordwise_loss_batch,
            imagewise=ft.partial(compute_notewise_loss, imagewise=True),
            mingreedy_notewise=compute_mingreedy_notewise_loss,
            maxgreedy_notewise=compute_maxgreedy_notewise_loss)[FLAGS.kind]
  
  # Retrieve model and hparams.
  if wrapped_model is None:
    hparam_updates = {'use_pop_stats': FLAGS.use_pop_stats}
    wrapped_model = retrieve_model_tools.retrieve_model(
        model_name=FLAGS.model_name, hparam_updates=hparam_updates)
  hparams = wrapped_model.hparams
  assert hparams.use_pop_stats == FLAGS.use_pop_stats
  
  print 'model_name', hparams.model_name
  print hparams.checkpoint_fpath
  # TODO: model_name in hparams is the conv spec class name, not retrieve model_name
  #assert wrapped_model.config.hparams.model_name == FLAGS.model_name
  
  # Get data to evaluate on. 
  if pianorolls is None:
    pianorolls = data_tools.get_data_as_pianorolls(FLAGS.input_dir, hparams, FLAGS.fold)
    print '\nRetrieving pianorolls from %s set of %s dataset.\n' % (
        FLAGS.fold, hparams.dataset)
  else:
    print '\n%s samples were passed in and to be evaluated on %s model.\n' % (
        sample_name, hparams.dataset)

  if isinstance(pianorolls, np.ndarray):
    print pianorolls.shape
  print '# of total pieces in evaluation set:', len(pianorolls)
  lengths = [len(roll) for roll in pianorolls]
  if 'image' not in hparams.dataset and len(np.unique(lengths)) > 1:
    print 'lengths', np.sort(lengths)
  print 'max_len', max(lengths)
  
  # Breaking up long pieces (that are outliers in length).
  pianorolls, max_len = breakup_long_pieces(pianorolls)
  # Batch size after breaking up long pieces. 
  B = len(pianorolls)

  print '# of current pieces used',
  print 'may be more than original count b/c of breaking up longer pieces:', len(pianorolls)
  print 'unique lengths', np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls))
  print 'shape', pianorolls[0].shape
  
  # Length of piece to evaluate. 
  eval_len = max_len if FLAGS.eval_len == 0 else FLAGS.eval_len
  
  # Length that convnet can see, if flags is 0 than use what it was trained on.
  convnet_len = hparams.crop_piece_len if FLAGS.convnet_len == 0 else FLAGS.convnet_len
  print 'updated convnet_len', convnet_len
  
  print FLAGS.convnet_len, hparams.crop_piece_len
  if convnet_len != hparams.crop_piece_len:
    print 'WARNING: convnet_len %r,  hparams.crop_piece_len %r, mismatch' % (convnet_len, hparams.crop_piece_len)
  # Warn if training and evaluation lengths are different.
  if not FLAGS.chronological and eval_len != convnet_len:
    assert False, 'Evaluating on a length (%d) that is different than train len (%d).  Might lead to more cold starts than necessary.' % (eval_len, convnet_len)
 
 
  # Updates eval_batch_size to be the number of pieces available unless FLAGS gives a smaller one.
  eval_batch_size = FLAGS.evaluation_batch_size if B > FLAGS.evaluation_batch_size else B
  if eval_batch_size != hparams.batch_size:
    print 'Using batch size %r for evaluation instead of %r' % (
        eval_batch_size, hparams.batch_size)
  
  if FLAGS.eval_test_mode:
    eval_batch_size = N = 4
    convnet_len = T = 5
    pianorolls = pianorolls[:N]
    pianorolls = [roll[:T] for roll in pianorolls]
    print 'WARNING: testing so only using %d examples' % (len(pianorolls))
    lengths = [len(roll) for roll in pianorolls]
    eval_len = max(lengths)
    assert eval_len == T
    # Try out non-divisible batch size.
    eval_batch_size = 3

  print 'resetting eval_batch_size', eval_batch_size

  # Get folder for previous runs for this config.
  dir_name = '%s-%s-num_rolls=%r-num_crops=%r-crop_len=%r-eval_len=%r--eval_bs=%r-chrono=%s-margin-%s-pitch_chrono=%s-use_pop_stats=%s-eval_test_mode=%r' % (
      FLAGS.fold, FLAGS.kind, B, FLAGS.num_crops, 
      convnet_len, eval_len, eval_batch_size, 
      FLAGS.chronological, FLAGS.chronological_margin, 
      FLAGS.pitch_chronological, hparams.use_pop_stats, FLAGS.eval_test_mode)
  print 'dir_name:', dir_name

  # Check to see if there's previous evaluation losses.
  eval_path = os.path.join(FLAGS.checkpoint_dir, dir_name)
  eval_fpath = os.path.join(eval_path, '%s-evaluations.npz' % sample_name)
  print eval_fpath
  if not os.path.exists(eval_path):
    os.mkdir(eval_path)
  if os.path.exists(eval_fpath):
    print '\nLoading previous log \n'
    print eval_fpath
    eval_data = np.load(eval_fpath)
  else:
    eval_data = None
    print '\nNo previous log.\n'

  # Evaluate!
  try:
    def predictor(xs, masks):
      model = wrapped_model.model
      sess = wrapped_model.sess
      input_data = [mask_tools.apply_mask_and_stack(x, mask) for x, mask in zip(xs, masks)]
      p = sess.run(model.predictions, feed_dict={model.input_data: input_data})
      return p

    mean_loss, sem_loss, N = fn(
        predictor, pianorolls,
        convnet_len=convnet_len, eval_len=eval_len, 
        num_crops=FLAGS.num_crops, 
        eval_data=eval_data,
        separate_instruments=hparams.separate_instruments,
        batch_size=eval_batch_size, eval_fpath=eval_fpath,
        chronological=FLAGS.chronological, 
        chronological_margin=FLAGS.chronological_margin,
        pitch_chronological=FLAGS.pitch_chronological,
        log_eval_progress=FLAGS.log_eval_progress,
        pad_mode=FLAGS.pad_mode)
  except InfiniteLoss:
    print "infinite loss"
    return np.inf, np.inf, np.inf, wrapped_model, eval_path
  print "%s done" % hparams.model_name
  return mean_loss, sem_loss, N, wrapped_model, eval_path

def main(argv):
  try:
    run()
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, KeyboardInterrupt):
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


if __name__ == '__main__':
  tf.app.run()
