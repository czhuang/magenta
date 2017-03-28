from collections import defaultdict

import os, sys, time
from datetime import datetime
import numpy as np, tensorflow as tf
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, generate_tools, data_tools

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

def sample_masks(shape, separate_instruments=None, pm=None, k=None):
  assert (pm is None) != (k is None)
  assert separate_instruments is not None
  # like mask_tools.get_random_all_time_instrument_mask except
  # produces a full batch of masks. the size of the mask follows a
  # binomial distribution, but all examples in the batch have the same
  # mask size. (this simplifies the sequential sampling logic.)
  B, T, P, I = shape
  assert separate_instruments or I == 1
  if separate_instruments:
    D = I
  else:
    D = P
  if k is None:
    k = (np.random.rand(T * D) < pm).sum()
  masks = np.zeros(shape, dtype=np.float32)
  for b in range(B):
    js = np.random.choice(T * D, size=k, replace=False)
    t = js / D
    i = js % D
    if separate_instruments:
      masks[b, t, :, i] = 1.
    else:
      masks[b, t, i, 0] = 1.
  if separate_instruments:
    assert np.allclose(masks.max(axis=2).sum(axis=(1,2)), k)
  else: 
    assert np.allclose(masks.sum(axis=(1,2,3)), k)
  return masks


def sample_bernoulli(p, temperature):
  B, T, P, I = p.shape
  assert I == 1
  if temperature == 0.:
    sampled = p > 0.5
  else:
    axis = 3
    pp = np.concatenate((p, (1-p)), axis=3)
    logpp = np.log(pp)
    logpp /= temperature
    logpp -= logpp.max(axis=axis, keepdims=True)
    #p = np.where(logpp > 0, 
    #             1 / (1 + np.exp(-logpp)), 
    #             np.exp(logpp) / (np.exp(logpp) + 1))
    p = np.exp(logpp)
    p /= p.sum(axis=axis, keepdims=True)
    p = p[:, :, :, :1]
    print "%.5f < %.5f < %.5f < %.5f < %.5g" % (np.min(p), np.percentile(p, 25), np.percentile(p, 50), np.percentile(p, 75), np.max(p))

    sampled = np.random.random(p.shape) < p
  return sampled


def sample_contiguous_masks(shape, pm=None):
  B, T, P, I = shape
  # (deterministically) determine how many 4-timestep chunks to mask out
  chunk_size = 4
  k = int(np.ceil(pm * T * I)) // chunk_size
  masks = np.zeros(shape, dtype=np.float32)
  for b in range(B):
    ms = None
    # m * chunk_size > T would cause overlap, which would cause the
    # mask size to be smaller than desired, which breaks sequential
    # sampling.
    while ms is None or any(m * chunk_size > T for m in ms):
      if ms is not None:
        print "resampling mask to avoid overlap"
      # assign chunks to instruments
      ms = np.random.multinomial(k, pvals=[1./I] * I)
    for i in range(I):
      dt = ms[i] * chunk_size
      t = np.random.choice(T)
      t = np.arange(t, t + dt) % T
      masks[b, t, :, i] = 1.
  return masks

def sample_masks_within_masks(shape, context_masks, pm=None, k=None):
  assert (pm is None) != (k is None)
  B, T, P, I = shape
  # Across batch, blankout size will be the same but positions will vary.
  assert np.unique(context_masks.sum(axis=(1,2,3))).size == 1

  # Use one mask to sample size of blankout.
  if k is None:
    context_mask = context_masks[0].max(axis=1).reshape((T*I,))
    temp_mask = np.random.rand(T * I) < pm
    k = int((context_mask * temp_mask).sum())
  print "k out of mask_size", k, context_mask.sum()
  
  masks = np.zeros_like(context_masks)
  for b in range(B):
    ts, is_ = np.where(context_masks[b].max(axis=1))
    js = np.random.choice(ts.size, size=k, replace=False)
    masks[b, ts[js], :, is_[js]] = 1.
  assert np.allclose(masks.max(axis=2).sum(axis=(1,2)), k)
  return masks

class BernoulliMasker(object):
  def __call__(self, shape, separate_instruments, pm=None):
    return sample_masks(shape, separate_instruments, pm=pm)

  def __repr__(self):
    return "BernoulliMasker()"

class BernoulliInpaintingMasker(object):
  def __init__(self, context_kind):
    self.context_kind = context_kind
    self._context_masks = None

  def context_masks(self, shape):
    # Only set context masks once.  Afterwards in Gibbs they are kept fixed.
    if self._context_masks is not None:
      return self._context_masks
    try:
      masker = getattr(self.__class__, "get_%s_masks" % self.context_kind)
    except KeyError:
      assert False, "ERROR: %s context_kind is not implemented" % self.context_kind
    self._context_masks = masker(shape)
    return self._context_masks

  def __call__(self, shape, separate_instruments, pm=None):
    if not separate_instruments:
      raise NotImplementedError()
    return sample_masks_within_masks(shape, self.context_masks(shape), pm=pm)

  def __repr__(self):
    return "BernoulliInpaintingMasker(context_kind=%r)" % (self.context_kind)

  @staticmethod
  def get_bernoulli_masks(shape):
    B, T, P, I = shape
    return sample_masks(shape, k=int(T*I*0.75))

  @staticmethod
  def get_harmonization_masks(shape):
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 1:] = 1.
    return masks
  
  @staticmethod
  def get_transition_masks(shape):
    B, T, P, I = shape
    masks = np.zeros(shape, dtype=np.float32)
    start = int(T/2. - T/4.)
    end = int(T/2. + T/4.)
    masks[:, start:end, :, :] = 1.
    assert masks.max(axis=2).sum() == B * T * I / 2
    return masks

  @staticmethod
  def get_inner_voices_masks(shape):
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 1:3] = 1.
    return masks
  
  @staticmethod
  def get_soprano_masks(shape):
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 0] = 1.
    return masks

  @staticmethod
  def get_alto_masks(shape):
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 1] = 1.
    return masks

  @staticmethod
  def get_tenor_masks(shape):
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 2] = 1.
    return masks

  @staticmethod
  def get_bass_masks(shape):
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 3] = 1.
    return masks

class ContiguousMasker(object):
  def __call__(self, shape, pm=None):
    return sample_contiguous_masks(shape, pm=pm)

  def __repr__(self):
    return "ContiguousMasker()"


class BachSampler(object):
  def __init__(self, **kwargs):
    pass

  def __call__(self, wmodel, pianorolls, masks):
    print "Loading validation pieces from %s..." % wmodel.hparams.dataset
    bach_pianorolls = data_tools.get_data_as_pianorolls(FLAGS.input_dir, wmodel.hparams, 'valid')
    shape = pianorolls.shape
    pianorolls = np.array([pianoroll[:shape[1]] for pianoroll in bach_pianorolls])[:shape[0]]
    yield pianorolls, masks, pianorolls

class ZeroSampler(object):
  def __init__(self, **kwargs):
    pass

  def __call__(self, wmodel, pianorolls, masks):
    yield 0 * pianorolls, masks, 0 * pianorolls

class UniformRandomSampler(object):
  def __init__(self, separate_instruments=None, **kwargs):
    assert isinstance(separate_instruments, bool)
    self.separate_instruments = separate_instruments

  def __call__(self, wmodel, pianorolls, masks):
    print 'random sampling...'
    #FIXME: a hack
    predictions = np.ones(pianorolls.shape) * 0.5
    if self.separate_instruments:
      #pianorolls = generate_tools.sample_onehot(
      #    1 + np.random.rand(B, T, P, I), axis=2)
      samples = generate_tools.sample_onehot(predictions, axis=2,
                                             temperature=1)
      assert (samples * masks).sum() == masks.max(axis=2).sum()
    else:
      samples = sample_bernoulli(predictions, temperature=1)

    pianorolls = np.where(masks, samples, pianorolls)
    yield pianorolls, masks, predictions

  def __repr__(self):
    return "RandomSampler"


class IndependentSampler(object):
  def __init__(self, temperature=1, separate_instruments=None):
    self.temperature = temperature
    assert separate_instruments is not None
    self.separate_instruments = separate_instruments

  def __call__(self, wmodel, pianorolls, masks):
    print 'independent sampling...'
    input_data = np.asarray([
        mask_tools.apply_mask_and_stack(pianoroll, mask)
        for pianoroll, mask in zip(pianorolls, masks)])
    predictions = wmodel.sess.run(wmodel.model.predictions,
                                  {wmodel.model.input_data: input_data})
    if self.separate_instruments:
      samples = generate_tools.sample_onehot(predictions, axis=2,
                                             temperature=self.temperature)
      assert (samples * masks).sum() == masks.max(axis=2).sum()
    else:
      samples = sample_bernoulli(predictions, self.temperature)

    #B, T, P, I = pianorolls.shape
    #assert samples.sum() == B * T * I
    pianorolls = np.where(masks, samples, pianorolls)
    yield pianorolls, masks, predictions

  def __repr__(self):
    return "IndependentSampler(temperature=%r)" % self.temperature

class ChronologicalSampler(object):
  def __init__(self, temperature=1, separate_instruments=None):
    self.temperature = temperature
    assert separate_instruments is not None
    self.separate_instruments = separate_instruments

  def __call__(self, wmodel, pianorolls, masks):
    B, T, P, I = pianorolls.shape
    assert self.separate_instruments or I == 1

    # determine how many model evaluations we need to make
    if self.separate_instruments:
      mask_size = np.unique(masks.max(axis=2).sum(axis=(1,2)))
    else:
      mask_size = np.unique(masks.sum(axis=(1,2,3)))
    # everything is better if mask sizes are the same throughout the batch
    assert mask_size.size == 1

    for s in range(mask_size):
      print '\tsequential step', s
      input_data = np.asarray([
          mask_tools.apply_mask_and_stack(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)])
      predictions = wmodel.sess.run(wmodel.model.predictions,
                                    {wmodel.model.input_data: input_data})

      # sample predictions
      if self.separate_instruments:
        samples = generate_tools.sample_onehot(
            predictions, axis=2, temperature=self.temperature)
        assert np.allclose(samples.max(axis=2), 1)
      else:
        samples = sample_bernoulli(predictions, self.temperature)

      # determine which variable to update
      if self.separate_instruments:
        # find index of first (t, i) with mask[:, t, :, i] == 1
        selection = np.argmax(np.transpose(masks, axes=[0, 2, 1, 3]).reshape((B, P, T * I)), axis=2)
        selection = np.transpose(np.eye(T * I)[selection].reshape((B, P, T, I)), axes=[0, 2, 1, 3])
      else:
        # find index of first (t, p) with mask[:, t, p, :] == 1
        selection = np.argmax(masks.reshape((B, T * P)), axis=1)
        selection = np.eye(T * P)[selection].reshape((B, T, P, I))

      pianorolls = np.where(selection, samples, pianorolls)
      previous_masks = masks.copy()
      masks = np.where(selection, 0., masks)
      yield pianorolls, previous_masks, predictions  
    assert masks.sum() == 0
    if self.separate_instruments:
      assert np.allclose(pianorolls.max(axis=2), 1)

  def __repr__(self):
    return "ChronologicalSampler(temperature=%r)" % self.temperature

class SequentialSampler(object):
  def __init__(self, temperature=1, separate_instruments=None):
    self.temperature = temperature
    assert separate_instruments is not None
    self.separate_instruments = separate_instruments

  def __call__(self, wmodel, pianorolls, masks):
    B, T, P, I = pianorolls.shape
    assert self.separate_instruments or I == 1

    # determine how many model evaluations we need to make
    if self.separate_instruments:
      mask_size = np.unique(masks.max(axis=2).sum(axis=(1,2)))
    else:
      mask_size = np.unique(masks.sum(axis=(1,2,3)))
    # everything is better if mask sizes are the same throughout the batch
    assert mask_size.size == 1

    for s in range(mask_size):
      print '\tsequential step', s
      input_data = np.asarray([
          mask_tools.apply_mask_and_stack(pianoroll, mask)
          for pianoroll, mask in zip(pianorolls, masks)])
      predictions = wmodel.sess.run(wmodel.model.predictions,
                                    {wmodel.model.input_data: input_data})
      if self.separate_instruments:
        samples = generate_tools.sample_onehot(
            predictions, axis=2, temperature=self.temperature)
        assert np.allclose(samples.max(axis=2), 1)
        # select one variable to sample. sample according to normalized mask;
        # is uniform as all masked out variables have equal positive weight.
        selection = masks.max(axis=2).reshape([B, T * I])
        selection = generate_tools.sample_onehot(selection, axis=1)
        selection = selection.reshape([B, T, 1, I])
      else:
        samples = sample_bernoulli(predictions, self.temperature)
        selection = masks.reshape([B, T * P])
        selection = generate_tools.sample_onehot(selection, axis=1)
        selection = selection.reshape([B, T, P, I])
      pianorolls = np.where(selection, samples, pianorolls)
      previous_masks = masks.copy()
      masks = np.where(selection, 0., masks)
      yield pianorolls, previous_masks, predictions  
    assert masks.sum() == 0
    if self.separate_instruments:
      assert np.allclose(pianorolls.max(axis=2), 1)

  def __repr__(self):
    return "SequentialSampler(temperature=%r)" % self.temperature

class BisectingSampler(object):
  def __init__(self, temperature=1):
    self.independent_sampler = IndependentSampler(temperature=temperature)

  def __call__(self, wmodel, pianorolls, masks):
    B, T, P, I = pianorolls.shape

    # determine how many variables need sampling
    mask_size = np.unique(masks.max(axis=2).sum(axis=(1,2)))
    # everything is better if mask sizes are the same throughout the batch
    assert mask_size.size == 1

    # determine bisection, a mask that is 1 where variables are to be
    # sampled independently
    bisection_size = int(np.ceil(mask_size / 2.))
    bisection = np.zeros_like(masks)
    for b in range(B):
      pm = masks[b].max(axis=1).reshape([T * I])
      pm /= pm.sum()
      js = np.random.choice(T * I, size=bisection_size, p=pm, replace=False)
      assert len(js) == bisection_size
      t = js / I
      i = js % I
      bisection[b, t, :, i] = 1.

    # sample one half independently
    ind_pianorolls, _, predictions = self.independent_sampler(wmodel, pianorolls, masks)
    pianorolls = np.where(bisection, ind_pianorolls, pianorolls)

    if bisection_size == mask_size:
      yield pianorolls, masks, predictions

    # sample the other half by recursive bisection
    pianorolls = self(wmodel, pianorolls, np.where(bisection, 0., masks))
    yield pianorolls, masks, predictions

  def __repr__(self):
    return "BisectingSampler(temperature=%r)" % self.independent_sampler.temperature

class YaoSchedule(object):
  def __init__(self, pmin=0.1, pmax=0.9, alpha=0.8):
    self.pmin = pmin
    self.pmax = pmax
    self.alpha = alpha

  def __call__(self, i, n):
    wat = (self.pmax - self.pmin) * i / n
    return max(self.pmin, self.pmax - wat / self.alpha)

  def __repr__(self):
    return ("YaoSchedule(pmin=%r, pmax=%r, alpha=%r)"
            % (self.pmin, self.pmax, self.alpha))

class ConstantSchedule(object):
  def __init__(self, p):
    self.p = p

  def __call__(self, i, n):
    return self.p

  def __repr__(self):
    return "ConstantSchedule(%r)" % self.p

class Gibbs(object):
  def __init__(self, num_steps, masker, sampler, schedule, separate_instruments):
    self.num_steps = num_steps
    self.masker = masker
    self.sampler = sampler
    self.schedule = schedule
    self.separate_instruments = separate_instruments

  def __call__(self, wmodel, pianorolls):
    B, T, P, I = pianorolls.shape
    print 'shape', pianorolls.shape
    for s in range(self.num_steps):
      pm = self.schedule(s, self.num_steps)
      masks = self.masker(pianorolls.shape, self.separate_instruments, pm)
      #pianorolls = self.sampler(wmodel, pianorolls, masks)
      #assert (pianorolls * masks).sum() == masks.max(axis=2).sum()
      ##assert pianorolls.sum() == B * T * I
      #yield pianorolls, masks
      for pianorolls, masks, predictions in self.sampler(wmodel, pianorolls, masks):
        yield pianorolls, masks, predictions

  def __repr__(self):
    return ("Gibbs(num_steps=%r, masker=%r, schedule=%r, sampler=%r)"
            % (self.num_steps, self.masker, self.schedule, self.sampler))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("gen_batch_size", 100, "num of samples to generate in a batch.")
tf.app.flags.DEFINE_integer("num_steps", None, "number of gibbs steps to take")
tf.app.flags.DEFINE_string("generation_type", None, "unconditioned, inpainting, voicewise")
tf.app.flags.DEFINE_string("sampler", None, "independent or sequential or chronological or bisecting")
tf.app.flags.DEFINE_string("masker", None, "bernoulli or contiguous or bernoulli_inpainting")
tf.app.flags.DEFINE_string("context_kind", None, "bernoulli, harmonization, transition, inner_voices, tenor")
tf.app.flags.DEFINE_string("schedule", None, "yao or constant")
tf.app.flags.DEFINE_float("schedule_yao_pmin", 0.1, "")
tf.app.flags.DEFINE_float("schedule_yao_pmax", 0.9, "")
tf.app.flags.DEFINE_float("schedule_yao_alpha", 0.7, "")
tf.app.flags.DEFINE_float("schedule_constant_p", None, "")
tf.app.flags.DEFINE_float("temperature", 1, "softmax temperature")
tf.app.flags.DEFINE_string("initialization", "random", "how to obtain initial piano roll; random or independent or sequential or bach or zero")
tf.app.flags.DEFINE_integer("piece_length", 32, "num of time steps in generated piece")
# already defined in generate_tools.py
#tf.app.flags.DEFINE_string(
#    "generation_output_dir", "/Tmp/cooijmat/autofill/generate",
#    "Output directory for storing the generated Midi.")
tf.app.flags.DEFINE_float("log_percent", 0.05, "Percentage of intermediate generation steps to save.")


def main(unused_argv):
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

  if FLAGS.generation_type is None:
    assert False, "Please specify generation type: unconditioned or inpainting"

  # Setup init sampler.
  init_sampler = dict(
      random=UniformRandomSampler,
      independent=IndependentSampler,
      sequential=SequentialSampler,
      chronological=ChronologicalSampler,
      bisecting=BisectingSampler,
      bach=BachSampler,
      zero=ZeroSampler,
  )[FLAGS.initialization](temperature=FLAGS.temperature, 
                          separate_instruments=FLAGS.separate_instruments)
  
  # Setup sampler.
  if FLAGS.sampler is None:
    sampler = None
  else:
    sampler = dict(
        independent=IndependentSampler,
        sequential=SequentialSampler,
        chronological=ChronologicalSampler,
        bisecting=BisectingSampler
    )[FLAGS.sampler](temperature=FLAGS.temperature,
                     separate_instruments=FLAGS.separate_instruments)

  # Setup schedule.
  if FLAGS.schedule == "yao":
    schedule = YaoSchedule(pmin=FLAGS.schedule_yao_pmin,
                        pmax=FLAGS.schedule_yao_pmax,
                        alpha=FLAGS.schedule_yao_alpha)
  elif FLAGS.schedule == "constant":
    schedule = ConstantSchedule(p=FLAGS.schedule_constant_p)
  elif FLAGS.num_steps == 0:
    schedule = None
  else:
    assert False, 'No schedule specified.'
 
  # Setup masker. 
  if FLAGS.masker is None:
    masker = None 
  else:
    # TODO: for NADE, because need context mask, would need to setup a BernoulliInpainting master too.  Should separate these different functions.
    masker = dict(bernoulli=BernoulliMasker(),
                  contiguous=ContiguousMasker(),
                  bernoulli_inpainting=BernoulliInpaintingMasker(FLAGS.context_kind),
    )[FLAGS.masker]

  hparam_updates = {'use_pop_stats': FLAGS.use_pop_stats}
  wmodel = retrieve_model_tools.retrieve_model(
      model_name=FLAGS.model_name, hparam_updates=hparam_updates)

  hparams = wmodel.hparams
  B = FLAGS.gen_batch_size
  T, P, I = hparams.raw_pianoroll_shape
  print B, T, P, I
  hparams.crop_piece_len = FLAGS.piece_length
  T, P, I = hparams.raw_pianoroll_shape
  print B, T, P, I
  
  intermediates = defaultdict(list)

  def log(pianorolls, masks, predictions, step_idx):
    intermediates["pianorolls"].append(pianorolls.copy())
    intermediates["masks"].append(masks.copy())
    intermediates["predictions"].append(predictions.copy())
    intermediates["step_idx"].append(step_idx)

  # Include initialization time.  Allows us to also time NADE sampling.
  start_time = time.time()

  pianorolls = np.zeros([B, T, P, I], dtype=np.float32)
  masks = np.ones([B, T, P, I], dtype=np.float32)

  log_denominator = 1
  if FLAGS.initialization == 'sequential':
    D = T * I if FLAGS.separate_instruments else T * P
    log_denominator = int(np.ceil(D * FLAGS.log_percent))
    last_step = D
    print 'log_denominator', log_denominator

  # FIXME since context_kind == "original" is now initialization == "bach", there is no way to do
  # ancestral inpainting on bach pieces.
  iter_idx = 0
  for pianorolls, masks, predictions in init_sampler(wmodel, pianorolls, masks):
    print iter_idx,
    if iter_idx % log_denominator == 0 or iter_idx == last_step - 1:
      print 'Logging step', iter_idx
      log(pianorolls=pianorolls, masks=masks, predictions=predictions, step_idx=iter_idx)
    iter_idx += 1

  gibbs = Gibbs(num_steps=FLAGS.num_steps,
                masker=masker, sampler=sampler, schedule=schedule,
                separate_instruments=FLAGS.separate_instruments)

  def do_gibbs(pianorolls):
    iter_idx = 0
    for pianorolls, masks, predictions in gibbs(wmodel, pianorolls):
      print iter_idx,
      if (iter_idx % int(FLAGS.log_percent * FLAGS.num_steps) == 0 or
          FLAGS.num_steps - 1 == iter_idx):
        print 'Logging step', iter_idx
        log(pianorolls=pianorolls, masks=masks, predictions=predictions, step_idx=iter_idx)
      iter_idx += 1
    return pianorolls

  if FLAGS.generation_type == "inpainting":
    # if doing inpainting, masker must expose inpainting masks
    # if doing inpainting, get initial masks and initialize masked-out portion
    masks = masker.context_masks(pianorolls.shape)
    # Logs context.
    log(pianorolls=pianorolls, masks=masks, predictions=np.zeros_like(pianorolls), step_idx=-1)
    do_gibbs(pianorolls)
  elif FLAGS.generation_type == "unconditioned":
    do_gibbs(pianorolls)
  elif FLAGS.generation_type == "voicewise":
    for voice in "soprano alto tenor bass".split():
      assert FLAGS.masker == "bernoulli_inpainting"
      masker = BernoulliInpaintingMasker(voice)
      masks = masker.context_masks(pianorolls.shape)
      gibbs.masker = masker # yuck
      pianorolls = do_gibbs(pianorolls)
  else:
    assert False, 'Generation type %s not yet supported.' % FLAGS.generation_type  

  time_taken = (time.time() - start_time) / 60.0 #  In minutes.
  label = "".join(c if c.isalnum() else "_" for c in repr(gibbs))
  label = "fromscratch_%s_init=%s_%s_%r_%s_%.2fmin" % (FLAGS.model_name, FLAGS.initialization, label, FLAGS.temperature, timestamp, time_taken)
  path = os.path.join(FLAGS.generation_output_dir, label + ".npz")
  print "Writing to", path  
  np.savez_compressed(path, **intermediates)

if __name__ == "__main__":
  with pdb_post_mortem():
    tf.app.run()
