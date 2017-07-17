import os, sys, time, contextlib, cPickle as pkl, gzip
import re
from collections import defaultdict
from datetime import datetime
import numpy as np
import tensorflow as tf

import pretty_midi

import mask_tools
import retrieve_model_tools
import data_tools
from npz_to_midi import pianoroll_to_midi
import util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("gen_batch_size", 100, "num of samples to generate in a batch.")
tf.app.flags.DEFINE_string("strategy", None, "")
tf.app.flags.DEFINE_float("temperature", 1, "softmax temperature")
tf.app.flags.DEFINE_integer("piece_length", 32, "num of time steps in generated piece")
tf.app.flags.DEFINE_string(
    "generation_output_dir", None,
    "Output directory for storing the generated Midi.")
tf.app.flags.DEFINE_string(
    "prime_midi_melody_fpath", None,
    "Path to midi melody to be harmonized.")


def main(unused_argv):
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

  hparam_updates = {'use_pop_stats': FLAGS.use_pop_stats}
  wmodel = retrieve_model_tools.retrieve_model(
      model_name=FLAGS.model_name, hparam_updates=hparam_updates)
  hparams = wmodel.hparams
  Globals.separate_instruments = hparams.separate_instruments

  B = FLAGS.gen_batch_size
  T, P, I = hparams.raw_pianoroll_shape
  print B, T, P, I
  hparams.crop_piece_len = FLAGS.piece_length
  T, P, I = hparams.raw_pianoroll_shape
  print B, T, P, I
  shape = [B, T, P, I]

  # Instantiates generation strategy.
  strategy = BaseStrategy.make(FLAGS.strategy, wmodel)
  Globals.bamboo = util.Bamboo()

  # Generates.
  start_time = time.time()
  pianorolls = np.zeros(shape, dtype=np.float32)
  masks = np.ones(shape, dtype=np.float32)
  pianorolls = strategy(pianorolls, masks)
  time_taken = (time.time() - start_time) / 60.0
  
  # Logs final step, without predictions (which are pianorolls here).
  Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)

  # Creates a folder for storing the process of the sampling.
  label = "sample_%s_%s_%s_T%g_l%i_%.2fmin" % (timestamp, FLAGS.strategy, 
    hparams.model_name, FLAGS.temperature, FLAGS.piece_length, time_taken)
  basepath = os.path.join(FLAGS.generation_output_dir, label)
  os.makedirs(basepath)
  
  # Stores all the (intermediate) steps.
  path = os.path.join(basepath, 'intermediate_steps.npz')
  with util.timing('writing_out_sample_npz'):
    print "Writing to", path
    Globals.bamboo.dump(path)
  
  # Makes function to save midi from pianorolls.
  def save_midi_from_pianorolls(rolls, label, midi_path):
    for i, pianoroll in enumerate(rolls):
      midi_fpath = os.path.join(midi_path, "%s_%i.midi" % (label, i))
      midi_data = pianoroll_to_midi(
          pianoroll, qpm=hparams.qpm, quantization_level=hparams.quantization_level, 
          pitch_offset=hparams.pitch_ranges[0])
      print midi_fpath
      midi_data.write(midi_fpath)

  # Saves the results as midi and npy.    
  midi_path = os.path.join(basepath, "midi")
  os.makedirs(midi_path)
  save_midi_from_pianorolls(pianorolls, label, midi_path)
  np.save(os.path.join(basepath, "generated_result.npy"), pianorolls)

  # Save the prime as midi and npy if in harmonization mode.
  # First, checks the stored npz for the first (context) and last step.
  foo = np.load(path)
  for key in foo.keys():
    if re.match(r"0_root/.*?_strategy/.*?_context/0_pianorolls", key):
      context_rolls = foo[key]
  if 'harm' in FLAGS.strategy:
    # Only synthesize the one prime if in Midi-melody-prime mode.
    primes = context_rolls
    if 'Melody' in FLAGS.strategy:
      primes = [context_rolls[0]]
    save_midi_from_pianorolls(primes, label + '_prime', midi_path)
  np.save(os.path.join(basepath, "context.npy"), context_rolls)


# decorator for timing and Globals.bamboo.log structuring
def instrument(label, printon=True, subsample_factor=None):
  def decorator(fn):
    def wrapped_fn(*args, **kwargs):
      with util.timing(label, printon=printon):
        with Globals.bamboo.scope(label, subsample_factor=subsample_factor):
          return fn(*args, **kwargs)
    return wrapped_fn
  return decorator


##################
### Strategies ###
##################
# Commonly used compositions of samplers, user-selectable through FLAGS.strategy

class BaseStrategy(util.Factory):
  def __init__(self, wmodel):
    self.wmodel = wmodel

class HarmonizeMidiMelodyStrategy(BaseStrategy):
  key = "harmonizeMidiMelody"

  def load_midi_melody(self):
    midi = pretty_midi.PrettyMIDI(FLAGS.prime_midi_melody_fpath)
    if len(midi.instruments) != 1:
      raise ValueError(
          'Only one melody/instrument allowed, %r given.' % (
              len(midi.instruments)))
    tempo_change_times, tempo_changes = midi.get_tempo_changes()
    assert len(tempo_changes) == 1
    tempo = tempo_changes[0]
    assert tempo in [60., 120.]
    #assert tempo_changes[0] == 60. or tempo_changes[0] == 120.
    # qpm=60, 16th notes, time taken=1/60 * 1/4
    # qpm=120, 16th notes, time taken=1/120 * /4
    # for 16th in qpm=120 to be rendered correctly in qpm=60, fs=2
    # shape: (128, t)
    if tempo == 120.:
      fs = 2 
    elif tempo == 60.:
      fs = 4
    else:
      assert False, 'Tempo %r not supported yet.' % tempo
    # Returns matrix of shape (128, time) with summed velocities.
    roll = midi.get_piano_roll(fs=fs)  # 16th notes
    roll = np.where(roll>0, 1, 0)
    print roll.shape
    roll = roll.T
    return roll
  
  def make_pianoroll_from_melody_roll(self, mroll, pitch_ranges,
                                      requested_shape):
    # mroll shape: time, pitch
    # requested_shape: batch, time, pitch, instrument
    B, T, P, I = requested_shape
    print 'requested_shape', requested_shape
    assert mroll.ndim == 2
    assert mroll.shape[1] == 128
    low, high = pitch_ranges
    requested_range = high - low + 1
    assert P == requested_range, '%r != %r' % (P, requested_range) 
    if T != mroll.shape[0]:
      print 'WARNING: requested T %r != prime T %r' % (T, mroll.shape[0])
    rolls = np.zeros((B, mroll.shape[0], P, I), dtype=np.float32)
    rolls[:, :, :, 0] = mroll[None, :, low:high+1]
    print 'resulting shape', rolls.shape
    return rolls

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    mroll = self.load_midi_melody()
    pianorolls = self.make_pianoroll_from_melody_roll(
        mroll, self.wmodel.hparams.pitch_ranges, pianorolls.shape)
    masks = HarmonizationMasker()(pianorolls.shape)
    num_steps = np.max(numbers_of_masked_variables(masks))
    print 'num_steps', num_steps
    gibbs = GibbsSampler(num_steps=num_steps,
                         masker=BernoulliMasker(),
                         sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                         schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7))

    with Globals.bamboo.scope("context"):
      context = np.array([mask_tools.apply_mask(pianoroll, mask)
                          for pianoroll, mask in zip(pianorolls, masks)])
      Globals.bamboo.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)

    return pianorolls

class ScratchUpsamplingStrategy(BaseStrategy):
  key = "scratch_upsampling"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    # start with an empty pianoroll of length 1, then repeatedly upsample
    desired_length = pianorolls.shape[1]
    pianorolls = np.zeros_like(pianorolls[:, :1])
    masks = np.ones_like(masks[:, :1])

    sampler = UpsamplingSampler(desired_length=desired_length,
                                sampler=GibbsSampler(masker=BernoulliMasker(),
                                                     sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                                                     schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7)))
    return sampler(pianorolls, masks)

class BachUpsamplingStrategy(BaseStrategy):
  key = "bach_upsampling"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    # optionally start with bach samples
    init_sampler = BachSampler(self.wmodel, temperature=FLAGS.temperature)
    pianorolls = init_sampler(pianorolls, masks)
    masks = np.ones_like(masks)
    desired_length = 4 * pianorolls.shape[1]
    sampler = UpsamplingSampler(desired_length=desired_length,
                                sampler=GibbsSampler(masker=BernoulliMasker(),
                                                     sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                                                     schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7)))
    return sampler(pianorolls, masks)

class RevoiceStrategy(BaseStrategy):
  key = "revoice"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    init_sampler = BachSampler(self.wmodel, temperature=FLAGS.temperature)
    pianorolls = init_sampler(pianorolls, masks)

    sampler = GibbsSampler(masker=BernoulliMasker(),
                           sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                           schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7))
    for i in range(pianorolls.shape[-1]):
      masks = InstrumentMasker(instrument=i)(masks.shape)
      with Globals.bamboo.scope("context"):
        context = np.array([mask_tools.apply_mask(pianoroll, mask)
                            for pianoroll, mask in zip(pianorolls, masks)])
        Globals.bamboo.log(pianorolls=context, masks=masks, predictions=context)
      pianorolls = sampler(pianorolls, masks)
    return pianorolls

class HarmonizationStrategy(BaseStrategy):
  key = "harmonization"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    init_sampler = BachSampler(self.wmodel, temperature=FLAGS.temperature)
    pianorolls = init_sampler(pianorolls, masks)

    masks = HarmonizationMasker()(masks.shape)
    num_steps = np.max(numbers_of_masked_variables(masks))
    gibbs = GibbsSampler(num_steps=num_steps,
                         masker=BernoulliMasker(),
                         sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                         schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7))

    with Globals.bamboo.scope("context"):
      context = np.array([mask_tools.apply_mask(pianoroll, mask)
                          for pianoroll, mask in zip(pianorolls, masks)])
      Globals.bamboo.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    with Globals.bamboo.scope("result"):
      Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)

    return pianorolls

class TransitionStrategy(BaseStrategy):
  key = "transition"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    init_sampler = BachSampler(self.wmodel, temperature=FLAGS.temperature)
    pianorolls = init_sampler(pianorolls, masks)

    masks = TransitionMasker()(masks.shape)
    num_steps = np.max(numbers_of_masked_variables(masks))
    gibbs = GibbsSampler(num_steps=num_steps,
                         masker=BernoulliMasker(),
                         sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                         schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7))

    with Globals.bamboo.scope("context"):
      context = np.array([mask_tools.apply_mask(pianoroll, mask)
                          for pianoroll, mask in zip(pianorolls, masks)])
      Globals.bamboo.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    return pianorolls

class ChronologicalStrategy(BaseStrategy):
  key = "chronological"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    sampler = AncestralSampler(self.wmodel, temperature=FLAGS.temperature,
                               selector=ChronologicalSelector())
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class OrderlessStrategy(BaseStrategy):
  key = "orderless"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    sampler = AncestralSampler(self.wmodel, temperature=FLAGS.temperature,
                               selector=OrderlessSelector())
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class IgibbsStrategy(BaseStrategy):
  key = "igibbs"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=IndependentSampler(self.wmodel, temperature=FLAGS.temperature),
                           schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class AgibbsStrategy(BaseStrategy):
  key = "agibbs"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=YaoSchedule(pmin=0.1, pmax=0.9, alpha=0.7))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class Agibbs99Strategy(BaseStrategy):
  key = "agibbs99"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    pm = .99
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class Agibbs95Strategy(BaseStrategy):
  key = "agibbs95"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    pm = .95
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class Agibbs90Strategy(BaseStrategy):
  key = "agibbs90"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    pm = .90
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class Agibbs75Strategy(BaseStrategy):
  key = "agibbs75"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    pm = .75
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class Agibbs50Strategy(BaseStrategy):
  key = "agibbs50"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    pm = .50
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=BernoulliMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class Cgibbs50Strategy(BaseStrategy):
  key = "cgibbs50"

  @instrument(key + "_strategy")
  def __call__(self, pianorolls, masks):
    pm = 0.50
    num_steps = np.max(numbers_of_masked_variables(masks))
    sampler = GibbsSampler(num_steps=num_steps,
                           masker=ContiguousMasker(),
                           sampler=AncestralSampler(self.wmodel, OrderlessSelector(), temperature=FLAGS.temperature),
                           schedule=ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


################
### Samplers ###
################
# Composable strategies for filling in a masked-out block

class BaseSampler(util.Factory):
  def __init__(self, wmodel, temperature=1, **kwargs):
    self.wmodel = wmodel
    self.temperature = temperature

  def predict(self, pianorolls, masks):
    # TODO: wrap in RobustPredictor from evaluation_tools
    input_data = np.asarray([
      mask_tools.apply_mask_and_stack(pianoroll, mask)
      for pianoroll, mask in zip(pianorolls, masks)])
    predictions = self.wmodel.sess.run(self.wmodel.model.predictions,
                                       {self.wmodel.model.input_data: input_data})
    return predictions

  @classmethod
  def __repr__(cls, self):
    return "samplers.%s" % cls.key

class BachSampler(BaseSampler):
  key = "bach"

  @instrument(key)
  def __call__(self, pianorolls, masks):
    print "Loading validation pieces from %s..." % self.wmodel.hparams.dataset
    bach_pianorolls = data_tools.get_data_as_pianorolls(FLAGS.data_dir, self.wmodel.hparams, 'valid')
    shape = pianorolls.shape
    pianorolls = np.array([pianoroll[:shape[1]] for pianoroll in bach_pianorolls])[:shape[0]]
    Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

class ZeroSampler(BaseSampler):
  key = "zero"

  @instrument(key)
  def __call__(self, pianorolls, masks):
    pianorolls = 0 * pianorolls
    Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

class UniformRandomSampler(BaseSampler):
  key = "uniform"

  @instrument(key)
  def __call__(self, pianorolls, masks):
    predictions = np.ones(pianorolls.shape)
    if Globals.separate_instruments:
      samples = util.sample_onehot(predictions, axis=2, temperature=1)
      assert (samples * masks).sum() == masks.max(axis=2).sum()
    else:
      samples = util.sample_bernoulli(0.5 * predictions, temperature=1)
    pianorolls = np.where(masks, samples, pianorolls)
    Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
    return pianorolls

class IndependentSampler(BaseSampler):
  key = "independent"

  @instrument(key, printon=False)
  def __call__(self, pianorolls, masks):
    predictions = self.predict(pianorolls, masks)
    if Globals.separate_instruments:
      samples = util.sample_onehot(predictions, axis=2,
                                   temperature=self.temperature)
      assert (samples * masks).sum() == masks.max(axis=2).sum()
    else:
      samples = util.sample_bernoulli(predictions, self.temperature)
    pianorolls = np.where(masks, samples, pianorolls)
    Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
    return pianorolls

class AncestralSampler(BaseSampler):
  key = "ancestral"

  def __init__(self, wmodel, selector, temperature=1.):
    self.wmodel = wmodel
    self.selector = selector
    self.temperature = temperature

  @instrument(key)
  def __call__(self, pianorolls, masks):
    B, T, P, I = pianorolls.shape
    assert Globals.separate_instruments or I == 1

    # determine how many model evaluations we need to make
    mask_size = np.max(numbers_of_masked_variables(masks))

    with Globals.bamboo.scope("sequence", subsample_factor=10):
      for _ in range(mask_size):
        predictions = self.predict(pianorolls, masks)
        if Globals.separate_instruments:
          samples = util.sample_onehot(predictions, axis=2, temperature=self.temperature)
          assert np.allclose(samples.max(axis=2), 1)
        else:
          samples = util.sample_bernoulli(predictions, self.temperature)
        selection = self.selector(predictions, masks)
        pianorolls = np.where(selection, samples, pianorolls)
        Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
        masks = np.where(selection, 0., masks)

    Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=predictions)
    assert masks.sum() == 0
    return pianorolls

class GibbsSampler(BaseSampler):
  key = "gibbs"

  def __init__(self, masker, sampler, schedule, num_steps=None):
    self.masker = masker
    self.sampler = sampler
    self.schedule = schedule
    self.num_steps = num_steps

  @instrument(key)
  def __call__(self, pianorolls, masks):
    B, T, P, I = pianorolls.shape
    print 'shape', pianorolls.shape
    num_steps = (np.max(numbers_of_masked_variables(masks))
                 if self.num_steps is None else self.num_steps)
    print 'num_steps', num_steps

    with Globals.bamboo.scope("sequence", subsample_factor=10):
      for s in range(num_steps):
        pm = self.schedule(s, num_steps)
        inner_masks = self.masker(pianorolls.shape, pm=pm, outer_masks=masks)
        pianorolls = self.sampler(pianorolls, inner_masks)
        if Globals.separate_instruments:
          # ensure the sampler did actually sample everything under inner_masks
          assert np.all(np.where(inner_masks.max(axis=2), np.isclose(pianorolls.max(axis=2), 1), 1))
        Globals.bamboo.log(pianorolls=pianorolls, masks=inner_masks, predictions=pianorolls)

    Globals.bamboo.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)
    return pianorolls

  def __repr__(self):
    return "samplers.gibbs(masker=%r, sampler=%r)" % (self.masker, self.sampler)

class UpsamplingSampler(BaseSampler):
  key = "upsampling"

  def __init__(self, desired_length, sampler):
    self.desired_length = desired_length
    self.sampler = sampler

  @instrument(key)
  def __call__(self, pianorolls, masks):
    if not np.all(masks):
      raise NotImplementedError()
    with Globals.bamboo.scope("sequence"):
      while pianorolls.shape[1] < self.desired_length:
        # upsample by zero-order hold and mask out every second time step
        pianorolls = np.repeat(pianorolls, 2, axis=1)
        masks = np.repeat(masks, 2, axis=1)
        masks[:, 1::2] = 1

        with Globals.bamboo.scope("context"):
          context = np.array([mask_tools.apply_mask(pianoroll, mask)
                              for pianoroll, mask in zip(pianorolls, masks)])
          Globals.bamboo.log(pianorolls=context, masks=masks, predictions=context)

        pianorolls = self.sampler(pianorolls, masks)
        masks = np.zeros_like(masks)
    return pianorolls


###############
### Maskers ###
###############
# Strategies for generating masks (possibly within masks).

class BaseMasker(util.Factory):
  @classmethod
  def __repr__(cls, self):
    return "maskers.%s" % cls.key

class BernoulliMasker(BaseMasker):
  key = "bernoulli"

  def __call__(self, shape, pm=None, outer_masks=1.):
    return sample_bernoulli_masks(shape, pm=pm, outer_masks=outer_masks)

class HarmonizationMasker(BaseMasker):
  key = "harmonization"

  def __call__(self, shape, outer_masks=1.):
    if not Globals.separate_instruments:
      raise NotImplementedError()
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, 1:] = 1.
    return masks * outer_masks

class TransitionMasker(BaseMasker):
  key = "transition"

  def __call__(self, shape, outer_masks=1.):
    masks = np.zeros(shape, dtype=np.float32)
    B, T, P, I = shape
    start = int(T * 0.25)
    end = int(T * 0.75)
    masks[:, start:end, :, :] = 1.
    return masks * outer_masks

class InstrumentMasker(BaseMasker):
  key = "instrument"

  def __init__(self, instrument):
    self.instrument = instrument

  def __call__(self, shape, outer_masks=1.):
    if not Globals.separate_instruments:
      raise NotImplementedError()
    masks = np.zeros(shape, dtype=np.float32)
    masks[:, :, :, self.instrument] = 1.
    return masks * outer_masks

class ContiguousMasker(BaseMasker):
  key = "contiguous"

  def __call__(self, shape, pm=None, outer_masks=1.):
    if not Globals.separate_instruments:
      raise NotImplementedError()
    return sample_contiguous_masks(shape, pm=pm, outer_masks=outer_masks)


#################
### Schedules ###
#################
# Used to anneal GibbsSampler.

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


#################
### Selectors ###
#################
# Used in ancestral sampling to determine which variable to sample next.
class BaseSelector(util.Factory):
  pass

class ChronologicalSelector(BaseSelector):
  key = "chronological"

  def __call__(self, predictions, masks):
    B, T, P, I = masks.shape
    # determine which variable to update
    if Globals.separate_instruments:
      # find index of first (t, i) with mask[:, t, :, i] == 1
      selection = np.argmax(np.transpose(masks, axes=[0, 2, 1, 3]).reshape((B, P, T * I)), axis=2)
      selection = np.transpose(np.eye(T * I)[selection].reshape((B, P, T, I)), axes=[0, 2, 1, 3])
    else:
      # find index of first (t, p) with mask[:, t, p, :] == 1
      selection = np.argmax(masks.reshape((B, T * P)), axis=1)
      selection = np.eye(T * P)[selection].reshape((B, T, P, I))
    # Intersect with mask to avoid selecting outside of the mask, e.g. in case some masks[b] is zero
    # everywhere. This can happen inside blocked Gibbs, where different examples have different
    # block sizes.
    return selection * masks

class OrderlessSelector(BaseSelector):
  key = "orderless"

  def __call__(self, predictions, masks):
    B, T, P, I = masks.shape
    if Globals.separate_instruments:
      # select one variable to sample. sample according to normalized mask;
      # is uniform as all masked out variables have equal positive weight.
      selection = masks.max(axis=2).reshape([B, T * I])
      selection = util.sample_onehot(selection, axis=1)
      selection = selection.reshape([B, T, 1, I])
    else:
      selection = masks.reshape([B, T * P])
      selection = util.sample_onehot(selection, axis=1)
      selection = selection.reshape([B, T, P, I])
    # Intersect with mask to avoid selecting outside of the mask, e.g. in case some masks[b] is zero
    # everywhere. This can happen inside blocked Gibbs, where different examples have different
    # block sizes.
    return selection * masks


####################################
### Raw mask-sampling procedures ###
####################################
def sample_bernoulli_masks(shape, pm=None, outer_masks=1.):
  assert pm is not None
  B, T, P, I = shape
  if Globals.separate_instruments:
    probs = np.tile(np.random.random([B, T, 1, I]), [1, 1, P, 1])
  else:
    assert I == 1
    probs = np.random.random([B, T, P, I]).astype(np.float32)  #, dtype=np.float32)
  masks = probs < pm
  return masks * outer_masks

def sample_contiguous_masks(shape, pm=None, outer_masks=1.):
  if not Globals.separate_instruments:
    raise NotImplementedError()
  # unclear how to make this work in the presence of outer_masks; in that case the masked-out
  # variables may themselves be discontiguous.
  if not np.all(outer_masks):
    raise NotImplementedError()
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
  return masks * outer_masks

def numbers_of_masked_variables(masks):
  if Globals.separate_instruments:
    return masks.max(axis=2).sum(axis=(1,2))
  else:
    return masks.sum(axis=(1,2,3))


###########################################
### Globals to keep complexity in check ###
###########################################

class Thing(object):
  pass

Globals = Thing()


if __name__ == "__main__":
  tf.app.run()
