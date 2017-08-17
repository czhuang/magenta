import os, sys, time, cPickle as pkl, gzip
import re
import numpy as np
import tensorflow as tf

import pretty_midi

import lib.mask
import lib.graph
import lib.data
import lib.util
import lib.logging
import lib.sampling
import lib.pianoroll


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
tf.app.flags.DEFINE_string('checkpoint', None, 'path to checkpoint file')


def main(unused_argv):
  wmodel = lib.graph.load_checkpoint(FLAGS.checkpoint)
  hparams = wmodel.hparams

  B = FLAGS.gen_batch_size
  T, P, I = hparams.pianoroll_shape
  print B, T, P, I
  hparams.crop_piece_len = FLAGS.piece_length
  T, P, I = hparams.pianoroll_shape
  print B, T, P, I
  shape = [B, T, P, I]

  # Instantiates generation strategy.
  logger = lib.logging.Logger()
  strategy = BaseStrategy.make(FLAGS.strategy, wmodel, logger)

  # Generates.
  start_time = time.time()
  pianorolls = strategy(shape)
  time_taken = (time.time() - start_time) / 60.0
  
  # Logs final step
  logger.log(pianorolls=pianorolls)

  # Creates a folder for storing the process of the sampling.
  label = "sample_%s_%s_%s_T%g_l%i_%.2fmin" % (lib.util.timestamp(), FLAGS.strategy, 
    hparams.architecture, FLAGS.temperature, FLAGS.piece_length, time_taken)
  basepath = os.path.join(FLAGS.generation_output_dir, label)
  os.makedirs(basepath)
  
  # Stores all the (intermediate) steps.
  path = os.path.join(basepath, 'intermediate_steps.npz')
  with lib.util.timing('writing_out_sample_npz'):
    print "Writing to", path
    logger.dump(path)
  
  # Makes function to save midi from pianorolls.
  def save_midi_from_pianorolls(rolls, label, midi_path):
    for i, pianoroll in enumerate(rolls):
      midi_fpath = os.path.join(midi_path, "%s_%i.midi" % (label, i))
      midi_data = lib.pianoroll.pianoroll_to_midi(pianoroll)
      #pianoroll, qpm=hparams.qpm, quantization_level=hparams.quantization_level, 
      #    pitch_offset=hparams.min_pitch)
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
      np.save(os.path.join(basepath, "context.npy"), context_rolls)
      if 'harm' in FLAGS.strategy:
        # Only synthesize the one prime if in Midi-melody-prime mode.
        primes = context_rolls
        if 'Melody' in FLAGS.strategy:
          primes = [context_rolls[0]]
        save_midi_from_pianorolls(primes, label + '_prime', midi_path)
      break


##################
### Strategies ###
##################
# Commonly used compositions of samplers, user-selectable through FLAGS.strategy

class BaseStrategy(lib.util.Factory):
  def __init__(self, wmodel, logger):
    self.wmodel = wmodel
    self.logger = logger

  def __call__(self, shape):
    label = "%s_strategy" % self.key
    with lib.util.timing(label):
      with self.logger.scope(label):
        return self.run(shape)

  def blank_slate(self, shape):
    return (np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32))

  # convenience function to avoid passing the same arguments over and over
  def make_sampler(self, key, **kwargs):
    kwargs.update(wmodel=self.wmodel, logger=self.logger)
    return lib.sampling.BaseSampler.make(key, **kwargs)

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
  
  def make_pianoroll_from_melody_roll(self, mroll, requested_shape):
    # mroll shape: time, pitch
    # requested_shape: batch, time, pitch, instrument
    B, T, P, I = requested_shape
    print 'requested_shape', requested_shape
    assert mroll.ndim == 2
    assert mroll.shape[1] == 128
    hparams = self.wmodel.hparams
    assert P == hparams.num_pitches, '%r != %r' % (P, hparams.num_pitches)
    if T != mroll.shape[0]:
      print 'WARNING: requested T %r != prime T %r' % (T, mroll.shape[0])
    rolls = np.zeros((B, mroll.shape[0], P, I), dtype=np.float32)
    rolls[:, :, :, 0] = mroll[None, :, hparams.min_pitch:hparams.max_pitch + 1]
    print 'resulting shape', rolls.shape
    return rolls

  def run(self, shape):
    mroll = self.load_midi_melody()
    pianorolls = self.make_pianoroll_from_melody_roll(mroll, shape)
    masks = HarmonizationMasker()(shape)
    gibbs = self.make_sampler(
        "gibbs", masker=lib.sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent",
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.YaoSchedule())

    with self.logger.scope("context"):
      context = np.array([lib.mask.apply_mask(pianoroll, mask)
                          for pianoroll, mask in zip(pianorolls, masks)])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)

    return pianorolls

class ScratchUpsamplingStrategy(BaseStrategy):
  key = "scratch_upsampling"

  def run(self, shape):
    # start with an empty pianoroll of length 1, then repeatedly upsample
    initial_shape = list(shape)
    desired_length = shape[1]
    initial_shape[1] = 1
    initial_shape = tuple(shape)

    pianorolls, masks = self.blank_slate(initial_shape)

    sampler = self.make_sampler(
        "upsampling",
        desired_length=desired_length,
        sampler=self.make_sampler(
            "gibbs",
            masker=lib.sampling.BernoulliMasker(),
            sampler=self.make_sampler("independent",
                                      temperature=FLAGS.temperature),
            schedule=lib.sampling.YaoSchedule()))

    return sampler(pianorolls, masks)

class BachUpsamplingStrategy(BaseStrategy):
  key = "bach_upsampling"

  def run(self, shape):
    # optionally start with bach samples
    init_sampler = self.make_sampler("bach", temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)
    desired_length = 4 * shape[1]
    sampler = self.make_sampler(
        "upsampling",
        desired_length=desired_length,
        sampler=self.make_sampler(
            "gibbs",
            masker=lib.sampling.BernoulliMasker(),
            sampler=self.make_sampler("independent",
                                      temperature=FLAGS.temperature),
            schedule=lib.sampling.YaoSchedule()))
    return sampler(pianorolls, masks)

class RevoiceStrategy(BaseStrategy):
  key = "revoice"

  def run(self, shape):
    init_sampler = self.make_sampler("bach", temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)

    sampler = self.make_sampler(
        "gibbs",
        masker=lib.sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent",
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.YaoSchedule())

    for i in range(shape[-1]):
      masks = lib.sampling.InstrumentMasker(instrument=i)(shape)
      with self.logger.scope("context"):
        context = np.array([lib.mask.apply_mask(pianoroll, mask)
                            for pianoroll, mask in zip(pianorolls, masks)])
        self.logger.log(pianorolls=context, masks=masks, predictions=context)
      pianorolls = sampler(pianorolls, masks)

    return pianorolls

class HarmonizationStrategy(BaseStrategy):
  key = "harmonization"

  def run(self, shape):
    init_sampler = self.make_sampler("bach", temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)

    masks = lib.sampling.HarmonizationMasker()(shape)

    gibbs = self.make_sampler(
        "gibbs",
        masker=lib.sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent",
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.YaoSchedule())

    with self.logger.scope("context"):
      context = np.array([lib.mask.apply_mask(pianoroll, mask)
                          for pianoroll, mask in zip(pianorolls, masks)])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    with self.logger.scope("result"):
      self.logger.log(pianorolls=pianorolls, masks=masks, predictions=pianorolls)

    return pianorolls

class TransitionStrategy(BaseStrategy):
  key = "transition"

  def run(self, shape):
    init_sampler = lib.sampling.BachSampler(
        wmodel=self.wmodel, temperature=FLAGS.temperature)
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = init_sampler(pianorolls, masks)

    masks = TransitionMasker()(shape)
    gibbs = self.make_sampler(
        "gibbs",
        masker=lib.sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent",
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.YaoSchedule())

    with self.logger.scope("context"):
      context = np.array([lib.mask.apply_mask(pianoroll, mask)
                          for pianoroll, mask in zip(pianorolls, masks)])
      self.logger.log(pianorolls=context, masks=masks, predictions=context)
    pianorolls = gibbs(pianorolls, masks)
    return pianorolls

class ChronologicalStrategy(BaseStrategy):
  key = "chronological"

  def run(self, shape):
    sampler = self.make_sampler(
        "ancestral",
        temperature=FLAGS.temperature,
        selector=lib.sampling.ChronologicalSelector())
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class OrderlessStrategy(BaseStrategy):
  key = "orderless"

  def run(self, shape):
    sampler = self.make_sampler(
        "ancestral",
        temperature=FLAGS.temperature,
        selector=lib.sampling.OrderlessSelector())
    pianorolls, masks = self.blank_slate(shape)
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class IgibbsStrategy(BaseStrategy):
  key = "igibbs"

  def run(self, shape):
    pianorolls, masks = self.blank_slate(shape)
    sampler = self.make_sampler(
        "gibbs",
        masker=lib.sampling.BernoulliMasker(),
        sampler=self.make_sampler("independent",
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.YaoSchedule())
    pianorolls = sampler(pianorolls, masks)
    return pianorolls

class AgibbsStrategy(BaseStrategy):
  key = "agibbs"

  def run(self, shape):
    pianorolls, masks = self.blank_slate(shape)
    sampler = self.make_sampler(
        "gibbs",
        masker=lib.sampling.BernoulliMasker(),
        sampler=self.make_sampler("ancestral",
                                  selector=lib.sampling.OrderlessSelector(),
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.YaoSchedule())
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


class Cgibbs50Strategy(BaseStrategy):
  key = "cgibbs50"

  def run(self, shape):
    pianorolls, masks = self.blank_slate(shape)
    pm = 0.50
    sampler = self.make_sampler(
        "gibbs",
        masker=lib.sampling.ContiguousMasker(),
        sampler=self.make_sampler("ancestral",
                                  selector=lib.sampling.OrderlessSelector(),
                                  temperature=FLAGS.temperature),
        schedule=lib.sampling.ConstantSchedule(pm))
    pianorolls = sampler(pianorolls, masks)
    return pianorolls


# Variations from the convergence plot in the paper
def _generate_convergence_strategies():
  strategies = []
  for maskout_percentage in [1, 50, 75, 90, 95, 99]:
    class Strategy(BaseStrategy):
      _maskout_percentage = maskout_percentage
      key = "agibbs_p%02i" % _maskout_percentage

      def run(self, shape):
        pianorolls, masks = self.blank_slate(shape)
        pm = self._maskout_percentage / 100.
        sampler = self.make_sampler(
            "gibbs",
            masker=lib.sampling.ContiguousMasker(),
            sampler=self.make_sampler(
                "ancestral",
                selector=lib.sampling.OrderlessSelector(),
                temperature=FLAGS.temperature),
            schedule=lib.sampling.ConstantSchedule(pm))
        pianorolls = sampler(pianorolls, masks)
        return pianorolls
    # keep a reference to the class so it stays alive
    strategies.append(Strategy)
  return strategies
_convergence_strategies = _generate_convergence_strategies()


if __name__ == "__main__":
  tf.app.run()
