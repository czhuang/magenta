"""Tools for generation.

Example usage:
    $ bazel run :generate_tools -- \
        --prime_fpath=/tmp/primes/prime.mid --validation_set_dir=/tmp/data \
        --output_dir=/tmp/generated
"""
from collections import namedtuple
from collections import defaultdict
from itertools import permutations
import os
import copy
import time

import cPickle as pickle

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools
from magenta.models.basic_autofill_cnn import retrieve_model_tools
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import seed_tools
#from magenta.models.basic_autofill_cnn.seed_tools import MELODY_VOICE_INDEX
from magenta.models.basic_autofill_cnn import plot_tools
from magenta.lib.midi_io import sequence_proto_to_midi_file
from magenta.lib.note_sequence_io import NoteSequenceRecordWriter
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS
# TODO(annahuang): Set the default input and output_dir to None for opensource.
# condition_on/sample1.mid
# highest.tfrecord
# /u/huangche/data/bach/high0.tfrecord
# 'prime_fpath', '/u/huangche/generated/useful/2016-10-06_17:56:31-DeepResidual/za_last_step_1_2_3_0.tfrecord',
#  'validation_set_dir', '/u/huangche/data/bach/filtered/instrs=4_duration=0.125_sep=True', 
#    'validation_set_dir', '/u/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True',
tf.app.flags.DEFINE_string(
    'prime_fpath', '/u/huangche/data/bach/bwv103.6.tfrecord',
    'Path to the Midi or MusicXML file that is used as a prime.')

tf.app.flags.DEFINE_string(
    'validation_set_dir', '/Tmp/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True',
    'Directory for validation set to use in batched prediction')
#    'generation_output_dir', '/Tmp/huangche/generation',
tf.app.flags.DEFINE_string(
    'generation_output_dir', '/data/lisatmp4/huangche/new_generated',
    'Output directory for storing the generated Midi.')

AutofillStep = namedtuple('AutofillStep', ['prediction', 'change_to_context',
                                           'generated_piece'])

# Enumerations for timestep generation order within a voice.
FORWARD, RANDOM = range(2)

# Enumerations for method used to pick a pitch for each timestep.
ARGMAX, SAMPLE = range(2)


def sample_pitch(prediction, time_step, instr_idx, num_pitches, temperature):
  # At the randomly choosen timestep, sample pitch.
  p = prediction[time_step, :, instr_idx]
  #print 'temperature', temperature
  if temperature == 0.:
    print 'Taking the argmax'
    pitch = np.argmax(p)
  else:
    p = np.exp(np.log(p) / temperature)
    p /= p.sum()
    if np.isnan(p).any():
      print p
    pitch = np.random.choice(range(num_pitches), p=p)
  return pitch


def regenerate_voice_by_voice(pianorolls, wrapped_model, config):
  """Rewrites a piece voice by voice.

  The generation process is as follows: start with an original piece, blank
      out one voice, ask the model to fill it back in, then blank out another
      voice and feed in the generated, ask the model to fill in another voice,
      until all voices are rewritten.
  """
  model = wrapped_model.model

  # Gets shapes.
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape
  pianoroll_shape = pianorolls[0].shape

  generated_pianoroll = np.zeros(pianoroll_shape)
  original_pianoroll = pianorolls[config.requested_index].copy()
  context_pianoroll = original_pianoroll.copy()
  autofill_steps = []

  # Generate instrument by instrument.
  print 'config.instr_ordering', config.instr_ordering
  if config.instr_ordering is not None:
    instr_ordering = config.instr_ordering
  else:
    instr_ordering = list(np.random.permutation(config.voices_to_regenerate))
  print 'instr_ordering', instr_ordering

  instr_ordering_str_list = [str(idx) for idx in instr_ordering]
  instr_ordering_str = '_'.join(instr_ordering_str_list)
  
  duplicated_instr_ordering = instr_ordering * config.num_rewrite_iterations
  print 'duplicated_instr_ordering', duplicated_instr_ordering 
  np.random.shuffle(duplicated_instr_ordering)
  print 'shuffled', duplicated_instr_ordering

  for instr_idx in duplicated_instr_ordering: #instr_ordering:
    print 'instr_idx', instr_idx
    mask_for_generation = mask_tools.get_instrument_mask(pianoroll_shape,
                                                         instr_idx)

    # Mask out the part that is going to be predicted.
    context_pianoroll *= 1 - mask_for_generation
    
    # Since might be regenerating multiple iterations, mask out the current
    # instrument in the generated pianoroll too.
    generated_pianoroll *= 1 - mask_for_generation

    # For each instrument, choose random ordering in time for filling in.
    if config.sequential_order_type == FORWARD:
      ordering = np.arange(num_timesteps)
    elif config.sequential_order_type == RANDOM:
      ordering = np.random.permutation(num_timesteps)
    else:
      raise ValueError('Unknown sequential order.')

    for time_step in ordering:
      # Update the context with the generated notes.
      context_pianoroll += generated_pianoroll
      context_pianoroll = np.clip(context_pianoroll, 0, 1)
      if not config.start_with_empty:
        assert np.allclose(np.unique(context_pianoroll), np.array([0, 1])) or (
            np.allclose(np.unique(context_pianoroll), np.array([0])))

      # Stack all pieces to create a batch.
      input_datas = []
      for data_index in range(batch_size):
        if data_index == config.requested_index:
          input_data = mask_tools.apply_mask_and_stack(context_pianoroll,
                                                       mask_for_generation)
        else:
          mask = mask_tools.get_random_instrument_mask(pianoroll_shape)
          input_data = mask_tools.apply_mask_and_stack(pianorolls[data_index],
                                                       mask)
        input_datas.append(input_data)
      input_datas = np.asarray(input_datas)

      raw_prediction = wrapped_model.sess.run(model.predictions,
                                              {model.input_data: input_datas})

      prediction = raw_prediction[config.requested_index]

      # At the randomly choosen timestep, sample pitch.
      pitch = sample_pitch(prediction, time_step, instr_idx, num_pitches,
                           config.temperature)
      generated_pianoroll[time_step, pitch, instr_idx] = 1
      mask_for_generation[time_step, :, instr_idx] = 0
      change_index = tuple([time_step, pitch, instr_idx])

      step = AutofillStep(prediction, (change_index, 1),
                          generated_pianoroll.copy())
      autofill_steps.append(step)
  print np.sum(generated_pianoroll), num_timesteps * num_instruments
  assert np.sum(generated_pianoroll) == num_timesteps * num_instruments
  return generated_pianoroll, autofill_steps, original_pianoroll, instr_ordering_str


def generate_gibbs_like(pianorolls, wrapped_model, config):
  model = wrapped_model.model
  assert pianorolls.ndim == 4
  # Gets shapes.
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape
  pianoroll_shape = pianorolls[0].shape

  generated_pianoroll = np.zeros(pianoroll_shape)
  original_pianoroll = pianorolls[config.requested_index].copy()
  context_pianoroll = np.zeros(pianoroll_shape)
  print config.prime_voices, original_pianoroll.shape
  context_pianoroll[:, :, tuple(config.prime_voices)] = original_pianoroll[:, :, tuple(config.prime_voices)]

  # To check if all was regenerated
  global_check = np.ones(pianoroll_shape)
  autofill_steps = []

  # Estimated number of prediction steps to cover the whole pianoroll.
  if config.condition_mask_size is None:
    raise ValueError('condition_mask_size is not yet set, still None.')
  inverse_blankout_ratio = int(np.ceil(
      float(num_timesteps) / config.condition_mask_size * (1 + config.sample_extra_ratio)))
  #print 'inverse_blankout_ratio', inverse_blankout_ratio
  num_steps_per_rewrite = num_instruments * inverse_blankout_ratio
  #print 'num_steps_per_rewrite', num_steps_per_rewrite

  #mask_border = config.condition_mask_size / 2
  num_maskout = 4
  #print mask_border, 'num_maskout', num_maskout
  mask_func = mask_tools.get_multiple_random_instrument_time_mask_by_mask_size

  # TODO: for debug
  counter = 0
  for _ in range(config.num_rewrite_iterations * num_steps_per_rewrite):
    
    # Create mask for one instrument, with blankout_timesteps blanked out.
    # TODO: Make it possible to use any mask.  Can just set a config parameter.
    # If blank out multiple instruments at the same time then can more easily resample the harmony.
    condition_mask = mask_func(pianoroll_shape, config.condition_mask_size, num_maskout, config.voices_to_regenerate)
    #condition_mask = mask_func(pianoroll_shape, mask_border, num_maskout)

    # Mask out the part that is going to be predicted.
    context_pianoroll *= 1 - condition_mask 
    
    # Since might be regenerating multiple iterations, mask out the current
    # instrument in the generated pianoroll too.
    generated_pianoroll *= 1 - condition_mask 

    #print 'condition_mask.shape', condition_mask.shape
    print np.sum(condition_mask), config.condition_mask_size * num_pitches * num_maskout
    #assert np.sum(condition_mask) == config.condition_mask_size * num_pitches
    # TODO(annahuang): they might overlap
    #assert np.sum(condition_mask) == config.condition_mask_size * num_pitches * num_maskout
    global_check -= condition_mask
    global_check = np.clip(global_check, 0, 1)
    #print 'np.sum(global_check)', np.sum(global_check)
    #if np.sum(global_check) == 0.:
    #  print 'first iter where all has been rewritten', counter

    # Need to generate one by one.
    resample_order = np.random.permutation(np.arange(int(np.sum(condition_mask)/num_pitches)))
    indices_to_resample = np.array(np.where(condition_mask[:, :1, :]>0)).T
    #print 'indices_to_resample.shape', indices_to_resample.shape   
    #print counter, indices_to_resample
    
    for resample_idx in resample_order:
      time_step, _, instr_idx = indices_to_resample[resample_idx]
      # TODO(annahuang): print for debugging
      counter += 1
      if counter % 100 == 0:
        print 'taken steps', counter, '# of pianoroll cells unvisited', np.sum(global_check)
     
      # Update the context with recently generated note. 
      context_pianoroll += generated_pianoroll
      context_pianoroll = np.clip(context_pianoroll, 0, 1)
      
      # Stack all pieces to create a batch.
      input_datas = []
      for data_index in range(batch_size):
        # The piece being generated.
        if data_index == config.requested_index:
          input_data = mask_tools.apply_mask_and_stack(context_pianoroll,
                                                       condition_mask)
        # The other pieces for batch statistics.
        else:
          # TODO: Maybe need to change this mask to match the mask used for generation.
          #mask = mask_tools.get_random_instrument_mask(pianoroll_shape)
          mask = mask_func(pianoroll_shape, config.condition_mask_size, num_maskout)
          input_data = mask_tools.apply_mask_and_stack(pianorolls[data_index],
                                                       mask)
        input_datas.append(input_data)
      input_datas = np.asarray(input_datas)
      #print 'sess.run...' 
      predictions = wrapped_model.sess.run(model.predictions,
                                              {model.input_data: input_datas})
  
      prediction = predictions[config.requested_index]
      pitch = sample_pitch(prediction, time_step, instr_idx, num_pitches,
                           config.temperature)
  
      generated_pianoroll[time_step, pitch, instr_idx] = 1
  
      change_index = tuple([time_step, pitch, instr_idx])
      step = AutofillStep(prediction, (change_index, 1),
                          generated_pianoroll.copy())
      autofill_steps.append(step)
  
      # Update.
      condition_mask[time_step, :, instr_idx] = 0

  print 'np.sum(global_check) should equal to zero', np.sum(global_check), 
  print 'ratio unvisited', np.sum(global_check)/np.product(pianoroll_shape)
  #assert np.sum(global_check) == 0.
  return generated_pianoroll, autofill_steps, original_pianoroll, None


def generate_routine(config, output_path):
  prime_fpath = config.prime_fpath
  requested_validation_piece_name = config.requested_validation_piece_name

  # Checks if there are inconsistencies in the types of priming requested.
  if prime_fpath is not None and requested_validation_piece_name is not None:
    raise ValueError(
        'Either prime generation with melody or piece from validation set.')
  start_with_empty = config.start_with_empty
  if start_with_empty and (prime_fpath is not None or
                           requested_validation_piece_name is not None):
    raise ValueError(
        'Generate from empty initialization requested but prime given.')

  # Gets name of pretrained model to be retrieved.
  model_name = config.model_name

  # Gets data.
  seeder = seed_tools.get_seeder(config.validation_path, model_name)
  seeder.crop_piece_len = config.requested_num_timesteps
  requested_index = config.requested_index

  # Gets unique output path.
  timestamp_str = config_tools.get_current_time_as_str()
  run_id = '%s-%s' % (timestamp_str, model_name)#, piece_name)
  output_path = os.path.join(output_path, run_id)
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # Save config, as .py so that can read with syntax highlighting.
  with open(os.path.join(output_path, 'config.py'), 'w') as p:
    p.writelines(str(config))

  # Gets model.
  print 'Retrieving %s model...' % model_name
  wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
  print 'Finished retrieving %s model.' % model_name

  # Generate and synths output.
  generate_method_name = config.generate_method_name
  for prime_idx in range(config.num_diff_primes):
    
    # Gets prime and batch.
    if start_with_empty:
      pianorolls = seeder.get_random_batch_with_empty_as_first()
      piece_name = 'empty'
    elif prime_fpath is not None:
      pianorolls = seeder.get_random_batch_with_prime(
          prime_fpath, config.prime_voices, config.prime_duration_ratio)
      #piece_name = 'magenta_theme'
      piece_name = os.path.split(os.path.basename(prime_fpath))[0]
    elif requested_validation_piece_name is not None:
      pianorolls = seeder.get_batch_with_piece_as_first(
          requested_validation_piece_name, 0)
      piece_name = requested_validation_piece_name
    else:
      for i in range(30):
        try:
          pianorolls, piece_names = seeder.get_random_batch(
              config.requested_index, return_names=True)
          break
        except:
          tf.logging.error('Prime piece shorter then requested crop length.')
          if i == 30 - 1:
            raise ValueError('Did not manage to find a prime piece sufficient for requested crop length') 
          
      #print'piece_names', piece_names[config.requested_index], piece_names
      #piece_name = os.path.split(piece_names[config.requested_index])[0]
      piece_name = piece_names[config.requested_index]
    print 'Piece name:', piece_name

    #seqs_by_ordering = defaultdict(list)
    # TODO(annahuang): Use consistent instrument or voice.
    instr_orderings = list(permutations(config.voices_to_regenerate))
    if config.num_samples_per_instr_ordering is not None:
      instr_orderings = instr_orderings * config.num_samples_per_instr_ordering
    elif config.num_samples is not None:
      # TODO(annahuang): A hack, instrument ordering not used directly for this case, instead just for looping
      instr_orderings = range(config.num_samples)
      #instr_orderings = instr_orderings[:config.num_samples]
    else:
      tf.log.warning('Should specify num_samples or num_samples_per_instr_ordering, otherwise assumes num_samples_per_instr_ordering to be 1')    
    
    for i, instr_ordering in enumerate(instr_orderings):	
      # TODO: hack to clear dictionary for pickling.
      seqs_by_ordering = defaultdict(list)
      start_time = time.time()
      # Generate.
      if isinstance(instr_ordering, list):
        config.instr_ordering = instr_ordering
      generated_results = globals()[generate_method_name](pianorolls,
                                                          wrapped_model, config)
      generated_pianoroll, autofill_steps, original_pianoroll, instr_ordering_str = generated_results
      time_taken = (time.time() - start_time) / 60.0 #  In minutes. 
 
      run_local_id = '%d-%.2fmin-%s-%d-%s-%s' % (i, time_taken, run_id, prime_idx, piece_name, instr_ordering_str)
      if config.run_description is not None:
        run_local_id = config.run_description + run_local_id
      
      # TODO(annahuang): Remove, just for debugging.
      if isinstance(instr_ordering, list):
        requested_instr_ordering_str = '_'.join(str(i) for i in instr_ordering)
        print 'requested', requested_instr_ordering_str, instr_ordering_str 
        if instr_ordering_str is not None and instr_ordering_str != requested_instr_ordering_str:
          raise ValueError('Instrument ordering mismatch')

      # Synths original, only for the first sample.
      if original_pianoroll is not None and not i:
        original_seq = seeder.encoder.decode(original_pianoroll)
        fpath = os.path.join(
            output_path, 'original-%s-run_id_%s.midi' % (
                generate_method_name, run_local_id))
        sequence_proto_to_midi_file(original_seq, fpath)
        print 'original', fpath
      elif original_pianoroll is None:
        original_seq = None     
 
      # Synths generated.
      # TODO(annahuang): Output sequence that merges prime and generated.
      generated_seq = seeder.encoder.decode(generated_pianoroll)
      fpath = os.path.join(
          output_path, 'generated-%s-run_id_%s.midi' % (
              generate_method_name, run_local_id))
      print 'generated', fpath
      sequence_proto_to_midi_file(generated_seq, fpath)
      tfrecord_fpath = os.path.splitext(fpath)[0] + '.tfrecord'
      writer = NoteSequenceRecordWriter(tfrecord_fpath)     
      writer.write(generated_seq)    

      seqs_by_ordering[instr_ordering_str].append([
          generated_seq, autofill_steps, original_seq, instr_ordering_str])

      if config.plot_process:
        plot_path = os.path.join(output_path, 'plots')
        if not os.path.exists(plot_path):
          os.mkdir(plot_path)
        plot_tools.plot_steps(autofill_steps, original_pianoroll, plot_path, run_local_id)

      # Pickle every sample for this prime's generated sequences.
      if not isinstance(instr_ordering, list):
        pickle_fname = '%s_' % str(instr_ordering)
      else:
        pickle_fname = ''
      pickle_fname += '%s-%s.pkl' % (generate_method_name, run_local_id)
      with open(os.path.join(output_path, pickle_fname), 'wb') as p:
        pickle.dump(seqs_by_ordering, p)

   
def main(unused_argv):
  print '..............................main..'
  #generate_routine(
  #     GENERATION_PRESETS['RegeneratePrimePieceByGibbsOnMeasures'],
  #     FLAGS.generation_output_dir)
  #generate_routine(
  #    GENERATION_PRESETS['RegenerateValidationPieceVoiceByVoiceConfig'],
  #    FLAGS.generation_output_dir)
  #generate_routine(GENERATION_PRESETS['RegeneratePrimePieceVoiceByVoiceConfig'],
  #                 FLAGS.generation_output_dir)
  #generate_routine(
  #    GENERATION_PRESETS['GenerateAccompanimentToPrimeMelodyConfig'],
  #    FLAGS.generation_output_dir)
  #generate_routine(GENERATION_PRESETS['GenerateFromScratchVoiceByVoice'],
  #                 FLAGS.generation_output_dir)

  generate_routine(GENERATION_PRESETS['GenerateGibbsLikeConfig'],
                   FLAGS.generation_output_dir)


class GenerationConfig(object):
  """Configurations for regenerating all voices voice by voice.

  Attributes:
    generate_method_name: A string that gives the name of the function used for
        generation.
    model_name: A string that gives the ...
  """
  _defaults = dict(
      run_description=None,
      generate_method_name='regenerate_voice_by_voice',
      model_name='DeepResidual',
      prime_fpath=None,
      prime_duration_ratio=1,
      validation_path=None,
      requested_validation_piece_name=None,
      start_with_empty=False,

      # Request index in batch.
      requested_index=0,

      # Generation parameters.
      prime_voices=None,
      voices_to_regenerate=range(4),
      instr_ordering=None,
      sequential_order_type=RANDOM,
      pitch_picking_method=SAMPLE,
      temperature=1,
      num_diff_primes=1,
      num_samples=1,  # None to specify count by permuations.
      num_samples_per_instr_ordering=None,  # Only used when we care about analyzing different instrument ordering as oppose to just getting more samples.
      requested_num_timesteps=8,
      num_rewrite_iterations=10,  # Number of times to regenerate all the voices.
      condition_mask_size=None,
      sample_extra_ratio=None,
      plot_process=False)

  def __init__(self, *args, **init_hparams):
    unknown_params = set(init_hparams) - set(GenerationConfig._defaults)
    if unknown_params:
      raise ValueError('Unknown hyperparameters: %s', unknown_params)

    # Update instance with default class variables.
    for key, value in GenerationConfig._defaults.items():
      if key in init_hparams:
        value = init_hparams[key]
      setattr(self, key, value)

  def __str__(self):
    config_str = 'config = dict(\n'
    for key, value in self.__dict__.items():
      if isinstance(value, str):
        config_str += '  %s="%s",\n' % (str(key), str(value))
      else:
        config_str += '  %s=%s,\n' % (str(key), str(value))
    config_str += ')'
    return config_str

_DEFAULT_MODEL_NAME = 'DeepResidualRandomMask'

GENERATION_PRESETS = {

    'RegeneratePrimePieceByGibbsOnMeasures': GenerationConfig(
        generate_method_name='generate_gibbs_like',
        model_name= _DEFAULT_MODEL_NAME, #'DeepResidual',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(3),
        voices_to_regenerate=range(3),
        sequential_order_type=RANDOM,
        num_samples=4,
        requested_num_timesteps=16, #16, #128, #64,
        num_rewrite_iterations=10, #20, #20,
        condition_mask_size=8, #8, #8,
        sample_extra_ratio=0, 
        temperature=1,
        plot_process=False),

    'RegenerateValidationPieceVoiceByVoiceConfig': GenerationConfig(
        generate_method_name='regenerate_voice_by_voice',
        model_name='DeepResidual',
        validation_path=FLAGS.validation_set_dir,
        #requested_validation_piece_name='bwv103.6.mxl',
        requested_validation_piece_name=None,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_diff_primes=100,
        num_samples=16,
        requested_num_timesteps=32*2, #128,
        plot_process=False),
    'RegeneratePrimePieceVoiceByVoiceConfig': GenerationConfig(
        generate_method_name='regenerate_voice_by_voice',
        model_name='DeepResidual',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_samples=1, #5,
        requested_num_timesteps=16, #16, #128, #64,
        num_rewrite_iterations=1, #20, #20,
        condition_mask_size=4, #8, #8,
        sample_extra_ratio=1, #10, #10,
        temperature=0.1,
        plot_process=False),
    # Configuration for generating an accompaniment to prime melody.
    'GenerateAccompanimentToPrimeMelodyConfig': GenerationConfig(
        generate_method_name='regenerate_voice_by_voice',
        model_name='DeepResidual',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_duration_ratio=1,
        prime_voices=[0],
        voices_to_regenerate=[1, 2, 3],  #list(set(range(4)) - set([MELODY_VOICE_INDEX])),
        sequential_order_type=RANDOM, #FORWARD,
        num_samples=30,
        requested_num_timesteps=32,
        plot_process=False),
    # Configurations for generating in random instrument cross timestep order.
    'GenerateFromScratchVoiceByVoice': GenerationConfig(
        generate_method_name='regenerate_voice_by_voice',
        model_name='DeepResidual',
        start_with_empty=True,
        validation_path=FLAGS.validation_set_dir,
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM, 
        num_samples=3,
        requested_num_timesteps=32, #32, #64, #16,
        num_rewrite_iterations=5,
        temperature=0, #0.1, #0.5, #1 # It seems forward requires a higher temperature, with 0.1 its holding on to same notes.
        plot_process=False),
    # Configurations for generating in random instrument cross timestep order.
    'GenerateGibbsLikeConfig': GenerationConfig(
        generate_method_name='generate_gibbs_like',
        model_name='DeepResidual',
        start_with_empty=True,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_samples=5, #5, #5,
        requested_num_timesteps=64, #64, #16, #128, #64,
        num_rewrite_iterations=40,  #40, #20, #20,
        condition_mask_size=8, #8, #8,
        sample_extra_ratio=0, #10, #10,
        temperature=0.1,
        plot_process=False),
    # Configurations for generating in random instrument cross timestep order.
    'InpaintingConfig': GenerationConfig(
        generate_method_name='generate_gibbs_like',
        model_name='DeepResidual',
        start_with_empty=False,
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        voices_to_regenerate=None, # Does not apply to this setting, because just fill in all that's empty
        sequential_order_type=RANDOM,
        num_samples=2,
        requested_num_timesteps=4,
        num_rewrite_iterations=2,
        plot_process=False)
}

if __name__ == '__main__':
  tf.app.run()
