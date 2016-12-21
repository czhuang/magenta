"""Tools for generation.

Example usage:
    $ bazel run :generate_tools -- \
        --prime_fpath=/tmp/primes/prime.mid --validation_set_dir=/tmp/data \
        --output_dir=/tmp/generated
"""
from collections import namedtuple
from collections import defaultdict
from itertools import permutations
import os, sys
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
#from magenta.models.basic_autofill_cnn import plot_tools
from magenta.music.midi_io import sequence_proto_to_midi_file
from magenta.music.note_sequence_io import NoteSequenceRecordWriter
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

#    'validation_set_dir', '/Tmp/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True',
tf.app.flags.DEFINE_string(
    'validation_set_dir', '/data/lisatmp4/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True',
    'Directory for validation set to use in batched prediction')

#    'generation_output_dir', '/Tmp/huangche/generation',
#    'generation_output_dir', '/data/lisatmp4/huangche/new_generated',
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
  if np.isnan(p).any():
    print 'nans in prediction'
    print p
  p = softmax(p, temperature=temperature)
  try:
    pitch = np.random.choice(range(num_pitches), p=p)
  except:
    import pdb; pdb.set_trace()
  return pitch

def softmax(p, axis=None, temperature=1):
  if axis is None:
    axis = p.ndim - 1
  if temperature == 0.:
    # NOTE: may have multiple equal maxima, normalized below
    p = p == np.max(p, axis=axis, keepdims=True)
  else:
    oldp = p
    logp = np.log(p)
    logp /= temperature
    logp -= logp.max(axis=axis, keepdims=True)
    p = np.exp(logp)
  p /= p.sum(axis=axis, keepdims=True)
  if np.isnan(p).any():
    import pdb; pdb.set_trace()
  return p

def sample_onehot(p, axis=None, temperature=1):
  if axis is None:
    axis = p.ndim - 1

  p = softmax(p, axis=axis, temperature=temperature)

  # temporary transpose/reshape to matrix
  if axis != p.ndim - 1:
    permutation = list(range(0, axis)) + list(range(axis + 1, p.ndim)) + [axis]
    p = np.transpose(p, permutation)
  pshape = p.shape
  p = p.reshape([-1, p.shape[-1]])

  assert np.allclose(p.sum(axis=1), 1)

  # sample in a loop i guess -_-
  x = np.zeros(p.shape, dtype=np.float32)
  for i in range(p.shape[0]):
    x[i, np.random.choice(p.shape[1], p=p[i])] = 1.
  
  # transpose/reshape back
  x = x.reshape(pshape)
  if axis != x.ndim - 1:
    x = np.transpose(x, permutation)

  assert np.allclose(x.sum(axis=axis), 1)
  return x

def regenerate_chronological_ti(pianorolls, wrapped_model, config):
  return regenerate_chronological(pianorolls, wrapped_model, config, order="ti")

def regenerate_chronological_it(pianorolls, wrapped_model, config):
  return regenerate_chronological(pianorolls, wrapped_model, config, order="it")

def regenerate_chronological(pianorolls, wrapped_model, config, order="ti"):
  model = wrapped_model.model
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape

  generated_pianoroll = np.zeros(pianorolls[0].shape)
  original_pianoroll = pianorolls[config.requested_index].copy()
  autofill_steps = []

  mask_for_generation = np.ones(pianorolls[0].shape)
  # mask generator for the rest of the batch
  batch_mask_function = getattr(mask_tools, "get_chronological_%s_mask" % order)

  for j in range(num_timesteps * num_instruments):
    if order == "ti":
      time_step = j // num_instruments
      instr_indx = j % num_instruments
    elif order == "it":
      instr_indx = j // num_timesteps
      time_step = j % num_timesteps
    else:
      assert False

    input_data = np.asarray([mask_tools.apply_mask_and_stack(generated_pianoroll, mask_for_generation)] +
                            [mask_tools.apply_mask_and_stack(pianoroll, batch_mask_function(pianoroll.shape))
                             for pianoroll in pianorolls])

    raw_prediction = wrapped_model.sess.run(model.predictions, {model.input_data: input_data})
    prediction = raw_prediction[config.requested_index]
    pitch = sample_pitch(prediction, time_step, instr_indx, num_pitches, config.temperature)

    generated_pianoroll[time_step, pitch, instr_indx] = 1
    mask_for_generation[time_step, :, instr_indx] = 0

    step = AutofillStep(prediction, ((time_step, pitch, instr_indx), 1),
                        generated_pianoroll.copy())
    autofill_steps.append(step)

    sys.stderr.write(".")
    sys.stderr.flush()
  sys.stderr.write("\n")
  print np.sum(generated_pianoroll), num_timesteps * num_instruments
  assert np.sum(generated_pianoroll) == num_timesteps * num_instruments
  return generated_pianoroll, autofill_steps, original_pianoroll, None


def regenerate_random_order(pianorolls, wrapped_model, config):
  model = wrapped_model.model
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape

  generated_pianoroll = np.zeros(pianorolls[0].shape)
  original_pianoroll = pianorolls[config.requested_index].copy()
  autofill_steps = []

  mask_for_generation = np.ones(pianorolls[0].shape)
  # mask generator for the rest of the batch
  batch_mask_function = getattr(mask_tools, "get_fixed_order_mask")

  order = np.arange(num_timesteps * num_instruments)
  np.random.shuffle(order)

  for j in order:
    time_step = j // num_instruments
    instr_indx = j % num_instruments

    input_data = np.asarray([mask_tools.apply_mask_and_stack(generated_pianoroll, mask_for_generation)] +
                            [mask_tools.apply_mask_and_stack(pianoroll, batch_mask_function(pianoroll.shape))
                             for pianoroll in pianorolls])

    raw_prediction = wrapped_model.sess.run(model.predictions, {model.input_data: input_data})
    prediction = raw_prediction[config.requested_index]
    pitch = sample_pitch(prediction, time_step, instr_indx, num_pitches, config.temperature)
  
    generated_pianoroll[time_step, pitch, instr_indx] = 1
    mask_for_generation[time_step, :, instr_indx] = 0
  
    step = AutofillStep(prediction, ((time_step, pitch, instr_indx), 1),
                        generated_pianoroll.copy())
    autofill_steps.append(step)
  
    sys.stderr.write(".")
    sys.stderr.flush()
  sys.stderr.write("\n")
  print np.sum(generated_pianoroll), num_timesteps * num_instruments
  assert np.sum(generated_pianoroll) == num_timesteps * num_instruments
  return generated_pianoroll, autofill_steps, original_pianoroll, None


def regenerate_fixed_order(pianorolls, wrapped_model, config):
  model = wrapped_model.model
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape

  generated_pianoroll = np.zeros(pianorolls[0].shape)
  original_pianoroll = pianorolls[config.requested_index].copy()
  autofill_steps = []

  mask_for_generation = np.ones(pianorolls[0].shape)
  # mask generator for the rest of the batch
  batch_mask_function = getattr(mask_tools, "get_fixed_order_mask")

  order = mask_tools.get_fixed_order_order(num_timesteps)

  for time_step in order:
    for instr_indx in range(num_instruments):
      input_data = np.asarray([mask_tools.apply_mask_and_stack(generated_pianoroll, mask_for_generation)] +
                              [mask_tools.apply_mask_and_stack(pianoroll, batch_mask_function(pianoroll.shape))
                               for pianoroll in pianorolls])
  
      raw_prediction = wrapped_model.sess.run(model.predictions, {model.input_data: input_data})
      prediction = raw_prediction[config.requested_index]
      pitch = sample_pitch(prediction, time_step, instr_indx, num_pitches, config.temperature)
  
      generated_pianoroll[time_step, pitch, instr_indx] = 1
      mask_for_generation[time_step, :, instr_indx] = 0
  
      step = AutofillStep(prediction, ((time_step, pitch, instr_indx), 1),
                          generated_pianoroll.copy())
      autofill_steps.append(step)
  
      sys.stderr.write(".")
      sys.stderr.flush()
  sys.stderr.write("\n")
  print np.sum(generated_pianoroll), num_timesteps * num_instruments
  assert np.sum(generated_pianoroll) == num_timesteps * num_instruments
  return generated_pianoroll, autofill_steps, original_pianoroll, None


def regenerate_voice_by_voice(pianorolls, wrapped_model, config):
  """Rewrites a piece voice by voice.

  The generation process is as follows: start with an original piece, blank
      out one voice, ask the model to fill it back in, then blank out another
      voice and feed in the generated, ask the model to fill in another voice,
      until all voices are rewritten.
  """
  # Only allow the regenerate to overlap when it is a full rewrite.
  if config.prime_voices == config.voices_to_regenerate and len(config.prime_voices) != 4:
    raise ValueError('In rewriting mode, all voices must be rewritten.')
  if config.prime_voices is None and not config.start_with_empty:
    raise ValueError('If no prime voices, then should be start with empty.')
  if config.prime_voices is None and len(config.voices_to_regenerate) != 4:
    raise ValueError('If not primed, must rewrite all voices.')
  # For inpainting mode
  if config.prime_voices is not None and (config.prime_voices != config.voices_to_regenerate) and (len(config.prime_voices) + len(config.voices_to_regenerate) != 4):
    raise ValueError('In inpainting mode, prime and regenerate voices must add up to 4 and not overlap')
    if set(config.prime_voices + config.voices_to_regenerate) != set(range(4)):
      raise ValueError('In inpainting mode, prime and regenerate must cover all 4 voices.')

  # Gets shapes.
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape
  pianoroll_shape = pianorolls[0].shape
  context_pianoroll = np.zeros(pianoroll_shape)
  generated_pianoroll = np.zeros(pianoroll_shape)
  if config.start_with_empty:
    original_pianoroll = np.zeros(pianoroll_shape)
  else:
    original_pianoroll = pianorolls[config.requested_index].copy()
    if original_pianoroll.ndim != 3:
      raise ValueError('Pianoroll should be 3 dimensions, time, pitch, instruments.')
    context_pianoroll[:, :, tuple(config.prime_voices)] = (
        original_pianoroll[:, :, tuple(config.prime_voices)].copy())

    expected_num_ons = len(config.prime_voices) * num_timesteps
    print 'expected_num_ons', expected_num_ons
    if expected_num_ons != np.sum(context_pianoroll):
      raise ValueError('Mismatch in amount of prime expected. %.2f, %.2f' % (expected_num_ons, np.sum(context_pianoroll)))


  model = wrapped_model.model
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
    print instr_idx, 'voice being regenerated, size of context', np.sum(context_pianoroll)  
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

    # Update for last timestep.
    context_pianoroll += generated_pianoroll
    context_pianoroll = np.clip(context_pianoroll, 0, 1)

  print np.sum(generated_pianoroll), num_timesteps * num_instruments
  print np.sum(generated_pianoroll, axis=(0,1))
  #return generated_pianoroll, autofill_steps, original_pianoroll, instr_ordering_str
  # Make sure context_pianoroll is now a full-fledged piece.
  print np.sum(context_pianoroll), num_timesteps * num_instruments
  if np.sum(context_pianoroll) != num_timesteps * num_instruments:
    raise ValueError('Some timesteps are still empty.')
  return context_pianoroll, autofill_steps, original_pianoroll, instr_ordering_str


def generate_gibbs_like(pianorolls, wrapped_model, config):
  model = wrapped_model.model
  assert pianorolls.ndim == 4
  # Gets shapes.
  batch_size, num_timesteps, num_pitches, num_instruments = pianorolls.shape
  pianoroll_shape = pianorolls[0].shape

  generated_pianoroll = np.zeros(pianoroll_shape)
  original_pianoroll = pianorolls[config.requested_index].copy()
  print 'original_pianoroll', original_pianoroll.shape
  context_pianoroll = np.zeros(pianoroll_shape)
  if config.prime_voices is not None:
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
  #mask_func = lambda shape, *_: mask_tools.get_random_all_time_instrument_mask(shape)

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
          # For denoising case.
          if config.start_with_random: 
            input_data = np.concatenate((context_pianoroll, condition_mask), 2)
          else:
            input_data = mask_tools.apply_mask_and_stack(context_pianoroll,
                                                         condition_mask)
        # The other pieces for batch statistics.
        else:
          # TODO: Maybe need to change this mask to match the mask used for generation.
          #mask = mask_tools.get_random_instrument_mask(pianoroll_shape)
          mask = mask_func(pianoroll_shape, config.condition_mask_size, num_maskout)
          if config.start_with_random:
            input_data = mask_tools.perturb_and_stack(pianorolls[data_index], mask)
          else:
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
  if prime_fpath is None:
    prime_fpath = []
  elif isinstance(prime_fpath, basestring):
    prime_fpath = [prime_fpath]
  requested_validation_piece_name = config.requested_validation_piece_name

  # Checks if there are inconsistencies in the types of priming requested.
  if prime_fpath and requested_validation_piece_name is not None:
    raise ValueError(
        'Either prime generation with melody or piece from validation set.')
  start_with_empty = config.start_with_empty
  start_with_random = config.start_with_random
  if (start_with_empty or start_with_random) and (
      prime_fpath or requested_validation_piece_name is not None):
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
    elif start_with_random:
      pianorolls = seeder.get_random_batch_with_random_as_first()
      piece_name = 'random'
    elif prime_fpath:
      pianorolls = seeder.get_random_batch_with_prime(
          prime_fpath[prime_idx], config.prime_voices, config.prime_duration_ratio)
      #piece_name = 'magenta_theme'
      piece_name = os.path.split(os.path.basename(prime_fpath[prime_idx]))[0]
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

      if config.plot_process:
        plot_path = os.path.join(output_path, 'plots')
        if not os.path.exists(plot_path):
          os.mkdir(plot_path)
        plot_tools.plot_steps(autofill_steps, original_pianoroll, plot_path, run_local_id)

      # Save generated pianorolls in steps.
      fname_prefix = ''
      if not isinstance(instr_ordering, list):
        fname_prefix= '%s_' % str(instr_ordering)
      fname_prefix += '%s-%s' % (generate_method_name, run_local_id)
      save_steps(autofill_steps, original_pianoroll, output_path, fname_prefix)


def save_steps(steps, original_pianoroll, output_path, fname_prefix):
  # Save generated pianorolls in steps.
  print '# of steps', len(steps)
  generated_pianorolls = [step.generated_piece for step in steps]
  predictions = [step.prediction for step in steps]
  step_indices = [step.change_to_context[0] for step in steps]
 
  npz_fpath = os.path.join(output_path, fname_prefix + '.npz')
  np.savez_compressed(npz_fpath, generated_pianorolls=generated_pianorolls,
                      predictions=predictions, step_indices=step_indices,
                      original_pianoroll=original_pianoroll)
  print 'NPZ written to', npz_fpath

   
def main(unused_argv):
  print '..............................main..'
  for model_name in "balanced_by_scaling".split():
    import gc
    gc.collect()
    generate_routine(GenerationConfig(
        generate_method_name='regenerate_voice_by_voice',
        model_name=model_name,
        prime_fpath=FLAGS.prime_fpath,
        prime_voices=range(4), 
        validation_path=FLAGS.validation_set_dir,
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_samples=4, #5,
        requested_num_timesteps=128, #16, #128, #64,
        #condition_mask_size=8, #8, #8,
        num_rewrite_iterations=1, #20, #20,
        sample_extra_ratio=0,
        temperature=0.01),
        FLAGS.generation_output_dir)

#  wrapped_model = retrieve_model_tools.retrieve_model(
#      model_name='balanced_by_scaling')
#  for _ in range(4):
#    from datetime import datetime
#    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#    start_time = time.time()
#    intermediates = generate_annealed_gibbs(
#        wrapped_model=wrapped_model,
#        num_steps=100,#2000,
#        temperature=0.01)
#    time_taken = (time.time() - start_time) / 60.0 #  In minutes.
#    run_local_id = 'annealed_gibbs_scratch_%s-%.2fmin' % (timestamp, time_taken)
#    np.savez_compressed(os.path.join(FLAGS.generation_output_dir, run_local_id + '.npz'),
#                        **intermediates)
#  return

  #generate_routine(GENERATION_PRESETS['GenerateGibbsLikeConfig'],
  #                 FLAGS.generation_output_dir)

  #generate_routine(GENERATION_PRESETS['GenerateGibbsLikeFromRandomConfig'],
  #                 FLAGS.generation_output_dir)

  #generate_routine(
  #     GENERATION_PRESETS['RegeneratePrimePieceByGibbsOnMeasures'],
  #     FLAGS.generation_output_dir)
  #generate_routine(
  #    GENERATION_PRESETS['RegenerateValidationPieceVoiceByVoiceConfig'],
  #    FLAGS.generation_output_dir)
  #generate_routine(GENERATION_PRESETS['RegeneratePrimePieceVoiceByVoiceConfig'],
  #                 FLAGS.generation_output_dir)
  #generate_routine(GENERATION_PRESETS['RegeneratePrimePieceFixedOrderConfig'],
  #                 FLAGS.generation_output_dir)
  #generate_routine(
  #    GENERATION_PRESETS['GenerateAccompanimentToPrimeMelodyConfig'],
  #    FLAGS.generation_output_dir)
  #generate_routine(GENERATION_PRESETS['GenerateFromScratchVoiceByVoice'],
  #                 FLAGS.generation_output_dir)

#  for model_name in "Denoising64_128 balanced_fc_mask_only".split():
#    import gc
#    gc.collect()
#    generate_routine(GenerationConfig(
#        generate_method_name='generate_gibbs_like',
#        model_name=model_name,
#        start_with_empty=True,
#        validation_path=FLAGS.validation_set_dir,
#        voices_to_regenerate=range(4),
#        sequential_order_type=RANDOM,
#        num_samples=4, #5,
#        requested_num_timesteps=32, #32, #16, #128, #64,
#        condition_mask_size=8, #8, #8,
#        num_rewrite_iterations=5, #5, #20, #20,
#        sample_extra_ratio=0,
#        temperature=0.00001),
#        FLAGS.generation_output_dir)


#  for model_name in "DeepResidual32_256 DeepResidual64_128 Denoising64_128".split():
#    import gc
#    gc.collect()
#    generate_routine(GenerationConfig(
#        generate_method_name='regenerate_random_order',
#        model_name=model_name,
#        start_with_empty=True,
#        validation_path=FLAGS.validation_set_dir,
#        voices_to_regenerate=range(4),
#        sequential_order_type=RANDOM,
#        num_samples=4, #5,
#        requested_num_timesteps=32, #16, #128, #64,
#        temperature=0.00001),
#        FLAGS.generation_output_dir)
#
#    generate_routine(GenerationConfig(
#        generate_method_name='regenerate_voice_by_voice',
#        model_name=model_name,
#        prime_fpath=FLAGS.prime_fpath,
#        validation_path=FLAGS.validation_set_dir,
#        prime_voices=range(4),
#        voices_to_regenerate=range(4),
#        sequential_order_type=RANDOM,
#        num_samples=4,
#        requested_num_timesteps=32, #16, #128, #64,
#        temperature=0.00001,
#        num_rewrite_iterations=1),
#        FLAGS.generation_output_dir)
#  return
#
#  for model_name, method_name in {"chronological_ti":
#"regenerate_chronological_ti",
#                                  "chronological_it":
#"regenerate_chronological_it",
#                                  "fixed_order_64-128":
#"regenerate_fixed_order"}:
#    generate_routine(GenerationConfig(
#        generate_method_name='regenerate_chronological_ti',
#        model_name='chronological_ti',
#        prime_fpath=FLAGS.prime_fpath,
#        validation_path=FLAGS.validation_set_dir,
#        prime_voices=range(4),
#        voices_to_regenerate=range(4),
#        num_samples=4,
#        requested_num_timesteps=64, #16, #128, #64,
#        temperature=0.00001,
#        num_rewrite_iterations=1),
#        FLAGS.generation_output_dir)
#
#  return
#  for model_name in "random_medium".split():
#    import gc
#    gc.collect()
#    generate_routine(GenerationConfig(
#        generate_method_name='generate_gibbs_like',
#        model_name=model_name,
#        start_with_empty=True,
#        validation_path=FLAGS.validation_set_dir,
#        voices_to_regenerate=range(4),
#        sequential_order_type=RANDOM,
#        num_samples=3, #5,
#        requested_num_timesteps=64, #16, #128, #64,
#        condition_mask_size=8, #8, #8,
#        num_rewrite_iterations=10, #20, #20,
#        sample_extra_ratio=0,
#        temperature=0.01),
#        FLAGS.generation_output_dir)




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

      # Prime setup.
      prime_fpath=None,
      prime_duration_ratio=1,
      validation_path=FLAGS.validation_set_dir,
      requested_validation_piece_name=None,
      start_with_empty=False,
      start_with_random=False,  
 
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
      num_rewrite_iterations=1,  # Number of times to regenerate all the voices.
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

#_DEFAULT_MODEL_NAME = 'DeepResidualRandomMaskTBF'
_DEFAULT_MODEL_NAME = 'Denoising'
_DEFAULT_MODEL_NAME = 'DeepResidual'
_DEFAULT_MODEL_NAME = 'DeepResidual64_128'
_DEFAULT_MODEL_NAME = 'DeepResidual32_256'
_DEFAULT_MODEL_NAME = 'DeepResidual64_128'
_DEFAULT_MODEL_NAME = 'Denoising64_128'


GENERATION_PRESETS = {

    'GenerateRandomOrderConfig': GenerationConfig(
        generate_method_name='regenerate_random_order',
        model_name='balanced_fc_mask_only',
        start_with_empty=True,
        validation_path=FLAGS.validation_set_dir,
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_samples=10, #5,
        requested_num_timesteps=64, #16, #128, #64,
        temperature=0.01,
        plot_process=False),
    
    # Configurations for generating in random instrument cross timestep order.
    'GenerateGibbsLikeConfig': GenerationConfig(
        generate_method_name='generate_gibbs_like',
        model_name='balanced',
        start_with_empty=True,
        validation_path=FLAGS.validation_set_dir,
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_samples=16, #5,
        requested_num_timesteps=32, #64, #16, #128, #64,
        num_rewrite_iterations=5, #20, #20,
        condition_mask_size=8, #8, #8,
        sample_extra_ratio=0, #10, #10,
        temperature=0.01,
        plot_process=False),

    'RegeneratePrimePieceByGibbsOnMeasures': GenerationConfig(
        generate_method_name='generate_gibbs_like',
        model_name= _DEFAULT_MODEL_NAME, #'DeepResidual',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(3), 
        voices_to_regenerate=range(3),
        sequential_order_type=RANDOM,
	    num_samples=4,
        requested_num_timesteps=32, #16, #128, #64,
        num_rewrite_iterations=2, #20, #20,
        condition_mask_size=8, #8, #8,
        sample_extra_ratio=0, 
        temperature=0.1,
        plot_process=False),
    
    'RegeneratePrimePieceVoiceByVoiceConfig': GenerationConfig(
        generate_method_name='regenerate_voice_by_voice',
        model_name='random_medium',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM,
        num_samples=2,
        requested_num_timesteps=64, #16, #128, #64,
        temperature=0,
        num_rewrite_iterations=1,
        plot_process=False),

    # sequential generation
    'RegeneratePrimePieceChronologicalTIConfig': GenerationConfig(
        generate_method_name='regenerate_chronological_ti',
        model_name='chronological_ti',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        num_samples=5,
        requested_num_timesteps=64, #16, #128, #64,
        temperature=0.01,
        num_rewrite_iterations=1,
        plot_process=False),

    'RegeneratePrimePieceChronologicalITConfig': GenerationConfig(
        generate_method_name='regenerate_chronological_it',
        model_name='chronological_it',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        num_samples=5,
        requested_num_timesteps=64, #16, #128, #64,
        temperature=0.01,
        num_rewrite_iterations=1,
        plot_process=False),
    'RegeneratePrimePieceFixedOrderConfig': GenerationConfig(
        generate_method_name='regenerate_fixed_order',
        model_name='fixed_order_64-128',
        prime_fpath=FLAGS.prime_fpath,
        validation_path=FLAGS.validation_set_dir,
        prime_voices=range(4),
        voices_to_regenerate=range(4),
        num_samples=5,
        requested_num_timesteps=64, #16, #128, #64,
        temperature=0.01,
        num_rewrite_iterations=1,
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
        model_name='random_medium',
        start_with_empty=True,
        validation_path=FLAGS.validation_set_dir,
        voices_to_regenerate=range(4),
        sequential_order_type=RANDOM, 
        num_samples=5,
        requested_num_timesteps=64, #32, #64, #16,
        num_rewrite_iterations=5,
        temperature=0.01, #0.1, #0.5, #1 # It seems forward requires a higher temperature, with 0.1 its holding on to same notes.
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

