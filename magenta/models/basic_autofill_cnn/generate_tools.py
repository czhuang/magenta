"""Tools for generation.

Example usage:
 $ bazel run :generate_tools -- \
  --prime_fpath=/tmp/primes/prime.mid --output_dir=/tmp/generated
"""
from collections import namedtuple
import os
import copy

 

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools
from magenta.models.basic_autofill_cnn import retrieve_model_tools
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import seed_tools
from magenta.models.basic_autofill_cnn.seed_tools import MELODY_VOICE_INDEX
from magenta.models.basic_autofill_cnn import plot_tools
from  magenta.lib.midi_io import sequence_proto_to_midi_file
from  magenta.protobuf import music_pb2


FLAGS f.app.flags.FLAGS
# TODO(annahuang): Set the default input and output_dir to None for opensource.
tf.app.flags.DEFINE_string(
 'prime_fpath',
 'condition_on/magenta_theme.xml',
 'Path to the Midi or MusicXML file that is used as rime.')
tf.app.flags.DEFINE_string(
 'output_dir',
 '/usr/local/ /home/annahuang/magenta_tmp/generated/',
 'Output directory for storing the generated Midi.')


AutofillStep amedtuple('AutofillStep', ['prediction',
           change_to_context',
           generated_piece'])

# Enumerations for timestep generation order within oice.
FORWARD, RANDOM ange(2)


def sample_pitch(prediction, time_step, instr_idx, num_pitches, temperature):
  t the randomly choosen timestep, sample pitch.
   prediction[time_step, :, instr_idx]
   np.exp(np.log(p) emperature)
  = p.sum()
  if np.isnan(p).any():
 print p
  pitch p.random.choice(range(num_pitches), p=p)
  return pitch


def regenerate_voice_by_voice(pianorolls, wrapped_model, config):
  """Rewrite iece voice by voice.

  The generation process is as follows: start with an original piece, blank
   out one voice, ask the model to fill it back in, then blank out another
   voice and feed in the generated, ask the model to fill in another voice,
   until all voices are rewritten.
  """
  model rapped_model.model

  ets shapes.
  batch_size, num_timesteps, num_pitches, num_instruments ianorolls.shape
  pianoroll_shape ianorolls[0].shape

  generated_pianoroll p.zeros(pianoroll_shape)
  original_pianoroll ianorolls[config.requested_index].copy()
  context_pianoroll riginal_pianoroll.copy()
  autofill_steps ]

  enerate instrument by instrument.
  instr_ordering p.random.permutation(config.voices_to_regenerate)
  for instr_idx in instr_ordering:
 print 'instr_idx', instr_idx
 mask_for_generation ask_tools.get_instrument_mask(
  pianoroll_shape, instr_idx)

 # Mask out the part that is going to be predicted.
 context_pianoroll *=  mask_for_generation

 # For each instrument, choose random ordering in time for filling in.
 if config.sequential_order_type == FORWARD:
   ordering p.arange(num_timesteps)
 elif config.sequential_order_type == RANDOM:
   ordering p.random.permutation(num_timesteps)
 else:
   raise ValueError("Unknown sequential order.")

 for time_step in ordering:
   pdate the context with the generated notes.
   context_pianoroll += generated_pianoroll
   context_pianoroll p.clip(context_pianoroll, 0, 1)
   if not config.start_with_empty:
  assert np.allclose(np.unique(context_pianoroll), np.array([0, 1])) or (
    np.allclose(np.unique(context_pianoroll), np.array([0])))

   tack all pieces to create atch.
   input_datas ]
   for data_index in range(batch_size):
  if data_index == config.requested_index:
    input_data ask_tools.apply_mask_and_stack(context_pianoroll, mask_for_generation)
  else:
    mask ask_tools.get_random_instrument_mask(pianoroll_shape)
    input_data ask_tools.apply_mask_and_stack(
     pianorolls[data_index], mask)
  input_datas.append(input_data)
   input_datas p.asarray(input_datas)

   raw_prediction rapped_model.sess.run(
    model.predictions,
    {model.input_data: input_datas}
   )

   prediction aw_prediction[config.requested_index]

   t the randomly choosen timestep, sample pitch.
   pitch ample_pitch(prediction, time_step, instr_idx, num_pitches,
       onfig.temperature)
   generated_pianoroll[time_step, pitch, instr_idx] 
   mask_for_generation[time_step, :, instr_idx] 
   change_index uple([time_step, pitch, instr_idx])

   step utofillStep(
    prediction, (change_index, 1), generated_pianoroll.copy())
   autofill_steps.append(step)

  return generated_pianoroll, autofill_steps, original_pianoroll


def generate_gibbs_like(pianorolls, wrapped_model, config):
  model rapped_model.model

  ets shapes.
  batch_size, num_timesteps, num_pitches, num_instruments ianorolls.shape
  pianoroll_shape ianorolls[0].shape

  generated_pianoroll p.zeros(pianoroll_shape)
  generated_mask p.zeros(pianoroll_shape)
  autofill_steps ]

  timestep_indices ange(num_timesteps)
  pair_indices (instr_idx, time_idx)
      for instr_idx in config.voices_to_regenerate
      for time_idx in timestep_indices]
  print 'len of pair_indices', len(pair_indices)
  pair_indices_copy ]
  for n range(config.num_regenerations):
 pair_indices_copy.extend(copy.copy(pair_indices))

  pair_indices air_indices_copy
  print 'len of pair_indices after', len(pair_indices)
  n place shuffling.
  np.random.shuffle(pair_indices)
  counter 
  for instr_idx, time_step in pair_indices:
 counter += 1
 if counter 00 == 0:
   print 'taken steps', counter
 #print 'instr_idx, time_step', instr_idx, time_step
 assert generated_mask.ndim == 3
 # Reset mask.
 generated_mask[:, :, :] 
 # Turn on mask for the step being generated.
 generated_mask[time_step, :, instr_idx] 
 # Remove the note to be regenerated.
 generated_pianoroll[time_step, :, instr_idx] 

 # Stack all pieces to create atch.
 input_datas ]
 for data_index in range(batch_size):
   if data_index == config.requested_index:
  input_data ask_tools.apply_mask_and_stack(
   generated_pianoroll, generated_mask)
   else:
  mask ask_tools.get_random_instrument_mask(pianoroll_shape)
  input_data ask_tools.apply_mask_and_stack(
   pianorolls[data_index], mask)
   input_datas.append(input_data)
 input_datas p.asarray(input_datas)

 raw_prediction rapped_model.sess.run(
  model.predictions,
  {model.input_data: input_datas}
 )

 prediction aw_prediction[config.requested_index]
 pitch ample_pitch(prediction, time_step, instr_idx, num_pitches,
       config.temperature)

 generated_pianoroll[time_step, pitch, instr_idx] 

 change_index uple([time_step, pitch, instr_idx])
 step utofillStep(
  prediction, (change_index, 1), generated_pianoroll.copy())
 autofill_steps.append(step)
  return generated_pianoroll, autofill_steps, None


def generate_routine(config, output_path):
  prime_fpath onfig.prime_fpath
  requested_validation_piece_name onfig.requested_validation_piece_name

  hecking if there's inconsistency in the type of priming requested.
  if prime_fpath is not None and requested_validation_piece_name is not None:
 raise ValueError(
  'Either prime generation with melody or piece from validation set.')
  start_with_empty onfig.start_with_empty
  if start_with_empty and (prime_fpath is not None or
       equested_validation_piece_name is not None):
 raise ValueError(
  'Generate from empty initialization requested but prime given.')

  model_name onfig.model_name
  requested_index onfig.requested_index

  et data.
  seeder eed_tools.get_seeder(model_name)
  seeder.crop_piece_len onfig.requested_num_timesteps

  et prime and batch.
  if start_with_empty:
 pianorolls eeder.get_random_batch_with_empty_as_first()
 piece_name empty'
  elif prime_fpath is not None:
 pianorolls eeder.get_random_batch_with_midi_prime(
  prime_fpath, config.prime_duration_ratio)
 #piece_name magenta_theme'
 piece_name s.path.split(os.path.basename(prime_fpath))[0]

  elif requested_validation_piece_name is not None:
 requested_validation_piece_name 
  requested_validation_piece_name.replace('N_A', 'N/A'))
 pianorolls eeder.get_batch_with_piece_as_first(
  requested_validation_piece_name, 0)
 piece_name equested_validation_piece_name
  else:
 pianorolls, piece_names eeder.get_random_batch(return_names=True)
 piece_name s.path.split(piece_names[config.requested_index])[0]

  ue to initial naming of piece names with N/A and the forward slash interpreted as irectory.
  print 'Name:', piece_name
  piece_name iece_name.replace('N/A', 'N_A')

  et unique output path.
  timestamp_str onfig_tools.get_current_time_as_str()
  run_id %s-%s-%s' timestamp_str, model_name, piece_name)
  output_path s.path.join(output_path, run_id)
  if not os.path.exists(output_path):
 os.makedirs(output_path)

  et model.
  wrapped_model etrieve_model_tools.retrieve_model(
   model_name=model_name)

  enerate and synth output.
  generate_method_name onfig.generate_method_name
  for n range(config.num_samples):
 # Generate.
 run_local_id %s-%d' run_id, i)
 generated_results lobals()[generate_method_name](
  pianorolls, wrapped_model, config)
 generated_pianoroll, autofill_steps, original_pianoroll enerated_results

 # Synth original.
 if original_pianoroll is not None:
   original_seq eeder.encoder.decode(original_pianoroll)
   fpath s.path.join(
    output_path, 'original-%s-run_id_%s.midi' 
        enerate_method_name, run_local_id))
   sequence_proto_to_midi_file(original_seq, fpath)

 # Synth generated.
 # TODO(annahuang): Output sequence that merges prime and generated.
 generated_seq eeder.encoder.decode(generated_pianoroll)
 fpath s.path.join(
  output_path, 'generated-%s-run_id_%s.midi' 
   generate_method_name, run_local_id))
 sequence_proto_to_midi_file(generated_seq, fpath)

 if config.plot_process:
   plot_tools.plot_steps(autofill_steps, original_pianoroll)


def main(unused_argv):
  generate_routine(
   GENERATION_PRESETS['RegenerateValidationPieceVoiceByVoiceConfig'],
   FLAGS.output_dir)
  generate_routine(
   GENERATION_PRESETS['RegeneratePrimePieceVoiceByVoiceConfig'],
   FLAGS.output_dir)
  generate_routine(
   GENERATION_PRESETS['GenerateAccompanimentToPrimeMelodyConfig'],
   FLAGS.output_dir)
  generate_routine(
   GENERATION_PRESETS['GenerateGibbsLikeConfig'], FLAGS.output_dir)


class GenerationConfig(object):
  """Configurations for regenerating all voices voice by voice.

  Attributes:
 generate_method_name: tring that gives the name of the function used for
  generation.
 model_name: tring that gives the ...
  """
  _defaults ict(
  generate_method_name regenerate_voice_by_voice',
  model_name DeepResidual',
  prime_fpath one,
  prime_duration_ratio ,
  requested_validation_piece_name one,
  start_with_empty alse,

  equest index in batch.
  requested_index ,

  eneration parameters.
  voices_to_regenerate ange(4),
  sequential_order_type ANDOM,
  temperature ,

  num_samples ,
  requested_num_timesteps ,
  num_regenerations 0,

  plot_process alse)

  def __init__(self, *args, **init_hparams):
 unknown_params et(init_hparams) et(GenerationConfig._defaults)
 if unknown_params:
   raise ValueError("Unknown hyperparameters: %s", unknown_params)

 # Update instance with default class variables.
 for key, value in GenerationConfig._defaults.items():
   if key in init_hparams:
  value nit_hparams[key]
   setattr(self, key, value)


GENERATION_PRESETS 
 'RegenerateValidationPieceVoiceByVoiceConfig': GenerationConfig(
  generate_method_name regenerate_voice_by_voice',
  model_name DeepResidual',
  requested_validation_piece_name 139822010622096-bwv103.6.mxl-N_A',
  voices_to_regenerate ange(4),
  sequential_order_type ANDOM,
  num_samples ,
  requested_num_timesteps ,
  plot_process alse),
 'RegeneratePrimePieceVoiceByVoiceConfig': GenerationConfig(
  generate_method_name regenerate_voice_by_voice',
  model_name DeepResidual',
  prime_fpath LAGS.prime_fpath,
  voices_to_regenerate ange(4),
  sequential_order_type ANDOM,
  num_samples ,
  requested_num_timesteps ,
  plot_process alse),
 # Configuration for generating an accompaniment to prime melody.
 'GenerateAccompanimentToPrimeMelodyConfig': GenerationConfig(
  generate_method_name regenerate_voice_by_voice',
  model_name DeepResidual',
  prime_fpath LAGS.prime_fpath,
  prime_duration_ratio ,
  voices_to_regenerate ist(set(range(4)) et([MELODY_VOICE_INDEX])),
  sequential_order_type ANDOM,
  num_samples ,
  requested_num_timesteps ,
  plot_process alse),
 # Configurations for generating in random instrument cross timestep order.
 'GenerateGibbsLikeConfig': GenerationConfig(
  generate_method_name generate_gibbs_like',
  model_name DeepResidual',
  start_with_empty rue,
  voices_to_regenerate ange(4),
  sequential_order_type ANDOM,
  num_samples ,
  requested_num_timesteps ,
  num_regenerations ,
  plot_process alse)
}


if __name__ == '__main__':
  tf.app.run()
