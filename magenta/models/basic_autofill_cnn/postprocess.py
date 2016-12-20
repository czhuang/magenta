import cPickle as pickle
import os
from collections import defaultdict

import tensorflow as tf
import numpy as np

from magenta.protobuf import music_pb2
from magenta.music.midi_io import sequence_proto_to_midi_file
from magenta.music.note_sequence_io import NoteSequenceRecordWriter

from magenta.models.basic_autofill_cnn.generate_tools import AutofillStep
from magenta.models.basic_autofill_cnn.plot_tools import plot_steps
from magenta.models.basic_autofill_cnn import pianorolls_lib


def retrieve_pickle(fpath):
  print '\nLoading pickle:', fpath
  with open(fpath, 'rb') as p:
    results = pickle.load(p)
  return results


def retrieve_generation_bundle(fpath):
  results = retrieve_pickle(fpath)
  requested_index = 0
  print results.keys(), type(results[None]), requested_index, type(results[None][requested_index])
  seq_bundle = results[None][requested_index]
  generated_seq, steps, original_seq, _ = seq_bundle 
  return generated_seq, steps, original_seq


def get_fpath():
  fpath = '/Tmp/huangche/new_generation/2016-11-10_17:50:25-balanced_by_scaling/0_regenerate_voice_by_voice-0-0.05min-2016-11-10_17:50:25-balanced_by_scaling-0-empty-3_2_0_1.npz'
  return fpath


def get_fpath_wrapper(fname_tag='', file_type='png'):
  source_fpath = get_fpath()
  dirname = os.path.dirname(source_fpath)
  fname = os.path.basename(source_fpath).split('.')[0]
  fpath = os.path.join(dirname, 
                       '%s%s.%s' % (fname_tag, fname, file_type))
  return fpath


def retrieve_process_npz(fpath):
  print '\nLoading NPZ:', fpath
  with open(fpath, 'rb') as p:
    np_dict = np.load(p)
    generated_pianorolls = np_dict['generated_pianorolls']
    predictions = np_dict['predictions']
    step_indices = np_dict['step_indices']
    original_pianoroll = np_dict['original_pianoroll']
  return predictions, step_indices, generated_pianorolls, original_pianoroll


def check_retrieve_process_npz():
  bundle = retrieve_process_npz(get_fpath())
  for item in bundle:
    print item.shape


def plot_process():
  fpath = get_fpath()
  path = os.path.dirname(fpath)
  steps_bundle = retrieve_process_npz(fpath)
  #predictions, step_indices, generated_pianorolls, original_pianoroll = steps_bundle
  #num_timesteps, num_pitches, num_instruments = prediction_shape[1:]

  plot_output_path = os.path.join(path, 'plots')
  last_pianoroll, intermediate_seqs = plot_steps(steps_bundle, plot_output_path, '',
      subplot_step_indices=None, subplots=False)
  
  # Synth last pianoroll.  
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  generated_seq = encoder.decode(last_pianoroll)
  print 'last_pianoroll', np.sum(last_pianoroll), last_pianoroll.shape
  print 'generated_seq', generated_seq.total_time, len(generated_seq.notes)
  
  midi_fpath = os.path.join(path, 'plots', 'z_last_step_%s.midi' % seq_key) 
  sequence_proto_to_midi_file(generated_seq, midi_fpath)
  return last_pianoroll, intermediate_seqs


def concatenate_seqs(seqs, gap_in_seconds=1.5):
  time_offset = 0
  concatenated_seq = music_pb2.NoteSequence()
  concatenated_seq.ticks_per_quarter = 220
  for i, seq in enumerate(seqs):
    if i:
      for note in seq.notes:
        note.start_time += time_offset
        note.end_time += time_offset
        concatenated_seq.notes.extend([note])
    else:
      concatenated_seq.notes.extend(seq.notes)
    time_offset += seq.total_time + gap_in_seconds 
  return concatenated_seq


def concatenate_process():
  path = '/data/lisatmp4/huangche/new_generated/2016-10-20_00:25:18-DeepResidualRandomMask'
  fname = '0_generate_gibbs_like-0-4.49min-2016-10-20_00:25:18-DeepResidualRandomMask-0--None.pkl' 
  #path = '/u/huangche/generated/2016-10-16_23:45:29-DeepResidual'
  #fname = '0_generate_gibbs_like-0-2016-10-16_23:45:29-DeepResidual-0--None.pkl'

  # # of steps: 38958 ~ 40iter *4instr *(64time_steps*4instrs) = 40*4*8 gibbs step = 1280 steps
  path = '/Tmp/huangche/generation/best_fromScratch-64-2016-10-26_01:00:30-DeepResidual'  
  fname = '0_generate_gibbs_like-0-285.03min-2016-10-26_01:00:30-DeepResidual-0-empty-None.pkl'
  output_path = path

  path = '/Tmp/cooijmat/autofill/generate/2016-10-30_13:49:29-balanced'
  fname = '0_generate_gibbs_like-0-253.44min-2016-10-30_13:49:29-balanced-0-empty-None.pkl'
  output_path = '/data/lisatmp4/huangche/TBF_generated'


  run_id = fname.split('.pkl')[0]
  requested_index = int(fname.split('_')[0])  
  print 'retrieving pickle...'
  results = retrieve_pickle(os.path.join(path, fname))
  print results.keys, type(results[None]), type(results[None][requested_index])
  seq_bundle = results[None][requested_index]
  generated_seq, steps, original_seq, _ = seq_bundle 
  print 'type(steps)', type(steps)
  num_steps = len(steps)
  print '# of steps:', num_steps
  pianoroll_shape = steps[0].prediction.shape  

  plot_indices = []
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  if 'empty' in fname:
    # TODO: encoder.encode not able to take (0, 53, 0)
    original_pianoroll = np.zeros(pianoroll_shape)
  else:
    original_pianoroll = encoder.encode(original_seq)
  last_seq, intermediate_seqs = plot_steps(
      steps, original_pianoroll, output_path, run_id, subplots=True, subplot_step_indices=plot_indices)
  
  inspect_seqs = []
  inspect_crop_len = 64 
  
  synth_num_steps = 15 
  synth_indices = range(0, num_steps, num_steps // synth_num_steps)
#  synth_interval = 16
#  synth_indices = range(0, num_steps, synth_interval) 

  print '# of intermediate seqs:', len(intermediate_seqs), '# of steps:', num_steps
  for i in synth_indices:
    pianoroll = encoder.encode(intermediate_seqs[i])
    print 'pianoroll.shape', pianoroll.shape, inspect_crop_len
    # TODO: not sure why when pianoroll is supposed to be 16 when uncropped but would be 11.
    #assert pianoroll.shape[0] >= inspect_crop_len
    seq = encoder.decode(pianoroll[:inspect_crop_len, :, :])
    inspect_seqs.append(seq)
  concated_seqs = concatenate_seqs(inspect_seqs)
  output_fname = 'zaa_%s_process_concat.midi' % run_id
  sequence_proto_to_midi_file(concated_seqs, os.path.join(output_path, output_fname))
  
  # Synth each intermediate seq separately
  synth_seqs(inspect_seqs, output_path, run_id)


def synth_seqs(seqs, path, run_id):
  for i, seq in enumerate(seqs):
    output_fname = 'z_summary_%s_synthed_step_%d.midi' % (run_id, i)
    sequence_proto_to_midi_file(seq, os.path.join(path, output_fname))


def concatenate_generated_seqs():
  path = '/u/huangche/generated/2016-09-30_15:57:51-DeepResidual'
  fname = 'regenerate_voice_by_voice-2016-09-30_15:57:51-DeepResidual-1--1.pkl'
  path = '/u/huangche/generated/2016-09-30_17:36:28-DeepResidual'
  fname = 'regenerate_voice_by_voice-2016-09-30_17:36:28-DeepResidual-0--47.pkl'

  fpath = os.path.join(path, fname)
  results = retrieve_pickle(fpath)
  for key, seq_bundles in results.iteritems():
    generated_seqs = []
    for seq_bundle in seq_bundles:
      generated_seqs.append(seq_bundle[0])
    print key, len(generated_seqs)    
    output_fname = 'z_%s_concatenated_%s.midi' % (key, os.path.splitext(fname)[0])
    output_fpath = os.path.join(path, output_fname)
    sequence_proto_to_midi_file(concatenate_seqs(generated_seqs), output_fpath)
    

def get_seq_bundle():
  # seq_key = None, results[seq_key][0] # first one is really nice, hence ending and interesting theme 
  path = '/u/huangche/generated/nice/2016-10-05_17:13:46-DeepResidual'
  fname = 'generate_gibbs_like-None-2016-10-05_17:13:46-DeepResidual-0-empty-4.pkl'
  seq_key = None
  requested_index = 0

  # [4] # fifth one, but the ending and beginning is not as nice.
  path = '/u/huangche/generated/nice/2016-10-05_20:52:02-DeepResidual'
  fname = 'regenerate_voice_by_voice-0_3_1_2-2016-10-05_20:52:02-DeepResidual-0-empty-4.pkl'
  seq_key = '0_3_1_2'
  requested_index = 0
  subplot_step_indices = [2, 33, 34, 47, 48, 51]
 
  # [4], also last one, great harmony, except the 'flare note' was too short.
  #path = '/u/huangche/generated/nice/2016-10-05_20:56:29-DeepResidual'
  #fname = 'regenerate_voice_by_voice-0_3_1_2-2016-10-05_20:56:29-DeepResidual-0-empty-4.pkl'
  #seq_key = '0_3_1_2'
  #requested_index = 0

  #path = '/u/huangche/generated/2016-10-04_17:58:01-DeepResidual'
  #fname = 'regenerate_voice_by_voice-3_1_2-2016-10-04_17:58:01-DeepResidual-0--4.pkl'
  #seq_key = '1_3_2'

  # With temperature 0.1
  path = '/u/huangche/generated/useful/2016-10-06_17:56:31-DeepResidual' 
  fname = 'regenerate_voice_by_voice-1_2_3_0-2016-10-06_17:56:31-DeepResidual-0-empty-9.pkl'
  seq_key = '1_0_2_3'
  requested_index = 0
  
  seq_key = '1_2_3_0'
  requested_index = 0
  # 16: switch to 2nd voice, see a lot of vertical structure.
  # 17: collapses to more melodic constraints.
  # 20: Had the chance of suspension but went for the B.
  # 29: went for suspension but not resolving down, bot voices moving up.
  # 32: 3rd voice comes in.
  # 37: has the chance to be a em or G, and decides to be an em.
  # 
  subplot_step_indices = [0, 1, 16, 17, 20, 29, 32, 37]#, 63]
 
  #subplot_step_indices = None 
  subplots = True
  #subplots = False

  #path = '/u/huangche/generated/2016-10-06_22:16:54-DeepResidual'
  #fname = 'regenerate_voice_by_voice-1_2_3_0-2016-10-06_22:16:54-DeepResidual-0-empty-9.pkl'
  #seq_key = '1_2_3_0'
  #requested_index = 0

  fpath = os.path.join(path, fname)
  results = retrieve_pickle(fpath)
  # plot one sequence first.
  print 'keys', results.keys()
  print 'num sequences available', len(results[seq_key])
  seq_bundle = results[seq_key][requested_index]  # [0]
  return seq_bundle, subplot_step_indices, path


def plot_process_main_for_pickle():
  seq_bundle, subplot_step_indices, path = get_seq_bundle()
  plot_process_for_pickle(seq_bundle, subplot_step_indices, path)


def plot_process_for_pickle(seq_bundle, subplot_step_indices, path):
  steps = seq_bundle[1]
  original_seq = seq_bundle[2]
  prediction_shape = steps[0].prediction.shape
  num_timesteps, num_pitches, num_instruments = prediction_shape
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  # TODO: Hack, should make a function in generator_tools for instantiating a new sequence, so that it has all the new settings such as ticks and source_type
  
  print 'original_seq, # of notes', len(original_seq.notes)
  if original_seq is not None and len(original_seq.notes) != 0:
    original_seq.source_info.source_type = 1
    # TODO: Assumes generation crop started from the beginning.
    original_pianoroll = encoder.encode(original_seq)[:num_timesteps]
  else:
    original_pianoroll = np.zeros(prediction_shape)
  print 'original_pianoroll', original_pianoroll.shape  
  plot_output_path = os.path.join(path, 'plots')
  last_pianoroll, intermediate_seqs = plot_steps(steps, original_pianoroll, plot_output_path, seq_key, 
      subplot_step_indices=subplot_step_indices, subplots=subplots)
  
  # Synth last pianoroll.  
  generated_seq = encoder.decode(last_pianoroll)
  print 'last_pianoroll', np.sum(last_pianoroll), last_pianoroll.shape
  print 'generated_seq', generated_seq.total_time, len(generated_seq.notes)
  
  midi_fpath = os.path.join(path, 'plots', 'z_last_step_%s.midi' % seq_key) 
  sequence_proto_to_midi_file(generated_seq, midi_fpath)
  return last_pianoroll, intermediate_seqs


def save_last_prediction_as_tfrecord():
  # The one used for plots.
  path = '/u/huangche/generated/useful/2016-10-06_17:56:31-DeepResidual' 
  fname = 'regenerate_voice_by_voice-1_2_3_0-2016-10-06_17:56:31-DeepResidual-0-empty-9.pkl'
  seq_key = '1_2_3_0'
  requested_index = 0
 
  # generated-regenerate_voice_by_voice-run_id_0_3_2_1-2016-10-12_01\:15\:16-DeepResidual-0--5.midi 
  path = '/u/huangche/generated/useful/2016-10-12_01:15:16-DeepResidual'
  fname = 'regenerate_voice_by_voice-3_0_2_1-2016-10-12_01:15:16-DeepResidual-0--49.pkl'
  seq_key = '0_3_2_1'
  requested_index = 2
 
  fpath = os.path.join(path, fname)
  results = retrieve_pickle(fpath)
  seq_bundle = results[seq_key][requested_index]  # [0]
  steps = seq_bundle[1]
  prediction_shape = steps[0].prediction.shape
  num_timesteps, num_pitches, num_instruments = prediction_shape
  
  # Get original pianoroll
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  original_seq = seq_bundle[2]
  if original_seq is not None and len(original_seq.notes) != 0:
    original_seq.source_info.source_type = 1
    # TODO: Assumes generation crop started from the beginning.
    original_pianoroll = encoder.encode(original_seq)[:num_timesteps]
  else:
    prediction_shape = steps[0].prediction.shape
    original_pianoroll = np.zeros(prediction_shape)
  print 'original_pianoroll', original_pianoroll.shape  

  # Just for building the final pianoroll from steps, not to print.
  plot_output_path = os.path.join(path)
  last_pianoroll = plot_steps(steps, original_pianoroll, plot_output_path, seq_key, 
      subplot_step_indices=[37, len(steps)-1], subplots=True)
  print 'last_pianoroll', np.sum(last_pianoroll), last_pianoroll.shape
  
  generated_seq = encoder.decode(last_pianoroll)
  print 'generated_seq', generated_seq.total_time, len(generated_seq.notes)
  
  # Synth last pianoroll.  
  midi_fpath = os.path.join(path,  'za_last_step_%s.midi' % seq_key) 
  sequence_proto_to_midi_file(generated_seq, midi_fpath)
  
  # Save as tfrecord
  tfrecord_fpath = os.path.join(path, 'za_last_step_%s.tfrecord' % seq_key)
  writer = NoteSequenceRecordWriter(tfrecord_fpath)
  writer.write(generated_seq)

def main(unused_args):
  #concatenate_generated_seqs()
  #plot_process()
  #save_last_prediction_as_tfrecord()
  try:
    plot_process()
    #check_retrieve_process_npz()
    #concatenate_process()
  except:
    import sys
    rahh = sys.exc_info()
    print '\nError msg:', repr(rahh[1])
    import pdb; pdb.post_mortem()
 # concatenate_process()

if __name__ == '__main__':
  tf.app.run()

