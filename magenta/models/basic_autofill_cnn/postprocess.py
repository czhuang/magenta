import cPickle as pickle
import os
from collections import defaultdict

import numpy as np

from magenta.protobuf import music_pb2
from magenta.lib.midi_io import sequence_proto_to_midi_file
from magenta.lib.note_sequence_io import NoteSequenceRecordWriter

from magenta.models.basic_autofill_cnn.generate_tools import AutofillStep
from magenta.models.basic_autofill_cnn.plot_tools import plot_steps
from magenta.models.basic_autofill_cnn import pianorolls_lib


def retrieve_pickle(fpath):
  with open(fpath, 'rb') as p:
    results = pickle.load(p)
  return results


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
    

def plot_process():
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
  last_pianoroll = plot_steps(steps, original_pianoroll, plot_output_path, seq_key, 
      subplot_step_indices=subplot_step_indices, subplots=subplots)
  
  # Synth last pianoroll.  
  generated_seq = encoder.decode(last_pianoroll)
  print 'last_pianoroll', np.sum(last_pianoroll), last_pianoroll.shape
  print 'generated_seq', generated_seq.total_time, len(generated_seq.notes)
  
  midi_fpath = os.path.join(path, 'plots', 'z_last_step_%s.midi' % seq_key) 
  sequence_proto_to_midi_file(generated_seq, midi_fpath)

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


if __name__ == '__main__':
  #concatenate_generated_seqs()
  #plot_process()
  save_last_prediction_as_tfrecord()
