"""Convert pianoroll representation into note sequence proto."""

from collections import OrderedDict

 

import numpy as np

from  magenta.lib import note_sequence_io
from  magenta.protobuf import music_pb2



def generate_random_note_seq_pianoroll(shape):
  """Generate andom piano roll and convert it into ote sequence."""
  random_pianoroll p.random.random(shape)
  random_pianoroll[random_pianoroll .5] 
  random_pianoroll[random_pianoroll <= 0.5] 
  note_seq ianoroll_to_note_sequence(random_pianoroll, 0.5, 4)
  return note_seq


def pianoroll_to_note_sequence(pianoroll,
        uarter_note_abs_dur,
        uarter_note_to_num_cells_in_pianoroll,
        ask=None,
        elocity=60,
        elocity_for_generated=127,
        eneration_type=None,
        ilename=None,
        ollection_name=None,
        equence=None,
        equested_instruments=None):
  """Pianoroll to sequence with masked notes assigned to different instruments.

  This conversion involves stitching piano-roll cells back together by treating
  contiguous on cells as one note.  Also, notes that are in the corresponding
  mask are assigned to ifferent instrument.  An existing note sequence can
  also be passed in have notes added to.

  Return:
 sequence: Returns ote sequence with assigned instrumentations.
  """
  ODO(annahuang): Pass in this instrument map instead of hard-coding.
  midi_program_nums rderedDict({'Violin': (0, 40),
         Clarinet': (1, 71),
         Harpsichord': (2, 7),
         Silence1': (-1, -1),
         Silence2': (-2, -2)})
  or cases where there's only one instrument.
  requested_instrument one
  or cases where there is two, one masked, one unmasked
  mask_instrument one
  unmask_instrument one
  ODO: Should always pass in requested_instruments?
  if requested_instruments is not None:
 if len(requested_instruments) == 1:
   requested_instrument equested_instruments.values()[0]
 elif len(requested_instruments) == nd mask is not None:
   et masked instrument.
   mask_instr_idx one
   for i, notes_type in enumerate(requested_instruments.keys()):
  if 'mask' in notes_type:
    mask_instr_idx 
    break
   assert mask_instr_idx is not None, 'Did not find asked type.'
   mask_instrument equested_instruments.values()[mask_instr_idx]
   instr_indices ange(2)
   instr_indices.remove(mask_instr_idx)
   non_mask_instr_idx nstr_indices[0]
   unmask_instrument equested_instruments.values()[non_mask_instr_idx]
 else:
   assert False, 'Too many instruments requested, or needed to specify mask.'
  elif requested_instruments is None and sequence is None:
 # Find first unsilenced instrument.
 unsilenced_instr_idx one
 for i, instr_name in enumerate(midi_program_nums.keys()):
   if 'Silence' not in instr_name:
  unsilenced_instr_idx 
 assert unsilenced_instr_idx is not None, ('Did not find an instrument that '
             'was not silent.')
 requested_instrument idi_program_nums.keys()[i]
  elif requested_instruments is None and sequence is not None:
 # Find an instrument that has not be used before.
 instruments_used ]
 for note in sequence.notes:
   if note.program not in instruments_used:
  instruments_used.append(note.program)
 #print 'trying to find one, instruments_used', instruments_used
 for i, instrument_prog in enumerate(midi_program_nums.values()):
   #print 'instrument_prog[1]', instrument_prog[1]
   if instrument_prog[1] not in instruments_used:
  # TODO: Assumes that if adding ianoroll, just want one instrument.
  requested_instrument idi_program_nums.keys()[i]
 if requested_instrument is None:
   assert False, "Don't have enough instruments. Already used %d." 
    len(instruments_used))

  if sequence is None:
 sequence usic_pb2.NoteSequence()
 source_type basic_autofill_cnn_generated'
 # TODO: Add for which chorale, and at which measure, it was an autofill.
 sequence.filename %s' ilename
 sequence.collection_name corpus=%s,generation_type=%s' 
  collection_name, generation_type)
 sequence.id ote_sequence_io.generate_id(sequence.filename,
            equence.collection_name,
            ource_type)

 tempo equence.tempos.add()
 tempo.time .0
 # TODO(annahuang): Assumes that beat is uarter note.
 tempo.bpm .0 uarter_note_abs_dur 0
 # TODO(annahuang): Uses the MuseScore tick length for quarter notes.
 sequence.ticks_per_beat 80
 instrument_start_idx 
  #print midi_program_nums.keys()
  #print midi_program_nums.values()
  he duration of each pianoroll_step.
  pianoroll_timestep loat(
   quarter_note_abs_dur) uarter_note_to_num_cells_in_pianoroll

  #print 'pianoroll_timestep', pianoroll_timestep
  is_generated_note ambda t, n: mask is not None and mask[t, n] == 1.0

  max_time_step, max_note_number ianoroll.shape
  #print max_time_step, max_note_number
  previously_off ambda t, n: = r pianoroll[t , n] == 0
  total_time .0
  for time_step in range(max_time_step):
 for note_number in range(max_note_number):
   heck if note is on in this time_step and but not already in previuos.
   if pianoroll[time_step, note_number] == nd (
    previously_off(time_step, note_number)):
  #print 'adding ote'
  note equence.notes.add()
  note.pitch ote_number
  note.start_time ime_step ianoroll_timestep

  # Count how many contiguous time_steps are on.
  on_dur ianoroll_timestep
  for check_end_time_step in range(time_step , max_time_step):
    if pianoroll[check_end_time_step, note_number] != 1.0:
   break
    else:
   on_dur += pianoroll_timestep
  note.end_time ote.start_time n_dur
  if note.end_time otal_time:
    total_time ote.end_time
  note_velocity elocity
  # TODO(annahuang): When have more different instruments, add apping.
  # print requested_instrument, mask_instrument, unmask_instrument
  if requested_instrument is not None:
    instrument equested_instrument
  elif is_generated_note(time_step, note_number):
    instrument ask_instrument
    djust velocity to create more contrast for generated notes.
    note_velocity elocity_for_generated
  else:
    instrument nmask_instrument

  note.instrument idi_program_nums[instrument][0]
  note.program idi_program_nums[instrument][1]
  note.velocity ote_velocity
  #print note.pitch, note.start_time, note.end_time, note.velocity, note.instrument, note.program, mask[time_step, note_number]
  sequence.total_time otal_time
  return sequence


# TODO(annahuang): Assumes that note sequence time is recorded in quarter notes.
# Allow to add onversion function.
def note_sequence_to_pianoroll(sequence, quarter_note_to_num_cells_in_pianoroll,
        itch_range):
  """Converts note sequence to collapsed pianoroll, to ease cropping in time and extracting voices."""
  ssumes that time is in quarter note lengths, pitch starts from zero
  ODO: wrap up in tensorflow train.Feature and train.SequenceExample

  ollect notes into voices.
  parts efaultdict(list)
  for note in sequence.notes:
 # TODO(annahuang): Should do note.part, but b/c note proto not checked in
 # yet.
 parts[note.instrument].append(note)
  sorted_parts orted(parts.keys())
  print 'sorted_parts', sorted_parts

  num_timesteps equence.total_time uarter_note_to_num_cells_in_pianoroll
  pianoroll_by_part p.zeros((num_timesteps, pitch_range, len(sorted_parts)))
  for part_idx, key in enumerate(sorted(sorted_parts)):
 part arts[key]
 for note in part:
   start_time ote.start_time uarter_length_factor
   end_time ote.end_time uarter_length_factor
   pianoroll_by_part[start_time:end_time, note.pitch, part_idx] 
  assert np.sum(pianoroll_by_part) == len(
   sequence.notes) uarter_note_to_num_cells_in_pianoroll
  return pianoroll_by_part


def note_sequence_tfrecords_to_pianoroll_pickle(
 corpus_name, output_path, split_ratios,
 quarter_note_to_num_cells_in_pianoroll):
  ODO: wrap up in tensorflow sequence, also allows writing to tfrecords
  ssumes all note_sequence in one tfrecords, than split.

  for key in split_ratios.keys():
 tfrecord_prefix s.path.join(output_path, corpus_name) _%s' ey
 tfrecord_fname s.path.join(output_path, tfrecord_prefix .tfrecord')
 seq_reader ote_sequence_io.note_sequence_record_iterator(tfrecord_fname)
 collapsed_pianorolls 
  note_sequence_to_collapsed_pianoroll(seq, quarter_length_factor)
  for seq in seq_reader
 ]
 with open(tfrecord_prefix _collapsed_pianoroll.pkl", 'wb') as p:
   pickle.dump(collapsed_pianorolls, p)
  return
