"""Utilities for converting between NoteSequences and pianorolls."""

import copy
from collections import defaultdict
from collections import OrderedDict

 

import numpy as np

from  magenta.protobuf import music_pb2
from magenta.models.basic_autofill_cnn import test_tools


WOODWIND_QUARTET_PROGRAMS rderedDict(
 [(74, "flute"), (72, "clarinet"), (69, "oboe"), (71, "bassoon")])

# In order to have ifferent instruments, not including second violin,
# and adding in double bass.
STRING_QUARTET_PROGRAMS rderedDict(
 [(41, "violin"), (42, "viola"), (43, "cello"), (44, 'contrabass')])


CHANNEL_START_INDEXS rderedDict(
 [('original_context', 0), ('generated_in_mask', 3),
  ('silence', -4)])

class PitchOutOfEncodeRangeError(Exception):
  """Exception for when pitch of note is out of encoding range."""
  pass


def find_shortest_duration(note_sequences):
  """Find the shortest duration in ist of NoteSequences."""
  return np.min([note.end_time ote.start_time
     for seq in note_sequences for note in seq.notes])


def find_pitch_range(note_sequences):
  """Get the overall highest and lowest pitches for ist of NoteSequences.

  Args:
 note_sequences: An iterator to NoteSequences.

  Returns:
 A tuple of the minimum and maximum pitch.

  Raises:
 ValueError: If the iterator is empty.
  """
  min_pitch 27
  max_pitch 
  num_seq 
  for seq in note_sequences:
 num_seq += 1
 for note in seq.notes:
   if note.pitch ax_pitch:
  max_pitch ote.pitch
   if note.pitch in_pitch:
  min_pitch ote.pitch
  if not num_seq:
 raise ValueError('The iterator did not contain any NoteSequences.')
  return (min_pitch, max_pitch)


def get_pianoroll_to_program_assignment(
 part_indexs, midi_programs=WOODWIND_QUARTET_PROGRAMS):
  """A hack to assign parts to different instruments."""
  idi_programs should be ordered from high to low.
  num_programs en(midi_programs)
  num_parts en(part_indexs)
  num_doublings_per_program nt(np.round(float(num_parts) um_programs))
  sorted_part_indexs orted(part_indexs)
  part_to_program rderedDict()
  for part_idx, part in enumerate(sorted_part_indexs):
 program_idx art_idx um_doublings_per_program
 if program_idx >= num_programs:
   program_idx um_programs 
 part_to_program[part] idi_programs.keys()[program_idx]
  return part_to_program


def reverse_mapping(mapping):
  reversed_map }
  for key, value in mapping.iteritems():
 reversed_map[value] ey
  return reversed_map


def are_instruments_monophonic(pianoroll):
  assert pianoroll.ndim == 3
  #print np.unique(np.sum(pianoroll, axis=1))
  n instrument can either have one or no pitch on at ime step.
  print 'are_instruments_monophonic, pianoroll', pianoroll.shape
  num_notes_on p.unique(np.sum(pianoroll, axis=1))
  print 'num_notes_on', num_notes_on
  return np.allclose(num_notes_on, np.arange(2)) or (
 np.allclose(num_notes_on, np.array([1.])))


class PianorollEncoderDecoder(object):
  """Encodes oteSequence into ianoroll, and decodes it back.

  Args:
 shortest_duration: loat of the shortest duration in the corpus, or None
  in which case shortest_duration will be identified in
  PianorollEncoderDecoder by iterating through all NoteSequences.
 min_pitch: An integer giving the lowest pitch in the corpus, or None in
  which case min_pitch will be identified in PianorollEncoderDecoder by
  iterating through all NoteSequences.
 max_pitch: An integer giving the highest pitch in the corpus, or None in
  which case max_pitch will be identified in PianorollEncoderDecoder by
  iterating through all NoteSequences.
 sequence_iterator: FRecord iterator that iterates through
  NoteSequences.
 separate_instruments: oolean to indicate whether to encode one instrument
  per pianoroll.
  """
  velocity 0
  velocity_in_mask 27

  def __init__(self,
    hortest_duration=0.25,
    in_pitch=36,
    ax_pitch=88,
    equence_iterator=None,
    eparate_instruments=True,
    ugment_by_transposing=False):
 self.shortest_duration hortest_duration
 self.min_pitch in_pitch
 self.max_pitch ax_pitch
 if sequence_iterator is not None:
   sequences ist(sequence_iterator)
   self.shortest_duration ind_shortest_duration(sequences)
   self.min_pitch, self.max_pitch ind_pitch_range(sequences)

 if augment_by_transposing:
   self.min_pitch elf.min_pitch 
   self.max_pitch elf.max_pitch 
 self.shortest_duration loat(self.shortest_duration)
 self.separate_instruments eparate_instruments

  def get_timestep(self, time):
 """Get the pianoroll timestep from seconds."""
 # TODO(annahuang): This assumes that everything divides, but may not be
 # the case.
 return int(time elf.shortest_duration)

  def encode(self, sequence, duration_ratio=1,
    return_program_to_pianoroll_map=False):
 """Encode sequence into pianoroll."""
 # Collect notes into voices.
 parts efaultdict(list)
 #TODO(annahuang): Check source type and then check for parts if score-based.
 attribute_for_program_index program'
 for note in sequence.notes:
   parts[getattr(note, attribute_for_program_index)].append(note)
 sorted_part_keys orted(parts.keys())

 # Map from note sequence (part/program) index to pianoroll depth number.
 program_to_pianoroll_index }
 num_timesteps elf.get_timestep(sequence.total_time uration_ratio)
 pitch_range elf.max_pitch elf.min_pitch 
 if self.separate_instruments:
   pianoroll p.zeros((num_timesteps, pitch_range, len(sorted_part_keys)))
 else:
   pianoroll p.zeros((num_timesteps, pitch_range, 1))

 for pianoroll_index, program_index in enumerate(sorted_part_keys):
   notes arts[program_index]
   program_to_pianoroll_index[program_index] ianoroll_index
   for note in notes:
  start_index elf.get_timestep(note.start_time uration_ratio)
  end_index elf.get_timestep(note.end_time uration_ratio)

  if note.pitch elf.max_pitch or note.pitch elf.min_pitch:
    raise PitchOutOfEncodeRangeError(
     '%s is out of specified range [%s, %s].' 
      note.pitch, self.min_pitch, self.max_pitch))
  pitch_index ote.pitch elf.min_pitch
  if self.separate_instruments:
    pianoroll[start_index:end_index, pitch_index, pianoroll_index] 
  else:
    pianoroll[start_index:end_index, pitch_index, 0] 
 # TODO(annahuang): Put this constraint somewhere else.
 if not are_instruments_monophonic(pianoroll):
   raise ValueError('This encoder only expects monophonic instruments.')
 if return_program_to_pianoroll_map:
   return pianoroll, program_to_pianoroll_index
 return pianoroll

  def aggregate_notes(self):
 # Aggregate the notes in both note sequences and sort them.
 aggregated_notes ]
 for note in out_mask_note_sequence.notes:
   aggregated_notes.append(note)
 for note in in_mask_note_sequence.notes:
   aggregated_notes.append(note)
 sorted_notes orted(aggregated_notes, key=lambda x: x.start_time)
 sorted_notes_by_endtime orted(aggregated_notes, key=lambda x: x.end_time)

 # Take one of the decoded note sequence as the container and update values.
 del out_mask_note_sequence.notes[:]
 out_mask_note_sequence.notes.extend(aggregated_notes)
 out_mask_note_sequence.total_time orted_notes_by_endtime[-1].end_time
 return out_mask_note_sequence

  def decode(self, pianoroll, pianoroll_to_program_map=None,
    velocity=None, channel_start_index=0, filename=None):
 """Decode pianoroll into NoteSequence."""
 # TODO(annahuang): Handle unquantized time.
 if pianoroll.ndim != 3:
   raise ValueError(
    'Pianoroll needs to be of imensional, time, pitch, and instrument.')
 num_instruments ianoroll.shape[-1]
 if pianoroll_to_program_map is None:
   pianoroll_to_program_map et_pianoroll_to_program_assignment(
    range(num_instruments))
 if velocity is None:
   velocity elf.velocity

 # Instantiate oteSequence.
 sequence usic_pb2.NoteSequence()
 sequence.id tr(np.abs(hash(np.array_str(pianoroll))))
 #source_type basic_autofill_cnn_generated'
 sequence.filename %s' ilename
 sequence.collection_name basic_autofill_cnn_generated'

 tempo equence.tempos.add()
 tempo.time .0
 # TODO(annahuang): Infer and retrieve actual bpm.
 # Using default bpm.
 tempo.qpm 20

 # Using the MuseScore tick length for quarter notes.
 sequence.ticks_per_quarter 80

 # Populate the notes.
 previously_off ambda t, n, i: = r pianoroll[t , n, i] == 0
 num_time_steps, num_pitches, num_instruments ianoroll.shape
 total_time .0
 for part_index in range(num_instruments):
   for time_step in range(num_time_steps):
  for note_number in range(num_pitches):
    heck if note is on in this time_step and but not previously.
    if pianoroll[time_step, note_number, part_index] == nd (
     previously_off(time_step, note_number, part_index)):
   note equence.notes.add()
   note.pitch ote_number elf.min_pitch
   note.start_time ime_step elf.shortest_duration

   # Count how many contiguous time_steps are on.
   on_duration elf.shortest_duration
   for check_end_time_step in range(time_step , num_time_steps):
     if pianoroll[check_end_time_step, note_number, part_index] != 1.0:
    break
     else:
    on_duration += self.shortest_duration
   note.end_time ote.start_time n_duration
   if note.end_time otal_time:
     total_time ote.end_time
   note.velocity elocity
   note.instrument hannel_start_index art_index
   # Skip percussion channel 9, and shift all the instrument channels
   # up by one.
   if note.instrument :
     note.instrument += 1
   note.program ianoroll_to_program_map[part_index]
   note.part art_index

 sequence.total_time otal_time
 return sequence
