"""Utilities for converting between NoteSequences and pianorolls."""

import copy
from collections import defaultdict
from collections import OrderedDict

import numpy as np

from magenta.protobuf import music_pb2
import test_tools

OLD_SETTINGS = True

if OLD_SETTINGS:
  #WOODWIND_QUARTET_PROGRAMS = OrderedDict(
  #    [(73, 'flute'), (71, 'clarinet'), (68, 'oboe'), (70, 'bassoon')])
  WOODWIND_QUARTET_PROGRAMS = OrderedDict(
      [(75, 'flute'), (73, 'clarinet'), (70, 'oboe'), (72, 'bassoon')])
else:
  WOODWIND_QUARTET_PROGRAMS = OrderedDict(
      [(74, 'flute'), (72, 'clarinet'), (69, 'oboe'), (71, 'bassoon')])

# ok this is better with timidity
WOODWIND_QUARTET_PROGRAMS = OrderedDict(
    [(69, 'oboe'), (70, 'english_horn'), (72, 'clarinet'), (71, 'bassoon')])

# In order to have 4 different instruments, not including second violin,
# and adding in double bass.
#STRING_QUARTET_PROGRAMS = OrderedDict(
#    [(41, 'violin'), (42, 'viola'), (43, 'cello'), (44, 'contrabass')])
STRING_QUARTET_PROGRAMS = OrderedDict(
    [(41, 'violin'), (41, 'violin'), (42, 'viola'), (43, 'cello')])
#STRING_QUARTET_PROGRAMS = OrderedDict(
#    [(43, 'violin'), (43, 'violin'), (42, 'viola'), (41, 'cello')])

CHANNEL_START_INDEXS = OrderedDict([('original_context', 0),
                                    ('generated_in_mask', 3), ('silence', -4)])


SYNTH_MODE = False
if SYNTH_MODE:
  _DEFAULT_QPM = 60 
else:
  _DEFAULT_QPM = 60
  _DEFAULT_QPM = 120

class PitchOutOfEncodeRangeError(Exception):
  """Exception for when pitch of note is out of encodings range."""
  pass


def make_note_sequence(fname='', collection_name=''):
  # Instantiate a NoteSequence.
  sequence = music_pb2.NoteSequence()
  # TODO: A hack.
  sequence.id = str(np.abs(hash(np.array_str(np.random.random((5,5))))))
  sequence.filename = '%s' % fname
  sequence.collection_name = collection_name
#  sequence.source_info.source_type = source_type

  tempo = sequence.tempos.add()
  tempo.time = 0.0
  tempo.qpm = _DEFAULT_QPM

  # Using the MuseScore tick length for quarter notes.
  sequence.ticks_per_quarter = 480
  return sequence


def find_shortest_duration(note_sequences):
  """Find the shortest duration in a list of NoteSequences."""
  return np.min([note.end_time - note.start_time
                 for seq in note_sequences for note in seq.notes])


def find_pitch_range(note_sequences):
  """Get the overall highest and lowest pitches for a list of NoteSequences.

  Args:
    note_sequences: An iterator to NoteSequences.

  Returns:
    A tuple of the minimum and maximum pitch.

  Raises:
    ValueError: If the iterator is empty.
  """
  min_pitch = 127
  max_pitch = 0
  num_seq = 0
  for seq in note_sequences:
    num_seq += 1
    for note in seq.notes:
      if note.pitch > max_pitch:
        max_pitch = note.pitch
      if note.pitch < min_pitch:
        min_pitch = note.pitch
  if not num_seq:
    raise ValueError('The iterator did not contain any NoteSequences.')
  return (min_pitch, max_pitch)


def get_pianoroll_to_program_assignment(
    part_indexs, midi_programs=WOODWIND_QUARTET_PROGRAMS):
  """A hack to assign parts to different instruments."""
  # midi_programs should be ordered from high to low.
  num_programs = len(midi_programs)
  num_parts = len(part_indexs)
  num_doublings_per_program = int(np.round(float(num_parts) / num_programs))
  if num_doublings_per_program == 0:
    num_doublings_per_program = 1
  sorted_part_indexs = sorted(part_indexs)
  part_to_program = OrderedDict()
  for part_idx, part in enumerate(sorted_part_indexs):
    print part_idx, num_doublings_per_program
    program_idx = part_idx / num_doublings_per_program
    if program_idx >= num_programs:
      program_idx = num_programs - 1
    # Subtract 1 because MIDI program numbers are 1 based.
    part_to_program[part] = midi_programs.keys()[program_idx] - 1
  return part_to_program


def reverse_mapping(mapping):
  reversed_map = {}
  for key, value in mapping.iteritems():
    reversed_map[value] = key
  return reversed_map


def are_instruments_monophonic(pianoroll):
  assert pianoroll.ndim == 3
  #print np.unique(np.sum(pianoroll, axis=1))
  # An instrument can either have one or no pitch on at a time step.
  #print 'are_instruments_monophonic, pianoroll', pianoroll.shape
  num_notes_on = np.unique(np.sum(pianoroll, axis=1))
  #print 'pianoroll shape', pianoroll.shape
  #print 'num_notes_on', num_notes_on
  return np.allclose(num_notes_on, np.arange(2)) or (
      np.allclose(num_notes_on, np.array([1.])))


class PianorollEncoderDecoder(object):
  """Encodes a NoteSequence into a pianoroll, and decodes it back.

  Args:
    shortest_duration: A float of the shortest duration in the corpus, or None
        in which case shortest_duration will be identified in
        PianorollEncoderDecoder by iterating through all NoteSequences.
    min_pitch: An integer giving the lowest pitch in the corpus, or None in
        which case min_pitch will be identified in PianorollEncoderDecoder by
        iterating through all NoteSequences.
    max_pitch: An integer giving the highest pitch in the corpus, or None in
        which case max_pitch will be identified in PianorollEncoderDecoder by
        iterating through all NoteSequences.
    sequence_iterator: A TFRecord iterator that iterates through
        NoteSequences.
    separate_instruments: A boolean to indicate whether to encode one instrument
        per pianoroll.
  """
  velocity = 85
  velocity_in_mask = 127

  def __init__(self,
               shortest_duration=0.125,
               min_pitch=36,
               max_pitch=88,
               sequence_iterator=None,
               separate_instruments=True,
               augment_by_transposing=False,
               num_instruments=None,
               encode_silences=None,
               quantization_level=None):
    assert num_instruments is not None
    assert encode_silences is not None
    self.shortest_duration = shortest_duration
    self.min_pitch = min_pitch
    self.max_pitch = max_pitch

    if sequence_iterator is not None:
      sequences = list(sequence_iterator)
      self.shortest_duration = find_shortest_duration(sequences)
      self.min_pitch, self.max_pitch = find_pitch_range(sequences)

    if augment_by_transposing:
      self.min_pitch = self.min_pitch - 5
      self.max_pitch = self.max_pitch + 6
    self.shortest_duration = float(self.shortest_duration)
    self.separate_instruments = separate_instruments
    self.num_instruments = num_instruments
    self.encode_silences = encode_silences

    self.quantization_level = quantization_level
    if quantization_level is None:
      self.quant_ratio = 1
    else:
      self.quant_ratio = self.shortest_duration / self.quantization_level
    print 'quantization_level, and ratio', self.quantization_level, self.quant_ratio

  def quantize(self, time):
    # TODO: not yet quantize, but just dividing.
    return time / self.shortest_duration * self.quant_ratio

  def get_timestep(self, time):
    """Get the pianoroll timestep from seconds."""
    # TODO(annahuang): This assumes that everything divides, but may not be
    # the case.
    quantized_timestep = self.quantize(time)
    assert quantized_timestep % 1.0 == 0.0
    return int(quantized_timestep)

  def get_quantized_on_off_timesteps(self, onset, offset):
    onsetq = self.quantize(onset)
    offsetq = self.quantize(offset)
    onset_is_onbeat = onsetq % 1.0 == 0.0
    closest_offbeat = np.ceil(offsetq)
    #print '--', onset, onsetq, onset_is_onbeat
    if not onset_is_onbeat:
      closest_onbeat = np.ceil(onsetq)
      if closest_offbeat - closest_onbeat > 0:
        return int(closest_onbeat), int(closest_offbeat)
      return None, None
    if closest_offbeat < onsetq:
      return None, None
    #if closest_offbeat == onsetq:
    #  return int(onsetq), int(closest_offbeat) + 1
    return int(onsetq), int(closest_offbeat)

  def get_quantized_on_off_timesteps_may_overlap_with_other_notes(self, onset, offset):
    onsetq = self.quantize(onset)
    offsetq = self.quantize(offset)
    # The closest current or future beat on quantized grid
    closest_onbeat = np.ceil(onsetq)
    if closest_onbeat > offsetq:
      closest_onbeat = None
    # The closest previous or current beat on quantized grid
    closest_offbeat = np.floor(offsetq)
    if closest_offbeat < onsetq:
      closest_offbeat = None
    if closest_onbeat is None or closest_offbeat is None:
      return None, None
    else:
      return int(closest_onbeat), int(closest_offbeat)

  def encode(self,
             sequence,
             duration_ratio=1,
             return_program_to_pianoroll_map=False,
             return_with_additional_encodings=False):
    """Encode sequence into pianoroll."""
    # list of lists
    if isinstance(sequence, np.ndarray) or (
        isinstance(sequence, list) and (
            isinstance(sequence[0], list) or isinstance(sequence[0], tuple))):
      return self.encode_list_of_lists(
          sequence, duration_ratio=duration_ratio,
          return_with_additional_encodings=return_with_additional_encodings)
    elif isinstance(sequence, music_pb2.NoteSequence):
      #return self.encode_NoteSequences(
      return self.encode_NoteSequences_with_quantization(
          sequence, duration_ratio=duration_ratio,
          return_program_to_pianoroll_map=return_program_to_pianoroll_map,
          return_with_additional_encodings=return_with_additional_encodings)
          
    else:
      assert False, 'Type %s not yet supported.' % type(sequence)
     
  def encode_list_of_lists(self, sequence, 
                           duration_ratio=1,
                           return_with_additional_encodings=False):
    #TODO: duration_ratio not yet used.
    assert duration_ratio == 1

    # TODO: rename shortest_duration as it means here the duration_unit.
    # Hence if quant level is higher than we want to include every so many notes.
    skip_interval = self.quantization_level / self.shortest_duration
    assert skip_interval % 1.0 == 0.0
    skip_interval = int(skip_interval)

    if isinstance(sequence, np.ndarray):
      assert sequence.shape[-1] == 4
    assert len(sequence) % skip_interval == 0.
    T = int(len(sequence) / skip_interval)
    P = self.max_pitch - self.min_pitch + 1
    # For silences.
    P += 1
    if self.separate_instruments:
      roll = np.zeros((T, P, self.num_instruments))
    else:
      roll = np.zeros((T, P, 1))
    overlap_counts = 0
    for t, chord in enumerate(sequence):
      # Only takes time steps that are on the quantization grid.
      if t % skip_interval != 0.0:
        continue
      t /= skip_interval
      assert t % 1. == 0.
      if self.separate_instruments:
        for i in range(self.num_instruments):
          # FIXME: Need better way of aligning voices for time steps that are not full voicing.
          # FIXME: this only holds for bach pieces with 4voice encoding.
          #assert len(chord) == self.num_instruments
          if i < len(chord):
            pitch = chord[i]
            if not np.isnan(pitch):
              if pitch > self.max_pitch or pitch < self.min_pitch:
                raise PitchOutOfEncodeRangeError(
                    '%r is out of specified range [%r, %r].' % (
                        pitch, self.min_pitch, self.max_pitch))
              p = pitch - self.min_pitch
            else:
              # Then it's a silence
              p = P - 1
          else:
            # Then it's a silence
            p = P - 1
      
          assert p % 1. == 0.
          p = int(p)
   
          roll[t, p, i] = 1
      else:
        for pitch in chord:
          if not np.isnan(pitch):
            if pitch > self.max_pitch or pitch < self.min_pitch:
              raise PitchOutOfEncodeRangeError(
                  '%r is out of specified range [%r, %r].' % (
                      pitch, self.min_pitch, self.max_pitch))
            p = pitch - self.min_pitch
          else:
            # Then it's a silence
            p = P - 1
      
          assert p % 1. == 0.
          p = int(p)
          
          # Account for multiple voices having the same pitch in case of instruments separated
          if roll[t, p, 0] == 1:
            overlap_counts += 1
          else:
            roll[t, p, 0] = 1

    num_notes = np.sum(len(chord) for t, chord in enumerate(sequence) if t % skip_interval == 0.0)
    if num_notes != np.sum(roll) + overlap_counts:
      assert False, 'There are %d overlaps, but still (%d != %d).' % (
          overlap_counts, num_notes, np.sum(roll) + overlap_counts)
    if not return_with_additional_encodings:
      return roll[:, :-1, :]
    else:
      return roll      
  
  def encode_NoteSequences_with_quantization(self,
             sequence,
             duration_ratio=1,
             return_program_to_pianoroll_map=False,
             return_with_additional_encodings=False):
    """Encode sequence into pianoroll."""
    # Collect notes into voices.
    parts = defaultdict(list)
    #TODO(annahuang): Check source type and then check for parts if score-based.
    #print 'encode: source_type is', sequence.source_info.source_type
    if (sequence.source_info.source_type == 
        music_pb2.NoteSequence.SourceInfo.SCORE_BASED):
      attribute_for_program_index = 'part'
    elif (sequence.source_info.encoding_type == 
          music_pb2.NoteSequence.SourceInfo.MIDI):
      attribute_for_program_index = 'program'
    else:
      raise ValueError(
          'Source type or encoding type of sequence not yet supported')
    for note in sequence.notes:
      parts[getattr(note, attribute_for_program_index)].append(note)
    sorted_part_keys = sorted(parts.keys())

    # Map from note sequence (part/program) index to pianoroll depth number.
    program_to_pianoroll_index = {}
    num_timesteps = self.get_timestep(sequence.total_time * duration_ratio)
    pitch_range = self.max_pitch - self.min_pitch + 1
    if self.separate_instruments:
      pianoroll = np.zeros((num_timesteps, pitch_range, len(sorted_part_keys)))
    else:
      pianoroll = np.zeros((num_timesteps, pitch_range, 1))

    for pianoroll_index, program_index in enumerate(sorted_part_keys):
      notes = parts[program_index]
      program_to_pianoroll_index[program_index] = pianoroll_index
      for note in notes:
        start_index, end_index = self.get_quantized_on_off_timesteps(
            note.start_time * duration_ratio, note.end_time * duration_ratio)
        #print note.start_time, start_index, note.end_time, end_index
        if start_index is None or end_index is None: 
          continue
        if note.pitch > self.max_pitch or note.pitch < self.min_pitch:
          raise PitchOutOfEncodeRangeError(
              '%s is out of specified range [%s, %s].' % (
                  note.pitch, self.min_pitch, self.max_pitch))
        pitch_index = note.pitch - self.min_pitch
        if self.separate_instruments:
          pianoroll[start_index:end_index, pitch_index, pianoroll_index] = 1
        else:
          pianoroll[start_index:end_index, pitch_index, 0] = 1
    # TODO(annahuang): Put this constraint somewhere else.
    if self.separate_instruments and not are_instruments_monophonic(pianoroll):
      raise ValueError('This encoder only expects monophonic instruments.')
    if return_program_to_pianoroll_map:
      return pianoroll, program_to_pianoroll_index
    return pianoroll


  def encode_NoteSequences(self,
             sequence,
             duration_ratio=1,
             return_program_to_pianoroll_map=False,
             return_with_additional_encodings=False):
    """Encode sequence into pianoroll."""
    # Collect notes into voices.
    parts = defaultdict(list)
    #TODO(annahuang): Check source type and then check for parts if score-based.
    #print 'encode: source_type is', sequence.source_info.source_type
    if (sequence.source_info.source_type == 
        music_pb2.NoteSequence.SourceInfo.SCORE_BASED):
      attribute_for_program_index = 'part'
    elif (sequence.source_info.encoding_type == 
          music_pb2.NoteSequence.SourceInfo.MIDI):
      attribute_for_program_index = 'program'
    else:
      raise ValueError(
          'Source type or encoding type of sequence not yet supported')
    for note in sequence.notes:
      parts[getattr(note, attribute_for_program_index)].append(note)
    sorted_part_keys = sorted(parts.keys())

    # Map from note sequence (part/program) index to pianoroll depth number.
    program_to_pianoroll_index = {}
    num_timesteps = self.get_timestep(sequence.total_time * duration_ratio)
    pitch_range = self.max_pitch - self.min_pitch + 1
    if self.separate_instruments:
      pianoroll = np.zeros((num_timesteps, pitch_range, len(sorted_part_keys)))
    else:
      pianoroll = np.zeros((num_timesteps, pitch_range, 1))

    for pianoroll_index, program_index in enumerate(sorted_part_keys):
      notes = parts[program_index]
      program_to_pianoroll_index[program_index] = pianoroll_index
      for note in notes:
        start_index = self.get_timestep(note.start_time * duration_ratio)
        end_index = self.get_timestep(note.end_time * duration_ratio)

        if note.pitch > self.max_pitch or note.pitch < self.min_pitch:
          raise PitchOutOfEncodeRangeError(
              '%s is out of specified range [%s, %s].' % (
                  note.pitch, self.min_pitch, self.max_pitch))
        pitch_index = note.pitch - self.min_pitch
        if self.separate_instruments:
          pianoroll[start_index:end_index, pitch_index, pianoroll_index] = 1
        else:
          pianoroll[start_index:end_index, pitch_index, 0] = 1
    # TODO(annahuang): Put this constraint somewhere else.
    if self.separate_instruments and not are_instruments_monophonic(pianoroll):
      raise ValueError('This encoder only expects monophonic instruments.')
    if return_program_to_pianoroll_map:
      return pianoroll, program_to_pianoroll_index
    return pianoroll

  def aggregate_notes(self):
    # Aggregate the notes in both note sequences and sort them.
    aggregated_notes = []
    for note in out_mask_note_sequence.notes:
      aggregated_notes.append(note)
    for note in in_mask_note_sequence.notes:
      aggregated_notes.append(note)
    sorted_notes = sorted(aggregated_notes, key=lambda x: x.start_time)
    sorted_notes_by_endtime = sorted(aggregated_notes, key=lambda x: x.end_time)

    # Take one of the decoded note sequence as the container and update values.
    del out_mask_note_sequence.notes[:]
    out_mask_note_sequence.notes.extend(aggregated_notes)
    out_mask_note_sequence.total_time = sorted_notes_by_endtime[-1].end_time
    return out_mask_note_sequence

  def decode(self,
             pianoroll,
             pianoroll_to_program_map=None,
             qpm=_DEFAULT_QPM,
             velocity=None,
             channel_start_index=0,
             filename=None,
             source_type=music_pb2.NoteSequence.SourceInfo.SCORE_BASED):
    """Decode pianoroll into NoteSequence."""
    # TODO(annahuang): Handle unquantized time.
    if pianoroll.ndim != 3:
      raise ValueError(
          'Pianoroll needs to be of 3 dimensional, time, pitch, and instrument.')
    num_instruments = pianoroll.shape[-1]
    if pianoroll_to_program_map is None:
      pianoroll_to_program_map = get_pianoroll_to_program_assignment(
          range(num_instruments))
    else:
      pianoroll_to_program_map = get_pianoroll_to_program_assignment(
          range(num_instruments), pianoroll_to_program_map)
      
    if velocity is None:
      velocity = self.velocity

    # Instantiate a NoteSequence.
    sequence = music_pb2.NoteSequence()
    sequence.id = str(np.abs(hash(np.array_str(pianoroll))))
    #source_type = 'basic_autofill_cnn_generated'
    sequence.filename = '%s' % filename
    sequence.collection_name = 'basic_autofill_cnn_generated'
    # TODO: Do not set default for source_type
    sequence.source_info.source_type = source_type

    tempo = sequence.tempos.add()
    tempo.time = 0.0
    # TODO(annahuang): Infer and retrieve actual bpm.
    # Using default bpm.
    tempo.qpm = qpm

    # Using the MuseScore tick length for quarter notes.
    sequence.ticks_per_quarter = 480

    # Populate the notes.
    previously_off = lambda t, n, i: t == 0 or pianoroll[t - 1, n, i] == 0
    num_time_steps, num_pitches, num_instruments = pianoroll.shape
    total_time = 0.0
    for part_index in range(num_instruments):
      for time_step in range(num_time_steps):
        for note_number in range(num_pitches):
          # Check if note is on in this time_step and but not previously.
          if pianoroll[time_step, note_number, part_index] == 1 and (
              previously_off(time_step, note_number, part_index)):
            note = sequence.notes.add()
            note.pitch = note_number + self.min_pitch
            note.start_time = time_step * self.shortest_duration
            if SYNTH_MODE:
              note.start_time *= 2

            # Count how many contiguous time_steps are on.
            on_duration = self.shortest_duration
            for check_end_time_step in range(time_step + 1, num_time_steps):
              if pianoroll[check_end_time_step, note_number, part_index] != 1.0:
                break
              else:
                on_duration += self.shortest_duration
            note.end_time = note.start_time + on_duration
            if SYNTH_MODE:
              note.end_time = note.start_time + on_duration * 2
            if note.end_time > total_time:
              total_time = note.end_time
            note.velocity = velocity
            note.instrument = channel_start_index + part_index
            # Skip percussion channel 9, and shift all the instrument channels
            # up by one.
            if note.instrument > 8:
              note.instrument += 1
            note.program = pianoroll_to_program_map[part_index]
            note.part = part_index

    sequence.total_time = total_time
    return sequence
