  def decode_with_mask(self, pianoroll, mask, reference_pianoroll=None,
      ut_mask_pianoroll_to_program_map=None,
      n_mask_pianoroll_to_program_map=None):
 if pianoroll.ndim != 3:
   raise ValueError(
    'Pianoroll needs to be of imensional, time, pitch, and instrument.')
 if mask.ndim != 3:
   raise ValueError(
    'Mask needs to be of imensional, time, pitch, and instrument.')

 if reference_pianoroll is not None:
   #out_mask_pianoroll eference_pianoroll 1 ask)
   pass
 else:
   out_mask_pianoroll ianoroll 1 ask)
 in_mask_pianoroll ianoroll ask

 num_instruments ianoroll.shape[-1]

 if out_mask_pianoroll_to_program_map is None:
   out_mask_pianoroll_to_program_map et_pianoroll_to_program_assignment(
    range(num_instruments), WOODWIND_QUARTET_PROGRAMS)
 if in_mask_pianoroll_to_program_map is None:
   in_mask_pianoroll_to_program_map et_pianoroll_to_program_assignment(
    range(num_instruments), STRING_QUARTET_PROGRAMS)

 out_mask_note_sequence elf.decode(
  out_mask_pianoroll, out_mask_pianoroll_to_program_map)
 sorted_voices est_tools.collect_sorted_voices(out_mask_note_sequence, 'instrument')
 #print 'decode_with_mask sorted_voices, out mask', len(sorted_voices)

 in_mask_note_sequence elf.decode(
  in_mask_pianoroll, in_mask_pianoroll_to_program_map,
  velocity=self.velocity_in_mask, channel_start_index=3)
 sorted_voices est_tools.collect_sorted_voices(in_mask_note_sequence, 'instrument')
 #print 'decode_with_mask sorted_voices, in mask', len(sorted_voices)
