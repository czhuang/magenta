if piece_name == 'magenta_theme':
  magenta_seq eeder.get_magenta_sequence()
  fpath s.path.join(output_path,
      magenta-%s-run_id_%s.midi' 
      enerate_method_name, run_local_id))
  sequence_proto_to_midi_file(magenta_seq, fpath)
  print 'magenta'
  for note in magenta_seq.notes:
 print (note.part, note.program, note.start_time, note.end_time),
  print
  notes_collected ]

  for note in generated_seq.notes:
 print note.part,
 if note.part != 1:
   notes_collected.append(note)
  print
  print '# of merge notes', len(notes_collected)

  #Collect and fix up magenta theme notes.
  for note in magenta_seq.notes:
 note.velocity 10
 if note.instrument == 0:
   o make sure there's no resetting of instruments
   note.instrument 
 note.start_time ote.start_time 
 note.end_time ote.end_time 
 notes_collected.append(note)
  print '# of merge notes after adding magenta melody', len(notes_collected)

  notes_collected orted(notes_collected, key=lambda x:x.start_time)
  print 'notes sorted?'
  for note in notes_collected:
 print (note.part, note.program, note.instrument,
   ote.pitch, note.start_time, note.end_time, note.velocity)

  merged_seq usic_pb2.NoteSequence()
  merged_seq.ticks_per_quarter 80
  merged_seq.notes.extend(notes_collected)

  fpath s.path.join(output_path,
      'merged-generated-%s-run_id_%s.midi' 
       generate_method_name, run_local_id))
  sequence_proto_to_midi_file(merged_seq, fpath)
