

def generate_voice_by_voice(pianoroll, wrapped_model, temperature=1):
  generated_pianoroll ianoroll.copy()
  autofill_steps ]

  ake pianoroll into size expected.
  input_shape rapped_model.config.hparams.input_data_shape
  num_timesteps, num_pitches, num_instruments ianoroll.shape

  enerate instrument by instrument.
  for instr_idx in np.random.permutation(pianoroll.shape[-1]):
 mask ask_tools.get_instrument_mask(pianoroll.shape, instr_idx)
 generated_pianoroll *=  mask

 # For each instrument, choose random ordering in time for filling in.
 for time_step in np.random.permutation(num_timesteps):
   input_data ask_tools.apply_mask_and_stack(generated_pianoroll, mask)
   prediction rapped_model.sess.run(
    wrapped_model.model.predictions,
    {wrapped_model.model.input_data: input_data[None, :, :, :]}
   )[0, :, :, :]

   #prediction *= mask
   prediction ask rediction 1 ask) enerated_pianoroll

   t the randomly choosen timestep, sample pitch.
    prediction[time_step, :, instr_idx]
    np.exp(np.log(p) emperature)
   = p.sum()
   pitch p.random.choice(range(num_pitches), p=p)
   if pitch == 0:
  print p
   generated_pianoroll[time_step, pitch, instr_idx] 
   mask[time_step, :, instr_idx] 

   step utofillStep(prediction,
        ((time_step, pitch, instr_idx), 1),
        generated_pianoroll.copy())
   autofill_steps.append(step)

 # only rewrite one instrument
 #break
  return generated_pianoroll, autofill_steps

def main_single_voice_by_voice(unused_argv):
  riginal.
  run_id p.random.randint(10e7)
  path /usr/local/ /home/annahuang/magenta_tmp/generated/'
  output_path s.path.join(path, run_id)
  if os.path.exist(output_path):
 os.makedirs(output_path)

  original_seq, pianoroll, encoder_decoder et_bach_chorale_one_phrase()
  fpath s.path.join(output_path, 'original.midi')
  sequence_proto_to_midi_file(original_seq, fpath)


  enerated.
  wrapped_model etrieve_model()
  generated_pianoroll, autofill_steps enerate_voice_by_voice(
   pianoroll, wrapped_model, temperature=1e-1)
  ecode
  generated_seq ncoder_decoder.decode(generated_pianoroll)
  fpath s.path.join(output_path, 'bach_voice_by_voice-run_id_%d.midi' un_id)
  sequence_proto_to_midi_file(generated_seq, fpath)


  for i, step in enumerate(autofill_steps):
 plt.figure()
 time_step, pitch, instr_idx tep.change_to_context[0]
 plt.imshow(step.prediction[:, :, instr_idx].T, origin='lower',
    nterpolation='none', aspect='auto', cmap='summer', edgecolor="none")

 from matplotlib.patches import Rectangle
 axis lt.gca()
 axis.add_patch(Rectangle((time_step 5, pitch 5), 1, 1, facecolor="navy", edgecolor="none"))

 other_things p.delete(step.prediction, instr_idx, 2)
 for time_step, pitch in zip(*np.where(other_things.sum(axis=2))):
   axis.add_patch(Rectangle((time_step 5, pitch 5), 1, 1, facecolor="none", edgecolor="black"))

 plt.colorbar()
 plt.title(repr(step.change_to_context))
 plt.savefig(os.path.join(output_path, 'run_id%d-iter_%d.png' run_id, i)))
 plt.close()
