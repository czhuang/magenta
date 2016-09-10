

def check_attention():
  heck attention
  attentions ]
  for step in autofill_steps:
 attentions.append(step.full_attention.flatten())
  all_attentions p.concatenate(attentions)
  #np.histogram(np.asarray(attentions))
  plt.figure()
  results lt.hist(all_attentions, bins=10)
  #import pdb; pdb.set_trace()
  print results
  print '# zeros:', np.sum(all_attentions==0.0)
  fpath s.path.join(output_path,
      histogram-full_attention-%s-run_id_%s.png' 
       enerate_method_name, run_local_id))
  plt.savefig(fpath)
  plt.close()

  plt.figure()
  log_attentions 0*np.log(1+all_attentions)
  results_log lt.hist(log_attentions, bins=10)
  print '# zeros:', np.sum(log_attentions==0.0)
  print '% zero', np.sum(log_attentions==0.0) loat(np.product(all_attentions.shape))
  print results_log
  fpath s.path.join(output_path,
      histogram-log-full_attention-%s-run_id_%s.png' 
       enerate_method_name, run_local_id))
  plt.savefig(fpath)
  plt.close()


def plot_attention():
  check_attention()
  def get_max_absolute_sum_instrs(autofill_steps, attribute_name, take_log):
 global_max np.inf
 for step in autofill_steps:
   if take_log:
  local_max p.max(np.abs(np.log(np.abs(1 etattr(step, attribute_name).sum(axis=2)))))
   else:
  local_max p.max(np.abs(getattr(step, attribute_name).sum(axis=2)))
   if local_max lobal_max:
  global_max ocal_max
 return global_max

  eed to collect all attention to put it on the same scale
  #attention_max et_max_absolute_sum_instrs(autofill_steps, 'attention', take_log)
  #full_attention_max et_max_absolute_sum_instrs(autofill_steps, 'full_attention', take_log)

  attention_max et_max_absolute_sum_instrs([step], 'attention', take_log)
  full_attention_max et_max_absolute_sum_instrs([step], 'full_attention', take_log)

  ttention.
  for attention_type in ['attention', 'full_attention']:
 plt.figure()
 axis lt.gca()

 attention etattr(step, attention_type)

 #all_context riginal_context.sum(axis=2) lready_generated_pianoroll.sum(axis=2)

 attention_to_plot ttention.sum(axis=2).T
 if take_log:
   attention_to_plot_signs p.sign(attention_to_plot)
   log_abs_attention_to_plot p.log(np.abs(1 ttention_to_plot))
   attention_to_plot og_abs_attention_to_plot ttention_to_plot_signs

 if attention_type == 'attention':
   plot_max ttention_max
 else:
   plot_max ull_attention_max
 plt.imshow(attention_to_plot, origin='lower',
    nterpolation='none', aspect='auto', cmap='BrBG',
    min=-plot_max, vmax=plot_max) #cmap='spectral')

 # What notes are in the input?
 # The original context that's not yet blanked out.
 for t, n zip(*np.where(original_context.sum(axis=2))):
   axis.add_patch(Rectangle((t 5,  .5), 1, 1,
        ill=None, edgecolor="navy"))

 # The generated pianoroll.
 for t, n zip(*np.where(previous_already_generated_pianoroll.sum(axis=2))):
   axis.add_patch(Rectangle((t 5,  .5), 1, 1,
        ill=None, edgecolor="magenta"))

 # Marks the current change.
 axis.add_patch(Rectangle((time_step 5, pitch 5), 1, 1,
        fill=None, edgecolor="cyan"))

 plt.colorbar()
 plt.title(attention_type epr(step.change_to_context))
 plt.savefig(os.path.join(
  output_path, 'att-run_id%s-iter_%d-summed-%s.png' run_id, i, attention_type)))
 plt.close()

  lot subplots of all attention.
  def get_max_absolute(autofill_steps, attribute_name, take_log):
 global_max np.inf
 for step in autofill_steps:
   if take_log:
  local_max p.max(np.abs(np.log(np.abs(1 etattr(step, attribute_name)))))
   else:
  local_max p.max(np.abs(getattr(step, attribute_name)))
   if local_max lobal_max:
  global_max ocal_max
 return global_max

  ingle instrument max.
  #single_instr_attention_max et_max_absolute(autofill_steps, 'full_attention', take_log)
  single_instr_attention_max et_max_absolute([step], 'full_attention', take_log)

  attention etattr(step, 'full_attention')
  assert attention.shape[-1] == 8
  for indices in [range(4), range(4, 8)]:
 plt.figure()
 for idx in indices:
   plt.subplot(2, 2, (idx ) )
   axis lt.gca()
   attention_to_plot ttention[:, :, idx]
   if take_log:
  attention_to_plot og_negative_positive_part_separately(attention_to_plot)
   print 'max, min', idx, np.min(attention_to_plot), \
   np.max(attention_to_plot)
   print 'single global max', single_instr_attention_max

   plt.imshow(attention_to_plot.T,
    origin='lower', interpolation='none', aspect='auto', cmap='BrBG',
    vmin=-single_instr_attention_max, vmax=single_instr_attention_max)
   if idx :
  tag pianoroll'
   else:
  tag mask'
   plt.title(tag tr(idx) epr(step.change_to_context))

   hat notes are in the input?
   he original context that's not yet blanked out.
   for t, n zip(*np.where(original_context.sum(axis=2))):
  axis.add_patch(Rectangle((t 5,  .5), 1, 1,
         fill=None, edgecolor="navy"))

   he generated pianoroll.
   for t, n zip(*np.where(previous_already_generated_pianoroll.sum(axis=2))):
  axis.add_patch(Rectangle((t 5,  .5), 1, 1,
         fill=None, edgecolor="magenta"))

   arks the current change.
   axis.add_patch(Rectangle((time_step 5, pitch 5), 1, 1,
        ill=None, edgecolor="cyan"))

 plt.colorbar()
 plt.savefig(os.path.join(
  output_path,
  'att_each-run_id%s-iter_%d-each_%d.png' run_id, i, idx)))
 plt.close()

 
def log_negative_positive_part_separately(mat):
  signs p.sign(mat)
  logged_abs p.log(np.abs(1+mat))
  return logged_abs igns
