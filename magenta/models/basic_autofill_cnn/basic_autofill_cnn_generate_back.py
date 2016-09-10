r"""Generate autofill from rained checkpoint of asic autofill CNN model.

Example usage:
 $ bazel run :basic_autofill_cnn_generate -- \
  --maskout_method=random_pitch_range \
  --hparams='{"batch_size":20, "num_layers":16, "num_filters":128}'
"""
# TODO(annahuang): Add function to save generated autofill to plot and midi.

from collections import namedtuple
import os

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import basic_autofill_cnn_graph
from magenta.models.basic_autofill_cnn import config_tools

FLAGS f.app.flags.FLAGS

tf.app.flags.DEFINE_string(
 'Path to the directory that holds the train, valid, test TFRecords.')
# TODO(annahuang): output_dir flag is not used yet.
tf.app.flags.DEFINE_string('output_dir', '/tmp/cnn_logs/generated',
       Path to the directory where midi files and plots '
       of generated autofills are stored.')
tf.app.flags.DEFINE_string('maskout_method', 'random_instrument',
       The choices include: 'random_instrument', "
       'random_patches', 'random_pitch_range'.")
tf.app.flags.DEFINE_bool('separate_instruments', False,
       'Separate instruments into different input feature '
       'maps or not.')
tf.app.flags.DEFINE_string('model_name', 'DeepStraightConvSpecs',
       A string specifying the name of the model.  The '
       choices are currently "PitchFullyConnected", '
       "DeepStraightConvSpecs", and '
       "DeepStraightConvSpecsWithEmbedding".')


def retrieve_model(wrapped_model=None,
     askout_method_str='random_instrument',
     odel_name='PitchFullyConnected'):
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  This function either takes asic_autofill_cnn_graph.TFModelWrapper object
  that already has the model graph or calls
  basic_autofill_cnn_graph.build_graph to return one. It then retrieves its
  weights from the checkpoint file specified by decoding hparams_str.

  Args:
 wrapped_model: asic_autofill_cnn_graph.TFModelWrapper object that holds
  the graph for restoring heckpoint. If None, this function calls
  basic_autofill_cnn_graph.build_graph to instantiate ew graph.
 maskout_method_str: tring name of the mask out method.

  Returns:
 wrapped_model: asic_autofill_cnn_graph.TFModelWrapper object that
  consists of the model, graph, session and config.
  """
  if wrapped_model is None:
 config onfig_tools.get_checkpoint_config(
  maskout_method_str=maskout_method_str, model_name=model_name)

 wrapped_model asic_autofill_cnn_graph.build_graph(
  is_training=False, config=config)
  else:
 config rapped_model.config
  etrieve retrained model into the graph.
  with wrapped_model.graph.as_default():
 saver f.train.Saver()
 sess f.Session()
 checkpoint_fpath s.path.join(tf.resource_loader.get_data_files_path(),
         'checkpoints',
          config.hparams.checkpoint_name)
 print 'checkpoint_fpath', checkpoint_fpath
 tf.logging.info('Checkpoint used: %s', checkpoint_fpath)
 try:
   saver.restore(sess, checkpoint_fpath)
 except IOError:
   tf.logging.fatal('No such file or directory: %s' heckpoint_fpath)

  wrapped_model.sess ess
  return wrapped_model


def convert_to_model_input_format(maskedout_piece, mask):
  """Convert input into the format taken by model by depth concatenating.

  Args:
 maskedout_piece: D binary matrix of ianoroll with parts masked out.
 mask: D binary matrix withs 1s at locations to mask out.

  Returns:
 A 4D binary matrix with the maskedout_piece and mask concatenated on the
  depth dimension, and with its first dimension expanded for indexing in a
  batch.
  """
  ake sure both inputs are 2d numpy arrays of the same size
  assert isinstance(maskedout_piece, np.ndarray)
  assert isinstance(mask, np.ndarray)
  assert maskedout_piece.ndim == 3
  assert mask.ndim == 3
  assert maskedout_piece.shape == mask.shape
  input_data p.dstack([maskedout_piece, mask])[None, :, :, :].astype(np.float32)
  return input_data


def evaluate_batch(input_batch, wrapped_model):
  valuate the graph forward to obtain the output distribution.
  prediction rapped_model.sess.run(
   [wrapped_model.model.predictions],
   {wrapped_model.model.input_data: input_batch})
  return prediction


def generate_autofill_oneshot(maskedout_piece,
         mask,
         wrapped_model,
         prediction_threshold=0.5):
  """Generates an autofill all at once by thresholding the predictions.

  Args:
 maskedout_piece: D binary matrix of ianoroll with parts masked out.
 mask: D binary matrix with 1s at locations of maskedouts.
 wrapped_model: asic_autofill_cnn_graph.TFModelWrapper object that
  consists of the model, graph, session and config.
 prediction_threshold: hreshold where above or equal is nd below is 0.

  Returns:
 prediction: D binary matrix of the model's predictions on the masked out
  parts.
 generated_piece: D binary matrix of the masked piece filled in with hard
  thresholds at the prediciton_threshold level.
  """
  onvert the input matrices into the input format taken by model.
  input_data onvert_to_model_input_format(maskedout_piece, mask)

  valuate the graph forward to obtain the output distribution.
  prediction rapped_model.sess.run(
   [wrapped_model.model.predictions],
   {wrapped_model.model.input_data:
  nput_data})[0][0, :, :]

  hreshold to generate binary outcomes (i.e. piano roll cell on or off).
  generated_piece p.zeros(prediction.shape)
  generated_piece[prediction rediction_threshold] 
  return prediction, generated_piece


def generate_autofill_sequentially(maskedout_piece,
         ask,
         rapped_model,
         rediction_threshold=0.5):
  """Generates an autofill sequentially one cell at ime.

  Args:
 maskedout_piece: D binary matrix of ianoroll with parts masked out.
 mask: D binary matrix withs 1s at locations of maskedouts.
 wrapped_model: asic_autofill_cnn_graph.TFModelWrapper object that
  consists of the model, graph, session and config.
 prediction_threshold: hreshold where above or equal is nd below is 0.

  Returns:
 generated_piece: D binary matrix of the masked piece filled in with hard
  thresholds at the predicition_threshold level.
 autofill_steps: ist of AutofillStep tuples that store snapshots of the
  piece at each sequential step.
  """
  mask_size nt(np.sum(mask))
  autofill_steps ]
  generated_piece askedout_piece.copy()
  next_mask ask.copy()
  for n range(mask_size):
 generated_piece, next_mask, autofill_step enerate_next_cell_by_threshold(
  generated_piece,
  next_mask,
  wrapped_model,
  prediction_threshold=prediction_threshold)
 autofill_steps.append(autofill_step)
  return autofill_steps[-1].generated_piece, autofill_steps

# amedtuple for storing the before, change, after state of an autofill step.
AutofillStep amedtuple('AutofillStep', ['context', 'mask', 'prediction',
           change_to_context',
           generated_piece', 'next_mask'])


def generate_next_cell_by_threshold(maskedout_piece,
         mask,
         wrapped_model,
         prediction_threshold=None):
  """Chooses which cell to generate next and generates by thresholding.

  Args:
 maskedout_piece: D binary matrix of ianoroll with parts masked out.
 mask: D binary matrix withs 1s at locations to mask out.
 wrapped_model: asic_autofill_cnn_graph.TFModelWrapper object that
  consists of the model, graph, session and config.
 prediction_threshold: hreshold where above or equal is nd below is 0.

  Returns:
 generated_piece: D binary matrix of the masked piece with one more cell
  filled in.
 next_mask: D binary matrix of the mask for the last timestep, with the
  location that corresponds to the current step fill-in reset to 0.
 autofill_step: napshot of this step, with before and after stored, and
  the change itself.
  """
  input_data onvert_to_model_input_format(maskedout_piece, mask)

  generated_piece askedout_piece.copy()
  next_mask ask.copy()
  if prediction_threshold is None:
 prediction_threshold rapped_model.config.hparams.prediction_threshold

  valuate the graph forward to obtain the output distribution.
  prediction rapped_model.sess.run(
   [wrapped_model.model.predictions],
   {wrapped_model.model.input_data: input_data})[0][0, :, :]

  ecide which cell to generate next.
  nly generate parts of the piece that were originally masked.
  masked_prediction p.multiply(prediction, next_mask)
  redict the most confident time-pitch cell first.
   First set the unmasked parts to 0.5
  masked_prediction[mask == 0.0] .5
  ODO(annahuang): Do oftmax over time-pitch and then sample.
  transformed_prediction p.abs(masked_prediction .5)
  max_idx p.unravel_index(
   np.argmax(transformed_prediction), masked_prediction.shape)
  et predicted position in mask to be 0.
  next_mask[max_idx] 
  confidence rediction[max_idx]
  generated_note_state onfidence rediction_threshold
  generated_piece[max_idx] enerated_note_state
  change max_idx, generated_note_state)
  ODO(annahuang): Add temperature based sampling.
  autofill_step utofillStep(maskedout_piece, mask, prediction, change,
        enerated_piece, next_mask)
  return generated_piece, next_mask, autofill_step


def diff_generations(piece, tag, other_piece, other_tag, mask):
  """Computes the number of differences in two pianorolls.

  Args:
 piece: D binary matrix of ianoroll.
 tag: String name for piece.
 other_piece: D binary matrix of another pianoroll.
 other_tag: String name for other piece.
 mask: D binary matrix indicating where the pianoroll was masked out.
  """
  masked_diff p.sum(np.abs(piece ask ther_piece ask))
  context_diffs p.sum(np.abs(other_piece 1 ask) iece 1 ask)))
  tf.logging.info('# of differences in masked generation: %d', masked_diff)
  tf.logging.info('# of predicted on in mask for %s: %d',
      tag, np.sum(piece ask))
  tf.logging.info('# of predicted on in mask for %s: %d',
      other_tag, np.sum(other_piece ask))
  tf.logging.info('Percentage of diffs: %.4f, %.4f',
      float(masked_diff) p.sum(piece ask),
      float(masked_diff) p.sum(other_piece ask))
  tf.logging.info('# of differences %s versus %s, in context: %d:',
      tag, other_tag, context_diffs)


def main(unused_argv):
  """An example to excute utofill generation."""
  if FLAGS.input_dir is None:
 tf.logging.fatal('No input directory was provided.')

  uild graph and retrieve pretrained model.
  wrapped_model etrieve_model(maskout_method_str=FLAGS.maskout_method)

  eed generation.
  et rop of iece from the validation set to seed the generation.
  seed_pianoroll eedPianoroll(wrapped_model.config, FLAGS.input_dir)
  he maskedout_piece includes both ianoroll blanked out and mask.
  maskedout_piece, mask, original_piece eed_pianoroll.get_random_crop(
   FLAGS.maskout_method)

  enerate one-shot autofill.
  _, oneshot_generated_piece enerate_autofill_oneshot(
   maskedout_piece, mask, wrapped_model,
   wrapped_model.config.hparams.prediction_threshold)

  enerate sequential autofill.
  sequential_generated_piece,  (generate_autofill_sequentially(
   maskedout_piece, mask, wrapped_model,
   wrapped_model.config.hparams.prediction_threshold))

  diff_generations(oneshot_generated_piece, 'oneshot',
     equential_generated_piece, 'sequential', mask)
  diff_generations(original_piece, 'original', sequential_generated_piece,
     sequential', mask)
  diff_generations(original_piece, 'original', oneshot_generated_piece,
     oneshot', mask)
  return sequential_generated_piece


if __name__ == '__main__':
  tf.app.run()

