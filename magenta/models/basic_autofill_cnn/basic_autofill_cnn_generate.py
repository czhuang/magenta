r"""Generate autofill from a trained checkpoint of a basic autofill CNN model.

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
from magenta.models.basic_autofill_cnn import data_tools

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_dir', None,
    'Path to the directory that holds the train, valid, test TFRecords.')
# TODO(annahuang): output_dir flag is not used yet.
tf.app.flags.DEFINE_string('output_dir', '/tmp/cnn_logs/generated',
                           'Path to the directory where midi files and plots '
                           'of generated autofills are stored.')
tf.app.flags.DEFINE_string('maskout_method', 'random_pitch_range',
                           "The choices include: 'random_instrument', "
                           "'random_patches', 'random_pitch_range'.")
tf.app.flags.DEFINE_bool('separate_instruments', False,
                         'Separate instruments into different input feature '
                         'maps or not.')


class SeedPianoroll(object):
  """Produces a pianoroll with maskouts to seed generation."""

  def __init__(self, config, path):
    valid_data = data_tools.get_raw_data(path, 'valid')
    # Only use validation (unseen) data to seed the generation.
    self._raw_data = list(valid_data)
    self._num_pieces = len(self._raw_data)
    self._config = config

  @property
  def config(self):
    return self._config

  def get_random_crop(self, maskout_method_str='random_pitch_range'):
    """Crops piece in the validation set, and blanks out a part of it."""
    # Randomly choose a piece.
    random_piece = np.random.choice(self._raw_data)
    self._config.maskout_method = maskout_method_str
    input_data, targets = data_tools.make_data_feature_maps(
        [random_piece], self._config)
    maskedout_piece = input_data[0, :, :, 0]
    mask = input_data[0, :, :, 1]
    target = targets[0, :, :]
    return maskedout_piece, mask, target


def retrieve_model(wrapped_model=None,
                   maskout_method_str='random_pitch_range'):
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  This function either takes a basic_autofill_cnn_graph.TFModelWrapper object
  that already has the model graph or calls
  basic_autofill_cnn_graph.build_graph to return one. It then retrieves its
  weights from the checkpoint file specified by decoding hparams_str.

  Args:
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that holds
        the graph for restoring a checkpoint. If None, this function calls
        basic_autofill_cnn_graph.build_graph to instantiate a new graph.
    maskout_method_str: A string name of the mask out method.

  Returns:
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that
        consists of the model, graph, session and config.
  """
  config = None
  if wrapped_model is None:
    config = config_tools.get_checkpoint_config(
        maskout_method_str=maskout_method_str)

    wrapped_model = basic_autofill_cnn_graph.build_graph(
        is_training=False, config=config)
  else:
    config = wrapped_model.config
  # Retrieve a pretrained model into the graph.
  with wrapped_model.graph.as_default():
    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_fpath = os.path.join(tf.resource_loader.get_data_files_path(),
                                    'checkpoints',
                                    '%s_best_model.ckpt' % config.hparams.name)
    tf.logging.info('Checkpoint used: %s', checkpoint_fpath)
    try:
      saver.restore(sess, checkpoint_fpath)
    except IOError:
      tf.logging.fatal('No such file or directory: %s' % checkpoint_fpath)

  wrapped_model.sess = sess
  return wrapped_model


def convert_to_model_input_format(maskedout_piece, mask):
  """Convert input into the format taken by model by depth concatenating.

  Args:
    maskedout_piece: A 2D binary matrix of a pianoroll with parts masked out.
    mask: A 2D binary matrix withs 1s at locations to mask out.

  Returns:
    A 4D binary matrix with the maskedout_piece and mask concatenated on the
        depth dimension, and with its first dimension expanded for indexing in a
        batch.
  """
  # Make sure both inputs are 2d numpy arrays of the same size
  assert isinstance(maskedout_piece, np.ndarray)
  assert isinstance(mask, np.ndarray)
  assert len(maskedout_piece.shape) == 2
  assert len(mask.shape) == 2
  assert maskedout_piece.shape == mask.shape
  return np.dstack([maskedout_piece, mask])[None, :, :, :]


def generate_autofill_oneshot(maskedout_piece,
                              mask,
                              wrapped_model,
                              prediction_threshold=0.5):
  """Generates an autofill all at once by thresholding the predictions.

  Args:
    maskedout_piece: A 2D binary matrix of a pianoroll with parts masked out.
    mask: A 2D binary matrix with 1s at locations of maskedouts.
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that
        consists of the model, graph, session and config.
    prediction_threshold: A threshold where above or equal is 1 and below is 0.

  Returns:
    prediction: A 2D binary matrix of the model's predictions on the masked out
        parts.
    generated_piece: A 2D binary matrix of the masked piece filled in with hard
        thresholds at the prediciton_threshold level.
  """
  # Convert the input matrices into the input format taken by model.
  input_data = convert_to_model_input_format(maskedout_piece, mask)

  # Evaluate the graph forward to obtain the output distribution.
  prediction = wrapped_model.sess.run(
      [wrapped_model.model.predictions],
      {wrapped_model.model.input_data: input_data})[0][0, :, :]

  # Threshold to generate binary outcomes (i.e. piano roll cell on or off).
  generated_piece = np.zeros(prediction.shape)
  generated_piece[prediction > prediction_threshold] = 1
  return prediction, generated_piece


def generate_autofill_sequentially(maskedout_piece,
                                   mask,
                                   wrapped_model,
                                   prediction_threshold=0.5):
  """Generates an autofill sequentially one cell at a time.

  Args:
    maskedout_piece: A 2D binary matrix of a pianoroll with parts masked out.
    mask: A 2D binary matrix withs 1s at locations of maskedouts.
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that
        consists of the model, graph, session and config.
    prediction_threshold: A threshold where above or equal is 1 and below is 0.

  Returns:
    generated_piece: A 2D binary matrix of the masked piece filled in with hard
        thresholds at the predicition_threshold level.
    autofill_steps: A list of AutofillStep tuples that store snapshots of the
        piece at each sequential step.
  """
  mask_size = int(np.sum(mask))
  autofill_steps = []
  generated_piece = maskedout_piece.copy()
  next_mask = mask.copy()
  for _ in range(mask_size):
    generated_piece, next_mask, autofill_step = generate_next_cell_by_threshold(
        generated_piece,
        next_mask,
        wrapped_model,
        prediction_threshold=prediction_threshold)
    autofill_steps.append(autofill_step)
  return autofill_steps[-1].generated_piece, autofill_steps

# A namedtuple for storing the before, change, after state of an autofill step.
AutofillStep = namedtuple('AutofillStep', ['context', 'mask', 'prediction',
                                           'change_to_context',
                                           'generated_piece', 'next_mask'])


def generate_next_cell_by_threshold(maskedout_piece,
                                    mask,
                                    wrapped_model,
                                    prediction_threshold=None):
  """Chooses which cell to generate next and generates by thresholding.

  Args:
    maskedout_piece: A 2D binary matrix of a pianoroll with parts masked out.
    mask: A 2D binary matrix withs 1s at locations to mask out.
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that
        consists of the model, graph, session and config.
    prediction_threshold: A threshold where above or equal is 1 and below is 0.

  Returns:
    generated_piece: A 2D binary matrix of the masked piece with one more cell
        filled in.
    next_mask: A 2D binary matrix of the mask for the last timestep, with the
        location that corresponds to the current step fill-in reset to 0.
    autofill_step: A snapshot of this step, with before and after stored, and
        the change itself.
  """
  input_data = convert_to_model_input_format(maskedout_piece, mask)

  generated_piece = maskedout_piece.copy()
  next_mask = mask.copy()
  if prediction_threshold is None:
    prediction_threshold = wrapped_model.config.hparams.prediction_threshold

  # Evaluate the graph forward to obtain the output distribution.
  prediction = wrapped_model.sess.run(
      [wrapped_model.model.predictions],
      {wrapped_model.model.input_data: input_data})[0][0, :, :]

  # Decide which cell to generate next.
  # Only generate parts of the piece that were originally masked.
  masked_prediction = np.multiply(prediction, next_mask)
  # Predict the most confident time-pitch cell first.
  #   First set the unmasked parts to 0.5
  masked_prediction[mask == 0.0] = 0.5
  # TODO(annahuang): Do a softmax over time-pitch and then sample.
  transformed_prediction = np.abs(masked_prediction - 0.5)
  max_idx = np.unravel_index(
      np.argmax(transformed_prediction), masked_prediction.shape)
  # Set predicted position in mask to be 0.
  next_mask[max_idx] = 0
  confidence = prediction[max_idx]
  generated_note_state = confidence > prediction_threshold
  generated_piece[max_idx] = generated_note_state
  change = (max_idx, generated_note_state)
  # TODO(annahuang): Add temperature based sampling.
  autofill_step = AutofillStep(maskedout_piece, mask, prediction, change,
                               generated_piece, next_mask)
  return generated_piece, next_mask, autofill_step


def diff_generations(piece, tag, other_piece, other_tag, mask):
  """Computes the number of differences in two pianorolls.

  Args:
    piece: A 2D binary matrix of a pianoroll.
    tag: String name for piece.
    other_piece: A 2D binary matrix of another pianoroll.
    other_tag: String name for other piece.
    mask: A 2D binary matrix indicating where the pianoroll was masked out.
  """
  masked_diff = np.sum(np.abs(piece * mask - other_piece * mask))
  context_diffs = np.sum(np.abs(other_piece * (1 - mask) - piece * (1 - mask)))
  tf.logging.info('# of differences in masked generation: %d', masked_diff)
  tf.logging.info('# of predicted on in mask for %s: %d',
                  tag, np.sum(piece * mask))
  tf.logging.info('# of predicted on in mask for %s: %d',
                  other_tag, np.sum(other_piece * mask))
  tf.logging.info('Percentage of diffs: %.4f, %.4f',
                  float(masked_diff) / np.sum(piece * mask),
                  float(masked_diff) / np.sum(other_piece * mask))
  tf.logging.info('# of differences %s versus %s, in context: %d:',
                  tag, other_tag, context_diffs)


def main(unused_argv):
  """An example to excute a autofill generation."""
  if FLAGS.input_dir is None:
    tf.logging.fatal('No input directory was provided.')

  # Build graph and retrieve pretrained model.
  wrapped_model = retrieve_model(maskout_method_str=FLAGS.maskout_method)

  # Seed generation.
  # Get a crop of a piece from the validation set to seed the generation.
  seed_pianoroll = SeedPianoroll(wrapped_model.config, FLAGS.input_dir)
  # The maskedout_piece includes both a pianoroll blanked out and mask.
  maskedout_piece, mask, original_piece = seed_pianoroll.get_random_crop(
      FLAGS.maskout_method)

  # Generate one-shot autofill.
  _, oneshot_generated_piece = generate_autofill_oneshot(
      maskedout_piece, mask, wrapped_model,
      wrapped_model.config.hparams.prediction_threshold)

  # Generate sequential autofill.
  sequential_generated_piece, _ = (generate_autofill_sequentially(
      maskedout_piece, mask, wrapped_model,
      wrapped_model.config.hparams.prediction_threshold))

  diff_generations(oneshot_generated_piece, 'oneshot',
                   sequential_generated_piece, 'sequential', mask)
  diff_generations(original_piece, 'original', sequential_generated_piece,
                   'sequential', mask)
  diff_generations(original_piece, 'original', oneshot_generated_piece,
                   'oneshot', mask)
  return sequential_generated_piece


if __name__ == '__main__':
  tf.app.run()
