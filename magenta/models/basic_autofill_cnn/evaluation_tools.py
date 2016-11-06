"""Evaluations for comparing against prior work."""
import os, sys, traceback
import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import seed_tools, pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools, retrieve_model_tools, data_tools

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'input_dir', os.environ["JSB_TFRECORDS_DIR"],
    'Path to the directory that holds the train, valid, test TFRecords.')
tf.app.flags.DEFINE_string(
    'model_name', None,
    'name of the model to evaluate')
tf.app.flags.DEFINE_string(
    'fold', None,
    'data fold on which to evaluate (valid or test)')
tf.app.flags.DEFINE_string(
    'kind', None,
    'notewise or chordwise loss')
tf.app.flags.DEFINE_integer(
    'num_crops', 10,
    'number of random crops to consider')


def compute_chordwise_loss(model_name, fold):
  wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
  config = wrapped_model.config
  model = wrapped_model.model
  session = wrapped_model.sess
  sequences = list(data_tools.get_note_sequence_data(FLAGS.input_dir, fold))
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  piano_rolls = [encoder.encode(sequence) for sequence in sequences]

  losses = []
  def report():
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    sys.stdout.write("%.5f+-%.5f" % (loss_mean, loss_std))

  num_crops = 5
  for _ in range(num_crops):
    xs = np.array([data_tools.random_crop_pianoroll(x, config.hparams.crop_piece_len)
                   for x in piano_rolls], dtype=np.float32)
    # working copy to fill in model's own predictions of chord notes
    xs_scratch = np.copy(xs)

    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)

    # each example has its own ordering
    orders = np.ones([B, 1], dtype=np.int32) * np.arange(T, dtype=np.int32)[None, :]
    # random time orderings
    for i in range(B):
      np.random.shuffle(orders[i])
    # random instrument orderings within each time step
    orders = orders[:, :, None] * I + np.arange(I, dtype=np.int32)[None, None, :]
    for i in range(B):
      for t in range(T):
        np.random.shuffle(orders[i, t])
    orders = orders.reshape([B, T * I])

    for bahh, j in enumerate(orders.T):
      # NOTE: j is a vector with an index for each example in the batch
      t = j // I
      i = j % I

      # replace model predictions with ground truth when starting a fresh timestep
      if bahh % I == 0:
        xs_scratch = np.copy(xs)

      input_data = [mask_tools.apply_mask_and_stack(x, m)
                    for x, m in zip(xs_scratch, mask)]
      p = session.run(model.predictions,
                      feed_dict={model.input_data: input_data})
      loss = -np.where(xs_scratch[np.arange(B), t, :, i], np.log(p[np.arange(B), t, :, i]), 0).sum(axis=1)
      losses.append(loss)

      # update xs_scratch to contain predictions
      xs_scratch[np.arange(B), t, :, i] = np.eye(P)[np.argmax(p[np.arange(B), t, :, i], axis=1)]

      mask[np.arange(B), t, :, i] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      #print
      #mask_tools.print_mask(mask[0])

      if len(losses) % 100 == 0:
        report()

      sys.stdout.write(".")
      sys.stdout.flush()
    assert np.allclose(mask, 0)
    report()
  sys.stdout.write("\n")
  print "%s done" % model_name

def compute_notewise_loss(model_name, fold):
  wrapped_model = retrieve_model_tools.retrieve_model(model_name=model_name)
  config = wrapped_model.config
  model = wrapped_model.model
  session = wrapped_model.sess
  sequences = list(data_tools.get_note_sequence_data(FLAGS.input_dir, fold))
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  piano_rolls = [encoder.encode(sequence) for sequence in sequences]

  losses = []
  def report():
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    sys.stdout.write("%.5f+-%.5f" % (loss_mean, loss_std))

  num_crops = 5
  for _ in range(num_crops):
    xs = np.array([data_tools.random_crop_pianoroll(x, config.hparams.crop_piece_len)
                   for x in piano_rolls], dtype=np.float32)

    B, T, P, I = xs.shape
    mask = np.ones([B, T, P, I], dtype=np.float32)

    # each example has its own ordering
    orders = np.ones([B, 1], dtype=np.int32) * np.arange(T * I, dtype=np.int32)
    # yuck
    for i in range(B):
      np.random.shuffle(orders[i])

    for j in orders.T:
      # NOTE: j is a vector with an index for each example in the batch
      t = j // I
      i = j % I
  
      input_data = [mask_tools.apply_mask_and_stack(x, m)
                    for x, m in zip(xs, mask)]
      p = session.run(model.predictions,
                      feed_dict={model.input_data: input_data})
      loss = -np.where(xs[np.arange(B), t, :, i], np.log(p[np.arange(B), t, :, i]), 0).sum(axis=1)
      #loss = -(np.log(p[np.arange(B), t, :, i]) * xs[np.arange(B), t, :, i]).sum(axis=1)
      losses.append(loss)
      mask[np.arange(B), t, :, i] = 0
      assert np.unique(mask.sum(axis=(1, 2, 3))).size == 1

      if len(losses) % 100 == 0:
        report()

      sys.stdout.write(".")
      sys.stdout.flush()
    assert np.allclose(mask, 0)
    report()
  sys.stdout.write("\n")
  print "%s done" % model_name

def main(argv):
  try:
    print FLAGS.model_name, FLAGS.fold, FLAGS.kind
    fn = dict(notewise=compute_notewise_loss,
              chordwise=compute_chordwise_loss)[FLAGS.kind]
    fn(FLAGS.model_name, FLAGS.fold)
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, KeyboardInterrupt):
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()

if __name__ == '__main__':
  tf.app.run()
