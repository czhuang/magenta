"""Tests the training process for the basic auto-fill CNN model."""

import os

 

import numpy as np
import tensorflow as tf

 .testing.pybase import flagsaver
from magenta.models.basic_autofill_cnn import basic_autofill_cnn_train

FLAGS = tf.app.flags.FLAGS


class BasicAutofillCnnTrainTest(tf.test.TestCase):
  """A preliminary test for checking that training runs."""

  @flagsaver.FlagSaver
  def testTrainingMain(self):
    """Testing running the main example from basic_autofill_cnn_train."""
    FLAGS.input_dir = os.path.join(
        tf.resource_loader.get_data_files_path(), 'testdata', 'jsb')
    FLAGS.log_progress = False
    FLAGS.num_epochs = 1
    FLAGS.num_filters = 8
    FLAGS.num_layers = 4
    FLAGS.batch_size = 2
    FLAGS.augment_by_transposing = 0
    best_validation_loss = basic_autofill_cnn_train.main(list())
    self.assertTrue(np.exp(-best_validation_loss) <= 1.0)


if __name__ == '__main__':
  tf.test.main()
