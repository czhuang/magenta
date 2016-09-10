"""Tests the training process for the basic auto-fill CNN model."""

import os

 

import numpy as np
import tensorflow as tf

from  .testing.pybase import flagsaver
from magenta.models.basic_autofill_cnn import basic_autofill_cnn_train

FLAGS f.app.flags.FLAGS


class BasicAutofillCnnTrainTest(tf.test.TestCase):
  """A preliminary test for checking that training runs."""

  @flagsaver.FlagSaver
  def testTrainingMain(self):
 """Testing running the main example from basic_autofill_cnn_train."""
 FLAGS.input_dir s.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb')
 FLAGS.log_progress alse
 FLAGS.num_epochs 
 FLAGS.num_filters 
 FLAGS.num_layers 
 FLAGS.batch_size 
 FLAGS.augment_by_transposing 
 best_validation_loss asic_autofill_cnn_train.main(list())
 self.assertTrue(np.exp(-best_validation_loss) <= 1.0)


if __name__ == '__main__':
  tf.test.main()
