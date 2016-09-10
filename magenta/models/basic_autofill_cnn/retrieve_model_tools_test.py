"""Tests the generation process for the basic auto-fill CNN model."""

import os

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import retrieve_model_tools
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import test_tools


class RetrieveModelTests(tf.test.TestCase):
  """Tests the generation process."""

  def setUp(self):
    """Sets up the configs and the model, and prepares random data."""
    self.model_name = 'DeepResidual'
    self.config = config_tools.get_checkpoint_config(model_name=self.model_name)
    self.input_data, self.targets = test_tools.generate_random_data(
        self.config.hparams)
    self.wrapped_model = test_tools.init_model(self.config)
    self.path = os.path.join(tf.resource_loader.get_data_files_path(),
                             'testdata', 'jsb')

  def checkRunModel(self, wrapped_model):
    """Check model losses are below 1.0 as independent sigmoid loss are used."""
    model = wrapped_model.model
    losses = wrapped_model.sess.run([
        model.loss, model.loss_total, model.loss_mask, model.loss_unmask
    ], {model.input_data: self.input_data,
        model.targets: self.targets})
    for loss in losses:
      self.assertTrue(np.exp(-loss) <= 1.0)

  def testRetrieveModelFromCheckpoint(self):
    """Test retrieval of particular pretrained CNN model from checkpoint."""
    # This function sets up the graph and initializes the variables.
    wrapped_model = test_tools.init_model(self.config)
    # This function loads the trained weights into the graph in wrapped_model.
    wrapped_model = retrieve_model_tools.retrieve_model(
        wrapped_model, model_name=self.model_name)
    self.assertTrue(wrapped_model is not None)
    self.assertTrue(isinstance(wrapped_model.sess, tf.Session))
    self.checkRunModel(wrapped_model)


if __name__ == '__main__':
  tf.test.main()
