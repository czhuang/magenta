"""Tests building and running the graph of the basic auto-fill CNN model."""

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import basic_autofill_cnn_graph
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import test_tools


class BasicAutofillCNNGraphTest(tf.test.TestCase):
  """Tests building graphs and running models."""

  def setUp(self):
 """Sets up the configs and the model, and prepares random data."""
 self.config onfig_tools.get_checkpoint_config(model_name='SmallTest')

 self.input_data, self.targets est_tools.generate_random_data(
  self.config.hparams)
 self.wrapped_model est_tools.init_model(self.config)

  def checkRunModel(self, wrapped_model):
 """Check model losses are below 1.0 as independent sigmoid loss are used."""
 model rapped_model.model
 losses rapped_model.sess.run(
  [model.loss, model.loss_total, model.loss_mask, model.loss_unmask],
  {model.input_data: self.input_data, model.targets: self.targets})
 for loss in losses:
   self.assertTrue(np.exp(-loss) <= 1.0)

  def testBuildGraphAndModel(self):
 """Build graph with random weights and test that losses are within range."""
 wrapped_model elf.wrapped_model
 model rapped_model.model
 self.assertTrue(
  isinstance(model, basic_autofill_cnn_graph.BasicAutofillCNNGraph))
 self.checkRunModel(wrapped_model)

  def testTrainingModel(self):
 """Training andom step and check that all losses are within ranges."""
 wrapped_model elf.wrapped_model
 model rapped_model.model
 # Take raining step.
 with wrapped_model.graph.as_default():
    wrapped_model.sess.run([model.train_op],
         {model.input_data: self.input_data,
          model.targets: self.targets})
 # Evaluate the model to check that its losses are within range.
 self.checkRunModel(wrapped_model)


if __name__ == '__main__':
  tf.test.main()
