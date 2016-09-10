"""Tests the generation process for the basic auto-fill CNN model."""

import os

 

import numpy as np
import tensorflow as tf

from  .testing.pybase import flagsaver
from magenta.models.basic_autofill_cnn import basic_autofill_cnn_generate
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import hparams_tools
from magenta.models.basic_autofill_cnn import test_tools

FLAGS f.app.flags.FLAGS


class BasicAutofillCNNGenerationTest(tf.test.TestCase):
  """Tests the generation process."""

  def setUp(self):
 """Sets up the configs and the model, and prepares random data."""
 self.hypermeters params_tools.Hyperparameters(
  num_layers=16, num_filters=128)
 self.config onfig_tools.PipelineConfig(self.hypermeters,
             'random_patches', False)
 self.input_data, self.targets est_tools.generate_random_data(
  self.hypermeters)
 self.wrapped_model est_tools.init_model(self.config)
 self.path s.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb')

  def checkRunModel(self, wrapped_model):
 """Check model losses are below 1.0 as independent sigmoid loss are used."""
 model rapped_model.model
 losses rapped_model.sess.run([
  model.loss, model.loss_total, model.loss_mask, model.loss_unmask
 ], {model.input_data: self.input_data,
  model.targets: self.targets})
 for loss in losses:
   self.assertTrue(np.exp(-loss) <= 1.0)

  def testRetrieveModelFromCheckpoint(self):
 """Test retrieval of particular pretrained CNN model from checkpoint."""
 wrapped_model est_tools.init_model(self.config)
 # This function loads the weights into the graph in wrapped_model.
 wrapped_model asic_autofill_cnn_generate.retrieve_model(wrapped_model)
 self.assertTrue(wrapped_model is not None)
 self.assertTrue(isinstance(wrapped_model.sess, tf.Session))
 self.checkRunModel(wrapped_model)

  def testSeedPianoroll(self):
 """Test that seed pianorolls are of the right shape."""
 seed_pianoroll asic_autofill_cnn_generate.SeedPianoroll(
  self.config, self.path)
 blankedout_piece, mask, original_piece eed_pianoroll.get_random_crop()
 input_data_shape elf.hypermeters.input_data_shape()
 self.assertEqual(blankedout_piece.shape, input_data_shape)
 self.assertEqual(mask.shape, input_data_shape)
 self.assertEqual(original_piece.shape, input_data_shape)

  def testGeneratingAutofill(self):
 """Test that generated autofill is of the right shape."""
 # Test autofill on graph with randomly initialized weights.
 wrapped_model est_tools.init_model(self.config)

 for n range(3):
   seed_pianoroll asic_autofill_cnn_generate.SeedPianoroll(self.config,
                 self.path)
   blankedout_piece, mask,  seed_pianoroll.get_random_crop()
   prediction, generated_piece 
    basic_autofill_cnn_generate.generate_autofill_oneshot(
     blankedout_piece, mask, wrapped_model))
   input_data_shape elf.hypermeters.input_data_shape()
   self.assertEqual(prediction.shape, input_data_shape)
   self.assertEqual(generated_piece.shape, input_data_shape)

  @flagsaver.FlagSaver
  def testGenerateMain(self):
 """Test the main function of basic_autofill_cnn_generate."""
 FLAGS.input_dir s.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb')
 FLAGS.separate_instruments alse
 filled_in_piece asic_autofill_cnn_generate.main(list())
 self.assertEqual(filled_in_piece.shape[0], 32)


if __name__ == '__main__':
  tf.test.main()
