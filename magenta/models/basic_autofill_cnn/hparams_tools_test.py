"""Tests that hyperparameters are set and updated correctly."""

 

import tensorflow as tf

from  .testing.pybase import flagsaver
from magenta.models.basic_autofill_cnn import hparams_tools
from magenta.models.basic_autofill_cnn.hparams_tools import ModelMisspecificationError


class HparamsToolsTest(tf.test.TestCase):
  """Tests the setting and updating of hyperparameters."""

  @flagsaver.FlagSaver
  def testSettingHparams(self):
 """Test overwriting default hyperparameters."""
 model_name DeepStraightConvSpecs'
 num_layers 
 num_pitches 8
 augment_by_transposing 
 # Test overwriting default hyperparameters.
 hparams params_tools.Hyperparameters(
  model_name=model_name, num_layers=num_layers, num_pitches=num_pitches,
  augment_by_transposing=augment_by_transposing)
 self.assertEqual(hparams.num_layers, num_layers)
 self.assertEqual(hparams.num_pitches, num_pitches)
 # Check that the ones not set is of its default values.
 self.assertEqual(hparams.num_filters, 256)
 self.assertEqual(hparams.batch_norm, True)
 self.assertTrue(hasattr(hparams, 'get_conv_arch'))

 # Test catching the model misspecification exception.
 num_layers 
 try:
   hparams params_tools.Hyperparameters(
    model_name=model_name, num_layers=num_layers,
    num_pitches=num_pitches)
 except ModelMisspecificationError:
   tf.logging.error('Model misspecification: too few layers.')


if __name__ == '__main__':
  tf.test.main()
