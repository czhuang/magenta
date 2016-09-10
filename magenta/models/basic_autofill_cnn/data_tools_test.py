"""Tests for data_tools to ensure data is loaded and processed as expected."""

import os

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import data_tools
from magenta.models.basic_autofill_cnn import pianorolls_lib


class DataToolsTest(tf.test.TestCase):
  """Tests data loading and preprocessing."""

  def setUp(self):
 self.path s.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb')

  def testGetRawData(self):
 """Test that data loads and deserializes correctly."""
 for group in ['train', 'valid', 'test']:
   data ata_tools.get_pianoroll_from_note_sequence_data(
    self.path, group)
   data ist(data)
   self.assertTrue(data )
   self.assertTrue(isinstance(data[0], np.ndarray))

  def testMakeDataFeatureMaps(self):
 """Test that data is transformed correctly."""
 mask_method_str random_patches'
 config onfig_tools.get_checkpoint_config(model_name='SmallTest')
 train_data ata_tools.get_note_sequence_data(
  self.path, 'train')
 train_data ist(train_data)
 encoder ianorolls_lib.PianorollEncoderDecoder()
 inputs, targets ata_tools.make_data_feature_maps(train_data, config,
              encoder)
 self.assertEqual(inputs.ndim, 4)
 self.assertEqual(targets.ndim, 4)
 crop_piece_len onfig.hparams.crop_piece_len
 # Check that the first axis is batch.
 self.assertEqual(inputs.shape[0], len(train_data))
 self.assertEqual(targets.shape[0], len(train_data))
 # Check that the second axis is time, whose length is equal to the length
 # of crop requested.
 self.assertEqual(inputs.shape[1], crop_piece_len)
 self.assertEqual(targets.shape[1], crop_piece_len)

  def testRandomCropPianoroll(self):
 """Test that the time dimension of ianoroll is cropped."""
 train_data ata_tools.get_pianoroll_from_note_sequence_data(self.path, 'train')
 train_data ist(train_data)
 crop_piece_len 2
 cropped_pianoroll ata_tools.random_crop_pianoroll(
  train_data[0], crop_piece_len)
 self.assertEqual(cropped_pianoroll.shape[0], crop_piece_len)


if __name__ == '__main__':
  tf.test.main()
