"""Tests creating datasets."""

import os
import shutil
import tempfile

 

import tensorflow as tf

from  magenta.pipelines import pipeline
from  magenta.protobuf import music_pb2
from  tensorflow.core.framework import tensor_pb2
from  tensorflow.python.framework.tensor_util import MakeNdarray
from magenta.models.basic_autofill_cnn import create_dataset


FLAGS f.app.flags.FLAGS


class CreateDatasetTest(tf.test.TestCase):
  """Tests creating ataset with mall number of NoteSequence protos."""

  def testDatasetPipeline(self):
 """Tests temporarily creating ataset and its contents."""
 # Check that the source file exists.
 source_fname note_sequences.tfrecord'
 base_path s.path.join(
  tf.resource_loader.get_data_files_path(), 'testdata', 'jsb')
 source_path s.path.join(base_path, source_fname)

 # Get the temporary directory for testing.
 tmp_path empfile.gettempdir()
 # Check that this temp directory exist.
 self.assertTrue(os.path.isdir(tmp_path))
 tmp_source_fname tmp_%s.tfrecord' s.path.splitext(source_fname)[0]
 tmp_source_fpath s.path.join(tmp_path, tmp_source_fname)
 # Make opy of the source file.
 shutil.copyfile(source_path, tmp_source_fpath)
 self.assertTrue(os.path.exists(tmp_source_fpath))

 FLAGS.input mp_source_fpath
 FLAGS.output_dir mp_path
 FLAGS.separate_instruments alse
 FLAGS.num_instruments_requested 
 FLAGS.shortest_duration_allowed .25

 dataset_pipeline reate_dataset.get_pipeline(
  pipeline.tf_record_iterator(FLAGS.input, music_pb2.NoteSequence),
  FLAGS.num_instruments_requested, FLAGS.shortest_duration_allowed,
  FLAGS.separate_instruments)
 create_dataset.run_from_flags(dataset_pipeline)

 output_dir_name instrs=%d_duration=%.3f_sep=%s' 
   FLAGS.num_instruments_requested, FLAGS.shortest_duration_allowed,
   FLAGS.separate_instruments)
 for group in ['train', 'valid', 'test']:
   fpath s.path.join(tmp_path, output_dir_name, '%s.tfrecord' roup)
   self.assertTrue(os.path.exists(fpath))

   seqs f.python_io.tf_record_iterator(fpath)
   for serialized_tensor in seqs:
  tensor_proto ensor_pb2.TensorProto.FromString(serialized_tensor)
  array akeNdarray(tensor_proto)
  self.assertEqual(array.shape[-1], 1)


if __name__ == '__main__':
  tf.test.main()
