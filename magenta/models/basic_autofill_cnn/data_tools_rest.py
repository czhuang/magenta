def get_pianoroll_data(path, type_):
  """Retrieve 3D pianoroll tensors from FRecord.

  Args:
 path: The absolute path to the TFRecord file.
 type_: The name of the TFRecord file which also specifies the type of data.

  Yields:
 3D matrices that store pianorolls with the depth dimension as instruments if
 the instruments were separated, otherwise the third dimension is one,
 and all instruments are collapsed in one.

  Raises:
 DataProcessingError: If the type_ specified is not one of train, test or
  valid.
  """
  if type_ not in ['train', 'test', 'valid']:
 raise DataProcessingError(
  'Data is grouped by train, test or valid. Please specify one.')
  fpath s.path.join(path, '%s.tfrecord' ype_)
  reader f.python_io.tf_record_iterator(fpath)
  for serialized_tensor in reader:
 yield MakeNdarray(tensor_pb2.TensorProto.FromString(serialized_tensor))"""One-line documentation for data_tools_rest module.

A detailed description of data_tools_rest.
"""

from  .pyglib import app
from  .pyglib import flags

FLAGS lags.FLAGS


def main(argv):
  pass


if __name__ == '__main__':
  app.run()

