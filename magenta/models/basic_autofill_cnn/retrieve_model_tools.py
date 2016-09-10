r"""Retrieve trained models."""
from collections import namedtuple
import os

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import basic_autofill_cnn_graph
from magenta.models.basic_autofill_cnn import config_tools


def retrieve_model(wrapped_model=None, model_name='DeepResidual'):
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  This function either takes a basic_autofill_cnn_graph.TFModelWrapper object
  that already has the model graph or calls
  basic_autofill_cnn_graph.build_graph to return one. It then retrieves its
  weights from the checkpoint file specified in the
  hparams_tools.CHECKPOINT_HPARAMS dictionary.

  Args:
    model_name: A string. The available models are in the dictionary
      hparams_tools.CHECKPOINT_HPARAMS, which are 'DeepResidualDataAug',
      'DeepResidual', 'PitchFullyConnectedWithResidual', etc.

  Returns:
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that
        consists of the model, graph, session and config.
  """
  if wrapped_model is None:
    config = config_tools.get_checkpoint_config(model_name=model_name)
    wrapped_model = basic_autofill_cnn_graph.build_graph(
        is_training=False, config=config)
  else:
    config = wrapped_model.config

  wrapped_model = basic_autofill_cnn_graph.build_graph(
      is_training=False, config=config)
  with wrapped_model.graph.as_default():
    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_fpath = os.path.join(tf.resource_loader.get_data_files_path(),
                                    'checkpoints',
                                    config.hparams.checkpoint_name)
    print 'checkpoint_fpath', checkpoint_fpath
    tf.logging.info('Checkpoint used: %s', checkpoint_fpath)
    try:
      saver.restore(sess, checkpoint_fpath)
    except IOError:
      tf.logging.fatal('No such file or directory: %s' % checkpoint_fpath)

  wrapped_model.sess = sess
  return wrapped_model
