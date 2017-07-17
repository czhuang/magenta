"""Retrieve trained models."""
from collections import namedtuple
import os

import yaml

import numpy as np
import tensorflow as tf

import graph
from hparams_tools import Hyperparameters
import train

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', None, 'Path to checkpoint directory.')


def retrieve_model(wrapped_model=None, model_name=None, placeholders=None, 
                   hparam_updates=None):
  # TODO: change function name to reflect the fact that it updates hparams
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  This function either takes a graph.TFModelWrapper object
  that already has the model graph or calls
  graph.build_graph to return one. It then retrieves its
  weights from the checkpoint file specified in the
  hparams_tools.CHECKPOINT_HPARAMS dictionary.

  Returns:
    wrapped_model: A graph.TFModelWrapper object that
        consists of the model, graph, session and hparams.
  """
  if wrapped_model is None:
    hparams = get_checkpoint_hparams(model_name=model_name)
  else:
    hparams = wrapped_model.hparams

  # update hparams
  if hparam_updates is not None:
    for key, val in hparam_updates.iteritems():
      if hasattr(hparams, key):
        print 'Update hparams %s to be %r' % (key, val)
        setattr(hparams, key, val)
      else:
        assert False, 'hparams does not have this parameters %s' % key

  wrapped_model = graph.build_graph(
      is_training=False, hparams=hparams, placeholders=placeholders)
  with wrapped_model.graph.as_default():
    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_fpath = hparams.checkpoint_fpath
    print 'checkpoint_fpath', checkpoint_fpath
    tf.logging.info('Checkpoint used: %s', checkpoint_fpath)
    try:
      saver.restore(sess, checkpoint_fpath)
    except IOError:
      tf.logging.fatal('No such file or directory: %s' % checkpoint_fpath)

  wrapped_model.sess = sess
  return wrapped_model


# TODO: Provide pre-trained model.
CHECKPOINT_HPARAMS = {
}


def get_checkpoint_hparams(model_name):
  """Returns the model architecture."""
  print 'model_name', model_name
  if model_name is None and FLAGS.checkpoint_dir is None:
    raise ModelMisspecificationError('No model name or checkpoint path specified.')
  if model_name is None:
    hparams_fpath = os.path.join(FLAGS.checkpoint_dir, 'config')
    with open(hparams_fpath, 'r') as p:
      hparams = yaml.load(p)
    hparams.checkpoint_fpath = os.path.join(
        FLAGS.checkpoint_dir,
        '%s-best_model.ckpt' % (hparams.conv_arch.name))
    print 'Will load checkpoint from ', hparams.checkpoint_fpath
    return hparams
 
  elif model_name not in CHECKPOINT_HPARAMS:
    raise ModelMisspecificationError('Model name %s does not exist.' % model_name)
  else:
    return CHECKPOINT_HPARAMS[model_name]


