"""Retrieve trained models."""
import os
import yaml

import tensorflow as tf

import graph
import lib.tfutil as tfutil


def retrieve_model(checkpoint_dir):
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  This function either takes a tfutil.WrappedModel object
  that already has the model graph or calls
  graph.build_wrapped_model to return one. It then retrieves its
  weights from the checkpoint file specified in the
  hparams_tools.CHECKPOINT_HPARAMS dictionary.

  Returns:
    wrapped_model: A tfutil.WrappedModel object that
        consists of the model, graph, session and hparams.
  """
  hparams = get_checkpoint_hparams(checkpoint_dir)
  placeholders, model = graph.build_graph(is_training=False, hparams=hparams)
  wmodel = tfutil.WrappedModel(model, model.loss.graph, hparams)
  with wmodel.graph.as_default():
    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_fpath = hparams.checkpoint_fpath
    print 'checkpoint_fpath', checkpoint_fpath
    tf.logging.info('Checkpoint used: %s', checkpoint_fpath)
    try:
      saver.restore(sess, checkpoint_fpath)
    except IOError:
      tf.logging.fatal('No such file or directory: %s' % checkpoint_fpath)
    wmodel.sess = sess
    wmodel.placeholders = placeholders
  return wmodel

def get_checkpoint_hparams(checkpoint_dir):
  hparams_fpath = os.path.join(checkpoint_dir, 'config')
  with open(hparams_fpath, 'r') as p:
    hparams = yaml.load(p)
  hparams.checkpoint_fpath = os.path.join(checkpoint_dir, 'best_model.ckpt')
  print 'Will load checkpoint from ', hparams.checkpoint_fpath
  return hparams
