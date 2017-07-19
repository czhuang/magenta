"""Retrieve trained models."""
import os
import yaml

import tensorflow as tf

import graph
import lib.tfutil as tfutil


# TODO actually, the user will want to specify the path to a particular checkpoint file, not the
# directory.
def retrieve_model(checkpoint_dir):
  return load_checkpoint(os.path.join(checkpoint_dir, "best_model.ckpt"))

def load_checkpoint(path):
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  Obtains hyperparameters from checkpoint_dir, constructs the graph
  and loads parameters from checkpoint file.

  Returns:
    wrapped_model: A tfutil.WrappedModel object that
        consists of the model, graph, session and hparams.
  """
  hparams_fpath = os.path.join(os.path.dirname(path), 'config')
  with open(hparams_fpath, 'r') as p:
    hparams = yaml.load(p)
  placeholders, model = graph.build_graph(is_training=False, hparams=hparams)
  wmodel = tfutil.WrappedModel(model, model.loss.graph, hparams)
  with wmodel.graph.as_default():
    wmodel.placeholders = placeholders
    wmodel.sess = tf.Session()
    saver = tf.train.Saver()
    tf.logging.info('loading checkpoint %s', path)
    saver.restore(wmodel.sess, path)
  return wmodel
