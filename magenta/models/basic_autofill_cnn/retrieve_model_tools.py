"""Retrieve trained models."""
from collections import namedtuple
import os

import yaml

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import basic_autofill_cnn_graph
from magenta.models.basic_autofill_cnn.hparams_tools import Hyperparameters
from magenta.models.basic_autofill_cnn import basic_autofill_cnn_train


tf.app.flags.DEFINE_string('checkpoint_dir', None, 'Path to checkpoint directory.')
#FIXME: remove later.
tf.app.flags.DEFINE_string('shellscript_fname', None, 'Path to shell script.')
FLAGS = tf.app.flags.FLAGS


import sys
import contextlib
@contextlib.contextmanager
def pdb_post_mortem():
  try:
    yield
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
      import traceback
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


def retrieve_model(wrapped_model=None, model_name=None, placeholders=None, 
                   hparam_updates=None):
  # TODO: change function name to reflect the fact that it updates hparams
  """Builds graph, retrieves checkpoint, and returns wrapped model.

  This function either takes a basic_autofill_cnn_graph.TFModelWrapper object
  that already has the model graph or calls
  basic_autofill_cnn_graph.build_graph to return one. It then retrieves its
  weights from the checkpoint file specified in the
  hparams_tools.CHECKPOINT_HPARAMS dictionary.

  Returns:
    wrapped_model: A basic_autofill_cnn_graph.TFModelWrapper object that
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

  wrapped_model = basic_autofill_cnn_graph.build_graph(
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


CHECKPOINT_HPARAMS = {
    'balanced_by_scaling': Hyperparameters(
       separate_instruments=True,
       num_layers=64,
       num_filters=128,
       use_residual=True,
       mask_indicates_context=True,
       model_name='DeepStraightConvSpecs',
       checkpoint_name="balanced_by_scaling_64-128.ckpt",
    ),
    #'SmallTest': Hyperparameters(
    #    batch_size=2,
    #    num_layers=4,
    #    num_filters=8,
    #    model_name='DeepStraightConvSpecs')
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


def load_shellscript_as_dict(fpath):
  print 'Reading from', fpath
  with open(fpath, 'r') as p:
    lines = p.readlines()
  flags = {}
  for line in lines[4:]:
    print line
    parts = line.strip().split(' ')
    key = parts[0][2:]
    value = parts[1]
    print '\t', key, value
    # FIXME: hack to recover the type.
    try:
      value = int(value)
      if value == 0:
        value = False
      elif value == 1:
        value = True
    except ValueError:
        try:
          value = float(value)
        except ValueError:
          if value == 'True':
            value = True
          elif value == 'False':
            value = False
          else:
            print '%s is a string then.' % value
    flags[key] = value
  print 'Printing content of flags...'
  for key, value in flags.iteritems():
    print key, value 
  return flags
            

def shellscript_to_yaml():
  path = '/u/huangche/magenta-autofill/magenta/models/basic_autofill_cnn'
  fpath = os.path.join(path, FLAGS.shellscript_fname)
  flags = load_shellscript_as_dict(fpath)
  hparams = Hyperparameters(**flags)
  
  assert os.path.exists(FLAGS.checkpoint_dir)
  output_fpath = os.path.join(FLAGS.checkpoint_dir, 'config')
  print 'Writing to', output_fpath
  with open(output_fpath, 'w') as p:
    yaml.dump(hparams, p)
  
  # Check.
  print 'Reading from', output_fpath
  with open(output_fpath, 'r') as p:
    retrieved_hparams = yaml.load(p)
  assert str(hparams) == str(retrieved_hparams)


def main(unused_argv):
  shellscript_to_yaml()


if __name__ == "__main__":
  with pdb_post_mortem():
    tf.app.run()
