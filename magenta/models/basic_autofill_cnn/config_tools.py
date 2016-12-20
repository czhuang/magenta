"""Configurations for creating datasets, training and synthesizing results."""

from datetime import datetime

 

import tensorflow as tf

from magenta.models.basic_autofill_cnn import hparams_tools


class ConfigSpecificationError(Exception):
  pass


def get_checkpoint_config(model_name='DeepResidual'):
  """Creates a configuration for retrieving the stored checkpoint."""
  hparams = hparams_tools.get_checkpoint_hparams(model_name)
  config = PipelineConfig(hparams)
  return config


def get_current_time_as_str():
  return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


class PipelineConfig(object):
  """Pipeline config parameters, also storing hyperparameters."""

  # Settings for masking.
  maskout_border = 4  # 8-step length
  num_maskout = 4
  initial_maskout_factor = 0.001
  maskout_method_strs = ['random_instrument', 'random_patches',
                         'random_pitch_range', 'random_time_range',
                         'random_multiple_instrument_time',
                         'random_all_time_instrument',
                         'chronological_ti',
                         'chronological_it',
                         'fixed_order', 'balanced', 'balanced_by_scaling',
                         'no_mask']
                           
  RANDOM_INSTRUMENT, RANDOM_PATCHES, RANDOM_PITCH_RANGE = range(3)
  RANDOM_TIME_RANGE, RANDOM_MULTIPLE_INSTRUMENT_TIME = range(3, 5)
  RANDOM_ALL_TIME_INSTRUMENT = 5
  CHRONOLOGICAL_TI = 6
  CHRONOLOGICAL_IT = 7
  FIXED_ORDER = 8
  BALANCED = 9
  BALANCED_BY_SCALING = 10
  NO_MASK = 11
  # Run time configurations.
  eval_freq = 5

  # A identifier for each run.
  run_id = get_current_time_as_str()

  def __init__(self,
               hparams,
               maskout_method_str='random_multiple_instrument_time',
               separate_instruments=True,
               num_instruments=4):
    """Sets hyperparameters, mask out method and instrument representation."""

    if num_instruments != 4:
      raise ValueError('Only tested on num_instruments == 4.')
    self.hparams = hparams
    conv_specs_str = hparams.conv_arch.name

    # Update maskout method related settings.
    print 'maskout_method_str', maskout_method_str
    self.maskout_method = maskout_method_str
    # make this available on hparams too
    hparams.maskout_method = maskout_method_str
    # Maskout border needs to be smaller for random patches so that won't mask
    # out the entire pianoroll.
    if self.maskout_method == self.RANDOM_PATCHES:
      self.maskout_border = 3

    self.separate_instruments = separate_instruments

    # Log directory name for Tensorflow supervisor.
    self.log_subdir_str = '%s,%s,%s' % (conv_specs_str, str(hparams),
                                        self.run_id)
    tf.logging.info('Model Specification: %s', self.log_subdir_str)
    # TODO(annahuang): Remove.
    print 'Model Specification: %s' % self.log_subdir_str

  @property
  def maskout_method(self):
    return self._maskout_method

  @maskout_method.setter
  def maskout_method(self, maskout_method_str):
    """Sets the maskout_method attribute with its string's enumeration."""
    try:
      self._maskout_method = self.maskout_method_strs.index(maskout_method_str)
    except ValueError:
      raise ConfigSpecificationError('Unknown maskout_method %s.' %
                                     maskout_method_str)

  def get_maskout_method_str(self):
    return self.maskout_method_strs[self.maskout_method]
