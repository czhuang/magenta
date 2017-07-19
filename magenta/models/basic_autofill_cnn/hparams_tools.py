"""Classes for defining hypermaters and model architectures."""
import itertools as it

class ModelMisspecificationError(Exception):
  """Exception for specifying a model that is not currently supported."""
  pass

class Hyperparameters(object):
  """Stores hyperparameters for initialization, batch norm and training."""
  _defaults = dict(
      # Data.
      dataset=None,
      quantization_level=0.125,
      shortest_duration=0.125,
      qpm=60,
      corrupt_ratio=0.25,
      # Input dimensions.
      batch_size=20,
      num_pitches=53,  #53 + 11
      pitch_ranges=[36, 81],
      
      crop_piece_len=64, #128, #64, #32,
      num_instruments=4,
      separate_instruments=False,
      #input_depth=None, #8,
      #output_depth=None, #4,
      # Batch norm parameters.
      batch_norm=True,
      batch_norm_variance_epsilon=1e-7,
      # Initialization.
      init_scale=0.1,
      # Model architecture.
      architecture=None,
      num_layers=28,
      num_filters=256,
      use_residual=True,
      checkpoint_name=None,
      # Loss setup.
      # TODO: currently maskout_method here is not functional, still need to go through config_tools.
      maskout_method='balanced_by_scaling',
      optimize_mask_only=False,
      #use_softmax_loss=True,
      rescale_loss=True,
      # Training.
      #learning_rate=2**-6,
      learning_rate=2**-4, #for sigmoids
      mask_indicates_context=False,
      eval_freq = 1,
      num_epochs = 0,
      patience = 5,
      # Runtime configs.
      run_dir=None,
      log_process=True,
      save_model_secs=30,
      run_id = '')

  def __init__(self, *args, **init_hparams):
    """Update the default parameters through string or keyword arguments.

    This __init__ provides two ways to initialize default parameters, either by
    passing a string representation of a a Python dictionary containing
    hyperparameter to value mapping or by passing those hyperparameter values
    directly as keyword arguments.

    Args:
      *args: A tuple of arguments. This first expected argument is a string
          representation of a Python dictionary containing hyperparameter
          to value mapping. For example, {"num_layers":8, "num_filters"=128}.
      **init_hparams: Keyword arguments for setting hyperparameters.

    """
    print 'Instantiating hparams:'
    unknown_params = set(init_hparams) - set(Hyperparameters._defaults)
    if unknown_params:
      raise ValueError('Unknown hyperparameters: %s', unknown_params)
    self.update(Hyperparameters._defaults)
    self.update(init_hparams)

  def update(self, dikt, **kwargs):
    for key, value in it.chain(dikt.iteritems(), kwargs.iteritems()):
      setattr(self, key, value)

  @property
  def input_depth(self):
    return 2 * (self.num_instruments if self.separate_instruments else 1)
  
  @property
  def output_depth(self):
    return self.num_instruments if self.separate_instruments else 1
  
  @property
  def conv_arch(self):
    return self.get_conv_arch()
  
  @property
  def log_subdir_str(self):
    return '%s_%s' % (self.conv_arch.name, self.__str__())
  
  @property
  def name(self):
    return self.conv_arch.name

  @property
  def input_shape(self):
    """Returns the shape of input data."""
    return [self.crop_piece_len, self.num_pitches, self.input_depth]

  @property
  def output_shape(self):
    """Returns the shape of output data."""
    return [self.crop_piece_len, self.num_pitches, self.output_depth]
  
  @property
  def raw_pianoroll_shape(self):
    """Returns the shape of raw pianorolls."""
    if self.separate_instruments:
      return [self.crop_piece_len, self.num_pitches, self.num_instruments]
    else:
      return [self.crop_piece_len, self.num_pitches, 1]

  @property
  def use_softmax_loss(self):
    if not self.separate_instruments and (
        self.num_instruments > 1 or self.num_instruments == 0):
      return False
    else: 
      return True

  def __str__(self):
    """Get all hyperparameters as a string."""
    param_keys = self.__dict__.keys() + ["use_softmax_loss", "num_pitches", "input_depth"]
    #param_keys = [key for key in dir(self) if '__' not in key and key != '_defaults']
    #print param_keys
    sorted_keys = sorted(param_keys)
    # Filter out some parameters so that string repr won't be too long for
    # directory name.
    # Want to show 'dataset', input_depth', and use_softmax_loss, learning rate, 'batch_size'
    blacklist = """
        num_layers num_filters eval_freq output_depth architecture checkpoint_name
        batch_norm_variance_epsilon batch_norm init_scale optimize_mask_only
        conv_arch mask_indicates_context run_dir num_epochs log_process
        save_model_secs batch_size input_depth num_instruments num_pitches
    """.split()
    lastlist = """maskout_method corrupt_ratio'""".split()
    shorthand = dict(
        batch_size='bs', learning_rate='lr', optimize_mask_only='mask_only',
        corrupt_ratio='corrupt', input_depth='in', crop_piece_len='len',
        use_softmax_loss='soft', num_instruments='num_i', num_pitches='n_pch',
        quantization_level='quant', use_residual='res',
        separate_instruments='sep', rescale_loss='rescale', 
        maskout_method='mm')

    def show_first(key):
      return (key not in blacklist and
              key not in lastlist)

    first_keys = [key for key in sorted_keys if show_first(key)]
    last_keys = [key for key in sorted_keys if key in lastlist]
    line = ','.join('%s=%s' % (shorthand.get(key, key), getattr(self, key))
                    for key in it.chain.from_iterable(first_keys + last_keys))
    return line

  def get_conv_arch(self):
    """Returns the model architecture."""
    return Architecture.make(
        self.architecture, 
        self.input_depth, self.num_layers, self.num_filters, 
        self.num_pitches, output_depth=self.output_depth)


class Architecture(util.Factory):
  pass


class Straight(Architecture):
  """A convolutional net where each layer has the same number of filters."""
  key = 'straight'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches, 
               output_depth, **kwargs):
    print self.key, input_depth, output_depth
    assert num_layers >= 4

    self.layers = []
    def _add(**kwargs):
      self.layers.append(kwargs)

    _add(filters=[3, 3, input_depth, num_filters])
    for _ in range(num_layers - 3):
      _add(filters=[3, 3, num_filters, num_filters])
    _add(filters=[2, 2, num_filters, num_filters])
    _add(filters=[2, 2, num_filters, output_depth],
        activation=util.identity)

    self.name = '%s-%d-%d' % (self.key, len(self.layers), num_filters)
  
  def __str__(self):
    #FIXME: a hack.
    return self.name
