"""Classes for defining hypermaters and model architectures."""


class ModelMisspecificationError(Exception):
  """Exception for specifying a model that is not currently supported."""
  pass


class Hyperparameters(object):
  """Stores hyperparameters for initialization, batch norm and training."""
  _defaults = dict(
      # Data augmentation.
      augment_by_transposing=0,
      augment_by_halfing_doubling_durations=0,
      corrupt_ratio=0.25,
      # Input dimensions.
      batch_size=20,
      num_pitches=53,  #53 + 11
      crop_piece_len=64, #128, #64, #32,
      num_instruments=4,
      separate_instruments=True,
      input_depth=None, #8,
      output_depth=None, #4,
      # Batch norm parameters.
      batch_norm=True,
      batch_norm_variance_epsilon=1e-7,
      batch_norm_gamma=0.1,
      # Initialization.
      init_scale=0.1,
      # Model architecture.
      num_layers=28,
      num_filters=256,
      model_name=None,
      checkpoint_name=None,
      use_residual=True,
      denoise_mode=False,
      # Loss setup.
      optimize_mask_only=False,
      use_softmax_loss=True,
      # Training.
      learning_rate=2**-6,
      mask_indicates_context=False,
      # Prediction threshold.
      prediction_threshold=0.5)

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
    unknown_params = set(init_hparams) - set(Hyperparameters._defaults)
    if unknown_params:
      raise ValueError('Unknown hyperparameters: %s', unknown_params)

    # Update instance with default class variables.
    for key, value in Hyperparameters._defaults.items():
      if key in init_hparams:
        value = init_hparams[key]
      setattr(self, key, value)

    if self.separate_instruments:
      self.input_depth = self.num_instruments * 2
    else:
      self.input_depth = 1 * 2
    self.output_depth = self.input_depth // 2    

    # Check if pitch range is expanded if data augmentation on pitch is desired.
    if self.augment_by_transposing and self.num_pitches != 53 + 11:
      self.num_pitches = 53 + 11
      #raise ValueError("num_pitches should be 53 + 11 if transposing")

    # If denoising mode, then masks are not feed into model.
    if self.denoise_mode:
      self.input_depth //= 2
      if self.input_depth != self.output_depth:
        raise ValueError('Legacy from before, Input depth needs to be twice as output depth initially to account for mask that is not used in convolution but for computing debug statistics.')
    else:
      if self.input_depth // 2 != self.output_depth:
        raise ValueError('Output depth should be half of that of input.')

    self.conv_arch = self.get_conv_arch()

  @property
  def input_data_shape(self):
    """Returns the shape of input data."""
    return (self.crop_piece_len, self.num_pitches, self.input_depth)

  @property
  def name(self):
    return self.conv_arch.name

  def __str__(self):
    """Get all hyperparameters as a string."""
    param_keys = self.__dict__.keys()
    print param_keys
    sorted_keys = sorted(param_keys)
    # Filter out some parameters so that string repr won't be too long for
    # directory name.
    keys_to_filter_out = [
        'batch_size', 'use_softmax_loss', 'instr_sep', 'border', 'num_layers',
        'input_depth', 'output_depth', 'model_name',
        'batch_norm_variance_epsilon', 'batch_norm_gamma', 'batch_norm',
        'init_scale', 'crop_piece_len', 'learning_rate',
        'prediction_threshold', 'optimize_mask_only', 'conv_arch',
        'augment_by_halfing_doubling_durations', 'augment_by_transposing',
        'mask_indicates_context',
    ]
    # Want to show 'input_depth'
    keys_to_filter_out = [
        'batch_size', 'use_softmax_loss', 'border', 'num_layers',
        'output_depth', 'model_name',
        'batch_norm_variance_epsilon', 'batch_norm_gamma', 'batch_norm',
        'init_scale', 'crop_piece_len', 'learning_rate',
        'prediction_threshold', 'optimize_mask_only', 'conv_arch',
        'augment_by_halfing_doubling_durations', 'augment_by_transposing',
        'mask_indicates_context',
    ]
    return (','.join('%s=%s' % (key, self.__dict__[key]) for key in sorted_keys
                     if key not in keys_to_filter_out))

  def get_conv_arch(self):
    """Returns the model architecture."""
    if self.model_name == 'PitchLocallyConnectedConvSpecs':
      return PitchLocallyConnectedConvSpecs(
          self.input_depth, self.num_layers, self.num_filters, self.num_pitches, self.output_depth)

    if self.model_name == 'PitchFullyConnectedConvSpecs':
      return globals()[self.model_name](self.input_depth, self.num_layers, self.num_filters, self.num_pitches, self.output_depth)

    if self.model_name == 'DeepStraightConvSpecs':
      return DeepStraightConvSpecs(self.input_depth, self.num_layers,
                                   self.num_filters, self.num_pitches, self.output_depth)
    elif self.model_name == 'DeepStraightConvSpecsWithEmbedding':
      return DeepStraightConvSpecsWithEmbedding(self.input_depth,
                                                self.num_layers,
                                                self.num_filters,
                                                self.num_pitches)
    elif self.model_name == 'PitchFullyConnected':
      return PitchFullyConnected(self.input_depth, self.num_layers,
                                 self.num_filters, self.num_pitches)
    elif self.model_name == 'PitchFullyConnectedWithResidual':
      return PitchFullyConnectedWithResidual(self.input_depth, self.num_layers,
                                             self.num_filters, self.num_pitches)
    else:
      raise ModelMisspecificationError('Model name does not exist.')


class ConvArchitecture(object):
  """Parent class for convnet architectures in condensed representations."""

  def __init__(self):
    self.condensed_specs = None

  def get_spec(self):
    """Expand the expanded convnet architeciture."""
    conv_specs_expanded = []
    for layer in self.condensed_specs:
      if isinstance(layer, tuple):
        for _ in range(layer[0]):
          conv_specs_expanded.append(layer[1])
      else:
        conv_specs_expanded.append(layer)
    return conv_specs_expanded


class DeepStraightConvSpecs(ConvArchitecture):
  """A convolutional net where each layer has the same number of filters."""
  model_name = 'DeepStraightConvSpecs'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches, output_depth):
    print self.model_name, input_depth, output_depth
    if num_layers < 4:
      raise ModelMisspecificationError(
          'The network needs to be at least 4 layers deep, %d given.' %
          num_layers)
    super(DeepStraightConvSpecs, self).__init__()
    self.condensed_specs = [
        dict(
            filters=[3, 3, input_depth, num_filters],
            conv_stride=1,
            conv_pad='SAME'), (num_layers - 3, dict(
                filters=[3, 3, num_filters, num_filters],
                conv_stride=1,
                conv_pad='SAME')), dict(
                    filters=[2, 2, num_filters, num_filters],
                    conv_stride=1,
                    conv_pad='SAME'), dict(
                        filters=[2, 2, num_filters, output_depth],
                        conv_stride=1,
                        conv_pad='SAME',
                        activation=lambda x: x)
    ]
    self.specs = self.get_spec()
    assert self.specs
    if input_depth != 2:
      self.name_prefix = '%s-multi_instr' % self.model_name
    else:
      self.name_prefix = '%s-col_instr' % self.model_name
    self.name = '%s_depth-%d_filter-%d-%d' % (self.name_prefix, len(self.specs),
                                              num_filters, num_filters)


class PitchLocallyConnectedConvSpecs(ConvArchitecture):
  """A convolutional net where each layer has the same number of filters."""
  model_name = 'PitchLocallyConnectedConvSpecs'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches, output_depth):
    num_instruments = output_depth
    if num_layers < 4:
      raise ModelMisspecificationError(
          'The network needs to be at least 4 layers deep, %d given.' %
          num_layers)
    super(PitchLocallyConnectedConvSpecs, self).__init__()
    bottom = [dict(filters=[3, 3, input_depth, num_filters])]
    middle = []
    for i in range(num_layers - 4):
      middle.append(dict(filters=[3, 3, num_filters, num_filters],
                         pitch_locally_connected=i % 8 == 7))
    top = [dict(filters=[3, 3, num_filters, num_instruments], pitch_locally_connected = True),
           dict(filters=[3, 3, num_instruments, num_instruments], pitch_locally_connected = True),
           dict(change_to_pitch_fully_connected=1,
                filters=[3, 1, num_pitches * num_instruments, num_pitches * num_instruments],
                activation=lambda x: x),
           dict(change_to_pitch_fully_connected=-1, activation=lambda x: x)]
    self.condensed_specs = bottom + middle + top
    # -1 because the last layer is just a reshape
    assert len(self.condensed_specs) - 1 == num_layers
    self.specs = self.get_spec()
    assert self.specs
    if input_depth != 2:
      self.name_prefix = '%s-multi_instr' % self.model_name
    else:
      self.name_prefix = '%s-col_instr' % self.model_name
    self.name = '%s_depth-%d_filter-%d-%d' % (self.name_prefix, len(self.specs),
                                              num_filters, num_filters)

class PitchFullyConnectedConvSpecs(ConvArchitecture):
  """A convolutional net where each layer has the same number of filters."""
  model_name = 'PitchFullyConnectedConvSpecs'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches, output_depth):
    num_instruments = output_depth
    if num_layers < 4:
      raise ModelMisspecificationError(
          'The network needs to be at least 4 layers deep, %d given.' %
          num_layers)
    super(PitchFullyConnectedConvSpecs, self).__init__()
    bottom = [dict(filters=[3, 3, input_depth, num_filters])]
    middle = []
    for i in range(num_layers - 2):
      middle.append(dict(filters=[3, 3, num_filters, num_filters]))
    top = [dict(change_to_pitch_fully_connected=1, activation=lambda x: x),
           dict(filters=[1, 1, num_pitches * num_filters, num_pitches * num_instruments],
                activation=lambda x: x),
           dict(change_to_pitch_fully_connected=-1, activation=lambda x: x)]
    self.condensed_specs = bottom + middle + top
    # -2 because two layers are just reshapes
    assert len(self.condensed_specs) - 2 == num_layers
    self.specs = self.get_spec()
    assert self.specs
    if input_depth != 2:
      self.name_prefix = '%s-multi_instr' % self.model_name
    else:
      self.name_prefix = '%s-col_instr' % self.model_name
    self.name = '%s_depth-%d_filter-%d-%d' % (self.name_prefix, len(self.specs),
                                              num_filters, num_filters)



class DeepStraightConvSpecsWithEmbedding(ConvArchitecture):
  """A convolutional net where each layer has the same number of filters."""
  model_name = 'DeepStraightConvSpecsWithEmbedding'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches):
    if num_layers < 4:
      raise ModelMisspecificationError(
          'The network needs to be at least 4 layers deep, %d given.' %
          num_layers)
    super(DeepStraightConvSpecsWithEmbedding, self).__init__()
    self.condensed_specs = [
        dict(
            filters=[5, num_pitches - 12, input_depth, 64],
            conv_stride=1,
            conv_pad='SAME'), dict(
                filters=[3, 3, 64, 256], conv_stride=1, conv_pad='SAME'),
        (num_layers - 4, dict(
            filters=[3, 3, num_filters, num_filters],
            conv_stride=1,
            conv_pad='SAME')), dict(
                filters=[2, 2, num_filters, num_filters],
                conv_stride=1,
                conv_pad='SAME'), dict(
                    filters=[2, 2, num_filters, input_depth / 2],
                    conv_stride=1,
                    conv_pad='SAME',
                    activation=lambda x: x)
    ]
    self.specs = self.get_spec()
    assert self.specs
    if input_depth != 2:
      self.name_prefix = '%s-multi_instr' % self.model_name
    else:
      self.name_prefix = '%s-col_instr' % self.model_name
    first_layer_str = ','.join(str(size_) for size_ in self.specs[0]['filters'])
    self.name = '%s_d-%d_f-%d_1st_%s' % (self.name_prefix, len(self.specs),
                                         num_filters, first_layer_str)
    print 'len of name:', len(self.name)


class PitchFullyConnected(ConvArchitecture):
  """A convolutional net where each layer has the same number of filters."""
  model_name = 'PitchFullyConnected'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches):
    if num_layers < 4:
      raise ModelMisspecificationError(
          'The network needs to be at least 4 layers deep, %d given.' %
          num_layers)
    super(PitchFullyConnected, self).__init__()

    reduced_depth_size = num_filters / 4
    expanded_depth_size = reduced_depth_size * num_pitches

    self.condensed_specs = [
        dict(
            filters=[5, 12, input_depth, num_filters],
            conv_stride=1,
            conv_pad='SAME'),
        (num_layers - 6, dict(
            filters=[3, 3, num_filters, num_filters],
            conv_stride=1,
            conv_pad='SAME')),
        dict(
            filters=[3, 3, num_filters, reduced_depth_size],
            conv_stride=1,
            conv_pad='SAME'),
        # Feature maps are being reshaped before performing the convolution.
        dict(
            change_to_pitch_fully_connected=1,
            filters=[3, 1, expanded_depth_size, expanded_depth_size],
            conv_stride=1,
            conv_pad='SAME'),
        (1, dict(
            filters=[3, 1, expanded_depth_size, expanded_depth_size],
            conv_stride=1,
            conv_pad='SAME')),
        # Feature maps are being reshaped before performing the convolution.
        dict(
            change_to_pitch_fully_connected=-1,
            filters=[2, 2, reduced_depth_size, num_filters],
            conv_stride=1,
            conv_pad='SAME'),
        dict(
            filters=[2, 2, num_filters, input_depth / 2],
            conv_stride=1,
            conv_pad='SAME',
            activation=lambda x: x)
    ]

    self.specs = self.get_spec()
    assert self.specs
    if input_depth != 2:
      self.name_prefix = '%s-multi_instr' % self.model_name
    else:
      self.name_prefix = '%s-col_instr' % self.model_name
    self.name = '%s_depth-%d_filter-%d-%d' % (self.name_prefix, len(self.specs),
                                              num_filters, num_filters)


class PitchFullyConnectedWithResidual(ConvArchitecture):
  """A convolutional net where each layer has the same number of filters."""
  model_name = 'PitchFullyConnected'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches):
    if num_layers < 4:
      raise ModelMisspecificationError(
          'The network needs to be at least 4 layers deep, %d given.' %
          num_layers)
    super(PitchFullyConnectedWithResidual, self).__init__()

    reduced_depth_size = num_filters / 4
    expanded_depth_size = reduced_depth_size * num_pitches

    self.condensed_specs = [
        dict(
            filters=[5, 12, input_depth, num_filters],
            conv_stride=1,
            conv_pad='SAME'),
        (num_layers - 6, dict(
            filters=[3, 3, num_filters, num_filters],
            conv_stride=1,
            conv_pad='SAME')),
        dict(
            filters=[3, 3, num_filters, reduced_depth_size],
            conv_stride=1,
            conv_pad='SAME'),
        # Feature maps are being reshaped before performing the convolution.
        dict(
            change_to_pitch_fully_connected=1,
            filters=[3, 1, expanded_depth_size, num_filters],
            conv_stride=1,
            conv_pad='SAME'),
        (2, dict(
            filters=[3, 1, num_filters, num_filters],
            conv_stride=1,
            conv_pad='SAME')),
        # Feature maps are being reshaped before performing the convolution.
        #dict(change_to_pitch_fully_connected=-1,
        #     filters=[2, 2, num_filters, num_filters], conv_stride=1,
        #     conv_pad='SAME'),
        dict(
            filters=[2, 1, num_filters, num_pitches * input_depth / 2],
            conv_stride=1,
            conv_pad='SAME',
            activation=lambda x: x),
        dict(change_to_pitch_fully_connected=-1)
    ]

    self.specs = self.get_spec()
    assert self.specs
    if input_depth != 2:
      self.name_prefix = '%s-multi_instr' % self.model_name
    else:
      self.name_prefix = '%s-col_instr' % self.model_name
    self.name = '%s_depth-%d_filter-%d-%d' % (self.name_prefix, len(self.specs),
                                              num_filters, num_filters)


CHECKPOINT_HPARAMS = {
    'DeepResidual': Hyperparameters(
        num_layers=28,
        num_filters=256,
        use_residual=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='DeepStraightConvSpecs-with_res-multi_instr_depth-28_filter-256-256-best_model.ckpt'
    ),
    'DeepResidual32_256': Hyperparameters(
        num_layers=32,
        num_filters=256,
        use_residual=True,
        mask_indicates_context=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='DeepStraightConvSpecs_d-32_f-256_best_model.ckpt',
    ),
    'DeepResidual64_128': Hyperparameters(
        num_layers=64,
        num_filters=128,
        use_residual=True,
        mask_indicates_context=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='DeepStraightConvSpecs_d-64_f-128_best_model.ckpt',
    ),
 
    'DeepResidualRandomMaskTBF': Hyperparameters(
        num_layers=64,
        num_filters=128,
        use_residual=True,
        mask_indicates_context=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name=''
    ),
    'Denoising64_128': Hyperparameters(
        num_layers=64,
        num_filters=128,
        use_residual=True,
        mask_indicates_context=True,
        denoise_mode=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='denoising_d-64_f-128.ckpt'
    ),
    'Denoising32_256': Hyperparameters(
        num_layers=32,
        num_filters=256,
        use_residual=True,
        mask_indicates_context=True,
        denoise_mode=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='',
    ),

   'random_medium': Hyperparameters(
        num_layers=64,
        num_filters=128,
        use_residual=True,
        mask_indicates_context=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name="random_64-128_0.5.ckpt",
    ),

    'balanced_fc_mask_only': Hyperparameters(
        num_layers=64,
        num_filters=128,
        use_residual=True,
        mask_indicates_context=True,
        model_name='PitchFullyConnectedConvSpecs',
        checkpoint_name="balanced_64-128_fc_mask_only.ckpt",
    ),
    'balanced_by_scaling': Hyperparameters(
       num_layers=64,
       num_filters=128,
       use_residual=True,
       mask_indicates_context=True,
       model_name='DeepStraightConvSpecs',
       checkpoint_name="balanced_by_scaling_64-128.ckpt",
    ),
    'DeepResidualDataAug': Hyperparameters(
        num_layers=28,
        num_filters=256,
        augment_by_transposing=1,
        augment_by_halfing_doubling_durations=1,
        num_pitches=53 + 11,
        use_residual=True,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='DeepStraightConvSpecs-with_res-with_aug-multi_instr_depth-28_filter-256-256-best_model.ckpt'
    ),
    'PitchFullyConnectedWithResidual': Hyperparameters(
        num_layers=28,
        num_filters=256,
        model_name='PitchFullyConnectedWithResidual',
        checkpoint_name='PitchFullyConnected-multi_instr_depth-29_filter-256-256-best_model.ckpt'
    ),
    'DeepStraight': Hyperparameters(
        num_layers=28,
        num_filters=256,
        use_residual=False,
        model_name='DeepStraightConvSpecs',
        checkpoint_name='DeepStraightConvSpecs-multi_instr_depth-28_filter-256-256-best_model.ckpt'
    ),
    'SmallTest': Hyperparameters(
        batch_size=2,
        num_layers=4,
        num_filters=8,
        model_name='DeepStraightConvSpecs')
}


def get_checkpoint_hparams(model_name):
  """Returns the model architecture."""
  if model_name not in CHECKPOINT_HPARAMS:
    raise ModelMisspecificationError('Model name %s does not exist.' % model_name)
  else:
    return CHECKPOINT_HPARAMS[model_name]
