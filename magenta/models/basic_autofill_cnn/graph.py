"""Defines the graph for a convolutional net designed for music autofill."""
import tensorflow as tf, numpy as np
from tensorflow.python.framework.function import Defun
from collections import OrderedDict
import lib.tfutil as tfutil


class BasicAutofillCNNGraph(object):
  """Model for predicting autofills given context."""

  def __init__(self, is_training, hparams, input_data,
               targets, lengths):  #, target_inspection_index):
    self.hparams = hparams
    self.batch_size = hparams.batch_size
    self.num_pitches = hparams.num_pitches
    self.num_instruments = hparams.num_instruments
    self.is_training = is_training
    self._input_data = input_data
    self._targets = targets
    self._lengths = lengths
    self._hiddens = []
    self.prediction_threshold = hparams.prediction_threshold
    self.popstats_by_batchstat = OrderedDict()
    self.build()

  def build(self):
    conv_specs = hparams.conv_arch.specs

    output = self.preprocess_input(self._input_data)
    self.residual_init()

    n = len(conv_specs)
    for i, specs in enumerate(conv_specs):
      with tf.variable_scope('conv%d' % i):
        self.residual_counter += 1
        self.residual_save(output)

        # Reshape output if moving into or out of being pitch fully connected.
        if specs.get('change_to_pitch_fully_connected', 0) == 1:
          bb, tt, pp, ii = tf.shape_n(output)
          output = tf.reshape(output, [bb, tt, 1, pp * ii])
          self.residual_reset()

        elif specs.get('change_to_pitch_fully_connected', 0) == -1:
          # assume it is the last layer and shape to the required output shape
          assert i == n - 1
          bb, tt, pp, _ = tf.shape_n(self._input_data)
          output = tf.reshape(output, [bb, tt, pp, self.num_instruments])
          self.residual_reset()
          continue

        output = self.apply_convolution(output, specs)
        output = self.apply_residual(output, is_first=i == 0, is_last=i == n - 1)
        output = self.apply_activation(output, specs)
        output = self.apply_pooling(output, specs)

        self._hiddens.append(output)

    self._logits = output
    self._predictions = self.compute_predictions(logits=self._logits, labels=self._targets)
    self._cross_entropy = self.compute_cross_entropy(logits=self._logits, labels=self._targets)

    self.compute_loss(cross_entropy)
    self.setup_optimizer()

  def preprocess_input(input_data):
    if self.hparams.mask_indicates_context:
      # flip meaning of mask for convnet purposes: after flipping, mask is hot
      # where values are known. this makes more sense in light of padding done
      # by convolution operations: the padded area will have zero mask,
      # indicating no information to rely on.
      def flip_mask(input):
        stuff, mask = tf.split(input, 2, axis=3)
        return tf.concat([stuff, 1 - mask], axis=3)
      input_data = flip_mask(input_data)

    # For denoising case, don't use masks in model
    if self.hparams.denoise_mode:
      input_data, _ = tf.split(input_data, 2, axis=3)

    return input_data

  def setup_optimizer(self):
    self.learning_rate = tf.Variable(self.hparams.learning_rate,
                                     name="learning_rate",
                                     trainable=False, dtype=tf.float32)

    # If not training, don't need to add optimizer to the graph.
    if not self.is_training:
      self._train_op = tf.no_op
      return

    # FIXME 0.5 -> hparams.decay_rate
    self.decay_op = tf.assign(self.learning_rate, 0.5*self.learning_rate)
    self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self._train_op = self._optimizer.minimize(self._loss)
    self._gradient_norms = [
        tf.sqrt(tf.reduce_sum(gradient[0]**2))
        for gradient in self._optimizer.compute_gradients(
            self._loss, var_list=tf.trainable_variables())
    ]

  def compute_loss(self, unreduced_loss):
    # construct mask due to identify zero padding that was introduced to make
    # the batch rectangular
    batch_duration = tf.shape(self._targets)[1]
    indices = tf.to_float(tf.range(batch_duration))
    self._pad_mask = tf.to_float(indices[None, :, None, None] <
                                 self.lengths[:, None, None, None])

    # construct mask and its complement, respecting pad mask
    self._mask = tf.split(self._input_data, 2, axis=3)[1]
    self._mask = self._mask * self._pad_mask
    self._unmask = (1 - self._mask) * self._pad_mask

    # Compute numbers of variables
    # #timesteps * #variables per timestep
    variable_axis = 3 if self.hparams.use_softmax_loss else 2
    D = (self.lengths[:, None, None, None] *
         tf.to_float(tf.shape(self._targets)[variable_axis]))
    reduced_D = tf.reduce_sum(D)

    # Compute numbers of variables to be predicted/conditioned on
    self._mask_size = tf.reduce_sum(
        self._mask, axis=[1, variable_axis], keep_dims=True)
    self._unmask_size = tf.reduce_sum(
        self._unmask, axis=[1, variable_axis], keep_dims=True)

    self._unreduced_loss = unreduced_loss * self._pad_mask
    if self.hparams.rescale_loss:
      self._unreduced_loss *= D / self._mask_size

    # Compute average loss over entire set of variables
    self._loss_total = tf.reduce_sum(self._unreduced_loss) / reduced_D

    # Compute loss for masked variables
    # NOTE: indexing the pitch dimension with 0 because the mask is constant
    # across pitch. Except in the sigmoid case, but then the pitch dimension
    # will have been reduced over.
    self._reduced_mask_size = tf.reduce_sum(self._mask_size[:, :, 0, :])
    self._loss_mask = (tf.reduce_sum(self._mask * self._unreduced_loss)
                       / self._reduced_mask_size)

    # Compute loss for out-of-mask (unmask) portion.
    self._reduced_unmask_size = tf.reduce_sum(self._unmask_size[:, :, 0, :])
    self._loss_unmask = (tf.reduce_sum(self._unmask * self._unreduced_loss)
                         / self._reduced_unmask_size)

    assert_partition_op = tf.group(
        tf.assert_equal(tf.reduce_sum(self._mask * self._unmask), 0),
        tf.assert_equal(self._reduced_mask_size + self._reduced_unmask_size,
                        reduced_D))
    with tf.control_dependencies([assert_partition_op]):
      self._loss_mask = tf.identity(self._loss_mask)
      self._loss_unmask = tf.identity(self._loss_unmask)

    # Check which loss to use as objective function.
    self._loss = (self._loss_mask if self.hparams.optimize_mask_only else
                  self._loss_total)

    # Use another loss altogether for particular maskout methods. In this case
    # all of the above work must be done anyway because things break if the
    # appropriate attributes are not available.
    self._maybe_compute_fixedorder_loss_instead()

  def maybe_compute_fixedorder_loss_instead(self):
    """Recompute and overwrite loss for special maskout methods."""
    if hparams.maskout_method not in "chronological_ti chronological_it fixed_order".split():
      self._lossmask = tf.no_op()
      return

    if self.hparams.rescale_loss:
      # If this is true then self._unreduced_losses will have been rescaled,
      # which biases the loss.
      raise ValueError("rescale_loss doesn't make sense with maskout_method %r"
                       % hparams.maskout_method)

    # These maskout methods correspond to fixed orderings and are incompatible
    # with the orderless NADE training procedure. Instead of training a whole
    # set of conditional distributions at once, we train exactly one. We use
    # the mask to determine which variable is next in the ordering.
    _, mask = tf.split(self._input_data, 2, axis=3)
    # If the variables are predicted in T, I order, then the number of known
    # variables corresponds to the index of the variable that would be sampled
    # next. Next, we transform this array to fit the maskout_method's ordering.
    flat_prediction_index = tf.to_int32(tf.reduce_sum(1 - mask[:, :, 0, :],
                                                      reduction_indices=(1, 2)))

    if hparams.maskout_method == "fixed_order":
      # Fixed order is a particular order in which the time steps are permuted
      # but the instruments are in order. We construct the permutation by
      # obtaining the time ordering and then expanding the array to squeeze in
      # the inner loop over instruments.
      time_order = mask_tools.get_fixed_order_order(hparams.crop_piece_len)
      flat_order = (self.num_instruments * np.array(time_order)[:, None]
                    + np.arange(self.num_instruments)[None, :])
      flat_order = flat_order.ravel()

      # Treat `flat_prediction_index` as an index into the permutation
      # `flat_order`. (`o_d` in NADE parlance.)
      flat_prediction_index = tf.gather(flat_order, flat_prediction_index)

    # Create a mask that has a single 1 for each example in the batch.
    batch_size, num_timesteps, _, num_instruments = tf.shape_n(mask)
    lossmask = tf.one_hot(flat_prediction_index,
                          depth=num_timesteps * num_instruments)
    # Get rid of that pesky pitch dimension.
    loss3d = tf.reduce_sum(self._unreduced_loss, reduction_indices=2)

    if hparams.maskout_method == "chronological_it":
      # This ordering loops over time first; permute the dimensions to match.
      loss3d = tf.transpose(loss3d, perm=[0, 2, 1])

    # Flatten the loss to match lossmask and reduce.
    flatloss = tf.reshape(loss3d, [batch_size, num_timesteps * num_instruments])
    self._loss = tf.reduce_sum(lossmask * flatloss) / tf.to_float(batch_size)
    self._lossmask = lossmask

  def residual_init(self):
    if not self.hparams.use_residual:
      return
    self.residual_period = 2
    self.output_for_residual = None
    # TODO figure out why this is initialized to -1
    self.residual_counter = -1

  def residual_reset(self):
    self.output_for_residual = None
    self.residual_counter = 0

  def residual_save(self, x):
    if not self.hparams.use_residual:
      return
    if self.residual_counter % self.residual_period == 1:
      self.output_for_residual = x

  def apply_residual(self, x, is_first, is_last):
    if not self.hparams.use_residual:
      return x
    if self.output_for_residual is None:
      return x
    if self.output_for_residual.get_shape()[-1] != x.get_shape()[-1]:
      # shape mismatch; e.g. change in number of filters
      self.residual_reset()
      return x
    if self.residual_counter % self.residual_period == 0:
      if not is_first and not is_last:
        x += self.output_for_residual
    return x

  def apply_convolution(self, x, specs):
    if "filters" not in specs:
      return x

    filter_shape = specs["filters"]
    # Instantiate or retrieve filter weights.
    fanin = tf.to_float(tf.reduce_prod(filter_shape[:-1]))
    stddev = tf.sqrt(tf.div(2.0, fanin))
    weights = tf.get_variable(
        'weights', filter_shape,
        initializer=tf.random_normal_initializer(0.0, stddev))
    stride = specs.get('conv_stride', 1)
    conv = tf.nn.conv2d(x, weights,
                        strides=[1, stride, stride, 1],
                        padding=specs.get('conv_pad', 'SAME'))

    # Compute batch normalization or add biases.
    if hparams.batch_norm:
      y = self.apply_batchnorm(conv)
    else:
      biases = tf.get_variable('bias', [conv.get_shape()[-1]],
                               initializer=tf.constant_initializer(0.0))
      y = tf.nn.bias_add(conv, biases)
    return y

  def apply_batchnorm(self, x):
    output_dim = x.get_shape()[-1]
    gammas = tf.get_variable('gamma', [1, 1, 1, output_dim],
                             initializer=tf.constant_initializer(1.))
    betas = tf.get_variable('beta', [output_dim],
                            initializer=tf.constant_initializer(0.))

    popmean = tf.get_variable(
        "popmean", shape=[1, 1, 1, output_dim], trainable=False,
        collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
        initializer=tf.constant_initializer(0.0))
    popvariance = tf.get_variable(
        "popvariance", shape=[1, 1, 1, output_dim], trainable=False,
        collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
        initializer=tf.constant_initializer(1.0))
    batchmean, batchvariance = tf.nn.moments(conv, [0, 1, 2], keep_dims=True)

    decay = 0.01
    if self.is_training:
      mean, variance = batchmean, batchvariance
      updates = [popmean.assign_sub(decay * (popmean - mean)),
                 popvariance.assign_sub(decay * (popvariance - variance))]
      # make update happen when mean/variance are used
      with tf.control_dependencies(updates):
        mean, variance = tf.identity(mean), tf.identity(variance)
    else:
      mean, variance = popmean, popvariance
      mean, variance = batchmean, batchvariance

    self.popstats_by_batchstat[batchmean] = popmean
    self.popstats_by_batchstat[batchvariance] = popvariance

    return tf.nn.batch_normalization(
        x, mean, variance, betas, gammas,
        self.hparams.batch_norm_variance_epsilon)

  def apply_activation(self, x, specs):
    activation_func = specs.get('activation', tf.nn.relu)
    return activation_func(x)

  def apply_pooling(self, x, specs):
    if 'pooling' not in specs:
      return x
    pooling = specs['pooling']
    return tf.nn.max_pool(
        x,
        ksize=[1, pooling[0], pooling[1], 1],
        strides=[1, pooling[0], pooling[1], 1],
        padding=specs['pool_pad'])

  def compute_predictions(self, logits):
      return (tf.nn.softmax(logits, dim=2)
              if self.hparams.use_softmax_loss else
              tf.nn.sigmoid(logits))

  def compute_cross_entropy(self, logits, labels):
      return (tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=labels,
                                                      dim=2)
              if self.hparams.use_softmax_loss else
              tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                      labels=labels))

  @property
  def gradient_norms(self):
    return self._gradient_norms

  @property
  def input_data(self):
    return self._input_data

  @property
  def lengths(self):
    return self._lengths

  @property
  def mask(self):
    return self._mask

  @property
  def reduced_mask_size(self):
    return self._reduced_mask_size

  @property
  def reduced_unmask_size(self):
    return self._reduced_unmask_size

  @property
  def targets(self):
    return self._targets

  @property
  def train_op(self):
    return self._train_op

  @property
  def logits(self):
    return self._logits

  @property
  def predictions(self):
    return self._predictions

  @property
  def loss_mask(self):
    return self._loss_mask

  @property
  def loss_unmask(self):
    return self._loss_unmask

  @property
  def loss_total(self):
    return self._loss_total

  @property
  def loss(self):
    return self._loss

def get_placeholders(hparams):
  # NOTE: fixed batch_size because einstein sum can only deal with up to 1 unknown dimension
  return dict(input_data=tf.placeholder(tf.float32, [None, None] + hparams.input_shape[-2:]),
              targets=tf.placeholder(tf.float32, [None, None] + hparams.output_shape[-2:]),
              lengths=tf.placeholder(tf.float32, [None]))

def build_placeholders_initializers_graph(is_training, hparams, placeholders=None):
  """Builds input and target placeholders, initializer, and training graph."""
  if placeholders is None:
    placeholders = get_placeholders(hparams)

  # Setup initializer.
  initializer = tf.random_uniform_initializer(-hparams.init_scale,
                                              hparams.init_scale)
  # Build training graph.
  with tf.variable_scope('model', reuse=None, initializer=initializer):
    train_model = BasicAutofillCNNGraph(
        is_training=is_training,
        hparams=hparams,
        **placeholders)
  return placeholders["input_data"], placeholders["targets"], placeholders["lengths"], initializer, train_model


def build_graph(is_training, hparams, placeholders=None):
  """Build BasicAutofillCNNGraph, input output placeholders, and initializer."""
  _, _, _, _, model = build_placeholders_initializers_graph(
      is_training, hparams, placeholders=placeholders)
  return tfutil.WrappedModel(model, model.loss.graph, hparams)
