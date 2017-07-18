"""Defines the graph for a convolutional net designed for music autofill."""
import tensorflow as tf, numpy as np
from tensorflow.python.framework.function import Defun
from collections import OrderedDict
import lib.tfutil as tfutil


def locally_connected_layer_2d_with_second_axis_shared(input_, filter_sizes):
  print 'locally_connected_layer_2d_with_second_axis_shared'
  filter_time_size, filter_pitch_size, num_in_channels, num_out_channels = filter_sizes
  print 'filter', filter_sizes
  assert filter_time_size == filter_pitch_size
  batch_size = input_.get_shape().as_list()[0]
  num_timesteps = tf.shape(input_)[1]
  num_pitches = input_.get_shape().as_list()[2]
  W_shape = (filter_time_size, filter_pitch_size, 
             num_pitches, num_in_channels, num_out_channels)
  print 'W_shape', W_shape, W_shape[:-1] 
  stddev = tf.sqrt(tf.div(2.0, tf.to_float(tf.reduce_prod(W_shape[:-1]))))
  W = tf.get_variable('locally_connected_weights', W_shape, 
                      initializer=tf.random_normal_initializer(0.0, stddev))
  # Defun needs tensor, and can't accept variables
  W = tf.convert_to_tensor(W)
 
  #@Defun(tf.float32, tf.float32)
  def compute_pitch_locally_connected_convolution(input_, W):
    # Defun input looses shape.
    W.set_shape(W_shape)
    num_timesteps = tf.shape(input_)[1]
    input_pad = tf.pad(input_, [[0, 0], [1,1], [1,1], [0, 0]])
    end = (filter_pitch_size - 1) // 2
    start = - (filter_pitch_size - 1 - end) 
    #print start, end
    Y = tf.zeros((batch_size, num_timesteps, num_pitches, num_out_channels))
    for dh in range(start, end+1):
      for dw in range(start, end+1):
        i = 1+dh
        j = 1+dw
        local_input = input_pad[:, i:i+num_timesteps, j:j+num_pitches, :]
        local_input.set_shape([batch_size, None, num_pitches, num_in_channels])
        #print tf.shape(W[i, j]).eval(), tf.shape(local_input).eval()
        # np.einsum('pio,btpi->btpw', W[i, j].eval(), local_X.eval())
        Y += tf.einsum('def,bcde->bcdf', W[i, j], local_input)
    return Y

  func_name = "localpitch_" + "_".join(map(str, np.random.choice(100, 10)))
  do_it = Defun(tf.float32, tf.float32, func_name=func_name)(lambda input_, W: compute_pitch_locally_connected_convolution(input_, W))

  #Y = compute_pitch_locally_connected_convolution(input_, W)
  Y = do_it(input_, W)
  # Defun output looses shape.
  Y.set_shape((batch_size, None, num_pitches, num_out_channels))
  return Y, W


class BasicAutofillCNNGraph(object):
  """Model for predicting autofills given context."""

  def __init__(self, is_training, hparams, input_data,
               targets, lengths):  #, target_inspection_index):
    self.hparams = hparams
    self.batch_size = hparams.batch_size
    self.num_pitches = hparams.num_pitches
    self.is_training = is_training
    self._input_data = input_data
    self._targets = targets
    self._lengths = lengths
    self._hiddens = []
    self.prediction_threshold = hparams.prediction_threshold
    self.popstats_by_batchstat = OrderedDict()
    self.build()

  def build(self):
    sym_batch_size, sym_batch_duration, sym_num_pitches, sym_num_instruments = tf.shape_n(self._targets)
    input_shape = tf.shape(self._input_data)
    conv_specs = hparams.conv_arch.specs

    output = self._input_data
    if hparams.mask_indicates_context:
      # flip meaning of mask for convnet purposes: after flipping, mask is hot
      # where values are known. this makes more sense in light of padding done
      # by convolution operations: the padded area will have zero mask,
      # indicating no information to rely on.
      def flip_mask(input):
        stuff, mask = tf.split(output, 2, axis=3)
        return tf.concat([stuff, 1 - mask], axis=3)
      output = flip_mask(output)

    # For denoising case, don't use masks in model
    if hparams.denoise_mode:
      output = tf.split(output, 2, axis=3)[0]
      input_shape = tf.shape(output)

    self.residual_init()

    n = len(conv_specs)
    for i, specs in enumerate(conv_specs):
      with tf.variable_scope('conv%d' % i):
        self.residual_counter += 1
        self.residual_save(output)

        # Reshape output if moving into or out of being pitch fully connected.
        if specs.get('change_to_pitch_fully_connected', 0) == 1:
          output_shape = tf.shape(output)
          output = tf.reshape(output, [output_shape[0], output_shape[1], 1,
                                       output_shape[2] * output_shape[3]])
          self.residual_reset()

        elif specs.get('change_to_pitch_fully_connected', 0) == -1:
          output = tf.reshape(
              output, [input_shape[0], input_shape[1], input_shape[2], -1])
          self.residual_reset()
          # Needs to be the layer about the last to do the reshaping
          assert specs is conv_specs[-1]
          continue

        output = self.apply_convolution(output, specs)
        output = self.apply_residual(output, is_first=i == 0, is_last=i == n - 1)
        output = self.apply_activation(output, specs)
        output = self.apply_pooling(output, specs)

        self._hiddens.append(output)

    self._logits = output
    self._predictions = self.compute_predictions(logits=self._logits, labels=self._targets)
    self._cross_entropy = self.compute_cross_entropy(logits=self._logits, labels=self._targets)

    # Adjust loss to not include padding part.
    indices = tf.to_float(tf.range(sym_batch_duration))
    pad_mask = tf.to_float(indices[None, :, None, None] <
                           self.lengths[:, None, None, None])
    self._cross_entropy *= pad_mask

    # Compute numbers of variables to be predicted
    # #timesteps * #variables per timestep
    variable_axis = 3 if hparams.use_softmax_loss else 2
    D = (self.lengths[:, None, None, None] *
         tf.to_float(tf.shape(self._targets)[variable_axis]))
    reduced_D = tf.reduce_sum(D)
    self._mask = tf.split(self._input_data, 2, axis=3)[1]
    self._mask_size = tf.reduce_sum(
        mask * pad_mask, axis=[1, variable_axis], keep_dims=True)

    self._unreduced_loss = self._cross_entropy
    if hparams.rescale_loss:
      self._unreduced_loss *= D / self._mask_size

    # Compute total loss.
    self._loss_total = tf.reduce_sum(self._unreduced_loss) / reduced_D

    # Compute loss for masked portion.
    self._reduced_mask_size = tf.reduce_sum(self._mask_size[:, :, 0, :])
    self._loss_mask = (tf.reduce_sum(self._mask * self._unreduced_loss)
                       / self._reduced_mask_size)

    # Compute loss for out-of-mask (unmask) portion.
    self._unmask = (1 - self._mask) * pad_mask
    self._reduced_unmask_size = tf.reduce_sum(self._unmask[:, :, 0, :])
    self._loss_unmask = (tf.reduce_sum(self._unmask * self._unreduced_loss)
                         / self._reduced_unmask_size)

    check_unmask_count_equal_op = tf.assert_equal(
        self._reduced_unmask_size, reduced_D - self._reduced_mask_size)
    with tf.control_dependencies([check_unmask_count_equal_op]):
      self._loss_unmask = tf.identity(self._loss_unmask)

    # Check which loss to use as objective function.
    self._loss = (self._loss_mask if hparams.optimize_mask_only else
                  self._loss_total)

    # FIXME put this ugly stuff into a big ugly method
    if "chronological" in hparams.maskout_method or "fixed_order" in hparams.maskout_method:
      _, mask = tf.split(self._input_data, 2, axis=3)
      flat_prediction_index = tf.to_int32(tf.reduce_sum(1 - mask[:, :, 0, :],
                                                        reduction_indices=(1, 2)))

      if "fixed_order" in hparams.maskout_method:
        num_instruments = 4
        import mask_tools, numpy as np
        time_order = mask_tools.get_fixed_order_order(hparams.crop_piece_len)
        flat_order = num_instruments * np.array(time_order)[:, None] + np.arange(num_instruments)[None, :]
        flat_order = flat_order.ravel()
        flat_prediction_index = tf.gather(flat_order, flat_prediction_index)

      lossmask = tf.one_hot(flat_prediction_index, depth=tf.shape(mask)[1] * tf.shape(mask)[3])
      loss3d = tf.reduce_sum(self._unreduced_loss, reduction_indices=2)

      if hparams.maskout_method.endswith("_ti") or "fixed_order" in hparams.maskout_method:
        pass
      elif hparams.maskout_method.endswith("_it"):
        loss3d = tf.transpose(loss3d, perm=[0, 2, 1])
      else:
        print hparams.maskout_method
        assert False
      self._lossmask = lossmask
      flatloss = tf.reshape(loss3d, (tf.shape(mask)[0], tf.shape(mask)[1] * tf.shape(mask)[3]))
      self._loss = tf.reduce_sum(lossmask * flatloss) / tf.to_float(tf.shape(mask)[0])
    else:
      self._lossmask = tf.no_op()

    self.learning_rate = tf.Variable(hparams.learning_rate, name="learning_rate", trainable=False, dtype=tf.float32)

    # If not training, don't need to add optimizer to the graph.
    if not is_training:
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

  def reshape_to_2d(self, data):
    # Collapse the batch, time, and instrument dimension of a tensor that has a
    # shape of (batch, time, pitch, instrument) into 2D.
    transposed_data = tf.transpose(data, perm=[0, 1, 3, 2])
    return tf.reshape(transposed_data, [-1, tf.shape(data)[2]])

  def reshape_back_to_4d(self, data_2d, shape):
    reshaped_data = tf.reshape(data_2d, [-1, shape[1], shape[3], shape[2]])
    return tf.transpose(reshaped_data, [0, 1, 3, 2])

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
    if specs.get('pitch_locally_connected', False):
      # Weight instantiation and initialization is wrapped inside.
      conv, weights = locally_connected_layer_2d_with_second_axis_shared(
          output, filter_shape)
    else:
      # Instantiate or retrieve filter weights.
      fanin = tf.to_float(tf.reduce_prod(filter_shape[:-1]))
      stddev = tf.sqrt(tf.div(2.0, fanin))
      weights = tf.get_variable(
          'weights', specs['filters'],
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
