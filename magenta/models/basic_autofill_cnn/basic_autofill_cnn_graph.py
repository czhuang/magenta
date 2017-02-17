"""Defines the graph for a convolutional net designed for music autofill."""
import tensorflow as tf, numpy as np
from tensorflow.python.framework.function import Defun
from collections import OrderedDict


class ConvLayerParams(object):
  """Stores the params for each convolutional layer."""

  def __init__(self, weights, biases=None, gammas=None, betas=None):
    self.weights = weights
    self.biases = biases
    self.gammas = gammas
    self.betas = betas


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
    self.batch_size = hparams.batch_size
    self.num_pitches = hparams.num_pitches
    self.is_training = is_training
    self._input_data = input_data
    self._targets = targets
    self._lengths = lengths
    self.prediction_threshold = hparams.prediction_threshold
    input_shape = tf.shape(self._input_data)
    output_depth = input_shape[3] / 2
    conv_specs = hparams.conv_arch.specs
    num_conv_layers = len(conv_specs) - 1

    residual_period = 2

    # Build convolutional layers.
    output = self._input_data
    if hparams.mask_indicates_context:
      def flip_mask(input):
        stuff, mask = tf.split(output, 2, axis=3)
        return tf.concat([stuff, 1 - mask], axis=3)
      output = flip_mask(output)

    # For denoising case, don't use masks in model
    if hparams.denoise_mode:
      output = tf.split(output, 2, axis=3)[0]
      input_shape = tf.shape(output)

    self.popstats_by_batchstat = OrderedDict()

    self._hiddens = []

    output_for_residual = None
    residual_counter = -1
    for i, specs in enumerate(conv_specs):
      with tf.variable_scope('conv%d' % i):
        residual_counter += 1
        # Save output from last layer for residual connections for odd layers.
        if hparams.use_residual and residual_counter % residual_period == 1:
          output_for_residual = tf.identity(output)

        # Reshape output if moving into or out of being pitch fully connected.
        if specs.get('change_to_pitch_fully_connected', 0) == 1:
          output_shape = tf.shape(output)
          output = tf.reshape(output, [output_shape[0], output_shape[1], 1,
                                       output_shape[2] * output_shape[3]])
          # When crossing from pitch not fully connected to yes, clear
          # kept output for residual.
          output_for_residual = None
          # Also reset residual counter.
          residual_counter = 0

        elif specs.get('change_to_pitch_fully_connected', 0) == -1:
          output = tf.reshape(
              output, [input_shape[0], input_shape[1], input_shape[2], -1])
          # When switching back pitch to not fully connected, also clear
          # kept output for residual as already quite close to the softmax
          # layer.
          output_for_residual = None
          # Needs to be the layer about the last to do the reshaping
          assert specs is conv_specs[-1]
          continue

        if "filters" in specs:
          # Compute convolution.
          if specs.get('pitch_locally_connected', False):
            # Weight instantiation and initialization is wrapped inside.
            conv, weights = locally_connected_layer_2d_with_second_axis_shared(
                output, specs['filters'])
            layer = ConvLayerParams(weights)
          else:
            # Instantiate or retrieve filter weights.
            stddev = tf.sqrt(
                tf.div(2.0, tf.to_float(tf.reduce_prod(specs['filters'][:-1]))))
            weights = tf.get_variable(
                'weights',
                specs['filters'],
                initializer=tf.random_normal_initializer(0.0, stddev))
            layer = ConvLayerParams(weights)
            stride = specs.get('conv_stride', 1)
            conv = tf.nn.conv2d(
                output,
                layer.weights,
                strides=[1, stride, stride, 1],
                padding=specs.get('conv_pad', 'SAME'))

          num_source_filters, num_target_filters = specs['filters'][-2:]
          if num_target_filters != num_source_filters:
            output_for_residual = None
            residual_counter = 0

          # Compute batch normalization or add biases.
          if not hparams.batch_norm:
            layer.biases = tf.get_variable(
                'bias', [num_target_filters],
                initializer=tf.constant_initializer(0.0))
            output = tf.nn.bias_add(conv, layer.biases)
          else:
            layer.gammas = tf.get_variable(
                'gamma', [1, 1, 1, num_target_filters],
                initializer=tf.constant_initializer(hparams.batch_norm_gamma))
            layer.betas = tf.get_variable(
                'beta', [num_target_filters],
                initializer=tf.constant_initializer(0.0))
            layer.popmean = tf.get_variable(
                "popmean", shape=[1, 1, 1, num_target_filters], trainable=False,
                collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
                initializer=tf.constant_initializer(0.0))
            layer.popvariance = tf.get_variable(
                "popvariance", shape=[1, 1, 1, num_target_filters], trainable=False,
                collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
                initializer=tf.constant_initializer(1.0))
            layer.batchmean, layer.batchvariance = tf.nn.moments(conv, [0, 1, 2], keep_dims=True)
            decay = 0.01
            if self.is_training:
              mean, variance = layer.batchmean, layer.batchvariance
              updates = [layer.popmean.assign_sub(decay * (layer.popmean - mean)),
                         layer.popvariance.assign_sub(decay * (layer.popvariance - variance))]
              # make update happen when mean/variance are used
              with tf.control_dependencies(updates):
                mean, variance = tf.identity(mean), tf.identity(variance)
            else:
              if hparams.use_pop_stats:
                mean, variance = layer.popmean, layer.popvariance
              else:
                mean, variance = layer.batchmean, layer.batchvariance

            self.popstats_by_batchstat[layer.batchmean] = layer.popmean
            self.popstats_by_batchstat[layer.batchvariance] = layer.popvariance

            output = tf.nn.batch_normalization(
                conv, mean, variance, layer.betas, layer.gammas,
                hparams.batch_norm_variance_epsilon)

        # Sum residual before nonlinearity if odd layer and residual exist.
        if hparams.use_residual and output_for_residual is not None and (
            i > 0 and i < num_conv_layers and
            residual_counter % residual_period == 0):
          # Was going to use residual_counter > 0 instead of i > 0,
          # but didn't since when residual is reset back to 0
          # output_for_residual is also set to None, which is already checked.
          output += output_for_residual

        # Pass through nonlinearity, except for the last layer.
        activation_func = specs.get('activation', tf.nn.relu)
        output = activation_func(output)

        # Perform pooling layer if specified in specs.
        if 'pooling' in specs:
          pooling = specs['pooling']
          output = tf.nn.max_pool(
              output,
              ksize=[1, pooling[0], pooling[1], 1],
              strides=[1, pooling[0], pooling[1], 1],
              padding=specs['pool_pad'])

        self._hiddens.append(output)

    # Compute total loss.
    self._logits = output

    # If treating each input instrument feature map as monophonic.
    if hparams.use_softmax_loss:
      self._logits = output
      softmax_2d = tf.nn.softmax(self.reshape_to_2d(self._logits))
      output_shape = [
          input_shape[0], input_shape[1], input_shape[2], output_depth
      ]

      self._predictions = self.reshape_back_to_4d(softmax_2d, output_shape)
      self._cross_entropy = -tf.log(self._predictions) * self._targets
    else:
      self._predictions = tf.sigmoid(self._logits)
      self._cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self._logits, labels=self._targets)

    self._unreduced_loss = self._cross_entropy

    # Adjust loss to not include padding part.
    shape = tf.shape(self._targets)
    mask = tf.split(self._input_data, 2, axis=3)[1]
    non_pad_indicators = tf.to_float(tf.range(shape[1])[None, :, None, None]) < (
        self.lengths[:, None, None, None])
    non_pad_indicators = tf.to_float(non_pad_indicators)
    non_pad_mask = mask * non_pad_indicators
    #TODO: Make sure the padded parts are zeroed out.
    self._unreduced_loss *= non_pad_indicators

    if hparams.use_softmax_loss:
      # #timesteps * #instruments
      D = self.lengths[:, None, None, None] * tf.to_float(shape[3])
      reduced_D = tf.reduce_sum(self.lengths) * tf.to_float(shape[3])
      # #masked out variables
      self._mask_size = tf.reduce_sum(
          non_pad_mask, reduction_indices=[1, 3], keep_dims=True)
    else:
      # #timesteps * #pitches
      D = self.lengths[:, None, None, None] * tf.to_float(shape[2])
      reduced_D = tf.reduce_sum(self.lengths) * tf.to_float(shape[2])
      # #masked out variables
      self._mask_size = tf.reduce_sum(
          non_pad_mask, reduction_indices=[1, 2], keep_dims=True)

    self.D = D
    self.reduced_D = reduced_D

    if hparams.rescale_loss:
      def compute_scale():
        return D / self._mask_size
      self._unreduced_loss *= compute_scale()

    # Compute total loss.
    self._loss_total = tf.reduce_sum(self._unreduced_loss) / reduced_D

    # Compute loss for masked portion.
    self._mask = tf.split(self._input_data, 2, axis=3)[1]
    self._reduced_mask_size = tf.reduce_sum(self._mask_size[:, :, 0, :])
    self._loss_mask = tf.reduce_sum(self._mask * self._unreduced_loss) / (
        self._reduced_mask_size)

    # Compute loss for out-of-mask (unmask) portion.
    self._unmask = 1 - self._mask
    self._unmask *= non_pad_indicators 
    self._reduced_unmask_size = reduced_D - self._reduced_mask_size
    tf.assert_equal(
        self._reduced_unmask_size, tf.reduce_sum(self._unmask[:, :, 0, :]))
    self._loss_unmask = tf.reduce_sum(
        self._unmask * self._unreduced_loss) / self._reduced_unmask_size

    # Check which loss to use as objective function.
    if hparams.optimize_mask_only:
      self._loss = self._loss_mask
    else:
      self._loss = self._loss_total

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
      # reduce_mean over pitch to be consistent with above
      loss3d = tf.reduce_mean(self._unreduced_loss, reduction_indices=2)

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


class TFModelWrapper(object):
  """A Wrapper for passing model related and other configs as one object."""

  def __init__(self, model, graph, hparams):
    self.model = model
    self.graph = graph
    self.hparams = hparams
    self._sess = None

  @property
  def sess(self):
    return self._sess

  @sess.setter
  def sess(self, sess):
    self._sess = sess


def build_graph(is_training, hparams, placeholders=None):
  """Build BasicAutofillCNNGraph, input output placeholders, and initializer."""
  _, _, _, _, model = build_placeholders_initializers_graph(
      is_training, hparams, placeholders=placeholders)
  return TFModelWrapper(model, model.loss.graph, hparams)
