"""Defines the graph for a convolutional net designed for music autofill."""

import tensorflow as tf


class ConvLayerParams(object):
  """Stores the params for each convolutional layer."""

  def __init__(self, weights, biases=None, gammas=None, betas=None):
    self.weights = weights
    self.biases = biases
    self.gammas = gammas
    self.betas = betas


def locally_connected_layer_2d_with_second_axis_shared(input_, filter_sizes):
  print 'locally_connected_layer_2d_with_second_axis_shared'
  filter_height, filter_width, num_in_channels, num_out_channels = filter_sizes
  input_shape = input_.get_shape().as_list()
  print 'input', input_shape
  print 'filter', filter_sizes
  batch, in_height, in_width = input_shape[0], input_shape[1], input_shape[2]
  input_pad = tf.pad(input_, [[0, 0], [1,1], [1,1], [0, 0]])
  W_shape = (filter_width, filter_width, 
             in_height, num_in_channels, num_out_channels)
  stddev = tf.sqrt(
      tf.div(2.0, tf.to_float(tf.reduce_prod(W_shape[:-1]))))
  W = tf.get_variable('locally_connected_weights', W_shape, 
                      initializer=tf.random_normal_initializer(0.0, stddev))
  end = (filter_width - 1) // 2
  start = - (filter_width - 1 - end) 
  #print start, end
  Y = tf.zeros((batch_size, in_height, in_width, num_out_channels))
  for dh in range(start, end+1):
    for dw in range(start, end+1):
      i = 1+dh
      j = 1+dw
      local_input = input_pad[:, i:i+in_height, j:j+in_width, :]
      #print tf.shape(W[i, j]).eval(), tf.shape(local_input).eval()
      # np.einsum('hio,bhwi->bhwo', W[i, j].eval(), local_X.eval())
      Y += tf.einsum('cef,bcde->bcdf', W[i, j], local_input)
  return Y, W


class BasicAutofillCNNGraph(object):
  """Model for predicting autofills given context."""

  def __init__(self, is_training, hparams, input_data,
               targets):  #, target_inspection_index):
    self.batch_size = hparams.batch_size
    self.num_pitches = hparams.num_pitches
    self.is_training = is_training
    self._input_data = input_data
    self._targets = targets
    self.prediction_threshold = hparams.prediction_threshold
    input_shape = tf.shape(self._input_data)
    conv_specs = hparams.conv_arch.specs
    num_conv_layers = len(conv_specs) - 1

    residual_period = 2
    locally_connected_period = 8
    locally_connected_stop_index = num_conv_layers - 5

    # Build convolutional layers.
    output = self._input_data
    if hparams.mask_indicates_context:
      def flip_mask(input):
        stuff, mask = tf.split(3, 2, output)
        return tf.concat(3, [stuff, 1 - mask])
      output = flip_mask(output)
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
          assert i == num_conv_layers
          continue

        if "filters" in specs:
          # Compute convolution.
          if 'pitch_locally_connected' in specs or (i != 0 and i % 8 == 0 and i < locally_connected_stop_index):
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

          # Compute batch normalization or add biases.
          num_target_filters = specs['filters'][-1]
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
            mean, variance = tf.nn.moments(conv, [0, 1, 2], keep_dims=True)
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

    # Compute total loss.
    self._logits = output

    # If treating each input instrument feature map as monophonic.
    if hparams.use_softmax_loss:
      self._logits = output
      softmax_2d = tf.nn.softmax(self.reshape_to_2d(self._logits))
      output_shape = [
          input_shape[0], input_shape[1], input_shape[2], input_shape[3] / 2
      ]

      self._predictions = self.reshape_back_to_4d(softmax_2d, output_shape)
      self._cross_entropy = -tf.log(self._predictions) * self._targets
    else:
      self._predictions = tf.sigmoid(self._logits)
      self._cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
          self._logits, self._targets)

    self._loss_total = tf.reduce_mean(self._cross_entropy)

    # Compute loss for masked portion.
    self._mask = tf.split(3, 2, self._input_data)[1]
    self._mask_size = tf.reduce_sum(self._mask)
    self._loss_mask = tf.reduce_sum(self._mask * self._cross_entropy) / (
        self._mask_size)

    # Compute loss for out-of-mask (unmask) portion.
    self._unmask = 1 - self._mask
    self._unmask_size = tf.reduce_sum(self._unmask)
    self._loss_unmask = tf.reduce_sum(self._unmask * self._cross_entropy) / (
        self._unmask_size)

    # Check which loss to use as objective function.
    if hparams.optimize_mask_only:
      self._loss = self._loss_mask
    else:
      self._loss = self._loss_total

    # If not training, don't need to add optimizer to the graph.
    if not is_training:
      self._train_op = tf.no_op
      return

    self._optimizer = tf.train.AdamOptimizer(
        learning_rate=hparams.learning_rate)
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
  def mask(self):
    return self._mask

  @property
  def mask_size(self):
    return self._mask_size

  @property
  def unmask_size(self):
    return self._unmask_size

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


def build_placeholders_initializers_graph(is_training, hparams):
  """Builds input and target placeholders, initializer, and training graph."""
  input_data = tf.placeholder(tf.float32, [None, None, None,
                                           hparams.input_depth])
  targets = tf.placeholder(tf.float32, [None, None, None,
                                        hparams.input_depth / 2])

  # Setup initializer.
  initializer = tf.random_uniform_initializer(-hparams.init_scale,
                                              hparams.init_scale)
  # Build training graph.
  with tf.variable_scope('model', reuse=None, initializer=initializer):
    train_model = BasicAutofillCNNGraph(
        is_training=is_training,
        hparams=hparams,
        input_data=input_data,
        targets=targets)
  return input_data, targets, initializer, train_model


class TFModelWrapper(object):
  """A Wrapper for passing model related and other configs as one object."""

  def __init__(self, model, graph, config):
    self.model = model
    self.graph = graph
    self.config = config
    self._sess = None

  @property
  def sess(self):
    return self._sess

  @sess.setter
  def sess(self, sess):
    self._sess = sess


def build_graph(is_training, config):
  """Build BasicAutofillCNNGraph, input output placeholders, and initializer."""
  graph = tf.Graph()
  with graph.as_default() as graph:
    _, _, _, model = build_placeholders_initializers_graph(
        is_training, config.hparams)
  return TFModelWrapper(model, graph, config)
