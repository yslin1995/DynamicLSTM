import tensorflow as tf
import numpy as np

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
    b = tf.get_variable('b', [output_dim])
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def tdnn(input_, kernels, kernel_features, output_size, scope='TDNN'):
  assert len(kernels) == len(kernel_features)


  max_length = input_.get_shape()[1]
  embed_size = input_.get_shape()[-1]

  input_ = tf.expand_dims(input_,-1) # size (batch, time, embed, 1)

  layers = []
  with tf.variable_scope(scope):
    for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
      reduced_length = max_length - kernel_size + 1

      # [batch_size, time, embed, 1]
      conv = conv2d(input_, kernel_feature_size, kernel_size, embed_size, name="kernel_%d" % kernel_size)
      # conv=[batch_size, reduced_length, 1, kernel_feature_size]
      pool = tf.nn.max_pool(tf.nn.relu(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')
      # pool=[batch_size, 1, 1, kernel_feature_size]
      layers.append(tf.squeeze(pool)) # size num_kernel * (batch, kernel)

    if len(kernels) > 1:
      #[batch_size, sum of num_kernels]
      output = tf.concat(layers,1)
    else:
      output = layers[0]

  with tf.variable_scope(scope or "SimpleLinear"):
    matrix = tf.Variable(tf.random_uniform([300, output_size]))
    #matrix = tf.get_variable("Matrix", [300, output_size], dtype=input_.dtype)
    bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

  output_resize = tf.nn.sigmoid(tf.matmul(output, matrix) + bias_term) #is the sigmoid suitable?
  #output_resize = tf.Print(output_resize, [output_resize], message='output_resize', summarize=28)
  # deform_rate = output_resize / tf.expand_dims(tf.reduce_max(output_resize, axis=1),1)
  deform_rate = output_resize * tf.cast(tf.reshape(tf.tile(tf.range(output_size),[tf.shape(output_resize)[0]]),[-1,output_size]),dtype=tf.float32)
  #deform_rate = tf.Print(deform_rate, [deform_rate], message='deform_rate', summarize=28)
  return deform_rate



# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def lstm_ortho_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    size_x = shape[0]
    size_h = shape[1]/4 # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h])*scale
    t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
    t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
    t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
    return tf.constant(t, dtype)
  return _initializer

def layer_norm_all(h, batch_size, base, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3, use_bias=True):
  # Layer Norm (faster version, but not using defun)
  #
  # Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
  #
  # Reshapes h in to perform layer norm in parallel
  h_reshape = tf.reshape(h, [batch_size, base, num_units])
  mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
  var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
  epsilon = tf.constant(epsilon)
  rstd = tf.rsqrt(var + epsilon)
  h_reshape = (h_reshape - mean) * rstd
  # reshape back to original
  h = tf.reshape(h_reshape, [batch_size, base * num_units])
  with tf.variable_scope(scope):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable('ln_gamma', [4*num_units], initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable('ln_beta', [4*num_units], initializer=tf.constant_initializer(0.0))
  if use_bias:
    return gamma*h + beta
  return gamma * h

def layer_norm(x, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3, use_bias=True):
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  x_shifted = x-mean
  var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims=True)
  inv_std = tf.rsqrt(var + epsilon)
  with tf.variable_scope(scope):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable('ln_gamma', [num_units], initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable('ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
  output = gamma*(x_shifted)*inv_std
  if use_bias:
    output = output + beta
  return output

def super_linear(x, output_size, scope=None, reuse=False,
  init_w="ortho", weight_start=0.0, use_bias=True, bias_start=0.0, input_size=None):
  # support function doing linear operation.  uses ortho initializer defined earlier.
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    if reuse == True:
      tf.get_variable_scope().reuse_variables()

    w_init = None # uniform
    if input_size == None:
      x_size = shape[1]
    else:
      x_size = input_size
    h_size = output_size
    if init_w == "zeros":
      w_init=tf.constant_initializer(0.0)
    elif init_w == "constant":
      w_init=tf.constant_initializer(weight_start)
    elif init_w == "gaussian":
      w_init=tf.random_normal_initializer(stddev=weight_start)
    elif init_w == "ortho":
      w_init=lstm_ortho_initializer(1.0)

    w = tf.get_variable("super_linear_w",
      [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable("super_linear_b", [output_size], tf.float32,
        initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)

def hyper_norm(layer, hyper_output, embedding_size, num_units,
               scope="hyper", use_bias=True):
  '''
  HyperNetwork norm operator
  
  provides context-dependent weights
  layer: layer to apply operation on
  hyper_output: output of the hypernetwork cell at time t
  embedding_size: embedding size of the output vector (see paper)
  num_units: number of hidden units in main rnn
  '''
  # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
  init_gamma = 0.10 # cooijmans' da man.
  with tf.variable_scope(scope):
    zw = super_linear(hyper_output, embedding_size, init_w="constant",
      weight_start=0.00, use_bias=True, bias_start=1.0, scope="zw")
    alpha = super_linear(zw, num_units, init_w="constant",
      weight_start=init_gamma / embedding_size, use_bias=False, scope="alpha")
    result = tf.multiply(alpha, layer)
  return result

def hyper_bias(layer, hyper_output, embedding_size, num_units,
               scope="hyper"):
  '''
  HyperNetwork norm operator
  
  provides context-dependent bias
  layer: layer to apply operation on
  hyper_output: output of the hypernetwork cell at time t
  embedding_size: embedding size of the output vector (see paper)
  num_units: number of hidden units in main rnn
  '''

  with tf.variable_scope(scope):
    zb = super_linear(hyper_output, embedding_size, init_w="gaussian",
      weight_start=0.01, use_bias=False, bias_start=0.0, scope="zb")
    beta = super_linear(zb, num_units, init_w="constant",
      weight_start=0.00, use_bias=False, scope="beta")
  return layer + beta
  
class LSTMCell(tf.contrib.rnn.RNNCell):
  """
  Layer-Norm, with Ortho Initialization and
  Recurrent Dropout without Memory Loss.
  https://arxiv.org/abs/1607.06450 - Layer Norm
  https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss
  derived from
  https://github.com/OlavHN/bnlstm
  https://github.com/LeavesBreathe/tensorflow_with_latest_papers
  """

  def __init__(self, num_units, RL, forget_bias=1.0, use_layer_norm=False,
    use_recurrent_dropout=False, dropout_keep_prob=1.0):
    """Initialize the Layer Norm LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
      deform_rate:  a list of c_state and h_state location
      C_state: a list of c_state
      H_state: a list of h_state
      cur_time: the time_step of current input
      time_step: total time step
    """
    self.num_units = num_units
    self.RL = RL
    self.forget_bias = forget_bias
    self.use_layer_norm = use_layer_norm
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.cur_time =0

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.num_units, self.num_units)

  def get_newch(self, state, x, cur_time, RL):
    self.C_state = tf.concat([self.C_state[1:], tf.expand_dims(state[0],0)], 0) # need add self.C_state[1:]?
    self.H_state = tf.concat([self.H_state[1:], tf.expand_dims(state[1],0)], 0) # store the recent 5 states for using
    RL_input = tf.concat([x, tf.reduce_mean(self.H_state, 0)], 1)  # shape (batch, embedding)

    actions = self.RL.choose_action(RL_input, cur_time) # shape (batch_size, 1)

    coord = tf.concat([tf.cast(actions, tf.int32), tf.expand_dims(tf.range(tf.shape(actions)[0]),1)],1)  #(b_sz,2)

    self.c_state = tf.gather_nd(self.C_state, coord)
    self.h_state = tf.gather_nd(self.H_state, coord)

    return self.c_state, self.h_state


  def __call__(self, x, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      batch_size = x.get_shape().as_list()[0]
      x_size = x.get_shape().as_list()[1]

      cur_time = self.cur_time
      self.cur_time+=1
      # each batch clear the matrix
      if cur_time == 0:
        self.RL.actions=[]
        self.RL.act_prob=[]
        c, h = state
        self.RL.choose_action(tf.concat([x,h],1), cur_time)
        # each batch clear the matrix
        self.C_state = tf.reshape(tf.tile(tf.reshape(tf.zeros_like(c, dtype=tf.float32),[-1]),[self.RL.n_actions]), [self.RL.n_actions, batch_size, self.num_units])
        self.H_state = tf.reshape(tf.tile(tf.reshape(tf.zeros_like(h, dtype=tf.float32),[-1]),[self.RL.n_actions]), [self.RL.n_actions, batch_size, self.num_units])

      else:
        c, h = self.get_newch(state, x, cur_time, self.RL) #compute the new states

      h_size = self.num_units


      
      w_init=None # uniform

      h_init=lstm_ortho_initializer()

      W_xh = tf.get_variable('W_xh',
        [x_size, 4 * self.num_units], initializer=w_init)

      W_hh = tf.get_variable('W_hh_i',
        [self.num_units, 4*self.num_units], initializer=h_init)

      W_full = tf.concat([W_xh, W_hh], 0)

      bias = tf.get_variable('bias',
        [4 * self.num_units], initializer=tf.constant_initializer(0.0))

      concat = tf.concat([x, h], 1) # concat for speed.
      concat = tf.matmul(concat, W_full) + bias
      
      # new way of doing layer norm (faster)
      if self.use_layer_norm:
        concat = layer_norm_all(concat, batch_size, 4, self.num_units, 'ln')

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      if self.use_layer_norm:
        new_h = tf.tanh(layer_norm(new_c, self.num_units, 'ln_c')) * tf.sigmoid(o)
      else:
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
    
    return new_h, tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

class HyperLSTMCell(tf.contrib.rnn.RNNCell):
  '''
  HyperLSTM, with Ortho Initialization,
  Layer Norm and Recurrent Dropout without Memory Loss.
  
  https://arxiv.org/abs/1609.09106
  '''

  def __init__(self, num_units, forget_bias=1.0,
    use_recurrent_dropout=False, dropout_keep_prob=0.90, use_layer_norm=True,
    hyper_num_units=160, hyper_embedding_size=32,
    hyper_use_recurrent_dropout=False):
    '''Initialize the Layer Norm HyperLSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
      use_layer_norm: boolean. (default True)
        Controls whether we use LayerNorm layers in main LSTM and HyperLSTM cell.
      hyper_num_units: int, number of units in HyperLSTM cell.
        (default is 128, recommend experimenting with 256 for larger tasks)
      hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
        (default is 4, recommend trying larger values but larger is not always better)
      hyper_use_recurrent_dropout: boolean. (default False)
        Controls whether HyperLSTM cell also uses recurrent dropout. (Not in Paper.)
        Recommend turning this on only if hyper_num_units becomes very large (>= 512)
    '''
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.use_layer_norm = use_layer_norm
    self.hyper_num_units = hyper_num_units
    self.hyper_embedding_size = hyper_embedding_size
    self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

    self.total_num_units = self.num_units + self.hyper_num_units

    self.hyper_cell=LSTMCell(hyper_num_units,
                             use_recurrent_dropout=hyper_use_recurrent_dropout,
                             use_layer_norm=use_layer_norm,
                             dropout_keep_prob=dropout_keep_prob)

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.num_units+self.hyper_num_units,
                                         self.num_units+self.hyper_num_units)

  def __call__(self, input_all, state, timestep = 0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      input_main, input_hyper = tf.split(input_all,num_or_size_splits=[900,10],axis=1)
      total_c, total_h = state
      c = total_c[:, 0:self.num_units]
      h = total_h[:, 0:self.num_units]
      hyper_state = tf.contrib.rnn.LSTMStateTuple(total_c[:,self.num_units:],
                                                  total_h[:,self.num_units:])

      w_init=None # uniform

      h_init=lstm_ortho_initializer(1.0)
      
      x_size = input_main.get_shape().as_list()[1]
      embedding_size = self.hyper_embedding_size
      num_units = self.num_units
      batch_size = input_hyper.get_shape().as_list()[0]

      W_xh = tf.get_variable('W_xh',
        [x_size, 4*num_units], initializer=w_init)
      W_hh = tf.get_variable('W_hh',
        [num_units, 4*num_units], initializer=h_init)
      bias = tf.get_variable('bias',
        [4*num_units], initializer=tf.constant_initializer(0.0))

      # concatenate the input and hidden states for hyperlstm input
      # hyper_input = tf.concat([x, h], 1)  #zanshi bu lianjie h
      hyper_input = input_hyper
      hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)

      xh = tf.matmul(input_main, W_xh)
      hh = tf.matmul(h, W_hh)

      # split Wxh contributions
      ix, jx, fx, ox = tf.split(xh, 4, 1)
      ix = hyper_norm(ix, hyper_output, embedding_size, num_units, 'hyper_ix')
      jx = hyper_norm(jx, hyper_output, embedding_size, num_units, 'hyper_jx')
      fx = hyper_norm(fx, hyper_output, embedding_size, num_units, 'hyper_fx')
      ox = hyper_norm(ox, hyper_output, embedding_size, num_units, 'hyper_ox')

      # split Whh contributions
      ih, jh, fh, oh = tf.split(hh, 4, 1)
      ih = hyper_norm(ih, hyper_output, embedding_size, num_units, 'hyper_ih')
      jh = hyper_norm(jh, hyper_output, embedding_size, num_units, 'hyper_jh')
      fh = hyper_norm(fh, hyper_output, embedding_size, num_units, 'hyper_fh')
      oh = hyper_norm(oh, hyper_output, embedding_size, num_units, 'hyper_oh')

      # split bias
      ib, jb, fb, ob = tf.split(bias, 4, 0) # bias is to be broadcasted.
      ib = hyper_bias(ib, hyper_output, embedding_size, num_units, 'hyper_ib')
      jb = hyper_bias(jb, hyper_output, embedding_size, num_units, 'hyper_jb')
      fb = hyper_bias(fb, hyper_output, embedding_size, num_units, 'hyper_fb')
      ob = hyper_bias(ob, hyper_output, embedding_size, num_units, 'hyper_ob')

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i = ix + ih + ib
      j = jx + jh + jb
      f = fx + fh + fb
      o = ox + oh + ob

      if self.use_layer_norm:
        concat = tf.concat([i, j, f, o], 1)
        concat = layer_norm_all(concat, batch_size, 4, num_units, 'ln_all')
        i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j) 

      new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
      if self.use_layer_norm:
        new_h = tf.tanh(layer_norm(new_c, num_units, 'ln_c')) * tf.sigmoid(o)
      else:
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
    
      hyper_c, hyper_h = hyper_new_state
      new_total_c = tf.concat([new_c, hyper_c], 1)
      new_total_h = tf.concat([new_h, hyper_h], 1)

    return new_h, tf.contrib.rnn.LSTMStateTuple(new_total_c, new_total_h)
