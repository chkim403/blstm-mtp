import tensorflow as tf

# Code Source: https://stackoverflow.com/questions/47745027/
class Wrapper(tf.nn.rnn_cell.RNNCell):
  def __init__(self, inner_cell):
     super(Wrapper, self).__init__()
     self._inner_cell = inner_cell

  @property
  def state_size(self):
     return self._inner_cell.state_size

  @property
  def output_size(self):
    return (self._inner_cell.state_size, self._inner_cell.output_size)

  def call(self, input, *args, **kwargs):
    output, next_state = self._inner_cell(input, *args, **kwargs)
    emit_output = (next_state, output)
    return emit_output, next_state