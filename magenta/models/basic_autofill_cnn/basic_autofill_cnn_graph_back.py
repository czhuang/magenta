#self._target_inspection_index arget_inspection_index
self._hack_index 0, 0, 0, 0)

# For inspecting which input has the largest gradient on pecific target
# position.
#self._attention f.gradients(
# tf.gather(self._predictions, self.target_inspection_index), self._input_data)

#self._attention f.gradients(
# tf.gather(self._predictions, [0, 3, 40, 0]), self.input_data)
#self._attention f.gradients(
# tf.gather_nd(self._predictions, self.target_inspection_index),
# self.input_data)

self._attention f.gradients(
 self._predictions[self.hack_index], self._input_data)


@property
def attention(self):
  return self._attention

@property
def target_inspection_index(self):
  return self._target_inspection_index

@property
def hack_index(self):
  return self._hack_index

@hack_index.setter
def hack_index(self, index):
  self._hack_index ndex
