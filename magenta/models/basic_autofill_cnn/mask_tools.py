"""Tools for masking out pianorolls in different ways, such as by instrument."""
 
import numpy as np


class MaskUseError(Exception):
  pass


def apply_mask_and_stack(pianoroll, mask):
  """Stack pianorolls and masks on the last dimension.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll and mask stacked.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  masked_pianoroll = pianoroll * (1 - mask)
  return np.concatenate([masked_pianoroll, mask], 2)


def apply_mask(pianoroll, mask):
  """Apply mask to pianoroll.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  return pianoroll * (1 - mask)


def print_mask(mask):
  # assert mask is constant across pitch
  assert np.equal(mask, mask[:, 0, :][:, None, :]).all()
  # get rid of pitch dimension and transpose to get landscape orientation
  mask = mask[:, 0, :].T
  print "\n".join("".join(str({True: 1, False: 0}[z]) for z in y) for y in mask)


def get_mask(maskout_method, *args, **kwargs):
  mask_fn = globals()['get_%s_mask' % maskout_method]
  return mask_fn(*args, **kwargs)


def get_bernoulli_mask(pianoroll_shape, separate_instruments=True, 
                       blankout_ratio=0.5, **kwargs):
  """
  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  T, P, I = pianoroll_shape
  if separate_instruments:
    mask = np.random.random([T, 1, I]) < blankout_ratio
    mask = mask.astype(np.float32)
    mask = np.tile(mask, [1, pianoroll_shape[1], 1])
  else:
    mask = np.random.random([T, P, I]) < blankout_ratio
    mask = mask.astype(np.float32)
  return mask


def get_chronological_ti_mask(pianoroll_shape):
  # ti means the class of masks corresponds to the time-major ordering
  # over the time/instrument matrix, i.e. s1a1t1b1s2a2t2b2s3a3t3b3
  T, P, I = pianoroll_shape
  mask = np.ones([T * I]).astype(np.float32)
  j = np.random.randint(len(mask))
  mask[:j] = 0.
  mask = mask.reshape([T, 1, I])
  mask = np.tile(mask, [1, P, 1])
  return mask


def get_chronological_it_mask(pianoroll_shape):
  # it means the class of masks corresponds to the instrument-major
  # ordering over the time/instrument matrix,
  # i.e. s1s2s3s4...a1a2a3a4...t1t2t3t4...b1b2b3b4
  T, P, I = pianoroll_shape
  mask = np.ones([T * I]).astype(np.float32)
  j = np.random.randint(len(mask))
  mask[:j] = 0.
  mask = mask.reshape([I, 1, T])
  mask = mask.T
  mask = np.tile(mask, [1, P, 1])
  return mask


def get_fixed_order_order(num_timesteps):
  order = []
  for step in reversed(range(int(np.log2(num_timesteps)))):
    for j in reversed(range(0, num_timesteps, 2**step)):
      if j not in order:
        order.append(j)
  assert len(order) == num_timesteps
  return order


def get_fixed_order_mask(pianoroll_shape):
  T, P, I = pianoroll_shape
  order = get_fixed_order_order(T)
  t = np.random.randint(T)
  i = np.random.randint(I)
  mask = np.ones(pianoroll_shape)
  mask[order[:t]] = 0.
  mask[order[t], :, :i] = 0.
  return mask


def get_balanced_by_scaling_mask(pianoroll_shape, separate_instruments=True, **kwargs):
  T, P, I = pianoroll_shape

  if separate_instruments:
    d = T * I
  else:
    assert I == 1
    d = T * P 
  # sample a mask size
  k = np.random.choice(d) + 1
  # sample a mask of size k
  i = np.random.choice(d, size=k, replace=False)
 
  mask = np.zeros(d, dtype=np.float32)
  mask[i] = 1.
  if separate_instruments:
    mask = mask.reshape((T, 1, I))
    mask = np.tile(mask, [1, P, 1])
  else:
    mask = mask.reshape((T, P, 1))
  return mask
