"""Tools for masking out pianorolls in different ways, such as by instrument."""
 
import numpy as np


class MaskUseError(Exception):
  pass


def apply_mask_and_interleave(pianoroll, mask):
  """Depth concatenate pianorolls and masks by interleaving them.

  Args:
    pianoroll: A 3D binary matrix with 2D slices of pianorolls. This is not
        modified.
    mask: A 3D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
    A 3D binary matrix with masked pianoroll and masks interleaved.

  Raises:
    MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  masked_pianoroll = pianoroll * (1 - mask)
  timesteps, pitch_range, num_instruments = masked_pianoroll.shape
  pianoroll_and_mask = np.zeros(
      (timesteps, pitch_range, num_instruments * 2), dtype=np.float32)
  for instr_idx in range(num_instruments):
    pianoroll_and_mask[:, :, instr_idx * 2] = masked_pianoroll[:, :, instr_idx]
    pianoroll_and_mask[:, :, instr_idx * 2 + 1] = mask[:, :, instr_idx]
  return pianoroll_and_mask


def apply_mask_and_stack(pianoroll, mask, pad=False):
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
  if pianoroll.shape != mask.shape and not pad:
    raise MaskUseError('Shape mismatch in pianoroll and mask.')
  if pianoroll.shape[1:] != mask.shape[1:]: 
    raise MaskUseError('Shape mismatch in pianoroll and mask.')

  T, P, I = pianoroll.shape
  pad_length = T - mask.shape[0]
  assert np.sum(pianoroll[(T-pad_length):, :, :]) == 0
  mask = np.pad(mask, [(0, pad_length)] + [(0, 0)] * (pianoroll.ndim - 1), 
                mode="constant", constant_values=1)
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

def perturb_and_stack_alt(pianoroll, mask):
  """Alternative implementation."""
  if pianoroll.ndim != 3:
    raise ValueError(
      'Shape needs to be 3 dimensional, consisting of time, pitch, and instrument.')
  if pianoroll.shape != mask.shape:
    raise ValueError(
      'Mismatch in pianoroll and mask shape')
  # Mask gives which time steps to perturb, and then for each time step, randomly sample a pitch.
  # First, blankout those positions and then add the new sampled random pitch in.
  masked_pianoroll = pianoroll * (1 - mask)
  mask_without_pitch_dim = np.sum(mask, axis=1)
  masked_indices = np.where(mask_without_pitch_dim > 0)
  num_pitches = pianoroll.shape[1]
  num_of_masked_positions = int(np.sum(mask_without_pitch_dim > 0))
  sampled_pitches = np.random.randint(num_pitches, size=num_of_masked_positions)
  pianoroll[masked_indices[0], :, masked_indices[1]] = (
      sampled_pitches[:, None] == np.arange(num_pitches)[None, :])
  pianoroll = pianoroll.astype(np.float32)
  num_notes_on = np.unique(np.sum(pianoroll, axis=1))
  print 'pianoroll shape', pianoroll.shape
  print 'num_notes_on', num_notes_on
  assert np.allclose(num_notes_on, np.arange(2)) or (
      np.allclose(num_notes_on, np.array([1.])))
  # Check that every timestep has no more than 1 pitch.  Can use the check from pianorolls_lib
  return np.concatenate([masked_pianoroll, mask], 2)


def perturb_and_stack(pianoroll, mask):
  """Alternative implementation."""
  if pianoroll.ndim != 3:
    raise ValueError(
      'Shape needs to be 3 dimensional, consisting of time, pitch, and instrument.')
  if pianoroll.shape != mask.shape:
    raise ValueError(
      'Mismatch in pianoroll and mask shape')
  num_timesteps, num_pitches, num_instrs = pianoroll.shape
  categorical_pianoroll = np.random.randint(num_pitches, size=(num_timesteps, num_instrs))
  onehot_pianoroll = np.transpose(np.eye(num_pitches)[categorical_pianoroll], axes=[0, 2, 1])
  masked_pianoroll = (1 - mask) * pianoroll + mask * onehot_pianoroll
  # Check to make sure monophonic.  
  num_notes_on = np.unique(np.sum(masked_pianoroll, axis=1))
  print 'pianoroll shape', masked_pianoroll.shape
  print 'num_notes_on', num_notes_on
  assert np.allclose(num_notes_on, np.arange(2)) or (
      np.allclose(num_notes_on, np.array([1.])))
  # Check that every timestep has no more than 1 pitch.  Can use the check from pianorolls_lib
  return np.concatenate([masked_pianoroll, mask], 2)


def get_no_mask(pianoroll_shape):
  return np.zeros((painoroll_shape))


def get_random_instrument_mask(pianoroll_shape):
  """Creates a mask to mask out a random instrument.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.

  Returns:
    A 3D binary mask.
  """
  instr_idx = np.random.randint(pianoroll_shape[-1])
  return get_instrument_mask(pianoroll_shape, instr_idx)


def get_instrument_mask(pianoroll_shape, instr_idx):
  """Creates a mask to mask out the instrument at given index.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    instr_idx: An integer index indicating which instrument to be masked out.

  Returns:
    A 3D binary mask.
  """
  mask = np.zeros((pianoroll_shape))
  mask[:, :, instr_idx] = np.ones(pianoroll_shape[:2])
  return mask


def get_random_all_time_instrument_mask(pianoroll_shape, blankout_ratio=0.5):
  """
  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  T, P, I = pianoroll_shape
  mask = np.random.random([T, 1, I]) < blankout_ratio
  mask = mask.astype(np.float32)
  mask = np.tile(mask, [1, pianoroll_shape[1], 1])
  return mask


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


def print_mask(mask):
  # assert mask is constant across pitch
  assert np.equal(mask, mask[:, 0, :][:, None, :]).all()
  # get rid of pitch dimension and transpose to get landscape orientation
  mask = mask[:, 0, :].T
  print "\n".join("".join(str({True: 1, False: 0}[z]) for z in y) for y in mask)


def get_balanced_mask(pianoroll_shape):
  T, P, I = pianoroll_shape

  d = T * I
  coeffs = np.zeros(T * I + 1, dtype=np.float32)
  coeffs[1:] = 1. / np.arange(1, d + 1)
  pk = coeffs / np.sum(coeffs)

  # sample a mask size
  k = np.random.choice(d + 1, p=pk)
  # sample a mask of size k
  i = np.random.choice(d, size=k, replace=False)

  mask = np.zeros(T * I, dtype=np.float32)
  mask[i] = 1.
  mask = mask.reshape((T, 1, I))
  mask = np.tile(mask, [1, P, 1])
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


def get_multiple_random_patch_mask(pianoroll_shape, mask_border,
                                   initial_maskout_factor):
  """Creates a mask with multiple random patches to be masked out.

  This function first randomly selects locations in the pianoroll. The number
  of such selections is given by the initial_maskout_factor * size of pianoroll.
  The masked patches are then the bordered square patches around the selections.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border of the mask in number of cells.
    initial_maskout_factor: The initial percentage of how much mask locations to
        generate.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape, dtype=np.bool)
  random_inds = np.random.permutation(mask.size)
  num_initial_blankouts = int(np.ceil(mask.size * initial_maskout_factor))
  blankout_inds = np.unravel_index(random_inds[:num_initial_blankouts],
                                   mask.shape)
  mask[blankout_inds] = 1
  # Set up a different mask for each instrument.
  for instr_idx in range(pianoroll_shape[-1]):
    for axis in [0, 1]:
      # Shift the mask to make sure some longer notes are blanked out.
      for shifts in range(-mask_border, mask_border + 1):
        mask[:, :, instr_idx] += np.roll(
            mask[:, :, instr_idx], shifts, axis=axis)
  return mask.astype(np.float32)


def get_random_pitch_range_mask(pianoroll_shape, mask_border):
  """Creates a mask to mask out all time steps in a random pitch range.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border below and above a randomly choosen pitch for masking
        out a range of pitches.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  _, pitch_range, num_instruments = pianoroll_shape
  instr_idx = np.random.randint(num_instruments)
  random_pitch_center = np.random.randint(pitch_range)
  upper_pitch = random_pitch_center + mask_border + 1
  lower_pitch = random_pitch_center - mask_border
  if lower_pitch < 0:
    lower_pitch = 0
  mask[:, lower_pitch:upper_pitch, instr_idx] = 1
  return mask


def get_random_time_range_mask(pianoroll_shape, mask_border):
  """Mask out all notes in a random time range across all pitches.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  time_range, _, num_instruments = pianoroll_shape
  # Mask out only one intrument.
  instr_idx = np.random.randint(num_instruments)
  random_time_center = np.random.randint(time_range)
  for time_shift in range(-mask_border, mask_border):
    time_idx = (time_shift + random_time_center) % time_range
    mask[time_idx, :, instr_idx] = 1
  return mask


def get_random_instrument_time_mask(pianoroll_shape, timesteps, voices_for_mask_candidate=None):
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  time_range, num_pitches, num_instruments = pianoroll_shape
  # Mask out only one intrument.
  if voices_for_mask_candidate is None:
    voices_for_mask_candidate = range(num_instruments)
  instr_idx = np.random.choice(voices_for_mask_candidate)
  random_start_idx = np.random.randint(time_range)
  end_idx = random_start_idx + timesteps
  #print 'random_start_idx, end_idx', random_start_idx, end_idx 
  for time_idx in range(random_start_idx, end_idx):
    time_idx %= time_range
    mask[time_idx, :, instr_idx] = 1
  assert np.sum(mask) == timesteps * num_pitches
  return mask


def get_multiple_random_instrument_time_mask_by_mask_size(pianoroll_shape, mask_size, 
                                             num_maskout, voices_for_mask_candidate=None):
  """Mask out multiple random time ranges, randomly across instruments.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  for i in range(num_maskout):
    mask += get_random_instrument_time_mask(pianoroll_shape, mask_size, voices_for_mask_candidate)
  return np.clip(mask, 0, 1)


def get_multiple_random_instrument_time_mask(pianoroll_shape, mask_border,
                                             num_maskout, voices_for_mask_candidate=None):
  """Mask out multiple random time ranges, randomly across instruments.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  for i in range(num_maskout):
    mask += get_random_time_range_mask(pianoroll_shape, mask_border)
  return np.clip(mask, 0, 1)


def get_multiple_random_instrument_time_mask_next(pianoroll_shape, mask_border,
                                                  num_maskout):
  """Mask out middle of pianoroll, across all instruments.

  Args:
    pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
        should be 3D, with dimensions representing time, pitch, and instrument.
    mask_border: The border before and after a randomly choosen timestep to mask
        out.

  Returns:
    A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
    raise ValueError(
        'Shape needs to of 3 dimensional, time, pitch, and instrument.')
  mask = np.zeros(pianoroll_shape)
  num_timesteps = pianoroll_shape[0]
  one_fourth_duration = num_timesteps / 4
  end_index = num_timesteps - one_fourth_duration
  mask[one_fourth_duration:end_index, :, :] = 1
  return mask


def get_distribution_on_num_diff_instr():
  """Check distribution of number of instruments masked in random selection."""
  num_tries = 1000
  num_instrs = []
  for i in range(num_tries):
    instrs = set()
    for j in range(4):
      instrs.add(np.random.randint(4))
    num_instrs.append(len(instrs))
  hist = np.histogram(num_instrs, bins=range(1, 5))
  return hist
