"""Tools for masking out pianorolls in different ways, such as by instrument."""


 
import numpy as np


class MaskUseError(Exception):
  pass


def apply_mask_and_interleave(pianoroll, mask):
  """Depth concatenate pianorolls and masks by interleaving them.

  Args:
 pianoroll: D binary matrix with 2D slices of pianorolls. This is not
  modified.
 mask: D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
 A 3D binary matrix with masked pianoroll and masks interleaved.

  Raises:
 MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
 raise MaskUseError('Shape mismatch in pianoroll and mask.')
  masked_pianoroll ianoroll 1 ask)
  timesteps, pitch_range, num_instruments asked_pianoroll.shape
  pianoroll_and_mask p.zeros(
   (timesteps, pitch_range, num_instruments ), dtype=np.float32)
  for instr_idx in range(num_instruments):
 pianoroll_and_mask[:, :, instr_idx ] asked_pianoroll[:, :, instr_idx]
 pianoroll_and_mask[:, :, instr_idx  ] ask[:, :, instr_idx]
  return pianoroll_and_mask


def apply_mask_and_stack(pianoroll, mask):
  """Stack pianorolls and masks on the last dimension.

  Args:
 pianoroll: D binary matrix with 2D slices of pianorolls. This is not
  modified.
 mask: D binary matrix with 2D slices of masks, one per each pianoroll.

  Returns:
 A 3D binary matrix with masked pianoroll and mask stacked.

  Raises:
 MaskUseError: If the shape of pianoroll and mask do not match.
  """
  if pianoroll.shape != mask.shape:
 raise MaskUseError('Shape mismatch in pianoroll and mask.')
  masked_pianoroll ianoroll 1 ask)
  return np.concatenate([masked_pianoroll, mask], 2)


def get_random_instrument_mask(pianoroll_shape):
  """Creates ask to mask out andom instrument.

  Args:
 pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
  should be 3D, with dimensions representing time, pitch, and instrument.

  Returns:
 A 3D binary mask.
  """
  instr_idx p.random.randint(pianoroll_shape[-1])
  return get_instrument_mask(pianoroll_shape, instr_idx)


def get_instrument_mask(pianoroll_shape, instr_idx):
  """Creates ask to mask out the instrument at given index.

  Args:
 pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
  should be 3D, with dimensions representing time, pitch, and instrument.
 instr_idx: An integer index indicating which instrument to be masked out.

  Returns:
 A 3D binary mask.
  """
  mask p.zeros((pianoroll_shape))
  mask[:, :, instr_idx] p.ones(pianoroll_shape[:2])
  return mask


def get_multiple_random_patch_mask(pianoroll_shape, mask_border,
         nitial_maskout_factor):
  """Creates ask with multiple random patches to be masked out.

  This function first randomly selects locations in the pianoroll. The number
  of such selections is given by the initial_maskout_factor ize of pianoroll.
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
    'Shape needs to of imensional, time, pitch, and instrument.')
  mask p.zeros(pianoroll_shape, dtype=np.bool)
  random_inds p.random.permutation(mask.size)
  num_initial_blankouts nt(np.ceil(mask.size nitial_maskout_factor))
  blankout_inds p.unravel_index(random_inds[:num_initial_blankouts],
         ask.shape)
  mask[blankout_inds] 
  et up ifferent mask for each instrument.
  for instr_idx in range(pianoroll_shape[-1]):
 for axis in [0, 1]:
   hift the mask to make sure some longer notes are blanked out.
   for shifts in range(-mask_border, mask_border ):
  mask[:, :, instr_idx] += np.roll(
   mask[:, :, instr_idx], shifts, axis=axis)
  return mask.astype(np.float32)


def get_random_pitch_range_mask(pianoroll_shape, mask_border):
  """Creates ask to mask out all time steps in andom pitch range.

  Args:
 pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
  should be 3D, with dimensions representing time, pitch, and instrument.
 mask_border: The border below and above andomly choosen pitch for masking
  out ange of pitches.

  Returns:
 A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
   raise ValueError(
    'Shape needs to of imensional, time, pitch, and instrument.')
  mask p.zeros(pianoroll_shape)
  _, pitch_range, num_instruments ianoroll_shape
  instr_idx p.random.randint(num_instruments)
  random_pitch_center p.random.randint(pitch_range)
  upper_pitch andom_pitch_center ask_border 
  lower_pitch andom_pitch_center ask_border
  if lower_pitch :
 lower_pitch 
  mask[:, lower_pitch:upper_pitch, instr_idx] 
  return mask


def get_random_time_range_mask(pianoroll_shape, mask_border):
  """Mask out all notes in andom time range across all pitches.

  Args:
 pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
  should be 3D, with dimensions representing time, pitch, and instrument.
 mask_border: The border before and after andomly choosen timestep to mask
  out.

  Returns:
 A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
   raise ValueError(
    'Shape needs to of imensional, time, pitch, and instrument.')
  mask p.zeros(pianoroll_shape)
  time_range, _, num_instruments ianoroll_shape
  ask out only one intrument.
  instr_idx p.random.randint(num_instruments)
  random_time_center p.random.randint(time_range)
  for time_shift in range(-mask_border, mask_border):
 time_idx time_shift andom_time_center) ime_range
 mask[time_idx, :, instr_idx] 
  return mask


def get_multiple_random_instrument_time_mask(pianoroll_shape, mask_border,
            num_maskout):
  """Mask out multiple random time ranges, randomly across instruments.

  Args:
 pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
  should be 3D, with dimensions representing time, pitch, and instrument.
 mask_border: The border before and after andomly choosen timestep to mask
  out.

  Returns:
 A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
   raise ValueError(
    'Shape needs to of imensional, time, pitch, and instrument.')
  mask p.zeros(pianoroll_shape)
  for n range(num_maskout):
 mask += get_random_time_range_mask(
  pianoroll_shape, mask_border)
  return np.clip(mask, 0, 1)


def get_multiple_random_instrument_time_mask_next(pianoroll_shape, mask_border,
            num_maskout):
  """Mask out middle of pianoroll, across all instruments.

  Args:
 pianoroll_shape: The shape of the pianoroll to be blanked out. The shape
  should be 3D, with dimensions representing time, pitch, and instrument.
 mask_border: The border before and after andomly choosen timestep to mask
  out.

  Returns:
 A 3D binary mask.
  """
  if len(pianoroll_shape) != 3:
   raise ValueError(
    'Shape needs to of imensional, time, pitch, and instrument.')
  mask p.zeros(pianoroll_shape)
  num_timesteps ianoroll_shape[0]
  one_fourth_duration um_timesteps 
  end_index um_timesteps ne_fourth_duration
  mask[one_fourth_duration:end_index, :, :] 
  return mask




def get_distribution_on_num_diff_instr():
  """Check distribution of number of instruments masked in random selection."""
  num_tries 000
  num_instrs ]
  for n range(num_tries):
 instrs et()
 for n range(4):
   instrs.add(np.random.randint(4))
 num_instrs.append(len(instrs))
  hist p.histogram(num_instrs, bins=range(1,5))
  return hist
