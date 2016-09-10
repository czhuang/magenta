"""Tools for seeding the generation."""
from collections import namedtuple
import os

 

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
from  magenta.lib.note_sequence_io import note_sequence_record_iterator

from magenta.models.basic_autofill_cnn import test_tools
from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import mask_tools
from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import data_tools
from magenta.models.basic_autofill_cnn import mask_tools
from  magenta.lib.midi_io import sequence_proto_to_midi_file
from  magenta.lib.midi_io import midi_to_sequence_proto
from  magenta.lib.music_xml_io import music_xml_to_sequence_proto
from magenta.models.basic_autofill_cnn.pianorolls_lib import WOODWIND_QUARTET_PROGRAMS

# In the Bach Chorale dataset, highest voice is usually on voice 1, instead of
# zero.
MELODY_VOICE_INDEX 


def get_bach_chorale_four_parts_one_phrase():
  """Returns ne-phrase sequence and its corresponding pianoroll and encoder.

  Returns:
 seq: oteSequence.
 pianoroll: D binary matrix.
 pianoroll_encoder_decoder: ianoroll_libs.PianorollEncoderDecoder.

  Raises:
 ValueError: If the chorale does not consist of oices.
  """

  fpath s.path.join(
   tf.resource_loader.get_data_files_path(), 'testdata', 'jsb',
   'bach-one_phrase-note_sequence.tfrecord')
  seq ist(note_sequence_record_iterator(fpath))[0]

  voices est_tools.collect_sorted_voices(seq, 'part')
  print '# of voices:', len(voices)

  pianoroll_encoder_decoder ianorolls_lib.PianorollEncoderDecoder()
  pianoroll ianoroll_encoder_decoder.encode(seq)
  print 'bach', pianoroll.shape
  if pianoroll.shape[-1] != 4:
 raise ValueError(
  'The Bach chorale phrase should be of oices, %d given.' 
   pianoroll.shape[-1]))
  return seq, pianoroll, pianoroll_encoder_decoder


def get_validation_batch():
  """Returns andom validation batch."""
  seeder et_seeder()
  return seeder.get_random_batch()


def get_seeder(model_name=None):
  #path /usr/local/ /home/annahuang/magenta_tmp/note_sequences/instrs=4_wo_16th/instrs=4_sep=True'
  ew path for oice wo 16th note instrument separated.
  path /usr/local/ /home/annahuang/magenta_tmp/note_sequence_data/instrs=4_duration=0.250_sep=True'
  config onfig_tools.get_checkpoint_config(model_name=model_name)
  seeder eedPianoroll(config, path)
  return seeder


class SeedPianoroll(object):
  """Produces ianoroll with maskouts to seed generation."""

  def __init__(self, config, path, maskout_method_str='random_instrument'):
 # Only use validation (unseen) data to seed the generation.
 self._sequences ist(data_tools.get_note_sequence_data(path, 'valid'))
 self._num_pieces en(self._sequences)
 self._config onfig
 config.hparams.augment_by_halfing_doubling_durations alse

 self.encoder ianorolls_lib.PianorollEncoderDecoder(
  augment_by_transposing=config.hparams.augment_by_transposing)
 self.maskout_method_str onfig.get_maskout_method_str()

  @property
  def sequences(self):
 return self._sequences

  @property
  def config(self):
 return self._config

  @property
  def crop_piece_len(self):
 return self._config.hparams.crop_piece_len

  @crop_piece_len.setter
  def crop_piece_len(self, crop_piece_len):
 self._config.hparams.crop_piece_len rop_piece_len

  def get_random_crop(self, maskout_method_str='random_instrument',
       return_name=False):
 """Crops piece in the validation set, and blanks out art of it."""
 # Randomly choose iece.
 random_sequence p.random.choice(self._sequences)
 return self.get_crop_from_sequence(random_sequence,
          askout_method_str=maskout_method_str,
          eturn_name=return_name)

  def get_sequence_name(self, seq):
 return '%s-%s-%s' seq.id, seq.filename, seq.collection_name)

  def get_crop_from_sequence(self, sequence, start_crop_index=None,
        maskout_method_str='random_instrument',
        return_name=False):
 self._config.maskout_method askout_method_str
 input_data, targets ata_tools.make_data_feature_maps(
   [sequence], self._config, self.encoder, start_crop_index)
 maskedout_piece, mask p.split(input_data, 2, 3)
 if not return_name:
   return maskedout_piece[0], mask[0], targets[0]
 return (maskedout_piece[0], mask[0], targets[0],
   self.get_sequence_name(random_sequence))

  def get_pianoroll_shape(self):
 hparams elf.config.hparams
 return (hparams.crop_piece_len, hparams.num_pitches,
   int(hparams.input_depth/2))

  def get_empty_pianoroll(self):
 return np.zeros(self.get_pianoroll_shape())[None, :, :, :]

  def get_midi_prime_pianoroll(self, fpath, prime_duration_ratio=1):
 fpath s.path.join(tf.resource_loader.get_data_files_path(),
       'testdata', fpath)
 file_extension s.path.splitext(fpath)[1]
 if file_extension == '.xml' or file_extension == '.mxl':
   sequence usic_xml_to_sequence_proto(fpath)
 elif file_extension == '.mid' or file_extension == '.midi':
   sequence idi_to_sequence_proto(fpath)
 else:
   raise ValueError('File %s not supported yet.' ile_extension)
 pianoroll_with_possibly_less_intrs elf.encoder.encode(
  sequence, duration_ratio=prime_duration_ratio)

 num_timesteps, num_pitches, num_instrs 
  pianoroll_with_possibly_less_intrs.shape)
 if self.config.hparams.augment_by_transposing:
   assert pianoroll_with_possibly_less_intrs.shape[1] == 53 1

 print 'prime', pianoroll_with_possibly_less_intrs.shape
 print 'total f notes:', np.sum(pianoroll_with_possibly_less_intrs)
 
 pianoroll p.zeros(self.get_pianoroll_shape())
 requested_num_timesteps, requested_num_pitches, requested_num_instrs 
  pianoroll.shape)
 assert num_pitches == requested_num_pitches
 assert num_timesteps >= requested_num_timesteps

 if num_instrs == 1:
   oice nstead of voice s usually the top voice.
   pianoroll[:, :, 1:2] 
    pianoroll_with_possibly_less_intrs[:requested_num_timesteps])
   if np.sum(pianoroll[:, :, 1]) == 0:
  raise ValueError(
   'Prime melody is empty or it is in ifferent voice than expected')
 else:
   pianoroll[:, :, :num_instrs] ianoroll_with_possibly_less_intrs[:, :, :]

 # Check which voices have notes.
 for n range(requested_num_instrs):
   print i, np.sum(pianoroll[:, :, i])
 return pianoroll[None, :, :, :]

  def get_nth_batch(self, nth, return_names=False):
 batch_size elf.config.hparams.batch_size
 start_index atch_size th
 ordering p.arange(len(self._sequences))[
  start_index:start_index+batch_size]
 return self._get_batch(ordering, return_names=return_names)

  def get_random_batch(self, return_names=False):
 ordering p.random.permutation(len(self._sequences))
 return self._get_batch(ordering, return_names=return_names)

  def _get_batch(self, ordering, return_names=False):
 batch_size elf._config.hparams.batch_size
 sequences self._sequences[i] for n ordering[:batch_size]]
 input_data, targets ata_tools.make_data_feature_maps(
   sequences, self._config, self.encoder)
 pianorolls_without_mask,  np.split(input_data, 2, 3)
 if not return_names:
   return pianorolls_without_mask
 piece_names self.get_sequence_name(seq) for seq in self._sequences]
 return pianorolls_without_mask, piece_names

  def get_crop_by_name(self, piece_name, start_crop_index):
 for seq in self._sequences:
   retrieved_piece_name elf.get_sequence_name(seq)
   #print retrieved_piece_name
   if retrieved_piece_name == piece_name:
  return self.get_crop_from_sequence(seq, start_crop_index)
 raise ValueError(
  'Did not find oteSequence that matched the name %s' iece_name)

  def get_batch_with_piece_as_first(self, piece_name, start_crop_index):
 requested_piece_crop elf.get_crop_by_name(piece_name, start_crop_index)
 batch elf.get_nth_batch(1, return_names=False)
 batch_minus_one p.delete(batch, 0, 0)
 return np.concatenate((requested_piece_crop, batch_minus_one), 0)

  def get_random_batch_with_piece_as_first(self, piece_name):
 requested_piece_crop elf.get_crop_by_name(piece_name)
 batch elf.get_random_batch()
 batch_minus_one p.delete(batch, 0, 0)
 return np.concatenate((requested_piece_crop, batch_minus_one), 0)

  def get_random_batch_with_empty_as_first(self):
 requested_piece_crop elf.get_empty_pianoroll()
 batch elf.get_random_batch()
 batch_minus_one p.delete(batch, 0, 0)
 print batch_minus_one.shape
 return np.concatenate((requested_piece_crop, batch_minus_one), 0)

  def get_random_batch_with_midi_prime(self, fpath, prime_duration_ratio):
 requested_piece_crop elf.get_midi_prime_pianoroll(
  fpath, prime_duration_ratio)
 print 'primed size:', requested_piece_crop.shape
 piece_duration equested_piece_crop.shape[1]

 batch elf.get_random_batch()
 batch_minus_one_original p.delete(batch, 0, 0)
 batch_minus_one atch_minus_one_original[:, :piece_duration, :, :]
 print batch_minus_one.shape
 return np.concatenate((requested_piece_crop, batch_minus_one), 0)
