"""Tests to check conversion between NoteSequences and pianorolls."""

import os

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import pianorolls_lib
from magenta.models.basic_autofill_cnn import test_tools
from magenta.models.basic_autofill_cnn import mask_tools


class PianorollsLibTest(tf.test.TestCase):
  """Tests to check conversion between NoteSequences and pianorolls."""

  def setUp(self):
    self.seqs = list(test_tools.get_small_bach_chorales_with_4_voices_dataset())

  def checkPianorollsFromMultipleConversionRounds(self, input_seq):
    """Check round trip starting from NoteSequence to pianoroll."""

    sorted_voices = test_tools.collect_sorted_voices(input_seq, 'program')
    shortest_duration = pianorolls_lib.find_shortest_duration([input_seq])
    tf.logging.info('# of voices: %d', len(sorted_voices))
    tf.logging.info('shortest_duration: %.2f', shortest_duration)

    # Go from note sequence to pianoroll.
    pianoroll_encoder_decoder = pianorolls_lib.PianorollEncoderDecoder()
    pianoroll = pianoroll_encoder_decoder.encode(input_seq)

    # Go from pianoroll back to note sequence.
    # pianoroll_to_program_map = pianorolls_lib.reverse_mapping(
    #    program_to_pianoroll_map)
    decoded_seq = pianoroll_encoder_decoder.decode(pianoroll)

    # Go from decoded note sequence to pianoroll again.
    pianoroll_from_decoded_seq = pianoroll_encoder_decoder.encode(decoded_seq)

    # Just comparing if all the notes are there.  Not yet checking if the
    # instruments are carrying the same notes.
    collapsed_pianoroll = np.sum(pianoroll, axis=2)
    collapsed_pianoroll_from_decoded_seq = np.sum(pianoroll_from_decoded_seq,
                                                  axis=2)
    #print 'diffs:', np.sum(
    #    np.abs(collapsed_pianoroll - collapsed_pianoroll_from_decoded_seq))
    # Check that the two pianorolls are the same
    self.assertTrue(
        np.allclose(collapsed_pianoroll, collapsed_pianoroll_from_decoded_seq))

    # Go from pianoroll_from_decoded_seq to note sequence again.
    second_decoded_seq = pianoroll_encoder_decoder.decode(
        pianoroll_from_decoded_seq)
    self.assertEqual(len(decoded_seq.notes), len(second_decoded_seq.notes))

  def testPianorollsFromMultipleConversionRounds(self):
    """Test round trip starting from NoteSequence to pianoroll."""
    for seq in self.seqs:
      self.checkPianorollsFromMultipleConversionRounds(seq)

  def testProgramAssignment(self):
    target = {0: 74, 1: 72, 2: 69, 3: 71}
    pianoroll_to_program = pianorolls_lib.get_pianoroll_to_program_assignment(
        range(4))
    for pianoroll_index, program_index in target.iteritems():
      self.assertEqual(program_index, pianoroll_to_program[pianoroll_index])


if __name__ == '__main__':
  tf.test.main()
