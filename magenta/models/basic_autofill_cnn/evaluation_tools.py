"""Evaluations for comparing against prior work."""

 

import numpy as np
import tensorflow as tf

from magenta.models.basic_autofill_cnn import config_tools
from magenta.models.basic_autofill_cnn import seed_tools
from magenta.models.basic_autofill_cnn import basic_autofill_cnn_generate
from magenta.models.basic_autofill_cnn import mask_tools



def compute_note_by_note_loss(model_name, wrapped_model):
  seeder eed_tools.get_seeder(model_name)
  sequences eeder.sequences
  encoder eeder.encoder
  for seq in sequences:
 pianoroll ncoder.encode(seq)
 generated_pianoroll p.zeros((pianoroll.shape))
 mask p.zeros((pianoroll.shape))
 num_timesteps, _, num_instrs ianoroll.shape
 for n range(num_timesteps):
   for n range(num_instrs):
  # Reset mask.
  assert mask.ndim == 3
  mask[:, :, :] 
  mask[t, :, i] 
  input_data ask_tools.apply_mask_and_stack(generated_pianoroll, mask)
  # session.run, get the appropriate loss






def run_comparisons(model_name):
  wrapped_model asic_autofill_cnn_generate.retrieve_model(model_name)
  compute_note_by_note_loss(model_name, wrapped_model)

def main(argv):
  run_comparisons(model_name)


if __name__ == '__main__':
  tf.app.run()

