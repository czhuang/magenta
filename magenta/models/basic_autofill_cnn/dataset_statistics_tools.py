"""One-line documentation for dataset_statistics_tools module.

A detailed description of dataset_statistics_tools.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf

import pretty_midi

from magenta.models.basic_autofill_cnn import data_tools
#from magenta.models.basic_autofill_cnn import seed_tools
#from magenta.models.basic_autofill_cnn import data_pipeline_tools
from magenta.models.basic_autofill_cnn import test_tools
from magenta.models.basic_autofill_cnn import pianorolls_lib

from magenta.music.note_sequence_io import note_sequence_record_iterator, NoteSequenceRecordWriter
from magenta.music.midi_io import sequence_proto_to_midi_file, midi_to_sequence_proto


import contextlib
@contextlib.contextmanager
def pdb_post_mortem():
  try:
    yield
  except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
      import traceback
      traceback.print_exception(exc_type, exc_value, exc_traceback)
      import pdb; pdb.post_mortem()


# shortest_duration, min_pitch, max_pitch
base_path = '/data/lisatmp4/huangche/data'
datasets = ['MuseData', 'Nottingham', 'Piano-midi.de', 'JSB Chorales']


#def test_pianorolls_lib():
#  data, encoder = data_tools.get_data(base_path, 'Piano-midi.de', 'train', separate_instruments=False)
#  rolls = [encoder.encode(seq) for seq in data]
#  note_densities = []
#  for roll in rolls:
#    note
#    print roll.shape, np.sum(roll), np.sum(roll) / roll.shape[0]
    
               
folds = ['train', 'valid', 'test']
dataset_params = dict()
def read_datasets():
  for dataset in datasets:
    fpath = os.path.join(base_path, dataset+'.npz')
    data = np.load(fpath)
    print data.keys(), type(data)
    min_pitch = 127
    max_pitch = 0
    for fold in data.keys():
      pieces = data[fold]
      print len(pieces)
      for piece in pieces:
        for chord in piece:
#          print np.min(chord).shape, np.min(chord)
          if len(chord) > 0:
            min_pitch = np.minimum(np.min(chord), min_pitch)
            max_pitch = np.maximum(np.max(chord), max_pitch)
          else:
            print 'empty chord'
    dataset_params[dataset] = dict(pitch_ranges=[min_pitch, max_pitch])
    print dataset, min_pitch, max_pitch
  print dataset_params

def read_midi(): 
  print pretty_midi.__version__
  fpath = '/data/lisatmp4/huangche/data/Nottingham/train/waltzes_simple_chords_8.mid'
  fpath = '/data/lisatmp4/huangche/data/Nottingham/train/morris_simple_chords_21.mid'
  seq = midi_to_sequence_proto(fpath)
  print len(seq.notes)
  num_instruments = set(note.instrument for note in seq.notes)
  num_programs = set(note.program for note in seq.notes)
  num_parts = set(note.part for note in seq.notes)
  print num_instruments, num_programs, num_parts


def get_note_sequences():
  path = '/u/huangche/data/bach'
  fname = 'bach_chorale_note_sequences.tfrecord'
  return list(note_sequence_record_iterator(os.path.join(path, fname)))


def get_the_prime_piece():
  fpath = '/data/lisatmp4/huangche/data/bach/bwv103.6.tfrecord'
  return list(note_sequence_record_iterator(fpath))
  

def check_num_of_pieces_in_tfrecord():
  path = '/u/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True'
  fnames = ['train', 'valid', 'test']
  num_of_pieces = dict()
  for fname in fnames:
    fpath = os.path.join(path, fname + '.tfrecord')
    num_of_pieces[fname] = len(list(note_sequence_record_iterator(fpath)))
  print num_of_pieces.items()
  print np.sum(num_of_pieces.values())


def get_num_of_voices_hist(seqs=None):
  if seqs is None:
    seqs = get_note_sequences()
  voicing_counts = defaultdict(int)
  for seq in seqs:
    voicing = set(note.part for note in seq.notes)
    voicing_counts[tuple(voicing)] += 1
  for voicing, counts in voicing_counts.iteritems():
    print voicing, counts
  return voicing_counts


def get_4_voice_sequences(seqs=None):
  if seqs is None:
    seqs = get_note_sequences()
  four_voice_seqs = []
  for seq in seqs:
    if len(set(note.part for note in seq.notes)) == 4:
      four_voice_seqs.append(seq)
  return four_voice_seqs



# nicolas dataset
# set, # of pieces, min pch, max pch
# train, 229, 43, 96
# valid, 76, 48, 96
# test, 77, 45, 96
# global min and max, 43, 96

# music21 dataset
# 36, 88

# min and max across both datasets
# 36, 96
MIN_PCH = 36
MAX_PCH = 96
# in nichola's, a piece is a list of list


def get_min_max_in_list_of_list(lll):
  min_max = [f(l) for ll in lll for l in ll for f in (min, max) if len(l)>0]
  return min(min_max), max(min_max)


def get_piece_min_max(ll):
  min_max = [f(l) for l in ll for f in (min, max) if len(l)>0]
  return min(min_max), max(min_max)


def get_piece_tensor(ll):
  min_, max_ = get_piece_min_max(ll)
  pch_range = max_ - min_ + 1
  condensed_pianoroll = np.zeros((len(ll), pch_range, 4))
  for t, l in enumerate(ll):
    transposed_pitches = np.array(l) - min_
    condensed_pianoroll[t, :, :len(l)] = np.transpose(np.eye(pch_range)[transposed_pitches], axes=[1, 0])
  return np.repeat(condensed_pianoroll, 4, axis=0)


def test_get_piece_tensor():
  ll = [[1,2], [0,2]]
  tensor = get_piece_tensor(ll)
  print tensor


def diff_piece_patch(piece, patch):
  assert piece.ndim == 3 and patch.ndim == 3
  diff_pitch_range = piece.shape[1] - patch.shape[1]
  assert piece.shape[1] < patch.shape[1]
  # check ordering if instrs, if soprano of piece is lower than bass of patch
  assert piece[0, :,0] > np.nonzero
  for i in range(8):
    pass 


def retrieve_nicolas_bach_pickle():
  path = '/data/lisatmp4/huangche/data/JSB Chorales.pickle'
  import cPickle as pickle
  with open(path, 'rb') as p:
    data = pickle.load(p)
  print len(data)
  piece_counts = np.array([float(len(dd)) for dd in data.values()])
  print 'split %', piece_counts/np.sum(piece_counts)  

  lengths = []
  num_voices_list = []
  piece_count = 0
  dataset_min_pch = 127
  dataset_max_pch = 0
  for k, dd in data.items():
    print k, len(dd)
    piece_count += len(dd) 
    print 'min, max', get_min_max_in_list_of_list(dd)
    for d in dd:
      lengths.append(len(d))
      num_voices = np.max([len(ch) for ch in d])
      num_voices_list.append(num_voices)
  print sorted(list(set(lengths)))
  print sorted(list(set(num_voices_list)))
  print '# of pieces:', piece_count
  print 'min, max', datset_min_pch, dataset_max_pch
  
  return data

def match_dataset_split():
  data_split = retrieve_nicolas_bach_pickle()
  print '---'
  seqs = get_4_voice_sequences()
  print len(seqs)
  encoder = pianorolls_lib.PianorollEncoderDecoder(
    min_pitch=MIN_PCH, max_pitch=MAX_PCH)
  lengths = []
  pianorolls = []
  for seq in seqs:
    pianoroll = encoder.encode(seq)
    pianorolls.append(pianoroll)
    length = pianoroll.shape[0]
    lengths.append(length)
  print sorted(list(set(lengths)))

  matches = {}  # from nicolas's to music21's index here
  no_matches_nicolas = []
  for key, pieces in data_split.items():
    for i, piece in enumerate(pieces):
      # match by length
      candidates = []
      for j, pianoroll in enumerate(pianorolls):
        #print len(piece) * 4, pianoroll.shape[0]
        if abs(len(piece)*4 - pianoroll.shape[0]) < 5:
          candidates.append(j)
      #print '# of candidates:', len(candidates)
      assert len(candidates) != 0
      # if matched by length
      if len(candidates) == 1:
        matches[(key, i)] = candidates[0]
      else: 
        # if didn't match by length then match pitch distribution
        for c in candidates:
          pianoroll = pianorolls[c]
          pdist = np.sum(pianoroll, axis=(0, 2))
          assert pdist.size == 53
          pdist /= np.sum(pdist)

        # make one hot by 
 
       
        no_matches_nicolas.append((key, i))        
#  print len(no_matches_nicolas), 'no_match', no_matches_nicolas
  print len(matches), 'matched', matches


def synth_start_of_note_sequences():
  path = '/data/lisatmp4/huangche/data/bach/primes'
#  seqs = get_4_voice_sequences()
  seqs = get_the_prime_piece()
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  synth_timesteps = 128
  short_seqs = []
  for seq in seqs:
    pianoroll = encoder.encode(seq)
    short_seq = encoder.decode(pianoroll[:synth_timesteps])
    fpath = os.path.join(path, seq.filename.split('.mxl')[0] + '.midi')
    sequence_proto_to_midi_file(short_seq, fpath) 


def synth_random_crop_from_valid():
  path = '/data/lisatmp4/huangche/data/bach/random_crops'
  path = '/data/lisatmp4/huangche/data/bach/fromScratch_random_crops'
  path = '/data/lisatmp4/huangche/data/listening/samplingMethod_unconditioned/bach'
 # valid_data = list(data_tools.get_note_sequence_data(FLAGS.input_dir, 'valid'))
 # print '# of valid_data:', len(valid_data)
 # encoder = pianorolls_lib.PianorollEncoderDecoder()
 # synth_timesteps = 32
 # short_seqs = []
 # config.hparams.batch_size = 4
 # input_data, targets = data_tools.make_data_feature_maps(
 #     valid_data, config, encoder)
 
  from pianorolls_lib import STRING_QUARTET_PROGRAMS, WOODWIND_QUARTET_PROGRAMS
  PROGRAMS = WOODWIND_QUARTET_PROGRAMS
  # Gets data.
  input_dir = '/Tmp/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True'
  valid_data = list(data_tools.get_note_sequence_data(input_dir, 'valid'))

#  validation_path = '/Tmp/huangche/data/bach/qbm120/instrs=4_duration=0.125_sep=True'
#  model_name = 'DeepResidual64_128'
#  seeder = seed_tools.get_seeder(validation_path, model_name, maskout_method_str='no_mask')
#  seeder.crop_piece_len = 32 
#  pianorolls, piece_names = seeder.get_random_batch(
#     0 , return_names=True)
  
  seqs = [valid_data[i] for i in np.random.choice(len(valid_data), size=4)]
  print [seq.filename for seq in seqs]
  encoder = pianorolls_lib.PianorollEncoderDecoder()

  # skip first one since starts from beginning.
  for i, seq in enumerate(seqs):
  #  assert target.shape == (32, 53, 4)
  #  decoded_seq = encoder.decode(target, PROGRAMS)
  #  fpath = os.path.join(path, piece_names[i].split('.mxl')[0] + '.midi')
  #  sequence_proto_to_midi_file(decoded_seq, fpath)
  #  

    # synth complete
    complete_seq = seq
    complete_pianoroll = encoder.encode(seq)
    complete_seq_decoded = encoder.decode(complete_pianoroll, PROGRAMS)
    complete_fpath = os.path.join(path, seq.filename.split('.mxl')[0] + 'complete_%d.midi'% i)
    sequence_proto_to_midi_file(complete_seq_decoded, complete_fpath)
    
    crop_len = 32
    start_index = np.random.randint(complete_pianoroll.shape[0] - crop_len)
    crop_pianoroll = complete_pianoroll[start_index:start_index + crop_len]

    crop_seq_decoded = encoder.decode(crop_pianoroll, PROGRAMS)
    crop_fpath = os.path.join(path, 'o_%d.midi' % i)
    sequence_proto_to_midi_file(crop_seq_decoded, crop_fpath)

    tfrecord_fpath = os.path.join(path, seq.filename.split('.mxl')[0] + '.tfrecord')
    writer = NoteSequenceRecordWriter(tfrecord_fpath)    
    writer.write(crop_seq_decoded)


def rearrange_instruments():
  path = '/data/lisatmp4/huangche/data/listening/fromScratch'
  print path
  from pianorolls_lib import STRING_QUARTET_PROGRAMS
  PROGRAMS = STRING_QUARTET_PROGRAMS
  encoder = pianorolls_lib.PianorollEncoderDecoder()
  rename_dict = {'19-balanced': 'b', 'Denoising': 'd',
                 'balanced_fc_mask_only': 'bfm', 'bwv': 'o',
                 'random_medium': 'rm'}
  rename_counts = defaultdict(int)
  for root, dirs, fnames in os.walk(path):
    print root, dirs, fnames
    for fname in fnames:
      if '.tfrecord' in fname:
        fpath = os.path.join(root, fname) 
        seqs = note_sequence_record_iterator(fpath)
        seq = list(seqs)[0]
        pianoroll = encoder.encode(seq)
        decoded_seq = encoder.decode(pianoroll, pianoroll_to_program_map=PROGRAMS)
        fname_prefix = fname.split('.tfrecord')[0]
        print fname_prefix
        rename_tag = None
       
        for key, abbr in rename_dict.items():
          if key in fname_prefix:
            rename_tag = abbr
            rename_counts[abbr] += 1
            break
        if rename_tag is None:
          assert False
        fpath = os.path.join(root, rename_tag+str(rename_counts[rename_tag])+'_sq.midi')

        sequence_proto_to_midi_file(decoded_seq, fpath)


def get_duration_hist():
  seqs = get_note_sequences()
  get_num_of_voices_hist(seqs)
  seqs = get_4_voice_sequences(seqs)
  tempo_used = set()
  duration_counts = defaultdict(int)
  piece_with_such_duration = defaultdict(set)
  for seq in seqs:
   tempo_used.add(seq.tempos[0].qpm)
   for note in seq.notes:
     duration = note.end_time - note.start_time
     duration_counts[duration] += 1
     piece_with_such_duration[duration].add(seq.filename)
  sorted_durations = sorted(duration_counts, key=lambda x: x)
  for duration in sorted_durations:
    print duration, duration_counts[duration], len(piece_with_such_duration[duration])
  print tempo_used


#def check_tessitura_hist_per_voice():
#  ust checking range for now.
#  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
#  voice_tessitura efaultdict(list)
#  ggregate tessitura for each voice across pieces.
#  for seq in seqs_reader:
# voices est_tools.collect_sorted_voices(seq, 'program')
# #voices est_tools.collect_sorted_voices(seq, 'part')
# #print voices.keys()
# #assert sorted(voices.keys()) == range(4)
# assert set(voices.keys()) == set([74, 72, 69, 71])
# pitch_duration_by_voice efaultdict(list)
# for part_index, notes in voices.iteritems():
#   pitches ]
#   durations ]
#   for note in notes:
#  pitches.append(note.pitch)
#  durations.append(note.end_time ote.start_time)
#   ggregate itch as the approximate tessitura.
#   voice_tessitura[part_index].append(np.ceil(
#    np.average(pitches, weights=durations)))
#  rint historgram of tessitura.
#  for part_index in [74, 72, 69, 71]:
# pitches, counts p.unique(voice_tessitura[part_index], return_counts=True)
# print '\nPart %d:' art_index
# print 'pitches', pitches
# print 'counts', counts
# sorted_indices p.argsort(pitches)
#
# #for index in sorted_indices:
# #  print '(%d, %d)' pitches[index], counts[index]),
#
#
def check_tessitura_ordering_hist():
  print 'check_tessitura_ordering_hist'
  seqs = get_4_voice_sequences()
  print 'len(seqs)', len(seqs)
  ordering_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
  ordering_counts = defaultdict(int)
  # aggregate tessitura for each voice across pieces.
  pieces = []
  for seq in seqs:
    voices = test_tools.collect_sorted_voices(seq, 'part')
    pieces.append(voices)

  tessitura = defaultdict(list)
  for voices in pieces:
    for part_index, notes in voices.iteritems():
      pitches = []
      durations = []
      for note in notes:
        pitches.append(note.pitch)
        durations.append(note.end_time-note.start_time)
      #aggregate itch as the approximate tessitura.
      tessitura[part_index].append(np.ceil(
          np.average(pitches, weights=durations)))

  for part_index, tess in tessitura.items():
    print part_index, len(tess)
 
  # Check tessitura ordering.
  for top_index, bottom_index in ordering_pairs:
    for piece_index in range(len(pieces)):
      if (tessitura[top_index][piece_index] 
        > tessitura[bottom_index][piece_index]):
        ordering_counts[(top_index, bottom_index)] += 1

  for ordering_pair in ordering_pairs:
    print ordering_pair, ordering_counts[ordering_pair]
#
#
#def check_voices():
#  seqs_reader ata_pipeline_tools.get_bach_chorales_with_4_voices_dataset()
#  for seq in seqs_reader:
# voices est_tools.collect_sorted_voices(seq, 'program')
# print voices.keys()


def check_dataset():
  from magenta.models.basic_autofill_cnn import hparams_tools
  hparams = hparams_tools.Hyperparameters(
      dataset='bach-16th-priorwork-4_voices',
      quantization_level=0.125)
  base_path = '/data/lisatmp4/huangche/data/'
  pianorolls = data_tools.get_data_as_pianorolls(base_path, hparams, 'test')
  print len(pianorolls)


def main(unused_argv):
  #get_duration_hist()
  #check_num_of_pieces_in_tfrecord()
  #check_tessitura_hist_per_voice()
  #check_voices()
  #check_tessitura_ordering_hist()
#  synth_random_crop_from_valid()
#  synth_start_of_note_sequences()
#  rearrange_instruments()
#  match_dataset_split()
#  retrieve_nicolas_bach_pickle()
#  test_get_piece_tensor()
#  read_midi()
#  read_datasets()
#  test_pianorolls_lib()
  check_dataset()


if __name__ == '__main__':
  with pdb_post_mortem():
    tf.app.run()


