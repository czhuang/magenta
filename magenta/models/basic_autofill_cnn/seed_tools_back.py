  def get_magenta_sequence(self):
 fpath / /src/cloud/annahuang/annahuang0-annahuang-basic_autofill_cnn-separate_voices-git5/magenta/models/basic_autofill_cnn/testdata/condition_on/magenta_theme.xml'
 return music_xml_to_sequence_proto(fpath)


def get_scale_pianoroll(self):
 pianoroll p.zeros(self.get_pianoroll_shape())
 major_scale 0, 2, 4, 5, 7, 9, 11, 12]
 duration 
 pitch_offset 0

 default_pianoroll_shape elf.get_pianoroll_shape()
 pianoroll p.zeros((duration*len(major_scale), default_pianoroll_shape[1],
        default_pianoroll_shape[2]))
 time_count 
 for scale_degree in major_scale:
   pianoroll[time_count:time_count+duration, pitch_offset+scale_degree, 0] 
   time_count += duration
 return pianoroll[None, :, :, :]
