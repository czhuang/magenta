## Deep Convolutional Net for Polyphonic Music Autofill

## Background and motivation

Music Autofill is analogous to inpainting for image completion, where the goal
is to predict the missing parts of the canvas based on surrounding context. The
canvas here is the piano roll, a binary matrix with discrete ordered pitches as
rows and quantized time as columns. The piano roll represents a piece of music
unfolding in time, and each cell corresponds to if a pitch is heard or not at
that time step. There can be multiple instruments playing at the same time,
which means the piano roll will be multiple hot at each time step. Instruments
may also overlap on the notes they play, in which case multiple piano rolls are
needed to represent the individual parts. In the current implementation, we
overlay all instruments onto one piano roll, while in the future, we plan to
have one instrument family per piano roll to better model the different
characteristics of different types of instruments.

Imagine a composer writing for a choir. She has the melody written out and also
the bass line. She’s laboriously working through the inner voices one by one.
What if the machine could suggest different ways to autocomplete the missing
parts? To achieve this, we use a deep convolutional net to learn the
interrelationships between voices by estimating hierarchical sets of filters. A
model is considered successful in this task when it is able to automatically add
a voice to the score in a way that is stylistically idiomatic, shares the
motives present in the other voices, fits into the implied harmonic progression,
captures the local interplay between voices, and that is coherent with the
larger narrative of the piece.

## Model

The model is currently a 16-layer convolutional network with batch
normalization, with 128 filters at each respective layer. The input takes two
128x32 binary matrices, a piano roll with patches blanked out and a mask
indicating where these regions are. 128 corresponds to the range of midi
pitches, while 32 corresponds to two measures of music where time ticks in 32th
notes. A blank out is a rectangular patch indicating which timesteps need to be
filled in and which pitch range they are most likely to occur in. For example,
if we want to blank out mostly the alto voice for two measures, the mask would
horizontally span all the columns and vertically span the range of 57 to 68. The
output is also a binary matrix of the same size. The model is trained to
reconstruct the entire score with the blank outs filled in. As the input and
output are of the same sizes and of a relative small size, there is no pooling
or striding. As we allow the input to be a longer segment of music, we will
likely have to use a conv-deconv architecture.

## How to try out the model?

*   To build all, bazel build :all.
*   To train, bazel run :basic_autofill_cnn_train.
*   To generate, blaze-run :basic_autofill_cnn_generate.

## TODO:

*   To demo how a user might interact with the generation, such as by adding
    one's own piano roll as context for autofill, a step-by-step example is
    given in basic_autofill_cnn_generate_demo.ipynb.
