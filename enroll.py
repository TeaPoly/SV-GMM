#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import argparse
import sys
import os

import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture

from features import extract_features


def main(_):
    features = np.asarray(())
    with open(FLAGS.file_list, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            # read the audio (wav)
            try:
                # using wav format to analysis preferentially
                sr, audio = read(line)
            except:
                # processing 16-bit integer pcm data types by default
                audio = np.memmap(line, dtype='int16', mode='r')
                sr = FLAGS.sample_rate

            # extract MFCC & delta MFCC features
            vector = extract_features(audio, sr)

            if features.size == 0:
                features = vector
            else:
                # concatenated features
                features = np.vstack((features, vector))

    # model training based on GMM
    gmm = GaussianMixture(n_components=FLAGS.gmm_number, max_iter=200,
                          covariance_type='diag', n_init=3)
    gmm.fit(features)

    # write the trained gaussian model to file.
    gmm_file = os.path.join(FLAGS.model_dir, FLAGS.speaker_id+".gmm")
    with open(gmm_file, 'wb') as fp:
        pickle.dump(gmm, fp)
        print("GMM model for speaker \"{}\" has writed in \"{}\".".format(
            FLAGS.speaker_id, gmm_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Speaker Verification with GMMs.')

    parser.add_argument('--file_list',
                        type=str,
                        default='')

    parser.add_argument('--model_dir',
                        type=str,
                        default='')

    parser.add_argument('--speaker_id',
                        type=str,
                        default='')

    parser.add_argument('--sample_rate',
                        type=int,
                        default=8000)

    parser.add_argument('--gmm_number',
                        type=int,
                        default=16)

    FLAGS = parser.parse_args()

    FLAGS, unparsed = parser.parse_known_args()

    sys.exit(main(sys.argv[:1] + unparsed))
