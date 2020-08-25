import os
import shutil

import pickle

import pandas as pd

from modules.utils.data_utils import preprocessing

DIRS = ['data\\inputs', 'data\\targets']
FRAC = 0.025

for dir in DIRS:
    if os.path.isdir(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)
    else:
        os.mkdir(dir)

df = pd.read_csv('data\\1_6_mil_twitter.csv', header=None)
df = df.sample(frac=FRAC).reset_index(drop=True)

list_sentences = list(df[5].values)
sentiments = df[0].map(
    {
        2: 0,
        4: 1,
        0: 2
    }
).values

encoder, decoder = preprocessing(
    list_sentences,
    sentiments,
    max_len=1000,
    language='english'
)

encoder_file = open('results\\objects\\encoder.pkl', 'wb')
pickle.dump(encoder, encoder_file)
encoder_file.close()

decoder_file = open('results\\objects\\decoder.pkl', 'wb')
pickle.dump(decoder, decoder_file)
decoder_file.close()
