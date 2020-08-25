import pandas as pd

from modules.utils.data_utils import preprocessing
from modules.utils.general_utils import dirs_creation, dump_pickle

DIRS = ['data\\inputs', 'data\\targets', 'results\\objects']
FRAC = 1.0

dirs_creation(
    dirs=DIRS,
    wipe_dir=True,
)

df = pd.read_csv('data\\csv\\cleaned\\airline_twitter.csv')
df = df.sample(frac=FRAC).reset_index(drop=True)
list_sentences = list(df['tweet'].values)
sentiments = df['sentiment'].map(
    {
        'neutral': 0,
        'positive': 1,
        'negative': 2
    }
).values

encoder, decoder = preprocessing(
    list_sentences,
    sentiments,
    max_len=1000
)

dump_pickle(
    objs=[encoder, decoder],
    paths=['results\\objects'] * 2,
    filenames=['encoder', 'decoder']
)
