import os

import numpy as np

import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping as es

from modules.models import language_sentiment_model
from modules.utils.data_utils import DataGenerator
from modules.utils.general_utils import dirs_creation

###############################################################

DECODER = pd.read_pickle('results\\objects\\decoder.pkl')

BTCH = [i for i in range(len(os.listdir('data\\inputs')))]
BTCH = np.random.choice(BTCH, len(BTCH), replace=False)

TR_BTCH = BTCH[: int(len(BTCH) * 0.8)]
TS_BTCH = BTCH[int(len(BTCH) * 0.8):]

DIRS = ['results\\models']

dirs_creation(
    dirs=DIRS,
    wipe_dir=False,
)

###############################################################

stopper = es(
    min_delta=0.0001,
    patience=5,
    monitor='val_loss',
    restore_best_weights=True
)
hp_dict = {
    'embedding_units': 250,
    'lstm_units': 100,
    'dense_units': 100,
    'dropout': 0.2
}

tr_generator = DataGenerator(
    TR_BTCH,
    shuffle=True
)

ts_generator = DataGenerator(
    TS_BTCH,
    shuffle=True
)

model = language_sentiment_model(
    len(DECODER) + 1,
    hp_dict,
    sentiment_bias=0.5
)

###############################################################

model.fit(
    tr_generator,
    epochs=30,
    verbose=1,
    callbacks=[stopper],
    validation_data=ts_generator
)

###############################################################

model.save('results\\models\\sentiment_estimator')

inp = model.get_layer('input').input
out = model.get_layer('lstm_encoder').output
extractor = Model(inp, out)

extractor.save('results\\models\\features_extractor')
