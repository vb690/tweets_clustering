import os

import numpy as np

import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping as es

from modules.models import sentiment_classifier
from modules.utils.data_utils import DataGenerator

DECODER = pd.read_pickle('results\\objects\\decoder.pkl')

BTCH = [i for i in range(len(os.listdir('data\\inputs')))]
BTCH = np.random.choice(BTCH, len(BTCH), replace=False)

TR_BTCH = BTCH[: int(len(BTCH) * 0.8)]
TS_BTCH = BTCH[int(len(BTCH) * 0.8):]

stopper = es(
    min_delta=0.0001,
    patience=5,
    monitor='val_loss',
    restore_best_weights=True
)
hp_dict = {
    'embedding_units': 250,
    'lstm_units': 50,
    'dense_units': 50,
    'noise': 0.4
}

tr_generator = DataGenerator(
    TR_BTCH,
    shuffle=True
)

ts_generator = DataGenerator(
    TS_BTCH,
    shuffle=True
)

model = sentiment_classifier(
    len(DECODER) + 1,
    hp_dict,
    gamma=0.5
)

model.fit(
    tr_generator,
    epochs=10,
    verbose=1,
    callbacks=[stopper],
    validation_data=ts_generator
)

model.save('results\\models\\sentiment_estimator')

inp = model.get_layer('input').input
out = model.get_layer('lstm_1').output
extractor = Model(inp, out)

extractor.save('results\\models\\features_extractor')
