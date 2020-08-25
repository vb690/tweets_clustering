from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import RepeatVector, SpatialDropout1D
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

from tensorflow.keras import backend as K


from tensorflow.keras.models import Model


def __repeat_vector(args):
    '''
    '''
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


def sentiment_classifier(voc_size, hp_dict, max_len=60, gamma=1.0):
    '''
    '''
    inp = Input(shape=(None,), name='input')

    vectorization = Embedding(
        input_dim=voc_size+1,
        output_dim=hp_dict['embedding_units'],
        name='emb'
    )(inp)
    vectorization = SpatialDropout1D(
        hp_dict['noise']
    )(vectorization)

    recurrence_1 = LSTM(
        units=hp_dict['lstm_units'],
        name='lstm_1',
        return_sequences=True
    )(vectorization)

    # classification
    out_1 = Dense(
        units=hp_dict['dense_units'],
        name='dense_1'
    )(recurrence_1)
    out_1 = Dropout(hp_dict['noise'])(out_1)
    out_1 = Dense(
        units=3,
        name='dense_out_1'
    )(out_1)
    out_1 = Activation('softmax', name='act_1')(out_1)

    # language model
    out_2 = Dense(
            units=hp_dict['dense_units'],
            name='dense_2'
        )(recurrence_1)
    out_2 = Dropout(hp_dict['noise'])(out_2)
    out_2 = Dense(
        units=voc_size,
        name='dense_out_2'
    )(out_2)
    out_2 = Activation('softmax', name='act_2')(out_2)

    model = Model(inp, [out_1, out_2])
    model.compile(
        optimizer='Adam',
        loss=['sparse_categorical_crossentropy'] * 2,
        loss_weights=[1.0 * gamma, 1.0],
        metrics=['acc']
    )
    return model
