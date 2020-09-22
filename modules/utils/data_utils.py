import numpy as np

from tensorflow.keras.utils import Sequence

from nltk.tokenize import word_tokenize


class DataGenerator(Sequence):
    '''
    Class implementing a data generator
    '''
    def __init__(self, list_batches, shuffle=True):
        '''
        '''
        self.list_batches = list_batches
        self.shuffle = shuffle
        self.root_dir = f'data'
        self.on_epoch_end()

    def __len__(self):
        '''
        Denotes the number of batches per epoch'
        '''
        return int(len(self.list_batches))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''
        # Pick a batch
        batch = self.list_batches[index]
        # Generate X and y
        X, y = self.__data_generation(batch)
        return X, y

    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        '''
        if self.shuffle is True:
            np.random.shuffle(self.list_batches)

    def __data_generation(self, batch):
        '''
        Generates data containing batch_size samples
        '''
        X = np.load(
            f'{self.root_dir}\\inputs\\{batch}.npy',
            allow_pickle=True
        )
        y_1 = np.load(
            f'{self.root_dir}\\targets\\{batch}_1.npy',
            allow_pickle=True
        )
        y_2 = np.load(
            f'{self.root_dir}\\targets\\{batch}_2.npy',
            allow_pickle=True
        )
        return [X], [y_1, y_2]


def padding_list(my_list, max_len, pad):
    """
    """
    if len(my_list) > max_len:
        my_list = my_list[:max_len]
        return my_list
    elif len(my_list) < max_len:
        my_list = my_list + ([pad] * (max_len - len(my_list)))
        return my_list
    else:
        return my_list


def tokenization(sentence):
    """
    """
    tokens = sentence.split()
    tokens = [token for token in tokens if token[0] != '@']
    sentence = ' '.join(tokens)
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = tokens + ['<EOS>']
    tokens = ['<START>'] + tokens
    return tokens


def encoding(bag_of_sentences, max_len):
    """
    """
    unique_words = set(
        [word for sentence in bag_of_sentences for word in sentence]
    )
    encoder = {word: code for code, word in enumerate(unique_words, 1)}
    decoder = {code: word for word, code in encoder.items()}
    return encoder, decoder


def preprocessing(list_sentences, targets, max_len=60,
                  max_batch=64):
    """
    """
    dict_sentences = {length: [] for length in range(2, max_len)}
    dict_targets = {length: [] for length in range(2, max_len)}
    bag_of_sentences = [
        tokenization(sentence) for sentence in list_sentences
    ]
    print(bag_of_sentences)
    encoder, decoder = encoding(
        bag_of_sentences=bag_of_sentences,
        max_len=max_len
    )

    for sentence, target in zip(bag_of_sentences, targets):

        if len(sentence) < 3:
            continue
        bag_of_words = [encoder[word] for word in sentence]
        dict_sentences[len(bag_of_words)].append(bag_of_words)

        dict_targets[len(bag_of_words)].append([target] * len(bag_of_words))

    batch_count = 0
    for length in range(2, max_len):

        sentence_batch = np.array(dict_sentences[length])

        if len(sentence_batch.shape) < 2:
            print(sentence_batch.shape)
            continue
        sentence_tar_batch = sentence_batch[:, 1:]
        sentence_batch = sentence_batch[:, :-1]

        class_tar_batch = np.array(dict_targets[length])
        class_tar_batch = class_tar_batch[:, -1]

        num_batches = (
            sentence_batch.shape[0] + max_batch - 1
        ) // max_batch

        for batch_index in range(num_batches):

            minimum = min(
                sentence_batch.shape[0],
                (batch_index + 1) * max_batch
            )

            sentence_sub_batch = sentence_batch[
                batch_index * max_batch: minimum
            ]
            sentence_sub_batch = np.float32(sentence_sub_batch)

            sentence_tar_sub_batch = sentence_tar_batch[
                batch_index * max_batch: minimum
            ]
            sentence_tar_sub_batch = np.float32(sentence_tar_sub_batch)

            class_tar_sub_batch = class_tar_batch[
                batch_index * max_batch: minimum
            ]
            class_tar_sub_batch = np.float32(class_tar_sub_batch)

            if len(sentence_sub_batch.shape) < 2:
                print(sentence_sub_batch.shape)

            np.save(f'data\\inputs\\{batch_count}', sentence_sub_batch)
            np.save(f'data\\targets\\{batch_count}_1', class_tar_sub_batch)
            np.save(f'data\\targets\\{batch_count}_2', sentence_tar_sub_batch)

            batch_count += 1

    return encoder, decoder
