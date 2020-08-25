import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def __cluster_reults(sentences, embedding, sentiments, labels, pad=0):
    """
    """
    decoder = pd.read_pickle('results\\objects\\decoder.pkl')
    sentiment_mapping = {
        0: 'Neutral',
        1: 'Positive',
        2: 'Negative'
    }
    results = []

    for cluster in np.unique(labels):

        df = pd.DataFrame(
            columns=[
                'sentence',
                'predicted_sentiment',
                'cluster',
                'embedding'
            ]
        )
        idx = np.where(labels == cluster)

        cluster_sentences = []
        df['predicted_sentiment'] = np.vectorize(
            sentiment_mapping.get
        )(sentiments[idx])
        df['embedding'] = embedding[idx].tolist()
        df['cluster'] = cluster
        for sentence in sentences[idx]:

            words = [decoder[word] for word in sentence if word != pad]
            sentence_string = ' '.join(words)
            cluster_sentences.append(sentence_string)

        df['sentence'] = np.array(cluster_sentences)
        results.append(df)

    results = pd.concat(results)
    return results


def __cluster_visualizer(manifold, labels, title, **kwargs):
    """
    """
    plt.figure(figsize=(10, 10))
    if manifold.shape[1] > 2:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    for label in np.unique(labels):

        idx = np.where(labels == label)
        if label == -1:
            alpha = 0.05
            label = 'Noise'
        else:
            alpha = 1
        if manifold.shape[1] > 2:
            ax.scatter3D(
                manifold[idx, 0],
                manifold[idx, 1],
                manifold[idx, 2],
                label=f'Cluster {label}',
                alpha=alpha,
                **kwargs
            )
        else:
            ax.scatter(
                manifold[idx, 0],
                manifold[idx, 1],
                label=f'Cluster {label}',
                alpha=alpha,
                **kwargs
            )

    ax.legend(markerscale=6)
    plt.title(title)

    plt.show()


def cluster_inspection(manifold, sentences, sentiments, labels, title,
                       embedding, pad=0, verbose=10, **kwargs):
    """
    """
    __cluster_visualizer(
        manifold,
        labels,
        title,
        **kwargs
    )
    results = __cluster_reults(
        sentences,
        embedding,
        sentiments,
        labels,
        pad=0
    )

    if verbose > 0:
        for cluster, slice in results.groupby('cluster'):

            print('')
            print(f'Cluster {cluster}')
            print('')
            sent, prop = np.unique(
                slice['predicted_sentiment'].values,
                return_counts=True
            )
            prop = prop / prop.sum()
            prop = [round(perc, 3) for perc in prop]
            for s, p in zip(sent, prop):

                print(s, ':', p)

            print('')
            sample_sentences = np.random.choice(
                slice['sentence'].values,
                verbose
            )
            for sentence in sample_sentences:

                print(f'* {sentence}')
                print('')

    return results
