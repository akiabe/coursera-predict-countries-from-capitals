import pickle
import pandas as pd
import matplotlib.pyplot as plt

#import metrics
import models

if __name__ == "__main__":
    # load train set
    data = pd.read_csv("../input/train.txt", delimiter=' ')
    data.columns = ['city1', 'country1', 'city2', 'country2']

    # load word embeddings
    word_embeddings = pickle.load(open("../input/word_embeddings_subset.p", "rb"))

    #accuracy = metrics.get_accuracy(word_embeddings, data)
    #print(f"Accuracy is {accuracy:.2f}")

    words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
             'village', 'country', 'continent', 'petroleum', 'joyful']

    # given a list of words and the embeddings
    X = models.get_vectors(word_embeddings, words)

    # plotting the vectors using pca
    result = models.compute_pca(X, 2)
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

    plt.show()
















