import pickle
from gensim.models import KeyedVectors

embeddings = KeyedVectors.load_word2vec_format('../input/GoogleNews-vectors-negative300.bin', binary = True)

def get_word_embeddings(embeddings):
    word_embeddings = {}

    for word in embeddings.vocab:
        word_embeddings[word] = embeddings[word]

    return word_embeddings

word_embeddings = get_word_embeddings(embeddings)
print(len(word_embeddings))
pickle.dump(word_embeddings, open("../input/word_embeddings_subset.p", "wb"))