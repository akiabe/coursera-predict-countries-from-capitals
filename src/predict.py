import numpy as np

def cosine_similarity(A, B):
    """
    :param A: a numpy array which corresponds to a word vector
    :param B: a numpy which corresponding to a word vector
    :return cos: numerical number representing the cosine similarity between A and B
    """
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (np.dot(norma, normb))

    return cos

def euclidean(A, B):
    """
    :param A: a numpy array which corresponds to a word vector
    :param B: a numpy array which corresponds to a word vector
    :return d: numerical number representing the Euclidean distance between A and B
    """
    # euclidean distance
    d = np.linalg.norm(A - B)

    return d

def get_country(city1, country1, city2, embeddings):
    """
    :param city1: a string (the capital city of country1)
    :param country1: a string (the country of capital1)
    :param city2: a string (the capital city of country2)
    :param embeddings: a dictionary where the keys are words and values are their embeddings
    :return countries: a dictionary with the most likely country and its similarity score
    """
    # store the city1, country1, city2 in a set
    group = set((city1, country1, city2))

    # get embeddings of city1, country 1, city2
    city1_emb = embeddings[city1]
    country1_emb = embeddings[country1]
    city2_emb = embeddings[city2]

    # get embeddings of country2
    vec = country1_emb - city1_emb + city2_emb

    # initialize the similarity
    similarity = -1

    # initialize country to an empty string
    country = ''

    # loop through all word in the embedding dictionary
    for word in embeddings.keys():
        # check word is not already in the group
        if word not in group:
            # get embeddings
            word_emb = embeddings[word]
            # calculate cosine similarity between country2 and the word
            cur_similarity = cosine_similarity(vec, word_emb)
            # if the cosine similarity is more similar than previous best similarity
            if cur_similarity > similarity:
                # update the similarity
                similarity = cur_similarity
                # store the country as a tuple
                country = (word, similarity)

    return country