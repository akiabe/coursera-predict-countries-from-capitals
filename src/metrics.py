import predict

def get_accuracy(word_embeddings, data):
    """
    :param word_embeddings: a dictionary where the key is a word and the value is its embedding
    :param data: a pandas dataframe containing all the country and capital city pairs
    :return accuracy: the accuracy of the model
    """
    # initialize num correct
    num_correct = 0

    # loop through the rows of the dataframe
    for i, row in data.iterrows():
        # get city and country name
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']

        # predict country2
        predicted_country2, _ = predict.get_country(city1, country1, city2, word_embeddings)

        # if the predicted country 2 is the same as the actual country2
        if predicted_country2 == country2:
            # increment the number of correct
            num_correct += 1

    # get the number of rows in the data dataframe
    m = len(data)

    accuracy = num_correct / m

    return accuracy