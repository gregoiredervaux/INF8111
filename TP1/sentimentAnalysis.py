from data_extraction import *
from model import *


def extract_sentiment(classifier, tweet):
    """
    Extract the tweet sentiment

    classifier: classifier object
    tweet: represents the tweet message, as a row of the BoW matrix

    :return: list of detected airline companies
    """

    try:
        return classifier.predict(tweet)
    except:
        raise NotImplementedError("")

if __name__ == "__main__":

    # Path of training dataset
    trainingPath = "./train_data.tsv"

    # Path of validation dataset
    validationPath = "./dev_data.tsv"

    training_X, training_Y = load_dataset(trainingPath)
    validation_X, validation_Y = load_dataset(validationPath)
    pre_process_pipe = PreprocessingPipeline(True, True, True)

    bow = CountBoW(pre_process_pipe, False, False)

    classifier, _, _ = train_evaluate(training_X, training_Y, validation_X, validation_Y, bow)

    raw_tweets = extract_tweet_content("saved_tweets.txt")
    bow_tweets = bow.transform(raw_tweets)

    sentiment_vector = classifier.predict(bow_tweets)

    print("verif: ")
    for i in range(len(sentiment_vector[:20])):
        print("\nraw: " + str(raw_tweets[i]))
        print("sentiment: " + str(sentiment_vector[i]))

    print(classifier.predict(bow_tweets[0]))



