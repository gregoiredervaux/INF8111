import json
import re

from TFIDFBoW import TFIDFBoW
from PreProcessing import PreprocessingPipeline

def extract_tweet_content(raw_tweet_file):
    """
    Extract the tweet content for each json object

    raw_tweet_file: file path that contains all json objects

    :return: a list with the tweet contents
    """
    try:
        tweet_list = []
        with open("e_corp_dataset.txt", "r") as reader:
            for line in reader:
                json_line = json.load(line)
                tweet_list.append(json_line)
        return tweet_list
    except:
        raise NotImplementedError("")


def detect_airline(tweet_msg):
    """
    Detect and return the airline companies mentioned in the tweet

    tweet_msg: represents the tweet message.

    :return: list of detected airline companies
    """
    try:
        compagnies = ["Air France",
                      "American Airways",
                      "British Airways",
                      "Delta",
                      "Southwest",
                      "United",
                      "Us Airways",
                      "Virgin America"]

        present_compagnies = []
        for compagnie in compagnies:
            if re.match(compagnie, tweet_msg):
                present_compagnies.append(compagnie)
        return present_compagnies

    except:
        raise NotImplementedError("")


def extract_sentiment(classifier, tweet):
    """
    Extract the tweet sentiment

    classifier: classifier object
    tweet: represents the tweet message. You should define the data type

    :return: list of detected airline companies
    """
    pre_process_pipe = PreprocessingPipeline(True, True, True)

    bow_tweet

    bow_tweet = bow_obj.fit_transform([tweet])

    raise NotImplementedError("")