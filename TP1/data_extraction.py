import json
import re

def extract_tweet_content(raw_tweet_file):
    """
    Extract the tweet content for each json object

    raw_tweet_file: file path that contains all json objects

    :return: a list with the tweet contents
    """
    tweet_list = []
    try:
        with open(raw_tweet_file, "r") as reader:
            for line in reader:
                json_line = json.loads(line)
                tweet_list.append(json_line["text"])
    except:
        raise NotImplementedError("")
    return tweet_list

def detect_airline(tweet_msg):
    """
    Detect and return the airline companies mentioned in the tweet

    tweet_msg: represents the tweet message.

    :return: list of detected airline companies
    """

    present_compagnies = []
    try:
        compagnies = "(Air ?France|" \
                     "American ?Airways|" \
                     "British ?Airways|" \
                     "Delta|" \
                     "Southwest|" \
                     "United|" \
                     "Us ?Airways|" \
                     "Virgin ?America)"
        if re.match(r'.+?' + compagnies, tweet_msg, re.IGNORECASE | re.DOTALL):
            present_compagnies.append(tweet_msg)

    except:
        raise NotImplementedError("")

    return present_compagnies

def save_airline_tweets():

    try:
        i = 0
        j = 0
        with open("e_corp_dataset.txt", "r") as reader:
            for line in reader:
                json_line = json.loads(line)
                with open("saved_tweets.txt", "a") as writer:

                    if len(detect_airline(json_line["text"])) != 0:
                        writer.write(line)
                        print("\r{} tweets selectionnés sur {}".format(i, j), end="")
                        i += 1
                    j += 1
    except:
        raise NotImplementedError("")

if __name__ == "__main__":
    save_airline_tweets()

