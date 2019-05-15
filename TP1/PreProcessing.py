import nltk
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk.stem.snowball import SnowballStemmer

import re

class SpaceTokenizer(object):
    """
    It tokenizes the tokens that are separated by whitespace (space, tab, newline).
    We consider that any tokenization was applied in the text when we use this tokenizer.

    For example: "hello\tworld of\nNLP" is split in ['hello', 'world', 'of', 'NLP']
    """

    def tokenize(self, text):
        try:
            tokens = text.split()
        except:
            raise NotImplementedError("space tokenize error")
        return tokens


class NLTKTokenizer(object):
    """
    This tokenizer uses the default function of nltk package (https://www.nltk.org/api/nltk.html) to tokenize the text.
    """

    def tokenize(self, text):
        try:
            tokens = nltk.word_tokenize(text)
        except:
            raise NotImplementedError("nltk tokenize error")

        # Have to return a list of tokens
        return tokens


class Stemmer(object):

    def __init__(self):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

    def stem(self, tokens):
        """
        tokens: a list of strings
        """
        try:
            tokens = [self.stemmer.stem(token) for token in tokens]
        except:
            raise NotImplementedError("")

        return tokens


class TwitterPreprocessing(object):

    def preprocess(self, tweet):
        """
        tweet: original tweet
        """
        try:
            if tweet == "Not Available":
                return tweet
            # on retire les mots qui commencent par des @ (tags) et http (liens)
            tweet = re.sub("(http|@)[^ \t\n\r]*([\s]|$)", "", tweet)
            # on retire les #
            tweet = re.sub("#", "", tweet)
            # on remplace les chiffres par des espaces
            tweet = re.sub("[0-9]", " ", tweet)

        except:
            raise NotImplementedError("")

        # return the preprocessed twitter
        return tweet


class PreprocessingPipeline:

    def __init__(self, tokenization, twitterPreprocessing, stemming):
        """
        tokenization: enable or disable tokenization.
        twitterPreprocessing: enable or disable twitter preprocessing.
        stemming: enable or disable stemming.
        """

        self.tokenizer = NLTKTokenizer() if tokenization else SpaceTokenizer()
        self.twitterPreprocesser = TwitterPreprocessing() if twitterPreprocessing else None
        self.stemmer = Stemmer() if stemming else None

    def preprocess(self, tweet):
        """
        Transform the raw data

        tokenization: boolean value.
        twitterPreprocessing: boolean value. Apply the
        stemming: boolean value.
        """
        if self.twitterPreprocesser:
            tweet = self.twitterPreprocesser.preprocess(tweet)

        tokens = self.tokenizer.tokenize(tweet)

        if self.stemmer:
            tokens = self.stemmer.stem(tokens)

        return tokens