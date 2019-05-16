from scipy.sparse import csc_matrix
import numpy as np

def bigram(tokens):
    """
    tokens: a list of strings
    """
    len_tokens = len(tokens)
    bigram = []
    i = 0
    while i < len_tokens - 1:
        bigram.append(tokens[i] + ' ' + tokens[i + 1])
        i += 1
    return bigram


def trigram(tokens):
    """
    tokens: a list of strings
    """
    len_tokens = len(tokens)
    trigram = []
    i = 0
    while i < len_tokens - 2:
        trigram.append(tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2])
        i += 1
    return trigram

class CountBoW(object):

    def __init__(self, pipeline, bigram=False, trigram=False):
        """
        pipelineObj: instance of PreprocesingPipeline
        bigram: enable or disable bigram
        trigram: enable or disable trigram
        """
        self.pipeline = pipeline
        self.bigram = bigram
        self.trigram = trigram
        self.tokens_index = []

    def get_tokens_list(self, tokens):
        tokens_list = []
        if self.bigram and self.trigram:
            tokens_list = bigram(tokens) + trigram(tokens)
        else:
            if self.bigram:
                tokens_list = bigram(tokens)
            elif self.trigram:
                self.tokens_list = tokens
            else:
                tokens_list = tokens
        return tokens_list

    def fit_transform(self, X):
        """
        This method preprocesses the data using the pipeline object, relates each unigram, bigram or trigram to a specific integer and
        transforms the text in a vector. Vectors are weighted using the token frequencies in the sentence.

        X: a list that contains tweet contents

        :return: a list of vectors
        """

        self.tokens_index = []
        fited_tweets = [[]]
        try:
            for x in X:
                if x != "Not Available":
                    tokens = self.pipeline.preprocess(x)
                    tokens_list = self.get_tokens_list(tokens)
                    fited_tweets.append([0 for i in range(len(fited_tweets[0]))])
                    for token in tokens_list:
                        if token in self.tokens_index:
                            fited_tweets[-1][self.tokens_index.index(token)] += 1
                        else:
                            for fited_tweet in fited_tweets:
                                fited_tweet.append(0)
                            fited_tweets[-1][-1] += 1
                            self.tokens_index.append(token)
        except:
            raise NotImplementedError("")
        return csc_matrix(fited_tweets[1:])

    def transform(self, X):
        """
        This method preprocesses the data using the pipeline object and  transforms the text in a list of integer.
        Vectors are weighted using the token frequencies in the sentence.

        X: a list of vectors

        :return: a list of vectors
        """
        fited_tweets = csc_matrix((len(X), len(self.tokens_index)), dtype=np.int8)
        try:
            index = 0
            for x in X:
                tokens = self.pipeline.preprocess(x)
                tokens_list = self.get_tokens_list(tokens)
                for token in tokens_list:
                    if token in self.tokens_index:
                        fited_tweets[index, self.tokens_index.index(token)] += 1
                index += 1
        except:
            raise NotImplementedError("")

        return csc_matrix(fited_tweets)
