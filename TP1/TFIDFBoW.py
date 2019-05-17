from CountBoW import CountBoW
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

class TFIDFBoW(CountBoW):

    def __init__(self, pipeline, bigram=False, trigram=False):
        """
        pipelineObj: instance of PreprocesingPipeline
        bigram: enable or disable bigram
        trigram: enable or disable trigram
        """
        self.pipeline = pipeline
        self.bigram = bigram
        self.trigram = trigram
        self.count_bow = CountBoW(pipeline, False, False)
        self.idf = []
        self.tokens_index = []

    def get_idf(self, bow_matrix):
        """
        this methode return the idf of the set of tokens

        bow_matrix: the matrix of the raw frequency
        :return: a list of the idf by index
        """

        idf = []
        for i in range(len(self.tokens_index)):
            dfi = 0
            for j in range(bow_matrix.shape[0]):
                # on compte le nombre de documents qui contiennent ce mot
                dfi += 1 if bow_matrix[j, i] != 0 else 0
            # on calcul et ajoute le idf a la liste des idf
            idf.append(math.log(bow_matrix.shape[0] / dfi))
        return idf

    def get_weighted_matrix(self, bow_matrix, save_idf = False):
        """
        this methode return the new matrix by tf-idf

        bow_matrix: the matrix of the raw frequency
        :return: the new matrix weigted
        """

        idf = self.get_idf(bow_matrix) if save_idf else self.idf
        if save_idf: self.idf = idf
        for j in range(bow_matrix.shape[0]):
            for i in range(len(self.tokens_index)):
                bow_matrix[j, i] = bow_matrix[j, i] * idf[i]
        return bow_matrix

    def fit_transform(self, X):
        """
        This method preprocesses the data using the pipeline object, calculates the IDF and TF and
        transforms the text in vectors. Vectors are weighted using TF-IDF method.

        X: a list that contains tweet contents

        :return: a list that contains the list of integers
        """
        self.tokens_index = []
        fited_tweets = [[]]
        try:
            for x in X:
                if x != "Not Available":
                    tokens = self.pipeline.preprocess(x)
                    tokens_list = self.get_tokens_list(tokens)
                    new_fited_tweet = [0 for i in range(len(fited_tweets[0]))]
                    for token in tokens_list:
                        if token in self.tokens_index:
                            new_fited_tweet[self.tokens_index.index(token)] += 1
                        else:
                            for fited_tweet in fited_tweets:
                                fited_tweet.append(0)
                            new_fited_tweet.append(1)
                            self.tokens_index.append(token)
                    fited_tweets.append(new_fited_tweet)
        except:
            raise NotImplementedError("")

        return csr_matrix(self.get_weighted_matrix(lil_matrix(fited_tweets[1:]), True))

    def transform(self, X):
        """
        This method preprocesses the data using the pipeline object and
            transforms the text in a list of integer.

        X: a list of vectors

        :return: a list of vectors
        """
        fited_tweets = lil_matrix((len(X), len(self.tokens_index)), dtype=np.int8)

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

        return lil_matrix(self.get_weighted_matrix(fited_tweets))
