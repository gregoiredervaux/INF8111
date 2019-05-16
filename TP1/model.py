import time
import codecs
import re

from CountBoW import CountBoW
from TFIDFBoW import TFIDFBoW
from PreProcessing import PreprocessingPipeline

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def load_dataset(path):
    dtFile = codecs.open(path, 'r')

    x = []
    y = []

    for l in dtFile:
        sid, uid, label, text = re.split(r"\s+", l, maxsplit=3)

        text = text.strip()

        # Remove not available
        if text == "Not Available":
            continue

        x.append(text)

        if label == "negative":
            y.append(0)
        elif label == "neutral":
            y.append(1)
        elif label == "positive":
            y.append(2)

    assert len(x) == len(y)

    return x, y

def train_evaluate(training_X, training_Y, validation_X, validation_Y, bowObj):
    """
    training_X: tweets from the training dataset
    training_Y: tweet labels from the training dataset
    validation_X: tweets from the validation dataset
    validation_Y: tweet labels from the validation dataset

    bowObj: Bag-of-word object

    :return: the classifier and its accuracy in the training and validation dataset.
    """

    classifier = LogisticRegression(solver='liblinear', multi_class='auto')

    debut_fit_trans = time.time()
    training_rep = bowObj.fit_transform(training_X)
    fin_fit_trans = time.time() - debut_fit_trans

    print("     taille dico: " + str(training_rep[0].shape[0]) + " nb tweet: " + str(training_rep.shape[0]))

    debut_train = time.time()
    classifier.fit(training_rep, training_Y)
    fin_train = time.time() - debut_train

    trainAcc = accuracy_score(training_Y, classifier.predict(training_rep))

    debut_valid = time.time()
    validationAcc = accuracy_score(validation_Y, classifier.predict(bowObj.transform(validation_X)))
    fin_valid = time.time() - debut_valid

    print("     temps: fit_trans: {}s training: {}s validation: {}s".format(fin_fit_trans, fin_train, fin_valid))

    return classifier, trainAcc, validationAcc


def custom_evaluate(training_X, training_Y, validation_X, validation_Y, options):
    """
        training_X: tweets from the training dataset
        training_Y: tweet labels from the training dataset
        validation_X: tweets from the validation dataset
        validation_Y: tweet labels from the validation dataset
        options: options on the programme

        :return: the classifier and its accuracy in the training and validation dataset.
        """
    pre_process_pipe = PreprocessingPipeline(options["tokenisation"],
                                             options["preprocess"],
                                             options["stemming"])
    if options["Bow_methode"] == "CountBow":
        bow = CountBoW(pre_process_pipe, options["bigram"], options["trigram"])
    else:
        bow = TFIDFBoW(pre_process_pipe, options["bigram"], options["trigram"])

    classifier = LogisticRegression(solver='liblinear', multi_class='auto')

    debut_fit_trans = time.time()
    training_rep = bow.fit_transform(training_X)
    fin_fit_trans = time.time() - debut_fit_trans

    print("     taille dico: " + str(training_rep[0].shape[0]) + " nb tweet: " + str(training_rep.shape[0]))

    debut_train = time.time()
    classifier.fit(training_rep, training_Y)
    fin_train = time.time() - debut_train

    trainAcc = accuracy_score(training_Y, classifier.predict(training_rep))

    debut_valid = time.time()
    validationAcc = accuracy_score(validation_Y, classifier.predict(bow.transform(validation_X)))
    fin_valid = time.time() - debut_valid

    print("     temps: fit_trans: {}s training: {}s validation: {}s".format(fin_fit_trans, fin_train, fin_valid))

    return classifier, trainAcc, validationAcc

def test_model(ls_options):

    for option in ls_options:
        try:
            print("\n" + option["name"] + " ----\n")
            _, trainAcc, validationAcc = custom_evaluate(training_X, training_Y, validation_X, validation_Y, option)
            print("\n     trainAcc: " + str(trainAcc))
            print("     validationAcc: " + str(validationAcc))

        except MemoryError:
            print("not enough memory for this one")



# Path of training dataset
trainingPath = "./train_data.tsv"

# Path of validation dataset
validationPath = "./dev_data.tsv"

training_X, training_Y = load_dataset(trainingPath)
validation_X, validation_Y = load_dataset(validationPath)
pre_process_pipe = PreprocessingPipeline(True, True, True)

bow = CountBoW(pre_process_pipe, False, False)

test_model([
    {
        "name": "CountBoW + SpaceTokenizer(without tokenizer) + unigram",
        "Bow_methode": "CountBow",
        "tokenisation": False,
        "preprocess": False,
        "stemming": False,
        "bigram": False,
        "trigram": False
    },
    {
        "name": "CountBoW + NLTKTokenizer + unigram",
        "Bow_methode": "CountBow",
        "tokenisation": True,
        "preprocess": False,
        "stemming": False,
        "bigram": False,
        "trigram": False
    },
    {
        "name": "TFIDFBoW + NLTKTokenizer + unigram",
        "Bow_methode": "CountBow",
        "tokenisation": True,
        "preprocess": False,
        "stemming": False,
        "bigram": False,
        "trigram": False
    },
    {
        "name": "TFIDFBoW + NLTKTokenizer + Stemming + unigram",
        "Bow_methode": "CountBow",
        "tokenisation": True,
        "preprocess": False,
        "stemming": True,
        "bigram": False,
        "trigram": False
    },
    {
        "name": "TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram",
        "Bow_methode": "CountBow",
        "tokenisation": True,
        "preprocess": True,
        "stemming": True,
        "bigram": False,
        "trigram": False
    },
    {
        "name": "TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram",
        "Bow_methode": "CountBow",
        "tokenisation": True,
        "preprocess": True,
        "stemming": True,
        "bigram": True,
        "trigram": False
    },
    {
        "name": "TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram + trigram",
        "Bow_methode": "CountBow",
        "tokenisation": True,
        "preprocess": True,
        "stemming": True,
        "bigram": True,
        "trigram": True
    },
])