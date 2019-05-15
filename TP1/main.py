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


# Path of training dataset
trainingPath = "./train_data.tsv"

# Path of validation dataset
validationPath = "./dev_data.tsv"

training_X, training_Y = load_dataset(trainingPath)
validation_X, validation_Y = load_dataset(validationPath)
pre_process_pipe = PreprocessingPipeline(True, True, True)

bow = CountBoW(pre_process_pipe, False, False)

def train_evaluate(training_X, training_Y, validation_X, validation_Y, bowObj):
    """
    data: [
        training_X: tweets from the training dataset
        training_Y: tweet labels from the training dataset
        validation_X: tweets from the validation dataset
        validation_Y: tweet labels from the validation dataset
        ]
    bowObj: Bag-of-word object

    :return: the classifier and its accuracy in the training and validation dataset.
    """

    classifier = LogisticRegression(solver='liblinear', multi_class='auto')
    debut_fit_trans = time.time()
    training_rep = bowObj.fit_transform(training_X)
    print("     taille dico: " + str(len(training_rep[0])) + " nb tweet: " + str(len(training_rep)) + " temps: " + str(time.time() - debut_fit_trans))
    debut_train = time.time()
    classifier.fit(training_rep, training_Y)
    fin_train = time.time() - debut_train
    print('     training: ' + str(fin_train) + "s")
    trainAcc = accuracy_score(training_Y, classifier.predict(training_rep))
    debut_valid = time.time()
    validationAcc = accuracy_score(validation_Y, classifier.predict(bowObj.transform(validation_X)))
    fin_valid = time.time() - debut_valid
    print('     validation: ' + str(fin_valid) + "s")
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
    print("     taille dico: " + str(len(training_rep[0])) + " nb tweet: " + str(len(training_rep)) + " temps: " + str(
        time.time() - debut_fit_trans))

    debut_train = time.time()
    classifier.fit(training_rep, training_Y)
    fin_train = time.time() - debut_train
    print('     training: ' + str(fin_train) + "s")

    trainAcc = accuracy_score(training_Y, classifier.predict(training_rep))

    debut_valid = time.time()
    validationAcc = accuracy_score(validation_Y, classifier.predict(bow.transform(validation_X)))
    fin_valid = time.time() - debut_valid
    print('     validation: ' + str(fin_valid) + "s")

    return classifier, trainAcc, validationAcc

# data testing
print("\ndata testing ----\n")
print("     data shape: trainX: {}, TrainY: {}, validX: {}, validY: {}".format(len(training_X),
                                                                         len(training_Y),
                                                                         len(validation_X),
                                                                         len(validation_Y)))

classifier, trainAcc, validationAcc = train_evaluate(training_X, training_Y, validation_X, validation_Y, bow)

print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# CountBoW + SpaceTokenizer(without tokenizer) + unigram

print("\nCountBoW + SpaceTokenizer(without tokenizer) + unigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "CountBow",
                                                 "tokenisation": False,
                                                 "preprocess": False,
                                                 "stemming": False,
                                                 "bigram": False,
                                                 "trigram": False
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# CountBoW + NLTKTokenizer + unigram

print("\nCountBoW + NLTKTokenizer + unigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "CountBow",
                                                 "tokenisation": True,
                                                 "preprocess": False,
                                                 "stemming": False,
                                                 "bigram": False,
                                                 "trigram": False
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# TFIDFBoW + NLTKTokenizer + unigram

print("\nTFIDFBoW + NLTKTokenizer + unigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "TFIDFBoW",
                                                 "tokenisation": True,
                                                 "preprocess": False,
                                                 "stemming": False,
                                                 "bigram": False,
                                                 "trigram": False
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# FIDFBoW + NLTKTokenizer + Stemming + unigram

print("\nFIDFBoW + NLTKTokenizer + Stemming + unigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "TFIDFBoW",
                                                 "tokenisation": True,
                                                 "preprocess": False,
                                                 "stemming": True,
                                                 "bigram": False,
                                                 "trigram": False
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram

print("\nTFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "TFIDFBoW",
                                                 "tokenisation": True,
                                                 "preprocess": True,
                                                 "stemming": True,
                                                 "bigram": False,
                                                 "trigram": False
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram

print("\nTFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "TFIDFBoW",
                                                 "tokenisation": True,
                                                 "preprocess": True,
                                                 "stemming": True,
                                                 "bigram": True,
                                                 "trigram": False
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

# TFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram + trigram

print("\nTFIDFBoW + NLTKTokenizer + Twitter preprocessing + Stemming  + unigram + bigram + trigram ----\n")
_, trainAcc, validationAcc = custom_evaluate(training_X,
                                             training_Y,
                                             validation_X,
                                             validation_Y,
                                             {
                                                 "Bow_methode": "TFIDFBoW",
                                                 "tokenisation": True,
                                                 "preprocess": True,
                                                 "stemming": True,
                                                 "bigram": True,
                                                 "trigram": True
                                             })
print("     trainAcc: " + str(trainAcc))
print("     validationAcc: " + str(validationAcc))

