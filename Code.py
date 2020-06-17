# Author: Srishti Chaudhary

import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm, metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid
import wittgenstein
import matplotlib.pyplot as plt
import seaborn


# Creating the Dataset Required

def makeDataset():
    emails = pandas.read_csv('C:/Users/Srishti Chaudhary/Desktop/AI/Project/Dataset.csv')

    split_data = emails["text"].str.split("\n", n=1)
    data = split_data.to_list()
    names = ["subject", "body"]

    Dataset = pandas.DataFrame(data, columns=names)
    Dataset['message'] = emails['text'].str[9:]
    Dataset['label'] = emails['label_num'].astype(int)

    # Final Dataset
    Dataset['subject'] = Dataset['subject'].str[9:]

    return Dataset


# Extracting Features

# Term Frequency Feature Matrix

def TF_FM (Dataset):
    # # Using Stop List
    # vect_Tf = TfidfVectorizer(analyzer='word', stop_words=ENGLISH_STOP_WORDS, min_df=3, max_features= 1000, use_idf = False)

    # Not using Stop List
    vect_Tf = TfidfVectorizer(analyzer='word', min_df=2, max_features= 1000, use_idf = False)

    X_Tf = vect_Tf.fit_transform(Dataset)

    return X_Tf

# TF-IDF Feature Matrix

def TFIDF_FM (Dataset):
    # # Using Stop List
    # vect_Tfidf = TfidfVectorizer(analyzer='word', stop_words=ENGLISH_STOP_WORDS, min_df=3, max_features= 1000)

    # Not using Stop List
    vect_Tfidf = TfidfVectorizer(analyzer='word', min_df=3, max_features= 1000)

    X_Tfid = vect_Tfidf.fit_transform(Dataset)

    return X_Tfid

# Binary Feature Matrix

def B_FM (Dataset):
    # Using Stop List
    vect_binary = CountVectorizer(analyzer='word', stop_words=ENGLISH_STOP_WORDS, binary=True, min_df=3, max_features= 1000)

    # # Not using Stop List
    # vect_binary = CountVectorizer(analyzer='word', stop_words=ENGLISH_STOP_WORDS, binary=True, min_df=3, max_features= 1000)

    X_binary = vect_binary.fit_transform(Dataset)

    return X_binary

# Support Vector Machine

def SVM (FeatureMatrix, Labels):

    XTrain, XTest, LabelTrain, LabelTest  = train_test_split (FeatureMatrix, Labels,  test_size = 0.1)

    # training model on dataset
    clf = svm.LinearSVC()
    clf.fit(XTrain, LabelTrain)

    # testing model on dataset
    expected = LabelTest
    predicted = clf.predict(XTest)

    return (expected, predicted)


# Boosting with Decision Tree

def BDT (FeatureMatrix, Labels):

    XTrain, XTest, LabelTrain, LabelTest  = train_test_split (FeatureMatrix, Labels,  test_size = 0.1)

    # training model on dataset
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(XTrain, LabelTrain)

    # testing model on dataset
    expected = LabelTest
    predicted = clf.predict(XTest)

    return (expected, predicted)

# RIPPER

def RIPPER (FeatureMatrix, Labels):

    FeatureMatrix = pandas.DataFrame.sparse.from_spmatrix(FeatureMatrix)

    XTrain, XTest, LabelTrain, LabelTest  = train_test_split (FeatureMatrix, Labels,  test_size = 0.1)
    
    # training model on dataset
    clf = wittgenstein.RIPPER()
    clf.fit(XTrain, LabelTrain, class_feat=None, pos_class='1')

    # testing model on dataset
    expected = LabelTest
    predicted = clf.predict(XTest)

    return (expected, predicted)

# ROCCHIO

def ROCCHIO (FeatureMatrix, Labels):

    samples, features = FeatureMatrix.shape

    XTrain, XTest, LabelTrain, LabelTest  = train_test_split (FeatureMatrix, Labels,  test_size = 0.1)

    # training model on dataset
    clf = NearestCentroid()
    clf.fit(XTrain, LabelTrain)

    # testing model on dataset
    expected = LabelTest
    predicted = clf.predict(XTest)

    return (expected, predicted)


# Calling the Dataset function

Dataset = makeDataset()

# Creating the feature matrix for subject, body or message

FeatureMatrix = B_FM(Dataset['subject'])

# Calculating the average miss rate and average false alarm rate

total_far = 0
total_mr = 0

for x in range(10):
    expected, predicted = RIPPER(FeatureMatrix, Dataset['label'])
    cm = metrics.confusion_matrix(expected, predicted)
    mr = cm[1][0] / (cm[1][0] + cm[1][1])
    total_mr = total_mr + mr
    far = cm[0][1] / (cm[0][0] + cm[0][1])
    total_far = total_far + far

MissRate = total_mr / 10
print(MissRate)

FalseAlarmRate = total_far / 10
print(FalseAlarmRate)












