#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(kernel='rbf', C=10000.0)

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
predictions = clf.predict(features_test)
print "training time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(predictions, labels_test)
print "accuracy: ", round(accuracy, 3)

print "answer 10: ", predictions[10]
print "answer 26: ", predictions[26]
print "answer 50: ", predictions[50]

# Count Chris
num_chris = 0
for p in predictions:
    if p == 1:
        num_chris += 1

print "Number of Chris' emails: ", num_chris

#########################################################
