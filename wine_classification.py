from __future__ import division
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split as tts

wine = datasets.load_wine()

features = wine.data
labels = wine.target

# print("Number of entries: ", len(features))
# for featurename in wine.feature_names:
#     print(featurename[:10], "\t\t", end=" ")
# print("Class")
# for feature, label in zip(features, labels):
#     for f in feature:
#         print(f, "\t\t", end=" ")
#     print(label)

train_feats, test_feats, train_labels, test_lables = tts(features, labels, test_size=0.2)

clf = svm.SVC()

# train
clf.fit(train_feats, train_labels)

#predictions
predictions = clf.predict(test_feats)
# print(predictions)

score = 0
for i in range(len(predictions)):
    if predictions[i] == test_lables[i]:
        score += 1
print("SVM algorithm prediction accuracy: ", score/len(predictions))

# Tree Algorithm
from sklearn import tree
clf = tree.DecisionTreeClassifier()

# train
clf.fit(train_feats, train_labels)

#predictions
predictions = clf.predict(test_feats)
# print(predictions)

score = 0
for i in range(len(predictions)):
    if predictions[i] == test_lables[i]:
        score += 1
print("Tree algorithm prediction accuracy: ", score/len(predictions))

# Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# train
clf.fit(train_feats, train_labels)

#predictions
predictions = clf.predict(test_feats)
# print(predictions)

score = 0
for i in range(len(predictions)):
    if predictions[i] == test_lables[i]:
        score += 1
print("Random Forest algorithm prediction accuracy: ", score/len(predictions))


