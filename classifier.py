########################################################################################################################

# Classification using KMeans
# from sklearn import datasets
# import matplotlib.pyplot as plt
#
# digits = datasets.load_digits()
#
# fig = plt.figure(figsize=(8,8))
# for i in range(64):
#     ax = fig.add_subplot(8,8,i+1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
#     ax.text(0,7,str(digits.target[i]))
#
# plt.show()

########################################################################################################################

# Basic Classification with optional dimensionality reduction
#
# from sklearn import cluster, datasets
# iris = datasets.load_iris()
# X_iris = iris.data
# y_iris = iris.target
#
# k_means = cluster.KMeans(n_clusters=3)
# k_means.fit(X_iris)
#
# print(k_means.labels_[::10])
#
# print(y_iris[::10])

########################################################################################################################

# Random Forest https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
#
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
#
# import pandas as pd
# import numpy as np
# from pprint import pprint
#
# iris = load_iris()
#
# # Create pandas df that contains data with columns as sepal length, width, etc.
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
#
# df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# print(df.head())
#
# df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
#
# train, test = df[df['is_train']==True], df[df['is_train']==False]
#
# # Create a list of the feature column's names
# features = df.columns[:4]
#
# # train['species'] contains the actual species names. Before we can use it,
# # we need to convert each species name into a digit. So, in this case there
# # are three species, which have been coded as 0, 1, or 2.
# y = pd.factorize(train['species'])[0]
# print(y)
#
# # Create a random forest classifier. By convention, clf means 'classifier'
# clf = RandomForestClassifier(n_jobs=2)
#
# # Train the classifier to take the training features and learn how they relate
# # to the training y (the species)
# clf.fit(train[features], y)
#
# # Apply the classifier we trained to the test data (which, remember, it has never seen before)
# clf.predict(test[features])
#
# # Create actual english names for the plants for each predicted plant class
# preds = iris.target_names[clf.predict(test[features])]
#
# # View the PREDICTED species for the first five observations
# print(preds[0:5])
#
# # View the ACTUAL species for the first five observations
# print(test['species'].head())
#
# # View a list of the features and their importance scores
# pprint(list(zip(train[features], clf.feature_importances_)))

########################################################################################################################

# Simulated Data for Supervised Learning - Classification

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

blank1 = []
blank2 =[]

def learn():

    df = pd.DataFrame(data=np.array([[1,1,0,1,1,0,1],[0,0,1,0,0,1,0],[1,1,1,1,1,0,1],[1,0,0,1,1,0,0],
                                     [0,0,1,0,1,1,0],[0,1,0,1,1,0,1],[0,0,0,1,1,0,1],[1,0,1,0,0,1,0],
                                     [1,1,0,0,1,0,1],[0,0,1,0,0,0,0]]),
                      index=['EID1', 'EID2', 'EID3', 'EID4', 'EID5', 'EID6', 'EID7', 'EID8', 'Test1', 'Test2'],
                      columns=['Q1','Q2','Q3','Q4','Q5','Q6','Q7'])
    df['Dx'] = ['ASD', 'SM', 'ASD', 'ASD', 'SM', 'ASD', 'ASD', 'SM', 'ASD', 'SM']
    df['subset'] = ['train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'test', 'test']

    # Create array with Dx converted to a digit
    # Access factorized array only with factorized[0]
    # Access codes (strings for factorized array) with factorized[1]
    factorized = pd.factorize(df['Dx'])
    # print(df)

    features = df.columns[:7]
    # print(features)

    clf = RandomForestClassifier(n_estimators=10)

    training_set = df[df['subset'] == 'train']

    clf.fit(training_set[features], factorized[0][:-2])

    test_set = df[df['subset'] == 'test']

    # print(clf.predict(test_set[features]))

    # print(clf.predict_proba(test_set[features]))
    # print(clf.predict_proba(test_set[features])[0][0])
    # print(clf.predict_proba(test_set[features])[1][0])

    blank1.append(clf.predict_proba(test_set[features])[0][0])
    blank2.append(clf.predict_proba(test_set[features])[1][0])


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# print(blank1)
# print(blank2)
#
# print(mean(blank1))
# print(mean(blank2))

if __name__ == "__main__":
    for l in range(10):
        for i in range(50):
            learn()
        print(mean(blank1))
        print(mean(blank2))





