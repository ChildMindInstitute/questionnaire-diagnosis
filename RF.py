# This sript is intended to implement random forest to the HBN dataset

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def RF():

    # Manually created simulated data for personal learning -- should be automated for training/testing
    df = pd.DataFrame(data=np.array([[1,1,0,1,1,0,1],[0,0,1,0,0,1,0],[1,1,1,1,1,0,1],[1,0,0,1,1,0,0],
                                     [0,0,1,0,1,1,0],[0,1,0,1,1,0,1],[0,0,0,1,1,0,1],[1,0,1,0,0,1,0],
                                     [1,1,0,0,1,0,1],[0,0,1,0,0,0,0]]),
                      index=['EID1', 'EID2', 'EID3', 'EID4', 'EID5', 'EID6', 'EID7', 'EID8', 'Test1', 'Test2'],
                      columns=['Q1','Q2','Q3','Q4','Q5','Q6','Q7'])
    df['Dx'] = ['ASD', 'SM', 'ASD', 'ASD', 'SM', 'ASD', 'ASD', 'SM', 'ASD', 'SM']
    df['subset'] = ['train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'test', 'test']

    filename = 'DSM_NaN_Replaced.xlsx'
    sheetname = 'DSM_Data'
    output = pd.ExcelFile(filename)
    df = output.parse(sheetname)
    df = pd.DataFrame(data=df)

    df = df.apply(lambda x: pd['Dx'].factorize(x)[0])
    print(df['Dx'])

    # Create array with Dx converted to a digit so that Dx is codified
    factorized = pd.factorize(df['Dx'])

    # Extract relevant features (questions in this case) from pd df
    features = df.columns[:7]

    # Initialize random forest classifier with various parameters
    rnd_clf = RandomForestClassifier(n_estimators=10, random_state=0)

    # Initialize logitistic regression
    log_clf = LogisticRegression()

    voting_clf = VotingClassifier(estimators=[('lr', log_clf),('rf',rnd_clf)], voting='hard')

    train_set = df[df['subset'] == 'train']
    test_set = df[df['subset'] == 'test']

    voting_clf.fit(train_set[features], factorized[0][:-2])
    rnd_clf.fit(train_set[features], factorized[0][:-2])

    print(voting_clf.predict(test_set[features]))
    print(rnd_clf.predict(test_set[features]))

    # Each model has a accuracy of 1.0 -- caused by data overfitting, will have to change simulated dataset
    for clf in (log_clf, rnd_clf, voting_clf):
        clf.fit(train_set[features], factorized[0][:-2])
        y_pred = clf.predict(test_set[features])
        print(clf.__class__.__name__, accuracy_score(y_pred, factorized[0][-2:]))

    # Print prediction probabilities for each Dx categorization for each "test" item
    print(rnd_clf.predict_proba(test_set[features]))

    # Print feature importance
    print(list(zip(train_set[features], rnd_clf.feature_importances_)))