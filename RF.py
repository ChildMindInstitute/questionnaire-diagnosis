# This sript is intended to implement random forest to the HBN dataset

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

def learn():

    # Manually created simulated data for personal learning -- should be automated for training/testing
    df = pd.DataFrame(data=np.array([[1,1,0,1,1,0,1],[0,0,1,0,0,1,0],[1,1,1,1,1,0,1],[1,0,0,1,1,0,0],
                                     [0,0,1,0,1,1,0],[0,1,0,1,1,0,1],[0,0,0,1,1,0,1],[1,0,1,0,0,1,0],
                                     [1,1,0,0,1,0,1],[0,0,1,0,0,0,0]]),
                      index=['EID1', 'EID2', 'EID3', 'EID4', 'EID5', 'EID6', 'EID7', 'EID8', 'Test1', 'Test2'],
                      columns=['Q1','Q2','Q3','Q4','Q5','Q6','Q7'])
    df['Dx'] = ['ASD', 'SM', 'ASD', 'ASD', 'SM', 'ASD', 'ASD', 'SM', 'ASD', 'SM']
    df['subset'] = ['train', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'test', 'test']

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

    print(train_set[features])

    voting_clf.fit(train_set[features], factorized[0][:-2])
    rnd_clf.fit(train_set[features], factorized[0][:-2])

    print(voting_clf.predict(test_set[features]))
    print(rnd_clf.predict(test_set[features]))

    # Each model has a accuracy of 1.0 -- caused by data overfitting, will have to change simulated dataset
    # for clf in (log_clf, rnd_clf, voting_clf):
    #     clf.fit(train_set[features], factorized[0][:-2])
    #     y_pred = clf.predict(test_set[features])
    #     print(clf.__class__.__name__, accuracy_score(y_pred, factorized[0][-2:]))

    # Print prediction probabilities for each Dx categorization for each "test" item
    print(rnd_clf.predict_proba(test_set[features]))

    # Print feature importance
    # print(list(zip(train_set[features], rnd_clf.feature_importances_)))


def load(num):

    train = 'DSM_Train' + str(num) + 'replaced.xlsx'
    test = 'DSM_Test' + str(num) + 'replaced.xlsx'

    output = pd.ExcelFile('/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/' + train)
    df_tr = output.parse('Train')
    df_tr = pd.DataFrame(data=df_tr)
    df_tr = df_tr.set_index('EID')

    output = pd.ExcelFile('/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/' + test)
    df_te = output.parse('Test')
    df_te = pd.DataFrame(data=df_te)
    df_te = df_te.set_index('EID')

    return df_tr, df_te


def RF(df_tr, df_te):

    train_set = df_tr
    test_set = df_te

    print(train_set.head)
    print(test_set.head)

    train_targets = list(train_set['Dx'])
    test_targets = list(test_set['Dx'])

    X_train = train_set.columns[2:-2]
    X_test = (test_set[X_train])
    X_train = (train_set[X_train])

    for row in range(train_set.shape[0]):
        train_targets[row] = literal_eval(train_targets[row])

    for row in range(test_set.shape[0]):
        test_targets[row] = literal_eval(test_targets[row])

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(train_targets)
    # print(Y[0])

    model = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()),
                      ('clf', OneVsRestClassifier(LinearSVC()))])

    clf = RandomForestClassifier()

    clf.fit(X_train, Y)

    print(clf.predict(X_test)[:10])
    #
    # print((clf.predict_proba(X_test)))

    labels = mlb.inverse_transform(clf.predict(X_test))

    print(X_test)

    from pprint import pprint

    pprint(labels)

    print(len(labels))

    # for item, labels in zip(X_test, labels):
    #     print('{1}'.format(item, ', '.join(labels)))


def OneVsRF():

    filename = 'DSM_NaN_Replaced.xlsx'
    sheetname = 'DSM_Data'
    output = pd.ExcelFile(filename)
    df = output.parse(sheetname)
    df = pd.DataFrame(data=df)
    df = df.set_index('EID')

    train_set = df[df['train'] == True][0:100]
    test_set = df[df['train'] == False][0:50]
    train_set['Anx'] = np.zeros(len(train_set)).astype('object')
    train_set['ADHD'] = np.zeros(len(train_set)).astype('object')
    test_set['Anx'] = np.zeros(len(test_set)).astype('object')
    test_set['ADHD'] = np.zeros(len(test_set)).astype('object')
    copy = train_set
    copy2 = test_set

    for row in range(train_set.shape[0]):
        if 'Anxiety' in str(train_set.iloc[row, 607]):
            copy.iloc[row, 609] = 'Anxiety'
            copy.iloc[row, 610] = 'NA'
        elif 'Attention' in str(train_set.iloc[row, 607]):
            copy.iloc[row, 610] = 'ADHD'
            copy.iloc[row, 609] = 'NA'
        else:
            copy.iloc[row, 609] = 'NA'
            copy.iloc[row, 610] = 'NA'

    for row in range(test_set.shape[0]):
        if 'Anxiety' in str(train_set.iloc[row, 607]):
            copy2.iloc[row, 609] = 'Anxiety'
            copy2.iloc[row, 610] = 'NA'
        elif 'Attention' in str(test_set.iloc[row, 607]):
            copy2.iloc[row, 610] = 'ADHD'
            copy2.iloc[row, 609] = 'NA'
        else:
            copy2.iloc[row, 609] = 'NA'
            copy2.iloc[row, 610] = 'NA'

    writer = pd.ExcelWriter('Train_Set.xlsx')
    writer2 = pd.ExcelWriter('Test_Set.xlsx')
    copy.to_excel(writer, 'DSM_Data')
    copy2.to_excel(writer2, 'DSM_Data')
    writer.save()
    writer2.save()

    clf2 = OneVsRestClassifier(LinearSVC())
    clf3 = OneVsRestClassifier(LinearSVC())

    Y = np.array(copy['Anx'])
    Targ = np.array(copy2['Anx'])
    Y2 = np.array(copy['ADHD'])
    Targ2 = np.array(copy2['ADHD'])

    X_train = train_set.columns[2:-4]
    X_test = (test_set[X_train])
    X_train = (train_set[X_train])

    clf2.fit(X_train, Y)
    clf3.fit(X_train, Y2)

    count = 0
    count2 = 0

    preds = clf2.predict(X_test)
    preds2 = clf3.predict(X_test)

    for item in range(len(preds)):
        if preds[item] == Targ[item]:
            count += 1

    print(preds)
    print(f1_score(Targ, preds, average='weighted'))
    print(count)

    for item in range(len(preds2)):
        if preds2[item] == Targ2[item]:
            count2 += 1

    print(preds2)
    print(f1_score(Targ2, preds2, average='weighted'))
    print(count2)

if __name__ == '__main__':

    [df_tr, df_te] = load(3)
    # RF(df_tr, df_te)
