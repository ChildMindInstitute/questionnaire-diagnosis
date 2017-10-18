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
import matplotlib.pyplot as plt

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

    train = '/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/DSM_Train' + str(num) + 'replaced.xlsx'
    test = '/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/DSM_Test' + str(num) + 'replaced.xlsx'

    output = pd.ExcelFile(train)
    df_tr = output.parse('Train')
    df_tr = pd.DataFrame(data=df_tr)
    df_tr = df_tr.set_index('EID')

    output = pd.ExcelFile(test)
    df_te = output.parse('Test')
    df_te = pd.DataFrame(data=df_te)
    df_te = df_te.set_index('EID')

    return df_tr, df_te


def get_stats(predictions, targets):

    correct = 0
    false_p = 0
    false_n = 0

    for num in range(len(predictions)):
        if predictions[num] == targets[num]:
            correct += 1
        elif predictions[num] == 1 and targets[num] == 0:
            false_p += 1
        elif predictions[num] == 0 and targets[num] == 1:
            false_n += 1

    correct /= len(predictions)
    false_p /= len(predictions)
    false_n /= len(predictions)

    return correct, false_p, false_n


def RF(df_tr, df_te, Dx, iters):

    train_targets = list(df_tr[Dx])
    test_targets = list(df_te[Dx])
    train_feat = df_tr[df_tr.columns[2:-5]]
    test_feat = df_te[df_te.columns[2:-5]]

    f1_list = []
    acc_list = []
    correct_list = []
    false_n_list = []
    false_p_list = []
    feature_dict = {}
    iteration = 0

    for num in range(iters):

        clf = RandomForestClassifier(n_estimators=100)#(random_state=0)

        clf.fit(train_feat, train_targets)

        predictions = clf.predict(test_feat)

        # print(f1_score(test_targets, predictions, average='weighted'), accuracy_score(test_targets, predictions))

        [correct, false_p, false_n] = get_stats(list(predictions), test_targets)
        print([correct, false_p, false_n])

        correct_list.append(correct)
        false_p_list.append(false_p)
        false_n_list.append(false_n)

        f1_list.append(f1_score(test_targets, predictions, average='weighted'))
        acc_list.append(accuracy_score(test_targets, predictions))

        features = list(zip(train_feat, clf.feature_importances_))

        threshold_features = []
        for question, importance in features:
            # If the feature importance is greater than 5 times the importance if all features were equally important
            if float(importance) > float(5/train_feat.shape[1]):
                threshold_features.append((question, importance))

        print(threshold_features)

        if len(threshold_features) > 0:

            (questions, importances) = zip(*threshold_features)
            questions = list(questions)
            importances = np.array(importances)
            indices = np.argsort(importances)[::-1]

            for num in range(len(questions)):
                if str(questions[num]) in feature_dict.keys():
                    feature_dict[str(questions[num])] = int(feature_dict[str(questions[num])]) + 1
                else:
                    feature_dict[questions[num]] = 1

        iteration += 1
        print('Iteration # ' + str(iteration))

        # Plot only important features

        if iters == 1:
            plt.figure(figsize=(5, 5))
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), questions, rotation=15)
            plt.xlim([-1, len(importances)])
            plt.axhline(y=float(5/train_feat.shape[1]), color='r', linestyle='--')
            plt.show()

    top_feat = sorted(feature_dict.items(), key=lambda attrib: attrib[1], reverse=True)

    print(top_feat)
    # print(np.array(f1_list).mean())
    # print(np.array(acc_list).mean())
    print(np.array(correct_list).mean())
    print(np.array(false_p_list).mean())
    print(np.array(false_n_list).mean())


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
    iters = 1000
    RF(df_tr, df_te, 'adhd', iters)
