import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
from itertools import groupby

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

    rnd_clf = RandomForestClassifier(n_estimators=10, random_state=0)
    log_clf = LogisticRegression()
    svm_clf = SVC()

    voting_clf = VotingClassifier(estimators=[('lr', log_clf),('rf',rnd_clf),('svc',svm_clf)], voting='hard')

    training_set = df[df['subset'] == 'train']

    voting_clf.fit(training_set[features], factorized[0][:-2])
    rnd_clf.fit(training_set[features], factorized[0][:-2])

    test_set = df[df['subset'] == 'test']

    print(voting_clf.predict(test_set[features]))
    print(rnd_clf.predict(test_set[features]))

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(training_set[features], factorized[0][:-2])
        y_pred = clf.predict(test_set[features])
        print(clf.__class__.__name__, accuracy_score(y_pred, factorized[0][-2:]))

    print(rnd_clf.predict_proba(test_set[features]))
    print(list(zip(training_set[features], rnd_clf.feature_importances_)))

def DSM():

    # Load Patient DSM Data
    filename = 'DSM_Data.xlsx'
    sheetname = 'DSM_Data.csv'
    output = pd.ExcelFile(filename)
    df = output.parse(sheetname)
    df = pd.DataFrame(data=df)
    df = df.set_index('EID')

    df = df.drop(['START_DATE.y', 'Study.y', 'Site.y', 'Year.y', 'Days_Baseline.y',
                  'Season.y'], axis=1)

    df = df.replace('NA', np.NaN)

    # Load Patient Dx
    filename = 'ConsensusDx.xlsx'
    sheetname = 'ConsensusDx'
    output = pd.ExcelFile(filename)
    dx = output.parse(sheetname)
    dx = pd.DataFrame(data=dx)
    dx = dx.set_index('EID')

    col_keep = []

    for num in range(1, 8):
        col_keep.append('DX_0' + str(num))
        col_keep.append('DX_0' + str(num) + '_Code')

    dx = dx[col_keep]

    # Remove EIDs that have no Dx (would just add noise, "No Diagnosis" classification is unhelpful)

    EID_drop_list = []

    for row in range(dx.shape[0]):
        if dx.iloc[row, 0] == 'No Diagnosis Given' or dx.iloc[row, 1] == 'No Diagnosis Given':
            EID_drop_list.append(int(row))

    dx = dx.drop(dx.index[EID_drop_list])

    # Create a dictionary containing codes and associated Dx
    code_dict = {}

    for col in range(dx.shape[1]):
        if col % 2 == 1:
            for row in range(dx.shape[0]):
                if dx.iloc[row, col] not in code_dict.keys():
                    code_dict[dx.iloc[row, col]] = dx.iloc[row, col-1]

    # Create a dictionary where key = EID and values = diagnosis / diagnoses

    EID_Dx_dict = {}

    for row in range(dx.shape[0]):
        Dx_EID_list = []
        for col in range(dx.shape[1]):
            if col % 2 == 1 and not isinstance(dx.iloc[row, col], float):
                if 'F' in dx.iloc[row, col]:
                    Dx_EID_list.append('F' + dx.iloc[row, col].split('F', 1)[1][:2])
                elif 'Z' in dx.iloc[row, col]:
                    Dx_EID_list.append('Z' + dx.iloc[row, col].split('Z', 1)[1][:2])
                else:
                    print('what')
        EID_Dx_dict[dx.index.values[row]] = Dx_EID_list

    print(EID_Dx_dict)

    # Remove patients from DSM Data dataframe if they were removed from the Dx dataframe

    # df['Dx'] = np.zeros(len(df))
    # df_Dx_match = df
    #
    # for num in range(dx.shape[0]):
    #     if df.index[num] in dx.index.values:
    #         df_Dx_match['Dx'][num] =
    #     else:
    #         print('remove EID from df')
    #
    #
    # df = df_Dx_match

    # Assign the set of Dx to each EID, transferring information from Dx to DSM dataframe

    # Remove questions / patients with missing data, replace other NaN with mode

    # mean_list = []
    mode_list = []

    for column in df:
        if df[column].isnull().sum() > round(.10 * df.shape[0], 0):
            df = df.drop(column, axis=1)

    for column in df:
        # mean_list.append(df[column].mean())
        freqs = groupby(Counter(column).most_common(), lambda x: x[1])
        modes = list(freqs)[0][0]
        mode_list.append(modes)

    NaN_droplist = []

    for num in range(df.shape[0]):
        if df.isnull().sum(axis=1)[num] > 50:
            NaN_droplist.append(int(num))

    df = df.drop(df.index[NaN_droplist])

    # print(df.isnull().sum(axis=1)) # Display the number of NaN values for each EID

    df = df.replace(np.NaN, 'NA')

    np.random.seed(seed=0)
    df['train'] = np.random.uniform(0, 1, len(df)) <= .20
    df_copy = df

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                # df_copy.iloc[row, col] = mean_list[col]
                df_copy.iloc[row, col] = mode_list[col]

    writer = pd.ExcelWriter('DSM_NaN_Replaced.xlsx')
    df_copy.to_excel(writer, 'DSM_Data')
    writer.save()

if __name__ == "__main__":
    DSM()
