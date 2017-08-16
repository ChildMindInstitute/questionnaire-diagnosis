import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
from itertools import groupby

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

    training_set = df[df['subset'] == 'train']
    test_set = df[df['subset'] == 'test']

    voting_clf.fit(training_set[features], factorized[0][:-2])
    rnd_clf.fit(training_set[features], factorized[0][:-2])

    print(voting_clf.predict(test_set[features]))
    print(rnd_clf.predict(test_set[features]))

    # Each model has a accuracy of 1.0 -- caused by data overfitting, will have to change simulated dataset
    for clf in (log_clf, rnd_clf, voting_clf):
        clf.fit(training_set[features], factorized[0][:-2])
        y_pred = clf.predict(test_set[features])
        print(clf.__class__.__name__, accuracy_score(y_pred, factorized[0][-2:]))

    # Print prediction probabilities for each Dx categorization for each "test" item
    print(rnd_clf.predict_proba(test_set[features]))

    # Print feature importance
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

    # Retain columns that are relevant for analysis, reduce pd df size
    col_keep = []
    for num in range(1, 8):
        col_keep.append('DX_0' + str(num) + '_Cat')
        col_keep.append('DX_0' + str(num) + '_Sub')
        col_keep.append('DX_0' + str(num) + '_Code')
    dx = dx[col_keep]

    # Create a dictionary containing codes and associated Dx
    code_dict = {}
    EID_Dx_dict = {}

    # for col in range(dx.shape[1]):
    for row in range(dx.shape[0]):
        Dx_EID_list = []
        for col in range(dx.shape[1]):
            if col % 3 == 2 and not isinstance(dx.iloc[row, col], float):
                # For Dx with ICD 10 Code that begins with 'F'
                if 'F' in dx.iloc[row, col]:
                    Dx_EID_list.append('F' + dx.iloc[row, col].split('F', 1)[1][:2])
                    if ('F' + dx.iloc[row, col].split('F', 1)[1][:2]) not in code_dict.keys():
                        if dx.iloc[row, col-2] == 'Neurodevelopmental Disorders':
                            code_dict['F' + dx.iloc[row, col].split('F', 1)[1][:2]] = dx.iloc[row, col - 1]
                        else:
                            code_dict['F' + dx.iloc[row, col].split('F', 1)[1][:2]] = dx.iloc[
                                row, col - 2]

                # For Dx with ICD 10 Code that begins with 'Z'
                elif 'Z' in dx.iloc[row, col]:
                    Dx_EID_list.append('Z' + dx.iloc[row, col].split('Z', 1)[1][:2])
                    if ('Z' + dx.iloc[row, col].split('Z', 1)[1][:2]) not in code_dict.keys():
                        if dx.iloc[row, col-2] == 'Neurodevelopmental Disorders':
                            code_dict['Z' + dx.iloc[row, col].split('Z', 1)[1][:2]] = dx.iloc[row, col - 1]
                        else:
                            code_dict['Z' + dx.iloc[row, col].split('Z', 1)[1][:2]] = dx.iloc[
                                row, col - 2]

                # For Dx that does not have an ICD 10 Code

                elif 'No Diagnosis' in dx.iloc[row, col]:
                    Dx_EID_list.append(dx.index[row])
                    code_dict[dx.index[row]] = dx.iloc[row, col - 2]
                else:
                    Dx_EID_list.append(dx.iloc[row, col-2])

        EID_Dx_dict[dx.index.values[row]] = Dx_EID_list

    dict_duplicates_removed = {a: list(set(b)) for a, b in EID_Dx_dict.items()}

    # Remove patients from DSM Data dataframe if they were removed from the Dx dataframe

    df['Dx'] = np.zeros(len(df))
    df = df.astype('object')
    df_Dx_match = df

    No_EID_drop_list = []

    for row in range(df.shape[0]):
        if df.index[row] in dx.index.values:
            print(str(df.index[row]))
            df_Dx_match.loc[str(df.index[row]), 'Dx'] = list(EID_Dx_dict[str(df.index[row])])
        else:
            No_EID_drop_list.append(int(row))

    df = df_Dx_match.drop(df_Dx_match.index[No_EID_drop_list])

    # Remove questions / patients with missing data, replace other NaN with mode

    mode_list = []

    for column in df:
        if df[column].isnull().sum() > round(.10 * df.shape[0], 0):
            df = df.drop(column, axis=1)

    for column in df:
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

    # Create MultiLabel Classifier and segment dataset for training/testing

    training_set = df[df['train'] == True]
    testing_set = df[df['train'] == False]

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                # df_copy.iloc[row, col] = mean_list[col]
                df_copy.iloc[row, col] = mode_list[col]

    features = df.columns[3:-2]

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                df_copy.iloc[row, col] = mode_list[col]

    writer = pd.ExcelWriter('DSM_NaN_Replaced.xlsx')
    df_copy.to_excel(writer, 'DSM_Data')
    writer.save()

if __name__ == "__main__":
    DSM()