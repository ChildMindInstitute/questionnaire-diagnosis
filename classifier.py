# This script is intended to process data for learning algorithms

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
from itertools import groupby

def get_responses():
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
    return df

def get_Dx():

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
    return dx

def pair_diagnoses(df, dx):

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
                    Dx_EID_list.append(dx.iloc[row, col - 2])
                    if ('F' + dx.iloc[row, col].split('F', 1)[1][:2]) not in code_dict.keys():
                        if dx.iloc[row, col - 2] == 'Neurodevelopmental Disorders':
                            code_dict['F' + dx.iloc[row, col].split('F', 1)[1][:2]] = dx.iloc[row, col - 1]
                        else:
                            code_dict['F' + dx.iloc[row, col].split('F', 1)[1][:2]] = dx.iloc[
                                row, col - 2]

                # For Dx with ICD 10 Code that begins with 'Z'
                elif 'Z' in dx.iloc[row, col]:
                    Dx_EID_list.append(dx.iloc[row, col - 2])
                    if ('Z' + dx.iloc[row, col].split('Z', 1)[1][:2]) not in code_dict.keys():
                        if dx.iloc[row, col - 2] == 'Neurodevelopmental Disorders':
                            code_dict['Z' + dx.iloc[row, col].split('Z', 1)[1][:2]] = dx.iloc[row, col - 1]
                        else:
                            code_dict['Z' + dx.iloc[row, col].split('Z', 1)[1][:2]] = dx.iloc[
                                row, col - 2]

                # For Dx with ICD 10 Code that begins with 'G'
                elif 'G' in dx.iloc[row, col]:
                    Dx_EID_list.append(dx.iloc[row, col - 2])
                    if ('G' + dx.iloc[row, col].split('G', 1)[1][:2]) not in code_dict.keys():
                        if dx.iloc[row, col - 2] == 'Neurodevelopmental Disorders':
                            code_dict['G' + dx.iloc[row, col].split('G', 1)[1][:2]] = dx.iloc[row, col - 1]
                        else:
                            code_dict['G' + dx.iloc[row, col].split('G', 1)[1][:2]] = dx.iloc[
                                row, col - 2]

                # For Dx that does not have an ICD 10 Code

                elif 'No Diagnosis' in dx.iloc[row, col]:
                    Dx_EID_list.append(dx.iloc[row, col - 2])
                    code_dict[dx.index[row]] = dx.iloc[row, col - 2]
                else:
                    Dx_EID_list.append(dx.iloc[row, col - 2])

        EID_Dx_dict[dx.index.values[row]] = set(Dx_EID_list)

    print(code_dict)
    print(set(code_dict.values()))

    dict_duplicates_removed = {a: list(set(b)) for a, b in EID_Dx_dict.items()}

    # Remove patients from DSM Data dataframe if they were removed from the Dx dataframe

    df['Dx'] = np.zeros(len(df))
    df = df.astype('object')
    df_Dx_match = df

    No_EID_drop_list = []

    for row in range(df.shape[0]):
        if df.index[row] in dx.index.values:
            # print(str(df.index[row]))
            df_Dx_match.loc[str(df.index[row]), 'Dx'] = list(EID_Dx_dict[str(df.index[row])])
        else:
            No_EID_drop_list.append(int(row))

    print(No_EID_drop_list)
    df = df_Dx_match.drop(df_Dx_match.index[No_EID_drop_list])

    return df

def feature_trim(df):

    # Remove questions / patients with missing data, replace other NaN with mode

    mode_list = []

    for column in df:
        if df[column].isnull().sum() > round(.30 * df.shape[0], 0):
            df = df.drop(column, axis=1)

    for column in df:
        freqs = groupby(Counter(column).most_common(), lambda x: x[1])
        modes = list(freqs)[0][0]
        mode_list.append(modes)

    NaN_droplist = []

    # for num in range(df.shape[0]):
    #     if df.isnull().sum(axis=1)[num] > 50:
    #         NaN_droplist.append(int(num))

    df = df.drop(df.index[NaN_droplist])

    # print(df.isnull().sum(axis=1)) # Display the number of NaN values for each EID

    df = df.replace(np.NaN, 'NA')

    np.random.seed(seed=0)
    df['train'] = np.random.uniform(0, 1, len(df)) <= .30
    df_copy = df

    # Create MultiLabel Classifier and segment dataset for training/testing

    training_set = df[df['train'] == True]
    testing_set = df[df['train'] == False]

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                # df_copy.iloc[row, col] = mean_list[col]
                df_copy.iloc[row, col] = mode_list[col]

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                df_copy.iloc[row, col] = mode_list[col]

    writer = pd.ExcelWriter('DSM_NaN_Replaced.xlsx')
    df_copy.to_excel(writer, 'DSM_Data')
    writer.save()

if __name__ == "__main__":
    df = get_responses()
    dx = get_Dx()
    df = pair_diagnoses(df, dx)
    feature_trim(df)