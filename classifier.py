# This script is intended to process data for learning algorithms

import pandas as pd
import numpy as np
from collections import Counter
from itertools import groupby, chain
import collections


def get_responses():

    # Load Patient DSM Data
    filename = 'test.xlsx'
    sheetname = 'Test'
    output = pd.ExcelFile(filename)
    df = output.parse(sheetname)
    df = pd.DataFrame(data=df)
    df = df.set_index('EID')
    # df = df.drop(['START_DATE.y', 'Study.y', 'Site.y', 'Year.y', 'Days_Baseline.y',
    #               'Season.y'], axis=1)
    df = df.replace('NA', np.NaN)
    df = split_age(df)

    return df


def get_Dx():

    # Load Patient Dx
    filename = 'ConsensusDx_2.xlsx'
    sheetname = 'ConsensusDx_2'
    output = pd.ExcelFile(filename)
    dx = output.parse(sheetname)
    dx = pd.DataFrame(data=dx)
    dx = dx.set_index('EID')

    # Retain columns that are relevant for analysis, reduce pd df size
    col_keep = []
    for num in range(1, 9):
        col_keep.append('DX_0' + str(num) + '_Cat')
        col_keep.append('DX_0' + str(num) + '_Sub')
        col_keep.append('DX_0' + str(num))
        col_keep.append('DX_0' + str(num) + '_Code')
    dx = dx[col_keep]

    return dx


def threshold(dx):

    print(dx.DX_01.value_counts())
    print(dx.DX_01_Sub.value_counts())
    print(dx.DX_01_Cat.value_counts())
    print(dx.DX_02_Cat.value_counts())
    print(dx['DX_01_Cat'].value_counts().add(dx.DX_02_Cat.value_counts(), fill_value=0))

    base = dx.DX_01_Cat.value_counts()

    for num in range(2, 9):
        mid = 'DX_0' + str(num) + '_Cat'
        base = base.add(dx[mid].value_counts(sort=True, ascending=False), fill_value=0)

    base = base.sort_values(ascending=False)
    print(base)

    return base


def pair_diagnoses(df, dx):

    # Create a dictionary containing codes and associated Dx
    code_dict = {}
    EID_Dx_dict = {}

    for row in range(dx.shape[0]):
        Dx_EID_list = []
        for col in range(dx.shape[1]):
            if col % 4 == 3 and not isinstance(dx.iloc[row, col], float):

                # ICD 10 Coding = dx.iloc[row, col]
                # DX = dx.iloc[row, col - 1]
                # DX_Sub = dx.iloc[row, col -2]
                # DX_Cat = dx.iloc[row, col - 3]

                # For Dx with ICD 10 Code that begins with 'F'
                if 'F' in dx.iloc[row, col]:
                    # Dx_EID_list.append(dx.iloc[row, col - 2])
                    if dx.iloc[row, col - 2] == 'Neurodevelopmental Disorders':
                        if str(dx.iloc[row, col - 1]) == 'nan':
                            Dx_EID_list.append(dx.iloc[row, col - 2])
                            code_dict['F' + dx.iloc[row, col].split('F', 1)[1][:2]] = dx.iloc[row, col - 1]
                        else:
                            Dx_EID_list.append(dx.iloc[row, col - 1])
                    else:
                        code_dict['F' + dx.iloc[row, col].split('F', 1)[1][:2]] = dx.iloc[
                            row, col - 2]
                        Dx_EID_list.append(dx.iloc[row, col - 2])

                # For Dx with ICD 10 Code that begins with 'Z'
                elif 'Z' in dx.iloc[row, col]:
                    # Dx_EID_list.append(dx.iloc[row, col - 2])
                    if dx.iloc[row, col - 2] == 'Neurodevelopmental Disorders':
                        Dx_EID_list.append(dx.iloc[row, col - 1])
                        code_dict['Z' + dx.iloc[row, col].split('Z', 1)[1][:2]] = dx.iloc[row, col - 1]
                    else:
                        code_dict['Z' + dx.iloc[row, col].split('Z', 1)[1][:2]] = dx.iloc[
                            row, col - 2]
                        Dx_EID_list.append(dx.iloc[row, col - 2])

                # For Dx with ICD 10 Code that begins with 'G'
                elif 'G' in dx.iloc[row, col]:
                    # Dx_EID_list.append(dx.iloc[row, col - 2])
                    if dx.iloc[row, col - 2] == 'Neurodevelopmental Disorders':
                        Dx_EID_list.append(dx.iloc[row, col - 1])
                        code_dict['G' + dx.iloc[row, col].split('G', 1)[1][:2]] = dx.iloc[row, col - 1]
                    else:
                        code_dict['G' + dx.iloc[row, col].split('G', 1)[1][:2]] = dx.iloc[
                            row, col - 2]
                        Dx_EID_list.append(dx.iloc[row, col - 2])

                # For Dx that does not have an ICD 10 Code

                elif 'No Diagnosis' in dx.iloc[row, col]:
                    Dx_EID_list.append(dx.iloc[row, col - 2])
                    code_dict[dx.index[row]] = dx.iloc[row, col - 2]
                else:
                    Dx_EID_list.append(dx.iloc[row, col - 2])

        EID_Dx_dict[dx.index.values[row]] = set(Dx_EID_list)

    dict_duplicates_removed = {a: list(set(b)) for a, b in EID_Dx_dict.items()}

    # Remove patients from DSM Data dataframe if they were removed from the Dx dataframe

    df['Dx'] = np.zeros(len(df))
    df = df.astype('object')
    df_Dx_match = df

    No_EID_drop_list = []

    for row in range(df.shape[0]):
        if df.index[row] in dx.index.values:
            print(df.index[row])
            print(list(EID_Dx_dict[(df.index[row])]))
            df_Dx_match.loc[str(df.index[row]), 'Dx'] = list(EID_Dx_dict[(df.index[row])])
        else:
            No_EID_drop_list.append(int(row))

    df = df_Dx_match.drop(df_Dx_match.index[No_EID_drop_list])

    return df


def split_age(df):

    df_mid = df[df['Age'] < 17.01]
    df_mid = df_mid[df_mid['Age'] >= 7.99]

    # Count frequency of each diagnosis

    # count_set = list(df_mid['Dx'])
    # freq = collections.defaultdict(int)
    # for x in chain.from_iterable(count_set):
    #     freq[x] += 1

    return df_mid


def remove_age_q(df_mid):

    drop_list = []

    for col in df_mid:
        for item in ['ACE', 'CDI', 'ASR', 'CAARS', 'STAI', 'YFAS', 'ICU', 'CBCL',
                     'CIS', 'SRS', 'TRF', 'WHODAS', 'Total', '_GD', '_SH',
                     '_SC', '_HY', '_PN', '_SP', '_IN']:
            if item in col:
                drop_list.append(col)

    df = df_mid.drop(drop_list, axis=1)

    return df


def feature_trim(df, param):

    for column in df:
        if df[column].isnull().sum() > round(float(param) * df.shape[0], 0):
            df = df.drop(column, axis=1)

    df = df.replace(np.NaN, 'NA')

    if param == 1.0:
        param = '1'
    elif param == .30:
        param = '2'
    elif param == .05:
        param = '3'

    return df, param


def replace_missing(df):

    # Replace missing values with the mode of each feature

    mode_list = []

    for column in df:
        freqs = groupby(Counter(column).most_common(), lambda x: x[1])
        modes = list(freqs)[0][0]
        mode_list.append(modes)

    df_copy = df

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                df_copy.iloc[row, col] = mode_list[col]

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                df_copy.iloc[row, col] = mode_list[col]

    return df


def anx(df):

    df['Anx'] = np.zeros(len(df)).astype('object')
    copy = df

    for row in range(df.shape[0]):
        if 'Anxiety' in str(df.iloc[row, copy.shape[1]-2]):
            copy.iloc[row, copy.shape[1]-1] = 1
        else:
            copy.iloc[row, copy.shape[1]-1] = 0

    anx_full = copy

    return anx_full


def adhd(df):

    df['adhd'] = np.zeros(len(df)).astype('object')
    copy = df

    for row in range(df.shape[0]):

        if 'Attention' in str(df.iloc[row, df.shape[1]-3]):
            copy.iloc[row, copy.shape[1]-1] = 1
        else:
            copy.iloc[row, copy.shape[1]-1] = 0

    adhd_full = copy

    return adhd_full


def asd(df):

    df['asd'] = np.zeros(len(df)).astype('object')
    copy = df

    for row in range(df.shape[0]):

        if 'Autism' in str(df.iloc[row, df.shape[1]-4]):
            copy.iloc[row, copy.shape[1]-1] = 1
        else:
            copy.iloc[row, copy.shape[1]-1] = 0

    final_df = copy

    return final_df


def train_test(df):

    np.random.seed(seed=0)
    df['train'] = np.random.uniform(0, 1, len(df)) <= .80
    train = df[df['train'] == True][0:round(df.shape[0]*2/5)]
    test = df[df['train'] == False][0:round(df.shape[0]/10)]

    print(df.shape)

    return train, test


def export(train, test, param, replace=False):

    if replace == True:
        train = replace_missing(train)
        test = replace_missing(test)
        param = param + 'replaced'

    path = '/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/DSM_Train' + param + '.xlsx'

    writer = pd.ExcelWriter(path)
    train.to_excel(writer, 'Train')
    writer.save()

    path = '/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/DSM_Test' + param + '.xlsx'

    writer = pd.ExcelWriter(path)
    test.to_excel(writer, 'Test')
    writer.save()

if __name__ == "__main__":
    df = get_responses()
    dx = get_Dx()
    base = threshold(dx)
    df = pair_diagnoses(df, dx)
    # df_mid = split_age(df)
    # df = remove_age_q(df_mid)
    # [df, param] = feature_trim(df, .05)
    # anx_full = anx(df)
    # adhd_full = adhd(anx_full)
    # final_df = asd(adhd_full)
    # [train, test] = train_test(final_df)
    # export(train, test, param, True)
