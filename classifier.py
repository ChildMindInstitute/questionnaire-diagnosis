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

    print(df)
    print(df.shape)

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
    for num in range(1, 10):
        col_keep.append('DX_0' + str(num) + '_Cat')
        col_keep.append('DX_0' + str(num) + '_Sub')
        col_keep.append('DX_0' + str(num))
        col_keep.append('DX_0' + str(num) + '_Code')
    dx = dx[col_keep]

    return dx

def threshold(dx):

    # Cat, Sub, Dx contain value counts for each column with the respective label

    print(dx.shape)

    Cat = dx['DX_01_Cat'].value_counts()
    Sub = dx['DX_01_Sub'].value_counts()
    Dx = dx['DX_01'].value_counts()

    for num in range(2, 10):

        iter_cat = 'DX_0' + str(num) + '_Cat'
        temp_Cat = dx[iter_cat].value_counts()
        iter_sub = 'DX_0' + str(num) + '_Sub'
        temp_Sub = dx[iter_sub].value_counts()
        iter_dx = 'DX_0' + str(num)
        temp_dx = dx[iter_dx].value_counts()

        Cat = Cat.add(temp_Cat, fill_value=0)
        Sub = Sub.add(temp_Sub, fill_value=0)
        Dx = Dx.add(temp_dx, fill_value=0)

    print(Cat)
    print(Sub)
    print(Dx)

    return Cat, Sub, Dx


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
                # DX_Sub = dx.iloc[row, col - 2]
                # DX_Cat = dx.iloc[row, col - 3]

                if dx.iloc[row, col - 3] == 'Neurodevelopmental Disorders':
                    Dx_EID_list.append(dx.iloc[row, col - 1])
                else:
                    Dx_EID_list.append(dx.iloc[row, col - 1])

        EID_Dx_dict[dx.index.values[row]] = set(Dx_EID_list)

    dict_duplicates_removed = {a: list(set(b)) for a, b in EID_Dx_dict.items()}

    # Remove patients from DSM Data dataframe if they were removed from the Dx dataframe

    df['Dx'] = np.zeros(len(df))
    df = df.astype('object')
    df_Dx_match = df

    No_EID_drop_list = []

    for row in range(df.shape[0]):
        if df.index[row] in dx.index.values:
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
                     'CIS', 'SRS', 'TRF', 'WHODAS', 'Total', '_GD', '_SH', '_T', '_LP',
                     '_SC', '_HY', '_PN', '_SP', '_IN', '_PP', '_OPD', '_PM', '_ID', '_INV',
                     'EEG', '_Raw', '_Scaled', '_Sum', '_Percentile', 'VCI', 'FRI', '_WMI', '_CP',
                     '_AG', '_FR', '_LP', '_AD', '_WD', '_SC', '_SP', '_TP', '_RBB', '_AB', '_OP', '_Ext', '_Int',
                     '_C', 'CELF', 'CGAS', 'CV_', 'CTOPP', 'TOWRE', 'KBIT', 'NIH5', 'NLES', 'percentile',
                     '_scaled', '_raw', '_desc', '_composite', '_absorption', '_regulation', '_tolerance',
                     '_Flanker', '_List', '_Pattern', '_Picture', 'appraisal', 'eeg', 'attempt', '_size', '_inion',
                     '_ear', '_break', 'WISC', 'WIAT', '_DC', '_PD']:
            if item in col:
                drop_list.append(col)

    df = df_mid.drop(drop_list, axis=1)

    return df


def feature_trim(df, param):

    for column in df:
        if df[column].isnull().sum() > round(float(param) * df.shape[0], 0):
            df = df.drop(column, axis=1)

    if param == 1.0:
        param = '1'
    elif param == .20:
        param = '2'

    return df, param


def replace_missing(df):

    # Replace missing values with the mode of each feature

    mode_list = []
    means_list = []

    print('step 1')

    for column in df:

        try:
            freqs = groupby(Counter(column).most_common(), lambda x: x[1])
            modes = list(freqs)[0][0]
            mode_list.append(modes)
            means = df[column].mean()
            means_list.append(means)

        except:
            continue

    df_copy = df

    df = df.replace(np.NaN, 'NA')

    print('step 2')

    num_replaced = 0

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iloc[row, col] == 'NA':
                try:
                    df_copy.iloc[row, col] = means_list[col]
                    num_replaced += 1
                except:
                    continue

    print('Number replaced: ' + str(num_replaced))

    return df_copy


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


def template(df):

    df['Dx_of_Interest'] = np.zeros(len(df)).astype('object')
    copy = df

    # for row in range(df.shape[0]):
    #
    #     if 'ADHD-Combined' in str(df.iloc[row, df.shape[1]-2]):
    #         copy.iloc[row, copy.shape[1]-1] = 0
    #     elif 'ADHD-Inattentive' in str(df.iloc[row, df.shape[1]-2]):
    #         copy.iloc[row, copy.shape[1]-1] = 1
    #     elif 'ADHD-Hyperactive' in str(df.iloc[row, df.shape[1]-2]):
    #         copy.iloc[row, copy.shape[1]-1] = 2
    #     else:
    #         copy.iloc[row, copy.shape[1]-1] = 'Nan'

    for row in range(df.shape[0]):

        if 'Impairment in Mathematics' in str(df.iloc[row, df.shape[1]-2]):
            copy.iloc[row, copy.shape[1]-1] = 0
        elif 'Impairment in Reading' in str(df.iloc[row, df.shape[1]-2]):
            copy.iloc[row, copy.shape[1]-1] = 1
        else:
            copy.iloc[row, copy.shape[1]-1] = 'Nan'

    for row in range(df.shape[0]):

        if 'Autism' in str(df.iloc[row, df.shape[1]-2]):
            copy.iloc[row, copy.shape[1]-1] = 1
        else:
            copy.iloc[row, copy.shape[1]-1] = 0

    return copy

def train_test(df):

    np.random.seed(seed=0)
    df['train'] = np.random.uniform(0, 1, len(df)) <= .90
    train = df[df['train'] == True]
    test = df[df['train'] == False]

    print(df.shape)

    return train, test


def export(train, test, param, replace=False):

    if replace == True:
        train = replace_missing(train)
        test = replace_missing(test)
        param = param + 'replaced'

    path = '/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/training' + param + '.xlsx'

    writer = pd.ExcelWriter(path)
    train.to_excel(writer, 'Train')
    writer.save()

    path = '/Users/jake.son/PycharmProjects/Dx_mvpa/Train_Test_Sets/testing' + param + '.xlsx'

    writer = pd.ExcelWriter(path)
    test.to_excel(writer, 'Test')
    writer.save()

    print(train.shape)

if __name__ == "__main__":
    df = get_responses()
    dx = get_Dx()
    [Cat, Sub, Dx] = threshold(dx)
    df = pair_diagnoses(df, dx)
    df = remove_age_q(df)
    [df, param] = feature_trim(df, 1.0)
    copy = template(df)
    # anx_full = anx(df)
    # adhd_full = adhd(anx_full)
    # final_df = asd(adhd_full)
    [train, test] = train_test(copy)
    export(train, test, param, True)
