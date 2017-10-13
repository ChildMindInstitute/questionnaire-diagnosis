# -*- coding: utf-8 -*-

# Copyright (c) 2015-2017 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import os
import pandas as pd
import pytest
import struct
import sys
import time

from bayeslite import bayesdb_open
from bayeslite import bayesdb_read_csv
from iventure.utils_bql import query

from testing_utils import mkdir


BQL_STR = '''
CREATE TABLE "temp" AS
  SELECT entropy, mi FROM
    (ESTIMATE MUTUAL INFORMATION OF
      ({columns}) WITH ({columns})
      USING {n_samples} SAMPLES AS entropy BY pop),
    (ESTIMATE MUTUAL INFORMATION OF
      ({columns}) WITH "{diagnosis}"
      USING {n_samples} SAMPLES AS mi BY pop)
'''

DESIRED_NUMBER_OF_QUESTIONS = 1

NUMBER_MC_SAMPLES = [1]
MINUTES_ANALYSIS = [1]
DIAGNOSIS = {
    'autism': "Autism Spectrum Disorder",
    'adhd': "Attention-Deficit/Hyperactivity Disorder"
}


@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
@pytest.mark.parametrize('minutes_analysis', MINUTES_ANALYSIS)
@pytest.mark.parametrize('diagnosis', DIAGNOSIS.keys())
@pytest.mark.parametrize('seed', range(1, 2))
def test_questinnaire_short(number_mc_samples, minutes_analysis, diagnosis, seed):
    experimetal_config = {
        'number_mc_samples' : str(number_mc_samples),
        'minutes_analysis' : str(minutes_analysis),
        'diagnosis': diagnosis,
        'seed' : seed,
    }

    mkdir('tests/bdb/')
    file_name = 'tests/bdb/test_questinnaire_{diagnosis}_short_mc={number_mc_samples}' +\
        '_minutes={minutes_analysis}_seed={seed}.bdb'
    bdb_file_name = file_name.format(**experimetal_config)
    # XXX Great. I neither haven an idea why on earth one would make setting the
    # seed so complicated, neither do I fully understand what struct.pack is
    # actually doing.
    byte_str_seed = struct.pack('<QQQQ', seed, seed, seed, seed)

    if os.path.exists(bdb_file_name):
        os.remove(bdb_file_name)
    bdb = bayesdb_open(bdb_file_name, seed=byte_str_seed)
    bdb.metamodels['cgpm'].set_multiprocess(False)

    bdb.execute('''
        CREATE TABLE data_table
            FROM 'resources/data/init_data.csv';
    ''')
    bdb.execute('''
        CREATE POPULATION pop FOR data_table WITH SCHEMA (
            GUESS STATTYPES FOR (*);
            MODEL
                 "Neurodevelopmental Disorders",
                 "Substance Related and Addictive Disorders",
                 "Adjustment Disorder",
                 "Feeding and Eating Disorders",
                 "SCQ_30",
                 "SCQ_01",
                 "Schizophrenia Spectrum and other Psychotic Disorders",
                 "Neurodevelopmental Disorder",
                 "Trauma and Stressor Related Disorders",
                 "Tic Disorder",
                 "Elimination Disorders",
                 "SCQ_28",
                 "Other Conditions That May Be a Focus of Clinical Attention",
                 "Bipolar and Related Disorders",
                 "Obsessive Compulsive and Related Disorders",
                 "Motor Disorder",
                 "Somatic Symptom and Related Disorders",
                 "Intellectual Disability",
                 "Sleep-Wake Disorders"
            AS
                NOMINAL;
            MODEL
                 "Age"
            AS
                NUMERICAL;
            IGNORE
                 "EID";
        );
    ''')
    bdb.execute('CREATE ANALYSIS SCHEMA cc FOR pop WITH BASELINE crosscat;')
    bdb.execute('INITIALIZE 1 MODELS FOR cc;')
    bdb.execute('''
        ANALYZE cc FOR {minutes_analysis} MINUTES WAIT(OPTIMIZED);
    '''.format(**experimetal_config))

    # Get the names of all candidate variables for the shortened questionnaire.
    df = query(bdb, 'SELECT * FROM data_table LIMIT 1')
    candidate_questions = df.columns.tolist()
    # Remove the diagnosis from the candidates.
    candidate_questions.remove(DIAGNOSIS[diagnosis])
    # Remove the subject ID from the candidates.
    candidate_questions.remove("EID")

    selected_questions = [] # Initialize: no questions selected yet.

    # While we don't have the desired number of quesions, keep searching.
    bdb.sql_execute('DROP TABLE IF EXISTS "cond_entropy"')
    bdb.sql_execute('CREATE TABLE cond_entropy(entropy REAL, mi REAL, columns TEXT)')

    while len(selected_questions) < DESIRED_NUMBER_OF_QUESTIONS:
        current_scores = []
        # Loop through all the candidate questions.

        if selected_questions:
            selected_columns = ','.join(selected_questions)
        else:
            selected_columns = 'None'

        for next_question in candidate_questions:
            columns = ','.join([
                    '"%s"' % question for question in
                    [next_question] + selected_questions
                ])

            current_bql_str = BQL_STR.format(
                columns=columns,
                n_samples=number_mc_samples,
                diagnosis= DIAGNOSIS[diagnosis]
            )
            bdb.execute('DROP TABLE IF EXISTS "temp"')
            bdb.execute(current_bql_str)
            bdb.sql_execute('INSERT INTO cond_entropy (entropy, mi, columns)' +
                '''SELECT entropy, mi, '{selected_columns}' FROM
                "temp"'''.format(selected_columns=selected_columns)
            )

        df = query(bdb, '''
            SELECT entropy - mi AS conditional_entropy FROM cond_entropy
                WHERE columns = '{selected_columns}'
        '''.format(selected_columns=selected_columns))

        new_question = candidate_questions[np.argmin(df['conditional_entropy'].values)]

        selected_questions.append(new_question)
        candidate_questions.remove(new_question)
    mkdir('tests/output/selected/childmind')
    output_file_name = 'tests/output/selected/childmind/questions_{diagnosis}_mc={number_mc_samples}' +\
        '_minutes={minutes_analysis}_seed={seed}.csv'
    pd.DataFrame({'selected':selected_questions}).to_csv(
        output_file_name.format(**experimetal_config),
        index=False
    )

