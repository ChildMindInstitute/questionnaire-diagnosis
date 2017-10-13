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

from utils import mkdir

LEAK_P_SPONTANEOUS = 0.95

N_SYMPTOMS = 4

def generate_data_disease(seed, N):

    prng = RandomState(seed)
    def sample_binary(p):
        return prng.uniform(0 , 1) < p

    disease = [sample_binary(0.5) for _ in range(N)]
    leak_p_symptoms = np.linspace(0.01, 0.49, N_SYMPTOMS)
    data = {}
    for symptom in range(N_SYMPTOMS):
        data['c_' + str(symptom)] = [
            sample_binary(1 - leak_p_symptoms[symptom] * LEAK_P_SPONTANEOUS)
            if disease else
            sample_binary(1 - LEAK_P_SPONTANEOUS)
            for disease_value in disease
        ]
    data['target'] = disease
    pd.DataFrame(data).to_csv(
        'tests/data/noisy_or_n=%d_seed=%d.csv' % (N, seed),
        index=False
    )
    return data

def test_data_gen_smoke():
    generate_data_disease(1, 3)
    df = pd.read_csv('tests/data/noisy_or_n=3_seed=1.csv')
    assert len(df) == 3
    assert df.columns.shape[0] == 5
    assert df.columns[0] == 'c_0'
    assert df.columns[1] == 'c_1'




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

def shorten_conditional_entropy(bdb, experimental_config):

    candidate_questions = experimental_config['column_names']
    # Remove the diagnosis from the candidates.
    candidate_questions.remove('target')
    selected_questions = [] # Initialize: no questions selected yet.

    # While we don't have the desired number of quesions, keep searching.
    bdb.sql_execute('DROP TABLE IF EXISTS "cond_entropy"')
    bdb.sql_execute('CREATE TABLE cond_entropy(entropy REAL, mi REAL, columns TEXT)')

    while len(selected_questions) < N_SYMPTOMS-1:
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
                n_samples=experimental_config['number_mc_samples'],
                diagnosis= 'target'
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

    return selected_questions

def prep_bdb(experimental_config):
    """ This functions prepare a bdb object.

        It reads the csv file, creates the population and runs analysis. 
    """
    mkdir('tests/bdb/')
    file_name = 'tests/bdb/test_noisy_or_mc={number_mc_samples}_n={number_datapoints}' +\
        '_iters={number_iterations}_seed={seed}.bdb'
    bdb_file_name = file_name.format(**experimental_config)
    # XXX Great. I neither haven an idea why on earth one would make setting the
    # seed so complicated, neither do I fully understand what struct.pack is
    # actually doing.
    byte_str_seed = struct.pack(
        '<QQQQ',
        experimental_config['seed'],
        experimental_config['seed'],
        experimental_config['seed'],
        experimental_config['seed'],
    )

    if os.path.exists(bdb_file_name):
        os.remove(bdb_file_name)

    bdb = bayesdb_open(bdb_file_name, seed=byte_str_seed)
    bdb.metamodels['cgpm'].set_multiprocess(False)
    bdb.execute('''
        CREATE TABLE data_table
            FROM 'tests/data/noisy_or_n={number_datapoints}_seed={seed}.csv';
    '''.format(**experimental_config))
    bdb.execute('''
        CREATE POPULATION pop FOR data_table WITH SCHEMA(
            MODEL {column_names}
            AS NOMINAL;
        );
    '''.format(column_names=','.join(experimental_config['column_names'])))

    bdb.execute('CREATE ANALYSIS SCHEMA cc FOR pop WITH BASELINE crosscat;')
    bdb.execute('INITIALIZE 1 MODELS FOR cc;')
    bdb.execute('''
        ANALYZE cc FOR {number_iterations} ITERATION  WAIT(OPTIMIZED);
    '''.format(**experimental_config))
    return bdb

DESIRED_NUMBER_OF_QUESTIONS = 10

NUMBER_MC_SAMPLES = [10, 100]
MINUTES_ANALYSIS = [60, 240]
DIAGNOSIS = {
    'autism': "Autism Spectrum Disorder",
    'adhd': "Attention-Deficit/Hyperactivity Disorder"
}

@pytest.mark.parametrize('number_mc_samples', [1])
@pytest.mark.parametrize('number_datapoints', [1])
@pytest.mark.parametrize('number_iterations', [1])
@pytest.mark.parametrize('seed', range(1, 2))
def test_noisy_or(number_mc_samples, number_datapoints, number_iterations, seed):

    experimental_config = {
        'number_mc_samples' : str(number_mc_samples),
        'number_datapoints' : str(number_datapoints),
        'number_iterations' : str(number_iterations),
        'seed'              : seed,
        'column_names'      : generate_data_disease(seed, number_datapoints).keys()
    }

    # Prepare bdb given an experimental configuration.
    bdb = prep_bdb(experimental_config)

    # Select columns.
    selected_questions = shorten_conditional_entropy(bdb, experimental_config)

    # Save result.
    mkdir('tests/output/selected/')
    output_file_name = 'tests/output/selected/questions_noisy_or_mc={number_mc_samples}' +\
        'n={number_datapoints}_iters={number_iterations}_seed={seed}.csv'
    pd.DataFrame({'selected':selected_questions}).to_csv(
        output_file_name.format(**experimental_config),
        index=False
    )
