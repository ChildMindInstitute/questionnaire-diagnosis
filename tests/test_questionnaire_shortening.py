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

def read_mi(df):
    return df['mi'].values[0]

def get_bql_pattern(next_column, diagnosis, n_samples, selected_questions):
    if selected_questions:
        already_selected_questions = ','.join(selected_questions)
        return '''ESTIMATE MUTUAL INFORMATION OF\n    "{next_column}" WITH
        "{diagnosis}"\n    GIVEN ({already_selected_questions})\n    USING
        {n_samples} SAMPLES AS mi \n    BY pop
                '''.format(
                    next_column=next_column,
                    diagnosis=diagnosis,
                    n_samples=n_samples,
                    already_selected_questions=already_selected_questions
        )
    else:
        return '''ESTIMATE MUTUAL INFORMATION OF\n   "{next_column}" WITH
        "{diagnosis}"\n    USING {n_samples} SAMPLES AS mi BY pop
            '''.format(
                next_column=next_column,
                diagnosis=diagnosis,
                n_samples=n_samples
        )

LEAK_P_SPONTANEOUS = 0.95

N_SYMPTOMS = 4

def generate_data_disease_symptoms(seed, N):
    """Generate data with a noisy-or net. The net is comprised of one parent
    (the disease) and a N_SYMPTOMS children (the symptomns). The prior for
    symptoms to be 'on' is the same for all symptoms. By fixing this prior, we
    can determine the order of questions that should be selected through a
    single parameter in the data generator (the leak parameter).
    """
    prng = RandomState(seed)
    def sample_binary(p):
        return prng.uniform(0 , 1) < p

    disease = [sample_binary(0.5) for _ in range(N)]
    leak_p_symptoms = np.linspace(0.01, 0.49, N_SYMPTOMS)
    data = {}
    for symptom in range(N_SYMPTOMS):
        data['c_' + str(symptom)] = [
            sample_binary(1 - leak_p_symptoms[symptom] * LEAK_P_SPONTANEOUS)
            if disease_value else
            sample_binary(1 - LEAK_P_SPONTANEOUS)
            for disease_value in disease
        ]
    data['target'] = disease
    pd.DataFrame(data).to_csv(
        'tests/data/noisy_or_disease_symptoms_n=%d_seed=%d.csv' % (N, seed),
        index=False
    )
    return data

def generate_data_symptoms_diagnosis(seed, N):
    """Generate data with a noisy-or net. The net is comprised of one child
    (the diagnosis) and N_SYMPTOMS parents (the symptomns). The prior for
    symptoms to be 'on' is the same for all symptoms. By fixing this prior, we
    can determine the order of questions that should be selected through a
    single parameter in the data generator (the leak parameter).
    """
    prng = RandomState(seed)
    def sample_binary(p):
        return prng.uniform(0 , 1) < p

    # Get symptoms.
    def get_col_name(index):
        return 'c_' + str(index)
    data = {
        get_col_name(symptom) : [sample_binary(0.5) for _ in range(N)]
        for symptom in range(N_SYMPTOMS)
    }
    leak_p_symptoms = np.linspace(0.01, 0.49, N_SYMPTOMS)
    data['target'] = []

    for row in range(N):
        leak = LEAK_P_SPONTANEOUS
        for symptom in range(N_SYMPTOMS):
            if data[get_col_name(symptom)][row]:
                leak *=  leak_p_symptoms[symptom]
        data['target'].append(sample_binary(1-leak))

    pd.DataFrame(data).to_csv(
        'tests/data/noisy_or_symptoms_diagnosis_n=%d_seed=%d.csv' % (N, seed),
        index=False
    )
    return data

def generate_data_mix_causal_directions(seed, N):
    """Generate data with a noisy-or net. 3 Nodes are parents to the target and
    3 nodes are children of the target.
    """
    prng = RandomState(seed)
    def sample_binary(p):
        return prng.uniform(0 , 1) < p

    # Get symptoms.
    def get_col_name(index):
        return 'c_' + str(index)

    # Leaks for links from columns 0, 1, and 2 to target.
    target_parent_leaks = [0.01, 0.1, 0.2]
    # Leaks for links from target to columns 3, 4, and 5.
    target_children_leaks = [0.02, 0.15, 0.25]

    # Sample the columns which are parents to the target.
    data = {
        get_col_name(symptom) : [sample_binary(0.5) for _ in range(N)]
        for symptom in range(len(target_parent_leaks))
    }

    # Sample the target.
    data['target'] = []
    for row in range(N):
        leak = LEAK_P_SPONTANEOUS
        for symptom in range(len(target_parent_leaks)):
            if data[get_col_name(symptom)][row]:
                leak *=  target_parent_leaks[symptom]
        data['target'].append(sample_binary(1-leak))

    # Sample the children of the target.
    child_indeces = range(
        len(target_parent_leaks),
        len(target_parent_leaks) + len(target_children_leaks)
    )
    for child in child_indeces:
        data['c_' + str(child)] = [
            sample_binary(
                1 - target_children_leaks[child-len(target_parent_leaks)] * LEAK_P_SPONTANEOUS
            )
            if target_value else
            sample_binary(1 - LEAK_P_SPONTANEOUS)
            for target_value in data['target']
        ]

    # Save and return data.
    pd.DataFrame(data).to_csv(
        'tests/data/noisy_or_causal_mix_n=%d_seed=%d.csv' % (N, seed),
        index=False
    )
    return data

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

def init_question_lists(experimental_config):
    """ Initialized lists search. """
    candidate_questions = experimental_config['column_names']
    # Remove the diagnosis from the candidates.
    candidate_questions.remove('target')
    selected_questions = [] # Initialize: no questions selected yet.
    return candidate_questions, selected_questions

def shorten_conditional_entropy(bdb, experimental_config):
    """Shorten a questionnaire usign conditional entropy as a metric."""
    # Selected questions is an empty list.
    candidate_questions, selected_questions = init_question_lists(
        experimental_config
    )

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


def shorten_conditional_mutual_information(bdb, experimental_config):
    """Shorten a questionnaire usign CMI as a metric."""
    # Selected questions is an empty list.
    candidate_questions, selected_questions = init_question_lists(
        experimental_config
    )
    # While we don't have the desired number of quesions, keep searching.
    while len(selected_questions) < DESIRED_NUMBER_OF_QUESTIONS:
            current_scores = []
            # Loop through all the candidate questions.
            for next_column in candidate_questions:
                current_bql_pattern = get_bql_pattern(
                    next_column,
                    'target',
                    experimental_config['number_mc_samples'],
                    selected_questions
                )
                df = query(bdb, current_bql_pattern)
                current_scores.append(read_mi(df))
            new_question = candidate_questions[np.argmax(current_scores)]
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
            FROM 'tests/data/noisy_or_{data_generator_name}_n={number_datapoints}_seed={seed}.csv';
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

DESIRED_NUMBER_OF_QUESTIONS = 3

SHORTENING_FUNCTIONS = {
    'cond_entropy' : shorten_conditional_entropy,
    'cmi' : shorten_conditional_mutual_information,
}
DATA_GENERATORS = {
    'disease_symptoms'   : generate_data_disease_symptoms,
    'symptoms_diagnosis' : generate_data_symptoms_diagnosis,
    'causal_mix'        : generate_data_mix_causal_directions,
}

@pytest.mark.parametrize('data_generator_name', DATA_GENERATORS)
def test_data_gen_smoke(data_generator_name):
    """Smoke test data generation.

    This test assess only whether the shape of the data generated is correct.
    """
    data = DATA_GENERATORS[data_generator_name](1, 3)
    df = pd.read_csv(
        'tests/data/noisy_or_%s_n=3_seed=1.csv' % (data_generator_name,)
    )

    assert df.equals(pd.DataFrame(data))
    assert len(df) == 3
    if data_generator_name == 'causal_mix':
        assert df.columns.shape[0] == 7
    else:
        assert df.columns.shape[0] == 5
    assert df.columns[0] == 'c_0'
    assert df.columns[1] == 'c_1'

NUMBER_MC_SAMPLES = [1]

@pytest.mark.parametrize('shortening_function_name', SHORTENING_FUNCTIONS)
@pytest.mark.parametrize('data_generator_name', DATA_GENERATORS)
@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
@pytest.mark.parametrize('number_datapoints', [1] )
@pytest.mark.parametrize('number_iterations', [1])
@pytest.mark.parametrize('seed', range(1, 2))
def test_noisy_or(
        shortening_function_name,
        data_generator_name,
        number_mc_samples,
        number_datapoints,
        number_iterations,
        seed
    ):

    experimental_config = {
        'shortening_function_name' : shortening_function_name,
        'data_generator_name' : data_generator_name,
        'number_mc_samples' : str(number_mc_samples),
        'number_datapoints' : str(number_datapoints),
        'number_iterations' : str(number_iterations),
        'seed'              : seed,
        'column_names'      : DATA_GENERATORS[data_generator_name](
            seed, number_datapoints
        ).keys()
    }

    # Prepare bdb given an experimental configuration.
    bdb = prep_bdb(experimental_config)

    # Select columns.
    selected_questions = SHORTENING_FUNCTIONS[shortening_function_name](
        bdb,
        experimental_config
    )

    # Save result.
    mkdir(
        'tests/output/noisy-or/{shortening_function_name}'.format(
            **experimental_config
         )
    )
    output_file_name = 'tests/output/noisy-or/{shortening_function_name}/' +\
        'selected_columns_mc={number_mc_samples}' +\
        'n={number_datapoints}_iters={number_iterations}_seed={seed}.csv'
    pd.DataFrame({'selected':selected_questions}).to_csv(
        output_file_name.format(**experimental_config),
        index=False
    )
