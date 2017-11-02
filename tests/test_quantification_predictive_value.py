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
import operator
import pandas as pd
import pytest
import struct
import sys
import time

from bayeslite import bayesdb_open
from bayeslite import bayesdb_read_csv

from testing_utils import mkdir

sys.path.append('src')
from quantify_predictive_value import select_columns
from quantify_predictive_value import get_all_scoring_functions

LEAK_P_SPONTANEOUS = 0.95

N_SYMPTOMS = 4

def generate_data_disease_symptoms(seed, N):
    """Generate data with a noisy-or net. The net is comprised of one parent
    (the disease) and a N_SYMPTOMS children (the symptomns). The prior for
    symptoms to be 'on' is the same for all symptoms. By fixing this prior, we
    can determine the order of columns that should be selected through a
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
    can determine the order of columns that should be selected through a
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

def prep_bdb(experimental_config):
    """ This functions prepare a bdb object.

        It reads the csv file, creates the population and runs analysis.
    """
    mkdir('tests/bdb/')
    file_name = 'tests/bdb/test_noisy_or_{data_generator_name}_' +\
        '{scoring_function_name}_mc={number_mc_samples}' +\
        '_n={number_datapoints}_iters={number_iterations}_seed={seed}.bdb'
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
NUMBER_DATAPOINTS = [5]
NUMBER_ITERATIONS = [1]
@pytest.mark.parametrize('target', ['target'])
@pytest.mark.parametrize('scoring_function_name', get_all_scoring_functions())
@pytest.mark.parametrize('data_generator_name', DATA_GENERATORS)
@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
@pytest.mark.parametrize('number_datapoints', NUMBER_DATAPOINTS)
@pytest.mark.parametrize('number_iterations', NUMBER_ITERATIONS)
@pytest.mark.parametrize('seed', range(1, 2))
def test_noisy_or(
        target,
        scoring_function_name,
        data_generator_name,
        number_mc_samples,
        number_datapoints,
        number_iterations,
        seed
    ):

    experimental_config = {
        'target' : target,
        'scoring_function_name' : scoring_function_name,
        'scoring_function' : get_all_scoring_functions()[scoring_function_name],
        'data_generator_name' : data_generator_name,
        'number_mc_samples' : str(number_mc_samples),
        'number_datapoints' : str(number_datapoints),
        'number_iterations' : str(number_iterations),
        'seed'              : seed,
        'desired_number_of_questions': DESIRED_NUMBER_OF_QUESTIONS,
        'column_names'      : DATA_GENERATORS[data_generator_name](
            seed, number_datapoints
        ).keys()
    }

    # Prepare bdb given an experimental configuration.
    bdb = prep_bdb(experimental_config)

    # Select columns.
    selected_columns = select_columns(bdb, experimental_config)

    # Save result.
    mkdir(
        'tests/output/noisy-or/{scoring_function_name}'.format(
            **experimental_config
         )
    )
    output_file_name = 'tests/output/noisy-or/{scoring_function_name}/' +\
        'selected_columns_{data_generator_name}_mc={number_mc_samples}' +\
        'n={number_datapoints}_iters={number_iterations}_seed={seed}.csv'
    pd.DataFrame({'selected':selected_columns}).to_csv(
        output_file_name.format(**experimental_config),
        index=False
    )


@pytest.mark.parametrize('target', ['target'])
@pytest.mark.parametrize('scoring_function_name', get_all_scoring_functions())
@pytest.mark.parametrize('data_generator_name', DATA_GENERATORS)
@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
@pytest.mark.parametrize('number_datapoints', NUMBER_DATAPOINTS)
@pytest.mark.parametrize('number_iterations', NUMBER_ITERATIONS)
def test_get_results(
        target,
        scoring_function_name,
        data_generator_name,
        number_mc_samples,
        number_datapoints,
        number_iterations,
    ):
    experimental_config = {
        'target' : target,
        'scoring_function_name' : scoring_function_name,
        'scoring_function' : get_all_scoring_functions()[scoring_function_name],
        'data_generator_name' : data_generator_name,
        'number_mc_samples' : str(number_mc_samples),
        'number_datapoints' : str(number_datapoints),
        'number_iterations' : str(number_iterations),
    }


    pattern = 'tests/output/noisy-or/{scoring_function_name}/' +\
        'selected_columns_{data_generator_name}_mc={number_mc_samples}' +\
        'n={number_datapoints}_iters={number_iterations}_seed=*.csv'
    print  ""

    all_results = [
        ', '.join(pd.read_csv(file)['selected'].values.tolist())
        for file in glob.glob(pattern.format(**experimental_config))
    ]
    results = dict(
        (ordered_selection, all_results.count(ordered_selection))
        for ordered_selection in set(all_results)
    )
    print  ""
    print  ""
    print scoring_function_name
    print data_generator_name
    max_prob_order = max(results.iteritems(), key=operator.itemgetter(1))
    print "MAP Order: %s with P(%.2f)" % (
        max_prob_order[0], float(max_prob_order[1])/len(all_results)
    )
    print  ""
    print  ""
    print  ""
    mkdir('tests/png/noisy-or')
    output_plot_name = 'tests/png/noisy-or/{scoring_function_name}_' +\
        'selected_columns_{data_generator_name}_mc={number_mc_samples}' +\
        'n={number_datapoints}_iters={number_iterations}.png'

    fig, ax = plt.subplots()
    pd.Series(results).plot(kind='barh', ax=ax)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Order')
    ax.set_title(
        'Orders for selected by %s with data from %s' %\
        (scoring_function_name, data_generator_name,)
    )
    fig.savefig(
        output_plot_name.format(**experimental_config),
        bbox_inches='tight'
    )
