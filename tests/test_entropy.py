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
sys.path.append('modules/NPEET')
from entropy_estimators import entropyd

P_A = [0.1, 0.3, 0.6]

P_B_given_A = {
    0: [0.1, 0.3, 0.6],
    1: [0.49, 0.01, 0.5],
    2: [0.6, 0.1, 0.3],
}

# TODO: P_A_B and P_B!

ENTROPY_PLOT_LIMITS = [-0.1, 1.4]

def generate_data(seed, N):
    prng = RandomState(seed)
    def sample_categorical(ps):
        return prng.multinomial(1, ps).argmax()
    col_a = [sample_categorical(P_A) for _ in range(N)]
    col_b = [
        sample_categorical(P_B_given_A[value_a]) for value_a in col_a
    ]
    mkdir('tests/data')
    df = pd.DataFrame({'a': col_a, 'b': col_b})
    df.to_csv('tests/data/synth_data_n=%d_seed=%d.csv' % (N, seed), index=False)
    return df

def test_data_gen_smoke():
    generate_data(1, 3)
    df = pd.read_csv('tests/data/synth_data_n=3_seed=1.csv')
    assert len(df) == 3
    assert df.columns[0] == 'a'
    assert df.columns[1] == 'b'

def compute_entropy():
    return - sum([np.log(p_a) * p_a for p_a in P_A])



NUMBER_MC_SAMPLES = [10, 20, 50, 100, 200, 500, 1000]
NUMBER_DATAPOINTS = [10, 50 , 100, 500]
NUMBER_ITERATIONS = [100, 500]

@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
@pytest.mark.parametrize('number_datapoints', NUMBER_DATAPOINTS)
@pytest.mark.parametrize('number_iterations', NUMBER_ITERATIONS)
@pytest.mark.parametrize('seed', range(1, 100))
def test_cc_entropy(number_mc_samples, number_datapoints, number_iterations, seed):
    experiment_dict = {
        'number_mc_samples' : str(number_mc_samples),
        'number_datapoints' : str(number_datapoints),
        'number_iterations' : str(number_iterations),
        'seed' : seed,
    }

    data  = generate_data(seed, number_datapoints)
    mkdir('tests/bdb/')
    file_name = 'tests/bdb/test_mc={number_mc_samples}_n={number_datapoints}' +\
        '_iters={number_iterations}_seed={seed}.bdb'
    bdb_file_name = file_name.format(**experiment_dict)
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
            FROM 'tests/data/synth_data_n={number_datapoints}_seed={seed}.csv';
    '''.format(**experiment_dict))
    bdb.execute('''
        CREATE POPULATION pop FOR data_table WITH SCHEMA(
            MODEL
                a,
                b
            AS NOMINAL;
        );
    ''')
    bdb.execute('CREATE ANALYSIS SCHEMA cc FOR pop WITH BASELINE crosscat;')
    bdb.execute('INITIALIZE 1 MODELS FOR cc;')
    bdb.execute('''
        ANALYZE cc FOR {number_iterations} ITERATION  WAIT(OPTIMIZED);
    '''.format(**experiment_dict))
    start = time.time()
    df = query(bdb, '''
        ESTIMATE MUTUAL INFORMATION OF a WITH a USING
        {number_mc_samples} SAMPLES AS entropy_cc BY pop;
    '''.format(**experiment_dict))
    df['entropy_true'] = compute_entropy()
    df['entropy_npeet'] = entropyd(data['a'].values.tolist())
    df['number_mc_samples'] = number_mc_samples
    df['number_iterations'] = number_iterations
    df['number_datapoints'] = number_datapoints
    df['time_seconds'] = time.time() - start
    mkdir('tests/output')
    output_file_name = 'tests/output/test_mc={number_mc_samples}' +\
        '_n={number_datapoints}_iters={number_iterations}_seed={seed}.csv'
    df.to_csv(output_file_name.format(**experiment_dict), index=False)


@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
#@pytest.mark.parametrize('number_datapoints', NUMBER_DATAPOINTS)
@pytest.mark.parametrize('number_iterations', NUMBER_ITERATIONS)
def test_plot_fixed_samples(number_mc_samples, number_iterations):
    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob('tests/output/*.csv')]
    )
    import seaborn as sns
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    temp = df[df['number_mc_samples'] == number_mc_samples]
    data_to_plot = temp[temp['number_iterations'] == number_iterations]
    with sns.axes_style("whitegrid"):
        ax = sns.stripplot(
            x="number_datapoints",
            y="entropy_cc",
            data=data_to_plot,
            ax=ax,
            order=NUMBER_DATAPOINTS,
            jitter=True
        )
    if np.isnan(df['entropy_true'].values[0]):
        true_entropy = 0
    else:
        true_entropy = df['entropy_true'].values[0]
    ax.axhline(
        y=true_entropy,
        color='r',
        linestyle='-',
        label='True entropy'
    )
    ax.set_ylim(ENTROPY_PLOT_LIMITS)
    ax.set_title(
        'Number of datapoints vs. entropy' +\
        '\nusing %d MC samples after %d iterations of analysis'\
            % (number_mc_samples, number_iterations)
    )
    fig.savefig('tests/png/entropy_fixed_samples_mc%d_iter%d.png' % \
        (number_mc_samples, number_iterations))

@pytest.mark.parametrize('number_datapoints', NUMBER_DATAPOINTS)
@pytest.mark.parametrize('number_iterations', NUMBER_ITERATIONS)
def test_plot_fixed_n(number_datapoints, number_iterations):
    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob('tests/output/*.csv')]
    )
    import seaborn as sns
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    temp = df[df['number_datapoints'] == number_datapoints]
    data_to_plot = temp[temp['number_iterations'] == number_iterations]
    ax = sns.stripplot(
        x='number_mc_samples',
        y="entropy_cc",
        data=data_to_plot,
        ax=ax,
        order=NUMBER_MC_SAMPLES,
        jitter=True
    )
    if np.isnan(df['entropy_true'].values[0]):
        true_entropy = 0
    else:
        true_entropy = df['entropy_true'].values[0]
    ax.axhline(
        y=true_entropy,
        color='r',
        linestyle='-',
        label='True entropy'
    )
    ax.legend()
    ax.set_ylim(ENTROPY_PLOT_LIMITS)
    ax.set_title(
        'Number of MC samples vs. entropy' +\
        '\nusing %d datapoints after %d iterations of analysis'\
            % (number_datapoints, number_iterations)
    )
    fig.savefig('tests/png/entropy_fixed_n%d_iter%d.png' % \
        (number_datapoints, number_iterations))

@pytest.mark.parametrize('number_datapoints', NUMBER_DATAPOINTS)
@pytest.mark.parametrize('number_iterations', NUMBER_ITERATIONS)
def test_plot_timing(number_datapoints, number_iterations):
    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob('tests/output/*.csv')]
    )
    fig, ax = plt.subplots()
    temp = df[df['number_datapoints'] == number_datapoints]
    data_to_plot = temp[temp['number_iterations'] == number_iterations]
    entropy_cc = data_to_plot['entropy_cc'].values
    if np.isnan(df['entropy_true'].values[0]):
        true_entropy = 0
    else:
        true_entropy = df['entropy_true'].values[0]

    error = np.abs(entropy_cc - true_entropy)
    ax.scatter(data_to_plot['time_seconds'], error)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Absolute error')
    ax.legend()
    #ax.set_ylim(ENTROPY_PLOT_LIMITS)
    ax.set_title(
        'Time vs. accuracy' +\
        '\nusing %d datapoints after %d iterations of analysis'\
            % (number_datapoints, number_iterations)
    )
    ax.grid(True)
    fig.savefig('tests/png/timing_vs_accuracy_fixed_n%d_iter%d.png' % \
        (number_datapoints, number_iterations))
