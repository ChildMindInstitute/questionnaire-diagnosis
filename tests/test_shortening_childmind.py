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

import sys
sys.path.append('src')
from quantify_predictive_value import select_columns
from quantify_predictive_value import get_all_scoring_functions
from testing_utils import mkdir

DESIRED_NUMBER_OF_QUESTIONS = 10

NUMBER_MC_SAMPLES = [100]
MINUTES_ANALYSIS = [60]
DIAGNOSIS = {
    #'autism': "Autism Spectrum Disorder",
    'adhd': "Attention-Deficit/Hyperactivity Disorder"
}

def prep_bdb(experimental_config):
    """ This functions prepare a bdb object.

        It reads the csv file, creates the population and runs analysis.
    """
    mkdir('tests/bdb/')
    file_name = 'tests/bdb/test_childmind_{target_name}_' +\
        '{scoring_function_name}_mc={number_mc_samples}' +\
        '_minutes={minutes_analysis}_seed={seed}.bdb'
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
    '''.format(**experimental_config))

    return bdb

@pytest.mark.parametrize('scoring_function_name', get_all_scoring_functions())
@pytest.mark.parametrize('number_mc_samples', NUMBER_MC_SAMPLES)
@pytest.mark.parametrize('minutes_analysis', MINUTES_ANALYSIS)
@pytest.mark.parametrize('diagnosis', DIAGNOSIS.keys())
@pytest.mark.parametrize('seed', range(1, 11))
def test_questionnaire_short(
        scoring_function_name,
        number_mc_samples,
        minutes_analysis,
        diagnosis,
        seed
    ):
    experimental_config = {
        'target' : DIAGNOSIS[diagnosis],
        'target_name' : diagnosis,
        'scoring_function_name' : scoring_function_name,
        'scoring_function' : get_all_scoring_functions()[scoring_function_name],
        'data_generator_name' : 'childmind',
        'number_mc_samples' : str(number_mc_samples),
        'minutes_analysis' : str(minutes_analysis),
        'seed'              : seed,
        'desired_number_of_questions': DESIRED_NUMBER_OF_QUESTIONS,
    }
    # Prepare bdb given an experimental configuration.
    bdb = prep_bdb(experimental_config)
    experimental_config['column_names'] = query(bdb, '''
        SELECT * FROM data_table LIMIT 1;
    ''').columns.tolist()
    experimental_config['column_names'].remove('EID')

    # Select columns.
    selected_columns = select_columns(bdb, experimental_config)

    # Save result.
    mkdir(
        'tests/output/childmind/{scoring_function_name}'.format(
            **experimental_config
         )
    )

    output_file_name = 'tests/output/childmind/{scoring_function_name}/' +\
        'selected_columns_childmind_mc={number_mc_samples}' +\
        '_minutes={minutes_analysis}_seed={seed}.csv'
    pd.DataFrame({'selected':selected_columns}).to_csv(
        output_file_name.format(**experimental_config),
        index=False
    )
