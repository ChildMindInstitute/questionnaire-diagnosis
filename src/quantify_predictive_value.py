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

import numpy as np

from iventure.utils_bql import query

def read_score(df):
    return df['score'].values[0]

def init_column_lists(experimental_config):
    """ Initialized lists search. """
    candidate_columns = experimental_config['column_names']
    # Remove the diagnosis from the candidates.
    candidate_columns.remove('target')
    selected_columns = [] # Initialize: no columns selected yet.
    return candidate_columns, selected_columns


def select_columns(bdb, experimental_config):
    """Select most predictively relevant columns."""
    # Selected columns is an empty list.
    candidate_columns, selected_columns = init_column_lists(
        experimental_config
    )
    # While we don't have the desired number of quesions, keep searching.
    while len(selected_columns) <\
        experimental_config['desired_number_of_questions']:
            current_scores = []
            # Loop through all the candidate columns.
            for next_column in candidate_columns:
                bql_str = experimental_config['scoring_function'](
                    bdb,
                    experimental_config['target'],
                    next_column,
                    selected_columns,
                    experimental_config['number_mc_samples']
                )
                print bql_str
                df = query(bdb, bql_str)
                current_scores.append(read_score(df))
            new_column = candidate_columns[np.argmax(current_scores)]
            selected_columns.append(new_column)
            candidate_columns.remove(new_column)
    return selected_columns

def score_columns_conditional_mutual_information(
        bdb,
        target,
        next_column,
        selected_columns,
        number_mc_samples,
    ):
    """Return BQL string to compute a conditional mutual information."""

    selected_columns_str = ','.join([
        '"%s"' % column for column in selected_columns
    ])
    if selected_columns_str == '':
        return '''ESTIMATE MUTUAL INFORMATION OF\n   "{next_column}" WITH
        "{target}"\n USING {n_samples} SAMPLES AS score BY pop
            '''.format(
                next_column=next_column,
                target=target,
                n_samples=number_mc_samples
        )
    else:
        return '''ESTIMATE MUTUAL INFORMATION OF\n    "{next_column}" WITH
        "{target}"\n  GIVEN ({already_selected_columns})\n    USING
        {n_samples} SAMPLES AS score \n    BY pop
                '''.format(
                    next_column=next_column,
                    target=target,
                    n_samples=number_mc_samples,
                    already_selected_columns=selected_columns_str
        )


def score_columns_conditional_entropy(
        bdb,
        target,
        next_column,
        selected_columns,
        number_mc_samples
    ):
    """Return BQL string to compute a conditional entropy score."""
    selected_columns_str = ','.join([
        '"%s"' % column for column in
        [next_column] + selected_columns
    ])
    return '''
      SELECT  mi - entropy AS score FROM
        (ESTIMATE MUTUAL INFORMATION OF
          ({selected_columns}) WITH ({selected_columns})
          USING {n_samples} SAMPLES AS entropy BY pop),
        (ESTIMATE MUTUAL INFORMATION OF
          ({selected_columns}) WITH "{target}"
          USING {n_samples} SAMPLES AS mi BY pop)
    '''.format(
        selected_columns=selected_columns_str,
        n_samples=number_mc_samples,
        target = target
    )
