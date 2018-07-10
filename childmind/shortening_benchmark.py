import datetime
import os
os.environ['LOOM_VERBOSITY'] = '0'
import numpy as np
import pandas as pd
import struct
import subprocess
import time

import statsmodels.formula.api as smf


CSV_FILE_PATH = 'questions_v4_depression_dmdd_asd_and_derived.csv'
SEED = 42
SIGNIFICANCE_LEVEL = 0.05

from bql_prog import bql_program

def execute_bql(bdb, bql_strings):
    for bql_string in bql_strings:
        print bql_string
        bdb.execute(bql_string)
        print '---------------------'

def search_for_n_nonredundant_predictors(bdb, n, outcome, potential_predictors):
    """This functions returns the n most relevant (and non-redundant predictors)
    for an outcome."""
    from iventure.utils_bql import query
    mi_str ='''
    ESTIMATE MUTUAL INFORMATION OF
        "{column}" WITH "{outcome}"
        USING 20 SAMPLES AS mi
        BY "questionnaire_data"
    '''
    cmi_str ='''
    ESTIMATE MUTUAL INFORMATION OF
        "{column}" WITH "{outcome}"
        GIVEN ({already_selected_columns})
        USING 20 SAMPLES AS mi
        BY "questionnaire_data"
    '''
    columns = potential_predictors['column'].values.tolist() # turn the
    # dataframe with potential predicots into a list.
    nonredundant_predictors = [] # Inniatilize the list of non-redundant
    #predictors.
    while len(nonredundant_predictors) < n: # As long as we haven't found
    # n non-redundant predictos: repeat.
        mi_values = [] # Initialize
        for column in columns:
            if not nonredundant_predictors: # If we haven't found an predictors
                # yet we run un-conditional mutual information.
                current_query = mi_str.format(column=column, outcome=outcome)
            else: # Otherwise, we condition the query on what we found previously:
                # We have to turn the list of found predictors into a string;
                # e.g ['a', 'b', 'a'] => 'a,b,c'.
                already_selected_columns = ','.join([
                    '"%s"' % col for col in nonredundant_predictors
                ])
                current_query = cmi_str.format(
                    column=column,
                    outcome=outcome,
                    already_selected_columns=already_selected_columns
                )
            # Print the query we are going to execute.
            print current_query
            print ''
            # Run query with BayesDB magics.
            df = query(bdb, current_query)
            # Turn the dataframe into an entry in a list.
            mi_values.append(df['mi'].values[0])
        # Find the maximally informative column and select it.
        selected_column = columns[np.argmax(mi_values)]
        # Add said selected column the list of non-redundant predictors.
        nonredundant_predictors.append(selected_column)
        # Remove said selected column from the list of columns.
        columns.remove(selected_column)
    return nonredundant_predictors

def feature_selection_with_bdb(
        target_column,
        number_of_predictors,
        csv_file_path=CSV_FILE_PATH
    ):
    print '################################## Targetting {} with BayesDB questionnaire ################################### '.format(target_column)
    # XXX: this
    import bayeslite
    from bayeslite.backends.loom_backend import LoomBackend
    from bayeslite import bayesdb_register_backend

    from iventure.utils_bql import nullify
    from iventure.utils_bql import query
    # Create a seed.
    byte_str_seed = struct.pack(
        '<QQQQ',
        SEED,
        SEED,
        SEED,
        SEED,
    )
    # Should we parametrize this further?
    time_stamp = datetime.datetime.now().strftime('%F-%H-%M')
    file_name = 'shortening-' + target_column + '-' + time_stamp
    bdb_file_name = file_name + '.bdb'
    # XXX: we probably don't this.
    if os.path.exists(bdb_file_name):
        os.remove(bdb_file_name)
    # Create bdb file.
    bdb = bayeslite.bayesdb_open(bdb_file_name, seed=byte_str_seed)
    # Configure loom.
    bayesdb_register_backend(
        bdb, LoomBackend(os.path.abspath('loom-files-{}/'.format(file_name)))
    )
    # Load data.
    bdb.execute('''
        CREATE TABLE questionnaire_data
            FROM '{csv_file_path}';
    '''.format(csv_file_path=csv_file_path))
    # Nullify.
    nullify(bdb, 'questionnaire_data', '')
    execute_bql(bdb, bql_program)
    # Get a set of potential predictors.
    bql_str = '''
    SELECT name0 AS "column" FROM dependencies
        WHERE ((name1 = "{}") AND (debprob >= 0.0) AND (name1 != name0))
            ORDER BY depprob DESC LIMIT 50
    '''.format(target_column.lower())
    print bql_str
    potential_predictors = query(bdb, bql_str)
    result = search_for_n_nonredundant_predictors(
        bdb,
        number_of_predictors,
        target_column,
        potential_predictors
    )
    print result
    return pd.Series(result)

def random_feature_subset(
        target_column,
        number_of_predictors,
        csv_file_path=CSV_FILE_PATH
    ):
    df = pd.read_csv(csv_file_path)
    candidates = df.columns.tolist()
    candidates.remove(target_column)
    return pd.Series(
        np.random.choice(candidates, number_of_predictors, replace=False)
    )

def produce_data_frame_without_missing_vals(df):
    # Delete all all-null columns.
    print 'Remove the following cols for varrrank:'
    for i, row in df.isnull().sum().iteritems():
        if row>=len(df)/2:
            del df[i]
    df_dropped_rows = df.dropna()
    del df_dropped_rows['EID']
    return df_dropped_rows

def varrank(
        target_column,
        number_of_predictors,
        csv_file_path=CSV_FILE_PATH
    ):

    r_tempfile = 'temp.csv'
    df = pd.read_csv(csv_file_path)
    df_no_missing_vals = produce_data_frame_without_missing_vals(df)
    df_no_missing_vals .to_csv(r_tempfile)

    #subprocess.call(['./run_varrank.sh'])
    r_command = 'Rscript run_varrank.R {} {} {}'.format(
        target_column,
        r_tempfile,
        number_of_predictors)
    print 'Running....'
    print r_command
    subprocess.call(r_command, shell=True)
    r_output = pd.read_csv(r_tempfile)
    os.remove(r_tempfile)
    return r_output['ordered.var']

def stepwise_log_reg(
        target_column,
        number_of_predictors,
        csv_file_path=CSV_FILE_PATH,
        reversed_order=True
    ):
    """Stepwise, linear logistic regression."""
    data = pd.read_csv(csv_file_path)
    data = data.applymap(
        lambda x: int(x) if isinstance(
            x,
            bool
        ) else x
    )
    candidates = list(data.columns)
    candidates.remove(target_column)
    if reversed_order:
        candidates = list(reversed(candidates))
        print 'reversed'
        print candidates
    selected = []
    failed_attempts = 0
    for candidate in candidates:
        formula = '{} ~ {} + 1'.format(target_column,
                                       ' + '.join(selected + [candidate]))
        print '##########################   Current hypothesis   ##################################'
        print formula
        print '####################################################################################'
        try:
            result = smf.logit(formula, data).fit()
        except:
            failed_attempts +=1
            continue
        print result.summary()
        if result.pvalues[candidate] < SIGNIFICANCE_LEVEL:
            selected.append(candidate)
            for previously_selected in selected:
                if result.pvalues[previously_selected] > SIGNIFICANCE_LEVEL:
                    selected.remove(previously_selected)
    print '####################################################################################'
    print '####################################################################################'
    print  '                          Final result:'
    formula = '{} ~ {} + 1'.format(
        target_column, ' + '.join(selected[:number_of_predictors])
    )
    print formula
    print '####################################################################################'
    print '####################################################################################'
    print 'Logistic regression failed {} times.'.format(failed_attempts)
    return pd.Series(selected[:number_of_predictors])

def shorten_questionnaire(
        shortening_method,
        target_columnm,
        number_of_predictors,
    ):
    start_time = time.time()
    time_stamp = datetime.datetime.now().strftime('%F-%H-%M')
    result = shortening_method(target_columnm, number_of_predictors)
    name = '-'.join([shortening_method.__name__, target_columnm, time_stamp])
    result.to_csv('results/' + name + '.csv')
    elapsed_time = time.time() - start_time
    print 'Running time (seconds):'
    print  elapsed_time


#shortening_method = feature_selection_with_bdb
#shortening_method = varrank
shortening_method = stepwise_log_reg
n = 5
target = 'Anxiety'
shorten_questionnaire(shortening_method, target,  n)
#shortening_methods = [feature_selection_with_bdb]
#target_columnms = [
#    'Major Depressive Disorder',
#    'Anxiety',
#    'Disruptive Mood Dysregulation Disorder',
#]
#
#number_of_predictors = 5
#for shortening_method in shortening_methods:
#    for target_column in target_columnms:
#        shorten_questionnaire(
#            shortening_method,
#            target_column.lower(),
#            number_of_predictors,
#        )
