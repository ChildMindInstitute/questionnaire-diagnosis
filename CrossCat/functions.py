import pandas as pd


def contingency(pred, act, var):
    """
    Function to build contingency table from
    predicted-value and actual-value DataFrames
    with matching dependent variable column name.
    
    Parameters
    ----------
    pred: DataFrame
        predicted-value DataFrame
    
    act: DataFrame
        actual-value DataFrame
    
    var: string
        name of column to compare
    """
    try:
        tp = ((pred[str(var)] == act[str(var)]) & (pred[str(var)] == 1)).value_counts().loc[True]
    except:
        tp = 0
    try:
        tn = ((pred[str(var)] == act[str(var)]) & (pred[str(var)] == 0)).value_counts().loc[True]
    except:
        tn = 0
    try:
        fp = ((pred[str(var)] > act[str(var)])).value_counts().loc[True]
    except:
        fp = 0
    try:
        fn = ((pred[str(var)] < act[str(var)])).value_counts().loc[True]
    except:
        fn = 0
    return (pd.DataFrame(
          [[tp, fn],
           [fp, tn]],
          columns=pd.MultiIndex.from_tuples([('actual', 'True'), ('actual', 'False')]),
          index=pd.MultiIndex.from_tuples([('predicted', 'True'), ('predicted', 'False')])
    ))