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


def dependent_variables(
    in_file,
    out_file=None
):
    """
    Function to create columns and subset files for
    models differentiating between
    ADHD subtypes (inattentive vs combined type) and
    between autism spectrum disorder vs no diagnosis
    
    Parameters
    ----------
    in_file: string
        csv path without extension
    
    out_file: string, optional
        csv path without extension
    
    Returns
    -------
    None
    
    Outputs
    -------
    out_file: csv
    """
    out_file = in_file if not out_file else out_file
    # read file
    data = pd.read_csv("{0}.csv".format(
        in_file
    ) if not in_file.endswith(
        ".csv"
    ) else in_file)
    # ADHD subtypes
    data["ADHD subtype"] = pd.concat([
        pd.Series(
            "ADHD-Combined Type",
            index=data.index
        ).loc[
            data[
                "Dx"
            ].str.contains(
                'ADHD-Combined Type'
            )
        ],
        pd.Series(
            "ADHD-Inattentive Type",
            index=data.index
        ).loc[
            data[
                "Dx"
            ].str.contains(
                'ADHD-Inattentive Type'
            )
        ]
    ])
    ADHD_subtypes = data.dropna(
        axis=0,
        subset=["ADHD subtype"]
    )
    # ASD
    data["ASD"] = data[
        "Dx"
    ].str.contains(
        "Autism Spectrum Disorder"
    )
    # write files
    data.dropna(
        axis=0,
        how='all'
    ).dropna(
        axis=1,
        how='all'
    ).drop(
        labels=['Dx', 'Anx', 'train', 'Dx2', 'ADHD subtype'],
        axis=1,
        errors='ignore'
    ).to_csv(
        "{0}_ASD.csv".format(
            out_file
        ),
        index=False
    )    
    ADHD_subtypes.dropna(
        axis=0,
        how='all'
    ).dropna(
        axis=1,
        how='all'
    ).drop(
        labels=['Dx', 'Anx', 'train', 'Dx2'],
        axis=1,
        errors='ignore'
    ).to_csv(
        "{0}_ADHD_subtypes.csv".format(
            out_file
        ),
        index=False
    )           