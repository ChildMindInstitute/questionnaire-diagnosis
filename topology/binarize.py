import pandas as pd
import random


def binarize(data, target_columns=['target']):
    """
    """
    new_data = data[
        [
            col for col in data.columns if \
            col not in target_columns
        ]
    ].copy().applymap(
        lambda x: 1 if x else 0
    )
    return(
        pd.concat(
            [
                new_data,
                data[target_columns]
            ],
            axis=1
        )
    )


def shuffle_dataframe_remove_labels(data, target_columns=['target']):
    """
    Function to remove index and column labels and shuffle both axes,
    putting target columns on the far right.
    
    Parameters
    ----------
    data : DataFrame
    
    target_columns : list of strings
        each string is a column label for a column to keep to the right.
        
    Returns
    -------
    new_data : DataFrame
    
    Example
    -------
    >>> import pandas as pd
    >>> import random
    >>> data = pd.DataFrame({
    ...     "Iron Man": [2008, 2010, 2013, None],
    ...     "Hulk": [2008, None, None, None],
    ...     "Thor": [2011, 2013, 2017, None],
    ...     "Captain America": [2011, 2014, 2016, None],
    ...     "Avengers": [2012, 2015, 2018, 2019]
    ... }).T
    >>> data["count"] = data.apply(lambda x: x.count(), axis=1)
    >>> data.equals(
    ...      shuffle_dataframe_remove_labels(data, ["count"])
    ... )
    False
    >>> data.shape == shuffle_dataframe_remove_labels(
    ...     data, ["count"]
    ... ).shape
    True
    >>> shuffle_dataframe_remove_labels(data, ["count"]).equals(
    ...     shuffle_dataframe_remove_labels(data, ["count"])
    ... )
    False
    >>> shuffle_dataframe_remove_labels(
    ...     data, ["count"]
    ... ).shape == shuffle_dataframe_remove_labels(
    ...     data, ["count"]
    ... ).shape
    True
    >>> shuffle_dataframe_remove_labels(
    ...     data, ["count"]
    ... ).columns[-1]
    'count'
    """
    random_index = list(
        range(
            len(
                list(
                    data.index
                )
            )
        )
    )
    random_columns = list(
        range(
            len(
                list(
                    data.columns
                )
            ) - len(
                target_columns
            )
        )
    )
    random.shuffle(random_columns)
    random_columns = [*random_columns, *target_columns]
    random.shuffle(random_index)
    data.index=random_index
    data.columns=random_columns
    data.sort_index(inplace=True)
    new_data = pd.concat(
        [
            data[
                    [
                        col for col in data.columns if \
                        col not in target_columns
                    ]
            ].copy(),
            data[target_columns].copy()
        ],
        axis=1
    )
    new_data_columns = {
        str(c): c for c in new_data.columns
    }
    new_data = new_data[
        [
            new_data_columns[
                col
            ] for col in sorted(
                new_data_columns
            )
        ]
    ].copy()
    return(new_data)