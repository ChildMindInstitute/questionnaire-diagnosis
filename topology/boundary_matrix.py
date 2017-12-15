from itertools import combinations
import numpy as np
import pandas as pd


def build_boundary_matrix(vv, max_dimensions=None):
    """
    Function to build a boundary matrix from vector of vectors (list of lists).
    
    Parameter
    ---------
    vv : list of lists
    
    max_dimensions : int or None
    
    Returns
    -------
    bmatrix : list of 2-tuples of (int, list of ints)
    """
    md = max([point for vector in vv for point in vector])
    all_boundaries = {0 : {str([p]):p for p in range(md + 1)}}
    # all_boundaries =
    #     {
    #         dimension :
    #         {
    #             str([point_indices]) :
    #                 boundary_element_index
    #         }
    #     } ∀ points ∈ range(max(point_indices))
    return(all_boundaries)

    #TODO
    i = 1
    md = md + 1 if not max_dimensions else max_dimensions
    for v in vv:
        while (i <= md):
            for com in list(combinations(v, i + 1)):
                print(com)
                all_boundaries[str(list(com))] = (
                    i,
                    len(all_boundaries)
                )
            i = i + 1
    # now we have a pointwise boundary matrix
    print(all_boundaries)
    bmatrix = [
        (
            all_boundaries[element][0],
            element if all_boundaries[element][0] > 0 else []
        ) for element in all_boundaries
    ]
    return(bmatrix)


def build_boundary_matrix_max_only_in_each_D(vv):
    """
    Function to build a boundary matrix from vector of vectors (list of lists).
    
    Parameter
    ---------
    vv : list of lists
    
    Returns
    -------
    bmatrix : list of 2-tuples of (int, list of ints)
    """
    all_boundaries = {
                         str([p]):p for p in range(
                             max([point for vector in vv for point in vector]) + 1
                         )
                     }
    bmatrix = list()
    for v in vv:
        all_boundaries = lower_dimensions_index(v, all_boundaries)
    for v in all_boundaries:
        z = list()
        for x in combinations(eval(v), len(eval(v)) - 1):
            if len(x):
                z.append(all_boundaries[str(list(x))])
        if len(z):
            bmatrix.append((len(z) - 1, z))
        else:
            bmatrix.append((0, []))
    return(bmatrix)


def lower_dimensions_index(vector, index_dict):
    """
    Function to index all lower-dimension structures needed for
    the most complex structures in a boundary matrix.
    
    Parameters
    ----------
    vector : list
    
    index_dict : dictionary of {vector (list): index (int)} {key: value} pairs
    
    Returns
    -------
    index_dict : dictionary of {vector (list): index (int)} {key: value} pairs
    """
    if str(vector) in index_dict or (type(vector) == "int" and vector in index_dict):
        return(index_dict)
    else:
        for v in list(combinations(vector, len(vector) - 1)):
            lv = list(v)
            if len(lv) > 1:
                lower_dimensions_index(lv, index_dict)
    index_dict[str(vector)] = len(index_dict)
    return(index_dict)


def point_up(all_boundaries, max_dimensions=None):
    """
    Function to convert a point-defined boundary matrix into one defined by immediately-lower-dimensional
    components.
    
    Parameters
    ----------
    all_boundaries: dictionary
        {
             str([point_indices]) :
                 (
                     dimension,
                     boundary_element_index
                 )
         } ∀ points ∈ range(max(point_indices))
    
    max_dimensions: int or None
    """
    pass


def replace_list(self, old, new, count=None):
    """
    Expansion of str.replace method to allow iterable of old to be replaced with the same new.
    
    Parameters
    ----------
    self : string
        string in which to replace old with new
    
    old : iterable of strings
        strings to replace with new
        
    new : string
        string with which to replace strings in old
        
    count : int (optional)
        If the optional argument count is given, only the first count occurrences are replaced.
        
    Returns
    -------
    self : string
        with all strings in old replaced with new
    """
    if(count):
        for oldstring in old:
            self = self.replace(oldstring, new, count)
    else:
        for oldstring in old:
            self = self.replace(oldstring, new)
    return(self)


def vector_vector(df):
    """
    Function to return a list of lists (vector of vectors) given a dataframe.
    
    Parameter
    ---------
    df : DataFrame
    
    Returns
    -------
    d : list of lists
    """
    d = list(list(df.loc[df[c]].index.values) for c in df.columns)
    return(d)


def vector_vector_T(df):
    """
    Function to return a list of lists (vector of vectors) given a transposed dataframe.
    
    Parameter
    ---------
    df : DataFrame
    
    Returns
    -------
    d : list of lists
    """
    df_T = df.T
    df_T.reset_index(inplace=True)
    d = list(list(df_T.loc[df_T[c]].index.values) for c in [c for c in df_T.columns if c != "index"])
    return(d)