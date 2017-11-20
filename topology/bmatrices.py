from boundary_matrix import build_boundary_matrix
from boundary_matrix import lower_dimensions_index
from boundary_matrix import replace_list
from boundary_matrix import vector_vector_T
import numpy as np
import os
import pandas as pd

data = os.path.join(os.pardir, "train_test_data", "unsplit_no_totals.csv")

df = pd.read_csv(data)

dxes = {("Anx", "adhd", "asd"):
           {(0,0,0),
           (1,0,0),
           (0,1,0),
           (0,0,1),
           (1,1,0),
           (1,0,1),
           (0,1,1),
           (1,1,1)}
       }
       
subdfs = {}
for k in dxes:
    for r in dxes[k]:
        d = set(zip(k, r))
        subdfs[str(dict(d))] = df.loc[
            eval(
                " & ".join(
                    {
                        "(df[\"{0}\"] == {1})".format(key, value) for key, value in d
                    }
                )
            )
        ].copy()
        
bmdir = os.path.join(os.pardir, os.pardir, "boundary_matrices", "train")
if not os.path.exists(bmdir):
    os.makedirs(bmdir)
    
for n in subdfs:
    df_in_use = subdfs[n].loc[subdfs[n]["train"]].drop(
        ["Dx", "Anx", "adhd", "asd", "train", "Sex", "Age", "EID"],
        axis=1
    ).copy()
    df_in_use.reset_index(inplace=True, drop=True)
    for col in [col for col in df_in_use.columns if col != "EID"]:
        df_in_use[col] = (df_in_use[col] == max(df_in_use[col]))
    bmatrix = build_boundary_matrix(vector_vector_T(df_in_use))
    with open(os.path.join(
        bmdir,
        "{0}.txt".format(replace_list(n.replace(" ", "_"),[":", ",", "'", "{", "}"], ""))
    ), "w") as of:
        of.write(str(bmatrix))