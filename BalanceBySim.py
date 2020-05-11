import pandas as pd
import numpy as np

from rdkit import Chem
from math import floor


def BalanceBySim(df, sim_thresh):
    # Calculate MACCS descriptors
    df['MACCS'] = df['Mol'].apply(Chem.rdMolDescriptors.GetMACCSKeysFingerprint)
    df['Outcome'] = df['Outcome'].astype(int)
    
    # Define larger class
    bigger_class = df.loc[df.Outcome == 0].copy()
    smaller_class = df.loc[df.Outcome == 1].copy()
    if len(smaller_class) > len(bigger_class):
        bigger_class, smaller_class = smaller_class, bigger_class

    comparison = smaller_class.MACCS.tolist()

    select_size = int(floor(len(smaller_class)/sim_thresh))

    bigger_class['Max_Tanimoto_Sim'] = bigger_class['MACCS'].apply(lambda x: max([Chem.DataStructs.TanimotoSimilarity(x, comparison[i]) for i in range(len(comparison))]))
    bigger_class.sort_values(by='Max_Tanimoto_Sim', ascending=False, inplace=True)
    bigger_class.reset_index(drop=False, inplace=True)

    bigger_closest = bigger_class.loc[:select_size,'index'].tolist()
    second_half_selection = np.linspace(start=select_size+1, stop=len(bigger_class), endpoint=False, num=select_size, dtype='int')
    bigger_linear = [int(_) for _ in bigger_class.loc[second_half_selection,'index'].tolist()]

    maccs_bal = smaller_class.index.tolist() + bigger_closest + bigger_linear
    
    df = df.drop(['MACCS'], axis=1)
    df['Set'] = np.where(df.index.isin(maccs_bal), 'train', 'ext')
    
    return df #(df.loc[df.index.isin(maccs_bal)], df.loc[~df.index.isin(maccs_bal)])
