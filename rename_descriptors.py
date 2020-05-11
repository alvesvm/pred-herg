import pandas as pd

def rename_descriptors(descriptors):
    renamed = descriptors.rename(columns={0: 'MolLogP', 1: 'MolMR', 2: 'FpDensityMorgan1',
                                          3: 'FpDensityMorgan2',4: 'FpDensityMorgan3', 5: 'FractionCSP3',
                                          6: 'HeavyAtomMolWt', 7: 'MaxAbsPartialCharge',
                                          8: 'MaxPartialCharge', 9: 'MinAbsPartialCharge',
                                          10: 'MinPartialCharge', 11: 'MolWt', 12: 'NumRadicalElectrons',
                                          13: 'NumValenceElectrons', 14: 'MaxAbsEStateIndex',
                                          15: 'MaxEStateIndex', 16: 'MinAbsEStateIndex',
                                          17: 'MinEStateIndex', 18: 'EState_VSA1',19: 'EState_VSA10',
                                          20: 'EState_VSA11', 21: 'EState_VSA2', 22: 'EState_VSA3',
                                          23: 'EState_VSA4', 24: 'EState_VSA5', 25: 'EState_VSA6',
                                          26: 'EState_VSA7', 27: 'EState_VSA8', 28: 'EState_VSA9',
                                          29: 'fr_Al_COO', 30: 'fr_Al_OH', 31: 'fr_Al_OH_noTert',
                                          32: 'fr_aldehyde', 33: 'fr_alkyl_carbamate', 34: 'fr_alkyl_halide',
                                          35: 'fr_allylic_oxid', 36: 'fr_amide', 37: 'fr_amidine',
                                          38: 'fr_aniline', 39: 'fr_Ar_COO', 40: 'fr_Ar_N', 41: 'fr_Ar_NH',
                                          42: 'fr_Ar_OH', 43: 'fr_ArN',  44: 'fr_aryl_methyl',
                                          45: 'fr_azide', 46: 'fr_azo', 47: 'fr_barbitur',
                                          48: 'fr_benzene', 49: 'fr_benzodiazepine', 50: 'fr_bicyclic',
                                          51: 'fr_C_O',  52: 'fr_C_O_noCOO', 53: 'fr_C_S', 54: 'fr_COO',
                                          55: 'fr_COO2', 56: 'fr_diazo', 57: 'fr_dihydropyridine',
                                          58: 'fr_epoxide', 59: 'fr_ester', 60: 'fr_ether',  61: 'fr_furan',
                                          62: 'fr_guanido', 63: 'fr_halogen', 64: 'fr_hdrzine',
                                          65: 'fr_hdrzone', 66: 'fr_HOCCN', 67: 'fr_imidazole',
                                          68: 'fr_imide',  69: 'fr_Imine', 70: 'fr_isocyan',
                                          71: 'fr_isothiocyan', 72: 'fr_ketone',  73: 'fr_ketone_Topliss',
                                          74: 'fr_lactam', 75: 'fr_lactone', 76: 'fr_methoxy',
                                          77: 'fr_morpholine', 78: 'fr_N_O', 79: 'fr_Ndealkylation1',
                                          80: 'fr_Ndealkylation2', 81: 'fr_NH0', 82: 'fr_NH1',
                                          83: 'fr_NH2', 84: 'fr_Nhpyrrole', 85: 'fr_nitrile',
                                          86: 'fr_nitro', 87: 'fr_nitro_arom',  88: 'fr_nitro_arom_nonortho',
                                          89: 'fr_nitroso', 90: 'fr_oxazole', 91: 'fr_oxime',
                                          92: 'fr_para_hydroxylation', 93: 'fr_phenol',
                                          94: 'fr_phenol_noOrthoHbond', 95: 'fr_phos_acid',
                                          96: 'fr_phos_ester', 97: 'fr_piperdine', 98: 'fr_piperzine',
                                          99: 'fr_priamide',  100: 'fr_prisulfonamd', 101: 'fr_pyridine',
                                          102: 'fr_quatN',  103: 'fr_SH', 104: 'fr_sulfide',
                                          105: 'fr_sulfonamd', 106: 'fr_sulfone',  107: 'fr_term_acetylene',
                                          108: 'fr_tetrazole', 109: 'fr_thiazole',  110: 'fr_thiocyan',
                                          111: 'fr_thiophene', 112: 'fr_unbrch_alkane', 113: 'fr_urea',
                                          114: 'BalabanJ', 115: 'BertzCT', 116: 'Chi0', 117: 'Chi0n',
                                          118: 'Chi0v', 119: 'Chi1', 120: 'Chi1n', 121: 'Chi1v',
                                          122: 'Chi2n', 123: 'Chi2v', 124: 'Chi3n', 125: 'Chi3v',
                                          126: 'Chi4n', 127: 'Chi4v',  128: 'HallKierAlpha', 129: 'Ipc',
                                          130: 'Kappa1', 131: 'Kappa2', 132: 'Kappa3', 133: 'HeavyAtomCount',
                                          134: 'NHOHCount', 135: 'NOCount',  136: 'NumAliphaticCarbocycles',
                                          137: 'NumAliphaticHeterocycles',  138: 'NumAliphaticRings',
                                          139: 'NumAromaticCarbocycles',  140: 'NumAromaticHeterocycles',
                                          141: 'NumAromaticRings', 142: 'NumHAcceptors', 143: 'NumHDonors',
                                          144: 'NumHeteroatoms', 145: 'NumRotatableBonds',
                                          146: 'NumSaturatedCarbocycles', 147: 'NumSaturatedHeterocycles',
                                          148: 'NumSaturatedRings', 149: 'RingCount', 150: 'LabuteASA',
                                          151: 'PEOE_VSA1', 152: 'PEOE_VSA10', 153: 'PEOE_VSA11',
                                          154: 'PEOE_VSA12', 155: 'PEOE_VSA13',  156: 'PEOE_VSA14',
                                          157: 'PEOE_VSA2', 158: 'PEOE_VSA3', 159: 'PEOE_VSA4',
                                          160: 'PEOE_VSA5', 161: 'PEOE_VSA6', 162: 'PEOE_VSA7',
                                          163: 'PEOE_VSA8',  164: 'PEOE_VSA9', 165: 'SlogP_VSA1',
                                          166: 'SlogP_VSA10', 167: 'SlogP_VSA11', 168: 'SlogP_VSA12',
                                          169: 'SlogP_VSA2', 170: 'SlogP_VSA3', 171: 'SlogP_VSA4',
                                          172: 'SlogP_VSA5', 173: 'SlogP_VSA6', 174: 'SlogP_VSA7',
                                          175: 'SlogP_VSA8',  176: 'SlogP_VSA9', 177: 'SMR_VSA1',
                                          178: 'SMR_VSA10', 179: 'SMR_VSA2',  180: 'SMR_VSA3',
                                          181: 'SMR_VSA4', 182: 'SMR_VSA5', 183: 'SMR_VSA6',184: 'SMR_VSA7',
                                          185: 'SMR_VSA8', 186: 'SMR_VSA9', 187: 'TPSA'})
    return renamed
