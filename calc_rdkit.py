import pandas as pd
import numpy as np
#from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, MolSurf, Lipinski, Fragments, EState, GraphDescriptors

def calc_rdkit(mol):
    descriptors = pd.Series(np.array([Crippen.MolLogP(mol),
                                      Crippen.MolMR(mol),
                                      Descriptors.FpDensityMorgan1(mol),
                                      Descriptors.FpDensityMorgan2(mol),
                                      Descriptors.FpDensityMorgan3(mol),
                                      Descriptors.FractionCSP3(mol),
                                      Descriptors.HeavyAtomMolWt(mol),
                                      Descriptors.MaxAbsPartialCharge(mol),
                                      Descriptors.MaxPartialCharge(mol),
                                      Descriptors.MinAbsPartialCharge(mol),
                                      Descriptors.MinPartialCharge(mol),
                                      Descriptors.MolWt(mol),
                                      Descriptors.NumRadicalElectrons(mol),
                                      Descriptors.NumValenceElectrons(mol),
                                      EState.EState.MaxAbsEStateIndex(mol),
                                      EState.EState.MaxEStateIndex(mol),
                                      EState.EState.MinAbsEStateIndex(mol),
                                      EState.EState.MinEStateIndex(mol),
                                      EState.EState_VSA.EState_VSA1(mol),
                                      EState.EState_VSA.EState_VSA10(mol),
                                      EState.EState_VSA.EState_VSA11(mol),
                                      EState.EState_VSA.EState_VSA2(mol),
                                      EState.EState_VSA.EState_VSA3(mol),
                                      EState.EState_VSA.EState_VSA4(mol),
                                      EState.EState_VSA.EState_VSA5(mol),
                                      EState.EState_VSA.EState_VSA6(mol),
                                      EState.EState_VSA.EState_VSA7(mol),
                                      EState.EState_VSA.EState_VSA8(mol),
                                      EState.EState_VSA.EState_VSA9(mol),
                                      Fragments.fr_Al_COO(mol),
                                      Fragments.fr_Al_OH(mol),
                                      Fragments.fr_Al_OH_noTert(mol),
                                      Fragments.fr_aldehyde(mol),
                                      Fragments.fr_alkyl_carbamate(mol),
                                      Fragments.fr_alkyl_halide(mol),
                                      Fragments.fr_allylic_oxid(mol),
                                      Fragments.fr_amide(mol),
                                      Fragments.fr_amidine(mol),
                                      Fragments.fr_aniline(mol),
                                      Fragments.fr_Ar_COO(mol),
                                      Fragments.fr_Ar_N(mol),
                                      Fragments.fr_Ar_NH(mol),
                                      Fragments.fr_Ar_OH(mol),
                                      Fragments.fr_ArN(mol),
                                      Fragments.fr_aryl_methyl(mol),
                                      Fragments.fr_azide(mol),
                                      Fragments.fr_azo(mol),
                                      Fragments.fr_barbitur(mol),
                                      Fragments.fr_benzene(mol),
                                      Fragments.fr_benzodiazepine(mol),
                                      Fragments.fr_bicyclic(mol),
                                      Fragments.fr_C_O(mol),
                                      Fragments.fr_C_O_noCOO(mol),
                                      Fragments.fr_C_S(mol),
                                      Fragments.fr_COO(mol),
                                      Fragments.fr_COO2(mol),
                                      Fragments.fr_diazo(mol),
                                      Fragments.fr_dihydropyridine(mol),
                                      Fragments.fr_epoxide(mol),
                                      Fragments.fr_ester(mol),
                                      Fragments.fr_ether(mol),
                                      Fragments.fr_furan(mol),
                                      Fragments.fr_guanido(mol),
                                      Fragments.fr_halogen(mol),
                                      Fragments.fr_hdrzine(mol),
                                      Fragments.fr_hdrzone(mol),
                                      Fragments.fr_HOCCN(mol),
                                      Fragments.fr_imidazole(mol),
                                      Fragments.fr_imide(mol),
                                      Fragments.fr_Imine(mol),
                                      Fragments.fr_isocyan(mol),
                                      Fragments.fr_isothiocyan(mol),
                                      Fragments.fr_ketone(mol),
                                      Fragments.fr_ketone_Topliss(mol),
                                      Fragments.fr_lactam(mol),
                                      Fragments.fr_lactone(mol),
                                      Fragments.fr_methoxy(mol),
                                      Fragments.fr_morpholine(mol),
                                      Fragments.fr_N_O(mol),
                                      Fragments.fr_Ndealkylation1(mol),
                                      Fragments.fr_Ndealkylation2(mol),
                                      Fragments.fr_NH0(mol),
                                      Fragments.fr_NH1(mol),
                                      Fragments.fr_NH2(mol),
                                      Fragments.fr_Nhpyrrole(mol),
                                      Fragments.fr_nitrile(mol),
                                      Fragments.fr_nitro(mol),
                                      Fragments.fr_nitro_arom(mol),
                                      Fragments.fr_nitro_arom_nonortho(mol),
                                      Fragments.fr_nitroso(mol),
                                      Fragments.fr_oxazole(mol),
                                      Fragments.fr_oxime(mol),
                                      Fragments.fr_para_hydroxylation(mol),
                                      Fragments.fr_phenol(mol),
                                      Fragments.fr_phenol_noOrthoHbond(mol),
                                      Fragments.fr_phos_acid(mol),
                                      Fragments.fr_phos_ester(mol),
                                      Fragments.fr_piperdine(mol),
                                      Fragments.fr_piperzine(mol),
                                      Fragments.fr_priamide(mol),
                                      Fragments.fr_prisulfonamd(mol),
                                      Fragments.fr_pyridine(mol),
                                      Fragments.fr_quatN(mol),
                                      Fragments.fr_SH(mol),
                                      Fragments.fr_sulfide(mol),
                                      Fragments.fr_sulfonamd(mol),
                                      Fragments.fr_sulfone(mol),
                                      Fragments.fr_term_acetylene(mol),
                                      Fragments.fr_tetrazole(mol),
                                      Fragments.fr_thiazole(mol),
                                      Fragments.fr_thiocyan(mol),
                                      Fragments.fr_thiophene(mol),
                                      Fragments.fr_unbrch_alkane(mol),
                                      Fragments.fr_urea(mol),
                                      GraphDescriptors.BalabanJ(mol),
                                      GraphDescriptors.BertzCT(mol),
                                      GraphDescriptors.Chi0(mol),
                                      GraphDescriptors.Chi0n(mol),
                                      GraphDescriptors.Chi0v(mol),
                                      GraphDescriptors.Chi1(mol),
                                      GraphDescriptors.Chi1n(mol),
                                      GraphDescriptors.Chi1v(mol),
                                      GraphDescriptors.Chi2n(mol),
                                      GraphDescriptors.Chi2v(mol),
                                      GraphDescriptors.Chi3n(mol),
                                      GraphDescriptors.Chi3v(mol),
                                      GraphDescriptors.Chi4n(mol),
                                      GraphDescriptors.Chi4v(mol),
                                      GraphDescriptors.HallKierAlpha(mol),
                                      GraphDescriptors.Ipc(mol),
                                      GraphDescriptors.Kappa1(mol),
                                      GraphDescriptors.Kappa2(mol),
                                      GraphDescriptors.Kappa3(mol),
                                      Lipinski.HeavyAtomCount(mol),
                                      Lipinski.NHOHCount(mol),
                                      Lipinski.NOCount(mol),
                                      Lipinski.NumAliphaticCarbocycles(mol),
                                      Lipinski.NumAliphaticHeterocycles(mol),
                                      Lipinski.NumAliphaticRings(mol),
                                      Lipinski.NumAromaticCarbocycles(mol),
                                      Lipinski.NumAromaticHeterocycles(mol),
                                      Lipinski.NumAromaticRings(mol),
                                      Lipinski.NumHAcceptors(mol),
                                      Lipinski.NumHDonors(mol),
                                      Lipinski.NumHeteroatoms(mol),
                                      Lipinski.NumRotatableBonds(mol),
                                      Lipinski.NumSaturatedCarbocycles(mol),
                                      Lipinski.NumSaturatedHeterocycles(mol),
                                      Lipinski.NumSaturatedRings(mol),
                                      Lipinski.RingCount(mol),
                                      MolSurf.LabuteASA(mol),
                                      MolSurf.PEOE_VSA1(mol),
                                      MolSurf.PEOE_VSA10(mol),
                                      MolSurf.PEOE_VSA11(mol),
                                      MolSurf.PEOE_VSA12(mol),
                                      MolSurf.PEOE_VSA13(mol),
                                      MolSurf.PEOE_VSA14(mol),
                                      MolSurf.PEOE_VSA2(mol),
                                      MolSurf.PEOE_VSA3(mol),
                                      MolSurf.PEOE_VSA4(mol),
                                      MolSurf.PEOE_VSA5(mol),
                                      MolSurf.PEOE_VSA6(mol),
                                      MolSurf.PEOE_VSA7(mol),
                                      MolSurf.PEOE_VSA8(mol),
                                      MolSurf.PEOE_VSA9(mol),
                                      MolSurf.SlogP_VSA1(mol),
                                      MolSurf.SlogP_VSA10(mol),
                                      MolSurf.SlogP_VSA11(mol),
                                      MolSurf.SlogP_VSA12(mol),
                                      MolSurf.SlogP_VSA2(mol),
                                      MolSurf.SlogP_VSA3(mol),
                                      MolSurf.SlogP_VSA4(mol),
                                      MolSurf.SlogP_VSA5(mol),
                                      MolSurf.SlogP_VSA6(mol),
                                      MolSurf.SlogP_VSA7(mol),
                                      MolSurf.SlogP_VSA8(mol),
                                      MolSurf.SlogP_VSA9(mol),
                                      MolSurf.SMR_VSA1(mol),
                                      MolSurf.SMR_VSA10(mol),
                                      MolSurf.SMR_VSA2(mol),
                                      MolSurf.SMR_VSA3(mol),
                                      MolSurf.SMR_VSA4(mol),
                                      MolSurf.SMR_VSA5(mol),
                                      MolSurf.SMR_VSA6(mol),
                                      MolSurf.SMR_VSA7(mol),
                                      MolSurf.SMR_VSA8(mol),
                                      MolSurf.SMR_VSA9(mol),
                                      MolSurf.TPSA(mol)]))
    return descriptors
