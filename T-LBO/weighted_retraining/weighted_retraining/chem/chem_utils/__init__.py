""" Contains many chem utils codes """
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen, QED
import networkx as nx
from rdkit.Chem import rdmolops

# My imports
from weighted_retraining.weighted_retraining.chem.chem_utils.SA_Score import sascorer


# Make rdkit be quiet
def rdkit_quiet():
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_mol(smiles_or_mol):
    '''                                                                                                                                       
    Loads SMILES/molecule into RDKit's object                                   
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def standardize_smiles(smiles):
    """ Get standard smiles without stereo information """
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def penalized_logP(smiles: str, min_score=-float("inf")) -> float:
    """ calculate penalized logP for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    sa = SA(mol)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
            (logp - 2.45777691) / 1.43341767
            + (-sa + 3.05352042) / 0.83460587
            + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, min_score)


def SA(mol):
    return sascorer.calculateScore(mol)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def QED_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol)
