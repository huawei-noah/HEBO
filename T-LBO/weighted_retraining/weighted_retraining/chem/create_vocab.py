"""
Creates vocab file for a dataset in the form of a smiles file
"""
import argparse

from tqdm.auto import tqdm

from weighted_retraining.weighted_retraining.chem.chem_data import get_vocab_from_smiles
# My imports
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    help="file of SMILES to use to make vocabulary",
    required=True,
)
parser.add_argument(
    "-o", "--output_file", type=str, help="vocab file to output to", required=True
)

if __name__ == "__main__":

    args = parser.parse_args()

    # Control rdkit's incessant logging
    rdkit_quiet()

    # Read input file
    print("Reading input file")
    with open(args.input_file) as f:
        input_smiles = f.readlines()

    # Make the vocab set
    print("Making vocab")
    cset = set()
    for line in tqdm(input_smiles):
        smiles = line.split()[0]
        cset |= get_vocab_from_smiles(smiles)
    print(f"Found {len(cset)} vocab items")

    # Output to a file
    print("Writing vocab file")
    with open(args.output_file, "w") as f:
        f.write("\n".join([x for x in sorted(cset)]))
