import os
import re
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

file_dir = "."

for file in os.listdir(file_dir):
    if file.endswith(".sdf"):
    # if re.match('7L1V_fragment2', file):
        print(file)
        file_f = os.path.join(file_dir, file)
        mol = Chem.SDMolSupplier(file_f)[0]
        smiles = Chem.MolToSmiles(mol)
        print(smiles)
        mol.RemoveAllConformers()
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('molAtomMapNumber', str(i))
        Draw.MolToImage(mol, size=(1000,1000))
        
        atoms_index = list(range(len(list(mol.GetAtoms()))))
        atoms_index_str = ""
        for i in atoms_index:
            atoms_index_str += str(i) + " "
        print('atoms_index_str=')
        print(atoms_index_str)
        Draw.MolToFile(mol,os.path.join(file_dir,file.replace(".sdf",".png")))

