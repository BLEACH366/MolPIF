import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def keep_scaffold_and_groups(mol, scaffold, attachment_atoms):
    """
    保留小分子中的骨架和与指定原子相连的基团
    :param mol: 需要处理的小分子
    :param scaffold: 骨架分子
    :param attachment_atoms: 需要保留的基团连接的原子索引列表
    :return: 处理后的小分子
    """

    # scaffold_matches = mol.GetSubstructMatch(scaffold)
    # atoms_to_keep = set(scaffold_matches)

    # atoms_to_keep = [i for i in range(scaffold.GetNumAtoms())]
    # scaffold_atom_num = scaffold.GetNumAtoms()

    confA = mol.GetConformer()
    confB = scaffold.GetConformer()
    coordsA = [confA.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coordsB = [confB.GetAtomPosition(j) for j in range(scaffold.GetNumAtoms())]

    atoms_to_keep = []
    mol2scaffold = {}
    for i, posA in enumerate(coordsA):
        for j, posB in enumerate(coordsB):
            if (posA.x == posB.x) and (posA.y == posB.y) and (posA.z == posB.z):
                atoms_to_keep.append(i)
                mol2scaffold[i] = j
                break  # 一旦匹配，就不用继续比对 B 的其他原子
    scaffold_atom_num = scaffold.GetNumAtoms()

    # print('atoms_to_keep',atoms_to_keep)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.Kekulize(scaffold, clearAromaticFlags=True)

    while(len(attachment_atoms) != 0):
        attachment_atoms_new = set()
        atoms_to_keep_new = set([i for i in atoms_to_keep])
        for idx in attachment_atoms:
            # 查找与指定原子直接相连的原子，并将它们标记为保留
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                atoms_to_keep_new.add(neighbor.GetIdx())
                if neighbor.GetIdx() not in atoms_to_keep:
                    attachment_atoms_new.add(neighbor.GetIdx())
                    # print('attachment_atoms_new', attachment_atoms_new, idx)
        attachment_atoms = attachment_atoms_new
        atoms_to_keep = atoms_to_keep_new
    # print('atoms_to_keep',atoms_to_keep)
    
    # 调整键的类型
    emol = Chem.RWMol(mol)
    for begin_atom_idx in range(emol.GetNumAtoms()):
        for end_atom_idx in range(begin_atom_idx + 1, emol.GetNumAtoms()):

            if begin_atom_idx in mol2scaffold.keys() and end_atom_idx in mol2scaffold.keys():
                bond_old = scaffold.GetBondBetweenAtoms(mol2scaffold[begin_atom_idx], mol2scaffold[end_atom_idx])
                # bond.SetIsAromatic(bond_old.GetIsAromatic())
                if bond_old:
                    # print(bond_old.GetBondType())
                    bond = emol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                    if not bond:
                        emol.AddBond(begin_atom_idx, end_atom_idx, bond_old.GetBondType())
                        bond = emol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                        # print(bond_old.GetBondType(), bond.GetBondType())
                    else:
                        bond.SetBondType(bond_old.GetBondType())

    # Draw.MolToFile(emol, 'test2.png', size=(1000,1000))
    # writer = Chem.SDWriter('test2.sdf')
    # writer.write(emol)
    # writer.close()

    # 使用EditMol删除未标记的原子
    for atom_idx in reversed(range(mol.GetNumAtoms())):
        if atom_idx not in atoms_to_keep:
            emol.RemoveAtom(atom_idx)

    # Draw.MolToFile(emol, 'test3.png', size=(1000,1000))
    # writer = Chem.SDWriter('test3.sdf')
    # writer.write(emol)
    # writer.close()

    try:
        Chem.SanitizeMol(emol)
        return emol.GetMol()
    except:
        return None


def process_sdf_files(molecule_sdf, scaffold, attachment_atoms):
    """
    处理SDF文件,保留骨架和指定基团
    :param scaffold_sdf: 骨架SDF文件
    :param molecule_sdf: 需要处理的小分子SDF文件
    :param attachment_atoms: 基团连接的原子索引列表
    """

    molecule_supplier = Chem.SDMolSupplier(molecule_sdf)
    
    mol = molecule_supplier[0]
    mol = Chem.RemoveHs(mol)
    # 保留骨架和基团
    new_mol = keep_scaffold_and_groups(mol, scaffold, attachment_atoms)

    # img = Chem.Draw.MolToImage(new_mol)
    # img.show()

    # writer = Chem.SDWriter(output_sdf)
    # writer.write(new_mol)
    # writer.close()
    return new_mol

def modify_scaffold(gen_dir, scaffold_mol, attachment_atoms, min_add_num, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    scaffold_atom_num = scaffold_mol.GetNumAtoms()
    SMILES_list = []

    for file in os.listdir(gen_dir):
        if file.endswith('.sdf'):
            molecule_sdf = os.path.join(gen_dir, file)
            new_mol = process_sdf_files(molecule_sdf, scaffold_mol, attachment_atoms)
            if not new_mol:
                continue
            if new_mol.GetNumAtoms() >= scaffold_atom_num + min_add_num:
                print(molecule_sdf)
                SMILES = Chem.MolToSmiles(new_mol)
                if SMILES not in SMILES_list:
                    SMILES_list.append(SMILES)

                    # Draw.MolToFile(new_mol, os.path.join(output_dir, f"filter_{count}.png"))

                    try:
                        output_file = os.path.join(output_dir, f"filter_{count+1}.sdf")
                        with Chem.SDWriter(output_file) as f:
                            f.write(new_mol)
                        count += 1
                        print(count)
                    except:
                        continue
    print(f'Get {count} mols!')

if __name__ == '__main__':
    data_dir = "."
    gen_dir = 'output_test3'
    scaffold_sdf = '7rbt_scaffold.sdf'
    attachment_atoms = [2,10,21,30]
    min_add_num = 7
    output_dir = os.path.join(gen_dir, 'frag_part_filter')
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    scaffold = Chem.SDMolSupplier(scaffold_sdf)[0]
    scaffold = Chem.RemoveHs(scaffold)
    scaffold_atom_num = scaffold.GetNumAtoms()
    SMILES_list = []

    fdit = os.path.join(data_dir, gen_dir)
    for file in os.listdir(fdit):

        # if file != '97.sdf':
        #     continue

        if file.endswith('.sdf'):
            molecule_sdf = os.path.join(fdit, file)
            new_mol = process_sdf_files(molecule_sdf, scaffold_sdf, attachment_atoms)
            if not new_mol:
                continue
            if new_mol.GetNumAtoms() >= scaffold_atom_num + min_add_num:
                print(molecule_sdf)
                SMILES = Chem.MolToSmiles(new_mol)
                if SMILES not in SMILES_list:
                    SMILES_list.append(SMILES)

                    # Draw.MolToFile(new_mol, os.path.join(output_dir, f"filter_{count}.png"))

                    try:
                        output_file = os.path.join(output_dir, f"filter_{count+1}.sdf")
                        with Chem.SDWriter(output_file) as f:
                            f.write(new_mol)
                        count += 1
                        print(count)
                    except:
                        continue
    print(f'Get {count} mols!')