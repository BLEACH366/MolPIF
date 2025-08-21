from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import ScalarFormatter

# 🧪生成指纹
def mols_to_fp(mols, method, radius=2, n_bits=1024):
    fps = []
    for m in mols:
        if method == 'Morgan':
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(bv, arr)
        elif method == 'RDKit':
            bv = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits).GetFingerprint(m)
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(bv, arr)
        elif method == 'USR-CAT':
            bv_list = rdMolDescriptors.GetUSRCAT(m)
            arr = np.array(bv_list, dtype=float)
        else:
            raise ValueError(method)
        fps.append(arr)
    return np.array(fps)

def compute_npr(mol):

    # 获取原子坐标和质量
    atom_coords = []
    atom_masses = []
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        mass = atom.GetMass()
        atom_coords.append([pos.x, pos.y, pos.z])
        atom_masses.append(mass)

    # 转换为 NumPy 数组
    atom_coords = np.array(atom_coords)
    atom_masses = np.array(atom_masses)

    # 计算质心
    total_mass = np.sum(atom_masses)
    center_of_mass = np.sum(atom_coords.T * atom_masses, axis=1) / total_mass

    # 计算惯性张量
    inertia_tensor = np.zeros((3, 3))
    for i in range(len(atom_masses)):
        r = atom_coords[i] - center_of_mass
        inertia_tensor += atom_masses[i] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    # 计算主惯性矩
    eigenvalues, _ = np.linalg.eigh(inertia_tensor)
    eigenvalues.sort()
    
    I1, I2, I3 = eigenvalues

    npr1 = I1 / I3
    npr2 = I2 / I3
    return npr1, npr2


def compute_pbf(mol):
    return rdMolDescriptors.CalcPBF(mol)


# 🎯JointGrid + KDE + scatter helper
def plot_tsne_joint(df, title, palette):
    g = sns.JointGrid(data=df, x='TSNE1', y='TSNE2', hue='label',
                      height=5, ratio=5, space=0.1, palette=palette)
    g.plot_joint(sns.scatterplot, s=20, alpha=0.6)
    sns.kdeplot(data=df, x='TSNE1', fill=True, ax=g.ax_marg_x, alpha=0.3, color='grey')
    sns.kdeplot(data=df, y='TSNE2', fill=True, ax=g.ax_marg_y, alpha=0.3, color='grey')
    g.ax_joint.set_title(title)
    g.ax_joint.legend(title='')
    return g



# === 主流程 ===
# 示例 molecules 和标签

# samples = torch.load('./MolPIF_pose_checked.pt',map_location='cpu')
# test = torch.load('./crossdocked_test_vina_docked.pt',map_location='cpu')
# mols = [i['mol'] for i in samples] + [i['mol'] for i in test]
# labels = ['MolPIF'] * len(samples) + ['Test'] * len(test)


samples1 = torch.load('./MolPIF_pose_checked.pt',map_location='cpu')
samples2 = torch.load('./molcraft_vina_docked_pose_checked.pt',map_location='cpu')
samples3 = torch.load('./targetdiff_vina_docked_pose_checked.pt',map_location='cpu')
test = torch.load('./crossdocked_test_vina_docked.pt',map_location='cpu')
mols = [i['mol'] for i in samples1] + [i['mol'] for i in samples2] +[i['mol'] for i in samples3] +[i['mol'] for i in test]
labels = ['MolPIF'] * len(samples1) +['MolCRAFT'] * len(samples2) +['TargetDiff'] * len(samples3) + ['Test'] * len(test)



# 处理三种指纹
fps_methods = ['Morgan','RDKit','USR-CAT']
recs = []
for method in fps_methods:
    fps = mols_to_fp(mols, method)
    tsne = TSNE(n_components=2,
                metric='jaccard' if method != 'USR-CAT' else 'euclidean',
                perplexity=30, random_state=0)
    coords = tsne.fit_transform(fps)
    for (x,y), lab in zip(coords, labels):
        recs.append({'fp': method, 'TSNE1': x, 'TSNE2': y, 'label': lab})
df = pd.DataFrame(recs)


# NPR and PBF
npr1_list, npr2_list, pbf_list, label_list = [], [], [], []
for m, label in zip(mols, labels):
    npr1, npr2 = compute_npr(m)
    pbf = compute_pbf(m)
    npr1_list.append(npr1)
    npr2_list.append(npr2)
    pbf_list.append(pbf)
    label_list.append(label)

shape_df = pd.DataFrame({
    'NPR1': npr1_list,
    'NPR2': npr2_list,
    'PBF': pbf_list,
    'Label': label_list
}).dropna()


# 绘图
sns.set(style='whitegrid')

fig, axes = plt.subplots(2, 3, figsize=(18,10))
for ax, method in zip(axes[0], fps_methods):
    subdf = df[df['fp'] == method]
    # 在同一个坐标轴上叠加 KDE 密度
    sns.kdeplot(
        data=subdf[subdf['label'] == 'MolPIF'], x='TSNE1', y='TSNE2', 
        fill=True, levels=20, cmap="Spectral", thresh=0.05, cbar=True, ax=ax
    )
    # 散点图
    sns.scatterplot(
        data=subdf[subdf['label'] == 'Test'], x='TSNE1', y='TSNE2', 
        color='black', alpha=0.6, ax=ax, label='Test'
    )
    ax.set_title(method)

for ax, group in zip(axes[1,:2], ['MolPIF', 'Test']):
    subdf = shape_df[shape_df['Label'] == group]
    sns.kdeplot(
        data=subdf, x='NPR1', y='NPR2',
        fill=True, cmap='Blues', levels=10, ax=ax, cbar=True
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.45, 1.05)
    ax.set_title(f"{group}", fontsize=12)
    ax.set_xlabel("NPR1")
    ax.set_ylabel("NPR2")
    ax.text(0.01, 0.95, "Rod", transform=ax.transAxes)
    ax.text(0.84, 0.95, "Sphere", transform=ax.transAxes)
    ax.text(0.45, 0.04, "Disc", transform=ax.transAxes)
    # 添加三角形边界
    ax.plot([0, 1], [1, 1], 'k--')
    ax.plot([0, 0.5], [1, 0.5], 'k--')
    ax.plot([0.5, 1], [0.5, 1], 'k--')


ax = axes[1,2]
sns.boxplot(data=shape_df, x='Label', y='PBF', ax=ax)
ax.set_xlabel('Sample labels', fontsize=12, weight='bold')
ax.set_ylabel('PBF', fontsize=12)


fig.tight_layout()
plt.show()

plt.savefig("result.png")