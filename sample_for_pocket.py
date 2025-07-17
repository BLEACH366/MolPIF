import argparse
import os
import json
import csv

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

# import datetime, pytz

from core.config.config import Config, parse_config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import NormalizerCallback, EMACallback
from core.callbacks.validation_callback_for_sample import (
    DockingTestCallback,
)

import core.utils.transforms as trans
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict
from core.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

from core.evaluation.utils import scoring_func
from core.evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np
from rdkit import Chem
from collections import defaultdict
import pandas as pd


def get_dataloader_from_pdb(cfg, fix_index=None):
    assert cfg.evaluation.protein_path is not None and cfg.evaluation.ligand_path is not None
    protein_fn, ligand_fn = cfg.evaluation.protein_path, cfg.evaluation.ligand_path

    # load protein and ligand
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)  # remove H here!
    lig_pos = ligand_dict["pos"]

    print('[DEBUG] get_dataloader')
    print(lig_pos.shape, lig_pos.mean(axis=0))

    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = pocket.to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn

    if fix_index:
        data.fix_index = fix_index

    # transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
    ]
    transform = Compose(transform_list)
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
            f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")

    # dataloader
    collate_exclude_keys = ["ligand_nbh_list"]
    test_set = [transform(data)] * cfg.evaluation.num_samples

    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )

    cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(protein_fn))
    print(f"protein root: {cfg.evaluation.docking_config.protein_root}")

    return test_loader


def call(protein_fn, ligand_fn, ckpt_path='./checkpoints/pretrained.ckpt',
         num_samples=10, sample_steps=100, sample_num_atoms='ref', 
        sampling_strategy='end_back_pmf', seed=1234,
         fix_index=None, out_fn='output', cfg_path=None):
    
    if cfg_path:
        cfg = Config(cfg_path)
    else: 
        cfg = Config(os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'config.yaml'))
    seed_everything(cfg.seed)
    
    cfg.evaluation.protein_path = protein_fn
    cfg.evaluation.ligand_path = ligand_fn
    cfg.evaluation.ckpt_path = ckpt_path
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.num_samples = num_samples
    cfg.evaluation.sample_steps = sample_steps
    cfg.evaluation.sample_num_atoms = sample_num_atoms # or 'prior'
    cfg.dynamics.sampling_strategy = sampling_strategy
    cfg.seed = seed
    cfg.train.max_grad_norm = 'Q'
    cfg.accounting.test_outputs_dir = out_fn

    print(f"The config of this process is:\n{cfg}")

    print(protein_fn, ligand_fn)
    test_loader = get_dataloader_from_pdb(cfg, fix_index=fix_index)
    # wandb_logger.log_hyperparams(cfg.todict())

    model = SBDDTrainLoop(config=cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        num_sanity_val_steps=0,
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
        ],
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)


class Metrics:
    def __init__(self, protein_fn, ref_ligand_fn, ligand_fn):
        self.protein_fn = protein_fn
        self.ref_ligand_fn = ref_ligand_fn
        self.ligand_fn = ligand_fn
        self.exhaustiveness = 16

    def vina_dock(self, mol):
        chem_results = {}

        try:
            # qed, logp, sa, lipinski, ring size, etc
            chem_results.update(scoring_func.get_chem(mol))
            chem_results['atom_num'] = mol.GetNumAtoms()

            # docking                
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename=self.ref_ligand_fn, protein_root='./', protein_path=self.protein_fn)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.exhaustiveness)
            docking_results = vina_task.run(mode='dock', exhaustiveness=self.exhaustiveness)

            chem_results['vina_score'] = score_only_results[0]['affinity']
            chem_results['vina_minimize'] = minimize_results[0]['affinity']
            chem_results['vina_dock'] = docking_results[0]['affinity']
            # chem_results['vina_dock_pose'] = docking_results[0]['pose']
            return chem_results
        except Exception as e:
            print(e)
        
        return chem_results

    def pose_check(self, mol):
        pc = PoseCheck()

        pose_check_results = {}

        protein_ready = False
        try:
            pc.load_protein_from_pdb(self.protein_fn)
            protein_ready = True
        except ValueError as e:
            return pose_check_results

        ligand_ready = False
        try:
            pc.load_ligands_from_mols([mol])
            ligand_ready = True
        except ValueError as e:
            return pose_check_results

        if ligand_ready:
            try:
                strain = pc.calculate_strain_energy()[0]
                pose_check_results['strain'] = strain
            except Exception as e:
                pass

        if protein_ready and ligand_ready:
            try:
                clash = pc.calculate_clashes()[0]
                pose_check_results['clash'] = clash
            except Exception as e:
                pass

            try:
                df = pc.calculate_interactions()
                columns = np.array([column[2] for column in df.columns])
                flags = np.array([df[column][0] for column in df.columns])
                
                def count_inter(inter_type):
                    if len(columns) == 0:
                        return 0
                    count = sum((columns == inter_type) & flags)
                    return count

                # ['Hydrophobic', 'HBDonor', 'VdWContact', 'HBAcceptor']
                hb_donor = count_inter('HBDonor')
                hb_acceptor = count_inter('HBAcceptor')
                vdw = count_inter('VdWContact')
                hydrophobic = count_inter('Hydrophobic')

                pose_check_results['hb_donor'] = hb_donor
                pose_check_results['hb_acceptor'] = hb_acceptor
                pose_check_results['vdw'] = vdw
                pose_check_results['hydrophobic'] = hydrophobic
            except Exception as e:
                pass

        for k, v in pose_check_results.items():
            mol.SetProp(k, str(v))

        return pose_check_results
    
    def evaluate(self):
        chem_results_total = defaultdict(list)
        dicts = []
        for sdf_file in os.listdir(self.ligand_fn):
            if not sdf_file.endswith('.sdf'):
                continue
            mol = Chem.SDMolSupplier(os.path.join(self.ligand_fn,sdf_file), removeHs=False)[0]
        
            chem_results = self.vina_dock(mol)
            pose_check_results = self.pose_check(mol)
            chem_results.update(pose_check_results)
            chem_results['filename'] = sdf_file
            dicts.append(chem_results)

        for d in dicts:
            for key, value in d.items():
                chem_results_total[key].append(value)
        return chem_results_total


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--protein_path", type=str, default="./examples/1umd/1umd_B_rec.pdb")
    parser.add_argument("--ligand_path", type=str, default="./examples/1umd/1umd_B_rec_1umb_tdp_lig_tt_docked_1.sdf")
    parser.add_argument("--ckpt_path", type=str, default="./logs/interpolation_para_randn_flow_uni_fulltype_geo_0009_noEMA2_pf/checkpoints/epoch12-val_loss12.04-mol_stable0.98-complete0.95-vina_score-7.27.ckpt")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--sample_num_atoms", type=str, default="ref")  # "ref", "prior"
    parser.add_argument("--fix_index", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])
    parser.add_argument("--out_fn", type=str, default="./examples/1umd/output_ref")
    parser.add_argument("--cfg_path", type=str, default=None)

    args = parser.parse_args()

    protein_path = args.protein_path
    ligand_path = args.ligand_path
    ckpt_path = args.ckpt_path
    num_samples = args.num_samples
    sample_steps = args.sample_steps
    sample_num_atoms = args.sample_num_atoms
    fix_index = args.fix_index
    out_fn = args.out_fn
    cfg_path = args.cfg_path


    call(protein_path, ligand_path, ckpt_path=ckpt_path, num_samples=num_samples, sample_steps=sample_steps,
         sample_num_atoms=sample_num_atoms, fix_index=fix_index, out_fn=out_fn, cfg_path=cfg_path)


    metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()

    num_rows = len(next(iter(metrics.values())))  # 任意一列的长度
    # 获取字段名（列名）
    fieldnames = list(metrics.keys())
    # 组合为一行一行的数据（按行取出）
    rows = [ {key: metrics[key][i] for key in fieldnames} for i in range(num_rows) ]

    # 写入 CSV
    with open(os.path.join(out_fn, 'metrics.csv'), mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

