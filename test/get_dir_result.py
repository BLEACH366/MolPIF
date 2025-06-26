import os
import torch
import pandas as pd

from rdkit import Chem

if __name__ == "__main__":
    mol_path = '/data1/jinyaowei/MolCRAFT2/logs/interpolation_para_randn_flow_uni_fulltype_geo_0009_noEMA2_pf/test_outputs_best/20250515-094010/mol_sdf/TRAR_RHIRD_1_234_0/1l3l_A_rec_1l3l_lae_lig_tt_min_0'
    generated_path = '/data1/jinyaowei/MolCRAFT2/logs/interpolation_para_randn_flow_uni_fulltype_geo_0009_noEMA2_pf/test_outputs_best/20250515-094010/vina_docked.pt'

    result = torch.load(generated_path, map_location='cpu')
    metrics = {'file_index':[],
                'mol_size':[],
                'qed':[],
                'sa':[],
                'vina_score':[],
                'strain':[],}
    file_index = 0
    for res in result:
        if os.path.basename(res['ligand_filename']) == os.path.basename(mol_path) + '.sdf':
            print(file_index)
            metrics['file_index'] += [file_index]
            metrics['mol_size'] += [res['chem_results']['atom_num']]
            metrics['qed'] += [res['chem_results']['qed']]
            metrics['sa'] += [res['chem_results']['sa']]
            metrics['vina_score'] += [res['vina']['score_only'][0]['affinity']]
            metrics['strain'] += [res['pose_check']['strain']]

            file_index += 1

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(mol_path, 'metrics_select.csv'), index=False)