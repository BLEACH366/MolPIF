import os
import json
import torch
import argparse

from core.config.config import Config, parse_config
from core.evaluation.metrics import CondMolGenMetric



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--generated_path", type=str)  # 

    parser.add_argument("--config_file", type=str, default="configs/default.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

    # train data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)    
    parser.add_argument("--pos_normalizer", type=float, default=2.0)    
    
    # train params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')  # '8.0' for

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=1.5)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf']) #vanilla or end_back

    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])

    # eval params
    parser.add_argument('--ckpt_path', type=str, default='best', help='path to the checkpoint')
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_dock", choices=['vina_score', 'vina_dock'])

    _args = parser.parse_args()
    if _args.ckpt_path.lstrip('./') == 'checkpoints/last.ckpt':
        _args.exp_name = 'official'
        _args.revision = 'default'
    else:
        print('trying to automatically parse experiment folder...')
        try:
            *_, exp_name, revision, _, ckpt_fn = _args.ckpt_path.split('/')
            _args.exp_name = exp_name
            _args.revision = revision
            print(f'change log dir to **/{exp_name}/{revision}')
        except Exception as e:
            pass

    cfg = Config(**_args.__dict__)

    
    if cfg.test_only:
        if os.path.exists(cfg.accounting.dump_config_path):
            # reload training config
            tr_cfg = Config(cfg.accounting.dump_config_path)
            tr_cfg.test_only = cfg.test_only
            tr_cfg.evaluation = cfg.evaluation
            tr_cfg.visual = cfg.visual
            tr_cfg.accounting = cfg.accounting
            # TODO: temporarily test different beta1 and sigma1_coord
            tr_cfg.dynamics.beta1 = cfg.dynamics.beta1
            tr_cfg.dynamics.sigma1_coord = cfg.dynamics.sigma1_coord
            tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
            tr_cfg.data = cfg.data
            tr_cfg.seed = cfg.seed
            tr_cfg.data.name = 'pl'
            cfg = tr_cfg
        if not hasattr(cfg.train, 'max_grad_norm'):
            cfg.train.max_grad_norm = 'Q'
    else:
        cfg.save2yaml(cfg.accounting.dump_config_path)

    metric = CondMolGenMetric(
        atom_decoder=cfg.data.atom_decoder,
        atom_enc_mode=cfg.data.transform.ligand_atom_mode,
        type_one_hot=False,
        single_bond=True,
        docking_config=cfg.evaluation.docking_config,
    )

    bad_case_dir = os.path.join(path, 'bad_cases_vina')
    os.makedirs(bad_case_dir, exist_ok=True)
    print(f'bad cases vina dumped to {bad_case_dir}')


    results = torch.load(os.path.join(_args.generated_path, f'generated.pt'), map_location='cpu')


    out_metrics = metric.evaluate(results, bad_case_dir)
    torch.save(results, os.path.join(path, f'vina_docked_new.pt'))
    out_metrics = {f'test/{k}': v for k, v in out_metrics.items()}

    out_metrics['test_outputs_dir'] = path
    print(json.dumps(out_metrics, indent=4))
    json.dump(out_metrics, open(os.path.join(path, 'metrics_new.json'), 'w'), indent=4)


