from absl import logging

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config.config import Struct
from core.models.common import compose_context, ShiftedSoftplus
from core.models.bfn_base import BFNBase
from core.models.uni_transformer import UniTransformerO2TwoUpdateGeneral
# from core.models.e3_transformer import E3_transformer

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RBF(nn.Module):
    def __init__(self, start, end, n_center):
        super().__init__()
        self.start = start
        self.end = end
        self.n_center = n_center
        self.centers = torch.linspace(start, end, n_center)
        self.width = (end - start) / n_center

    def forward(self, x):
        assert x.ndim >= 2
        out = (x - self.centers.to(x.device)) / self.width
        ret = torch.exp(-0.5 * out**2)
        return F.normalize(ret, dim=-1, p=1) * 2 - 1


class TimeEmbedLayer(nn.Module):
    def __init__(self, time_emb_mode, time_emb_dim):
        super().__init__()
        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim

        if self.time_emb_mode == "simple":
            assert self.time_emb_dim == 1
            self.time_emb = lambda x: x
        elif self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        elif self.time_emb_mode == "rbf":
            self.time_emb = RBF(0, 1, self.time_emb_dim)
        elif self.time_emb_mode == "rbfnn":
            self.time_emb = nn.Sequential(
                RBF(0, 1, self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t):
        return self.time_emb(t)


class PIF4SBDDScoreModel(BFNBase):
    def __init__(
        self,
        net_config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        device="cuda",
        condition_time=True,
        sigma1_coord=0.02,
        beta1=3.0,
        use_discrete_t=False,
        discrete_steps=1000,
        t_min=0.0001,
        node_indicator=True,
        time_emb_mode='simple',
        time_emb_dim=1,
        center_pos_mode='protein',
        pos_init_mode='zero',
        destination_prediction = False,
        sampling_strategy = "vanilla",
        use_random_mask = True,
        pm = 0.3,  # specify the probability of mask
        pam = 0.3,  # specify the probability of atom mask
    ):
        super().__init__()
        net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
        # elif net_config.name == 'e3_transformer':
        #     self.unio2net = E3_transformer({})
        else:
            raise NotImplementedError
        
        self.use_random_mask = use_random_mask
        self.pm = pm
        self.pam = pam

        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]


        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy

    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        return_all=False,  # legacy from targetdiff
        fix_x=False,
        gen_flag_lig=None,
    ):
        """
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [node_num x batch_size, protein_atom_feature_dim] := [N_protein, 27]
            batch_protein: [node_num x batch_size] := [N_protein]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        K = self.num_classes  # ligand_atom_feature_dim

        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]

        # ---------for targetdiff-----------
        batch_size = batch_protein.max().item() + 1
        # init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        init_ligand_v = theta_h_t
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]
        # init_ligand_h = input_ligand_feat # TODO: no embedding for ligand atoms, check whether this make sense.

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim ]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]


        if gen_flag_lig is None:
            h_all, pos_all, batch_all, mask_ligand = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=mu_pos_t,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
            )

            outputs = self.unio2net(
                h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
            )
        else:
            h_all, pos_all, batch_all, mask_ligand, mask_gen = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=mu_pos_t,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                gen_flag_lig=gen_flag_lig,
            )
            outputs = self.unio2net(
                h_all, pos_all, mask_gen, batch_all, return_all=return_all, fix_x=fix_x
            )

        final_pos, final_h = (
            outputs["x"],
            outputs["h"],
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]

        if not self.destination_prediction:  # True for self.destination_prediction
            raise ValueError(f'not implement for no destination_prediction!')
        else:
            coord_pred = final_ligand_pos #add destination prediction. 

        p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)  # [N_ligand, 13]

        return coord_pred, p0_h

    def reconstruction_loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        return self.loss_one_step(
            t, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand
        )

    def loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):
        K = self.num_classes

        assert ligand_v.max().item() < K, f"Error: {ligand_v.max().item()} >= {K}"
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]


        mu_coord = self.continuous_var_interpolation_update(
            t, x=ligand_pos, sigma1=self.sigma1_coord
        )  # [N, 3], [N, 1], gamma_coord is not used
        theta = self.discrete_var_interpolation_update(
            t, x=ligand_v, K=K, sigma1=self.sigma1_coord
        )  # [N, K]


        gen_flag_lig = torch.ones([batch_ligand.size(0)], device=batch_ligand.device, dtype=torch.float32) # float, 1.0 for generate
        use_random_mask = self.use_random_mask
        pm = self.pm
        pam = self.pam
        if use_random_mask:
            if torch.rand(1) >= pm:
                pass
            else:
                mask = torch.rand_like(gen_flag_lig) < pam
                gen_flag_lig[mask] = 0.0
        
        mu_coord = mu_coord * gen_flag_lig[...,None] + ligand_pos * (1.0 - gen_flag_lig[...,None])
        theta = theta * gen_flag_lig[...,None] + ligand_v * (1.0 - gen_flag_lig[...,None])


        coord_pred, p0_h = self.interdependency_modeling(
            time=t,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gen_flag_lig=gen_flag_lig,
        )  # [N, 3], [N, K], [?]


        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:  # True for self.use_discrete_t
            closs = self.ctime4continuous_loss(
                t=t,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]
        else:
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            # closs = self.dtime4continuous_loss(
            #     i=i,
            #     N=self.discrete_steps,
            #     sigma1=self.sigma1_coord,
            #     x_pred=coord_pred,
            #     x=ligand_pos,
            #     segment_ids=batch_ligand,
            # )

            # dloss = self.dtime4discrete_loss_prob(
            #     i=i,
            #     N=self.discrete_steps,
            #     beta1=self.beta1,
            #     one_hot_x=ligand_v,
            #     p_0=p0_h,
            #     K=K,
            #     segment_ids=batch_ligand,
            # )

            closs = self.dtime4continuous_interpolation_loss(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )

            dloss = self.dtime4discrete_interpolation_loss_prob(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                p_0=p0_h,
                one_hot_x=ligand_v,
                K=K,
                segment_ids=batch_ligand,
            )

        discretized_loss = torch.zeros_like(closs)  # not used

        return closs, dloss, discretized_loss

    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        n_nodes,  # B
        sample_steps=1000,
        desc='',
        ligand_pos=None,
        ligand_v=None,
        gen_flag_lig=None,
    ):
        """
        The function implements the sampling procedure
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """

        K = self.num_classes

        # TODO test
        if self.pos_init_mode == 'zero':
            mu_pos_t = torch.zeros((n_nodes, 3)).to(
                self.device
            )  # [N, 3] coordinates prior N(0, 1)
        elif self.pos_init_mode == 'randn':
            mu_pos_t = torch.randn((n_nodes, 3)).to(self.device)


        a_dirichlet = torch.ones((n_nodes, K)) / K
        dirichlet_dist = torch.distributions.Dirichlet(a_dirichlet)
        theta_h_t = dirichlet_dist.sample().to(self.device)


        theta_traj = []

        # TODO: debug
        mu_pos_t = mu_pos_t[batch_ligand]
        theta_h_t = theta_h_t[batch_ligand]


        if gen_flag_lig is not None:
            ligand_v_onehot = F.one_hot(ligand_v, K).float()
            
            # mu_pos_t += torch.mean(ligand_pos * gen_flag_lig[...,None], dim=0)
            # mu_pos_t = mu_pos_t * gen_flag_lig[...,None] + ligand_pos * (1.0 - gen_flag_lig[...,None])
            # theta_h_t = theta_h_t * gen_flag_lig[...,None] + ligand_v_onehot * (1.0 - gen_flag_lig[...,None])

            mu_pos_t = ligand_pos
            theta_h_t = ligand_v_onehot



        for i in trange(1, sample_steps + 1, desc=f'{desc}'):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            t = t[batch_ligand]

            coord_pred, p0_h_pred = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_t,
                mu_pos_t=mu_pos_t, 
                gen_flag_lig=gen_flag_lig,
            )

            # maintain theta_traj
            theta_traj.append((mu_pos_t, theta_h_t))
            # TODO delete the following condition
            if not torch.all(p0_h_pred.isfinite()):
                p0_h_pred = torch.where(
                    p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred)
                )
                logging.warn("p0_h_pred is not finite")
            p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)


            if "end_back" in self.sampling_strategy:  # self.sampling_strategy is end_back_pmf
                t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                t = t[batch_ligand]
                if self.sampling_strategy == "end_back":
                    # theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=sample_pred, K=K)
                    raise NotImplementedError(f"sampling strategy end_back not implemented")
                elif self.sampling_strategy == "end_back_pmf":

                    # theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)
                    theta_h_t = self.discrete_var_interpolation_update(t, x=p0_h_pred, K=K, sigma1=self.sigma1_coord)


                elif self.sampling_strategy == "end_back_mode":
                    p0_mode = torch.argmax(p0_h_pred, dim=-1)
                    mode_pred = F.one_hot(p0_mode, num_classes=K).float()
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=mode_pred, K=K)
                else:
                    raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")

                # mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                mu_pos_t = self.continuous_var_interpolation_update(t, x=coord_pred, sigma1=self.sigma1_coord)


                if gen_flag_lig is not None:
                    mu_pos_t = mu_pos_t * gen_flag_lig[...,None] + ligand_pos * (1.0 - gen_flag_lig[...,None])
                    theta_h_t = theta_h_t * gen_flag_lig[...,None] + ligand_v_onehot * (1.0 - gen_flag_lig[...,None])

            else:
                raise NotImplementedError

        mu_pos_final, p0_h_final = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_t,
            mu_pos_t=mu_pos_t,
            gen_flag_lig=gen_flag_lig,
        )

        if gen_flag_lig is not None:
            mu_pos_final = mu_pos_final * gen_flag_lig[...,None] + ligand_pos * (1.0 - gen_flag_lig[...,None])
            p0_h_final = p0_h_final * gen_flag_lig[...,None] + ligand_v_onehot * (1.0 - gen_flag_lig[...,None])


        # TODO delete the following condition
        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(
                p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final)
            )
            logging.warn("p0_h_pred is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)

        theta_traj.append((mu_pos_final, p0_h_final))


        return theta_traj
