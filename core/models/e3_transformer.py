import torch.nn as nn
import torch
from torch_scatter import scatter, scatter_softmax
import e3nn
import e3nn.o3
from torch_geometric.nn import radius_graph, knn_graph
import numpy as np


def get_irrep_seq(ns, nv, use_second_order_repr = False, reduce_pseudoscalars = False):
    if use_second_order_repr:
        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
            f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {nv if reduce_pseudoscalars else ns}x0o'
        ]
    else:
        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o',
            f'{ns}x0e + {nv}x1o + {nv}x1e',
            f'{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o'
        ]
    return irrep_seq

def irrep_to_size(irrep):
    irreps = irrep.split(' + ')
    size = 0
    for ir in irreps:
        m, (l, p) = ir.split('x')
        size += int(m) * (2 * int(l) + 1)
    return size



# class AdaLN(nn.Module):
#     def __init__(self, hidden_size = 1):
#         super().__init__()
#         self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # Do not perform affine transformation
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 3)
#         )
 
#     def forward(self, h, c):
#         shift_c, scale_c, gate_c = self.adaLN_modulation(c[..., None]).chunk(3, dim=1)  # (n, 1) * 3
#         h = torch.nn.functional.sigmoid(gate_c) * (self.norm(h) * scale_c + shift_c)
#         return h



class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, d_pair, dropout=0.0, fc_factor=4, use_layer_norm=False,
                 init_para='normal',use_internal_weights=False, use_bias=False):
        super().__init__()
        self.in_irreps = in_irreps
        self.sh_irreps = sh_irreps
        self.out_irreps = out_irreps
        self.fc_factor = fc_factor
        self.use_bias = use_bias

        self.use_internal_weights = use_internal_weights
        if self.use_internal_weights:
            self.tp = e3nn.o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=True,
                                                          internal_weights=True)
        else:
            self.tp = e3nn.o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)
            self.fc = nn.Sequential(
                nn.LayerNorm(d_pair) if use_layer_norm else nn.Identity(),
                nn.Linear(d_pair, self.fc_factor * d_pair, bias=self.use_bias),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.fc_factor * d_pair, self.tp.weight_numel, bias=self.use_bias)
            )

            if init_para == 'normal':
                pass
            elif init_para == 'zero': 
                with torch.no_grad():
                    for para in self.fc.parameters():
                        para.fill_(0)
            else:
                raise ValueError(f'Unknown init {init_para}')

    def forward(self, node_attr_src, edge_sh, edge_attr=None):
        if self.use_internal_weights:
            out = self.tp(node_attr_src, edge_sh)
        else:
            if edge_attr is None:
                raise ValueError(f'No edge_attr for TensorProductConvLayer')
            out = self.tp(node_attr_src, edge_sh, self.fc(edge_attr))
        return out

class E3_transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
 
        self.d_node = cfg.get('d_node', 128)
        self.cutoff_mode = cfg.get('cutoff_mode', 'knn')
        self.cut_off = cfg.get('cut_off', 24)
        self.d_edge = cfg.get('d_edge', 32)
        self.edge_type_num =  cfg.get('edge_type_num', 4)
        self.d_rbf = cfg.get('d_rbf', 32)
        self.num_classes = cfg.get('num_classes', None)
        self.num_blocks = cfg.get('num_blocks', 6)
        self.use_bias = cfg.get('use_bias', False)
        
        self.d_l0 = cfg.get('d_l0', 16)
        self.d_lx = cfg.get('d_lx', 4)
        self.sh_lmax = cfg.get('sh_lmax', 2)
        self.use_second_order_repr = cfg.get('use_second_order_repr', False)  # False
        irrep_seq = get_irrep_seq(self.d_l0, self.d_lx, self.use_second_order_repr)
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax=self.sh_lmax)

        self.proj_input_l0 = nn.Linear(self.d_node, self.d_l0, bias=self.use_bias)
        self.trunk = nn.ModuleDict()
        for idx in range(self.num_blocks):
            irreps_input = irrep_seq[min(idx, len(irrep_seq) - 1)]
            irreps_out = irrep_seq[min(idx + 1, len(irrep_seq) - 1)]

            irreps_query = irreps_input
            irreps_key = irreps_input
            irreps_value = irreps_out

            self.trunk[f'proj_edge_{idx}'] = nn.Linear(self.edge_type_num + self.d_rbf, self.d_edge, bias=self.use_bias)

            self.trunk[f'h_q_{idx}'] = e3nn.o3.Linear(irreps_input, irreps_query, biases=self.use_bias)  # same as o3.FullyConnectedTensorProduct(irreps_input, "1x0e", irreps_query, shared_weights=False)
            self.trunk[f'tpc_k_{idx}'] = TensorProductConvLayer(irreps_input, self.irreps_sh, irreps_key, self.d_edge, use_bias=self.use_bias)
            self.trunk[f'tpc_v_{idx}'] = TensorProductConvLayer(irreps_input, self.irreps_sh, irreps_value, self.d_edge, use_bias=self.use_bias)

            self.trunk[f'dot_qk_{idx}'] = e3nn.o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "1x0e")
        
            self.trunk[f'proj_h_{idx}'] = e3nn.o3.Linear(irreps_value, '1x1o', biases=self.use_bias)
        self.proj_out_l0 = e3nn.o3.Linear(irreps_value, f'{self.d_node}x0e', biases=self.use_bias)

        if self.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.d_node, self.d_node, bias=self.use_bias),
                nn.SiLU(),
                nn.Linear(self.d_node, self.num_classes, bias=self.use_bias),
            )
        else:
            self.classifier = None

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.cut_off, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.cut_off, batch=batch, flow='source_to_target')
        # elif self.cutoff_mode == 'hybrid':
        #     edge_index = batch_hybrid_edge_connection(
        #         x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    def _build_edge_type(self, edge_index, mask_ligand, edge_type_num=4):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = torch.nn.functional.one_hot(edge_type, num_classes=edge_type_num)
        return edge_type

    def _build_edge_dist(self, edge_vec, D_min = 0., D_max = 10., D_count = 24):
        # Distance radial basis function
        edge_length = edge_vec.norm(dim=-1, keepdim=True)  # (edge, 1)
        D_mu = torch.linspace(D_min, D_max, D_count).to(edge_length.device)[None, ...]  # (1, D_count)
        D_sigma = (D_max - D_min) / D_count
        edge_dist = torch.exp(-((edge_length - D_mu) / D_sigma)**2)  # (edge, D_count)
        return edge_dist

    def forward(self, h, x, lig_flag, batch_idx, gen_flag=None, return_all=False, fix_x=False):
        if gen_flag is None:
            gen_flag = lig_flag

        h_l_hid = self.proj_input_l0(h)

        edge_index = self._connect_edge(x, lig_flag, batch_idx)

        for idx in range(self.num_blocks):


            # edge_index = self._connect_edge(x, lig_flag, batch_idx)


            src, dst = edge_index
            edge_vec = x[src] - x[dst]
            edge_type = self._build_edge_type(edge_index, lig_flag, edge_type_num = self.edge_type_num)
            edge_dist = self._build_edge_dist(edge_vec, D_count = self.d_rbf)
            edge_feat = torch.concat([edge_type, edge_dist], dim=-1)
            edge_feat = self.trunk[f'proj_edge_{idx}'](edge_feat)
            edge_sh = e3nn.o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')

            q = self.trunk[f'h_q_{idx}'](h_l_hid)[dst]
            k = self.trunk[f'tpc_k_{idx}'](h_l_hid[src], edge_sh, edge_feat)
            v = self.trunk[f'tpc_v_{idx}'](h_l_hid[src], edge_sh, edge_feat)

            qk_logits = self.trunk[f'dot_qk_{idx}'](q, k)/np.sqrt(k.shape[-1])  # compute the numerator (num_edges, 1)
            # qk_logits = self.trunk[f'dot_qk_{idx}'](q, k)  # compute the numerator (num_edges, 1)

            alpha = scatter_softmax(qk_logits, dst, dim=0)  # (num_edges, 1)

            h_l_hid = scatter(alpha * v, dst, dim=0, dim_size=x.shape[0], reduce="sum")  # (num_nodes, irreps_output)

            if not fix_x:
                x = x + self.trunk[f'proj_h_{idx}'](h_l_hid) * gen_flag.unsqueeze(-1)
        h = self.proj_out_l0(h_l_hid)

        if self.classifier is not None:
            c = self.classifier(h)
            outputs = {'x': x, 'h': h, 'c':c}
            return x, h, c
        else:
            outputs = {'x': x, 'h': h}
            return outputs