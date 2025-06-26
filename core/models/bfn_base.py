import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torchdiffeq import odeint
from torch_scatter import scatter_mean, scatter_sum
import torch.distributions as dist

import numpy as np


LOG2PI = np.log(2 * np.pi)

def dirichlet_kl_batch(alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
    """
    计算批量 Dirichlet 分布 KL 散度。
    alpha1, alpha2: 张量形状 [B, K]，表示 B 个样本，每个样本有 K 个浓度参数。
    返回: 形状 [B] 的 KL 散度。
    """
    # 1) 计算每个批次的浓度参数和，形状 [B]
    sum1 = alpha1.sum(dim=1)       # 
    sum2 = alpha2.sum(dim=1)       # 

    # 2) ln Γ(∑α1) - ln Γ(∑α2)，形状 [B]
    term1 = torch.lgamma(sum1) - torch.lgamma(sum2)

    # 3) ∑ [ln Γ(α2_i) - ln Γ(α1_i)]，形状 [B]
    term2 = torch.lgamma(alpha2).sum(dim=1) - torch.lgamma(alpha1).sum(dim=1)

    # 4) ∑ (α1_i - α2_i) [ψ(α1_i) - ψ(∑α1)]，形状 [B]
    term3 = ((alpha1 - alpha2) *
            (torch.digamma(alpha1) - torch.digamma(sum1).unsqueeze(1))
    ).sum(dim=1)

    return term1 + term2 + term3

class BFNBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(BFNBase, self).__init__(*args, **kwargs)

    # def zero_center_of_mass(self, x_pos, segment_ids):
    #     size = x_pos.size()
    #     assert len(size) == 2  # TODO check this
    #     seg_means = scatter_mean(x_pos, segment_ids, dim=0)
    #     mean_for_each_segment = seg_means.index_select(0, segment_ids)
    #     x = x_pos - mean_for_each_segment

    #     return x

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_l

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)

        return f_




    def continuous_var_interpolation_update(self, t, x, sigma1, prior=None):
        # """
        # x: [N, D]
        # """
        if prior is None:
            prior = [torch.zeros_like(x).to(x.device), torch.ones_like(x).to(x.device)]

        gamma = 1 - torch.pow(sigma1, t)  # [0,1]
        # gamma = t  # [0,1]
        
        x_flow = [gamma * x, (1 - gamma) * prior[1]]  # flow


        mu = x_flow[0] + torch.sqrt(x_flow[1]) * torch.randn_like(x).to(x.device)  # std = sqrt(1 - gamma**2)

        # laplace_dist = torch.distributions.Laplace(x_flow[0], x_flow[1])  # laplace prior
        # mu = laplace_dist.sample().to(self.device)

        return mu



    def discrete_var_interpolation_update(self, t, x, K, sigma1, prior=None):
        # """
        # x: [N, K]
        # """
        if prior is None:
            # prior = torch.ones_like(x).to(x.device)  
            prior = torch.ones_like(x).to(x.device) / K  

        gamma = 1 - torch.pow(sigma1, t)   # (B,)
        # gamma = t  # [0,1]

        x_flow = gamma * x + (1 - gamma) * prior

        dirichlet_dist = torch.distributions.Dirichlet(x_flow)
        theta = dirichlet_dist.sample().to(x.device)

        return theta




    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        x: [N, K]
        """
        # Eq.(182): β(t) = t**2 β(1)
        beta = beta1 * (t**2)  # (B,)

        # Eq.(185): p_F(θ|x;t) = E_{N(y | β(t)(Ke_x−1), β(t)KI)} δ (θ − softmax(y))
        # can be sampled by first drawing y ~ N(y | β(t)(Ke_x−1), β(t)KI)
        # then setting θ = softmax(y)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def discreteised_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        Note, this is identical to the continuous_var_bayesian_update
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def ctime4continuous_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        # Eq.(101): L∞(x) = −ln(σ1) * E_{t∼U (0,1), p_F(θ|x;t)} [|x − x_hat(θ,t)|**2 / (σ_1**2)**t]
        if segment_ids is not None:
            loss = scatter_mean(
                torch.pow(sigma1, -2 * t.view(-1))
                * ((x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)),
                segment_ids,
                dim=0,
            )
        else:
            loss = torch.pow(sigma1, -2 * t.view(-1)) * (x_pred - x).view(
                x.shape[0], -1
            ).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss



    def dtime4continuous_interpolation_loss(self, i, N, sigma1, x_pred, x, segment_ids=None):
        t = (i - 1 + 1)/ N
        gamma = 1 - torch.pow(sigma1, t)
        # gamma = t
        
        # weight = gamma ** 2 / (2 * (1 - gamma))
        weight = np.sqrt(N) * gamma ** 2 / (2 * (1 - gamma))

        if segment_ids is not None:
            loss = scatter_mean(
                weight.view(-1) * ((x_pred - x) ** 2).sum(-1), segment_ids, dim=0
            )


        # b = 1 - gamma  # laplace prior
        # if segment_ids is not None:
        #     loss = scatter_mean(
        #         np.sqrt(N) * (torch.exp(-torch.abs(x_pred - x)/b) + torch.abs(x_pred - x)/b).sum(-1), segment_ids, dim=0
        #     )

        return loss

    def dtime4discrete_interpolation_loss_prob(
        self, i, N, sigma1, p_0, one_hot_x, K, segment_ids=None
    ):
        t = (i - 1 + 1)/ N
        gamma = 1 - torch.pow(sigma1, t)
        # gamma = t
        
        alpha1 = gamma * p_0  + (1 - gamma) * 1/K
        alpha2 = gamma * one_hot_x  + (1 - gamma) * 1/K

        if segment_ids is not None:
            loss = scatter_mean(
                dirichlet_kl_batch(alpha1, alpha2), segment_ids, dim=0
            )
        return loss


    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None):
        # TODO not debuged yet
        weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        # print(x_pred.shape, x.shape , i.shape,weight.shape)
        # print(segment_ids)
        if segment_ids is not None:
            loss = scatter_mean(
                weight.view(-1) * ((x_pred - x) ** 2).sum(-1), segment_ids, dim=0
            )
        else:
            loss = (
                N
                * (1 - torch.pow(sigma1, 2 / N))
                / (2 * torch.pow(sigma1, 2 * i / N))
                * (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
            )

        # print(loss.shape)
        return loss

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K, segment_ids=None):
        # Eq.(205): L∞(x) = Kβ(1) E_{t∼U (0,1), p_F (θ|x,t)} [t|e_x − e_hat(θ, t)|**2,
        # where e_hat(θ, t) = (\sum_k p_O^(1) (k | θ; t)e_k, ..., \sum_k p_O^(D) (k | θ; t)e_k)
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        assert e_x.size() == e_hat.size()
        if segment_ids is not None:
            L_infinity = scatter_mean(
                K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1),
                segment_ids,
                dim=0,
            )
        else:
            L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None
    ):
        # this is based on the official implementation of BFN.
        # import pdb
        # pdb.set_trace()
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)
        alpha = beta1 * (2 * i - 1) / N**2  # [D]
        alpha = alpha.view(-1, 1) # [D, 1]
        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = dist.Independent(
            dist.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = dist.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = dist.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        sender_dist = dist.Independent( dist.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )
        # loss = (
        #         (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
        #         .mean(0)
        #         .flatten(start_dim=1)
        #         .mean(1, keepdims=True)
        #     )
        # #
        return loss.mean()

    def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # i in {1,n}
        # Algorithm 7 in BFN
        e_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D, K)
        assert e_x.size() == e_hat.size()
        alpha = beta1 * (2 * i - 1) / N**2  # [D]

        # print(alpha.shape)
        mean_ = alpha * (K * e_x - 1)  # [D, K]
        std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
        eps = torch.randn_like(mean_)  # [D,K,]
        y_ = mean_ + std_ * eps
        # modify this line:
        matrix_ek = torch.eye(K, K).unsqueeze(0).to(e_x.device)
        matrix_ek.repeat(alpha.size(0), 1, 1)  # [D,K,K]
        mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [D,K,K]
        std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
        likelihood = (
            torch.exp(
                -((y_.unsqueeze(1).repeat(1, K, 1) - mean_matrix) ** 2)
                / (2 * std_matrix**2)
            )
            / (std_matrix * np.sqrt(2 * np.pi))
        ).prod(
            -1
        )  # [D,K]

        if segment_ids is not None:
            L_N = -scatter_mean(
                torch.log((likelihood * e_hat).sum(dim=-1)), segment_ids, dim=0
            )
        else:
            L_N = -torch.log((likelihood * e_hat).sum(dim=-1))  # [D]
        # print(L_N.shape)
        #
        return N * L_N

    def dtime4discrete_loss_gjj(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # i in {1,n}
        # Algorithm 7 in BFN
        e_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D, K)
        assert e_x.size() == e_hat.size()
        alpha = beta1 * (2 * i - 1) / N**2  # [D]

        # print(alpha.shape)
        mean_ = alpha * (K * e_x - 1)  # [D, K]
        std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
        eps = torch.randn_like(mean_)  # [D,K,]
        y_ = mean_ + std_ * eps  # [D, K]
        # modify this line:
        matrix_ek = torch.eye(K, K).to(e_x.device)  # [K, K]
        mean_matrix = K * matrix_ek - 1  # [K, K]
        std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
        _log_gaussians = (  # [D, K]
            (-0.5 * LOG2PI - torch.log(std_matrix))
            - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
        ).sum(-1)

        _inner_log_likelihood = torch.log(
            torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1)
        )  # (D,)

        _inner_log_likelihood = torch.log(e_hat) + _log_gaussians  # [D, K]
        log_likelihood = torch.logsumexp(_inner_log_likelihood, dim=-1)  # [D]

        if segment_ids is not None:
            L_N = -scatter_mean(log_likelihood, segment_ids, dim=0)
        else:
            L_N = -log_likelihood.sum(dim=-1)  # [D]
        # print(L_N.shape)
        #
        return N * L_N

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        if segment_ids is not None:
            loss = scatter_sum(
                (x_pred - x).view(x.shape[0], -1).abs().pow(2), segment_ids, dim=0
            )
        else:
            raise NotImplementedError
            loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

