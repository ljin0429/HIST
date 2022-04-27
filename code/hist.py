import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import binarize


def pairwise_mahalanobis(X, means, log_vars):
    """
    Computes pairwise squared Mahalanobis distances between X (data points) and a set of distributions
    :param X: [N, F] where N is the batch size and F is the feature dimension
    :param means: [C, F] C is the number of classes
    :param log_vars: [C, F] C is the number of classes, we assume a diagonal covariance matrix
    :return: pairwise squared Mahalanobis distances... [N, C, F] matrix
    i.e., M_ij = (x_i-means_j)\top * inv_cov_j * (x_i - means_j)

    """
    sz_batch = X.size(0)
    nb_classes = means.size(0)

    new_X = torch.unsqueeze(X, dim=1)  # [N, 1, F]
    new_X = new_X.expand(-1, nb_classes, -1)  # [N, C, F]

    new_means = torch.unsqueeze(means, dim=0)  # [1, C, F]
    new_means = new_means.expand(sz_batch, -1, -1)  # [N, C, F]

    # pairwise distances
    diff = new_X - new_means

    # convert log_var to covariance
    covs = torch.unsqueeze(torch.exp(log_vars), dim=0)  # [1, C, F]

    # the squared Mahalanobis distances
    M = torch.div(diff.pow(2), covs).sum(dim=-1)  # [N, C]

    return M


# Class Distributions to Hypergraph
class CDs2Hg(nn.Module):
    def __init__(self, nb_classes, sz_embed, tau=32, alpha=0.9):
        super(CDs2Hg, self).__init__()
        # Parameters (means and covariance)
        self.means = nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda())
        self.log_vars = nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda())

        # Initialization
        nn.init.kaiming_normal_(self.means, mode='fan_out')
        nn.init.kaiming_normal_(self.log_vars, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.tau = tau
        self.alpha = alpha

    def forward(self, X, T):
        mu = self.means
        log_vars = self.log_vars
        log_vars = F.relu6(log_vars)

        # L2 normalize
        X = F.normalize(X, p=2, dim=-1)
        mu = F.normalize(mu, p=2, dim=-1)

        # Labels of each distributions (NxC matrix)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)

        # Compute pairwise mahalanobis distances (NxC matrix)
        distance = pairwise_mahalanobis(X, mu, log_vars)

        # Distribution loss
        mat = F.softmax(-1 * self.tau * distance, dim=1)
        loss = torch.sum(mat * P_one_hot, dim=1)
        non_zero = loss != 0
        loss = -torch.log(loss[non_zero])

        # Hypergraph construction
        class_within_batch = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        exp_term = torch.exp(-1 * self.alpha * distance[:, class_within_batch])
        H = P_one_hot[:, class_within_batch] + exp_term * (1 - P_one_hot[:, class_within_batch])

        return loss.mean(), H


# Hypergraph Neural Networks (AAAI 2019)
class HGNN(nn.Module):
    def __init__(self, nb_classes, sz_embed, hidden):
        super(HGNN, self).__init__()

        self.theta1 = nn.Linear(sz_embed, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lrelu = nn.LeakyReLU(0.1)

        self.theta2 = nn.Linear(hidden, nb_classes)

    def compute_G(self, H):
        # the number of hyperedge
        n_edge = H.size(1)
        # the weight of the hyperedge
        we = torch.ones(n_edge).cuda()
        # the degree of the node
        Dv = (H * we).sum(dim=1)
        # the degree of the hyperedge
        De = H.sum(dim=0)

        We = torch.diag(we)
        inv_Dv_half = torch.diag(torch.pow(Dv, -0.5))
        inv_De = torch.diag(torch.pow(De, -1))
        H_T = torch.t(H)

        # propagation matrix
        G = torch.chain_matmul(inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half)

        return G

    def forward(self, X, H):
        G = self.compute_G(H)

        # 1st layer
        X = G.matmul(self.theta1(X))
        X = self.bn1(X)
        X = self.lrelu(X)

        # 2nd layer
        out = G.matmul(self.theta2(X))

        return out
