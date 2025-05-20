import torch


def get_logp_a(C, z, logdet_J):
    _GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
    logp = C * _GCONST_ - 0.5*torch.sum((z-1)**2, 1) + logdet_J
    logp = logp / C
    return logp


def get_normal_boundary(logps, mask, pos_beta=0.05):
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
    """
    normal_logps = logps[mask == 0]

    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]
    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary

    return b_n