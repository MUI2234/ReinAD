import torch


_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_theta = torch.nn.LogSigmoid()


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp


def neg_relu(logps):
    zeros = logps.new_zeros(logps.shape)
    logps = torch.min(logps, zeros)
    
    return logps