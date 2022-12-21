import torch
import math

def compute_out_dimension(out_dimension, l):
    if isinstance(l, torch.nn.ReLU):
        return out_dimension
    elif isinstance(l, torch.nn.Identity):
        return out_dimension
    elif isinstance(l, torch.nn.Linear):
        return (1, 1, l.out_features)
    elif isinstance(l, torch.nn.BatchNorm2d):
        return out_dimension
    elif isinstance(l, torch.nn.Conv2d):
        w_dim = math.floor(
            (out_dimension[1] - l.kernel_size[0] + 2 * l.padding[0]) / l.stride[0]) + 1
        h_dim = math.floor(
            (out_dimension[2] - l.kernel_size[1] + 2 * l.padding[1]) / l.stride[1]) + 1

        out_dimension = (l.out_channels, w_dim, h_dim)
        return out_dimension
    else:
        for la in l.path_a:
            out_dimension = compute_out_dimension(out_dimension, la)
        return out_dimension
