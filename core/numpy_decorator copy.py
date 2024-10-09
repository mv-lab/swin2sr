import numpy as np
import torch


def allow_numpy(func):
    """
    All optional arguements are passed as is. positional arguments are checked. if they are numpy array,
    they are converted to torch Tensor.
    """

    def numpy_wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.Tensor(arg)
            new_args.append(arg)
        new_args = tuple(new_args)

        output = func(*new_args, **kwargs)
        return output

    return numpy_wrapper
