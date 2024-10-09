import math

import torch


class StableExponential:
    """
    Here, the idea is that everything is done on the tensor which you've given in the constructor.
    when exp() is called, what that means is that we want to compute self._tensor.exp()
    when log() is called, we want to compute torch.log(self._tensor.exp())
    
    What is done here is that definition of exp() has been changed. This, naturally, has changed the result of log. 
    but the log is still the mathematical log, that is, it takes the math.log() on whatever comes out of exp().
    """

    def __init__(self, tensor):
        self._raw_tensor = tensor
        posneg_dic = self.posneg_separation(self._raw_tensor)
        self.pos_f, self.neg_f = posneg_dic['filter']
        self.pos_data, self.neg_data = posneg_dic['value']

    def posneg_separation(self, tensor):
        pos = tensor > 0
        pos_tensor = torch.clip(tensor, min=0)

        neg = tensor <= 0
        neg_tensor = torch.clip(tensor, max=0)

        return {'filter': [pos, neg], 'value': [pos_tensor, neg_tensor]}

    def exp(self):
        return torch.exp(self.neg_data) * self.neg_f + (1 + self.pos_data) * self.pos_f

    def log(self):
        """
        Note that if you have the output from exp(). You could simply apply torch.log() on it and that should give
        identical numbers.
        """
        return self.neg_data * self.neg_f + torch.log(1 + self.pos_data) * self.pos_f


def log_prob(nn_output_mu, nn_output_logvar, x):
    """
    This computes the log_probablity of a Normal distribution.
    Args:
        nn_output_mu: mean of the distribution
        nn_output_logvar: log(variance) of the distribution. Note that for numerical stablity, this is no longer a
            log(variance). We define a different function to get variance from this value. This is done this way for
            stability.
        x: input for which the log_prob needs to be computed.
    """
    assert False, "This code is not compatible with Stable exponential. Ideally, StableLogVar should be passed here."
    mu = nn_output_mu
    # compute the
    var_gen = StableExponential(nn_output_logvar)
    var = var_gen.exp()
    logstd = 1 / 2 * var_gen.log()
    return -((x - mu)**2) / (2 * var) - logstd - math.log(math.sqrt(2 * math.pi))


if __name__ == '__main__':
    stable = StableExponential(torch.Tensor([-0.1]).mean())
    print(stable.exp())