import numpy as np
from scipy.stats import rv_continuous


class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e

    def _cdf(self, x):
        return np.log(x/self.a)/np.log(self.b/self.a)

def logUniform(a=1, b=np.exp(10)):
    return logUniform_gen(a=a, b=b, name="logUniiform")
