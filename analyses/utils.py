import numpy as np
from scipy.special import hyp2f1, gammaln


def rpdf(rho, n, rs):
    """ rho = population correlation coefficient. If rho = 0, then this function simplifies to the above function """
    lnum = np.log(n-2) + gammaln(n-1) + np.log((1-rho**2)**(.5*(n-1))) + np.log((1-rs**2)**(.5*(n-4)))
    lden = np.log(np.sqrt(2*np.pi)) + gammaln(n-.5) + np.log((1-rho*rs)**(n-3/2))
    fac = lnum - lden
    hyp = hyp2f1(.5, .5, (2*n-1)/2, (rho*rs+1)/2)
    return np.exp(fac) * hyp
