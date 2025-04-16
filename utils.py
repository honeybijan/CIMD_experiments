## Implementing CSD for linear function and gaussian noise

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from conditional_independence import partial_correlation_suffstat, partial_correlation_test
import pandas as pd
import itertools

def min_solve(f, x_range):
    """Find the minimum of a function f over a range x_range"""
    eps = .03
    min_val = eps
    min_x = None
    for x in x_range:
        val = np.abs(f(x))
        if val < min_val:
            min_val = val
            min_x = x
    return [min_val, min_x]

# Compute the conditional covariance matrix on indices a given indices c, starting with covariance matrix S
def conditional_covariance(S, a, c):
    """Compute the conditional covariance matrix on indices `a` given indices `c`, starting with covariance matrix `S`"""
    S_AA = S[np.ix_(a, a)]
    S_AC = S[np.ix_(a, c)]
    S_CC = S[np.ix_(c, c)]
    S_CA = S_AC.T
    S_CC_inv = np.linalg.inv(S_CC)
    return S_AA - S_AC @ S_CC_inv @ S_CA

# Compute new covariance matrix by projecting old one into conditional independence on A, B given C
def covariance_projection(S, a, b, c):
    """Project the covariance matrix `S` into conditional independence on `A`, `B` given `C`"""
    #S_CC is a submatrix of S using indices from C
    S_CC = S[np.ix_(c, c)]
    S_ABC = S[np.ix_(a + b, c)]
    S_CAB = S[np.ix_(c, a + b)]
    # x is the list of indices of S that are not in a, b, or c
    x = [i for i in range(S.shape[0]) if i not in a + b + c]
    S_XABC = S[np.ix_(x, a + b + c)]
    S_ABAB_cond = conditional_covariance(S, a + b, c)
    #print("Conditioned sub-cov matrix: {}".format(S_ABAB_cond))

    new_S_ABAB = S_ABC @ np.linalg.inv(S_CC) @ S_CAB
    # replace the submatrix at indices a, b with new_S_ABAB
    new_S = S.copy()
    # the diagonal of S_ABAB
    diag_S_ABAB_cond = np.diag(np.diag(S_ABAB_cond))
    new_S[np.ix_(a + b, a + b)] = new_S_ABAB + diag_S_ABAB_cond
    new_S_XX_change = S_XABC @ (la.inv(new_S[np.ix_(a + b + c, a + b + c)]) - la.inv(S[np.ix_(a + b + c, a + b + c)])) @ S_XABC.T
    new_S[np.ix_(x, x)] += new_S_XX_change
    return new_S

# Compute mutual information using 2x2 covariance matrix S
def mutual_information(S):
    """Compute the mutual information between indices `a` and `b` using covariance matrix `S`"""
    a = [0]
    b = [1]
    S_AA = S[np.ix_(a, a)]
    S_BB = S[np.ix_(b, b)]
    return .5 * (np.linalg.slogdet(S_AA)[1] + np.linalg.slogdet(S_BB)[1] - np.linalg.slogdet(S)[1])

# Compute the conditional mutual information on indices a, b given c using covariance matrix S
def conditional_mutual_information(S, a, b, c):
    """Compute the conditional mutual information on indices `a`, `b` given `c` using covariance matrix `S`"""
    cond_S = conditional_covariance(S, a + b , c)
    return mutual_information(cond_S)

# Compute the conditional independence meta dependence (CIMD)
def CIMD(S, a1, b1, c1, a2, b2, c2, max_value=np.inf):
    """Compute the conditional independence meta dependence (CIMD) between indices `a1`, `b1` and indices `a2`, `b2` given indices `c1`, `c2` using covariance matrix `S`"""
    S_projected = covariance_projection(S, a2, b2, c2)
    MI1 = conditional_mutual_information(S, a1, b1, c1)
    MI2 = conditional_mutual_information(S_projected, a1, b1, c1)
    #print("MI1: {} MI2: {}".format(MI1, MI2))
    return min(MI1 - MI2, max_value)

# Compute the conditional independence meta dependence (CIMD)
def CIMD_limited_normed(S, a1, b1, c1, a2, b2, c2, max_value=np.inf):
    """Compute the conditional independence meta dependence (CIMD) between indices `a1`, `b1` and indices `a2`, `b2` given indices `c1`, `c2` using covariance matrix `S`"""
    S_projected = covariance_projection(S, a2, b2, c2)
    MI1 = conditional_mutual_information(S, a1, b1, c1)
    MI2 = conditional_mutual_information(S_projected, a1, b1, c1)
    #print("MI1: {} MI2: {}".format(MI1, MI2))
    if MI1 > .1 or MI2 > .1:
        return 0
    if MI1 < .00001:
        return 0
    return min((MI1 - MI2)/MI1, max_value)

# Compute the conditional independence meta dependence (CIMD)
def CIMD_limited(S, a1, b1, c1, a2, b2, c2, max_value=np.inf):
    """Compute the conditional independence meta dependence (CIMD) between indices `a1`, `b1` and indices `a2`, `b2` given indices `c1`, `c2` using covariance matrix `S`"""
    S_projected = covariance_projection(S, a2, b2, c2)
    MI1 = conditional_mutual_information(S, a1, b1, c1)
    MI2 = conditional_mutual_information(S_projected, a1, b1, c1)
    #print("MI1: {} MI2: {}".format(MI1, MI2))
    if MI1 > .1 or MI2 > .1:
        return 0
    return min((MI1 - MI2), max_value)

# Compute the FS-CID (difference between the product of the probability of faithfulness violations and the joint probability of faithfulness violation)
def CI_test_dependence_lim(S, a1, b1, c1, a2, b2, c2, n=20):
    """Compute the difference between the product of the probability of faithfulness violations and the joint probability of faithfulness violation"""
    # Sample n points using covariance matrix S and find a stochastic version of S
    S_projected = covariance_projection(S, a2, b2, c2)
    MI1 = conditional_mutual_information(S, a1, b1, c1)
    MI2 = conditional_mutual_information(S_projected, a1, b1, c1)
    #print("MI1: {} MI2: {}".format(MI1, MI2))
    if MI1 > .1 or MI2 > .1:
        return 0
    double_rejects = 0
    t1_rejects = 0
    t2_rejects = 0
    total = 1000
    for i in range(total):
        data = np.random.multivariate_normal([0, 0, 0], S, n)
        suffstat = partial_correlation_suffstat(data)
        t1 = partial_correlation_test(suffstat, a1, b1, set(c1))
        t2 = partial_correlation_test(suffstat, a2, b2, set(c2))
        if (not t1['reject']) and (not t2['reject']):
            double_rejects += 1
        if not t1['reject']:
            t1_rejects += 1
        if not t2['reject']:
            t2_rejects += 1
    return double_rejects / total - ((t1_rejects / total) * (t2_rejects / total))

def CI_test_dependence(S, a1, b1, c1, a2, b2, c2, n=20):
    """Compute the difference between the product of the probability of faithfulness violations and the joint probability of faithfulness violation"""
    double_rejects = 0
    t1_rejects = 0
    t2_rejects = 0
    total = 1000
    for i in range(total):
        data = np.random.multivariate_normal([0, 0, 0], S, n)
        suffstat = partial_correlation_suffstat(data)
        t1 = partial_correlation_test(suffstat, a1, b1, set(c1))
        t2 = partial_correlation_test(suffstat, a2, b2, set(c2))
        if (not t1['reject']) and (not t2['reject']):
            double_rejects += 1
        if not t1['reject']:
            t1_rejects += 1
        if not t2['reject']:
            t2_rejects += 1
    return double_rejects / total - ((t1_rejects / total) * (t2_rejects / total))


def stoch_covariance_matrix(alpha1, alpha2, beta, n):
    """Compute the covariance matrix for a linear function and Gaussian noise using monte carlo"""
    A = np.random.normal(0, 1, n)
    B = A * alpha1 + np.random.normal(0, 1, n)
    C = A * alpha2 + B * beta + np.random.normal(0, 1, n)
    S = np.cov(np.array([A, B, C]))
    return S

def covariance_matrix(alpha1, alpha2, beta):
    """Compute the covariance matrix for a linear function and Gaussian noise"""
    cov_ab = alpha1
    cov_ac = alpha2 + (alpha1 * beta)
    cov_bc = alpha1 * (alpha2 + (alpha1 * beta)) + beta
    S = np.array([[1, cov_ab, cov_ac],
                  [cov_ab, 1 + (alpha1**2), cov_bc],
                  [cov_ac, cov_bc, 1 + (beta**2) + ((alpha2 + (beta * alpha1))**2)]])
    return S

def stoch_covariance_matrix_normed(alpha1, alpha2, beta, n):
    """Compute the covariance matrix for a linear function and Gaussian noise using monte carlo"""
    A = np.random.normal(0, 1, n)
    B = A * alpha1 + np.random.normal(0, np.sqrt(1 - (alpha1**2)), n)
    C = A * alpha2 + B * beta + np.random.normal(0, np.sqrt(1 - (alpha2**2) - (beta**2)), n)
    S = np.cov(np.array([A, B, C]))
    print(S)
    return S

# Adjusted so variance is the same for every variable
def covariance_matrix_normed(alpha1, alpha2, beta):
    """Compute the covariance matrix for a linear function and Gaussian noise"""
    #return stoch_covariance_matrix_normed(alpha1, alpha2, beta, 1000)
    cov_ab = alpha1
    cov_ac = alpha2 + (alpha1 * beta)
    cov_bc = alpha1 * (alpha2 + (alpha1 * beta)) + beta*(1 - (alpha1**2))
    S = np.array([[1, cov_ab, cov_ac],
                  [cov_ab, 1, cov_bc],
                  [cov_ac, cov_bc, 1]])
    return S

# takes a list of numbers and returns all possible triples of subsets of size 1, 1, and k
def all_subsets(lst, k):
    """Return all possible triples of subsets of size 1, 1, and k"""
    subsets = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            # now iterate over all subsets of lst - [lst[i], lst[j] of size k
            for k_subset in itertools.combinations([x for x in lst if x != lst[i] and x != lst[j]], k):
                subsets.append(([lst[i]], [lst[j]], list(k_subset)))
    return subsets

# prepare dataset into a dataframe
def dataset_to_pd(d, dropped):
    """Compute the covariance matrix for a dataset"""
    df1 = pd.DataFrame(data = d.data, columns = d.feature_names)
    df2 = pd.DataFrame(data = d.target, columns = d.target_names)
    df = pd.concat([df1, df2], axis=1)
    for drop in dropped:
        df = df.drop(drop, axis=1)
    return df

# Returns a numpy covariance matrix
def dataframe_to_cov(df, subset_size = None):
    """Compute the covariance matrix for a dataframe"""
    if subset_size is not None:
        # find a random sample of the dataframe of size subset_size
        df = df.sample(n=subset_size)
    return df.cov().to_numpy()

def correlCo(someList1, someList2):

    # First establish the means and standard deviations for both lists.
    xMean = np.mean(someList1)
    yMean = np.mean(someList2)
    xStandDev = np.std(someList1)
    yStandDev = np.std(someList2)
    # r numerator
    rNum = 0.0
    for i in range(len(someList1)):
        rNum += (someList1[i]-xMean)*(someList2[i]-yMean)

    # r denominator
    rDen = xStandDev * yStandDev

    r =  rNum/rDen
    return r

