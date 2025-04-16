## Implementing CSD for linear function and gaussian noise

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.colors import LinearSegmentedColormap
from utils import *
from scipy.optimize import fsolve
from sklearn.datasets import fetch_california_housing

plt.rcParams['text.usetex'] = True

def plot_covariances(alpha1_range, alpha2_range, beta_range, a, b):
    """Plot covariance matrices for varying alpha1, alpha2 and beta (one must be length 1)"""
    correlation_values = np.zeros((len(alpha1_range), len(alpha2_range), len(beta_range)))
    for i, alpha1 in enumerate(alpha1_range):
        for j, alpha2 in enumerate(alpha2_range):
            for k, beta in enumerate(beta_range):
                S = covariance_matrix_normed(alpha1, alpha2, beta)
                correlation_values[i, j, k] = S[a, b]
    if len(alpha1_range) == 1:
        print("alpha1 range is 1")
        matrix_to_plot = correlation_values[0, :, :]
        x_range = beta_range
        y_range = alpha2_range
    elif len(alpha2_range) == 1:
        print("alpha2 range is 1")
        matrix_to_plot = correlation_values[:, 0, :]
        x_range = beta_range
        y_range = alpha1_range
    elif len(beta_range) == 1:
        print("beta range is 1")
        matrix_to_plot = correlation_values[:, :, 0]
        x_range = alpha1_range
        y_range = alpha2_range
    plt.imshow(matrix_to_plot, origin='lower', interpolation='nearest', extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]))
    plt.colorbar()
    plt.show()


def plot_heatmap(test_type_used, alpha1_range, alpha2_range, beta_range, dependence_function, a1, b1, c1, a2, b2, c2, custom_vmax=None):
    """Plot a heatmap for varying alpha1, alpha2 and beta (one must be length 1)"""
    # find the fixed variable
    variable_labels = [r'A', r'B', r'C']
    parameter_labels = [r'\alpha_1', r'\alpha_2', r'\beta']
    ind = r'\perp\!\!\!\!\perp'
    test1_label = f'${variable_labels[a1[0]]} {ind} {variable_labels[b1[0]]}'
    test1_filename = f'{variable_labels[a1[0]]}_ind_{variable_labels[b1[0]]}'
    if c1:
        test1_label += f' | {",".join([variable_labels[c] for c in c1])}'
        test1_filename += f'_given_{",".join([variable_labels[c] for c in c1])}'
    test1_label += "$"
    test2_label = f'${variable_labels[a2[0]]} {ind} {variable_labels[b2[0]]}'
    test2_filename = f'{variable_labels[a2[0]]}_ind_{variable_labels[b2[0]]}'
    if c2:
        test2_label += f' | {",".join([variable_labels[c] for c in c2])}'
        test2_filename += f'_given_{",".join([variable_labels[c] for c in c2])}'
    test2_label += "$"
    function_of_S = lambda S: dependence_function(S, a1, b1, c1, a2, b2, c2)
    #print("Bottom left corner should be: {}".format(function_of_S(covariance_matrix(1., -1, -1))))

    correlation_values = np.zeros((len(alpha1_range), len(alpha2_range), len(beta_range)))
    for i, alpha1 in enumerate(alpha1_range):
        for j, alpha2 in enumerate(alpha2_range):
            for k, beta in enumerate(beta_range):
                S = covariance_matrix_normed(alpha1, alpha2, beta)
                correlation_values[i, j, k] = function_of_S(S)
    # Find the dimension of the fixed variable
    if len(alpha1_range) == 1:
        print("alpha1 range is 1")
        matrix_to_plot = correlation_values[0, :, :]
        print(matrix_to_plot.shape)
        fixed_string = f'${parameter_labels[0]} = {alpha1_range[0]}$'
        fixed_filename = f'{parameter_labels[0][1:]}_eq_{alpha1_range[0]}'
        x_range = beta_range
        y_range = alpha2_range
        x_label = parameter_labels[2]
        y_label = parameter_labels[1]
        precision_t1 = lambda alpha2, beta: la.inv(covariance_matrix_normed(alpha1_range[0], alpha2, beta)[np.ix_(a1+b1+c1, a1+b1+c1)])
        precision_t2 = lambda alpha2, beta: la.inv(covariance_matrix_normed(alpha1_range[0], alpha2, beta)[np.ix_(a2+b2+c2, a2+b2+c2)])
        faithfulness_violations_xs = beta_range
        faithfulness_violation_ys_t1 = [min_solve(lambda alpha2: precision_t1(alpha2, beta)[0, 1], alpha2_range) for beta in beta_range]
        faithfulness_violation_ys_t2 = [min_solve(lambda alpha2: precision_t2(alpha2, beta)[0, 1], alpha2_range) for beta in beta_range]
    elif len(alpha2_range) == 1:
        print("alpha2 range is 1")
        matrix_to_plot = correlation_values[:, 0, :]
        print(matrix_to_plot.shape)
        fixed_string = f'${parameter_labels[1]} = {alpha2_range[0]}$'
        fixed_filename = f'{parameter_labels[1][1:]}_eq_{alpha2_range[0]}'
        x_range = beta_range
        y_range = alpha1_range
        x_label = parameter_labels[2]
        y_label = parameter_labels[0]
        precision_t1 = lambda alpha1, beta: la.inv(covariance_matrix_normed(alpha1, alpha2_range[0], beta)[np.ix_(a1+b1+c1, a1+b1+c1)])
        precision_t2 = lambda alpha1, beta: la.inv(covariance_matrix_normed(alpha1, alpha2_range[0], beta)[np.ix_(a2+b2+c2, a2+b2+c2)])
        faithfulness_violations_xs = beta_range
        faithfulness_violation_ys_t1 = [min_solve(lambda alpha1: precision_t1(alpha1, beta)[0, 1], alpha1_range) for beta in beta_range]
        faithfulness_violation_ys_t2 = [min_solve(lambda alpha1: precision_t2(alpha1, beta)[0, 1], alpha1_range) for beta in beta_range]
    elif len(beta_range) == 1:
        print("beta range is 1")
        matrix_to_plot = correlation_values[:, :, 0]
        print(matrix_to_plot.shape)
        fixed_string = f'${parameter_labels[2]} = {beta_range[0]}$'
        fixed_filename = f'{parameter_labels[2][1:]}_eq_{beta_range[0]}'
        x_range = alpha1_range
        y_range = alpha2_range
        x_label = parameter_labels[1]
        y_label = parameter_labels[0]
        precision_t1 = lambda alpha1, alpha2: la.inv(covariance_matrix_normed(alpha1, alpha2, beta_range[0])[np.ix_(a1+b1+c1, a1+b1+c1)])
        precision_t2 = lambda alpha1, alpha2: la.inv(covariance_matrix_normed(alpha1, alpha2, beta_range[0])[np.ix_(a2+b2+c2, a2+b2+c2)])
        faithfulness_violations_xs = alpha2_range
        faithfulness_violation_ys_t1 = [min_solve(lambda alpha1: precision_t1(alpha1, alpha2)[0, 1], alpha1_range) for alpha2 in alpha2_range]
        faithfulness_violation_ys_t2 = [min_solve(lambda alpha1: precision_t2(alpha1, alpha2)[0, 1], alpha1_range) for alpha2 in alpha2_range]
    else:
        raise ValueError("One of the ranges must have length 1")

    # plot the heatmap
    plt.imshow(matrix_to_plot, origin='lower', interpolation='nearest', extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]))
    plt.colorbar()
    not_none = [i for i, item in enumerate(faithfulness_violation_ys_t1) if item[1] is not None]
    if len(not_none) > -1:
        plt.plot(faithfulness_violations_xs, [y[1] for y in faithfulness_violation_ys_t2], label=test2_label, color='blue')
    elif len(not_none) > 0:
        # find minimum of the values that are not none
        index = min([(y[0], i) for i, y in enumerate(faithfulness_violation_ys_t2) if y[1] is not None])[1]
        plt.axvline(x=faithfulness_violations_xs[index], color='blue', label=test2_label)
    if len(not_none) > -1:
        plt.plot(faithfulness_violations_xs, [y[1] for y in faithfulness_violation_ys_t1], label=test1_label, color='red')
    elif len(not_none) > 0:
        # find minimum of the values that are not none
        index = min([(y[0], i) for i, y in enumerate(faithfulness_violation_ys_t1) if y[1] is not None])[1]
        plt.axvline(x=faithfulness_violations_xs[index], color='red', label=test1_label)
    not_none = [i for i, item in enumerate(faithfulness_violation_ys_t2) if item[1] is not None]
    plt.legend()
    plt.xlabel("${}$".format(x_label), fontsize=20)
    plt.ylabel("${}$".format(y_label), fontsize=20)
    plt.title('{} between {} and {}, {}'.format(test_type_used, test1_label, test2_label, fixed_string), fontsize=20)
    filename = test_type_used + "_" + test1_filename + '_vs_' + test2_filename + '_fixed_' + fixed_filename
    plt.savefig(f'plots/{filename}.pdf')

def percent_negative_experiment(alpha1_range, alpha2_range, beta_range):
    counter = 0
    neg_counter = 0
    cimds = []
    for i, alpha1 in enumerate(alpha1_range):
        for j, alpha2 in enumerate(alpha2_range):
            for k, beta in enumerate(beta_range):
                S = covariance_matrix(alpha1, alpha2, beta)
                cimd = CIMD(S, [0], [1], [], [0], [2], [])
                cimds.append(cimd)
                counter+=1
    # histogram of cimds
    plt.hist(cimds)
    plt.show()

def test_text(a, b, c, labels):
    result = labels[a[0]] + r"$\perp\!\!\!\!\perp$" + labels[b[0]]
    if len(c) > 0:
        result += r"$ | $" + ",".join([labels[ci] for ci in c])
    return result

# Takes a dataframe and finds CIMD for all pairs of CI tests
def compare_all_ci_tests(df, dataset_title):
    labels = df.columns.tolist()
    cov = dataframe_to_cov(df)
    num_variables = len(labels)
    CIMDs = list()
    test_order = list()
    all_tests = []
    for k in range(2):
        all_tests_k = all_subsets(list(range(num_variables)), k)
        for a, b, c in all_tests_k:
            test_order.append(test_text(a, b, c, labels))
        all_tests = all_tests + all_tests_k
    counter = 0
    for a1, b1, c1 in all_tests:
        counter += 1
        print("{} out of {}".format(counter, len(all_tests)))
        CIMDs.append([])
        for a2, b2, c2 in all_tests:
            CIMDs[-1].append(CIMD_limited(cov, a1, b1, c1, a2, b2, c2))

    # plot comparison of pairwise CIMD, labeling test order
    # Create the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(np.matrix(CIMDs), interpolation='nearest')
    # Add colorbar
    plt.colorbar(label='CIMD')
    # Add labels
    tick_marks = np.arange(len(test_order))
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(tick_marks, test_order, rotation=45, ha='right')
    plt.yticks(tick_marks, test_order)

    # Add title
    plt.title('CIMD-lim Between Pairs of CI Tests, {} Dataset'.format(dataset_title))
    plt.savefig(f'plots/{dataset_title.replace(" ", "_")}.pdf', bbox_inches='tight')

# Takes a dataframe and finds FS-CID for all pairs of CI tests
def compare_all_ci_tests_fs_cid(df, dataset_title, subset_size=50):
    labels = df.columns.tolist()
    num_variables = len(labels)
    all_tests = []
    test_order = []
    for k in range(2):
        all_tests_k = all_subsets(list(range(num_variables)), k)
        for a, b, c in all_tests_k:
            test_order.append(test_text(a, b, c, labels))
        all_tests = all_tests + all_tests_k

    iterations = 1000
    T1_Counts = np.zeros((len(all_tests), len(all_tests)))
    T2_Counts = np.zeros((len(all_tests), len(all_tests)))
    T1T2_Counts = np.zeros((len(all_tests), len(all_tests)))
    for iter in range(iterations):
        print("Iteration {} / {}".format(iter, iterations))
        data = df.sample(n=subset_size).to_numpy()
        for counter1, (a1, b1, c1) in enumerate(all_tests):
            for counter2, (a2, b2, c2) in enumerate(all_tests):
                suffstat = partial_correlation_suffstat(data)
                t1 = partial_correlation_test(suffstat, a1, b1, set(c1))
                t2 = partial_correlation_test(suffstat, a2, b2, set(c2))
                if (not t1['reject']) and (not t2['reject']):
                    T1T2_Counts[counter1, counter2] += 1
                if not t1['reject']:
                    T1_Counts[counter1, counter2] += 1
                if not t2['reject']:
                    T2_Counts[counter1, counter2] += 1
    T1_Counts = T1_Counts / iterations
    T2_Counts = T2_Counts / iterations
    T1T2_Counts = T1T2_Counts / iterations
    # plot comparison of pairwise FS-CID, labeling test order
    FSCIDs = T1T2_Counts - np.multiply(T1_Counts, T2_Counts)
    # Create the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(np.matrix(FSCIDs), interpolation='nearest')
    # Add colorbar
    plt.colorbar(label='FS-CID')
    # Add labels
    tick_marks = np.arange(len(test_order))
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(tick_marks, test_order, rotation=45, ha='right')
    plt.yticks(tick_marks, test_order)

    # Add title
    plt.title('FS-CID Between Pairs of CI Tests, {} Dataset'.format(dataset_title))
    plt.savefig(f'plots/FSCID_{dataset_title.replace(" ", "_")}.pdf', bbox_inches='tight')      


################################ Do Experiments with Section 4 Structural Equation Parameters (Figure 1) ##############################

# Constant is alpha1 = 1, test A \indep C and B \indep C
#plot_heatmap('CIMD', [.5], np.linspace(-.5, .5, 100), np.linspace(-.5,.5, 100), CIMD, [0], [2], [], [1], [2], [])
#plot_heatmap('CIMD-lim', [.5], np.linspace(-.5, .5, 100), np.linspace(-.5,.5, 100), CIMD_limited, [0], [2], [], [1], [2], [])
#plot_heatmap('FS-CID', [.5], np.linspace(-.5, .5, 100), np.linspace(-.5,.5, 100), CI_test_dependence_lim, [0], [2], [], [1], [2], [])


############################# Do experiments on Real Data (Figure 2 and Figure 3) ###############################
# CIMDlim tests
def real_data_CIMD_matrix_california_housing(): 
    housing = fetch_california_housing()
    dropped = ['Latitude', 'Longitude', 'AveBedrms', 'AveOccup']
    df = dataset_to_pd(housing, dropped)
    compare_all_ci_tests(df, dataset_title="California Housing")

def real_data_CIMD_matrix_apple_watch_fitbit(): 
    df = pd.read_csv('datasets/aw_fb_data.csv')
    df = df[['height', 'weight', 'steps', 'resting_heart', 'hear_rate']]
    compare_all_ci_tests(df, dataset_title="Apple Watch and Fitbit")

def real_data_CIMD_matrix_auto_mpg(): 
    df = pd.read_csv('datasets/auto_mpg.txt', sep='\t')
    df = df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']]
    compare_all_ci_tests(df, dataset_title="Auto MPG")

plt.clf()
real_data_CIMD_matrix_california_housing()
plt.clf()
real_data_CIMD_matrix_apple_watch_fitbit()
plt.clf()
real_data_CIMD_matrix_auto_mpg()
plt.clf()

# FS-CID Tests
def real_data_FS_CID_matrix_california_housing(): 
    housing = fetch_california_housing()
    dropped = ['Latitude', 'Longitude', 'AveBedrms', 'AveOccup']
    df = dataset_to_pd(housing, dropped)
    compare_all_ci_tests_fs_cid(df, dataset_title="California Housing")

def real_data_FS_CID_matrix_apple_watch_fitbit(): 
    df = pd.read_csv('datasets/aw_fb_data.csv')
    df = df[['height', 'weight', 'steps', 'resting_heart', 'hear_rate']]
    compare_all_ci_tests_fs_cid(df, dataset_title="Apple Watch and Fitbit")

def real_data_FS_CID_matrix_auto_mpg(): 
    df = pd.read_csv('datasets/auto_mpg.txt', sep='\t')
    df = df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']]
    compare_all_ci_tests_fs_cid(df, dataset_title="Auto MPG")


real_data_FS_CID_matrix_california_housing()
plt.clf()
real_data_FS_CID_matrix_apple_watch_fitbit()
plt.clf()
real_data_FS_CID_matrix_auto_mpg()
plt.clf()


############################ Do Experiments with various coefficients set to 0 (Figure 4) ########################
# beta = 0 (C <- A -> B)
def beta_zero():
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), [0], CIMD, [0], [1], [], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), [0], CIMD, [1], [0], [], [1], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), [0], CIMD, [2], [1], [], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), [0], CIMD, [0], [1], [], [0], [1], [2])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), [0], CIMD, [0], [2], [], [0], [2], [1])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), [0], CIMD, [1], [2], [], [1], [2], [0])
    plt.clf()

# alpha1 = 0 (A -> C <- B)
def alpha1_zero():
    plot_heatmap('CIMD', [0], np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), CIMD, [0], [1], [], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', [0], np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), CIMD, [1], [0], [], [1], [2], [])
    plt.clf()
    plot_heatmap('CIMD', [0], np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), CIMD, [2], [1], [], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', [0], np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), CIMD, [0], [1], [], [0], [1], [2])
    plt.clf()
    plot_heatmap('CIMD', [0], np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), CIMD, [0], [2], [], [0], [2], [1])
    plt.clf()
    plot_heatmap('CIMD', [0], np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), CIMD, [1], [2], [], [1], [2], [0])
    plt.clf()

# alpha2 = 0 (A -> B -> C)
def alpha2_zero():
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), [0], np.linspace(-1, 1, 100), CIMD, [0], [1], [], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), [0], np.linspace(-1, 1, 100), CIMD, [1], [0], [], [1], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), [0], np.linspace(-1, 1, 100), CIMD, [2], [1], [], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), [0], np.linspace(-1, 1, 100), CIMD, [0], [1], [], [0], [1], [2])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), [0], np.linspace(-1, 1, 100), CIMD, [0], [2], [1], [0], [2], [])
    plt.clf()
    plot_heatmap('CIMD', np.linspace(-1, 1, 100), [0], np.linspace(-1, 1, 100), CIMD, [1], [2], [], [1], [2], [0])
    plt.clf()


######################
def CIMD_mixture():
    alpha1 = 0
    alpha2 = 1
    beta = 1 
    S = covariance_matrix(alpha1, alpha2, beta)
    S2 = covariance_matrix(0,0,0)
    n=50
    data = np.random.multivariate_normal([0, 0, 0], S, n)
    data2 = np.random.multivariate_normal([0, 0, 0], S2, n)
    d = np.concatenate((data, data2), axis=0)
    df = pd.DataFrame(d, columns=['A', 'B', 'C'])
    compare_all_ci_tests(df, dataset_title="Mixture")

def FS_CID_mixture(): 
    alpha1 = 0
    alpha2 = 1
    beta = 1 
    S = covariance_matrix(alpha1, alpha2, beta)
    S2 = covariance_matrix(0,0,0)
    n=50
    data = np.random.multivariate_normal([0, 0, 0], S, n)
    data2 = np.random.multivariate_normal([0, 0, 0], S2, n)
    d = np.concatenate((data, data2), axis=0)
    df = pd.DataFrame(d, columns=['A', 'B', 'C'])
    compare_all_ci_tests_fs_cid(df, dataset_title="Mixture")

#CIMD_mixture()
#FS_CID_mixture()