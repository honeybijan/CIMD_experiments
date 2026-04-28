# Project Title

This code is used to generate the figures in the paper: "Meta-Dependence in Conditional Independence Testing"

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)

## Requirements

In order to run this code, you must have the following python packages installed

numpy
scipy
utils
matplotlib
itertools
conditional-independence
pandas

## Usage

Examples of how to use the project.

```bash
# Example usage
python3 cimd.py
```

The following lines should be un-commented to generate each figure.

Figure 1:

```python
# Constant is alpha1 = .5, test A \indep C and B \indep C
plot_heatmap('CIMD', [.5], np.linspace(-.5, .5, 100), np.linspace(-.5,.5, 100), CIMD, [0], [2], [], [1], [2], [])
plot_heatmap('CIMD-lim', [.5], np.linspace(-.5, .5, 100), np.linspace(-.5,.5, 100), CIMD_limited, [0], [2], [], [1], [2], [])
plot_heatmap('FS-CID', [.5], np.linspace(-.5, .5, 100), np.linspace(-.5,.5, 100), CI_test_dependence_lim, [0], [2], [], [1], [2], [])
```

Each instance of plot_heatmap takes in a constant value for the leftout variable (in this case, .5), two linspaces which give it the range of the parameter values for the heat map, and a function to test (CIMD, CIMD_limited, and CI_test_dependence_lim). There are then 6 lists which give the variables whose CI is being tested. [0], [2], [], [1], [2], [] means we are testing whether X[0] is independent from X[2] conditioned on [] and comparing it to testing X[1] independent from X[2] conditioned on []. These values can be changed to test different conditional independencies.

Figure 2 and 3:

```python
real_data_FS_CID_matrix_california_housing()
plt.clf()
real_data_FS_CID_matrix_apple_watch_fitbit()
plt.clf()
real_data_FS_CID_matrix_auto_mpg()
plt.clf()
```