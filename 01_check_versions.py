"""
-------------------------------------------------------------------------------
 Machine learning tutorial analyzing iris flowers. This script displays
 version numbers for the packages installed in the Conda environment.

 Adapted from:
 https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
-------------------------------------------------------------------------------
"""

import sys

import matplotlib
import numpy
import pandas
import sklearn
import scipy


if __name__ == '__main__':
    print(f'\nPython: {sys.version}')
    print(f'matplotlib: {matplotlib.__version__}')
    print(f'numpy: {numpy.__version__}')
    print(f'pandas: {pandas.__version__}')
    print(f'scikit-learn/sklearn: {sklearn.__version__}')
    print(f'scipy: {scipy.__version__}')
