[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/calekochenour/machine-learning-tutorial/main)

# Machine Learning Tutorial

Provides a tutorial for working with machine learning algorithms in Python. Adapted from the following [tutorial](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/).

## Project Structure 

```
machine-learning
│   01_check_versions.py
│   02_check_imports.py
│   03_analyze_flowers.py
│   environment.yml
│   Makefile
│   README.md
│
├───data
│       iris.csv
│
└───figures
        figure-01-box-plot.png
        figure-02-histogram.png
        figure-03-scatter-matrix.png
        figure-04-spot-check.png
```

### `01_check_versions.py`

Displays version numbers for the packages installed in the Conda environment.

### `02_check_imports.py`

Checks that all packages necessary for the iris flower workflow import correctly.

### `03_analyze_flowers.py`

Machine learning tutorial analyzing iris flowers.

### `environment.yml`

Provides recipe for the Conda environment.

### `Makefile`

Provides recipes for automating the project.

### `data/`

Contains input data used by the `03_analyze_flowers.py` script.

### `figures/`

Contains all figures created by the `03_analyze_flowers.py` script. Note this directory is empty by default. The project tree diagram shows what it looks like once the `03_analyze_flowers.py` script has completed.

## Project Setup - Local

Install Miniconda: 

* Download the latest [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) for your operating system
* Follow the Miniconda installation instructions ([Windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html), [Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html), [macOS](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html))

Open terminal:

* Open the operating system default terminal

Clone GitLab repository:

```commandline
(base) > git clone https://github.com/calekochenour/machine-learning-tutorial.git
```

Create Conda environment:

```commandline
(base) > cd machine-learning-tutorial
(base) > conda env create -f environment.yml
```

Activate Conda environment:

```commandline
(base) > conda activate machine-learning-tutorial
```

## Project Setup - Binder

Launching the repository with Binder will open a JupyterLab instance and set up the Conda enviroment automatically. The scripts can then be run direclty within this JupyterLab instance. Note that the Binder is non-persistent; once closed, any file changes will not be saved. 

Click the icon below to launch the project with Binder (also listed at the top of the repository). It may take some time to set up and congfigure the environment.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/calekochenour/machine-learning-tutorial/main)

## Run Scripts

### Manual - With Terminal Commands

These examples provide syntax to run all scripts, followed by expected outputs. Note that some results for `03_analyze_flowers.py` may differ slightly from what is provided here, due to inherent randomness within the classification algorithms.

Script 1 - check versions:

```commandline
(machine-learning-tutorial) > python 01_check_versions.py

Python: 3.10.13 | packaged by conda-forge | (main, Oct 26 2023, 18:01:37) [MSC v.1935 64 bit (AMD64)]
matplotlib: 3.8.1
numpy: 1.26.0
pandas: 2.1.3
scikit-learn/sklearn: 1.3.2
scipy: 1.11.3
```

Script 2 - check imports:

```commandline
(machine-learning-tutorial) > python 02_check_imports.py

SUCCESS: Imported packages without error.
```

Script 3 - analyze flowers:

```commandline
(machine-learning-tutorial) > python 03_analyze_flowers.py
Shape: (150, 5)

First 20 records: 
    sepal-length  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa

Summary: 
       sepal-length  sepal-width  petal-length  petal-width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

Class distribution: class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64


Algorithm evaluation:
LR: 0.942 (0.065)
LDA: 0.975 (0.038)
KNN: 0.958 (0.042)
CART: 0.933 (0.05)
NB: 0.95 (0.055)
SVM: 0.983 (0.033)


Algorithm prediction:
Accuracy score: 0.967
Confusion matrix:
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
classification report:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30
```

### Automated - With Makefile Recipes

To run the scripts in a more automated way, use the Makefile recipes. There is one recipie for each script. In addition, there is a recipe to run all three scripts in succession. Finally, there is a recipe to delete the figures created by the `03_analyze_flowers.py` script. Outputs are omitted from these commands, as they will match the outputs shown in the previous section.

Script 1 - check versions:

```commandline
(machine-learning-tutorial) > make versions
```

Script 2 - check imports:

```commandline
(machine-learning-tutorial) > make imports
```

Script 3 - analyze flowers:

```commandline
(machine-learning-tutorial) > make flowers
```

Run all three scripts in succession:

```commandline
(machine-learning-tutorial) > make all
```

or

```commandline
(machine-learning-tutorial) > make
```

Delete figures created by script 3 - analyze flowers (Windows):

```commandline
(machine-learning-tutorial) > make clean-windows
```

Delete figures created by script 3 - analyze flowers (Linux - for use with Binder):

```commandline
(machine-learning-tutorial) > make clean-linux
```
