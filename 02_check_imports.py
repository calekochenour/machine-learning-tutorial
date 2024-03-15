"""
-------------------------------------------------------------------------------
 Machine learning tutorial analyzing iris flowers. This script checks
 that all packages necessary for the full workflow import correctly.
 
 Adapted from:
 https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
-------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    try:
        from matplotlib.pyplot import boxplot, close, savefig, title
        from pandas import read_csv
        from pandas.plotting import scatter_matrix
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as error:
        print(f'\nERROR: {error}')
    else:
        print('\nSUCCESS: Imported packages without error.')
