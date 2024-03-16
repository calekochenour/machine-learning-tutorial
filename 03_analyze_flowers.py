"""
-------------------------------------------------------------------------------
 Machine learning tutorial analyzing iris flowers. This script provides
 the full workflow.

 Adapted from:
 https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

 Steps include:
   - Environment setup
   - Data loading
   - Data summary
   - Data visualization
   - Algorithm evaluation
   - Algorithm prediction

 Information about accuracy scores.

 Link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

 precision ~ errors of commission
   - 'The precision is intuitively the ability of the classifier not to label a
      negative sample as positive.'

 recall ~ errors of omission
   - 'The recall is intuitively the ability of the classifier to find all the
      positive samples.'
-------------------------------------------------------------------------------
"""

# -------------------------1.0 ENVIRONMENT SETUP----------------------------- #
from matplotlib.pyplot import boxplot, close, savefig, title
from pathlib import Path
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# -------------------------2.0 DATA LOADING---------------------------------- #
# Load dataset with column names
columns = [
    'sepal-length',
    'sepal-width',
    'petal-length',
    'petal-width',
    'class',
]
dataframe = read_csv(
    filepath_or_buffer=Path('data/iris.csv'), names=columns, header=None
)


# -------------------------3.0 DATA SUMMARY---------------------------------- #
# Review dataframe
print(f'Shape: {dataframe.shape}\n')
print(f'First 20 records: \n{dataframe.head(20)}\n')
print(f'Summary: \n{dataframe.describe()}\n')
print(f'Class distribution: {dataframe.groupby("class").size()}')
print('\n')


# -------------------------4.0 DATA VISUALIZATION---------------------------- #
# Visualize data to see trends
# Box plot
dataframe.plot(
    kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False
)
savefig(Path('figures/figure-01-box-plot.png'), dpi=300)
close()

# Histogram
dataframe.hist()
savefig(Path('figures/figure-02-histogram.png'), dpi=300)
close()

# Scatter matrix
scatter_matrix(dataframe)
savefig(Path('figures/figure-03-scatter-matrix.png'), dpi=300)
close()


# -------------------------5.0 ALGORITHM EVALUATION-------------------------- #
# Extract all data to array
array = dataframe.values
# Isolate numerical values (x values: all rows, all but last/5th column)
x_values = array[:, 0:-1]
# Isolate flower/class names (y values: all rows, last/5th column only)
y_values = array[:, -1]
# Split data into 80% for training, 20% for validation/testing
x_train, x_validation, y_train, y_validation = train_test_split(
    x_values, y_values, test_size=0.20, random_state=1, shuffle=True
)

# Spot check algorithms
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto')),
]

# Evaluate each model
results = []
column_names = []
print('Algorithm evaluation:')
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, x_train, y_train, cv=kfold, scoring='accuracy'
    )
    results.append(cv_results)
    column_names.append(name)
    print(
        f'{name}: {round(cv_results.mean(), 3)} ({round(cv_results.std(), 3)})'
    )
print('\n')

# Compare algorithms
boxplot(results, labels=column_names)
title('Algorithm Comparison')
savefig(Path('figures/figure-04-spot-check.png'), dpi=300)
close()


# -------------------------6.0 ALGORITHM PREDICTION-------------------------- #
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# Evaluate predictions
print('Algorithm prediction:')
print(f'Accuracy score: {round(accuracy_score(y_validation, predictions), 3)}')
print(f'Confusion matrix:\n{confusion_matrix(y_validation, predictions)}')
print(
    f'classification report:\n{classification_report(y_validation, predictions)}'
)
