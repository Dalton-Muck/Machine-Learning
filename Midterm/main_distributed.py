import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Binarizer
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as pltjo
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.offline as pyo
import joblib
import time
import itertools
from dask.distributed import Client

print('Connecting to scheduler')
client = Client("tcp://odd01.cs.ohio.edu:8786")
print('Connected to scheduler')

csv_files = [
    './data/mutations_chi2_300.csv',
    './data/mutations_f_classif_300.csv',
    './data/mutations_f_regression_300.csv',
    './data/mutations_mutual_info_classif_300.csv',
    './data/mutations_mutual_info_regression_300.csv',
]

best_model = None
best_accuracy = 0
best_params = None
best_report = None
best_conf_matrix = None

param_grid = [
    {
        'C': [.05, 0.1, 1],
        'kernel': ['poly'],
        'gamma': ['auto', .05, 0.1, 0.01],
        'degree': [2, 3, 4],  # Removed 'auto' since 'degree' requires an integer
        'coef0': [2.0, 3.0, 4.0],
        'tol': [0.0001]
    },
    {
        'C': [.1, 1],
        'kernel': ['sigmoid'],
        'gamma': ['scale', 'auto', .001, .01, .1],
        'coef0': [0.0, 1.0],
        'tol': [0.0001]
    },
    {
        'C': [.01, .1, 1],
        'kernel': ['rbf'],
        'tol': [0.0001],
        'gamma': ['scale', 'auto'],
        'cache_size': [500],
        'decision_function_shape': ['ovo', 'ovr']
    },
    {
        'C': [.01, .1, 1, 10],
        'kernel': ['linear'],
        'tol': [0.0001]
    }
]

# Compute total parameter combinations
cv_folds = 3
num_combinations = sum(len(list(itertools.product(*grid.values()))) for grid in param_grid)
total_iterations = num_combinations * cv_folds
print(f"Total parameter combinations: {num_combinations}")
print(f"Cross-validation folds: {cv_folds}")
print(f"Total iterations (param_combinations * cv): {total_iterations}")

for csv_file in csv_files:
    print(f"\nProcessing {csv_file}...")
    
    # Load data
    dataSet = pd.read_csv(csv_file)
    
    # Split into features (X) and target variable (y)
    X = dataSet.drop(columns=['targets'])
    y = dataSet['targets']
    KVAL = 3000
    featureSelection = SelectKBest(k=KVAL).fit_transform(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(featureSelection, y, test_size=0.2)

    # Initialize the SVM model
    svm = SVC()

    # Run timed grid search
    start_time = time.time()
    print('Running Grid Search')
    with joblib.parallel_backend('dask'):
        grid_search = GridSearchCV(svm, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Grid search took {elapsed_time:.2f} seconds")

    # Best parameters from grid search
    best_params_current = grid_search.best_params_
    print(f'Best parameters: {best_params_current}')

    # Train the SVM model with best parameters
    best_svm_current = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_svm_current.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['brca', 'prad', 'luad'], output_dict=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Compare and store the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = best_svm_current
        best_params = best_params_current
        best_report = report
        best_conf_matrix = conf_matrix

# Close Dask client
client.close()

# Output stats for the best model
print("\nBest Model Overall:")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print(f"Best Parameters: {best_params}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['brca', 'prad', 'luad']))

print("\nConfusion Matrix:")
print(best_conf_matrix)

#Create a figure for the radar chart
plot = go.Figure()

brcaData = [report['brca']['precision'], report['brca']['recall'], report['brca']['f1-score']]
pradData = [report['prad']['precision'], report['prad']['recall'], report['prad']['f1-score']]
luadData = [report['luad']['precision'], report['luad']['recall'], report['luad']['f1-score']]

#Add a radar graph of brca
plot.add_trace(go.Scatterpolar(
    r = brcaData,
    theta = ['precision', 'recall', 'f1-score'],
    fill = 'toself',
    name = 'brca'
))

#Add a radar graph of prad
plot.add_trace(go.Scatterpolar(
    r = pradData,
    theta = ['precision', 'recall', 'f1-score'],
    fill = 'toself',
    name = 'prad'
))

#Add a radar graph of luad
plot.add_trace(go.Scatterpolar(
    r = luadData,
    theta = ['precision', 'recall', 'f1-score'],
    fill = 'toself',
    name = 'luad'
))

#Update the figure to have an appropriate format
plot.update_layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
            range= [0, 1]
        )),
    showlegend = True
)

#Show the radar graph
plot.show()


print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
