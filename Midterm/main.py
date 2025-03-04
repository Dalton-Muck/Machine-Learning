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

# Load the data
dataSet = pd.read_csv('../mutations/mutations_chi2_300.csv')  # Load dataset from CSV file
#dataSet = pd.read_csv('../genes_f_classif_300.csv')  # Load dataset from CSV file


# Select the best features

# Split into features (X) and target variable (y)
X = dataSet.drop(columns = ['targets'])  # Separate features from the target variable
y = dataSet['targets']  # Target variable
KVAL = 3000
featureSelection = SelectKBest(k=KVAL).fit_transform(X, y)

# Support Vector Machine
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(featureSelection, y, test_size=0.2)

# Initialize the SVM model
svm = SVC()

# Implement Grid Search
# param_grid = {
#     'C': [.05, 0.1, 1],
#     'kernel': ['poly'],
#     'gamma': ['auto', .05, 0.1, 0.01],
#     # only for poly
#     'degree': ['auto', 2, 3, 4],
#     'coef0': [2.0, 3.0, 4.0],
#     'tol': [ 0.0001]
# }

# param_grid = {
#     'C': [.1, 1],
#     'kernel': ['sigmoid'],
#     'gamma': ['scale', 'auto', .001, .01, .1],
#     'coef0': [0.0, 1.0],
#     'tol': [0.0001]
# }

RBF_param_grid = {
    'C': [.01, .1, 1],
    'kernel': ['rbf'],
    'tol' : [0.0001],
    'gamma' : ['scale', 'auto'],
    'cache_size' : [500],
    'decision_function_shape' : ['ovo', 'ovr']
}

Linear_param_grid = {
    'C': [.01, .1, 1, 10],
    'kernel': ['linear'],
    'tol' : [0.0001],
}

grid_search = GridSearchCV(svm, RBF_param_grid, cv=3, scoring='accuracy', n_jobs= -1)
grid_search.fit(X_train, y_train)

print(KVAL)
# Best parameters from grid search
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')


# Train the SVM model with best parameters
best_svm = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

#Make a classification report to use to output the data
print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names= ['brca', 'prad', 'luad']))
report = classification_report(y_test, y_pred, target_names= ['brca', 'prad', 'luad'], output_dict=True)

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
