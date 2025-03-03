import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Binarizer
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load the data
dataSet = pd.read_csv('../genes_mutual_info_classif_300.csv')  # Load dataset from CSV file

# Separate features from the target variable
X = dataSet.drop(columns=['targets'])
y = dataSet['targets']

# #Select the 300 best features
# selector = SelectKBest(mutual_info_classif, k=3000)
# X_selected = selector.fit_transform(X, y)

# Binazier
# binarizer = Binarizer()
# X_selected = binarizer.fit_transform(X)

# Split into training and testing sets using the selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=35)

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
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs= -1)
grid_search.fit(X_train, y_train)

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
