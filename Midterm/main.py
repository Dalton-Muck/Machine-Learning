import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Binarizer
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load the data
dataSet = pd.read_csv('/Users/tm033520/Documents/4830/Machine-Learning/genes_f_classif_300.csv')  # Load dataset from CSV file
# Split into features (X) and target variable (y)
X = dataSet.drop(columns = ['targets'])  # Separate features from the target variable
y = dataSet['targets']  # Target variable

# Support Vector Machine
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Initialize the SVM model
svm = SVC()

# Implement Grid Search
param_grid = {
    'C': [10],
    'kernel': ['sigmoid'],
    'gamma': ['auto']
}
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
