import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Binarizer
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load the data
dataSet = pd.read_csv('/Users/tm033520/Documents/4830/Machine-Learning/dataset.csv')  # Load dataset from CSV file

# Data cleaning
# Drop rows with missing values / invalid data
dataSet = dataSet.dropna()  # Remove rows with missing values

# Feature Selection
# Drop columns with only one unique value
dataSet = dataSet.loc[:, dataSet.nunique() > 1]  # Keep columns with more than one unique value

# Feature Scaling
# Normalize the data on a scale of 0 to 1
scaler = MinMaxScaler()  # Initialize MinMaxScaler for feature normalization

# Encode labels
# Group types of cancer
label = LabelEncoder()  # Initialize LabelEncoder for encoding labels
# Rename the first column to Names
dataSet = dataSet.rename(columns={'Unnamed: 0': 'Names'})  # Rename the first column

# Remove numbers from names so we can classify them better
# Regular expression
dataSet['Names'] = dataSet['Names'].apply(lambda x: re.sub(r'\d+', '', x))  # Remove digits from Names column

# Transform the names into numbers
dataSet['Names'] = label.fit_transform(dataSet['Names'])  # Encode the Names column as numeric labels

# Split into features (X) and target variable (y)
X = dataSet.drop(columns=['Names'])  # Separate features from the target variable
# Scale X values
X = scaler.fit_transform(X)  # Normalize feature values
y = dataSet['Names']  # Target variable

# Feature Selection
# Select the top k features
k = 10000 # Number of top features to select
# mutual_info_classif https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html 
selector = SelectKBest(score_func=mutual_info_classif, k=k)  # Initialize SelectKBest with mutual information
X = selector.fit_transform(X, y)  # Select top k features
# Binarize the data
binarizer = Binarizer()
X = binarizer.fit_transform(X)

# Support Vector Machine
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Initialize the SVM model
svm = SVC()

# Implement Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
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
