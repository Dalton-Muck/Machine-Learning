import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # For scaling features and encoding labels
import re  # For handling regular expressions
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.neighbors import KNeighborsClassifier  # For k-NN model
from sklearn.metrics import accuracy_score  # For evaluating model performance
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning and cross-validation
from sklearn.feature_selection import SelectKBest, f_classif  # For selecting the most relevant features

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

# Select the k best features
features = 10  # Adjust k based on the desired number of features
selector = SelectKBest(f_classif, k = features)  # Initialize SelectKBest with ANOVA F-value
# ANOVA F-value measures the linear dependency between the feature and the target variable
# https://datascience.stackexchange.com/questions/74465/how-to-understand-anova-f-for-feature-selection-in-python-sklearn-selectkbest-w 
X = selector.fit_transform(X, y)  # Reduce dataset to top-k features

# Split data into training and testing sets
# Takes portion of X and y to train and test
# "seed" random state
# test_size = size of the test set; 1 - test_size = size of the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets

# Logistic Regression Model
logistic_model = LogisticRegression()  # Initialize logistic regression model
# Fit the model on the training data (sigmoid function happens here)
logistic_model.fit(X_train, y_train)  # Train logistic regression model

# Make predictions using logistic regression
y_pred_logistic = logistic_model.predict(X_test)  # Predict using logistic regression model

# Evaluate logistic regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)  # Compute accuracy of logistic regression model
print("Logistic Regression Accuracy:", accuracy_logistic)  # Print logistic regression accuracy

# Define a parameter grid
param_grid = {
    'n_neighbors': [i for i in range(1, features, 2)],  # Different values for number of neighbors
    'weights': ['uniform', 'distance'],  # Weighting schemes for k-NN
    'metric': ['euclidean', 'manhattan']  # Distance metrics for k-NN
}


# Perform GridSearchCV to find the best hyperparameters
# https://dev.to/anurag629/gridsearchcv-in-scikit-learn-a-comprehensive-guide-2a72#:~:text=GridSearchCV%20works%20by%20defining%20a,combination%20is%20called%20cross%2Dvalidation 
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=8, scoring='accuracy')  # Initialize grid search for k-NN
grid_search.fit(X_train, y_train)  # Fit grid search on training data

# Get the best parameters and evaluate on test set
best_knn = grid_search.best_estimator_  # Retrieve the best k-NN model

y_pred_knn = best_knn.predict(X_test)  # Predict using the best k-NN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)  # Compute accuracy of k-NN model

print("Best Parameters for k-NN:", grid_search.best_params_)  # Print best hyperparameters for k-NN
print("k-NN Accuracy: ", accuracy_knn)  # Print k-NN accuracy with best parameters