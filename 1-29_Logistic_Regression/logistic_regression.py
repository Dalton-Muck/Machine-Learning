# Dalton Muck
# CS 4830 Activity #3

import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
dataSet = pd.read_csv('/Users/tm033520/Documents/4830/dataset.csv')


# Save the first column (Patient ID) for later
#first_column = dataSet.iloc[:, 0]  # Select the first column

# Drop the first column for numeric computations
#dataSet = dataSet.drop(dataSet.columns[0], axis=1)

# Data cleaning
# Drop rows with missing values / invalid data
dataSet = dataSet.dropna()

# Feature Selection
# Drop columns with only one unique value
dataSet = dataSet.loc[:, dataSet.nunique() > 1]

# Feature Scaling
# Normalize the data on a scale of 0 to 1
scaler = MinMaxScaler()
#dataSet = pd.DataFrame(scaler.fit_transform(dataSet), columns=dataSet.columns)




# Encode labels (assuming the target column name is 'diagnosis')
label = LabelEncoder()
#Rename
dataSet = dataSet.rename(columns = { 'Unnamed: 0': 'Names' })

#remove digits from names so we can classify them better
#regular expression
dataSet['Names'] = dataSet['Names'].apply(lambda x: re.sub(r'\d+', '', x))

#transform the names into numbers
dataSet['Names'] = label.fit_transform(dataSet['Names'])


# Split into features (X) and target variable (y)
X = dataSet.drop(columns = ['Names'])
#scale X values
X = scaler.fit_transform(X)
y = dataSet['Names']

# Split data into training and testing sets
# takes portion of X and Y to train and test
#"seed" random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data (sigmoid function happens here)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# # Add the first column back if desired
# dataSet.insert(0, first_column.name, first_column)

# # Output the processed DataFrame with Patient IDs
# print(dataSet)
