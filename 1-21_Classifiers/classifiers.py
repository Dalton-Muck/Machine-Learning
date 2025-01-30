#Dalton Muck
#CS 4830 Activity #2

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
dataSet = pd.read_csv('/Users/tm033520/Documents/4830/dataset.csv')

# Save the first column (Patient ID) for later
first_column = dataSet.iloc[:, 0]  # Select the first column

# Drop the first column for numeric computations
dataSet = dataSet.drop(dataSet.columns[0], axis=1)

# Data cleaning
# Drop rows with missing values / invalid data
dataSet = dataSet.dropna()

#feature Selection
# Drop columns with only one unique value
dataSet = dataSet.loc[:, dataSet.nunique() > 1]

#Feature Scaling
# Normalize the data on a scale of 0 to 1
scaler = MinMaxScaler()
dataSet = pd.DataFrame(scaler.fit_transform(dataSet), columns=dataSet.columns)

# Add the first column back if desired
#dataSet.insert(0, first_column.name, first_column)

# Output the processed DataFrame
print(dataSet)




