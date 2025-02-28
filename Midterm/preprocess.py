import re

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# load genesset
print('loading genes...')
genes = pd.read_csv('../dataset.csv')
print('loading mutations...')
mutations: pd.DataFrame = pd.read_csv('../mutations.csv')

# encode targets
print('encoding gene targets...')
genes.rename(columns={'Unnamed: 0': 'targets'}, inplace=True)
genes['targets'] = genes['targets'].apply(lambda x: re.sub(r'\d+', '', x))
encoder = LabelEncoder()
genes['targets'] = encoder.fit_transform(genes['targets'])

print('encoding mutation targets...')
mutations.rename(columns={'Unnamed: 0': 'targets'}, inplace=True)
mutations['targets'] = mutations['targets'].apply(
    lambda x: re.sub(r'\d+', '', x))
mutations['targets'] = LabelEncoder().fit_transform(mutations['targets'])

# ensure equal samples for each class
print('resampling genes...')
class_counts = genes['targets'].value_counts().min()
genes = pd.concat([
    resample(genes[genes['targets'] == cls], replace=False,
             n_samples=class_counts, random_state=420)
    for cls in genes['targets'].unique()
])
print('resampling mutations...')
mutations = pd.concat([
    resample(mutations[mutations['targets'] == cls], replace=False,
             n_samples=class_counts, random_state=420)
    for cls in mutations['targets'].unique()
])

features = genes.drop('targets', axis=1)
targets = genes['targets']

# Apply SelectKBest
selector = SelectKBest(score_func=f_classif, k=300)
selected_features = selector.fit_transform(features, targets)

# Get the mask of selected features
mask = selector.get_support()

# Get the selected feature names
genes_features = features.columns[mask].tolist()

print('filtering mutations based off genes...')
print('number of mutations before: ', mutations.shape[1])
# Filter mutations to keep only the columns that match the selected gene features
pattern = '|'.join(genes_features)
mutations_filtered = mutations.loc[:, mutations.columns.str.contains(pattern)]
print('number of mutations after: ', mutations_filtered.shape[1])

print('saving mutations to csv...')
mutations_filtered.index = mutations['targets']
mutations_filtered.to_csv('filtered_mutations.csv', index=True)
