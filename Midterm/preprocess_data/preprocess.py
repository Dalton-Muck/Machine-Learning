import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                       f_regression, mutual_info_classif,
                                       mutual_info_regression)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

print('loading genes...')
genes = pd.read_csv('../dataset.csv')

print('loading mutations...')
mutations_targets = np.genfromtxt(
    '../mutations.csv',
    delimiter=',',
    names=True,
    dtype=None,
    usecols=[0]
)
max_features = np.genfromtxt(
    '../mutations.csv',
    delimiter=',',
    max_rows=1,
).shape[0]
mutations_features = np.genfromtxt(
    '../mutations.csv',
    delimiter=',',
    dtype=None,
    names=True,
    usecols=range(1, max_features)
)
mutations = pd.DataFrame(
    mutations_features,
    columns=mutations_features.dtype.names
)
mutations_targets = pd.DataFrame(
    mutations_targets,
    columns=mutations_targets.dtype.names
)
mutations = pd.concat([mutations_targets, mutations], axis=1)

# encode targets
print('encoding gene targets...')
genes.rename(columns={'Unnamed: 0': 'targets'}, inplace=True)
genes['targets'] = genes['targets'].apply(lambda x: re.sub(r'\d+', '', x))
encoder = LabelEncoder()
genes['targets'] = encoder.fit_transform(genes['targets'])

print('encoding mutation targets...')
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
classif_functions = [
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
]

for func in classif_functions:
    print(f'Applying {func.__name__} k=300')
    selector = SelectKBest(score_func=func, k=300)
    selector.fit_transform(features, targets)
    filtered_genes = features.columns[selector.get_support()].tolist()
    print('filtering mutations based off genes...')
    print('number of mutations before: ', mutations.shape[1] - 1)
    # Filter mutations to keep only the columns that match the selected gene features
    pattern = '|'.join(filtered_genes)
    mutations_filtered = mutations.loc[:,
                                       mutations.columns.str.contains(pattern)]
    mutations_filtered = mutations_filtered.loc[:,
                                                mutations_filtered.nunique() > 1]
    print('number of mutations after: ', mutations_filtered.shape[1])

    print('saving mutations to csv...')
    mutations_filtered.index = mutations['targets']
    mutations_filtered.to_csv(
        f'../mutations_{func.__name__}_300.csv',
        index=True
    )
