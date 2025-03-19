import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Binarizer
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import os
# Load the data
dataSet = pd.read_csv('/Users/tm033520/Documents/4830/Machine-Learning/dataset.csv')  # Load dataset from CSV file

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

# Split into features (X)
X = dataSet.drop(columns=['Names'])  # Separate features from the target variable

# Feature Selection
# Create training data for different numbers of top features
feature_counts = [10, 100, 500, 1000, 10000]  # List of feature counts to select
training_data = {}  # Dictionary to store training data for each feature count

for k in feature_counts:
    selector = SelectKBest(score_func=mutual_info_classif, k=k)  # Initialize SelectKBest with mutual information
    X_selected = selector.fit_transform(X, dataSet['Names'])  # Select top k features
    training_data[k] = X_selected  # Store the training data for the current feature count

# K-Means Clustering for each feature count
for k, X_selected in training_data.items():
    print(f"\nEvaluating K-Means Clustering for top {k} features:")
    
    # Initialize the K-Means model
    n_clusters = 3  # Set the number of clusters (adjust based on your dataset)
    kmeans = KMeans(n_clusters=n_clusters, random_state=35)
    
    # Fit the model
    kmeans.fit(X_selected)
    
    # Get cluster labels
    cluster_labels = kmeans.labels_
    
    # Evaluate the clustering using various metrics
    silhouette_avg = silhouette_score(X_selected, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_selected, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_selected, cluster_labels)
    rand_index = adjusted_rand_score(dataSet['Names'], cluster_labels)
    adjusted_rand_index = adjusted_rand_score(dataSet['Names'], cluster_labels)
    mutual_info = mutual_info_classif(dataSet['Names'].values.reshape(-1, 1), cluster_labels).mean()
    normalized_mutual_info = normalized_mutual_info_score(dataSet['Names'], cluster_labels)
    
    # Store evaluation metrics in a dictionary
    metrics = {
        'Features': k,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin,
        'Rand Index': rand_index,
        'Adjusted Rand Index': adjusted_rand_index,
        'Mutual Information': mutual_info,
        'Normalized Mutual Information': normalized_mutual_info,
        'Silhouette Score': silhouette_avg
    }
    
    # Append metrics to a DataFrame for tabular display
    if 'results_df' not in locals():
        results_df = pd.DataFrame(columns=metrics.keys())
    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)

import matplotlib.pyplot as plt

# Create and display line graphs for each metric
# Directory to save the plots
output_dir = '/Users/tm033520/Documents/4830/Machine-Learning/3-19_Clustering/plots'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

metrics_to_plot = [
    'Calinski-Harabasz Index',
    'Davies-Bouldin Index',
    'Rand Index',
    'Adjusted Rand Index',
    'Mutual Information',
    'Normalized Mutual Information',
    'Silhouette Score'
]

for metric in metrics_to_plot:
    plt.figure()
    plt.plot(results_df['Features'], results_df[metric], marker='o', label=metric)
    plt.title(f'{metric} vs Number of Features')
    plt.xlabel('Number of Features (k)')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'{metric.replace(" ", "_")}_vs_Features.png')
    plt.savefig(plot_path)
    plt.close() 
