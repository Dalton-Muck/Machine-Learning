# Description: This script loads the Iris dataset, calculates the maximum, minimum, and average values for the first two features of each class, and then visualizes the dataset using a scatter plot.
# for retreicing the dataset
from sklearn import datasets
# for plotting
import matplotlib.pyplot as plt
# for numerical operations
import numpy as np
#sepal is the first part of the flower to form!
# Load the Iris dataset
iris_data = datasets.load_iris()

# Using sklearn
# Separate the data points based on their class labels
setosa_data = iris_data.data[iris_data.target == 0]
versicolor_data = iris_data.data[iris_data.target == 1]
virginica_data = iris_data.data[iris_data.target == 2]

# Initialize a dictionary to store statistics for each class
class_stats = {}

# Calculate max, min, and average for each class (first two features)
# Using Numpy
for idx, (flower_name, flower_data) in enumerate(
    zip(iris_data.target_names, [setosa_data, versicolor_data, virginica_data])
):
    class_stats[flower_name] = {
        "maximum": np.max(flower_data[:, :2], axis=0),
        "minimum": np.min(flower_data[:, :2], axis=0),
        "average": np.mean(flower_data[:, :2], axis=0),
    }

# Print the statistics for each flower class
for flower_name, values in class_stats.items():
    print(f"Flower Class: {flower_name}")
    print(f"  Maximum (in inches): {values['maximum'] * 0.393701}")
    print(f"  Minimum (in inches): {values['minimum'] * 0.393701}")
    print(f"  Average (in inches): {values['average'] * 0.393701}")
    print()

# Create the scatter plot
fig, plot_axis = plt.subplots()
scatter_plot = plot_axis.scatter(
    iris_data.data[:, 0] * 0.393701,
    iris_data.data[:, 1] * 0.393701,
    c=iris_data.target,
    cmap="viridis",
)

# Set axis labels, title, and legend
plot_axis.set(
    xlabel="Sepal Length (inches)",
    ylabel="Sepal Width (inches)",
    title="Iris Dataset Visualization",
)

#plot 2D graph
plot_axis.legend(
    scatter_plot.legend_elements()[0],
    iris_data.target_names,
    loc="lower right",
    title="Flower Classes",
)

# Display the plot
plt.show()
