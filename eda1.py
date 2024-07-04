import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


def feature_evaluation(data_path):
    # load the data
    data = pd.read_csv(data_path, encoding='ISO-8859-8')

    # Calculate the mean and median of passengers_up
    mean_passengers_up = data['passengers_up'].mean()
    median_passengers_up = data['passengers_up'].median()

    print(f"Mean of passengers_up: {mean_passengers_up}")
    print(f"Median of passengers_up: {median_passengers_up}")

    return data


def plot_feature_correlation(data, curr_feature):
    """
    Plot the correlation between a given feature and passengers_up.

    Parameters:
    - data: DataFrame containing the dataset
    - curr_feature: str, name of the feature column to correlate with passengers_up
    """

    correlation, _ = pearsonr(data[curr_feature], data['passengers_up'])

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[curr_feature], y=data['passengers_up'])

    plt.title(f'Correlation between {curr_feature} and passengers_up\nPearson Correlation: {correlation:.2f}')
    plt.xlabel(curr_feature)
    plt.ylabel('passengers_up')
    plt.savefig(f'pearson_passengers_up_{curr_feature}.png')
    plt.close()


def plot_categorical_correlation(data, categorical_feature):
    """
    Plot the correlation between a categorical feature and a numerical feature using box plot and violin plot.

    Parameters:
    - data: DataFrame containing the dataset
    - categorical_feature: str, name of the categorical feature
    - numerical_feature: str, name of the numerical feature
    """
    numerical_feature = "passengers_up"
    # Group by the categorical feature and sum the numerical feature
    grouped_data = data.groupby(categorical_feature)[numerical_feature].sum().reset_index()

    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categorical_feature, y=numerical_feature, hue=categorical_feature, data=grouped_data, palette="viridis", dodge=False, legend=False)
    plt.title(f'Sum of {numerical_feature} by {categorical_feature}')
    plt.xlabel(categorical_feature)
    plt.ylabel(f'Sum of {numerical_feature}')
    plt.savefig(f'sum_{categorical_feature}_{numerical_feature}.png')
    plt.close()


def plot_hexbin_scatter(data):
    plt.figure(figsize=(12, 8))
    # Create the hexbin plot
    hb = plt.hexbin(data['longitude'], data['latitude'], C=data['passengers_up'], gridsize=50, cmap='viridis_r',
                    reduce_C_function=np.sum, vmin=0, vmax=50)

    # Add color bar to show intensity of passengers_up
    cbar = plt.colorbar(hb)
    cbar.set_label('Number of Passengers Up')

    # Set plot title and labels
    plt.title('Hexbin Plot of Passengers Up by Location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save and show the plot
    plt.savefig('hexbin_passengers_up.png')
    plt.close()


def plot_correlation(data):
    # Define colors for direction 1 and 2
    colors = {1: 'b', 2: 'g'}
    labels = {1: 'Direction 1', 2: 'Direction 2'}

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the scatter plot
    for direction in [1, 2]:
        subset = data[data['direction'] == direction]
        ax.scatter(subset['arrival_time'], subset['passengers_up'],
                   c=colors[direction], label=labels[direction], alpha=0.6, edgecolors='w', s=30)

    # Set plot title and labels
    ax.set_title('Scatter Plot of Arrival Time, Passengers Up, and Direction')
    ax.set_xlabel('Arrival Time (minutes)')
    ax.set_ylabel('Passengers Up')

    # Add legend
    ax.legend(title='Direction')

    # Show the plot
    plt.savefig('passengers_up.png')
    plt.close()


if __name__ == '__main__':
    data = feature_evaluation("preprocess_precent_5_of_datatrain_bus_schedule.csv")

    # plots the correlation per feature

    plot_categorical_correlation(data, 'direction')
    plot_hexbin_scatter(data)
    plot_correlation(data)

