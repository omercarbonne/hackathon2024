import csv
from typing import NoReturn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_total_time_vs_num_of_passengers_colored(df_path: str):
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    
    # Define colors for clusters
    cluster_colors = {
        0: 'magenta', 1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5: 'orange',
        6: 'brown', 7: 'pink', 8: 'gray', 9: 'olive', 10: 'cyan'
    }
    
     # Plotting
    plt.figure(figsize=(10, 6))
    for cluster, color in cluster_colors.items():
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['total_passengers'], cluster_data['total_time'], color=color, label=f'Cluster {cluster}', alpha=0.6)

    plt.title('Total Time vs Number of Passengers (Colored by Clusters)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Passengers', fontsize=12)
    plt.ylabel('Total Time', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Total_Time_vs_Number_of_Passengers.png")

def plot_total_time_vs_num_of_stations(df_path: str):
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df['num_of_stations'], df['total_time'], color='skyblue', alpha=0.6)
    plt.title('Total Time vs Number of Stations', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Stations', fontsize=12)
    plt.ylabel('Total Time', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Total_Time_vs_Number_of_Stations.png")

def plot_total_time_mean_by_cluster(df_path: str):
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    
    cluster_mapping = ["אונו-אלעד", "בת ים-רמת גן", "שרון חולון מרחבי", "דרומי-ראשלצ-חולון", 
                       "מזרחי-רמת גן", "פת-תא", "דרומי-בת ים", "השרון", 
                       "חולון עירוני ומטרופוליני+תחרות חולון", "תל אביב", "מזרחי-בני ברק"]

    # Map cluster numbers to names
    df['cluster_name'] = df['cluster'].map(lambda x: cluster_mapping[x])

    # Group by 'cluster_name' and calculate mean of 'total_time'
    cluster_means = df.groupby('cluster_name')['total_time'].mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_means['cluster_name'], cluster_means['total_time'], color='skyblue')
    plt.title('Mean Total Time by Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Mean Total Time', fontsize=12)
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig("Mean_Total_Time_by_Cluster.png")

def plot_correlations(df_path: str):
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    
    # Filter out columns 'trip_id_unique' and 'departure_time'
    columns_to_exclude = ['trip_id_unique', 'departure_time']
    numeric_columns = [col for col in df.columns if col not in columns_to_exclude]
    # Calculate correlations
    correlations = df[numeric_columns].corr()['total_time'].drop('total_time')  # Exclude total_time itself
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(correlations.index, correlations.values, color='skyblue')
    ax.set_title('Correlation with Total Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.grid(axis='y')
    ax.axhline(0, color='gray', linewidth=0.5)  # Add horizontal line at zero
    # Adjust x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig("Correlation_with_Total_Time.png")

def plot_unique_trip_id_part_by_cluster(df_path: str):
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")

    # Mapping list
    cluster_mapping = ["", "אונו-אלעד", "בת ים-רמת גן", "שרון חולון מרחבי", "דרומי-ראשלצ-חולון", 
                       "מזרחי-רמת גן", "פת-תא", "דרומי-בת ים", "השרון", 
                       "חולון עירוני ומטרופוליני+תחרות חולון", "תל אביב", "מזרחי-בני ברק"]

    # Create a dictionary from the mapping list
    cluster_dict = {str(i): cluster_mapping[i] for i in range(len(cluster_mapping))}
    
    # Replace cluster numbers with names
    df['cluster'] = df['cluster'].astype(str).map(cluster_dict)
    
    # Check for unmapped clusters
    if df['cluster'].isnull().any():
        print("Unmapped clusters found:")
        print(df[df['cluster'].isnull()]['cluster'].unique())
    
    # Drop rows with unmapped clusters
    df = df.dropna(subset=['cluster'])
    
    # Create a new column for the (trip_id, part) pair
    df['trip_id_part'] = df['trip_id'].astype(str) + '_' + df['part'].astype(str)
    
    # Group by 'cluster' and count unique 'trip_id_part'
    unique_trip_part_counts = df.groupby('cluster')['trip_id_part'].nunique().reset_index()
    
    # Sort values by trip_id_part for better visualization
    unique_trip_part_counts = unique_trip_part_counts.sort_values(by='trip_id_part', ascending=False)
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    plt.bar(unique_trip_part_counts['cluster'], unique_trip_part_counts['trip_id_part'], color='skyblue')
    plt.title('Number of Unique Trip ID-Part Pairs by Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Unique Trip ID-Part Pairs', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for better readability
    plt.grid(axis='y')
    plt.tight_layout()  # Adjust layout to fit everything properly
    plt.savefig("Number_of_Unique_Trip_ID-Part_Pairs_by_Cluster.png")

def minutes_to_time(minutes):
    # Convert minutes to hours and minutes
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def plot_passengers_by_arrival_time(df_path: str):
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    
    # Convert 'arrival_time' from minutes after midnight to 'HH:MM' format
    df['arrival_time'] = df['arrival_time'].apply(minutes_to_time)
    
    # Convert 'arrival_time' to datetime and extract the hour
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M').dt.hour
    
    # Group by 'arrival_time' and sum 'passengers_up'
    passenger_counts = df.groupby('arrival_time')['passengers_up'].sum().reset_index()
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(passenger_counts['arrival_time'], passenger_counts['passengers_up'], marker='o')
    plt.title('Number of Passengers Up by Arrival Time')
    plt.xlabel('Hour of Arrival Time')
    plt.ylabel('Number of Passengers Up')
    plt.grid(True)
    plt.xticks(range(0, 24))  # Set x-axis ticks for each hour
    plt.show()

def minutes_to_time(minutes):
    # Convert minutes to hours and minutes
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def feature_evaluation(df_path: str):
    cluster_mapping = ["","אונו-אלעד","בת ים-רמת גן","שרון חולון מרחבי","דרומי-ראשלצ-חולון","מזרחי-רמת גן","פת-תא","דרומי-בת ים","השרון","חולון עירוני ומטרופוליני+תחרות חולון","תל אביב","מזרחי-בני ברק"]
    # Load the data
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    # Create a dictionary from the mapping list
    cluster_dict = {str(i): cluster_mapping[i] for i in range(len(cluster_mapping))}
    # Replace cluster numbers with names
    df['cluster'] = df['cluster'].astype(str).map(cluster_dict)
    # Check for unmapped clusters
    if df['cluster'].isnull().any():
        print("Unmapped clusters found:")
        print(df[df['cluster'].isnull()]['cluster'].unique())
    # Drop rows with unmapped clusters
    df = df.dropna(subset=['cluster'])
    # Group by 'cluster' and sum 'passengers_up'
    passenger_counts = df.groupby('cluster')['passengers_up'].sum().reset_index()
    # Sort values by passengers_up for better visualization
    passenger_counts = passenger_counts.sort_values(by='passengers_up', ascending=False)
    # Plot the data
    plt.figure(figsize=(12, 8))
    plt.bar(passenger_counts['cluster'], passenger_counts['passengers_up'], color='skyblue')
    plt.title('Number of Passengers Up by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Passengers Up')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("Number_of_Passengers_Up_by_Cluster.png")


if __name__ == '__main__':
    plot_total_time_vs_num_of_passengers_colored("preprocess_train_bus_schedule.csv")