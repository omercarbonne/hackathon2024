import csv
import random

import numpy as np
import pandas as pd
from datetime import datetime


def time_to_seconds(t):
    """
    Convert a time string in HH:MM:SS format to seconds since the start of the day.
    """
    if pd.isna(t) or t.strip() == '':
        return None
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

     # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def calculate_dist(trip):
    sum_dist = 0
    trip_sorted = trip.sort_values(by='station_index').reset_index(drop=True)
    for i in range(len(trip_sorted) - 1):
        row1 = trip_sorted.iloc[i]
        row2 = trip_sorted.iloc[i + 1]
        lon1, lat1 = row1['longitude'], row1['latitude']
        lon2, lat2 = row2['longitude'], row2['latitude']
        sum_dist += haversine(lat1, lon1, lon2, lat2)
    return sum_dist

def take_x_percent(df_path, num : int) -> str: 
    # initialising
    x_precent_df_path = "precent_"+str(num)+"_of_data"+df_path
    percentage = num / 100
    # start the process
    with open(df_path, mode='r', newline='', encoding="ISO-8859-8") as input_file:
        csv_reader = csv.reader(input_file)
        header = next(csv_reader)
        # Calculate the target number of rows to process
        total_rows = sum(1 for _ in csv_reader)
        rows_to_process = int(total_rows * percentage)
        # Reset the file pointer to the beginning of the file and skip the header again
        input_file.seek(0)
        next(csv_reader)
        # Reservoir sampling
        sample = []
        for i, row in enumerate(csv_reader):
            if i < rows_to_process:
                sample.append(row)
            else:
                j = random.randint(0, i)
                if j < rows_to_process:
                    sample[j] = row
    # Write the sampled rows to the output file
    with open(x_precent_df_path, mode='w', newline='', encoding="ISO-8859-8") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header)
        for row in sample:
            csv_writer.writerow(row)    
    return x_precent_df_path

def prepare_data(group):
    num_of_stations = group['station_index'].max()
    first_station_exist = group['station_index'].min() # should be 1
    dep_time = group.loc[group['station_index'] == first_station_exist, 'arrival_time'].values[0]
    arr_time = group.loc[group['station_index'] == num_of_stations, 'arrival_time'].values[0]
    total_time_seconds = time_to_seconds(arr_time) - time_to_seconds(dep_time)
    total_time = total_time_seconds / 60
    total_passengers = sum(group['passengers_up'])
    cluster = group['cluster'].values[0]
    line_id = group['line_id'].values[0]
    total_dist = calculate_dist(group)
    res = pd.DataFrame({
        'trip_id_unique': group['trip_id_unique'].values[0],  # Take the first value, assuming all are the same
        'total_time': [total_time],
        'num_of_stations': [num_of_stations],
        'total_passengers': [total_passengers],
        'line_id': [line_id],
        'departure_time': [dep_time],
        'cluster': [cluster],
        'total_distance': [total_dist]})
    return res
def preprocess_train_task_1(df_path: str) -> str:
    """
    preprocess training data.
    Parameters
    ----------
    df_path: String
        the loaded data path
    preprocess_df_path: preprocess data path

    Returns
    -------
    A clean, preprocessed version of the data in a new csv file
    """





    preprocess_df_path = "preprocess_"+df_path

    # Read the data into a DataFrame to calculate mean gap
    df = pd.read_csv(df_path, encoding="ISO-8859-8")

    # remove non relevant columns
    columns_to_keep = ['trip_id_unique',
                       'station_index',
                       'arrival_time',
                       'passengers_up',
                       'cluster',
                       'line_id',
                       'longitude',
                       'latitude']
    # change the cluster to categorical number
    df_filtered = df[columns_to_keep]
    clusters_list = list()
    for c in df_filtered['cluster']:
        if c not in clusters_list:
            clusters_list.append(c)
    df_filtered['cluster'] = df_filtered['cluster'].apply(lambda x: clusters_list.index(x))
    # makes sure the time format is ok, if not set to NaT (NaN)
    #df_filtered['arrival_time'] = pd.to_datetime(df_filtered['arrival_time'], format='%H:%M:%S', errors='coerce')
    # removes all the rows containing NaN values or empty cells
    df_filtered = df_filtered.dropna()
    df_filtered.reset_index()

    grouped = df_filtered.groupby("trip_id_unique").apply(prepare_data).reset_index(drop=True)
    grouped.to_csv(preprocess_df_path, index=False, encoding="ISO-8859-8")



if __name__ == '__main__':
    # part_df = "precent_5_of_datatrain_bus_schedule.csv"
    part_df = "train_bus_schedule (1).csv"
    preprocess_df_path = preprocess_train_task_1(part_df)