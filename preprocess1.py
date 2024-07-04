import csv
import random
import pandas as pd
from datetime import datetime

cluster_map = dict()
cluster_counter = list()

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

def time_to_seconds(t):
    """
    Convert a time string in HH:MM:SS format to seconds since the start of the day.
    """
    if pd.isna(t) or t.strip() == '':
        return None
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(s):
    """
    Convert a number of seconds since the start of the day to a time string in HH:MM:SS format.
    """
    if s is None:
        return None
    h = s // 3600
    s %= 3600
    m = s // 60
    s %= 60
    return f"{int(h):02}:{int(m):02}:{int(s):02}"

def fix_cluster(row):
    if row[7] in cluster_map:
        row[7] = cluster_map[row[7]]
    else:
        cluster_map[row[7]] = cluster_counter[0]
        row[7] = cluster_map[row[7]]
        cluster_counter[0]+=1
    return row

def fix_part(row):
    valid_row = True
    if(row[1]=='א'):
        row[1] = 1
    if(row[1]=='ב'):
        row[1] = 2
    if(row[1]=='ג'):
        row[1] = 3
    else:
        valid_row = False
    return row, valid_row

def fix_arrival_time(row, valid_row, arrival_mean_seconds):
    arrival_time_col_index = 11
    # Check if arrival_time is in time format
    if row[arrival_time_col_index] is not None and row[arrival_time_col_index].strip() != '':
        try:
            row[arrival_time_col_index] = datetime.strptime(row[arrival_time_col_index], '%H:%M:%S').strftime('%H:%M:%S')
            row[arrival_time_col_index] = time_to_seconds(row[arrival_time_col_index])
        except ValueError:
            valid_row = False
    else:
        row[arrival_time_col_index] = arrival_mean_seconds
    return row, valid_row

def fix_door_time(row,valid_row,door_mean_gap_minuets):
    # Column indices
    arrival_time_col_index = 11
    door_closing_time_col_index = 12
    # check if door_closing_time is in time format and not null
    if row[door_closing_time_col_index] is not None and row[door_closing_time_col_index].strip() != '':
        try:
            row[door_closing_time_col_index] = datetime.strptime(row[door_closing_time_col_index], '%H:%M:%S').strftime('%H:%M:%S')
            # change the door_closing_time to the gap of seconds
            door_closing_time_seconds = time_to_seconds(row[door_closing_time_col_index])
            arrival_time_seconds = row[arrival_time_col_index]
            row[door_closing_time_col_index] = max(0,(door_closing_time_seconds - arrival_time_seconds) // 60)
            row[door_closing_time_col_index] = min(5,row[door_closing_time_col_index])
            row[arrival_time_col_index] = max(0,row[arrival_time_col_index] // 60)
        except ValueError:
            valid_row = False
    else:
        row[door_closing_time_col_index] = door_mean_gap_minuets
        row[arrival_time_col_index] = max(0,row[arrival_time_col_index] // 60)
    return row, valid_row

def fix_row_task1(row,door_mean_gap_seconds ,arrival_mean_seconds):
    row = fix_cluster(row)
    row, valid_row = fix_part(row)
    row, valid_row = fix_arrival_time(row,valid_row,arrival_mean_seconds)
    row, valid_row = fix_door_time(row,valid_row,door_mean_gap_seconds) 
    return row, valid_row

def delete_row_task1(row,test):
    # remove staion id and name
    del row[10]
    del row[9]
    # remove the alternative column
    del row[6]
    # remove the trip id unique station and trip id unique
    del row[3]
    if not test:
        del row[2]
    return row

def preprocess_row_task_1(row ,door_mean_gap_seconds ,arrival_mean_seconds,test):
    row, valid_row = fix_row_task1(row,door_mean_gap_seconds,arrival_mean_seconds)
    row = delete_row_task1(row,test)
    return row, valid_row

def preprocess_train_task_1(df_path: str, test=False) -> str:
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
    df['arrival_time_seconds'] = df['arrival_time'].apply(time_to_seconds)
    df['door_closing_time_seconds'] = df['door_closing_time'].apply(time_to_seconds)
    df['time_gap'] = df['door_closing_time_seconds'] - df['arrival_time_seconds']
    arrival_mean_seconds = max(0,df['arrival_time_seconds'].dropna().mean())
    door_mean_gap_minuets = max(0,(df['time_gap'].dropna().mean()) // 60)
    door_mean_gap_minuets = min(5,door_mean_gap_minuets)

    # initialise cluster counter
    cluster_counter.append(1)

    # process the data
    with open(df_path, mode='r', newline='', encoding="ISO-8859-8") as input_file, \
    open(preprocess_df_path, mode='w', newline='',encoding="ISO-8859-8") as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)
        # write the header to the output file if needed
        header = next(csv_reader)
        del header[10]
        del header[9]
        del header[6]
        del header[3]
        if not test:
            del header[2]
        csv_writer.writerow(header)
        # Process the data
        for row in csv_reader:
            processed_row, valid_row = preprocess_row_task_1(row,door_mean_gap_minuets,arrival_mean_seconds,test)
            if(valid_row):
                csv_writer.writerow(processed_row)
    return preprocess_df_path

def preprocess_text_task_1(df_path: str):
    preprocess_df = preprocess_train_task_1(df_path,True)
    df = pd.read_csv(preprocess_df)
    X,y = df.drop("trip_id_unique_station", axis=1), df.trip_id_unique_station
    X.to_csv("test_to_model.csv", index=False, encoding="ISO-8859-8")
    y.to_csv("xais.csv", index=False, encoding="ISO-8859-8")
    return "test_to_model.csv", "xais.csv"

def preprocess_train_task_1_base_line(df_path: str) -> str:
    preprocess_df_path = "baseline_preprocess_"+df_path
    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    # delete null rows
    df = df.dropna()
    # delete all colums except trip_id_unique_station, passengers_continue, arrival_is_estimated,mekadem_nipuach_luz,passengers_continue_menupach
    delete_colum = {"trip_id","part","trip_id_unique_station","trip_id_unique","line_id","direction","alternative","cluster","station_index","station_id","station_name","arrival_time","door_closing_time","latitude","longitude"}
    for colum in delete_colum:
        df=df.drop(colum,axis=1)
    df.to_csv(preprocess_df_path, index=False, encoding="ISO-8859-8")
    return preprocess_df_path

#if __name__ == '__main__':
#    part_df = "precent_5_of_datatrain_bus_schedule.csv"
#    X_path = preprocess_train_task_1(part_df)