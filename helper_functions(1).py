#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd
import numpy as np
from datetime import timedelta
import pygeohash as geo
import scipy.spatial.distance as dist


# In[5]:


def read_csv_files_in_folder(folder_path):
    data_frames = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            data_frames[file_name] = df
    return data_frames


# In[1]:


def read_csv_files_in_participant_folders(main_folder_path):
    all_participants_data = {}
    
    # Iterate over participant folders
    for participant_folder in os.listdir(main_folder_path):
        participant_folder_path = os.path.join(main_folder_path, participant_folder)
        
        # Check if it's a directory
        if os.path.isdir(participant_folder_path):
            # Read CSV files in participant folder
            participant_data = read_csv_files_in_folder(participant_folder_path)
            all_participants_data[participant_folder] = participant_data
    
    return all_participants_data


# In[2]:


def resample_multiple_sensors(df, sensor_columns, desired_frequency):
    # Resample each sensor separately and concatenate the results
    resampled_dfs = []
    for sensor_column in sensor_columns:
        resampled_df = df[[sensor_column]].resample(desired_frequency).mean()
        resampled_dfs.append(resampled_df)

    # Concatenate resampled DataFrames along the columns axis
    resampled_df = pd.concat(resampled_dfs, axis=1)

    return resampled_df


# In[3]:


def round_timestamps(data_frames, file_names_to_round, timestamp_column='timestamp'):
    for file_name in file_names_to_round:
        if file_name in data_frames:
            df = data_frames[file_name]
            if timestamp_column in df.columns:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column]).dt.round('1s')
                data_frames[file_name] = df
            else:
                print(f"Warning: Timestamp column '{timestamp_column}' not found in DataFrame '{file_name}'.")
        else:
            print(f"Warning: DataFrame '{file_name}' not found in data_frames.")
    return data_frames


# In[1]:


def set_index_to_timestamp(all_participants_data, timestamp_column='timestamp'):
    for participant_id, participant_data in all_participants_data.items():
        for dataframe_name, df in participant_data.items():
            if timestamp_column in df.columns:
                df.set_index(timestamp_column, inplace=True)
                all_participants_data[participant_id][dataframe_name] = df
            else:
                print(f"Warning: Timestamp column '{timestamp_column}' not found in DataFrame '{dataframe_name}' of participant '{participant_id}'. Index not set.")
    return all_participants_data


# In[6]:


def merge_dataframes_across_timestamps(data_frames):
    # Find the earliest and latest timestamps across all dataframes
    timestamps = []
    for df in data_frames.values():
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps.extend(df.index)
    if not timestamps:
        print("No timestamp index found in any dataframe.")
        return None
    earliest_timestamp = min(timestamps)
    latest_timestamp = max(timestamps)
    
    # Create a new dataframe with timestamps spanning from earliest to latest
    all_timestamps = pd.date_range(start=earliest_timestamp, end=latest_timestamp, freq='1s')
    full_df = pd.DataFrame(index=all_timestamps)
    
    # Merge the original dataframes with the full dataframe
    merged_df = full_df
    for key, df in data_frames.items():
        if isinstance(df.index, pd.DatetimeIndex):
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='left')
    
    return merged_df


# In[1]:


def drop_columns_in_participant_data(all_participants_data, columns_to_drop_per_df):
    modified_participants_data = {}
    
    # Iterate over each participant's data
    for participant_id, participant_data in all_participants_data.items():
        modified_data = {}
        
        # Iterate over each DataFrame in the participant's data
        for df_name, df in participant_data.items():
            # Drop specified columns
            columns_to_drop = columns_to_drop_per_df.get(df_name, [])
            modified_df = df.drop(columns=columns_to_drop, errors='ignore')
            modified_data[df_name] = modified_df
        
        modified_participants_data[participant_id] = modified_data
    
    return modified_participants_data


# In[2]:


def convert_to_korean_time(df, timestamp_columns):
    for column in timestamp_columns:
        datetime_objs = pd.to_datetime(df[column], unit='ms')
        datetime_korean = datetime_objs.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        df[column] = datetime_korean

    return df


# In[3]:


def convert_to_korean_time_all(all_participants_data, timestamp_columns):
    for participant_data in all_participants_data.values():
        for df_name, df in participant_data.items():
            if 'timestamp' in df.columns and df['timestamp'].dtype == 'int64':
                for column in timestamp_columns:
                    if column in df.columns:
                        datetime_objs = pd.to_datetime(df[column], unit='ms')
                        datetime_korean = datetime_objs.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
                        df[column] = datetime_korean
    return all_participants_data


# In[4]:


def extract_data_before_esm(all_participants_data, esm_responses, window_size_minutes):
    before_esm_data = {}

    # Iterate over each participant's first ESM response
    for participant_id, first_esm_response in esm_responses.groupby('Pcode').first().iterrows():
        # Get the timestamp of the first ESM response
        esm_timestamp = first_esm_response['ResponseTime']
        
        # Get the participant's data
        participant_data = all_participants_data.get(participant_id, {})
        
        # Iterate over each DataFrame in participant's data
        for df_name, df in participant_data.items():
            # Check if the dataframe has a timestamp column and it's in datetime format
            if not df.empty and 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                # Selecting only the data within the window size before the ESM timestamp
                window_end_time = esm_timestamp
                window_start_time = window_end_time - pd.Timedelta(minutes=window_size_minutes)
                data_within_window = df[(df['timestamp'] >= window_start_time) & (df['timestamp'] <= window_end_time)]
                before_esm_data.setdefault(participant_id, {})[df_name] = data_within_window.copy()

    return before_esm_data


# In[11]:


def calculate_static_features(windowed_data):
    static_features = {
        'mean': windowed_data.mean(),
        'std': windowed_data.std(),
        'min': windowed_data.min(),
        'max': windowed_data.max()
    }
    return static_features


# In[10]:


def extract_data_before_esm(all_participants_data, esm_responses, window_sizes):
    tabular_data = pd.DataFrame()

    # Iterate over each ESM response
    for _, esm_response in esm_responses.iterrows():
        participant_id = esm_response['Pcode']
        esm_timestamp = esm_response['ResponseTime']
        participant_data = all_participants_data.get(participant_id)

        participant_features = pd.DataFrame()

        # Iterate over each DataFrame in participant's data
        for df_name, df in participant_data.items():
            # Check if the dataframe has a timestamp column and it's in datetime format
            if not df.empty and 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                # Extract data before ESM response
                data_before_esm = df[df['timestamp'] < esm_timestamp]

                if not data_before_esm.empty:
                    # Calculate static features for the extracted data for each window size
                    for window_size_minutes in window_sizes:
                        window_start_time = esm_timestamp - pd.Timedelta(minutes=window_size_minutes)
                        windowed_data = data_before_esm[data_before_esm['timestamp'] >= window_start_time]

                        if not windowed_data.empty:
                            static_features = calculate_static_features(windowed_data)
                            # Add windowed static features to participant_features
                            participant_features[f"{df_name}_{window_size_minutes}_mean"] = static_features['mean']
                            participant_features[f"{df_name}_{window_size_minutes}_std"] = static_features['std']
                            participant_features[f"{df_name}_{window_size_minutes}_min"] = static_features['min']
                            participant_features[f"{df_name}_{window_size_minutes}_max"] = static_features['max']

        # Append participant features to tabular_data
        if not participant_features.empty:
            participant_features['participant_id'] = participant_id
            participant_features['esm_timestamp'] = esm_timestamp
            tabular_data = pd.concat([tabular_data, participant_features], ignore_index=True)

    return tabular_data


# In[14]:


def drop_columns(dataframe, columns_to_drop):
    return dataframe.drop(columns=columns_to_drop, errors='ignore')

