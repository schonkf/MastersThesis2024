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


# In[2]:


def read_csv_files_in_participant_folders(main_folder_path):
    all_participants_data = {}
    for participant_folder in os.listdir(main_folder_path):
        participant_folder_path = os.path.join(main_folder_path, participant_folder)
        
        # Check if it's a directory
        if os.path.isdir(participant_folder_path):
            # Read CSV files in participant folder
            participant_data = read_csv_files_in_folder(participant_folder_path)
            all_participants_data[participant_folder] = participant_data
    
    return all_participants_data


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
    if 'ResponseTime' in df.columns and df['ResponseTime'].dtype == 'int64':
        for column in timestamp_columns:
            if column in df.columns:
                datetime_objs = pd.to_datetime(df[column], unit='ms')
                datetime_korean = datetime_objs.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
                df[column] = datetime_korean
    return df


# In[7]:


def convert_to_korean_time_general(df, timestamp_columns):
    for column in timestamp_columns:
        if column in df.columns:
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


# In[14]:


def drop_columns(dataframe, columns_to_drop):
    return dataframe.drop(columns=columns_to_drop, errors='ignore')

