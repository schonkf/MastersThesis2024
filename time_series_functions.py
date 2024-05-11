#!/usr/bin/env python
# coding: utf-8

# In[51]:

import os
import pandas as pd
import numpy as np
from datetime import timedelta

def resample_participant_data(all_participants_data, resample_dfs):
    for participant, participant_data in all_participants_data.items():
        resampled_participant_data = {}
        for dataframe_name, dataframe in participant_data.items():
            if dataframe_name in resample_dfs.keys():
                data_to_be_resampled = participant_data[dataframe_name]
                resample_rate = resample_dfs[dataframe_name]
                if data_to_be_resampled.index.duplicated().any():
                    data_to_be_resampled = data_to_be_resampled[~data_to_be_resampled.index.duplicated(keep='first')]
                resampled_data = data_to_be_resampled.resample(resample_rate).ffill()
                resampled_participant_data[dataframe_name] = resampled_data
            else:
                resampled_participant_data[dataframe_name] = dataframe
        all_participants_data[participant] = resampled_participant_data
    return all_participants_data


# In[2]:


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


# In[3]:


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


# In[4]:


def calculate_statistics_per_second(dataframe):
    def agg_func(x):
        stats = {}
        for col, m, s in zip(x.columns, x.mean(), x.std()):
            stats[f"{col}_mean"] = m
            stats[f"{col}_std"] = s
        stats['size'] = len(x)  # Add size calculation outside the loop
        return pd.Series(stats)

    resampled_df = dataframe.resample('S').agg(agg_func)
    return resampled_df

def calculate_stats_for_all(all_participants_data, resample_dfs):
    for participant, participant_data in all_participants_data.items():
        resampled_participant_data = {}
        for dataframe_name, dataframe in participant_data.items():
            if dataframe_name in resample_dfs:
                data_to_be_resampled = participant_data[dataframe_name]
                resamppled_df = calculate_statistics_per_second(data_to_be_resampled)
                resampled_participant_data[dataframe_name] = resamppled_df
            else:
                resampled_participant_data[dataframe_name] = dataframe
        all_participants_data[participant] = resampled_participant_data
    return all_participants_data


# In[10]:


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

def merge_dataframes_across_timestamps_for_participant(participant_data):
    # Find the earliest and latest timestamps across all dataframes for the participant
    timestamps = []
    for df in participant_data.values():
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
    for df in participant_data.values():
        if isinstance(df.index, pd.DatetimeIndex):
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='left')
    
    return merged_df

def merge_dataframes_for_all_participants(all_participants_data):
    # Round timestamps to the nearest second for specified dataframes
    file_names_to_round = ['ActivityEvent.csv', 'ActivityTransition.csv', 'AppUsageEvent.csv', 'CallEvent.csv', 'MessageEvent.csv', 'WiFi']  # Adjust as needed
    all_participants_data = round_timestamps(all_participants_data, file_names_to_round)

    merged_all_participants_data = {}

    # Iterate over each participant
    for participant, participant_data in all_participants_data.items():
        # Merge dataframes for the participant across timestamps
        merged_dataframe = merge_dataframes_across_timestamps_for_participant(participant_data)
        # Store the merged dataframe for the participant
        merged_all_participants_data[participant] = merged_dataframe

    return merged_all_participants_data


# In[102]:


#MessageEvent.csv
def extract_message_time_series(df, windows, timestamp):
    windowed_messageevent = {}
    if df.empty:
        for i in range(49):  # Assuming 49 timesteps
            windowed_messageevent[i] = [0, 0, 0, 0, 0]
        return windowed_messageevent

    before_esm = df[df.index <= timestamp]
    timestamp = pd.Timestamp(timestamp)
    start_time = timestamp - pd.Timedelta(hours=8)
    end_time = timestamp
    max_timestamp = max(before_esm.index.min(), timestamp - pd.Timedelta(hours=8))

    sequence_number = 0
    current_time = end_time  # Start from the end_time
    while current_time >= max_timestamp:  # Iterate backwards
        window_start = current_time - pd.Timedelta(minutes=10)
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index < current_time)]

        unique_numbers_outgoing = []
        unique_numbers_incoming = []
        messages_outgoing = 0
        messages_incoming = 0 
        unique_messegers_outgoing = 0
        unique_messegers_incoming = 0 
        total_messages = 0 
        for message_time, message_type, number, in windowed_data[['messageBox', 'number']].itertuples(index=True):
            total_messages += 1
            if message_type == 'SENT':
                messages_outgoing += 1
                if number not in unique_numbers_outgoing:
                    unique_messegers_outgoing += 1
                    unique_numbers_outgoing.append(number)
            elif message_type == 'INBOX':
                messages_incoming += 1
                if number not in unique_numbers_incoming:
                    unique_messegers_incoming += 1
                    unique_numbers_incoming.append(number)
        
        windowed_messageevent[sequence_number] = [messages_outgoing, messages_incoming, unique_messegers_outgoing, unique_messegers_incoming, total_messages]
        
        current_time -= pd.Timedelta(minutes=10)  # Decrement current_time
        sequence_number += 1

    return windowed_messageevent


# In[104]:


def extract_deviceevent_time_series(df, windows, timestamp):
    windowed_deviceevent = {}
    if df.empty:
        for i in range(49):  # Assuming 49 timesteps
            windowed_deviceevent[i] = [0, 0]
        return windowed_deviceevent

    before_esm = df[df.index <= timestamp]
    timestamp = pd.Timestamp(timestamp)
    start_time = timestamp - pd.Timedelta(hours=8)
    end_time = timestamp
    max_timestamp = max(before_esm.index.min(), timestamp - pd.Timedelta(hours=8))

    sequence_number = 0
    current_time = end_time  # Start from the end_time
    while current_time >= max_timestamp:  # Iterate backwards
        window_size = windows['10min']  # Retrieve window size from the windows dictionary
        window_start = current_time - pd.Timedelta(minutes=10)
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index < current_time)]

        times_unlocked = 0
        time_spent_on_phone = 0
        unlock_time = None  # Variable to store the timestamp of the last unlock event
        for event_time, event_type in windowed_data[['type']].itertuples(index=True):
            if event_type == 'UNLOCK':
                times_unlocked += 1
                unlock_time = event_time  # Update the unlock time
            elif event_type == 'SCREEN_OFF' and unlock_time is not None:
                # Calculate the time spent on phone by subtracting unlock time from screen off time
                time_spent_on_phone += (event_time - unlock_time).total_seconds()
                unlock_time = None  # Reset the unlock time
        proportion_time_spent_on_phone = time_spent_on_phone / window_size
        # Store the calculated values in the windowed_deviceevent dictionary
        windowed_deviceevent[current_time] = [times_unlocked, proportion_time_spent_on_phone]
        
        current_time -= pd.Timedelta(minutes=10)  # Decrement current_time
        sequence_number += 1

    return windowed_deviceevent


# In[105]:


def entropy(labels):
    n_labels = len(labels)
    
    if n_labels <= 1:
        return 0
    
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy

def extract_appusage_time_series(df, windows, timestamp):
    windowed_appevent = {}
    if df.empty:
        for i in range(49):  # Assuming 49 timesteps
            windowed_appevent[i] = []
        return windowed_appevent
    
    before_esm = df[df.index <= timestamp]
    timestamp = pd.Timestamp(timestamp)
    end_time = timestamp
    max_timestamp = max(before_esm.index.min(), timestamp - pd.Timedelta(hours=8))
    
    sequence_number = 0
    current_time = end_time  # Start from the end_time
    while current_time >= max_timestamp:  # Iterate backwards
        window_size = windows['10min']  # Retrieve window size from the windows dictionary
        window_start = current_time - pd.Timedelta(minutes=10)
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index <= current_time)]
        
        if len(windowed_data) == 0:
            windowed_appevent[sequence_number] = [np.nan] * 4
        else:
            # Find the top 5 categories
            top_categories = windowed_data['category'].value_counts().head(5).index
            most_common_category = windowed_data['category'].mode().iloc[0]
            
            # Calculate the time spent on each app category
            app_category_time_spent = {}
            app_category_count = {}
            for category in top_categories:
                category_data = windowed_data[windowed_data['category'] == category]
                move_to_foreground_indices = category_data[category_data['type'] == 'MOVE_TO_FOREGROUND'].index
                move_to_background_indices = category_data[category_data['type'] == 'MOVE_TO_BACKGROUND'].index

                category_time_spent = 0
                for foreground_index in move_to_foreground_indices:
                    next_background_index = min(move_to_background_indices[move_to_background_indices > foreground_index], default=None)
                    if next_background_index is not None:
                        category_time_spent += (next_background_index - foreground_index).total_seconds()

                app_category_time_spent[category] = category_time_spent
                app_category_count[category] = len(category_data)

            # Calculate the entropy of the app category distribution
            app_category_entropy = entropy(windowed_data['category'])
        
            # Store the calculated values in the windowed_appevent dictionary
            sequence_data = []
            for category, time_spent in app_category_time_spent.items():
                sequence_data.extend([time_spent / 60, app_category_count[category]])
            
            # Add entropy and most common category to the sequence data
            sequence_data.extend([app_category_entropy, most_common_category])
            
            windowed_appevent[sequence_number] = sequence_data
        
        current_time -= pd.Timedelta(minutes=10)  # Decrement current_time
        sequence_number += 1

    return windowed_appevent


# In[107]:


def extract_call_timeseries(df, windows, timestamp):
    windowed_callevent = {}

    # Check if the DataFrame is empty
    if df.empty:
        # If the DataFrame is empty, return a dictionary with zeros
        for i in range(49):  # Assuming 49 timesteps
            windowed_callevent[i] = [0, 0, 0]
        return windowed_callevent

    # Process the DataFrame if it's not empty
    before_esm = df[df.index <= timestamp]
    timestamp = pd.Timestamp(timestamp)
    start_time = timestamp - pd.Timedelta(hours=8)
    end_time = timestamp
    max_timestamp = max(before_esm.index.min(), timestamp - pd.Timedelta(hours=8))

    sequence_number = 0
    current_time = end_time  # Start from the end_time
    while current_time >= max_timestamp:  # Iterate backwards
        window_start = current_time - pd.Timedelta(minutes=10)
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index < current_time)]

        unique_callers_outgoing = []
        unique_callers_incoming = []
        time_spent_calling = 0

        for call_time, number, call_type, duration in windowed_data[['number', 'type', 'duration']].itertuples(index=True):
            time_spent_calling += duration / 60  # Accumulate duration in minutes
            if call_type == 'OUTGOING' and number not in unique_callers_outgoing:
                unique_callers_outgoing.append(number)
            elif call_type == 'INCOMING' and number not in unique_callers_incoming:
                unique_callers_incoming.append(number)

        windowed_callevent[current_time] = [len(unique_callers_outgoing), len(unique_callers_incoming), time_spent_calling]

        current_time -= pd.Timedelta(minutes=10)  # Decrement current_time
        sequence_number += 1

    return windowed_callevent


# In[99]:


#Location.csv
def calculate_entropy(cluster_counts):
    total_time = cluster_counts.sum()
    cluster_proportions = cluster_counts / total_time
    entropy = -np.sum([p * np.log2(p) for p in cluster_proportions.values if p != 0])
    return entropy

def calculate_normalised_entropy(cluster_counts):
    if len(cluster_counts) == 0:
        return np.nan
        
    total_time = cluster_counts.sum()
    cluster_proportions = cluster_counts / total_time
    entropy = -np.sum([p * np.log2(p) for p in cluster_proportions.values if p != 0])
    
    max_entropy = -np.log2(1 / len(cluster_counts))
    
    if max_entropy == 0 or np.isnan(entropy):
        return np.nan
    
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy

def extract_location_time_series(df, windows, timestamp):
    windowed_location = {}
    if df.empty:
        for i in range(49):  # Assuming 49 timesteps
            windowed_location[i] = [np.nan, np.nan, np.nan]
        return windowed_location

    before_esm = df[df.index <= timestamp]
    timestamp = pd.Timestamp(timestamp)
    start_time = timestamp - pd.Timedelta(hours=8)
    end_time = timestamp
    max_timestamp = max(before_esm.index.min(), timestamp - pd.Timedelta(hours=8))
    
    sequence_number = 0
    current_time = end_time  # Start from the end_time
    while current_time >= max_timestamp:  # Iterate backwards
        window_start = current_time - pd.Timedelta(minutes=10)
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index < current_time)]
        
        if len(windowed_data) == 0:
            most_common_cluster = np.nan
            window_entropy = np.nan
            window_normalised_entropy = np.nan
        else:
            cluster_counts = windowed_data['cluster'].value_counts()
            most_common_cluster = windowed_data['cluster'].mode().iloc[0]
            window_entropy = calculate_entropy(cluster_counts)
            window_normalised_entropy = calculate_normalised_entropy(cluster_counts)
        
        windowed_location[current_time] = [most_common_cluster, window_entropy, window_normalised_entropy]
        
        current_time -= pd.Timedelta(minutes=10)  # Decrement current_time
        sequence_number += 1

    return windowed_location


# In[106]:


def generic_entropy(data):
    value_counts = data.value_counts()
    probabilities = value_counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def extract_generic_time_series(df, windows, timestamp):
    windowed_features_dict = {}
    if df.empty:
        # If the DataFrame is empty, return a dictionary with zeros
        for i in range(49):  # Assuming 49 timesteps
            windowed_features_dict[i] = [0, 0, 0, 0]
        return windowed_features_dict

    before_esm = df[df.index <= timestamp]
    timestamp = pd.Timestamp(timestamp)
    start_time = timestamp - pd.Timedelta(hours=8)
    end_time = timestamp
    max_timestamp = max(before_esm.index.min(), timestamp - pd.Timedelta(hours=8))
    
    sequence_number = 0
    current_time = end_time  # Start from the end_time
    while current_time >= max_timestamp:  # Iterate backwards
        window_start = current_time - pd.Timedelta(minutes=10)
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index < current_time)]
        
        windowed_features = []  # Initialize list to store statistics for each column
        numeric_cols = windowed_data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            col_mean = windowed_data[col].mean()
            col_median = windowed_data[col].median()
            col_std = windowed_data[col].std()
            col_entropy = generic_entropy(windowed_data[col])
            windowed_features.extend([col_mean, col_median, col_std, col_entropy])  # Extend the list with statistics
            
        windowed_features_dict[current_time] = windowed_features
        current_time -= pd.Timedelta(minutes=10)  # Decrement current_time
        sequence_number += 1

    return windowed_features_dict


# In[111]:


# def sequence_creation(all_participants_data, esm_responses, user_info):
#     desired_structure = {}
#     days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#     window = {
#         '10min': 60 * 10
#     }

#     for participant_id, participant_data in all_participants_data.items():
#         participant_sequences = {}
#         sleep_proxies = participant_data['sleep_proxies']

#         participant_esm_responses = esm_responses[esm_responses['Pcode'] == participant_id]
#         for index, esm_response in participant_esm_responses.iterrows():
#             timestamp = esm_response['ResponseTime']
#             timestamp = pd.Timestamp(timestamp)
#             sequence_name = f"{timestamp}_{index}"

#             day_of_week_index = timestamp.weekday()
#             day_of_week = days_of_week[day_of_week_index]
#             p_code = esm_response['Pcode']
#             stress = esm_response['Stress_binary']
#             valence = esm_response['Valence_binary']
#             arousal = esm_response['Arousal_binary']
#             user_info_row = user_info[user_info['Pcode'] == participant_id].iloc[0]
#             age = user_info_row['Age']
#             gender = user_info_row['Gender']
#             openness = user_info_row['Openness']
#             conscientiousness = user_info_row['Conscientiousness']
#             neuroticism = user_info_row['Neuroticism']
#             extraversion = user_info_row['Extraversion']
#             agreeableness = user_info_row['Agreeableness']
#             pss10 = user_info_row['PSS10']
#             phq9 = user_info_row['PHQ9']
#             ghq12 = user_info_row['GHQ12']     
        
#             target_value = [stress, valence, arousal]
            
#             # Initialize an empty list to hold the features for each timestep
#             timestep_features = []

#             for dataframe_name, dataframe in participant_data.items():
#                 if dataframe_name == 'CallEvent.csv':
#                     # Call the external function to get the timestep features
#                     timestep_features1 = call_timeseries(dataframe, window, timestamp)  

#                     # Append the timestep features to the list
#                     for i in range(49):
#                         timestep_features.append(timestep_features1.get(i, [0, 0, 0]))

#             # Add static features to each timestep
#             static_features = [day_of_week, age, gender, openness, conscientiousness, neuroticism, agreeableness, pss10, phq9, ghq12]
#             for i in range(49):
#                 timestep_features[i] = static_features + timestep_features[i]

#             # Create the sequence data dictionary
#             sequence_data = {'features': timestep_features, 'target': target_value}

#             # Add sequence data to participant sequences
#             participant_sequences[sequence_name] = sequence_data

#         # Add participant sequences to desired structure
#         desired_structure[participant_id] = participant_sequences

#     return desired_structure


def sequence_creation(all_participants_data, esm_responses, user_info):
    desired_structure = {}
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    window = {
        '10min': 60 * 10
    }
    
    # Define a list of tuples with external function names and corresponding DataFrame names
    external_functions = [
        (extract_generic_time_series, 'Calorie.csv'),
        (extract_generic_time_series, 'SkinTemperature.csv'),  # Example: Add more functions and DataFrame names here
        (extract_generic_time_series, 'AmbientLight.csv'),
        (extract_generic_time_series, 'RRI.csv'),
        (extract_generic_time_series, 'StepCount.csv'),
        (extract_message_time_series, 'MessageEvent.csv'),
        (extract_call_timeseries, 'CallEvent.csv'),
        (extract_generic_time_series, 'ActivityEvent.csv'),
        (extract_location_time_series, 'Location.csv'),
        (extract_generic_time_series, 'HR.csv'),
        (extract_generic_time_series, 'Distance.csv'),
        (extract_appusage_time_series, 'AppUsageEvent.csv'),
        (extract_generic_time_series, 'Acceleration.csv'),
        (extract_generic_time_series, 'UltraViolet.csv'),
        (extract_deviceevent_time_series, 'DeviceEvent.csv')
        
        # Add more tuples as needed for additional external functions
    ]

    for participant_id, participant_data in all_participants_data.items():
        participant_sequences = {}
        sleep_proxies = participant_data['sleep_proxies']

        participant_esm_responses = esm_responses[esm_responses['Pcode'] == participant_id]
        for index, esm_response in participant_esm_responses.iterrows():
            timestamp = esm_response['ResponseTime']
            timestamp = pd.Timestamp(timestamp)
            sequence_name = f"{timestamp}_{index}"

            day_of_week_index = timestamp.weekday()
            day_of_week = days_of_week[day_of_week_index]
            p_code = esm_response['Pcode']
            stress = esm_response['Stress_binary']
            valence = esm_response['Valence_binary']
            arousal = esm_response['Arousal_binary']
            user_info_row = user_info[user_info['Pcode'] == participant_id].iloc[0]
            age = user_info_row['Age']
            gender = user_info_row['Gender']
            openness = user_info_row['Openness']
            conscientiousness = user_info_row['Conscientiousness']
            neuroticism = user_info_row['Neuroticism']
            extraversion = user_info_row['Extraversion']
            agreeableness = user_info_row['Agreeableness']
            pss10 = user_info_row['PSS10']
            phq9 = user_info_row['PHQ9']
            ghq12 = user_info_row['GHQ12']     
        
            target_value = [stress, valence, arousal]
            
            # Initialize an empty list to hold the features for each timestep
            timestep_features = [[] for _ in range(49)]

            # Iterate through each external function and its corresponding DataFrame
            for func, dataframe_name in external_functions:
                dataframe = participant_data.get(dataframe_name)  # Get the DataFrame
                if dataframe is not None:
                    # Call the external function to get the timestep features
                    timestep_features_func = func(dataframe, window, timestamp)  

                    # Append the timestep features to the list
                    for i in range(49):
                        timestep_features[i].extend(timestep_features_func.get(i, [0, 0, 0]))

            # Add static features to each timestep
            static_features = [day_of_week, age, gender, openness, conscientiousness, neuroticism, agreeableness, pss10, phq9, ghq12]
            for i in range(49):
                timestep_features[i] = static_features + timestep_features[i]

            # Create the sequence data dictionary
            sequence_data = {'features': timestep_features, 'target': target_value}

            # Add sequence data to participant sequences
            participant_sequences[sequence_name] = sequence_data

        # Add participant sequences to desired structure
        desired_structure[participant_id] = participant_sequences

    return desired_structure


# In[ ]:




