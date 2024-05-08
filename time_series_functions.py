#!/usr/bin/env python
# coding: utf-8

# In[51]:


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

