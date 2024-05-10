#!/usr/bin/env python
# coding: utf-8



# In[1]:

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import pygeohash as geo
import scipy.spatial.distance as dist


def acceleration_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        acceleration_df = participant_data.get('Acceleration.csv')
        if acceleration_df is None:
            print(f"No acceleration data found for participant {participant_id}")
            continue
        
        # Calculate magnitude and add it as a new column
        acceleration_df['mag'] = np.sqrt(np.square(acceleration_df['X']) + np.square(acceleration_df['Y']) + np.square(acceleration_df['Z']))


# In[2]:


def wifi_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        wifi_df = participant_data.get('WiFi.csv')
        if wifi_df is None:
            print(f"No WiFi data found for participant {participant_id}")
            continue
        
        # Sort by timestamp
        wifi_df.sort_values(by='timestamp', inplace=True)

        # Calculate features
        wifi_df['bssid'] = wifi_df['bssid'].astype(str) + '-' + wifi_df['frequency'].astype(str)
        wifi_df['prev_bssid'] = wifi_df.groupby('bssid')['bssid'].shift(1)  # Create prev_bssid column
        wifi_df['prev_rssi'] = wifi_df.groupby('bssid')['rssi'].shift(1)
        wifi_df['intersect'] = wifi_df.apply(lambda row: len(np.intersect1d([row['bssid']], [row['prev_bssid']])), axis=1)
        wifi_df['union'] = wifi_df.apply(lambda row: len(np.union1d([row['bssid']], [row['prev_bssid']])), axis=1)
        wifi_df['w'] = 1 / wifi_df['intersect']
        wifi_df['cosine'] = wifi_df.apply(lambda row: 1 - dist.cosine([row['prev_rssi']], [row['rssi']]) if row['intersect'] > 0 else 0, axis=1)
        wifi_df['euclidean'] = wifi_df.apply(lambda row: 1 / (1 + dist.euclidean([row['prev_rssi']], [row['rssi']], [row['w']])) if row['intersect'] > 0 else 0, axis=1)
        wifi_df['manhattan'] = wifi_df.apply(lambda row: 1 / (1 + dist.cityblock([row['prev_rssi']], [row['rssi']], [row['w']])) if row['intersect'] > 0 else 0, axis=1)
        wifi_df['jaccard'] = wifi_df['intersect'] / wifi_df['union'] if (wifi_df['union'] > 0).any() else 0

        # Drop unnecessary columns
        wifi_df.drop(columns=['prev_bssid', 'prev_rssi', 'intersect', 'union', 'w', 'rssi', 'bssid', 'ssid', 'frequency'], inplace=True)


# In[3]:


#StepCount.csv
def step_count_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'StepCount.csv' in participant_data:
            data = participant_data['StepCount.csv'].sort_index(axis=0, level='timestamp').assign(
                steps=lambda x: (x['TotalSteps'] - x['TotalSteps'].shift(1))
            )
            participant_data['StepCount.csv'] = data


# In[4]:


#MessageEvent.csv
def messageevent_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'MessageEvent.csv' in participant_data:
            data = participant_data['MessageEvent.csv'].sort_index(axis=0, level='timestamp')
            # Update the StepCount.csv dataframe in the participant's data
            participant_data['MessageEvent.csv'] = data


# In[5]:


#Distance.csv
def distance_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'Distance.csv' in participant_data:
            data = participant_data['Distance.csv'].sort_index(axis=0, level='timestamp').assign(
                steps=lambda x: (x['TotalDistance'] - x['TotalDistance'].shift(1))
            )
            # Update the StepCount.csv dataframe in the participant's data
            participant_data['Distance.csv'] = data


# In[6]:


def calculate_top_sleep_proxies(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'DeviceEvent.csv' in participant_data:
            data = participant_data['DeviceEvent.csv']
            sleep_proxies = {}
            unlock_times = data[data['type'] == 'UNLOCK'].index

            for timestamp, event in data[data['type'] == 'SCREEN_OFF'].iterrows():
                screen_off_times = timestamp
              # Vectorized approach to find next unlock time after screen off
                next_unlock_time = unlock_times[unlock_times > screen_off_times].min()
                if not pd.isna(next_unlock_time):
                    time_diff = next_unlock_time - screen_off_times
                    if time_diff > pd.Timedelta(0):
                        date_key = screen_off_times.date()
                        sleep_proxies[date_key] = max(sleep_proxies.get(date_key, pd.Timedelta(0)), time_diff)

            sorted_proxies = sorted(sleep_proxies.items(), key=lambda x: x[0])
            proxies_df = pd.DataFrame(sorted_proxies, columns=['Date', 'SleepProxy'])
            proxies_df.set_index('Date', inplace = True)
          # Update participant's data with proxies_df (assuming key)
            participant_data['sleep_proxies'] = proxies_df


# In[7]:


def delete_preprocess(all_participants_data, df_name):
    # Iterate through each participant
    for participant_id, participant_data in all_participants_data.items():
        # Check if the participant has any of the specified dataframes
            if df_name in participant_data:
                del participant_data[df_name]


# In[26]:


#ActivityEvent.csv
def activityevent_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'ActivityEvent.csv' in participant_data:
            activity_event_data = participant_data['ActivityEvent.csv']
            # Apply the transformation
            transformed_df = activity_event_data.pivot_table(index='timestamp', columns='type', values='confidence', fill_value=0)
            #transformed_df.set_index('timestamp', inplace = True)
            # Store the transformed data in the participant's dictionary
            participant_data['ActivityEvent.csv'] = transformed_df


# In[9]:


#Calorie.csv
def calorie_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'Calorie.csv' in participant_data:
            data = participant_data['Calorie.csv'].sort_index(axis=0, level='timestamp').assign(
                steps=lambda x: (x['TotalCalories'] - x['TotalCalories'].shift(1))
            )
            # Update the StepCount.csv dataframe in the participant's data
            participant_data['Calories.csv'] = data


# In[10]:


#HR.csv
def hr_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'HR.csv' in participant_data: 
            data = participant_data['HR.csv'].sort_index(axis=0, level='timestamp')
            data = data[(data['BPM'] >= 30) & (data['BPM'] <= 200)]
            data = data.sort_values(by='timestamp')


# In[11]:


#SkinTemperature.csv
def skintemp_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'SkinTemperature.csv' in participant_data: 
            data = participant_data['SkinTemperature.csv'].sort_index(axis=0, level='timestamp')
            data = data[(data['Temperature'] >= 31) & (data['Temperature'] <= 38)]
            data = data.sort_values(by='timestamp')


# In[12]:


#Location.csv
def location_preprocess(all_participants_data):
    for participant_id, participant_data in all_participants_data.items():
        if 'Location.csv' in participant_data:
            data = participant_data['Location.csv'].sort_index(axis=0, level='timestamp')
            data['cluster'] = data.apply(lambda row: geo.encode(row['longitude'], row['latitude'], precision=7), axis=1)
            participant_data['Location.csv'] = data

