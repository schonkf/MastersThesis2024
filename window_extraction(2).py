#!/usr/bin/env python
# coding: utf-8

# In[122]:


#MessageEvent.csv
def extract_message(df, windows, timestamp):
    #messages sent / received / total texts
    windowed_messages = {}
    before_esm = df[df.index <= timestamp]
    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        windowed_data = before_esm[before_esm.index >= (timestamp - window_timedelta)]
        sent = 0
        received = 0
        total = 0
        for message in windowed_data.messageBox:
            if message == 'INBOX':
                sent += 1
                total+= 1
            else:
                received += 1
                total += 1
        windowed_messages[window_name + '_messages_sent'] = sent
        windowed_messages[window_name + '_messages_received'] = received
        windowed_messages[window_name + '_messages_total'] = total
    return windowed_messages

def extract_message(df, windows, timestamp):
    #messages sent / received / total texts
    windowed_messageevent = {}
    before_esm = df[df.index <= timestamp]
    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        windowed_data = before_esm[before_esm.index >= (timestamp - window_timedelta)]
        unique_numbers_outgoing = []
        unique_numbers_incoming = []
        unique_messegers_outgoing = 0
        unique_messegers_incoming = 0 
        total_messages = 0 
        for message_time, message_type, number, in windowed_data[['messageBox', 'number']].itertuples(index=True):
            total_messages += 1
            if message_type == 'SENT':
                if number not in unique_numbers_outgoing:
                    unique_messegers_outgoing += 1
                    unique_numbers_outgoing.append(number)
            elif message_type == 'INBOX':
                if number not in unique_numbers_incoming:
                    unique_messegers_incoming += 1
                    unique_numbers_incoming.append(number)
        windowed_messageevent[f'{window_name}_unique_messegers_outgoing'] = unique_messegers_outgoing
        windowed_messageevent[f'{window_name}_unique_messegers_incoming'] = unique_messegers_incoming
        windowed_messageevent[f'{window_name}_total_messages'] = total_messages
        
    return windowed_messageevent


# In[123]:


#DeviceEvent.csv
def extract_deviceevent(df, windows, timestamp):
    # Number of times a phone is unlocked / time spent on phone
    windowed_deviceevent = {}
    before_esm = df[df.index <= timestamp]
    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        windowed_data = before_esm[before_esm.index >= (timestamp - window_timedelta)]
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
        windowed_deviceevent[f'{window_name}_times_unlocked'] = times_unlocked
        windowed_deviceevent[f'{window_name}_proportion_of_time_spent_on_phone'] = proportion_time_spent_on_phone
    
    return windowed_deviceevent


# In[2]:


#AppUsageEvent
def entropy(labels):
    n_labels = len(labels)
    
    if n_labels <= 1:
        return 0
    
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy

def extract_appusage(df, windows, timestamp):
    windowed_appevent = {}
    before_esm = df[df.index <= timestamp]
    
    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        window_start = timestamp - window_timedelta
        windowed_data = before_esm[(before_esm.index >= window_start) & (before_esm.index <= timestamp)]
        
        # Count the number of events associated with each app category
        app_category_counts = windowed_data['category'].value_counts().head(5)
        
        # Calculate the total number of app interactions in the window
        total_interactions = len(windowed_data)
        
        # Calculate the proportion of time spent using each app category
        app_category_proportions = app_category_counts / total_interactions
        
        # Calculate the entropy of the app category distribution
        app_category_entropy = entropy(windowed_data['category'])
        
        # Create keys for each category in the windowed_appevent dictionary
        for category, count in app_category_counts.items():
            windowed_appevent[f'{window_name}_{category}_number_of_events'] = count
            #windowed_appevent[f'{window_name}_{category}_proportion_of_time'] = app_category_proportions[category]
        
        # Add entropy to the windowed_appevent dictionary
        windowed_appevent[f'{window_name}_category_entropy'] = app_category_entropy
        
    return windowed_appevent


# In[125]:


#CallEvent.csv
def extract_call(df, windows, timestamp):
    #messages sent / received / total texts
    windowed_callevent = {}
    before_esm = df[df.index <= timestamp]
    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        windowed_data = before_esm[before_esm.index >= (timestamp - window_timedelta)]
        unique_numbers_outgoing = []
        unique_numbers_incoming = []
        unique_callers_outgoing = 0
        unique_callers_incoming = 0 
        time_spent_calling = 0
        for call_time, number, call_type, duration in windowed_data[['number', 'type', 'duration']].itertuples(index=True): 
            time_spent_calling += duration
            if call_type == 'OUTGOING':
                if number not in unique_numbers_outgoing:
                    unique_callers_outgoing += 1
                    unique_numbers_outgoing.append(number)
            elif call_type == 'INCOMING':
                if number not in unique_numbers_incoming:
                    unique_callers_incoming += 1
                    unique_numbers_incoming.append(number)
        windowed_callevent[f'{window_name}_unique_callers_outgoing'] = unique_callers_outgoing
        windowed_callevent[f'{window_name}_unique_callers_incoming'] = unique_callers_incoming
        windowed_callevent[f'{window_name}_time_spent_calling'] = time_spent_calling
    return windowed_callevent


# In[5]:


#Location.csv
def calculate_entropy(cluster_counts):
    total_time = cluster_counts.sum()
    cluster_proportions = cluster_counts / total_time
    entropy = -np.sum([p * np.log2(p) for p in cluster_proportions.values if p != 0])
    return entropy

def calculate_normalised_entropy(cluster_counts):
    if len(cluster_counts) == 0:
        return np.nan  # or set to some default value
        
    total_time = cluster_counts.sum()
    cluster_proportions = cluster_counts / total_time
    entropy = -np.sum([p * np.log2(p) for p in cluster_proportions.values if p != 0])
    
    # Calculate the maximum possible entropy
    max_entropy = -np.log2(1 / len(cluster_counts))
    
    # Check if maximum entropy is 0 or entropy is NaN
    if max_entropy == 0 or np.isnan(entropy):
        return np.nan  # or set to some default value
    
    # Normalize entropy
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy


def extract_location(df, windows, timestamp):
    windowed_location = {}
    before_esm = df[df.index <= timestamp]
    # Iterate through each window
    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        windowed_data = before_esm[before_esm.index >= (timestamp - window_timedelta)]
        # Calculate cluster counts within the window
        cluster_counts = windowed_data['cluster'].value_counts()
        
        # Calculate entropy for the window
        window_entropy = calculate_entropy(cluster_counts)
        window_normalised_entropy = calculate_normalised_entropy(cluster_counts)
        
        # Store entropy in the windowed_location dictionary
        windowed_location[window_name + '_location_entropy'] = window_entropy
        windowed_location[f'{window_name}_normalised_location_entropy'] = window_normalised_entropy
    
    return windowed_location


# In[7]:


def extract_generic_window_features(df, windows, timestamp):
    windowed_features = {}
    before_esm = df[df.index <= timestamp]

    for window_name, window_size in windows.items():
        window_timedelta = pd.Timedelta(seconds=window_size)
        windowed_data = before_esm[before_esm.index >= (timestamp - window_timedelta)]

        # Calculate mean, median, and standard deviation for numeric columns
        numeric_cols = windowed_data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            col_mean = windowed_data[col].mean()
            col_median = windowed_data[col].median()
            col_std = windowed_data[col].std()
            windowed_features[f'{col}_mean_{window_name}'] = col_mean
            windowed_features[f'{col}_median_{window_name}'] = col_median
            windowed_features[f'{col}_std_{window_name}'] = col_std
        
            # Calculate entropy and time-series complexity estimate (you need to implement these)
            # entropy = calculate_entropy(windowed_data)
            # complexity_estimate = calculate_complexity_estimate(windowed_data)
            # windowed_features[f'entropy_{window_name}'] = entropy
            # windowed_features[f'complexity_estimate_{window_name}'] = complexity_estimate
    
    return windowed_features


# In[6]:


def extract_windowed_data(all_participants_data, esm_responses, processing_functions, windows):
    windowed_data_list = []  # List to store dictionaries of windowed features
    
    # Iterate through each participant's data dictionary
    for participant_id, participant_data in all_participants_data.items():
        # Get all ESM responses for the current participant
        participant_esm_responses = esm_responses[esm_responses['Pcode'] == participant_id]
        
        # Iterate through each ESM response for the participant
        for index, esm_response in participant_esm_responses.iterrows():
            timestamp = index
            timestamp = pd.Timestamp(timestamp)
            p_code = esm_response['Pcode']
            stress = esm_response['Stress_binary']
            valence = esm_response['Valence_binary']
            arousal = esm_response['Arousal_binary']
            
            windowed_features = {
                'ResponseTime': timestamp,
                'Pcode': p_code,
                'Stress_binary': stress,
                'Valence_binary': valence,
                'Arousal_binary': arousal
            }
            
            # Iterate through each dataframe in the participant's data dictionary
            for dataframe_name, dataframe in participant_data.items():
                # Check if current dataframe needs specific windowed feature extraction
                if dataframe_name in processing_functions:
                    # Get the processing function for the current dataframe
                    processing_function = processing_functions[dataframe_name]
                    additional_features = processing_function(dataframe, windows, timestamp)
                    windowed_features.update(additional_features)
                else:
                    # Apply generic window feature extraction for numeric dataframes
                    generic_features = extract_generic_window_features(dataframe, windows, timestamp)
                    windowed_features.update(generic_features)
            
            windowed_data_list.append(windowed_features)
    
    # Convert the list of dictionaries into a DataFrame
    windowed_data_df = pd.DataFrame(windowed_data_list)
    
    return windowed_data_df


# In[8]:


def all_participants(all_participants_data, esm_responses, user_info, processing_functions, windows):
    all_windowed_data_list = []  # List to store DataFrames of windowed features
    
    # Load user information
    user_info_df = user_info
    
    # Merge user information with ESM responses based on participant code
    esm_responses = pd.merge(esm_responses, user_info_df, on='Pcode')
    
    # Iterate through each participant's data dictionary
    for participant_id, participant_data in all_participants_data.items():
        windowed_data_list = []  # List to store dictionaries of windowed features
        
        # Get all ESM responses for the current participant
        participant_esm_responses = esm_responses[esm_responses['Pcode'] == participant_id]
        
        # Iterate through each ESM response for the participant
        for index, esm_response in participant_esm_responses.iterrows():
            timestamp = index
            p_code = esm_response['Pcode']
            stress = esm_response['Stress_binary']
            valence = esm_response['Valence_binary']
            arousal = esm_response['Arousal_binary']
            
            # Extract user information for the current participant from user_info_df
            user_info_row = user_info_df[user_info_df['Pcode'] == participant_id].iloc[0]
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
            
            # Find the sleep proxy for the current date from participant_data
            sleep_proxy_df = participant_data.get('SleepProxies.csv')
            if sleep_proxy_df is not None:
                sleep_proxy_row = sleep_proxy_df[sleep_proxy_df['Date'] == timestamp.date()]
                sleep_proxy = sleep_proxy_row['SleepProxy'].iloc[0] if not sleep_proxy_row.empty else pd.NaT
            else:
                sleep_proxy = pd.NaT
            
            windowed_features = {
                'ResponseTime': timestamp,
                'Pcode': p_code,
                'Stress_binary': stress,
                'Valence_binary': valence,
                'Arousal_binary': arousal,
                'Age': age,
                'Gender': gender,
                'Openness': openness,
                'Conscientiousness': conscientiousness,
                'Neuroticism': neuroticism,
                'Extraversion': extraversion,
                'Agreeableness': agreeableness,
                'PSS10': pss10,
                'PHQ9': phq9,
                'GHQ12': ghq12,
                'SleepProxy': sleep_proxy
            }
            
            # Iterate through each dataframe in the participant's data dictionary
            for dataframe_name, dataframe in participant_data.items():
                # Check if current dataframe needs specific windowed feature extraction
                if dataframe_name in processing_functions:
                    # Get the processing function for the current dataframe
                    processing_function = processing_functions[dataframe_name]
                    
                    # Apply the processing function to compute additional features
                    additional_features = processing_function(dataframe, windows, timestamp)
                    
                    # Store additional features in the windowed features dictionary
                    windowed_features.update(additional_features)
                else:
                    # Apply generic window feature extraction for numeric dataframes
                    additional_features = extract_generic_window_features(dataframe, windows, timestamp)
                    
                    # Store additional features in the windowed features dictionary
                    windowed_features.update(additional_features)
            
            # Append the windowed features dictionary to the list for this participant
            windowed_data_list.append(windowed_features)
        
        # Convert the list of dictionaries into a DataFrame for this participant
        windowed_data_df = pd.DataFrame(windowed_data_list)
        
        # Append the windowed data for this participant to the list for all participants
        all_windowed_data_list.append(windowed_data_df)
    
    # Concatenate all windowed dataframes for each participant into a single dataframe
    all_windowed_data_df = pd.concat(all_windowed_data_list, ignore_index=True)
    
    return all_windowed_data_df

