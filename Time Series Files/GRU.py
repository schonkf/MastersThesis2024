#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder 
from collections import defaultdict
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Embedding, LSTM, concatenate, Dense, Embedding, TimeDistributed, Flatten, Masking, Dropout, GRU, Concatenate, RNN
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score as f1_score_sklearn, accuracy_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit



# In[24]:


numerical_sequences_train_flat = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/numerical_sequences_train_flat.npy')
numerical_sequences_test_flat = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/numerical_sequences_test_flat.npy')
categorical_sequences_train_flat = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/categorical_sequences_train_flat.npy')
categorical_sequences_test_flat = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/categorical_sequences_test_flat.npy')
arousal_train = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/arousal_train.npy')
valence_train = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/valence_train.npy')
stress_train = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/stress_train.npy')
arousal_test = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/arousal_test.npy')
valence_test = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/valence_test.npy')
stress_test = np.load('/Users/finnschonknecht/Desktop/k-emo_notebooks/stress_test.npy')


# In[15]:


embedding_info = {0: (7, 4), 1: (2, 2), 2: (2398, 50), 3: (31, 16), 4: (31, 16), 5: (31, 16), 6: (31, 16), 7: (31, 16), 8: (31, 16)}


# In[22]:


es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

best_models = []
train_f1_scores = []
val_f1_scores = []
train_losses = []
val_losses = []

num_samples, num_timesteps, num_num_features = numerical_sequences_train_flat.shape

# Scaling and reshaping numerical data
scaler = MinMaxScaler()
numerical_sequences_reshaped = numerical_sequences_train_flat.reshape(-1, num_num_features)
numerical_sequences_normalized = scaler.fit_transform(numerical_sequences_reshaped)
numerical_sequences = numerical_sequences_normalized.reshape(num_samples, num_timesteps, num_num_features)

# Prepare the input data
X_numerical = numerical_sequences_train_flat
X_categorical = [categorical_sequences_train_flat[:, :, i] for i in range(categorical_sequences_train_flat.shape[2])]
X = [X_numerical] + X_categorical
y = np.array(valence_train)  # Assuming targets are already defined

# TimeSeriesSplit initialization
tscv = TimeSeriesSplit(n_splits=5)

for num_units in [32, 64, 128, 256]:  # Example hyperparameter range for LSTM units
    for batch_size in [32, 64, 128, 256]:  # Example hyperparameter range for batch size
        for dropout_rate in [0.1, 0.2, 0.4]:
            for train_index, val_index in tscv.split(X_numerical):

                # Split data
                X_train_numerical, X_val_numerical = X_numerical[train_index], X_numerical[val_index]
                X_train_categorical = [cat[train_index] for cat in X_categorical]
                X_val_categorical = [cat[val_index] for cat in X_categorical]
                y_train, y_val = y[train_index], y[val_index]

                # Define the inputs for numerical and categorical features
                num_input = Input(shape=(num_timesteps, num_num_features))
                cat_inputs = [Input(shape=(num_timesteps,), name=f'cat_var_{i}') for i in range(categorical_sequences_train_flat.shape[2])]

                # Masking layer
                masking_layer = Masking(mask_value=-1999)
                masked_num_input = masking_layer(num_input)

                # Create embedding layers for each categorical variable
                embeddings = []
                for i in range(categorical_sequences_train_flat.shape[2]):
                    embed_info = embedding_info[i]
                    embed = Embedding(input_dim=embed_info[0], output_dim=embed_info[1], input_length=num_timesteps)(cat_inputs[i])
                    embeddings.append(embed)

                # Concatenate all embeddings and numerical features
                embeddings_concat = concatenate(embeddings, axis=-1)
                masked_cat_input = masking_layer(embeddings_concat)

                # Concatenate numerical and categorical inputs
                combined_input = concatenate([masked_num_input, masked_cat_input], axis=-1)

                # Single LSTM layer with L2 regularization
                lstm_output = LSTM(units=num_units, activation='tanh', return_sequences=False)(combined_input)

                # Dropout layer
                dropout_rate = 0.4 # Increased dropout rate
                dropout_layer = Dropout(rate=dropout_rate)(lstm_output)

                # Output layer
                output_layer = Dense(1, activation='sigmoid')(dropout_layer)

                # Define and compile the model
                model = Model(inputs=[num_input] + cat_inputs, outputs=output_layer)
                model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

                # Fit the model
                history = model.fit([X_train_numerical] + X_train_categorical, y_train, 
                                    epochs=50, batch_size=batch_size, 
                                    validation_data=([X_val_numerical] + X_val_categorical, y_val),
                                    callbacks=[es])

                # Predict on validation data
                y_val_pred = (model.predict([X_val_numerical] + X_val_categorical) > 0.5).astype("int32")
                val_f1_score = f1_score(y_val, y_val_pred)
                val_loss = model.evaluate([X_val_numerical] + X_val_categorical, y_val, verbose=0)[0]

                # Predict on training data
                y_train_pred = (model.predict([X_train_numerical] + X_train_categorical) > 0.5).astype("int32")
                train_f1_score = f1_score(y_train, y_train_pred)
                train_loss = model.evaluate([X_train_numerical] + X_train_categorical, y_train, verbose=0)[0]

                train_f1_scores.append(train_f1_score)
                val_f1_scores.append(val_f1_score)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                best_models.append((model, val_f1_score, train_f1_score, train_loss, val_loss))

# Calculate mean training F1-score
mean_train_f1_score = np.mean(train_f1_scores)
print("Mean Training F1 Score:", mean_train_f1_score)

# Select the best model based on validation F1-score
best_model, best_val_f1_score, best_train_f1_score, best_train_loss, best_val_loss = max(best_models, key=lambda x: x[1])

# Save the best model in the Keras format
best_model.save('model/best_LSTM_model.keras')

print("Best Validation F1 Score:", best_val_f1_score)
print("Corresponding Training F1 Score:", best_train_f1_score)
print("Best Training Loss:", best_train_loss)
print("Best Validation Loss:", best_val_loss)
