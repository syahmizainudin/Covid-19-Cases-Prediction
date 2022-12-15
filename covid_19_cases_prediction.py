# %%
import os
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# %% 1. Data loading
DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

train_df = pd.read_csv(os.path.join(DATASET_PATH, 'cases_malaysia_train.csv'))

# %% 2. Data inspection
train_df.info() # cases_new column is an object type which needs to be convert to int type
train_df.describe()

train_df.isna().sum() # the inital assesment shows no NaN in the cases column but cases_new column needs to be check again after it had been converted into int
train_df.duplicated().sum() # no duplicates

# %% 3. Data cleaning
# Convert cases_new column to int
train_df['cases_new'] = pd.to_numeric(train_df['cases_new'], errors='coerce')
train_df['cases_new'].isna().sum() # 12 NaN values in new_cases

# Fill NaN values in cases_new column with interpolation
train_df['cases_new'] = train_df['cases_new'].interpolate(method='polynomial', order=2).astype(np.int64)
train_df['cases_new'].isna().sum() # no more NaN values
train_df['cases_new'].dtype # cases_new column is now an dtype int

# Plot cases_new column to visualize the data's pattern
plt.figure()
plt.plot(train_df['cases_new'])
plt.xlabel('Day')
plt.ylabel('New Covid-19 Cases')
plt.title('New Covid-19 Cases Per Day')
plt.show()

# %% 4. Features selection
# Define the data that will be used for training
train_data = train_df['cases_new']

# %% 5. Data pre-processing
# Expand the dimension of the train_data
train_data_exp = np.expand_dims(train_data, -1)

# Normalization
mm_scaler = MinMaxScaler()
train_data_exp = mm_scaler.fit_transform(train_data_exp)

# Cut the data into specific time frame
TIME_FRAME = 30
train_features = [] 
train_targets = []

for i in range(TIME_FRAME, len(train_data_exp)):
    train_features.append(train_data_exp[i-TIME_FRAME:i])
    train_targets.append(train_data_exp[i])

# Convert list into numpy array
train_features = np.array(train_features)
train_targets = np.array(train_targets)

# Train-validation split
SEED = 12345

X_train, X_test, y_train, y_test = train_test_split(train_features, train_targets, random_state=SEED)

# %% Model development
# Build the Sequential model
model = Sequential()
model.add(LSTM(64, input_shape=X_train.shape[1:], return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1))

# Model summary
model.summary()
plot_model(model, to_file='resources/model_architecture.png', show_shapes=True, show_layer_names=True)

# Model compile
model.compile(optimizer='adam', loss='mse', metrics=['mape', 'mae'])

# Define callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tb = TensorBoard(log_dir=LOG_DIR)
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Model training
EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tb, es, reduce_lr])

# %% Model evaluation
# Load test data
test_df = pd.read_csv(os.path.join(DATASET_PATH, 'cases_malaysia_test.csv'))

# Inspect the test dataset
test_df.info() # cases_new column is a float dtype which needs to be convert to int
test_df.isna().sum() # there is 1 NaN values which needs to be fill in
test_df.duplicated().sum() # no duplicates

# Plot the cases_new column to visualize the trend
plt.figure()
plt.plot(test_df['cases_new'])
plt.xlabel('Day')
plt.ylabel('New Covid-19 Cases')
plt.title('New Covid-19 Cases Per Day')
plt.show()

# Define the data that will be used for testing
test_data = test_df['cases_new']

# Fill the NaN values with interpolation
test_data = test_data.interpolate(method='polynomial', order=2).astype(np.int64)
test_data.isna().sum() # There is no more NaN values
test_data.dtype # Test data dtype is now int

# Visualize the test data after it had been filled
plt.figure()
plt.plot(test_data)
plt.xlabel('Day')
plt.ylabel('New Covid-19 Cases')
plt.title('New Covid-19 Cases Per Day')
plt.show()

# Concatenate the train data and the test data
concat_data = pd.concat((train_data, test_data))
test_data = concat_data.loc[len(concat_data)-len(test_data)-TIME_FRAME:]

# Expand the dimension of the test data
test_data = np.expand_dims(test_data, -1)

# Normalize the test data
test_data = mm_scaler.transform(test_data)

# Cut the test data into specific timeframe
test_features = []
test_targets = []

for i in range(TIME_FRAME, len(test_data)):
    test_features.append(test_data[i-TIME_FRAME:i])
    test_targets.append(test_data[i])

test_features = np.array(test_features)
test_targets = np.array(test_targets)

# Do prediction with the model on the test data
y_pred = mm_scaler.inverse_transform(model.predict(test_features))
y_true = mm_scaler.inverse_transform(test_targets)

# Plot a graph for the prediction againts the true values
plt.figure()
plt.plot(y_true, color='b', label='Actual Cases')
plt.plot(y_pred, color='r', label='Predicted Cases')
plt.xlabel('Day')
plt.ylabel('Number of Cases')
plt.title('Actual Covid-19 Cases Againts Predicted Covid-19 Cases')
plt.legend()
plt.show()

# Evaluate the predictions
print('MAE:', mean_absolute_error(y_true, y_pred))
print('MAPE:', mean_absolute_percentage_error(y_true, y_pred))

# %% Model saving
# Save scaler
with open('mm_scaler.pkl','wb') as f:
    pickle.dump(mm_scaler, f)

# Save model
model.save('model.h5')
