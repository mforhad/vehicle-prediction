import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Replace 'data/Melbourn.csv' with the actual path to your CSV file
df = pd.read_csv('data/Melbourn.csv')

# Extract the numerical values for the model
data = df.iloc[:, 6:].values  # Selecting columns from '00:00' to the end

# Create labels representing the number of vehicles in the next timeframe
labels = np.roll(data, shift=-1, axis=1)
labels[:, -1] = 0  # Set the last timeframe's label to 0 as there is no next timeframe

# Reshape data for the Conv1D layer
data = data.reshape((len(df), len(df.columns) - 6, 1))
labels = labels.reshape((len(df), len(df.columns) - 6, 1))

print(labels.shape)
print(data.shape)

# Split the data into training and temporary sets (80% training, 20% temporary)
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train_temp.shape)
print(y_train_temp.shape)

# Split the temporary set into validation and test sets (50% validation, 50% test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_val.shape)
print(y_val.shape)

# # Define the CNN model
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=15, activation='relu', input_shape=(X_train_temp.shape[1], 1)))
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
# model.add(Dense(32, activation='relu'))  # Adjust the number of units as needed
# model.add(Dense(units=1, activation='sigmoid'))
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=15, activation='relu', input_shape=(X_train_temp.shape[1], 1)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(X_train_temp.shape[1], activation='relu'))
model.add(Dense(units=1, activation='linear'))  # Use linear activation for regression


# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train your model using the training and validation sets
num_epochs = 15  # Adjust as needed
batch_size = 8  # Adjust as needed

model.fit(X_train_temp, y_train_temp, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate your model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100}')
