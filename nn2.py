import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load the MovieLens 1M dataset
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=header, engine='python', encoding='latin-1')

# Preprocess the dataset
df['rating'] = df['rating'].astype(float)

# Merge demographic information with the dataset
user_info = pd.read_csv('./ml-1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zipcode'], engine='python', encoding='latin-1')
df = df.merge(user_info, on='user_id')

# Feature engineering
user_features = df.groupby('user_id')['rating'].agg(['count', 'mean'])
df = df.merge(user_features, how='left', on='user_id')

# Convert gender and occupation to numerical labels
df['gender'] = df['gender'].map({'M': 0, 'F': 1})
df['occupation'] = df['occupation'].astype('category').cat.codes

X = df[['user_id', 'item_id', 'count', 'mean', 'age', 'gender', 'occupation']].values
y = df['rating'].values

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the deep neural network architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)
