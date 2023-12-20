#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import classification_report, accuracy_score

# Load and preprocess the dataset
df = pd.read_csv('C:\\Users\\Shaurya\\Downloads\\data.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
y = to_categorical(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pad the data to get 36 features (6x6)
X_padded = np.pad(X_scaled, ((0, 0), (0, 36 - X_scaled.shape[1])), 'constant')

# Reshape the data to 6x6 and then pad it to 32x32
X_reshaped = np.reshape(X_padded, (-1, 6, 6))
X_padded = np.pad(X_reshaped, ((0, 0), (0, 26), (0, 26)), 'constant')

# Add an additional dimension to mimic 3 channels (RGB)
X_padded = np.stack((X_padded,)*3, axis=-1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Load the DenseNet121 model
base_model = DenseNet121(include_top=False, input_shape=(32, 32, 3), pooling='avg')
base_model.trainable = False

# Build and compile the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[2]:


# Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


# In[ ]:




