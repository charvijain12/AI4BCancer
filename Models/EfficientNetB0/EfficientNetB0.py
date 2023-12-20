#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Load the dataset
df = pd.read_csv('C:\\Users\\Shaurya\\Downloads\\data.csv')

# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Extract features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convert 'M' to 1 and 'B' to 0

# Convert target to categorical
y = to_categorical(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[9]:


import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# After training the model
# Predict on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate and print the classification report
report = classification_report(y_true, y_pred)
print("Classification Report:\n", report)


# In[ ]:




