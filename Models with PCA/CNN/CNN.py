#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


# In[2]:


# Load the dataset
file_path = r"C:\Users\Shaurya\Downloads\data.csv"
df = pd.read_csv(file_path)


# In[3]:


# Preprocess the data
df = df.drop(columns=['id', 'Unnamed: 32'])
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
df = df.dropna()
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']


# In[4]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


# Apply PCA
n_components = 10  # Adjust based on your requirement
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[6]:


# Reshape data for CNN input
X_train_pca = X_train_pca.reshape(X_train_pca.shape[0], n_components, 1, 1)
X_test_pca = X_test_pca.reshape(X_test_pca.shape[0], n_components, 1, 1)


# In[7]:


# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(n_components, 1, 1)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[8]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[9]:


# Train the model
model.fit(X_train_pca, y_train, epochs=50, batch_size=32, validation_split=0.2)


# In[10]:


# Evaluate the model
y_pred_prob = model.predict(X_test_pca)
y_pred = (y_pred_prob > 0.5).astype(int)


# In[11]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


# In[ ]:




